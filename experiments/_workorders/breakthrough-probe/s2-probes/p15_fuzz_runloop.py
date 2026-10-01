"""Differential fuzz of runtime.run_draw (real TwoStage + real loop + real recompute) vs an independent ground-truth oracle.
Fake module/env (no model): positions are scripted, so ground truth is known by construction."""
import sys, os, math, random
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='')
import numpy as np, torch
import common, runtime, harvest
from common import H, RHO
from types import SimpleNamespace
W=np.array([10.,10.]); GOAL=np.array([40.,40.])
class Space:
    def seed(self,s): self.np_random=np.random.default_rng(s)
class Env:
    def __init__(self): self.action_space=Space(); self.unwrapped=self
    def load(self,script,succ): self.script=script; self.succ=succ
    def reset(self,seed,options):
        self.np_random=np.random.default_rng(seed); self.t=0
        return self.script[0].copy(), {'goal':GOAL.copy()}
    def step(self,a):
        self.t+=1; t=self.t
        return self.script[t].copy(), 0., False, t>=H, {'success': (self.succ is not None and t==self.succ)}
mod=SimpleNamespace(torch=torch,CHUNK=4,_GRAD_CACHE={'u':None},_reseed_shuf=lambda i:None,_RDIR=[1.0],
    condvec=lambda s,g: torch.zeros(1,4), normstate=lambda x: torch.zeros(1,2), goal_to_obs=lambda g: g,
    sample_plan=lambda n,c,a: torch.zeros(1,8,256), ahead=lambda c,u: torch.zeros(1,4,2), _q=lambda u:u)
def dist(a,b): return float(np.linalg.norm(np.asarray(a,float)-np.asarray(b,float)))
def make_script(rng):
    T=H+1
    pos=np.zeros((T,2)); p=np.array([0.,0.]); pos[0]=p
    th=rng.choice([None]*3+list(range(1,300)))                   # intended hit time
    mode=rng.choice(['walk','still_start','still_mid'])
    still_from=rng.randint(5,80); still_len=rng.randint(10,60)
    for t in range(1,T):
        if mode=='still_start' and t<=still_len+25: pos[t]=pos[t-1]; continue
        if mode=='still_mid' and still_from<=t<still_from+still_len: pos[t]=pos[t-1]; continue
        ang=rng.uniform(0,2*math.pi); step=rng.uniform(0.0,0.8)
        pos[t]=pos[t-1]+step*np.array([math.cos(ang),math.sin(ang)])
        # keep away from W unless it's the intended hit time (avoid accidental early hits)
        if dist(pos[t],W)<RHO+0.3 and t!=th: pos[t]=W+ (pos[t]-W)/max(dist(pos[t],W),1e-9)*(RHO+0.4)
    if th is not None:
        pos[th]=W+np.array([rng.uniform(-0.8,0.8),rng.uniform(-0.8,0.8)])
        # after hit let it move on freely (but not teleport out-of-range issues)
    succ=rng.choice([None]*4+list(range(1,H+1))) if rng.random()<0.5 else rng.choice([None,None,None]+list(range(1,400)))
    return pos,th,succ
def truth(pos,succ,arm,cap,sw,sd):
    """independent oracle: returns dict(steps, reason, w_step, g_step)"""
    stage='w' if arm!='C' else 'g'; w=None; g=None
    for t in range(1,H+1):
        p=pos[t]
        if stage=='w' and dist(p,W)<RHO: w=t; stage='g'
        if succ is not None and t==succ: g=t
        if g is not None: return dict(steps=t,reason='success',w=w,g=g)
        if t>=H: return dict(steps=t,reason='truncated',w=w,g=g)
        if stage=='w':
            win=pos[max(0,t-sw):t+1]
            stuck = (t+1)>sw and len(win)==sw+1 and np.max(np.linalg.norm(win-win[0],axis=1))<=sd
            if stuck: return dict(steps=t,reason='stuck',w=w,g=g)
            if t>=cap: return dict(steps=t,reason='cap',w=w,g=g)
    raise AssertionError
rng=random.Random(7); env=Env()
bad=[]; cats={}; n=0
for trial in range(1500):
    arm=rng.choice('ABC'); pos,th,succ=make_script(rng)
    cap=rng.choice([20,37,48,96,150,400,1000]); sw=rng.choice([10,20,30]); sd=rng.choice([0.2,0.5])
    env.load(pos,succ)
    stuck=runtime.calibrated_stuck(sw,sd)
    plan=dict(task=4,episode=4,draw=64,arm=arm)
    row=runtime.run_draw(mod,env,plan,move=3,waypoint=None if arm=='C' else W,cap_steps=cap,stuck_check=stuck)
    row['synthetic']=True
    tr=truth(pos,succ,arm,cap,sw,sd)
    out=harvest.recompute_move3(row,None if arm=='C' else W,lambda r_:True)
    n+=1
    errs=[]
    if row['observed_steps']!=tr['steps']: errs.append(('steps',row['observed_steps'],tr['steps']))
    if row['termination_reason']!=tr['reason']: errs.append(('reason',row['termination_reason'],tr['reason']))
    if out['invalid']: errs.append(('invalid',out['invalid']))
    if out['w_step']!=(tr['w'] if arm!='C' else None): errs.append(('w',out['w_step'],tr['w']))
    if out['g_step']!=tr['g']: errs.append(('g',out['g_step'],tr['g']))
    if len(row['flow_calls'])>math.ceil(H/4)+2: errs.append(('calls',len(row['flow_calls'])))
    if len(row['xy'])!=row['observed_steps']+1: errs.append(('tracelen',))
    if row['runtime_errors']: errs.append(('runtime_errors',row['runtime_errors']))
    cats[(arm,out['category'])]=cats.get((arm,out['category']),0)+1
    if errs: bad.append((trial,arm,cap,sw,sd,th,succ,errs))
print('trials',n,'failures',len(bad))
for b in bad[:8]: print(b)
print('category coverage:',dict(sorted(cats.items())))
