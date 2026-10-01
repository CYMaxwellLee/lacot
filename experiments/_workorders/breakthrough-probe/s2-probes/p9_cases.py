"""S2 own judgment cases against rules.py / harvest.py (mirror copy, unmodified)."""
import sys, random, itertools
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
import numpy as np, rules as r, harvest, common
from math import ceil
res=[]
def chk(name, got, exp):
    ok = got==exp; res.append(ok); print(('PASS' if ok else 'FAIL'), name, '' if ok else f'got={got!r} expected={exp!r}')
c=['failure']*16
# ---- Case A: tie / ambiguity
chk('A1 tie d3=16,d5=16 -> AMBIGUOUS', r.question_grid(['d3']*8+['d5']*8,['d3']*8+['d5']*8,c)['grid'], r.AMBIGUOUS)
chk('A2 tie d2=16,d3=16 (each arm d2=8, not >50%) -> AMBIGUOUS not INSUFFICIENT', r.question_grid(['d2']*8+['d3']*8,['d2']*8+['d3']*8,c)['grid'], r.AMBIGUOUS)
chk('A3 d1-only both arms -> AMBIGUOUS (no unique max)', r.question_grid(['d1']*16,['d1']*16,c)['grid'], r.AMBIGUOUS)
chk('A4 d2 unique max but reliable -> INSUFFICIENT & not world-counted', (lambda g:(g['grid'],g['world_counted']))(r.question_grid(['d2']*8+['d3']*4+['d5']*4,['d2']*8+['d3']*4+['d5']*4,c)), (r.INSUFFICIENT,False))
chk('A5 C has 7 valid fails + tie elsewhere -> INSUFFICIENT precedes AMBIGUOUS', r.question_grid(['d3']*8+['d5']*8,['d3']*8+['d5']*8,['failure']*7+[None]*9)['grid'], r.INSUFFICIENT)
chk('A6 E4 top two within 1 -> mixed row', r.e4([r.BOTH]*6+[r.A_ONLY]*5+[r.DEEP]*3,8,0,'跟隨者')['row'], '混合格/幾何間分歧')
chk('A7 E4 margin exactly 2 -> dominant', r.e4([r.A_ONLY]*6+[r.DEEP]*4,0,0,'跟隨者')['row'], '時程形主導')
chk('A8 E4 only AMBIGUOUS/INSUFFICIENT -> 無合格題', r.e4([r.AMBIGUOUS]*7+[r.INSUFFICIENT]*7,0,0,'跟隨者')['row'], '無合格題')
# ---- Case B: budget-exceeded flow
chk('B1 9 invalid + 7 valid d4 on A, B fine, C all fail -> INSUFFICIENT', r.question_grid([None]*9+['d4']*7,['d4']*16,c)['grid'], r.INSUFFICIENT)
chk('B2 C success draw over budget cannot rescue', r.classify('C',calls=253,chunk_steps=4,w_step=None,g_step=5,crossing=None), None)
chk('B3 invalid draws excluded from w-miss denominator: A d2=5,valid=8,invalid=8 -> unreliable', r.arm_stats(['d2']*5+['d3']*3+[None]*8)['w_unreliable'], True)
chk('B4 A d2=4 of valid 8 not unreliable, not insufficient', (lambda s:(s['w_unreliable'],s['insufficient']))(r.arm_stats(['d2']*4+['d3']*4+[None]*8)), (False,False))
chk('B5 all 16 invalid -> valid 0, w_miss None, insufficient', (lambda s:(s['valid'],s['w_miss_rate'],s['insufficient']))(r.arm_stats([None]*16)), (0,None,True))
# harvest-level: over-budget ledger on a real-shaped row
H=common.H
def mkrow(arm, w=None, g=None, steps=40, extra_calls=0, chunk=4):
    xy=np.zeros((steps+1,2)); 
    if w is not None: xy[w:]=[4.,0.]
    gs=[False]*(steps+1)
    if g is not None: gs[g]=True; 
    row=dict(arm=arm,xy=xy.tolist(),goal_success=gs,observed_steps=steps,chunk_steps=chunk,synthetic=True)
    exp=[];t=0
    while t<steps:
        tgt='w' if arm!='C' and (w is None or t<w) else 'g'
        exp.append(dict(step=t,target=tgt)); t = w if (tgt=='w' and w is not None and t<w<t+chunk) else t+chunk
    row['flow_calls']=exp + [dict(step=steps-1,target='g')]*extra_calls
    return row
row=mkrow('C',steps=1000); row['goal_success']=[False]*1000+[True]; 
row['flow_calls']=[dict(step=t,target='g') for t in range(0,1000,4)]
out=harvest.recompute_move3(row,None,lambda r_:True)
chk('B6 C 1000-step success with exact 250 calls valid -> success', out['category'], 'success')
row['flow_calls']=row['flow_calls']+[dict(step=999,target='g')]*3   # 253 calls
out=harvest.recompute_move3(row,None,lambda r_:True)
chk('B7 same with 253 calls -> invalid (None) with budget-exceeded', (out['category'], 'budget-exceeded' in out['invalid']), (None,True))
# ---- Case C: bypass / same-step / d4 d5 differential vs rules.classify
random.seed(1); bad=0; n=0
for trial in range(4000):
    arm=random.choice('AB'); steps=random.choice([30,64,200])
    w=random.choice([None]+list(range(1,steps+1))); g=random.choice([None]+list(range(1,steps+1)))
    if g is not None:   # episode ends at success
        steps=g
        if w is not None and w>steps: w=None
    crossing=random.choice([True,False])
    row=mkrow(arm,w,g,steps)
    out=harvest.recompute_move3(row,[4.,0.],lambda r_:crossing)
    if out['invalid']: continue
    ref=r.classify(arm,calls=len(row['flow_calls']),chunk_steps=4,w_step=out['w_step'],g_step=out['g_step'],crossing=crossing if (out['w_step'] is not None and out['g_step'] is not None) else None)
    n+=1; bad+= (ref!=out['category'])
print('C-differential (harvest.recompute vs rules.classify) trials',n,'disagreements',bad); res.append(bad==0)
chk('C1 same-step w&g (w=g=7) -> d4 when crossing (not bypass)', harvest.recompute_move3(mkrow('A',7,7,7),[4.,0.],lambda r_:True)['category'], 'd4')
chk('C2 rules.classify same-step -> d4', r.classify('A',calls=2,chunk_steps=4,w_step=7,g_step=7,crossing=True), 'd4')
chk('C3 bypass: g before w', harvest.recompute_move3(mkrow('B',None,9,9),[4.,0.],lambda r_:True)['category'], 'd1')
# d1-heavy corner (contract gap, report only)
g=r.question_grid(['d1']*15+['d3'],['d1']*16,c)
print('INFO C4 A=15xd1+1xd3, B=16xd1, C all fail ->', g['grid'], '(single d3 draw yields DEEP; card silent on d1-dominant cells)')
# ---- Case D: behavior_gate rounding, E2/E3 bounds, exact p
for q,p,rr,exp in [(5,4,0,'passed'),(5,3,0,'行為資格不足'),(6,5,0,'passed'),(6,4,0,'行為資格不足'),(7,6,2,'passed'),(7,5,0,'行為資格不足'),(8,6,3,'gate-content-insensitive'),(4,4,0,'gate 樣本不足'),(0,0,0,'gate 樣本不足')]:
    chk(f'D behavior_gate({q},{p},{rr})', r.behavior_gate(q,p,rr)['status'], exp)
chk('D e2 task5 3/6 not stopped', r.e2([(True,False)]*3+[(False,False)]*3,5)['stopped'], False)
chk('D e2 task5 4/6 stopped', r.e2([(True,False)]*4+[(False,False)]*2,5)['stopped'], True)
chk('D ceiling task5 4/6 false, 5/6 true', (r.ceiling(['A']*4+['O']*2,5), r.ceiling(['A']*5+['O'],5)), (False,True))
chk('D exact p (5,0)', abs(r.exact_one_sided(5,0)-1/32)<1e-12, True)
chk('D exact p (10,3)<.05', r.exact_one_sided(10,3)<.05, True)
chk('D exact p (0,0)=1', r.exact_one_sided(0,0), 1.0)
# ---- Case E: E4 first-hit, B_ONLY handling, stop-loss flag
chk('E1 BOTH dominant, 5:0 -> 岔路可執行', r.e4([r.BOTH]*9+[r.DEEP]*3,5,0,'跟隨者')['row'], '岔路可執行')
chk('E2 BOTH dominant, 4:0 -> B d4 主導但未站穩', r.e4([r.BOTH]*9+[r.DEEP]*3,4,0,'跟隨者')['row'], 'B d4 主導但未站穩')
chk('E3 B_ONLY dominant, 6:1 (p=.0625) -> 未站穩', r.e4([r.B_ONLY]*8+[r.DEEP]*3,6,1,'跟隨者')['row'], 'B d4 主導但未站穩')
chk('E4 stop-loss flag forward=2', r.e4([r.DEEP]*8,2,0,'跟隨者')['axis1_deprioritized'], True)
chk('E5 d5 dominant -> UNVERIFIED row regardless of p', r.e4([r.UNVERIFIED]*7+[r.BOTH]*2,9,0,'跟隨者')['row'], r.UNVERIFIED)
chk('E6 insensitive column for 岔路可執行', r.e4([r.BOTH]*9,6,0,'吸附')['action'], '修讀出端（Astra 分段介面）')
chk('E7 directional column for 岔路可執行', r.e4([r.BOTH]*9,6,0,'方向性')['action'], '加題補證後重判')
print('SUMMARY', sum(res), 'pass /', len(res))
