"""u 編碼想像軌跡。B1 population probe and honest selection/GD controls."""
import math,json,time
from toy import *
from acceptance import assess_run
for sigma in (.02,.05,.1):
    logp=1024*math.log(math.erf(.05/(sigma*math.sqrt(2))))
    print(f'B1 KD=1024 sigma={sigma} box_probability={math.exp(logp):.8g} logP={logp:.4f}')
# Population minimizer under v1: no data-band supervision outside the box,
# conservative penalty uniquely selects identity there. Same actual flow samples.
stats,(flow,refine,head,c,u,q)=train(seed=0)
old=u.clone()
inside=((u-torch.stack([q.route,q.goal],-1)).abs()<=.05).all((1,2))
old[inside]=torch.stack([q.route,q.goal],-1)[inside]
ratio=(q(old).mean()/q(u).mean()).item()
identity=dict(stats,quality=[q(u).mean().item()]*7)
assert not assess_run(identity)[0]
assert ratio>.95
print(f'PASS B1 old population optimum correction_ratio={ratio:.6f}; identity rejected by same gate')
# Compare fixed 512 conditions. This is measured, not a claimed fair walltime win.
rows=[]
def timed(name,fn,compute):
    start=time.perf_counter();v=fn(); elapsed=time.perf_counter()-start
    with torch.no_grad():
        cost=q(v).mean().item(); plus=(v[...,0]>0).float().mean().item()
    rows.append(dict(method=name,cost=cost,seconds_excluding_shared_initial_flow=elapsed,compute=compute,positive_mode_fraction=plus))
for r in (0,1,3):
    def run(r=r):
        with torch.no_grad():
            v=u.clone()
            for _ in range(r): v=refine(c,v)
            return v
    timed('learned-R'+str(r),run,dict(flow=1,refine=r))
def gd():
    v=u.detach().clone().requires_grad_()
    for _ in range(3):
        g=torch.autograd.grad(q(v).sum(),v)[0]
        v=(v-.25*g).detach().requires_grad_()
    return v.detach()
timed('gradient-3',gd,dict(flow=1,decoder_forward=3,decoder_backward=3))
def bon():
    # Preserve each base sample's mode for this paired comparison; reject opposite branches.
    with torch.no_grad():
        candidates=torch.stack([u]+[flow.sample(len(c),c) for _ in range(3)])
        costs=torch.stack([q(v) for v in candidates])
        costs=costs.masked_fill(candidates[...,0].squeeze(-1).sign()!=q.route.squeeze(-1),float('inf'))
        best=costs.argmin(0)
        return candidates[best,torch.arange(len(c))]
timed('best-of-4-same-mode',bon,dict(flow=4,decoder_forward=4))
print('BASELINES '+json.dumps(rows,sort_keys=True))
print('NOTE toy oracle exact; BoN mode mask uses toy branch identity. Production BoN uses qualified quality scorer and must also report unrestricted mode coverage.')
print('PASS B1/scaling/baselines probe (no universal learned-vs-selection claim)')
