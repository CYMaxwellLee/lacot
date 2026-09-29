"""u 編碼想像軌跡。Additional real RefineOperator, frozen true Flow, analytic plan world."""
from toy import *
from lacot.model import RefineOperator
from lacot.nf_head import Flow
import json
results=[]
for seed in (0,1,2):
    torch.manual_seed(seed)
    flow=Flow(4,1,n_blocks=1,d_hidden=8,n_layers=1,n_heads=2,cond_dim=1).eval().requires_grad_(False)
    model=RefineOperator(1,1,4,hidden=32)
    teacher=copy.deepcopy(model).requires_grad_(False)
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    direction=torch.tensor([1.,-1.,1.,-1.]).reshape(1,1,4)
    def quality_for(u):
        # Two lawful unit-scale route plans; choose this sampled plan's branch.
        route=(u*direction).sum(-1,keepdim=True).sign().detach()
        target=route*direction
        return lambda x:(x-target).square().mean((1,2)),target
    cond=torch.zeros(64,1)
    for step in range(700):
        u=flow.sample(64,cond)
        q,target=quality_for(u)
        terms=refinement_terms(model,teacher,q,cond,target,u,rounds=1+step%3,noise=.08*torch.randn_like(u))
        loss=terms.quality+.1*terms.consistency
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            for t,p in zip(teacher.parameters(),model.parameters()):t.lerp_(p,.05)
    cond=torch.zeros(512,1)
    with torch.no_grad():
        u=flow.sample(512,cond);q,target=quality_for(u);costs=[q(u).mean().item()];errors=[0.]
        for r in range(6):
            u=model(cond,u);costs.append(q(u).mean().item())
            errors.append(((u*direction).sum(-1).sign()!=target[...,0].sign()).float().mean().item())
    row=dict(seed=seed,optimizer_steps=700,quality=costs,mode_error=errors)
    print(json.dumps(row),flush=True);results.append(row)
passed=all(r['quality'][3]<.15*r['quality'][0] and r['quality'][3]<.12 and r['mode_error'][3]<.05 for r in results)
print(('PASS' if passed else 'FAIL')+' real RefineOperator analytic-world off-manifold CPU training; no feedback features added')
raise SystemExit(0 if passed else 1)
