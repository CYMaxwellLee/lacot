"""u 編碼想像軌跡。Main-b/EMA-specific gate, intentionally separate from top-level acceptance."""
from toy import *
from conformance import fixture,objective_suite,composition_suite
from fakes import refinement_terms,compose
objective_suite(refinement_terms); composition_suite(compose)
cl,s,n,c,p,f,t,q=fixture()
x=refinement_terms(f,t,q,c,cl,s,rounds=1,noise=n)
assert torch.autograd.grad(x.consistency,p)[0].norm()>1e-6
from lacot.model import RefineOperator
op=RefineOperator(1,1,4,hidden=8)
teacher=copy.deepcopy(op).requires_grad_(False)
with torch.no_grad(): teacher.net[-1].bias.add_(torch.tensor([.2,-.1,.1,-.2]))
u=torch.tensor([[[1.,-1.,.7,-.7]], [[-.8,.8,-1.,1.]]]); cond=torch.zeros(2,1)
z=refinement_terms(op,teacher,lambda u:u.square().mean((1,2)),cond,u,u,rounds=1,noise=torch.ones_like(u)*.03)
g=torch.autograd.grad(z.consistency,tuple(op.parameters()))
assert sum(v.square().sum() for v in g)>1e-8
assert all(p.grad is None for p in teacher.parameters())
print('PASS F5 real RefineOperator R1 stale EMA target -> nonzero student gradient; teacher detached')
print('NOTE synchronized teacher/student may have zero consistency gradient; quality must teach first')
