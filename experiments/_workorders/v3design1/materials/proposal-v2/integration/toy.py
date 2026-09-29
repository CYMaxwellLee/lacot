"""u 編碼想像軌跡。CPU-only small training harness; v1 factorial routes retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import copy
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'contracts'))
from fakes import refinement_terms,compose
from api import Adapter
from oracle_fake import Quality,gauge
REPO=Path('/home/cymaxwelllee/Projects/lacot')
sys.path.insert(0,str(REPO))
torch.set_num_threads(1)
class TinyRefine(nn.Module):
    # Toy geometry feedback (corridor side/sign) is explicit. This changed after
    # raw-latent training failed; it is NOT the existing production RefineOperator.
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,24),nn.Tanh(),nn.Linear(24,2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,c,u):
        return u+self.net(torch.cat([c[:,:1],u.flatten(1),u.flatten(1).sign()],-1)).reshape_as(u)
class TinyHead(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Linear(2,1)
        with torch.no_grad(): self.net.weight.copy_(torch.tensor([[.7,.2]])); self.net.bias.zero_()
    def forward(self,u): return self.net(u)
class ReplayFlow(nn.Module):
    def __init__(self,u): super().__init__(); self.u=u
    def sample(self,b,c): assert b==len(self.u); return self.u.clone()
    def nll(self,u,c): return u.square().mean()
def batch(n=64):
    assert n % 4 == 0
    # Four repeated conditions allow the exact data-route x sampled-route factorial.
    cond=torch.linspace(-.5,.5,n//4).repeat_interleave(4).reshape(n,1)
    data=torch.tensor([1.,-1.]).repeat(n//2).reshape(n,1,1)
    clean=torch.cat([data,cond[:,None]],-1)
    return cond,clean,data

def real_flow():
    from lacot.nf_head import Flow
    flow=Flow(2,1,n_blocks=1,d_hidden=8,n_layers=1,n_heads=2,cond_dim=1).eval()
    flow.requires_grad_(False)
    return flow

def ref_loss(refine,teacher,head,cond,clean,labels,sampled,rounds,noise):
    quality=Quality(sampled,cond)
    adapter=Adapter(lambda:clean.square().mean(),lambda x:(head(x)-labels).square().mean(),
        lambda:sampled,lambda u:(head(u)-quality.action(u).detach()).square().mean(),lambda u:u)
    return compose(adapter,refine,teacher,lambda sampled,c:Quality(sampled,c),cond,clean,rounds=rounds,noise=noise)

def train(mode='reference',seed=0,steps=600,current=None):
    torch.manual_seed(seed)
    flow=real_flow()
    refine=TinyRefine(); teacher=copy.deepcopy(refine).requires_grad_(False)
    head=TinyHead()
    opt=torch.optim.Adam(list(refine.parameters())+list(head.parameters()),lr=.003)
    c,clean,labels=batch()
    for step in range(steps):
        sampled=flow.sample(len(c),c)
        noise=torch.randn_like(sampled)*.08
        r=1+step%3
        if mode=='reference':
            total,_,_=ref_loss(refine,teacher,head,c,clean,labels,sampled,r,noise)
        else:
            terms=current(refine,teacher,head,c,clean,labels,sampled)
            total=terms['total']
        opt.zero_grad(); total.backward(); opt.step()
        with torch.no_grad():
            for t,p in zip(teacher.parameters(),refine.parameters()): t.lerp_(p,.05)
    # Fresh actual Flow.sample: no post-hoc perturbation of held-out samples.
    c,clean,labels=batch(512)
    with torch.no_grad():
        u=flow.sample(len(c),c); quality=Quality(u,c); states=[u]
        for _ in range(6): states.append(refine(c,states[-1]))
        q=[quality(x).mean().item() for x in states]
        mode_error=[(x[...,0].sign()!=quality.route).float().mean().item() for x in states]
        g=gauge(head,states[:4],clean,labels,quality)
        ratio=q[3]/q[0]
    return dict(seed=seed,steps=steps,quality=q,mode_error=mode_error,
        correction_ratio=ratio,exposure=g), (flow,refine,head,c,u,quality)
