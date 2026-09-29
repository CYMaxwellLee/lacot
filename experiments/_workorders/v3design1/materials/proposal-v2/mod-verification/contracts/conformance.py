"""u 編碼想像軌跡。Independent recurrence and gradient oracles."""
import torch
from torch import nn
from api import Adapter
from fakes import refinement_terms

def fixture():
    clean=torch.tensor([1.,-1.],dtype=torch.float64).reshape(2,1,1).requires_grad_()
    sample=torch.tensor([2.,-.4],dtype=torch.float64).reshape(2,1,1).requires_grad_()
    noise=torch.tensor([.13,-.07],dtype=torch.float64).reshape(2,1,1).requires_grad_()
    cond=torch.tensor([[.3],[-.2]],dtype=torch.float64,requires_grad=True)
    theta=torch.tensor([.6,.2],dtype=torch.float64,requires_grad=True)
    refine=lambda c,u:theta[0]*u+theta[1]*c[:,None]
    teacher=lambda c,u:.8*u+.1*c[:,None]
    quality=lambda u:(u-1.4).square().mean((1,2))
    return clean,sample,noise,cond,theta,refine,teacher,quality

def objective_suite(kernel):
    cl,s,n,c,p,f,t,q=fixture()
    for rounds in (0,1,3):
        got=kernel(f,t,q,c,cl,s,rounds=rounds,noise=n)
        u=s.detach(); v=u+n.detach(); eq=cl.new_zeros(()); ec=cl.new_zeros(())
        for _ in range(rounds):
            un=p[0]*u+p[1]*c[:,None]; v=p[0]*v+p[1]*c[:,None]
            eq=eq+(((un-1.4)**2).mean()+.25*((v-1.4)**2).mean())/1.25
            ec=ec+(un-(.8*u+.1*c[:,None]).detach()).square().mean()
            u=un
        eq/=max(rounds,1); ec/=max(rounds,1)
        assert torch.allclose(got.quality,eq,atol=1e-10),'quality recurrence/target/noise/rounds'
        assert torch.allclose(got.consistency,ec,atol=1e-7),'EMA formula'
        assert len(got.states)==rounds+1,'rounds count'
        if rounds:
            actual=torch.autograd.grad(got.quality,(p,c,cl,s,n),allow_unused=True,retain_graph=True)
            expected=torch.autograd.grad(eq,(p,c),retain_graph=True)
            assert torch.allclose(actual[0],expected[0],atol=1e-10),'full BPTT gradient'
            assert actual[1] is not None and torch.allclose(actual[1],expected[1]),'cond gradient'
            assert all(g is None for g in actual[2:]),'stop sample/clean/noise'
    mixed=kernel(f,t,q,c.bfloat16(),cl,s,rounds=1,noise=n)
    assert torch.isfinite(mixed.quality),'mixed cond dtype allowed'
    # Actual autocast Linear output may be bf16, not latent input dtype.
    layer=nn.Linear(1,1).float()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        kernel(lambda c,u:layer(u),lambda c,u:u,q,c.float(),cl.float(),s.float(),rounds=1,noise=n.float())
    bad=s.detach().clone(); bad[0]=float('nan')
    got=kernel(f,t,q,c,cl,bad,rounds=1,noise=n)
    assert not torch.isfinite(got.quality),'nonfinite propagates to scaler, no raise'
    for r in (-1,True,1.5):
        try: kernel(f,t,q,c,cl,s,rounds=r,noise=n)
        except ValueError: pass
        else: raise AssertionError('rounds validation')
    print('PASS objective: C1 mixed dtype/autocast/nonfinite; target/noise/R0/1/3/full BPTT/cond')

def composition_suite(compose):
    cl,s,n,c,p,f,t,q=fixture()
    a=torch.tensor(2.,requires_grad=True,dtype=torch.float64)
    h=torch.tensor(.7,requires_grad=True,dtype=torch.float64)
    calls=[]
    adapter=Adapter(lambda:a*a,lambda features:h*h,lambda:(calls.append('sample') or s),
        lambda u:h*h+u.square().mean(),lambda u:u)
    total,logs,states=compose(adapter,f,t,lambda sampled,cond:q,c,cl,rounds=3,noise=n,lam_cons=.4)
    expected=logs['l_nf']+logs['l_act_anchor']+logs['l_plan_quality']+.4*logs['l_cons']+.25*logs['l_head_exposure']
    assert torch.allclose(total,expected),'total includes all weighted terms'
    ga,gh=torch.autograd.grad(total,(a,h),retain_graph=True)
    assert torch.allclose(ga,2*a),'density gradient'
    assert torch.allclose(gh,2*h*1.25),'anchor-gradient plus separately attributed exposure'
    assert calls==['sample'],'one real sample'
    calls.clear()
    total,logs,states=compose(adapter,f,t,lambda sampled,cond:q,c,cl,rounds=0,noise=n)
    assert torch.allclose(total,a*a+h*h) and not calls and not states,'R2 zero semantics'
    assert logs['l_cons']==logs['l_act_refine']==logs['l_plan_quality']==logs['l_head_exposure']==0
    print('PASS composition: total, anchor-gradient attribution, clean/sample separation, R2')

def oracle_suite(Quality,gauge):
    cl=torch.tensor([[[1.,0.]], [[-1.,0.]]]); s=cl+torch.tensor([[[.4,.8]]])
    c=torch.zeros(2,1); q=Quality(s,c)
    assert q(cl).max()<1e-9 and q(s).min()>.1
    assert q(-cl).min()>3,'route swap detectable'
    u=s.clone().requires_grad_(); assert torch.autograd.grad(q(u).sum(),u)[0].norm()>0
    good=lambda u:u[...,:1]; bad=lambda u:torch.zeros_like(u[...,:1])
    a=gauge(good,[s],cl,cl[...,:1],q); b=gauge(bad,[s],cl,cl[...,:1],q)
    assert a['exposure/r0_mse']==0 and b['exposure/r0_mse']>.1
    assert not any(isinstance(v,torch.Tensor) and v.requires_grad for v in b.values())
    # Perfect on clean, deliberately unreadable outside clean support.
    exposed=lambda u:torch.where(u[...,1:].abs()<.01,u[...,:1],torch.zeros_like(u[...,:1]))
    g=gauge(exposed,[s],cl,cl[...,:1],q)
    assert g['exposure/anchor_mse']==0 and g['exposure/r0_gap']>.1,'exposure bias bridge'
    print('PASS oracle: same-plan labels, frozen differentiable decoder, no-grad exposure gauge')
