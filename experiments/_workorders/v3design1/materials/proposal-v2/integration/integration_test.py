"""u 編碼想像軌跡。Design-independent F1/off-manifold gates, no consistency formula gate."""
import argparse,json,hashlib
import torch
from toy import *
from current_inline import load_inline
from acceptance import assess_run

def main():
    p=argparse.ArgumentParser(); p.add_argument('--reference',action='store_true')
    p.add_argument('--codebase',default=str(REPO)); args=p.parse_args()
    torch.manual_seed(1729)
    print('mode='+('PROPOSAL_REFERENCE' if args.reference else 'CURRENT_MAINLINE')+' CPU')
    current=load_inline(args.codebase)
    print(f'inline_source={current.source} lines={current.lines} total_line={current.total_line} sha256={hashlib.sha256(current.source.read_bytes()).hexdigest()}')
    failures=[]
    def check(name,ok,detail):
        print(('PASS ' if ok else 'FAIL ')+name+': '+str(detail))
        if not ok: failures.append(name)
    c,clean,labels=batch()
    sampled=clean.clone()
    sampled[...,0]=torch.tensor([1.,1.,-1.,-1.]).repeat(len(c)//4).reshape(-1,1)
    sampled[...,1]+=.8
    assert (sampled[...,0]*labels[...,0]).mean()==0  # exact factorial, not approximate independence
    r=TinyRefine(); t=copy.deepcopy(r).requires_grad_(False); h=TinyHead()
    def grad(labels):
        if args.reference: val=ref_loss(r,t,h,c,clean,labels,sampled,3,torch.ones_like(sampled)*.08)[0]
        else:
            ns=current(r,t,h,c,clean,labels,sampled); val=ns['total']
        return torch.cat([g.flatten() for g in torch.autograd.grad(val,tuple(r.parameters()))])
    delta=(grad(labels)-grad(-labels)).norm().item()
    check('inline-F1-label-isolation',delta<1e-7,f'gradient delta={delta:.6g}')
    # Both actual actor methods are also exercised, using the reusable v1 tiny layout.
    import sys
    sys.path.insert(0,str(Path(args.codebase)))
    from lacot.model import LaCoTActor,LaCoTActorState
    from types import SimpleNamespace
    class Head:
        def __call__(self,x): return x[:,-2:-1].reshape(-1,1,1)
        def nll(self,p,a): return (p-a).square().mean((1,2))
    for cls in (LaCoTActor,LaCoTActorState):
        obj=SimpleNamespace(k=1,d_model=2,flow=ReplayFlow(sampled),action_head=Head())
        def rounds(c,u,n):
            states=[u]
            for _ in range(n): states.append(r(c,states[-1]))
            return states
        obj.refine_rounds=rounds
        def g(a):
            if args.reference: val=ref_loss(r,t,h,c,clean,a,sampled,3,torch.zeros_like(sampled))[0]
            else: val=cls.losses_given(obj,c,clean,a,rounds=3)[0]
            return torch.cat([x.flatten() for x in torch.autograd.grad(val,tuple(r.parameters()))])
        d=(g(labels)-g(-labels)).norm().item()
        check(cls.__name__+'-F1-label-isolation',d<1e-7,f'gradient delta={d:.6g}')
    for seed in (0,1,2):
        stats,_=train('reference' if args.reference else 'current',seed,current=current)
        print('TRAIN '+json.dumps(stats,sort_keys=True))
        q=stats['quality']; m=stats['mode_error']; e=stats['exposure']
        check(f'off-manifold-seed{seed}',assess_run(stats)[0],
            f'R0/1/3={q[0]:.5f}/{q[1]:.5f}/{q[3]:.5f}, mode_error={m[3]:.5f}')
        check(f'exposure-seed{seed}',e['exposure/r3_mse']<.08,e)
    identity=dict(stats,quality=[q[0]]*7)
    check('identity-must-fail',not assess_run(identity)[0],'identity states measured at R0 cost for every round; correction_ratio=1')
    print(('FAIL' if failures else 'PASS')+f' integration: {len(failures)} failed checks')
    return bool(failures)
if __name__=='__main__': raise SystemExit(main())
