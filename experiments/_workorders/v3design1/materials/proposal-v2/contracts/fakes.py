"""u 編碼想像軌跡。Small executable v2 specification, NOT production repair."""
import torch
import math
from api import Terms

def validate(cond, clean, sampled, noise, rounds):
    if type(rounds) is not int or rounds < 0:
        raise ValueError('rounds must be nonnegative integer')
    if clean.ndim != 3 or cond.ndim != 2 or min(clean.shape) < 1 or min(cond.shape) < 1 or cond.shape[0] != clean.shape[0]:
        raise ValueError('shape')
    if sampled.shape != clean.shape or noise.shape != clean.shape:
        raise ValueError('shape')
    if any(not x.is_floating_point() or x.device != clean.device for x in (cond,clean,sampled,noise)):
        raise ValueError('floating device')
    if sampled.dtype != clean.dtype or noise.dtype != clean.dtype:
        raise ValueError('latent dtype')
    # cond may be bf16. No value/finiteness sync or raise in the hot path.

def refinement_terms(refine, teacher, quality, cond, clean, sampled, *, rounds, noise):
    validate(cond,clean,sampled,noise,rounds)
    u = sampled.detach()
    v = u + noise.detach()
    q = clean.new_zeros(())
    c = clean.new_zeros(())
    states = [u]
    for _ in range(rounds):
        un = refine(cond,u)
        v = refine(cond,v)
        if un.shape != clean.shape or v.shape != clean.shape:
            raise ValueError('refine shape')
        # Frozen teacher parameters AND no graph through its targets.
        with torch.no_grad():
            target = teacher(cond,u.detach())
        q = q + (quality(un).mean()+.25*quality(v).mean())/1.25
        c = c + (un.float()-target.float()).square().mean()
        u = un
        states.append(u)
    return Terms(q/max(rounds,1),c/max(rounds,1),states)

def compose(adapter, refine, teacher, quality_factory, cond, clean, *, rounds, noise, lam_cons=.1,
            kernel=refinement_terms):
    if type(rounds) is not int or rounds < 0:
        raise ValueError('rounds')
    if not isinstance(lam_cons,(int,float)) or not math.isfinite(lam_cons) or lam_cons < 0:
        raise ValueError('lam_cons')
    nf, anchor = adapter.density(), adapter.anchor(adapter.features(clean))
    zero = nf.new_zeros(())
    if rounds == 0:
        terms=Terms(zero,zero,[])
        exposure=zero
    else:
        sampled=adapter.sample().detach()
        quality=quality_factory(sampled,cond)
        terms=kernel(refine,teacher,quality,cond,clean,sampled,rounds=rounds,noise=noise)
        # Head learns from the SAME decoded sampled plan; no gradients to refiner.
        exposure=torch.stack([adapter.exposure(u.detach()) for u in terms.states[1:]]).mean()
    total=nf+anchor+terms.quality+lam_cons*terms.consistency+.25*exposure
    return total, dict(l_nf=nf,l_act_anchor=anchor,l_plan_quality=terms.quality,
        l_cons=terms.consistency,l_head_exposure=exposure,l_act_refine=zero), terms.states
