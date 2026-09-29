"""u 編碼想像軌跡。Execute six actual mutant implementations against independent checks."""
import inspect
from toy import *
import fakes
from conformance import objective_suite,composition_suite
mutants=[
 ('total-missing-quality','compose','total=nf+anchor+terms.quality+lam_cons*terms.consistency+.25*exposure','total=nf+anchor+lam_cons*terms.consistency+.25*exposure',composition_suite),
 ('wrong-u_target','refinement_terms','u = sampled.detach()','u = clean.detach()',objective_suite),
 ('ignore-rounds','refinement_terms','for _ in range(rounds):','for _ in range(min(rounds,1)):',objective_suite),
 ('zero-noise','refinement_terms','v = u + noise.detach()','v = u + 0*noise.detach()',objective_suite),
 ('truncated-BPTT','refinement_terms','u = un\n','u = un.detach()\n',objective_suite),
 ('detach-cond','refinement_terms','validate(cond,clean,sampled,noise,rounds)','validate(cond,clean,sampled,noise,rounds)\n    cond = cond.detach()',objective_suite)]
for name,fn,old,new,suite in mutants:
    source=inspect.getsource(getattr(fakes,fn)); assert old in source
    ns=dict(vars(fakes)); exec(source.replace(old,new),ns)
    try: suite(ns[fn])
    except AssertionError as e: print(f'CAUGHT {name}: {e}')
    else: raise AssertionError('SURVIVED '+name)
print('PASS 6/6 mutation kill matrix')
