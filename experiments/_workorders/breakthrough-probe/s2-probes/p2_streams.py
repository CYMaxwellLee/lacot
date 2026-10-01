import json, sys
sys.path.insert(0,'/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe')
sys.dont_write_bytecode=True
import common, harness_move3 as m3, harness_move1 as m1
b=json.load(open('/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/builder.json'))
rows=json.load(open('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/gate0b-result.json'))['rows']
allq=[(r['task'],r['episode']) for r in rows]
# 1) formula vs independent
for (t,e) in allq:
    for d in range(64,80):
        s=common.seeds(t,e,d)
        base=7*t+e
        assert s['base_seed']==base and s['stream_seed']==base+1000003*d, (t,e,d,s)
        assert s['flow_seed']==s['stream_seed'] and s['noise_seed']==s['stream_seed'] and s['torch_seed']==s['stream_seed']
        assert s['env_seed']==1000*t+e
        si=common.seeds(t,e,d,injected=True)
        assert si['flow_seed'] is None and si['stream_seed']==s['stream_seed']
# draw guard
for bad in (0,63,80,-1,1000):
    try: common.seeds(4,4,bad); print('NO GUARD for',bad)
    except ValueError: pass
# 2) intersection with the old 0..63 for ALL 50 questions (not only main set), both flow-seed families
old_stream={(7*t+e+1000003*d) for (t,e) in allq for d in range(64)}
old_flow_B={7*t+e for (t,e) in allq}          # B-type collector flow seeds = base
new_stream={(7*t+e+1000003*d) for (t,e) in allq for d in range(64,80)}
print('old stream n',len(old_stream),'new n',len(new_stream),'inter',len(old_stream&new_stream),'inter with B base',len(old_flow_B&new_stream))
# 3) collisions across questions inside the new plan (contract says expected: (4,30)/(5,23) base=58)
from collections import defaultdict
byseed=defaultdict(set)
for (t,e) in allq: byseed[7*t+e].add((t,e))
print('base collisions among all 50:',{k:sorted(v) for k,v in byseed.items() if len(v)>1})
# restrict to question sets that run
plan3=m3.rollout_plan(b); plan1=m1.rollout_plan(b)
print('len plan3',len(plan3),'len plan1',len(plan1))
q3=sorted({(p['task'],p['episode']) for p in plan3}); 
col=defaultdict(set)
for (t,e) in q3: col[7*t+e].add((t,e))
print('collisions inside Move3 question set:',{k:sorted(v) for k,v in col.items() if len(v)>1})
# 4) plan content checks
from collections import Counter
print(Counter((p['task'],p['arm']) for p in plan3))
print(Counter((p['group'],p['arm'],p['kind']) for p in plan1))
assert all(p['draw'] in range(64,80) for p in plan3)
assert all(p['draw']==64 for p in plan1)
# the plan's embedded seeds equal fresh computation
for p in plan3: assert p['seeds']==common.seeds(p['task'],p['episode'],p['draw'])
for p in plan1: assert p['seeds']==common.seeds(p['task'],p['episode'],64,p['arm']!='N')
print('plan seeds consistent')
# 5) Move3 plan: gate questions absent? control only A,C?
assert not any(p['task'] in (1,3) for p in plan3)
print('task2 arms',sorted({p['arm'] for p in plan3 if p['task']==2}))
# 6) Move1 plan: R donors / sources
src=Counter((p['group'],p['arm'],p['source']) for p in plan1)
print(src)
