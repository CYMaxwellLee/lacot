import json, math, sys, os
os.environ['MUJOCO_GL']='egl'
import numpy as np, ogbench
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
a=json.load(open(SCR+'/e2e-move3/result.json')); b=a['builder']
env=ogbench.make_env_and_datasets('pointmaze-large-stitch-v0', env_only=True)
x2c,c2x=env.unwrapped.xy_to_ij, env.unwrapped.ij_to_xy
RHO=1.875; mism=0; n=0
for r in a['rows']:
    g=b['geometries'][str(r['task'])]
    xy=np.array(r['xy'],np.float64)
    if r['arm']=='C':
        w=None
    else:
        wp=np.array(c2x(g['w'][r['arm']]),np.float64)
        d=np.linalg.norm(xy-wp,axis=1); idx=np.flatnonzero(d<RHO); w=int(idx[0]) if len(idx) else None
    gs=r['goal_success']; gt=next((i for i,v in enumerate(gs) if v),None)
    # independent expected category
    if r['arm']=='C': cat='success' if gt is not None else 'failure'
    elif gt is not None and (w is None or gt<w): cat='d1'
    elif w is None: cat='d2'
    elif gt is None: cat='d3'
    else:
        vis={tuple(x2c(p)) for p in xy}
        own=set(map(tuple,g['exclusive'][r['arm']])); oth=set(map(tuple,g['exclusive']['B' if r['arm']=='A' else 'A']))
        cat='d4' if (vis&own and not (vis&oth)) else 'd5'
    # independent ledger expectation
    exp=[];t=0;chunk=4
    while t<r['observed_steps']:
        tgt='w' if (r['arm']!='C' and (w is None or t<w)) else 'g'
        exp.append({'step':t,'target':tgt}); t = w if (tgt=='w' and w is not None and t<w<t+chunk) else t+chunk
    ok = (cat==r['independent_readout']['category'] and r['flow_calls']==exp and w==r['independent_readout']['w_step'])
    n+=1; mism+= (not ok)
print('rows compared',n,'mismatches',mism)
# also: runtime-state events vs recompute for A/B rows (hit step)
bad=0
for r in a['rows']:
    ev=[e for e in r['events'] if e['event']=='reached']
    w=r['independent_readout']['w_step']
    if (w is None) != (not ev) or (ev and ev[0]['step']!=w): bad+=1
print('runtime reached-event vs trace-recompute w_step mismatches:',bad)
# duplicates in keys & sizes
ks=[(r['task'],r['episode'],r['arm'],r['draw']) for r in a['rows']]; print('unique keys',len(set(ks)),'of',len(ks))
# stuck rows: when does stuck fire?
st=[(r['task'],r['draw'],r['observed_steps']) for r in a['rows'] if r['termination_reason']=='stuck']
print('stuck steps (task2 A):', sorted(set(s for _,_,s in st)))
xs=np.array([r['xy'][:60] for r in a['rows'] if r['termination_reason']=='stuck'][0]); print('task2 A first rows xy drift over first 40 steps:', np.linalg.norm(xs[40]-xs[0]).round(3))
env.close()
