import sys, os, json
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='egl'
import numpy as np, ogbench, harvest, common
from builder import build
env=ogbench.make_env_and_datasets(common.ENV, env_only=True)
x2c,c2x=env.unwrapped.xy_to_ij, env.unwrapped.ij_to_xy
b=build()
def poly(cells,per=8):
    pts=[]
    P=[np.array(c2x(c),float) for c in cells]
    for a,bb in zip(P[:-1],P[1:]):
        for t in np.linspace(0,1,per,endpoint=False): pts.append(a+(bb-a)*t)
    pts.append(P[-1]); return np.array(pts)
ok=True
for task in ('4','5'):
    g=b['geometries'][task]
    for arm,oth in (('A','B'),('B','A')):
        own=poly(g['routes'][arm]); other=poly(g['routes'][oth])
        row=dict(arm=arm,xy=own.tolist())
        got_own=harvest.exclusive_crossing(row,g,x2c)
        row=dict(arm=arm,xy=other.tolist()); got_other=harvest.exclusive_crossing(row,g,x2c)
        # hybrid: start along own route to merge midpoint of own exclusive, then hop to the other route's exclusive segment
        k=g['exclusive_depths'][0]+2
        hyb=np.vstack([poly(g['routes'][arm][:k+1]), poly(g['routes'][oth][k:])])
        row=dict(arm=arm,xy=hyb.tolist()); got_hyb=harvest.exclusive_crossing(row,g,x2c)
        # ε-jitter near cell centers must not change membership (±1.5 inside a 4-unit cell)
        rng=np.random.default_rng(0); jit=own+rng.uniform(-1.5,1.5,own.shape)
        row=dict(arm=arm,xy=jit.tolist()); got_jit=harvest.exclusive_crossing(row,g,x2c)
        print(f'task{task} arm{arm}: own-route={got_own} (exp True) | other-route={got_other} (exp False) | hybrid(own then other)={got_hyb} (exp False) | jitter±1.5={got_jit} (info)')
        ok &= got_own and (not got_other) and (not got_hyb)
print('REAL-MAPPING CROSSING OK' if ok else 'REAL-MAPPING CROSSING FAIL')
env.close()
