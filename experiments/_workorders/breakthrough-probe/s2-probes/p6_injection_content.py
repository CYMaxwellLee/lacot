"""Independent check of WHICH tensor each Move-1 row actually injected (harvest only verifies main P/Q)."""
import sys, os, json
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np
import common, runtime, harness_move1 as m1
from common import digest_array
art=json.load(open(SCR+'/e2e-move1/result.json'))
b=art['builder']
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
env=module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
c2x=env.unwrapped.ij_to_xy
def route(task,arm): return [c2x(c) for c in b['geometries'][str(task)]['routes'][arm]]
def H(u): return digest_array(u.detach().cpu().numpy())
# capture exact normalized input given to encode_u, compare against my own resampling
captured=[]
orig=module.encode_u
def spy(pts): captured.append(pts.detach().cpu().numpy().copy()); return orig(pts)
module.encode_u=spy
def my_resample(raw):
    p=np.asarray(raw,np.float64)
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1); L=np.r_[0,np.cumsum(seg)]
    keep=np.r_[True,np.diff(L)>0]; p,L=p[keep],L[keep]
    t=np.linspace(0,L[-1],128)
    r=np.stack([np.interp(t,L,p[:,i]) for i in range(2)],1)
    return ((r-module.MU_XY)/module.SD_XY).astype(np.float32)
cands={}
for t in range(1,6):
    for arm in [a for a in 'AB' if a in b['geometries'][str(t)]['routes']]:
        u=m1.encode_trajectory(module, route(t,arm)); cands[H(u)]=f'route-{arm}@task{t}'
        assert np.array_equal(captured[-1][0], my_resample(route(t,arm))), 'resample differs'
print('encode_trajectory resample == independent arc-length resample for all 10 routes: OK')
# duplicates robustness: N trace with stationary segments
rows=art['rows']
byk={(r['task'],r['episode'],r['arm']):r for r in rows}
for (t,e,arm),r in sorted(byk.items()):
    kind=None
    if arm in 'PQR':
        exp=None
        if r['injection_source']=='route-A': exp=f'route-A@task{t}'
        elif r['injection_source']=='route-B': exp=f'route-B@task{t}'
        elif r['injection_source']=='donor-route-A': exp=f"route-A@task{b['R_donor'][str(t)]['task']}"
        elif r['injection_source']=='self-rollout-u':
            n=byk[(t,e,'N')]; u=m1.encode_trajectory(module,n['xy']); cands[H(u)]=f'selfroll@{t},{e}'; exp=f'selfroll@{t},{e}'
        got=cands.get(r['encoded_u_sha256'],'<<UNKNOWN TENSOR>>')
        print(f'{(t,e,arm)} source={r["injection_source"]:14s} injected={got:18s} expected={exp:18s} {"OK" if got==exp else "MISMATCH"}')
# degenerate-trace behaviour
tr=np.array(byk[(1,10,'N')]['xy'],np.float64)
dups=np.sum(np.linalg.norm(np.diff(tr,axis=0),axis=1)==0)
print('N trace (1,10): points',len(tr),'zero-length steps',int(dups))
u1=m1.encode_trajectory(module,tr); print('encode finite:',bool(np.isfinite(u1.cpu().numpy()).all()))
# explicitly feed a trace with duplicated points
d=np.repeat(tr[:50],3,axis=0); u2=m1.encode_trajectory(module,d); u3=m1.encode_trajectory(module,tr[:50])
print('dup-expanded vs plain same u:', bool(np.array_equal(u2.cpu().numpy(),u3.cpu().numpy())))
env.close()
