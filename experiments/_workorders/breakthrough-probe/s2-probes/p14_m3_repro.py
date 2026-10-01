import sys, os, json
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np, common, runtime
from builder import build
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
env=module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
b=build(); c2x=env.unwrapped.ij_to_xy
stuck=runtime.calibrated_stuck(20,0.2)
def go(task,ep,draw,arm):
    g=b['geometries'][str(task)]
    wp=None if arm=='C' else c2x(g['w'][arm])
    return runtime.run_draw(module,env,dict(task=task,episode=ep,draw=draw,arm=arm),move=3,waypoint=wp,cap_steps=96,stuck_check=stuck)
a1=go(4,4,64,'A'); other=go(5,5,70,'C'); a2=go(4,4,64,'A')
c1=go(4,4,64,'C'); c2=go(4,4,64,'C')
eq=lambda x,y: x['xy']==y['xy'] and x['flow_calls']==y['flow_calls'] and x['rng_before']==y['rng_before'] and x['rng_after']==y['rng_after']
print('A rerun after an intervening other-geometry rollout identical (xy, ledger, RNG snapshots):', eq(a1,a2))
print('C rerun identical:', eq(c1,c2))
# pairing: A and C share the same action-noise stream (same stream_seed) -> noise states equal at start
print('A/C share initial noise state:', a1['rng_before']['noise']==c1['rng_before']['noise'], '| share initial reset (fingerprints):', a1['initial_sha256']==c1['initial_sha256'])
# different draw -> different noise stream
d=go(4,4,65,'C'); print('different draw -> different noise state:', d['rng_before']['noise']!=c1['rng_before']['noise'])
env.close()
