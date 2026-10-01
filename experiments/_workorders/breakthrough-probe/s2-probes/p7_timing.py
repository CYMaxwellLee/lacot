import sys, os, json, time
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np, common, runtime, harvest
from builder import build
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
env=module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
b=build(); x2c,c2x=env.unwrapped.xy_to_ij, env.unwrapped.ij_to_xy
stuck=runtime.calibrated_stuck(20,0.2)
for arm in 'CAB':
    geom=b['geometries']['4']
    wp=None if arm=='C' else c2x(geom['w'][arm])
    plan=dict(task=4,episode=4,draw=64,arm=arm)
    t=time.time()
    row=runtime.run_draw(module,env,plan,move=3,waypoint=wp,cap_steps=96,stuck_check=stuck)
    dt=time.time()-t
    rd=harvest.recompute_move3(row,wp,geometry=geom,xy_to_cell=x2c)
    print(arm,'steps',row['observed_steps'],'reason',row['termination_reason'],'calls',len(row['flow_calls']),'secs',round(dt,2),'events',row['events'][:3],'cat',rd['category'],rd['invalid'],rd['w_step'],rd['g_step'])
env.close()
