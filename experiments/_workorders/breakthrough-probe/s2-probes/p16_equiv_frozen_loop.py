"""Protocol equivalence: harness N arm (run_draw, Move 1) vs the FROZEN source's own oracle rollout path
(policy_chunk with ORACLE_COLLECTOR A-arm), same seeds, CPU, first 160 steps."""
import sys, os
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np, common, runtime, harness_move3 as m3
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
torch=module.torch
env=module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
LIM=160
# frozen _bon_plan is defined AFTER the import boundary (line 2738); BON_N<=1 branch == sample_plan verbatim -> stub for the reference path only
module._bon_plan=lambda n,cond,anc,s,g: module.sample_plan(n,cond,anc,s,g)
runtime.H=LIM; m3.H=LIM
# --- harness side: record actions given to env.step
rec_h=[]
orig_step=env.step
def spy(a): rec_h.append(np.array(a,copy=True)); return orig_step(a)
env.step=spy
allok=True
sys.path.insert(0,str(common.REPO))
from experiments._workorders.ucontrast1.collector import Collector
for (task,ep,draw) in [(4,4,64),(1,10,64),(5,5,71)]:
    rec_h.clear()
    row=runtime.run_draw(module,env,dict(task=task,episode=ep,draw=draw,arm='N'),move=1)
    h_actions=list(rec_h); h_xy=np.array(row['xy'],np.float32)
    # --- frozen side: replicate rollout() oracle branch exactly (lines ~3030-3086 of the frozen source)
    env.step=orig_step
    col=Collector('A',0.,16); module.ORACLE_COLLECTOR=col
    seed=1000*task+ep
    np.random.seed(seed)
    try: env.action_space.seed(seed)
    except Exception: pass
    obs,info=env.reset(seed=seed,options={'task_id':task,'render_goal':False}); goal=info['goal']
    torch.manual_seed(7*task+ep)            # historical per-episode seed (overridden by begin)
    col.begin(task,ep,draw,obs,goal)
    module._reset_grad_cache(); module._reseed_shuf(1000*task+ep)
    f_actions=[]; f_xy=[np.asarray(obs[:2],np.float32)]; steps=0; success=False
    while steps<LIM and not success:
        for a in module.policy_chunk(obs,goal,0,True):
            a=col.action(a)                 # A-arm: identity
            f_actions.append(np.array(a,copy=True))
            obs,rew,term,trunc,info=env.step(a); steps+=1; f_xy.append(np.asarray(obs[:2],np.float32))
            if info.get('success'): success=True
            if success or term or trunc or steps>=LIM: break
    module.ORACLE_COLLECTOR=None
    env.step=spy
    same_actions = len(h_actions)==len(f_actions) and all(np.array_equal(x,y) for x,y in zip(h_actions,f_actions))
    same_xy = np.array_equal(h_xy,np.array(f_xy,np.float32))
    print((task,ep,draw),'steps',row['observed_steps'],'harness==frozen actions:',same_actions,'| xy traces identical:',same_xy,'| first_u equal:', col.first_u is not None)
    allok &= same_actions and same_xy
print('PROTOCOL EQUIVALENCE (harness N arm == frozen oracle A-arm path):', allok)
env.close()
