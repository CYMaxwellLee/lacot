import sys, os, json
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np, common, runtime, harness_move3 as m3
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
env=module.ogbench.make_env_and_datasets(common.ENV, env_only=True)
torch=module.torch
stash={}
orig=m3.make_collector
def spy(*a,**k):
    c=orig(*a,**k); stash['c']=c; return c
m3.make_collector=spy
for (task,ep,draw) in [(4,4,64),(5,5,67),(1,10,64)]:
    plan=dict(task=task,episode=ep,draw=draw,arm='N')
    # run only a handful of steps by using cap through horizon? use a tiny episode: run 1 chunk by exploiting move=1 N path with full run is long; instead stop after first get_u via exception
    class Stop(Exception): pass
    cc={}
    real_get=None
    def run_first_u():
        obs,info=common.reset_env(env,task,ep)
        goal=np.asarray(info['goal'])
        module._GRAD_CACHE['u']=None
        runtime.reset_policy(module,task,ep,draw)
        col=orig(task,ep,draw,obs,goal)
        with torch.no_grad():
            cond=runtime.condition(module,obs,goal)
            u=col.get_u(lambda: module.sample_plan(1,cond,None))
        return col.first_u, common.digest_array(u.detach().cpu().numpy())
    fu, hu = run_first_u()
    # independent: global torch seed = stream_seed, nothing else
    obs,info=common.reset_env(env,task,ep); goal=np.asarray(info['goal'])
    torch.manual_seed(7*task+ep+1000003*draw)
    with torch.no_grad():
        s=module.normstate(obs); g=module.normstate(goal)
        cond=module.condvec(s,g)
        u2=module.flow.sample(1,cond)
    h2=common.digest_array(u2.detach().cpu().numpy())
    # legacy (wrong) branches for contrast
    torch.manual_seed(7*task+ep)
    with torch.no_grad(): u3=module.flow.sample(1,cond)
    print((task,ep,draw),'harness-collector first_u == independent stream-seeded flow.sample:', fu==h2, '| differs from legacy torch.manual_seed(7*task+ep):', fu!=common.digest_array(u3.detach().cpu().numpy()))
env.close()
