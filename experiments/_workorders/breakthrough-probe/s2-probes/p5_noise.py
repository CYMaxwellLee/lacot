import sys, os
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
import numpy as np, torch
import common, harness_move3 as m3
rng=np.random.default_rng(0)
ok=True
for (task,ep,draw) in [(4,4,64),(5,5,79),(2,1,70)]:
    col=m3.make_collector(task,ep,draw,np.zeros(2),np.ones(2))
    g=torch.Generator(device='cpu'); g.manual_seed(7*task+ep+1000003*draw)
    for step in range(50):
        a=rng.uniform(-1.5,1.5,size=2).astype(np.float32)
        got=m3.noisy_action(col,a)
        eps=torch.randn((2,),generator=g).numpy()
        exp=np.clip(np.clip(a,-1.,1.).astype(np.float32)+np.float32(0.05)*eps,-1.,1.).astype(np.float32)
        if not (got.dtype==np.float32 and np.array_equal(got,exp)):
            ok=False; print('MISMATCH',task,ep,draw,step,got,exp)
    # flow seed branch
    print((task,ep,draw),'collector.flow_seed',col.flow_seed,'== stream',7*task+ep+1000003*draw, col.flow_seed==7*task+ep+1000003*draw, 'arm',col.arm,'sigma',col.sigma)
print('noise-equivalence', ok)
# encode_trajectory robustness is checked elsewhere (needs model)
