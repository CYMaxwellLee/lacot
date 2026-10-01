import sys, os
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl')
import numpy as np, hashlib, common, runtime, harness_move3 as m3
module=runtime.load_frozen('/home/cymaxwelllee/data/ogbench')
col=m3.make_collector(4,4,64,np.zeros(2),np.ones(2))
f=sys.modules['experiments._workorders.ucontrast1.collector'].__file__
print('collector module file:', f)
print('sha256:', hashlib.sha256(open(f,'rb').read()).hexdigest()[:16], 'equals common.COLLECTOR sha:', hashlib.sha256(open(f,'rb').read()).hexdigest()==common.file_sha(common.COLLECTOR))
print('os.path realpath of imported collector:', os.path.realpath(f))
print('sys.path head after load:', sys.path[:3])
