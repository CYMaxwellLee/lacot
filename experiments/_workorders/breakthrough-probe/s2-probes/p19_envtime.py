import os, time, numpy as np
os.environ['MUJOCO_GL']='egl'
import ogbench
env=ogbench.make_env_and_datasets('pointmaze-large-stitch-v0', env_only=True)
env.reset(seed=1004,options={'task_id':4,'render_goal':False})
rng=np.random.default_rng(0); t=time.time()
for _ in range(2000): env.step(rng.uniform(-1,1,2).astype(np.float32))
dt=time.time()-t; print(f'env.step: {dt/2000*1000:.3f} ms/step -> {dt/2000*1000:.2f} s per 1000 steps')
