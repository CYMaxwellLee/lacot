import sys, os, json, time, resource
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='egl'
import harvest, common, ogbench
from builder import build
t=time.time(); a=json.load(open(SCR+'/e2e-move3/result.json')); tl=time.time()-t
fps={tuple(map(int,k.split(','))):tuple(v) for k,v in json.load(open(SCR+'/n64_fingerprints.json')).items()}
env=ogbench.make_env_and_datasets(common.ENV, env_only=True)
b=build()
t=time.time(); out=harvest.validate_artifact(a,b,fps,env.unwrapped.ij_to_xy,env.unwrapped.xy_to_ij); tv=time.time()-t
t=time.time(); data=common.canonical(a); tc=time.time()-t
print(f'128 rows: json.load {tl:.2f}s validate_artifact {tv:.2f}s canonical() {tc:.2f}s size {len(data)/1e6:.1f}MB  -> x10 extrapolation: load {tl*10:.0f}s validate {tv*10:.0f}s canonical {tc*10:.0f}s size {len(data)*10/1e6:.0f}MB')
print('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
# size breakdown per row
r=a['rows'][0]; 
import collections
sizes={k:len(json.dumps(v)) for k,v in r.items()}
print(sorted(sizes.items(), key=lambda kv:-kv[1])[:6])
print('rng_before parts', {k:len(json.dumps(v)) for k,v in r['rng_before'].items()})
env.close()
