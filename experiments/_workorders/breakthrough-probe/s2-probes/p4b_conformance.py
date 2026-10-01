import sys, os, json, copy
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a'+'/scratchpad'
sys.path.insert(0,SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'); sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='egl'
import numpy as np, harvest, common, ogbench
from builder import build
import rules
fps={tuple(map(int,k.split(','))):tuple(v) for k,v in json.load(open(SCR+'/n64_fingerprints.json')).items()}
env=ogbench.make_env_and_datasets(common.ENV, env_only=True)
b=build(); art=json.load(open(SCR+'/e2e-move1/result.json'))
a=copy.deepcopy(art)
r=next(r for r in a['conformance_rows'] if r['arm']=='P')
r['xy'][3][0]+=1e-3
r['trace_sha256']=common.digest_array(np.asarray(r['xy'],np.float32))
out=harvest.validate_artifact(a,b,fps,env.unwrapped.ij_to_xy,env.unwrapped.xy_to_ij)
print('conformance after tamper:', out['conformance'])
print('does anything downstream degrade the exact-zero language? content_sensitivity not called by harvest ->', 'content_sensitivity' in open(SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe/harvest.py').read())
print('rules.content_sensitivity(n_diff=1, {same_u_rerun: False}) =', rules.content_sensitivity(1, {'same_u_rerun': False}))
import re
for name in ['content_sensitivity','exits(','e4(','e2(','ceiling(','adsorption(','exact_one_sided','behavior_gate','paired_response','question_grid']:
    hits=[]
    for fn in ['harvest.py','runtime.py','harness_move1.py','harness_move3.py','smoke.py']:
        t=open(SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe/'+fn).read()
        if re.search(r'\b'+re.escape(name), t): hits.append(fn)
    print(f'{name:22s} referenced in:', hits)
env.close()
