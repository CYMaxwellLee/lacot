import sys, os, json, copy
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a'+'/scratchpad'
MIR=SCR+'/mirror/lacot/experiments/_workorders/breakthrough-probe'
sys.path.insert(0,MIR); sys.dont_write_bytecode=True
os.environ['MUJOCO_GL']='egl'
import numpy as np
import harvest, common, ogbench
from builder import build
fps={tuple(map(int,k.split(','))):tuple(v) for k,v in json.load(open(SCR+'/n64_fingerprints.json')).items()}
env=ogbench.make_env_and_datasets(common.ENV, env_only=True)
x2c,c2x=env.unwrapped.xy_to_ij, env.unwrapped.ij_to_xy
b=build()
art=json.load(open(SCR+'/e2e-move1/result.json'))
def validate(a): return harvest.validate_artifact(a,b,fps,c2x,x2c)
base=validate(copy.deepcopy(art))
print('baseline validates OK; gate', base['gate'])
def case(name, mut, expect=harvest.Rejected):
    a=copy.deepcopy(art)
    try:
        mut(a)
    except Exception as e:
        print(f'{name:55s} MUTATION-ERROR {e!r}'); return
    try:
        validate(a); print(f'{name:55s} ACCEPTED  <-- (expected reject)' if expect else f'{name:55s} accepted (as expected)')
    except expect as e:
        print(f'{name:55s} rejected: {str(e)[:70]}')
    except Exception as e:
        print(f'{name:55s} OTHER-EXC {type(e).__name__}: {str(e)[:80]}')
row=lambda a,t,arm,kind='rollout',ep=None: next(r for r in (a['rows'] if kind=='rollout' else a['conformance_rows']) if r['task']==t and r['arm']==arm and (ep is None or r['episode']==ep))
case('drop gate P row', lambda a: a['rows'].remove(row(a,1,'P')))
case('drop main R row', lambda a: a['rows'].remove(row(a,4,'R')))
case('duplicate a row', lambda a: a['rows'].append(copy.deepcopy(a['rows'][0])))
case('wrong seed on N', lambda a: row(a,4,'N')['seeds'].__setitem__('flow_seed',5))
case('wrong seed: injected row with flow seed', lambda a: row(a,4,'P')['seeds'].__setitem__('flow_seed', row(a,4,'P')['seeds']['stream_seed']))
case('reset fingerprint swapped', lambda a: row(a,4,'P').__setitem__('initial_sha256','0'*64))
case('xy tamper w/o hash fix', lambda a: row(a,4,'P')['xy'][7].__setitem__(0, 99.))
def tamper_consistent(a):
    r=row(a,4,'P'); r['xy'][500][0]+=1.0
    r['trace_sha256']=common.digest_array(np.asarray(r['xy'],np.float32))
case('xy tamper WITH consistent hash (undetectable?)', tamper_consistent)
case('success flipped w/o reason change (truncated+succ)', lambda a: row(a,4,'P')['goal_success'].__setitem__(-1,True))
case('injection_source mislabel', lambda a: row(a,4,'P').__setitem__('injection_source','route-B'))
case('R donor tampered', lambda a: row(a,4,'R').__setitem__('donor',{'task':2,'episode':1,'route':'A'}))
case('P overwritten by flow (flow_calls non-empty)', lambda a: row(a,4,'P').__setitem__('flow_calls',[{'step':0,'target':'g'}]))
case('head hash mismatch mid-episode', lambda a: row(a,4,'Q')['head_u_hashes'].__setitem__(100,'f'*64))
case('head hashes truncated', lambda a: row(a,4,'Q').__setitem__('head_u_hashes', row(a,4,'Q')['head_u_hashes'][:-1]))
case('P encoded_u hash != representation u_A', lambda a: (row(a,4,'P').__setitem__('encoded_u_sha256','a'*64), row(a,4,'P').__setitem__('head_u_hashes',['a'*64]*250)))
case('skipped entry removed', lambda a: a['skipped'].pop())
case('skipped entry fabricated for a run row', lambda a: a['skipped'].append(dict(a['skipped'][0], task=4, episode=4, arm='P', kind='rollout')))
case('receipt code hash tampered', lambda a: a['receipt']['code_sha256'].__setitem__('rules.py','0'*64))
case('receipt smoke flag flipped', lambda a: a['receipt'].__setitem__('smoke', False))
case('builder altered', lambda a: a['builder']['questions']['main'].pop())
case('representation evidence swapped (u_A<->u_B)', lambda a: a['representation_evidence']['4'].update(u_A=a['representation_evidence']['4']['u_B'], u_B=a['representation_evidence']['4']['u_A']))
case('noise_band set huge so rep fails but rows exist', lambda a: a['receipt']['calibration']['noise_band'].__setitem__('4', 1e9))
case('gate P source_trace hash wrong', lambda a: row(a,1,'P').__setitem__('source_trace_sha256','x'))
case('rerun xy altered (conformance must be False not reject?)', lambda a: (row(a,4,'P','rep').__setitem__('xy', [[v[0]+0.0, v[1]] for v in row(a,4,'P','rep')['xy']]), row(a,4,'P','rep')['xy'][3].__setitem__(0, row(a,4,'P','rep')['xy'][3][0]+1e-3), row(a,4,'P','rep').__setitem__('trace_sha256', common.digest_array(np.asarray(row(a,4,'P','rep')['xy'],np.float32)))), expect=None)
env.close()
