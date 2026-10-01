import json, sys, os
os.environ['MUJOCO_GL']='egl'
import numpy as np
sys.dont_write_bytecode=True
import ogbench
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
a=json.load(open(SCR+'/e2e-move1/result.json'))
env=ogbench.make_env_and_datasets('pointmaze-large-stitch-v0', env_only=True)
x2c=env.unwrapped.xy_to_ij; c2x=env.unwrapped.ij_to_xy
b=a['builder']
for t in ('4','5'):
    g=b['geometries'][t]
    ex={k:set(map(tuple,v)) for k,v in g['exclusive'].items()}
    print('task',t,'branch',g['branch'],'merge',g['merge'],'exclusive A',g['exclusive']['A'],'B',g['exclusive']['B'])
    for arm in 'AB':
        pts=np.array(a['representation_evidence'][t]['decoded_'+arm])
        cells=[tuple(x2c(p)) for p in pts]
        comp=[cells[0]]
        for c in cells[1:]:
            if c!=comp[-1]: comp.append(c)
        tag=lambda c: 'A' if c in ex['A'] else ('B' if c in ex['B'] else '.')
        print(' decoded',arm,'first pt',pts[0].round(2),'last',pts[-1].round(2))
        print('   cell seq:',[ (c,tag(c)) for c in comp][:25])
        nA=sum(c in ex['A'] for c in cells); nB=sum(c in ex['B'] for c in cells)
        print('   points in A-excl:',nA,'B-excl:',nB, 'of',len(cells))
    # route centers
    for arm in 'AB':
        print(' route',arm,[tuple(c) for c in g['routes'][arm]][:8], 'xy of first 3', [np.round(c2x(c),2).tolist() for c in g['routes'][arm][:3]])
