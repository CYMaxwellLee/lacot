import json, torch, numpy as np, random
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
for name in ('e2e-move3','e2e-move1'):
    a=json.load(open(f'{SCR}/{name}/result.json')); bad=0; n=0
    for r in a['rows']:
        s=r['seeds']; stream=s['stream_seed']
        torch.manual_seed(stream); ref_cpu=torch.get_rng_state().tolist()
        g=torch.Generator(device='cpu').manual_seed(stream); ref_noise=g.get_state().tolist()
        b=r['rng_before']
        ok_cpu=(b['torch_cpu']==ref_cpu)
        ok_noise=('noise' not in b) or (b['noise']==ref_noise)
        # numpy global state: seeded with env seed by reset_env
        np.random.seed(s['env_seed']); st=np.random.get_state(); ref_np=[st[0],st[1].tolist(),*st[2:]]
        n+=1; bad+= not (ok_cpu and ok_noise)
    print(name,'rows',n,'rng_before torch_cpu/noise equal recomputed-from-seed states; mismatches:',bad)
