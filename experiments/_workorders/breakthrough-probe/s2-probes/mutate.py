"""S2 mutation harness: copy the audited dir per mutation, apply ONE textual change, run selftest, record failures."""
import os, re, shutil, subprocess, sys, json, concurrent.futures as cf
SCR='/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad'
SRC='/home/cymaxwelllee/Projects/lacot'
ORIG=SRC+'/experiments/_workorders/breakthrough-probe'
MUT=[]  # (id, file, old, new, desc)
def m(id, file, old, new, desc): MUT.append((id,file,old,new,desc))

# ---- common / seeds / reset
m('C1','common.py',"    if actual != PIN:\n        raise Blocked(", "    if False:\n        raise Blocked(", 'verify_source never raises')
m('C2','common.py',"if draw not in range(64, 80):","if draw not in range(63, 80):",'draw guard admits 63 (old stream)')
m('C3','common.py',"1_000_003 * draw","1_000_033 * draw",'stream multiplier changed')
m('C4','common.py',"base_seed=7 * task + episode,","base_seed=7 * task + episode + 1,",'base_seed off by one (reported only)')
m('C5','common.py',"flow_seed=None if injected else stream","flow_seed=stream",'injected rows get flow seed')
m('C6','common.py',"noise_seed=stream, python_seed=stream, torch_seed=stream","noise_seed=stream+1, python_seed=stream, torch_seed=stream",'noise seed differs from stream')
m('C7','common.py',"    np.random.seed(seed)\n    try:\n        env.action_space.seed(seed)","    try:\n        env.action_space.seed(seed)\n    except Exception:\n        pass\n    np.random.seed(seed)\n    try:\n        pass",'reset order swapped')
m('C8','common.py',"    seed = 1000 * task + episode\n    np.random.seed(seed)","    seed = 1000 * task + episode + 1\n    np.random.seed(seed)",'reset seed off by one')
# ---- rules.classify
m('R1','rules.py',"(w_step is None or g_step < w_step)","(w_step is None or g_step <= w_step)",'classify: equal-step counts as bypass')
m('R2','rules.py',"if calls > ceil(H/chunk_steps)+2:","if calls >= ceil(H/chunk_steps)+2:",'classify: budget boundary 252 invalid')
m('R3','rules.py',"return 'd4' if crossing is True else 'd5'","return 'd4' if crossing is not None else 'd5'",'classify: crossing False -> d4')
m('R4','rules.py',"insufficient=valid < 8","insufficient=valid < 7",'insufficient threshold 7')
m('R5','rules.py',"w_unreliable=counts['d2'] > valid/2","w_unreliable=counts['d2'] >= valid/2",'w_unreliable >= 50%')
m('R6','rules.py',"    if stats['C']['counts'].get('success', 0):\n        grid = C_RESCUE\n    elif any(","    if False:\n        grid = C_RESCUE\n    elif any(",'C success no longer first')
m('R7','rules.py',"if len(winners) == 1 else AMBIGUOUS)","if True else AMBIGUOUS)",'tie not ambiguous')
m('R8','rules.py',"{'d2': INSUFFICIENT, 'd3': DEEP, 'd5': UNVERIFIED}[winners[0]]","{'d2': INSUFFICIENT, 'd3': DEEP, 'd5': DEEP}[winners[0]]",'d5 dominance folded into DEEP')
m('R9','rules.py',"grid = BOTH if ar and br else A_ONLY if ar else B_ONLY","grid = BOTH if ar and br else B_ONLY if ar else A_ONLY",'A_only/B_only swapped')
m('R10','rules.py',"elif p_reproduced < ceil(qualified*3/4):","elif p_reproduced < int(qualified*3/4):",'behavior gate floor instead of ceil')
m('R11','rules.py',"elif r_reproduced > 2:","elif r_reproduced >= 2:",'R threshold >=2')
m('R12','rules.py',"    if qualified < 5:","    if qualified < 4:",'gate min qualified 4')
m('R13','rules.py',"return dict(numerator=count, denominator=expected, stopped=count > expected/2)","return dict(numerator=count, denominator=expected, stopped=count >= expected/2)",'E2 >= 50%')
m('R14','rules.py',"{4: 11, 5: 5}[task]","{4: 11, 5: 4}[task]",'E3 task5 threshold 4')
m('R15','rules.py',"ordered[0][1] -\n                (ordered[1][1] if len(ordered) > 1 else 0) >= 2","ordered[0][1] -\n                (ordered[1][1] if len(ordered) > 1 else 0) >= 1",'E4 dominance margin 1')
m('R16','rules.py',"switch >= 5 and anti == 0","switch >= 4 and anti == 0",'follower threshold 4')
m('R17','rules.py',"threshold = ceil(len(pairs)/2)","threshold = len(pairs)//2",'adsorption floor threshold')
m('R18','rules.py',"axis1_deprioritized=forward < 3","axis1_deprioritized=forward < 2",'stop-loss <2')
m('R19','rules.py',"elif dominant in (BOTH, B_ONLY) and p < .05:","elif dominant in (BOTH, B_ONLY) and p < .5:",'p threshold .5')
m('R20','rules.py',"    if not delivery_ok:\n        return dict(exit='E0'","    if False:\n        return dict(exit='E0'",'E0 disabled')
# ---- harness_move3 state machine
m('S1','harness_move3.py',"hit = self.stage == 'w' and dist(xy, self.w) < RHO","hit = self.stage == 'w' and dist(xy, self.w) <= RHO",'reach <= rho')
m('S2','harness_move3.py',"if stuck or self.t >= self.cap:","if stuck or self.t > self.cap:",'cap off by one')
m('S3','harness_move3.py',"RHO\n","RHO\n",'noop sentinel')  # control: should SURVIVE
m('S4','harness_move3.py',"                self.reason = 'stuck' if stuck else 'cap'","                self.reason = 'cap'",'stuck labeled cap')
m('S5','harness_move3.py',"eps = torch.randn(action.shape, generator=collector.noise).numpy()\n    return np.clip(action + SIGMA*eps, -1., 1.).astype(np.float32)","eps = torch.randn(action.shape, generator=collector.noise).numpy()\n    return np.clip(action + 0*eps, -1., 1.).astype(np.float32)",'noise zeroed (draws kept)')
m('S6','harness_move3.py',"    collector = Collector('A', 0., 16)","    collector = Collector('B', 0.05, 16)",'collector arm B')
m('S7','harness_move3.py',"for arm in ('AC' if q['task'] == 2 else 'ABC')","for arm in ('ABC')",'task2 gets B arm')
m('S8','harness_move3.py',"for draw in range(64, 80)]","for draw in range(64, 79)]",'only 15 draws')
m('S9','harness_move3.py',"    if not smoke and not PRODUCTION_READY:","    if False:",'PRODUCTION_READY gate removed (m3)')
# ---- harness_move1
m('M1','harness_move1.py',"torch.use_deterministic_algorithms(True, warn_only=False)","torch.use_deterministic_algorithms(True, warn_only=True)",'warn_only=True')
m('M2','harness_move1.py',"torch.backends.cuda.matmul.allow_tf32 = False","torch.backends.cuda.matmul.allow_tf32 = True",'tf32 on')
m('M3','harness_move1.py',"        if delivered != self.expected:\n            raise ValueError('E0: injection changed before head')\n","        pass\n",'pre-head hash check removed')
m('M4','harness_move1.py',"        if digest_array(array_of(self.u)) != self.expected:\n            raise ValueError('E0: head mutated injection')\n","        pass\n",'post-head hash check removed')
m('M5','harness_move1.py',"conditional_on_N_success=group == 'gate' and arm != 'N',","conditional_on_N_success=False,",'gate P/R not conditional on N')
m('M6','harness_move1.py',"conditional_on_representation=group == 'main' and arm in 'PQ',","conditional_on_representation=False,",'P/Q not conditional on representation')
m('M7','harness_move1.py',"keep = np.r_[True, np.diff(lengths) > 0]","keep = np.r_[True, np.diff(lengths) >= 0]",'zero-length points kept')
m('M8','harness_move1.py',"passed = distance > noise_band and decoded_identity_a == 'A' and decoded_identity_b == 'B'","passed = distance > noise_band",'manipulation check ignores round trip')
m('M9','harness_move1.py',"    if not smoke and not PRODUCTION_READY:","    if False:",'PRODUCTION_READY gate removed (m1)')
m('M10','harness_move1.py',"for arm in 'PQ':\n                original","for arm in 'P':\n                original",'only P reruns')
m('M11','harness_move1.py',"source=('self-rollout-u' if group == 'gate' and arm == 'P' else","source=('route-A' if group == 'gate' and arm == 'P' else",'gate P source route-A')
# ---- harvest
m('H1','harvest.py',"        require(row['trace_sha256'] == digest_array(xy), 'trace hash mismatch')\n","",'trace hash check removed')
m('H2','harvest.py',"require(type(steps) is int and 0 < steps <= H, 'invalid observed steps')","require(type(steps) is int and 0 < steps, 'invalid observed steps')",'steps<=H check removed')
m('H3','harvest.py',"            require(len(row['head_u_hashes']) == ceil(steps/row['chunk_steps']), 'missing head chunks')\n","",'head chunk count check removed')
m('H4','harvest.py',"    return bool(visited & own) and not bool(visited & other)","    return bool(visited & own)",'crossing ignores other-arm visit')
m('H5','harvest.py',"    return bool(visited & own) and not bool(visited & other)","    return not bool(visited & other)",'crossing ignores own-arm visit')
m('H6','harvest.py',"    if calls > ceil(H/chunk)+2:\n        invalid.append('budget-exceeded')","    if calls > ceil(H/chunk)+3:\n        invalid.append('budget-exceeded')",'harvest budget +3')
m('H7','harvest.py',"    elif g is not None and (w is None or g < w):\n        category = 'd1'","    elif g is not None and (w is None or g <= w):\n        category = 'd1'",'harvest equal-step bypass')
m('H8','harvest.py',"    if row['flow_calls'] != expected:\n        invalid.append('mid-chunk-switch-or-call-ledger-anomaly')","    if False:\n        invalid.append('mid-chunk-switch-or-call-ledger-anomaly')",'ledger anomaly check removed')
m('H9','harvest.py',"t = w if target == 'w' and w is not None and t < w < t+chunk else t+chunk","t = t+chunk",'expected schedule ignores mid-chunk reswitch')
m('H10','harvest.py',"        require(row['seeds'] == seeds(k[0], k[1], k[3], move == 1 and k[2] != 'N'), 'wrong seed')\n","",'seed check removed')
m('H11','harvest.py',"        require(row.get('polluted') is False, 'polluted row')\n","",'polluted check removed')
m('H12','harvest.py',"        require(not any(row['goal_success'][:-1]), 'observations after first success')\n","",'post-success observation check removed')
m('H13','harvest.py',"            reproduced[arm] += bool(n != 'O' and entries[k]['route'] == n and any(by_key[k]['goal_success']))","            reproduced[arm] += bool(n != 'O' and entries[k]['route'] == n)",'reproduction ignores success')
m('H14','harvest.py',"qualified = [p for p in gate if any(by_key[key(p)]['goal_success'])]","qualified = [p for p in gate]",'all gate Qs qualified')
m('H15','harvest.py',"(p['group'] == 'main' and not smoke and len(qualified) < 5)","(p['group'] == 'main' and not smoke and len(qualified) < 4)",'main skip threshold 4')
m('H16','harvest.py',"require(row['donor'] == builder['R_donor'][str(p['task'])], 'wrong donor')","pass",'donor check removed')
m('H17','harvest.py',"require(row['source_trace_sha256'] == n['trace_sha256'], 'gate P is not self-rollout-u')","pass",'gate P source check removed')
m('H18','harvest.py',"n_diff=sum(d['different'] for d in details),","n_diff=0,",'n_diff forced 0')
m('H19','harvest.py',"conformance[str(k)] = a.shape == b.shape and a.tobytes() == b.tobytes()","conformance[str(k)] = True",'conformance always true')
m('H20','harvest.py',"require(sources[0] == sources[1], 'historical A/B fingerprints disagree')","pass",'A/B fingerprint agreement removed')
# ---- builder
m('B1','builder.py',"depth = (lo + hi) // 2","depth = (lo + hi + 1) // 2",'w depth ceil')
m('B2','builder.py',"paths = sorted(walk(s))","paths = sorted(walk(s), reverse=True)",'route naming reversed')
m('B3','builder.py',"for task, other, group in [(4, 5, 'main'), (5, 4, 'main'), (1, 3, 'gate'), (3, 1, 'gate')]:","for task, other, group in [(4, 5, 'main'), (5, 4, 'main'), (1, 3, 'gate'), (3, 1, 'gate'), (2, 4, 'main')]:",'task 2 donor added')
# ---- runtime (orchestration; selftest has no e2e)
m('X1','runtime.py',"            if reason is not None or state.switch_required:\n                break","            if reason is not None:\n                break",'no mid-chunk re-plan on reach')
m('X2','runtime.py',"cap_steps=min(H, int(np.ceil(geom['w_depth']*step_budget*2))),","cap_steps=min(H, int(np.ceil(geom['w_depth']*step_budget*1))),",'cap factor 1 not 2')
m('X3','runtime.py',"kwargs['u'] = encode(donor['task'], donor['route'])","kwargs['u'] = encode(task, 'A')",'R injects own route A instead of donor')
m('X4','runtime.py',"action = (noisy_action(collector, action) if move == 3 else","action = (np.clip(action, -1., 1.).astype(np.float32) if move == 3 else",'Move3 without noise')
m('X5','runtime.py',"stuck = bool(stuck_check(state.xy + [list(obs[:2])])) if stuck_check else False","stuck = False",'stuck never fires')
m('X6','runtime.py',"                target = waypoint if state.stage == 'w' else goal\n","                target = goal\n",'conditions always on goal')
m('X7','runtime.py',"    target = waypoint if state.stage == 'w' else goal\n","    target = goal\n",'conditions always on goal (alt indent)')
m('X8','runtime.py',"    reset_policy(module, task, episode, draw)\n","    pass\n",'policy RNG/caches not reset')
m('X9','runtime.py',"        if p['conditional_on_N_success'] and not n_rows[k]['success']:","        if False:",'N-failed gate rows run anyway')
m('X10','runtime.py',"                if p['group'] == 'main' and not smoke and sum(v['success'] for v in n_rows.values()\n                        if v['task'] in (1, 3)) < 5:","                if False:",'main never gated by >=5')
m('X11','runtime.py',"kwargs['u'] = (m1.encode_trajectory(module, m1.gate_p_source(n_rows[k])['xy'])\n                                   if p['group'] == 'gate' else encode(task, 'A' if arm == 'P' else 'B'))","kwargs['u'] = encode(task, 'A' if arm == 'P' else 'B')",'gate P uses route not self-rollout')
m('X12','runtime.py',"        interface = interface_check(module, route(4, 'A'))","        interface = None",'interface check skipped')
m('X13','runtime.py',"    obs, info = reset_env(env, task, episode)","    obs, info = env.reset(seed=1000*task+episode, options={'task_id': task, 'render_goal': False})",'reset_env bypassed (no np/action_space seeds)')
m('X14','runtime.py',"        encoded[k] = m1.encode_trajectory(module, route(task, arm))" if False else "            encoded[k] = m1.encode_trajectory(module, route(task, arm))","            encoded[k] = m1.encode_trajectory(module, route(task, arm)[::-1])",'routes encoded reversed (goal->start)')

def run(entry):
    id,file,old,new,desc=entry
    d=f'{SCR}/mut/{id}/lacot'
    shutil.rmtree(f'{SCR}/mut/{id}', ignore_errors=True)
    os.makedirs(d+'/experiments/_workorders')
    shutil.copytree(ORIG, d+'/experiments/_workorders/breakthrough-probe')
    for n in ('lacot','results','.venv'): os.symlink(f'{SRC}/{n}', f'{d}/{n}')
    os.symlink(f'{SRC}/experiments/_workorders/ucontrast1', d+'/experiments/_workorders/ucontrast1')
    p=d+'/experiments/_workorders/breakthrough-probe/'+file
    t=open(p).read(); cnt=t.count(old)
    if id!='S3' and cnt!=1: return (id,desc,'BAD-PATTERN count=%d'%cnt,[])
    if id!='S3': t=t.replace(old,new)
    open(p,'w').write(t)
    r=subprocess.run(['python3','-B','selftest.py'],cwd=d+'/experiments/_workorders/breakthrough-probe',capture_output=True,text=True,
                     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=900)
    out=r.stdout+r.stderr
    failed=sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+)',out,re.M)))
    status='SURVIVED' if r.returncode==0 else 'KILLED'
    if r.returncode!=0 and not failed: status='CRASH'; failed=[out[-300:].replace('\n',' | ')]
    shutil.rmtree(f'{SCR}/mut/{id}', ignore_errors=True)
    return (id,desc,status,failed)
if __name__=='__main__':
    sel=sys.argv[1:] 
    todo=[e for e in MUT if not sel or e[0] in sel]
    res=[]
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run,todo): 
            print(f'{r[0]:4s} {r[2]:9s} {r[1]:55s} {",".join(x.replace("test_","t") for x in r[3])[:120]}', flush=True); res.append(r)
    json.dump(res,open(SCR+'/mutation-results.json','w'),ensure_ascii=False,indent=1)
    print('SUMMARY killed',sum(r[2]=='KILLED' for r in res),'survived',sum(r[2]=='SURVIVED' for r in res),'other',sum(r[2] not in('KILLED','SURVIVED') for r in res))
