"""PREREG wallpen v1.1/v1.2/v1.3 interpretation, CPU Stage O + model JSON only."""
import sys
sys.dont_write_bytecode = True
import argparse
import collections
import copy
import json
from pathlib import Path
import numpy as np
import model_metrics as M

TASKS = ('2', '4', '5')
CLASSES = ('方向對只是不準', '分不清', '太近沒前進', '走錯邊')
CATS = ('VALID', 'WALL', 'FAR', 'NEAR', 'OFF')
PAIRS = (('C1', 'B0'), ('C2-hi', 'C1'), ('C2-lo', 'C1'), ('C2-hi', 'B0'), ('C2-lo', 'B0'))
N_BOOT = 10000


def reaches(value, threshold):
    # Decimal preregistered thresholds; avoid 0.35-0.20 rounding below .15.
    return value >= threshold or abs(value-threshold) <= 1e-12


def validate_stage(o):
    if o.get('schema') != 'selfsub-v1-r1' or len(o.get('cells', [])) != 50:
        raise ValueError('Stage O schema/inventory mismatch: selfsub-v1-r1, 50 cells required')
    expected_ruler = dict(smooth_window=9, convolution='valid', anchor=True, arc=12.0, cell_size=4.0, M=16)
    if o['ruler'] != expected_ruler:
        raise ValueError('Stage O ruler mismatch: 9-point valid smoothing, anchor, arc12, M16 required')
    if o['gate_evidence_cache']['key'] != [o['provenance']['ckpt_sha256'], '1ce8c6e8daee5e8691cb024f25679fc66871214b3fa9e21b4a5c8c97ddb17405']:
        raise ValueError('Stage O checkpoint/probe SHA256 evidence key mismatch')
    seen = set()
    for c in o['cells']:
        key = (str(c['task']), tuple(c['cell']))
        if key in seen or key[0] not in TASKS or not isinstance(c['away'], bool):
            raise ValueError(f'Invalid/duplicate cell {key}')
        seen.add(key)
        if not isinstance(c['seed'], int) or len(c['flow']) != 16:
            raise ValueError(f'Stage O requires seed + 16 draws at {key}')
        if len(set(d['draw'] for d in c['flow'])) != 16:
            raise ValueError(f'Duplicate draws at {key}')
        for d in c['flow']:
            sm, p = np.asarray(d['smooth_anchored_xy']), np.asarray(d['p_xy'])
            if d['cat'] not in CATS or sm.shape != (120, 2) or p.shape != (2,) or not np.isfinite(sm).all() or not np.isfinite(p).all():
                raise ValueError(f'Invalid draw at {key}')
    for g in ('G1', 'G2', 'G3'):
        if 'status' not in o['gates'][g]:
            raise ValueError(f'Missing gate {g} status')
    for t in TASKS:
        for away, label in ((True, 'away'), (False, 'nonaway')):
            ds = [d for c in o['cells'] if str(c['task']) == t and c['away'] == away for d in c['flow']]
            counts = {cat: sum(d['cat'] == cat for d in ds) for cat in CATS}
            s = o['summary'][t][label]
            if not ds or s['n'] != len(ds) or s['counts'] != counts or s['valid_rate'] != counts['VALID']/len(ds):
                raise ValueError(f'Summary mismatch task {t} {label}')
        if o['summary'][t]['r'] != o['summary'][t]['away']['valid_rate']:
            raise ValueError(f'Summary r mismatch task {t}')


def assert_paired(a, b):
    ka = [(str(c['task']), c['cell'], c['away']) for c in a['cells']]
    kb = [(str(c['task']), c['cell'], c['away']) for c in b['cells']]
    if ka != kb:
        raise ValueError('Pairing refused: ordered cell lists differ')
    for ca, cb in zip(a['cells'], b['cells']):
        if ca['seed'] != cb['seed'] or [d['draw'] for d in ca['flow']] != [d['draw'] for d in cb['flow']]:
            raise ValueError(f'Pairing refused: seed/draw sequence differs at {ca["task"]} {ca["cell"]}')


def bootstrap(x, y=None, *, paired=True):
    """Two levels; identical index tensors applied to both arms (differences x-y)."""
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 16 or x.shape[0] == 0:
        raise ValueError('Bootstrap expects [trap cells,16]')
    if y is not None and np.shape(y) != x.shape:
        raise ValueError('Bootstrap shape mismatch')
    rng = np.random.default_rng(M.SEED)
    n = len(x)
    ci = rng.integers(n, size=(N_BOOT, n, 1))
    di = rng.integers(16, size=(N_BOOT, n, 16))
    rx = x[ci, di].mean(axis=(1, 2))
    if y is not None:
        y = np.asarray(y, np.float64)
        if not paired:  # Mutation injection used ONLY by the E5 killer.
            ci = rng.integers(n, size=(N_BOOT, n, 1))
            di = rng.integers(16, size=(N_BOOT, n, 16))
        rx -= y[ci, di].mean(axis=(1, 2))
    lo, hi = np.percentile(rx, [2.5, 97.5])
    return dict(estimate=float(x.mean()-(np.mean(y) if y is not None else 0)),
                ci95=[float(lo), float(hi)], n_boot=N_BOOT, seed=M.SEED,
                method='two-level paired percentile' if y is not None else 'two-level percentile')


def failure_classifier(geo, routes):
    # breakdown_h3.py: free landing unchanged; wall landing nearest free center,
    # ambiguous if nearest center from a different class is <1 world unit farther.
    free = sorted({tuple(int(v) for v in c) for c in np.argwhere(geo.mm == 0)})
    fs = set(free)
    cen = np.asarray([geo.c2xy(c) for c in free])
    distances = {}

    def D(a, b):
        if a not in distances:
            dd, q = {a: 0}, collections.deque([a])
            while q:
                c = q.popleft()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = c[0]+di, c[1]+dj
                    if n not in dd and n in fs:
                        dd[n] = dd[c]+1
                        q.append(n)
            distances[a] = dd
        return distances[a].get(b, 10**9)

    def classify(t, c, p):
        g = tuple(routes[t]['g'])
        def klass(s):
            det, prog = D(c, s)+D(s, g)-D(c, g), D(c, g)-D(s, g)
            return CLASSES[3] if det > 0 else (CLASSES[0] if prog >= 1 else CLASSES[2])
        pc = geo.xy2c(p)
        if pc in fs:
            return klass(pc)
        dist = np.linalg.norm(cen-np.asarray(p), axis=1)
        order = np.argsort(dist)
        k = klass(free[order[0]])
        other = next((j for j in order[1:] if klass(free[j]) != k), None)
        if other is not None and dist[other]-dist[order[0]] < 1.0:
            k = CLASSES[1]
        return k
    return classify


def arm_metrics(o, model, geo, tau=M.TAU):
    classify = failure_classifier(geo, o['routes'])
    out = dict(gates={g: o['gates'][g]['status'] for g in ('G1', 'G2', 'G3')},
               gate_pass=all(o['gates'][g]['status'] == 'PASS' for g in ('G1', 'G2', 'G3')),
               positive_control=model['positive_control'], straight=model['straight'], tasks={},
               synthetic=model.get('synthetic', False), model_provenance=model['provenance'])
    arrays = {}
    for t in TASKS:
        cells = [c for c in o['cells'] if str(c['task']) == t and c['away']]
        rows, per_draw = [], []
        counts = collections.Counter()
        failures = collections.Counter({k: 0 for k in CLASSES})
        for c in cells:
            rr = []
            for d in c['flow']:
                md, rd = geo.stats(np.asarray(d['smooth_anchored_xy']))
                valid, clean = d['cat'] == 'VALID', md <= tau
                counts[d['cat']] += 1
                klass = None if valid else classify(t, tuple(c['cell']), d['p_xy'])
                if klass is not None:
                    failures[klass] += 1
                rr.append([valid, clean, valid and clean, rd > 0, d['cat'] == 'NEAR'])
                per_draw.append(dict(cell=c['cell'], seed=c['seed'], draw=d['draw'], cat=d['cat'],
                                     max_occupancy_depth=md, max_true_wall_depth=rd, failure_class=klass))
            rows.append(rr)
        a = np.asarray(rows, np.float64)
        arrays[t] = a
        metrics = {name: dict(bootstrap(a[:, :, k]), count=int(a[:, :, k].sum()), n=int(a.shape[0]*16))
                   for k, name in enumerate(('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near'))}
        nonaway = [d for c in o['cells'] if str(c['task']) == t and not c['away'] for d in c['flow']]
        nc = sum(d['cat'] == 'VALID' for d in nonaway)
        nf = sum(failures.values())
        out['tasks'][t] = dict(metrics, counts={cat: counts[cat] for cat in CATS},
                               nonaway_valid=dict(count=nc, n=len(nonaway), rate=nc/len(nonaway)),
                               failures=dict(n=nf, counts=dict(failures),
                                             proportions={k: (failures[k]/nf if nf else None) for k in CLASSES}),
                               per_draw=per_draw)
    return out, arrays


def guards(arm, base):
    sa, sb = arm['straight'], base['straight']
    if sa['n'] != 2048 or sb['n'] != 2048 or sa['coordinate_sha256'] != sb['coordinate_sha256']:
        raise ValueError('Straight guard refused: n != 2048 or coordinate sha256 differs')
    if not np.isfinite(sa['mse']) or not np.isfinite(sb['mse']) or sa['mse'] < 0 or sb['mse'] <= 0:
        raise ValueError('Invalid straight MSE (positive B0 required)')
    checks = [dict(name='直路重建 MSE 守門', value=sa['mse'], limit=sb['mse']*1.10,
                   ratio=sa['mse']/sb['mse'], status='PASS' if sa['mse'] <= sb['mse']*1.10 else 'FAIL')]
    for t in TASKS:
        drop = base['tasks'][t]['nonaway_valid']['rate']-arm['tasks'][t]['nonaway_valid']['rate']
        near = arm['tasks'][t]['near']['estimate']-base['tasks'][t]['near']['estimate']
        checks.extend([dict(name=f'非陷阱格 VALID 守門 task {t}', value=drop, limit=.10,
                            status='PASS' if drop <= .10+1e-12 else 'FAIL'),
                       dict(name=f'NEAR 守門 task {t}', value=near, limit=.15,
                            status='PASS' if near <= .15+1e-12 else 'FAIL')])
    return dict(pass_all=all(c['status'] == 'PASS' for c in checks), checks=checks,
                failed=[c['name'] for c in checks if c['status'] != 'PASS'])


def eligible(arm):
    return arm['gate_pass'] and arm['guards']['pass_all']


def blockers(arm):
    return [f'Stage O {g}={s}' for g, s in arm['gates'].items() if s != 'PASS'] + [
        f'{name}=FAIL' for name in arm['guards']['failed']]


def assign_guards(arms):
    for name, arm in arms.items():
        reference = 'C1' if name in ('C2-hi', 'C2-lo') else 'B0'
        arm['guard_reference'] = reference
        arm['guards_vs_b0'] = guards(arm, arms['B0'])
        arm['guards'] = (guards(arm, arms[reference]) if reference in arms else
                         dict(pass_all=False, checks=[], failed=['C1 未提供，守門無法計算']))
        arm['no_conclusion_reasons'] = blockers(arm)


def select_primary(arms):
    hi, lo = arms.get('C2-hi'), arms.get('C2-lo')
    if hi is None:
        return None, '不出結論：C2-hi 未提供，無法檢查預先指定的主判'
    if eligible(hi):
        return 'C2-hi', '主判 C2-hi：Stage O 閘與對 C1 守門全 PASS（不看主指標）'
    why = '、'.join(blockers(hi))
    if lo is not None and eligible(lo):
        return 'C2-lo', f'主判由 hi 改 lo，因為 hi 的 {why}（不看主指標）'
    return None, f'不出結論：hi 的 {why}；' + ('lo 亦不通過：'+'、'.join(blockers(lo)) if lo else 'lo 未提供')


def class_branch(arm_task, c1_task):
    f, b = arm_task['failures'], c1_task['failures']
    if f['n'] == 0 or b['n'] == 0:
        return '待討論：無失敗樣本，失敗比例無法比較'
    if reaches(f['proportions'][CLASSES[2]]-b['proportions'][CLASSES[2]], .15):
        return '教成原地踏步、調懲罰'
    maximum = max(f['counts'].values())
    winners = [k for k, n in f['counts'].items() if n == maximum]
    if len(winners) != 1:
        return '待討論：最大類並列（'+'、'.join(winners)+'）'
    return {CLASSES[3]: '走錯邊為主 ⇒ value／搜尋',
            CLASSES[0]: '方向對只是不準為主 ⇒ 容量或「看得到」',
            CLASSES[1]: '分不清為主 ⇒ 待討論',
            CLASSES[2]: '太近沒前進為主，但增幅未達 .15 ⇒ 待討論'}[winners[0]]


def noise_report(arms):
    if 'B0s' not in arms:
        return dict(status='未提供', by_task={}, same_scale=False)
    sample, base = arms['B0s'], arms['B0']
    reasons = [f'{name} Stage O {g}={v}' for name, arm in (('B0', base), ('B0s', sample))
               for g, v in arm['gates'].items() if v != 'PASS']
    if reasons:
        return dict(status='不出結論：'+'、'.join(reasons), by_task={}, same_scale=False)
    ratio = abs(sample['straight']['mse']/base['straight']['mse']-1.0)
    ts = {t: dict(valid=abs(sample['tasks'][t]['valid']['estimate']-base['tasks'][t]['valid']['estimate']),
                  clean=abs(sample['tasks'][t]['clean']['estimate']-base['tasks'][t]['clean']['estimate']),
                  straight_mse_ratio=ratio,
                  nonaway_valid=abs(sample['tasks'][t]['nonaway_valid']['rate']-base['tasks'][t]['nonaway_valid']['rate']),
                  near=abs(sample['tasks'][t]['near']['estimate']-base['tasks'][t]['near']['estimate']))
          for t in ('4', '5')}
    return dict(status='僅描述，不進主判', by_task=ts,
                same_scale=any(reaches(ts[t]['valid'], .10) for t in ts))


def interpret(arms, differences):
    primary, reason = select_primary(arms)
    decisions = dict(primary=primary, reason=reason, by_task={},
                     hi_primary_metrics=({t: arms['C2-hi']['tasks'][t]['valid'] for t in TASKS} if 'C2-hi' in arms else None),
                     arm_status={name: ('可判讀' if eligible(a) else '不出結論') for name, a in arms.items()})
    base_ok = arms['B0']['gate_pass']
    c1 = arms.get('C1')
    for t in TASKS:
        if t == '2':
            decisions['by_task'][t] = dict(c1='task 2 只描述', c2='task 2 只描述')
            continue
        c1_text = '不出結論：C1 未提供'
        if c1:
            if not base_ok or not eligible(c1):
                c1_text = '不出結論：B0 Stage O 閘或 C1 Stage O 閘／守門 FAIL'
            else:
                delta = differences['C1-B0'][t]['valid']['estimate']
                c1_text = ('模仿擦牆的 teacher 是一大塊原因' if reaches(delta, .15) else 'C1 VALID 增幅未達 .15')
        c2_text = reason
        branch = None
        if primary:
            a = arms[primary]
            if not base_ok or not a['gate_pass'] or c1 is None or not c1['gate_pass']:
                c2_text = '不出結論：B0／C1／主判 Stage O 閘未過或 C1 未提供'
            else:
                ds = differences[primary+'-C1'][t]
                dv, dc = ds['valid']['estimate'], ds['clean']['estimate']
                if reaches(dv, .15):
                    c2_text = '沒有牆的概念是原因、訓練時罰有用'
                elif reaches(dc, .30):
                    c2_text = '學到牆、近子目標沒變好'
                    branch = class_branch(a['tasks'][t], c1['tasks'][t])
                else:
                    c2_text = '懲罰沒教到牆（權重太小或被躲掉）'
        decisions['by_task'][t] = dict(c1=c1_text, c2=c2_text, failure_branch=branch)
    rollout = primary is not None and all(
        arms[primary]['tasks'][t]['valid']['estimate'] >= .70 for t in ('4', '5'))
    decisions['rollout'] = 'task 4 與 5 各自 VALID ≥ .70，下一步可考慮 rollout（另呈主人）' if rollout else '尚不建議考慮 rollout'
    decisions['conclusion_blockers'] = [f'{name} Stage O {g}={v}' for name in ('B0', 'C1') if name in arms
                                       for g, v in arms[name]['gates'].items() if v != 'PASS']
    if c1 is None:
        decisions['conclusion_blockers'].append('C1 未提供')
    else:
        decisions['conclusion_blockers'].extend('C1 對 B0 '+name+'=FAIL' for name in c1['guards']['failed'])
    if primary is None:
        decisions['conclusion_blockers'].append(reason)
    warning = '⚠️ C1 對 B0 已有副作用，C2 對 C1 通過不代表整套沒問題'
    c1_bad = c1 is not None and not c1['guards']['pass_all']
    decisions['c2_guard_conclusions'] = {}
    for name in ('C2-hi', 'C2-lo'):
        if name not in arms:
            continue
        arm = arms[name]
        def layer(g, reference):
            if not base_ok or c1 is None or not c1['gate_pass'] or not arm['gate_pass']:
                return f'不出結論：{reference} 比較的 Stage O 閘未過或 C1 未提供'
            return ('PASS：未見副作用' if g['pass_all'] else 'FAIL：'+'、'.join(g['failed']))
        decisions['c2_guard_conclusions'][name] = dict(
            penalty_side_effects=layer(arm['guards'], 'C2 對 C1'),
            overall_degradation=layer(arm['guards_vs_b0'], 'C2 對 B0'))
    noise = noise_report(arms)
    decisions['noise'] = noise
    suffix = '訓練雜訊跟門檻同量級，要更多 seed 才能定' if noise['same_scale'] else ''
    def annotate(text, c2=False):
        return text + ('；'+warning if c2 and c1_bad else '') + ('；'+suffix if suffix else '')
    decisions['reason'] = annotate(decisions['reason'], True)
    decisions['rollout'] = annotate(decisions['rollout'])
    for t, row in decisions['by_task'].items():
        row['c1'] = annotate(row['c1'])
        row['c2'] = annotate(row['c2'], True)
        if row.get('failure_branch'):
            row['failure_branch'] = annotate(row['failure_branch'], True)
    for columns in decisions['c2_guard_conclusions'].values():
        for key in columns:
            columns[key] = annotate(columns[key], True)
    return decisions


def analyze(inputs, geo):
    if 'B0' not in inputs or any(n not in ('B0', 'B0s', 'G', 'C1', 'C2-lo', 'C2-hi') for n in inputs):
        raise ValueError('B0 required; supported arms B0/B0s/G/C1/C2-lo/C2-hi')
    base_o = inputs['B0'][0]
    for name, (o, model) in inputs.items():
        validate_stage(o)
        assert_paired(o, base_o)
        if o['maze'] != geo.mm.tolist() or o['routes'] != base_o['routes']:
            raise ValueError(f'{name}: maze/routes mismatch')
        for key in ('normalization_mu', 'normalization_sd', 'dataset_sha256'):
            if model['provenance'][key] != o['provenance'][key]:
                raise ValueError(f'{name}: model/Stage O provenance mismatch {key}')
        if not model.get('synthetic', False) and model['provenance']['ckpt_sha256'] != o['provenance']['ckpt_sha256']:
            raise ValueError(f'{name}: checkpoint identity mismatch')
        positive = model['positive_control']
        if model['schema'] != 'wallpen-model-v1' or model['device'] != 'cpu' or (positive is None and not model.get('synthetic')) or (positive is not None and positive.get('tau') != M.TAU):
            raise ValueError(f'{name}: model metrics schema/device/tau mismatch')
        if not np.array_equal(np.asarray(o['provenance']['normalization_mu']), geo.mu) or not np.array_equal(np.asarray(o['provenance']['normalization_sd']), geo.sd):
            raise ValueError(f'{name}: geometry normalization mismatch')
        if o['provenance']['dataset_sha256'] != M.DATASET_PIN:
            raise ValueError(f'{name}: DATASET_PIN mismatch')
    if 'G' in inputs and inputs['G'][0]['provenance']['ckpt_sha256'] != base_o['provenance']['ckpt_sha256']:
        raise ValueError('G is not bitwise H3: checkpoint SHA256 mismatch')
    arms, arrays = {}, {}
    for name, (o, model) in inputs.items():
        arms[name], arrays[name] = arm_metrics(o, model, geo)
    assign_guards(arms)
    differences = {}
    for left, right in PAIRS:
        if left not in arms or right not in arms:
            continue
        differences[left+'-'+right] = {t: {metric: bootstrap(arrays[left][t][:, :, k], arrays[right][t][:, :, k])
                                           for k, metric in enumerate(('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near'))}
                                      for t in TASKS}
    return dict(schema='wallpen-analysis-v1', tau=M.TAU, arms=arms, differences=differences,
                interpretation=interpret(arms, differences), prereg_version='v1.3+supplement',
                notes=['判定 task 4 與 5 各自判；task 2 只描述；不跨 task 合併。',
                       '只描述落點與路，不是 rollout 成功率。',
                       '尺的 9 點平滑本身也會切角（teacher 原始 max .1008 → 平滑後 .121）；各臂共用同尺，深度絕對值會偏深。',
                       '定標 λ／ρ／pen-MSE log 不在這兩份 JSON 的輸入契約中：未提供，不推定權重有效。'],
                synthetic=any(a['synthetic'] for a in arms.values()))


def render(result):
    lines = ['wallpen v1.1 + v1.2 + v1.3 | CPU | substitute example' if result['synthetic'] else 'wallpen v1.1 + v1.2 + v1.3 | CPU',
             'arm     task VALID(n/N)   r [95% CI]             clean(n/N)  r [95% CI]             joint truewall',
             '-'*110]
    for name, arm in result['arms'].items():
        for t in TASKS:
            task = arm['tasks'][t]
            v, c = task['valid'], task['clean']
            lines.append(f'{name:<7} {t:>4} {v["count"]:>3}/{v["n"]:<3}     {v["estimate"]:.4f} [{v["ci95"][0]:.4f},{v["ci95"][1]:.4f}]  '
                         f'{c["count"]:>3}/{c["n"]:<3}   {c["estimate"]:.4f} [{c["ci95"][0]:.4f},{c["ci95"][1]:.4f}]  '
                         f'{task["valid_and_clean"]["count"]:>3}   {task["true_wall_entry"]["count"]:>3}')
        lines.append(f'  Stage O: {arm["gates"]}; {result["interpretation"]["arm_status"][name]}')
        lines.append(f'  守門參照: {arm["guard_reference"]}')
        for check in arm['guards']['checks']:
            lines.append(f'  {check["status"]} {check["name"]}: {check["value"]:.8f} <= {check["limit"]:.8f}')
        if name in ('C2-hi', 'C2-lo'):
            lines.append('  獨立欄 C2 對 B0：')
            for check in arm['guards_vs_b0']['checks']:
                lines.append(f'    {check["status"]} {check["name"]}: {check["value"]:.8f} <= {check["limit"]:.8f}')
        p = arm['positive_control']
        if arm['synthetic']:
            lines.append('  MODEL SYNTHETIC: '+str(arm['model_provenance'].get('synthetic_note', 'see model JSON')))
        tm = arm['straight']['turn_mse']
        lines.append(f'  straight MSE={arm["straight"]["mse"]:.9f} n={arm["straight"]["n"]} batches={arm["straight"]["batches"]} '
                     f'turn MSE={tm if tm is not None else "未量"} n={arm["straight"]["turn_n"]}')
        lines.append(f'  straight sha256={arm["straight"]["coordinate_sha256"]}')
        lines.append(f'  positive control: {p["clean_count"]}/{p["n"]}; p50={p["p50"]:.4f} p95={p["p95"]:.4f} max={p["max"]:.4f}' if p is not None else '  positive control: 未量（Hinf 無指定 checkpoint）')
        for e in p['named'] if p is not None else []:
            lines.append(f'    task {e["task"]} {e["cell"]} route {e["route_id"]}: occupancy={e["max_occupancy_depth"]:.4f} truewall={e["max_true_wall_depth"]:.4f}')
        for t in ('4', '5'):
            lines.append(f'  task {t} failures: {arm["tasks"][t]["failures"]}')
    lines.append('\nPaired differences: estimate [95% CI]')
    for pair, ts in result['differences'].items():
        for t, ms in ts.items():
            lines.append(f'{pair:<11} task {t}: '+ '; '.join(f'{k} {ms[k]["estimate"]:+.4f} [{ms[k]["ci95"][0]:+.4f},{ms[k]["ci95"][1]:+.4f}]' for k in ('valid', 'clean')))
    d = result['interpretation']
    lines.extend(['', d['reason'], 'hi 主指標（完整保留）: '+json.dumps(d['hi_primary_metrics'], ensure_ascii=False)])
    for t, row in d['by_task'].items():
        lines.append(f'task {t}: C1: {row["c1"]}; C2: {row["c2"]}'+(f'; {row["failure_branch"]}' if row.get('failure_branch') else ''))
    for name, columns in d['c2_guard_conclusions'].items():
        lines.append(f'{name} 懲罰有沒有副作用（C2 對 C1）：{columns["penalty_side_effects"]}')
        lines.append(f'{name} 整套跟原本比有沒有變差（C2 對 B0）：{columns["overall_degradation"]}')
    lines.append('B0s |B0s − B0|：'+json.dumps(d['noise'], ensure_ascii=False))
    for pair in result['differences']:
        for t in ('4', '5'):
            lines.append(f'{pair} task {t} B0s 雜訊欄：'+json.dumps(d['noise']['by_task'].get(t, d['noise']['status']), ensure_ascii=False))
    lines.append(d['rollout'])
    lines.append('其他擋住結論的原因：'+('、'.join(d['conclusion_blockers']) or '無'))
    lines.extend(result['notes'])
    return '\n'.join(lines)+'\n'


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    names = list(result['arms'])
    width = .8/len(names)
    x = np.arange(3)
    for ax, metric in zip(axs, ('valid', 'clean')):
        for i, name in enumerate(names):
            vs = [result['arms'][name]['tasks'][t][metric] for t in TASKS]
            pos = x-.4+width*(i+.5)
            ax.bar(pos, [v['estimate'] for v in vs], width=width, label=name)
            # Draw CI endpoints directly: percentile interval need not contain point estimate.
            ax.vlines(pos, [v['ci95'][0] for v in vs], [v['ci95'][1] for v in vs], color='black')
            for sign in (0, 1):
                ax.hlines([v['ci95'][sign] for v in vs], pos-width*.15, pos+width*.15, color='black')
        ax.set_xticks(x, ['Task 2 (descriptive)', 'Task 4', 'Task 5'])
        ax.set_title('Trap VALID rate' if metric == 'valid' else 'Path clean rate (tau=.0664)')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
    axs[0].set_ylabel('Rate / 95% two-level bootstrap interval')
    axs[1].legend()
    fig.suptitle('Substitute arms only; no new-arm evaluation' if result['synthetic'] else 'wallpen v1')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', action='append', required=True, metavar='NAME=STAGEO_JSON,MODEL_JSON')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=M.DATASET)
    args = p.parse_args()
    env = None
    try:
        inputs, sources = {}, {}
        for value in args.arm:
            name, files = value.split('=', 1)
            sp, mp = map(Path, files.split(',', 1))
            if name in inputs:
                raise ValueError(f'Duplicate arm {name}')
            inputs[name] = (json.loads(sp.read_text()), json.loads(mp.read_text()))
            sources[name] = dict(stageo_path=str(sp), stageo_sha256=M.sha_file(sp), model_path=str(mp), model_sha256=M.sha_file(mp))
        # Refuse schema mismatches before expensive geometry construction.
        for o, _ in inputs.values():
            validate_stage(o)
        geo, env = M.geometry_from_dataset(args.dataset)
        result = analyze(inputs, geo)
        result['sources'] = sources
        args.outdir.mkdir(parents=True, exist_ok=True)
        M.write_json(args.outdir/'wallpen-analysis.json', result)
        (args.outdir/'wallpen-analysis.txt').write_text(render(result))
        plot(result, args.outdir/'wallpen-analysis.png')
        print(render(result))
    except Exception as e:
        p.exit(2, f'BLOCKED: {type(e).__name__}: {e}\n')
    finally:
        if env:
            env.close()


if __name__ == '__main__':
    main()
