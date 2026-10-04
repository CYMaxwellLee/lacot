"""CPU implementation of PREREG-wallpen-v2, including section VI.

All v1 metrics, paired bootstrap, route ruler and failure classes are imported.
No checkpoint is loaded. Missing draws remain unmeasured, never fabricated.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import numpy as np
import analyze_wallpen as V1

M = V1.M
TASKS = V1.TASKS
SEEDS = ('C1p', 'C1p-s34', 'C1p-s35')
METRICS = ('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near')
PREREG_SHA = '2131461b93bd339d79eec94ec043687a757dc31e86c88f4c08795e0b10309563'
NOISE_NOTE = '訓練雜訊與門檻同量級'


def distance_metrics(rows):
    a = np.asarray(rows, dtype=np.float64)
    rng = np.random.default_rng(M.SEED)
    n = len(a)
    ci = rng.integers(n, size=(V1.N_BOOT, n, 1))
    di = rng.integers(16, size=(V1.N_BOOT, n, 16))
    samples = np.median(a[ci, di].reshape(V1.N_BOOT, -1), axis=1)
    median = dict(estimate=float(np.median(a)), ci95=np.percentile(samples, [2.5, 97.5]).tolist(),
                  n=int(a.size), n_boot=V1.N_BOOT, seed=M.SEED, method='two-level median percentile')
    floor = (a >= 5.65) & (a <= 6.5)
    return dict(median_distance=median, floor_fraction=dict(V1.bootstrap(floor), count=int(floor.sum()), n=int(a.size)))


def floor_subset_cells(o):
    # wallpen2 r3: every shortest-route suffix must have a far-enough target.
    # Raw cell centers differ by cell_size per grid coordinate (origin cancels).
    subset = {t: [] for t in TASKS}
    for c in o['cells']:
        if not c['away']:
            continue
        t, start = str(c['task']), c['cell']
        paths = [p for p in o['routes'][t]['paths'] if start in p]
        if not paths:
            raise ValueError('Floor subset cell missing from routes: '+str((t, start)))
        targets = [p[min(p.index(start)+3, len(p)-1)] for p in paths]
        if all(o['ruler']['cell_size'] * np.linalg.norm(np.asarray(end)-start) >= 5.65 for end in targets):
            subset[t].append(start)
    return subset


def descriptors(o, subset):
    result = {t: {label: distance_metrics([
        [float(np.sqrt(sum((float(d['p_xy'][j])-float(d['smooth_anchored_xy'][0][j]))**2
                           for j in (0, 1)))) for d in c['flow']]
        for c in o['cells'] if str(c['task']) == t and c['away'] == away])
        for away, label in ((True, 'away'), (False, 'nonaway'))} for t in TASKS}
    # wallpen2 r3: only the trap floor fraction changes; medians keep all cells.
    for t in TASKS:
        distances = np.asarray([
            [float(np.sqrt(sum((float(d['p_xy'][j])-float(d['smooth_anchored_xy'][0][j]))**2
                               for j in (0, 1)))) for d in c['flow']]
            for c in o['cells'] if str(c['task']) == t and c['away'] and c['cell'] in subset[t]])
        floor = (distances >= 5.65) & (distances <= 6.5)
        result[t]['away']['floor_fraction'] = dict(V1.bootstrap(floor), count=int(floor.sum()), n=int(floor.size))
    return result


def has_draws(o):
    return any(c.get('flow') for c in o.get('cells', []))


def validate_no_draw(o, reference):
    # V1's full validator requires 16 draws. Only its no-draw envelope is special.
    if o.get('schema') != reference['schema'] or o.get('ruler') != reference['ruler']:
        raise ValueError('No-draw Stage O schema/ruler mismatch')
    if o['gate_evidence_cache']['key'] != [o['provenance']['ckpt_sha256'], reference['gate_evidence_cache']['key'][1]]:
        raise ValueError('No-draw Stage O evidence key mismatch')
    def inventory(stage):
        return [(str(c['task']), c['cell'], c['away'], c['seed']) for c in stage['cells']]
    if inventory(o) != inventory(reference) or any(not isinstance(c['away'], bool) or not isinstance(c['seed'], int) for c in o['cells']):
        raise ValueError('No-draw Stage O inventory/seed mismatch')
    if any(c.get('flow') != [] for c in o['cells']):
        raise ValueError('No-draw Stage O requires explicitly empty flow arrays')
    for g in ('G1', 'G2', 'G3'):
        if 'status' not in o['gates'][g]:
            raise ValueError('No-draw Stage O missing gate '+g)


def validate_model(o, model, geo):
    for key in ('normalization_mu', 'normalization_sd', 'dataset_sha256'):
        if model['provenance'][key] != o['provenance'][key]:
            raise ValueError('Model/Stage O provenance mismatch '+key)
    if not model.get('synthetic', False) and model['provenance']['ckpt_sha256'] != o['provenance']['ckpt_sha256']:
        raise ValueError('Model/Stage O checkpoint identity mismatch')
    positive = model['positive_control']
    if model['schema'] != 'wallpen-model-v1' or model['device'] != 'cpu' or (positive is None and not model.get('synthetic')) or (positive is not None and positive.get('tau') != M.TAU):
        raise ValueError('Model schema/device/tau mismatch')
    if not np.array_equal(o['provenance']['normalization_mu'], geo.mu) or not np.array_equal(o['provenance']['normalization_sd'], geo.sd):
        raise ValueError('Geometry normalization mismatch')
    if o['provenance']['dataset_sha256'] != M.DATASET_PIN:
        raise ValueError('DATASET_PIN mismatch')
    s = model['straight']
    if s['n'] != 2048 or not np.isfinite(s['mse']) or s['mse'] < 0:
        raise ValueError('Invalid straight MSE/sample count')


def quantities(task):
    q = {k: (task[k]['estimate'], task[k]['n']) for k in METRICS}
    q['nonaway_valid'] = (task['nonaway_valid']['rate'], task['nonaway_valid']['n'])
    for label in ('away', 'nonaway'):
        for k in ('median_distance', 'floor_fraction'):
            m = task['descriptors'][label][k]
            q[label+'_'+k] = (m['estimate'], m['n'])
    return q


def noise_report(arms):
    by_task = {}
    for t in TASKS:
        qs = [quantities(arms[n]['tasks'][t]) for n in SEEDS]
        row = {}
        for k, (_, n) in qs[0].items():
            if any(q[k][1] != n for q in qs):
                raise ValueError('Seed sample counts differ: '+k)
            values = [q[k][0] for q in qs]
            raw = max(values)-min(values)
            row[k] = dict(values=values, min=min(values), max=max(values), raw_range=raw,
                          R=raw if raw != 0 else 1/n, n=n, fallback=raw == 0)
        by_task[t] = row
    values = [arms[n]['straight']['mse'] for n in SEEDS]
    raw = max(values)-min(values)
    return dict(by_task=by_task, straight_mse=dict(values=values, min=min(values), max=max(values),
                raw_range=raw, R=raw if raw != 0 else 1/2048, n=2048, fallback=raw == 0),
                same_scale=any(V1.reaches(by_task[t]['valid']['R'], .10) for t in ('4', '5')))


def guards(arm, base, noise):
    if not arm['has_draws']:
        return dict(pass_all=False, checks=[], failed=['沒有 draw，守門未量'])
    checks = []
    for t in TASKS:
        a, b, ns = arm['tasks'][t], base['tasks'][t], noise['by_task'][t]
        for key, value, label in (
            ('nonaway_valid', b['nonaway_valid']['rate']-a['nonaway_valid']['rate'], '非陷阱 VALID'),
            ('near', a['near']['estimate']-b['near']['estimate'], '陷阱 NEAR')):
            limit = 2*ns[key]['R']
            checks.append(dict(task=t, metric=key, name=f'{label} 守門 task {t}', value=value, limit=limit,
                               status='FAIL' if value > limit+1e-12 else 'PASS'))
    return dict(pass_all=all(c['status'] == 'PASS' for c in checks), checks=checks,
                failed=[c['name'] for c in checks if c['status'] == 'FAIL'])


def blockers(arm):
    return ([f'Stage O {g}={s}' for g, s in arm['gates'].items() if s != 'PASS'] +
            ([] if arm['has_draws'] else ['沒有 draw']) + [s+'=FAIL' for s in arm['guards']['failed']])


def eligible(arm):
    return arm['gate_pass'] and arm['has_draws'] and arm['guards']['pass_all']


def select_primary(arms):
    if eligible(arms['C3-b']):
        return 'C3-b', '主判 C3-b：Stage O 閘與守門全 PASS（不看主指標）'
    why = '、'.join(blockers(arms['C3-b']))
    if eligible(arms['C3-a']):
        return 'C3-a', '主判由 C3-b 改 C3-a，因為 C3-b 的 '+why+'（不看主指標）'
    return None, '不出結論：C3-b 的 '+why+'；C3-a 的 '+'、'.join(blockers(arms['C3-a']))


def descriptor_checks(task, ns):
    med = task['descriptors']['away']['median_distance']['estimate']
    floor = task['descriptors']['away']['floor_fraction']['estimate']
    md, fl = ns['away_median_distance'], ns['away_floor_fraction']
    band = [md['min']-md['R'], md['max']+md['R']]
    limit = fl['max'] + 2*fl['R']
    squeezed = floor > limit + 1e-12
    return dict(median=med, noise_band=band, in_band=band[0]-1e-12 <= med <= band[1]+1e-12,
                shrinking=med < band[0]-1e-12, floor_fraction=floor, floor_limit=limit, squeezed=squeezed)


def interpret(arms, differences, noise):
    primary, reason = select_primary(arms)
    def annotate(text, checks=None):
        return text + ('；仍在縮' if checks and checks['shrinking'] else '') + ('；'+NOISE_NOTE if noise['same_scale'] else '')
    by_arm = {}
    for name in ('C3-b', 'C3-a'):
        rows = {}
        for t in TASKS:
            if not arms[name]['has_draws']:
                rows[t] = dict(conclusion=annotate('不出結論：'+'、'.join(blockers(arms[name]))))
                continue
            checks = descriptor_checks(arms[name]['tasks'][t], noise['by_task'][t])
            ds = differences[name+'-C1p'][t]
            dv, dc = ds['valid']['estimate'], ds['clean']['estimate']
            branch = None
            if t == '2':
                text = 'task 2 只描述'
            elif not eligible(arms[name]):
                text = '不出結論：'+'、'.join(blockers(arms[name]))
            elif V1.reaches(dv, .15):
                if checks['in_band'] and not checks['squeezed']:
                    text = '不碰牆又走得遠的計畫學得到'
                else:
                    text = 'VALID 上來了，但步長分布塌在下限／仍在縮'
            elif V1.reaches(dc, .30):
                if checks['squeezed']:
                    text = '學成最小步'
                else:
                    branch = V1.class_branch(arms[name]['tasks'][t], arms['C1p']['tasks'][t])
                    text = branch
            else:
                text = '懲罰沒教到牆'
            rows[t] = dict(conclusion=annotate(text, checks), failure_branch=annotate(branch, checks) if branch else None,
                           valid_delta=dv, clean_delta=dc, descriptor_checks=checks)
        by_arm[name] = rows
    return dict(primary=primary, reason=annotate(reason), by_arm=by_arm,
                by_task=by_arm[primary] if primary else {t: dict(conclusion=annotate(reason)) for t in TASKS},
                b_primary_metrics={t: arms['C3-b']['tasks'][t]['valid'] for t in TASKS} if arms['C3-b']['has_draws'] else None)


def analyze(inputs, geo):
    if set(inputs) != set(SEEDS+('C3-b', 'C3-a')):
        raise ValueError('Exactly C1p, C1p-s34, C1p-s35, C3-b, C3-a required')
    base_o = inputs['C1p'][0]
    V1.validate_stage(base_o)
    # wallpen2 r3: one route-defined subset shared by both arms and all seeds.
    subset = floor_subset_cells(base_o)
    arms, arrays = {}, {}
    for name, (o, model) in inputs.items():
        draw = has_draws(o)
        if draw:
            V1.validate_stage(o)
            V1.assert_paired(o, base_o)
        else:
            validate_no_draw(o, base_o)
        if o['maze'] != geo.mm.tolist() or o['routes'] != base_o['routes']:
            raise ValueError(name+': maze/routes mismatch')
        validate_model(o, model, geo)
        if model['straight']['coordinate_sha256'] != inputs['C1p'][1]['straight']['coordinate_sha256']:
            raise ValueError(name+': straight coordinate set mismatch')
        if name in SEEDS and (not draw or any(o['gates'][g]['status'] != 'PASS' for g in ('G1', 'G2', 'G3'))):
            raise ValueError(name+': seed reference requires PASS gates and measured draws')
        if draw:
            arms[name], arrays[name] = V1.arm_metrics(o, model, geo)
            desc = descriptors(o, subset)
            for t in TASKS:
                arms[name]['tasks'][t]['descriptors'] = desc[t]
        else:
            gates = {g: o['gates'][g]['status'] for g in ('G1', 'G2', 'G3')}
            arms[name] = dict(gates=gates, gate_pass=all(s == 'PASS' for s in gates.values()),
                              tasks={}, straight=model['straight'], positive_control=model['positive_control'],
                              synthetic=model.get('synthetic', False), model_provenance=model['provenance'])
        arms[name]['has_draws'] = draw
    noise = noise_report(arms)
    warnings = []
    for name, arm in arms.items():
        arm['guards'] = guards(arm, arms['C1p'], noise)
        equal = arm['straight']['mse'] == arms['C1p']['straight']['mse']
        arm['straight_equal_to_C1p'] = equal
        if not equal:
            warnings.append(name+'：警告，直路 MSE 不等於 C1p（只列，不當副作用判準）')
    differences = {name+'-C1p': {t: {metric: V1.bootstrap(arrays[name][t][:, :, k], arrays['C1p'][t][:, :, k])
                    for k, metric in enumerate(METRICS)} for t in TASKS} for name in arrays if name != 'C1p'}
    return dict(schema='wallpen2-analysis-v2', prereg_sha256=PREREG_SHA, tau=M.TAU, arms=arms,
                floor_subset_cells=subset,  # wallpen2 r3: audit the selected cells.
                noise=noise, differences=differences, interpretation=interpret(arms, differences, noise), warnings=warnings,
                synthetic=any(a['synthetic'] for a in arms.values()),
                notes=['task 4、5 各自判；task 2 只描述。非陷阱描述量不參與判定。',
                       '只描述落點與路；不是 rollout 成功率。',
                       '中位距離區間為兩層 percentile bootstrap（10000 次，seed 20261004）；比例沿用 v1 bootstrap。',
                       '替身資料不代表 v2 實驗臂結果；三 seed 替身的雜訊不是 v2 訓練雜訊。'])


def render(result):
    lines = ['wallpen v2 | CPU | '+('SUBSTITUTE / SYNTHETIC' if result['synthetic'] else 'measured JSON'),
             'arm       task VALID [95% CI]          clean [95% CI]          joint    med-trap [95% CI]       floor-trap [95% CI]', '-'*135]
    # wallpen2 r3: include the common floor subset even for arms without draws.
    for t, cells in result['floor_subset_cells'].items():
        lines.append(f'task {t} floor_subset_cells: '+json.dumps(cells))
    for name, arm in result['arms'].items():
        if not arm['has_draws']:
            lines.append(f'{name:<10} 未量（沒有 draw）')
        else:
            for t in TASKS:
                a = arm['tasks'][t]
                def fmt(m):
                    return f'{m["estimate"]:.4f} [{m["ci95"][0]:.4f},{m["ci95"][1]:.4f}]'
                d = a['descriptors']
                lines.append(f'{name:<10} {t:>4} {fmt(a["valid"]):<24} {fmt(a["clean"]):<24} {a["valid_and_clean"]["estimate"]:.4f}   {fmt(d["away"]["median_distance"]):<25} {fmt(d["away"]["floor_fraction"])}')
                lines.append(f'  task {t} nontrap: VALID={a["nonaway_valid"]["rate"]:.4f}; median={fmt(d["nonaway"]["median_distance"])}; floor={fmt(d["nonaway"]["floor_fraction"])}')
                lines.append(f'  task {t} failures: '+json.dumps(a['failures'], ensure_ascii=False))
        lines.append(f'  Stage O {arm["gates"]}; straight MSE={arm["straight"]["mse"]:.12g}; equal C1p={arm["straight_equal_to_C1p"]}')
        for c in arm['guards']['checks']:
            lines.append(f'  {c["status"]} {c["name"]}: {c["value"]:.8f} <= {c["limit"]:.8f}')
    lines.append('\nSeed noise (C1p, C1p-s34, C1p-s35):')
    for t, row in result['noise']['by_task'].items():
        for k, ns in row.items():
            lines.append(f'task {t} {k:<25} R={ns["R"]:.10f} raw={ns["raw_range"]:.10f} N={ns["n"]} fallback={ns["fallback"]} min={ns["min"]:.6f} max={ns["max"]:.6f}')
    lines.append('straight MSE noise: '+json.dumps(result['noise']['straight_mse']))
    lines.append('\nPaired differences vs C1p:')
    for name, tasks in result['differences'].items():
        for t, ms in tasks.items():
            lines.append(name+' task '+t+': '+ '; '.join(k+' '+str(ms[k]) for k in ('valid', 'clean')))
    lines.extend(result['warnings'])
    lines.append(result['interpretation']['reason'])
    for name, tasks in result['interpretation']['by_arm'].items():
        for t, row in tasks.items():
            lines.append(name+' task '+t+': '+row['conclusion'])
            if 'descriptor_checks' in row:
                lines.append('  '+json.dumps(row['descriptor_checks'], ensure_ascii=False))
    lines.extend(result['notes'])
    return '\n'.join(lines)+'\n'


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    names = [n for n, a in result['arms'].items() if a['has_draws']]
    width, x = .8/len(names), np.arange(3)
    for ax, key, title in zip(axs.flat, ('valid', 'clean', 'median_distance', 'floor_fraction'),
            ('Trap VALID', 'Path clean (tau=.0664)', 'Trap median landing distance', 'Trap floor fraction [5.65,6.5]')):
        if key == 'median_distance':
            for j, t in enumerate(TASKS):
                ns = result['noise']['by_task'][t]['away_median_distance']
                ax.fill_between([j-.45, j+.45], ns['min']-ns['R'], ns['max']+ns['R'], color='grey', alpha=.25)
        for i, name in enumerate(names):
            ts = result['arms'][name]['tasks']
            vs = [ts[t][key] if key in ('valid', 'clean') else ts[t]['descriptors']['away'][key] for t in TASKS]
            pos = x-.4+width*(i+.5)
            ax.bar(pos, [v['estimate'] for v in vs], width=width, label=name)
            ax.vlines(pos, [v['ci95'][0] for v in vs], [v['ci95'][1] for v in vs], color='black')
            for end in (0, 1):
                ax.hlines([v['ci95'][end] for v in vs], pos-width*.15, pos+width*.15, color='black')
        ax.set_xticks(x, ['Task 2 (description)', 'Task 4', 'Task 5'])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        if key != 'median_distance':
            ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=.2)
    axs[0, 0].legend()
    missing = [n for n, a in result['arms'].items() if not a['has_draws']]
    fig.suptitle('wallpen v2 | '+('SUBSTITUTE ONLY | ' if result['synthetic'] else '')+'95% two-level intervals'+(' | no draws: '+','.join(missing) if missing else ''))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, metavar='C1p=STAGEO,MODEL')
    p.add_argument('--seed', action='append', required=True, metavar='C1p-s34=STAGEO,MODEL')
    p.add_argument('--arm', action='append', required=True, metavar='C3-b=STAGEO,MODEL')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=M.DATASET)
    args = p.parse_args()
    env = None
    try:
        inputs, sources = {}, {}
        for flag, values, allowed in (('base', [args.base], ('C1p',)), ('seed', args.seed, SEEDS[1:]), ('arm', args.arm, ('C3-b', 'C3-a'))):
            for value in values:
                name, files = value.split('=', 1)
                sp, mp = map(Path, files.split(',', 1))
                if name not in allowed or name in inputs:
                    raise ValueError('Invalid/duplicate '+flag+' '+name)
                inputs[name] = (json.loads(sp.read_text()), json.loads(mp.read_text()))
                sources[name] = dict(stageo_path=str(sp), stageo_sha256=M.sha_file(sp), model_path=str(mp), model_sha256=M.sha_file(mp))
        geo, env = M.geometry_from_dataset(args.dataset)
        result = analyze(inputs, geo)
        result['sources'] = sources
        args.outdir.mkdir(parents=True, exist_ok=True)
        M.write_json(args.outdir/'wallpen2-analysis.json', result)
        (args.outdir/'wallpen2-analysis.txt').write_text(render(result))
        plot(result, args.outdir/'wallpen2-analysis.png')
        print(render(result), end='')
    except Exception as e:
        p.exit(2, f'BLOCKED: {type(e).__name__}: {e}\n')
    finally:
        if env:
            env.close()


if __name__ == '__main__':
    main()
