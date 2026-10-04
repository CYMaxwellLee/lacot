"""Ten v2 tests. Tested code is executed only in fresh subprocesses.

Mutants are source copies executed in subprocesses; no preload or monkeypatch.
Synthetic decision aggregates are marked separately from measured JSON fixtures.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import math
from fractions import Fraction
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import traceback
import numpy as np
import analyze_wallpen as V1

HERE = Path(__file__).resolve().parent
TARGET = HERE/'wallpen2_analyze.py'
BG = V1.M.BACKGROUND
SEEDS = ('C1p', 'C1p-s34', 'C1p-s35')
TASKS = V1.TASKS
NOTE = '訓練雜訊與門檻同量級'


def worker(candidate, payload, output):
    # Execute the exact requested source in this new process, including mutant files.
    # No tested module is imported by the parent, cached, or substituted in sys.modules.
    mod = runpy.run_path(str(candidate))
    data = json.loads(payload.read_text())
    arms = data['arms']
    noise = mod['noise_report'](arms)
    for arm in arms.values():
        arm['guards'] = mod['guards'](arm, arms['C1p'], noise)
    differences = {n+'-C1p': {t: {k: dict(estimate=arms[n]['tasks'][t][k]['estimate']-arms['C1p']['tasks'][t][k]['estimate'])
                    for k in ('valid', 'clean')} for t in TASKS} for n in ('C3-b', 'C3-a') if arms[n]['has_draws']}
    for pair, ts in data.get('difference_overrides', {}).items():
        for t, metrics in ts.items():
            differences[pair][t].update({k: dict(estimate=v) for k, v in metrics.items()})
    result = dict(noise=noise, arms=arms, differences=differences,
                  interpretation=mod['interpret'](arms, differences, noise))
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))


class Suite:
    def __init__(self):
        self.tests = self.killers = self.checks = self.serial = 0
        self.tmp = Path(tempfile.mkdtemp(prefix='wallpen2_selftest_'))
        print('wallpen2-eval-v1 | CPU | substitute data + explicitly synthetic branch fixtures', flush=True)
        print('Artifacts retained (no deletion): '+str(self.tmp), flush=True)

    def check(self, condition, text):
        self.checks += 1
        assert condition, text

    def passed(self, text):
        self.tests += 1
        print(text+' PASS', flush=True)

    def mutant(self, label, old, new):
        source = TARGET.read_text()
        self.check(source.count(old) == 1, 'unique mutation site: '+label)
        path = self.tmp/('wallpen2_mutant_'+label+'.py')
        path.write_text(source.replace(old, new))
        return path

    def logic(self, arms, candidate=TARGET, overrides=None):
        self.serial += 1
        payload, output = self.tmp/f'wallpen2_payload_{self.serial}.json', self.tmp/f'wallpen2_result_{self.serial}.json'
        payload.write_text(json.dumps(dict(arms=arms, difference_overrides=overrides or {}), allow_nan=False))
        cp = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(candidate),
                             '--payload', str(payload), '--output', str(output)], capture_output=True, text=True)
        if cp.returncode:
            raise AssertionError('Worker exit '+str(cp.returncode)+'\n'+cp.stdout+cp.stderr)
        return json.loads(output.read_text())

    def cli(self, bindings, candidate=TARGET, outdir=None):
        self.serial += 1
        outdir = outdir or self.tmp/f'wallpen2_cli_{self.serial}'
        cmd = [sys.executable, '-B', str(candidate)]
        for n in SEEDS+('C3-b', 'C3-a'):
            sp, mp = bindings[n]
            flag = '--base' if n == 'C1p' else '--seed' if n in SEEDS else '--arm'
            cmd.extend([flag, n+'='+str(sp)+','+str(mp)])
        cmd.extend(['--outdir', str(outdir)])
        env = os.environ.copy()
        env['PYTHONPATH'] = str(HERE)+os.pathsep+env.get('PYTHONPATH', '')
        cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (self.tmp/f'wallpen2_cli_{self.serial}_output.txt').write_text(cp.stdout+cp.stderr)
        if cp.returncode:
            raise AssertionError('CLI exit '+str(cp.returncode)+'\n'+cp.stdout+cp.stderr)
        return json.loads((outdir/'wallpen2-analysis.json').read_text())

    def kill(self, label, baseline, mutant):
        baseline()  # Same invariant must pass the original before mutation.
        try:
            mutant()
        except AssertionError as e:
            self.killers += 1
            print(f'KILLER {label}: mutant FAIL (expected): {e}', flush=True)
        else:
            raise AssertionError('Killer survived: '+label)


def fixture_arms(measured):
    """Explicit synthetic aggregates, never reported as measured/bootstrap evidence."""
    a = copy.deepcopy(measured['arms']['C1p'])
    a['synthetic'] = True
    a['gates'] = dict(G1='PASS', G2='PASS', G3='PASS')
    a['gate_pass'] = a['has_draws'] = True
    for t in TASKS:
        q = a['tasks'][t]
        for k, v in (('valid', .2), ('clean', .2), ('near', .1)):
            q[k]['estimate'] = v
        q['nonaway_valid']['rate'] = .8
        q['descriptors']['away']['median_distance']['estimate'] = 8.
        q['descriptors']['away']['floor_fraction']['estimate'] = .1
        q['failures'] = failure([30, 20, 10, 40])
    return {n: copy.deepcopy(a) for n in SEEDS+('C3-b', 'C3-a')}


def failure(counts):
    total = sum(counts)
    return dict(n=total, counts=dict(zip(V1.CLASSES, counts)),
                proportions=dict(zip(V1.CLASSES, [n/total for n in counts])) if total else dict.fromkeys(V1.CLASSES))


def recount(o):
    for t in TASKS:
        for away, label in ((True, 'away'), (False, 'nonaway')):
            ds = [d for c in o['cells'] if str(c['task']) == t and c['away'] == away for d in c['flow']]
            s = o['summary'][t][label]
            s.update(n=len(ds), counts={cat: sum(d['cat'] == cat for d in ds) for cat in V1.CATS},
                     valid_rate=sum(d['cat'] == 'VALID' for d in ds)/len(ds))
        o['summary'][t]['r'] = o['summary'][t]['away']['valid_rate']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--payload', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.worker:
        worker(args.worker, args.payload, args.output)
        return 0
    s, env, status = Suite(), None, 'BLOCKED'
    try:
        stages = {n: BG/'wallpen/stageO'/f'{n}.json' for n in ('C1', 'C2lo', 'C2hi', 'B0s')}
        stages['H3'] = BG/'hsweep/stageO/H3.json'
        models = {n: BG/'wallpen/model'/f'{n}-model.json' for n in ('C1', 'C2lo', 'C2hi', 'B0s')}
        models['H3'] = HERE/'H3-model.json'
        bindings = {n: (stages[k], models[k]) for n, k in zip(SEEDS+('C3-b', 'C3-a'), ('C1', 'H3', 'B0s', 'H3', 'C2lo'))}
        measured = s.cli(bindings, outdir=HERE/'wallpen2_example')
        geo, env = V1.M.geometry_from_dataset()
        raw = {n: json.loads(path.read_text()) for n, path in stages.items()}
        reference, arrays = {}, {}
        for n in ('C1', 'H3', 'C2lo', 'B0s'):
            reference[n], arrays[n] = V1.arm_metrics(raw[n], json.loads(models[n].read_text()), geo)
        # V1: compare every imported metric and classifier, then all paired deltas.
        for name, old in (('C1p', 'C1'), ('C3-b', 'H3'), ('C3-a', 'C2lo')):
            for t in TASKS:
                q = measured['arms'][name]['tasks'][t]
                for k in ('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near', 'failures', 'per_draw'):
                    s.check(q[k] == reference[old]['tasks'][t][k], f'V1 exact {name} task {t} {k}')
                if name != 'C1p':
                    for i, k in enumerate(('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near')):
                        expected = V1.bootstrap(arrays[old][t][:, :, i], arrays['C1'][t][:, :, i])
                        s.check(measured['differences'][name+'-C1p'][t][k] == expected, f'V1 exact paired {name} {t} {k}')
        s.passed('V1 v1 exact metrics / paired intervals / four classes')
        # V2: independent raw draw calculation; never calls tested descriptors.
        for name, old in (('C3-b', 'H3'), ('C3-a', 'C2lo')):
            for t in TASKS:
                for away, label in ((True, 'away'), (False, 'nonaway')):
                    distances = [np.sqrt(sum((float(d['p_xy'][j])-float(d['smooth_anchored_xy'][0][j]))**2 for j in (0, 1)))
                                 for c in raw[old]['cells'] if str(c['task']) == t and c['away'] == away for d in c['flow']]
                    med = float(np.median(distances))
                    floor = sum(5.65 <= x <= 6.5 for x in distances)/len(distances)
                    got = measured['arms'][name]['tasks'][t]['descriptors'][label]
                    s.check(got['median_distance']['estimate'] == med and got['floor_fraction']['estimate'] == floor,
                            f'V2 independent {old} task {t} {label}')
        s.passed('V2 independent H3/C2lo distances and inclusive floor fraction')
        # V3: independent extrema for every measured quantity, including 1/N.
        def hand_quant(stage, task, t):
            ds = {label: [d for c in stage['cells'] if str(c['task']) == t and c['away'] == away for d in c['flow']]
                  for away, label in ((True, 'away'), (False, 'nonaway'))}
            q = {k: task[k]['estimate'] for k in ('valid', 'clean', 'valid_and_clean', 'true_wall_entry', 'near')}
            q['nonaway_valid'] = sum(d['cat'] == 'VALID' for d in ds['nonaway'])/len(ds['nonaway'])
            for label, draws in ds.items():
                distances = [float(np.linalg.norm(np.asarray(d['p_xy'])-np.asarray(d['smooth_anchored_xy'])[0])) for d in draws]
                q[label+'_median_distance'] = float(np.median(distances))
                q[label+'_floor_fraction'] = sum(5.65 <= d <= 6.5 for d in distances)/len(distances)
            return q
        for t in TASKS:
            qs = [hand_quant(raw[n], reference[n]['tasks'][t], t) for n in ('C1', 'H3', 'B0s')]
            for k, ns in measured['noise']['by_task'][t].items():
                vals = [q[k] for q in qs]
                raw_r = max(vals)-min(vals)
                s.check(ns['values'] == vals and ns['raw_range'] == raw_r and ns['R'] == (raw_r if raw_r else 1/ns['n']), f'V3 hand R {t} {k}')
        same = dict.fromkeys(SEEDS+('C3-b', 'C3-a'), (stages['C1'], models['C1']))
        zero = s.cli(same)
        for row in zero['noise']['by_task'].values():
            s.check(all(ns['R'] == 1/ns['n'] and ns['raw_range'] == 0 and ns['fallback'] for ns in row.values()), 'V3 identical JSON R=1/N')
        s.check(zero['noise']['straight_mse']['R'] == 1/2048, 'V3 MSE zero fallback')
        s.passed('V3 hand seed ranges and three identical JSON 1/N fallback')
        # V4/V5 raw fixture: all trap endpoints 5.8 and VALID, summaries updated.
        floor_raw = copy.deepcopy(raw['C1'])
        for c in floor_raw['cells']:
            if c['away']:
                for d in c['flow']:
                    start = d['smooth_anchored_xy'][0]
                    d['p_xy'] = [start[0]+5.8, start[1]]
                    d['cat'] = 'VALID'
        recount(floor_raw)
        floor_path = s.tmp/'wallpen2_floor_stage.json'
        floor_path.write_text(json.dumps(floor_raw))
        constructed = dict(same, **{'C3-b': (floor_path, models['C1'])})
        floor_result = s.cli(constructed)
        for t in TASKS:
            s.check(floor_result['interpretation']['by_arm']['C3-b'][t]['descriptor_checks']['squeezed'], 'V4 all endpoints 5.8 squeezed '+t)
        a = fixture_arms(measured)
        for t in TASKS:
            for n, fraction in zip(SEEDS, (.1, .2, .3)):
                a[n]['tasks'][t]['descriptors']['away']['floor_fraction']['estimate'] = fraction
            a['C3-b']['tasks'][t]['descriptors']['away']['floor_fraction']['estimate'] = .5
        def boundary(candidate=TARGET):
            r = s.logic(a, candidate)
            s.check(not r['interpretation']['by_arm']['C3-b']['4']['descriptor_checks']['squeezed'], 'V4 .5 <= max(.1,.2,.3)+2*.2=.7; must not compare C1p alone')
        mut = s.mutant('V4_floor_reference', 'squeezed = floor > limit + 1e-12', "squeezed = floor > ns['away_floor_fraction']['values'][0]")
        s.kill('V4_floor_reference', boundary, lambda: boundary(mut))
        # Strict > at the cutoff and inclusive distance band boundaries.
        for value, expected in ((.7, False), (.700001, True)):
            b = copy.deepcopy(a)
            b['C3-b']['tasks']['4']['descriptors']['away']['floor_fraction']['estimate'] = value
            r = s.logic(b)
            s.check(r['interpretation']['by_arm']['C3-b']['4']['descriptor_checks']['squeezed'] == expected, 'V4 strict floor boundary')
        s.passed('V4 raw all-5.8 fixture / max+2R boundary / killer')
        for t in ('4', '5'):
            row = floor_result['interpretation']['by_arm']['C3-b'][t]
            s.check(row['valid_delta'] >= .15 and row['conclusion'].startswith('VALID 上來了，但步長分布塌在下限／仍在縮'), 'V5 raw VALID gain must still fail descriptor '+t)
        def collapsed(candidate=TARGET):
            b = fixture_arms(measured)
            for t in ('4', '5'):
                b['C3-b']['tasks'][t]['valid']['estimate'] = .4
                b['C3-b']['tasks'][t]['descriptors']['away']['floor_fraction']['estimate'] = 1.
            r = s.logic(b, candidate)
            s.check(all(r['interpretation']['by_arm']['C3-b'][t]['conclusion'].startswith('VALID 上來了，但') for t in ('4', '5')), 'V5 successful VALID needs both descriptors')
        mut = s.mutant('V5_success_descriptors', "if checks['in_band'] and not checks['squeezed']:", 'if True:')
        s.kill('V5_success_descriptors', collapsed, lambda: collapsed(mut))
        # Exhaustive decision paths and task-local annotations on independent aggregates.
        for t in ('4', '5'):
            for dv, dc, med, floor, expected in (
                (.15, 0, 8., .1, '不碰牆又走得遠的計畫學得到'),
                (.15, 0, 7., .1, 'VALID 上來了，但'),
                (.15, 0, 9., .1, 'VALID 上來了，但'),
                (.149, .30, 8., 1., '學成最小步'),
                (.149, .30, 8., .1, '走錯邊為主'),
                (.149, .299, 8., .1, '懲罰沒教到牆')):
                b = fixture_arms(measured)
                q = b['C3-b']['tasks'][t]
                q['descriptors']['away']['median_distance']['estimate'] = med
                q['descriptors']['away']['floor_fraction']['estimate'] = floor
                overrides = {'C3-b-C1p': {t: dict(valid=dv, clean=dc)}}
                r = s.logic(b, overrides=overrides)
                row = r['interpretation']['by_arm']['C3-b'][t]
                s.check(row['conclusion'].startswith(expected), f'V5 task {t} branch {dv}/{dc}/{med}/{floor}')
                s.check(('；仍在縮' in row['conclusion']) == (med == 7.), 'V5 shrinking only below lower band')
                s.check(r['interpretation']['by_task']['2']['conclusion'].startswith('task 2 只描述'), 'V5 task 2 description')
            for counts, base, expected in (([20, 10, 60, 10], [30, 20, 10, 40], '教成原地踏步'),
                ([60, 10, 10, 20], [30, 20, 10, 40], '方向對只是不準為主'),
                ([10, 60, 10, 20], [30, 20, 10, 40], '分不清為主'),
                ([10, 20, 40, 30], [30, 20, 30, 20], '太近沒前進為主'),
                ([40, 10, 10, 40], [30, 20, 10, 40], '待討論：最大類並列'),
                ([0, 0, 0, 0], [30, 20, 10, 40], '待討論：無失敗樣本')):
                b = fixture_arms(measured)
                b['C3-b']['tasks'][t]['failures'] = failure(counts)
                b['C1p']['tasks'][t]['failures'] = failure(base)
                r = s.logic(b, overrides={'C3-b-C1p': {t: dict(valid=0., clean=.30)}})
                s.check(r['interpretation']['by_task'][t]['conclusion'].startswith(expected), 'V5 fourclass '+expected)
        b = fixture_arms(measured)
        b['C1p-s34']['tasks']['4']['valid']['estimate'] = .3
        r = s.logic(b)
        s.check(NOTE in r['interpretation']['reason'] and all(NOTE in row['conclusion'] for tasks in r['interpretation']['by_arm'].values() for row in tasks.values()), 'V5 >=.10 annotates every conclusion')
        b['C1p-s34']['tasks']['4']['valid']['estimate'] = .299
        s.check(not s.logic(b)['noise']['same_scale'], 'V5 below .10 no noise note')
        s.passed('V5 success descriptors / every decision and class branch / task-local shrinking / noise note / killer')
        # V6: the real G2 FAIL JSON with empty flow must succeed through the CLI.
        no_draw = dict(same, **{'C3-b': (stages['C2hi'], models['C2hi'])})
        def fallback(candidate=TARGET):
            r = s.cli(no_draw, candidate, HERE/'wallpen2_example_nodraw' if candidate == TARGET else None)
            s.check(r['interpretation']['primary'] == 'C3-a' and 'G2' in r['interpretation']['reason'] and not r['arms']['C3-b']['has_draws'], 'V6 real G2 fail => C3-a; no exit or fabricated draw')
            s.check(r['interpretation']['b_primary_metrics'] is None, 'V6 no draw metrics remain null')
        fallback()
        mut = s.mutant('V6_nodraw_exit', 'validate_no_draw(o, base_o)', "raise ValueError('mutant: no draw exits')")
        s.kill('V6_nodraw_exit', fallback, lambda: fallback(mut))
        # All gates, both guard kinds and exact 2R edges at every task.
        for t in TASKS:
            for metric in ('nonaway_valid', 'near'):
                b = fixture_arms(measured)
                q = b['C3-b']['tasks'][t]
                n = q['nonaway_valid']['n'] if metric == 'nonaway_valid' else q['near']['n']
                if metric == 'nonaway_valid':
                    q['nonaway_valid']['rate'] = .8-2/n
                else:
                    q['near']['estimate'] = .1+2/n
                s.check(s.logic(b)['interpretation']['primary'] == 'C3-b', 'V6 equality 2R guard PASS')
                if metric == 'nonaway_valid':
                    q['nonaway_valid']['rate'] -= 1e-6
                else:
                    q['near']['estimate'] += 1e-6
                r = s.logic(b)
                s.check(r['interpretation']['primary'] == 'C3-a' and 'task '+t in r['interpretation']['reason'], 'V6 >2R guard fallback '+metric+' '+t)
                b['C3-a']['tasks'][t] = copy.deepcopy(q)
                s.check(s.logic(b)['interpretation']['primary'] is None, 'V6 both guards FAIL no conclusion')
        for g in ('G1', 'G2', 'G3'):
            b = fixture_arms(measured)
            b['C3-b']['gates'][g] = 'FAIL'
            b['C3-b']['gate_pass'] = False
            r = s.logic(b)
            s.check(r['interpretation']['primary'] == 'C3-a' and g in r['interpretation']['reason'], 'V6 gate fallback '+g)
            b['C3-a']['gates'][g] = 'FAIL'
            b['C3-a']['gate_pass'] = False
            s.check(s.logic(b)['interpretation']['primary'] is None, 'V6 both gates FAIL')
        # Existing substitute decoder mismatch must be warning only.
        s.check(any('警告' in w and '直路 MSE 不等於 C1p' in w for w in measured['warnings']),
                'V6 straight mismatch emits warning')
        def straight_excluded(result):
            allowed = {f'{label} 守門 task {t}' for t in TASKS for label in ('非陷阱 VALID', '陷阱 NEAR')}
            s.check(all(
                set(a['guards']['failed']) <= allowed and
                all(c['metric'] in ('nonaway_valid', 'near') for c in a['guards']['checks']) and
                a['guards']['pass_all'] == (not a['guards']['failed'])
                for a in result['arms'].values()),
                'V6 straight MSE excluded from failed guards and pass_all')
        straight_excluded(measured)
        def straight_guard(candidate=TARGET):
            straight_excluded(s.logic(measured['arms'], candidate))
        mut = s.mutant('V6_straight_mse_guard', 'checks = []\n    for t in TASKS:',
                       "checks = [dict(metric='straight_mse', name='直路 MSE', "
                       "status='PASS' if arm['straight']['mse'] == base['straight']['mse'] else 'FAIL')]\n    for t in TASKS:")
        s.kill('V6_straight_mse_guard', straight_guard, lambda: straight_guard(mut))
        s.passed('V6 real no-draw G2 CLI / every gate and task guard / both fail / warning-only MSE / killer')
        # V7: same gate/guard status, high and low VALID; b always remains primary.
        def ignores_valid(candidate=TARGET):
            selections = []
            for rate in (.9, .01):
                b = fixture_arms(measured)
                for t in TASKS:
                    b['C3-b']['tasks'][t]['valid']['estimate'] = rate
                    b['C3-a']['tasks'][t]['valid']['estimate'] = .5
                r = s.logic(b, candidate)
                selections.append(r['interpretation']['primary'])
                s.check(r['interpretation']['b_primary_metrics']['4']['estimate'] == rate, 'V7 b metrics still reported')
            s.check(selections == ['C3-b', 'C3-b'], 'V7 selection must ignore VALID high/low; actual='+str(selections))
        ignores_valid()
        mut = s.mutant('V7_metric_selection', 'def select_primary(arms):\n',
                       "def select_primary(arms):\n    return max(('C3-b', 'C3-a'), key=lambda n: arms[n]['tasks']['4']['valid']['estimate']), 'mutant: chooses VALID'\n")
        s.kill('V7_metric_selection', ignores_valid, lambda: ignores_valid(mut))
        missing_outdir = subprocess.run([sys.executable, '-B', str(TARGET), '--base', 'C1p=unread,unread'], capture_output=True, text=True)
        s.check(missing_outdir.returncode == 2 and '--outdir' in missing_outdir.stderr, 'V7 outdir mandatory')
        s.passed('V7 high/low VALID cannot change primary / b metrics retained / mandatory outdir / killer')
        # V8: integer counts with an independent exact rational cutoff. Both
        # decision tasks must keep equality learnable and reject the next count.
        def count_boundaries(candidate=TARGET):
            for n, seed_counts, cutoff_count in ((80, (0, 0, 3), 9),
                                                 (80, (7, 7, 7), 9),
                                                 (112, (0, 0, 3), 9)):
                raw_count = max(seed_counts)-min(seed_counts)
                # R=0 falls back to 1/N: (7 + 2*1)/80 = 9/80.
                exact_r = Fraction(raw_count or 1, n)
                exact_limit = Fraction(max(seed_counts), n) + 2*exact_r
                s.check(exact_limit == Fraction(cutoff_count, n), 'V8 hand count cutoff')
                for arm_count in (cutoff_count-1, cutoff_count, cutoff_count+1):
                    b = fixture_arms(measured)
                    counts = dict(zip(SEEDS, seed_counts), **{'C3-b': arm_count, 'C3-a': arm_count})
                    for name, count in counts.items():
                        for t in TASKS:
                            q = b[name]['tasks'][t]
                            q['descriptors']['away']['floor_fraction'].update(estimate=count/n, count=count, n=n)
                            valid_count = n//4 if name in SEEDS else n//2
                            q['valid'].update(estimate=valid_count/n, count=valid_count, n=n)
                    r = s.logic(b, candidate)
                    expected = Fraction(arm_count, n) > exact_limit
                    case = f'n={n} seeds={seed_counts} arm={arm_count}'
                    for t in ('4', '5'):
                        ns = r['noise']['by_task'][t]['away_floor_fraction']
                        s.check(ns['n'] == n and ns['values'] == [c/n for c in seed_counts] and
                                ns['fallback'] == (raw_count == 0) and abs(ns['R']-float(exact_r)) <= 1e-12,
                                'V8 count-derived noise '+case+' task '+t)
                        for name in ('C3-b', 'C3-a'):
                            row = r['interpretation']['by_arm'][name][t]
                            s.check(row['descriptor_checks']['squeezed'] == expected,
                                    'V8 exact strict floor boundary '+case+' '+name+' task '+t)
                            conclusion = 'VALID 上來了，但' if expected else '不碰牆又走得遠的計畫學得到'
                            s.check(row['conclusion'].startswith(conclusion), 'V8 conclusion '+case+' '+name+' task '+t)
                    print('V8 counts '+case+f': squeezed={expected} PASS', flush=True)
            # Distance band in exact 1/80 units: [639/80, 642/80].
            # Computing the extrema and R in floats rounds both edges inward.
            for med_count, expected_band, expected_shrinking in ((638, False, True), (639, True, False),
                                                                 (642, True, False), (643, False, False)):
                b = fixture_arms(measured)
                for t in TASKS:
                    for name, count in zip(SEEDS, (640, 640, 641)):
                        b[name]['tasks'][t]['descriptors']['away']['median_distance'].update(estimate=count/80, n=80)
                    for name in ('C3-b', 'C3-a'):
                        q = b[name]['tasks'][t]
                        q['valid']['estimate'] = 32/80
                        q['descriptors']['away']['median_distance'].update(estimate=med_count/80, n=80)
                r = s.logic(b, candidate)
                for t in ('4', '5'):
                    for name in ('C3-b', 'C3-a'):
                        row = r['interpretation']['by_arm'][name][t]
                        checks = row['descriptor_checks']
                        s.check(checks['in_band'] == expected_band and checks['shrinking'] == expected_shrinking,
                                f'V8 distance band {med_count}/80 {name} task {t}')
                        s.check(row['conclusion'].startswith('不碰牆又走得遠的計畫學得到' if expected_band else 'VALID 上來了，但') and
                                ('；仍在縮' in row['conclusion']) == expected_shrinking, 'V8 distance band conclusion')
                print(f'V8 distance {med_count}/80: in_band={expected_band} shrinking={expected_shrinking} PASS', flush=True)
        mut = s.mutant('V8_floor_tolerance', 'squeezed = floor > limit + 1e-12', 'squeezed = floor > limit')
        s.kill('V8_floor_tolerance', count_boundaries, lambda: count_boundaries(mut))
        s.passed('V8 count-derived n=80/n=112 equality / 1/N fallback / next count / tolerance killer')
        # V9: independent route arithmetic on H3, plus explicit route fixtures.
        # Never import or call the candidate's subset/descriptors in this process.
        def independent_subset(stage):
            selected, displacements = {t: [] for t in TASKS}, {}
            for c in stage['cells']:
                if not c['away']:
                    continue
                t, start = str(c['task']), c['cell']
                ds = []
                for path in stage['routes'][t]['paths']:
                    if start in path:
                        end = path[min(path.index(start)+3, len(path)-1)]
                        ds.append(math.hypot(end[0]-start[0], end[1]-start[1])*4.0)
                s.check(bool(ds), 'V9 independent route coverage')
                displacements[t, tuple(start)] = ds
                if min(ds) >= 5.65:
                    selected[t].append(start)
            return selected, displacements

        def subset_matches(stage, result, txt):
            expected, ds = independent_subset(stage)
            s.check(result['floor_subset_cells'] == expected, 'V9 exact independent subset list')
            for t in TASKS:
                s.check(f'task {t} floor_subset_cells: '+json.dumps(expected[t]) in txt,
                        'V9 TXT subset list '+t)
            return ds

        h3_only = dict.fromkeys(SEEDS+('C3-b', 'C3-a'), (stages['H3'], models['H3']))
        h3_out = s.tmp/'wallpen2_v9_h3'
        h3_result = s.cli(h3_only, outdir=h3_out)
        ds = subset_matches(raw['H3'], h3_result, (h3_out/'wallpen2-analysis.txt').read_text())
        print('V9 H3 independent per-trap 3-step displacements: '+str(ds), flush=True)

        # Explicit synthetic routes: the first route for [3,8] qualifies, but
        # the second has a three-step U prefix ending at [3,9], displacement 4.
        # These route fixtures test the JSON contract, not measured maze routes.
        u_raw = copy.deepcopy(raw['C1'])
        p2 = u_raw['routes']['4']['paths'][1]
        u_raw['routes']['4']['paths'][1] = [[3,8], [4,8], [4,9], [3,9]]+p2[2:]
        for c in u_raw['cells']:
            if c['away']:
                for d in c['flow']:
                    start = d['smooth_anchored_xy'][0]
                    d['p_xy'] = [start[0]+8., start[1]]
        u_path = s.tmp/'wallpen2_v9_u.json'
        u_path.write_text(json.dumps(u_raw))
        u_bindings = dict.fromkeys(SEEDS+('C3-b', 'C3-a'), (u_path, models['C1']))
        u_arm = copy.deepcopy(u_raw)
        # Excluded cell: all sixteen endpoints at 5.8. Included cells: exactly
        # two endpoints at 5.8. Thus the subset cutoff is exactly 2/64.
        for c in u_arm['cells']:
            if str(c['task']) == '4' and c['away']:
                for d in (c['flow'] if c['cell'] == [3,8] else c['flow'][:2] if c['cell'] == [2,8] else []):
                    start = d['smooth_anchored_xy'][0]
                    d['p_xy'] = [start[0]+5.8, start[1]]
        u_arm_path = s.tmp/'wallpen2_v9_u_arm.json'
        u_arm_path.write_text(json.dumps(u_arm))
        u_bindings['C3-b'] = (u_arm_path, models['C1'])

        def u_invariant(candidate=TARGET):
            out = s.tmp/('wallpen2_v9_u_'+candidate.stem)
            result = s.cli(u_bindings, candidate, out)
            ds = subset_matches(u_raw, result, (out/'wallpen2-analysis.txt').read_text())
            s.check(ds['4', (3,8)] == [math.sqrt(80), 4.0] and
                    [3,8] not in result['floor_subset_cells']['4'], 'V9 one U route excludes cell despite qualifying other route')
            for name in SEEDS+('C3-b', 'C3-a'):
                floor = result['arms'][name]['tasks']['4']['descriptors']['away']['floor_fraction']
                s.check(floor['n'] == 64 and floor['count'] == (2 if name == 'C3-b' else 0),
                        'V9 all arms/seeds share subset denominator/count '+name)
            ns = result['noise']['by_task']['4']['away_floor_fraction']
            checks = result['interpretation']['by_arm']['C3-b']['4']['descriptor_checks']
            s.check(ns['n'] == 64 and ns['R'] == 1/64 and ns['fallback'] and
                    checks['floor_fraction'] == 2/64 and checks['floor_limit'] == 2/64 and not checks['squeezed'],
                    'V9 subset R=1/64 and max+2R equality')
            return result

        u_result = u_invariant()
        mut = s.mutant('V9_all_traps',
                       "if all(o['ruler']['cell_size'] * np.linalg.norm(np.asarray(end)-start) >= 5.65 for end in targets):",
                       'if True:')
        s.kill('V9_all_traps', u_invariant, lambda: u_invariant(mut))
        next_raw = copy.deepcopy(u_arm)
        cell = next(c for c in next_raw['cells'] if str(c['task']) == '4' and c['cell'] == [2,8])
        d = cell['flow'][2]
        d['p_xy'] = [d['smooth_anchored_xy'][0][0]+5.8, d['smooth_anchored_xy'][0][1]]
        next_path = s.tmp/'wallpen2_v9_next.json'
        next_path.write_text(json.dumps(next_raw))
        next_result = s.cli(dict(u_bindings, **{'C3-b': (next_path, models['C1'])}))
        checks = next_result['interpretation']['by_arm']['C3-b']['4']['descriptor_checks']
        s.check(checks['floor_fraction'] == 3/64 and checks['squeezed'], 'V9 subset next count squeezed')
        # Routes shorter than three steps use their last cell: one step is 4,
        # two perpendicular steps are sqrt(32), on the >=5.65 side.
        short_raw = copy.deepcopy(u_raw)
        short_raw['routes']['4']['paths'] += [[[3,8], [3,9]], [[3,9], [3,10], [4,10]],
                                             [[2,8], [3,8], [3,9]]]
        short_path = s.tmp/'wallpen2_v9_short.json'
        short_path.write_text(json.dumps(short_raw))
        short_out = s.tmp/'wallpen2_v9_short'
        short_result = s.cli(dict.fromkeys(SEEDS+('C3-b', 'C3-a'), (short_path, models['C1'])), outdir=short_out)
        ds = subset_matches(short_raw, short_result, (short_out/'wallpen2-analysis.txt').read_text())
        s.check(ds['4', (3,8)][-1] == 4 and ds['4', (3,9)][-1] == 0 and
                math.sqrt(32) in ds['4', (3,9)] and [3,9] not in short_result['floor_subset_cells']['4'] and
                ds['4', (2,8)][-1] == math.sqrt(32) and [2,8] in short_result['floor_subset_cells']['4'],
                'V9 short/terminal routes use last cell')
        s.passed('V9 H3 independent subset / U all-routes exclusion / short routes / shared N,R,max+2R / killer')

        # V10: pinned original source, executed by the real CLI on identical
        # paths. Remove only floor-related fields; compare all remaining values.
        cp = subprocess.run(['git', 'show', 'eccece8:experiments/_workorders/wallpen_eval/wallpen2_analyze.py'],
                            cwd=HERE, capture_output=True, check=True)
        s.check(hashlib.sha256(cp.stdout).hexdigest() == '68038b856fc6a08dad16506f6444799d93dd7df110df56823876cea359aff8da',
                'V10 pinned 68038b85 original source')
        old_target = s.tmp/'wallpen2_analyze_r2_reference.py'
        old_target.write_bytes(cp.stdout)
        def without_floor(result):
            result = copy.deepcopy(result)
            result.pop('floor_subset_cells', None)
            for arm in result['arms'].values():
                for task in arm['tasks'].values():
                    task['descriptors']['away'].pop('floor_fraction')
            for task in result['noise']['by_task'].values():
                task.pop('away_floor_fraction')
            for tasks in (list(result['interpretation']['by_arm'].values())+[result['interpretation']['by_task']]):
                for row in tasks.values():
                    for k in ('floor_fraction', 'floor_limit', 'squeezed'):
                        row.get('descriptor_checks', {}).pop(k, None)
            return result
        for label, inputs, current, current_out in (
                ('measured', bindings, measured, HERE/'wallpen2_example'),
                ('U subset', u_bindings, u_result, s.tmp/'wallpen2_v9_u_wallpen2_analyze')):
            old_out = s.tmp/('wallpen2_v10_'+label.replace(' ', '_'))
            original = s.cli(inputs, old_target, old_out)
            s.check(without_floor(current) == without_floor(original), 'V10 every non-floor JSON value equal '+label)
            if label == 'measured':
                txt = (current_out/'wallpen2-analysis.txt').read_text()
                txt = '\n'.join(line for line in txt.split('\n') if ' floor_subset_cells: ' not in line)
                s.check(txt == (old_out/'wallpen2-analysis.txt').read_text(), 'V10 all original TXT values/lines unchanged on H3 subset')
                s.check((current_out/'wallpen2-analysis.png').read_bytes() == (old_out/'wallpen2-analysis.png').read_bytes(),
                        'V10 original PNG unchanged on H3 subset')
        s.passed('V10 pinned r2 CLI / all non-floor JSON values equal (real and U fixtures) / original TXT+PNG equal')
        s.check(s.tests == 10 and s.killers == 7, 'exactly ten tests and seven killed subprocess mutants')
        print('SECOND PASS: independent subsets, pinned r2 non-floor equality and all same-invariant subprocess killers passed', flush=True)
        status = 'DONE'
    except Exception:
        print(traceback.format_exc(), flush=True)
    finally:
        if env:
            env.close()
        print(f'n_tests={s.tests} n_killers={s.killers} n_checks={s.checks}', flush=True)
        print('STATUS: '+status, flush=True)
    return 0 if status == 'DONE' else 2


if __name__ == '__main__':
    sys.exit(main())
