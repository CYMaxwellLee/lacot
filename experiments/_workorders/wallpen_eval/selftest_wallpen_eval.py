"""Seven workorder tests and twelve deliberately failing mutant killers, CPU only.

First failure is recorded unchanged; do not adapt expected values to results.
Default recomputes the three specified checkpoints in separate subprocesses.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import json
import inspect
import hashlib
from types import SimpleNamespace
import subprocess
from pathlib import Path
import traceback
import numpy as np
import model_metrics as M
import analyze_wallpen as A

PINS = dict(H3='c42e4389a6f29188a9f8d5f1f3bbad732ccc8597bd26faae00d7fbef9d924d4f',
            H4='5bc82bf8d72b77d4d9649406224045b815e312d260b4682e0a63be726c5cc46a',
            H8='2304379c7c1408cdffd73b7f558c1acbd57dda2325d1a0f8ccd7df6d9f67ee15')


class Suite:
    def __init__(self):
        self.lines, self.checks, self.killers, self.tests = [], 0, 0, 0

    def emit(self, text):
        print(text, flush=True)
        self.lines.append(text)

    def check(self, condition, text):
        self.checks += 1
        if not condition:
            raise AssertionError(text)

    def rejected(self, call, text):
        try:
            call()
        except ValueError as e:
            self.check(True, text)
            self.emit(f'  refusal PASS: {text}: {e}')
        else:
            self.check(False, text)

    def killer(self, invariant, mutant, text):
        try:
            assert invariant(mutant), text
        except AssertionError as e:
            self.killers += 1
            self.emit(f'  KILLER {self.killers} mutant FAIL (expected): {type(e).__name__}: {e}; actual={mutant}')
        else:
            raise AssertionError(f'Killer did not FAIL: {text}')

    def mutation(self, owner, name, mutant, invariant, text):
        original = getattr(owner, name)
        invariant()  # The exact invariant must PASS before applying the mutation.
        try:
            setattr(owner, name, mutant)
            try:
                invariant()
            except AssertionError as e:
                self.killers += 1
                self.emit(f'  KILLER {self.killers} mutant FAIL (expected): {type(e).__name__}: {e}; {text}')
            else:
                raise AssertionError(f'Killer did not FAIL: {text}')
        finally:
            setattr(owner, name, original)

    def passed(self, name):
        self.tests += 1
        self.emit(name+' PASS')


def valid_array(o, t):
    return np.asarray([[d['cat'] == 'VALID' for d in c['flow']] for c in o['cells']
                       if str(c['task']) == t and c['away']], float)


def decision_diffs(arms):
    # Synthetic decision fixtures only: no bootstrap claim from edited aggregates.
    return {l+'-'+r: {t: {m: dict(estimate=arms[l]['tasks'][t][m]['estimate']-arms[r]['tasks'][t][m]['estimate'])
                          for m in ('valid', 'clean')} for t in A.TASKS}
            for l, r in A.PAIRS if l in arms and r in arms}


def recalc_guards(arms):
    A.assign_guards(arms)


def source_mutant(function, old, new):
    source = inspect.getsource(function)
    if old not in source:
        raise AssertionError(f'Mutation site missing: {old}')
    namespace = {}
    exec(source.replace(old, new), function.__globals__, namespace)
    return namespace[function.__name__]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reuse-models', action='store_true', help='reuse locally computed pinned H3/H4/H8 JSON; identity checked')
    args = p.parse_args()
    suite = Suite()
    env = None
    status = 'BLOCKED'
    try:
        suite.emit('wallpen-eval-v1 selftest | CPU | H3/H4/H8/Hinf substitutes only')
        for h, pin in PINS.items():
            ck = M.BACKGROUND/'hsweep/ckpt-local'/f'{h}.pt'
            out = M.HERE/f'{h}-model.json'
            suite.check(M.sha_file(ck) == pin, f'{h} checkpoint pin')
            if not args.reuse_models:
                cmd = [sys.executable, '-B', str(M.HERE/'model_metrics.py'), '--ckpt', str(ck),
                       '--ckpt-sha256', pin, '--out', str(out)]
                suite.emit(f'COMPUTE {h}: '+ ' '.join(cmd))
                with (M.HERE/f'{h}-model-output.txt').open('w') as log:
                    cp = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
                suite.check(cp.returncode == 0, f'{h} model_metrics process failed; see {h}-model-output.txt')
            suite.check(out.is_file(), f'{h} model JSON unavailable')
        # E7 repeat is ALWAYS fresh, including --reuse-models; never compare a deepcopy.
        repeat_path = M.HERE/'H3-repeat-model.json'
        repeat_cmd = [sys.executable, '-B', str(M.HERE/'model_metrics.py'), '--ckpt',
                      str(M.BACKGROUND/'hsweep/ckpt-local/H3.pt'), '--ckpt-sha256', PINS['H3'], '--out', str(repeat_path)]
        suite.emit('COMPUTE H3 repeat (fresh process): '+' '.join(repeat_cmd))
        with (M.HERE/'H3-repeat-model-output.txt').open('w') as log:
            cp = subprocess.run(repeat_cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        suite.check(cp.returncode == 0, 'H3 repeat fresh process failed')
        repeat_model = json.loads(repeat_path.read_text())
        stages = {h: json.loads((M.BACKGROUND/'hsweep/stageO'/f'{h}.json').read_text()) for h in ('H3', 'H4', 'H8', 'Hinf')}
        models = {h: json.loads((M.HERE/f'{h}-model.json').read_text()) for h in PINS}
        for h, model in models.items():
            suite.check(model['provenance']['ckpt_sha256'] == PINS[h] and not model.get('synthetic'), f'{h} model identity')
            suite.check('>=3 distinct cells' in model['straight']['definition'], f'{h} r1 distinct-cell model required')
        for h, o in stages.items():
            A.validate_stage(o)
            suite.emit(f'SCHEMA {h}: {o["schema"]} cells={len(o["cells"])} draws/cell=16')
        # E1 independent raw-category recount AND summary crosscheck.
        for t, n, k in (('4', 80, 19), ('5', 64, 29), ('2', 112, 78)):
            x = valid_array(stages['H3'], t)
            summary = stages['H3']['summary'][t]
            suite.check(x.size == n and x.sum() == k, f'E1 task {t} {k}/{n}')
            suite.check(x.mean() == summary['r'] == summary['away']['valid_rate'], f'E1 summary task {t}')
            suite.emit(f'E1 task {t}: {int(x.sum())}/{x.size}={x.mean():.10f}; summary.r={summary["r"]}')
        suite.passed('E1')

        geo, env = M.geometry_from_dataset()
        # Synthetic model is explicit: Hinf straight MSE only; no invented positive controls.
        hinf = dict(schema='wallpen-model-v1', device='cpu', synthetic=True,
                    provenance=copy.deepcopy(stages['Hinf']['provenance']), positive_control=None,
                    straight=copy.deepcopy(models['H3']['straight']))
        hinf['provenance']['synthetic_note'] = 'Only straight MSE constructed equal to H3; coordinate set reused from pinned real-data sampler. Hinf positive controls and turning MSE unmeasured.'
        hinf['straight']['mse'] = models['H3']['straight']['mse']
        hinf['straight']['per_trajectory_mse'] = None
        hinf['straight']['turn_mse'] = None
        hinf['straight']['turn_n'] = None
        M.write_json(M.HERE/'Hinf-model-synthetic.json', hinf)
        models['Hinf'] = hinf
        metrics = {h: A.arm_metrics(stages[h], models[h], geo)[0] for h in stages}
        expected = dict(H3={'4': (20, 4), '5': (35, 21), '2': (55, 50)},
                        Hinf={'4': (0, 0), '5': (6, 6), '2': (13, 12)})
        for h, ts in expected.items():
            for t, (clean, joint) in ts.items():
                a = metrics[h]['tasks'][t]
                suite.check((a['clean']['count'], a['valid_and_clean']['count']) == (clean, joint), f'E2 {h} task {t}: expected clean/joint={clean}/{joint}, got {a["clean"]["count"]}/{a["valid_and_clean"]["count"]}')
                suite.emit(f'E2 {h} task {t}: clean {a["clean"]["count"]}/{a["clean"]["n"]}; valid+clean {a["valid_and_clean"]["count"]}/{a["valid_and_clean"]["n"]}')
        for t, k in (('4', 6), ('5', 15), ('2', 53)):
            suite.check(metrics['Hinf']['tasks'][t]['valid']['count'] == k, f'E2 Hinf VALID task {t}')
        mutated = A.arm_metrics(stages['H3'], models['H3'], geo, tau=.0572)[0]
        old = [metrics['H3']['tasks'][t]['clean']['count'] for t in A.TASKS]
        new = [mutated['tasks'][t]['clean']['count'] for t in A.TASKS]
        suite.killer(lambda m: m == old, new, f'E2 tau .0572 violates lead clean counts {old} (task order 2,4,5)')
        suite.passed('E2')

        pos = models['H3']['positive_control']
        suite.emit(f'E3 H3 positive n={pos["n"]} p50={pos["p50"]:.8f} p95={pos["p95"]:.8f} max={pos["max"]:.8f}')
        suite.check(pos['n'] == 62, 'E3 n=62')
        for name, v in (('p50', .0073), ('p95', .0664), ('max', .1431)):
            suite.check(abs(pos[name]-v) <= .001, f'E3 {name} expected {v}, got {pos[name]}')
        suite.check(pos['clean_count'] == sum(e['max_occupancy_depth'] <= M.TAU for e in pos['entries']), 'E3 positive pass count uses tau')
        for t, c, v in (('2', [5, 6], .085), ('2', [5, 1], .128), ('4', [3, 9], .007)):
            es = [e for e in pos['named'] if e['task'] == t and e['cell'] == c]
            suite.check(len(es) == 1 and abs(es[0]['max_occupancy_depth']-v) <= .001, f'E3 named {t} {c} expected {v}')
            suite.emit(f'E3 named task {t} {c}: '+json.dumps(es, ensure_ascii=False))
            if c == [5, 1]:
                suite.check(es[0]['max_true_wall_depth'] > 0, 'E3 (5,1) true wall entry')
        suite.passed('E3')

        for t, exp in (('4', [27, 11, 10, 13]), ('5', [14, 7, 11, 3])):
            f = metrics['H3']['tasks'][t]['failures']
            suite.check([f['counts'][k] for k in A.CLASSES] == exp and f['n'] == sum(exp), f'E4 task {t}: expected {exp}, got {f}')
            suite.emit(f'E4 task {t}: failures {f["n"]}; '+str(f['counts']))
        suite.passed('E4')

        for t in A.TASKS:
            x, y = valid_array(stages['H3'], t), valid_array(stages['H4'], t)
            zero, forward, reverse = A.bootstrap(x, x), A.bootstrap(x, y), A.bootstrap(y, x)
            suite.check(zero['estimate'] == 0 and zero['ci95'] == [0, 0], f'E5 identical task {t}')
            suite.check(forward['estimate'] == -reverse['estimate'] and np.array_equal(forward['ci95'], [-reverse['ci95'][1], -reverse['ci95'][0]]), f'E5 swapped interval task {t}')
            # Change exactly k VALID draws in trap cells to NEAR, preserving all pairing identifiers.
            mutated_o = copy.deepcopy(stages['H3'])
            k = 3
            done = 0
            for c in mutated_o['cells']:
                if str(c['task']) == t and c['away']:
                    for d in c['flow']:
                        if done < k and d['cat'] == 'VALID':
                            d['cat'] = 'NEAR'
                            done += 1
            A.assert_paired(stages['H3'], mutated_o)
            delta = A.bootstrap(x, valid_array(mutated_o, t))
            suite.check(done == k and abs(delta['estimate']-k/x.size) < 1e-15, f'E5 k/n task {t}')
            suite.emit(f'E5 task {t}: identical={zero}; H3-H4={forward}; H4-H3={reverse}; change k={k}, n={x.size}, delta={delta["estimate"]:.10f}')
        x = valid_array(stages['H3'], '4')
        independent = A.bootstrap(x, x, paired=False)
        suite.killer(lambda m: m['ci95'][1]-m['ci95'][0] == 0, independent, 'E5 independent indices break zero-width identical-arm interval')
        # Equal per-cell differences: outer-only bootstrap is degenerate, draw layer is not.
        dx, dy = np.zeros((5, 16)), np.zeros((5, 16))
        dx[:, 0] = 1
        def inner_draw_invariant():
            value = A.bootstrap(dx, dy)
            assert value['estimate'] == 1/16 and value['ci95'][1] > value['ci95'][0], 'E5 inner draw resampling must give nonzero CI for sparse within-cell differences'
        def outer_only(x, y=None, **kwargs):
            rng = np.random.default_rng(M.SEED)
            rows = np.asarray(x).mean(1) - (np.asarray(y).mean(1) if y is not None else 0)
            draws = rows[rng.integers(len(rows), size=(A.N_BOOT, len(rows)))].mean(1)
            return dict(estimate=float(rows.mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist())
        suite.emit('E5 sparse inner draws: two-level CI='+str(A.bootstrap(dx, dy)['ci95'])+'; one-level CI='+str(outer_only(dx, dy)['ci95']))
        # Defer killer until after legacy KILLER 3 to preserve its number.
        for kind in ('cell', 'seed', 'draw'):
            bad = copy.deepcopy(stages['H3'])
            if kind == 'cell':
                bad['cells'][0], bad['cells'][1] = bad['cells'][1], bad['cells'][0]
            elif kind == 'seed':
                bad['cells'][0]['seed'] += 1
            else:
                bad['cells'][0]['flow'].reverse()
            suite.rejected(lambda: A.assert_paired(stages['H3'], bad), f'E5 pairing {kind}')
        suite.passed('E5')

        inputs = {n: (stages[h], models[h]) for n, h in [('B0', 'H3'), ('C1', 'H4'), ('C2-hi', 'H8'), ('C2-lo', 'Hinf')]}
        example = A.analyze(inputs, geo)
        example['sources'] = {n: dict(stageo_path=str(M.BACKGROUND/'hsweep/stageO'/f'{h}.json'),
                                     stageo_sha256=M.sha_file(M.BACKGROUND/'hsweep/stageO'/f'{h}.json'),
                                     model_path=str(M.HERE/(f'{h}-model.json' if h != 'Hinf' else 'Hinf-model-synthetic.json')))
                              for n, h in [('B0', 'H3'), ('C1', 'H4'), ('C2-hi', 'H8'), ('C2-lo', 'Hinf')]}
        # Independently counted fractions, v1.3 C2 compares H4, not H3.
        # hi nonaway drops (74-61)/144, (146-123)/208, (126-137)/192:
        # .090278, .110577, -.057292. Only task4 FAIL.
        # lo drops (74-62)/144, (146-137)/208, (126-140)/192:
        # .083333, .043269, -.072917. All PASS; NEAR differences <=0.
        # C1 vs B0 still FAIL MSE and task2. hi -> lo; lo valid deltas
        # (6-14)/80=-.10 and (15-28)/64=-.203125. Clean deltas also <.30.
        suite.check(example['interpretation']['primary'] == 'C2-lo', 'E6 hand primary lo: hi nonaway task4 FAIL')
        hand_failed = {'B0': [], 'C1': ['直路重建 MSE 守門', '非陷阱格 VALID 守門 task 2'],
                       'C2-hi': ['非陷阱格 VALID 守門 task 4'], 'C2-lo': []}
        for n, expected_failed in hand_failed.items():
            suite.check(example['arms'][n]['guards']['failed'] == expected_failed, f'E6 exact hand-calculated guard table {n}')
        for n, counts in (('C2-hi', (61, 123, 137)), ('C2-lo', (62, 137, 140))):
            for t, reference, count, denom in zip(A.TASKS, (74, 146, 126), counts, (144, 208, 192)):
                check = next(c for c in example['arms'][n]['guards']['checks'] if c['name'] == f'非陷阱格 VALID 守門 task {t}')
                suite.check(abs(check['value']-(reference-count)/denom) < 1e-15, f'E6 {n} hand fraction task{t}')
            suite.check(example['arms'][n]['guards_vs_b0']['failed'] == ['非陷阱格 VALID 守門 task 2'], f'E6 {n} separate overall B0 layer')
        for t in ('4', '5'):
            row = example['interpretation']['by_task'][t]
            suite.check(row['c1'].startswith('不出結論') and row['c2'].startswith('懲罰沒教到牆'), f'E6 substitute task {t} hand conclusion')
            suite.check('⚠️ C1 對 B0 已有副作用' in row['c2'], f'E6 C1 warning task{t}')
        for n, columns in example['interpretation']['c2_guard_conclusions'].items():
            suite.check(columns['overall_degradation'].startswith('FAIL') and '⚠️' in columns['overall_degradation'], f'E6 {n} overall warning')
        suite.check(example['interpretation']['c2_guard_conclusions']['C2-lo']['penalty_side_effects'].startswith('PASS'), 'E6 lo penalty PASS, overall FAIL')
        suite.check(example['interpretation']['by_task']['2']['c1'] == 'task 2 只描述' and example['interpretation']['by_task']['2']['c2'].startswith('task 2 只描述'), 'E6 task2 descriptive')
        suite.check(example['interpretation']['noise']['status'] == '未提供', 'E6 B0s absent explicit')
        suite.check(example['interpretation']['conclusion_blockers'] == [
            'C1 對 B0 直路重建 MSE 守門=FAIL', 'C1 對 B0 非陷阱格 VALID 守門 task 2=FAIL'],
            'E6 C1 guard blockers listed separately from rollout threshold')
        suite.emit('E6 hand: C1 FAIL MSE/task2; hi vs C1 nonaway drops 13/144=.090278, 23/208=.110577 FAIL, -11/192=-.057292; lo drops 12/144=.083333, 9/208=.043269, -14/192=-.072917 all PASS. NEAR <=0. Primary lo; task4/5 penalty not learned, C1 warning; both C2 vs B0 FAIL task2.')
        suite.emit('E6 actual: '+json.dumps(example['interpretation'], ensure_ascii=False))
        M.write_json(M.HERE/'wallpen-analysis.json', example)
        (M.HERE/'wallpen-analysis.txt').write_text(A.render(example))
        (M.HERE/'example-analysis.txt').write_text(A.render(example))
        A.plot(example, M.HERE/'wallpen-analysis.png')

        # Explicit synthetic aggregates for decision branch coverage, not substitute measurement.
        synthetic = copy.deepcopy(example['arms'])
        for n, arm in synthetic.items():
            arm['straight']['mse'] = synthetic['B0']['straight']['mse']
            for t in A.TASKS:
                arm['tasks'][t]['nonaway_valid'] = copy.deepcopy(synthetic['B0']['tasks'][t]['nonaway_valid'])
                arm['tasks'][t]['near'] = copy.deepcopy(synthetic['B0']['tasks'][t]['near'])
        recalc_guards(synthetic)
        suite.check(A.select_primary(synthetic)[0] == 'C2-hi', 'E6 defaults to hi when guard PASS')
        sa = copy.deepcopy(synthetic)
        sa['C2-hi']['tasks']['4']['near']['estimate'] += .20
        sa['C2-hi']['tasks']['4']['valid']['estimate'] = .9
        sa['C2-lo']['tasks']['4']['valid']['estimate'] = .5
        recalc_guards(sa)
        selected, reason = A.select_primary(sa)
        suite.check(selected == 'C2-lo' and 'NEAR 守門' in reason, 'E6 (a) hi NEAR +.20 => lo, explicit reason')
        suite.emit(f'E6 synthetic (a): {selected}; {reason}; hi r=.9 > lo r=.5')
        def selection_invariant():
            assert A.select_primary(sa)[0] == 'C2-lo', 'E6 primary selection must ignore hi r=.9 > lo r=.5'
        suite.mutation(A, 'select_primary',
                       lambda arms: (max(('C2-hi', 'C2-lo'), key=lambda n: arms[n]['tasks']['4']['valid']['estimate']), 'mutant metric selection'),
                       selection_invariant, 'E6 monkeypatch actual select_primary to look at primary metric')
        sb = copy.deepcopy(sa)
        sb['C2-lo']['tasks']['4']['near']['estimate'] += .20
        recalc_guards(sb)
        suite.check(A.select_primary(sb)[0] is None, 'E6 (b) both guards FAIL => no conclusion')
        suite.emit('E6 synthetic (b): '+str(A.select_primary(sb)))
        # Test every v1.2 threshold branch on task 4 and 5 independently.
        for t in ('4', '5'):
            for dv, dc, text in ((.15, 0, '沒有牆的概念是原因、訓練時罰有用'),
                                 (.149, .30, '學到牆、近子目標沒變好'),
                                 (.149, .299, '懲罰沒教到牆（權重太小或被躲掉）')):
                sc = copy.deepcopy(synthetic)
                sc['C1']['tasks'][t]['valid']['estimate'] = .3
                sc['C2-hi']['tasks'][t]['valid']['estimate'] = .3+dv
                sc['C1']['tasks'][t]['clean']['estimate'] = .2
                sc['C2-hi']['tasks'][t]['clean']['estimate'] = .2+dc
                dec = A.interpret(sc, decision_diffs(sc))
                suite.check(dec['by_task'][t]['c2'] == text, f'E6 task{t} dv={dv} dc={dc}')
        def failure(counts):
            return dict(failures=dict(n=sum(counts), counts=dict(zip(A.CLASSES, counts)),
                                     proportions=dict(zip(A.CLASSES, np.asarray(counts)/sum(counts)))))
        for counts, base, text in (([20, 10, 60, 10], [30, 20, 10, 40], '教成原地踏步'),
                                  ([20, 10, 10, 60], [30, 20, 10, 40], 'value／搜尋'),
                                  ([60, 10, 10, 20], [30, 20, 10, 40], '容量或「看得到」'),
                                  ([10, 60, 10, 20], [30, 20, 10, 40], '待討論'),
                                  ([35, 20, 35, 10], [40, 20, 20, 20], '教成原地踏步'),
                                  ([10, 20, 40, 30], [30, 20, 30, 20], '待討論'),
                                  ([40, 10, 10, 40], [30, 20, 10, 40], '並列')):
            actual = A.class_branch(failure(counts), failure(base))
            suite.check(text in actual, f'E6 fourclass {counts}, expected {text}, got {actual}')
        sc = copy.deepcopy(synthetic)
        for t in ('4', '5'):
            sc['C1']['tasks'][t]['valid']['estimate'] = sc['B0']['tasks'][t]['valid']['estimate']+.15
            sc['C2-hi']['tasks'][t]['valid']['estimate'] = .70
        dec = A.interpret(sc, decision_diffs(sc))
        suite.check(all('一大塊原因' in dec['by_task'][t]['c1'] for t in ('4', '5')), 'E6 C1 >=.15 boundary')
        suite.check('下一步可考慮 rollout' in dec['rollout'], 'E6 rollout both tasks .70')
        sc['C2-hi']['tasks']['5']['valid']['estimate'] = .699
        suite.check('尚不建議' in A.interpret(sc, decision_diffs(sc))['rollout'], 'E6 rollout requires both tasks')
        for gate in ('G1', 'G2', 'G3'):
            sc = copy.deepcopy(synthetic)
            sc['C2-hi']['gates'][gate] = 'FAIL'
            sc['C2-hi']['gate_pass'] = False
            dec = A.interpret(sc, decision_diffs(sc))
            suite.check(dec['arm_status']['C2-hi'] == '不出結論' and dec['primary'] == 'C2-lo' and gate in dec['reason'], f'E6 {gate} FAIL falls back lo explicitly')
            sc['C2-lo']['gates'][gate] = 'FAIL'
            sc['C2-lo']['gate_pass'] = False
            suite.check(A.interpret(sc, decision_diffs(sc))['by_task']['4']['c2'].startswith('不出結論'), f'E6 both {gate} FAIL blocks conclusion')
        sc = copy.deepcopy(synthetic)
        sc['B0']['gate_pass'] = False
        suite.check(A.interpret(sc, decision_diffs(sc))['by_task']['4']['c1'].startswith('不出結論'), 'E6 B0 gate prevents attribution')
        for keep in (('B0',), ('B0', 'C1'), ('B0', 'C1', 'C2-hi'), ('B0', 'C2-lo')):
            sc = {n: copy.deepcopy(synthetic[n]) for n in keep}
            recalc_guards(sc)
            dec = A.interpret(sc, decision_diffs(sc))
            suite.check(dec['primary'] == ('C2-hi' if 'C2-hi' in keep and 'C1' in keep else None), f'E6 partial arms {keep}')
        sc = copy.deepcopy(synthetic)
        sc['C2-hi']['straight']['mse'] = sc['B0']['straight']['mse']*1.10001
        recalc_guards(sc)
        suite.check(A.select_primary(sc)[0] == 'C2-lo', 'E6 MSE FAIL switches to lo')
        # Guard equality and no-failure discussion edge cases.
        sc = copy.deepcopy(synthetic)
        sc['C2-hi']['tasks']['4']['near']['estimate'] = sc['B0']['tasks']['4']['near']['estimate']+.15
        sc['C2-hi']['tasks']['4']['nonaway_valid']['rate'] = sc['B0']['tasks']['4']['nonaway_valid']['rate']-.10
        sc['C2-hi']['straight']['mse'] = sc['B0']['straight']['mse']*1.10
        recalc_guards(sc)
        suite.check(sc['C2-hi']['guards']['pass_all'], 'E6 inclusive guard equality boundaries')
        suite.check('待討論' in A.class_branch({'failures': {'n': 0}}, failure([1, 1, 1, 1])), 'E6 zero failures explicitly unresolved')
        suite.mutation(A, 'bootstrap', outer_only, inner_draw_invariant, 'B1 one-level bootstrap removes draw resampling')
        g3_inputs = copy.deepcopy(inputs)
        g3_inputs['B0'][0]['gates']['G3']['status'] = 'FAIL'
        def g3_invariant():
            result = A.analyze(g3_inputs, geo)
            assert not result['arms']['B0']['gate_pass'] and all(result['interpretation']['by_task'][t]['c2'].startswith('不出結論') for t in ('4', '5')), 'B1 only G3 FAIL must block all attribution'
        gate_mutant = source_mutant(A.arm_metrics,
            "gate_pass=all(o['gates'][g]['status'] == 'PASS' for g in ('G1', 'G2', 'G3'))",
            "gate_pass=all(o['gates'][g]['status'] == 'PASS' for g in ('G1', 'G2'))")
        suite.mutation(A, 'arm_metrics', gate_mutant, g3_invariant, 'B1 actual gate_pass omits G3 on Stage O copy')

        class_arms = copy.deepcopy(synthetic)
        class_arms['C2-hi']['tasks']['4'].update(failure([45, 10, 30, 15]))
        class_arms['C1']['tasks']['4'].update(failure([40, 10, 20, 30]))
        class_arms['B0']['tasks']['4'].update(failure([60, 10, 5, 25]))
        class_diffs = decision_diffs(class_arms)
        class_diffs['C2-hi-C1']['4']['valid']['estimate'] = 0
        class_diffs['C2-hi-C1']['4']['clean']['estimate'] = .30
        def class_invariant():
            branch = A.interpret(class_arms, class_diffs)['by_task']['4']['failure_branch']
            assert '容量或「看得到」' in branch, 'B1 fourclass must follow C1 too-close .20 (delta .10), not B0 .05 (delta .25)'
        class_mutant = source_mutant(A.interpret, "class_branch(a['tasks'][t], c1['tasks'][t])",
                                    "class_branch(a['tasks'][t], arms['B0']['tasks'][t])")
        suite.mutation(A, 'interpret', class_mutant, class_invariant, 'B1 actual interpretation uses B0 fourclass reference')

        # Exact binary float constants at prereg thresholds; each strict mutant really FAILs.
        exact_arms = copy.deepcopy(synthetic)
        exact_diffs = decision_diffs(exact_arms)
        reaches_original = A.reaches
        for threshold, dv, dc, expected_text in ((.15, .15, 0, '沒有牆的概念'), (.30, 0, .30, '學到牆')):
            exact_diffs['C2-hi-C1']['4']['valid']['estimate'] = dv
            exact_diffs['C2-hi-C1']['4']['clean']['estimate'] = dc
            def decision_boundary():
                assert A.interpret(exact_arms, exact_diffs)['by_task']['4']['c2'].startswith(expected_text), f'B1 inclusive decision boundary {threshold}'
            suite.mutation(A, 'reaches', lambda v, limit: v > limit if limit == threshold else reaches_original(v, limit),
                           decision_boundary, f'B1 >= {threshold} changed to >')

        noise_arms = copy.deepcopy(synthetic)
        noise_arms['B0s'] = copy.deepcopy(noise_arms['B0'])
        noise_arms['B0']['tasks']['4']['valid']['estimate'] = 0
        noise_arms['B0s']['tasks']['4']['valid']['estimate'] = .10
        def noise_boundary():
            result = A.interpret(noise_arms, decision_diffs(noise_arms))
            assert result['noise']['same_scale'] and all('訓練雜訊跟門檻同量級，要更多 seed 才能定' in row['c1'] and '訓練雜訊跟門檻同量級，要更多 seed 才能定' in row['c2'] for row in result['by_task'].values()), 'B1 B0s exact .10 annotates all conclusions'
        suite.mutation(A, 'reaches', lambda v, limit: v > limit if limit == .10 else reaches_original(v, limit),
                       noise_boundary, 'B1 B0s >= .10 changed to >')
        for site, replacement, label in (
            ("sa['mse'] <= sb['mse']*1.10", "sa['mse'] < sb['mse']*1.10", 'MSE 1.10x'),
            ('drop <= .10+1e-12', 'drop < .10', 'nonaway .10'),
            ('near <= .15+1e-12', 'near < .15', 'NEAR .15')):
            boundary_base, boundary_arm = copy.deepcopy(synthetic['C1']), copy.deepcopy(synthetic['C2-hi'])
            boundary_base['straight']['mse'] = 1.0
            boundary_arm['straight']['mse'] = 1.10
            boundary_base['tasks']['4']['nonaway_valid']['rate'] = .10
            boundary_arm['tasks']['4']['nonaway_valid']['rate'] = 0
            boundary_base['tasks']['4']['near']['estimate'] = 0
            boundary_arm['tasks']['4']['near']['estimate'] = .15
            def guard_boundary():
                assert A.guards(boundary_arm, boundary_base)['pass_all'], f'B1 inclusive guard boundary {label}'
            suite.mutation(A, 'guards', source_mutant(A.guards, site, replacement), guard_boundary, f'B1 <= changed to <: {label}')

        # B0s is optional, paired and gated, never selects primary.
        noise_inputs = dict(inputs, B0s=copy.deepcopy(inputs['B0']))
        noise_result = A.analyze(noise_inputs, geo)
        suite.check(noise_result['interpretation']['primary'] == example['interpretation']['primary'], 'A4 B0s never enters primary selection')
        suite.check(all(set(values) == {'valid', 'clean', 'straight_mse_ratio', 'nonaway_valid', 'near'} and all(value == 0 for value in values.values()) for values in noise_result['interpretation']['noise']['by_task'].values()), 'A4 identical B0s five exact zero values task4/5')
        for kind in ('cell', 'seed', 'draw'):
            bad_noise = copy.deepcopy(noise_inputs)
            o = bad_noise['B0s'][0]
            if kind == 'cell':
                o['cells'][0], o['cells'][1] = o['cells'][1], o['cells'][0]
            elif kind == 'seed':
                o['cells'][0]['seed'] += 1
            else:
                o['cells'][0]['flow'].reverse()
            suite.rejected(lambda: A.analyze(bad_noise, geo), f'A4 B0s pairing {kind}')
        bad_noise = copy.deepcopy(noise_inputs)
        bad_noise['B0s'][0]['gates']['G3']['status'] = 'FAIL'
        gated_noise = A.analyze(bad_noise, geo)
        suite.check(gated_noise['interpretation']['noise']['status'].startswith('不出結論') and not gated_noise['interpretation']['noise']['by_task'] and gated_noise['interpretation']['primary'] == 'C2-lo', 'A4 B0s gate failure blocks only noise description')
        # Independently hand-calculated nonzero noise quantities, including ratio difference.
        ns = copy.deepcopy(synthetic)
        ns['B0s'] = copy.deepcopy(ns['B0'])
        ns['B0s']['straight']['mse'] *= 1.05
        for t in ('4', '5'):
            ns['B0s']['tasks'][t]['valid']['estimate'] = ns['B0']['tasks'][t]['valid']['estimate']+.10
            ns['B0s']['tasks'][t]['clean']['estimate'] = ns['B0']['tasks'][t]['clean']['estimate']+.20
            ns['B0s']['tasks'][t]['nonaway_valid']['rate'] = ns['B0']['tasks'][t]['nonaway_valid']['rate']-.03
            ns['B0s']['tasks'][t]['near']['estimate'] = ns['B0']['tasks'][t]['near']['estimate']+.04
        nr = A.interpret(ns, decision_diffs(ns))
        for t in ('4', '5'):
            suite.check(all(abs(nr['noise']['by_task'][t][k]-v) < 1e-12 for k, v in dict(valid=.10, clean=.20, straight_mse_ratio=.05, nonaway_valid=.03, near=.04).items()), f'A4 hand noise five quantities task{t}')
        warning_text = '訓練雜訊跟門檻同量級，要更多 seed 才能定'
        suite.check(warning_text in nr['reason'] and warning_text in nr['rollout'] and all(warning_text in s for cs in nr['c2_guard_conclusions'].values() for s in cs.values()), 'A4 noise annotates selection and both conclusion layers')
        ns['B0s']['tasks']['4']['valid']['estimate'] = ns['B0']['tasks']['4']['valid']['estimate']+.099
        ns['B0s']['tasks']['5']['valid']['estimate'] = ns['B0']['tasks']['5']['valid']['estimate']+.099
        suite.check(not A.noise_report(ns)['same_scale'], 'A4 noise below .10 no annotation')
        for blocked in ('B0', 'C1'):
            sr = copy.deepcopy(synthetic)
            for t in ('4', '5'):
                sr['C2-hi']['tasks'][t]['valid']['estimate'] = .70
            sr[blocked]['gates']['G3'] = 'FAIL'
            sr[blocked]['gate_pass'] = False
            rr = A.interpret(sr, decision_diffs(sr))
            suite.check('下一步可考慮 rollout' in rr['rollout'] and any(blocked+' Stage O G3' in reason for reason in rr['conclusion_blockers']) and rr['by_task']['4']['c2'].startswith('不出結論'), f'A5 rollout .70 only, {blocked} blocker separate')
        cli = subprocess.run([sys.executable, '-B', str(M.HERE/'analyze_wallpen.py'), '--arm', 'B0=not-read,not-read'], capture_output=True, text=True)
        suite.check(cli.returncode == 2 and '--outdir' in cli.stderr and 'required' in cli.stderr, 'B3 outdir required before reading input')
        suite.emit('E6 v1.3: two guard layers/C1 warning; hi gate fallback; B0s five metrics, pairing/gates/noise annotation; rollout has separate blockers; required --outdir PASS')
        suite.passed('E6')

        b = copy.deepcopy(metrics['H3'])
        independent_h3 = copy.deepcopy(b)
        independent_h3['straight'] = repeat_model['straight']
        g = A.guards(independent_h3, b)
        suite.check(g['checks'][0]['ratio'] == 1.0 and g['pass_all'], 'E7 independent fresh-process H3/H3 ratio exactly 1')
        suite.check(repeat_model['straight'] == models['H3']['straight'], 'E7 full independent straight reconstruction matches')
        suite.check(models['H3']['straight']['coordinate_sha256'] == models['H4']['straight']['coordinate_sha256'], 'E7 H3/H4 coordinate sha equal')
        suite.check(all(models[h]['straight']['n'] == 2048 for h in PINS), 'E7 straight n=2048 each specified ckpt')
        suite.check(models['H3']['straight']['coordinate_sha256'] == models['H8']['straight']['coordinate_sha256'], 'E7 H3/H8 sha equal')
        bad = copy.deepcopy(b)
        bad['straight']['coordinate_sha256'] = '0'*64
        suite.rejected(lambda: A.guards(bad, b), 'E7 refuse different coordinate set')
        suite.emit('E7 H3/H3 ratio=1.0; H3/H4/H8 straight_n=2048, sha256='+models['H3']['straight']['coordinate_sha256'])
        for h in PINS:
            s = models[h]['straight']
            suite.emit(f'E7 {h}: batches={s["batches"]}, straight MSE={s["mse"]:.12f}, turn_n={s["turn_n"]}, turn MSE={s["turn_mse"]:.12f}')
        # B2 trajectory-level regression: A->B->A must be skipped, A->B->C kept.
        import torch
        xy_aba = np.resize(np.asarray([[0., 0.], [1., 0.], [0., 0.]], dtype=np.float32), (128, 2))
        xy_abc = np.resize(np.asarray([[0., 0.], [1., 0.], [2., 0.]], dtype=np.float32), (128, 2))
        xy_turn = np.resize(np.asarray([[0., 0.], [1., 0.], [1., 1.]], dtype=np.float32), (128, 2))
        batches = [0]
        def fake_batch(rng, teacher_mix):
            assert teacher_mix == 0.0
            xy = xy_aba if batches[0] == 0 else xy_abc
            batches[0] += 1
            traj = torch.tensor(np.concatenate([np.repeat(xy[None], 2048, axis=0), xy_turn[None]]))
            return traj, torch.zeros((2049, 128), dtype=torch.bool), traj, None, None
        module = SimpleNamespace(torch=torch, B=2049, make_batch=fake_batch, etarget=lambda traj, mask: traj, _dec=lambda encoded, start: encoded)
        fake_geo = SimpleNamespace(mu=np.zeros(2), sd=np.ones(2), xy2c=lambda p: tuple(int(x) for x in p))
        sampled = M.straight_reconstruction(module, fake_geo)
        expected_sha = hashlib.sha256(np.repeat(xy_abc[None].astype('<f8'), 2048, axis=0).tobytes()).hexdigest()
        suite.check(sampled['batches'] == 2 and sampled['n'] == 2048 and sampled['coordinate_sha256'] == expected_sha, 'B2 ABA skipped; >=3 distinct aligned cells selected')
        suite.check(sampled['turn_n'] == 2, 'B2 turning description still every sampled batch')
        suite.passed('E7')
        suite.check(suite.tests == 7 and suite.killers == 12, 'all 7 tests / 12 killers required')
        status = 'DONE'
    except Exception:
        suite.emit(traceback.format_exc())
    finally:
        if env:
            env.close()
        suite.emit(f'n_tests={suite.tests} n_killers={suite.killers} n_checks={suite.checks}')
        suite.emit('STATUS: '+status)
        (M.HERE/'selftest-output.txt').write_text('\n'.join(suite.lines)+'\n')
    return 0 if status == 'DONE' else 2


if __name__ == '__main__':
    sys.exit(main())
