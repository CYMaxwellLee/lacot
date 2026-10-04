"""wallpen2 r1: X1-X4 direct CLI checks and the target-mask mutation killer."""
from pathlib import Path
import ast
import hashlib
import json
import sys

from selftest_wallpen2 import TRAIN, C1, DEFAULT_SHA, command, environment, sha

PREVIOUS_SHA = '5d640811913b0e349f71489f82c325d4ccf8733fbcbe7a64c0759d03472c28a6'


def previous_source():
    """Read the verified v2 baseline; never reconstruct it from r1 implementation."""
    baseline = Path(json.loads((TRAIN.parent / 'r1-baseline-location.json').read_text())['path'])
    source = (baseline / 'train_wallpen.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == PREVIOUS_SHA, 'X3 previous source pin mismatch'
    return source


def run_suite(run_dir):
    count = dict(tests=0, killers=0)
    default = next((run_dir / 'W1-default').glob('*.pt'))
    assert sha(default) == DEFAULT_SHA
    print(f'X1 PASS W1/T1 unchanged default sha={sha(default)}', flush=True)
    count['tests'] += 1

    config = dict(TEACHER_CLEAN=1, WALLPEN_KAPPA=0.1, WALLPEN_STAGES='2', WALLPEN_FLOOR=8)
    def active_check(source):
        status, log = command([sys.executable, '-u', str(source), '--wallpen-cpu-test=floor'], environment(config))
        assert status == 0, 'X2 fixture CLI failed'
        row = json.loads(next(x[len('ACTIVE_RESULT '):] for x in log.splitlines() if x.startswith('ACTIVE_RESULT ')))
        assert row['active'] == [True, True, False, False, True, False, False], 'X2 target eligibility mask failed'
        assert abs(row['penalty'] - 2.) < 1e-5, 'X2 active-row mean failed'
        for key in ('boundary_equal', 'inactive_grad_zero', 'prefix_grad_zero', 'gradient_formula',
                    'start_target_detached', 'empty_zero', 'all_real_zero', 'rng_equal'):
            assert row[key], 'X2 ' + key + ' failed'
        assert row['teacher_count'] == 5, 'X2 teacher descriptive subset changed'
        return row
    fixture = active_check(TRAIN)
    mutant = run_dir / 'killer-X2-remove-target-mask.py'
    source = TRAIN.read_text()
    old = 'active = teacher & (target_distance >= WALLPEN_FLOOR)'
    assert source.count(old) == 1
    mutant.write_text(source.replace(old, 'active = teacher  # wallpen2 r1: killer restores v2.0', 1))
    try:
        active_check(mutant)
    except AssertionError as exc:
        assert str(exc) == 'X2 target eligibility mask failed', str(exc)
        print(f'KILLER X2: observed FAIL: {exc}', flush=True)
        count['killers'] += 1
    else:
        raise AssertionError('X2 killer undetected')
    print('X2 PASS active teacher-only mean, exact >= boundary, inactive/real zero gradients, '
          'differentiable empty zero, detached target/start, no RNG; fixture=' + json.dumps(fixture), flush=True)
    count['tests'] += 1

    previous = previous_source()
    old_path = run_dir / 'train_wallpen-5d640811.py'
    old_path.write_text(previous)
    print(f'X3 previous trainer sha={sha(old_path)} current trainer sha={sha(TRAIN)}', flush=True)

    # Observation copies add exactly one read-only call after the EMA update, before logging.
    # No preload, trace, hook, or monkey patch. Removing the insertion restores pinned source.
    anchor = '        # ⭐ 散度時間序列（LACOT_DIV_LOG_EVERY>0）：'
    insertion = ('        # wallpen2 r1: temporary CLI parameter/RNG observation only.\n'
                 '        from cpu_probe_wallpen2 import observe_step\n'
                 '        observe_step(globals(), locals())\n')
    copies = []
    for name, text in (('old', previous), ('new', source)):
        assert text.count(anchor) == 1
        first = text.index(anchor)
        measured = text[:first] + insertion + text[first:]
        assert measured.replace(insertion, '', 1) == text
        ast.parse(measured)
        path = run_dir / ('X3-observe-' + name + '.py')
        path.write_text(measured)
        copies.append(path)
    shared = dict(S1_FROM=str(C1), STEPS1=3, STEPS2=100, TEACHER_CLEAN=1,
                  WALLPEN_STAGES='2', WALLPEN_KAPPA=0.1, WALLPEN_FLOOR=5.65)
    def train(name, path):
        out = run_dir / name
        out.mkdir()
        status, log = command([sys.executable, '-u', str(path), '--wallpen-cpu-test=train'],
                              environment(shared | {'OUT_DIR': str(out)}))
        (run_dir / (name + '.log')).write_text(log)
        assert status == 0, 'X3 ' + name + ' CLI failed'
        assert 'stage 1 載入並跳過訓練' in log and '  DIAG stp ' not in log
        return Path(next(x[9:] for x in log.splitlines() if x.startswith('ARTIFACT ')))
    exact_old = train('X3-exact-old', old_path)
    observed_old = train('X3-observed-old', copies[0])
    observed_new = train('X3-observed-new', copies[1])
    exact_new = next((run_dir / 'W4-S1-from').glob('*.pt'))
    assert sha(exact_old) == sha(observed_old), 'X3 old observer changed checkpoint bytes'
    assert sha(exact_new) == sha(observed_new), 'X3 new observer changed checkpoint bytes'
    before = json.loads(Path(str(observed_old) + '.r1-observations.json').read_text())
    after = json.loads(Path(str(observed_new) + '.r1-observations.json').read_text())
    assert len(before) == len(after) == 100
    warm_cal = 20 + 50
    assert all(a['parameter_sha'] == b['parameter_sha'] for a,b in zip(before[:warm_cal], after[:warm_cal])), 'X3 warm/cal parameter hash differs'
    # wallpen2 r1: previous failure was a test error: sampled d can change after
    # parameter divergence, so excluded_short_count is comparable only before then.
    assert all(a['excluded_short_count'] == b['excluded_short_count']
               for a,b in zip(before[:warm_cal], after[:warm_cal])), 'X3 warm/cal floor predicate differs'
    for a,b in zip(before,after):
        for key in ('step', 'batch_sha', 'global_torch_sha', 'penalty_torch_sha', 'active_count',
                    'main_numpy_sha', 'teacher_numpy_sha', 'excluded_teacher_count'):
            assert a[key] == b[key], 'X3 unrelated stream changed: ' + key
    excluded = sum(a['excluded_short_count'] for a in before)
    first = next((a['step'] for a,b in zip(before,after) if a['parameter_sha'] != b['parameter_sha']), None)
    if not excluded:
        assert sha(exact_old) == sha(exact_new), 'X3 no conflicting rows but final checkpoint differs'
    else:
        assert first is not None and first >= warm_cal, 'X3 missing/early first divergence'
        a, b = before[first], after[first]
        print('X3 first divergence zero-based step=' + str(first) + ' (update=' + str(first+1) + '): '
              'teacher dT<F and sampled d<F excluded from floor; active-only denominator and floor rho/lambda reflect this rule. '
              'old=' + json.dumps(a, sort_keys=True) + ' new=' + json.dumps(b, sort_keys=True), flush=True)
        assert any(a['floor_pen'] != b['floor_pen'] for a,b in zip(before[:warm_cal],after[:warm_cal])), 'X3 eligibility never changed measured floor'
    side = json.loads(Path(str(exact_new) + '.wallpen.json').read_text())
    activity = side['floor_active_logs']
    assert len(activity) == 10
    for row in activity:
        expected = after[row['step']]['active_count']
        assert row['active_count'] == expected and row['active_fraction'] == expected / 64
    log_new = (run_dir / 'W4-S1-from.log').read_text()
    log_rows = [line for line in log_new.splitlines() if line.startswith('wallpen2: stp ')]
    assert len(log_rows) == len(activity)
    for line, row in zip(log_rows, activity):
        assert 'active_count=' + str(row['active_count']) in line
        assert 'active_fraction=' + str(row['active_fraction']) in line
    print(f'X3 PASS 70 warmup/measurement parameter hashes identical; all 100 batch/global/penalty RNG hashes identical; '
          f'excluded_short_rows={excluded} first_divergence={first}; observer CKPT byte identity old/new PASS; '
          f'old_ckpt={sha(exact_old)} new_ckpt={sha(exact_new)}; sidecar/log activity 10 points PASS', flush=True)
    count['tests'] += 1
    print('X4 PASS original W1-W7, T1-T8 and T6d completed in this same execution; '
          'W3 normal-calibration F=5.65 has eligible targets (F=100 would exclude every target under v2.1).', flush=True)
    count['tests'] += 1
    return count
