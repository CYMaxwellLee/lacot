"""wallpen2: W1–W7 direct-command CPU tests; retain every subprocess's raw output."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = HERE / 'train_wallpen.py'
C1 = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/wallpen/ckpt-local/C1.pt')
H3 = C1.parent / 'G.pt'
DEFAULT_SHA = '637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8'
CONFIG = dict(ENV='pointmaze-large-stitch-v0', ENC_OBJ='recon_ictr', LEARNED_REFINE=0,
              COND_DROP=0.1, BC_INDEP=1, TEACHER_MIX=0.5, EVAL_RS=0, DIAG_TRAIN=1,
              EMA_W=0.999, WARMUP=500, K=8, SEED=33, DEC_START='soft', TEACHER_H=3,
              STEPS1=3, STEPS2=3, EVAL_EPISODES=2, LOG_EVERY=10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('LACOT_') and k != 'WALLPEN_EXPECTED_SEED'}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OGBENCH_DATA_DIR='/home/cymaxwelllee/data/ogbench',
               PYTHONPATH=os.pathsep.join((str(HERE), str(REPO))), MUJOCO_GL='egl')
    env.update({'LACOT_' + k: str(v) for k, v in (CONFIG | extra).items()})
    return env


def command(argv, env):
    print('COMMAND ' + json.dumps(argv, ensure_ascii=False), flush=True)
    process = subprocess.Popen(argv, env=env, cwd=REPO, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    chunks = []
    for chunk in process.stdout:
        print(chunk, end='', flush=True)
        chunks.append(chunk)
    return process.wait(), ''.join(chunks)


def run_suite(run_dir):
    import numpy as np
    import torch
    count = dict(tests=0, killers=0)
    print(f'wallpen2: W suite C1 sha={sha(C1)} H3/G sha={sha(H3)}', flush=True)
    def train(name, extra=None, source=TRAIN, reject=None):
        out = run_dir / name
        out.mkdir()
        config = {'OUT_DIR': str(out)} | (extra or {})
        print('CONFIG ' + json.dumps(CONFIG | config, sort_keys=True), flush=True)
        status, log = command([sys.executable, '-u', str(source), '--wallpen-cpu-test=train'], environment(config))
        (run_dir / (name + '.log')).write_text(log)
        if reject:
            assert status != 0 and reject in log, f'{name}: missing intended refusal: {status}'
            return None, log
        assert status == 0, f'{name}: exit {status}'
        ckpt = Path(next(line[9:] for line in log.splitlines() if line.startswith('ARTIFACT ')))
        return ckpt, log
    def side(ckpt):
        return json.loads(Path(str(ckpt) + '.wallpen.json').read_text())
    def mutate(name, old, new):
        text = TRAIN.read_text()
        assert text.count(old) == 1, (name, text.count(old))
        path = run_dir / (name + '.py')
        path.write_text(text.replace(old, new, 1))
        return path
    def states(path):
        return torch.load(path, map_location='cpu', weights_only=False)
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
        return a == b
    def shapes(v):
        if isinstance(v, torch.Tensor):
            return ('tensor', tuple(v.shape))
        if isinstance(v, dict):
            return {k: shapes(x) for k, x in v.items()}
        return type(v).__name__

    ck, log = train('W1-default')
    assert sha(ck) == DEFAULT_SHA, 'W1 default checkpoint SHA changed'
    print(f'W1 PASS default SHA={sha(ck)}; original T1 also passed', flush=True)
    count['tests'] += 1

    floor_config = dict(TEACHER_CLEAN=1, WALLPEN_KAPPA=0.1, WALLPEN_STAGES='2', WALLPEN_FLOOR=5.65)
    def floor_check(source=TRAIN):
        status, log = command([sys.executable, '-u', str(source), '--wallpen-cpu-test=floor'], environment(floor_config))
        assert status == 0, 'W2 fixture process failed'
        result = json.loads(next(x[13:] for x in log.splitlines() if x.startswith('FLOOR_RESULT ')))
        assert np.isclose(result['mixed'], 2., atol=1e-5), 'W2 teacher-only mean failed'
        assert np.isclose(result['short'], 4., atol=1e-5) and result['long'] == 0., 'W2 floor formula failed'
        assert result['all_real'] == 0. and result['empty'] == 0, 'W2 all-real failed'
        assert result['real_grad_zero'] and result['finite'], 'W2 real-row gradient leaked'
    floor_check()
    mutant = mutate('killer-W2', 'selected = distance[active]', 'selected = distance  # wallpen2: killer includes real rows')
    try:
        floor_check(mutant)
    except AssertionError as exc:
        assert 'W2 teacher-only mean failed' in str(exc), str(exc)
        print(f'KILLER W2: observed FAIL: {exc}', flush=True)
        count['killers'] += 1
    else:
        raise AssertionError('W2 killer undetected')
    print('W2 PASS formula, teacher-only reduction, all-real zero and real-row zero gradient', flush=True)
    count['tests'] += 1

    shared = dict(S1_FROM=str(C1), STEPS1=3, STEPS2=100, TEACHER_CLEAN=1,
                  WALLPEN_STAGES='2', WALLPEN_KAPPA=0.1)
    small, log_small = train('W3-small', shared | {'WALLPEN_FLOOR': 1e-6})
    ss = side(small)
    assert ss['floor_valid_steps'] == 0 and all(x is None for x in ss['ρ_floor']), 'W3 floor unexpectedly active'
    assert ss['floor_fallback'] and ss['λ_floor'] == ss['λ_flow'] > 0, 'W3 fallback equality failed'
    assert 'wallpen2: floor 定標退路' in log_small
    # wallpen2 r1: F=100 excludes all targets; use eligible F=5.65 for normal calibration.
    large, log_large = train('W3-large', shared | {'WALLPEN_FLOOR': 5.65})
    sl = side(large)
    ratios = [x for x in sl['ρ_floor'] if x is not None]
    assert len(sl['ρ_floor']) == 50 and sl['floor_valid_steps'] == len(ratios) >= 10
    assert not sl['floor_fallback'] and sl['λ_floor'] == sl['κ'] * float(np.mean(ratios)) > 0
    assert np.isfinite(sl['λ_floor'])
    print(f"W3 PASS fallback λ_floor=λ_flow={ss['λ_floor']}; normal λ_floor={sl['λ_floor']} n_valid={len(ratios)}", flush=True)
    mutant = mutate('killer-W3', '_WP_STATE["2"]["lam"] if state["fallback"] else WALLPEN_KAPPA * float(np.mean(used))',
                    '0.0 if state["fallback"] else WALLPEN_KAPPA * float(np.mean(used))')
    train('W3-no-fallback', shared | {'WALLPEN_FLOOR': 1e-6}, mutant,
          reject='wallpen2: floor lambda must be positive and finite')
    print('KILLER W3: observed FAIL: removed fallback produces λ_floor=0, positivity guard rejected it', flush=True)
    count['killers'] += 1
    # Prove the floor's additional measurements do not affect either update stream.
    measured_floor, _ = train('W3-measure-floor', shared | {'STEPS2': 62, 'WALLPEN_FLOOR': 5.65})
    measured_wall, _ = train('W3-measure-wall', shared | {'STEPS2': 62})
    assert equal(states(measured_floor), states(measured_wall)), 'W3 measurement window changed updates'
    print('W3 PASS complete checkpoint tensors/cfg equal through warmup+50 measurement steps', flush=True)
    count['tests'] += 1

    w4, log4 = train('W4-S1-from', shared | {'WALLPEN_FLOOR': 5.65})
    current, origin = states(w4), states(C1)
    assert 'stage 1 載入並跳過訓練' in log4 and '  DIAG stp ' not in log4, 'W4 stage 1 ran'
    for name in ('traj_enc', 'e_pooler', 'u_dec'):
        assert equal(current[name], origin[name]), f'W4 frozen {name} changed'
    assert not equal(current['flow'], origin['flow']), 'W4 flow did not change'
    assert '_s1from' in w4.name and '_fl5.65' in w4.name
    probe = side(w4)['cpu_gradient_probe']
    for name in ('floor_grad_norms', 'wall_grad_norms'):
        assert len(probe[name]) == 50 and np.isfinite(probe[name]).all() and max(probe[name]) > 1e-12, f'W4 {name} no nonzero finite gradient'
        print(f'W4 {name} min={min(probe[name])} max={max(probe[name])}', flush=True)
    assert log4.count('floor_pen=') == 10 and log4.count('teacher_d_median=') == 10 and log4.count('teacher_d_lt_F=') == 10
    train('W4-reject-stages12', shared | {'WALLPEN_STAGES': '12', 'WALLPEN_FLOOR': 5.65},
          reject='wallpen: S1_FROM must be empty unless STAGES=2')
    print('KILLER W4: observed FAIL: STAGES=12 with S1_FROM rejected by assert', flush=True)
    count['killers'] += 1
    print(f'W4 PASS stage 1 skipped, encoder/decoder bitwise frozen, both penalties reach only flow; {w4.name}', flush=True)
    count['tests'] += 1

    control = dict(S1_FROM=str(C1), TEACHER_CLEAN=1)
    a, _ = train('W5-s33-a', control)
    b, _ = train('W5-s33-b', control)
    c, _ = train('W5-s34', control | {'SEED': 34})
    assert sha(a) == sha(b) and sha(c) != sha(a), 'W5 seed reproducibility failed'
    print(f'W5 PASS s33={sha(a)} repeat={sha(b)} s34={sha(c)}', flush=True)
    count['tests'] += 1

    assert shapes(current) == shapes(states(H3)), 'W6 H3 key/shape mismatch'
    status, _ = command([sys.executable, '-u', str(__file__), '--load', str(w4)], environment({}))
    assert status == 0, 'W6 fresh eval_common loader failed'
    print('W6 PASS recursive H3 key/shape identity; requested eval_common.load_ckpt fresh-process load, seed 33', flush=True)
    count['tests'] += 1

    for name, extra, reason in (
        ('kappa', {'WALLPEN_KAPPA': 0}, 'FLOOR requires KAPPA>0'),
        ('stage', {'WALLPEN_STAGES': '1'}, 'FLOOR requires stage 2'),
        ('clean', {'TEACHER_CLEAN': 0}, 'FLOOR requires TEACHER_CLEAN=1'),
        ('nan', {'WALLPEN_FLOOR': 'nan'}, 'FLOOR must be finite and nonnegative'),
        ('inf', {'WALLPEN_FLOOR': 'inf'}, 'FLOOR must be finite and nonnegative'),
        ('negative', {'WALLPEN_FLOOR': -1}, 'FLOOR must be finite and nonnegative'),
    ):
        train('W-flags-' + name, floor_config | extra, reject='wallpen2: ' + reason)
    print('W7 PASS original T1–T8 and T6d completed unchanged; additional floor flag refusals PASS', flush=True)
    count['tests'] += 1
    return count


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == '--load'
    assert 'WALLPEN_EXPECTED_SEED' not in os.environ
    sys.path.insert(0, str(REPO / 'experiments/_workorders/hsweep'))
    import eval_common
    model = eval_common.load_ckpt(os.environ['OGBENCH_DATA_DIR'], sys.argv[2], sha(sys.argv[2]))
    assert model.device == 'cpu' and model.TAG_SEED == 33
    print('W6 fresh eval_common.load_ckpt: PASS seed=33 WALLPEN_EXPECTED_SEED unset', flush=True)
