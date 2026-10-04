"""CPU-only T1–T8 and executable mutation killers. No rollout or formal training.
Only module-boundary tracing skips legacy evaluation; training/save source is unmodified.
Temporary artifacts are retained (no deletion)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / 'experiments/_workorders/hsweep/train_hsweep.py'
TRAIN = HERE / 'train_wallpen.py'
DATA = Path(os.environ.get('OGBENCH_DATA_DIR', '/home/cymaxwelllee/data/ogbench'))
REF = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/ckpt-local/H3.pt')
EVAL = Path('/home/cymaxwelllee/Projects/lacot-hsweep-eval-frozen/experiments/_workorders/hsweep')
WRAPPER = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/wallpen/run_train_wallpen.sh')
OLD_SHA = '418670fd14d1851e35a9cd77461b7bf156550aaa93cec58c8af2ce1660b0ff91'
DEFAULT_SHA = '637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8'
sys.path.insert(0, str(REPO))
import numpy as np
import torch
# r1: every worker is exec'd in a fresh process. The frozen loader owns ALL
# torch setup in the load worker: no thread setters or parallel torch work first.
LOAD_WORKER = '--worker' in sys.argv and sys.argv[sys.argv.index('--worker') + 1] == 'load'
if not LOAD_WORKER:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

CONFIG = dict(ENV='pointmaze-large-stitch-v0', ENC_OBJ='recon_ictr', LEARNED_REFINE=0,
              COND_DROP=0.1, BC_INDEP=1, TEACHER_MIX=0.5, EVAL_RS=0, DIAG_TRAIN=1,
              EMA_W=0.999, WARMUP=500, K=8, SEED=33, DEC_START='soft', TEACHER_H=3,
              STEPS1=0, STEPS2=0, EVAL_EPISODES=2, LOG_EVERY=10)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def configure(extra):
    for key in list(os.environ):
        if key.startswith('LACOT_'):
            del os.environ[key]
    os.environ.update({'LACOT_' + k: str(v) for k, v in (CONFIG | extra).items()})
    os.environ['OGBENCH_DATA_DIR'] = str(DATA)


def load(path, mode='batch', extra=None):
    """Execute exact source through a traced boundary; no AST/source rewriting for baseline.

    batch: stop before model construction. flow: stop after stage2 definitions (zero steps).
    train: run both real loops, stop before legacy eval, execute original tag/save slices.
    """
    configure(extra or {})
    source = path.read_text()
    lines = source.splitlines()
    def line(text):
        found = [i + 1 for i, s in enumerate(lines) if s.startswith(text)]
        assert len(found) == 1, (text, found)
        return found[0]
    model = line('traj_enc = sota_mlp')
    done = line('TAG_SEED = SEED')
    evalstart = line('if LOAD_CKPT:')
    tagdef = line('def _tag_extra(')
    outstart = line('out["n_intent_noroute"]')
    save = line('ck = os.path.join(os.path.dirname(dst)')
    guard_line = line('if (not LOAD_CKPT or CONT_TRAIN) and os.path.exists(_hsweep_ck +') if (extra or {}).get('GUARD_TARGET') else None
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location('wallpen_test_module', path))
    class Boundary(Exception):
        pass
    def trace(frame, event, arg):
        if frame.f_code.co_filename == str(path) and frame.f_code.co_name == '<module>':
            if event == 'line':
                if (mode == 'batch' and frame.f_lineno == model) or (mode == 'flow' and frame.f_lineno == done):
                    raise Boundary()
                if mode == 'train' and frame.f_lineno == evalstart:
                    raise Boundary()
                if mode == 'train' and frame.f_lineno == guard_line:
                    target = frame.f_globals['_hsweep_ck']
                    if extra['GUARD_TARGET'] == 'sidecar':
                        target += '.wallpen.json'
                    Path(target).parent.mkdir(parents=True, exist_ok=True)
                    Path(target).write_text('occupied target')
            return trace
        return None
    old = sys.gettrace()
    sys.settrace(trace)
    try:
        try:
            module.__spec__.loader.exec_module(module)
        except Boundary:
            if mode == 'train':
                module.SEEDS = CONFIG['EVAL_EPISODES']
                # Exact original statements, with original line numbers, no eval/save payload rewriting.
                for first, last in ((tagdef, outstart), (save, len(lines) + 1)):
                    sys.settrace(trace)  # Trace exceptions disable tracing; re-enable guard fixture.
                    fragment = '\n' * (first - 1) + '\n'.join(lines[first-1:last-1]) + '\n'
                    exec(compile(fragment, str(path), 'exec'), module.__dict__)
    finally:
        sys.settrace(old)
    assert module.device == 'cpu', 'GPU forbidden'
    return module


def interp(p, n, scale=1.0):
    cum = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0) * scale, axis=1))]
    tt = np.linspace(0., cum[-1], n)
    return np.stack([np.interp(tt, cum, p[:, k]) for k in (0, 1)], 1).astype(np.float32)


def check_teacher(m, n=2048):
    accepted = []
    original = m._wallpen_clean_teacher
    def record(p, shifted):
        result = original(p, shifted)
        accepted.append(result.copy())
        return result
    m._wallpen_clean_teacher = record
    result, goals = m._teacher_traj(np.random.default_rng(33), n)
    full = np.stack([interp(p, 128, m.SD_XY) for p in accepted])
    depth = max(m._WP_GEO_CPU.wall_depth(torch.from_numpy(full)).max().item(),
                m._WP_GEO_CPU.wall_depth(torch.from_numpy(result)).max().item(),
                m._WP_GEO_CPU.wall_depth(torch.from_numpy(goals[:, None])).max().item())
    print(f'T2 full-route/actual-target/g max_depth={depth:.10f} counts={m._CLEAN_COUNTS}', flush=True)
    assert depth <= m.TAU_DATA, 'T2 dirty teacher accepted'
    assert sum(m._CLEAN_COUNTS.values()) == n
    # Force the specified fallback branch with an unshifted valid route and failed translated candidates.
    original_ok = m._wallpen_teacher_ok
    route = m._TCH[0]
    m._wallpen_teacher_ok = lambda p: np.array_equal(p, route) and original_ok(p)
    before = m._CLEAN_COUNTS['fallback']
    fallback = original(route, route + 10.)
    assert np.array_equal(fallback, route) and m._CLEAN_COUNTS['fallback'] == before + 1
    m._wallpen_teacher_ok = original_ok
    print('T2 fallback exercised: PASS', flush=True)


def pair_batches(a, b, batches=70, clean=False):
    class Routes(list):
        def __init__(self, source):
            super().__init__(source)
            self.indices = []
        def __getitem__(self, key):
            self.indices.append(key)
            return super().__getitem__(key)
    a._TCH, b._TCH = Routes(a._TCH), Routes(b._TCH)
    ra, rb = np.random.default_rng(33), np.random.default_rng(33)
    for _ in range(batches):
        aa = a.make_batch(ra, teacher_mix=0.5)
        bb = b.make_batch(rb, teacher_mix=0.5)
        for x, y in zip(aa, bb):
            assert torch.equal(x[:32] if clean else x, y[:32] if clean else y), 'paired real data/batch differs'
    assert ra.bit_generator.state == rb.bit_generator.state, 'main rng differs'
    assert a._TCH.indices == b._TCH.indices, 'teacher route choices differ'
    print(f'{"T3" if clean else "T1"} {batches} batches: main rng, real data, {len(a._TCH.indices)} route choices PASS', flush=True)


def check_pen(m):
    pts = ((m.OBS_XY[:20000] - m.MU_XY) / m.SD_XY).astype(np.float32)
    assert m.pen(torch.from_numpy(pts)[None]).item() == 0., 'real data penalty'
    import ogbench
    env, _, _ = ogbench.make_env_and_datasets('pointmaze-large-stitch-v0', dataset_dir=str(DATA))
    try:
        a, b = env.unwrapped.ij_to_xy((5, 6)), env.unwrapped.ij_to_xy((5, 4))
        print(f'T5 task4 negative control ij_to_xy: {a} -> {b}', flush=True)
        neg = ((np.linspace(a, b, 128) - m.MU_XY) / m.SD_XY).astype(np.float32)
        p = m.pen(torch.from_numpy(neg)[None]).item()
        diluted = m.pen(torch.from_numpy(np.concatenate([neg, pts[:100]]))[None]).item()
        print(f'T5 real=0 negative={p:.12g} appended-free={diluted:.12g} delta={abs(p-diluted):.12g}', flush=True)
        assert p > 0, 'negative control zero'
        assert abs(p - diluted) < 1e-9, 'T5c dilution detected'
    finally:
        env.close()


def check_sample(m):
    # Use full-size reused model with nonidentity affine weights, both 2D and 3D prefix conditions.
    ref = torch.load(REF, map_location='cpu', weights_only=False)
    m.flow.load_state_dict(ref['flow'])
    batch = m.make_batch(np.random.default_rng(33), teacher_mix=0.5)
    _, _, s, g, _ = batch
    for cdim in (2, 3):
        c = m.flow_cond(m.condvec(s[:4], g[:4], None), None)
        if cdim == 3:
            c = c[:, None].expand(-1, 3, -1)
        gen = torch.Generator().manual_seed(2_000_036)
        state, global_state = gen.get_state(), torch.get_rng_state()
        actual = m._wallpen_sample(m.flow, len(c), c.detach(), gen)
        assert torch.equal(global_state, torch.get_rng_state()), 'global RNG changed'
        gen.set_state(state)
        expected = m.flow.sample(len(c), c.detach(), generator=gen)
        delta = (actual - expected).abs().max().item()
        print(f'T6a cond_dim={cdim} max_delta={delta:.10g}', flush=True)
        assert delta <= 1e-6
    for model in (m.flow, m.u_dec, m.cond_enc, m.cond_head):
        model.zero_grad(set_to_none=True)
    c = m.flow_cond(m.condvec(s[:4], g[:4], None), None).detach()
    global_state = torch.get_rng_state()
    u = m._wallpen_sample(m.flow, len(c), c, torch.Generator().manual_seed(2_000_036))
    penalty = m.pen(m._intent_inv(m._dec(m._q(u), s[:4]), None))
    print(f'T6b decoded penalty={penalty.item():.12g} requires_grad={penalty.requires_grad}', flush=True)
    assert penalty.requires_grad, 'T6b sample has no gradients'
    penalty.backward()
    grads = [p.grad for p in m.flow.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads) and any(x.abs().max() > 0 for x in grads)
    assert all(p.grad is None for p in m.u_dec.parameters())
    assert all(p.grad is None for model in (m.cond_enc, m.cond_head) for p in model.parameters())
    assert torch.equal(global_state, torch.get_rng_state())
    print('T6b/c flow finite nonzero; decoder/cond grads None; global RNG unchanged: PASS', flush=True)


def check_gradcheck(source):
    """T6d: execute the actual sampler AST, with randn replaced by explicit z.

    functional_call exposes every parameter as a gradcheck input (including the
    unused conditional start). No training import or copied inverse formula.
    """
    from lacot.nf_head import Flow
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == '_wallpen_sample')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
    inverse = namespace['_wallpen_sample']
    torch.manual_seed(260604)
    flow = Flow(token_dim=1, seq_len=3, n_blocks=2, d_hidden=4,
                n_layers=1, n_heads=1, cond_dim=2).double()
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.uniform_(-0.3, 0.3)  # includes nonzero to_params weights/bias

    class Inverse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow = flow
        def forward(self, z, cond):
            # Both production and reference receive exactly the supplied base z.
            with patch.object(torch, 'randn', return_value=z):
                return inverse(self.flow, len(z), cond, torch.Generator())

    wrapped = Inverse()
    names = tuple(name for name, _ in wrapped.named_parameters())
    parameters = tuple(p.detach().clone().requires_grad_() for p in wrapped.parameters())
    for cdim in (2, 3):
        z = torch.randn(1, 3, 1, dtype=torch.float64).requires_grad_()
        shape = (1, 2) if cdim == 2 else (1, 2, 2)
        cond = torch.randn(*shape, dtype=torch.float64).requires_grad_()
        with patch.object(torch, 'randn', return_value=z):
            expected = flow.sample(1, cond)
        assert torch.equal(wrapped(z, cond), expected), 'T6d forward differs from Flow.sample'
        print(f'T6d cond_dim={cdim} float64 randomized ALL weights; forward bitwise equal; '
              f'n_params={len(parameters)} n_scalars={sum(p.numel() for p in parameters)}', flush=True)
        # fast_mode=False checks the full numerical/analytical Jacobians.
        assert torch.autograd.gradcheck(lambda base: wrapped(base, cond), (z,), fast_mode=False)
        print(f'T6d cond_dim={cdim} gradcheck(z): PASS', flush=True)
        def joint(base, condition, *weights):
            return torch.func.functional_call(wrapped, dict(zip(names, weights)), (base, condition), strict=True)
        assert torch.autograd.gradcheck(joint, (z, cond, *parameters), fast_mode=False)
        print(f'T6d cond_dim={cdim} gradcheck(z, cond, ALL flow parameters): PASS', flush=True)


def worker(args):
    extra = json.loads(args.extra)
    if args.case == 'gradcheck':
        check_gradcheck(Path(args.source))
    elif args.case in ('train', 'guard'):
        m = load(Path(args.source), 'train', extra)
        print('ARTIFACT ' + m.ck, flush=True)
    elif args.case == 'load':
        configure({})
        sys.path.insert(0, str(EVAL))
        import eval_common
        m = eval_common.load_ckpt(str(DATA), args.source, sha(args.source))
        assert m.device == 'cpu'
        print('T8 frozen eval load_ckpt: PASS (no evaluation)', flush=True)
    else:
        m = load(Path(args.source), 'flow' if args.case == 'sample' else 'batch', extra)
        if args.case == 'teacher':
            check_teacher(m)
        elif args.case == 'rng':
            baseline = load(BASE)
            pair_batches(baseline, m, clean=True)
        elif args.case == 'batch':
            baseline = load(BASE)
            pair_batches(baseline, m, batches=6)
            assert m._WP_GEO is m._WP_GEO_CPU is m._CLEAN_RNG is m._WP_TGEN is None
        elif args.case == 'tau':
            assert abs(m.TAU_DATA - 0.05722857) <= 1e-5
            print(f'T4 τ_data={m.TAU_DATA:.10f}: PASS', flush=True)
        elif args.case == 'pen':
            check_pen(m)
        elif args.case == 'sample':
            check_sample(m)
        elif args.case == 'dryrun':
            # The batch boundary precedes the step-count definitions and all models.
            print(f'R4 formal arm={extra["ARM"]} STEPS1={os.environ["LACOT_STEPS1"]} '
                  f'STEPS2={os.environ["LACOT_STEPS2"]} '
                  f'CLEAN={m.TEACHER_CLEAN} KAPPA={m.WALLPEN_KAPPA}: asserts PASS (no training)', flush=True)


def main():
    run_dir = Path(tempfile.mkdtemp(prefix='wallpen-selftest-'))
    print(f'CPU SELFTEST artifacts retained: {run_dir}', flush=True)
    print(f'Python={sys.version.split()[0]} torch={torch.__version__} threads=1 CUDA_VISIBLE_DEVICES=empty', flush=True)
    print(f'baseline sha256={sha(BASE)} prereg=d2238aef400f6fb981026640e5c915fc702c55b7d4c9c6e03952cc071d03ca92', flush=True)
    print(f'dataset sha256={sha(DATA / (CONFIG["ENV"] + ".npz"))} reference ckpt sha256={sha(REF)}', flush=True)
    count = dict(tests=0, killers=0)
    # Recover the pre-r1 source by reverting ONLY the two allowed trainer edits;
    # the pin proves this is the original delivery, not an approximate baseline.
    current = TRAIN.read_text()
    start = current.index('    # wallpen: r1 forbid overlapping experiments')
    end = current.index('WARM_FRAC, CAL_STEPS, RETRY', start)
    old_source = (current[:start] + current[end:]).replace(
        'open(ck + ".wallpen.json", "x", encoding="utf-8")',
        'open(ck + ".wallpen.json", "x")', 1)
    old_train = run_dir / 'train_wallpen-v1.py'
    old_train.write_text(old_source, encoding='utf-8')
    assert sha(old_train) == OLD_SHA, 'r1 original source pin mismatch'
    print(f'r1 original trainer sha256={sha(old_train)} current={sha(TRAIN)}', flush=True)
    def run(case, source=TRAIN, extra=None, killer=None):
        config = extra or {}
        process = subprocess.Popen([sys.executable, '-u', str(__file__), '--worker', case,
                                    '--source', str(source), '--extra', json.dumps(config)],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        chunks = []
        for chunk in process.stdout:
            print(chunk, end='', flush=True)
            chunks.append(chunk)
        result = subprocess.CompletedProcess(process.args, process.wait(), ''.join(chunks))
        if killer:
            assert result.returncode != 0, f'{killer} failed to catch mutation'
            assert killer in result.stdout, f'{killer} failed for unrelated reason'
            count['killers'] += 1
            print(f'KILLER {killer}: observed FAIL (exit {result.returncode}), expected PASS', flush=True)
        else:
            assert result.returncode == 0, f'{case} exit={result.returncode}'
        return result.stdout
    def mutate(name, old, new):
        source = TRAIN.read_text()
        assert old in source
        path = run_dir / name / 'train_wallpen.py'
        path.parent.mkdir()
        path.write_text(source.replace(old, new, 1))
        ast.parse(path.read_text())
        return path
    def train(name, source=TRAIN, steps=3, extra=None):
        out = run_dir / name
        out.mkdir()
        log = run('train', source, {'STEPS1': steps, 'STEPS2': steps, 'OUT_DIR': str(out)} | (extra or {}))
        (run_dir / (name + '.log')).write_text(log)
        return Path(next(x[len('ARTIFACT '):] for x in log.splitlines() if x.startswith('ARTIFACT '))), log

    run('batch')
    b, _ = train('T1-base', BASE)
    w, _ = train('T1-default')
    assert sha(b) == sha(w), 'T1 checkpoint sha mismatch'
    assert sha(w) == DEFAULT_SHA, 'T1 previous-delivery checkpoint sha changed'
    print(f'T1 baseline/default ckpt sha256={sha(w)}: PASS', flush=True)
    mutant = mutate('killer-T1', '    _warm_lr(opt1, stp)', '    torch.rand(1)  # killer: global RNG drift\n    _warm_lr(opt1, stp)')
    k, _ = train('T1-mutant', mutant)
    try:
        assert sha(k) == sha(b), 'T1 checkpoint sha mismatch'
    except AssertionError as exc:
        print(f'KILLER T1: observed FAIL: {exc}; mutant sha256={sha(k)}', flush=True)
        count['killers'] += 1
    else:
        raise AssertionError('T1 killer undetected')
    count['tests'] += 1

    run('teacher', extra={'TEACHER_CLEAN': 1})
    mutant = mutate('killer-T2', 'return _WP_GEO_CPU.wall_depth(torch.from_numpy(_wallpen_teacher_points(p))[None]).max().item() <= TAU_DATA', 'return True  # killer: accept first jitter')
    run('teacher', mutant, {'TEACHER_CLEAN': 1}, killer='T2 dirty teacher accepted')
    count['tests'] += 1

    run('rng', extra={'TEACHER_CLEAN': 1})
    source = TRAIN.read_text().replace('def _wallpen_clean_teacher(p, shifted):', 'def _wallpen_clean_teacher(p, shifted, rng):').replace('_wallpen_clean_teacher(_wp_route, p)', '_wallpen_clean_teacher(_wp_route, p, rng)').replace('_CLEAN_RNG.uniform(', 'rng.uniform(')
    mutant = run_dir / 'killer-T3.py'
    mutant.write_text(source)
    run('rng', mutant, {'TEACHER_CLEAN': 1}, killer='paired real data/batch differs')
    count['tests'] += 1

    run('tau', extra={'TEACHER_CLEAN': 1})
    mutant = mutate('killer-T4', '_WP_GEO_CPU = _wallpen_geo("cpu")', '_WP_GEO_CPU = _wallpen_geo("cpu", res=4)')
    run('tau', mutant, {'TEACHER_CLEAN': 1}, killer='τ_data mismatch')
    count['tests'] += 1

    run('pen', extra={'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1})
    mutant = mutate('killer-T5', 'return (ex.square().sum(1) / (ex > 0).sum(1).clamp_min(1)).mean()', 'return ex.square().mean()')
    run('pen', mutant, {'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1}, killer='T5c dilution detected')
    count['tests'] += 1

    run('sample', extra={'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1})
    mutant = mutate('killer-T6', '    z = torch.randn(n, flow.seq_len', '    return flow.sample(n, cond, generator=generator)  # killer: no_grad\n    z = torch.randn(n, flow.seq_len')
    run('sample', mutant, {'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1}, killer='T6b sample has no gradients')
    run('gradcheck')
    # Detach killers preserve the full forward, but must fail a Jacobian check.
    for label, old, new in (
        ('history', 'block.embed(history)', 'block.embed(history.detach())'),
        ('z', '    u = z\n', '    u = z.detach()\n'),
        ('mu', ' + mu[:, -1])', ' + mu[:, -1].detach())'),
    ):
        mutant = mutate('killer-T6d-' + label, old, new)
        run('gradcheck', mutant, killer='Jacobian mismatch')
        print(f'T6d {label}.detach(): forward bitwise equal, gradcheck FAIL observed', flush=True)
    count['tests'] += 1

    c1, log1 = train('T7-C1', steps=100, extra={'TEACHER_CLEAN': 1})
    c2, log2 = train('T7-C2', steps=100, extra={'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1})
    old_c2, _ = train('r1-C2-old', old_train, steps=100, extra={'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1})
    assert sha(c2) == sha(old_c2), 'r1 old/new C2 checkpoint sha mismatch'
    print(f'r1 behavior unchanged C2 CPU 100+100 old/new ckpt sha256={sha(c2)}: PASS', flush=True)
    count['tests'] += 1
    def diagnostics(log):
        return {int(line.split()[2]): line for line in log.splitlines() if line.startswith('  DIAG stp ')}
    d1, d2 = diagnostics(log1), diagnostics(log2)
    assert all(d1[i] == d2[i] for i in range(0, 70, 10)), 'T7 warm/calibration update changed'
    assert any(d1[i] != d2[i] for i in range(70, 100, 10)), 'T7 penalty has no effect'
    side = json.loads(Path(str(c2) + '.wallpen.json').read_text())
    for stage, key in (('1', 'λ_dec'), ('2', 'λ_flow')):
        values = [x for x in side['ρ'][stage] if x is not None]
        assert len(side['ρ'][stage]) == 50
        assert np.isfinite(side[key]) and side[key] > 0
        assert side[key] == side['κ'] * float(np.mean(values))
        print(f'T7 {key}={side[key]:.12g} κ*meanρ={float(np.mean(values)):.12g} n_used={len(values)} PASS', flush=True)
    print('T7 C1/C2 stage1 DIAG 0–60 identical; >=70 diverges: PASS', flush=True)
    # Literal workorder killer: validate real startup tau, then temporarily set shared tau to 1e9.
    mutant = mutate('killer-T7', '    TAU_DATA = _wallpen_tau(_WP_GEO_CPU)',
                    '    TAU_DATA = _wallpen_tau(_WP_GEO_CPU)\n    TAU_DATA = 1e9  # killer: zero signal')
    zero, log0 = train('T7-zero-signal', mutant, steps=100, extra={'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1})
    side0 = json.loads(Path(str(zero) + '.wallpen.json').read_text())
    assert side0['τ_data'] == 1e9
    assert log0.count('無可定標') == 2
    assert side0['λ_dec'] == side0['λ_flow'] == 0
    assert all(side0['no_calibration'].values())
    assert all(x is None for seq in side0['ρ'].values() for x in seq)
    try:
        assert side0['λ_dec'] > 0 and side0['λ_flow'] > 0, 'T7 positive-calibration gate FAIL (zero signal)'
    except AssertionError as exc:
        print(f'KILLER T7: observed FAIL: {exc}; both λ=0, 無可定標, JSON finite PASS', flush=True)
        count['killers'] += 1
    else:
        raise AssertionError('T7 killer undetected')
    count['tests'] += 1

    def shapes(value):
        if isinstance(value, torch.Tensor):
            return ('tensor', tuple(value.shape))
        if isinstance(value, dict):
            return {k: shapes(v) for k, v in value.items()}
        return type(value).__name__
    actual, expected = torch.load(c2, map_location='cpu', weights_only=False), torch.load(REF, map_location='cpu', weights_only=False)
    assert shapes(actual) == shapes(expected), 'T8 checkpoint keys/shapes mismatch'
    required = {'τ_data', 'κ', 'STAGES', 'λ_dec', 'λ_flow', 'ρ', 'clean_teacher', 'no_calibration'}
    assert required <= set(side) and '_tc1' in c2.name and '_wp1' in c2.name
    print(f'T8 recursive key/shape schema matches H3; sidecar fields={sorted(side)} filename={c2.name}: PASS', flush=True)
    for target in ('ckpt', 'sidecar'):
        out = run_dir / ('T8-existing-' + target)
        out.mkdir()
        # Same zero-step filename obtained from the original pure naming source via a save-only worker.
        marker_log = subprocess.run([sys.executable, '-u', str(__file__), '--worker', 'guard', '--source', str(TRAIN),
                                    '--extra', json.dumps({'OUT_DIR': str(out), 'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1, 'GUARD_TARGET': target})],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(marker_log.stdout, end='', flush=True)
        assert marker_log.returncode != 0 and 'already exists' in marker_log.stdout
        print(f'T8 existing {target}: refusal PASS (exit {marker_log.returncode})', flush=True)
    run('load', c2)
    count['tests'] += 1
    for arm in ('G', 'C1', 'C2lo', 'C2hi'):
        dry = subprocess.run(['sh', str(WRAPPER), arm, str(run_dir / ('formal-' + arm)), 'dryrun'],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(dry.stdout, end='', flush=True)
        assert dry.returncode == 0, f'R4 wrapper {arm} dryrun failed'
        formal = {line.split('=', 1)[0][6:]: line.split('=', 1)[1]
                  for line in dry.stdout.splitlines() if line.startswith('LACOT_')}
        assert formal['STEPS1'] == '1500' and formal['STEPS2'] == '8000'
        run('dryrun', extra=formal | {'ARM': arm})
    # Real process refusals, before any optional checkpoint/bootstrap reads.
    for key, value, reason in (
        ('INTENT', 'embed', 'INTENT must be empty'),
        ('FSQ', '2,4', 'FSQ must be off'),
        ('FSQ_FIT', '/tmp/unused-fsq.pt', 'FSQ must be off'),
        ('FSQ_LOAD', '/tmp/unused-fsq.pt', 'FSQ must be off'),
        ('VQ', 4, 'VQ must be off'), ('LO_W', 1, 'LO_W must not be positive'),
        ('CONT_TRAIN', 1, 'CONT_TRAIN must be 0'),
        ('LOAD_CKPT', '/tmp/unused.pt', 'LOAD_CKPT must be empty'),
        ('S1_FROM', '/tmp/unused.pt', 'S1_FROM must be empty'),
        ('BOOT_DATA', '/tmp/unused.npz', 'BOOT_DATA must be empty'),
    ):
        incompatible = {'TEACHER_CLEAN': 1, 'WALLPEN_KAPPA': 1, key: value}
        if key == 'CONT_TRAIN':
            incompatible['LOAD_CKPT'] = '/tmp/unused.pt'  # satisfy the legacy continuation prerequisite
        rejected = subprocess.run([sys.executable, '-u', str(__file__), '--worker', 'dryrun',
                                   '--extra', json.dumps(incompatible)],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(rejected.stdout, end='', flush=True)
        assert rejected.returncode != 0 and 'wallpen: ' + reason in rejected.stdout
        print(f'R4 forbidden {key}: refusal PASS (exit {rejected.returncode})', flush=True)
    count['tests'] += 1
    print('T1–T8 (including T6d), r1 behavior equality and R4 asserts: PASS', flush=True)
    print('SECOND PASS: T6d tests the exact sampler AST with explicit z and ALL functional parameters; '
          'T8 loader ran once in a fresh worker without pre-load torch setup. '
          'CUDA inverse/grid_sample behavior remains outside this CPU test.', flush=True)
    print(f'n_tests={count["tests"]} n_killers={count["killers"]}', flush=True)
    print('STATUS: DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', dest='case')
    parser.add_argument('--source', default=str(TRAIN))
    parser.add_argument('--extra', default='{}')
    args = parser.parse_args()
    if args.case:
        worker(args)
    else:
        main()
