"""Move 1 fixed injection; CPU integration verified separately from GPU release."""
import argparse
import os
from pathlib import Path
import random
import numpy as np
from common import Blocked, digest_array, seeds, verify_source

from common import release_check


def rollout_plan(builder, smoke=False, include_conformance=True):
    main, gate = builder['questions']['main'], builder['questions']['gate']
    if smoke:
        main = [min((q for q in main if q['task'] == t), key=lambda q: q['episode']) for t in (4, 5)]
        gate = [min((q for q in gate if q['task'] == t), key=lambda q: q['episode']) for t in (1, 3)]
    rows = []
    for group, questions, arms in [('main', main, 'NPQR'), ('gate', gate, 'NPR')]:
        for q in questions:
            for arm in arms:
                rows.append(dict(**q, group=group, arm=arm, draw=64,
                                 kind='rollout', additional_draw=True,
                                 conditional_on_representation=group == 'main' and arm in 'PQ',
                                 conditional_on_N_success=group == 'gate' and arm != 'N',
                                 seeds=seeds(q['task'], q['episode'], 64, arm != 'N'),
                                 source=('self-rollout-u' if group == 'gate' and arm == 'P' else
                                         'route-A' if arm == 'P' else 'route-B' if arm == 'Q' else
                                         'donor-route-A' if arm == 'R' else 'fresh-flow')))
    if include_conformance:
        for task in (4, 5):
            episode = min(q['episode'] for q in main if q['task'] == task)
            for arm in 'PQ':
                original = next(r for r in rows if (r['task'], r['episode'], r['arm']) ==
                                (task, episode, arm))
                rows.append(dict(original, kind='determinism-verification', additional_draw=False))
    return rows


def configure_determinism(torch):
    """Official randomness checklist; call before model/CUDA initialization.

    https://docs.pytorch.org/docs/stable/notes/randomness.html
    CUBLAS configuration is also set in sbatch before importing torch.
    """
    if torch.cuda.is_initialized():
        raise Blocked('BLOCKED: determinism configured after CUDA initialization')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise Blocked('BLOCKED: CUBLAS_WORKSPACE_CONFIG must be fixed before startup')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                cudnn_version=torch.backends.cudnn.version(), deterministic=True,
                warn_only=False, cudnn_benchmark=False, cudnn_deterministic=True,
                tf32=False, sdpa='math', dtype='float32', batch=1,
                fill_uninitialized_memory=True, threads=1, dataloader_workers=0,
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])


def seed_policy(torch, task, episode):
    record = seeds(task, episode, 64)
    random.seed(record['python_seed'])
    torch.manual_seed(record['torch_seed'])
    return record


def rng_snapshot(torch, env, noise=None):
    """Archive full states, not just seed integers (caller writes the JSON)."""
    np_state = np.random.get_state()
    result = dict(python=random.getstate(),
                  numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                  torch_cpu=torch.get_rng_state().tolist(),
                  torch_cuda=[s.tolist() for s in torch.cuda.get_rng_state_all()],
                  environment=env.unwrapped.np_random.bit_generator.state,
                  action_space=env.action_space.np_random.bit_generator.state)
    if noise is not None:
        result['noise'] = noise.get_state().tolist()
    return result


def array_of(u):
    return u.detach().cpu().numpy() if hasattr(u, 'detach') else np.asarray(u)


class FixedInjection:
    """Records the tensor at the head call itself, not a pre-flow variable."""
    def __init__(self, head, u):
        self.head = head
        self.u = u.detach().clone() if hasattr(u, 'detach') else np.array(u, copy=True)
        self.expected = digest_array(array_of(self.u))
        self.head_hashes = []

    def __call__(self, cond):
        delivered = digest_array(array_of(self.u))
        if delivered != self.expected:
            raise ValueError('E0: injection changed before head')
        self.head_hashes.append(delivered)
        output = self.head(cond, self.u)
        if digest_array(array_of(self.u)) != self.expected:
            raise ValueError('E0: head mutated injection')
        return output


def encode_trajectory(module, raw_points):
    """128-point arc-length interpolation, normalized XY, encode_u's false mask.

    The pinned encode_u creates the all-False mask and calls etarget directly.
    """
    torch = module.torch
    if module.T_CAP != 128:
        raise Blocked('BLOCKED: T_CAP must be 128')
    points = np.asarray(raw_points, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('invalid XY trajectory')
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 0]
    points, lengths = points[keep], lengths[keep]
    if lengths[-1] <= 0:
        raise ValueError('degenerate trajectory')
    times = np.linspace(0., lengths[-1], 128)
    raw = np.stack([np.interp(times, lengths, points[:, i]) for i in range(2)], axis=1)
    normalized = ((raw-module.MU_XY)/module.SD_XY).astype(np.float32)
    with torch.no_grad():
        u = module.encode_u(torch.tensor(normalized, device=module.device)[None])
    if tuple(u.shape) != (1, module.K, module.D_MODEL):
        raise Blocked(f'BLOCKED: encode_u shape {tuple(u.shape)} does not match head contract')
    # Bind provenance at the actual encoder input, including gate P and donor R.
    u.probe_trajectory_sha256 = digest_array(np.asarray(raw_points, np.float64))
    return u


def manipulation_check(u_a, u_b, noise_band, decoded_identity_a, decoded_identity_b):
    if noise_band is None or not np.isfinite(noise_band) or noise_band < 0:
        raise Blocked('BLOCKED: same-route resampling noise band requires gate calibration')
    distance = float(np.linalg.norm(array_of(u_a).astype(np.float64)-array_of(u_b)))
    passed = distance > noise_band and decoded_identity_a == 'A' and decoded_identity_b == 'B'
    return dict(distance=distance, noise_band=float(noise_band),
                roundtrip_A=decoded_identity_a, roundtrip_B=decoded_identity_b,
                passed=passed, status='passed' if passed else '表示層不可判')


def rerun_conformance(run, u):
    """run must reset environment AND policy caches/RNG, on actual P/Q path."""
    first, second = run(u), run(u)
    a, b = array_of(first), array_of(second)
    return dict(passed=a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                first_sha256=digest_array(a), second_sha256=digest_array(b),
                kind='determinism-verification', additional_draw=False)


def gate_p_source(n_row):
    if not n_row['success']:
        return None
    if n_row['arm'] != 'N' or n_row['draw'] != 64:
        raise ValueError('gate P requires this question N draw 64')
    return dict(source='self-rollout-u', task=n_row['task'], episode=n_row['episode'],
                xy=n_row['xy'], source_trace_sha256=n_row['trace_sha256'])


def production(outdir=None, calibration=None, *, smoke=False, resume_from=None):
    verify_source()
    release_check(smoke)
    if calibration is None:
        raise Blocked('BLOCKED: explicit smoke calibration required')
    from runtime import execute
    return execute(1, outdir, calibration, smoke=smoke, resume_from=resume_from)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume-from', type=Path, help='reuse verified shards into a fresh output directory')
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--calibration', type=Path)
    args = p.parse_args()
    try:
        import json
        production(args.outdir, json.loads(args.calibration.read_text()) if args.calibration else None,
                   smoke=args.smoke, resume_from=args.resume_from)
    except Blocked as e:
        p.exit(2, str(e)+'\n')


if __name__ == '__main__':
    main()
