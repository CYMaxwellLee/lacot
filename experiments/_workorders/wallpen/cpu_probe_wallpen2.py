"""wallpen2: native CPU CLI probes, no preload, tracing, or model monkey-patching."""
from pathlib import Path
import json


def floor_probe(ns):
    # wallpen2: nontrivial raw origin/scales; first decoded point differs from condition start.
    torch = ns['torch']
    floor = ns['WALLPEN_FLOOR']
    start_raw = torch.tensor([[3., 7.]]).expand(4, -1)
    distance = torch.tensor([floor + 1., floor - 2., 0., floor - 1.])
    end_raw = start_raw + torch.stack((distance, distance * 0), -1)
    normalize = lambda x: (x - ns['MU_XY_T']) / ns['SD_XY_T']
    points = torch.stack((normalize(start_raw + 4.), normalize(end_raw)), 1).requires_grad_()
    starts = normalize(start_raw)
    # wallpen2 r1: W2 retains its formula/reduction checks with eligible targets.
    target_end = start_raw + torch.tensor([floor + 1., 0.])
    targets = torch.stack((normalize(start_raw + 9.), normalize(target_end)), 1)
    mixed, ds, _ = ns['_wallpen_floor'](points, starts, torch.tensor([0., 0., 1., 1.]), targets)
    all_real, empty, _ = ns['_wallpen_floor'](points, starts, torch.ones(4), targets)
    long, _, _ = ns['_wallpen_floor'](points[:1], starts[:1], torch.zeros(1), targets[:1])
    short, _, _ = ns['_wallpen_floor'](points[1:2], starts[1:2], torch.zeros(1), targets[1:2])
    grads = torch.autograd.grad(mixed, points, retain_graph=True)[0]
    print('FLOOR_RESULT ' + json.dumps(dict(mixed=mixed.item(), all_real=all_real.item(),
        long=long.item(), short=short.item(), distances=ds.tolist(), empty=empty.numel(),
        real_grad_zero=bool((grads[2:] == 0).all()), finite=bool(torch.isfinite(grads).all()))), flush=True)
    active_probe(ns)


def save_only(ns):
    # wallpen2: run the trainer's unmodified naming/guard/save statements after real loops.
    # No rollout result is fabricated; this CLI emits a checkpoint and sidecar only.
    path = Path(ns['__file__'])
    lines = path.read_text().splitlines()
    def line(prefix):
        hits = [i for i, value in enumerate(lines) if value.startswith(prefix)]
        assert len(hits) == 1, (prefix, hits)
        return hits[0]
    ns['LOAD_EMA'] = 0
    ns['SEEDS'] = int(ns['os'].environ.get('LACOT_EVAL_EPISODES', '2'))
    for first, last in ((line('def _tag_extra('), line('out["n_intent_noroute"]')),
                        (line('ck = os.path.join(os.path.dirname(dst)'), len(lines))):
        statements = '\n' * first + '\n'.join(lines[first:last]) + '\n'
        exec(compile(statements, str(path), 'exec'), ns)
    # wallpen2 r1: only temporary X3 observation copies populate this list.
    if _STEP_OBSERVATIONS:
        Path(ns['ck'] + '.r1-observations.json').write_text(json.dumps(_STEP_OBSERVATIONS, indent=2))
    print('ARTIFACT ' + ns['ck'], flush=True)


def active_probe(ns):
    """wallpen2 r1: mixed targets, exact >= boundary and all inactive gradients."""
    torch, np = ns['torch'], ns['np']
    floor = ns['WALLPEN_FLOOR']
    mu, sd = ns['MU_XY_T'], ns['SD_XY_T']
    normalize = lambda x: (x - mu) / sd
    # At F=8 these offsets round-trip exactly even with the real nonunit scales.
    raw_start = mu.expand(7, -1)
    decoded_d = torch.tensor([floor-1, floor-2, floor-3, floor-4, floor-1, floor-2, floor-3])
    target_d = torch.tensor([floor+1, floor, floor-1, floor-2, floor+2, floor+1, floor-1])
    endpoint = lambda d: raw_start + torch.stack((d, torch.zeros_like(d)), -1)
    starts = normalize(raw_start).clone().requires_grad_()
    points = torch.stack((normalize(raw_start + 17), normalize(endpoint(decoded_d))), 1).requires_grad_()
    targets = torch.stack((normalize(raw_start - 13), normalize(endpoint(target_d))), 1).requires_grad_()
    real_w = torch.tensor([0., 0., 0., 0., 0., 1., 1.])
    torch_rng, numpy_rng = torch.get_rng_state().clone(), np.random.get_state()
    penalty, teacher_d, active = ns['_wallpen_floor'](points, starts, real_w, targets)
    grad, start_grad, target_grad = torch.autograd.grad(penalty, (points, starts, targets), allow_unused=True)
    empty, _, empty_active = ns['_wallpen_floor'](points[2:4], starts[2:4], real_w[2:4], targets[2:4])
    empty_grad = torch.autograd.grad(empty, points)[0]
    real_zero, _, _ = ns['_wallpen_floor'](points, starts, torch.ones(7), targets)
    real_grad = torch.autograd.grad(real_zero, points)[0]
    actual_target_d = torch.linalg.vector_norm(targets[:, -1].detach()*sd + mu - raw_start, dim=-1)
    expected = torch.zeros_like(points)
    expected[[0, 1, 4], -1, 0] = -2 * (floor-decoded_d[[0, 1, 4]]) * sd[0] / 3
    numpy_now = np.random.get_state()
    rng_equal = (torch.equal(torch_rng, torch.get_rng_state()) and numpy_rng[0] == numpy_now[0]
                 and np.array_equal(numpy_rng[1], numpy_now[1]) and numpy_rng[2:] == numpy_now[2:])
    print('ACTIVE_RESULT ' + json.dumps(dict(
        penalty=penalty.item(), active=active.tolist(), target_distances=actual_target_d.tolist(),
        boundary_equal=bool(actual_target_d[1] == floor),
        inactive_grad_zero=bool((grad[[2, 3, 5, 6]] == 0).all()),
        prefix_grad_zero=bool((grad[:, 0] == 0).all()),
        gradient_formula=bool(torch.allclose(grad, expected, atol=2e-5, rtol=1e-5)),
        start_target_detached=start_grad is None and target_grad is None,
        empty_zero=empty.item() == 0 and not bool(empty_active.any()) and bool((empty_grad == 0).all()),
        all_real_zero=real_zero.item() == 0 and bool((real_grad == 0).all()),
        teacher_count=teacher_d.numel(), rng_equal=bool(rng_equal))), flush=True)


_STEP_OBSERVATIONS = []


def observe_step(ns, local):
    """wallpen2 r1: read-only hashes after updates in temporary direct CLI copies."""
    import hashlib
    torch = ns['torch']
    def digest(tensors):
        result = hashlib.sha256()
        for tensor in tensors:
            result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()
    modules = ns['f_mods'] + [ns['bc_head'], ns['traj_enc'], ns['e_pooler'], ns['u_dec']]
    parameters = [p for module in modules if module is not None for p in module.parameters()]
    parameters += [p for ema, _ in ns['_EMA_PAIRS'] for p in ema.parameters()]
    mu, sd = ns['MU_XY_T'], ns['SD_XY_T']
    start = ns['to_xy'](local['s']).detach()*sd + mu
    target_d = torch.linalg.vector_norm(local['traj'][:, -1].detach()*sd + mu - start, dim=-1)
    sampled_d = torch.linalg.vector_norm(local['_wp_pts'][:, -1].detach()*sd + mu - start, dim=-1)
    teacher = ns['_REAL_W'][0] == 0
    active = teacher & (target_d >= ns['WALLPEN_FLOOR'])
    conflicting = teacher & ~active & (sampled_d < ns['WALLPEN_FLOOR'])
    state_sha = lambda rng: hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()
    row = dict(step=local['stp'], parameter_sha=digest(parameters),
        batch_sha=digest([local['traj'], local['s'], local['g'], ns['_REAL_W'][0]]),
        global_torch_sha=digest([torch.get_rng_state()]),
        penalty_torch_sha=digest([ns['_WP_TGEN'].get_state()]),
        main_numpy_sha=state_sha(ns['rng']), teacher_numpy_sha=state_sha(ns['_CLEAN_RNG']),
        active_count=int(active.sum()), excluded_short_count=int(conflicting.sum()),
        excluded_teacher_count=int((teacher & ~active).sum()),
        floor_pen=float(local['_wpf2'].detach()), floor_lambda=ns['_WP_FLOOR_STATE']['lam'],
        wall_lambda=ns['_WP_STATE']['2']['lam'])
    _STEP_OBSERVATIONS.append(row)
    print('STEP_OBSERVATION ' + json.dumps(row, sort_keys=True), flush=True)
