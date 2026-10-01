"""Read-only M9 import and shared probe runtime; release remains lead-controlled."""
import ast
import importlib.util
import os
from pathlib import Path
import random
import sys
import numpy as np
from common import (Blocked, ENV, H, PIN, REPO, SOURCE, digest_array, file_sha,
                    reset_env, seeds, verify_source)

CKPT = REPO / ('results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_'
               'eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt')
DATASET_PIN = '9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8'
OUTPUT_ROOT = Path('/archive/cymaxwelllee/breakthrough1')

CKPT_PIN = '88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787'


def import_boundary():
    """Locate, but do not rewrite, the legacy environment/rollout entry."""
    verify_source()
    tree = ast.parse(SOURCE.read_bytes())
    matches = [n.lineno for n in tree.body if isinstance(n, ast.Assign)
               and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == 'make_env_and_datasets']
    if len(matches) != 1:
        raise Blocked('BLOCKED: pinned import boundary is not unique')
    return matches[0]


def load_frozen(dataset_dir):
    """Execute original M9 bytes through its s33/EMA loader, never legacy eval.

    A trace exception at the module boundary leaves the imported definitions and
    loaded weights intact. No sed, AST transformation, source-copy or flat-guard
    modification is used. Call once in a fresh process (determinism setup).
    """
    boundary = import_boundary()  # before torch, dataset, checkpoint, imports
    if not CKPT.is_file() or file_sha(CKPT) != CKPT_PIN:
        raise Blocked('BLOCKED: s33 checkpoint SHA256 mismatch/unavailable')
    dataset = Path(dataset_dir) / (ENV + '.npz')
    if not dataset.is_file():
        raise Blocked(f'BLOCKED: normalization dataset unavailable: {dataset}')
    if file_sha(dataset) != DATASET_PIN:
        raise Blocked('BLOCKED: normalization dataset SHA256 mismatch')
    import torch
    from harness_move1 import configure_determinism
    settings = configure_determinism(torch)
    config = dict(ENV=ENV, CONS='self', K=8, COND=256, CHUNK=4, TCAP=128,
                  ENC_OBJ='recon_ictr', LEARNED_REFINE=0, COND_DROP=0.1,
                  BC_INDEP=1, TEACHER_MIX=0.5, WARMUP=500, DEC_START='soft',
                  LOAD_EMA=1, CONT_TRAIN=0, STEPS1=0, STEPS2=0, DEV_EVAL=0,
                  PREREQ=0, GRAD_REFINE=0, SUBGOAL='', FINISH_R=0, INTENT='',
                  U_SOURCE='flow', BON_N=0, LOAD_CKPT=str(CKPT), SEED=33,
                  BOOT_DATA='', S1_FROM='', FLOW_PROBE=0, DIAG_DUMP=0)
    saved_env, saved_path, saved_trace = dict(os.environ), list(sys.path), sys.gettrace()
    saved_bytecode = sys.dont_write_bytecode
    class ModelLoaded(Exception):
        pass
    spec = importlib.util.spec_from_file_location('breakthrough_frozen_m9', SOURCE)
    module = importlib.util.module_from_spec(spec)
    def stop_before_legacy_eval(frame, event, arg):
        if frame.f_code.co_filename == str(SOURCE) and frame.f_code.co_name == '<module>':
            if event == 'line' and frame.f_lineno == boundary:
                raise ModelLoaded()
            return stop_before_legacy_eval
        return None
    try:
        for key in list(os.environ):
            if key.startswith('LACOT_'):
                del os.environ[key]
        os.environ.update({'LACOT_'+k: str(v) for k, v in config.items()})
        os.environ['OGBENCH_DATA_DIR'] = str(dataset_dir)
        sys.path.insert(0, str(REPO))
        sys.dont_write_bytecode = True
        sys.settrace(stop_before_legacy_eval)
        try:
            spec.loader.exec_module(module)
        except ModelLoaded:
            pass
        else:
            raise Blocked('BLOCKED: frozen module crossed legacy evaluation boundary')
    finally:
        sys.settrace(saved_trace)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
        os.environ.clear()
        os.environ.update(saved_env)
    if not (module.LOAD_EMA == 1 and module.TAG_SEED == 33 and module.T_CAP == 128
            and module.STEPS2 == module._S1 == 0 and module.CONT_TRAIN == 0):
        raise Blocked('BLOCKED: s33/EMA/no-training contract mismatch')
    # Source loads EMA for the consumer chain, raw checkpoint encoder/decoder.
    names = ('cond_enc', 'cond_head', 'flow', 'ahead', 'bc_head',
             'traj_enc', 'e_pooler', 'u_dec')
    for name in names:
        model = getattr(module, name)
        model.eval()
        model.requires_grad_(False)
    for name, state in module._ck['ema'].items():
        actual = getattr(module, name).state_dict()
        if any(not torch.equal(actual[k], v.to(module.device)) for k, v in state.items()):
            raise Blocked('BLOCKED: EMA weights not delivered')
    module.probe_provenance = dict(source_path=str(SOURCE), source_sha256=verify_source(),
        ckpt_path=str(CKPT), ckpt_sha256=CKPT_PIN, dataset_path=str(dataset),
        dataset_sha256=file_sha(dataset), normalization_mu=module.MU_XY.tolist(),
        normalization_sd=module.SD_XY.tolist(), import_stop_line=boundary,
        config=config, deterministic_settings=settings)
    return module


def reset_policy(module, task, episode, draw):
    record = seeds(task, episode, draw)
    random.seed(record['python_seed'])
    module.torch.manual_seed(record['torch_seed'])
    module._GRAD_CACHE['u'] = None
    module._reseed_shuf(1000*task+episode)
    module._RDIR[0] = 1.0


def condition(module, obs, goal):
    return module.condvec(module.normstate(obs), module.normstate(module.goal_to_obs(goal)))


def interface_check(module, raw_points):
    """Actual pinned encode_u -> etarget -> loaded EMA ahead; no simulator."""
    from harness_move1 import FixedInjection, encode_trajectory
    torch = module.torch
    masks = []
    def observe_mask(model, args, kwargs):
        mask = kwargs['key_padding_mask']
        masks.append((tuple(mask.shape), bool(mask.any())))
    handle = module.e_pooler.register_forward_pre_hook(observe_mask, with_kwargs=True)
    try:
        u = encode_trajectory(module, raw_points)
    finally:
        handle.remove()
    injected = FixedInjection(module.ahead, u)
    with torch.no_grad():
        out = injected(condition(module, raw_points[0], raw_points[-1]))
    if masks != [((1, 128), False)] or tuple(out.shape) != (1, module.CHUNK, module.ADIM):
        raise Blocked('BLOCKED: real encoder-mask/head interface failed')
    if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(out).all()):
        raise Blocked('BLOCKED: nonfinite encoder/head output')
    return dict(encoder_shape=list(u.shape), head_shape=list(out.shape), mask_all_false=True,
                encoded_u_sha256=injected.expected, head_u_hashes=injected.head_hashes,
                source_sha256=PIN, ckpt_sha256=CKPT_PIN, production_evidence=False)


def run_draw(module, env, plan, *, move, waypoint=None, cap_steps=None, stuck_check=None, u=None):
    """Shared real env loop. No padding or reset after termination; fixed u bypasses flow."""
    from harness_move1 import FixedInjection, rng_snapshot
    from harness_move3 import TwoStage, make_collector, noisy_action
    task, episode, draw, arm = (plan[k] for k in ('task', 'episode', 'draw', 'arm'))
    if move == 1 and ((arm == 'N') != (u is None)):
        raise ValueError('Move 1 injection/arm mismatch')
    if move == 3 and u is not None:
        raise ValueError('Move 3 must use fresh flow')
    obs, info = reset_env(env, task, episode)
    initial, goal = np.asarray(obs).copy(), np.asarray(info['goal']).copy()
    reset_policy(module, task, episode, draw)
    collector = make_collector(task, episode, draw, obs, goal) if u is None else None
    injected = None if u is None else FixedInjection(module.ahead, u)
    state = TwoStage(arm if move == 3 else 'C', waypoint,
                     chunk_steps=module.CHUNK, cap_steps=cap_steps, initial_xy=initial[:2])
    before = rng_snapshot(module.torch, env, collector.noise if collector else None)
    successes, terminated, truncated, reason = [False], False, False, None
    while state.t < H and reason is None:
        target = waypoint if state.stage == 'w' else goal
        with module.torch.no_grad():
            cond = condition(module, obs, target)
            if injected is not None:
                actions = injected(cond)[0].cpu().numpy()
            else:
                state.sample(state.stage)
                sampled = collector.get_u(lambda: module.sample_plan(1, cond, None))
                actions = module.ahead(cond, module._q(sampled))[0].cpu().numpy()
        for action in actions:
            action = (noisy_action(collector, action) if move == 3 else
                      np.clip(action, -1., 1.).astype(np.float32))
            obs, _, terminated, truncated, info = env.step(action)
            success = bool(info.get('success', False))
            stuck = bool(stuck_check(state.xy + [list(obs[:2])])) if stuck_check else False
            state.observe(obs[:2], goal_success=success, stuck=stuck)
            successes.append(success)
            reason = ('success' if success else 'terminated' if terminated else
                      'truncated' if truncated else 'stuck' if move == 1 and stuck else state.reason if state.stage == 'stopped' else
                      'horizon' if state.t == H else None)
            if reason is not None or state.switch_required:
                break
    xy = np.asarray(state.xy, np.float32)
    row = dict(task=task, episode=episode, arm=arm, draw=draw,
        kind=plan.get('kind', 'rollout'), additional_draw=plan.get('additional_draw', True),
        synthetic=False, polluted=False, source_sha256=PIN,
        seeds=seeds(task, episode, draw, injected is not None),
        initial=initial.tolist(), goal=goal.tolist(), observation_dtype=str(initial.dtype),
        goal_dtype=str(goal.dtype), initial_sha256=digest_array(initial), goal_sha256=digest_array(goal),
        xy=xy.tolist(), trace_sha256=digest_array(xy), observed_steps=state.t,
        termination_reason=reason, terminated=bool(terminated), truncated=bool(truncated),
        success=any(successes), goal_success=successes, chunk_steps=module.CHUNK,
        flow_calls=state.calls, events=state.events, runtime_errors=state.errors,
        head_u_hashes=injected.head_hashes if injected else [],
        encoded_u_sha256=injected.expected if injected else None,
        waypoint_xy=None if waypoint is None else np.asarray(waypoint).tolist(),
        stuck_check_enabled=stuck_check is not None,
        rng_before=before, rng_after=rng_snapshot(module.torch, env, collector.noise if collector else None))
    return row


def calibrated_stuck(window, distance):
    if type(window) is not int or window < 1 or not np.isfinite(distance) or distance < 0:
        raise ValueError('explicit positive stuck window/nonnegative distance required')
    def check(xy):
        return len(xy) > window and bool(np.max(np.linalg.norm(
            np.asarray(xy[-window-1:]) - xy[-window-1], axis=1)) <= distance)
    return check


def execute(move, outdir, calibration, *, smoke=False, resume_from=None):
    """Plan -> model -> real runtime -> independent validation -> exclusive write.

    Production entry points enforce PRODUCTION_READY; smoke uses the same path
    with the smaller frozen inventory. Calibration is explicit, never guessed.
    """
    import json
    from builder import build
    from common import GATE, receipt, write_json
    import harness_move1 as m1
    import harness_move3 as m3
    import harvest
    from calibrate import validate_calibration
    from common import release_check
    from artifacts import save_row, resume_rows
    release_check(smoke)
    validate_calibration(calibration)  # before model/data loading or output creation
    builder = build()
    # Fail before model loading if independent reset evidence is unavailable.
    fingerprints = harvest.historical_fingerprints(json.loads(GATE.read_text()))
    outdir = Path(outdir)
    if outdir.resolve().is_relative_to(OUTPUT_ROOT) is False:
        raise Blocked('BLOCKED: runtime output must be under /archive/cymaxwelllee/breakthrough1')
    if outdir.exists():
        raise Blocked('BLOCKED: choose a fresh output directory')
    step_budget = calibration['steps_per_cell']
    stuck = calibrated_stuck(calibration['stuck_window'], calibration['stuck_distance'])
    module = load_frozen(calibration['dataset_dir'])
    env = module.ogbench.make_env_and_datasets(ENV, env_only=True)
    rows, repeats, skipped, representation, conformance = [], [], [], {}, []
    representation_evidence, calibration_samples = {}, {}
    refs, repeat_refs = [], []
    manifest = dict(receipt(builder), runtime=module.probe_provenance,
                    calibration=calibration, interface=None, smoke=smoke)
    reusable = {}
    if resume_from is not None:
        header, completed = resume_rows(resume_from)
        harvest.validate_receipt(header['receipt'], builder)
        harvest.require(header['move'] == move and header['builder'] == builder and
            all(header['receipt'][k] == manifest[k] for k in ('runtime', 'calibration', 'smoke')),
            'resume protocol/provenance mismatch')
        for row in completed:
            identity = (*harvest.key(row), row['kind'])
            harvest.require(identity not in reusable, 'duplicate resume row')
            reusable[identity] = row
    outdir.mkdir(parents=True, exist_ok=False)
    def raw_artifact():
        return dict(move=move, builder=builder, receipt=manifest, rows=refs,
                    conformance_rows=repeat_refs, skipped=skipped,
                    representation_evidence=representation_evidence,
                    calibration_samples=calibration_samples, storage='shards-v1', production_ready=False)
    encoded = {}
    xy_to_cell, cell_to_xy = env.unwrapped.xy_to_ij, env.unwrapped.ij_to_xy
    def route(task, arm):
        return [cell_to_xy(c) for c in builder['geometries'][str(task)]['routes'][arm]]
    def encode(task, arm):
        k = task, arm
        if k not in encoded:
            encoded[k] = m1.encode_trajectory(module, route(task, arm))
        return encoded[k]
    try:
        interface = interface_check(module, route(4, 'A'))
        manifest['interface'] = interface
        manifest['hard_prerequisites'] = dict(
            torch_deterministic=module.probe_provenance['deterministic_settings']['deterministic'],
            environment_policy_reset=True, rng_archived=True, dtype_batch_backend_frozen=True,
            head_deterministic=True, paired_protocol=True)
        write_json(outdir/'checkpoint.json', raw_artifact())
        plan = (m3 if move == 3 else m1).rollout_plan(builder, smoke)
        if move == 1:
            if smoke:
                for task in (1, 3, 4, 5):
                    calibration_samples[str(task)] = {arm: [m1.array_of(m1.encode_trajectory(
                        module, route(task, arm))).tolist() for _ in range(3)] for arm in 'AB'}
            for task in (4, 5):
                ua, ub = encode(task, 'A'), encode(task, 'B')
                decoded, decoded_xy = [], []
                for arm, u in zip('AB', (ua, ub)):
                    with module.torch.no_grad():
                        points = module._dec(u, module.normstate(route(task, arm)[0]))[0]
                        points = (points * module.SD_XY_T + module.MU_XY_T).cpu().numpy()
                    decoded.append(harvest.first_entry(points, builder['geometries'][str(task)]['exclusive'],
                                                       xy_to_cell)['route'])
                    decoded_xy.append(points.tolist())
                representation[str(task)] = m1.manipulation_check(
                    ua, ub, calibration['noise_band'][str(task)], *decoded)
                representation_evidence[str(task)] = dict(
                    u_A=m1.array_of(ua).tolist(), u_B=m1.array_of(ub).tolist(),
                    decoded_A=decoded_xy[0], decoded_B=decoded_xy[1])
            # N->P/R gate precedes main; original ordering within each group is stable.
            plan.sort(key=lambda p: (p['group'] != 'gate', p['kind'] != 'rollout'))
        n_rows, original_rows = {}, {}
        for p in plan:
            task, episode, arm = (p[k] for k in ('task', 'episode', 'arm'))
            k = task, episode
            kwargs = dict(stuck_check=stuck)
            if move == 3:
                geom = builder['geometries'][str(task)]
                kwargs = dict(waypoint=None if arm == 'C' else cell_to_xy(geom['w'][arm]),
                              cap_steps=min(H, int(np.ceil(geom['w_depth']*step_budget*2))),
                              stuck_check=stuck)
            else:
                if p['group'] == 'main' and not smoke and sum(v['success'] for v in n_rows.values()
                        if v['task'] in (1, 3)) < 5:
                    skipped.append(dict(p, skip_reason='gate-sample-insufficient'))
                    continue
                if p['conditional_on_N_success'] and not n_rows[k]['success']:
                    skipped.append(dict(p, skip_reason='gate-N-failed'))
                    continue
                if p['conditional_on_representation'] and not representation[str(task)]['passed']:
                    skipped.append(dict(p, skip_reason='representation-failed'))
                    continue
                if arm == 'R':
                    donor = builder['R_donor'][str(task)]
                    kwargs['u'] = encode(donor['task'], donor['route'])
                elif arm in 'PQ':
                    kwargs['u'] = (m1.encode_trajectory(module, m1.gate_p_source(n_rows[k])['xy'])
                                   if p['group'] == 'gate' else encode(task, 'A' if arm == 'P' else 'B'))
            identity = (*harvest.key(p), p.get('kind', 'rollout'))
            row = reusable.pop(identity, None)
            if row is None:
                row = run_draw(module, env, p, move=move, **kwargs)
            elif move == 1 and arm != 'N':
                harvest.require(row['encoded_u_sha256'] == digest_array(m1.array_of(kwargs['u']))
                    and row['injection_traj_sha256'] == kwargs['u'].probe_trajectory_sha256,
                    'resume injection differs')
            if move == 1:
                row['injection_source'] = p['source']
                if arm != 'N':
                    row['injection_traj_sha256'] = kwargs['u'].probe_trajectory_sha256
                if arm == 'R':
                    row['donor'] = builder['R_donor'][str(task)]
                if p['group'] == 'gate' and arm == 'P':
                    row['source_trace_sha256'] = n_rows[k]['trace_sha256']
            ref = save_row(outdir, row, len(refs)+len(repeat_refs))
            (repeat_refs if p.get('kind') == 'determinism-verification' else refs).append(ref)
            count = len(refs)+len(repeat_refs)
            if count % 50 == 0:
                print(f'PROGRESS move={move} rollouts={count}/{len(plan)} task={task} episode={episode}', flush=True)
            harvest.validate_rows([row], {harvest.key(row)}, fingerprints, move=move)
            if move == 3:
                row['independent_readout'] = harvest.recompute_move3(
                    row, kwargs['waypoint'], geometry=geom, xy_to_cell=xy_to_cell)
            else:
                row['first_entry'] = harvest.first_entry(row['xy'], builder['geometries'][str(task)]['exclusive'],
                                                        xy_to_cell)
                if arm == 'N':
                    n_rows[k] = row
            if p.get('kind') == 'determinism-verification':
                original = original_rows[(task, episode, arm)]
                a, b = np.asarray(original['xy'], np.float32), np.asarray(row['xy'], np.float32)
                conformance.append(dict(task=task, episode=episode, arm=arm, additional_draw=False,
                    kind='determinism-verification', passed=a.shape == b.shape and a.tobytes() == b.tobytes(),
                    first_sha256=original['trace_sha256'], second_sha256=row['trace_sha256']))
                repeats.append(row)
            else:
                rows.append(row)
                original_rows[(task, episode, arm)] = row
        harvest.require(not reusable, 'extra/conditional-skipped resume rows')
        raw = raw_artifact()
        write_json(outdir/'raw.json', raw)
        artifact = dict(raw, rows=rows, conformance_rows=repeats)
        artifact['readout'] = harvest.validate_artifact(artifact, builder, fingerprints,
                                                      cell_to_xy, xy_to_cell)
        write_json(outdir/'result.json', dict(raw, readout=artifact['readout']))
        return artifact
    except Exception as error:
        # Every completed rollout has already been persisted, including a bad row.
        if not (outdir/'raw.json').exists():
            write_json(outdir/'raw.json', raw_artifact())
        write_json(outdir/'validation_error.json', dict(error_type=type(error).__name__,
                   message=str(error), raw='raw.json', completed_rollouts=len(refs)+len(repeat_refs),
                   delivery_exit='E0' if 'interface' in str(error) or 'head' in str(error) else None))
        raise
    finally:
        env.close()
