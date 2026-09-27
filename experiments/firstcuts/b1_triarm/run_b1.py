#!/usr/bin/env python
"""B1: matched-snapshot T/N/L, clipped action doses, and float32-ULP ensemble."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
HERE = Path(__file__).resolve().parent
FIRSTCUTS = HERE.parent
WV = FIRSTCUTS.parent / 'walk_verify'
sys.path.insert(0, str(WV))
sys.path.insert(0, str(WV / 'teacher_relay'))
import mujoco
import numpy as np
import torch
import wv_common as wv
import run_teacher_relay_byleg as relay
from p1_replay import sim_obs

RHO = 1.875
SEED = 20260908
HORIZON = 200
ALPHAS = (0., 0.9, 0.99, 1.)
N_ULP = 24
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
PREREGISTERED = {
    'rho': RHO, 'horizon': HORIZON, 'leg': 1, 'outcome': 'reached',
    'expected_main_tasks': 200, 'expected_L_failures': 43,
    'alpha': list(ALPHAS), 'n_ulp': N_ULP, 'stable_failure': '0 successes / 24',
    'residual_reduction_max_ratio': 0.1, 'paired_bootstrap_confidence': 0.95,
    'equivalence_failure_rate_margin': 0.05,
    'bootstrap_replicates': 10000, 'precision_e_data_fraction': 0.5,
    'negative_offset_steps': 3, 'negative_action_linf_min': 1e-6,
    'negative_qpos_linf_min': 1e-6,
    'status': 'preregistered',
    'churn': {'rescued_min': 12, 'harmed_max': 2, 'null_hits': '23/43',
              'status': 'churn-calibrated'},
    'old_new': {'workorder': {'alpha': [0, 0.1, 1], 'CI': 0.95, 'precision': 0.5},
                'implemented': {'alpha': list(ALPHAS), 'CI': 0.95, 'precision': 0.5},
                'reason': '原網格無十倍小殘差點'},
}


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, default=jsonable, indent=2, allow_nan=False) + '\n')


def wrappers(env):
    while True:
        yield env
        if env is env.unwrapped:
            break
        env = env.env


def rng_state(env):
    legacy = np.random.get_state()
    return dict(python=random.getstate(), numpy=[legacy[0], legacy[1].tolist(), *legacy[2:]],
                torch=torch.get_rng_state().tolist(),
                generators=[copy.deepcopy(x.np_random.bit_generator.state) for x in
                            (env.unwrapped, env.action_space, env.observation_space)])


def restore_rng(env, st):
    def tuples(x):
        return tuple(tuples(v) for v in x) if isinstance(x, (tuple, list)) else x
    random.setstate(tuples(st['python']))
    n = st['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(torch.tensor(st['torch'], dtype=torch.uint8))
    for obj, state in zip((env.unwrapped, env.action_space, env.observation_space), st['generators']):
        obj.np_random.bit_generator.state = copy.deepcopy(state)


def save_state(env, leg):
    """VQOracle save/restore convention, extended with integration inputs and RNG."""
    u, d = env.unwrapped, env.unwrapped.data
    integration = np.empty(mujoco.mj_stateSize(u.model, STATE_SPEC))
    mujoco.mj_getState(u.model, d, integration, STATE_SPEC)
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), act=d.act.copy(),
                warmstart=d.qacc_warmstart.copy(), time=float(d.time),
                integration=integration, rng=rng_state(env), leg=copy.deepcopy(leg),
                wrappers=[{k: copy.deepcopy(v) for k, v in vars(w).items() if k in
                           ('_elapsed_steps', '_has_reset', 'checked_reset', 'checked_step',
                            'checked_render', 'close_called')} for w in wrappers(env)],
                env_task={k: copy.deepcopy(getattr(u, k)) for k in
                          ('cur_task_id', 'cur_task_info', 'cur_goal_xy')})


def restore_state(env, st):
    u, d = env.unwrapped, env.unwrapped.data
    # Same set_state -> act/time/warmstart ordering as lacot_vqo.py.
    # INTEGRATION additionally retains ctrl, applied forces, mocap, eq_active, etc.
    mujoco.mj_setState(u.model, d, st['integration'], STATE_SPEC)
    u.set_state(st['qpos'], st['qvel'])
    d.act[:] = st['act']
    d.time = st['time']
    d.qacc_warmstart[:] = st['warmstart']
    for w, values in zip(wrappers(env), st['wrappers']):
        for key, value in values.items():
            setattr(w, key, copy.deepcopy(value))
    for key, value in st['env_task'].items():
        setattr(u, key, copy.deepcopy(value))
    u.set_goal(goal_xy=np.asarray(st['env_task']['cur_goal_xy']))
    restore_rng(env, st['rng'])
    return copy.deepcopy(st['leg'])


def save_snapshot(path, st):
    numeric = ('qpos', 'qvel', 'act', 'warmstart', 'time', 'integration')
    np.savez_compressed(path, **{k: st[k] for k in numeric},
                        metadata_json=np.asarray(json.dumps(
                            {k: v for k, v in st.items() if k not in numeric}, default=jsonable)))


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as z:
        st = json.loads(str(z['metadata_json']))
        st.update({k: z[k].copy() for k in ('qpos', 'qvel', 'act', 'warmstart', 'integration')})
        st['time'] = float(z['time'])
    return st


def initial_state(env, raw, task):
    env.reset()
    env.unwrapped.set_state(raw['qpos'][task['s0']].copy(), raw['qvel'][task['s0']].copy())
    return save_state(env, dict(active_leg=1, global_step=0, first_hit=-1,
                               leg_active_step={'1': 0}, leg_reach_step={},
                               leg_start_xy={'1': raw['qpos'][task['s0'], :2].tolist()},
                               target_xy=task['wp_xy'][0].tolist()))


def encode_tape(model, actions):
    # One chunk at a time: retain relay's exact inference/batch shape.
    with torch.no_grad():
        return np.array([int(model.encode_idx(torch.from_numpy(
            a.reshape(1, -1).astype(np.float32))).item()) for a in actions.reshape(-1, 4, 8)])


def decode(model, code, obs, lo, hi):
    with torch.no_grad():
        a = model.decode_from_idx(torch.tensor([int(code)]),
                                  torch.from_numpy(np.asarray(obs, np.float32)[None])).numpy().reshape(4, 8)
    return np.clip(a, lo, hi).astype(np.float64)


def reference_tape(model, codes, ref, lo, hi, offset=0):
    indices = np.arange(len(codes)) * 4 + offset
    if indices.min() < 0 or indices.max() >= len(ref['obs']):
        raise ValueError('Reference observation offset leaves R; never silently clamp/wrap')
    inputs = ref['obs'][indices].copy()
    actions = np.concatenate([decode(model, c, obs, lo, hi) for c, obs in zip(codes, inputs)])
    return actions, inputs, indices


def mix_actions(t, n, alpha):
    if alpha not in ALPHAS:
        raise ValueError('Only preregistered alpha values are allowed')
    return (1 - alpha) * n + alpha * t


def torso_yaw(qpos):
    quat = qpos[:, 3:7]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    w, x, y, z = quat.T
    return np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))


def body_shape(u):
    """All robot body COM positions in root-centered, yaw-removed coordinates."""
    # mj_step's derived xpos can lag the final state; compute on scratch MjData.
    d = mujoco.MjData(u.model)
    d.qpos[:] = u.data.qpos
    d.qvel[:] = u.data.qvel
    mujoco.mj_forward(u.model, d)
    yaw = torso_yaw(d.qpos[None])[0]
    points = d.xipos[1:].copy() - d.qpos[:3]
    c, s = np.cos(yaw), np.sin(yaw)
    return points @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rollout(env, st, actions=None, *, model=None, codes=None, capture_step=None):
    """Fixed 200-step horizon, even after hit, so every arm has aligned full streams."""
    leg = restore_state(env, st)
    u = env.unwrapped
    n = len(actions) if actions is not None else len(codes)*4
    lo, hi = env.action_space.low.astype(float), env.action_space.high.astype(float)
    qpos, qvel, obs, shape = [], [], [], []
    act, decoder_obs, decoder_steps, leg_rows = [], [], [], []
    midpoint = None
    for t in range(n + 1):
        qpos.append(u.data.qpos.copy())
        qvel.append(u.data.qvel.copy())
        obs.append(sim_obs(u))
        shape.append(body_shape(u))
        leg_rows.append([leg['active_leg'], leg['global_step'], leg['first_hit']])
        if t == capture_step:
            midpoint = save_state(env, leg)
        if t == n:
            break
        if actions is None:
            if t % 4 == 0:
                decoder_obs.append(obs[-1].copy())
                decoder_steps.append(leg['global_step'])
                chunk = decode(model, codes[t // 4], obs[-1], lo, hi)
            a = chunk[t % 4]
        else:
            a = actions[t]
        env.step(a)
        act.append(a.copy())
        step = leg['global_step']
        if leg['first_hit'] < 0 and np.linalg.norm(u.data.qpos[:2] - leg['target_xy']) <= RHO:
            leg['first_hit'] = step + 1  # first-hit count is one based, same as byleg rows.
            leg['active_leg'] = 2
            leg['leg_reach_step']['1'] = step
        leg['global_step'] += 1
    stream = dict(qpos=np.asarray(qpos), qvel=np.asarray(qvel), obs=np.asarray(obs),
                  action=np.asarray(act), xy=np.asarray(qpos)[:, :2],
                  yaw=torso_yaw(np.asarray(qpos)), shape=np.asarray(shape),
                  decoder_obs=np.asarray(decoder_obs, dtype=np.float32).reshape(-1, 29),
                  decoder_ref_step=np.asarray(decoder_steps, dtype=np.int64),
                  decoder_obs_source=np.asarray('own' if actions is None else 'none'),
                  leg_bookkeeping=np.asarray(leg_rows),
                  first_hit=np.asarray(leg['first_hit']), reached=np.asarray(leg['first_hit'] >= 0))
    return stream, midpoint


def metrics(stream, ref):
    dyaw = stream['yaw'] - ref['yaw']
    stream['xy_error'] = np.linalg.norm(stream['xy'] - ref['xy'], axis=1)
    stream['yaw_error'] = np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
    stream['shape_error'] = np.sqrt(np.mean(np.sum((stream['shape']-ref['shape'])**2, axis=-1), axis=1))
    return {key: dict(mean=float(stream[key][1:].mean()), final=float(stream[key][-1]),
                      max=float(stream[key].max())) for key in ('xy_error', 'yaw_error', 'shape_error')}


def perturb_ulp(st, model, rng):
    out = copy.deepcopy(st)
    for key in ('qpos', 'qvel'):
        f = st[key].astype(np.float32)
        direction = np.where(rng.integers(0, 2, size=f.shape), np.float32(np.inf), np.float32(-np.inf))
        # One float32 spacing about the original f64 point; no wholesale f32 cast.
        out[key] = st[key] + (np.nextafter(f, direction).astype(np.float64) - f.astype(np.float64))
    for j, jt in enumerate(model.jnt_type):
        if jt in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            start = int(model.jnt_qposadr[j]) + (3 if jt == mujoco.mjtJoint.mjJNT_FREE else 0)
            quat = out['qpos'][start:start+4]
            quat /= np.linalg.norm(quat)
    return out


def bootstrap_difference(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(SEED + 101)
    means = diff[rng.integers(0, len(diff), size=(PREREGISTERED['bootstrap_replicates'], len(diff)))].mean(axis=1)
    return dict(mean=float(diff.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                paired=True, n_pairs=len(diff))


def baseline_gate(n_tasks, failures):
    if n_tasks == 200 and failures != 43:
        raise RuntimeError(f'L baseline mismatch: {failures} failures != 43; no other arms run')
    return 'verified_43_of_200' if n_tasks == 200 else 'not_evaluated_smoke_only'


def local_data(path):
    path = Path(path).expanduser().resolve()
    file = path / (relay.DATASET + '-val.npz')
    if not file.is_file():
        raise FileNotFoundError(file)
    fs = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(file)], text=True).strip()
    if fs in ('nfs', 'nfs4') or fs.startswith(('cifs', 'smb', 'fuse.sshfs')):
        raise ValueError(f'NFS/network datasets forbidden: {file} ({fs})')
    return path, fs


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-dir', default=os.environ.get('OGBENCH_DATA_DIR', '/home/cymaxwelllee/.ogbench/data'))
    p.add_argument('--n-tasks', type=int, choices=[1, 2, 3, 4, 200], default=2)
    p.add_argument('--main', action='store_true', help='Explicit future authorization for 200 tasks; not used in delivery')
    p.add_argument('--ckpt', default=relay.DEFAULT_CKPT)
    p.add_argument('--ruler', default=relay.DEFAULT_RULER)
    p.add_argument('--out-dir', type=Path, default=HERE / 'results')
    return p


def load_inputs(args):
    if args.n_tasks == 200 and not args.main:
        raise ValueError('200-task main experiment disabled without --main')
    data, fs = local_data(args.data_dir)
    if not args.out_dir.resolve().is_relative_to(FIRSTCUTS):
        raise ValueError('Outputs must remain under experiments/firstcuts/')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / 'summary.json').exists():
        raise FileExistsError('Use a fresh --out-dir; never overwrite an experiment')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    raw = wv.load_npz(str(data), relay.DATASET, 'val')
    model, cfg = wv.load_dict_v1(args.ckpt)
    if cfg['seg_len'] != 4 or cfg['k'] != 32:
        raise ValueError('B1 requires L4K32')
    tasks, _ = relay.build_tasks(raw, args.n_tasks, SEED, json.loads(Path(args.ruler).read_text()), RHO)
    if len(tasks) != args.n_tasks or any(t['n_chunks'] * 4 != HORIZON for t in tasks):
        raise ValueError('Wrong task count or horizon; expected original 200-step tasks')
    env = wv.make_env(str(data), relay.DATASET)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    if env.unwrapped._teleport_info is not None:
        raise ValueError('Only non-teleport ant medium-stitch is supported')
    provenance = dict(host=socket.gethostname(), mujoco=mujoco.__version__, numpy=np.__version__,
                      torch=torch.__version__, data_path=raw['_path'], filesystem=fs,
                      seed=SEED, frame_skip=env.unwrapped.frame_skip,
                      snapshot_convention='lacot-vqoracle/experiments/lacot_vqo.py:86 + RNG/leg/INTEGRATION',
                      hashes={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in
                              [('data', raw['_path']), ('ckpt', args.ckpt), ('ruler', args.ruler)]})
    return raw, model, tasks, env, provenance


def assert_identical(a, b):
    for key in ('qpos', 'qvel', 'obs', 'action'):
        if not np.array_equal(a[key], b[key]):
            raise AssertionError(f'Full restore not exact: {key}, max={np.max(np.abs(a[key]-b[key]))}')


def run_experiment(args):
    raw, model, tasks, env, provenance = load_inputs(args)
    summary = dict(status='running', n_tasks=len(tasks), n_arms=3, n_ulp=N_ULP,
                   n_dose_arms=len(ALPHAS), n_control_rollouts_per_task=2,
                   arm_names=['T', 'N', 'L'], control_names=['T_replay', 'N_offset3'],
                   n_steps_total=0, n_rollouts=0, preregistered=PREREGISTERED, provenance=provenance)
    write_json(args.out_dir / 'preregistered.json', summary)
    starts, baselines, rows = [], [], []
    total_steps = 0
    try:
        # ALL L tasks precede T/N/doses/ULP. A 200-task baseline mismatch halts here.
        for ti, task in enumerate(tasks):
            td = args.out_dir / f'task_{ti:03d}_ep{task["episode"]}'
            td.mkdir()
            write_json(td / 'task.json', task)
            st = initial_state(env, raw, task)
            save_snapshot(td / 'snapshot.npz', st)
            teacher = np.clip(raw['actions'][task['s0']:task['s0']+HORIZON].astype(float),
                              env.action_space.low, env.action_space.high)
            codes = encode_tape(model, raw['actions'][task['s0']:task['s0']+HORIZON])
            l, _ = rollout(env, st, model=model, codes=codes)
            total_steps += HORIZON
            np.savez_compressed(td / 'L.npz', **l, codes=codes)
            starts.append((st, teacher, codes, td))
            baselines.append(l)
        failures = sum(not bool(l['reached']) for l in baselines)
        summary['baseline'] = dict(L_failures=failures, status=baseline_gate(len(tasks), failures))
        # One direct old-code comparison, including every common action/xy, not only outcome.
        old = relay.run_one(env, env.unwrapped, model, raw, tasks[0], 'schedule', RHO)
        total_steps += old['n_steps_run']
        common = old['n_steps_run']
        l = baselines[0]
        if bool(l['reached']) != (1 in old['leg_reach_step']):
            raise AssertionError('L vs original relay leg-1 reached mismatch')
        if not np.array_equal(l['action'][:common], old['acts']) or not np.array_equal(l['xy'][1:common+1], old['xy']):
            raise AssertionError('L vs original relay action/xy mismatch')
        summary['relay_check'] = dict(episode=tasks[0]['episode'], reached=bool(l['reached']),
                                      original_reached=1 in old['leg_reach_step'], common_steps=common,
                                      action_xy_bit_exact=True)
        for ti, (task, bundle, l) in enumerate(zip(tasks, starts, baselines)):
            st, teacher, codes, td = bundle
            ref, mid = rollout(env, st, teacher, capture_step=HORIZON//2)
            replay, _ = rollout(env, st, teacher)
            total_steps += 2*HORIZON
            assert_identical(ref, replay)
            save_snapshot(td / 'R_mid_snapshot.npz', mid)
            np.savez_compressed(td / 'T_replay.npz', **replay)
            n_tape, n_obs, n_idx = reference_tape(model, codes, ref, env.action_space.low, env.action_space.high)
            n, _ = rollout(env, st, n_tape)
            n.update(decoder_obs=n_obs, decoder_ref_step=n_idx, decoder_obs_source=np.asarray('teacher_R'))
            wrong_tape, wrong_obs, wrong_idx = reference_tape(model, codes, ref, env.action_space.low,
                                                            env.action_space.high, offset=3)
            wrong, _ = rollout(env, st, wrong_tape)
            wrong.update(decoder_obs=wrong_obs, decoder_ref_step=wrong_idx,
                         decoder_obs_source=np.asarray('teacher_R_offset3'))
            total_steps += 2*HORIZON
            da = float(np.max(np.abs(n['action']-wrong['action'])))
            dq = float(np.max(np.abs(n['qpos']-wrong['qpos'])))
            if da <= 1e-6 or dq <= 1e-6:
                raise AssertionError('N offset+3 negative control has no visible action/state effect')
            arm_metrics = {}
            for name, stream in [('T', ref), ('N', n), ('L', l), ('N_offset3', wrong)]:
                arm_metrics[name] = metrics(stream, ref)
                np.savez_compressed(td / f'{name}.npz', **stream, codes=codes)
            doses = []
            for alpha in ALPHAS:
                tape = mix_actions(teacher, n_tape, alpha)
                d, _ = rollout(env, st, tape)
                total_steps += HORIZON
                if alpha in (0., 1.):
                    assert_identical(d, n if alpha == 0. else ref)
                dm = metrics(d, ref)
                residual = float(np.sqrt(np.mean((tape-teacher)**2)))
                doses.append(dict(alpha=alpha, reached=bool(d['reached']), action_rms_vs_T=residual,
                                  **dm))
                np.savez_compressed(td / f'dose_{alpha:g}.npz', **d, codes=codes)
            probabilities = []
            for ui in range(N_ULP):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, ti, ui]))
                perturbed = perturb_ulp(st, env.unwrapped.model, rng)
                ulp, _ = rollout(env, perturbed, model=model, codes=codes)
                total_steps += HORIZON
                metrics(ulp, ref)
                np.savez_compressed(td / f'ulp_{ui:02d}.npz', **ulp, codes=codes,
                                    initial_delta_qpos=perturbed['qpos']-st['qpos'],
                                    initial_delta_qvel=perturbed['qvel']-st['qvel'])
                probabilities.append(bool(ulp['reached']))
            row = dict(ti=ti, episode=task['episode'], T=bool(ref['reached']), N=bool(n['reached']),
                       L=bool(l['reached']), T_replay_max_error=0.0,
                       negative_offset3=dict(action_linf=da, qpos_linf=dq), metrics=arm_metrics,
                       doses=doses, ulp_successes=sum(probabilities),
                       ulp_success_probability=float(np.mean(probabilities)),
                       stable_failure=not any(probabilities))
            rows.append(row)
            write_json(td / 'summary.json', row)
            print(f'task={ti} episode={task["episode"]} T/N/L={int(row["T"])}/{int(row["N"])}/{int(row["L"])} '
                  f'ULP={sum(probabilities)}/{N_ULP} offset3_action_linf={da:.6g}', flush=True)
        table = [[sum(r['L'] == l and r['N'] == n for r in rows) for n in (False, True)] for l in (False, True)]
        dose_summary = []
        for i, alpha in enumerate(ALPHAS):
            cur = [r['doses'][i] for r in rows]
            bs = bootstrap_difference([not d['reached'] for d in cur], [not r['N'] for r in rows])
            ratios = [d['action_rms_vs_T']/r['doses'][0]['action_rms_vs_T']
                      if r['doses'][0]['action_rms_vs_T'] > 0 else None for d, r in zip(cur, rows)]
            dose_summary.append(dict(alpha=alpha, failure_rate=float(np.mean([not d['reached'] for d in cur])),
                                     residual_ratios=ratios, failure_difference_vs_N=bs,
                                     continuous_means={k: float(np.mean([d[k]['mean'] for d in cur]))
                                                       for k in ('xy_error', 'yaw_error', 'shape_error')}))
        tenfold = next(d for d in dose_summary if d['alpha'] == 0.9)
        ci = tenfold['failure_difference_vs_N']['ci95']
        equivalent = max(abs(bound) for bound in ci) <= PREREGISTERED['equivalence_failure_rate_margin']
        verdict = ('equivalent_failure_within_margin' if equivalent else
                   'difference_detected' if ci[0] > 0 or ci[1] < 0 else 'difference_not_detected')
        summary.update(status='complete', n_steps_total=total_steps,
                       n_rollouts=len(tasks)*(9+N_ULP)+1, per_task=rows,
                       rescue_table=dict(rows='L fail, L success', columns='N fail, N success', counts=table),
                       rescued=table[0][1], harmed=table[1][0],
                       churn_threshold_met=(table[0][1] >= 12 and table[1][0] <= 2) if len(tasks)==200 else None,
                       dose_curve=dose_summary,
                       saturation_verdict=verdict,
                       ulp_success_probability=[r['ulp_success_probability'] for r in rows],
                       stable_failure_rate=float(np.mean([r['stable_failure'] for r in rows])),
                       inferential_scope='main' if len(tasks)==200 else 'smoke_only_no_scientific_verdict')
        write_json(args.out_dir / 'summary.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', error=str(exc), n_steps_total=total_steps)
        write_json(args.out_dir / 'summary.json', summary)
        raise
    finally:
        env.close()


def main():
    args = parser(__doc__).parse_args()
    result = run_experiment(args)
    print(f'B1 SMOKE PASS n_tasks={result["n_tasks"]} n_arms={result["n_arms"]} '
          f'n_ulp={result["n_ulp"]} n_steps_total={result["n_steps_total"]}' if args.n_tasks <= 4 else
          f'B1 MAIN COMPLETE n_tasks=200 n_arms={result["n_arms"]} '
          f'n_ulp={result["n_ulp"]} n_steps_total={result["n_steps_total"]}')


if __name__ == '__main__':
    main()
