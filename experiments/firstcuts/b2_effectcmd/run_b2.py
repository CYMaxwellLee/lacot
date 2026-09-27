#!/usr/bin/env python
"""B2: measured four-step effect table and fixed-route B / feedback C selectors."""
import argparse
import os
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
HERE = Path(__file__).resolve().parent
FIRSTCUTS = HERE.parent
sys.path.insert(0, str(FIRSTCUTS / 'b1_triarm'))
import run_b1 as b
import mujoco
import numpy as np

CHUNK = 4
K = 32
N_CALIBRATION = 40
N_PERTURB = 40
BOOTSTRAP = 10000
PREREGISTERED = {
    'status': 'preregistered', 'leg': 1, 'horizon': 200, 'chunk': CHUNK,
    'table_codes': K, 'table_steps_each': CHUNK,
    'expected_A_failures_200': 43, 'rescue_min': 12, 'harm_max': 2,
    'churn_status': 'churn-calibrated', 'churn_null_hits': '23/43',
    'paired_bootstrap_confidence': .95, 'bootstrap_replicates': BOOTSTRAP,
    'weight_calibration_tasks': N_CALIBRATION, 'perturbation_tasks': N_PERTURB,
    'perturbation_error_ratio_max': .5,
    'pulse_steps': 2, 'pulse_fraction_nominal_4step_rms': .5,
    'pulse_relative_tolerance': .01, 'pulse_max_directions': 16,
    'distance': 'body-frame xy metres squared + (frozen yaw metres_per_radian * wrap(yaw radians)) squared',
}


def write_json(path, value):
    b.write_json(path, value)


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def pose(qpos):
    q = np.asarray(qpos)
    return np.r_[q[:2], b.torso_yaw(q[None])[0]]


def relative(left, right):
    """SE(2) left inverse times right; translation is in left body coordinates."""
    left, right = np.asarray(left), np.asarray(right)
    dx, dy = right[:2] - left[:2]
    c, s = np.cos(left[2]), np.sin(left[2])
    return np.array([c*dx+s*dy, -s*dx+c*dy, wrap(right[2]-left[2])])


def score(effects, target, yaw_weight):
    d = np.asarray(effects) - target
    return np.sum(d[:, :2]**2, axis=1) + (yaw_weight*wrap(d[:, 2]))**2


def choose(effects, target, yaw_weight):
    return int(np.argmin(score(effects, target, yaw_weight)))


def advance(leg, env):
    step = leg['global_step']
    if leg['first_hit'] < 0 and np.linalg.norm(env.unwrapped.data.qpos[:2] - leg['target_xy']) <= b.RHO:
        leg['first_hit'] = step + 1
        leg['active_leg'] = 2
        leg['leg_reach_step']['1'] = step
    leg['global_step'] += 1


def measured_table(env, st, model):
    """Every complete K32 tuple is physically executed for four steps from st."""
    effects, shape_changes, endpoints = [], [], []
    for code in range(K):
        stream, _ = b.rollout(env, st, model=model, codes=np.array([code]))
        effects.append(relative(pose(stream['qpos'][0]), pose(stream['qpos'][-1])))
        shape_changes.append(stream['shape'][-1] - stream['shape'][0])
        endpoints.append(stream['qpos'][-1])
    return dict(codes=np.arange(K), effect=np.asarray(effects),
                shape_change=np.asarray(shape_changes), endpoint_qpos=np.asarray(endpoints),
                measured_steps=np.asarray(K*CHUNK))


def freeze_yaw_weight(references):
    """Use only teacher route increments from the preregistered calibration prefix."""
    xy, yaw = [], []
    for ref in references:
        poses = np.array([pose(q) for q in ref['qpos'][::CHUNK]])
        increments = np.array([relative(a, z) for a, z in zip(poses[:-1], poses[1:])])
        xy.extend(np.linalg.norm(increments[:, :2], axis=1))
        yaw.extend(np.abs(increments[:, 2]))
    xy_rms = float(np.sqrt(np.mean(np.square(xy))))
    yaw_rms = float(np.sqrt(np.mean(np.square(yaw))))
    if yaw_rms <= 1e-12:
        raise ValueError('Yaw calibration has no nonzero teacher increments')
    return xy_rms / yaw_rms


def run_arm(env, st, model, codes, table, ref, weight, arm, pulse=None):
    """B uses fixed g* left element; C alone uses g_live left element."""
    if arm not in ('A', 'B', 'C'):
        raise ValueError(arm)
    leg = b.restore_state(env, st)
    u = env.unwrapped
    lo, hi = env.action_space.low.astype(float), env.action_space.high.astype(float)
    records = {k: [] for k in ('qpos', 'qvel', 'obs', 'shape', 'leg_bookkeeping')}
    actions, selected, targets, live_poses, decoder_obs, pulse_realized = [], [], [], [], [], []
    teacher_pose = np.asarray([pose(q) for q in ref['qpos'][::CHUNK]])
    for t in range(b.HORIZON+1):
        records['qpos'].append(u.data.qpos.copy())
        records['qvel'].append(u.data.qvel.copy())
        records['obs'].append(b.sim_obs(u))
        records['shape'].append(b.body_shape(u))
        records['leg_bookkeeping'].append([leg['active_leg'], leg['global_step'], leg['first_hit']])
        if t == b.HORIZON:
            break
        if t % CHUNK == 0:
            k = t // CHUNK
            live = pose(u.data.qpos)
            left = live if arm == 'C' else teacher_pose[k]
            target = relative(left, teacher_pose[k+1])
            code = int(codes[k]) if arm == 'A' else choose(table['effect'], target, weight)
            seen = records['obs'][-1]
            chunk = b.decode(model, code, seen, lo, hi)
            selected.append(code)
            targets.append(target)
            live_poses.append(live)
            decoder_obs.append(seen)
        base = chunk[t % CHUNK]
        action = np.clip(base + pulse, lo, hi) if pulse is not None and t < 2 else base
        if pulse is not None and t < 2:
            pulse_realized.append(action-base)
        env.step(action)
        actions.append(action.copy())
        advance(leg, env)
    stream = {k: np.asarray(v) for k, v in records.items()}
    stream.update(action=np.asarray(actions), xy=np.asarray(records['qpos'])[:, :2],
                  yaw=b.torso_yaw(np.asarray(records['qpos'])),
                  selected_codes=np.asarray(selected), target_effect=np.asarray(targets),
                  live_pose_at_chunk=np.asarray(live_poses), teacher_pose=teacher_pose,
                  decoder_obs=np.asarray(decoder_obs),
                  realized_pulse=np.asarray(pulse_realized).reshape(-1, 8),
                  first_hit=np.asarray(leg['first_hit']), reached=np.asarray(leg['first_hit'] >= 0))
    return stream


def effect_errors(stream, table, weight):
    """Save command fit and a shared next-teacher-pose tracking error."""
    actual = np.array([relative(pose(stream['qpos'][k*CHUNK]),
                                pose(stream['qpos'][(k+1)*CHUNK])) for k in range(b.HORIZON//CHUNK)])
    common_targets = np.array([relative(live, target) for live, target in
        zip(stream['live_pose_at_chunk'], stream['teacher_pose'][1:])])
    command_errors = np.sqrt(score(actual, stream['target_effect'], weight))
    errors = np.sqrt(score(actual, common_targets, weight))
    stream['realized_effect'] = actual
    stream['common_target_effect'] = common_targets
    stream['command_effect_error'] = command_errors
    stream['effect_error'] = errors
    return float(np.mean(errors))


def bootstrap_paired(b_values, c_values):
    d = np.asarray(b_values, float) - np.asarray(c_values, float)
    rng = np.random.default_rng(b.SEED + 202)
    means = d[rng.integers(len(d), size=(BOOTSTRAP, len(d)))].mean(axis=1)
    return dict(B_minus_C=float(d.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                n_pairs=len(d), unit='whole_task', paired=True)


def pulse_calibration(env, st, model, code, nominal, task_index):
    """C2-style bounded first-two-action pulse; target is half nominal four-step tangent RMS."""
    q0, q4 = nominal['obs'][0], nominal['obs'][4]
    dq = np.empty(env.unwrapped.model.nv)
    mujoco.mj_differentiatePos(env.unwrapped.model, dq, 1., q0[:15], q4[:15])
    target = .5*float(np.sqrt(np.mean(dq**2)))
    if target <= 1e-12:
        raise ValueError('Zero nominal four-step displacement')
    rng = np.random.default_rng(np.random.SeedSequence([b.SEED, task_index, 202]))
    directions = rng.normal(size=(16, 8))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    count = 0
    lo, hi = env.action_space.low, env.action_space.high
    def trial(direction, amplitude):
        nonlocal count
        tape = nominal['action'][:4].copy()
        tape[:2] = np.clip(tape[:2] + direction*amplitude, lo, hi)
        result, _ = b.rollout(env, st, tape)
        delta = np.empty(env.unwrapped.model.nv)
        mujoco.mj_differentiatePos(env.unwrapped.model, delta, 1., q4[:15], result['obs'][-1, :15])
        count += 4
        return float(np.sqrt(np.mean(delta**2)))
    for direction in directions:
        lower, upper = 0., .25
        achieved = trial(direction, upper)
        while achieved < target and upper < 128.:
            lower, upper = upper, upper*2
            achieved = trial(direction, upper)
        if achieved < target:
            continue
        for _ in range(32):
            amplitude = (lower+upper)/2
            achieved = trial(direction, amplitude)
            if abs(achieved/target-1) <= .01:
                return direction*amplitude, dict(target=target, achieved=achieved,
                    relative_error=abs(achieved/target-1), n_steps=count)
            if achieved < target:
                lower = amplitude
            else:
                upper = amplitude
    raise RuntimeError('C2 pulse calibration failed; no fallback')


def load_inputs(args):
    return b.load_inputs(args)


def run_experiment(args):
    raw, model, tasks, env, provenance = load_inputs(args)
    out = args.out_dir
    if out.resolve().is_relative_to(FIRSTCUTS) is False:
        raise ValueError('Output must be under firstcuts')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists() or (out/'preregistered.json').exists():
        raise FileExistsError('Use a fresh output directory')
    summary = dict(status='running', n_tasks=len(tasks), n_arms=3, n_table_entries=0,
                   n_steps_total=0, preregistered=PREREGISTERED, provenance=provenance,
                   host=socket.gethostname())
    write_json(out/'preregistered.json', summary)
    steps = 0
    try:
        bundles, refs, baselines = [], [], []
        # All A baselines precede table and alternate arms. Main mismatch halts here.
        for ti, task in enumerate(tasks):
            td = out/f'task_{ti:03d}_ep{task["episode"]}'
            td.mkdir()
            write_json(td/'task.json', task)
            st = b.initial_state(env, raw, task)
            b.save_snapshot(td/'snapshot.npz', st)
            tape = raw['actions'][task['s0']:task['s0']+b.HORIZON]
            codes = b.encode_tape(model, tape)
            a, _ = b.rollout(env, st, model=model, codes=codes)
            steps += b.HORIZON
            np.savez_compressed(td/'A.npz', **a, codes=codes)
            bundles.append((td, st, codes, tape))
            baselines.append(a)
        failures = sum(not bool(a['reached']) for a in baselines)
        if len(tasks) == 200 and failures != 43:
            raise RuntimeError(f'A baseline mismatch: {failures} failures != 43; no other arms run')
        summary['baseline'] = dict(A_failures=failures,
            status='verified_43_of_200' if len(tasks)==200 else 'not_evaluated_smoke_only')
        old = b.relay.run_one(env, env.unwrapped, model, raw, tasks[0], 'schedule', b.RHO)
        common = old['n_steps_run']
        steps += common
        first = baselines[0]
        if bool(first['reached']) != (1 in old['leg_reach_step']) or not (
                np.array_equal(first['action'][:common], old['acts']) and
                np.array_equal(first['xy'][1:common+1], old['xy'])):
            raise AssertionError('A arm differs from original relay')
        summary['relay_check'] = dict(action_xy_bit_exact=True, common_steps=common)
        for td, st, codes, tape in bundles:
            teacher = np.clip(tape.astype(float), env.action_space.low, env.action_space.high)
            ref, _ = b.rollout(env, st, teacher)
            steps += b.HORIZON
            refs.append(ref)
            np.savez_compressed(td/'teacher_R.npz', **ref)
        ncal = min(len(tasks), N_CALIBRATION)
        weight = freeze_yaw_weight(refs[:ncal])
        summary['yaw_weight'] = dict(metres_per_radian=weight, n_calibration_tasks=ncal,
                                     status='frozen_before_effect_selection',
                                     inference='main' if len(tasks)==200 else 'smoke_only')
        rows = []
        for ti, ((td, st, codes, tape), ref, a) in enumerate(zip(bundles, refs, baselines)):
            table = measured_table(env, st, model)
            steps += K*CHUNK
            np.savez_compressed(td/'effect_table.npz', **table)
            arms = {}
            for name in ('B', 'C'):
                flow = run_arm(env, st, model, codes, table, ref, weight, name)
                steps += b.HORIZON
                effect_errors(flow, table, weight)
                b.metrics(flow, ref)
                np.savez_compressed(td/f'{name}.npz', **flow)
                arms[name] = flow
            same = int(np.sum(arms['B']['selected_codes'] == codes))
            row = dict(task_index=ti, episode=tasks[ti]['episode'], A=bool(a['reached']),
                       B=bool(arms['B']['reached']), C=bool(arms['C']['reached']),
                       B_schedule_matches=same, B_schedule_chunks=len(codes),
                       effect_error={name: float(np.mean(arms[name]['effect_error'])) for name in arms},
                       command_effect_error={name: float(np.mean(arms[name]['command_effect_error'])) for name in arms})
            if ti < min(len(tasks), N_PERTURB):
                # Match C2: calibrate a first-two-action pulse against the nominal first four steps.
                nominal, _ = b.rollout(env, st, model=model, codes=codes[:1])
                steps += 4
                pulse, calibration = pulse_calibration(env, st, model, codes[0], nominal, ti)
                steps += calibration['n_steps']
                write_json(td/'pulse_calibration.json', calibration)
                pert = {}
                for name in ('A', 'B', 'C'):
                    flow = run_arm(env, st, model, codes, table, ref, weight, name, pulse=pulse)
                    steps += b.HORIZON
                    effect_errors(flow, table, weight)
                    b.metrics(flow, ref)
                    np.savez_compressed(td/f'{name}_pulse.npz', **flow, requested_pulse=pulse)
                    pert[name] = dict(mean_effect_error=float(np.mean(flow['effect_error'])),
                                      switched_codes=int(np.sum(flow['selected_codes'] !=
                                          (codes if name == 'A' else arms[name]['selected_codes']))))
                row['perturbation'] = pert
            rows.append(row)
            write_json(td/'summary.json', row)
            summary['n_table_entries'] += K
            print(f'task={ti} A/B/C={int(row["A"])}/{int(row["B"])}/{int(row["C"])} '
                  f'B_schedule={same}/{len(codes)}', flush=True)
        decisions = {}
        for name in ('B', 'C'):
            rescued = sum(not r['A'] and r[name] for r in rows)
            harmed = sum(r['A'] and not r[name] for r in rows)
            decisions[name] = dict(rescued=rescued, harmed=harmed,
                threshold_met=(rescued >= 12 and harmed <= 2) if len(tasks)==200 else None,
                status='churn-calibrated' if len(tasks)==200 else 'smoke_only')
        pert_rows = [r for r in rows if 'perturbation' in r]
        perturbation = {name: dict(mean_effect_error=float(np.mean([r['perturbation'][name]['mean_effect_error'] for r in pert_rows])),
                    switched_codes=sum(r['perturbation'][name]['switched_codes'] for r in pert_rows)) for name in ('A', 'B', 'C')}
        for name in ('B', 'C'):
            base = perturbation['A']['mean_effect_error']
            perturbation[name]['error_ratio_vs_A_pulse'] = perturbation[name]['mean_effect_error']/base if base else None
            perturbation[name]['half_error_met'] = (perturbation[name]['error_ratio_vs_A_pulse'] <= .5
                and perturbation[name]['switched_codes'] > 0
                if len(pert_rows)==N_PERTURB and base else None)
        summary.update(status='complete', n_steps_total=steps, per_task=rows, decisions=decisions,
                       selection_consistency=dict(matches=sum(r['B_schedule_matches'] for r in rows),
                           chunks=sum(r['B_schedule_chunks'] for r in rows)),
                       B_C_failure_difference=bootstrap_paired([not r['B'] for r in rows], [not r['C'] for r in rows]),
                       perturbation=perturbation,
                       inferential_scope='main' if len(tasks)==200 else 'smoke_only_no_scientific_verdict')
        write_json(out/'summary.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', n_steps_total=steps, error=str(exc))
        write_json(out/'summary.json', summary)
        raise
    finally:
        env.close()


def main():
    p = b.parser(__doc__)
    p.set_defaults(out_dir=HERE/'results')
    args = p.parse_args()
    if args.n_tasks == 200 and not args.main:
        p.error('200 tasks require explicit --main')
    result = run_experiment(args)
    print(f'B2 COMPLETE n_tasks={result["n_tasks"]} n_arms={result["n_arms"]} '
          f'n_table_entries={result["n_table_entries"]} n_steps_total={result["n_steps_total"]}')


if __name__ == '__main__':
    main()
