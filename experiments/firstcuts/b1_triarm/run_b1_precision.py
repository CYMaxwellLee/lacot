#!/usr/bin/env python
"""Native-f64 snapshot replay: exact restore / qpos+qvel f32 cast / warmstart zero."""
import copy
import json
from pathlib import Path

import run_b1 as b
import numpy as np


def residuals(stream, ref):
    qp, qv = stream['qpos'][1:]-ref['qpos'][1:], stream['qvel'][1:]-ref['qvel'][1:]
    errors = dict(xy=np.linalg.norm(qp[:, :2], axis=1),
                  qpos_linf=np.max(np.abs(qp), axis=1),
                  qvel_linf=np.max(np.abs(qv), axis=1))
    report = {k: dict(one_step=float(v[0]), full_segment_rms=float(np.sqrt(np.mean(v*v))),
                      final=float(v[-1]), max=float(v.max())) for k, v in errors.items()}
    return errors, report


def run_precision(args):
    raw, _, tasks, env, provenance = b.load_inputs(args)
    summary = dict(status='running', n_tasks=len(tasks), n_arms=3, n_ulp=0,
                   arm_names=['i', 'ii', 'iii'], n_data_comparators=1,
                   n_steps_total=0, preregistered=b.PREREGISTERED, provenance=provenance,
                   primary_residual='xy L2; criterion compares means of paired one-step residuals',
                   e_data_definition='at same dataset index: qpos/qvel from val; warmstart=0 (unobserved); same clipped action',
                   controls={'i': 'complete native-f64 restore',
                             'ii': 'only qpos/qvel float64 -> float32 -> float64; no quaternion renormalization',
                             'iii': 'native-f64 qpos/qvel; only warmstart zeroed'})
    b.write_json(args.out_dir / 'preregistered.json', summary)
    total_steps, rows, full_restore_max_error = 0, [], 0.0
    try:
        source_summary = None
        if args.source_dir:
            source_summary = json.loads((args.source_dir / 'summary.json').read_text())
            if source_summary['status'] != 'complete' or source_summary['n_tasks'] != len(tasks):
                raise ValueError('Source B1 run must be complete and have the same tasks')
            for key in ('hashes', 'mujoco', 'seed'):
                if source_summary['provenance'][key] != provenance[key]:
                    raise ValueError(f'Source provenance mismatch: {key}')
        for ti, task in enumerate(tasks):
            td = args.out_dir / f'task_{ti:03d}_ep{task["episode"]}'
            td.mkdir()
            mid_step = b.HORIZON // 2
            if args.source_dir:
                src = args.source_dir / td.name
                st = b.load_snapshot(src / 'R_mid_snapshot.npz')
                with np.load(src / 'T.npz', allow_pickle=False) as z:
                    full_ref = {k: z[k].copy() for k in ('qpos', 'qvel', 'obs', 'action')}
            else:
                start = b.initial_state(env, raw, task)
                teacher = np.clip(raw['actions'][task['s0']:task['s0']+b.HORIZON].astype(float),
                                  env.action_space.low, env.action_space.high)
                full_ref, st = b.rollout(env, start, teacher, capture_step=mid_step)
                total_steps += b.HORIZON
                b.save_snapshot(td / 'initial_snapshot.npz', start)
                np.savez_compressed(td / 'R_full.npz', **full_ref)
            if st['qpos'].dtype != np.float64 or st['leg']['global_step'] != mid_step:
                raise AssertionError('Precision source must be a native f64 mid-R snapshot')
            b.save_snapshot(td / 'native_mid_snapshot.npz', st)
            ref = {k: full_ref[k][mid_step:] for k in ('qpos', 'qvel', 'obs', 'action')}
            tape = ref['action']
            np.savez_compressed(td / 'reference.npz', **ref)
            row = dict(ti=ti, episode=task['episode'], mid_step=mid_step, controls={})
            for name in ('i', 'ii', 'iii'):
                changed = copy.deepcopy(st)
                if name == 'ii':
                    for key in ('qpos', 'qvel'):
                        changed[key] = changed[key].astype(np.float32).astype(np.float64)
                elif name == 'iii':
                    changed['warmstart'][:] = 0
                b.save_snapshot(td / f'{name}_snapshot.npz', changed)
                stream, _ = b.rollout(env, changed, tape)
                total_steps += len(tape)
                errors, report = residuals(stream, ref)
                if name == 'i':
                    full_restore_max_error = max(full_restore_max_error, *(
                        float(np.max(np.abs(stream[key] - ref[key])))
                        for key in ('qpos', 'qvel', 'obs', 'action')))
                    b.assert_identical(stream, ref)
                    if any(np.any(v != 0) for v in errors.values()):
                        raise AssertionError('(i) complete restore must have exactly zero residual')
                report['initial_qpos_linf'] = float(np.max(np.abs(changed['qpos']-st['qpos'])))
                report['initial_qvel_linf'] = float(np.max(np.abs(changed['qvel']-st['qvel'])))
                report['initial_warmstart_linf'] = float(np.max(np.abs(changed['warmstart']-st['warmstart'])))
                if name == 'ii' and report['initial_qpos_linf'] == report['initial_qvel_linf'] == 0:
                    raise AssertionError('The f32 cast did not change the native state')
                row['controls'][name] = report
                np.savez_compressed(td / f'{name}.npz', **stream,
                                    **{f'residual_{k}': v for k, v in errors.items()})
            # Dataset lacks native warmstart: explicitly zero, do not inherit R's solver state.
            data_st = copy.deepcopy(st)
            idx = task['s0'] + mid_step
            data_st['qpos'] = raw['qpos'][idx].copy()
            data_st['qvel'] = raw['qvel'][idx].copy()
            data_st['warmstart'][:] = 0
            data_st['leg']['first_hit'] = -1
            data_st['leg']['active_leg'] = 1
            data_st['leg']['leg_reach_step'] = {}
            b.save_snapshot(td / 'data_snapshot.npz', data_st)
            data_replay, _ = b.rollout(env, data_st, tape)
            total_steps += len(tape)
            data_ref = dict(qpos=raw['qpos'][idx:idx+len(tape)+1],
                            qvel=raw['qvel'][idx:idx+len(tape)+1])
            if idx+len(tape) > task['e0']:
                raise ValueError('e_data reference crosses episode boundary')
            errors, data_report = residuals(data_replay, data_ref)
            np.savez_compressed(td / 'data_replay.npz', **data_replay,
                                data_qpos=data_ref['qpos'], data_qvel=data_ref['qvel'],
                                data_obs=raw['observations'][idx:idx+len(tape)+1],
                                **{f'residual_{k}': v for k, v in errors.items()})
            row['e_data'] = data_report
            for name in ('i', 'ii', 'iii'):
                den = data_report['xy']['one_step']
                num = row['controls'][name]['xy']['one_step']
                row['controls'][name]['one_step_fraction_of_e_data'] = num/den if den > 0 else None
            rows.append(row)
            b.write_json(td / 'summary.json', row)
            print(f'precision task={ti} i_xy={row["controls"]["i"]["xy"]["one_step"]:.9g} '
                  f'ii_xy={row["controls"]["ii"]["xy"]["one_step"]:.9g} '
                  f'iii_xy={row["controls"]["iii"]["xy"]["one_step"]:.9g} '
                  f'e_data_xy={data_report["xy"]["one_step"]:.9g}', flush=True)
        means = {k: float(np.mean([r['controls'][k]['xy']['one_step'] for r in rows])) for k in ('i', 'ii', 'iii')}
        edata = float(np.mean([r['e_data']['xy']['one_step'] for r in rows]))
        ratio = means['ii']/edata if edata > 0 else None
        summary.update(status='complete', n_steps_total=total_steps,
                       n_rollouts=len(tasks)*(4 if args.source_dir else 5), per_task=rows,
                       mean_one_step_xy={**means, 'e_data': edata},
                       precision_fraction_of_e_data=ratio,
                       precision_threshold_met=ratio >= .5 if ratio is not None else None,
                       inferential_scope='smoke_only_no_scientific_verdict' if len(tasks)<=4 else 'main',
                       full_restore_max_error=full_restore_max_error)
        b.write_json(args.out_dir / 'summary.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', error=str(exc), n_steps_total=total_steps)
        b.write_json(args.out_dir / 'summary.json', summary)
        raise
    finally:
        env.close()


def main():
    p = b.parser(__doc__)
    p.set_defaults(out_dir=b.HERE / 'precision_results')
    p.add_argument('--source-dir', type=Path, help='Optional completed B1 run with native mid-R snapshots')
    args = p.parse_args()
    summary = run_precision(args)
    print(f'B1 PRECISION PASS n_tasks={summary["n_tasks"]} n_arms=3 n_ulp=0 '
          f'n_steps_total={summary["n_steps_total"]} '
          f'full_restore_max_error={summary["full_restore_max_error"]:g}')


if __name__ == '__main__':
    main()
