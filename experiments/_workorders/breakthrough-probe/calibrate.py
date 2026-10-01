"""Estimate provisional calibration from hashed smoke artifacts; no guessed defaults.

Bootstrap smoke requires explicit provisional values using --template. Estimates
use maximum same-route encoding distance, observed waypoint steps/cell (90th
percentile), and low-displacement dwell on traces that demonstrably progressed.
These are estimators for smoke review, not a claim of GPU qualification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import Blocked, H, RHO, file_sha, write_json

REQUIRED = ('dataset_dir', 'steps_per_cell', 'stuck_window', 'stuck_distance', 'noise_band')


def validate_calibration(value):
    if not isinstance(value, dict):
        raise Blocked('BLOCKED: calibration must be an object')
    missing = [k for k in REQUIRED if k not in value]
    noise = value.get('noise_band', {})
    missing += ['noise_band.'+t for t in ('4', '5') if not isinstance(noise, dict) or t not in noise]
    if missing:
        raise Blocked('BLOCKED: missing calibration keys: '+', '.join(missing))
    def number(x, positive=False):
        return type(x) in (int, float) and np.isfinite(x) and (x > 0 if positive else x >= 0)
    if (not isinstance(value['dataset_dir'], str) or not value['dataset_dir'] or
        not number(value['steps_per_cell'], True) or type(value['stuck_window']) is not int or
        not 1 <= value['stuck_window'] < H or not number(value['stuck_distance']) or
        any(not number(noise[t]) for t in ('4', '5'))):
        raise Blocked('BLOCKED: invalid calibration values (template placeholders must be filled)')
    return value


def estimate(move1, move3):
    if move1['move'] != 1 or move3['move'] != 3 or not all(a['receipt']['smoke'] for a in (move1, move3)):
        raise Blocked('BLOCKED: calibration requires Move 1 and Move 3 smoke artifacts')
    samples = move1.get('calibration_samples', {})
    bands = {}
    for task in ('1', '3', '4', '5'):
        distances = []
        for arm in 'AB':
            values = np.asarray(samples.get(task, {}).get(arm, []), np.float64)
            if values.ndim != 4 or values.shape[0] < 3 or values.shape[1:] != (1, 8, 256) or not np.isfinite(values).all():
                raise Blocked(f'BLOCKED: missing same-route resampling evidence: {task}.{arm}')
            distances += [float(np.linalg.norm(a-b)) for i, a in enumerate(values) for b in values[i+1:]]
        bands[task] = max(distances)
    rates, progressing = [], []
    for row in move3['rows']:
        xy = np.asarray(row['xy'], float)
        if len(xy) < 2 or not np.isfinite(xy).all():
            raise Blocked('BLOCKED: missing/nonfinite smoke XY trace')
        if np.linalg.norm(xy[-1]-xy[0]) > RHO:
            progressing.append(xy)
        if row['arm'] == 'C':
            continue
        geom = move3['builder']['geometries'][str(row['task'])]
        # Runtime records the actual environment waypoint mapping for calibration.
        w = row.get('waypoint_xy')
        if w is None:
            raise Blocked('BLOCKED: missing waypoint_xy calibration evidence')
        hit = next((i for i, p in enumerate(xy) if np.linalg.norm(p-np.asarray(w)) < RHO), None)
        if hit is not None and hit > 0:
            rates.append(hit/geom['w_depth'])
    if not rates or not progressing:
        raise Blocked('BLOCKED: cap/stuck estimation needs reached waypoints and progressing traces')
    displacements = np.concatenate([np.linalg.norm(np.diff(x, axis=0), axis=1) for x in progressing])
    positive = displacements[displacements > 0]
    if not len(positive):
        raise Blocked('BLOCKED: no measurable progress for stuck calibration')
    distance = float(np.quantile(positive, .1))
    longest = 0
    for xy in progressing:
        start = 0
        for end in range(1, len(xy)):
            if np.linalg.norm(xy[end]-xy[start]) > distance:
                longest = max(longest, end-start-1)
                start = end
    window = min(H-1, max(1, 2*(longest+1)))
    result = dict(dataset_dir=move1['receipt']['calibration']['dataset_dir'],
                  steps_per_cell=float(np.quantile(rates, .9)), stuck_window=window,
                  stuck_distance=distance, noise_band={t: max(bands[t], bands['1'], bands['3']) for t in ('4', '5')},
                  calibrated=True, provisional=True,
                  estimator=dict(noise='max pairwise same-route re-encoding L2; gate 1/3 conservative floor',
                                 steps_per_cell='p90 observed waypoint steps / depth; runtime cap multiplies by 2',
                                 stuck='twice longest low-displacement dwell + 2; displacement p10 positive steps',
                                 reached_waypoints=len(rates), progressing_traces=len(progressing), same_route_bands=bands))
    return validate_calibration(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', action='store_true')
    p.add_argument('--move1', type=Path)
    p.add_argument('--move3', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.template:
            result = dict(dataset_dir=None, steps_per_cell=None, stuck_window=None, stuck_distance=None,
                          noise_band={'4': None, '5': None}, calibrated=False, provisional=True)
        else:
            if not args.move1 or not args.move3:
                raise Blocked('BLOCKED: --move1 and --move3 smoke artifacts required')
            from artifacts import load_artifact
            from builder import build
            from common import ENV, GATE
            from harvest import validate_artifact, historical_fingerprints, require
            import ogbench
            one, three = load_artifact(args.move1), load_artifact(args.move3)
            require(one['receipt'].get('runtime') == three['receipt'].get('runtime'),
                    'calibration runtime mismatch')
            builder = build()
            fingerprints = historical_fingerprints(json.loads(GATE.read_text()))
            env = ogbench.make_env_and_datasets(ENV, env_only=True)
            try:
                for artifact in (one, three):
                    validate_artifact(artifact, builder, fingerprints, env.unwrapped.ij_to_xy, env.unwrapped.xy_to_ij)
                result = estimate(one, three)
            finally:
                env.close()
            result['inputs'] = {str(p): file_sha(p) for p in (args.move1, args.move3)}
        write_json(args.out, result)
    except (Blocked, ValueError, KeyError) as e:
        p.exit(2, str(e)+'\n')


if __name__ == '__main__':
    main()
