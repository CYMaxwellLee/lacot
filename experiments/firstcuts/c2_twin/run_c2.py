#!/usr/bin/env python
"""C2 A: physical twins, mismatched innovations, and frozen-metric Gamma ledger."""
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
sys.path.insert(0, str(FIRSTCUTS / 'b1_triarm'))
import run_b1 as b1
import mujoco
import numpy as np
import torch
import wv_common as wv
import run_teacher_relay_byleg as relay
from p1_replay import sim_obs

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

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



def local_data(path):
    path = Path(path).expanduser().resolve()
    file = path / (relay.DATASET + '-val.npz')
    if not file.is_file():
        raise FileNotFoundError(file)
    fs = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(file)], text=True).strip()
    if fs in ('nfs', 'nfs4') or fs.startswith(('cifs', 'smb', 'fuse.sshfs')):
        raise ValueError(f'NFS/network datasets forbidden: {file} ({fs})')
    return path, fs


SEED = 20260927
CHUNK = 4
PULSE_STEPS = 2
PREFIX = 4
WINDOW = 32
DOSES = (0., .5, 1.)
CONDITIONS = (('0', 0.), ('0.5', .5), ('1', 1.), ('flip', -1.))
SPACES = ('gait', 'xy', 'yaw')
PRIMARY_CODEC = 'p0_dict_v1'
CODEC_LABELS = (PRIMARY_CODEC, 'sigma0', 'sigma1.0')
FALL_LINE_P1 = .25730210542678833
SUBSPACE_NOTE = ('C2 gait=39 維 body-COM 形狀（W 加權、含 z、去 yaw、無速度）；'
                 'xy/yaw 也經 W 加權。C1 project() 是 SE(2)-quotient＋速度、'
                 '訓練 sd 正規化的子空間；兩邊 per-subspace 數字不可同表比。')
PREREGISTERED = dict(
    status='preregistered', leg=1, branch_step=0, pulse_steps=2,
    pulse_fraction_of_nominal_4step_displacement_rms=.5,
    calibration_relative_tolerance=.01, calibration_max_trials=16,
    delta=list(DOSES), flip=-1., common_cutoffs_chunks=[1, 2, 4, 8],
    primary='gait', secondary=['xy', 'yaw'], recovery_error='endpoint weighted squared distance',
    error_reduction_gte=.20, paired_bootstrap_ci=.95, benefit_ci_lower_gt=0.,
    gamma_ci_upper_lt=0., fall_rate_increase_max=0., fall_torso_z_lt=FALL_LINE_P1,
    fall_line_source='experiments/walk_verify/results/ruler_pack.json:torso_z.fall_line_p1',
    primary_codec=PRIMARY_CODEC, primary_dose='1', primary_cutoff_chunks=4,
    primary_space='gait', primary_verdict_rule='N and M criteria met, and gait Gamma paired CI upper < 0',
    codec_roles={PRIMARY_CODEC:'B1 primary codec', 'sigma0':'mechanism comparison',
                 'sigma1.0':'mechanism comparison'},
    subspace_measurement=SUBSPACE_NOTE,
    gait_primary_reason=('C1 判決碼-效果通道在 gait 子空間驗活（碼層事實）；本件 dict_v1 decoder 無 ŝ 頭，'
                         'C1 的 ŝ-頭病理由不適用於此處，特此記明。'),
    n_calibration=40, n_evaluation=160, bootstrap_replicates=10000,
    zero_state_tolerance=1e-10, mutant_collapse_ratio_max=1e-3,
    mismatch_state_linf_min=1e-6, W='diagonal inverse nominal 4-step displacement variance',
    W_variance_floor=1e-8, inference='each dose and cutoff separately; no best-cell selection',
    innovation_matching='one scalar, whole 8-chunk tangent-sequence L2 norm',
    mechanism_pulse='sigma0 delta1 pulse projected onto BOTH codecs action headroom',
    action_limits=[-1.,1.], mechanism_clipping='identical realized additive pulse, no post-hoc clipping',
    old_new={'workorder': {'reduction': .20, 'CI': .95, 'gamma_ci_upper': 0.,
                           'fall_torso_z_lt': .2, 'verdict_composition': '原版未定義合成規則'},
             'implemented': {'reduction': .20, 'CI': .95, 'gamma_ci_upper': 0.,
                             'fall_torso_z_lt': FALL_LINE_P1,
                             'verdict_composition': 'p0_dict_v1 × δ=1 × 4-chunk × gait'},
             'reason': 'fall line 改採艦隊尺；原版未定義合成規則，現預註冊單格主判決'})
MECHANISM_CAVEAT = 'seed 與訓練噪音劑量綁定；兩 checkpoint 對比不得單獨歸因訓練噪音'


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def dump(path, stream):
    np.savez_compressed(path, **stream)


def state_obs(env):
    # Preserve f64 for geometry; only the decoder casts to f32, just as B1.
    return np.r_[env.unwrapped.data.qpos.copy(), env.unwrapped.data.qvel.copy()]


def innovation(model, reference, actual):
    """28D tangent: qpos displacement via SO(3) log, followed by velocity delta."""
    if np.array_equal(reference, actual):
        return np.zeros(2*model.nv)
    dq = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1., reference[:15], actual[:15])
    return np.r_[dq, actual[15:] - reference[15:]]


def inject_innovation(model, reference, delta):
    if not np.any(delta):
        return reference.copy()
    q = reference[:15].copy()
    mujoco.mj_integratePos(model, q, delta[:model.nv], 1.)
    return np.r_[q, reference[15:] + delta[model.nv:]]


def mismatched_sequence(target, donor):
    """One scalar per complete donor sequence preserves its event/time structure."""
    target_norm, donor_norm = np.linalg.norm(target), np.linalg.norm(donor)
    if target_norm == 0:
        return np.zeros_like(target), 0.
    if donor_norm <= 1e-14:
        raise ValueError('Nonzero target innovation has a zero donor; cannot amplitude-match')
    scale = target_norm / donor_norm
    return donor * scale, float(scale)


def advance_leg(leg, env):
    step = leg['global_step']
    if leg['first_hit'] < 0 and np.linalg.norm(env.unwrapped.data.qpos[:2] - leg['target_xy']) <= b1.RHO:
        leg['first_hit'] = step + 1
        leg['active_leg'] = 2
        leg['leg_reach_step']['1'] = step
    leg['global_step'] += 1


def simulate(env, st, codes, codec, *, arm='R', reference=None, wrong=None,
             actions=None, capture_step=None):
    """Single physics path for nominal, replay, R/N/M and pulse calibration."""
    leg = restore_state(env, st)
    n = len(actions) if actions is not None else len(codes)*CHUNK
    records = {k: [] for k in ('qpos', 'qvel', 'obs', 'shape', 'leg_bookkeeping')}
    acts, inputs, indices, integrations = [], [], [], []
    captured = None
    for t in range(n + 1):
        o = state_obs(env)
        for key, val in [('qpos', o[:15]), ('qvel', o[15:]), ('obs', o),
                         ('shape', b1.body_shape(env.unwrapped)),
                         ('leg_bookkeeping', [leg['active_leg'], leg['global_step'], leg['first_hit']])]:
            records[key].append(np.asarray(val).copy())
        integration = np.empty(mujoco.mj_stateSize(env.unwrapped.model, STATE_SPEC))
        mujoco.mj_getState(env.unwrapped.model, env.unwrapped.data, integration, STATE_SPEC)
        integrations.append(integration)
        if t == capture_step:
            captured = save_state(env, leg)
        if t == n:
            break
        if actions is None and t % CHUNK == 0:
            if arm == 'R':
                seen = o
            elif arm == 'N':
                seen = reference['obs'][t]
            elif arm == 'M':
                seen = inject_innovation(env.unwrapped.model, reference['obs'][t], wrong[t//CHUNK])
            else:
                raise ValueError(arm)
            inputs.append(seen.copy())
            indices.append(leg['global_step'])
            chunk = b1.decode(codec, codes[t//CHUNK], seen,
                              env.action_space.low, env.action_space.high)
        a = np.asarray(actions[t] if actions is not None else chunk[t % CHUNK], float)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite decoder action')
        env.step(a)
        advance_leg(leg, env)
        acts.append(a.copy())
    stream = {k: np.asarray(v) for k, v in records.items()}
    if not all(np.isfinite(v).all() for v in stream.values()):
        raise ValueError('Nonfinite physical stream')
    stream.update(action=np.asarray(acts), integration=np.asarray(integrations),
                  decoder_obs=np.asarray(inputs).reshape(-1, 29),
                  decoder_ref_step=np.asarray(indices, dtype=int),
                  decoder_obs_source=np.asarray(arm if actions is None else 'fixed_action'),
                  codes=np.asarray(codes), first_hit=np.asarray(leg['first_hit']))
    return stream, captured, save_state(env, leg)


def trim_reference(full):
    return {k: full[k][PREFIX:] for k in ('qpos', 'qvel', 'obs', 'shape', 'integration')}


def positive_check(r, n, reference):
    error = max(float(np.max(np.abs(r[k]-n[k]))) for k in ('qpos', 'qvel', 'integration'))
    replay = max(float(np.max(np.abs(r[k]-reference[k]))) for k in ('qpos', 'qvel'))
    if error > PREREGISTERED['zero_state_tolerance'] or replay > PREREGISTERED['zero_state_tolerance']:
        raise AssertionError(f'delta=0 wiring failed: R/N={error}, nominal={replay}')
    return dict(R_N_state_linf=error, nominal_replay_state_linf=replay)


def assert_mismatch_visible(r, m):
    gap = max(float(np.max(np.abs(r[k]-m[k]))) for k in ('qpos', 'qvel'))
    if gap <= PREREGISTERED['mismatch_state_linf_min']:
        raise AssertionError(f'M mutant caught: R/M state gap collapsed ({gap:g})')
    return gap


def calibrate(env, st, codes, codec, nominal, rng, directory):
    """Action-space pulse, measured with rotation tangent displacement, never q+quat."""
    q0, q4 = nominal['obs'][0], nominal['obs'][4]
    dq = innovation(env.unwrapped.model, q0, q4)[:env.unwrapped.model.nv]
    nominal_rms = float(np.sqrt(np.mean(dq**2)))
    target = .5 * nominal_rms
    if target <= 1e-12:
        raise ValueError('Zero nominal four-step displacement; pulse calibration undefined')
    count = 0
    trace = []
    # Direction is drawn in advance from task seed, independent of response outcomes.
    directions = rng.normal(size=(PREREGISTERED['calibration_max_trials'], 8))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    def trial(direction, amplitude):
        nonlocal count
        tape = nominal['action'][:PREFIX].copy()
        tape[:PULSE_STEPS] = np.clip(tape[:PULSE_STEPS] + amplitude*direction, -1., 1.)
        flow, _, _ = simulate(env, st, codes[:1], codec, actions=tape)
        achieved = float(np.sqrt(np.mean(innovation(env.unwrapped.model, q4, flow['obs'][-1])[:14]**2)))
        dump(directory/f'calibration_{count:04d}.npz', flow)
        trace.append(dict(trial=count, amplitude=amplitude, achieved=achieved))
        count += 1
        return achieved
    for di, direction in enumerate(directions):
        lower, upper = 0., .25
        value = trial(direction, upper)
        while value < target and upper < 128.:
            lower, upper = upper, upper*2
            value = trial(direction, upper)
        if value < target:
            continue
        for _ in range(32):
            amplitude = (lower+upper)/2
            value = trial(direction, amplitude)
            if abs(value/target-1) <= PREREGISTERED['calibration_relative_tolerance']:
                return direction*amplitude, dict(nominal_4step_rms=nominal_rms, target=target,
                    achieved=value, relative_error=abs(value/target-1), direction_index=di,
                    pulse=direction*amplitude, n_steps=count*PREFIX, trials=trace)
            if value < target:
                lower = amplitude
            else:
                upper = amplitude
    raise RuntimeError('Cannot calibrate bounded two-step pulse to 50%; no silent fallback')


def features(stream, reference=None):
    yaw = np.unwrap(b1.torso_yaw(stream['qpos']))[:, None]
    if reference is not None:
        anchor = b1.torso_yaw(reference['qpos'][:1])[0]
        yaw -= 2*np.pi*np.round((yaw[0,0]-anchor)/(2*np.pi))
    return dict(gait=stream['shape'].reshape(len(stream['shape']), -1),
                xy=stream['qpos'][:, :2], yaw=yaw)


def fit_weights(nominals, calibration_ids):
    """Only nominal calibration trajectories enter W, never perturbed/test outcomes."""
    weights = {}
    for space in SPACES:
        rows = []
        for name in nominals:
            for ti in calibration_ids:
                y = features(nominals[name][ti])[space]
                rows.extend(y[CHUNK:] - y[:-CHUNK])
        w = 1. / np.maximum(np.var(rows, axis=0), PREREGISTERED['W_variance_floor'])
        w /= len(w)
        w.setflags(write=False)
        weights[space] = w
    return weights


def gamma_ledger(reference, n, r, weights, cutoff):
    ref_y, ny, ry = features(reference), features(n,reference), features(r,reference)
    ledger = {}
    unchanged = np.array_equal(n['action'][:cutoff], r['action'][:cutoff])
    for space in SPACES:
        residual = ny[space][cutoff] - ref_y[space][cutoff]
        delta = ry[space][cutoff] - ny[space][cutoff]
        w = weights[space]
        cross = float(2*np.dot(residual*w, delta))
        square = float(np.dot(delta*w, delta))
        gamma = cross + square
        kind = ('no_action_change' if unchanged else 'outward' if cross >= 0 else
                'correction_overwhelmed' if gamma >= 0 else 'true_correction')
        ledger[space] = dict(r=residual, d=delta, cross=cross, d_squared_W=square, gamma=gamma, kind=kind)
    return ledger


def readout(stream, reference, weights, cutoff, goal, initial_xy):
    sf, rf = features(stream,reference), features(reference)
    result = {s: float(np.dot((sf[s][cutoff]-rf[s][cutoff])**2, weights[s])) for s in SPACES}
    result.update(fell=bool(np.any(stream['qpos'][:cutoff+1, 2] < PREREGISTERED['fall_torso_z_lt'])),
                  goal_progress=float(np.linalg.norm(initial_xy-goal)-np.linalg.norm(stream['qpos'][cutoff, :2]-goal)))
    return result


def bootstrap(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED+100)
    samples = values[rng.integers(len(values), size=(PREREGISTERED['bootstrap_replicates'], len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                n_pairs=len(values), paired=True)


def pulse_readout(env, nominal, prefix, requested, realized, target):
    displacement = innovation(env.unwrapped.model, nominal['obs'][PREFIX], prefix['obs'][-1])[:env.unwrapped.model.nv]
    rms = float(np.sqrt(np.mean(displacement**2)))
    proposed = nominal['action'][:PULSE_STEPS] + requested
    clipped = np.abs(proposed-realized-nominal['action'][:PULSE_STEPS]) > 1e-12
    return dict(realized_displacement_rms=rms, realized_displacement_ratio=rms/target,
                clip_fraction=float(np.mean(clipped)), any_clip=bool(np.any(clipped)))


def aggregate(rows, evaluation_ids, scientific):
    result = []
    selected = [r for r in rows if r['task_id'] in evaluation_ids]
    for codec in sorted({r['codec'] for r in selected}):
        for dose, _ in CONDITIONS:
            for chunks in PREREGISTERED['common_cutoffs_chunks']:
                cell = [r for r in selected if r['codec']==codec and r['dose']==dose and r['chunks']==chunks]
                entry = dict(codec=codec, dose=dose, chunks=chunks, spaces={},
                    verdict_role='primary' if (codec==PREREGISTERED['primary_codec'] and
                        dose==PREREGISTERED['primary_dose'] and
                        chunks==PREREGISTERED['primary_cutoff_chunks']) else 'secondary')
                if 'pulse' in cell[0]:
                    entry['realized_displacement_ratio'] = bootstrap(
                        [r['pulse']['realized_displacement_ratio'] for r in cell])
                    entry['realized_displacement_ratio_median'] = float(np.median(
                        [r['pulse']['realized_displacement_ratio'] for r in cell]))
                    entry['clip_fraction'] = float(np.mean([r['pulse']['clip_fraction'] for r in cell]))
                    entry['any_clip_fraction'] = float(np.mean([r['pulse']['any_clip'] for r in cell]))
                for space in SPACES:
                    comparisons = {}
                    for control in ('N', 'M'):
                        rc = np.array([r['arms'][control][space] for r in cell])
                        rr = np.array([r['arms']['R'][space] for r in cell])
                        benefit = bootstrap(rc-rr)
                        reduction = float(1-rr.mean()/rc.mean()) if rc.mean()>0 else None
                        fall_delta = float(np.mean([r['arms']['R']['fell']-int(r['arms'][control]['fell']) for r in cell]))
                        comparisons[control] = dict(benefit=benefit, relative_reduction=reduction,
                            fall_rate_delta=fall_delta,
                            criterion_met=(reduction is not None and reduction>=PREREGISTERED['error_reduction_gte']
                                and benefit['ci95'][0]>PREREGISTERED['benefit_ci_lower_gt']
                                and fall_delta<=PREREGISTERED['fall_rate_increase_max']))
                    gamma = bootstrap([r['gamma'][space]['gamma'] for r in cell])
                    kinds = ('no_action_change', 'outward', 'correction_overwhelmed', 'true_correction')
                    counts = {kind:sum(r['gamma'][space].get('kind')==kind for r in cell) for kind in kinds}
                    cross = (bootstrap([r['gamma'][space]['cross'] for r in cell])
                             if 'cross' in cell[0]['gamma'][space] else None)
                    d_squared = (bootstrap([r['gamma'][space]['d_squared_W'] for r in cell])
                                 if 'd_squared_W' in cell[0]['gamma'][space] else None)
                    is_primary = entry['verdict_role']=='primary' and space==PREREGISTERED['primary_space']
                    entry['spaces'][space] = dict(comparisons=comparisons, gamma=gamma,
                        kind_counts=counts, cross=cross, d_squared_W=d_squared,
                        feedback_corrects=(gamma['ci95'][1]<PREREGISTERED['gamma_ci_upper_lt'])
                            if scientific and dose!='0' and space=='gait' else None,
                        primary_verdict=(all(c['criterion_met'] for c in comparisons.values()) and
                            gamma['ci95'][1]<PREREGISTERED['gamma_ci_upper_lt'])
                            if scientific and is_primary else None,
                        role='primary' if is_primary else 'secondary')
                entry['arms'] = {a: dict(fall_rate=float(np.mean([r['arms'][a]['fell'] for r in cell])),
                    goal_progress=float(np.mean([r['arms'][a]['goal_progress'] for r in cell]))) for a in ('R','N','M')}
                result.append(entry)
    return result


def plot_dose_response(entries, path):
    """Gait N/M error reduction against measured displacement, by codec and window."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(CODEC_LABELS), 4, figsize=(16, 9), sharey=True)
    colors = {'N':'tab:blue', 'M':'tab:orange'}
    for i, codec in enumerate(CODEC_LABELS):
        for j, chunks in enumerate(PREREGISTERED['common_cutoffs_chunks']):
            ax = axes[i,j]
            cell = [r for r in entries if r['codec']==codec and r['chunks']==chunks]
            regular = sorted((r for r in cell if r['dose']!='flip'),
                             key=lambda r:r['realized_displacement_ratio_median'])
            flipped = next(r for r in cell if r['dose']=='flip')
            for control in ('N','M'):
                y = [r['spaces']['gait']['comparisons'][control]['relative_reduction'] for r in regular]
                x = [r['realized_displacement_ratio_median'] for r in regular]
                ax.plot(x,y,'o-',color=colors[control],label=control)
                fy = flipped['spaces']['gait']['comparisons'][control]['relative_reduction']
                ax.plot(flipped['realized_displacement_ratio_median'],fy,'x',color=colors[control])
            ax.axhline(PREREGISTERED['error_reduction_gte'],color='gray',linestyle=':')
            ax.set_title(f'{codec}, {chunks} chunk')
            ax.grid(alpha=.2)
            if j==0:
                ax.set_ylabel('gait relative error reduction')
            if i==len(CODEC_LABELS)-1:
                ax.set_xlabel('realized displacement / calibration target')
    axes[0,0].legend(title='control; x=flip')
    fig.suptitle('C2 dose response (evaluation tasks; smoke has no scientific verdict)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_fixture(directory, n_tasks):
    """Real Ant physics, synthetic starts/tapes, persisted with the B1 source schema."""
    directory.mkdir()
    env = wv.make_env('/tmp', relay.DATASET)
    try:
        codec, _ = wv.load_dict_v1(relay.DEFAULT_CKPT)
        for ti in range(n_tasks):
            env.reset(seed=SEED+ti)
            env.action_space.seed(SEED+ti)
            env.observation_space.seed(SEED+ti)
            leg = dict(active_leg=1, global_step=0, first_hit=-1, target_xy=[100., 100.],
                       leg_active_step={'1': 0}, leg_reach_step={}, leg_start_xy={'1': [0., 0.]})
            st = save_state(env, leg)
            tape = np.random.default_rng(SEED+ti).uniform(-1, 1, (200, 8)).astype(np.float32)
            codes = b1.encode_tape(codec, tape)
            # Supply the real B1 L schema. This fixture is not official dataset evidence.
            flow, _ = b1.rollout(env, st, model=codec, codes=codes)
            td = directory/f'task_{ti:03d}_ep{ti}'
            td.mkdir()
            save_snapshot(td/'snapshot.npz', st)
            dump(td/'L.npz', dict(**flow, codes=codes))
            write_json(td/'task.json', dict(episode=ti, s0=ti*201, e0=ti*201+200,
                T=201, n_chunks=50, M=1, wp_xy=[[100.,100.]]))
    finally:
        env.close()


def source_inputs(args):
    if args.n_tasks == 200 and (not args.main or args.synthetic):
        raise ValueError('200 tasks require explicit --main and official B1 source')
    if args.main and args.n_tasks != 200:
        raise ValueError('--main requires exactly 200 tasks')
    if args.synthetic and args.n_tasks > 4:
        raise ValueError('Synthetic fixture limited to four tasks')
    out = args.out_dir.resolve()
    if not out.is_relative_to(FIRSTCUTS):
        raise ValueError('Outputs must remain under experiments/firstcuts/')
    if out.exists():
        raise FileExistsError('Use a fresh --out-dir; never overwrite an experiment')
    if not args.synthetic and socket.gethostname().split('.')[0] != 'zeldajr':
        raise ValueError('Official C2 runs require zeldajr')
    data, fs = (Path('/tmp'), 'synthetic_no_dataset') if args.synthetic else local_data(args.data_dir)
    out.mkdir(parents=True)
    if args.synthetic:
        args.source_dir = out/'synthetic_b1_source'
        make_fixture(args.source_dir, args.n_tasks)
    directories = sorted(args.source_dir.glob('task_*'))[:args.n_tasks]
    if len(directories) != args.n_tasks:
        raise ValueError('B1 source does not yet contain requested task count')
    source = []
    for ti, td in enumerate(directories):
        st, flow = load_snapshot(td/'snapshot.npz'), npz(td/'L.npz')
        task = json.loads((td/'task.json').read_text())
        if td.name != f'task_{ti:03d}_ep{task["episode"]}':
            raise ValueError('B1 source task indices must be contiguous and in original order')
        if st['leg']['active_leg']!=1 or st['leg']['global_step']!=0:
            raise ValueError('Expected B1 leg-1 initial snapshot')
        if flow['codes'].shape!=(50,) or flow['qpos'].shape!=(201,15):
            raise ValueError('Expected B1 L4K32, 200-step full stream schema')
        if not np.array_equal(st['qpos'], flow['qpos'][0]):
            raise ValueError('B1 snapshot/L mismatch')
        source.append(dict(st=st, codes=flow['codes'], task=task, directory=td))
    codecs, configs = {}, {}
    codec_paths = [(PRIMARY_CODEC, args.ckpt_p0), ('sigma0', args.ckpt_sigma0),
                   ('sigma1.0', args.ckpt_sigma1)]
    for label, path in codec_paths:
        codecs[label], configs[label] = wv.load_dict_v1(path)
        if configs[label]['seg_len']!=4 or configs[label]['k']!=32:
            raise ValueError('C2 requires L4K32')
    # Reused B1 code IDs are valid only with the frozen B1 encoder and codebook.
    base = codecs[PRIMARY_CODEC]
    p0_sha = hashlib.sha256(args.ckpt_p0.read_bytes()).hexdigest()
    if p0_sha != 'b6df04119899942abb7d9d073bdf6a65915eaf38d1dfeb6459309300b25bae53':
        raise ValueError('B1 primary codec checkpoint SHA256 mismatch')
    for model in codecs.values():
        for key, value in base.state_dict().items():
            if key.startswith(('encoder.', 'vq.')) and not torch.equal(value, model.state_dict()[key]):
                raise ValueError('Checkpoint codebook/encoder differs from B1; cannot reuse codes')
    provenance = dict(host=socket.gethostname(), filesystem=fs, data_dir=str(data),
        source_dir=str(args.source_dir.resolve()), synthetic=args.synthetic, seed=SEED,
        mujoco=mujoco.__version__, numpy=np.__version__, torch=torch.__version__,
        configs=configs, mechanism_caveat=MECHANISM_CAVEAT,
        checkpoint_sha256={label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for label,path in codec_paths},
        source_sha256={str(s['directory']/name): hashlib.sha256((s['directory']/name).read_bytes()).hexdigest()
                       for s in source for name in ('snapshot.npz','L.npz','task.json')})
    return source, codecs, wv.make_env(str(data), relay.DATASET), provenance


def run_experiment(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    source, codecs, env, provenance = source_inputs(args)
    out = args.out_dir
    n = len(source)
    split = 40 if n==200 else max(1,n//2)
    calibration_ids, evaluation_ids = list(range(split)), list(range(split,n))
    donors = {}
    # Main donors stay within calibration/evaluation partitions to avoid leakage.
    # Tiny two-task smoke necessarily crosses partitions, and cannot yield inference.
    groups = [calibration_ids,evaluation_ids] if min(split,n-split)>=2 else [list(range(n))]
    for group in groups:
        donors.update({ti:group[(j+1)%len(group)] for j,ti in enumerate(group)})
    summary = dict(status='running', n_tasks=n, n_arms=3, n_doses=3, n_auxiliary_doses=1,
        n_codecs=len(codecs), n_steps_total=0, n_steps_fixture=n*200 if args.synthetic else 0,
        preregistered=PREREGISTERED, provenance=provenance,
        subspace_measurement=SUBSPACE_NOTE,
        calibration_ids=calibration_ids, evaluation_ids=evaluation_ids, donor_map=donors,
        inferential_scope='main' if n==200 else 'smoke_only_no_scientific_verdict',
        mechanism_caveat=MECHANISM_CAVEAT)
    write_json(out/'preregistered.json', PREREGISTERED)
    write_json(out/'summary.json', summary)
    nominals, starts, arms, rows, zero, mutants, pulse_metrics = {}, {}, {}, [], [], [], {}
    def run(*a, **kw):
        flow, mid, end = simulate(env,*a,**kw)
        summary['n_steps_total'] += len(flow['action'])
        return flow, mid, end
    try:
        for name, codec in codecs.items():
            nominals[name] = []
            for ti, src in enumerate(source):
                td = out/name/f'task_{ti:03d}'
                td.mkdir(parents=True)
                flow, mid, _ = run(src['st'], src['codes'][:9], codec, capture_step=PREFIX)
                replay, _, _ = run(src['st'], src['codes'][:9], codec)
                b1.assert_identical(flow, replay)
                dump(td/'nominal.npz',flow)
                dump(td/'nominal_replay.npz',replay)
                save_snapshot(td/'nominal_boundary.npz',mid)
                nominals[name].append(flow)
        weights = fit_weights(nominals,calibration_ids)
        dump(out/'frozen_W.npz',dict(**weights, calibration_ids=calibration_ids, evaluation_ids=evaluation_ids))
        w_hash = hashlib.sha256((out/'frozen_W.npz').read_bytes()).hexdigest()
        summary['W'] = dict(status='frozen',sha256=w_hash,n_fit_tasks=split,
            official_40_160=(n==200), calibration_ids=calibration_ids)
        pulses, calibrations, targets = {}, [], {}
        for name, codec in codecs.items():
            for ti, src in enumerate(source):
                td = out/name/f'task_{ti:03d}'/'calibration'
                td.mkdir()
                pulse, cal = calibrate(env,src['st'],src['codes'],codec,nominals[name][ti],
                    np.random.default_rng(np.random.SeedSequence([SEED,ti])),td)
                pulses[name,ti] = pulse
                targets[name,ti] = cal['target']
                cal.update(codec=name,task_id=ti)
                calibrations.append(cal)
                summary['n_steps_total'] += cal['n_steps']
                write_json(td/'calibration.json',cal)
        for name, codec in codecs.items():
            for ti, src in enumerate(source):
                ref = trim_reference(nominals[name][ti])
                for dose, multiplier in CONDITIONS:
                    td = out/name/f'task_{ti:03d}'/f'dose_{dose}'
                    td.mkdir()
                    tape = nominals[name][ti]['action'][:PREFIX].copy()
                    requested = multiplier*pulses[name,ti]
                    tape[:2] = np.clip(tape[:2]+requested,-1.,1.)
                    prefix, _, post = run(src['st'],src['codes'][:1],codec,actions=tape)
                    realized = tape[:2]-nominals[name][ti]['action'][:2]
                    pulse_metrics[name,ti,dose] = pulse_readout(env,nominals[name][ti],prefix,
                        requested,realized,targets[name,ti])
                    dump(td/'pulse.npz',dict(**prefix,requested_pulse=requested,
                        realized_pulse=realized))
                    save_snapshot(td/'post_pulse_snapshot.npz',post)
                    starts[name,ti,dose] = post
                    for arm in ('R','N'):
                        flow,_,_ = run(post,src['codes'][1:9],codec,arm=arm,reference=ref)
                        dump(td/f'{arm}.npz',flow)
                        arms[name,ti,dose,arm]=flow
                    if dose=='0':
                        zero.append(dict(codec=name,task_id=ti,**positive_check(
                            arms[name,ti,dose,'R'],arms[name,ti,dose,'N'],ref)))
        for name, codec in codecs.items():
            for ti, src in enumerate(source):
                ref = trim_reference(nominals[name][ti])
                donor_id = donors[ti]
                donor_ref = trim_reference(nominals[name][donor_id])
                for dose,_ in CONDITIONS:
                    td = out/name/f'task_{ti:03d}'/f'dose_{dose}'
                    r,narm = (arms[name,ti,dose,a] for a in ('R','N'))
                    donor = arms[name,donor_id,dose,'R']
                    target_innov = np.array([innovation(env.unwrapped.model,a,b) for a,b in
                        zip(ref['obs'][:-1:4],r['obs'][:-1:4])])
                    donor_innov = np.array([innovation(env.unwrapped.model,a,b) for a,b in
                        zip(donor_ref['obs'][:-1:4],donor['obs'][:-1:4])])
                    wrong, scale = mismatched_sequence(target_innov,donor_innov)
                    m,_,_ = run(starts[name,ti,dose],src['codes'][1:9],codec,arm='M',reference=ref,wrong=wrong)
                    m.update(innovation=wrong,target_innovation=target_innov,donor_innovation=donor_innov,
                             donor_task=np.asarray(donor_id),donor_scale=np.asarray(scale))
                    dump(td/'M.npz',m)
                    arms[name,ti,dose,'M']=m
                    # One mutant per nonzero full-dose cell, through identical simulator/decoder.
                    if dose=='1' and not args.main:
                        mutant,_,_ = run(starts[name,ti,dose],src['codes'][1:9],codec,
                            arm='M',reference=ref,wrong=target_innov)
                        dump(td/'M_correct_innovation_mutant.npz',mutant)
                        good_gap = assert_mismatch_visible(r,m)
                        mutant_gap = max(float(np.max(np.abs(r[k]-mutant[k]))) for k in ('qpos','qvel'))
                        try:
                            assert_mismatch_visible(r,mutant)
                        except AssertionError:
                            caught = True
                        else:
                            raise AssertionError('Negative-control test failed to catch M mutant')
                        if mutant_gap/good_gap > PREREGISTERED['mutant_collapse_ratio_max']:
                            raise AssertionError('M mutant difference did not collapse')
                        mutants.append(dict(codec=name,task_id=ti,normal_gap=good_gap,
                            mutant_gap=mutant_gap,collapse_ratio=mutant_gap/good_gap,caught=caught))
                    for chunks in PREREGISTERED['common_cutoffs_chunks']:
                        cutoff=chunks*4
                        row = dict(codec=name,task_id=ti,dose=dose,chunks=chunks,
                            pulse=pulse_metrics[name,ti,dose],
                            arms={a:readout(arms[name,ti,dose,a],ref,weights,cutoff,
                                np.asarray(src['st']['leg']['target_xy']),src['st']['qpos'][:2]) for a in ('R','N','M')},
                            gamma=gamma_ledger(ref,narm,r,weights,cutoff),
                            action_response_rms=float(np.sqrt(np.mean((r['action'][:cutoff]-narm['action'][:cutoff])**2))))
                        # Include the physical prefix in fall history for every arm.
                        prefix_fell = bool(np.any(npz(td/'pulse.npz')['qpos'][:,2]<PREREGISTERED['fall_torso_z_lt']))
                        for a in row['arms'].values():
                            a['fell'] = a['fell'] or prefix_fell
                        rows.append(row)
        comparison = []
        # Primary doses calibrate against EACH codec's own nominal displacement.
        # For mechanism comparison enforce identical REALIZED additive action pulses.
        # Project sigma0's calibrated proposal onto the intersection of actuator headroom.
        for ti, src in enumerate(source):
            baselines = np.array([nominals[name][ti]['action'][:2] for name in ('sigma0','sigma1.0')])
            shared = np.clip(pulses['sigma0',ti],np.max(-1.-baselines,axis=0),
                             np.min(1.-baselines,axis=0))
            if np.linalg.norm(shared)<=1e-12:
                raise ValueError('No common actuator headroom for a mechanism pulse')
            mechanism = {}
            for name in ('sigma0','sigma1.0'):
                codec = codecs[name]
                td = out/name/f'task_{ti:03d}'/'mechanism_shared_pulse'
                td.mkdir()
                ref = trim_reference(nominals[name][ti])
                tape = nominals[name][ti]['action'][:PREFIX].copy()
                tape[:2] += shared
                if np.max(np.abs(tape))>1.+1e-14:
                    raise AssertionError('Shared pulse exceeds actuator range')
                prefix,_,post = run(src['st'],src['codes'][:1],codec,actions=tape)
                dump(td/'pulse.npz',dict(**prefix,requested_pulse=shared,
                    unprojected_proposal=pulses['sigma0',ti],
                    realized_pulse=tape[:2]-nominals[name][ti]['action'][:2]))
                save_snapshot(td/'post_pulse_snapshot.npz',post)
                pair = {}
                for a in ('R','N'):
                    pair[a],_,_ = run(post,src['codes'][1:9],codec,arm=a,reference=ref)
                    dump(td/f'{a}.npz',pair[a])
                mechanism[name] = (ref,pair)
            for chunks in PREREGISTERED['common_cutoffs_chunks']:
                entry = dict(task_id=ti,chunks=chunks,pulse_source='sigma0_delta1_common_headroom',
                    split='evaluation' if ti in evaluation_ids else 'calibration',
                    action_response_rms={},gamma={})
                for name,(ref,pair) in mechanism.items():
                    gamma = gamma_ledger(ref,pair['N'],pair['R'],weights,chunks*4)
                    entry['action_response_rms'][name] = float(np.sqrt(np.mean(
                        (pair['R']['action'][:chunks*4]-pair['N']['action'][:chunks*4])**2)))
                    entry['gamma'][name] = {s:gamma[s]['gamma'] for s in SPACES}
                comparison.append(entry)
        if hashlib.sha256((out/'frozen_W.npz').read_bytes()).hexdigest()!=w_hash:
            raise AssertionError('Frozen W changed')
        mechanism_aggregate = []
        for chunks in PREREGISTERED['common_cutoffs_chunks']:
            cell = [r for r in comparison if r['chunks']==chunks and r['split']=='evaluation']
            mechanism_aggregate.append(dict(chunks=chunks, difference='sigma1.0 minus sigma0',
                action_response=bootstrap([r['action_response_rms']['sigma1.0']-r['action_response_rms']['sigma0'] for r in cell]),
                gamma={s:bootstrap([r['gamma']['sigma1.0'][s]-r['gamma']['sigma0'][s] for r in cell]) for s in SPACES},
                causal_attribution_to_training_noise=False))
        aggregated = aggregate(rows,evaluation_ids,n==200)
        primary_cell = next(r for r in aggregated if r['verdict_role']=='primary')
        plot_dose_response(aggregated,out/'dose_response_gait.svg')
        summary.update(status='complete',zero_checks=zero,mutant_checks=mutants,calibrations=calibrations,
            per_task=rows,aggregate=aggregated,
            primary_verdict=primary_cell['spaces']['gait']['primary_verdict'],
            primary_verdict_cell=dict(codec=PRIMARY_CODEC,dose='1',chunks=4,space='gait'),
            dose_response_plot='dose_response_gait.svg',mechanism_pairs=comparison,
            mechanism_aggregate=mechanism_aggregate, n_mechanism_extra_arms=2,
            n_steps_including_fixture=summary['n_steps_total']+summary['n_steps_fixture'])
        write_json(out/'summary.json',summary)
        return summary
    except Exception as exc:
        summary.update(status='failed',error=str(exc))
        write_json(out/'summary.json',summary)
        raise
    finally:
        env.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=FIRSTCUTS/'b1_triarm/results_main')
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--data-dir',default=os.environ.get('OGBENCH_DATA_DIR','/home/cymaxwelllee/.ogbench/data'))
    p.add_argument('--n-tasks',type=int,choices=[2,3,4,200],default=4)
    p.add_argument('--main',action='store_true',help='Future explicit main-run authorization only')
    p.add_argument('--synthetic',action='store_true')
    ckpt=FIRSTCUTS.parent/'robust_decoder/results'
    p.add_argument('--ckpt-p0',type=Path,default=Path(relay.DEFAULT_CKPT))
    p.add_argument('--ckpt-sigma0',type=Path,default=ckpt/'ckpt_sigma0.pt')
    p.add_argument('--ckpt-sigma1',type=Path,default=ckpt/'ckpt_sigma1.0.pt')
    return p


def main():
    args=parser().parse_args()
    result=run_experiment(args)
    label='SYNTHETIC' if args.synthetic else 'MAIN' if args.main else 'OFFICIAL_SMOKE'
    print(f'C2 {label} COMPLETE n_tasks={result["n_tasks"]} n_arms=3 n_doses=3 '
          f'n_auxiliary_doses=1 n_codecs={result["n_codecs"]} n_steps_total={result["n_steps_total"]}')
    print('C2 scientific_verdict='+result['inferential_scope'])


if __name__=='__main__':
    main()
