#!/usr/bin/env python3
"""Checkpoint rollout evaluation; C calls PB Q-C directly, SG splits conditions."""
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
import random
import traceback
from pathlib import Path
import numpy as np
from eval_common import load_ckpt, PB, smooth_anchor
from common import ENV, H, PIN, Blocked, digest_array, reset_env, verify_source, file_sha
from runtime import condition
from harness_detour import (decode, seeds_detour, make_collector_detour,
                            run_draw_detour, PROTOCOL)
import common
import runtime
import harness_move3


def assert_seed_bindings():
    assert runtime.seeds is common.seeds, 'runtime.seeds binding changed'
    assert harness_move3.seeds is common.seeds, 'harness_move3.seeds binding changed'


assert_seed_bindings()


def make_plan(question, arm, draw):
    return dict(question, arm=arm, draw=draw,
                seeds=seeds_detour(question['task'], question['episode'], draw))


TRAP = Path('/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/readout-2026-10-02/harvest-m3.json')
TRAP_PIN = '46de59ed8d46f7990e514c4c7fd9e52a4e1c0ecc50c9c31cf07504678fe421ce'
BUILDER = PB / 'builder.json'
BUILDER_PIN = '66a6957d7958729ba2775418cbcf2b86e5cec9571098db550d028d4760799846'


def tensor_sha(value):
    return digest_array(value.detach().cpu().numpy())


class StepLimitEnv:
    """Short CPU budget at env termination boundary; preserves all actual steps."""
    def __init__(self, env, max_steps):
        if type(max_steps) is not int or not 0 < max_steps <= H:
            raise ValueError('max_steps must be an integer in 1..1000')
        self.env, self.max_steps = env, max_steps
        self.steps = 0
        self.obs = self.goal = None
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.obs, info = self.env.reset(**kwargs)
        self.goal = np.asarray(info['goal']).copy()
        self.steps, self.actions = 0, []
        return self.obs, info

    def step(self, action):
        if self.steps >= self.max_steps:
            raise ValueError('step beyond requested budget')
        self.actions.append(np.asarray(action).copy().tolist())
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.max_steps < H and self.steps == self.max_steps:
            truncated = True
        return self.obs, reward, terminated, truncated, info


def cell(env, xy):
    return [int(v) for v in env.unwrapped.xy_to_ij(np.asarray(xy))]


class WiringAudit:
    """Observe actual sample_plan/flow.sample entries and the real ahead input."""
    def __init__(self, module, env, arm):
        self.module, self.env, self.arm = module, env, arm
        self.chunks = []
        self.expected_near = None
        self.sample_code = module.sample_plan.__code__
        self.flow_code = module.flow.sample.__func__.__code__

    def profile(self, frame, event, arg):
        assert_seed_bindings()
        if event == 'call' and frame.f_code is self.sample_code:
            cond = frame.f_locals['cond']
            self.chunks.append(dict(step=self.env.steps,
                current_xy=np.asarray(self.env.obs[:2]).tolist(),
                current_cell=cell(self.env, self.env.obs[:2]),
                w_xy=self.env.goal[:2].tolist() if self.arm == 'C' else None,
                w_cell=cell(self.env, self.env.goal[:2]) if self.arm == 'C' else None,
                sample_plan_calls=1, flow_calls=0,
                cond_far_sha256=tensor_sha(cond), cond_near_sha256=None,
                conditions_differ=False, head_calls=0, head_cond_sha256=None,
                head_consumed_near=None))
            if self.arm == 'C':
                self.expected_near = cond
        elif event == 'call' and frame.f_code is self.flow_code:
            if not self.chunks:
                raise AssertionError('flow.sample without sample_plan')
            self.chunks[-1]['flow_calls'] += 1
        if self.saved_profile is not None:
            self.saved_profile(frame, event, arg)

    def near(self, far, near, w):
        row = self.chunks[-1]
        self.expected_near = near
        row.update(w_xy=np.asarray(w).tolist(), w_cell=cell(self.env, w),
                   cond_near_sha256=tensor_sha(near),
                   conditions_differ=not self.module.torch.equal(far, near))
        assert row['cond_far_sha256'] == tensor_sha(far), 'flow must receive cond_far'

    def head(self, head, args):
        row = self.chunks[-1]
        row['head_calls'] += 1
        row['head_cond_sha256'] = tensor_sha(args[0])
        row['head_consumed_near'] = (args[0] is self.expected_near and
            row['head_cond_sha256'] == tensor_sha(self.expected_near))
        if not row['head_consumed_near']:
            raise AssertionError('SG wiring FAIL: action head did not consume cond_near')

    def __enter__(self):
        self.saved_profile = sys.getprofile()
        self.handle = self.module.ahead.register_forward_pre_hook(self.head)
        sys.setprofile(self.profile)
        return self

    def __exit__(self, kind, value, tb):
        sys.setprofile(self.saved_profile)
        self.handle.remove()
        assert_seed_bindings()

    def validate(self, require_distinct=False):
        assert self.chunks, 'no chunks observed'
        for row in self.chunks:
            assert row['sample_plan_calls'] == row['flow_calls'] == row['head_calls'] == 1, row
            assert row['w_xy'] is not None and row['w_cell'] is not None, row
            assert row['head_consumed_near'], row
            if require_distinct:
                assert row['conditions_differ'], row


def _run_draw_sg(module, env, plan, *, audit, move, waypoint=None, cap_steps=None, stuck_check=None, u=None):
    """Shared real env loop. No padding or reset after termination; fixed u bypasses flow."""
    from harness_move1 import FixedInjection, rng_snapshot
    from harness_move3 import TwoStage, noisy_action
    task, episode, draw, arm = (plan[k] for k in ('task', 'episode', 'draw', 'arm'))
    if move == 1 and ((arm == 'N') != (u is None)):
        raise ValueError('Move 1 injection/arm mismatch')
    if move == 3 and u is not None:
        raise ValueError('Move 3 must use fresh flow')
    obs, info = reset_env(env, task, episode)
    initial, goal = np.asarray(obs).copy(), np.asarray(info['goal']).copy()
    seed_record = seeds_detour(task, episode, draw)
    random.seed(seed_record['python_seed']); module.torch.manual_seed(seed_record['torch_seed'])
    module._GRAD_CACHE['u'] = None
    module._reseed_shuf(1000*task+episode); module._RDIR[0] = 1.
    collector = make_collector_detour(task, episode, draw, obs, goal) if u is None else None
    injected = None if u is None else FixedInjection(module.ahead, u)
    state = TwoStage('C', waypoint,
                     chunk_steps=module.CHUNK, cap_steps=cap_steps, initial_xy=initial[:2])
    before = rng_snapshot(module.torch, env, collector.noise if collector else None)
    successes, terminated, truncated, reason = [False], False, False, None
    while state.t < H and reason is None:
        target = waypoint if state.stage == 'w' else goal
        with module.torch.no_grad():
            cond_far = condition(module, obs, goal)
            state.sample(state.stage)
            sampled = collector.get_u(lambda: module.sample_plan(1, cond_far, None))
            raw = decode(module, sampled, np.asarray(obs[:2]))
            points = smooth_anchor(raw, obs[:2])
            w = points[-1].copy()
            cond_near = condition(module, obs, w)
            audit.near(cond_far, cond_near, w)
            actions = module.ahead(cond_near, module._q(sampled))[0].cpu().numpy()
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
        seeds=seeds_detour(task, episode, draw, injected is not None),
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


class DetourView:
    """Supply Q-C's unused oracle table without adding attributes to the model."""
    detour_e1_table = {}

    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)


def c_compat_row(raw, plan):
    """Serialize Q-C observations in the existing PB run_draw output contract.

    Q-C records environment xy as float64; the existing contract hashes float32.
    Keep Q-C's original trace hash alongside the unchanged float32 contract.
    No rollout or RNG operations occur here.
    """
    xy = np.asarray(raw['xy'], np.float32)
    fields = ('task', 'episode', 'draw', 'synthetic', 'source_sha256', 'seeds',
              'initial', 'goal', 'observation_dtype', 'goal_dtype', 'initial_sha256',
              'goal_sha256', 'observed_steps', 'termination_reason', 'terminated',
              'truncated', 'success', 'goal_success', 'chunk_steps', 'rng_before', 'rng_after')
    row = {key: raw[key] for key in fields}
    first_success = next((i for i, hit in enumerate(raw['goal_success']) if hit), None)
    row.update(arm='C', kind=plan.get('kind', 'rollout'),
        additional_draw=plan.get('additional_draw', True), polluted=False,
        xy=xy.tolist(), trace_sha256=digest_array(xy),
        flow_calls=[dict(step=c['step'], target='g') for c in raw['flow_calls']],
        events=[] if first_success is None else [dict(step=first_success, event='g')],
        runtime_errors=[], head_u_hashes=[], encoded_u_sha256=None,
        waypoint_xy=None, stuck_check_enabled=False,
        detour_trace_sha256=raw['trace_sha256'])
    return row


def execution_record(module, load_environment):
    return dict(module_device=str(module.device),
                torch_cuda_is_available=bool(module.torch.cuda.is_available()),
                **load_environment)


def load_environment_record():
    return {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_GL')}


def evaluate_draw(module, env, plan, *, max_steps=1000, load_environment=None):
    assert_seed_bindings()
    if plan['arm'] not in ('SG', 'C'):
        raise ValueError('arm must be SG or C')
    expected = seeds_detour(plan['task'], plan['episode'], plan['draw'])
    if plan['seeds'] != expected:
        raise ValueError('plan seed record mismatch')
    if load_environment is None:
        load_environment = load_environment_record()
    limited = StepLimitEnv(env, max_steps)
    try:
        with WiringAudit(module, limited, plan['arm']) as audit:
            if plan['arm'] == 'C':
                detour_plan = dict(plan, arm='Q-C', protocol=PROTOCOL)
                raw = run_draw_detour(DetourView(module), limited, detour_plan, 'Q-C')
                row = c_compat_row(raw, plan)
            else:
                row = _run_draw_sg(module, limited, plan, audit=audit, move=3)
        audit.validate(require_distinct=plan['arm'] == 'SG')
        row.update(chunks=audit.chunks, final_cell=cell(limited, limited.obs[:2]),
                   actions=limited.actions, max_steps=max_steps,
                   provenance=module.probe_provenance,
                   execution=execution_record(module, load_environment))
        return row
    finally:
        assert_seed_bindings()



def questions(group):
    if group == 'trap':
        if file_sha(TRAP) != TRAP_PIN:
            raise Blocked(f'BLOCKED: question SHA256 mismatch: {TRAP}')
        rows = json.loads(TRAP.read_text())['questions']
        expected = 30
    elif group == 'gate':
        if file_sha(BUILDER) != BUILDER_PIN:
            raise Blocked(f'BLOCKED: question SHA256 mismatch: {BUILDER}')
        rows = json.loads(BUILDER.read_text())['questions']['gate']
        expected = 8
    else:
        raise ValueError('questions must be trap or gate')
    assert len(rows) == expected, 'question inventory mismatch'
    return [dict(task=q['task'], episode=q['episode'], group=group) for q in rows]


def output_paths(out):
    out = Path(out)
    return out, Path(str(out) + '.summary.json')


def ensure_fresh(out):
    for path in output_paths(out):
        if path.exists() or path.is_symlink():
            raise Blocked(f'BLOCKED: output already exists: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--ckpt-sha256', required=True)
    p.add_argument('--arm', choices=('SG', 'C'), required=True)
    p.add_argument('--questions', choices=('trap', 'gate'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=H)
    p.add_argument('--dataset-dir', type=Path, default=Path('/home/cymaxwelllee/data/ogbench'))
    p.add_argument('--task', type=int, help='CPU selftest inventory filter')
    p.add_argument('--episode', type=int, help='CPU selftest inventory filter')
    p.add_argument('--draw', type=int, help='CPU selftest draw filter')
    args = p.parse_args()
    assert_seed_bindings()
    try:
        ensure_fresh(args.out)
        if not 0 < args.max_steps <= H:
            raise ValueError('--max-steps must be in 1..1000')
        qs = questions(args.questions)
        qs = [q for q in qs if (args.task is None or q['task'] == args.task)
              and (args.episode is None or q['episode'] == args.episode)]
        draws = list(range(80, 84 if args.questions == 'trap' else 82))
        if args.draw is not None:
            if args.draw not in draws:
                raise ValueError('--draw outside requested inventory')
            draws = [args.draw]
        plans = [make_plan(q, args.arm, d) for q in qs for d in draws]
        if not plans:
            raise ValueError('no questions selected')
        for plan in plans:
            seeds_detour(plan['task'], plan['episode'], plan['draw'])
        verify_source()
        load_environment = load_environment_record()
        assert_seed_bindings()
        module = load_ckpt(args.dataset_dir, args.ckpt, args.ckpt_sha256)
        assert_seed_bindings()
        env = module.ogbench.make_env_and_datasets(ENV, env_only=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary = dict(schema='hsweep-rollout-v1', arm=args.arm, questions=args.questions,
                       max_steps=args.max_steps, provenance=module.probe_provenance,
                       execution=execution_record(module, load_environment), errors=0, by_arm_task={})
        try:
            with args.out.open('x') as stream:
                for plan in plans:
                    try:
                        row = evaluate_draw(module, env, plan, max_steps=args.max_steps,
                                            load_environment=load_environment)
                    except Exception:
                        row = dict(plan, success=False, error=True,
                                   traceback=traceback.format_exc(),
                                   provenance=module.probe_provenance,
                                   execution=execution_record(module, load_environment))
                        summary['errors'] += 1
                        print(row['traceback'], file=sys.stderr, end='', flush=True)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    key = f"{row['arm']}:{row['task']}"
                    tally = summary['by_arm_task'].setdefault(key, dict(n=0, successes=0, errors=0))
                    tally['n'] += 1
                    tally['successes'] += int(row['success'])
                    tally['errors'] += int(row.get('error', False))
                    print(f"{key} episode={row['episode']} draw={row['draw']} success={row['success']} steps={row.get('observed_steps')} error={row.get('error', False)}", flush=True)
            with output_paths(args.out)[1].open('x') as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        finally:
            env.close()
            assert_seed_bindings()
        return 1 if summary['errors'] else 0
    except (Blocked, ValueError, FileExistsError, AssertionError) as exc:
        p.exit(2, str(exc) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
