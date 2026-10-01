"""Move 3 state-machine and shared pinned runtime; GPU release remains closed.

No model or environment is imported until the source pin has passed. Calibration
arguments are required explicitly; the implementation does not invent cap/stuck
constants or a passage-verification metric absent from the approved card.
"""
import argparse
from math import ceil, dist
from pathlib import Path
from common import H, RHO, SIGMA, Blocked, seeds, verify_source
from rules import classify

from common import release_check


def rollout_plan(builder, smoke=False):
    questions = builder['questions']['main'] + builder['questions']['control']
    if smoke:
        questions = [min((q for q in questions if q['task'] == t), key=lambda q: q['episode'])
                     for t in (4, 5, 2)]
    return [dict(**q, arm=arm, draw=draw, seeds=seeds(q['task'], q['episode'], draw))
            for q in questions for arm in ('AC' if q['task'] == 2 else 'ABC')
            for draw in range(64, 80)]


def make_collector(task, episode, draw, obs, goal):
    """Use the read-only Collector's A fresh-flow branch and paired noise RNG."""
    verify_source()
    import sys
    from common import REPO
    sys.path.insert(0, str(REPO))
    from experiments._workorders.ucontrast1.collector import Collector
    collector = Collector('A', 0., 16)
    collector.begin(task, episode, draw, obs, goal)
    expected = seeds(task, episode, draw)
    if collector.flow_seed != expected['flow_seed']:
        raise RuntimeError('fresh-flow branch mismatch')
    return collector


def noisy_action(collector, action):
    """All Move 3 arms use this identical transformation and CPU noise stream."""
    np, torch = collector.np, collector.torch
    action = np.clip(action, -1., 1.).astype(np.float32)
    eps = torch.randn(action.shape, generator=collector.noise).numpy()
    return np.clip(action + SIGMA*eps, -1., 1.).astype(np.float32)


class TwoStage:
    def __init__(self, arm, waypoint, *, chunk_steps, cap_steps, initial_xy):
        if arm not in 'ABC' or len(arm) != 1 or chunk_steps < 1:
            raise ValueError('arm/chunk')
        if (arm == 'C') != (waypoint is None):
            raise ValueError('waypoint/arm mismatch')
        if arm != 'C' and (cap_steps is None or not 0 < cap_steps <= H):
            raise ValueError('explicit calibrated cap required')
        self.arm, self.w = arm, waypoint
        self.chunk_steps, self.cap = chunk_steps, cap_steps
        self.stage = 'g' if waypoint is None else 'w'
        self.t, self.calls = 0, []
        self.w_step = self.g_step = None
        self.reason = None
        self.switch_required = False
        self.errors = []
        self.xy = [list(initial_xy)]
        self.events = []

    def sample(self, target):
        """Call once at the actual flow.sample boundary, never on head.forward."""
        if self.stage == 'stopped' or target != self.stage:
            self.errors.append('mid-chunk-switch-anomaly')
        if self.calls and not self.switch_required and self.t-self.calls[-1]['step'] != self.chunk_steps:
            self.errors.append('chunk-cadence-anomaly')
        self.calls.append(dict(step=self.t, target=target))
        self.switch_required = False

    def observe(self, xy, *, goal_success, stuck=False):
        if self.stage == 'stopped' or self.t >= H:
            raise ValueError('observation after stop/budget')
        if self.switch_required:
            self.errors.append('mid-chunk-switch-anomaly')
        self.t += 1
        self.xy.append(list(xy))
        # The source reach test is strict < rho. Latch every step, not per chunk.
        hit = self.stage == 'w' and dist(xy, self.w) < RHO
        if hit:
            self.w_step, self.reason, self.stage = self.t, 'reached', 'g'
            self.switch_required = True
            self.events.append(dict(step=self.t, event='reached'))
        if goal_success and self.g_step is None:
            self.g_step = self.t
            self.events.append(dict(step=self.t, event='g'))
        if self.stage == 'w' and not goal_success:
            if stuck or self.t >= self.cap:
                # No switch to g on cap/stuck; end with an actual steps+1 trace.
                self.reason = 'stuck' if stuck else 'cap'
                self.stage = 'stopped'
                self.events.append(dict(step=self.t, event=self.reason))
        return self.stage

    def summary(self, crossing):
        result = dict(w_step=self.w_step, g_step=self.g_step, reason=self.reason,
                      bypass_g=self.g_step is not None and
                      (self.w_step is None or self.g_step < self.w_step),
                      calls=len(self.calls), call_limit=ceil(H/self.chunk_steps)+2,
                      events=self.events, flow_calls=self.calls,
                      actual_trace_points=len(self.xy), errors=list(self.errors))
        result['classification'] = (None if self.errors else classify(
            self.arm, calls=len(self.calls), chunk_steps=self.chunk_steps,
            w_step=self.w_step, g_step=self.g_step, crossing=crossing))
        return result


def production(outdir=None, calibration=None, *, smoke=False, resume_from=None):
    verify_source()
    release_check(smoke)
    if calibration is None:
        raise Blocked('BLOCKED: explicit smoke calibration required')
    from runtime import execute
    return execute(3, outdir, calibration, smoke=smoke, resume_from=resume_from)


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
