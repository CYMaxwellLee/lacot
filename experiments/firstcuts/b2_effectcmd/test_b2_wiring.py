#!/usr/bin/env python
"""Two-way B2 checks: measured physical table, relay A, and a caught row mutant."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
import run_b2 as b2

b = b2.b


def synthetic_inputs(n_tasks):
    """Real Ant physics and codec, generated state/action tapes; no official-val claims."""
    random.seed(b.SEED)
    np.random.seed(b.SEED)
    torch.manual_seed(b.SEED)
    torch.set_num_threads(1)
    env = b.wv.make_env('/tmp', b.relay.DATASET)
    try:
        qpos, qvel, obs, tapes, tasks = [], [], [], [], []
        for ti in range(n_tasks):
            env.reset(seed=b.SEED+ti)
            leg = dict(active_leg=1, global_step=0, first_hit=-1,
                target_xy=[100., 100.], leg_active_step={'1': 0}, leg_reach_step={},
                leg_start_xy={'1': env.unwrapped.data.qpos[:2].tolist()})
            st = b.save_state(env, leg)
            actions = np.random.default_rng(33+ti).uniform(-1, 1, (200, 8)).astype(np.float32)
            ref, _ = b.rollout(env, st, actions)
            start = ti*201
            qpos.append(ref['qpos'])
            qvel.append(ref['qvel'])
            obs.append(ref['obs'])
            tapes.append(np.vstack([actions, actions[-1]]))
            tasks.append(dict(s0=start, e0=start+200, T=201, n_chunks=50, M=1,
                              wp_xy=ref['qpos'][100:101, :2], episode=ti))
    finally:
        env.close()
    raw = dict(qpos=np.concatenate(qpos), qvel=np.concatenate(qvel),
               observations=np.concatenate(obs), actions=np.concatenate(tapes),
               terminals=np.tile(np.r_[np.zeros(200, dtype=bool), True], n_tasks))
    model, _ = b.wv.load_dict_v1(b.relay.DEFAULT_CKPT)
    def load(args):
        if args.out_dir.resolve().is_relative_to(b.FIRSTCUTS) is False:
            raise ValueError('Outside firstcuts')
        args.out_dir.mkdir(parents=True, exist_ok=False)
        provenance = dict(fixture='synthetic engineering check only', seed=b.SEED)
        return raw, model, tasks, b.wv.make_env('/tmp', b.relay.DATASET), provenance
    return load


class Wiring(unittest.TestCase):
    args = None

    def test_01_se2_wrap_and_fixed_route(self):
        left = np.array([2., 3., np.pi-.01])
        right = np.array([1., 3., -np.pi+.02])
        d = b2.relative(left, right)
        self.assertAlmostEqual(d[2], .03, places=12)
        self.assertAlmostEqual(np.linalg.norm(d[:2]), 1., places=12)
        # An off-route live pose must alter C only; B is frozen at teacher[k].
        teacher = np.array([[0., 0., 0.], [1., 0., 0.]])
        live = np.array([.5, 1., .2])
        b_target = b2.relative(teacher[0], teacher[1])
        c_target = b2.relative(live, teacher[1])
        np.testing.assert_allclose(b_target, [1., 0., 0.])
        self.assertGreater(np.linalg.norm(b_target-c_target), .5)

    def test_02_mutant_caught_before_full_wiring(self):
        # The actual measured table is checked below. This controlled target
        # proves a row/tuple mismatch destroys BOTH selector arms' one-chunk score.
        effects = np.array([[0., 0., 0.], [1., 0., .1], [0., 1., -.1]])
        for arm in ('B', 'C'):
            target = effects[1].copy()
            code = b2.choose(effects, target, 1.)
            mutant = effects[[2, 0, 1]]
            mutant_code = b2.choose(mutant, target, 1.)
            self.assertEqual(code, 1)
            self.assertNotEqual(mutant_code, code)
            normal_error = b2.score(effects[code:code+1], target, 1.)[0]
            mutant_realized_error = b2.score(effects[mutant_code:mutant_code+1], target, 1.)[0]
            self.assertEqual(normal_error, 0.)
            self.assertGreater(mutant_realized_error, .5, arm)
        # If the teacher code's measured effect is available, B recovers it.
        for teacher_code in range(len(effects)):
            self.assertEqual(b2.choose(effects, effects[teacher_code], 1.), teacher_code)

    def test_03_full_synthetic_physics(self):
        s = b2.run_experiment(self.args)
        self.assertEqual(s['status'], 'complete')
        self.assertEqual((s['n_tasks'], s['n_arms'], s['n_table_entries']),
                         (self.args.n_tasks, 3, self.args.n_tasks*32))
        self.assertTrue(s['relay_check']['action_xy_bit_exact'])
        self.assertEqual(s['baseline']['status'], 'not_evaluated_smoke_only')
        self.assertEqual(s['inferential_scope'], 'smoke_only_no_scientific_verdict')
        self.assertGreater(s['n_steps_total'], self.args.n_tasks*(200+200+128+400))
        model, _ = b.wv.load_dict_v1(self.args.ckpt)
        env = b.wv.make_env('/tmp', b.relay.DATASET)
        try:
            for td in sorted(self.args.out_dir.glob('task_*')):
                st = b.load_snapshot(td/'snapshot.npz')
                with np.load(td/'effect_table.npz') as table:
                    self.assertEqual(table['effect'].shape, (32, 3))
                    self.assertEqual(table['shape_change'].shape[0], 32)
                    # Real measured table: swapping the most distant rows makes
                    # both B/C choose a physically wrong tuple at the same start.
                    effects = table['effect']
                    distances = np.array([b2.score(effects, target, 1.)
                                          for target in effects])
                    i, j = np.unravel_index(np.argmax(distances), distances.shape)
                    mutant = effects.copy()
                    mutant[[i, j]] = mutant[[j, i]]
                    for arm in ('B', 'C'):
                        target = effects[i]
                        normal = b2.choose(effects, target, 1.)
                        wrong = b2.choose(mutant, target, 1.)
                        self.assertEqual(normal, i, arm)
                        self.assertEqual(wrong, j, arm)
                        self.assertGreater(b2.score(effects[wrong:wrong+1], target, 1.)[0],
                                           b2.score(effects[normal:normal+1], target, 1.)[0], arm)
                    for code in (0, 17, 31):
                        actual, _ = b.rollout(env, st, model=model, codes=np.array([code]))
                        np.testing.assert_allclose(table['effect'][code],
                            b2.relative(b2.pose(actual['qpos'][0]), b2.pose(actual['qpos'][-1])), atol=0, rtol=0)
                with (np.load(td/'A.npz') as a, np.load(td/'teacher_R.npz') as ref,
                      np.load(td/'B.npz') as bb, np.load(td/'C.npz') as cc):
                    np.testing.assert_array_equal(bb['teacher_pose'], cc['teacher_pose'])
                    np.testing.assert_array_equal(bb['target_effect'], [b2.relative(x, y)
                        for x, y in zip(bb['teacher_pose'][:-1], bb['teacher_pose'][1:])])
                    np.testing.assert_array_equal(cc['target_effect'], [b2.relative(x, y)
                        for x, y in zip(cc['live_pose_at_chunk'], cc['teacher_pose'][1:])])
                    np.testing.assert_array_equal(cc['target_effect'], cc['common_target_effect'])
                    self.assertEqual(a['action'].shape, (200, 8))
                    for flow in (bb, cc):
                        self.assertEqual(flow['qpos'].shape, (201, 15))
                        self.assertEqual(flow['effect_error'].shape, (50,))
                        self.assertEqual(flow['command_effect_error'].shape, (50,))
                        self.assertTrue(np.isfinite(flow['effect_error']).all())
                for name in ('A', 'B', 'C'):
                    with np.load(td/f'{name}_pulse.npz') as p:
                        self.assertEqual(p['action'].shape, (200, 8))
                        self.assertEqual(p['realized_pulse'].shape, (2, 8))
                cal = json.loads((td/'pulse_calibration.json').read_text())
                self.assertLessEqual(cal['relative_error'], .01)
        finally:
            env.close()
        type(self).summary = s


def main():
    p = b.parser(__doc__)
    p.set_defaults(out_dir=b2.HERE/'synthetic_wiring')
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    if args.n_tasks > 4 or args.main:
        p.error('Wiring suite is limited to at most four tasks')
    if args.synthetic:
        b2.load_inputs = synthetic_inputs(args.n_tasks)
    Wiring.args = args
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Wiring)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    s = Wiring.summary
    label = 'SYNTHETIC' if args.synthetic else 'WIRING'
    report = (f'B2 {label} PASS tests={result.testsRun} n_tasks={s["n_tasks"]} '
              f'n_arms={s["n_arms"]} n_table_entries={s["n_table_entries"]} '
              f'n_steps_total={s["n_steps_total"]}\n'
              f'B2 MUTANT CAUGHT both_arms=true\n'
              f'B2 A_RELAY_MATCH action_xy_bit_exact=true\n'
              f'B2 MAIN NOT_RUN expected_A_failures=43/200')
    if args.synthetic:
        report += '\nB2 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true'
    (args.out_dir/'wiring_result.txt').write_text(report+'\n')
    print(report)


if __name__ == '__main__':
    main()
