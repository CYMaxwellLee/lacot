#!/usr/bin/env python
"""Small real-MuJoCo wiring suite; at most four official relay tasks."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import run_b1 as b
import run_b1_precision as precision
import numpy as np
import torch


def synthetic_inputs_factory():
    """Explicit engineering fixture; never a substitute for the official-val gate."""
    random.seed(b.SEED)
    np.random.seed(b.SEED)
    torch.manual_seed(b.SEED)
    torch.set_num_threads(1)
    env = b.wv.make_env('/tmp', b.relay.DATASET)
    try:
        leg = dict(active_leg=1, global_step=0, first_hit=-1, target_xy=[100., 100.],
                   leg_active_step={'1': 0}, leg_reach_step={}, leg_start_xy={'1': [0., 0.]})
        st = b.save_state(env, leg)
        actions = np.random.default_rng(33).uniform(-1, 1, (200, 8)).astype(np.float32)
        ref, _ = b.rollout(env, st, actions)
    finally:
        env.close()
    raw = dict(qpos=ref['qpos'], qvel=ref['qvel'], observations=ref['obs'],
               actions=np.vstack([actions, actions[-1]]), terminals=np.r_[np.zeros(200, dtype=bool), True])
    task = dict(s0=0, e0=200, T=201, n_chunks=50, M=1, wp_xy=ref['qpos'][100:101, :2], episode=0)
    model, _ = b.wv.load_dict_v1(b.relay.DEFAULT_CKPT)

    def inputs(args):
        if not args.out_dir.resolve().is_relative_to(b.FIRSTCUTS):
            raise ValueError('Outputs must remain under experiments/firstcuts/')
        args.out_dir.mkdir(parents=True, exist_ok=False)
        provenance = dict(mujoco=b.mujoco.__version__, hashes={'data': 'SYNTHETIC_NOT_OFFICIAL'},
                          seed=b.SEED, fixture='synthetic engineering check only')
        return raw, model, [task], b.wv.make_env('/tmp', b.relay.DATASET), provenance
    return inputs


class SnapshotTests(unittest.TestCase):
    """No dataset I/O: test complete snapshots, persisted RNG, legal ULP and gate."""
    def test_snapshot_rng_and_ulp(self):
        env = b.wv.make_env('/tmp', b.relay.DATASET)
        env.action_space.seed(b.SEED)
        env.observation_space.seed(b.SEED)
        try:
            env.step(np.full(8, .1))
            leg = dict(active_leg=1, global_step=1, first_hit=-1,
                       target_xy=[1., 1.], leg_active_step={'1': 0},
                       leg_reach_step={}, leg_start_xy={'1': [0., 0.]})
            st = b.save_state(env, leg)
            def draws():
                return [random.random(), np.random.random(), float(torch.rand(())),
                        float(env.unwrapped.np_random.random()),
                        env.action_space.sample().copy(), env.observation_space.np_random.random()]
            expected = draws()
            with tempfile.TemporaryDirectory(dir=b.HERE) as tmp:
                b.save_snapshot(Path(tmp)/'state.npz', st)
                loaded = b.load_snapshot(Path(tmp)/'state.npz')
                env.step(np.zeros(8))
                actual_leg = b.restore_state(env, loaded)
                self.assertEqual(actual_leg, leg)
                for a, e in zip(draws(), expected):
                    np.testing.assert_array_equal(a, e)
                again = b.save_state(env, actual_leg)
                for key in ('qpos', 'qvel', 'act', 'warmstart', 'time', 'integration', 'wrappers'):
                    np.testing.assert_equal(again[key], st[key])
            for ui in range(b.N_ULP):
                perturbed = b.perturb_ulp(st, env.unwrapped.model, np.random.default_rng(ui))
                self.assertAlmostEqual(np.linalg.norm(perturbed['qpos'][3:7]), 1., places=14)
                self.assertTrue(np.any(perturbed['qpos'] != st['qpos']))
                self.assertTrue(np.any(perturbed['qvel'] != st['qvel']))
                # Outside the quaternion, coordinate delta is one adjacent f32 spacing.
                for key, mask in [('qpos', np.r_[0:3, 7:15]), ('qvel', np.arange(14))]:
                    f = st[key].astype(np.float32)
                    delta = np.abs(perturbed[key]-st[key])
                    ulp_max = np.maximum(np.nextafter(f, np.float32(np.inf)).astype(float)-f,
                                         f.astype(float)-np.nextafter(f, np.float32(-np.inf)))
                    self.assertTrue(np.all(delta[mask] <= ulp_max[mask]))
        finally:
            env.close()

    def test_main_baseline_gate(self):
        self.assertEqual(b.baseline_gate(200, 43), 'verified_43_of_200')
        with self.assertRaisesRegex(RuntimeError, 'no other arms run'):
            b.baseline_gate(200, 42)
        self.assertEqual(b.baseline_gate(2, 0), 'not_evaluated_smoke_only')

    def test_native_f64_initial_and_midpoint_replay(self):
        env = b.wv.make_env('/tmp', b.relay.DATASET)
        try:
            leg = dict(active_leg=1, global_step=0, first_hit=-1,
                       target_xy=[100., 100.], leg_active_step={'1': 0},
                       leg_reach_step={}, leg_start_xy={'1': [0., 0.]})
            st = b.save_state(env, leg)
            tape = np.random.default_rng(17).uniform(-1, 1, (200, 8))
            ref, mid = b.rollout(env, st, tape, capture_step=100)
            replay, _ = b.rollout(env, st, tape)
            b.assert_identical(replay, ref)
            replay, _ = b.rollout(env, mid, tape[100:])
            b.assert_identical(replay, {k: ref[k][100:] for k in ('qpos', 'qvel', 'obs', 'action')})
        finally:
            env.close()


class WiringTests(unittest.TestCase):
    args = None

    @classmethod
    def setUpClass(cls):
        cls.summary = b.run_experiment(cls.args)
        p = copy.copy(cls.args)
        p.source_dir = cls.args.out_dir
        p.out_dir = cls.args.out_dir / 'precision'
        cls.precision = precision.run_precision(p)
        cls.taskdirs = sorted(cls.args.out_dir.glob('task_*'))

    def test_full_streams_and_completion_anchors(self):
        s = self.summary
        self.assertEqual(s['n_tasks'], self.args.n_tasks)
        self.assertEqual((s['n_arms'], s['n_dose_arms'], s['n_ulp']), (3, 4, 24))
        self.assertEqual(s['n_steps_total'], self.args.n_tasks*33*200+s['relay_check']['common_steps'])
        self.assertEqual(s['baseline']['status'], 'not_evaluated_smoke_only')
        self.assertIsNone(s['churn_threshold_met'])
        for td in self.taskdirs:
            files = [td/f'{name}.npz' for name in ('T', 'T_replay', 'N', 'L', 'N_offset3',
                                                   'dose_0', 'dose_0.9', 'dose_0.99', 'dose_1')]
            files += sorted(td.glob('ulp_*.npz'))
            self.assertEqual(len(files), 33)
            for path in files:
                with np.load(path, allow_pickle=False) as z:
                    for key, shape in [('qpos', (201, 15)), ('qvel', (201, 14)), ('obs', (201, 29)), ('action', (200, 8))]:
                        self.assertEqual(z[key].shape, shape, (path, key))
                        self.assertTrue(np.isfinite(z[key]).all(), (path, key))
                    self.assertTrue(np.all(np.abs(z['action']) <= 1))

    def test_T_replay_exact_and_L_matches_relay(self):
        self.assertTrue(self.summary['relay_check']['action_xy_bit_exact'])
        self.assertEqual(self.summary['relay_check']['reached'], self.summary['relay_check']['original_reached'])
        for td in self.taskdirs:
            with np.load(td/'T.npz') as t, np.load(td/'T_replay.npz') as r:
                b.assert_identical(t, r)

    def test_N_teacher_obs_timing_and_own_physics(self):
        model, _ = b.wv.load_dict_v1(self.args.ckpt)
        env = b.wv.make_env('/tmp', b.relay.DATASET)
        lo, hi = env.action_space.low, env.action_space.high
        env.close()
        for td in self.taskdirs:
            with np.load(td/'T.npz') as t, np.load(td/'N.npz') as n, np.load(td/'L.npz') as l:
                np.testing.assert_array_equal(n['codes'], l['codes'])
                np.testing.assert_array_equal(n['decoder_ref_step'], np.arange(0, 200, 4))
                np.testing.assert_array_equal(n['decoder_obs'], t['obs'][::4][:-1])
                decoded = np.concatenate([b.decode(model, code, obs, lo, hi)
                                          for code, obs in zip(n['codes'], n['decoder_obs'])])
                np.testing.assert_array_equal(n['action'], decoded)
                self.assertGreater(np.max(np.abs(n['action']-t['action'])), 1e-6)
                self.assertGreater(np.max(np.abs(n['qpos']-t['qpos'])), 1e-6)
                # N's observations come from its own physical state, distinct from decoder input.
                np.testing.assert_array_equal(n['obs'], np.concatenate([n['qpos'], n['qvel']], axis=1).astype(np.float32))
                self.assertGreater(np.max(np.abs(n['obs'][::4][:-1]-n['decoder_obs'])), 1e-6)

    def test_negative_offset3_changes_behavior(self):
        for td in self.taskdirs:
            with np.load(td/'T.npz') as t, np.load(td/'N_offset3.npz') as w, np.load(td/'N.npz') as n:
                np.testing.assert_array_equal(w['decoder_ref_step'], np.arange(0, 200, 4)+3)
                np.testing.assert_array_equal(w['decoder_obs'], t['obs'][np.arange(0, 200, 4)+3])
                self.assertGreater(np.max(np.abs(w['action']-n['action'])), 1e-6)
                self.assertGreater(np.max(np.abs(w['qpos']-n['qpos'])), 1e-6)

    def test_clipped_action_doses_and_endpoints(self):
        for td in self.taskdirs:
            with np.load(td/'T.npz') as t, np.load(td/'N.npz') as n:
                for alpha in b.ALPHAS:
                    with np.load(td/f'dose_{alpha:g}.npz') as d:
                        np.testing.assert_array_equal(d['action'], (1-alpha)*n['action']+alpha*t['action'])
                        if alpha in (0., 1.):
                            b.assert_identical(d, n if alpha == 0. else t)

    def test_precision_single_factor_and_exact_restore(self):
        self.assertEqual(self.precision['full_restore_max_error'], 0.)
        self.assertEqual(self.precision['n_steps_total'], self.args.n_tasks*400)
        for td in sorted((self.args.out_dir/'precision').glob('task_*')):
            native = b.load_snapshot(td/'native_mid_snapshot.npz')
            for name in ('i', 'ii', 'iii'):
                st = b.load_snapshot(td/f'{name}_snapshot.npz')
                for key in ('qpos', 'qvel', 'warmstart', 'act', 'time', 'integration', 'rng', 'leg', 'wrappers'):
                    expected = native[key]
                    if name == 'ii' and key in ('qpos', 'qvel'):
                        expected = expected.astype(np.float32).astype(np.float64)
                    elif name == 'iii' and key == 'warmstart':
                        expected = np.zeros_like(expected)
                    np.testing.assert_equal(st[key], expected)
            with np.load(td/'i.npz') as exact, np.load(td/'reference.npz') as ref:
                b.assert_identical(exact, ref)
            with np.load(td/'ii.npz') as reduced:
                np.testing.assert_array_equal(reduced['qpos'][0], native['qpos'].astype(np.float32).astype(float))
                np.testing.assert_array_equal(reduced['qvel'][0], native['qvel'].astype(np.float32).astype(float))

    def test_precision_standalone_native_reference(self):
        args = copy.copy(self.args)
        args.source_dir = None
        args.out_dir = self.args.out_dir / 'precision_standalone'
        summary = precision.run_precision(args)
        self.assertEqual(summary['n_steps_total'], self.args.n_tasks*600)
        self.assertEqual(summary['full_restore_max_error'], 0.)
        for td in sorted(args.out_dir.glob('task_*')):
            native = b.load_snapshot(td/'native_mid_snapshot.npz')
            prior = b.load_snapshot(self.args.out_dir/'precision'/td.name/'native_mid_snapshot.npz')
            for key in ('qpos', 'qvel'):
                np.testing.assert_array_equal(native[key], prior[key])
        type(self).standalone = summary


def main():
    p = b.parser(__doc__)
    p.set_defaults(out_dir=b.HERE/'smoke')
    p.add_argument('--snapshot-only', action='store_true', help='No dataset read; does not count as full wiring smoke')
    p.add_argument('--synthetic', action='store_true', help='One generated fixture, NOT official-val smoke; requires --n-tasks 1')
    args = p.parse_args()
    if args.n_tasks > 4 or args.main:
        p.error('Wiring test may never run more than four tasks')
    if args.synthetic:
        if args.n_tasks != 1 or args.snapshot_only:
            p.error('--synthetic requires --n-tasks 1 and cannot combine with --snapshot-only')
        b.load_inputs = synthetic_inputs_factory()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SnapshotTests)
    if not args.snapshot_only:
        WiringTests.args = args
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(WiringTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.snapshot_only:
        print('B1 SNAPSHOT PASS full_wiring=NOT_RUN')
    else:
        s, p = WiringTests.summary, WiringTests.precision
        label = 'SYNTHETIC' if args.synthetic else 'WIRING'
        report = (f'B1 {label} PASS tests={result.testsRun} n_tasks={s["n_tasks"]} '
                  f'n_arms={s["n_arms"]} n_ulp={s["n_ulp"]} n_steps_total={s["n_steps_total"]}\n'
                  f'B1 PRECISION PASS n_tasks={p["n_tasks"]} n_arms=3 n_ulp=0 '
                  f'n_steps_total={p["n_steps_total"]} full_restore_max_error={p["full_restore_max_error"]:g}\n'
                  f'B1 PRECISION_STANDALONE PASS n_tasks={WiringTests.standalone["n_tasks"]} '
                  f'n_arms=3 n_ulp=0 n_steps_total={WiringTests.standalone["n_steps_total"]} '
                  f'full_restore_max_error={WiringTests.standalone["full_restore_max_error"]:g}\n'
                  'B1 MAIN NOT_RUN expected_L_failures=43/200')
        if args.synthetic:
            report += '\nB1 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true'
        (args.out_dir/'smoke_result.txt').write_text(report+'\n')
        print(report)


if __name__ == '__main__':
    main()
