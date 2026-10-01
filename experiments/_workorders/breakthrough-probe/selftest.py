"""CPU-only acceptance tests. The real source-pin check is NEVER waived.

Exit 1 if any mandatory check fails, including unavailable approved source.
Mock assertions are printed separately and are not GPU smoke evidence.
"""
import ast
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
# The strengthened harvester verifies actual torch RNG bytes on CPU.
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
except ModuleNotFoundError:
    python = os.environ.get('PROBE_TEST_PYTHON', str(Path(__file__).resolve().parents[3]/'.venv/bin/python'))
    os.execv(python, [python, '-B', *sys.argv])
import numpy as np
from builder import build, geometry
from common import (HERE, SOURCE, PIN, COLLECTOR, H, RHO, Blocked, canonical,
                    digest_array, file_sha, seeds, sha, verify_source, reset_env)
import harness_move3 as m3
import harness_move1 as m1
import harvest
import rules as r


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = build()

    def test_00_actual_frozen_source_pin_REQUIRED(self):
        self.assertEqual(file_sha(SOURCE), PIN, 'BLOCKED: the required frozen source is not present at the specified path')

    def test_01_pin_guard_rejects_before_import(self):
        import runtime
        with tempfile.TemporaryDirectory() as temp:
            wrong = Path(temp)/'wrong.py'
            wrong.write_text('raise RuntimeError("must not import")')
            with patch('common.SOURCE', wrong):
                with self.assertRaisesRegex(Blocked, 'SHA256'):
                    verify_source()
                with self.assertRaisesRegex(Blocked, 'SHA256'):
                    runtime.load_frozen('/nonexistent')
        for module in (m1, m3):
            with self.assertRaisesRegex(Blocked, 'SHA256'):
                with patch('common.SOURCE', wrong):
                    # Supply an existing bad file for the pin gate.
                    with patch('common.file_sha', return_value='0'*64):
                        module.production()

    def test_02_builder_double_run_bytes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            outputs = [Path(temp)/name for name in ('one.json', 'two.json')]
            for output in outputs:
                subprocess.run([sys.executable, str(HERE/'builder.py'), '--out', str(output)],
                               capture_output=True, check=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            for output in outputs:
                self.assertEqual(file_sha(output), Path(str(output)+'.sha256').read_text().strip())
        self.assertEqual(canonical(self.builder), canonical(build()))

    def test_03_geometry_and_machine_question_sets(self):
        b = self.builder
        self.assertEqual([q['episode'] for q in b['questions']['gate'] if q['task'] == 1],
                         [10, 15, 20, 23, 25, 32, 34])
        self.assertEqual(b['geometries']['4']['w'], {'A': [1, 6], 'B': [5, 10]})
        self.assertEqual(b['geometries']['5']['w'], {'A': [1, 3], 'B': [3, 1]})
        self.assertEqual(b['geometries']['2']['w']['A'], [3, 5])
        for geom in b['geometries'].values():
            paths = list(geom['routes'].values())
            self.assertEqual(paths, sorted(paths))
            self.assertTrue(all(len(p) == geom['D']+1 for p in paths))
            self.assertEqual(geom['demo_label'], 'unverified')
            self.assertTrue(set(map(tuple, geom['exclusive']['A'])).isdisjoint(map(tuple, geom['exclusive']['B'])))
        self.assertEqual(b['R_donor']['4'], dict(task=5, episode=5, route='A'))
        self.assertEqual(b['R_donor']['5'], dict(task=4, episode=4, route='A'))
        self.assertEqual(b['R_donor']['1']['task'], 3)
        self.assertEqual(b['R_donor']['3']['task'], 1)
        self.assertNotIn('2', b['R_donor'])
        self.assertEqual(set(b['R_donor']), {'1', '3', '4', '5'})
        self.assertEqual(b['blockers'], [])

    def test_04_draw_plans_streams(self):
        plan = m3.rollout_plan(self.builder)
        self.assertEqual(len(plan), 1280)
        self.assertEqual({p['draw'] for p in plan}, set(range(64, 80)))
        self.assertEqual(len({(p['task'], p['episode'], p['arm'], p['draw']) for p in plan}), 1280)
        for p in plan:
            self.assertEqual(p['seeds']['flow_seed'], p['seeds']['stream_seed'])
            self.assertEqual(p['seeds']['noise_seed'], p['seeds']['stream_seed'])
        old = {7*q['task']+q['episode']+1000003*d
               for group in self.builder['questions'].values() for q in group for d in range(64)}
        self.assertTrue(old.isdisjoint(p['seeds']['stream_seed'] for p in plan))
        move1 = m1.rollout_plan(self.builder)
        self.assertEqual(len(move1), 108)  # v6.2: 80 main + 8 N + 16 P/R + 4 reruns
        self.assertTrue(all(p['seeds']['flow_seed'] is None for p in move1 if p['arm'] != 'N'))
        self.assertTrue(all(p['seeds']['flow_seed'] == p['seeds']['stream_seed'] for p in move1 if p['arm'] == 'N'))
        self.assertTrue(all(p['source'] == 'self-rollout-u' for p in move1 if p['group'] == 'gate' and p['arm'] == 'P'))

    def test_05_reset_three_lines_and_collector_digest(self):
        events = []
        class ActionSpace:
            def seed(self, value):
                events.append(('action_space.seed', value, int(np.random.get_state()[1][0])))
        class Env:
            action_space = ActionSpace()
            def reset(self, **kwargs):
                events.append(('reset', kwargs))
                return np.zeros(2), {'goal': np.ones(2)}
        reset_env(Env(), 4, 30)
        self.assertEqual(events[0], ('action_space.seed', 4030, 4030))
        self.assertEqual(events[1][1], dict(seed=4030, options={'task_id': 4, 'render_goal': False}))
        spec = importlib.util.spec_from_file_location('read_only_collector', COLLECTOR)
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        array = np.array([1., 2.], dtype=np.float32)
        self.assertEqual(digest_array(array), collector.digest_array(array))

    def test_06_d1_d5_and_251_calls(self):
        base = dict(calls=251, chunk_steps=4, crossing=False)
        cases = [(None, 10, False, 'd1'), (20, 10, True, 'd1'),
                 (None, None, False, 'd2'), (3, None, False, 'd3'),
                 (3, 10, True, 'd4'), (3, 10, False, 'd5')]
        for arm in 'AB':
            for w, g, crossing, expected in cases:
                with self.subTest(arm=arm, w=w, g=g, crossing=crossing):
                    self.assertEqual(r.classify(arm, **dict(base, w_step=w, g_step=g, crossing=crossing)), expected)
        self.assertEqual(r.classify('C', **dict(base, w_step=None, g_step=3)), 'success')
        self.assertIsNone(r.classify('A', **dict(base, calls=253, w_step=1, g_step=2)))

    def test_07_invalid_draw_denominators_and_c_priority(self):
        stat = r.arm_stats(['d2']*4+['d3']*4+[None]*8)
        self.assertEqual(stat['w_miss_rate'], .5)
        self.assertFalse(stat['w_unreliable'])
        self.assertFalse(stat['insufficient'])
        self.assertTrue(r.arm_stats(['d3']*7+[None]*9)['insufficient'])
        self.assertEqual(r.question_grid([None]*16, [None]*16, ['success']+[None]*15)['grid'], r.C_RESCUE)
        self.assertEqual(r.question_grid(['d4']*16, ['d4']*16, ['failure']*7+[None]*9)['grid'], r.INSUFFICIENT)
        self.assertEqual(r.question_grid(['d2']*9+['d4']*7, ['d4']*16, ['failure']*16)['grid'], r.INSUFFICIENT)

    def test_08_grids_v5_tie_and_d5(self):
        c = ['failure']*16
        for a, b, expected in [('d4', 'd4', r.BOTH), ('d4', 'd3', r.A_ONLY),
                               ('d3', 'd4', r.B_ONLY), ('d3', 'd3', r.DEEP),
                               ('d5', 'd5', r.UNVERIFIED)]:
            self.assertEqual(r.question_grid([a]*16, [b]*16, c)['grid'], expected)
        tie = r.question_grid(['d3']*8+['d5']*8, ['d3']*8+['d5']*8, c)
        self.assertEqual(tie['grid'], r.AMBIGUOUS)
        self.assertFalse(tie['world_counted'])
        self.assertEqual(r.question_grid(['d1']*16, ['d1']*16, c)['grid'], r.C_RESCUE)
        # v6: unique maximum, not the geometry margin>=2 rule.
        self.assertEqual(r.question_grid(['d1']*3+['d3']*7+['d5']*6,
                                        ['d1']*4+['d3']*6+['d5']*6, c)['grid'], r.DEEP)

    def test_09_state_machine_reach_cap_stuck_and_midchunk(self):
        state = m3.TwoStage('A', [4., 0.], chunk_steps=4, cap_steps=20, initial_xy=[0., 0.])
        state.sample('w')
        self.assertEqual(state.observe([4., 0.], goal_success=False), 'g')
        state.sample('g')  # immediate at step 1; no remaining w chunk execution
        for t in range(2, H+1):
            if t > 2 and (t-2) % 4 == 0:
                state.sample('g')
            state.observe([4., 0.], goal_success=t == H)
        report = state.summary(True)
        self.assertEqual(report['calls'], 251)
        self.assertEqual(report['classification'], 'd4')
        self.assertEqual(report['actual_trace_points'], H+1)
        self.assertEqual(report['errors'], [])
        for stuck, reason in [(False, 'cap'), (True, 'stuck')]:
            short = m3.TwoStage('A', [4., 0.], chunk_steps=4, cap_steps=1, initial_xy=[0., 0.])
            short.sample('w')
            short.observe([0., 0.], goal_success=False, stuck=stuck)
            self.assertEqual(short.reason, reason)
            self.assertEqual(short.stage, 'stopped')
        bad = m3.TwoStage('B', [4., 0.], chunk_steps=4, cap_steps=20, initial_xy=[0., 0.])
        bad.sample('w'); bad.observe([4., 0.], goal_success=False)
        bad.observe([4., 0.], goal_success=False)
        self.assertIn('mid-chunk-switch-anomaly', bad.errors)
        self.assertIsNone(bad.summary(True)['classification'])

    def test_10_strict_reach_and_bypass(self):
        state = m3.TwoStage('A', [4., 0.], chunk_steps=4, cap_steps=20, initial_xy=[0., 0.])
        state.sample('w')
        state.observe([4.-RHO, 0.], goal_success=True)
        self.assertIsNone(state.w_step)
        self.assertEqual(state.summary(False)['classification'], 'd1')
        self.assertTrue(state.summary(False)['bypass_g'])

    def test_11_behavior_delivery_gates_separate(self):
        self.assertEqual(r.behavior_gate(8, 8, 8)['status'], 'gate-content-insensitive')
        self.assertTrue(r.behavior_gate(8, 8, 8)['run_main'])
        self.assertEqual(r.behavior_gate(8, 5, 0)['status'], '行為資格不足')
        self.assertEqual(r.behavior_gate(8, 6, 2)['status'], 'passed')
        self.assertFalse(r.behavior_gate(4, 4, 0)['run_main'])

    def test_12_paired_response_nine_cells(self):
        pairs = [(p, q) for p in 'ABO' for q in 'ABO']
        self.assertEqual(set(r.paired_response(pairs)['table'].values()), {1})
        self.assertEqual(r.paired_response([('A', 'B')]*5)['action'], '跟隨者')
        self.assertEqual(r.paired_response([('A', 'B')]*4)['action'], '方向性訊號')
        self.assertEqual(r.paired_response([('A', 'B')]*5+[('B', 'A')])['action'], '方向性訊號')
        self.assertIsNone(r.paired_response(pairs)['calibrated_p'])

    def test_13_e2_pair_denominator_and_e3(self):
        self.assertTrue(r.e2([(True, False)]*8+[(False, False)]*6, 4)['stopped'])
        self.assertFalse(r.e2([(True, False)]*7+[(False, False)]*7, 4)['stopped'])
        self.assertTrue(r.ceiling(['A']*11+['O']*3, 4))
        self.assertFalse(r.ceiling(['A']*10+['O']*4, 4))
        self.assertTrue(r.ceiling(['B']*5+['O'], 5))
        self.assertFalse(r.adsorption([('B', 'B')]*14, 'A', False)['established'])
        self.assertFalse(r.adsorption([('A', 'A')]*14, 'unverified', False)['enabled'])
        self.assertTrue(r.adsorption([('A', 'A')]*14, 'A', False)['established'])

    def test_14_e4_first_match_v4_and_d5(self):
        grids = [r.BOTH]*6+[r.A_ONLY]*4+[r.B_ONLY]*2+[r.DEEP]*2
        self.assertEqual(r.e4(grids, 8, 0, '跟隨者')['row'], '岔路可執行')
        self.assertEqual(r.e4([r.UNVERIFIED]*14, 0, 0, '跟隨者')['row'], r.UNVERIFIED)
        self.assertEqual(r.e4([r.B_ONLY]*4+[r.DEEP]*2, 4, 0, '跟隨者')['row'], 'B d4 主導但未站穩')
        self.assertEqual(r.exact_one_sided(5, 0), .03125)
        self.assertEqual(r.exact_one_sided(4, 0), .0625)
        for grid, expected in [(r.A_ONLY, '時程形主導'), (r.DEEP, '深執行洞候選/alt 介入未見效'),
                               (r.C_RESCUE, 'C 對照救回主導'), (r.AMBIGUOUS, '無合格題')]:
            self.assertEqual(r.e4([grid]*14, 0, 0, '跟隨者')['row'], expected)

    def test_15_exit_priority(self):
        args = dict(delivery_ok=True, representation_ok=True, failures=[(False, False)]*14,
                    n_entries=['O']*14, task=4, grids=[r.BOTH]*14, forward=8, reverse=0, move1='跟隨者')
        self.assertEqual(r.exits(**args)['exit'], 'E4')
        for changes, expected in [({'n_entries': ['A']*14}, 'E3'),
                                  ({'failures': [(True, False)]*14}, 'E2'),
                                  ({'representation_ok': False}, 'E1'),
                                  ({'delivery_ok': False, 'representation_ok': False}, 'E0')]:
            self.assertEqual(r.exits(**dict(args, **changes))['exit'], expected)

    def test_16_fixed_head_input_mutation_and_manipulation(self):
        u = np.ones((1, 2, 3), np.float32)
        got = []
        inject = m1.FixedInjection(lambda cond, value: got.append(value.copy()), u)
        for _ in range(3):
            inject(None)
        self.assertEqual(inject.head_hashes, [digest_array(u)]*3)
        self.assertTrue(all(np.array_equal(v, u) for v in got))
        inject.u[0, 0, 0] = 2.
        with self.assertRaisesRegex(ValueError, 'E0'):
            inject(None)
        self.assertTrue(m1.manipulation_check(u, u+1, .01, 'A', 'B')['passed'])
        self.assertFalse(m1.manipulation_check(u, u+1, .01, 'A', 'A')['passed'])
        self.assertFalse(m1.manipulation_check(u, u, .01, 'A', 'B')['passed'])
        self.assertTrue(m1.rerun_conformance(lambda value: value.copy(), u)['passed'])
        count = [0]
        def nondeterministic(value):
            count[0] += 1
            return value+count[0]
        self.assertFalse(m1.rerun_conformance(nondeterministic, u)['passed'])
        self.assertEqual(r.content_sensitivity(1, {'same_u_rerun': False}), '行為差異觀測（不可歸因）')

    @staticmethod
    def rng(seeds):
        state = torch.Generator().manual_seed(seeds['torch_seed']).get_state().tolist()
        value = dict(python=[3, [1], None], numpy=['MT19937', [1], 0, 0, 0.],
                     torch_cpu=state, torch_cuda=[], environment={'state': 1}, action_space={'state': 1})
        if seeds['flow_seed'] is not None:
            value['noise'] = torch.Generator().manual_seed(seeds['noise_seed']).get_state().tolist()
        return value

    @staticmethod
    def interface():
        return dict(encoder_shape=[1, 8, 256], head_shape=[1, 4, 2], mask_all_false=True,
                    source_sha256=PIN, encoded_u_sha256='a'*64, head_u_hashes=['a'*64])

    def fixture(self, arm='A', move=3):
        initial, goal = np.zeros(2, np.float32), np.array([20., 20.], np.float32)
        xy = np.zeros((H+1, 2), np.float32)
        row = dict(task=4, episode=4, arm=arm, draw=64, source_sha256=PIN,
                   synthetic=True, polluted=False, observed_steps=H, termination_reason='horizon',
                   seeds=seeds(4, 4, 64, move == 1 and arm != 'N'),
                   initial=initial.tolist(), goal=goal.tolist(), observation_dtype='float32', goal_dtype='float32',
                   initial_sha256=digest_array(initial), goal_sha256=digest_array(goal),
                   xy=xy.tolist(), trace_sha256=digest_array(xy), goal_success=[False]*(H+1), chunk_steps=4,
                   flow_calls=[dict(step=t, target='w' if arm != 'C' else 'g') for t in range(0, H, 4)])
        row.update(success=False, rng_before=self.rng(row['seeds']), rng_after=self.rng(row['seeds']))
        return row, {harvest.key(row)}, {(4, 4): (row['initial_sha256'], row['goal_sha256'])}

    def test_17_harvester_negative_matrix(self):
        row, expected, fp = self.fixture()
        harvest.validate_rows([row], expected, fp, move=3, fixture=True)
        bad_cases = [([], 'missing'), ([row, row], 'duplicate')]
        mutations = [('seed', lambda v: v['seeds'].__setitem__('flow_seed', 0)),
                     ('reset', lambda v: v.__setitem__('initial_sha256', '0'*64)),
                     ('pin', lambda v: v.__setitem__('source_sha256', '0'*64)),
                     ('pollution', lambda v: v.__setitem__('polluted', True)),
                     ('foreign-question', lambda v: v.__setitem__('episode', 999)),
                     ('decimation', lambda v: v['xy'].pop()),
                     ('padding', lambda v: v.__setitem__('observed_steps', 1)),
                     ('trace-tamper', lambda v: v['xy'][2].__setitem__(0, 3.)),
                     ('raw-reset-tamper', lambda v: v['initial'].__setitem__(0, 3.)),
                     ('synthetic-laundering', lambda v: v.__setitem__('synthetic', False))]
        for name, mutation in mutations:
            bad = copy.deepcopy(row); mutation(bad); bad_cases.append(([bad], name))
        for rows, name in bad_cases:
            with self.subTest(name=name), self.assertRaises(harvest.Rejected):
                harvest.validate_rows(rows, expected, fp, move=3, fixture=True)
        with self.assertRaises(harvest.Rejected):
            harvest.validate_rows([row], expected, fp, move=3)

    def test_18_independent_recompute_all_d_and_switch_mutant(self):
        for arm in 'AB':
            for expected, hit, success, crossing in [('d1', False, True, False), ('d2', False, False, False),
                                                     ('d3', True, False, False), ('d4', True, True, True),
                                                     ('d5', True, True, False)]:
                row, _, _ = self.fixture(arm)
                if hit:
                    row['xy'][1:] = [[4., 0.]]*H
                    row['flow_calls'] = [dict(step=0, target='w')]+[dict(step=t, target='g') for t in range(1, H, 4)]
                if success:
                    row['goal_success'][-1] = True
                row['classification'] = 'deliberately-wrong'
                result = harvest.recompute_move3(row, [4., 0.], lambda unused: crossing)
                self.assertEqual(result['category'], expected)
        row['flow_calls'][1]['step'] = 4
        self.assertIsNone(harvest.recompute_move3(row, [4., 0.], lambda unused: True)['category'])

    def test_19_overwritten_injection_rejected(self):
        row, expected, fp = self.fixture('P', 1)
        row.update(flow_calls=[], head_u_hashes=['a'*64]*250, encoded_u_sha256='a'*64)
        harvest.validate_rows([row], expected, fp, move=1, fixture=True)
        row['head_u_hashes'][17] = 'b'*64
        with self.assertRaisesRegex(harvest.Rejected, 'delivery'):
            harvest.validate_rows([row], expected, fp, move=1, fixture=True)
        row['head_u_hashes'][17] = 'a'*64
        row['flow_calls'] = [dict(step=0, target='g')]
        with self.assertRaisesRegex(harvest.Rejected, 'overwritten'):
            harvest.validate_rows([row], expected, fp, move=1, fixture=True)

    def test_20_representation_pass_does_not_hide_rollout_failure(self):
        self.assertTrue(m1.manipulation_check(np.zeros(2), np.ones(2), .01, 'A', 'B')['passed'])
        geom = self.builder['geometries']['4']
        row, _, _ = self.fixture('P', 1)
        result = harvest.recompute_move1(row, row, geom, lambda xy: (99, 99))
        self.assertEqual((result['P']['route'], result['Q']['route']), ('O', 'O'))
        self.assertTrue(r.e2([(True, True)]*14, 4)['stopped'])

    def test_21_gate_source_and_smoke_coverage(self):
        row, _, _ = self.fixture('N', 1)
        row['success'] = True
        self.assertEqual(m1.gate_p_source(row)['xy'], row['xy'])
        self.assertEqual(m1.gate_p_source(row)['source'], 'self-rollout-u')
        row['success'] = False
        self.assertIsNone(m1.gate_p_source(row))
        self.assertEqual({(p['task'], p['arm']) for p in m3.rollout_plan(self.builder, True)},
                         {(4, a) for a in 'ABC'} | {(5, a) for a in 'ABC'} | {(2, a) for a in 'AC'})
        self.assertEqual({(p['task'], p['arm']) for p in m1.rollout_plan(self.builder, True)},
                         {(t, a) for t in (4, 5) for a in 'NPQR'} | {(t, a) for t in (1, 3) for a in 'NPR'})

    def test_22_deterministic_flags_mock(self):
        flags = {}
        def save(name):
            return lambda *args, **kwargs: flags.__setitem__(name, (args, kwargs))
        cuda_backend = SimpleNamespace(matmul=SimpleNamespace(),
            enable_flash_sdp=save('flash'), enable_mem_efficient_sdp=save('efficient'),
            enable_math_sdp=save('math'), enable_cudnn_sdp=save('cudnn_sdp'))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
            use_deterministic_algorithms=save('deterministic'),
            backends=SimpleNamespace(cuda=cuda_backend, cudnn=SimpleNamespace(version=lambda: 'mock')),
            utils=SimpleNamespace(deterministic=SimpleNamespace()),
            set_num_threads=save('threads'), set_num_interop_threads=save('interop'),
            __version__='mock', version=SimpleNamespace(cuda='mock'))
        prior = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        try:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            record = m1.configure_determinism(torch)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(flags['deterministic'], ((True,), {'warn_only': False}))
            self.assertEqual(flags['flash'][0], (False,))
            self.assertEqual(flags['math'][0], (True,))
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            self.assertEqual(record['dtype'], 'float32')
            self.assertEqual(record['batch'], 1)
        finally:
            if prior is None:
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            else:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = prior

    def test_23_production_adapter_REQUIRED(self):
        # Implementation acceptance and GPU release are independent requirements.
        python = os.environ.get('PROBE_TEST_PYTHON', str(HERE.parents[2]/'.venv/bin/python'))
        with tempfile.TemporaryDirectory() as temp:
            run = subprocess.run([python, '-B', str(HERE/'runtime_selftest.py')],
                capture_output=True, text=True, timeout=180,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8', MUJOCO_GL='egl',
                         TMPDIR=temp, MPLCONFIGDIR=temp, OMP_NUM_THREADS='1'))
        self.assertEqual(run.returncode, 0, run.stdout+'\n'+run.stderr)
        self.assertIn('REAL_CPU_PASS:', run.stdout)
        print('\n'+run.stdout, end='')
        from common import read_flags
        self.assertEqual(set(read_flags()), {'PRODUCTION_READY', 'SMOKE_ENABLED'})
        for module in (m1, m3):
            with patch('common.read_flags', return_value={'PRODUCTION_READY': False, 'SMOKE_ENABLED': True}):
                with self.assertRaisesRegex(Blocked, 'PRODUCTION_READY=False'):
                    module.production()

    def test_24_move1_108_limit_and_conditional_rows(self):
        plan = m1.rollout_plan(self.builder)
        self.assertEqual(len(plan), 108)
        counts = Counter((p['group'], p['kind']) for p in plan)
        self.assertEqual(counts, {('main', 'rollout'): 80, ('gate', 'rollout'): 24,
                                  ('main', 'determinism-verification'): 4})
        self.assertEqual(sum(p['group'] == 'gate' and p['arm'] == 'N' for p in plan), 8)
        self.assertEqual(sum(p['conditional_on_N_success'] for p in plan), 16)
        self.assertEqual(len({(p['task'], p['episode'], p['arm'], p['kind']) for p in plan}), 108)
        self.assertTrue(all(not p['additional_draw'] for p in plan if p['kind'] != 'rollout'))
        self.assertFalse(any(p['task'] == 2 for p in plan))

    def test_25_short_trace_contract(self):
        for steps in (1, 5, 999, 1000):
            for reason in ('success', 'terminated', 'truncated', 'cap', 'stuck'):
                row, expected, fp = self.fixture('P', 1)
                row.update(observed_steps=steps, termination_reason=reason, flow_calls=[],
                           head_u_hashes=['a'*64]*((steps+3)//4), encoded_u_sha256='a'*64)
                row['xy'] = row['xy'][:steps+1]
                row['trace_sha256'] = digest_array(np.asarray(row['xy'], np.float32))
                row['goal_success'] = [False]*steps+[reason == 'success']
                row['success'] = reason == 'success'
                harvest.validate_rows([row], expected, fp, move=1, fixture=True)
                for field, bad in [('observed_steps', steps-1), ('termination_reason', None),
                                   ('padded', True), ('spliced', True)]:
                    mutant = dict(row, **{field: bad})
                    with self.subTest(steps=steps, reason=reason, field=field), self.assertRaises(harvest.Rejected):
                        harvest.validate_rows([mutant], expected, fp, move=1, fixture=True)

    def test_26_d4_exclusive_crossing_symmetric(self):
        for task in ('4', '5'):
            geom = self.builder['geometries'][task]
            for arm, other in [('A', 'B'), ('B', 'A')]:
                own, opposite = geom['exclusive'][arm][0], geom['exclusive'][other][0]
                row = dict(arm=arm, xy=[geom['s'], own, geom['g']])
                self.assertTrue(harvest.exclusive_crossing(row, geom, lambda xy: xy))
                for points in ([geom['s'], geom['g']], [geom['s'], opposite, geom['g']],
                               [geom['s'], own, opposite, geom['g']]):
                    row['xy'] = points
                    self.assertFalse(harvest.exclusive_crossing(row, geom, lambda xy: xy))
                # Exercise the actual d4/d5 classifier with short observed traces,
                # including a goal reached in the same step as the waypoint.
                row, _, _ = self.fixture(arm)
                w = geom['w'][arm]
                row.update(xy=[[0., 0.], w], observed_steps=1, termination_reason='success',
                           goal_success=[False, True], flow_calls=[dict(step=0, target='w')])
                result = harvest.recompute_move3(row, w, geometry=geom, xy_to_cell=lambda xy: xy)
                self.assertEqual(result['category'], 'd4')
                self.assertEqual(result['invalid'], [])
                row.update(xy=[[0., 0.], w, opposite], observed_steps=2,
                           goal_success=[False, False, True],
                           flow_calls=[dict(step=0, target='w'), dict(step=1, target='g')])
                result = harvest.recompute_move3(row, w, geometry=geom, xy_to_cell=lambda xy: xy)
                self.assertEqual(result['category'], 'd5')

    def test_27_m9_pin_and_import_boundary(self):
        import runtime
        self.assertEqual(SOURCE.parts[-2:], ('M9', 'scratch_lacot_rollout.py'))
        self.assertEqual(verify_source(), PIN)
        self.assertEqual(runtime.import_boundary(), 2480)

    def full_fixture(self):
        from common import receipt
        b = self.builder
        evidence = {}
        for task in ('4', '5'):
            geom = b['geometries'][task]
            evidence[task] = dict(u_A=np.zeros((1, 8, 256), np.float32).tolist(),
                                  u_B=np.ones((1, 8, 256), np.float32).tolist(),
                                  decoded_A=[geom['exclusive']['A'][0]]*128,
                                  decoded_B=[geom['exclusive']['B'][0]]*128)
        rows, repeats, fp = [], [], {}
        for p in m1.rollout_plan(b):
            row, _, _ = self.fixture(p['arm'], 1)
            task, episode, arm = (p[k] for k in ('task', 'episode', 'arm'))
            xy = [row['xy'][0], b['geometries'][str(task)]['exclusive']['A'][0]]
            success = p['group'] == 'gate'
            encoded = digest_array(np.full((1, 8, 256), arm == 'Q', np.float32))
            row.update(task=task, episode=episode, seeds=p['seeds'], xy=xy,
                       trace_sha256=digest_array(np.asarray(xy, np.float32)), observed_steps=1,
                       goal_success=[False, success], termination_reason='success' if success else 'truncated',
                       flow_calls=[dict(step=0, target='g')] if arm == 'N' else [],
                       head_u_hashes=[] if arm == 'N' else [encoded], encoded_u_sha256=encoded,
                       injection_source=p['source'], kind=p['kind'], success=success,
                       stuck_check_enabled=True, rng_before=self.rng(p['seeds']), rng_after=self.rng(p['seeds']))
            if arm != 'N':
                points = (xy if p['group'] == 'gate' and arm == 'P' else
                          b['geometries'][str(b['R_donor'][str(task)]['task'])]['routes']['A'] if arm == 'R' else
                          b['geometries'][str(task)]['routes']['A' if arm == 'P' else 'B'])
                row['injection_traj_sha256'] = digest_array(np.asarray(points, np.float64))
            if arm == 'R':
                row['donor'] = b['R_donor'][str(task)]
            if p['group'] == 'gate' and arm == 'P':
                row['source_trace_sha256'] = row['trace_sha256']
            fp[(task, episode)] = row['initial_sha256'], row['goal_sha256']
            (rows if p['kind'] == 'rollout' else repeats).append(row)
        artifact = dict(builder=b, move=1, rows=rows, conformance_rows=repeats, skipped=[],
                        representation_evidence=evidence,
                        receipt=dict(receipt(b), smoke=False, interface=self.interface(),
                            hard_prerequisites={k: True for k in r.HARD_PREREQUISITES},
                            runtime={'deterministic_settings': dict(deterministic=True, warn_only=False,
                                cudnn_benchmark=False, cudnn_deterministic=True, tf32=False, sdpa='math',
                                fill_uninitialized_memory=True, threads=1, dataloader_workers=0,
                                dtype='float32', batch=1, cublas_workspace_config=':4096:8')},
                            calibration={'noise_band': {'4': .01, '5': .01}, 'calibrated': True}))
        return artifact, fp

    def test_28_full_artifact_conditional_coverage_and_donors(self):
        b = self.builder
        artifact, fp = self.full_fixture()
        def validate(value):
            return harvest.validate_artifact(value, b, fp, lambda c: c, lambda xy: xy, fixture=True)
        result = validate(artifact)
        self.assertEqual(result['gate']['qualified'], 8)
        self.assertEqual(result['gate']['status'], 'gate-content-insensitive')
        self.assertEqual(len(result['conformance']), 4)
        self.assertTrue(all(result['conformance'].values()))
        self.assertEqual(len(result['layers']['4']['pairs']), 14)
        self.assertEqual(len(result['layers']['5']['pairs']), 6)
        bad = copy.deepcopy(artifact)
        next(row for row in bad['rows'] if row['arm'] == 'R')['donor'] = {'task': 2}
        with self.assertRaisesRegex(harvest.Rejected, 'donor'):
            validate(bad)
        bad = copy.deepcopy(artifact)
        bad['conformance_rows'].pop()
        with self.assertRaisesRegex(harvest.Rejected, 'missing'):
            validate(bad)
        bad = copy.deepcopy(artifact)
        bad['conformance_rows'][0].update(encoded_u_sha256='b'*64, head_u_hashes=['b'*64])
        with self.assertRaisesRegex(harvest.Rejected, 'same u'):
            validate(bad)
        bad = copy.deepcopy(artifact)
        next(row for row in bad['rows'] if row['task'] == 1 and row['arm'] == 'P')['source_trace_sha256'] = 'bad'
        with self.assertRaisesRegex(harvest.Rejected, 'self-rollout'):
            validate(bad)
        # N failures remove only their conditional gate P/R; <5 closes main.
        reduced = copy.deepcopy(artifact)
        reduced['rows'] = [row for row in reduced['rows'] if row['task'] in (1, 3) and row['arm'] == 'N']
        for row in reduced['rows']:
            row.update(goal_success=[False, False], termination_reason='truncated', success=False)
        reduced['conformance_rows'] = []
        reduced['skipped'] = [p for p in m1.rollout_plan(b) if not (p['group'] == 'gate' and p['arm'] == 'N')]
        result = validate(reduced)
        self.assertEqual(result['gate']['qualified'], 0)
        self.assertEqual(result['gate']['status'], 'gate 樣本不足')
        self.assertEqual(result['layers']['4']['pairs'], [])

    def test_29_generated_contract_snapshots(self):
        from smoke import inventory
        for name, value in [('builder.json', self.builder), ('smoke-plan.json', inventory(self.builder)),
                            ('move1-plan.json', m1.rollout_plan(self.builder))]:
            path = HERE/name
            self.assertEqual(path.read_bytes(), canonical(value))
            self.assertEqual(Path(str(path)+'.sha256').read_text().strip(), file_sha(path))


def main():
    print('CPU contract selftest; mock tests are not GPU smoke evidence.', flush=True)
    print(f'Frozen source expected={PIN}', flush=True)
    print(f'Frozen source actual={file_sha(SOURCE)}', flush=True)
    from fourth_selftest import FourthTests
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(FourthTests)])
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    print('STATUS: DONE' if result.wasSuccessful() else 'STATUS: BLOCKED — mandatory acceptance checks failed', flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
