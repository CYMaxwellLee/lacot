"""S2 regression probes: real coordination, hostile artifacts, independent loop oracle."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import artifacts
import calibrate
import common
import harvest
import harness_move1 as m1
import harness_move3 as m3
import rules
import runtime
from builder import build


class Space:
    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)


class ScriptEnv:
    def __init__(self):
        self.action_space = Space()
        self.unwrapped = self
        self.actions = []
        self.script = np.zeros((61, 2), np.float32)
        self.succeed = None
    def reset(self, seed, options):
        self.np_random = np.random.default_rng(seed)
        self.t = 0
        self.actions = []
        return self.script[0].copy(), {'goal': np.array([99., 99.], np.float32)}
    def step(self, action):
        self.actions.append(action.copy())
        self.t += 1
        return self.script[self.t].copy(), 0., False, self.t == len(self.script)-1, {'success': self.t == self.succeed}
    def close(self):
        pass


def stub_module():
    targets = []
    def condvec(s, g):
        targets.append(np.asarray(g).copy())
        return torch.zeros(1, 4)
    return SimpleNamespace(torch=torch, CHUNK=4, _GRAD_CACHE={'u': None},
        _reseed_shuf=lambda n: None, _RDIR=[1.], condvec=condvec,
        normstate=lambda x: x, goal_to_obs=lambda x: x,
        sample_plan=lambda *a: torch.zeros(1, 8, 256),
        ahead=lambda c, u: torch.full((1, 4, 2), .25), _q=lambda u: u, targets=targets)


class FourthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from selftest import ContractTests
        cls.helper = ContractTests()
        cls.helper.builder = build()
        cls.builder = cls.helper.builder

    def validate(self, a, fp):
        return harvest.validate_artifact(a, self.builder, fp, lambda c: c, lambda p: p, fixture=True)

    def test_30_freeze_flags_and_fixed_hash_inventory(self):
        self.assertEqual(common.verify_contract(), common.file_sha(common.CARD))
        before = common.code_hashes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in common.CODE_FILES:
                (root/name).write_bytes((common.HERE/name).read_bytes())
            with patch.object(common, 'HERE', root):
                (root/'unreviewed_extra.py').write_text('not part of runtime')
                self.assertEqual(common.code_hashes(), before)
                for ready in (False, True):
                    for smoke_enabled in (False, True):
                        flags = dict(PRODUCTION_READY=ready, SMOKE_ENABLED=smoke_enabled)
                        (root/'flags.json').write_text(json.dumps(flags))
                        self.assertEqual(common.code_hashes(), before)
                        for mod in (m1, m3):
                            for smoke in (False, True):
                                with patch.object(runtime, 'execute', return_value='ran') as execute:
                                    if smoke_enabled if smoke else ready:
                                        self.assertEqual(mod.production(root, {}, smoke=smoke), 'ran')
                                        execute.assert_called_once()
                                    else:
                                        with self.assertRaises(common.Blocked):
                                            mod.production(root, {}, smoke=smoke)
                                        execute.assert_not_called()
                (root/'flags.json').write_text('{"PRODUCTION_READY": 1, "SMOKE_ENABLED": true}')
                with self.assertRaises(common.Blocked):
                    common.read_flags()

    def test_31_move1_positive_difference_and_conformance_downgrade(self):
        a, fp = self.helper.full_fixture()
        row = next(r for r in a['rows'] if r['task'] == 4 and r['arm'] == 'Q')
        row['xy'][-1] = self.builder['geometries']['4']['exclusive']['B'][0]
        row['trace_sha256'] = common.digest_array(np.asarray(row['xy'], np.float32))
        repeat = next(r for r in a['conformance_rows'] if r['task'] == 4 and r['arm'] == 'Q')
        repeat.update(xy=copy.deepcopy(row['xy']), trace_sha256=row['trace_sha256'])
        out = self.validate(a, fp)
        layer = out['layers']['4']
        self.assertEqual(layer['n_diff'], 1)
        self.assertEqual(layer['response']['table']['AB'], 1)
        self.assertTrue(layer['first_layer']['exact_zero_qualified'])
        self.assertEqual(layer['first_layer']['language'], '行為對注入張量有因果敏感性')
        repeat['xy'][-1][0] += .001
        repeat['trace_sha256'] = common.digest_array(np.asarray(repeat['xy'], np.float32))
        out = self.validate(a, fp)
        self.assertFalse(all(out['conformance'].values()))
        self.assertEqual(out['layers']['4']['first_layer']['language'], '行為差異觀測（不可歸因）')
        for key in rules.HARD_PREREQUISITES:
            prerequisites = dict.fromkeys(rules.HARD_PREREQUISITES, True)
            prerequisites.pop(key)
            self.assertFalse(rules.exact_zero_qualified(prerequisites), key)
        self.assertEqual(rules.content_sensitivity(1, {'anything': True}), '行為差異觀測（不可歸因）')
        for value in (1, 'true', {}, []):
            self.assertFalse(rules.exact_zero_qualified(dict.fromkeys(rules.HARD_PREREQUISITES, value)))

    def test_32_reproduction_and_gate_threshold_four_five(self):
        for qualified in (4, 5):
            a, fp = self.helper.full_fixture()
            gate = [p for p in m1.rollout_plan(self.builder) if p['group'] == 'gate' and p['arm'] == 'N']
            good = {(p['task'], p['episode']) for p in gate[:qualified]}
            keep, skip = [], []
            for p in m1.rollout_plan(self.builder):
                if (p['group'] == 'main' and qualified < 5) or (p['group'] == 'gate' and p['arm'] != 'N' and
                                                              (p['task'], p['episode']) not in good):
                    skip.append(p)
                else:
                    keep.append((harvest.key(p), p['kind']))
            a['rows'] = [r for r in a['rows'] if (harvest.key(r), r['kind']) in keep]
            a['conformance_rows'] = [r for r in a['conformance_rows'] if (harvest.key(r), r['kind']) in keep]
            a['skipped'] = skip
            for row in a['rows']:
                if row['task'] in (1, 3) and ((row['task'], row['episode']) not in good or row['arm'] == 'P'):
                    row.update(success=False, goal_success=[False, False], termination_reason='truncated')
            out = self.validate(a, fp)
            self.assertEqual(out['gate']['qualified'], qualified)
            self.assertEqual(out['gate']['p_reproduced'], 0, 'same route without success is not reproduction')
            self.assertEqual(len(out['layers']['4']['pairs']), 0 if qualified == 4 else 14)
        for q in range(5, 9):
            cutoff = (3*q+3)//4
            self.assertEqual(rules.behavior_gate(q, cutoff-1, 0)['status'], '行為資格不足')
            self.assertEqual(rules.behavior_gate(q, cutoff, 2)['status'], 'passed')
        self.assertEqual(rules.behavior_gate(4, 4, 0)['status'], 'gate 樣本不足')

    def test_33_harvest_hostile_evidence(self):
        a, fp = self.helper.full_fixture()
        mutations = [
            lambda a: a['receipt'].pop('interface'),
            lambda a: a['receipt']['interface'].__setitem__('mask_all_false', False),
            lambda a: a['receipt']['interface'].__setitem__('head_shape', [1, 3, 2]),
            lambda a: next(r for r in a['rows'] if r['arm'] == 'N').__setitem__('flow_calls', []),
            lambda a: next(r for r in a['rows'] if r['arm'] == 'P')['head_u_hashes'].clear(),
            lambda a: a['rows'][0].__setitem__('chunk_steps', 8),
            lambda a: a['rows'][0]['rng_before']['torch_cpu'].__setitem__(0, 0),
            lambda a: a['rows'][0]['rng_before']['noise'].__setitem__(0, 0),
            lambda a: a['rows'][0].__setitem__('termination_reason', 'success'),
            lambda a: a['rows'][0]['goal_success'].__setitem__(-1, True),
        ]
        for arm, group in [('P', 'gate'), ('R', 'gate'), ('R', 'main'), ('Q', 'main')]:
            mutations.append(lambda a, arm=arm, group=group: next(r for r in a['rows'] if r['arm'] == arm and
                ((r['task'] in (1, 3)) == (group == 'gate'))).__setitem__('injection_traj_sha256', 'bad'))
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(a)
            mutate(bad)
            with self.subTest(i=i), self.assertRaises(harvest.Rejected):
                self.validate(bad, fp)
        row, expected, fps = self.helper.fixture('P', 1)
        row.update(flow_calls=[], head_u_hashes=['a'*64]*251, encoded_u_sha256='a'*64,
                   observed_steps=1001, xy=[[0., 0.]]*1002, goal_success=[False]*1002)
        row['trace_sha256'] = common.digest_array(np.asarray(row['xy'], np.float32))
        with self.assertRaisesRegex(harvest.Rejected, 'steps'):
            harvest.validate_rows([row], expected, fps, move=1, fixture=True)
        row, expected, fps = self.helper.fixture('P', 1)
        row.update(flow_calls=[], head_u_hashes=['a'*64]*250, encoded_u_sha256='a'*64)
        row['goal_success'][5] = True
        with self.assertRaisesRegex(harvest.Rejected, 'first success'):
            harvest.validate_rows([row], expected, fps, move=1, fixture=True)

    def test_34_historical_A_B_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = dict(receipt={}, rows=[dict(task=4, episode=4)])
            for arm, fingerprint in [('A', 'a'), ('B', 'b')]:
                path = Path(tmp)/f'{arm}.json'
                common.write_json(path, dict(tasks=[dict(task=4, episode=4, draws=[dict(draw=d,
                    initial_sha256=fingerprint, goal_sha256='g') for d in range(64)])]))
                gate['receipt']['src_'+arm] = dict(path=str(path), sha256=common.file_sha(path))
            with self.assertRaisesRegex(harvest.Rejected, 'A/B fingerprints'):
                harvest.historical_fingerprints(gate)

    def test_35_new_card_and_rule_boundaries(self):
        self.assertEqual(rules.question_grid(['d1']*15+['d3'], ['d1']*16, ['failure']*16)['grid'], rules.C_RESCUE)
        for n in (7, 8):
            q = rules.question_grid(['d1']*n+['d3']*(16-n), ['d3']*16, ['failure']*16)
            self.assertEqual(q['grid'], rules.DEEP if n == 7 else rules.C_RESCUE)
            self.assertEqual(q['bypass_version'], n == 8)
        self.assertEqual(rules.arm_stats(['d1']*4+['d2']*4+[None]*8)['w_miss_rate'], .5)
        self.assertFalse(rules.ceiling(['A']*4+['O']*2, 5))
        self.assertTrue(rules.ceiling(['A']*5+['O'], 5))
        self.assertIsNone(rules.e4([rules.DEEP]*3+[rules.BOTH]*2, 2, 0, '')['dominant'])
        self.assertEqual(rules.e4([rules.DEEP]*4+[rules.BOTH]*2, 3, 0, '')['dominant'], rules.DEEP)
        self.assertTrue(rules.e4([], 2, 0, '')['axis1_deprioritized'])
        self.assertFalse(rules.e4([], 3, 0, '')['axis1_deprioritized'])
        self.assertFalse(rules.adsorption([('A', 'A')]*2+[('O', 'O')]*3, 'A', False)['established'])
        self.assertTrue(rules.adsorption([('A', 'A')]*3+[('O', 'O')]*2, 'A', False)['established'])
        geom = self.builder['geometries']['2']
        row = dict(task=2, arm='A', xy=[[0., 0.], geom['w']['A']], goal_success=[False, True],
                   observed_steps=1, chunk_steps=4, flow_calls=[dict(step=0, target='w')])
        self.assertEqual(harvest.recompute_move3(row, geom['w']['A'], geometry=geom, xy_to_cell=lambda p:p)['category'], 'd4')
        self.assertFalse(harvest.exclusive_crossing(dict(arm='A', xy=[[99, 99]]), geom, lambda p:p))

    def test_36_combined_exits_from_full_artifacts(self):
        a, fp = self.helper.full_fixture()
        one = self.validate(a, fp)
        three = dict(move=3, smoke=False, synthetic=True, layers={str(t): dict(grids=[rules.BOTH]*n,
            discordant_forward=8 if t == 4 else 5, discordant_reverse=0, axis1_deprioritized=False)
            for t, n in [(4, 14), (5, 6)]})
        for exit_name in ('E0', 'E1', 'E2', 'E3', 'E4'):
            v = copy.deepcopy(one)
            for t, n in [('4', 14), ('5', 6)]:
                layer = v['layers'][t]
                layer['n_entries'] = ['O']*n
                if exit_name == 'E0': layer['E0']['passed'] = False
                if exit_name == 'E1': layer['representation']['passed'] = False
                if exit_name == 'E2': layer['failures'] = [(True, False)]*n
                if exit_name == 'E3': layer['n_entries'] = ['A']*n
            result = harvest.combine(v, three)
            self.assertEqual([l['result']['exit'] for l in result['layers'].values()], [exit_name]*2)
        v = copy.deepcopy(one)
        v['layers']['4']['E0']['passed'] = False
        self.assertEqual([l['result']['exit'] for l in harvest.combine(v, three)['layers'].values()], ['E0', 'E0'])
        # Actual trace evidence feeds E2 numerator and N ceiling, not producer labels.
        for row in a['rows']:
            if row['task'] == 4 and row['arm'] == 'P':
                row['termination_reason'] = 'stuck'
        one = self.validate(a, fp)
        self.assertEqual(one['layers']['4']['E2']['result']['numerator'], 14)
        self.assertTrue(one['layers']['4']['E3']['n_ceiling'])
        self.assertEqual(harvest.combine(one, three)['layers']['4']['result']['exit'], 'E2')

    def test_37_stub_fuzz_runloop_noise_switch_target_stuck(self):
        # Adapted from S2 p15. Scripted positions give independent ground truth.
        rng = np.random.default_rng(2007)
        env, module = ScriptEnv(), stub_module()
        for trial in range(180):
            arm = 'ABC'[trial % 3]
            hit = int(rng.integers(1, 45)) if trial % 4 else None
            success = int(rng.integers(1, 60)) if trial % 3 else None
            cap = int(rng.integers(3, 55))
            window = int(rng.integers(2, 8))
            script = np.zeros((61, 2), np.float32)
            if trial % 5:
                script[:, 0] = np.arange(61)*.3
            if hit is not None: script[hit] = [40., 40.]
            env.script, env.succeed = script, success
            module.targets.clear()
            plan = dict(task=4, episode=4, draw=64+trial%16, arm=arm)
            row = runtime.run_draw(module, env, plan, move=3, waypoint=None if arm == 'C' else [40., 40.],
                                   cap_steps=cap, stuck_check=runtime.calibrated_stuck(window, .1))
            w, g, reason = None, None, None
            for t in range(1, 61):
                if arm != 'C' and w is None and np.linalg.norm(script[t]-[40., 40.]) < common.RHO: w=t
                if success == t: g=t
                stuck = t >= window and np.max(np.linalg.norm(script[t-window:t+1]-script[t-window], axis=1)) <= .1
                if g is not None: reason='success'
                elif t == 60: reason='truncated'
                elif arm != 'C' and w is None and stuck: reason='stuck'
                elif arm != 'C' and w is None and t >= cap: reason='cap'
                if reason: break
            self.assertEqual((row['observed_steps'], row['termination_reason']), (t, reason))
            self.assertEqual(row['runtime_errors'], [])
            row['synthetic'] = True
            report = harvest.recompute_move3(row, None if arm == 'C' else [40., 40.], lambda r: trial%2 == 0)
            self.assertEqual((report['w_step'], report['g_step'], report['invalid']), (w, g, []))
            noise = torch.Generator().manual_seed(common.seeds(4,4,64+trial%16)['noise_seed'])
            expected_actions = [np.clip(np.full(2,.25,np.float32)+.05*torch.randn((2,),generator=noise).numpy(),-1,1).astype(np.float32) for _ in range(t)]
            self.assertEqual(np.asarray(env.actions).tobytes(), np.asarray(expected_actions).tobytes())
            expected_targets = [np.array([40.,40.]) if c['target']=='w' else np.array([99.,99.]) for c in row['flow_calls']]
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(module.targets,expected_targets)))
        env.script = np.zeros((61,2), np.float32); env.succeed=None
        row = runtime.run_draw(module, env, dict(task=4,episode=4,draw=64,arm='P'), move=1,
                               u=torch.zeros(1,8,256), stuck_check=runtime.calibrated_stuck(3,.1))
        self.assertEqual((row['termination_reason'], row['observed_steps']), ('stuck',3))

    def coordinator(self, tmp, *, move=1, qualified=8, fail_validation=False, resume_from=None, outname="run"):
        """Real execute/encode/run_draw/harvest/storage; stub only model and environment."""
        b = self.builder
        env = ScriptEnv()
        env.xy_to_ij = lambda p: np.rint(np.asarray(p)/10).astype(int)
        env.ij_to_xy = lambda c: np.asarray(c, np.float32)*10
        module = stub_module()
        module.T_CAP, module.K, module.D_MODEL, module.device = 128, 8, 256, 'cpu'
        module.MU_XY = module.MU_XY_T = np.zeros(2, np.float32)
        module.SD_XY = module.SD_XY_T = np.ones(2, np.float32)
        decoded = {}
        def encode_u(points):
            # Content-addressed stub embedding, preserving route order and donor identity.
            raw = points.numpy().astype(np.float32)
            digest = bytes.fromhex(common.digest_array(raw))
            u = torch.tensor(list(digest)*64, dtype=torch.float32).reshape(1,8,256)
            decoded[common.digest_array(u.numpy())] = points
            return u
        module.encode_u = encode_u
        module._dec = lambda u, initial: decoded[common.digest_array(u.numpy())]
        module.ogbench = SimpleNamespace(make_env_and_datasets=lambda *a, **k: env)
        fixture, _ = self.helper.full_fixture()
        module.probe_provenance = fixture['receipt']['runtime']
        fp = {}
        for group in b['questions'].values():
            for q in group:
                initial = env.ij_to_xy(b['geometries'][str(q['task'])]['s'])
                fp[q['task'], q['episode']] = common.digest_array(initial), common.digest_array(np.array([99.,99.],np.float32))
        gate = [(p['task'],p['episode']) for p in m1.rollout_plan(b) if p['group']=='gate' and p['arm']=='N']
        actual = runtime.run_draw
        called, observed = [], []
        def draw(mod, env, p, **kwargs):
            called.append(dict(p))
            task, arm = p['task'], p['arm']
            geom = b['geometries'][str(task)]
            initial = env.ij_to_xy(geom['s'])
            entry = env.ij_to_xy((geom['exclusive']['A'] or geom['routes']['A'][1:-1])[0])
            env.script = np.stack([initial]+[entry+np.array([i*.2,0],np.float32) for i in range(5)])
            env.succeed = 5 if (task,p['episode']) in gate[:qualified] else None
            self.assertIsNotNone(kwargs.get('stuck_check'))
            if move == 3:
                self.assertEqual(kwargs['cap_steps'], min(common.H, int(np.ceil(geom['w_depth']*2.5*2))))
            if move == 1 and arm != 'N':
                if arm == 'R':
                    donor = b['R_donor'][str(task)]
                    points = [env.ij_to_xy(c) for c in b['geometries'][str(donor['task'])]['routes']['A']]
                elif p['group'] == 'gate':
                    n = next((r for r in observed if r['task']==task and r['episode']==p['episode'] and r['arm']=='N'), None)
                    if n is None:
                        _, old_rows = artifacts.resume_rows(resume_from)
                        n = next(r for r in old_rows if r['task']==task and r['episode']==p['episode'] and r['arm']=='N')
                    points = n['xy']
                else:
                    points = [env.ij_to_xy(c) for c in geom['routes']['A' if arm=='P' else 'B']]
                expected = m1.encode_trajectory(module, points)
                self.assertTrue(torch.equal(kwargs['u'], expected), 'actual injected content differs from independent source')
            row = actual(mod,env,p,**kwargs)
            observed.append(row)
            return row
        calibration = dict(dataset_dir='stub',steps_per_cell=2.5,stuck_window=20,stuck_distance=.01,
                           noise_band={'4':0.,'5':0.},calibrated=False)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runtime,'OUTPUT_ROOT',Path(tmp)))
            stack.enter_context(patch.object(runtime,'load_frozen',return_value=module))
            interface = stack.enter_context(patch.object(runtime,'interface_check',return_value=self.helper.interface()))
            stack.enter_context(patch.object(runtime,'run_draw',side_effect=draw))
            stack.enter_context(patch.object(harvest,'historical_fingerprints',return_value=fp))
            stack.enter_context(patch.object(common,'read_flags',return_value=dict(PRODUCTION_READY=True,SMOKE_ENABLED=True)))
            if fail_validation:
                stack.enter_context(patch.object(harvest,'validate_rows',side_effect=harvest.Rejected('injected validation failure')))
            out = Path(tmp)/outname
            result = runtime.execute(move,out,calibration,smoke=move==3,resume_from=resume_from)
            interface.assert_called_once()
        return result, called, fp

    def test_38_runtime_coordinator_sources_gates_shards_heartbeat(self):
        for qualified in (0,4,5,8):
            with self.subTest(qualified=qualified), tempfile.TemporaryDirectory() as tmp:
                result, called, fp = self.coordinator(tmp,qualified=qualified)
                gate_pr = [p for p in called if p['group']=='gate' and p['arm']!='N']
                main = [p for p in called if p['group']=='main']
                self.assertEqual(len(gate_pr),2*qualified)
                self.assertEqual(len(main),84 if qualified>=5 else 0)
                self.assertEqual(result['readout']['gate']['qualified'],qualified)
                saved = artifacts.load_artifact(Path(tmp)/'run/result.json')
                self.assertEqual(len(saved['rows']),len(result['rows']))
                reread = harvest.validate_artifact(saved,self.builder,fp,lambda c:np.asarray(c,np.float32)*10,
                                                   lambda p:np.rint(np.asarray(p)/10).astype(int))
                self.assertEqual(reread,result['readout'])
                raw = json.loads((Path(tmp)/'run/raw.json').read_text())
                self.assertNotIn('readout',raw)
                self.assertEqual(raw['storage'],'shards-v1')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            result,called,fp = self.coordinator(tmp,move=3)
            self.assertEqual(len(called),128)
            self.assertIn('rollouts=50/',output.getvalue())
            self.assertIn('rollouts=100/',output.getvalue())

    def test_39_raw_survives_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(harvest.Rejected,'injected validation'):
                self.coordinator(tmp,fail_validation=True)
            root = Path(tmp)/'run'
            self.assertTrue((root/'raw.json').is_file())
            self.assertFalse((root/'result.json').exists())
            raw = artifacts.load_artifact(root/'raw.json')
            self.assertEqual(len(raw['rows']),1)
            self.assertIn('rng_before',raw['rows'][0])
            self.assertIn('injected validation',json.loads((root/'validation_error.json').read_text())['message'])
            with self.assertRaises(FileExistsError):
                common.write_json(root/'raw.json',{})

    def test_40_calibration_fail_fast_and_estimators(self):
        for value in ({}, {'dataset_dir':'data'}, {'dataset_dir':'data','noise_band':{'4':0}}):
            with patch.object(runtime,'load_frozen') as load, self.assertRaisesRegex(common.Blocked,'missing calibration'):
                runtime.execute(1,'/not-used',value,smoke=True)
            load.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            # Construct explicit smoke observations, including same-route repeated samples.
            a,fp = self.helper.full_fixture()
            a['receipt']['smoke']=True
            a['receipt']['calibration']['dataset_dir']='data'
            a['calibration_samples']={t:{arm:[np.full((1,8,256),v,np.float32).tolist() for v in (0,.01,.02)]
                for arm in 'AB'} for t in ('1','3','4','5')}
            row=dict(task=4,arm='A',xy=[[0.,0.],[2.,0.],[4.,0.]],waypoint_xy=[4.,0.])
            b=dict(move=3,receipt={'smoke':True},builder=self.builder,rows=[row])
            result=calibrate.estimate(a,b)
            self.assertAlmostEqual(result['noise_band']['4'],np.sqrt(2048)*.02,places=6)
            self.assertEqual(result['steps_per_cell'],.5)
            self.assertTrue(result['calibrated'])
            a['calibration_samples']['1']['A']=[]
            with self.assertRaisesRegex(common.Blocked,'same-route'):
                calibrate.estimate(a,b)

    def test_41_remaining_boundary_and_preprocessing_mutants(self):
        for draw in (63,80):
            with self.assertRaises(ValueError): common.seeds(4,4,draw)
        for calls in (251,252):
            self.assertEqual(rules.classify('A',calls=calls,chunk_steps=4,w_step=1,g_step=1,crossing=True),'d4')
        self.assertIsNone(rules.classify('A',calls=253,chunk_steps=4,w_step=1,g_step=1,crossing=True))
        row,_,_ = self.helper.fixture()
        row['flow_calls'] += [dict(step=999,target='w')]*3
        self.assertIn('budget-exceeded',harvest.recompute_move3(row,[10.,10.],lambda r:False)['invalid'])
        u=np.zeros((1,8,256),np.float32)
        seen=[]
        fixed=m1.FixedInjection(lambda c,u:seen.append(1),u)
        fixed.u[0,0,0]=1
        with self.assertRaises(ValueError): fixed(None)
        self.assertEqual(seen,[], 'pre-head guard must reject before consumer executes')
        def mutate(c,u): u[0,0,0]+=1
        fixed=m1.FixedInjection(mutate,u)
        with self.assertRaisesRegex(ValueError,'head mutated'): fixed(None)
        module=SimpleNamespace(torch=torch,T_CAP=128,K=8,D_MODEL=256,device='cpu',MU_XY=0.,SD_XY=1.,
                               encode_u=lambda points:torch.zeros(1,8,256))
        real_interp=np.interp
        def strict_interp(t,x,y):
            self.assertTrue(np.all(np.diff(x)>0),'zero-length segments reached interpolation')
            return real_interp(t,x,y)
        with patch.object(np,'interp',side_effect=strict_interp):
            m1.encode_trajectory(module,[[0,0],[0,0],[1,0],[1,0],[2,0]])
        # Direct runtime API obeys release flags too.
        with patch.object(common,'read_flags',return_value=dict(PRODUCTION_READY=False,SMOKE_ENABLED=True)):
            with self.assertRaisesRegex(common.Blocked,'PRODUCTION_READY=False'):
                runtime.execute(1,'unused',{})

    def test_42_checkpoint_resume_without_rerunning_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(harvest.Rejected):
                self.coordinator(tmp,qualified=8,fail_validation=True)
            source=Path(tmp)/'run'
            result,called,fp=self.coordinator(tmp,qualified=8,resume_from=source,outname='resumed')
            self.assertEqual(len(called),107)
            self.assertEqual(result['readout']['gate']['qualified'],8)
            self.assertEqual(len(result['rows'])+len(result['conformance_rows']),108)
            result2,called2,_=self.coordinator(tmp,qualified=8,resume_from=Path(tmp)/'resumed',outname='resumed-again')
            self.assertEqual(called2,[])
            self.assertEqual(result['readout'],result2['readout'])
            sidecar=next((source/'shards').glob('*.rng.json.gz'))
            sidecar.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):
                artifacts.resume_rows(source)

    def test_43_move3_discordance_both_directions_from_traces(self):
        rows=[]
        geom=self.builder['geometries']['4']
        for episode,b_success,c_success in [(4,True,False),(5,False,True),(6,True,True)]:
            for arm in 'ABC':
                for draw in range(64,80):
                    success=b_success if arm=='B' else c_success if arm=='C' else False
                    point=geom['w'][arm] if arm!='C' else [99,99]
                    rows.append(dict(task=4,episode=episode,arm=arm,draw=draw,
                        xy=[[99,99],point],goal_success=[False,success],observed_steps=1,
                        chunk_steps=4,flow_calls=[dict(step=0,target='g' if arm=='C' else 'w')]))
        result=harvest.summarize_move3(rows,self.builder,lambda c:c,lambda p:p)
        layer=result['layers']['4']
        self.assertEqual((layer['discordant_forward'],layer['discordant_reverse']),(1,1))
        self.assertEqual(layer['exact_p'],.75)
        self.assertTrue(layer['axis1_deprioritized'])
        self.assertEqual(layer['paired_questions'],3)
