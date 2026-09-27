#!/usr/bin/env python
"""C2 real-Ant wiring tests; synthetic or official B1 source, at most four tasks."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_c2 as c
import numpy as np


class WiringTests(unittest.TestCase):
    args = None

    @classmethod
    def setUpClass(cls):
        cls.summary = c.run_experiment(cls.args)
        cls.out = cls.args.out_dir
        cls.env = c.wv.make_env('/tmp', c.relay.DATASET)
        cls.weights = c.npz(cls.out/'frozen_W.npz')

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def cells(self):
        for name in c.CODEC_LABELS:
            for ti in range(self.args.n_tasks):
                td = self.out/name/f'task_{ti:03d}'
                nominal = c.npz(td/'nominal.npz')
                for dose,_ in c.CONDITIONS:
                    dd = td/f'dose_{dose}'
                    yield name,ti,dose,dd,nominal,{a:c.npz(dd/f'{a}.npz') for a in ('R','N','M')}

    def test_00_mutant_is_caught_before_accepting_wiring(self):
        for name,ti,dose,dd,nominal,arms in self.cells():
            if dose!='1':
                continue
            mutant = c.npz(dd/'M_correct_innovation_mutant.npz')
            np.testing.assert_array_equal(mutant['decoder_obs_source'],np.asarray('M'))
            expected=np.array([c.inject_innovation(self.env.unwrapped.model,o,d)
                for o,d in zip(nominal['obs'][4:36:4],arms['M']['target_innovation'])])
            np.testing.assert_array_equal(mutant['decoder_obs'],expected)
            normal_gap = c.assert_mismatch_visible(arms['R'],arms['M'])
            with self.assertRaisesRegex(AssertionError,'M mutant caught'):
                c.assert_mismatch_visible(arms['R'],mutant)
            c.b1.assert_identical(arms['R'],mutant)
            self.assertGreater(normal_gap,c.PREREGISTERED['mismatch_state_linf_min'])
        self.assertTrue(all(r['caught'] and r['collapse_ratio']<=1e-3 for r in self.summary['mutant_checks']))

    def test_zero_dose_and_nominal_self_replay(self):
        for name,ti,dose,dd,nominal,arms in self.cells():
            if dose!='0':
                continue
            check=c.positive_check(arms['R'],arms['N'],c.trim_reference(nominal))
            self.assertLessEqual(check['R_N_state_linf'],1e-10)
            self.assertEqual(check['nominal_replay_state_linf'],0.)
            c.b1.assert_identical(nominal,c.npz(dd.parent/'nominal_replay.npz'))
            np.testing.assert_array_equal(arms['M']['innovation'],0.)
            c.b1.assert_identical(arms['R'],arms['M'])

    def test_three_arms_share_post_pulse_physics_and_code_tape(self):
        for name,ti,dose,dd,nominal,arms in self.cells():
            post=c.load_snapshot(dd/'post_pulse_snapshot.npz')
            prefix=c.npz(dd/'pulse.npz')
            for stream in arms.values():
                for key in ('qpos','qvel','integration'):
                    np.testing.assert_array_equal(stream[key][0],post[key])
                    np.testing.assert_array_equal(stream[key][0],prefix[key][-1])
                np.testing.assert_array_equal(stream['codes'],nominal['codes'][1:9])
            if dose!='0':
                self.assertGreater(np.max(np.abs(prefix['qpos'][-1]-nominal['qpos'][4])),1e-8)

    def test_N_uses_own_codec_nominal_simultaneous_obs(self):
        models={name:c.wv.load_dict_v1(path)[0] for name,path in
                [('p0_dict_v1',self.args.ckpt_p0),('sigma0',self.args.ckpt_sigma0),
                 ('sigma1.0',self.args.ckpt_sigma1)]}
        own_physics_diff=[]
        for name,ti,dose,dd,nominal,arms in self.cells():
            n=arms['N']
            np.testing.assert_array_equal(n['decoder_ref_step'],np.arange(4,36,4))
            np.testing.assert_array_equal(n['decoder_obs'],nominal['obs'][4:36:4])
            np.testing.assert_array_equal(arms['R']['decoder_obs'],arms['R']['obs'][:-1:4])
            decoded=np.concatenate([c.b1.decode(models[name],code,obs,-1.,1.)
                for code,obs in zip(n['codes'],n['decoder_obs'])])
            np.testing.assert_array_equal(n['action'],decoded)
            if dose=='1':
                own_physics_diff.append(np.max(np.abs(n['obs'][:-1:4]-n['decoder_obs'])))
        self.assertTrue(all(v>1e-6 for v in own_physics_diff))

    def test_M_is_scaled_other_event_sequence_in_rotation_tangent(self):
        for name,ti,dose,dd,nominal,arms in self.cells():
            m=arms['M']
            donor_id=int(m['donor_task'])
            self.assertNotEqual(ti,donor_id)
            donor_dir=self.out/name/f'task_{donor_id:03d}'
            donor_ref=c.npz(donor_dir/'nominal.npz')['obs'][4:36:4]
            donor_r=c.npz(donor_dir/f'dose_{dose}'/'R.npz')['obs'][:-1:4]
            expected=np.array([c.innovation(self.env.unwrapped.model,a,b) for a,b in zip(donor_ref,donor_r)])
            np.testing.assert_array_equal(m['donor_innovation'],expected)
            np.testing.assert_allclose(m['innovation'],expected*float(m['donor_scale']),rtol=0,atol=0)
            self.assertAlmostEqual(np.linalg.norm(m['innovation']),np.linalg.norm(m['target_innovation']),places=10)
            inputs=np.array([c.inject_innovation(self.env.unwrapped.model,o,d)
                for o,d in zip(nominal['obs'][4:36:4],m['innovation'])])
            np.testing.assert_array_equal(m['decoder_obs'],inputs)
            np.testing.assert_allclose(np.linalg.norm(inputs[:,3:7],axis=1),1.,atol=1e-12)
            if dose=='1':
                self.assertGreater(np.linalg.norm(m['innovation']-m['target_innovation']),1e-6)

    def test_pulse_duration_calibration_doses_and_flip(self):
        for name,ti,dose,dd,nominal,arms in self.cells():
            cal=c.json.loads((dd.parent/'calibration/calibration.json').read_text())
            pulse=c.npz(dd/'pulse.npz')
            multiplier=dict(c.CONDITIONS)[dose]
            np.testing.assert_array_equal(pulse['requested_pulse'],multiplier*np.array(cal['pulse']))
            expected=nominal['action'][:4].copy()
            expected[:2]=np.clip(expected[:2]+pulse['requested_pulse'],-1.,1.)
            np.testing.assert_array_equal(pulse['action'],expected)
            self.assertLessEqual(cal['relative_error'],.01)
            self.assertAlmostEqual(cal['target'],cal['nominal_4step_rms']*.5)
            if dose=='1':
                delta=c.innovation(self.env.unwrapped.model,nominal['obs'][4],pulse['obs'][-1])[:14]
                self.assertAlmostEqual(np.sqrt(np.mean(delta**2)),cal['achieved'])

    def test_frozen_W_no_evaluation_leak_and_split(self):
        nominal={name:[c.npz(self.out/name/f'task_{ti:03d}'/'nominal.npz') for ti in range(self.args.n_tasks)]
                 for name in c.CODEC_LABELS}
        expected=c.fit_weights(nominal,self.summary['calibration_ids'])
        for s in c.SPACES:
            np.testing.assert_array_equal(expected[s],self.weights[s])
            self.assertFalse(expected[s].flags.writeable)
        altered=copy.deepcopy(nominal)
        for flows in altered.values():
            for ti in self.summary['evaluation_ids']:
                flows[ti]['shape'][:]=1e8
                flows[ti]['qpos'][:,:3]=1e8
        unchanged=c.fit_weights(altered,self.summary['calibration_ids'])
        for s in c.SPACES:
            np.testing.assert_array_equal(unchanged[s],expected[s])
        self.assertFalse(set(self.summary['calibration_ids']) & set(self.summary['evaluation_ids']))
        self.assertFalse(self.summary['W']['official_40_160'])

    def test_Gamma_identity_and_cutoff_bookkeeping(self):
        for row in self.summary['per_task']:
            for space in c.SPACES:
                g=row['gamma'][space]
                r,d,w=np.array(g['r']),np.array(g['d']),self.weights[space]
                expected=float(np.dot((r+d)**2,w)-np.dot(r**2,w))
                self.assertAlmostEqual(g['gamma'],expected,delta=1e-8*max(1,abs(expected)))
                errors=row['arms']
                self.assertAlmostEqual(g['gamma'],errors['R'][space]-errors['N'][space],
                                       delta=1e-8*max(1,abs(expected)))
        self.assertEqual({r['chunks'] for r in self.summary['per_task']},{1,2,4,8})
        self.assertEqual(len(self.summary['per_task']),self.args.n_tasks*3*4*4)
        for row in self.summary['per_task']:
            self.assertIn('realized_displacement_ratio',row['pulse'])
            self.assertGreaterEqual(row['pulse']['clip_fraction'],0.)
        for cell in self.summary['aggregate']:
            for space in c.SPACES:
                self.assertEqual(sum(cell['spaces'][space]['kind_counts'].values()),
                                 len(self.summary['evaluation_ids']))
                self.assertEqual(cell['spaces'][space]['cross']['n_pairs'],
                                 len(self.summary['evaluation_ids']))
                self.assertEqual(cell['spaces'][space]['d_squared_W']['n_pairs'],
                                 len(self.summary['evaluation_ids']))

    def test_mechanism_uses_same_realized_pulse_not_equal_normalized_dose(self):
        for ti in range(self.args.n_tasks):
            sigma0=c.npz(self.out/'sigma0'/f'task_{ti:03d}'/'mechanism_shared_pulse/pulse.npz')
            sigma1=c.npz(self.out/'sigma1.0'/f'task_{ti:03d}'/'mechanism_shared_pulse/pulse.npz')
            np.testing.assert_array_equal(sigma0['requested_pulse'],sigma1['requested_pulse'])
            np.testing.assert_allclose(sigma0['realized_pulse'],sigma1['realized_pulse'],rtol=0,atol=1e-15)
            for flow in (sigma0,sigma1):
                np.testing.assert_allclose(flow['realized_pulse'],flow['requested_pulse'],rtol=0,atol=1e-15)
        self.assertIn('不得單獨歸因',self.summary['mechanism_caveat'])
        self.assertEqual(len(self.summary['mechanism_pairs']),self.args.n_tasks*4)

    def test_full_streams_completion_anchors_and_no_smoke_verdict(self):
        s=self.summary
        self.assertEqual((s['n_tasks'],s['n_arms'],s['n_doses'],s['n_codecs']), (self.args.n_tasks,3,3,3))
        total=0
        for path in self.out.rglob('*.npz'):
            if 'synthetic_b1_source' in path.parts:
                continue
            stream=c.npz(path)
            if 'action' not in stream:
                continue
            n=len(stream['action'])
            total+=n
            for key,shape in [('qpos',(n+1,15)),('qvel',(n+1,14)),('obs',(n+1,29))]:
                self.assertEqual(stream[key].shape,shape,str(path))
                self.assertTrue(np.isfinite(stream[key]).all())
            self.assertTrue((np.abs(stream['action'])<=1).all())
        self.assertEqual(s['n_steps_total'],total)
        self.assertEqual(s['status'],'complete')
        self.assertEqual(s['primary_verdict_cell'],dict(codec='p0_dict_v1',dose='1',chunks=4,space='gait'))
        self.assertIsNone(s['primary_verdict'])
        self.assertTrue((self.out/s['dose_response_plot']).is_file())
        for cell in s['aggregate']:
            for space in c.SPACES:
                self.assertIsNone(cell['spaces'][space]['primary_verdict'])
                self.assertIsNone(cell['spaces'][space]['feedback_corrects'])
        self.assertEqual(s['preregistered']['status'],'preregistered')
        self.assertEqual(s['preregistered']['error_reduction_gte'],.20)


class ContractTests(unittest.TestCase):
    def test_snapshot_and_findmnt_guard_are_copied_verbatim(self):
        def definitions(path):
            text=path.read_text()
            return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body
                    if isinstance(node,ast.FunctionDef)}
        old,new=definitions(c.FIRSTCUTS/'b1_triarm/run_b1.py'),definitions(c.HERE/'run_c2.py')
        for name in ('wrappers','rng_state','restore_rng','save_state','restore_state',
                     'save_snapshot','load_snapshot','local_data'):
            self.assertEqual(old[name],new[name])

    def test_readme_first_paragraph_verbatim(self):
        work=(c.FIRSTCUTS/'_workorders/WORK-C2-twin.md').read_text().split('\n\n')[1]
        self.assertEqual((c.HERE/'README.md').read_text().split('\n\n')[0],work)

    def test_findmnt_rejects_network_dataset(self):
        with tempfile.TemporaryDirectory(dir=c.HERE) as tmp:
            (Path(tmp)/(c.relay.DATASET+'-val.npz')).touch()
            for fs in ('nfs','nfs4','cifs','smb3','fuse.sshfs'):
                with patch.object(c.subprocess,'check_output',return_value=fs):
                    with self.assertRaisesRegex(ValueError,'network datasets forbidden'):
                        c.local_data(tmp)

    def test_so3_roundtrip_and_quaternion_double_cover(self):
        env=c.wv.make_env('/tmp',c.relay.DATASET)
        try:
            o=c.state_obs(env)
            d=np.random.default_rng(9).normal(0,.03,28)
            changed=c.inject_innovation(env.unwrapped.model,o,d)
            np.testing.assert_allclose(c.innovation(env.unwrapped.model,o,changed),d,atol=1e-14)
            same=o.copy()
            same[3:7]*=-1
            np.testing.assert_allclose(c.innovation(env.unwrapped.model,o,same),0,atol=1e-14)
        finally:
            env.close()

    def test_no_fall_tradeoff_or_xy_primary_and_paired_statistics(self):
        rows=[]
        for codec in c.CODEC_LABELS:
            for dose,_ in c.CONDITIONS:
                for chunks in (1,2,4,8):
                    for ti in range(2):
                        rows.append(dict(codec=codec,dose=dose,chunks=chunks,task_id=ti,
                            arms={a:dict(gait=1. if a=='R' else 2.,xy=1.,yaw=1.,
                                  fell=a=='R',goal_progress=0.) for a in ('R','N','M')},
                            gamma={s:dict(gamma=-1.,cross=-2.,d_squared_W=1.,
                                          kind='true_correction') for s in c.SPACES}))
        out=c.aggregate(rows,[0,1],True)
        for cell in out:
            self.assertIsNone(cell['spaces']['xy']['primary_verdict'])
            if cell['verdict_role']=='primary':
                self.assertFalse(cell['spaces']['gait']['primary_verdict'])
            else:
                self.assertIsNone(cell['spaces']['gait']['primary_verdict'])
            self.assertEqual(cell['spaces']['gait']['gamma']['n_pairs'],2)
        stat=c.bootstrap([1.,1.])
        self.assertEqual(stat['ci95'],[1.,1.])

    def test_yaw_branch_cut_has_common_chart(self):
        def stream(angle):
            q=np.zeros((5,15))
            q[:,3]=np.cos(angle/2)
            q[:,6]=np.sin(angle/2)
            return dict(qpos=q,shape=np.zeros((5,1,3)),action=np.ones((4,8))*angle)
        ref=stream(np.pi-.02)
        n=stream(-np.pi+.03)
        r=stream(-np.pi+.01)
        w=dict(gait=np.ones(3),xy=np.ones(2),yaw=np.ones(1))
        g=c.gamma_ledger(ref,n,r,w,4)['yaw']
        self.assertAlmostEqual(float(g['r'][0]),.05)
        self.assertAlmostEqual(float(g['d'][0]),-.02)
        self.assertAlmostEqual(g['gamma'],.03**2-.05**2)


def main():
    args=c.parser().parse_args()
    if args.main or args.n_tasks>4:
        raise SystemExit('Wiring test forbids main runs and more than four tasks')
    WiringTests.args=args
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(WiringTests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    s=WiringTests.summary
    label='SYNTHETIC' if args.synthetic else 'OFFICIAL_WIRING'
    report=(f'C2 {label} PASS tests={result.testsRun} n_tasks={s["n_tasks"]} n_arms={s["n_arms"]} '
        f'n_doses={s["n_doses"]} n_auxiliary_doses=1 n_codecs={s["n_codecs"]} n_steps_total={s["n_steps_total"]}\n'
        f'C2 ZERO PASS max_R_N_state_linf={max(r["R_N_state_linf"] for r in s["zero_checks"]):g} '
        f'max_nominal_replay_state_linf={max(r["nominal_replay_state_linf"] for r in s["zero_checks"]):g}\n'
        f'C2 MUTANT CAUGHT cells={len(s["mutant_checks"])} '
        f'max_collapse_ratio={max(r["collapse_ratio"] for r in s["mutant_checks"]):g}\n'
        f'C2 W FROZEN calibration_tasks={len(s["calibration_ids"])} evaluation_tasks={len(s["evaluation_ids"])} '
        'official_40_160=NOT_RUN\n'
        'C2 MAIN NOT_RUN scientific_verdict=NONE\n'
        'C2 B DESIGN_FROZEN NOT_IMPLEMENTED')
    if args.synthetic:
        report+='\nC2 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true'
    (args.out_dir/'wiring_result.txt').write_text(report+'\n')
    print(report)


if __name__=='__main__':
    main()
