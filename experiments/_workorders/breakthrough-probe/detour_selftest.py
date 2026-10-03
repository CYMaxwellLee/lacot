"""CPU factory equivalence, mock protocol tests and short real-model wiring; no GPU evidence."""
import contextlib
import ast
import copy
import inspect
import io
import json
import os
# Self-contained CPU entry; fix determinism before importing torch/M9.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import common
import harness_detour as d
import harvest_detour as h
import detour_plan as plans
import harness_move1 as m1
from artifacts import save_row, resume_rows, load_artifact


class Space:
    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)


class ScriptEnv:
    """Position follows an explicit script; actions are recorded, never used to fake model effects."""
    def __init__(self, script=None, goal=None, success_at=None, dtype=np.float32):
        self.dtype = dtype
        self.unwrapped = self
        self.action_space = Space()
        self.maze_map = np.asarray(d.checked_json(d.HERE/'builder.json')['maze'])
        self._maze_unit, self._offset_x, self._offset_y = 4.,4.,4.
        self._success_timing, self._add_noise_to_goal, self._goal_tol = 'post',True,1.
        self.script, self.fixed_goal, self.success_at = script,goal,success_at
        self.actions = []
    def ij_to_xy(self, c):
        return (4.*c[1]-4.,4.*c[0]-4.)
    def xy_to_ij(self, xy):
        return (int((xy[1]+6.)/4.),int((xy[0]+6.)/4.))
    def reset(self, seed, options):
        self.np_random = np.random.default_rng(seed)
        self.t, self.actions = 0,[]
        t = d.load_trapset()['tasks'][str(options['task_id'])]
        self.points = np.asarray(self.script if self.script is not None else [self.ij_to_xy(t['s'])]*1001,self.dtype)
        goal = self.fixed_goal if self.fixed_goal is not None else self.ij_to_xy(t['g'])
        return self.points[0].copy(),dict(goal=np.asarray(goal,self.dtype))
    def step(self, action):
        self.actions.append(action.copy()); self.t += 1
        return self.points[self.t].copy(),0.,False,self.t==len(self.points)-1,dict(success=self.t==self.success_at)
    def close(self):
        pass


def mock_module():
    seen = []; encode_inputs = []; cond_inputs = []; flow_count = []
    def encode(points):
        encode_inputs.append(points.detach().cpu().numpy().copy())
        u = torch.zeros((1,8,256),dtype=torch.float32)
        u.reshape(-1)[:256] = points.reshape(-1)
        return u
    def head(cond,u):
        digest = common.digest_array(u.detach().cpu().numpy()); seen.append(digest)
        # u AND observations affect actions; different delivered u is observable.
        values = torch.tensor([int(digest[:6],16)/0xffffff-.5,int(digest[6:12],16)/0xffffff-.5])
        return (values + cond[0,:2]*.001)[None,None,:].repeat(1,4,1)
    def sample(*args):
        flow_count.append(1)
        return torch.rand((1,8,256))
    def cond(s,g):
        cond_inputs.append((s.clone(),g.clone()))
        return torch.cat([s,g],dim=1)
    flow=SimpleNamespace(sample=sample)
    return SimpleNamespace(torch=torch,OBS_DIM=2,CHUNK=4,T_CAP=128,K=8,D_MODEL=256,
        MU_XY=np.zeros(2),SD_XY=np.ones(2),MU_XY_T=torch.zeros(2),SD_XY_T=torch.ones(2),device='cpu',
        encode_u=encode,_dec=lambda u,s:u.reshape(-1)[:256].reshape(1,128,2),
        ahead=head,_q=lambda u:u,normstate=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).reshape(1,2),
        goal_to_obs=lambda x:x,condvec=cond,flow=flow,sample_plan=lambda *a:flow.sample(*a),_GRAD_CACHE={'u':None},
        _reseed_shuf=lambda n:None,_RDIR=[1.],detour_mock=True,
        probe_provenance=dict(mock=True,device='cpu',deterministic_settings={'mock':True}),
        head_seen=seen,encode_inputs=encode_inputs,cond_inputs=cond_inputs,flow_count=flow_count)



class DetourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.module = mock_module()
        cls.env = ScriptEnv()
        cls.table = d.build_e1_table(cls.module,env=cls.env)
        cls.plan = plans.build_plan()
        cls.traps,cls.builder,cls.walls = d.geometry_context(cls.env)

    def run_row(self, arm='O3', env=None, plan=None):
        module = mock_module(); module.detour_e1_table = self.table
        env = env or ScriptEnv(script=[(28.,8.)]*10)
        plan = plan or dict(protocol=d.PROTOCOL,task=4,episode=4,draw=80,arm=arm,group='main')
        row = d.run_draw_detour(module,env,plan,arm)
        return row,module,env,plan

    def validate(self, row, env, plan):
        h.validate_row(row,plan,self.table,env.xy_to_ij,env.ij_to_xy,fixture=True)

    def killer(self, name, operation, exception, message):
        with self.assertRaisesRegex(exception, message):
            operation()
        print('TWO_STATE '+name+': correct=PASS; killer/mutant=FAIL (observed rejection)')

    def test_01_pins_flags_seeds(self):
        d.verify_pins()
        self.assertEqual(d.read_flags(),dict(PRODUCTION_READY=False,SMOKE_ENABLED=False))
        for smoke in (False,True):
            self.killer('release-'+str(smoke),lambda:d.release_check(smoke),common.Blocked, 'SMOKE_ENABLED=False' if smoke else 'PRODUCTION_READY=False')
        for ready in (False,True):
            for enabled in (False,True):
                with patch.object(d,'read_flags',return_value=dict(PRODUCTION_READY=ready,SMOKE_ENABLED=enabled)):
                    for smoke in (False,True):
                        if enabled if smoke else ready:
                            d.release_check(smoke)
                        else:
                            with self.assertRaisesRegex(common.Blocked,'SMOKE_ENABLED=False' if smoke else 'PRODUCTION_READY=False'): d.release_check(smoke)
        for task in range(1,6):
            for draw in range(80,84):
                self.assertEqual(d.seeds_detour(task,4,draw)['stream_seed'],7*task+4+1000003*draw)
                with self.assertRaisesRegex(ValueError,r'^draw outside 64\.\.79$'): common.seeds(task,4,draw)
        for draw in (79,84,80.,True):
            with self.assertRaisesRegex(ValueError,r'^draw outside 80\.\.83$'): d.seeds_detour(4,4,draw)
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'HERE',Path(tmp)):
            for value in ({},{'SMOKE_ENABLED':0,'PRODUCTION_READY':False}):
                (Path(tmp)/'detour-flags.json').write_text(json.dumps(value))
                with self.assertRaisesRegex(common.Blocked,r'^BLOCKED: detour-flags\.json requires exactly two boolean release flags$'): d.read_flags()

    def test_02_plan_machine_only_and_inventory(self):
        self.assertEqual(self.plan,d.checked_json(d.HERE/'detour-plan.json'))
        self.assertEqual(len(self.plan['formal']),408)
        self.assertEqual(len(self.plan['smoke']),15)
        self.assertEqual(len({h.key(p) for p in self.plan['formal']}),408)
        self.assertEqual({p['task'] for p in self.plan['smoke']},{1,2,3,4,5})
        self.assertEqual({p['draw'] for p in self.plan['formal'] if p['group']=='gate'},{80,81})
        bad=copy.deepcopy(self.plan); bad['formal'].pop()
        self.killer('plan-inventory',lambda:plans.validate_plan(bad),ValueError, r'^plan differs from frozen detour-plan\.json$')

    def test_03_table_all_cells_and_roundtrip(self):
        h.validate_table(self.table,fixture=True)
        self.assertEqual(sum(map(len,self.table['of_cells'].values())),225)
        for task,t in self.traps['tasks'].items():
            expected={k for k,v in t['oracle_table'].items() if len(v['route'])>4}
            self.assertEqual(set(self.table['cells'][task]),expected)
            self.assertTrue(all(c['qualification']=='PASS' for c in self.table['cells'][task].values()))
        self.killer('mock-production-reject',lambda:h.validate_table(self.table),h.Rejected, r'^mock E1 is not production evidence$')
        bad=copy.deepcopy(self.table); bad['cells']['4']['5,4']=next(iter(bad['cells']['4'].values()))
        self.killer('tail-in-static-table',lambda:h.validate_table(bad,fixture=True),h.Rejected, r'^E1 missing/extra static cells \(w=g excluded\)$')
        self.assertEqual(self.table['cpu_comparison']['compared'],sum(map(len,self.table['cells'].values())))

    def test_04_exact_segment_square(self):
        for a,b,want in [((-3,0),(3,0),0),((2,2),(3,3),0),((0,0),(0,0),0),
                         ((4,0),(4,0),2),((3,3),(3,3),np.sqrt(2)),((-4,3),(4,3),1)]:
            self.assertAlmostEqual(d.segment_box_distance(a,b,(0,0)),want)
            self.assertAlmostEqual(d.clearance([a,b],[(0,0)]),want)
        gen=np.random.default_rng(91)
        for _ in range(100):
            a,b,c=gen.normal(size=(3,2))*5
            self.assertAlmostEqual(d.segment_box_distance(a,b,c),d.clearance([a,b],[c]),places=12)

    def test_05_all_checker_killers_and_boundaries(self):
        controls=self.table['controls']
        self.assertEqual(controls['positive_pass'],225)
        for row in controls['negatives']:
            for kind in ('final_goal','reverse'):
                self.assertEqual(row[kind]['qualification'],'FAIL')
        self.assertEqual(controls['boundaries']['-1e-07']['qualification'],'FAIL')
        self.assertEqual(controls['boundaries']['1e-07']['qualification'],'PASS')
        self.assertTrue(all(v['qualification']=='FAIL' for v in controls['zeros']))
        self.assertEqual(len(controls['goal_positives']),5)
        self.assertTrue(all(v['qualification']=='PASS' for v in controls['goal_positives']))
        self.assertTrue(controls['known_limitations'])
        for v in controls['known_limitations']:
            self.assertEqual(v['qualification'],'PASS');self.assertLess(v['clearance'],.7)
        print('TWO_STATE E1-final-g: true-tau=PASS; all departure-cell final-g straight lines=FAIL')
        print('TWO_STATE E1-reverse: true-tau=PASS; all departure-cell reversed first directions=FAIL')
        print('TWO_STATE E1-boundary: p10+1e-7=PASS; p10-1e-7=FAIL')
        print('TWO_STATE E1-zero: nonzero=PASS; D/P/G zero=FAIL')
        # A checker that blindly accepts must be killed by its own controls.
        actual=d.direction_metrics
        def always(*args): return dict(actual(*args),qualification='PASS')
        with patch.object(d,'direction_metrics',always):
            self.killer('E1-always-pass',lambda:d.checker_controls(self.traps,self.builder,self.env.ij_to_xy,self.walls),common.Blocked, r'^BLOCKED: E1 direction controls failed$')

    def test_06_sampled_clearance_mutant(self):
        def sampled(points,walls):
            points=np.asarray(points)[:,None,:]; centers=np.asarray(walls)[None,:,:]
            return float(np.linalg.norm(np.maximum(np.abs(points-centers)-2.,0.),axis=2).min())
        tau=d.resample(d.oracle_points(self.traps,4,(2,6),self.env.ij_to_xy))
        bad=tau.copy(); bad[7:9]=[[18.528416,6.459119],[18.459119,6.528416]]
        self.assertEqual(d.e1_check(bad,tau,self.walls)['qualification'],'FAIL')
        with patch.object(d,'clearance',sampled):
            self.assertEqual(d.e1_check(bad,tau,self.walls)['qualification'],'PASS')
            # Legacy clearance stays descriptive; the direction gate is independent.
            self.assertEqual(d.direction_check(bad,d.oracle_points(self.traps,4,(2,6),self.env.ij_to_xy)[:4],self.env.ij_to_xy((5,4)))['qualification'],'PASS')

    def test_07_unordered_frechet_mutant(self):
        a=np.asarray([[0.,0.],[4.,0.],[4.,4.]])
        self.assertGreater(d.discrete_frechet(a,a[::-1]),0)
        self.assertEqual(d.discrete_frechet(a,a),0)

    def test_08_provider_head_and_noise_forward(self):
        row,module,env,plan=self.run_row()
        self.validate(row,env,plan)
        self.assertEqual(module.head_seen,[c['delivered_sha256'] for c in row['chunks']])
        self.assertEqual(len(module.flow_count),0)
        self.assertEqual(row['flow_calls'],[])
        self.assertEqual(len(module.cond_inputs),len(row['chunks']))
        self.assertFalse(np.array_equal(row['actions'],np.clip(row['head_actions'],-1,1)))
        self.assertEqual(row['rng_before']['noise'],torch.Generator().manual_seed(row['seeds']['stream_seed']).get_state().tolist())
        u1=m1.encode_trajectory(module,[[0,0],[4,0]])
        u2=m1.encode_trajectory(module,[[0,0],[0,4]])
        self.assertFalse(torch.equal(module.ahead(torch.zeros(1,4),u1),module.ahead(torch.zeros(1,4),u2)))

    def test_09_goal_zero_wall_script(self):
        script=[(28.,8.)]*4+[(12.,15.2)]*4+[(12.,16.8)]*4+[(14.1,-1.9)]*4+[(17.9,1.9)]*5
        env=ScriptEnv(script,goal=(12.,16.8))
        row,module,env,plan=self.run_row(env=env)
        self.validate(row,env,plan)
        self.assertEqual([c['mode'] for c in row['chunks']],['static','goal','零長度沿用','static','static'])
        self.assertEqual(row['chunks'][1]['u_sha256'],row['chunks'][2]['u_sha256'])
        self.assertEqual(row['chunks'][1]['qualification'],row['chunks'][2]['qualification'])
        self.assertEqual(row['chunks'][3]['c_raw'],[1,5]); self.assertEqual(row['chunks'][4]['c_raw'],[1,5])
        self.assertEqual(row['chunks'][3]['c_eff'],[1,4]); self.assertEqual(row['chunks'][4]['c_eff'],[1,6])
        self.assertNotEqual(row['chunks'][3]['u_sha256'],row['chunks'][4]['u_sha256'])
        provider=d.Provider(module,env,self.table)
        provider.get(4,np.array([12.,15.2]),np.array([12.,16.8]))
        provider.get(4,np.array([12.,15.4]),np.array([12.,16.8]))
        self.assertEqual(provider.cache,{})
        self.assertEqual(d.effective_cell(env,np.array([16.,0.]),d.free_cells(self.traps,4))[1],(1,4))

    def test_10_zero_inherits_failure_and_no_previous(self):
        module=mock_module(); env=ScriptEnv(); provider=d.Provider(module,env,self.table)
        with patch.object(module,'_dec',return_value=torch.tensor([[[12.,15.2]]*128])):
            _,prev=provider.get(4,np.array([12.,15.2]),np.array([12.,16.8]))
        self.assertEqual(prev['qualification'],'FAIL')
        _,now=provider.get(4,np.array([12.,16.8]),np.array([12.,16.8]))
        self.assertEqual(now['qualification'],'FAIL')
        self.assertEqual(now['u_sha256'],prev['u_sha256'])
        self.killer('zero-no-previous',lambda:d.Provider(module,env,self.table).get(4,np.array([12.,16.8]),np.array([12.,16.8])),common.Blocked, r'^BLOCKED: zero-length goal has no previous chunk u$')

    def test_11_provider_flow_mutant(self):
        actual=d.Provider.get
        def flow(self,*args):
            _,record=actual(self,*args)
            return self.module.sample_plan(1,None,None),record
        with patch.object(d.Provider,'get',flow):
            self.killer('provider-flow-E0',lambda:self.run_row(),ValueError, r'^E0: actual head delivery differs from expected$')

    def test_12_noise_removed_mutant(self):
        with patch.object(d.m3,'noisy_action',lambda collector,a:np.clip(a,-1,1).astype(np.float32)):
            row,_,env,plan=self.run_row()
        self.killer('noise-removed',lambda:self.validate(row,env,plan),h.Rejected, r'^noise action sequence mismatch$')
        # Old harvest guard is deliberately reintroduced; the acceptance assertion must fail.
        original=h.validate_noise
        def old_guard(row):
            if row['seeds']['flow_seed'] is not None:
                original(row)
        def must_reject():
            with self.assertRaisesRegex(h.Rejected,r'^noise action sequence mismatch$'): self.validate(row,env,plan)
        with patch.object(h,'validate_noise',old_guard):
            self.killer('flow-seed-conditional-validator',must_reject,AssertionError, r'Rejected not raised')

    def test_13_raw_cache_mutant(self):
        source=textwrap.dedent(inspect.getsource(d.Provider.get)).replace('key = task, eff','key = task, raw')
        namespace=dict(vars(d)); exec(compile(source,'<M3-cache-mutant>','exec'),namespace)
        script=[(14.1,-1.9)]*4+[(17.9,1.9)]*5
        with patch.object(d.Provider,'get',namespace['get']):
            self.killer('M3-c_raw-cache',lambda:self.run_row(env=ScriptEnv(script)),ValueError, r'^E0: cache source differs from frozen E1 table$')

    def test_14_targets_flow_and_quantization(self):
        for arm in ('O3','F3','OF','Q-C','Q-O3','Q-F3'):
            row,module,env,plan=self.run_row(arm,ScriptEnv([(28.,8.)]*4+[(20.,8.)]*5))
            self.validate(row,env,plan)
            self.assertEqual(len(module.flow_count),len(row['chunks']) if arm in d.FLOW_ARMS else 0)
            for chunk,(_,g) in zip(row['chunks'],module.cond_inputs):
                np.testing.assert_array_equal(g.numpy()[0],np.asarray(chunk['target_xy'],np.float32))
            bad=copy.deepcopy(row);bad['chunks'][0]['cond_target_xy']=[-999.,-999.]
            if arm in d.FLOW_ARMS:bad['flow_calls'][0]['target_xy']=[-999.,-999.]
            self.killer('target-'+arm,lambda:self.validate(bad,env,plan),h.Rejected, r'^E0 supplied w/cond target$')
            bad=copy.deepcopy(row);bad['protocol']='unknown';bad_plan=dict(plan,protocol='unknown')
            self.killer('unknown-protocol-harvest-'+arm,lambda:self.validate(bad,env,bad_plan),h.Rejected, r'^unknown protocol$')
            self.killer('unknown-protocol-worker-'+arm,lambda:d.run_draw_detour(module,env,bad_plan,arm),ValueError, r'^unknown detour protocol$')
        with patch.object(d,'decode',wraps=d.decode) as decoder:
            row,module,env,plan=self.run_row('F3',ScriptEnv([(28.123456789,8.012345678)]*9,dtype=np.float64))
        self.validate(row,env,plan)
        self.assertEqual(decoder.call_count,len(row['chunks']))
        for call,chunk in zip(decoder.call_args_list,row['chunks']):
            np.testing.assert_array_equal(call.args[2],chunk['xy'])
        bad=copy.deepcopy(row);bad['chunks'][0]['target_xy']=row['goal'];bad['chunks'][0]['cond_target_xy']=row['goal']
        bad['flow_calls'][0]['target_xy']=row['goal']
        self.killer('F3-final-g-target',lambda:self.validate(bad,env,plan),h.Rejected, r'^E0 supplied w/cond target$')
        # Mutate the real worker condition call, not merely a serialized field.
        source=textwrap.dedent(inspect.getsource(d._run_draw_detour))
        needle='recorded_condition(module,obs,target)'
        self.assertIn(needle,source)
        namespace=dict(vars(d));exec(source.replace(needle,'recorded_condition(module,obs,goal)'),namespace)
        with patch.object(d,'_run_draw_detour',namespace['_run_draw_detour']):
            wrong=d.run_draw_detour(module,env,plan,'F3')
        self.assertEqual(wrong['chunks'][0]['cond_target_xy'],wrong['goal'])
        self.killer('F3-worker-final-g-condition',lambda:self.validate(wrong,env,plan),h.Rejected, r'^E0 supplied w/cond target$')
        bad=copy.deepcopy(row);bad['flow_calls'].pop()
        self.killer('F3-call-count',lambda:self.validate(bad,env,plan),h.Rejected, r'^E0 actual flow\.sample calls$')
        # Actual sample_plan bypassing flow.sample must fail at the worker.
        with patch.object(module,'sample_plan',return_value=torch.zeros((1,8,256))):
            self.killer('F3-fake-sampling',lambda:d.run_draw_detour(module,env,plan,'F3'),ValueError, r'^E0: actual flow\.sample count$')
        row['chunks'][0]['goal_distance']=1000.
        self.validate(row,env,plan)  # F3 endpoint is never a gate.
        row,module,env,plan=self.run_row('O3')
        bad=copy.deepcopy(row);bad['chunks'][0]['tau_endpoint']=[0.,0.]
        self.killer('O3-encoding-endpoint',lambda:self.validate(bad,env,plan),h.Rejected, r'^encoding input endpoint$')
        with patch.object(module,'_q',lambda u:u+1):
            self.killer('head-must-equal-frozen-quantization',lambda:d.run_draw_detour(module,env,plan,'O3'),ValueError, r'^E0: provider differs from frozen E1 table$')
        self.killer('unknown-arm',lambda:d.run_draw_detour(module,env,dict(plan,arm='Z'),'Z'),ValueError, r'^unknown/mismatched detour arm$')

    def test_15_reject_artifact_matrix(self):
        row,_,env,plan=self.run_row()
        h.validate_rows([row],[plan],self.table,env.xy_to_ij,env.ij_to_xy,True)
        for name,rows in [('missing',[]),('duplicate',[row,row])]:
            self.killer(name,lambda:h.validate_rows(rows,[plan],self.table,env.xy_to_ij,env.ij_to_xy,True),h.Rejected, r'^missing rows$' if name=='missing' else r'^duplicate row$')
        changes=[('seed',lambda r:r['seeds'].__setitem__('noise_seed',0)),
                 ('draw',lambda r:r.__setitem__('draw',79)),
                 ('u-self-certify',lambda r:r['chunks'][0].update(u_sha256='a'*64,delivered_sha256='a'*64,expected_delivered_sha256='a'*64)),
                 ('missing-qualification',lambda r:r['chunks'][0].pop('qualification')),
                 ('noise-state',lambda r:r['rng_after'].__setitem__('noise',r['rng_before']['noise'])),
                 ('missing-chunk',lambda r:r['chunks'].pop()),
                 ('false-success',lambda r:r.__setitem__('success',True)),
                 ('early-stop',lambda r:r.update(terminated=False,truncated=False)),
                 ('missing-action',lambda r:r['actions'].pop())]
        messages = dict(seed=r'^seed mismatch$', draw=r'^row identity$',
                        **{'u-self-certify':r'^E0 frozen table u_sha256$',
                           'missing-qualification':r'^missing/malformed row contract:.*qualification',
                           'noise-state':r'^noise final state mismatch$',
                           'missing-chunk':r'^missing/duplicate chunk$',
                           'false-success':r'^success termination timing$',
                           'early-stop':r'^early stop / reason$',
                           'missing-action':r'^noise action evidence shape$'})
        for name,change in changes:
            bad=copy.deepcopy(row); change(bad)
            self.killer(name,lambda:self.validate(bad,env,plan),(h.Rejected,ValueError), messages[name])

    def test_16_receipt_and_exclusive_shards(self):
        row,module,env,plan=self.run_row()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); table_path=root/'e1-table-v8.json'
            common.write_json(table_path,self.table)
            receipt=d.receipt(module,env,self.plan,table_path)
            h.validate_receipt(receipt,self.plan,table_path,fixture=True)
            for key in ('card_sha256','code_sha256','collector_sha256','e1_sha256','plan_sha256'):
                bad=copy.deepcopy(receipt); bad[key]='wrong'
                self.killer('receipt-'+key,lambda:h.validate_receipt(bad,self.plan,table_path,fixture=True),h.Rejected, r'^receipt '+key+'$')
            refs=[save_row(root,row,0)]
            header=dict(storage='shards-v1',rows=[],conformance_rows=[],receipt=receipt)
            common.write_json(root/'checkpoint.json',header)
            _,restored=resume_rows(root)
            self.assertEqual(restored,[json.loads(common.canonical(row))])
            self.killer('exclusive-write',lambda:common.write_json(table_path,self.table),FileExistsError, r'File exists')
            common.write_json(root/'raw.json',dict(header,rows=refs))
            self.assertEqual(load_artifact(root/'raw.json')['rows'],restored)
            (root/'shards/0000.json').write_text('{}')
            self.killer('shard-tamper',lambda:resume_rows(root),ValueError, r'^artifact file SHA mismatch$')

    def test_17_question_layers_thresholds(self):
        self.assertTrue(d.question_success([True,True,False,False]))
        self.assertFalse(d.question_success([True,False,False,False]))
        for n,win,lose in [(14,10,2),(6,4,1),(10,7,1)]:
            self.assertEqual(d.layer_result(n,win),'救回'); self.assertEqual(d.layer_result(n,win-1),'部分')
            self.assertEqual(d.layer_result(n,lose),'未救回'); self.assertEqual(d.layer_result(n,lose+1),'部分')
        for n in (0,1,2): self.assertEqual(d.layer_result(n,n),'觀測不足')
        for c in (0,7): self.assertEqual(d.q_result(16,16,c)['label'],'Q 觀測不足')
        self.assertEqual(d.q_result(6,6,8)['statements']['O3'],'本預算下沒有比 Q-C 少超過 2 次')
        self.assertEqual(d.q_result(5,5,8)['annotations'],['格心短計畫在易題也變差','flow＋近目標在易題也變差'])
        self.assertEqual(d.q_result(0,0,7)['annotations'],['機制檢查無參考線'])

    def test_18_failure_partition_norm_and_compliance(self):
        row,_,env,_=self.run_row()
        rows=[]
        for cell,qual in [((3,4),'PASS'),((3,4),'FAIL'),((7,10),'PASS')]:
            r=copy.deepcopy(row);r['xy'][-1]=env.ij_to_xy(cell)
            r['chunks'][0]['qualification']=qual;rows.append(r)
        result=d.failure_types(rows,self.traps['tasks']['4']['traps_g'],env.xy_to_ij)
        self.assertEqual(result['counts'],dict(i=1,ii=1,iii=1)); self.assertEqual(result['denominator'],3)
        for counts,want in [(dict(i=4,ii=0,iii=1),'停在陷阱（首方向合格）'),(dict(i=0,ii=4,iii=1),'近陷阱但首方向不合格'),
                            (dict(i=0,ii=0,iii=0),'N/A')]:
            self.assertEqual(d.failure_label(counts),want)
        for args in [([0,0],[1,0],[0,1]),([1,0],[0,0],[0,1]),([1,0],[0,1],[0,0])]:
            self.assertEqual(d.compliance(*args)['status'],'N/A')
        self.assertTrue(d.compliance([1,0],[1,0],[-1,0])['preferred'])
        rows[0]['xy'][-1]=env.ij_to_xy((2,3))
        cheb=d.failure_types(rows[:1],[[3,4]],env.xy_to_ij)
        manh=d.failure_types(rows[:1],[[3,4]],env.xy_to_ij,'manhattan')
        self.assertNotEqual(cheb['label'],manh['label'])

    def test_19_exact_verdicts_and_exclusivity(self):
        expected={}
        for line in d.CARD.read_text().splitlines():
            if line.startswith('| R'):
                parts=[x.strip() for x in line.split('|')[1:-1]]
                expected[parts[0]]=parts[-1].replace('**','')
        self.assertEqual(d.RULES,expected)
        labels=('救回','未救回','部分','觀測不足')
        for o in labels:
            for f in labels:
                rule={('救回','救回'):'R1',('救回','未救回'):'R2',('未救回','救回'):'R3',('未救回','未救回'):'R4'}.get((o,f),'R5')
                result=d.conclusion(o,f,d.q_result(0,0,8))
                self.assertEqual(result['rule'],rule)
                self.assertEqual(result['statement'],expected[rule])
                self.assertEqual(result['annotations'],['格心短計畫在易題也變差','flow＋近目標在易題也變差'])
        original=d.RULES['R3']
        with patch.dict(d.RULES,R3=original+'病因已定位'):
            self.killer('exact-R3-wording',lambda:self.assertEqual(d.conclusion('未救回','救回',{})['statement'],expected['R3']),AssertionError,r'病因已定位')

    def test_20_full_horizon_fifteen_mock_rollouts(self):
        rows=[]
        for plan in self.plan['smoke']:
            module=mock_module();module.detour_e1_table=self.table;env=ScriptEnv()
            row=d.run_draw_detour(module,env,plan,plan['arm']);rows.append(row)
            self.assertEqual(row['observed_steps'],1000)
            self.assertEqual(len(row['chunks']),250)
            self.assertEqual(len(module.flow_count),250 if plan['arm'] in d.FLOW_ARMS else 0)
        h.validate_rows(rows,self.plan['smoke'],self.table,env.xy_to_ij,env.ij_to_xy,True)
        result=h.readout(rows,self.plan,self.table,env.xy_to_ij,env.ij_to_xy,True)
        self.assertEqual(set(result['layers']),{'4','5','2'})
        self.assertTrue(all(a['label']=='觀測不足' for l in result['layers'].values() for a in l['arms'].values()))
        self.assertEqual(result['Q']['label'],'Q 未跑')
        print('MOCK_FIFTEEN_PASS: 15 x H=1000; real detour collector factory, no factory patch; NOT production smoke')

    def test_21_original_factory_rejects_and_detour_passes(self):
        obs, goal = np.zeros(2,np.float32), np.ones(2,np.float32)
        # This unchanged rejection is why the lead approved a copied factory.
        with self.assertRaisesRegex(ValueError,'draw outside 64..79'):
            d.m3.make_collector(4,4,80,obs,goal)
        with self.assertRaisesRegex(ValueError,r'^draw outside 64\.\.79$'): common.seeds(4,4,80)
        for draw in range(80,84):
            col = d.make_collector_detour(4,4,draw,obs,goal)
            self.assertEqual(col.flow_seed,d.seeds_detour(4,4,draw)['flow_seed'])
        d.collector_compatibility_check()
        row,module,env,plan = self.run_row('Q-C')
        self.validate(row,env,plan)
        self.assertEqual(len(row['flow_calls']),len(row['chunks']))
        self.assertEqual(len(module.flow_count),len(row['chunks']))
        print('FACTORY_RESOLVED: original draw80 rejects; copied factory draw80..83 and real C path PASS')


    @contextlib.contextmanager
    def coordinator_fixture(self, root):
        """All production gate patches are visible here; no fixture switch in the CLI."""
        module=mock_module()
        env=ScriptEnv()
        module.ogbench=SimpleNamespace(make_env_and_datasets=lambda *a,**kw:env)
        original_table, original_receipt = h.validate_table,h.validate_receipt
        original_row, original_rows = h.validate_row,h.validate_rows
        def table(value,runtime=None,fixture=False): return original_table(value,runtime,fixture=True)
        def receipt(*a,**kw): kw['fixture']=True; return original_receipt(*a,**kw)
        def row(*a,**kw): kw['fixture']=True; return original_row(*a,**kw)
        def rows(*a,**kw): kw['fixture']=True; return original_rows(*a,**kw)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(d,'OUTPUT_ROOT',root))
            stack.enter_context(patch('socket.gethostname',return_value='jasmine'))
            stack.enter_context(patch.object(d,'read_flags',return_value=dict(PRODUCTION_READY=True,SMOKE_ENABLED=True)))
            stack.enter_context(patch.object(d,'require_real_module'))
            stack.enter_context(patch.object(d.runtime,'load_frozen',return_value=module))
            stack.enter_context(patch.object(h,'validate_table',side_effect=table))
            stack.enter_context(patch.object(h,'validate_receipt',side_effect=receipt))
            stack.enter_context(patch.object(h,'validate_row',side_effect=row))
            stack.enter_context(patch.object(h,'validate_rows',side_effect=rows))
            yield module,env

    def test_22_coordinator_freeze_fifteen_shards_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.coordinator_fixture(root):
                result=d.execute(root/'smoke','mock',smoke=True)
                artifact=load_artifact(root/'smoke/result.json')
                self.assertEqual(len(artifact['rows']),15)
                self.assertEqual(artifact['readout'],result)
                self.assertTrue(all(r['synthetic'] for r in artifact['rows']))
                self.assertEqual(artifact['receipt']['e1_sha256'],common.file_sha(root/'smoke/e1-table-v8.json'))
                with patch.object(d,'run_draw_detour',side_effect=AssertionError('must reuse shards')):
                    resumed=d.execute(root/'formal-r2','mock',smoke=True,resume_from=root/'smoke')
                self.assertEqual(resumed,result)
                timing=d.checked_json(root/'smoke/timing.json')
                self.assertGreater(timing['weighted_formal_seconds'],0)
                self.assertEqual(timing['proposed_time_limit_seconds'],2*timing['weighted_formal_seconds'])
                with self.assertRaisesRegex(common.Blocked,r'^BLOCKED: choose a fresh output directory$'): d.execute(root/'smoke','mock',smoke=True)
        print('MOCK_COORDINATOR_PASS: E1 exclusive freeze -> 15 shards -> independent harvest -> resume without rerun; gates patched in test only')

    def test_23_partial_failure_resume_and_fail_closed_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.coordinator_fixture(root):
                actual=d.run_draw_detour;count=[]
                def interrupted(*args):
                    if len(count)==2: raise RuntimeError('scripted interruption')
                    count.append(1);return actual(*args)
                with patch.object(d,'run_draw_detour',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'scripted interruption'):
                        d.execute(root/'smoke','mock',smoke=True)
                raw=load_artifact(root/'smoke/raw.json')
                self.assertEqual(len(raw['rows']),2)
                self.assertFalse((root/'smoke/result.json').exists())
                with patch.object(d,'run_draw_detour',wraps=actual) as run:
                    d.execute(root/'formal-r2','mock',smoke=True,resume_from=root/'smoke')
                    self.assertEqual(run.call_count,13)
        with patch.object(d,'read_flags',return_value=dict(PRODUCTION_READY=True,SMOKE_ENABLED=True)),patch('socket.gethostname',return_value='local-cpu'),patch.object(d.runtime,'load_frozen') as load:
            with self.assertRaisesRegex(common.Blocked,'require jasmine'):
                d.execute('/archive/cymaxwelllee/breakthrough1/detour-u/smoke','unused',smoke=True)
            load.assert_not_called()

    def assert_factory_equivalent(self, factory):
        """Capture immediately after begin, with the same unrelated initial RNG state."""
        obs = np.array([12.,16.],np.float32)
        goal = np.array([20.,8.],np.float32)
        saved = torch.get_rng_state().clone()
        try:
            for task,episode in ((4,4),(1,10)):
                for draw in (64,71,79):
                    torch.manual_seed(314159)
                    original = d.m3.make_collector(task,episode,draw,obs,goal)
                    original_rng = torch.get_rng_state().numpy().tobytes()
                    torch.manual_seed(314159)
                    copied = factory(common.seeds,task,episode,draw,obs,goal)
                    copied_rng = torch.get_rng_state().numpy().tobytes()
                    for key in ('stream_seed','flow_seed'):
                        self.assertEqual(getattr(original,key).to_bytes(8,'little'),
                                         getattr(copied,key).to_bytes(8,'little'))
                    self.assertEqual(original.noise.get_state().numpy().tobytes(),
                                     copied.noise.get_state().numpy().tobytes())
                    self.assertEqual(original_rng,copied_rng)
                    self.assertEqual((original.arm,original.sigma,original.draws),
                                     (copied.arm,copied.sigma,copied.draws))
        finally:
            torch.set_rng_state(saved)

    def test_25_factory_byte_equivalence_and_line_copy(self):
        self.assert_factory_equivalent(d._make_collector_with)
        old = ast.parse(textwrap.dedent(inspect.getsource(d.m3.make_collector))).body[0]
        new = ast.parse(textwrap.dedent(inspect.getsource(d._make_collector_with))).body[0]
        expected = copy.deepcopy(old)
        # The lead permits precisely this one body change, including the docstring.
        expected.body[8].value.func.id = 'seed_fn'
        self.assertEqual(ast.dump(ast.Module(body=expected.body,type_ignores=[])),
                         ast.dump(ast.Module(body=new.body,type_ignores=[])))
        lines = inspect.getsource(d._make_collector_with).splitlines()
        self.assertEqual(len(lines),13)
        for line,number in zip(lines,range(26,39)):
            self.assertIn(f'# harness_move3.py:{number}',line)
        print('FACTORY_BYTE_EQUIVALENCE_PASS: 2 task/episode pairs x draws 64,71,79; stream/flow, noise bytes, global torch RNG bytes')

    def test_26_factory_equivalence_killers(self):
        source = textwrap.dedent(inspect.getsource(d._make_collector_with))
        for name,old,new,exception,message in (
            ('factory-arm-B',"Collector('A', 0., 16)","Collector('B', 0., 16)",RuntimeError,r'^fresh-flow branch mismatch$'),
            ('factory-sigma',"Collector('A', 0., 16)","Collector('A', .01, 16)",ValueError,r'^A must have sigma=0$'),
            ('factory-noise-state','return collector',
             'collector.noise.manual_seed(collector.stream_seed + 1); return collector',AssertionError,r' != ')):
            namespace = dict(d.__dict__)
            self.assertIn(old,source)
            exec(source.replace(old,new),namespace)
            self.killer(name,lambda:self.assert_factory_equivalent(namespace['_make_collector_with']),exception, message)
        with self.assertRaisesRegex(RuntimeError,'fresh-flow branch mismatch'):
            d._make_collector_with(lambda *args:{'flow_seed':-1},4,4,80,
                                   np.zeros(2,np.float32),np.ones(2,np.float32))

    def test_27_three_arms_noise_bytes_and_factory(self):
        sequences = []
        original = d.m3.noisy_action
        for arm in ('O3','OF','Q-C'):
            noise,collectors = [],[]
            def record(collector,action):
                if not collectors: collectors.append(collector)
                self.assertIs(collector,collectors[0])
                generator = torch.Generator(device='cpu')
                generator.set_state(collector.noise.get_state())
                noise.append(torch.randn(action.shape,generator=generator).numpy().tobytes())
                sent = original(collector,action)
                self.assertEqual(collector.noise.get_state().numpy().tobytes(),
                                 generator.get_state().numpy().tobytes())
                return sent
            with patch.object(d,'H',40),patch.object(d,'make_collector_detour',wraps=d.make_collector_detour) as factory,patch.object(d.m3,'noisy_action',side_effect=record):
                row,module,env,plan = self.run_row(arm,env=ScriptEnv())
            factory.assert_called_once()
            h.validate_noise(row)
            self.assertEqual(len(noise),40)
            self.assertEqual(collectors[0].u_calls,10 if arm in d.FLOW_ARMS else 0)
            self.assertEqual(len(module.flow_count),10 if arm in d.FLOW_ARMS else 0)
            self.assertEqual(row['seeds']['flow_seed'],collectors[0].flow_seed if arm in d.FLOW_ARMS else None)
            sequences.append(b''.join(noise))
        self.assertEqual(sequences[0],sequences[1])
        self.assertEqual(sequences[0],sequences[2])
        reference = torch.Generator(device='cpu').manual_seed(d.seeds_detour(4,4,80)['noise_seed'])
        self.assertEqual(sequences[0],b''.join(torch.randn((2,),generator=reference).numpy().tobytes() for _ in range(40)))
        print('THREE_ARM_NOISE_BYTES_PASS: O3/OF/Q-C same task4 episode4 draw80; 40 per-step samples; flow calls 0/0/10')

    def test_28_cpu_real_short_wiring(self):
        """Real M9/ckpt + real maze, in memory only; short wiring, never a score/E1 artifact."""
        dataset_dir = os.environ.get('PROBE_TEST_DATASET_DIR','/home/cymaxwelllee/data/ogbench')
        if not d.runtime.CKPT.is_file() or not (Path(dataset_dir)/(common.ENV+'.npz')).is_file():
            self.fail('UNVERIFIED-CPU-REAL: pinned checkpoint or normalization dataset unavailable')
        module = self.real_cpu_module(dataset_dir)
        self.assertEqual(str(module.device),'cpu')
        env = module.ogbench.make_env_and_datasets(common.ENV,env_only=True)
        try:
            traps,builder,walls = d.geometry_context(env)
            tables={'cells':{},'of_cells':{}}
            for task in (1,4):
                cells,of_cells = {},{}
                for cell in d.free_cells(traps,task):
                    g=traps['tasks'][str(task)]['g']
                    if list(cell)==g: continue
                    points=d.oracle_points(traps,task,cell,env.unwrapped.ij_to_xy)
                    u=m1.encode_trajectory(module,points)
                    of_cells[d.cell_key(cell)]=dict(d.e1_check(d.decode(module,u,points[0]),d.resample(points),walls),tau_sha256=common.digest_array(points),u_sha256=common.digest_array(m1.array_of(u)),delivered_sha256=common.digest_array(m1.array_of(module._q(u))))
                    w,target,points=d.short_points(traps,task,cell,env.unwrapped.ij_to_xy,env.unwrapped.ij_to_xy(cell),env.unwrapped.ij_to_xy(g))
                    if w==g:continue
                    u=m1.encode_trajectory(module,points)
                    cells[d.cell_key(cell)]=dict(d.short_record(module,u,points,env.unwrapped.ij_to_xy(g),walls,d.direction_threshold()),tau_sha256=common.digest_array(points),u_sha256=common.digest_array(m1.array_of(u)),delivered_sha256=common.digest_array(m1.array_of(module._q(u))))
                tables['cells'][str(task)]=cells;tables['of_cells'][str(task)]=of_cells
            module.detour_e1_table=tables
            rows,noise_sequences = [],[]
            original_noise,original_head = d.m3.noisy_action,module.ahead
            for arm in ('Q-C','O3','F3','OF'):
                noises,seen,collectors = [],[],[]
                def noise(collector,action):
                    if not collectors: collectors.append(collector)
                    self.assertIs(collector,collectors[0])
                    generator = torch.Generator(device='cpu')
                    generator.set_state(collector.noise.get_state())
                    noises.append(torch.randn(action.shape,generator=generator).numpy().tobytes())
                    sent = original_noise(collector,action)
                    self.assertEqual(collector.noise.get_state().numpy().tobytes(),generator.get_state().numpy().tobytes())
                    return sent
                def head(cond,u):
                    seen.append(common.digest_array(m1.array_of(u)))
                    return original_head(cond,u)
                plan=next(p for p in self.plan['smoke'] if p['arm']==arm and p['task']==(1 if arm=='Q-C' else 4))
                with patch.object(d,'H',40),patch.object(d.m3,'noisy_action',side_effect=noise),patch.object(module,'ahead',side_effect=head),patch.object(module,'sample_plan',wraps=module.sample_plan) as flow:
                    row = d.run_draw_detour(module,env,plan,arm)
                self.assertEqual(row['observed_steps'],40)
                self.assertFalse(row['synthetic'])
                self.assertEqual(flow.call_count,len(row['chunks']) if arm in d.FLOW_ARMS else 0)
                self.assertEqual(collectors[0].u_calls,flow.call_count)
                self.assertEqual(len(row['flow_calls']),flow.call_count)
                self.assertEqual(seen,[c['delivered_sha256'] for c in row['chunks']])
                self.assertTrue(all(c['delivered_sha256']==c['expected_delivered_sha256'] for c in row['chunks']))
                self.assertTrue(all(c['qualification'] in ('N/A','PASS','FAIL') for c in row['chunks']))
                with patch.object(d,'H',40):
                    h.validate_row(row,plan,module.detour_e1_table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy)
                reference = torch.Generator(device='cpu').manual_seed(row['seeds']['noise_seed'])
                self.assertEqual(b''.join(noises),b''.join(torch.randn((2,),generator=reference).numpy().tobytes() for _ in range(40)))
                rows.append(row);noise_sequences.append(b''.join(noises))
            self.assertTrue(all(r['initial_sha256']==rows[1]['initial_sha256'] and r['goal_sha256']==rows[1]['goal_sha256'] for r in rows[1:]))
            self.assertTrue(all(x==noise_sequences[1] for x in noise_sequences[1:]))
            print('DETOUR_REAL_CPU_PASS: M9+s33/EMA, real maze, Q-C task1 episode10 and O3/F3/OF task4 episode4 draw80 H=40; full validate_row PASS, flow=10/0/10/0, noise bytes paired, delivered u hashes and qualification fields checked; score=false, GPU=false, archive_writes=0')
        finally:
            env.close()

    def test_29_float64_full_harvest(self):
        script=[(28.123456789,8.012345678)]*4+[(12.123456789,15.234567891)]*4+[(12.123456789,16.876543219)]*4+[(14.123456789,-1.987654321)]*4+[(17.987654321,1.123456789)]*5
        for arm in ('O3','F3','OF','Q-C'):
            with self.subTest(arm=arm):
                row,_,env,plan=self.run_row(arm,ScriptEnv(script,goal=script[8],dtype=np.float64))
                self.validate(row,env,plan)
                self.assertEqual(row['observation_dtype'],'float64')
                self.assertEqual(row['xy'],[list(x) for x in script])
                bad=copy.deepcopy(row)
                bad['xy']=np.asarray(bad['xy'],np.float32).tolist()
                bad['trace_sha256']=common.digest_array(np.asarray(bad['xy'],np.float64))
                self.killer('float32-trace-'+arm,lambda:self.validate(bad,env,plan),h.Rejected, r'^initial trace$')
                if arm not in d.FLOW_ARMS:
                    self.assertEqual([c['mode'] for c in row['chunks']][:3],['static','goal','零長度沿用'])
                    bad=copy.deepcopy(row)
                    bad['chunks'][1]['tau_sha256']=common.digest_array(np.asarray([script[4],script[8]],np.float32))
                    self.killer('goal-tau-dtype-'+arm,lambda:self.validate(bad,env,plan),h.Rejected, r'^runtime/static tau hash$')

    def test_30_tail_runtime_two_goals_and_draws(self):
        cell=(6,4); start=self.env.ij_to_xy(cell)
        records=[]
        for goal in ((12.123456789,16.234567891),(11.876543219,15.876543219)):
            row,_,env,plan=self.run_row('O3',ScriptEnv([start]*9,goal=goal,dtype=np.float64))
            self.validate(row,env,plan)
            rec=row['chunks'][0]; records.append(rec)
            self.assertEqual(rec['mode'],'tail');self.assertEqual(rec['qualification_source'],'runtime')
            self.assertEqual(rec['w'],[5,4]);self.assertEqual(rec['tau_endpoint'],list(goal))
            self.assertNotIn('6,4',self.table['cells']['4'])
            bad=copy.deepcopy(row);bad['chunks'][0]['mode']='static'
            self.killer('tail-static-hash',lambda:self.validate(bad,env,plan),h.Rejected, r'^qualification mode$')
            bad=copy.deepcopy(row);bad['chunks'][0]['qualification']='FAIL' if rec['qualification']=='PASS' else 'PASS'
            self.killer('tail-runtime-qualification',lambda:self.validate(bad,env,plan),h.Rejected, r'^E1 direction qualification$')
        self.assertNotEqual(records[0]['u_sha256'],records[1]['u_sha256'])
        # Reuse the same model over draws: provider/cache must remain draw-local.
        module=mock_module(); module.detour_e1_table=self.table
        hashes=[]
        for draw,goal in zip((80,81),((12.123456789,16.234567891),(11.876543219,15.876543219))):
            env=ScriptEnv([start]*5,goal=goal,dtype=np.float64)
            plan=dict(protocol=d.PROTOCOL,task=4,episode=4,draw=draw,arm='O3',group='main')
            row=d.run_draw_detour(module,env,plan,'O3'); self.validate(row,env,plan)
            hashes.append(row['chunks'][0]['u_sha256'])
        self.assertNotEqual(*hashes)
        bad=copy.deepcopy(row);bad['chunks'][0]['tau_sha256']=records[0]['tau_sha256']
        with self.assertRaisesRegex(h.Rejected,r'^runtime/static tau hash$'):self.validate(bad,env,plan)
        print('TWO_STATE tail-goal-u: separate draws distinct runtime hashes=PASS; stale first-draw hash=FAIL')

    def test_31_ruler_threshold_is_live_and_pinned(self):
        ruler=json.loads(d.RULER.read_text())
        want=float(np.quantile([r['cos_4'] for r in ruler['real'] if r['cos_4'] is not None],.1))
        self.assertEqual(d.direction_threshold(),want)
        changed=copy.deepcopy(ruler)
        for r in changed['real']:r['cos_4']=.9
        with patch.object(d,'checked_json',return_value=changed):
            self.assertEqual(d.direction_threshold(),.9)
            self.assertEqual(d.direction_check(d.resample([[0,0],[4,0]]),np.asarray([[0,0],[4,0]]),[4,0])['qualification'],'PASS')
        bad=copy.deepcopy(self.table);bad['threshold']=.814
        self.killer('rounded-hardcoded-p10',lambda:h.validate_table(bad,fixture=True),h.Rejected, r'^E1 threshold/h$')
        # Frozen comparison only enumerates shared static cells, and records every disagreement.
        cpu=d.checked_json(d.CPU_E1,d.CPU_E1_PIN)['table']
        want=[(int(t),k) for t,cells in self.table['cells'].items() for k,c in cells.items()
              if k in cpu[t] and (c['qualification']=='PASS')!=cpu[t][k]['pass_']]
        self.assertEqual([(r['task'],r['cell']) for r in self.table['cpu_comparison']['disagreements']],want)

    def test_32_direction_second_clause_truth_table(self):
        threshold=d.direction_threshold()
        cases=[
            ('plan-wins',[1,0],[1,0],[-1,0],'PASS'),
            ('same-direction-exception',[1,.2],[1,0],[1,0],'PASS'),
            ('neither',[1,.6],[1,0],[.5,np.sqrt(3)/2],'FAIL'),
            ('first-clause-fails',[0,1],[1,0],[1,0],'FAIL'),
            ('cos45-475',[1,.6],[1,0],[1,1],'FAIL'),
            ('cos45-476',[1,.6],[1,0],[np.cos(np.pi/4),np.sin(np.pi/4)],'FAIL'),
        ]
        for name,D,P,G,want in cases:
            with self.subTest(case=name):
                record=d.direction_metrics(D,P,G,threshold)
                if name in ('neither','cos45-475','cos45-476'):
                    self.assertGreaterEqual(record['cos_plan'],threshold)
                    self.assertLessEqual(record['cos_plan'],record['cos_greedy'])
                if name.startswith('cos45-'):
                    self.assertEqual(record['cos_plan_greedy'],
                                     .7071067811865475 if name.endswith('475') else .7071067811865476)
                    self.assertLessEqual(record['cos_plan_greedy'],np.cos(np.pi/4))
                self.assertEqual(record['qualification'],want,name)
        for index in range(3):
            vectors=[[1,0],[1,0],[1,0]];vectors[index]=[0,0]
            with self.subTest(zero=index):
                self.assertEqual(d.direction_metrics(*vectors,threshold),
                                 dict(cos_plan=None,cos_greedy=None,cos_plan_greedy=None,qualification='FAIL'))
        print('TWO_STATE N1-second-clause: plan-wins/same-direction=PASS; neither/cos45-475/cos45-476/zeros=FAIL')

    def test_33_real_cpu_e1_common_static_zero_flips(self):
        """Real frozen model; only the GPU/hostname entry guards are patched for this CPU fixture."""
        dataset_dir=os.environ.get('PROBE_TEST_DATASET_DIR','/home/cymaxwelllee/data/ogbench')
        module=self.real_cpu_module(dataset_dir)
        self.assertEqual(str(module.device),'cpu')
        self.assertFalse(getattr(module,'detour_mock',False))
        env=module.ogbench.make_env_and_datasets(common.ENV,env_only=True)
        try:
            # No execute(), out=, archive write, or formal/production receipt.
            with patch.object(d,'require_real_module') as guard,patch('socket.gethostname',return_value='jasmine'):
                table=d.build_e1_table(module,env=env)
            guard.assert_called_once_with(module)
            reference=d.checked_json(d.CPU_E1,d.CPU_E1_PIN)['table']
            shared=[(t,k,c,reference[t][k]) for t,cells in table['cells'].items()
                    for k,c in cells.items() if k in reference[t]]
            self.assertEqual(len(shared),205)
            flips=[(t,k) for t,k,c,r in shared if (c['qualification']=='PASS')!=r['pass_']]
            self.assertEqual(flips,[],'CPU common-static qualification flips')
            delta=max(abs(c['cos_plan']-r['cos_plan']) for _,_,c,r in shared)
            self.assertLessEqual(delta,1e-6,'CPU common-static cos_plan drift')
            self.assertEqual(table['cpu_comparison']['compared'],len(shared))
            self.assertEqual(table['cpu_comparison']['disagreements'],[])
            h.validate_table(table)
            print(f'REAL_CPU_E1_COMMON_STATIC_PASS: compared={len(shared)} flips={len(flips)} max_delta_cos={delta:.17g}; production_evidence=false GPU=false archive_writes=0')
        finally:
            env.close()

    @classmethod
    def real_cpu_module(cls, dataset_dir):
        # configure_determinism sets torch interop threads once per process.
        # Reuse the exact frozen model for CPU wiring and the full E1 fixture.
        if not hasattr(cls,'_real_cpu_module'):
            cls._real_cpu_module=d.runtime.load_frozen(dataset_dir)
        return cls._real_cpu_module

    def reject_row_tamper(self, name, row, env, plan, change, message):
        self.validate(row,env,plan)
        bad=copy.deepcopy(row);change(bad)
        bad['failed_qualification_chunks']=sum(c['qualification']=='FAIL' for c in bad['chunks'])
        self.killer(name,lambda:self.validate(bad,env,plan),h.Rejected,message)

    def test_34_runtime_qualification_recomputed_not_counted(self):
        row,_,env,plan=self.run_row('F3')
        def flip(r):
            c=r['chunks'][0];c['qualification']='FAIL' if c['qualification']=='PASS' else 'PASS'
        self.reject_row_tamper('N2-runtime-qualification-count-repaired',row,env,plan,flip,
                               r'^E1 direction qualification$')

    def test_35_direction_D_recomputed(self):
        row,_,env,plan=self.run_row('F3')
        self.reject_row_tamper('N2-direction-D',row,env,plan,
                               lambda r:r['chunks'][0].update(direction_D=[0.,0.]),r'^E1 direction cos_plan$')

    def test_36_static_u_frozen_authority(self):
        row,_,env,plan=self.run_row()
        self.reject_row_tamper('N2-static-u',row,env,plan,
                               lambda r:r['chunks'][0].update(u_sha256='a'*64),r'^E0 frozen table u_sha256$')

    def test_37_static_tau_geometry_and_frozen_authority(self):
        row,_,env,plan=self.run_row()
        self.reject_row_tamper('N2-static-tau-geometry',row,env,plan,
                               lambda r:r['chunks'][0].update(tau_sha256='a'*64),r'^runtime/static tau hash$')
        # A canonical static tau cannot differ from a correct frozen table. Corrupt the
        # table fixture separately to reach the frozen-tau guard after geometry passes.
        bad_table=copy.deepcopy(self.table)
        c=row['chunks'][0]
        bad_table['cells'][str(row['task'])][d.cell_key(c['c_eff'])]['tau_sha256']='a'*64
        self.killer('N2-static-tau-frozen',lambda:h.validate_row(row,plan,bad_table,env.xy_to_ij,env.ij_to_xy,fixture=True),
                    h.Rejected,r'^E0 frozen table tau_sha256$')

    def test_38_static_qualification_frozen_authority(self):
        row,_,env,plan=self.run_row()
        def flip(r):
            c=r['chunks'][0];c['qualification']='FAIL' if c['qualification']=='PASS' else 'PASS'
        self.reject_row_tamper('N2-static-qualification-count-repaired',row,env,plan,flip,
                               r'^E0 frozen table qualification$')

    def runtime_reference_tamper(self, arm):
        start=self.env.ij_to_xy((6,4))
        row,_,env,plan=self.run_row(arm,ScriptEnv([start]*9,goal=(12.123456789,16.234567891),dtype=np.float64))
        if arm=='O3':self.assertEqual(row['chunks'][0]['mode'],'tail')
        else:self.assertEqual(row['chunks'][0]['qualification_source'],'runtime-flow')
        def change(r):
            c=r['chunks'][0]
            c['direction_G']=[c['direction_G'][0]+1.,c['direction_G'][1]]
            # Internal cosines/qualification/count remain consistent. Only the
            # independent runtime reference can reject this row.
            c.update(d.direction_metrics(c['direction_D'],c['direction_P'],c['direction_G'],d.direction_threshold()))
        self.reject_row_tamper('N2-runtime-reference-'+arm,row,env,plan,change,r'^runtime direction reference$')

    def test_39_F3_tail_runtime_reference(self):
        self.runtime_reference_tamper('F3')

    def test_40_O3_tail_runtime_reference(self):
        self.runtime_reference_tamper('O3')

    def test_41_zero_length_rule(self):
        row,_,env,plan=self.run_row()
        def change(r):r['chunks'][1].update(mode='零長度沿用',qualification_source='runtime')
        self.reject_row_tamper('N2-zero-length-rule',row,env,plan,
                               change,r'^zero-length inheritance rule$')

    def test_42_zero_length_keeps_qualification(self):
        script=[(28.,8.)]*4+[(12.,15.2)]*4+[(12.,16.8)]*5
        row,_,env,plan=self.run_row(env=ScriptEnv(script,goal=(12.,16.8)))
        self.assertEqual(row['chunks'][2]['mode'],'零長度沿用')
        def flip(r):
            c=r['chunks'][2];c['qualification']='FAIL' if c['qualification']=='PASS' else 'PASS'
        self.reject_row_tamper('N2-zero-length-inherited-qualification',row,env,plan,flip,
                               r'^zero-length lost qualification/input$')

    def test_43_same_question_reset_fingerprint(self):
        row,_,env,plan=self.run_row()
        plan2=dict(plan,draw=81)
        paired,_,env2,_=self.run_row(plan=plan2)
        h.validate_rows([row,paired],[plan,plan2],self.table,env.xy_to_ij,env.ij_to_xy,True)
        changed,_,env2,_=self.run_row(env=ScriptEnv([(28.,8.)]*10,goal=(12.125,16.25)),plan=plan2)
        self.validate(changed,env2,plan2)
        self.assertNotEqual(row['goal_sha256'],changed['goal_sha256'])
        self.killer('N2-unpaired-reset',lambda:h.validate_rows([row,changed],[plan,plan2],self.table,env.xy_to_ij,env.ij_to_xy,True),
                    h.Rejected,r'^unpaired environment resets$')

    def test_44_early_stop_cannot_claim_horizon(self):
        row,_,env,plan=self.run_row()
        self.assertLess(row['observed_steps'],d.H)
        self.reject_row_tamper('N2-early-horizon',row,env,plan,
                               lambda r:r.update(terminated=False,truncated=False,termination_reason='horizon'),r'^early stop / reason$')

    def test_45_success_only_last_step(self):
        row,_,env,plan=self.run_row(env=ScriptEnv([(28.,8.)]*10,success_at=9))
        self.assertTrue(row['goal_success'][-1])
        self.reject_row_tamper('N2-success-before-last',row,env,plan,
                               lambda r:r['goal_success'].__setitem__(1,True),r'^success termination timing$')

    def test_46_w_field_independently_checked(self):
        row,_,env,plan=self.run_row()
        self.reject_row_tamper('N2-w-field',row,env,plan,
                               lambda r:r['chunks'][0].update(w=[-1,-1]),r'^E0 supplied w/cond target$')

    def test_47_O3_actual_encoder_endpoint_is_target(self):
        script=[(28.,8.)]*4+[self.env.ij_to_xy((6,4))]*4+[(12.,15.2)]*5
        with patch.object(m1,'encode_trajectory',wraps=m1.encode_trajectory) as encode:
            row,module,env,plan=self.run_row(env=ScriptEnv(script,goal=(12.125,16.25)))
        self.validate(row,env,plan)
        self.assertEqual([c['mode'] for c in row['chunks']],['static','tail','goal'])
        self.assertEqual(len(module.encode_inputs),len(row['chunks']))
        calls=[call for call in encode.call_args_list if call.args[0] is module]
        self.assertEqual(len(calls),len(row['chunks']))
        for chunk,call,captured in zip(row['chunks'],calls,module.encode_inputs):
            np.testing.assert_array_equal(call.args[1][-1],chunk['target_xy'],err_msg='O3 raw tau endpoint = target')
            self.assertEqual(common.digest_array(call.args[1]),chunk['tau_sha256'])
            self.assertEqual(captured.shape,(1,128,2))
            # mock normalization is identity: inspect the tensor at encode_u itself.
            np.testing.assert_array_equal(captured[0,-1],np.asarray(chunk['target_xy'],np.float32),
                                          err_msg='O3 captured encode_u endpoint = target')
        self.assertNotEqual(row['chunks'][0]['target_xy'],row['goal'])

    def test_48_killer_requires_expected_exception_and_message(self):
        def wrong_exception():raise RuntimeError('unrelated setup failure')
        def wrong_message():raise h.Rejected('failed qualification count')
        with self.assertRaisesRegex(RuntimeError,r'^unrelated setup failure$'):
            self.killer('N7-wrong-exception',wrong_exception,h.Rejected,r'^E1 direction qualification$')
        with self.assertRaisesRegex(AssertionError,r'E1 direction qualification.*does not match.*failed qualification count'):
            self.killer('N7-wrong-message',wrong_message,h.Rejected,r'^E1 direction qualification$')

    def test_24_formal_readout_denominators_and_q_pairs(self):
        base,_,env,_=self.run_row()
        rows=[]
        rank={t:{q['episode']:i for i,q in enumerate([q for q in self.plan['questions']['main'] if q['task']==t])} for t in (4,5,2)}
        gate_keys=sorted({(p['task'],p['episode'],p['draw']) for p in self.plan['formal'] if p['group']=='gate'})
        for p in self.plan['formal']:
            row=copy.deepcopy(base);row.update({k:p[k] for k in ('protocol','task','episode','arm','draw','group')})
            if p['group']=='main':
                threshold={4:10,5:0,2:4}[p['task']] if p['arm']=='O3' else 0
                row['success']=rank[p['task']][p['episode']]<threshold and p['draw']<82
                row['xy'][-1]=env.ij_to_xy(self.traps['tasks'][str(p['task'])]['traps_g'][0])
                if p['task']==5: row['chunks'][0]['qualification']='FAIL'
            else:
                row['success']=gate_keys.index((p['task'],p['episode'],p['draw']))<{'Q-C':8,'Q-O3':6,'Q-F3':0}[p['arm']]
            rows.append(row)
        result=h.readout(rows,self.plan,self.table,env.xy_to_ij,env.ij_to_xy)
        self.assertEqual([result['layers'][str(t)]['arms']['O3']['label'] for t in (4,5,2)],['救回','未救回','部分'])
        self.assertEqual([result['layers'][str(t)]['arms']['O3']['n'] for t in (4,5,2)],[14,6,10])
        self.assertEqual(result['layers']['5']['failure']['denominator'],24)
        self.assertEqual(result['layers']['5']['failure']['label'],'近陷阱但首方向不合格')
        self.assertEqual(result['layers']['4']['rule'],'R2')
        self.assertEqual(result['Q']['annotations'],['flow＋近目標在易題也變差'])
        self.assertEqual(result['layers']['4']['OF']['d_h'],20)
        table=copy.deepcopy(self.table)
        for cells in table['cells'].values():
            for c in cells.values():c['qualification']='FAIL'
        self.assertEqual(h.readout(rows,self.plan,table,env.xy_to_ij,env.ij_to_xy)['layers']['4']['arms']['O3'],result['layers']['4']['arms']['O3'])


if __name__=='__main__':
    unittest.main(verbosity=2)
