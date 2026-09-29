"""CPU wiring witnesses, not teacher qualification or a refiner learning verdict.

The scratch fixture executes its COMPLETE production _stage2_loop AST unchanged;
only dataset/environment globals are fixtures. Mutants replace executable compose
or kernel source in memory, then run that same production loop and witnesses.
"""
import ast
import contextlib
import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lacot.model import LaCoTActor, LaCoTActorState, RefineOperator
from lacot.nf_head import Flow
from lacot import refine_objective as objective
from lacot import refine_training as training
from lacot.refine_quality import QualityReport, QualityService

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / 'experiments/scratch_lacot_rollout.py'
B, K, D, C = 3, 2, 4, 4


def frozen(module):
    return module.eval().requires_grad_(False)


def exposure_result(loss):
    return loss, dict(valid=True, n_valid=B, n_total=B, reason=None)


class WiringOracle(QualityService):
    """Deliberately TEST ONLY: no learned/qualified production teacher is claimed."""
    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer('weights', torch.tensor([.4, .8, 1.3, 1.9]))
        self.calibration = SimpleNamespace(teacher_fingerprint='cpu-wiring-only')
        self.authorizations = 0
        self.valid = torch.tensor([True, False, True])
        self.eval()

    def authorize_training(self, horizon, *, same_plan=False):
        self.authorizations += 1
        assert horizon == 4 and same_plan

    def forward(self, u):
        # Asymmetric quadratic gives a nonconstant Jacobian through real LayerNorm.
        cost = ((u.float() - self.weights) ** 2 * self.weights).mean((1, 2))
        return QualityReport(cost, {'quadratic': cost}, self.valid.to(u.device),
                             torch.zeros_like(cost), torch.ones_like(cost),
                             self.calibration.teacher_fingerprint)


def services_for(refine, head, *, jitter=0.):
    oracle = WiringOracle()
    teacher = frozen(copy.deepcopy(refine))
    with torch.no_grad():
        next(teacher.parameters()).add_(.09)  # stale, nontrivial EMA target
    service = training.TrainingServices(
        teacher, oracle, lambda sampled, cond: oracle,
        # Test-only same-plan action labels; never a data-action closure.
        lambda u, features: exposure_result(head.nll(head(features), u[:, :1, :2].detach().tanh()).mean()),
        (frozen(nn.Identity()),), jitter_std=jitter,
        noise_generator=torch.Generator().manual_seed(483) if jitter else None)
    return service


def real_actor(kind):
    torch.manual_seed(76)
    if kind == 'image':
        actor = LaCoTActor(frozen(nn.Identity()), C // 2, D, K, 2, 1,
                           num_bins=8, n_flow_blocks=1, refine_hidden=12)
    else:
        actor = LaCoTActorState(2, D, K, 2, 1, enc_hidden=12, enc_out=8,
                                cond_dim=C, n_flow_blocks=1, refine_hidden=12)
    # Still real TARFlow, smaller width for the CPU exam.
    actor.flow = Flow(D, K, n_blocks=1, d_hidden=8, n_layers=1, n_heads=2, cond_dim=C)
    with torch.no_grad():
        actor.refine.net[-1].weight.normal_(0, .13)
    cond = torch.randn(B, C, requires_grad=True)
    clean = torch.randn(B, K, D, requires_grad=True)
    actions = torch.tensor([[[-.8, -.1]], [[.2, .6]], [[.9, -.4]]])
    return actor, cond, clean, actions


def load_scratch(namespace, source=None):
    tree = ast.parse(SCRATCH.read_text() if source is None else source)
    names = {'_stage2_loop', '_flow_nll', 'sample_plan', '_amp_step', '_amp_update'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names, 'production AST entry missing'
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRATCH), 'exec'), namespace)
    return namespace


def load_scratch_provenance(namespace):
    """Execute only production config, eval CONS gate, and filename expressions."""
    tree = ast.parse(SCRATCH.read_text())
    wanted = {'CONS', 'OBJECTIVE_VERSION', 'tag'}
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))
             or (isinstance(node, ast.FunctionDef) and node.name in {'_cons_from_ckpt_cfg', '_tag_extra'})]
    assert len(nodes) == 5
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRATCH), 'exec'), namespace)
    return namespace


class ScratchHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(C + K * D, 2)

    def forward(self, cond, u):
        return self.net(torch.cat([cond, u.flatten(1)], 1)).reshape(B, 1, 2)


class ScratchBC(nn.Linear):
    def forward(self, cond):
        return super().forward(cond).reshape(B, 1, 2)


class SmallFSQ:
    """Representation fixture: z has 2 dims; decoded u has 4, visibly different."""
    d = 2
    def z_of(self, u):
        return u[..., :2] + 2.7
    def dequant_z(self, u):
        return self.z_of(u) + torch.rand_like(u[..., :2]) * .1
    def snap(self, u):
        return u * .6 - .3


class ZFlow:
    def __init__(self, flow):
        self.inner = flow
    def sample(self, n, cond):
        z = self.inner.sample(n, cond)
        return torch.cat([z, z + .7], -1)
    def nll(self, u, cond):
        return self.inner.nll(u, cond)


def scratch_fixture(*, learned=True, jitter=0., fsq=False, intent=False, bc_indep=False, div=0., source=None):
    torch.manual_seed(87)
    condnet = nn.Linear(4, C)
    head = ScratchHead()
    bc = ScratchBC(C, 2)
    refine = RefineOperator(C, K, D, hidden=12)
    with torch.no_grad():
        refine.net[-1].weight.normal_(0, .12)
    inner = Flow(2 if fsq else D, K, n_blocks=1, d_hidden=8, n_layers=1,
                 n_heads=2, cond_dim=C)
    flow = ZFlow(inner) if fsq else inner
    clean = torch.randn(B, K, D)
    state, goal = torch.randn(B, 2), torch.randn(B, 2)
    actions = torch.tensor([[[-.8, -.1]], [[.2, .6]], [[.9, -.4]]])
    weights = torch.tensor([1., .2, 0.])
    svc = services_for(refine, nn.Identity(), jitter=jitter)
    svc.exposure = lambda u, features: exposure_result((head(*features) - u[:, :1, :2].detach().tanh()).square().mean())
    svc.batch_factory = lambda **context: svc  # test oracle has no state dependence
    params = [condnet, head, refine, inner] + ([] if bc_indep else [bc])
    opt = torch.optim.SGD([p for m in params for p in m.parameters()], lr=.01)
    ns = dict(torch=torch, refine_training=training, device='cpu', B=B, K=K, DIM=K*D,
              LEARNED_REFINE=int(learned), CONS='ema', EMA_M=.99, refine=refine,
              refine_ema=svc.teacher, REFINE_TRAINING_SERVICES=svc if learned else None,
              u_dec=None, s_embed=None, flow=flow, rng=None, _NLL_C={'fn': None},
              _amp_ctx=contextlib.nullcontext, _amp_unscale=lambda opt: None,
              _SCALER=None, _AMP_ST={'steps': 0, 'skip': 0},
              _warm_lr=lambda *a: None, opt2=opt, f_mods=params,
              opt_bc=torch.optim.SGD(bc.parameters(), lr=.01) if bc_indep else None,
              bc_head=bc, ahead=head,
              TEACHER_MIX=0, BC_INDEP=int(bc_indep), BC_OWN=0, LO_W=0., GRPO_W=0.,
              DIV_W=div, DIV_M=.3, DIV_LOG_EVERY=0, _EMA_PAIRS=[], LOG_EVERY=1000,
              INTENT_GUID_W=0., intent_ad=SimpleNamespace() if intent else None,
              INTENT='embed' if intent else '', INTENT_DROP=.2 if intent else 0.,
              COND_DROP=.4, fsq=SmallFSQ() if fsq else None, FSQ_SPACE='z' if fsq else 'u',
              FSQ_TGT='dequant' if fsq else 'snap', _REAL_W=[weights],
              _q=lambda u: u, etarget=lambda traj, mask: traj,
              intent_anchors_of=lambda traj: traj.mean(1),
              _intent_cond=lambda anc: anc,
              flow_cond=lambda cond, anc: cond if anc is None else cond + .1*anc,
              condvec=lambda s, g, ix: condnet(torch.cat([s, g], 1)) + (0 if ix is None else .1*ix))
    ns['make_batch'] = lambda rng, **kw: (clean, torch.zeros(B, K, dtype=torch.bool), state, goal, actions)
    backwards = []
    cond_values = []
    original_condvec = ns['condvec']
    def condvec(s, g, ix):
        result = original_condvec(s, g, ix)
        cond_values.append(result.detach().clone())
        return result
    ns['condvec'] = condvec
    def backward(loss):
        backwards.append(loss.detach().clone())
        loss.backward()
    ns['_amp_backward'] = backward
    load_scratch(ns, source)
    return SimpleNamespace(ns=ns, service=svc, refine=refine, flow=flow, head=head,
                           actions=actions, clean=clean, backwards=backwards, opt=opt,
                           cond_values=cond_values)


def grads(value, inputs):
    if not value.requires_grad:
        return [torch.zeros_like(x) for x in inputs]
    return [torch.zeros_like(x) if g is None else g for g, x in
            zip(torch.autograd.grad(value, inputs, retain_graph=True, allow_unused=True), inputs)]


class Trace:
    """Independent numeric + Jacobian witness, while the actual caller composes."""
    def __init__(self, case, flow, service, *, verify=True):
        self.case, self.flow, self.service, self.verify = case, flow, service, verify
        self.rows, self.samples = [], []
        self.original = training.compose
        self.original_sample = flow.sample

    def sample(self, *a, **kw):
        u = self.original_sample(*a, **kw)
        self.samples.append(u.detach().clone())
        return u

    def compose(self, adapter, refine, cond, clean, **kw):
        count = len(self.samples)
        noise_state = None if self.service.noise_generator is None else self.service.noise_generator.get_state()
        result = self.original(adapter, refine, cond, clean, **kw)
        total, logs, states = result
        rounds = kw['rounds']
        row = dict(caller=inspect.currentframe().f_back.f_code.co_name, rounds=rounds,
                   total=total, logs=logs, states=states, cond=cond, clean=clean,
                   features=adapter.features(clean), sample_count=len(self.samples)-count)
        self.rows.append(row)
        if not rounds:
            torch.testing.assert_close(total, logs['l_nf'] + logs['l_act_anchor'], rtol=0, atol=0)
            assert not states and row['sample_count'] == 0
            return result
        assert row['sample_count'] == 1, 'fresh flow sample once per optimizer step'
        sampled = self.samples[-1]
        noise = None
        if noise_state is not None:
            gen = torch.Generator().set_state(noise_state)
            noise = torch.randn(sampled.shape, generator=gen, dtype=sampled.dtype) * self.service.jitter_std
        if self.verify:
            assert len(states) == rounds + 1, 'rounds must be honored'
            torch.testing.assert_close(states[0], sampled, rtol=0, atol=0, msg='sample provenance must be fresh flow, not clean target')
            u, v = sampled, None if noise is None else sampled + noise
            expected_q, expected_c = 0, 0
            for r in range(rounds):
                previous = u
                u = refine(cond, u)
                torch.testing.assert_close(states[r+1], u, rtol=0, atol=0)
                qr = self.service.quality_service(u)
                q = qr.cost[qr.valid].mean()
                if v is not None:
                    v = refine(cond, v)
                    vr = self.service.quality_service(v)
                    q = (q + .25 * vr.cost[vr.valid].mean()) / 1.25
                expected_q = expected_q + q / rounds
                with torch.no_grad():
                    target = self.service.teacher(cond, previous)
                expected_c = expected_c + (u - target).square().mean() / rounds
            torch.testing.assert_close(logs['l_plan_quality'], expected_q, rtol=1e-5, atol=1e-6, msg='main + genuine auxiliary noise quality')
            torch.testing.assert_close(logs['l_cons'], expected_c, rtol=1e-5, atol=1e-6)
            expected_total = (logs['l_nf'] + logs['l_act_anchor'] + logs['l_plan_quality']
                              + kw['lam_cons'] * logs['l_cons'] + .25 * logs['l_head_exposure'])
            torch.testing.assert_close(total, expected_total, rtol=0, atol=0, msg='base_total includes each term exactly once')
            inputs = list(refine.parameters()) + [cond]
            actual_grad, expected_grad = grads(logs['l_plan_quality'], inputs), grads(expected_q, inputs)
            for i, (a, e) in enumerate(zip(actual_grad, expected_grad)):
                torch.testing.assert_close(a, e, rtol=3e-5, atol=2e-6,
                                           msg=f'full BPTT / cond Jacobian input {i}')
            assert actual_grad[-1].abs().sum() > 0, 'cond witness must be nonzero'
        row['quality_grads'] = grads(logs['l_plan_quality'], list(refine.parameters()))
        row['noise'] = noise
        return result

    @contextlib.contextmanager
    def active(self):
        with patch.object(self.flow, 'sample', new=self.sample), patch.object(training, 'compose', new=self.compose):
            yield self


class WiringTests(unittest.TestCase):
    def test_cons_default_preserves_old_filename_byte_for_byte(self):
        for env_cons, expected in ((None, 'medium-stitch_self_K4_c256_ch4_st2000_T128_ep50_gu_s0'),
                                   ('ema', 'medium-stitch_ema_K4_c256_ch4_st2000_T128_ep50_gu_s0')):
            env = {} if env_cons is None else {'LACOT_CONS': env_cons}
            ns = dict(os=os, refine_training=training, LEARNED_REFINE=1,
                      ENV_NAME='pointmaze-medium-stitch-v0', K=4, COND=256,
                      CHUNK=4, STEPS2=2000, T_CAP=128, SEEDS=50, TAG_SEED=0,
                      _extra='')
            with patch.dict(os.environ, env, clear=True):
                load_scratch_provenance(ns)
            self.assertEqual(ns['_tag_extra'](), ns['_extra'])
            self.assertEqual(ns['tag'], expected)
            self.assertEqual(ns['CONS'], 'self' if env_cons is None else 'ema')
            self.assertEqual(ns['OBJECTIVE_VERSION'], training.OBJECTIVE_VERSION)

    def test_eval_only_cons_comes_from_ckpt_cfg_in_both_states(self):
        ns = dict(os=os, refine_training=training, LEARNED_REFINE=1,
                  ENV_NAME='pointmaze-medium-stitch-v0', K=4, COND=256,
                  CHUNK=4, STEPS2=0, T_CAP=128, SEEDS=50, TAG_SEED=0,
                  _extra='')
        with patch.dict(os.environ, {}, clear=True):
            load_scratch_provenance(ns)
        for saved in ('self', 'ema'):
            self.assertEqual(ns['_cons_from_ckpt_cfg']({'CONS': saved}, None), saved)
            self.assertEqual(ns['_cons_from_ckpt_cfg']({'CONS': saved}, saved), saved)
            other = 'ema' if saved == 'self' else 'self'
            with self.assertRaisesRegex(AssertionError, '不一致'):
                ns['_cons_from_ckpt_cfg']({'CONS': saved}, other)
        with self.assertRaisesRegex(AssertionError, '缺少有效 CONS'):
            ns['_cons_from_ckpt_cfg']({}, None)
        tree = ast.parse(SCRATCH.read_text())
        load_block = next(n for n in tree.body if isinstance(n, ast.If) and
                          isinstance(n.test, ast.Name) and n.test.id == 'LOAD_CKPT')
        self.assertIn('_cons_from_ckpt_cfg(_cfg, os.environ.get(\'LACOT_CONS\'))',
                      ast.unparse(load_block))

    def test_objective_version_is_recorded_without_changing_tag(self):
        tree = ast.parse(SCRATCH.read_text())
        out = next(n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'out' for t in n.targets))
        self.assertIn('objective_version=OBJECTIVE_VERSION', ast.unparse(out))
        cfg_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == 'dict' and
                     {'CONS', 'OBJECTIVE_VERSION'} <= {k.arg for k in n.keywords}]
        self.assertEqual(len(cfg_calls), 1)
        tag = next(n for n in tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == 'tag' for t in n.targets))
        self.assertNotIn('OBJECTIVE_VERSION', ast.unparse(tag))

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_real_callers_fresh_flow_own_rounds_and_gradients(self):
        for kind in ('image', 'state', 'scratch'):
            if kind == 'scratch':
                f = scratch_fixture()
                trace = Trace(kind, f.flow, f.service)
                with trace.active():
                    # Full production loop, 4 tiny CPU optimizer steps, depth cycling.
                    f.ns['_stage2_loop'](4)
            else:
                actor, cond, clean, actions = real_actor(kind)
                svc = services_for(actor.refine, actor.action_head)
                trace = Trace(kind, actor.flow, svc)
                optimizer = torch.optim.SGD(actor.parameters(), lr=.001)
                with trace.active():
                    for rounds in (0, 1, 2, 3):
                        optimizer.zero_grad(set_to_none=True)
                        total, _ = actor.losses_given(cond, clean, actions, rounds, .1, training_services=svc)
                        self.assertIs(total, trace.rows[-1]['total'])
                        total.backward()
                        svc.after_step(actor.refine, succeeded=training.optimizer_step(optimizer))
            self.assertEqual([r['rounds'] for r in trace.rows], [0, 1, 2, 3])
            self.assertEqual(len(trace.samples), 3)
            self.assertFalse(torch.equal(trace.samples[0], trace.samples[1]))
            print(f'TRACE {kind}: caller={trace.rows[0]["caller"]} compose=4 R=0/1/2/3 sample=fresh-flow(3) states=0/2/3/4 full-BPTT=PASS', flush=True)

    def test_action_permutation_does_not_change_quality_gradients(self):
        for kind in ('image', 'state', 'scratch'):
            results = []
            for permute in (False, True):
                if kind == 'scratch':
                    f = scratch_fixture(jitter=.3)
                    if permute:
                        f.actions.copy_(f.actions.roll(1, 0))
                    trace = Trace(kind, f.flow, f.service)
                    with trace.active():
                        f.ns['_stage2_loop'](1, step_off=3)
                else:
                    actor, cond, clean, actions = real_actor(kind)
                    svc = services_for(actor.refine, actor.action_head, jitter=.3)
                    trace = Trace(kind, actor.flow, svc)
                    with trace.active():
                        actor.losses_given(cond, clean, actions.roll(1, 0) if permute else actions,
                                           3, .1, training_services=svc)
                results.append(trace.rows[0])
            for a, b in zip(results[0]['quality_grads'], results[1]['quality_grads']):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertNotEqual(float(results[0]['logs']['l_act_anchor']), float(results[1]['logs']['l_act_anchor']))
            print(f'PERMUTATION {kind}: refiner-quality gradient bitwise unchanged; anchor changes', flush=True)

    def test_scratch_masks_fsq_and_outer_total(self):
        for fsq in (False, True):
            for bc_indep in (False, True):
                f = scratch_fixture(fsq=fsq, bc_indep=bc_indep, div=.07)
                trace = Trace('scratch', f.flow, f.service)
                density = []
                orig = f.flow.nll
                def nll(u, cond):
                    density.append((u.detach().clone(), cond.detach().clone()))
                    return orig(u, cond)
                bc_values, head_values = [], []
                h1 = f.head.register_forward_hook(lambda m, i, o: head_values.append(o.detach().clone()))
                h2 = f.ns['bc_head'].register_forward_hook(lambda m, i, o: bc_values.append(o.detach().clone()))
                bound_contexts = []
                def bind(**context):
                    bound_contexts.append(context)
                    return f.service
                f.service.batch_factory = bind
                try:
                    with patch.object(f.flow, 'nll', side_effect=nll), trace.active():
                        f.ns['_stage2_loop'](1, step_off=3)
                finally:
                    h1.remove(); h2.remove()
                row = trace.rows[0]
                ca, head_u = row['features']
                self.assertTrue((ca == 0).any())
                torch.testing.assert_close(head_u, f.ns['fsq'].snap(f.clean) if fsq else f.clean)
                self.assertEqual(density[0][0].shape[-1], 2 if fsq else 4)
                torch.testing.assert_close(density[0][1], row['cond'], rtol=0, atol=0)
                self.assertEqual(len(bound_contexts), 1)
                self.assertEqual(bound_contexts[0]['state'].shape, (B, 2))
                self.assertIsNone(bound_contexts[0]['anchors'])
                w = f.ns['_REAL_W'][0]
                anchor = ((head_values[0]-f.actions).square().flatten(1).mean(1)*w).sum()/w.sum()
                torch.testing.assert_close(row['logs']['l_act_anchor'], anchor, rtol=0, atol=0)
                bc_loss = ((bc_values[0]-f.actions).square().flatten(1).mean(1)*w).sum()/w.sum()
                ci, cz = f.cond_values[-2:]
                rel = torch.sqrt((ci-cz).square().sum(-1)+1e-24)/(torch.sqrt(cz.square().sum(-1)+1e-24)+1e-8)
                div_loss = torch.relu(.3-rel).mean()
                expected = row['total'].detach() + (0.*bc_loss if bc_indep else bc_loss)
                expected = expected + .07*div_loss
                if bc_indep:
                    expected = expected + bc_loss
                torch.testing.assert_close(f.backwards[0], expected, rtol=0, atol=0,
                                           msg='scratch outer total: base once + original BC/div weights')
                self.assertEqual(len(f.backwards), 1)
                self.assertEqual(row['logs']['l_act_refine'], 0)

    def test_scratch_refine_rejects_intent_before_service_or_sample(self):
        f = scratch_fixture(intent=True)
        with patch.object(f.flow, 'sample', side_effect=AssertionError('sampled before intent guard')):
            with self.assertRaisesRegex(ValueError, 'VERDICT-EXP2-S2 M2'):
                f.ns['_stage2_loop'](1, step_off=3)
        f.ns['REFINE_TRAINING_SERVICES'] = None
        with self.assertRaisesRegex(ValueError, 'same-intent-as-deployment'):
            f.ns['_stage2_loop'](1, step_off=3)

    def test_r0_no_dependencies_no_rng_and_bitwise_anchor(self):
        for kind in ('image', 'state'):
            actor, cond, clean, actions = real_actor(kind)
            nf = actor.flow.nll(clean.detach(), cond)/(K*D)
            features = clean.flatten(1) if kind == 'image' else torch.cat([cond, clean.flatten(1)], 1)
            anchor = actor.action_head.nll(actor.action_head(features), actions).mean()
            rng = torch.get_rng_state().clone()
            with patch.object(actor.flow, 'sample', side_effect=AssertionError('R0 sample')), \
                 patch.object(actor.refine, 'forward', side_effect=AssertionError('R0 refine')):
                total, logs = actor.losses_given(cond, clean, actions, 0, training_services=object())
            torch.testing.assert_close(total, nf+anchor, rtol=0, atol=0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name in ('l_cons', 'l_act_refine', 'l_plan_quality', 'l_head_exposure'):
                self.assertEqual(logs[name], 0.)
        f = scratch_fixture(learned=False)
        with patch.object(f.flow, 'sample', side_effect=AssertionError('R0 sample')), \
             patch.object(f.refine, 'forward', side_effect=AssertionError('R0 refine')), \
             patch.object(training, 'require_services', side_effect=AssertionError('R0 service')):
            f.ns['_stage2_loop'](1)

    def test_no_services_or_unqualified_services_refuse_before_sampling(self):
        for kind in ('image', 'state'):
            actor, cond, clean, actions = real_actor(kind)
            with patch.object(actor.flow, 'sample', side_effect=AssertionError('must reject before sample')):
                with self.assertRaisesRegex(ValueError, 'qualified'):
                    actor.losses_given(cond, clean, actions, 1)
                svc = services_for(actor.refine, actor.action_head)
                with patch.object(svc.quality_service, 'authorize_training', side_effect=ValueError('A action NMSE >.1')):
                    with self.assertRaisesRegex(ValueError, 'A action NMSE'):
                        actor.losses_given(cond, clean, actions, 1, training_services=svc)
        f = scratch_fixture()
        f.ns['REFINE_TRAINING_SERVICES'] = None
        with patch.dict(f.ns, make_batch=lambda *a, **kw: self.fail('qualification before first batch')):
            with self.assertRaisesRegex(ValueError, 'qualified'):
                f.ns['_stage2_loop'](1)
        f = scratch_fixture()
        f.service.exposure = None
        with self.assertRaisesRegex(ValueError, 'exposure unavailable'):
            f.ns['_stage2_loop'](1)

    def test_invalid_fingerprint_freeze_and_exposure_gradient_routing(self):
        actor, cond, clean, actions = real_actor('state')
        svc = services_for(actor.refine, actor.action_head)
        svc.quality_service.valid[:] = False
        with self.assertRaisesRegex(ValueError, 'all candidates invalid'):
            actor.losses_given(cond, clean, actions, 1, training_services=svc)
        svc.quality_service.valid[:] = True
        original_factory = svc.quality_factory
        def wrong_report(u):
            report = svc.quality_service(u)
            report.teacher_fingerprint = 'wrong-version'
            return report
        svc.quality_factory = lambda sampled, cond: wrong_report
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            actor.losses_given(cond, clean, actions, 1, training_services=svc)
        svc.quality_factory = original_factory
        original_exposure = svc.exposure
        svc.exposure = lambda u, features: (u.sum()*0, dict(valid=False, n_valid=0, n_total=B, reason='no valid labels'))
        with self.assertRaisesRegex(ValueError, 'exposure invalid'):
            actor.losses_given(cond, clean, actions, 1, training_services=svc)
        svc.exposure = original_exposure
        svc.teacher.train()
        with self.assertRaisesRegex(ValueError, 'frozen and eval'):
            actor.losses_given(cond, clean, actions, 1, training_services=svc)
        svc.teacher.eval()
        trace = Trace('state', actor.flow, svc)
        with trace.active():
            total, _ = actor.losses_given(cond, clean, actions, 3, .1, training_services=svc)
        logs = trace.rows[0]['logs']
        params = list(actor.action_head.parameters())
        for total_g, exp_g, anchor_g in zip(grads(total, params), grads(logs['l_head_exposure'], params), grads(logs['l_act_anchor'], params)):
            torch.testing.assert_close(total_g-.25*exp_g, anchor_g, rtol=2e-5, atol=1e-6)
        for g in grads(logs['l_head_exposure'], list(actor.refine.parameters())):
            self.assertEqual(g.abs().sum(), 0)
        self.assertGreater(sum(g.abs().sum() for g in grads(logs['l_cons'], list(actor.refine.parameters()))), 0)
        total.backward()
        self.assertIsNone(clean.grad)
        self.assertTrue(all(p.grad is None for p in svc.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in svc.quality_service.parameters()))

    def test_kernel_float64_cond_dtype_and_cpu_autocast(self):
        # Same v2 float64 recurrence witness, now QualityReport reductions are fp32.
        cond = torch.full((B, C), .3, dtype=torch.float64, requires_grad=True)
        clean = torch.zeros(B, K, D, dtype=torch.float64, requires_grad=True)
        sample = torch.ones_like(clean, requires_grad=True)
        noise = torch.full_like(clean, .2, requires_grad=True)
        weight = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        op = lambda c, u: weight*u + c[:, None, :]
        teacher = lambda c, u: u*.7
        terms = objective.refinement_terms(op, teacher, WiringOracle(), cond, clean, sample, rounds=3, noise=noise)
        terms.quality.backward()
        self.assertGreater(cond.grad.abs().sum(), 0)
        self.assertGreater(weight.grad.abs(), 0)
        self.assertIsNone(sample.grad); self.assertIsNone(noise.grad); self.assertIsNone(clean.grad)
        actor, cond, clean, actions = real_actor('state')
        svc = services_for(actor.refine, actor.action_head)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            total, _ = actor.losses_given(cond.to(torch.bfloat16), clean, actions, 3, training_services=svc)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        for rounds in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                actor.losses_given(cond, clean, actions, rounds)

    def test_structural_checks_and_nonfinite_left_to_scaler(self):
        cond, clean = torch.ones(B, C), torch.zeros(B, K, D)
        oracle = WiringOracle()
        op = lambda c, u: u + c[:, None, :]
        for sample, noise in [(clean.double(), None), (clean[:, :1], None),
                              (None, None), (clean, torch.zeros(B, K, D, dtype=torch.int64))]:
            with self.assertRaises(ValueError):
                objective.refinement_terms(op, op, oracle, cond, clean, sample, rounds=1, noise=noise)
        poisoned = torch.full_like(clean, float('nan'))
        terms = objective.refinement_terms(op, op, oracle, cond, clean, poisoned, rounds=1)
        self.assertTrue(torch.isnan(terms.quality))  # no hot-path finite raise

    def test_scratch_amp_skip_controls_ema_at_actual_call_site(self):
        class ControlledScaler:
            def __init__(self, skipped):
                self.skipped = skipped
            def step(self, opt):
                if opt is not self.skipped:
                    opt.step()
            def get_scale(self):
                return 16.
            def update(self):
                pass
        for skip_main in (False, True):
            f = scratch_fixture(bc_indep=True)
            before = copy.deepcopy(f.service.teacher.state_dict())
            f.ns['_SCALER'] = ControlledScaler(f.opt if skip_main else f.ns['opt_bc'])
            f.ns['_stage2_loop'](1, step_off=3)
            self.assertEqual(f.service.successful_steps, int(not skip_main))
            self.assertEqual(f.service.attempted_steps, 1)
            if skip_main:
                for k, v in f.service.teacher.state_dict().items():
                    torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            else:
                self.assertTrue(any(not torch.equal(v, before[k]) for k, v in f.service.teacher.state_dict().items()))
            print(f'SCRATCH AMP main_skip={skip_main}: actual _amp_step(opt2) -> after_step PASS', flush=True)

    def test_six_semantic_mutations_fail_then_restored_pass_on_real_scratch(self):
        mutations = [
            ('total_missing_quality', training, 'compose',
             'total = nf + anchor + terms.quality +', 'total = nf + anchor +'),
            ('wrong_clean_target', training, 'compose',
             'sampled = adapter.sample().detach()', 'sampled = adapter.sample().detach()\n    sampled = clean.detach()'),
            ('ignore_rounds', objective, 'refinement_terms',
             'for _ in range(rounds):', 'for _ in range(1):'),
            ('zero_noise', training, 'compose',
             'noise = services.noise_for(sampled)', 'noise = torch.zeros_like(services.noise_for(sampled))'),
            ('truncate_bptt', objective, 'refinement_terms',
             'u = un\n', 'u = un.detach()\n'),
            ('detach_cond', objective, 'refinement_terms',
             'refine(cond,', 'refine(cond.detach(),'),
        ]
        for name, module, symbol, old, new in mutations:
            original = getattr(module, symbol)
            source = inspect.getsource(original)
            self.assertIn(old, source)
            scope = dict(module.__dict__)
            exec(compile(source.replace(old, new), f'<semantic-mutant:{name}>', 'exec'), scope)
            mutant = scope[symbol]
            with patch.object(module, symbol, mutant):
                f = scratch_fixture(jitter=.35)
                trace = Trace('scratch', f.flow, f.service)
                with trace.active():
                    with self.assertRaises(AssertionError) as caught:
                        f.ns['_stage2_loop'](1, step_off=3)
            print(f'MUTATION {name}: FAIL (caught semantic witness: {str(caught.exception).splitlines()[0]})', flush=True)
            f = scratch_fixture(jitter=.35)
            trace = Trace('scratch', f.flow, f.service)
            with trace.active():
                f.ns['_stage2_loop'](1, step_off=3)
            self.assertEqual(trace.rows[0]['sample_count'], 1)
            self.assertGreater(trace.rows[0]['noise'].abs().sum(), 0)
            print(f'MUTATION {name}: RESTORED PASS (_stage2_loop -> compose -> objective, R3)', flush=True)

    def test_optimizer_specific_amp_skip_ema_buffers_and_resume(self):
        class WithBuffer(nn.Linear):
            def __init__(self):
                super().__init__(2, 2)
                self.register_buffer('count', torch.tensor(3))
        class ControlledScaler:
            def __init__(self, skip):
                self.skip = skip
            def step(self, opt):
                if opt is not self.skip:
                    opt.step()
        for skip_main in (False, True):
            student = WithBuffer()
            svc = services_for(student, nn.Identity(), jitter=.2)
            bc = nn.Linear(2, 2)
            mainopt, bcopt = torch.optim.SGD(student.parameters(), lr=.1), torch.optim.SGD(bc.parameters(), lr=.1)
            for p in [*student.parameters(), *bc.parameters()]:
                p.grad = torch.ones_like(p)
            before = copy.deepcopy(svc.teacher.state_dict())
            student.count.add_(4)
            scaler = ControlledScaler(mainopt if skip_main else bcopt)
            succeeded = training.optimizer_step(mainopt, scaler)
            bc_succeeded = training.optimizer_step(bcopt, scaler)
            svc.after_step(student, succeeded=succeeded)
            self.assertEqual(succeeded, not skip_main)
            self.assertEqual(bc_succeeded, skip_main)
            for k, v in svc.teacher.state_dict().items():
                expected = before[k] if skip_main else (student.state_dict()[k] if k == 'count' else before[k]*.99+student.state_dict()[k]*.01)
                torch.testing.assert_close(v, expected)
            state = svc.checkpoint_state()
            expected_noise = svc.noise_for(torch.zeros(B, K, D))
            resumed = services_for(student, nn.Identity(), jitter=.2)
            resumed.restore(state)
            torch.testing.assert_close(resumed.noise_for(torch.zeros(B, K, D)), expected_noise, rtol=0, atol=0)
            self.assertEqual(resumed.successful_steps, int(not skip_main))
            self.assertEqual(resumed.attempted_steps, 1)
            for k, v in resumed.teacher.state_dict().items():
                torch.testing.assert_close(v, svc.teacher.state_dict()[k], rtol=0, atol=0)
            for bad in (None, {'metadata': {'objective_version': 'legacy'}}):
                with self.assertRaisesRegex(ValueError, 'migration'):
                    resumed.restore(bad)
            print(f'EMA main_skip={skip_main} bc_skip={not skip_main}: optimizer-specific update, buffers, resume/noise RNG PASS', flush=True)

    def test_real_cpu_gradscaler_distinguishes_optimizers(self):
        for skip_main in (False, True):
            main, bc = nn.Linear(2, 2), nn.Linear(2, 2)
            opt = torch.optim.SGD(main.parameters(), lr=.1)
            bcopt = torch.optim.SGD(bc.parameters(), lr=.1)
            scaler = torch.amp.GradScaler('cpu')
            x = torch.ones(2, 2)
            scaler.scale(main(x).sum() + bc(x).sum()).backward()
            next((main if skip_main else bc).parameters()).grad.fill_(float('inf'))
            scaler.unscale_(opt); scaler.unscale_(bcopt)
            self.assertEqual(training.optimizer_step(opt, scaler), not skip_main)
            self.assertEqual(training.optimizer_step(bcopt, scaler), skip_main)
            previous_scale = scaler.get_scale()
            scaler.update()
            self.assertLess(scaler.get_scale(), previous_scale)
            print(f'REAL CPU GradScaler main_skip={skip_main} bc_skip={not skip_main}: PASS', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
