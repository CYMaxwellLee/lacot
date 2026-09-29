"""CPU-only artifact/binder tests. Toy evidence NEVER qualifies production."""
import ast
import contextlib
import copy
import io
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from lacot import refine_service_builder as builder
from lacot.refine_models import toy_arrays, build_records, DEFAULT_SPLIT_SEED
from lacot.refine_quality import QualityReport, readings_digest
from lacot.refine_training import require_services
from test_refine_wiring_v3 import scratch_fixture

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "experiments/scratch_lacot_rollout.py"
BEGIN = "# BEGIN opt-in W_phi quality builder (exposure remains unavailable).\n"
END = "# END opt-in W_phi quality builder.\n"


def hook_source():
    source = SCRATCH.read_text()
    assert source.count(BEGIN) == source.count(END) == 1
    return source.split(BEGIN)[1].split(END)[0]


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory(prefix="refine-builder-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.directory = Path(cls.tmp.name)
        cls.paths = {}
        for domain in ("pointmaze", "ant"):
            out = cls.directory / domain
            run = subprocess.run([sys.executable, "-m", "experiments.refine_v3.train_offline",
                "--domain", domain, "--toy", "--out", str(out), "--steps", "200",
                "--hidden", "64", "--threads", "1"], cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
            if run.returncode:
                raise AssertionError(run.stdout + run.stderr)
            cls.paths[domain] = out / "oracle.pt"
        cls.saved = torch.load(cls.paths["pointmaze"], map_location="cpu", weights_only=True)
        cls.records = build_records(*toy_arrays("pointmaze", seed=DEFAULT_SPLIT_SEED),
            domain="pointmaze", dataset_hash="a"*64, split_seed=DEFAULT_SPLIT_SEED,
            t_cap=128, max_windows=1024)["test"]

    def load(self, path=None, **kwargs):
        return builder.load_quality_teacher(path or self.paths["pointmaze"],
            domain="pointmaze", diagnostic_cpu=True, **kwargs)

    def build(self, **overrides):
        args = dict(checkpoint=self.paths["pointmaze"], domain="pointmaze",
            teacher=nn.Identity().eval(), decoder=lambda u, s: u,
            intent_inverse=lambda p, a: p, frozen_modules=(nn.Identity().eval(),),
            state_mean=torch.zeros(4), state_scale=torch.ones(4), t_cap=128,
            diagnostic_cpu=True)
        args.update(overrides)
        return builder.build_training_services(**args)

    def batch(self, service):
        state = self.records["state"][:8].clone()
        u = self.records["path"][:8].clone().requires_grad_()
        goal = self.records["path"][:8, -1].clone()
        bound = service.for_batch(state=state, goal=goal, anchors=None)
        return bound.quality_factory(u.detach(), torch.zeros(8, 3)), u

    def test_checkpoint_load_cpu_frozen_eval_fingerprint_rng(self):
        rng = torch.get_rng_state().clone()
        with patch.object(torch, "load", wraps=torch.load) as load:
            service = self.load(expected_fingerprint=self.saved["calibration"]["teacher_fingerprint"])
        self.assertEqual(load.call_args.kwargs, dict(map_location="cpu", weights_only=True))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        service.train()
        self.assertTrue(all(not m.training for m in service.modules()))
        self.assertTrue(all(not p.requires_grad and p.device.type == "cpu" for p in service.parameters()))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.load(expected_fingerprint="0"*64)
        self.load(expected_fingerprint=self.saved["calibration"]["teacher_fingerprint"])
        print("LOAD: valid toy CPU artifact PASS; wrong expected fingerprint REJECT; restored PASS")

    def test_bad_checkpoint_two_states(self):
        cases = {
            "weights": lambda s: s["models"]["world"]["members.0.linear.weight"].add_(1),
            "support": lambda s: s["support_banks"][4].add_(1),
            "readings": lambda s: s["readings"].update(controller_action_nmse=0.),
            "split": lambda s: s["calibration"].update(split_hash="0"*64),
            "version": lambda s: s.update(version="old"),
            "schema": lambda s: s.pop("models"),
            "domain": lambda s: s["metadata"].update(domain="ant"),
        }
        path = self.directory / "mutant.pt"
        for name, mutate in cases.items():
            with self.subTest(name=name):
                saved = copy.deepcopy(self.saved)
                mutate(saved)
                torch.save(saved, path)
                with self.assertRaises(ValueError):
                    self.load(path)
                self.load()
                print(f"ARTIFACT {name}: tampered REJECT -> restored PASS")
        path.write_bytes(b"not a torch checkpoint")
        with self.assertRaises(ValueError):
            self.load(path)
        with self.assertRaises(ValueError):
            self.load(self.directory / "missing.pt")

    def test_qualification_two_states_no_empty_or_nonfinite_pass(self):
        for name, update in {
            "NMSE": lambda r: r["horizons"]["4"]["subspaces"]["xy"].update(world_nmse=1e6),
            "CI": lambda r: r["horizons"]["4"]["subspaces"]["xy"].update(paired_improvement_ci95=[0., 1.]),
            "missing subspace": lambda r: r["horizons"]["4"].update(subspaces={}),
            "unavailable": lambda r: r["horizons"]["4"].update(available=False),
            "null CI": lambda r: r["horizons"]["4"]["subspaces"]["xy"].update(paired_improvement_ci95=[None, None]),
            "split hash": lambda r: r.update(split_hash="0"*64),
        }.items():
            saved = copy.deepcopy(self.saved)
            update(saved["readings"])
            saved["readings_digest"] = readings_digest(saved["readings"])
            path = self.directory / "unqualified.pt"
            torch.save(saved, path)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load(path)
            self.load()
            print(f"QUALIFICATION {name}: insufficient REJECT -> restored PASS")
        saved = copy.deepcopy(self.saved)
        saved["readings"]["controller_action_nmse"] = float("nan")
        torch.save(saved, path)
        with self.assertRaises(ValueError):
            self.load(path)

    def test_toy_never_production_and_all_three_gates_recorded(self):
        with self.assertRaisesRegex(ValueError, "production"):
            builder.load_quality_teacher(self.paths["pointmaze"], domain="pointmaze")
        svc = self.build()
        info = svc.metadata()["quality_teacher"]
        self.assertEqual(info["source_kind"], "toy_offline")
        self.assertIn("A action NMSE >.1", info["training_blocks"])
        self.assertIn("generated validity missing/below threshold", info["training_blocks"])
        self.assertIsNone(svc.exposure)
        with self.assertRaisesRegex(ValueError, "diagnostic/unqualified"):
            require_services(svc)
        # Even hypothetical future full quality authorization cannot fill exposure.
        with patch.object(svc.quality_service, "authorize_training"):
            with self.assertRaisesRegex(ValueError, "exposure unavailable"):
                require_services(svc)
        with self.assertRaisesRegex(ValueError, "requires CPU"):
            self.load(device="cuda")
        print("TRAINING: W toy measurements PASS; A/generated/toy authorization REJECT; exposure NONE")

    def test_revised_pi_validity_are_health_not_gates(self):
        saved = copy.deepcopy(self.saved)
        saved["readings"]["horizons"]["4"]["data_validity"] = 0.
        for metric in saved["readings"]["horizons"]["4"]["subspaces"].values():
            metric["pi90_coverage"] = 0.
        saved["readings_digest"] = readings_digest(saved["readings"])
        path = self.directory / "health-only.pt"
        torch.save(saved, path)
        service = self.load(path)
        self.assertEqual(service.readings["horizons"]["4"]["data_validity"], 0.)
        with self.assertRaisesRegex(ValueError, "diagnostic/unqualified"):
            service.authorize_training(4, same_plan=True)

    def test_quality_fp32_finite_and_gradient_to_u_only(self):
        svc = self.build()
        quality, u = self.batch(svc)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            report = quality(u)
            loss = report.cost[report.valid].mean()
        self.assertIsInstance(report, QualityReport)
        self.assertEqual(report.cost.dtype, torch.float32)
        self.assertEqual(report.cost.shape, (8,))
        self.assertTrue(torch.isfinite(report.cost).all())
        loss.backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertGreater(float(u.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in svc.quality_service.parameters()))
        self.assertEqual(report.teacher_fingerprint, svc.metadata()["teacher_fingerprint"])
        # Input perturbation must change the result; a constant teacher cannot pass.
        shifted = quality(u.detach() + .001)
        self.assertFalse(torch.equal(report.cost, shifted.cost))
        first = next(svc.quality_service.world.parameters())
        first.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen and eval"):
            quality(u)
        first.requires_grad_(False)
        torch.testing.assert_close(quality(u).cost, report.cost)
        print(f"QUALITY: fp32 finite; valid={int(report.valid.sum())}/8; u grad norm={float(u.grad.norm()):.9f}")

    def test_batch_coordinates_anchors_and_multichunk_current_state(self):
        mean, scale = torch.tensor([2., -3., 1., 4.]), torch.tensor([2., 3., 4., 5.])
        state = self.records["state"][:8]
        paths = self.records["path"][:8]
        anc = torch.full_like(paths, .013)
        calls = []
        def decoder(u, current):
            calls.append(current.detach().clone())
            return u
        svc = self.build(state_mean=mean, state_scale=scale, decoder=decoder,
                         intent_inverse=lambda p, a: p+a, horizon=16)
        u = ((paths-mean[:2])/scale[:2] - anc).requires_grad_()
        bound = svc.for_batch(state=(state-mean)/scale,
            goal=(paths[:, -1]-mean[:2])/scale[:2], anchors=anc)
        anc.add_(100)  # Binder must own its anchors, not retain a mutable caller view.
        report = bound.quality_factory(u, torch.zeros(8, 3))(u)
        expected = svc.quality_service.same_plan(paths, state, paths[:, -1], lambda p, s: p, horizon=16)
        torch.testing.assert_close(report.cost, expected.cost, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(calls), 3*4)
        torch.testing.assert_close(calls[0], (state-mean)/scale)
        self.assertFalse(torch.equal(calls[0], calls[1]))
        grad = torch.autograd.grad(report.cost[report.valid].mean(), u)[0]
        self.assertGreater(float(grad.norm()), 0.)
        # A different bound goal cannot leak into the previous closure.
        bound2 = svc.for_batch(state=(state-mean)/scale,
            goal=(paths[:, -1]+.5-mean[:2])/scale[:2], anchors=torch.full_like(paths, .013))
        report2 = bound2.quality_factory(u, torch.zeros(8, 3))(u)
        self.assertFalse(torch.equal(report.cost, report2.cost))
        self.assertEqual(bound.metadata(), svc.metadata())

    def test_real_scratch_decoder_frozen_parameters_keep_latent_gradient(self):
        from lacot.traj_decoder import TrajDecoder
        from lacot.refine_models import freeze
        torch.manual_seed(31)
        decoder = freeze(TrajDecoder(8, 128, num_layers=1, num_heads=2))
        embed = freeze(nn.Linear(2, 8))
        # Keep this random decoder's toy outputs inside the fixed support region;
        # no calibration thresholds/readings are altered to admit the fixture.
        with torch.no_grad():
            decoder.head.weight.mul_(.1)
            decoder.head.bias.zero_()
        ns = dict(torch=torch, DEC_START="hard", u_dec=decoder, s_embed=embed,
                  to_xy=lambda x: x[..., :2], XY_DIM=2, INTENT="", intent_ad=None)
        nodes = [n for n in ast.parse(SCRATCH.read_text()).body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_dec", "_intent_inv")]
        self.assertEqual(len(nodes), 2)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRATCH), "exec"), ns)
        svc = self.build(decoder=ns["_dec"], intent_inverse=ns["_intent_inv"],
                         frozen_modules=(decoder, embed))
        u = torch.randn(8, 4, 8, requires_grad=True)
        bound = svc.for_batch(state=torch.zeros(8,4), goal=torch.full((8,2), .2), anchors=None)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            report = bound.quality_factory(u.detach(), torch.zeros(8,3))(u)
        report.cost[report.valid].mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertGreater(float(u.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for m in (decoder, embed) for p in m.parameters()))
        print(f"REAL _dec/_intent_inv + TrajDecoder: u grad norm={float(u.grad.norm()):.9f}; teacher/decoder grads NONE")

    def test_ant_same_interface_but_refinement_rejected(self):
        svc = self.build(checkpoint=self.paths["ant"], domain="ant",
                         state_mean=torch.zeros(29), state_scale=torch.ones(29))
        self.assertEqual(svc.quality_service.calibration.domain, "ant")
        with self.assertRaisesRegex(ValueError, "ant R>0 unsupported"):
            svc.for_batch(state=torch.zeros(8, 29), goal=torch.zeros(8, 2), anchors=None)
        with self.assertRaises(ValueError):
            require_services(svc)

    def test_metadata_checkpoint_and_early_contract_rejections(self):
        svc = self.build()
        state = svc.checkpoint_state()
        self.assertEqual(state["metadata"], svc.metadata())
        info = state["metadata"]["quality_teacher"]
        for key in ("teacher_fingerprint", "split_hash", "dataset_hash", "readings_digest"):
            self.assertEqual(len(info[key]), 64)
        self.assertTrue(info["frozen"] and info["eval"])
        info["split_hash"] = "altered copy"
        self.assertNotEqual(info, svc.metadata()["quality_teacher"])
        for overrides in (dict(t_cap=32), dict(state_scale=torch.zeros(4)),
                          dict(teacher=nn.Linear(2, 2)), dict(frozen_modules=())):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.build(**overrides)
        with self.assertRaisesRegex(ValueError, "for_batch"):
            svc.quality_factory(None, None)
        quality, u = self.batch(svc)
        with patch.object(svc.quality_service, "same_plan", wraps=svc.quality_service.same_plan) as forward:
            report = forward(u, self.records["state"][:8], self.records["path"][:8,-1], lambda u,s:u)
            report.valid.zero_()
            forward.return_value = report
            with self.assertRaisesRegex(ValueError, "all candidates invalid"):
                quality(u)

    def test_env_missing_bad_and_no_diagnostic_escape(self):
        for env in ({}, {"LACOT_REFINE_QUALITY_CKPT": ""}, {"LACOT_REFINE_QUALITY_CKPT": "  "}):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "is required"):
                builder.build_from_env(environ=env)
        with self.assertRaisesRegex(ValueError, "cannot enable diagnostic"):
            builder.build_from_env(environ={"LACOT_REFINE_QUALITY_CKPT": str(self.paths["pointmaze"])}, diagnostic_cpu=True)
        with patch.object(builder, "build_training_services", return_value="bound") as build:
            self.assertEqual(builder.build_from_env(environ={"LACOT_REFINE_QUALITY_CKPT": "/explicit/oracle.pt"}, domain="ant"), "bound")
            build.assert_called_once_with(checkpoint="/explicit/oracle.pt", domain="ant")

    def test_scratch_hook_opt_in_cons_intent_and_original_decoder(self):
        svc = self.build()
        calls = []
        ns = dict(LEARNED_REFINE=1, STEPS2=1, CONT_STEPS=0, INTENT="", CONS="ema",
            os=os, ENV_FAMILY="pointmaze", refine_ema=svc.teacher, device="cpu", EMA_M=.996,
            u_dec=nn.Identity().eval(), s_embed=None, vq=None, fsq=None, intent_ad=None,
            MU=torch.zeros(4), SD=torch.ones(4), T_CAP=128, REFINE_TRAINING_SERVICES=None,
            _q=lambda u: u+.1, _dec=lambda u,s: calls.append((u,s)) or u+.2,
            _intent_inv=lambda p,a:p)
        with patch.dict(os.environ, {"LACOT_REFINE_QUALITY_CKPT": str(self.paths["pointmaze"])}, clear=True):
            for key, value, error in (("CONS", "self", "LACOT_CONS=ema"), ("INTENT", "residual", "M2")):
                with self.subTest(key=key), patch.object(builder, "build_from_env") as build:
                    with self.assertRaisesRegex(ValueError, error):
                        exec(hook_source(), {**ns, key:value})
                    build.assert_not_called()
            with patch.object(builder, "build_from_env", return_value=svc) as build, contextlib.redirect_stdout(io.StringIO()):
                exec(hook_source(), ns)
                args = build.call_args.kwargs
                self.assertIs(ns["REFINE_TRAINING_SERVICES"], svc)
                u, state = torch.ones(2,3,2), torch.ones(2,4)
                torch.testing.assert_close(args["decoder"](u,state), u+.3)
                self.assertIs(calls[0][1], state)
                self.assertIs(args["intent_inverse"], ns["_intent_inv"])
                self.assertEqual(args["ema_decay"], .996)
            # Real hook rejects the unqualified toy artifact, without a mock builder.
            with self.assertRaisesRegex(ValueError, "production"):
                exec(hook_source(), ns)
        for value in ("", str(self.directory / "absent.pt")):
            with patch.dict(os.environ, {"LACOT_REFINE_QUALITY_CKPT": value}, clear=True):
                with self.assertRaises(ValueError):
                    exec(hook_source(), ns)
        for overrides in (dict(LEARNED_REFINE=0), dict(STEPS2=0, CONT_STEPS=0)):
            with patch.dict(os.environ, {"LACOT_REFINE_QUALITY_CKPT": "/unreadable"}, clear=True):
                with patch.object(builder, "build_from_env") as build:
                    exec(hook_source(), {**ns, **overrides})
                    build.assert_not_called()

    def test_flag_off_bitwise_full_scratch_loop_16_cells(self):
        source = SCRATCH.read_text()
        baseline = source.split(BEGIN)[0] + source.split(END)[1]
        # Own hook is the ONLY difference; complete old/new stage2 ASTs execute.
        for fsq, intent, bc, div in itertools.product((False, True), (False, True), (False, True), (0., .07)):
            results = []
            for text in (baseline, source):
                f = scratch_fixture(learned=False, fsq=fsq, intent=intent, bc_indep=bc, div=div, source=text)
                f.ns.update(os=os, STEPS2=1, CONT_STEPS=0)
                before_rng = torch.get_rng_state().clone()
                if text == source:
                    with patch.dict(os.environ, {}, clear=True), patch.object(builder, "build_from_env") as build:
                        exec(hook_source(), f.ns)
                        build.assert_not_called()
                    self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                with contextlib.redirect_stdout(io.StringIO()):
                    f.ns["_stage2_loop"](1)
                tensors = [*f.backwards, torch.get_rng_state()]
                for module in [*f.ns["f_mods"], f.ns["bc_head"]]:
                    for p in module.parameters():
                        tensors.extend([p.detach().clone(), None if p.grad is None else p.grad.clone()])
                results.append(tensors)
            self.assertEqual(len(results[0]), len(results[1]))
            self.assertGreater(len(results[0]), 2)
            for old, new in zip(*results):
                if old is None:
                    self.assertIsNone(new)
                else:
                    self.assertTrue(torch.equal(old, new))
        # Env absent + learned enabled must neither import/load nor allow training.
        f = scratch_fixture(learned=True)
        f.ns.update(os=os, STEPS2=1, CONT_STEPS=0, REFINE_TRAINING_SERVICES=None)
        with patch.dict(os.environ, {}, clear=True), patch.object(builder, "build_from_env") as build:
            exec(hook_source(), f.ns)
            build.assert_not_called()
            with self.assertRaisesRegex(ValueError, "requires qualified"):
                f.ns["_stage2_loop"](1)
        print("FLAG-OFF: 16/16 loss/grad/weights/RNG BITWISE PASS; missing env R>0 REJECT")


if __name__ == "__main__":
    unittest.main()
