#!/usr/bin/env python3
"""CPU offline training/calibration entrypoint. GPU runs need a later F5 workorder.

Toy: python -m experiments.refine_v3.train_offline --domain pointmaze --toy --out DIR
Real: --data /explicit/pointmaze-large-stitch-v0.npz --out DIR.
The repository DATA-MANIFEST.json is the dataset whitelist. No evaluation input option.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from lacot.refine_models import (VERSION, DEFAULT_SPLIT_SEED, load_offline_npz, build_records,
    toy_arrays, train_models, Normalizer, ConsequenceEnsemble, PairedController, freeze)
from lacot.refine_quality import OfflineCalibrator, Calibration, QualityService, qualification_reasons, readings_digest


def save_checkpoint(path, models, calibrator, readings, *, hidden):
    torch.save({"version": VERSION, "models": {k: v.state_dict() for k, v in models.items()},
                "hidden": hidden, "t_cap": models["controller"].t_cap,
                "metadata": calibrator.train["metadata"], "calibration": asdict(calibrator.artifact),
                "support_banks": calibrator.banks, "readings": readings,
                "readings_digest": calibrator._readings_digest}, path)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["version"] != VERSION:
        raise ValueError("checkpoint representation version mismatch")
    w = saved["models"]["world"]
    sn = Normalizer(w["state_norm.mean"], w["state_norm.scale"])
    an = Normalizer(w["action_norm.mean"], w["action_norm.scale"])
    models = {"world": ConsequenceEnsemble(sn, an, hidden=saved["hidden"]),
              "obs_only": ConsequenceEnsemble(sn, an, hidden=saved["hidden"], obs_only=True),
              "controller": PairedController(sn, an, saved["t_cap"], hidden=saved["hidden"])}
    for name, model in models.items():
        model.load_state_dict(saved["models"][name]); freeze(model)
    cal = OfflineCalibrator.__new__(OfflineCalibrator)
    cal.models = models; cal.world = models["world"]; cal.train = {"metadata": saved["metadata"]}
    cal.banks = saved["support_banks"]; cal.reference_nll = None; cal.reference_fingerprint = None
    cal.artifact = Calibration(**saved["calibration"])
    cal._readings_digest = saved["readings_digest"]
    if readings_digest(saved["readings"]) != cal._readings_digest:
        raise ValueError("checkpoint qualification readings digest mismatch")
    if cal.artifact.reference_fingerprint is not None:
        raise ValueError("reference flow checkpoint requires its explicit frozen adapter")
    # Construction verifies model+normalizer fingerprints before any caller can use it.
    QualityService(cal, readings=saved["readings"], diagnostic=True)
    return models, cal, saved["readings"]


def run(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.toy:
        if args.data:
            raise ValueError("toy cannot be mixed with a dataset path")
        arrays = toy_arrays(args.domain, seed=args.split_seed)
        digest = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
        records = build_records(*arrays, domain=args.domain, dataset_hash=digest,
                                split_seed=args.split_seed, t_cap=args.t_cap, max_windows=args.max_windows)
    else:
        if not args.data:
            raise ValueError("offline ingestion requires explicit --data from the manifest")
        arrays = load_offline_npz(args.data, expected_domain=args.domain)
        digest = arrays.dataset_hash
        records = build_records(arrays, split_seed=args.split_seed,
                                t_cap=args.t_cap, max_windows=args.max_windows)
    source = records["train"]["metadata"]["source_kind"]
    print(json.dumps({"phase": "data", "domain": args.domain, "source": source,
                      "windows": {k: len(v["state"]) for k, v in records.items()}, "device": "cpu"}), flush=True)
    models, curves = train_models(records["train"], steps=args.steps, hidden=args.hidden,
                                  seed=args.seed, batch_size=args.batch_size)
    for name, curve in curves.items():
        print(json.dumps({"phase": "fit", "model": name, **curve}), flush=True)
    calibrator = OfflineCalibrator(models, records["train"])
    artifact = calibrator.fit(records["calibration"])
    readings = calibrator.evaluate(records["test"], seed=args.seed)
    service = QualityService(calibrator, readings=readings, diagnostic=True)
    r = records["test"]
    report = service.report(r["state"][:8], r["actions"][:8, :4], r["path"][:8, -1])
    # Both directions: forward candidate->actions->consequences->report;
    # backward quality->actions/plan with all teacher parameters frozen.
    actions = r["actions"][:8, :4].clone().requires_grad_()
    action_report = service.report(r["state"][:8], actions, r["path"][:8, -1])
    grad = torch.autograd.grad(action_report.cost.sum(), actions)[0]
    wiring = {"action_to_quality_backward": bool(torch.isfinite(grad).all() and grad.abs().sum() > 0),
              "teachers_frozen": all(not p.requires_grad for m in models.values() for p in m.parameters()),
              "same_plan_backward": None, "ant_refine_rejected": None}
    if args.domain == "pointmaze":
        plan = r["path"][:8].clone().requires_grad_()
        plan_report = service.same_plan(plan, r["state"][:8], r["path"][:8, -1], lambda u, s: u)
        g = torch.autograd.grad(plan_report.cost.sum(), plan)[0]
        wiring["same_plan_backward"] = bool(torch.isfinite(g).all() and g.abs().sum() > 0)
    else:
        try:
            service.same_plan(r["path"][:8], r["state"][:8], r["path"][:8, -1], lambda u, s: u)
        except ValueError:
            wiring["ant_refine_rejected"] = True
    losses_decreased = all(v["final"] < v["initial"] for v in curves.values())
    summary = {"scope": "CPU toy wiring only" if args.toy else "offline measurements only",
               "domain": args.domain, "seed": args.seed, "split_seed": args.split_seed,
               "split_hash": records["train"]["metadata"]["split_hash"],
               "partial_chunks_dropped": {k: v["metadata"]["partial_chunks_dropped"] for k, v in records.items()},
               "device": "cpu", "steps_per_model": args.steps,
               "ensemble_members": 3, "horizons_physical_steps": [4, 16, 32],
               "t_cap_interpolation_points": args.t_cap, "chunk_original_actions": 4,
               "dataset_hash": digest, "curves": curves, "losses_decreased": losses_decreased,
               "readings": readings, "calibration": artifact.json(), "quality_report": report.json(),
               "wiring": wiring, "qualification_blocks": {str(h): qualification_reasons(readings, h, require_controller=args.domain == "pointmaze") for h in (4, 16, 32)},
               "physical_fall_freeze_labels": None, "counterfactual_accuracy": None,
               "elapsed_seconds": time.monotonic()-started}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fail rather than overwrite another experiment's evidence.
    with (out/"report.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with (out/"metadata.json").open("x") as stream:
        json.dump({k: {"metadata": v["metadata"], "episodes": sorted(set(v["episode_id"].tolist())),
                       "n_windows": len(v["state"]), "short_windows": {str(h): int((~v["mask"][:, :h].all(1)).sum()) for h in (4, 16, 32)}} for k, v in records.items()}, stream, indent=2)
    save_checkpoint(out/"oracle.pt", models, calibrator, readings, hidden=args.hidden)
    restored, restored_cal, _ = load_checkpoint(out/"oracle.pt")
    with torch.no_grad():
        torch.testing.assert_close(restored["world"](r["state"][:8], r["actions"][:8, :4]), models["world"](r["state"][:8], r["actions"][:8, :4]))
    if not losses_decreased or not wiring["action_to_quality_backward"] or not wiring["teachers_frozen"]:
        raise RuntimeError("CPU smoke failed; inspect report")
    if args.domain == "pointmaze" and not wiring["same_plan_backward"]:
        raise RuntimeError("same-plan gradient wiring failed")
    print(json.dumps({"phase": "complete", "report": str(out/"report.json"), "losses_decreased": losses_decreased,
                      "checkpoint_roundtrip": True, "wiring": wiring, "seconds": summary["elapsed_seconds"],
                      "scientific_verdict": "not_run"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["pointmaze", "ant"], required=True)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=1024)
    parser.add_argument("--t-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 8:
        parser.error("CPU smoke threads must be 1..8")
    run(args)


if __name__ == "__main__":
    main()
