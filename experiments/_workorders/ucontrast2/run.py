"""Frozen-code launcher for s35 replication and preregistered N=64 subset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments._workorders.ucontrast1 import run as old
from experiments._workorders.ucontrast2 import rollout_n64

OLD = ROOT / "experiments/_workorders/ucontrast1"
WORK = ROOT / "experiments/_workorders/ucontrast2"
S35_CAL_ROOT = Path("/archive/cymaxwelllee/ucontrast1/s35-calibration")
EVIDENCE_ROOT = Path("/home/cymaxwelllee/Projects/lacot")
QUESTION_SEED = 20260929
PINNED_HASHES = {
    "experiments/scratch_lacot_rollout.py": rollout_n64.PIN,
    "experiments/_workorders/ucontrast1/collector.py":
        "7d8e2bf3b8760388ffa0cdad5460b4ea298c738325d5bf25bc084a0a248c26a5",
    "experiments/_workorders/ucontrast1/run.py":
        "e78e2a704317bf47bde81d6397c07dde4ddc9158a0f20df949baca1b3e7b7a06",
}


def sample_questions():
    """Portable SHA256 keyed ranking: a uniform 50-subset of the 5 x 40 pool."""
    pool = [(task, episode) for task in range(1, 6) for episode in range(40)]
    def rank(pair):
        task, episode = pair
        return hashlib.sha256(f"{QUESTION_SEED}:{task}:{episode}".encode("ascii")).digest()
    return sorted(sorted(pool, key=rank)[:50])


def registered_questions():
    manifest = old.read(WORK / "n64-questions.json")
    expected = [dict(task=t, episode=e) for t, e in sample_questions()]
    if (manifest.get("seed") != QUESTION_SEED
            or manifest.get("method") != "SHA256 ASCII seed:task:episode; 50 smallest; task/episode sorted"
            or manifest.get("questions") != expected):
        raise ValueError("N64 preregistered question manifest differs from sampler")
    return sample_questions()


def relocated(path):
    path = Path(path)
    try:
        return ROOT / path.relative_to(EVIDENCE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Evidence path outside recorded root: {path}") from exc


def evidence_gate(path, stage, provenance_sha, flagoff_sha):
    obj = old.read(path)
    if (obj.get("stage") != stage or obj.get("status") != "PASS"
            or obj.get("code_hashes") != PINNED_HASHES
            or obj.get("provenance_sha256") != provenance_sha
            or obj.get("flagoff_proof", {}).get("sha256") != flagoff_sha
            or old.sha(relocated(obj["result_path"])) != obj["result_sha256"]):
        raise ValueError(f"Missing, failed, or stale {stage} gate")
    return obj


def validate_inputs():
    if old.code_hashes() != PINNED_HASHES:
        raise ValueError("BLOCKED: frozen code hash differs from pinned gate")
    provenance_path = OLD / "provenance.json"
    provenance = old.read(provenance_path)
    old.calibration_contract(provenance)
    flagoff_path = OLD / "smoke/flagoff-proof.json"
    old.flagoff_contract(flagoff_path, 33)
    provenance_sha = old.sha(provenance_path)
    flagoff_sha = old.sha(flagoff_path)
    cg = evidence_gate(OLD / "smoke/calibration/calibration-gate.json",
                       "calibration", provenance_sha, flagoff_sha)
    mg = evidence_gate(OLD / "smoke/mutant/mutant-gate.json",
                       "mutant", provenance_sha, flagoff_sha)
    if (cg["checkpoint"]["sha256"] != old.CKPT_SHA256[33]
            or mg["checkpoint"]["sha256"] != old.CKPT_SHA256[33]):
        raise ValueError("s33 calibration/mutant checkpoint mismatch")
    selection = old.read(OLD / "smoke/sigma-selection.json")
    if (selection.get("status") != "SELECTED"
            or selection.get("seed") != 33 or selection.get("sigma") != .05
            or selection.get("ckpt_sha256") != old.CKPT_SHA256[33]
            or selection.get("code_hashes") != PINNED_HASHES
            or len(selection.get("inputs", {})) != 4):
        raise ValueError("Missing/stale selected sigma=0.05 evidence")
    for path, digest in selection["inputs"].items():
        if old.sha(relocated(path)) != digest:
            raise ValueError(f"Changed sigma sweep evidence: {path}")
    return provenance, provenance_sha, flagoff_sha


def s35_mutant_gate(cal_root, provenance_sha, flagoff_sha):
    path = cal_root / "mutant/B/ucontrast2_s35_mutant_s35_B_sig0.json"
    gate = old.read(path.with_name("mutant-gate.json"))
    preflight = old.read(path.with_suffix(".preflight.json"))
    result = old.read(path)
    if (gate.get("status") != "PASS" or gate.get("stage") != "mutant"
            or gate.get("code_hashes") != PINNED_HASHES
            or gate.get("provenance_sha256") != provenance_sha
            or gate.get("flagoff_sha256") != flagoff_sha
            or gate.get("checkpoint", {}).get("sha256") != old.CKPT_SHA256[35]
            or gate.get("result_path") != str(path)
            or gate.get("result_sha256") != old.sha(path)
            or gate.get("preflight_sha256") != old.sha(path.with_suffix(".preflight.json"))
            or preflight.get("stage") != "mutant"
            or result.get("negative_mutant_passed") is not True):
        raise ValueError("Missing/stale s35 sigma-zero mutant gate")
    return path.with_name("mutant-gate.json")


def s35_selection(cal_root, provenance_sha, flagoff_sha):
    s35_mutant_gate(cal_root, provenance_sha, flagoff_sha)
    path = cal_root / "sigma-selection.json"
    selection = old.read(path)
    expected = {
        str(cal_root / "sweep/A/ucontrast2_s35_sweep_s35_A_sig0.json"),
        *(str(cal_root / f"sweep/B-sig{sigma:g}/ucontrast2_s35_sweep_s35_B_sig{sigma:g}.json")
          for sigma in old.SIGMAS),
    }
    if (selection.get("status") != "SELECTED" or selection.get("seed") != 35
            or selection.get("sigma") not in old.SIGMAS
            or selection.get("metric") != "pooled oracle@8"
            or selection.get("tie_break") != "smaller sigma"
            or selection.get("ckpt_sha256") != old.CKPT_SHA256[35]
            or selection.get("code_hashes") != PINNED_HASHES
            or set(selection.get("inputs", {})) != expected):
        raise ValueError("Missing/stale s35 independent sigma selection")
    for result_path, digest in selection["inputs"].items():
        result_path = Path(result_path)
        if old.sha(result_path) != digest:
            raise ValueError(f"Changed s35 sigma sweep evidence: {result_path}")
        verify_s35_sweep_result(result_path, provenance_sha, flagoff_sha)
    b_rows = [old.read(cal_root / f"sweep/B-sig{s:g}/ucontrast2_s35_sweep_s35_B_sig{s:g}.json")
              for s in old.SIGMAS]
    winner = max(b_rows, key=lambda row: (row["pooled_oracle"], -row["sigma"]))
    if selection["sigma"] != winner["sigma"]:
        raise ValueError("s35 selection differs from pooled oracle@8 winner")
    return selection


def verify_s35_sweep_result(result_path, provenance_sha, flagoff_sha):
    preflight = result_path.with_suffix(".preflight.json")
    verified = result_path.with_suffix(".verified.json")
    flight = old.read(preflight)
    receipt = old.read(verified)
    if (flight.get("stage") != "smoke" or flight.get("mode") != "s35"
            or flight.get("provenance_sha256") != provenance_sha
            or flight.get("flagoff_sha256") != flagoff_sha
            or flight.get("checkpoint", {}).get("sha256") != old.CKPT_SHA256[35]
            or receipt.get("status") != "PASS"
            or receipt.get("result_sha256") != old.sha(result_path)
            or receipt.get("preflight_sha256") != old.sha(preflight)):
        raise ValueError("Missing/stale s35 sweep receipt or preflight")


def reserve_outdir(base, arm):
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ("A", "B") or not p.is_dir() for p in base.iterdir()):
        raise ValueError("OUT base contains pre-existing unexpected files")
    outdir = base / arm
    outdir.mkdir(exist_ok=False)
    return outdir


def launch(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU eval requires sbatch allocation (gpu:1)")
    if args.index not in (0, 1):
        raise ValueError("Array index must be 0 or 1")
    provenance, provenance_sha, flagoff_sha = validate_inputs()
    seed = 33 if args.mode == "n64" else 35
    ck = old.checkpoint(seed, provenance)
    arm = "A" if args.index == 0 else "B"
    sigma = 0. if arm == "A" else (.05 if args.mode == "n64" else
                                     s35_selection(Path(args.calibration_root),
                                                   provenance_sha, flagoff_sha)["sigma"])
    if args.mode == "s35" and not args.smoke and arm == "A":
        s35_selection(Path(args.calibration_root), provenance_sha, flagoff_sha)
    smoke = args.smoke
    draws = 2 if args.mode == "n64" and smoke else 64 if args.mode == "n64" else 8
    episodes = 5 if smoke else 40
    stage = "smoke" if smoke else "main"
    questions = registered_questions() if args.mode == "n64" and not smoke else None
    derived_hash = rollout_n64.source_hash() if questions is not None else None
    tag = f"ucontrast2_{args.mode}_{stage}_s{seed}_{arm}_sig{sigma:g}"
    # A and B get distinct exclusive-write directories. A repeat of either
    # index, including an incomplete previous attempt, is refused before work.
    outdir = reserve_outdir(args.outdir, arm)
    result = outdir / f"{tag}.json"
    config = dict(ENV="pointmaze-large-stitch-v0", CONS="self", K=8, COND=256,
                  CHUNK=4, TCAP=128, ENC_OBJ="recon_ictr", LEARNED_REFINE=0,
                  COND_DROP=.1, BC_INDEP=1, TEACHER_MIX=.5, WARMUP=500,
                  DEC_START="soft", LOAD_EMA=1, CONT_TRAIN=0, DEV_EVAL=0,
                  PREREQ=0, EVAL_EPISODES=episodes, EVAL_RS=1, GRAD_REFINE=0,
                  SUBGOAL="", FINISH_R=0, INTENT="", U_SOURCE="flow",
                  SUB_ESEL=0, INTENT_GUID_W=0, DEC_ANCHOR=0, BON_MODE="plan",
                  BON_N=0, BOOT_DATA="", S1_FROM="", FLOW_PROBE=0,
                  DIAG_DUMP=0, LOAD_CKPT=old.CKPTS[seed], SEED=seed,
                  BOOT_TAG=tag, OUT_DIR=str(outdir), ORACLE_ARM=arm,
                  ORACLE_SIGMA=sigma, ORACLE_DRAWS=draws,
                  ORACLE_OUT=str(result), ORACLE_EPISODE_OFFSET=40 if smoke else 0,
                  ORACLE_STREAM_SALT=old.SWEEP_SALT if smoke else 0)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LACOT_")}
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({"LACOT_" + k: str(v) for k, v in config.items()})
    if questions is not None:
        env["LACOT_UCONTRAST2_QUESTIONS_JSON"] = json.dumps(
            [dict(task=t, episode=e) for t, e in questions], separators=(",", ":"))
    source = ("experiments/_workorders/ucontrast2/rollout_n64.py"
              if questions is not None else "experiments/scratch_lacot_rollout.py")
    metadata = dict(stage=stage, mode=args.mode, arm=arm, sigma=sigma,
                    checkpoint=ck, code_hashes=old.code_hashes(),
                    derived_source_sha256=derived_hash,
                    launcher_sha256=old.sha(__file__),
                    provenance_sha256=provenance_sha, flagoff_sha256=flagoff_sha,
                    submitted_job=os.environ["SLURM_JOB_ID"], config=config,
                    question_seed=QUESTION_SEED if questions else None,
                    sampled_questions=[dict(task=t, episode=e) for t, e in questions]
                    if questions else None,
                    started_unix=time.time(), training_use_prohibited=True)
    if args.mode == "s35":
        metadata["selection_sha256"] = old.sha(
            Path(args.calibration_root) / "sigma-selection.json")
    old.write(outdir / f"{tag}.preflight.json", metadata)
    subprocess.run([sys.executable, "-u", source], cwd=ROOT, env=env, check=True)
    data = old.read(result)
    count = 50 if questions else 5 * episodes
    if (data["n_tasks"] != count or data["n_draws"] != count * draws
            or data["draws_per_task"] != draws
            or data["ckpt"] != old.CKPTS[seed]
            or list(data)[-3:] != ["n_tasks", "n_draws", "n_success"]):
        raise ValueError("Abnormal result shape/schema")
    if args.mode == "s35" and (data["arm"] != arm or data["sigma"] != sigma):
        raise ValueError("s35 result arm/sigma differs from selected plan")
    pairs = [(t["task"], t["episode"]) for t in data["tasks"]]
    expected = (questions if questions is not None else
                [(t, e) for t in range(1, 6)
                 for e in range(40, 45) if smoke] if smoke else
                [(t, e) for t in range(1, 6) for e in range(40)])
    if pairs != expected:
        raise ValueError("Result question list differs from preregistration")
    if (questions is not None
            and (data.get("sampled_questions") != metadata["sampled_questions"]
                 or data.get("question_seed") != QUESTION_SEED)):
        raise ValueError("N64 question list missing from result JSON")
    oracle, quality, successes = old.independent_oracle(data["tasks"], draws)
    if (oracle != data["oracle_at_k"] or quality != data["per_draw_quality"]
            or successes != data["n_success"] or oracle[str(draws)] != data["pooled_oracle"]):
        raise ValueError("Oracle differs from raw draws")
    old.write(outdir / f"{tag}.verified.json",
              dict(status="PASS", result_sha256=old.sha(result),
                   preflight_sha256=old.sha(outdir / f"{tag}.preflight.json")))


def s35_calibration(args):
    cal_root = Path(args.outdir).resolve()
    if args.s35_stage == "select":
        provenance, provenance_sha, flagoff_sha = validate_inputs()
        old.checkpoint(35, provenance)
        gate = s35_mutant_gate(cal_root, provenance_sha, flagoff_sha)
        results = [cal_root / "sweep/A/ucontrast2_s35_sweep_s35_A_sig0.json"]
        results += [cal_root / f"sweep/B-sig{s:g}/ucontrast2_s35_sweep_s35_B_sig{s:g}.json"
                    for s in old.SIGMAS]
        for result_path in results:
            verify_s35_sweep_result(result_path, provenance_sha, flagoff_sha)
        old.select(argparse.Namespace(results=[str(p) for p in results],
                                      mutant_gate=str(gate), seed=35,
                                      output=str(cal_root / "sigma-selection.json")))
        s35_selection(cal_root, provenance_sha, flagoff_sha)
        return
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU eval requires sbatch allocation (gpu:1)")
    provenance, provenance_sha, flagoff_sha = validate_inputs()
    ck = old.checkpoint(35, provenance)
    if args.s35_stage == "mutant":
        if args.index not in (None, 0):
            raise ValueError("s35 mutant index must be 0")
        arm, sigma, stage = "B", 0., "mutant"
        outdir = reserve_outdir(cal_root / "mutant", arm)
    else:
        if args.index not in range(4):
            raise ValueError("s35 sweep index must be 0..3")
        arm = "A" if args.index == 0 else "B"
        sigma = 0. if arm == "A" else old.SIGMAS[args.index - 1]
        stage = "smoke"
        s35_mutant_gate(cal_root, provenance_sha, flagoff_sha)
        label = "A" if arm == "A" else f"B-sig{sigma:g}"
        outdir = cal_root / "sweep" / label
        outdir.mkdir(parents=True, exist_ok=False)
    tag = f"ucontrast2_s35_{args.s35_stage}_s35_{arm}_sig{sigma:g}"
    result = outdir / f"{tag}.json"
    config = dict(ENV="pointmaze-large-stitch-v0", CONS="self", K=8, COND=256,
                  CHUNK=4, TCAP=128, ENC_OBJ="recon_ictr", LEARNED_REFINE=0,
                  COND_DROP=.1, BC_INDEP=1, TEACHER_MIX=.5, WARMUP=500,
                  DEC_START="soft", LOAD_EMA=1, CONT_TRAIN=0, DEV_EVAL=0,
                  PREREQ=0, EVAL_EPISODES=5, EVAL_RS=1, GRAD_REFINE=0,
                  SUBGOAL="", FINISH_R=0, INTENT="", U_SOURCE="flow",
                  SUB_ESEL=0, INTENT_GUID_W=0, DEC_ANCHOR=0, BON_MODE="plan",
                  BON_N=0, BOOT_DATA="", S1_FROM="", FLOW_PROBE=0,
                  DIAG_DUMP=0, LOAD_CKPT=old.CKPTS[35], SEED=35,
                  BOOT_TAG=tag, OUT_DIR=str(outdir), ORACLE_ARM=arm,
                  ORACLE_SIGMA=sigma, ORACLE_DRAWS=8, ORACLE_OUT=str(result),
                  ORACLE_EPISODE_OFFSET=40, ORACLE_STREAM_SALT=old.SWEEP_SALT)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LACOT_")}
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({"LACOT_" + k: str(v) for k, v in config.items()})
    metadata = dict(stage=stage, mode="s35", arm=arm, sigma=sigma,
                    checkpoint=ck, code_hashes=old.code_hashes(),
                    launcher_sha256=old.sha(__file__),
                    provenance_sha256=provenance_sha, flagoff_sha256=flagoff_sha,
                    submitted_job=os.environ["SLURM_JOB_ID"], config=config,
                    started_unix=time.time(), training_use_prohibited=True)
    preflight = outdir / f"{tag}.preflight.json"
    old.write(preflight, metadata)
    subprocess.run([sys.executable, "-u", "experiments/scratch_lacot_rollout.py"],
                   cwd=ROOT, env=env, check=True)
    data = old.read(result)
    questions = [(t, e) for t in range(1, 6) for e in range(40, 45)]
    if (data["arm"] != arm or data["sigma"] != sigma
            or data["ckpt"] != old.CKPTS[35] or data["n_tasks"] != 25
            or data["n_draws"] != 200 or data["draws_per_task"] != 8
            or [(t["task"], t["episode"]) for t in data["tasks"]] != questions
            or list(data)[-3:] != ["n_tasks", "n_draws", "n_success"]):
        raise ValueError("Abnormal s35 calibration result")
    oracle, quality, successes = old.independent_oracle(data["tasks"], 8)
    if (oracle != data["oracle_at_k"] or quality != data["per_draw_quality"]
            or successes != data["n_success"] or oracle["8"] != data["pooled_oracle"]):
        raise ValueError("s35 calibration oracle differs from raw draws")
    if args.s35_stage == "mutant":
        passed = data.get("negative_mutant_passed") is True
        old.write(outdir / "mutant-gate.json",
                  dict(stage="mutant", status="PASS" if passed else "BLOCKED",
                       code_hashes=old.code_hashes(), checkpoint=ck,
                       provenance_sha256=provenance_sha, flagoff_sha256=flagoff_sha,
                       result_path=str(result), result_sha256=old.sha(result),
                       preflight_sha256=old.sha(preflight)))
        if not passed:
            raise ValueError("s35 sigma-zero mutant failed; STOP")
    else:
        old.write(outdir / f"{tag}.verified.json",
                  dict(status="PASS", result_sha256=old.sha(result),
                       preflight_sha256=old.sha(preflight)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("n64", "s35"))
    parser.add_argument("--s35-stage", choices=("mutant", "sweep", "select"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--calibration-root", default=str(S35_CAL_ROOT))
    args = parser.parse_args()
    try:
        if args.s35_stage:
            if args.mode != "s35" or args.smoke:
                raise ValueError("s35 calibration stages require mode=s35 without --smoke")
            s35_calibration(args)
        else:
            launch(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
