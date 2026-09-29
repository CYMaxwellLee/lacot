"""Slurm stage launcher and gates. Evaluation stays in scratch_lacot_rollout.py."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "experiments/_workorders/ucontrast1"
SIGMAS = (0.05, 0.10, 0.20)
CKPTS = {s: f"results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s{s}.pt"
         for s in (33, 35)}
PROTOCOL = "official-flat-R1-reached-T1-EMA"
SWEEP_SALT = 2_000_011
SWEEP_EPISODE_OFFSET = 40
ANCHOR_R1 = .350
ANCHOR_TOLERANCE = .08
SMOKE_R1_RANGE = (.15, .6)
CKPT_SHA256 = {
    33: "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787",
    35: "ba5009ec197d1d11b9902a2b92917d1ca9198e8b32882c442638380786ab0d7d",
}


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def code_hashes():
    paths = [ROOT / "experiments/scratch_lacot_rollout.py", WORK / "collector.py", WORK / "run.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def checkpoint(seed, provenance):
    """Frozen input asset: exact path and SHA256, with no training-time gate."""
    entry = provenance["checkpoints"][str(seed)]
    if (entry["path"] != CKPTS[seed] or entry["sha256"] != CKPT_SHA256[seed]
            or entry["training_started_unix"] is not None):
        raise ValueError("Frozen checkpoint path/SHA256/provenance mismatch")
    pinned = ROOT / CKPTS[seed]
    actual = sha(pinned)
    if actual != CKPT_SHA256[seed]:
        raise ValueError("Frozen checkpoint bytes differ from pinned SHA256")
    return dict(path=CKPTS[seed], sha256=actual, mtime=pinned.stat().st_mtime,
                training_started_unix=None, training_start_note="frozen input asset; not applicable")


def calibration_contract(provenance):
    anchor = provenance["anchor"]
    if (anchor["protocol"] != PROTOCOL or anchor["expected_r1"] != ANCHOR_R1
            or anchor["tolerance"] != ANCHOR_TOLERANCE
            or anchor["episodes_per_task"] != 40
            or anchor["source"] != "j_signal noclimb s33"):
        raise ValueError("J-signal R1 anchor contract mismatch")
    return anchor


def flagoff_contract(path, seed):
    """Check lead-side GPU baseline/current bitwise trajectory evidence."""
    proof = read(path)
    pairs = proof.get("pairs", [])
    baseline = subprocess.check_output(
        ["git", "show", "HEAD:experiments/scratch_lacot_rollout.py"], cwd=ROOT)
    if (proof.get("code_hashes") != code_hashes()
            or proof.get("ckpt_sha256") != CKPT_SHA256[seed]
            or proof.get("device") != "cuda"
            or proof.get("baseline_source_sha256") != hashlib.sha256(baseline).hexdigest()
            or proof.get("current_source_sha256") != code_hashes()["experiments/scratch_lacot_rollout.py"]
            or len(pairs) != 3
            or len({(p["task"], p["episode"]) for p in pairs}) != 3
            or any(not p["baseline_trajectory_sha256"]
                   or p["baseline_trajectory_sha256"] != p["current_trajectory_sha256"]
                   for p in pairs)):
        raise ValueError("GPU flag-off three-question bitwise proof missing/mismatched")
    return dict(path=str(path), sha256=sha(path))


def independent_oracle(tasks, draws):
    """Recompute from raw draw successes without calling collector.summarize."""
    if not tasks:
        raise ValueError("Empty oracle tasks")
    sums = [0] * draws
    quality = [0] * draws
    for task in tasks:
        raw = sorted(task["draws"], key=lambda d: d["draw"])
        if [d["draw"] for d in raw] != list(range(draws)):
            raise ValueError("Missing oracle draw")
        bits = [int(d["success"]) for d in raw]
        if bits != task["success_bits"]:
            raise ValueError("Stored success bits differ from draws")
        seen = 0
        for i, bit in enumerate(bits):
            quality[i] += bit
            seen |= bit
            sums[i] += seen
    n = len(tasks)
    return ({str(k + 1): sums[k] / n for k in range(draws)},
            [q / n for q in quality], sum(quality))


def gate(path, stage):
    obj = read(path)
    if (obj.get("stage") != stage or obj.get("status") != "PASS"
            or obj.get("code_hashes") != code_hashes()):
        raise ValueError(f"Missing, failed, or stale {stage} gate")
    if sha(obj["result_path"]) != obj["result_sha256"]:
        raise ValueError(f"Changed {stage} evidence")
    return obj


def launch(args):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU eval requires sbatch allocation (gpu:1)")
    stage = args.stage
    provenance = read(args.provenance)
    anchor = calibration_contract(provenance)
    flagoff = flagoff_contract(args.flagoff_proof, 33)
    if stage != "calibration":
        cg = gate(args.calibration_gate, "calibration")
        if cg["provenance_sha256"] != sha(args.provenance):
            raise ValueError("Calibration provenance changed")
        if cg["flagoff_proof"] != flagoff:
            raise ValueError("GPU flag-off evidence changed")
    if stage in ("smoke", "main"):
        mg = gate(args.mutant_gate, "mutant")
        if mg["provenance_sha256"] != sha(args.provenance):
            raise ValueError("Mutant provenance changed")
    seed = 33 if stage == "calibration" else args.seed
    ck = checkpoint(seed, provenance)
    if stage in ("smoke", "main") and mg["checkpoint"]["sha256"] != ck["sha256"]:
        raise ValueError("Sigma-zero mutant used a different checkpoint")
    arm, sigma, draws, episodes = "A", 0., 8, 5
    if stage == "calibration":
        draws, episodes = 1, anchor["episodes_per_task"]
    elif stage == "mutant":
        arm = "B"
    elif stage == "smoke":
        if not 0 <= args.index <= 3:
            raise ValueError("Smoke array index must be 0..3")
        arm = "A" if args.index == 0 else "B"
        sigma = 0. if args.index == 0 else SIGMAS[args.index - 1]
    elif stage == "main":
        # A review artifact; this work order never invokes this stage.
        selection = read(args.selection)
        if (selection["code_hashes"] != code_hashes() or selection["status"] != "SELECTED"
                or selection["sigma"] not in SIGMAS or selection["seed"] != seed
                or selection["ckpt_sha256"] != ck["sha256"]):
            raise ValueError("Missing/stale sigma selection for this checkpoint")
        for path, hashed in selection["inputs"].items():
            if sha(path) != hashed:
                raise ValueError("Sigma sweep evidence changed")
        if args.index not in (0, 1):
            raise ValueError("Main array index must be 0..1")
        episodes, arm = 40, ("A" if args.index == 0 else "B")
        sigma = 0. if arm == "A" else selection["sigma"]
    episode_offset = SWEEP_EPISODE_OFFSET if stage in ("mutant", "smoke") else 0
    stream_salt = SWEEP_SALT if stage in ("mutant", "smoke") else 0
    tag = f"ucontrast1_{stage}_s{seed}_{arm}_sig{sigma:g}"
    outdir = Path(args.outdir).resolve()
    result = outdir / f"{tag}.json"
    if result.exists():
        raise ValueError(f"Refusing to overwrite {result}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LACOT_")}
    config = dict(ENV="pointmaze-large-stitch-v0", CONS="self", K=8, COND=256,
                  CHUNK=4, TCAP=128, ENC_OBJ="recon_ictr", LEARNED_REFINE=0,
                  COND_DROP=0.1, BC_INDEP=1, TEACHER_MIX=0.5, WARMUP=500,
                  DEC_START="soft", LOAD_EMA=1,
                  CONT_TRAIN=0, DEV_EVAL=0, PREREQ=0, EVAL_EPISODES=episodes,
                  EVAL_RS=1, GRAD_REFINE=0, SUBGOAL="", FINISH_R=0, INTENT="",
                  U_SOURCE="flow", SUB_ESEL=0, INTENT_GUID_W=0, DEC_ANCHOR=0,
                  BON_MODE="plan", BON_N=0, BOOT_DATA="", S1_FROM="", FLOW_PROBE=0,
                  DIAG_DUMP=0, LOAD_CKPT=CKPTS[seed], SEED=seed, BOOT_TAG=tag,
                  OUT_DIR=str(outdir), ORACLE_ARM=arm, ORACLE_SIGMA=sigma,
                  ORACLE_DRAWS=draws, ORACLE_OUT=str(result),
                  ORACLE_EPISODE_OFFSET=episode_offset, ORACLE_STREAM_SALT=stream_salt)
    env.update({"LACOT_" + k: str(v) for k, v in config.items()})
    metadata = dict(stage=stage, arm=arm, sigma=sigma, ckpt=CKPTS[seed], tag=tag,
                    checkpoint=ck, code_hashes=code_hashes(),
                    flagoff_proof=flagoff,
                    provenance_sha256=sha(args.provenance),
                    submitted_job=os.environ["SLURM_JOB_ID"], config=config,
                    started_unix=time.time(), training_use_prohibited=True)
    write(outdir / f"{tag}.preflight.json", metadata)
    subprocess.run([sys.executable, "-u", "experiments/scratch_lacot_rollout.py"],
                   cwd=ROOT, env=env, check=True)
    data = read(result)
    if data["n_tasks"] != 5 * episodes or data["n_draws"] != 5 * episodes * draws:
        raise ValueError("Abnormal smoke result shape")
    if stage in ("calibration", "mutant"):
        passed = (abs(data["per_draw_quality"][0] - ANCHOR_R1) <= ANCHOR_TOLERANCE
                  if stage == "calibration" else data["negative_mutant_passed"])
        record = dict(metadata, status="PASS" if passed else "BLOCKED",
                      result_path=str(result), result_sha256=sha(result),
                      observed=data["per_draw_quality"][0] if stage == "calibration" else data["pooled_oracle"],
                      anchor=anchor if stage == "calibration" else None)
        write(outdir / f"{stage}-gate.json", record)
        if not passed:
            raise ValueError(f"{stage} failed; STOP, no subsequent stages")


def select(args):
    """CPU-only aggregation. Select only among the three 25-question B bins."""
    files = [Path(p).resolve() for p in args.results]
    rows = [read(p) for p in files]
    if len(rows) != 4:
        raise ValueError("Need A and all three B sigma bins")
    a = [r for r in rows if r["arm"] == "A"]
    b = [r for r in rows if r["arm"] == "B"]
    if len(a) != 1 or sorted(r["sigma"] for r in b) != list(SIGMAS):
        raise ValueError("Incomplete/duplicate sigma sweep")
    zero = read(gate(args.mutant_gate, "mutant")["result_path"])
    if zero["ckpt"] != CKPTS[args.seed]:
        raise ValueError("Sigma-zero mutant checkpoint differs from sweep")
    def paired(r):
        return [(t["task"], t["episode"], [(d["env_seed"], d["stream_seed"],
                  d["initial_sha256"], d["goal_sha256"]) for d in t["draws"]]) for t in r["tasks"]]
    proof = []
    for path, row in zip(files, rows):
        p = read(path.with_name(path.stem + ".preflight.json"))
        if p["code_hashes"] != code_hashes() or p["stage"] != "smoke":
            raise ValueError("Not current smoke evidence")
        proof.append(p)
        if (row["n_tasks"] != 25 or row["n_draws"] != 200 or row["draws_per_task"] != 8
                or row["ckpt"] != CKPTS[args.seed] or paired(row) != paired(a[0])):
            raise ValueError("Unpaired/wrong-size smoke evidence")
        oracle, quality, n_success = independent_oracle(row["tasks"], 8)
        if (oracle != row["oracle_at_k"] or oracle["8"] != row["pooled_oracle"]
                or quality != row["per_draw_quality"] or n_success != row["n_success"]):
            raise ValueError("Oracle differs from raw draw successes")
        if any(not (SWEEP_EPISODE_OFFSET <= t["episode"] < SWEEP_EPISODE_OFFSET + 5)
               for t in row["tasks"]):
            raise ValueError("Sweep overlaps main episode IDs")
    if len({p["checkpoint"]["sha256"] for p in proof}) != 1:
        raise ValueError("Different checkpoint bytes")
    for row in b:
        if [t["draws"][0]["first_u_sha256"] for t in row["tasks"]] != [
                t["draws"][0]["first_u_sha256"] for t in a[0]["tasks"]]:
            raise ValueError("B first u not paired with A draw 0")
        if [(t["task"], t["episode"]) for t in zero["tasks"]] != [
                (t["task"], t["episode"]) for t in row["tasks"]]:
            raise ValueError("Positive-sigma and sigma-zero questions differ")
        changed = sum(any(d["trajectory_sha256"] != z["draws"][i]["trajectory_sha256"]
                          for i, d in enumerate(t["draws"]))
                      for t, z in zip(row["tasks"], zero["tasks"]))
        if changed == 0:
            raise ValueError("B positive sigma matched sigma-zero trajectories")
    if not SMOKE_R1_RANGE[0] <= a[0]["per_draw_quality"][0] <= SMOKE_R1_RANGE[1]:
        raise ValueError("A smoke draw-0 R1 outside sanity band")
    winner = max(b, key=lambda r: (r["pooled_oracle"], -r["sigma"]))
    write(args.output, dict(status="SELECTED", seed=args.seed, sigma=winner["sigma"],
                           metric="pooled oracle@8", tie_break="smaller sigma",
                           ckpt_sha256=proof[0]["checkpoint"]["sha256"],
                           inputs={str(p): sha(p) for p in files}, code_hashes=code_hashes(),
                           note="Selection only; no A-vs-B verdict", training_use_prohibited=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration", "mutant", "smoke", "main", "select"))
    parser.add_argument("--provenance", default=str(WORK / "provenance.json"))
    parser.add_argument("--flagoff-proof", default=str(WORK / "smoke/flagoff-proof.json"))
    parser.add_argument("--calibration-gate", default=str(WORK / "smoke/calibration/calibration-gate.json"))
    parser.add_argument("--mutant-gate", default=str(WORK / "smoke/mutant/mutant-gate.json"))
    parser.add_argument("--selection", default=str(WORK / "smoke/sigma-selection.json"))
    parser.add_argument("--seed", type=int, choices=(33, 35), default=33)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--outdir", default=str(WORK / "smoke"))
    parser.add_argument("--results", nargs="*", default=[])
    parser.add_argument("--output", default=str(WORK / "smoke/sigma-selection.json"))
    args = parser.parse_args()
    try:
        select(args) if args.stage == "select" else launch(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
