"""CPU-only paired readout after both GPU arms have verified results."""
import argparse
from math import comb
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments._workorders.ucontrast1 import run as old
from experiments._workorders.ucontrast2 import run as launcher


def load_arm(base, mode, arm):
    seed = 33 if mode == "n64" else 35
    selection = None
    if mode == "s35":
        _, provenance_sha, flagoff_sha = launcher.validate_inputs()
        selection = launcher.s35_selection(launcher.S35_CAL_ROOT,
                                           provenance_sha, flagoff_sha)
    selected_sigma = .05 if mode == "n64" else selection["sigma"]
    sigma = "0" if arm == "A" else f"{selected_sigma:g}"
    tag = f"ucontrast2_{mode}_main_s{seed}_{arm}_sig{sigma}"
    directory = base / arm
    result = directory / f"{tag}.json"
    verified = old.read(directory / f"{tag}.verified.json")
    preflight = directory / f"{tag}.preflight.json"
    if (verified["status"] != "PASS" or verified["result_sha256"] != old.sha(result)
            or verified["preflight_sha256"] != old.sha(preflight)):
        raise ValueError(f"Missing or changed verified evidence for {arm}")
    flight = old.read(preflight)
    if (flight["mode"] != mode or flight["stage"] != "main"
            or flight["arm"] != arm or flight["code_hashes"] != launcher.PINNED_HASHES
            or flight["checkpoint"]["sha256"] != old.CKPT_SHA256[seed]):
        raise ValueError(f"Wrong preflight evidence for {arm}")
    if mode == "s35" and flight.get("selection_sha256") != old.sha(
            launcher.S35_CAL_ROOT / "sigma-selection.json"):
        raise ValueError("s35 selection changed after main preflight")
    row = old.read(result)
    if (row["arm"] != arm or row["sigma"] != (0. if arm == "A" else selected_sigma)
            or row["ckpt"] != old.CKPTS[seed]):
        raise ValueError(f"Wrong arm/sigma/checkpoint for {arm}")
    draws = 64 if mode == "n64" else 8
    oracle, quality, successes = old.independent_oracle(row["tasks"], draws)
    if (row["oracle_at_k"] != oracle or row["per_draw_quality"] != quality
            or row["n_success"] != successes or row["pooled_oracle"] != oracle[str(draws)]):
        raise ValueError(f"Raw draws disagree with summary for {arm}")
    return row, result


def analyze(mode, base):
    a, apath = load_arm(base, mode, "A")
    b, bpath = load_arm(base, mode, "B")
    n = 50 if mode == "n64" else 200
    draws = 64 if mode == "n64" else 8
    if a["n_tasks"] != n or b["n_tasks"] != n or a["n_draws"] != n * draws or b["n_draws"] != n * draws:
        raise ValueError("Wrong experiment size")
    pairs_a = [(t["task"], t["episode"]) for t in a["tasks"]]
    pairs_b = [(t["task"], t["episode"]) for t in b["tasks"]]
    if pairs_a != pairs_b:
        raise ValueError("Unpaired question lists")
    if mode == "n64" and (a["sampled_questions"] != b["sampled_questions"]
                          or a["sampled_questions"] !=
                          [dict(task=t, episode=e) for t, e in pairs_a]
                          or pairs_a != launcher.registered_questions()
                          or a.get("question_seed") != launcher.QUESTION_SEED
                          or b.get("question_seed") != launcher.QUESTION_SEED):
        raise ValueError("N64 question manifest mismatch")
    for ta, tb in zip(a["tasks"], b["tasks"]):
        for da, db in zip(ta["draws"], tb["draws"]):
            for key in ("env_seed", "stream_seed", "initial_sha256", "goal_sha256"):
                if da[key] != db[key]:
                    raise ValueError(f"A/B unpaired {key} at {(ta['task'], ta['episode'])}")
        if ta["draws"][0]["first_u_sha256"] != tb["draws"][0]["first_u_sha256"]:
            raise ValueError("A/B draw-zero first u differs")
    a_only = sum(any(x["success_bits"]) and not any(y["success_bits"])
                 for x, y in zip(a["tasks"], b["tasks"]))
    b_only = sum(any(y["success_bits"]) and not any(x["success_bits"])
                 for x, y in zip(a["tasks"], b["tasks"]))
    both = sum(any(x["success_bits"]) and any(y["success_bits"])
               for x, y in zip(a["tasks"], b["tasks"]))
    neither = n - a_only - b_only - both
    discordant = a_only + b_only
    a_unlocked = sum(not t["success_bits"][0] and any(t["success_bits"][1:])
                     for t in a["tasks"])
    b_unlocked = sum(not t["success_bits"][0] and any(t["success_bits"][1:])
                     for t in b["tasks"])
    p_one_sided = (sum(comb(discordant, i) for i in range(a_only, discordant + 1))
                   / (2 ** discordant)) if discordant else 1.
    delta_pp = 100 * (a["pooled_oracle"] - b["pooled_oracle"])
    summary = dict(mode=mode, n_questions=n, draws_per_question=draws,
                   a_result=str(apath), a_sha256=old.sha(apath),
                   b_result=str(bpath), b_sha256=old.sha(bpath),
                   a_oracle=a["pooled_oracle"], b_oracle=b["pooled_oracle"],
                   delta_pp=delta_pp, a_only=a_only, b_only=b_only,
                   both=both, neither=neither, unlocked_a=a_unlocked,
                   unlocked_b=b_unlocked, mcnemar_one_sided_p=p_one_sided,
                   training_use_prohibited=True)
    if mode == "n64":
        summary["criterion"] = "A-B >= +5pp and paired one-sided McNemar p<.05"
        summary["u_stands"] = delta_pp >= 5 - 1e-9 and p_one_sided < .05
        summary["otherwise"] = "retain 2026-09-28 not-established verdict"
    else:
        summary["criterion"] = "replication measurement only; no pass threshold"
        summary["direction"] = "A>B" if delta_pp > 0 else "A=B" if delta_pp == 0 else "A<B"
        summary["unlock_structure"] = f"{a_unlocked}v{b_unlocked}"
        summary["reference_2026_09_28"] = dict(delta_pp=3.0, paired_discordant="6:0",
                                                  unlock_structure="9v2")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("n64", "s35"))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    base = Path(args.outdir).resolve()
    try:
        old.write(base / f"{args.mode}-paired-summary.json", analyze(args.mode, base))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")


if __name__ == "__main__":
    main()
