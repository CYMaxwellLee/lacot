"""Static checks for the six J-signal eval jobs; never starts a rollout."""

import re
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/j_signal_trial/run_三臂.sbatch"


def test_j_trial_config():
    script = SCRIPT.read_text()
    assert "#SBATCH --array=0-5" in script
    assert 'case "$((SLURM_ARRAY_TASK_ID % 3))" in' in script
    assert 'case "$((SLURM_ARRAY_TASK_ID / 3))" in' in script
    arms = re.findall(
        r"^  ([012])\) ARM=(\w+); GRAD_REFINE=([01]); BON_N=(\d+); GRAD_MODE=(\w+); SEL_N=(\d+) ;;$",
        script, re.M,
    )
    assert arms == [
        ("0", "noclimb", "0", "0", "climb", "8"),
        ("1", "climb", "1", "0", "climb", "8"),
        ("2", "select", "1", "0", "select", "8"),
    ]
    assert re.findall(r"^  ([01])\) CKPT_SEED=(\d+) ;;$", script, re.M) == [
        ("0", "33"), ("1", "35")
    ]
    jobs = [(arms[i % 3][1], ("33", "35")[i // 3]) for i in range(6)]
    assert len(set(jobs)) == 6
    assert 'CKPT="results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s${CKPT_SEED}.pt"' in script
    assert 'LACOT_GRAD_REFINE="$GRAD_REFINE"' in script
    assert 'LACOT_BON_N="$BON_N"' in script
    assert 'LACOT_GRAD_MODE="$GRAD_MODE"' in script
    assert 'LACOT_SEL_N="$SEL_N"' in script
    assert 'LACOT_BOOT_TAG="jsignal_${ARM}"' in script
    assert 'LACOT_OUT_DIR="$OUT"' in script
    assert 'OUT=results/j_signal_trial' in script
    assert 'LACOT_SEED="$CKPT_SEED"' in script
    assert 'LACOT_LOAD_CKPT="$CKPT"' in script
    assert 'LACOT_EVAL_EPISODES=40' in script
    assert 'LACOT_EVAL_RS=1' in script
    assert 'LACOT_ENV=pointmaze-large-stitch-v0' in script
    assert 'LACOT_LEARNED_REFINE=0' in script
    assert 'LACOT_LOAD_EMA=1 LACOT_CONT_TRAIN=0' in script
    assert "wall = self.wall_depth(pts).mean(1)" in script


if __name__ == "__main__":
    test_j_trial_config()
    print("PASS: six J-signal jobs have isolated arm and seed identities")
