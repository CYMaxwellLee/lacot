#!/usr/bin/env bash
# Lead executes this script. Do not run from the implementation workspace.
set -euo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target="${1:-$HOME/Projects/lacot-u-frozen}"
baseline_commit=5b256d4bafdd9f1fdee0673fb147cae49d24e6c3
baseline_sha=21f4cf306a2987241114fe41339eb830f88249bf3fa565dab35fe69016fa6705
frozen_sha=276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
source_copy="$source_root/experiments/_workorders/ucontrast1/smoke/mutants/M9/scratch_lacot_rollout.py"

test ! -e "$target" || { echo "BLOCKED: target already exists: $target" >&2; exit 2; }
test "$(git -C "$source_root" show "$baseline_commit:experiments/scratch_lacot_rollout.py" | sha256sum | cut -d' ' -f1)" = "$baseline_sha" || {
  echo 'BLOCKED: baseline commit source SHA differs' >&2; exit 2;
}
test "$(sha256sum "$source_copy" | cut -d' ' -f1)" = "$frozen_sha" || {
  echo 'BLOCKED: frozen source copy SHA differs' >&2; exit 2;
}
for seed in 33 35; do
  ckpt="results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s${seed}.pt"
  test -f "$source_root/$ckpt" || { echo "BLOCKED: missing $ckpt" >&2; exit 2; }
  expected=88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787
  if [ "$seed" = 35 ]; then
    expected=ba5009ec197d1d11b9902a2b92917d1ca9198e8b32882c442638380786ab0d7d
  fi
  test "$(sha256sum "$source_root/$ckpt" | cut -d' ' -f1)" = "$expected" || {
    echo "BLOCKED: frozen checkpoint SHA differs: $ckpt" >&2; exit 2;
  }
done

git -C "$source_root" worktree add --detach "$target" "$baseline_commit"
mkdir -p "$target/experiments/_workorders" "$target/results"
cp -a "$source_root/experiments/_workorders/ucontrast1" "$target/experiments/_workorders/"
cp -a "$source_root/experiments/_workorders/ucontrast2" "$target/experiments/_workorders/"
cp "$source_copy" "$target/experiments/scratch_lacot_rollout.py"
for seed in 33 35; do
  ckpt="ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s${seed}.pt"
  ln -s "$source_root/results/$ckpt" "$target/results/$ckpt"
done

cd "$target"
test "$(sha256sum experiments/scratch_lacot_rollout.py | cut -d' ' -f1)" = "$frozen_sha"
test "$(sha256sum experiments/_workorders/ucontrast1/run.py | cut -d' ' -f1)" = e78e2a704317bf47bde81d6397c07dde4ddc9158a0f20df949baca1b3e7b7a06
test "$(sha256sum experiments/_workorders/ucontrast1/collector.py | cut -d' ' -f1)" = 7d8e2bf3b8760388ffa0cdad5460b4ea298c738325d5bf25bc084a0a248c26a5
sha256sum experiments/scratch_lacot_rollout.py experiments/_workorders/ucontrast1/{run.py,collector.py}
echo "READY: frozen worktree $target"
