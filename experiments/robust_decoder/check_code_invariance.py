#!/usr/bin/env python
"""驗收標準 #2：code 不變斷言 —— fine-tune 後 encoder 對 10 個抽樣段的 code
與原 ckpt 完全一致（freeze 真的 freeze 了）。

【在證明什麼、判準是什麼】
證明：train_robust_decoder.py 的 freeze 是真的（不是「大機率沒動到」）。
判準：(a) 對 10 個固定抽樣段（seed=0，來自 train split），4 顆 fine-tune 後的
      encode_idx() 輸出跟原 ckpt 逐一相等；(b) 更嚴的版本 —— encoder 與 vq
      的整個 state_dict 逐 tensor bit-exact 相等（不是機率上很像，是完全相等）。
      任何一項不通過就印出差異、exit(1) —— 這是 sanity 對照①的前置：
      如果 code 都不一樣了，教材字串就變了，跟 .554 的對照邏輯直接垮掉。

⛔ 不改動任何既有檔案；只 import 上一層 walk_verify 的 wv_common。
⛔ CPU-only（跟原 P0/teacher_relay 同慣例，10 個段的 forward 是毫秒級）。
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.normpath(os.path.join(HERE, "..", "walk_verify"))
sys.path.insert(0, WV_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402 - 既有共用模組，未修改

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
DEFAULT_BASE_CKPT = os.path.join(WV_DIR, "results", "p0_dict_v1_L4K32_50k.pt")
RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_TAGS = ["sigma0", "sigma0.1", "sigma0.35", "sigma1.0"]
SAMPLE_SEED = 0
N_SAMPLE = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=str, default=DEFAULT_BASE_CKPT)
    ap.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS))
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "code_invariance.json"))
    args = ap.parse_args()
    tags = args.tags.split(",")

    print(f"=== code 不變斷言：{tags} vs base={os.path.basename(args.base_ckpt)} "
          f"(n_sample={N_SAMPLE}, sample_seed={SAMPLE_SEED}) ===")

    base, base_cfg = wv.load_dict_v1(args.base_ckpt)

    raw = wv.load_npz(args.data_dir, DATASET, "train")
    seg_starts = wv.cut_segments(raw, base_cfg["seg_len"])
    segs, cond = wv.build_seg_tensors(raw, seg_starts, base_cfg["seg_len"])
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = rng.choice(len(segs), size=N_SAMPLE, replace=False)
    x = torch.from_numpy(segs[pick])
    idx_base = base.encode_idx(x)
    print(f"抽樣段 index（seg_starts 陣列的 index，非資料 global index）：{pick.tolist()}")
    print(f"base code idx: {idx_base.tolist()}")

    base_sd = base.state_dict()
    frozen_keys = [k for k in base_sd if k.startswith("encoder.") or k.startswith("vq.")]

    results = {}
    all_pass = True
    for tag in tags:
        ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_{tag}.pt")
        if not os.path.exists(ckpt_path):
            print(f"⛔ 找不到 {ckpt_path}，這顆還沒訓練完 / 路徑不對，停手回報。")
            results[tag] = dict(status="MISSING_CKPT", path=ckpt_path)
            all_pass = False
            continue
        ft, ft_cfg = wv.load_dict_v1(ckpt_path)
        idx_ft = ft.encode_idx(x)
        code_match = bool(torch.equal(idx_base, idx_ft))

        ft_sd = ft.state_dict()
        diffs = [k for k in frozen_keys if not torch.equal(base_sd[k], ft_sd[k])]
        weights_bitexact = len(diffs) == 0
        dec_keys = [k for k in base_sd if k.startswith("decoder.")]
        n_dec_changed = sum(1 for k in dec_keys if not torch.equal(base_sd[k], ft_sd[k]))

        status = "PASS" if (code_match and weights_bitexact) else "FAIL"
        if status == "FAIL":
            all_pass = False
        results[tag] = dict(
            status=status, code_match=code_match, code_idx_ft=idx_ft.tolist(),
            encoder_vq_weights_bitexact=weights_bitexact, changed_frozen_keys=diffs,
            n_decoder_tensors_changed=n_dec_changed, n_decoder_tensors_total=len(dec_keys),
            sigma_xy=ft_cfg.get("finetune", {}).get("sigma_xy"))
        print(f"[{tag}] code_match={code_match}  encoder/vq bitexact={weights_bitexact} "
              f"(changed={diffs})  decoder tensors changed={n_dec_changed}/{len(dec_keys)}  "
              f"=> {status}")

    print(f"\n=== 總結：{'PASS' if all_pass else 'FAIL'}（{sum(1 for r in results.values() if r.get('status')=='PASS')}/"
          f"{len(tags)} 顆通過）===")
    wv.save_json(dict(base_ckpt=args.base_ckpt, n_sample=N_SAMPLE, sample_seed=SAMPLE_SEED,
                      sample_pick_idx=pick.tolist(), base_code_idx=idx_base.tolist(),
                      results=results, all_pass=all_pass), args.out)
    print(f"saved: {args.out}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
