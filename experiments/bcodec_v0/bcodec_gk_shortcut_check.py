#!/usr/bin/env python
"""行為字典 v0 補跑 shortcut 檢查：G×K 平行頭版。跟 bcodec_shortcut_check.py
同一套判準與量法，只有 idx 形狀是 (n,G) 不是 (n,)——「異碼」定義為每一組都
各自加一個 1..K-1 的隨機 offset（保證每組都變、整個 tuple 必然不同）。

【判準，開工前寫死，跟 v0 單本版同一套】
ratio = median(metric_A 同碼異s_t) / median(metric_B 異碼同s_t)
metric_B median<=1e-12 時為 SUSPECT（量測退化，不判通過）。
ratio<=0.5 通過／0.5~1.0 灰帶／>=1.0 不通過（疑似 shortcut）。
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bcodec_common as bc  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
RATIO_PASS, RATIO_GRAY = 0.5, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    print(f"=== bcodec-gk shortcut check  ckpt={os.path.basename(args.ckpt)} n={args.n_samples} seed={args.seed} ===")
    print(f"判準（開工前寫死）：metric_B median<=1e-12 為 SUSPECT；ratio<={RATIO_PASS} 通過，{RATIO_PASS}~{RATIO_GRAY} 灰帶，>= {RATIO_GRAY} 不通過")

    model, cfg, obs_mu, obs_sd = bc.load_bcodec_gk_ckpt(args.ckpt)
    G, K = cfg["G"], cfg["K"]
    data_path = os.path.join(args.data_dir, f"{bc.DATASET_NAME}.npz")
    data = bc.load_bcodec_segments(data_path, seg_len=cfg["seg_len"], split_seed=cfg["split_seed"],
                                   val_frac=cfg["val_frac"])
    val_idx = data["val_idx"]
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(val_idx))
    pick = rng.choice(val_idx, size=n, replace=False)

    obs_mu_np, obs_sd_np = obs_mu.numpy(), obs_sd.numpy()
    s_seg_norm = bc._norm_seg(data["s_seg"][pick], obs_mu_np, obs_sd_np, cfg["seg_len"], cfg["obs_dim"])
    a_seg = data["a_seg"][pick]
    obs_t_norm = ((data["obs_start"][pick] - obs_mu_np) / obs_sd_np).astype(np.float32)

    perm = rng.permutation(n)
    if n < 2:
        raise ValueError("cross-state permutation requires at least 2 samples")
    fixed = perm == np.arange(n)
    for _ in range(1000):
        if not fixed.any():
            break
        perm = rng.permutation(n)
        fixed = perm == np.arange(n)
    else:
        raise RuntimeError("failed to draw a derangement in 1000 attempts")
    other_obs_t_norm = obs_t_norm[perm]

    with torch.no_grad():
        x_s = torch.from_numpy(s_seg_norm)
        x_a = torch.from_numpy(a_seg)
        x_o = torch.from_numpy(obs_t_norm)
        x_o_other = torch.from_numpy(other_obs_t_norm)

        z_e = model.encode(x_s, x_a)
        _, idx, _ = model.vq(z_e, training=False)  # (n, G)

        a_same, _ = model.decode_codes(idx, x_o)
        a_cross_state, _ = model.decode_codes(idx, x_o_other)

        gen = torch.Generator().manual_seed(args.seed + 1)
        offset = torch.randint(1, K, (n, G), generator=gen)  # 每組都加非零 offset，保證整個 tuple 不同
        idx_diff = (idx + offset) % K
        a_diff_code, _ = model.decode_codes(idx_diff, x_o)

        metric_a = ((a_cross_state - a_same) ** 2).mean(dim=1).numpy()
        metric_b = ((a_diff_code - a_same) ** 2).mean(dim=1).numpy()

    med_a, med_b = float(np.median(metric_a)), float(np.median(metric_b))
    ratio = med_a / max(med_b, 1e-12)
    if med_b <= 1e-12:
        verdict = "SUSPECT（metric_B 太小，量測退化）"
    elif ratio <= RATIO_PASS:
        verdict = "通過（碼對 â 影響 > s_t，非明顯 shortcut）"
    elif ratio < RATIO_GRAY:
        verdict = "灰帶（訊號模糊）"
    else:
        verdict = "不通過（疑似 shortcut：s_t 對 â 的影響 >= 碼）"

    print(f"metric_A（同碼異 s_t）median={med_a:.6f} mean={metric_a.mean():.6f}")
    print(f"metric_B（異碼同 s_t）median={med_b:.6f} mean={metric_b.mean():.6f}")
    print(f"ratio = {ratio:.4f}  =>  {verdict}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = dict(ckpt=args.ckpt, dict_cfg=cfg, n_samples=n, seed=args.seed,
              criterion=dict(ratio_pass=RATIO_PASS, ratio_gray=RATIO_GRAY, metric_b_min=1e-12),
              metric_a=dict(median=med_a, mean=float(metric_a.mean())),
              metric_b=dict(median=med_b, mean=float(metric_b.mean())),
              ratio=ratio, verdict=verdict)
    bc.save_json(out, os.path.join(args.out_dir, f"{args.tag}_shortcut_check.json"))
    print(f"saved: {os.path.join(args.out_dir, f'{args.tag}_shortcut_check.json')}")


if __name__ == "__main__":
    main()
