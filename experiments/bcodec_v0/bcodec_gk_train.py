#!/usr/bin/env python
"""行為字典 v0 補跑（主人 2026-09-12 裁）：平行頭版（G×K，跟昨天 G16K16/G32K32
完全同容量）× (s,a) 輸入／雙頭 (â,ŝ) 輸出。

在證明什麼：同容量下，行為詞 (s,a) 是否 >= 動作詞——這次 G,K,D 都跟昨天 G×K
純 action 版一模一樣（G16K16: G=16,K=16,D=8,64bits；G32K32: G=32,K=32,D=8,
160bits），只有 encoder 輸入／decoder 輸出／loss 這三處改法跟 v0 一樣（見
bcodec_common.py 檔頭）。λs=0 臂輸入仍是 (s,a)、只是 loss 不算 recon_s——用來
跟昨天純 action 版比「輸入端加 s」的效果；λs=1 臂再對比 λs=0＝「輸出端重建 s」
的效果（空白格②主消融）。

用法：
  python bcodec_gk_train.py --g 16 --k 16 --lam-s 1.0 --steps 50000
  python bcodec_gk_train.py --g 32 --k 32 --lam-s 0.0 --steps 50000

seed 公式：seed = 70000 + G*1000 + K*10 + (1 if lam_s>0 else 0)
split_seed 固定 42（跟昨天 G16K16/G32K32、跟 v0 全部同一套切分）。

⛔ CPU-only。OOM 教訓（v0 的 K8192 單本兩顆炸過，見 handoff）已經在
bcodec_common.chunked_forward 修好、這裡直接沿用；平行頭單組 K 只有 16/32，
預期不會再炸，但還是走 chunked path 保險。
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import bcodec_common as bc

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")


def fmt_lam(lam_s):
    return ("%.1f" % lam_s).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, required=True, choices=[16, 32])
    ap.add_argument("--k", type=int, required=True, choices=[16, 32])
    ap.add_argument("--lam-s", type=float, required=True, choices=[0.0, 1.0])
    ap.add_argument("--seg-len", type=int, default=4)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--d", type=int, default=None, help="每組 embedding 維度，預設沿用昨天 group_dim=8")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 70000 + args.g * 1000 + args.k * 10 + (1 if args.lam_s > 0 else 0)

    torch.set_num_threads(max(1, args.threads))

    tag = f"G{args.g}K{args.k}_lam{fmt_lam(args.lam_s)}"
    print(f"=== bcodec_v0 gk-train {tag} (seed={args.seed}, split_seed={args.split_seed}, steps={args.steps}) ===")

    data_path = os.path.join(args.data_dir, f"{bc.DATASET_NAME}.npz")
    if not os.path.exists(data_path):
        print(f"⛔ 找不到資料檔：{data_path}")
        sys.exit(1)

    t0 = time.time()
    data = bc.load_bcodec_segments(data_path, seg_len=args.seg_len, split_seed=args.split_seed, val_frac=args.val_frac)
    print(f"segments: total={len(data['seg_starts'])} train={len(data['train_idx'])} val={len(data['val_idx'])} "
          f"act_dim={data['act_dim']} obs_dim={data['obs_dim']} (extract took {time.time()-t0:.1f}s)")

    model, results, extra = bc.train_and_eval_gk(
        data, G=args.g, K=args.k, D=args.d, lam_s=args.lam_s, steps=args.steps, batch=args.batch, lr=args.lr,
        beta=args.beta, hidden=args.hidden, decay=args.decay, dead_steps=args.dead_steps,
        seed=args.seed, log_every=args.log_every,
    )
    m = results["metrics"]
    print(f"recon_a: train={m['recon_a_train']:.5f} val={m['recon_a_val']:.5f} "
          f"baseline_val={m['baseline_mse_a_val']:.5f} improve={m['val_improve_a_pct']:.1f}%")
    print(f"recon_s: train={m['recon_s_train']:.5f} val={m['recon_s_val']:.5f} "
          f"baseline_val={m['baseline_mse_s_val']:.5f} "
          f"improve={(m['val_improve_s_pct'] if m['val_improve_s_pct'] is not None else float('nan')):.1f}% "
          f"(lam_s={args.lam_s})")
    print(f"bits/chunk={m['bits_per_chunk']:.0f}  mean_perplexity={m['mean_perplexity']:.2f}/{args.k}  "
          f"mean_active={m['mean_active_codes']:.1f}/{args.k}  mean_top1={m['mean_top1_usage_frac']*100:.1f}%  "
          f"any_collapsed={m['any_group_collapsed']}  train_seconds={m['train_seconds']:.1f}")
    print("loss_curve head/tail:")
    for row in (results["loss_curve"][:3] + (["..."] if len(results["loss_curve"]) > 6 else []) + results["loss_curve"][-3:]):
        print(f"  {row}")

    results["config"] = dict(
        G=args.g, K=args.k, D=(args.d if args.d is not None else 8), lam_s=args.lam_s, seg_len=args.seg_len,
        steps=args.steps, batch=args.batch, lr=args.lr, beta=args.beta, hidden=args.hidden, decay=args.decay,
        dead_steps=args.dead_steps, seed=args.seed, split_seed=args.split_seed, val_frac=args.val_frac,
        act_dim=data["act_dim"], obs_dim=data["obs_dim"], dataset=bc.DATASET_NAME,
    )
    results["data"] = dict(n_segments_total=int(len(data["seg_starts"])), n_train=int(len(data["train_idx"])),
                           n_val=int(len(data["val_idx"])))

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, f"{tag}.json")
    bc.save_json(results, out_json)
    print(f"saved: {out_json}")

    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save(dict(state_dict=model.state_dict(), config=results["config"],
                    obs_mu=extra["obs_mu"], obs_sd=extra["obs_sd"]), ckpt_path)
    print(f"saved: {ckpt_path}")
    print(f"=== done {tag} (wall {time.time()-t0:.1f}s) ===")


if __name__ == "__main__":
    main()
