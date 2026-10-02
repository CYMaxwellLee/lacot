"""G16-② 診斷：同一顆 ckpt、只換 val 切法，量「同軌跡洩漏」灌了多少水。

對每種 split_by ∈ {segment(舊、跟 train 同軌跡), trajectory(新、整軌隔離)}：
在該切分的 val 集上算 recon_a / recon_s（用 ckpt 自帶的 obs_mu/obs_sd 當同一把尺）。
差距 = 舊內部診斷的樂觀幅度。⛔ E_indep 不在此列（官方 val、本來就乾淨）。

用法：python diag_split_leak.py --ckpt <pt> --kind {vq,gk} --data-path <npz> --out-json <p>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bcodec_common as bc  # noqa: E402


def eval_split(model, data, obs_mu, obs_sd, seg_len, obs_dim):
    val_idx = data["val_idx"]
    s_val = torch.from_numpy(bc._norm_seg(data["s_seg"][val_idx], obs_mu.numpy(), obs_sd.numpy(), seg_len, obs_dim))
    a_val = torch.from_numpy(data["a_seg"][val_idx])
    o_val = torch.from_numpy(((data["obs_start"][val_idx] - obs_mu.numpy()) / obs_sd.numpy()).astype(np.float32))
    snext_val = torch.from_numpy(bc._norm_seg(data["s_next"][val_idx], obs_mu.numpy(), obs_sd.numpy(), seg_len, obs_dim))
    with torch.no_grad():
        a_hat, s_hat, _ = bc.chunked_forward(model, s_val, a_val, o_val)
        recon_a = float(F.mse_loss(a_hat, a_val).item())
        recon_s = float(F.mse_loss(s_hat, snext_val).item())
    # 洩漏統計：val 段所屬軌跡與 train 段所屬軌跡的交集
    ep_len = data["ep_len"]
    traj = data["seg_starts"] // ep_len
    val_traj = set(traj[val_idx].tolist())
    train_traj = set(traj[data["train_idx"]].tolist())
    overlap = len(val_traj & train_traj)
    return dict(recon_a=recon_a, recon_s=recon_s, n_val=int(len(val_idx)),
                n_val_traj=len(val_traj), traj_overlap_with_train=overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["vq", "gk"], required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    load = bc.load_bcodec_gk_ckpt if args.kind == "gk" else bc.load_bcodec_ckpt
    model, cfg, obs_mu, obs_sd = load(args.ckpt)
    seg_len, obs_dim = cfg["seg_len"], cfg["obs_dim"]

    out = dict(ckpt=os.path.basename(args.ckpt), kind=args.kind, config=cfg, split={})
    for split_by in ("segment", "trajectory"):
        data = bc.load_bcodec_segments(args.data_path, seg_len=seg_len, split_by=split_by)
        out["split"][split_by] = eval_split(model, data, obs_mu, obs_sd, seg_len, obs_dim)
        print(f"[{os.path.basename(args.ckpt)}] {split_by}: {out['split'][split_by]}", flush=True)

    seg, trj = out["split"]["segment"], out["split"]["trajectory"]
    out["delta"] = dict(
        recon_a_pct=100.0 * (trj["recon_a"] - seg["recon_a"]) / max(seg["recon_a"], 1e-12),
        recon_s_pct=100.0 * (trj["recon_s"] - seg["recon_s"]) / max(seg["recon_s"], 1e-12),
    )
    # 兩向錨：舊切分必有洩漏、新切分必為零 —— 量測鏈本身的健康檢查
    assert seg["traj_overlap_with_train"] > 0, "舊切分應有同軌跡重疊，量到 0 ＝載入鏈壞了"
    assert trj["traj_overlap_with_train"] == 0, "新切分應零重疊，量到非零＝R11 修法或本診斷壞了"
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"OK -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
