#!/usr/bin/env python
"""「單碼本推薦格的字」原始段 gif（humanoidmaze-medium-stitch-v0）。

任務視覺化第二部分：「另 render 8 個『單碼本推薦格的字』的原始段 gif（讓主人看
humanoid 的步法字長什麼樣）」。

⚠️ 自訂假定（任務沒寫死，這裡記清楚）：推薦格是 G 個獨立 codebook 因子化組合，
「單碼本」取推薦格裡的 group 0（--group 可override）當代表，不是重新訓一個
G=1 的模型。挑「用量前 8 高」的字（同 route_dict 的「挑用量前幾高保證有真實段」
邏輯，K>=8 恆成立所以一定挑得滿 8 個），每個字挑 1 條真實段（該字在 val+train
合併池裡隨機抽 1 條、固定 seed 可重跑），播的是「真實」qpos/qvel 序列本身
（不是換字合成、不需要重新 step 物理引擎，最忠實地呈現這個字在資料裡實際長什麼樣）。

用法：
  python render_dict_words.py                 # 讀 summary.json 推薦格，group=0
  python render_dict_words.py --g 16 --k 16 --group 0
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ["MUJOCO_GL"] = "osmesa"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import hd_common as hc

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
N_WORDS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--group", type=int, default=0, help="推薦格裡當『單碼本』代表的 group index")
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=HERE)
    args = ap.parse_args()

    if args.g is None or args.k is None:
        summary_path = os.path.join(HERE, "results", "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        rec = summary["recommendation"]["cell"]
        if rec is None:
            print("⛔ summary.json 裡沒有推薦格，無法做字典字 render")
            sys.exit(1)
        args.g, args.k = rec["G"], rec["K"]
        print(f"從 summary.json 讀到推薦格：G={args.g} K={args.k}")

    if args.group >= args.g:
        print(f"⛔ --group {args.group} 超出範圍（推薦格只有 G={args.g} 組），改用 group 0")
        args.group = 0

    tag = f"G{args.g}_K{args.k}"
    ckpt_path = os.path.join(HERE, "ckpt", f"{tag}.pt")
    if not os.path.exists(ckpt_path):
        print(f"⛔ 找不到 checkpoint：{ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    G, K, D = cfg["G"], cfg["K"], cfg["group_dim"]

    model = hc.FactorizedCondVQVAE(
        seg_dim=cfg["seg_len"] * hc.ACT_DIM, obs_dim=hc.OBS_DIM, G=G, K=K, D=D, hidden=cfg["hidden"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_path = os.path.join(args.data_dir, f"{hc.DATASET_NAME}.npz")
    data = hc.load_segments_with_obs(data_path, cfg["seg_len"], split_seed=cfg["split_seed"], val_frac=cfg["val_frac"])
    d_raw = np.load(data_path)
    qpos_all = np.asarray(d_raw["qpos"], dtype=np.float64)
    qvel_all = np.asarray(d_raw["qvel"], dtype=np.float64)

    all_idx = np.concatenate([data["train_idx"], data["val_idx"]])
    with torch.no_grad():
        x_all = torch.from_numpy(data["segs"][all_idx]).float()
        z_e = model.encoder(x_all)
        _, codes, _ = model.vq(z_e, training=False)  # (N, G)
    codes_g = codes[:, args.group].numpy()

    counts = np.bincount(codes_g, minlength=K)
    order = np.argsort(-counts)
    top_codes = [int(c) for c in order[:min(N_WORDS, K)]]
    print(f"group={args.group} usage top-{len(top_codes)} codes (of K={K}): "
          f"{[(c, int(counts[c])) for c in top_codes]}")

    rng = np.random.default_rng(cfg["seed"] + 987)

    env_state = None
    render_env_ok = True
    render_env_err = None
    try:
        env_pack = hc.make_render_env(args.data_dir)
        env_state = dict(u=env_pack["u"], renderer=env_pack["renderer"])
    except Exception as e:  # noqa: BLE001
        render_env_ok = False
        render_env_err = repr(e)
        print(f"⛔ MuJoCo render env 建立失敗，全面退回 Pillow 動作曲線圖: {render_env_err}")

    viz_dir = os.path.join(args.out_dir, "viz_words")
    os.makedirs(viz_dir, exist_ok=True)
    seg_starts_all = data["seg_starts"][all_idx]
    seg_len = cfg["seg_len"]

    manifest = []
    for rank, code in enumerate(top_codes):
        members = np.flatnonzero(codes_g == code)
        usage_frac = float(counts[code] / max(counts.sum(), 1))
        pick_local = int(rng.choice(members))
        seg_i_global = int(all_idx[pick_local])
        s0 = int(seg_starts_all[pick_local])

        entry = dict(
            rank=rank, group=args.group, code=code, usage_frac=usage_frac,
            n_members=int(counts[code]), seg_index=seg_i_global, seg_start=s0,
            method=None, file=None,
        )

        used_gif = False
        if render_env_ok:
            try:
                qpos_seq = [qpos_all[s0 + t] for t in range(seg_len + 1)]
                qvel_seq = [qvel_all[s0 + t] for t in range(seg_len + 1)]
                out_path = os.path.join(viz_dir, f"word_group{args.group}_rank{rank}_code{code}.gif")
                hc.render_single_traj_gif(
                    env_state, qpos_seq, qvel_seq, out_path,
                    label=f"g{args.group} code={code} usage={usage_frac*100:.1f}%",
                )
                entry["method"] = "mujoco_gif_real_segment"
                entry["file"] = os.path.relpath(out_path, args.out_dir)
                used_gif = True
            except Exception as e:  # noqa: BLE001
                print(f"⛔ code={code} MuJoCo render 失敗，退回 Pillow 動作曲線圖: {e!r}")

        if not used_gif:
            actions = data["segs"][seg_i_global].reshape(seg_len, hc.ACT_DIM)
            out_path = os.path.join(viz_dir, f"word_group{args.group}_rank{rank}_code{code}_fallback.png")
            hc.draw_single_action_plot(
                actions, out_path,
                title=f"G{G}K{K} group={args.group} code={code} usage={usage_frac*100:.1f}% (real segment actions)",
            )
            entry["method"] = "fallback_plot"
            entry["file"] = os.path.relpath(out_path, args.out_dir)

        manifest.append(entry)

    n_gif = sum(1 for e in manifest if e["method"] == "mujoco_gif_real_segment")
    n_fallback = sum(1 for e in manifest if e["method"] == "fallback_plot")
    print(f"dict word viz for {tag} group={args.group}: {len(manifest)} words, "
          f"{n_gif} via mujoco_gif, {n_fallback} via fallback_plot")

    out_manifest = dict(
        cell=dict(G=G, K=K), group=args.group, render_env_ok=render_env_ok,
        render_env_error=render_env_err, n_words=len(manifest),
        n_mujoco_gif=n_gif, n_fallback_plot=n_fallback, words=manifest,
    )
    out_json = os.path.join(args.out_dir, "results", "dict_words_manifest.json")
    hc.save_json(out_manifest, out_json)
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
