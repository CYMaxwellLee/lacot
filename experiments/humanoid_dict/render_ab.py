#!/usr/bin/env python
"""推薦格 A/B 對比視覺化（humanoidmaze-medium-stitch-v0）：同一段、固定其他組、
只換第 g 組的字，每組 2 對。跟 gk_scan/render_ab.py（ant 版）同一套方法：優先
MuJoCo osmesa 並排 gif（真的 step 物理引擎跑合成動作），環境建不起來就整批退回
Pillow 曲線對比圖。

用法：
  python render_ab.py            # 讀 results/summary.json 決定用哪一格
  python render_ab.py --g 4 --k 32   # 手動指定（略過 summary.json）
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
N_PAIRS_PER_GROUP = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=HERE)
    args = ap.parse_args()

    if args.g is None or args.k is None:
        summary_path = os.path.join(HERE, "results", "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)
        rec = summary["recommendation"]["cell"]
        if rec is None:
            print("⛔ summary.json 裡沒有推薦格（可能全部格都塌陷），無法做 A/B render")
            sys.exit(1)
        args.g, args.k = rec["G"], rec["K"]
        print(f"從 summary.json 讀到推薦格：G={args.g} K={args.k} "
              f"(fallback={summary['recommendation']['fallback']})")

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
    obs_mu, obs_sd = ckpt["obs_mu"], ckpt["obs_sd"]

    data_path = os.path.join(args.data_dir, f"{hc.DATASET_NAME}.npz")
    data = hc.load_segments_with_obs(data_path, cfg["seg_len"], split_seed=cfg["split_seed"], val_frac=cfg["val_frac"])
    d_raw = np.load(data_path)
    qpos_all = np.asarray(d_raw["qpos"], dtype=np.float64)
    qvel_all = np.asarray(d_raw["qvel"], dtype=np.float64)

    rng = np.random.default_rng(cfg["seed"] + 12345)
    pick_pool = data["val_idx"]
    n_pairs_total = G * N_PAIRS_PER_GROUP
    pick = rng.choice(pick_pool, size=min(n_pairs_total, len(pick_pool)), replace=False)
    pick_by_group = {g: pick[g * N_PAIRS_PER_GROUP : (g + 1) * N_PAIRS_PER_GROUP] for g in range(G)}

    seg_starts = data["seg_starts"]
    seg_len = cfg["seg_len"]

    env_state = None
    render_env_ok = True
    render_env_err = None
    try:
        env_pack = hc.make_render_env(args.data_dir)
        env_state = dict(u=env_pack["u"], renderer=env_pack["renderer"])
    except Exception as e:  # noqa: BLE001
        render_env_ok = False
        render_env_err = repr(e)
        print(f"⛔ MuJoCo render env 建立失敗，全面退回 Pillow 曲線對比圖: {render_env_err}")

    viz_dir = os.path.join(args.out_dir, "viz_recommended")
    os.makedirs(viz_dir, exist_ok=True)

    manifest = []
    with torch.no_grad():
        for g in range(G):
            for p, seg_i in enumerate(pick_by_group[g]):
                obs_i = data["obs_start"][seg_i]
                obs_norm = torch.from_numpy(((obs_i - obs_mu[0]) / obs_sd[0]).astype(np.float32)).unsqueeze(0)
                x = torch.from_numpy(data["segs"][seg_i]).float().unsqueeze(0)

                z_e = model.encoder(x)
                _, orig_idx, _ = model.vq(z_e, training=False)
                orig_recon = model.decode_codes(orig_idx, obs_norm)[0].numpy().reshape(seg_len, hc.ACT_DIM)

                mod_idx = orig_idx.clone()
                mod_idx[0, g] = (orig_idx[0, g] + K // 2) % K
                mod_recon = model.decode_codes(mod_idx, obs_norm)[0].numpy().reshape(seg_len, hc.ACT_DIM)

                s0 = int(seg_starts[seg_i])
                qpos0, qvel0 = qpos_all[s0], qvel_all[s0]

                entry = dict(
                    group=g, pair=p, seg_index=int(seg_i), seg_start=s0,
                    orig_code=int(orig_idx[0, g].item()), swapped_code=int(mod_idx[0, g].item()),
                    method=None, file=None,
                )

                used_gif = False
                if render_env_ok:
                    try:
                        qpos_a, qvel_a = hc.rollout_from_actions(env_state, qpos0, qvel0, orig_recon)
                        qpos_b, qvel_b = hc.rollout_from_actions(env_state, qpos0, qvel0, mod_recon)
                        out_path = os.path.join(viz_dir, f"group{g:02d}_pair{p}.gif")
                        hc.render_side_by_side_gif(
                            env_state, qpos_a, qvel_a, qpos_b, qvel_b, out_path,
                            label_a=f"g{g} orig={entry['orig_code']}",
                            label_b=f"g{g} swap={entry['swapped_code']}",
                        )
                        entry["method"] = "mujoco_gif_side_by_side"
                        entry["file"] = os.path.relpath(out_path, args.out_dir)
                        used_gif = True
                    except Exception as e:  # noqa: BLE001
                        print(f"⛔ group{g} pair{p} MuJoCo render 失敗，退回 Pillow 曲線圖: {e!r}")

                if not used_gif:
                    out_path = os.path.join(viz_dir, f"group{g:02d}_pair{p}_fallback.png")
                    hc.draw_ab_fallback_plot(
                        orig_recon, mod_recon, out_path,
                        title=f"G{G}K{K} group={g} pair={p}: orig_code={entry['orig_code']} vs swapped_code={entry['swapped_code']}",
                    )
                    entry["method"] = "fallback_plot"
                    entry["file"] = os.path.relpath(out_path, args.out_dir)

                manifest.append(entry)

    n_gif = sum(1 for e in manifest if e["method"] == "mujoco_gif_side_by_side")
    n_fallback = sum(1 for e in manifest if e["method"] == "fallback_plot")
    print(f"A/B viz for {tag}: {len(manifest)} pairs, {n_gif} via mujoco_gif, {n_fallback} via fallback_plot")

    out_manifest = dict(
        cell=dict(G=G, K=K), render_env_ok=render_env_ok, render_env_error=render_env_err,
        n_pairs_per_group=N_PAIRS_PER_GROUP, n_pairs_total=len(manifest),
        n_mujoco_gif=n_gif, n_fallback_plot=n_fallback, pairs=manifest,
    )
    out_json = os.path.join(args.out_dir, "results", "ab_render_manifest.json")
    hc.save_json(out_manifest, out_json)
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
