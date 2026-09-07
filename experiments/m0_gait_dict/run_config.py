#!/usr/bin/env python
"""M0 偵察：訓一格 VQ-VAE 步法字典（seg_len × K）並量測。

用法：
  python run_config.py --seg-len 4 --k 16 --seed 42 --visualize

⛔ CPU-only（本檔內部強制 device='cpu'，不碰 torch.cuda / GPU）。
固定 seed 可重跑；每次執行只做一格（seg_len, K），--visualize 額外產出
「每個 code 抽 3 段 render 成 gif」的視覺化（設計上只在 4 步×K=16 那格開啟）。

輸出：
  results/L{seg_len}_K{k}.json        ：六格對照表用的數字
  results/hist_L{seg_len}_K{k}.png    ：code 使用率直方圖
  viz_L{seg_len}_K{k}/code%02d_ex%d.gif（或 fallback 的 code%02d.png）
  logs/L{seg_len}_K{k}.log            ：由 run_all.sh 導向，這裡只印 stdout
"""
import argparse
import os
import sys
import time

# 一律 CPU-only：GPU 只能走 slurm 排卡，這支腳本不經 slurm 只准 CPU。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import m0_common as mc

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET_NAME = "antmaze-medium-stitch-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-len", type=int, required=True, choices=[4, 8])
    ap.add_argument("--k", type=int, required=True, choices=[8, 16, 32])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="預設由 seg_len,k 決定（見下方），固定可重現")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n-examples", type=int, default=3, help="每個 code 抽幾段做視覺化")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 10000 + args.seg_len * 100 + args.k  # 每格不同、但固定可重現

    torch.set_num_threads(max(1, args.threads))

    tag = f"L{args.seg_len}_K{args.k}"
    print(f"=== M0 gait-dict run {tag} (seed={args.seed}, split_seed={args.split_seed}) ===")

    data_path = os.path.join(args.data_dir, f"{DATASET_NAME}.npz")
    if not os.path.exists(data_path):
        print(f"⛔ 找不到資料檔：{data_path}")
        sys.exit(1)

    t0 = time.time()
    data = mc.load_segments(data_path, args.seg_len, split_seed=args.split_seed, val_frac=args.val_frac)
    print(
        f"segments: total={len(data['seg_starts'])} train={len(data['train_idx'])} "
        f"val={len(data['val_idx'])} seg_dim={data['segs'].shape[1]} "
        f"fallback_segmentation_used={data['fallback_segmentation_used']} "
        f"(extract took {time.time()-t0:.1f}s)"
    )

    model, results, all_codes = mc.train_and_eval(
        data,
        K=args.k,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        beta=args.beta,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        decay=args.decay,
        dead_steps=args.dead_steps,
        seed=args.seed,
    )
    m = results["metrics"]
    print(
        f"train_mse={m['recon_mse_train']:.5f} val_mse={m['recon_mse_val']:.5f} "
        f"baseline_train={m['baseline_mse_train']:.5f} baseline_val={m['baseline_mse_val']:.5f} "
        f"perplexity={m['perplexity']:.2f}/{args.k} active_codes={m['n_active_codes']} "
        f"top1={m['top1_usage_frac']*100:.1f}% top3={m['top3_usage_frac']*100:.1f}% "
        f"restarts={m['restarts_total']} train_seconds={results['timing']['train_seconds']:.1f}"
    )

    results["config"] = dict(
        seg_len=args.seg_len,
        k=args.k,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        beta=args.beta,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        decay=args.decay,
        dead_steps=args.dead_steps,
        seed=args.seed,
        split_seed=args.split_seed,
        val_frac=args.val_frac,
        dataset=DATASET_NAME,
    )
    results["data"] = dict(
        n_segments_total=int(len(data["seg_starts"])),
        n_train=int(len(data["train_idx"])),
        n_val=int(len(data["val_idx"])),
        act_dim=int(data["act_dim"]),
        seg_dim=int(data["segs"].shape[1]),
        n_episodes=int(data["n_episodes"]),
        fallback_segmentation_used=bool(data["fallback_segmentation_used"]),
    )

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    hist_png = os.path.join(results_dir, f"hist_{tag}.png")
    mc.draw_usage_histogram(
        m["usage_frac"], hist_png, title=f"code usage - seg_len={args.seg_len} K={args.k}"
    )
    results["artifacts"] = dict(usage_hist_png=os.path.relpath(hist_png, args.out_dir))

    viz_manifest = None
    if args.visualize:
        viz_manifest = run_visualize(args, data, all_codes, tag)
        results["artifacts"]["visualization"] = viz_manifest

    out_json = os.path.join(results_dir, f"{tag}.json")
    mc.save_json(results, out_json)
    print(f"saved: {out_json}")
    print(f"saved: {hist_png}")
    print(f"=== done {tag} (wall {time.time()-t0:.1f}s) ===")


def run_visualize(args, data, all_codes, tag):
    """對每個 code 抽 --n-examples 段，優先 MuJoCo render 成 gif；失敗才退回 Pillow 曲線圖。"""
    os.environ["MUJOCO_GL"] = "osmesa"
    viz_dir = os.path.join(args.out_dir, f"viz_{tag}")
    os.makedirs(viz_dir, exist_ok=True)

    d = np.load(os.path.join(args.data_dir, f"{DATASET_NAME}.npz"))
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    qvel = np.asarray(d["qvel"], dtype=np.float64)
    actions = np.asarray(d["actions"], dtype=np.float32)
    seg_starts = data["seg_starts"]  # 對應 all_codes 的原始索引？見下方組合

    # all_codes 的順序是 [train_idx 段..., val_idx 段...]（見 m0_common.train_and_eval）
    combined_idx = np.concatenate([data["train_idx"], data["val_idx"]])
    combined_starts = seg_starts[combined_idx]

    rng = np.random.default_rng(args.seed + 999)

    env_state = None
    render_env_ok = True
    render_env_err = None
    try:
        env_pack = mc.make_render_env(args.data_dir)
        env_state = dict(u=env_pack["u"], renderer=env_pack["renderer"])
    except Exception as e:  # noqa: BLE001 (M0 偵察，render 環境建不起來要能繼續退回 fallback)
        render_env_ok = False
        render_env_err = repr(e)
        print(f"⛔ MuJoCo render env 建立失敗，全面退回 Pillow 曲線圖: {render_env_err}")

    per_code = []
    for k in range(args.k):
        pos = np.flatnonzero(all_codes == k)
        n_avail = len(pos)
        n_pick = min(args.n_examples, n_avail)
        entry = dict(code=k, n_available_in_dataset=int(n_avail), n_examples=n_pick, method=None, files=[])
        if n_pick == 0:
            entry["method"] = "none"
            entry["note"] = "此 code 在整個資料集（train+val）用量為 0，無法產生視覺化"
            per_code.append(entry)
            continue

        pick = rng.choice(pos, size=n_pick, replace=False)
        starts_for_code = combined_starts[pick]

        used_gif = False
        if render_env_ok:
            try:
                files = []
                for ei, s0 in enumerate(starts_for_code):
                    out_path = os.path.join(viz_dir, f"code{k:02d}_ex{ei}.gif")
                    mc.render_segment_gif(env_state, int(s0), args.seg_len, out_path, qpos, qvel)
                    files.append(os.path.relpath(out_path, args.out_dir))
                entry["method"] = "mujoco_gif"
                entry["files"] = files
                used_gif = True
            except Exception as e:  # noqa: BLE001
                print(f"⛔ code {k} MuJoCo render 失敗，退回 Pillow 曲線圖: {e!r}")

        if not used_gif:
            examples = []
            for s0 in starts_for_code:
                s0 = int(s0)
                disp_xy = (qpos[s0 + args.seg_len][:2] - qpos[s0][:2]).tolist()
                acts = actions[s0 : s0 + args.seg_len]
                examples.append(dict(disp_xy=disp_xy, actions=acts))
            out_path = os.path.join(viz_dir, f"code{k:02d}.png")
            mc.draw_code_fallback_plot(examples, out_path, k, args.seg_len, data["act_dim"])
            entry["method"] = "fallback_plot"
            entry["files"] = [os.path.relpath(out_path, args.out_dir)]

        per_code.append(entry)

    n_gif = sum(1 for e in per_code if e["method"] == "mujoco_gif")
    n_fallback = sum(1 for e in per_code if e["method"] == "fallback_plot")
    n_none = sum(1 for e in per_code if e["method"] == "none")
    print(f"visualize: {n_gif} codes via mujoco_gif, {n_fallback} via fallback_plot, {n_none} empty (0 usage)")

    return dict(
        viz_dir=os.path.relpath(viz_dir, args.out_dir),
        render_env_ok=render_env_ok,
        render_env_error=render_env_err,
        per_code=per_code,
        n_codes_mujoco_gif=n_gif,
        n_codes_fallback_plot=n_fallback,
        n_codes_empty=n_none,
    )


if __name__ == "__main__":
    main()
