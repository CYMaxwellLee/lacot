#!/usr/bin/env python
"""通用版單段保真度（E_indep）量測——供任何 gk_scan FactorizedCondVQVAE ckpt 使用。

追加工單（2026-09-11，派工者已獲主人核可）：補「譜容量→單段落點準度」曲線第三點
G32_K32（DreamerV2 原版配置，32 組×32 類，每 chunk 160 bits）。跟 G16K16 同一套
管線、同一批 200 task×50 chunk 段集，只換 ckpt。

【在證明什麼、判準是什麼】
證明：G32_K32 的單段 E_indep（xy p25/p50/p75/mean, pooled n=10000）與 recon MSE，
跟 K32／G16K16 放進同一張表。只回報數字，不下判讀（判讀派工者做）。

⛔ 不改任何既有檔案，包括今天稍早新增的 run_drift_analysis_g16k16.py（本檔是它的
參數化泛化版——差異只有：ckpt/ref-val-mse 走 CLI 而非硬編 G16K16 特定值、移除
「cfg必須是G=16,K=16」的斷言改成印實際值、多一段 task 指紋自檢）。import 對象跟
run_drift_analysis_g16k16.py 完全相同：wv_common / p1_replay / run_teacher_relay /
run_teacher_relay_g16k16（借它的 load_g16k16，本身就不寫死 G/K，見該檔 load_g16k16()
函式本體只從 ckpt 的 cfg 讀，斷言在它的 main() 裡才寫死——本檔不呼叫那個 main()）
/ gk_common。

段集指紋自檢：build_tasks() 不吃 ckpt/模型當參數，理論上任何字典都拿到同一批
200 task；仍然實測算一次 sha256（跟今天 check_task_comparability.py 用同一種雜湊
方式：episode,s0,e0,M 逐 task 串接），對照今天已驗證過的值
450be2402fcc5e8fdc5d99f6c81d453e815e32d4a6ab26c3dbf64318f88f46ff，不符就停手。

⛔ CPU-only。sbatch 見同目錄 run_drift_analysis_gkdict.sbatch。不畫圖。
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))                       # drift_analysis_g16k16/
WV_DIR = os.path.dirname(HERE)                                          # walk_verify/
EXP_DIR = os.path.dirname(WV_DIR)                                       # experiments/
TR_DIR = os.path.join(WV_DIR, "teacher_relay")
TRG16_DIR = os.path.join(WV_DIR, "teacher_relay_g16k16")
GK_DIR = os.path.join(EXP_DIR, "gk_scan")
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)
sys.path.insert(0, TRG16_DIR)
sys.path.insert(0, GK_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q  # noqa: E402
import run_teacher_relay as trk32  # noqa: E402
import run_teacher_relay_g16k16 as trg16  # noqa: E402 - 只借 load_g16k16()/DD/RHO_DEFAULT，未修改
import gk_common as gc  # noqa: E402

DD = trg16.DD
DEFAULT_RULER = trg16.DEFAULT_RULER
DELTA_SUB = trk32.DELTA_SUB
RHO_DEFAULT = trg16.RHO_DEFAULT
K32_DRIFT_SUMMARY_JSON = os.path.join(WV_DIR, "drift_analysis", "results", "drift_summary.json")
EXPECTED_TASK_HASH = "450be2402fcc5e8fdc5d99f6c81d453e815e32d4a6ab26c3dbf64318f88f46ff"

J_GRID_PRINT = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 49]


def task_hash(tasks):
    h = hashlib.sha256()
    for t in tasks:
        h.update(f"{t['episode']},{t['s0']},{t['e0']},{t['M']}|".encode())
    return h.hexdigest()


def run_reset_chunks_gkdict(env, u, model, obs_mu, obs_sd, raw, task, code_rng=None, garbage_k0=False):
    """跟 run_drift_analysis_g16k16.run_reset_chunks_g16k16 同協定，僅去除 G16K16 專屬命名。"""
    s0, n_chunks = task["s0"], task["n_chunks"]
    act, qpos_arr, qvel_arr = raw["actions"], raw["qpos"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    Gd, Kd = model.G, model.K
    xy_err, recon_mse, codes = [], [], []
    xy_err_garbage_k0 = None
    for k in range(n_chunks):
        s = s0 + 4 * k
        env.reset()
        u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
        with torch.no_grad():
            x = torch.from_numpy(act[s:s + 4].reshape(1, -1).astype(np.float32))
            z_e = model.encoder(x)
            _, idx, _ = model.vq(z_e, training=False)
            o_raw = torch.from_numpy(sim_obs(u)[None, :])
            o_norm = (o_raw - obs_mu) / obs_sd
            raw_a_t = model.decode_codes(idx, o_norm)
            mse = float(torch.mean((raw_a_t - x) ** 2).item())
            raw_a = raw_a_t.numpy().reshape(4, gc.ACT_DIM)
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
        for t in range(4):
            env.step(a_chunk[t])
        end_xy = np.asarray(u.data.qpos)[:2].copy()
        true_end_xy = qpos_arr[s + 4, :2]
        xy_err.append(float(np.linalg.norm(end_xy - true_end_xy)))
        recon_mse.append(mse)
        codes.append(idx.numpy().reshape(-1).copy())

        if garbage_k0 and k == 0:
            assert code_rng is not None
            env.reset()
            u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
            with torch.no_grad():
                idx_g = torch.from_numpy(code_rng.integers(0, Kd, size=(1, Gd), dtype=np.int64))
                o_raw_g = torch.from_numpy(sim_obs(u)[None, :])
                o_norm_g = (o_raw_g - obs_mu) / obs_sd
                raw_a_g = model.decode_codes(idx_g, o_norm_g).numpy().reshape(4, gc.ACT_DIM)
            a_chunk_g = np.clip(raw_a_g, lo, hi).astype(np.float64)
            for t in range(4):
                env.step(a_chunk_g[t])
            end_xy_g = np.asarray(u.data.qpos)[:2].copy()
            xy_err_garbage_k0 = float(np.linalg.norm(end_xy_g - true_end_xy))

    return dict(xy_err=np.asarray(xy_err), recon_mse=np.asarray(recon_mse),
                codes=np.asarray(codes, dtype=np.int64), xy_err_garbage_k0=xy_err_garbage_k0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--garbage-seed", type=int, default=20260911)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--ref-val-mse", type=float, default=None,
                    help="gk_scan results/<tag>.json 記錄的 val MSE，用來做載入自檢（不給就跳過自檢）")
    ap.add_argument("--ref-json-path", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== drift-analysis-gkdict  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} garbage_seed={args.garbage_seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg, obs_mu, obs_sd = trg16.load_g16k16(args.ckpt)
    assert cfg["seg_len"] == 4, f"⛔ 預期 seg_len=4，實際 cfg={cfg}"
    print(f"dict: G={cfg['G']} K={cfg['K']} group_dim={cfg['group_dim']} seg_len={cfg['seg_len']} "
          f"hidden={cfg['hidden']}  bits/chunk={cfg['G']*np.log2(cfg['K']):.0f}")

    # ================================================================
    # 載入自檢（跟 G16K16 版同款：整個 val 切分重建 MSE 對照 gk_scan 記錄值）
    # ================================================================
    if args.ref_val_mse is not None:
        train_npz = os.path.join(args.data_dir, f"{trk32.DATASET}.npz")
        data = gc.load_segments_with_obs(train_npz, seg_len=cfg["seg_len"],
                                         split_seed=cfg["split_seed"], val_frac=cfg["val_frac"])
        val_x = torch.from_numpy(data["segs"][data["val_idx"]])
        val_o_raw = torch.from_numpy(data["obs_start"][data["val_idx"]])
        val_o_norm = (val_o_raw - obs_mu) / obs_sd
        with torch.no_grad():
            val_recon, _, _ = model(val_x, val_o_norm, training=False)
            recon_mse_val_check = float(F.mse_loss(val_recon, val_x).item())
        val_mse_diff = abs(recon_mse_val_check - args.ref_val_mse)
        val_mse_match = val_mse_diff < 1e-4
        print(f"[載入自檢] 整個 val 切分（{len(data['val_idx'])} 段）重建 MSE={recon_mse_val_check:.6f} "
              f"vs {args.ref_json_path} 記錄值 {args.ref_val_mse:.6f}（差={val_mse_diff:.2e}）—— "
              f"{'吻合，載入正確' if val_mse_match else '⛔ 對不起來，停手回報'}")
        if not val_mse_match:
            sys.exit(1)
    else:
        recon_mse_val_check, val_mse_diff, val_mse_match = None, None, None
        print("[載入自檢] 未提供 --ref-val-mse，跳過（明標，不假裝驗過）")

    # ================================================================
    # 段集 + 指紋自檢
    # ================================================================
    with open(args.ruler) as f:
        ruler = json.load(f)
    raw = wv.load_npz(args.data_dir, trk32.DATASET, "val")
    tasks, skipped = trk32.build_tasks(raw, args.n_traj, args.seed, ruler, args.rho)
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}")
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報"
    n_chunks_all = np.array([t["n_chunks"] for t in tasks])
    assert (n_chunks_all == n_chunks_all[0]).all(), f"⛔ n_chunks 不一致: {np.unique(n_chunks_all)}"
    N_CHUNKS = int(n_chunks_all[0])

    my_hash = task_hash(tasks)
    hash_match = (my_hash == EXPECTED_TASK_HASH)
    print(f"[段集指紋自檢] sha256={my_hash}")
    print(f"[段集指紋自檢] 對照今天 check_task_comparability.py 驗證過的值 {EXPECTED_TASK_HASH}"
          f" —— {'一致' if hash_match else '⛔ 不一致，停手回報'}")
    if not hash_match:
        sys.exit(1)

    env = wv.make_env(args.data_dir, trk32.DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================
    # 200 題 x 每題全部 chunk
    # ================================================================
    print("\n=== 200 題 x 每題全部 chunk（reset 對照，E_indep）===")
    ta = time.time()
    code_rng = np.random.default_rng(args.garbage_seed)
    reset_xy_err_by_k = {k: [] for k in range(N_CHUNKS)}
    reset_recon_mse_by_k = {k: [] for k in range(N_CHUNKS)}
    garbage_k0_xy = []
    for ti, task in enumerate(tasks):
        rr = run_reset_chunks_gkdict(env, u, model, obs_mu, obs_sd, raw, task,
                                     code_rng=code_rng, garbage_k0=True)
        for k in range(len(rr["xy_err"])):
            reset_xy_err_by_k[k].append(float(rr["xy_err"][k]))
            reset_recon_mse_by_k[k].append(float(rr["recon_mse"][k]))
        garbage_k0_xy.append(rr["xy_err_garbage_k0"])
    print(f"done in {time.time()-ta:.1f}s")

    pooled_xy = np.concatenate([np.asarray(v) for v in reset_xy_err_by_k.values()])
    pooled_mse = np.concatenate([np.asarray(v) for v in reset_recon_mse_by_k.values()])
    e_indep_xy_q = q(pooled_xy)
    mse_q = q(pooled_mse)
    print(f"\nE_indep（n={len(pooled_xy)}）：xy p25/p50/p75/mean = "
          f"{e_indep_xy_q['25']:.4f}/{e_indep_xy_q['50']:.4f}/{e_indep_xy_q['75']:.4f}/{pooled_xy.mean():.4f}")
    print(f"重建 MSE（n={len(pooled_mse)}）：p25/p50/p75/mean = "
          f"{mse_q['25']:.5f}/{mse_q['50']:.5f}/{mse_q['75']:.5f}/{pooled_mse.mean():.5f}")

    k0_xy = np.asarray(reset_xy_err_by_k[0])
    garbage_k0_xy = np.asarray(garbage_k0_xy, dtype=float)
    print(f"\n[爛錨對照@k=0] 真字 median={np.median(k0_xy):.4f}m (n={len(k0_xy)})  vs "
          f"隨機字 median={np.median(garbage_k0_xy):.4f}m (n={len(garbage_k0_xy)})  "
          f"比值(隨機/真字)={np.median(garbage_k0_xy)/max(np.median(k0_xy),1e-12):.2f}")

    accounting_rows = []
    for k in J_GRID_PRINT:
        if k >= N_CHUNKS:
            continue
        v = np.asarray(reset_xy_err_by_k[k])
        m = np.asarray(reset_recon_mse_by_k[k])
        accounting_rows.append(dict(k=k, n=int(len(v)), xy_p25=float(np.percentile(v, 25)),
                                    xy_p50=float(np.percentile(v, 50)), xy_p75=float(np.percentile(v, 75)),
                                    mse_p50=float(np.percentile(m, 50))))

    with open(K32_DRIFT_SUMMARY_JSON) as f:
        k32_drift = json.load(f)
    k32_p50 = k32_drift["e_indep_baseline"]["xy"]
    pct_change = (e_indep_xy_q["50"] - k32_p50) / k32_p50 * 100.0
    print(f"\nE_indep p50={e_indep_xy_q['50']:.4f}m vs K32 archived p50={k32_p50:.4f}m  Δ%={pct_change:+.1f}%")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(
        config=dict(vars(args)), delta_sub=DELTA_SUB, dict_cfg=cfg,
        n_tasks_used=len(tasks), n_chunks_per_task=N_CHUNKS,
        task_fingerprint=dict(sha256=my_hash, expected=EXPECTED_TASK_HASH, match=hash_match),
        load_self_check=dict(full_val_recon_mse_check=recon_mse_val_check,
                             ref_val_mse=args.ref_val_mse, diff=val_mse_diff, match=val_mse_match),
        e_indep_baseline=dict(xy_p25=e_indep_xy_q["25"], xy_p50=e_indep_xy_q["50"],
                              xy_p75=e_indep_xy_q["75"], xy_mean=float(pooled_xy.mean()), n=int(len(pooled_xy))),
        recon_mse_baseline=dict(p25=mse_q["25"], p50=mse_q["50"], p75=mse_q["75"],
                                mean=float(pooled_mse.mean()), n=int(len(pooled_mse))),
        garbage_code_control_k0=dict(true_code_xy_p50=float(np.median(k0_xy)),
                                     garbage_code_xy_p50=float(np.median(garbage_k0_xy)),
                                     n=int(len(k0_xy)), seed=args.garbage_seed),
        accounting_table=accounting_rows,
        k32_reference=dict(source=K32_DRIFT_SUMMARY_JSON, e_indep_xy_p50=k32_p50,
                          pct_change_vs_k32_p50=pct_change),
    ), js_path)
    print(f"saved: {js_path}")

    npz_path = os.path.join(args.out_dir, f"{args.tag}_raw.npz")
    xy_grid = np.full((len(tasks), N_CHUNKS), np.nan)
    mse_grid = np.full((len(tasks), N_CHUNKS), np.nan)
    for k in range(N_CHUNKS):
        xy_grid[:len(reset_xy_err_by_k[k]), k] = reset_xy_err_by_k[k]
        mse_grid[:len(reset_recon_mse_by_k[k]), k] = reset_recon_mse_by_k[k]
    np.savez_compressed(npz_path, xy_err_grid=xy_grid, recon_mse_grid=mse_grid,
                        garbage_k0_xy=garbage_k0_xy, k0_xy=k0_xy)
    print(f"saved: {npz_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
