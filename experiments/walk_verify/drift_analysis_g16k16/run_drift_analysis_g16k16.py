#!/usr/bin/env python
"""G16K16 單段保真度（E_indep，同 K32 定義）—— 譜更細會不會讓單段執行落點更準。

工單（2026-09-11）：證明或否證「譜更細 ⇒ 單段執行落點更準」。已知 K32 字典
（p0_dict_v1_L4K32_50k.pt）的 E_indep（中位數，逐 chunk 在 0.043~0.053 打轉，
overall≈0.0491m，見 drift_analysis/results/drift_summary.json 的
e_indep_baseline / accounting_table）。本檔補 G16K16（experiments/gk_scan/
ckpt/G16_K16.pt，16 組×16 類 grouped categorical，每段資訊量 12.8×）的同一款
量測，用同一批 200 條評測段（見同目錄 check_task_comparability.py 的 hash 核對）。

【在證明什麼、判準是什麼】
證明：G16K16 的單段量化+單步展開 xy 落點偏差（E_indep）跟 K32 相比是「紅利
成立 / 打平 / 更差」。判準（工單發出前已定，不因為看到數字而改）：
  G16K16 p50 比 K32 p50 低 >30% => 紅利成立
  落在 K32 p25~p75 帶內          => 打平
  高於 K32 p75                   => 更差
本檔只回報數字與套用結果，不下最終判讀（判讀留給派工者）。

量法＝跟 drift_analysis/run_drift_analysis.py 的 run_reset_chunks()（臂2 / indep
重置對照）同一協定：對每個 task 的每個 chunk k，reset 回教材真起點
（env.reset()+set_state(qpos[s],qvel[s])）、用真動作段 encode 選字、用剛 reset
（＝真 obs）當 decoder 條件、decode 展開、clip、執行 4 步，量 xy 落點跟真實
段尾 xy 的偏差。⛔ 唯一差異＝ encode/decode 介面換成 G16K16（見下方 import
與 encode/decode 呼叫，介面差異逐項對照見 docs/NOTE-2026-09-09-teacher-g16k16.md
與 teacher_relay_g16k16/run_teacher_relay_g16k16.py 檔頭 docstring）。

⛔ 不修改任何既有檔案，只 import：
  - wv_common.py（load_npz/make_env，純函式）
  - p1_replay.py（sim_obs/q，純函式）
  - teacher_relay/run_teacher_relay.py（trk32：build_tasks/DELTA_SUB/DATASET——
    跟 K32 側 drift_analysis 實際用的 run_teacher_relay_byleg.build_tasks 逐欄位
    hash 對過，見 check_task_comparability.py 的 PASS 結果）
  - teacher_relay_g16k16/run_teacher_relay_g16k16.py（trg16：load_g16k16()——
    G16K16 模型載入，obs_mu/obs_sd 從 ckpt 讀，不重算）
  - gk_scan/gk_common.py（gc：ACT_DIM=8，跟 wv_common.ACT_DIM 相同）

額外控制（measure-trustworthy 的「爛錨對照」慣例，本專案 G16K16 相關量測一律配這個
控制）：k=0（首 chunk，等於剛從真起點 reset）額外量一次「16 組各自 uniform 隨機
取字」的版本，證明真字選出來的低誤差不是量測量不出差異造成的假象。

⛔ CPU-only（MuJoCo 物理 + 小 MLP encode/decode）。sbatch 見同目錄
run_drift_analysis_g16k16.sbatch。不畫圖（工單只要數字表，省時間少留檔）。
"""
import argparse
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
TR_DIR = os.path.join(WV_DIR, "teacher_relay")                          # walk_verify/teacher_relay/
TRG16_DIR = os.path.join(WV_DIR, "teacher_relay_g16k16")                # walk_verify/teacher_relay_g16k16/
GK_DIR = os.path.join(EXP_DIR, "gk_scan")                               # experiments/gk_scan/
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)
sys.path.insert(0, TRG16_DIR)
sys.path.insert(0, GK_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402 - 既有共用模組，未修改
from p1_replay import sim_obs, q  # noqa: E402 - 純函式重用，未修改
import run_teacher_relay as trk32  # noqa: E402 - build_tasks/DELTA_SUB/DATASET，未修改一行
import run_teacher_relay_g16k16 as trg16  # noqa: E402 - load_g16k16()，未修改一行
import gk_common as gc  # noqa: E402 - ACT_DIM 等常數/類別，未修改一行

DD = trg16.DD
DEFAULT_CKPT = trg16.DEFAULT_CKPT
DEFAULT_RULER = trg16.DEFAULT_RULER
DELTA_SUB = trk32.DELTA_SUB
RHO_DEFAULT = trg16.RHO_DEFAULT

# K32 對照一手來源（本檔不算，只讀）
K32_DRIFT_SUMMARY_JSON = os.path.join(WV_DIR, "drift_analysis", "results", "drift_summary.json")

J_GRID_PRINT = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 49]


# ------------------------------------------------------------------
# 每 chunk 重置對照（跟 run_drift_analysis.run_reset_chunks 同協定，encode/decode 換成 G16K16）
# ------------------------------------------------------------------
def run_reset_chunks_g16k16(env, u, model, obs_mu, obs_sd, raw, task, code_rng=None, garbage_k0=False):
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
            o_raw = torch.from_numpy(sim_obs(u)[None, :])  # 剛 reset，等於真 obs
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
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)          # 同 K32 側，保證同一批 200 題
    ap.add_argument("--garbage-seed", type=int, default=20260911)  # 爛錨對照專用，跟題目抽樣種子分開
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", type=str, default="g16k16_20260911")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== drift-analysis-g16k16  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} garbage_seed={args.garbage_seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg, obs_mu, obs_sd = trg16.load_g16k16(args.ckpt)
    assert cfg["seg_len"] == 4 and cfg["G"] == 16 and cfg["K"] == 16, (
        f"⛔ 預期 G16K16 seg_len=4 G=16 K=16，實際 cfg={cfg}")
    print(f"dict: G={cfg['G']} K={cfg['K']} group_dim={cfg['group_dim']} seg_len={cfg['seg_len']} "
          f"hidden={cfg['hidden']}")

    # ================================================================
    # 載入自檢（同 run_teacher_relay_g16k16.py 的作法，本次重跑一次，不假設上次的還有效）
    # ================================================================
    train_npz = os.path.join(args.data_dir, f"{trk32.DATASET}.npz")
    data = gc.load_segments_with_obs(train_npz, seg_len=cfg["seg_len"],
                                     split_seed=cfg["split_seed"], val_frac=cfg["val_frac"])
    val_x = torch.from_numpy(data["segs"][data["val_idx"]])
    val_o_raw = torch.from_numpy(data["obs_start"][data["val_idx"]])
    val_o_norm = (val_o_raw - obs_mu) / obs_sd
    import torch.nn.functional as F
    with torch.no_grad():
        val_recon, _, _ = model(val_x, val_o_norm, training=False)
        recon_mse_val_check = float(F.mse_loss(val_recon, val_x).item())
    val_mse_diff = abs(recon_mse_val_check - trg16.REF_VAL_MSE_JSON)
    val_mse_match = val_mse_diff < 1e-4
    print(f"[載入自檢] 整個 val 切分（{len(data['val_idx'])} 段）重建 MSE={recon_mse_val_check:.6f} "
          f"vs {trg16.GK_JSON_REF} 記錄值 {trg16.REF_VAL_MSE_JSON:.6f}（差={val_mse_diff:.2e}）—— "
          f"{'吻合，載入正確' if val_mse_match else '⛔ 對不起來，停手回報'}")
    if not val_mse_match:
        sys.exit(1)

    # ================================================================
    # 段集（跟 K32/drift_analysis 側同一顆 seed，見 check_task_comparability.py 的 hash PASS）
    # ================================================================
    with open(args.ruler) as f:
        ruler = json.load(f)
    raw = wv.load_npz(args.data_dir, trk32.DATASET, "val")
    tasks, skipped = trk32.build_tasks(raw, args.n_traj, args.seed, ruler, args.rho)
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}")
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報（不用預設值填）"
    n_chunks_all = np.array([t["n_chunks"] for t in tasks])
    assert (n_chunks_all == n_chunks_all[0]).all(), (
        f"⛔ 預期所有 task 的 n_chunks 一致，實際看到 {np.unique(n_chunks_all)}")
    N_CHUNKS = int(n_chunks_all[0])
    print(f"n_chunks per task = {N_CHUNKS}")

    env = wv.make_env(args.data_dir, trk32.DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================
    # 200 題 x 每題全部 chunk（跟 K32 側 e_indep_baseline 同一種池化方式）
    # ================================================================
    print("\n=== 200 題 x 每題全部 chunk（reset 對照，E_indep）===")
    ta = time.time()
    code_rng = np.random.default_rng(args.garbage_seed)
    reset_xy_err_by_k = {k: [] for k in range(N_CHUNKS)}
    reset_recon_mse_by_k = {k: [] for k in range(N_CHUNKS)}
    garbage_k0_xy = []
    per_task_codes = []
    for ti, task in enumerate(tasks):
        rr = run_reset_chunks_g16k16(env, u, model, obs_mu, obs_sd, raw, task,
                                     code_rng=code_rng, garbage_k0=True)
        for k in range(len(rr["xy_err"])):
            reset_xy_err_by_k[k].append(float(rr["xy_err"][k]))
            reset_recon_mse_by_k[k].append(float(rr["recon_mse"][k]))
        garbage_k0_xy.append(rr["xy_err_garbage_k0"])
        per_task_codes.append(rr["codes"])
    print(f"done in {time.time()-ta:.1f}s")

    pooled_xy = np.concatenate([np.asarray(v) for v in reset_xy_err_by_k.values()])
    pooled_mse = np.concatenate([np.asarray(v) for v in reset_recon_mse_by_k.values()])
    e_indep_xy_overall = float(np.median(pooled_xy))
    e_indep_xy_q = q(pooled_xy)
    mse_q = q(pooled_mse)
    print(f"\nE_indep（全部 chunk 合併，n={len(pooled_xy)}）：xy p25/p50/p75/mean = "
          f"{e_indep_xy_q['25']:.4f}/{e_indep_xy_q['50']:.4f}/{e_indep_xy_q['75']:.4f}/{pooled_xy.mean():.4f}")
    print(f"重建 MSE（decode 前 clip、跟同一批真動作段比，n={len(pooled_mse)}）：p25/p50/p75/mean = "
          f"{mse_q['25']:.5f}/{mse_q['50']:.5f}/{mse_q['75']:.5f}/{pooled_mse.mean():.5f}")

    k0_xy = np.asarray(reset_xy_err_by_k[0])
    garbage_k0_xy = np.asarray(garbage_k0_xy, dtype=float)
    print(f"\n[爛錨對照@k=0] 真字 median={np.median(k0_xy):.4f}m (n={len(k0_xy)})  vs "
          f"隨機字 median={np.median(garbage_k0_xy):.4f}m (n={len(garbage_k0_xy)})  "
          f"比值(隨機/真字)={np.median(garbage_k0_xy)/max(np.median(k0_xy),1e-12):.2f}")

    print(f"\n{'k':>4} {'n':>5} {'xy_p25':>10} {'xy_p50':>10} {'xy_p75':>10} {'mse_p50':>10}")
    accounting_rows = []
    for k in J_GRID_PRINT:
        if k >= N_CHUNKS:
            continue
        v = np.asarray(reset_xy_err_by_k[k])
        m = np.asarray(reset_recon_mse_by_k[k])
        row = dict(k=k, n=int(len(v)), xy_p25=float(np.percentile(v, 25)),
                  xy_p50=float(np.percentile(v, 50)), xy_p75=float(np.percentile(v, 75)),
                  mse_p50=float(np.percentile(m, 50)))
        accounting_rows.append(row)
        print(f"{k:>4} {row['n']:>5} {row['xy_p25']:>10.4f} {row['xy_p50']:>10.4f} "
              f"{row['xy_p75']:>10.4f} {row['mse_p50']:>10.5f}")

    # ================================================================
    # 套用工單預註冊判準（跟 K32 archived e_indep_baseline 比；本檔只算、不判讀）
    # ================================================================
    with open(K32_DRIFT_SUMMARY_JSON) as f:
        k32_drift = json.load(f)
    k32_p50 = k32_drift["e_indep_baseline"]["xy"]
    pct_change = (e_indep_xy_overall - k32_p50) / k32_p50 * 100.0
    # 需要 K32 的 p25/p75 才能判「打平」帶——K32 archived json 沒有直接存 e_indep 池化後的
    # p25/p75（只存了 accounting_table 逐 k 的 p50），所以這裡的判準套用在報告階段用
    # REUSE CHECK 重放時另外算的池化 p25/p75（見報告）；這裡只印 delta%，避免本檔猜資料結構。
    print(f"\nG16K16 E_indep p50={e_indep_xy_overall:.4f}m  vs K32 archived p50={k32_p50:.4f}m  "
          f"Δ%={pct_change:+.1f}%")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(
        config=dict(vars(args)), delta_sub=DELTA_SUB,
        n_tasks_used=len(tasks), n_chunks_per_task=N_CHUNKS,
        load_self_check=dict(full_val_recon_mse_check=recon_mse_val_check,
                             ref_val_mse_json=trg16.REF_VAL_MSE_JSON,
                             diff=val_mse_diff, match=val_mse_match),
        e_indep_baseline=dict(xy_p25=e_indep_xy_q["25"], xy_p50=e_indep_xy_q["50"],
                              xy_p75=e_indep_xy_q["75"], xy_mean=float(pooled_xy.mean()),
                              n=int(len(pooled_xy))),
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
    max_k = N_CHUNKS
    xy_grid = np.full((len(tasks), max_k), np.nan)
    mse_grid = np.full((len(tasks), max_k), np.nan)
    for k in range(N_CHUNKS):
        vals_xy = reset_xy_err_by_k[k]
        vals_mse = reset_recon_mse_by_k[k]
        xy_grid[:len(vals_xy), k] = vals_xy
        mse_grid[:len(vals_mse), k] = vals_mse
    np.savez_compressed(npz_path, xy_err_grid=xy_grid, recon_mse_grid=mse_grid,
                        garbage_k0_xy=garbage_k0_xy, k0_xy=k0_xy)
    print(f"saved: {npz_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
