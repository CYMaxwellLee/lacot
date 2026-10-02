#!/usr/bin/env python
"""行為字典 v0 關1 量測：E_indep 同協定（DESIGN §四 關1）。

在證明什麼／判準：(s,a) 段碼的物理保真 ≥ 純 action 碼。E_indep 同協定 p50
<= .034（16x16 級）起算有戲、<=.027（32x32 級）全勝；recon_a / recon_s 分開報；
爛錨對照要在；段集跟昨天同批（hash 開頭 450be2...）。

跟 experiments/walk_verify/drift_analysis_g16k16/run_drift_analysis_gkdict.py
（昨天 job 28033 用的那份，未改一行、本檔不 import 它只是同協定）的協定完全相同：
  - 同一顆 build_tasks（trk32，seed=20260908 / ruler 預設 / rho 預設 / n_traj=200）
  - 同一種 task 指紋 sha256（episode,s0,e0,M 逐 task 串接），對照同一個期望值
    450be2402fcc5e8fdc5d99f6c81d453e815e32d4a6ab26c3dbf64318f88f46ff
  - 同一種「reset 到真實 (qpos,qvel) → decode → 執行 4 步 → 量終點 xy 誤差」協定
  - 同一種爛錨對照（k=0 時額外跑一次隨機碼、同題目同 obs conditioning）
  - 同一種 pooled percentile（p1_replay.q，25/50/75/mean）
差異只在模型 IO（這正是本工單要測的東西，不是量法本身）：
  - encoder 輸入多了 state 段（hindsight，來自 raw["observations"]，跟 action 段
    一樣是「真值」——跟原本 x=act[s:s+4] 是同一種「hindsight 編碼」邏輯，只是
    多帶一份 s_{t:t+4}）。
  - decode_codes 吐 (â, ŝ) 兩個 tensor，原本只吐一個。â 拿去跟原本完全一樣地
    clip、step、量 xy；ŝ 是新增的重建目標，只拿來算 recon_s，不影響物理執行。
  - decoder 的條件輸入（正規化 s_t）跟原本一模一樣：來自 sim_obs(u)（reset 後、
    跟 raw["observations"][s] 理論上一致，E_indep 定義就是「每段獨立從真狀態
    reset」，跟原本協定同一個假設）。

⛔ CPU-only。不畫圖、不 render gif（跟 run_drift_analysis_gkdict.py 同款精簡）。
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

HERE = os.path.dirname(os.path.abspath(__file__))                       # bcodec_v0/
EXP_DIR = os.path.dirname(HERE)                                          # experiments/
WV_DIR = os.path.join(EXP_DIR, "walk_verify")
TR_DIR = os.path.join(WV_DIR, "teacher_relay")
TRG16_DIR = os.path.join(WV_DIR, "teacher_relay_g16k16")
GK_DIR = os.path.join(EXP_DIR, "gk_scan")
for _p in (HERE, WV_DIR, TR_DIR, TRG16_DIR, GK_DIR):
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q  # noqa: E402
import run_teacher_relay as trk32  # noqa: E402 - build_tasks/DELTA_SUB/DATASET，未改動
import run_teacher_relay_g16k16 as trg16  # noqa: E402 - 只借 DEFAULT_RULER/RHO_DEFAULT/DD 常數
import gk_common as gc  # noqa: E402 - ACT_DIM/OBS_DIM 常數對照用
import bcodec_common as bc  # noqa: E402

DD = trg16.DD
DEFAULT_RULER = trg16.DEFAULT_RULER
DELTA_SUB = trk32.DELTA_SUB
RHO_DEFAULT = trg16.RHO_DEFAULT
EXPECTED_TASK_HASH = "450be2402fcc5e8fdc5d99f6c81d453e815e32d4a6ab26c3dbf64318f88f46ff"

# 昨天（2026-09-11）三側對照，一手來源見各自 summary json（本檔開工前讀過，數字照抄，非本次量測）
YESTERDAY_REF = dict(
    K32=dict(xy_p50=0.04908665063046568, source="walk_verify/drift_analysis/results/drift_summary.json"),
    G16K16=dict(xy_p50=0.03412693774160658, recon_p50=0.014411837328225374,
               garbage_ratio=0.1273311014554535 / 0.028954974197453853,
               source="walk_verify/drift_analysis_g16k16/results/g16k16_20260911_summary.json"),
    G32K32=dict(xy_p50=0.02683719765283904, recon_p50=0.0007967779529280961,
               garbage_ratio=0.1301038987141143 / 0.015017577926152614,
               source="walk_verify/drift_analysis_g16k16/results/g32k32_20260911_summary.json"),
)


def task_hash(tasks):
    h = hashlib.sha256()
    for t in tasks:
        h.update(f"{t['episode']},{t['s0']},{t['e0']},{t['M']}|".encode())
    return h.hexdigest()


def run_reset_chunks_bcodec(env, u, model, obs_mu, obs_sd, raw, task, code_rng=None, garbage_k0=False):
    s0, n_chunks = task["s0"], task["n_chunks"]
    act, obs_arr, qpos_arr, qvel_arr = raw["actions"], raw["observations"], raw["qpos"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    K = model.K
    xy_err, recon_a_mse, recon_s_mse, codes = [], [], [], []
    xy_err_garbage_k0 = None
    obs_mu_np, obs_sd_np = obs_mu.numpy().reshape(-1), obs_sd.numpy().reshape(-1)

    for k in range(n_chunks):
        s = s0 + 4 * k
        env.reset()
        u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
        with torch.no_grad():
            s_seg_true = obs_arr[s:s + 4].reshape(-1).astype(np.float32)             # s_{t:t+4}，hindsight
            s_seg_norm = ((s_seg_true.reshape(4, -1) - obs_mu_np) / obs_sd_np).reshape(1, -1).astype(np.float32)
            a_seg_true = act[s:s + 4].reshape(1, -1).astype(np.float32)              # a_{t:t+4}，hindsight
            x_s = torch.from_numpy(s_seg_norm)
            x_a = torch.from_numpy(a_seg_true)
            z_e = model.encode(x_s, x_a)
            _, idx, _ = model.vq(z_e, training=False)

            o_raw = torch.from_numpy(sim_obs_pad(u, obs_dim=model.obs_dim)[None, :])
            o_norm = (o_raw - obs_mu) / obs_sd
            a_hat_t, s_hat_t = model.decode_codes(idx, o_norm)
            mse_a = float(torch.mean((a_hat_t - x_a) ** 2).item())

            s_next_true = obs_arr[s + 1:s + 5].reshape(-1).astype(np.float32)
            s_next_norm = ((s_next_true.reshape(4, -1) - obs_mu_np) / obs_sd_np).reshape(1, -1).astype(np.float32)
            mse_s = float(torch.mean((s_hat_t - torch.from_numpy(s_next_norm)) ** 2).item())

            raw_a = a_hat_t.numpy().reshape(4, gc.ACT_DIM)
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
        for t in range(4):
            env.step(a_chunk[t])
        end_xy = np.asarray(u.data.qpos)[:2].copy()
        true_end_xy = qpos_arr[s + 4, :2]
        xy_err.append(float(np.linalg.norm(end_xy - true_end_xy)))
        recon_a_mse.append(mse_a)
        recon_s_mse.append(mse_s)
        codes.append(int(idx.item()))

        if garbage_k0 and k == 0:
            assert code_rng is not None
            env.reset()
            u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
            with torch.no_grad():
                idx_g = torch.from_numpy(code_rng.integers(0, K, size=(1,), dtype=np.int64))
                o_raw_g = torch.from_numpy(sim_obs_pad(u, obs_dim=model.obs_dim)[None, :])
                o_norm_g = (o_raw_g - obs_mu) / obs_sd
                a_hat_g, _ = model.decode_codes(idx_g, o_norm_g)
                raw_a_g = a_hat_g.numpy().reshape(4, gc.ACT_DIM)
            a_chunk_g = np.clip(raw_a_g, lo, hi).astype(np.float64)
            for t in range(4):
                env.step(a_chunk_g[t])
            end_xy_g = np.asarray(u.data.qpos)[:2].copy()
            xy_err_garbage_k0 = float(np.linalg.norm(end_xy_g - true_end_xy))

    return dict(xy_err=np.asarray(xy_err), recon_a_mse=np.asarray(recon_a_mse),
                recon_s_mse=np.asarray(recon_s_mse), codes=np.asarray(codes, dtype=np.int64),
                xy_err_garbage_k0=xy_err_garbage_k0)


def sim_obs_pad(u, obs_dim):
    """sim_obs() 固定回傳 29 維（p1_replay 寫死 qpos[:15]+qvel[:14]）。本檔的模型
    obs_dim 就是 29（gc.OBS_DIM），這裡只是把介面講清楚、不做真的 padding。"""
    o = sim_obs(u)
    assert o.shape[0] == obs_dim, f"⛔ sim_obs 維度={o.shape[0]} 跟模型 obs_dim={obs_dim} 不符"
    return o


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
    ap.add_argument("--ref-recon-a", type=float, default=None, help="bcodec_train 記錄的 recon_a_val，載入自檢用")
    ap.add_argument("--ref-recon-s", type=float, default=None, help="bcodec_train 記錄的 recon_s_val，載入自檢用")
    ap.add_argument("--ref-json-path", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== bcodec-drift-eval  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} garbage_seed={args.garbage_seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg, obs_mu, obs_sd = bc.load_bcodec_ckpt(args.ckpt)
    assert cfg["seg_len"] == 4, f"⛔ 預期 seg_len=4，實際 cfg={cfg}"
    print(f"dict: K={cfg['K']} D={cfg['D']} lam_s={cfg['lam_s']} seg_len={cfg['seg_len']} "
          f"hidden={cfg['hidden']}  bits/chunk={np.log2(cfg['K']):.0f}")

    # ================================================================
    # 載入自檢（跟 gkdict 版同款：整個 val 切分重建 recon_a/recon_s 對照訓練當時記錄值）
    # ================================================================
    if args.ref_recon_a is not None:
        train_npz = os.path.join(args.data_dir, f"{trk32.DATASET}.npz")
        data = bc.load_bcodec_segments(train_npz, seg_len=cfg["seg_len"], split_seed=cfg["split_seed"],
                                       val_frac=cfg["val_frac"])
        obs_mu_np, obs_sd_np = obs_mu.numpy(), obs_sd.numpy()
        s_seg_norm = bc._norm_seg(data["s_seg"][data["val_idx"]], obs_mu_np, obs_sd_np, cfg["seg_len"], cfg["obs_dim"])
        s_next_norm = bc._norm_seg(data["s_next"][data["val_idx"]], obs_mu_np, obs_sd_np, cfg["seg_len"], cfg["obs_dim"])
        obs_start_norm = (data["obs_start"][data["val_idx"]] - obs_mu_np) / obs_sd_np
        val_a = torch.from_numpy(data["a_seg"][data["val_idx"]])
        with torch.no_grad():
            # chunked_forward：K=8192 時整個 val 切分（25000 段）一次 forward 會有
            # (25000,8192) 量級的 dist/onehot 暫存，訓練那邊已經因為同款全批 forward
            # 在 N=225000 時 OOM 過一次（見 bcodec_common.chunked_forward 註解），這裡
            # 分塊處理，數值等價、防同一個坑在量測端重演。
            a_hat, s_hat, _ = bc.chunked_forward(model, torch.from_numpy(s_seg_norm), val_a,
                                                 torch.from_numpy(obs_start_norm.astype(np.float32)))
            recon_a_check = float(F.mse_loss(a_hat, val_a).item())
            recon_s_check = float(F.mse_loss(s_hat, torch.from_numpy(s_next_norm)).item())
        diff_a, diff_s = abs(recon_a_check - args.ref_recon_a), abs(recon_s_check - (args.ref_recon_s or 0.0))
        match_a = diff_a < 1e-4
        match_s = (args.ref_recon_s is None) or diff_s < 1e-4
        print(f"[載入自檢] val recon_a={recon_a_check:.6f} vs {args.ref_json_path} 記錄 {args.ref_recon_a:.6f}"
              f"（差={diff_a:.2e}）—— {'吻合' if match_a else '⛔ 對不起來，停手回報'}")
        if args.ref_recon_s is not None:
            print(f"[載入自檢] val recon_s={recon_s_check:.6f} vs 記錄 {args.ref_recon_s:.6f}"
                  f"（差={diff_s:.2e}）—— {'吻合' if match_s else '⛔ 對不起來，停手回報'}")
        if not (match_a and match_s):
            sys.exit(1)
    else:
        recon_a_check = recon_s_check = None
        print("[載入自檢] 未提供 --ref-recon-a，跳過（明標，不假裝驗過）")

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
    is_full_scale = (args.n_traj == 200 and args.seed == 20260908)
    hash_match = (my_hash == EXPECTED_TASK_HASH)
    print(f"[段集指紋自檢] sha256={my_hash}")
    if is_full_scale:
        print(f"[段集指紋自檢] 對照昨天驗證過的值 {EXPECTED_TASK_HASH} —— {'一致' if hash_match else '⛔ 不一致，停手回報'}")
        if not hash_match:
            sys.exit(1)
    else:
        print(f"[段集指紋自檢] n_traj={args.n_traj}!=200 或 seed 非預設 —— smoke/縮小規模跑，本來就不該跟 450be2... 比對，不強制")

    env = wv.make_env(args.data_dir, trk32.DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== N 題 x 每題全部 chunk（reset 對照，E_indep）===")
    ta = time.time()
    code_rng = np.random.default_rng(args.garbage_seed)
    reset_xy_by_k = {k: [] for k in range(N_CHUNKS)}
    reset_a_by_k = {k: [] for k in range(N_CHUNKS)}
    reset_s_by_k = {k: [] for k in range(N_CHUNKS)}
    garbage_k0_xy = []
    for task in tasks:
        rr = run_reset_chunks_bcodec(env, u, model, obs_mu, obs_sd, raw, task, code_rng=code_rng, garbage_k0=True)
        for k in range(len(rr["xy_err"])):
            reset_xy_by_k[k].append(float(rr["xy_err"][k]))
            reset_a_by_k[k].append(float(rr["recon_a_mse"][k]))
            reset_s_by_k[k].append(float(rr["recon_s_mse"][k]))
        garbage_k0_xy.append(rr["xy_err_garbage_k0"])
    print(f"done in {time.time()-ta:.1f}s")

    pooled_xy = np.concatenate([np.asarray(v) for v in reset_xy_by_k.values()])
    pooled_a = np.concatenate([np.asarray(v) for v in reset_a_by_k.values()])
    pooled_s = np.concatenate([np.asarray(v) for v in reset_s_by_k.values()])
    e_indep_q = q(pooled_xy)
    a_q = q(pooled_a)
    s_q = q(pooled_s)
    print(f"\nE_indep（n={len(pooled_xy)}）：xy p25/p50/p75/mean = "
          f"{e_indep_q['25']:.4f}/{e_indep_q['50']:.4f}/{e_indep_q['75']:.4f}/{pooled_xy.mean():.4f}")
    print(f"recon_a（n={len(pooled_a)}）：p25/p50/p75/mean = {a_q['25']:.5f}/{a_q['50']:.5f}/{a_q['75']:.5f}/{pooled_a.mean():.5f}")
    print(f"recon_s（n={len(pooled_s)}）：p25/p50/p75/mean = {s_q['25']:.5f}/{s_q['50']:.5f}/{s_q['75']:.5f}/{pooled_s.mean():.5f}")

    k0_xy = np.asarray(reset_xy_by_k[0])
    garbage_k0_xy = np.asarray(garbage_k0_xy, dtype=float)
    ratio = np.median(garbage_k0_xy) / max(np.median(k0_xy), 1e-12)
    print(f"\n[爛錨對照@k=0] 真字 median={np.median(k0_xy):.4f}m (n={len(k0_xy)})  vs "
          f"隨機字 median={np.median(garbage_k0_xy):.4f}m (n={len(garbage_k0_xy)})  比值(隨機/真字)={ratio:.2f}")

    verdicts = []
    p50 = e_indep_q["50"]
    if p50 <= 0.027:
        verdict = "全勝（<= .027，32x32 級）"
    elif p50 <= 0.034:
        verdict = "有戲（<= .034，16x16 級起算）"
    else:
        verdict = "未過線（> .034）"
    print(f"\n判準：E_indep p50={p50:.4f}  <=.034 有戲 / <=.027 全勝  =>  {verdict}")
    for name, ref in YESTERDAY_REF.items():
        if "xy_p50" in ref:
            print(f"  對照 {name}: p50={ref['xy_p50']:.4f}  Δ%={(p50-ref['xy_p50'])/ref['xy_p50']*100:+.1f}%")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(
        config=dict(vars(args)), delta_sub=DELTA_SUB, dict_cfg=cfg,
        n_tasks_used=len(tasks), n_chunks_per_task=N_CHUNKS, is_full_scale=is_full_scale,
        task_fingerprint=dict(sha256=my_hash, expected=EXPECTED_TASK_HASH, match=hash_match),
        load_self_check=dict(recon_a_check=recon_a_check, recon_s_check=recon_s_check,
                             ref_recon_a=args.ref_recon_a, ref_recon_s=args.ref_recon_s),
        e_indep_baseline=dict(xy_p25=e_indep_q["25"], xy_p50=e_indep_q["50"], xy_p75=e_indep_q["75"],
                              xy_mean=float(pooled_xy.mean()), n=int(len(pooled_xy))),
        recon_a_baseline=dict(p25=a_q["25"], p50=a_q["50"], p75=a_q["75"], mean=float(pooled_a.mean()), n=int(len(pooled_a))),
        recon_s_baseline=dict(p25=s_q["25"], p50=s_q["50"], p75=s_q["75"], mean=float(pooled_s.mean()), n=int(len(pooled_s))),
        garbage_code_control_k0=dict(true_code_xy_p50=float(np.median(k0_xy)),
                                     garbage_code_xy_p50=float(np.median(garbage_k0_xy)),
                                     ratio=float(ratio), n=int(len(k0_xy)), seed=args.garbage_seed),
        verdict=verdict, yesterday_reference=YESTERDAY_REF,
    ), js_path)
    print(f"saved: {js_path}")

    npz_path = os.path.join(args.out_dir, f"{args.tag}_raw.npz")
    np.savez_compressed(npz_path, pooled_xy=pooled_xy, pooled_a=pooled_a, pooled_s=pooled_s,
                        garbage_k0_xy=garbage_k0_xy, k0_xy=k0_xy)
    print(f"saved: {npz_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
