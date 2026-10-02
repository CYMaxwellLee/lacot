#!/usr/bin/env python
"""行為字典 v0 補跑關1：G×K 平行頭版（同容量對照）E_indep 量測。

跟 bcodec_drift_eval.py 協定完全相同（task/hash/reset/garbage-anchor/percentile
都一樣），唯一差異＝模型換成 BehaviorCodecFactorizedVQVAE：
  - model.vq(z_e, training=False) 回傳 idx 形狀 (1,G) 不是 (1,)
  - 爛錨對照的隨機碼要對每一組各自 uniform 抽（size=(1,G)），跟
    run_teacher_relay_g16k16.py 的 random 模式、run_drift_analysis_gkdict.py 的
    garbage_k0 完全同一種做法
  - model.decode_codes(idx,obs_norm) 一樣回傳 (â,ŝ) tuple（跟 v0 單本版介面相同）

⛔ CPU-only。
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

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(HERE)
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
import run_teacher_relay as trk32  # noqa: E402
import run_teacher_relay_g16k16 as trg16  # noqa: E402
import gk_common as gc  # noqa: E402
import bcodec_common as bc  # noqa: E402

DD = trg16.DD
DEFAULT_RULER = trg16.DEFAULT_RULER
DELTA_SUB = trk32.DELTA_SUB
RHO_DEFAULT = trg16.RHO_DEFAULT
EXPECTED_TASK_HASH = "450be2402fcc5e8fdc5d99f6c81d453e815e32d4a6ab26c3dbf64318f88f46ff"

YESTERDAY_REF = dict(
    K32=dict(xy_p50=0.04908665063046568),
    G16K16=dict(xy_p50=0.03412693774160658, recon_p50=0.014411837328225374,
               garbage_ratio=0.1273311014554535 / 0.028954974197453853),
    G32K32=dict(xy_p50=0.02683719765283904, recon_p50=0.0007967779529280961,
               garbage_ratio=0.1301038987141143 / 0.015017577926152614),
)


def task_hash(tasks):
    h = hashlib.sha256()
    for t in tasks:
        h.update(f"{t['episode']},{t['s0']},{t['e0']},{t['M']}|".encode())
    return h.hexdigest()


def run_reset_chunks(env, u, model, obs_mu, obs_sd, raw, task, code_rng=None, garbage_k0=False):
    s0, n_chunks = task["s0"], task["n_chunks"]
    act, obs_arr, qpos_arr, qvel_arr = raw["actions"], raw["observations"], raw["qpos"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    G, K = model.G, model.K
    xy_err, recon_a_mse, recon_s_mse, codes = [], [], [], []
    xy_err_garbage_k0 = None
    obs_mu_np, obs_sd_np = obs_mu.numpy().reshape(-1), obs_sd.numpy().reshape(-1)

    for k in range(n_chunks):
        s = s0 + 4 * k
        env.reset()
        u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
        with torch.no_grad():
            s_seg_true = obs_arr[s:s + 4].reshape(-1).astype(np.float32)
            s_seg_norm = ((s_seg_true.reshape(4, -1) - obs_mu_np) / obs_sd_np).reshape(1, -1).astype(np.float32)
            a_seg_true = act[s:s + 4].reshape(1, -1).astype(np.float32)
            x_s = torch.from_numpy(s_seg_norm)
            x_a = torch.from_numpy(a_seg_true)
            z_e = model.encode(x_s, x_a)
            _, idx, _ = model.vq(z_e, training=False)  # (1, G)

            o_raw = torch.from_numpy(sim_obs(u)[None, :])
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
        codes.append(idx.numpy().reshape(-1).copy())

        if garbage_k0 and k == 0:
            assert code_rng is not None
            env.reset()
            u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
            with torch.no_grad():
                idx_g = torch.from_numpy(code_rng.integers(0, K, size=(1, G), dtype=np.int64))
                o_raw_g = torch.from_numpy(sim_obs(u)[None, :])
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
    ap.add_argument("--ref-recon-a", type=float, default=None)
    ap.add_argument("--ref-recon-s", type=float, default=None)
    ap.add_argument("--ref-json-path", type=str, default=None)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== bcodec-gk-drift-eval  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} garbage_seed={args.garbage_seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg, obs_mu, obs_sd = bc.load_bcodec_gk_ckpt(args.ckpt)
    assert cfg["seg_len"] == 4, f"⛔ 預期 seg_len=4，實際 cfg={cfg}"
    print(f"dict: G={cfg['G']} K={cfg['K']} D={cfg['D']} lam_s={cfg['lam_s']} seg_len={cfg['seg_len']} "
          f"bits/chunk={cfg['G']*np.log2(cfg['K']):.0f}")

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
            a_hat, s_hat, _ = bc.chunked_forward(model, torch.from_numpy(s_seg_norm), val_a,
                                                 torch.from_numpy(obs_start_norm.astype(np.float32)))
            recon_a_check = float(F.mse_loss(a_hat, val_a).item())
            recon_s_check = float(F.mse_loss(s_hat, torch.from_numpy(s_next_norm)).item())
        diff_a = abs(recon_a_check - args.ref_recon_a)
        diff_s = abs(recon_s_check - (args.ref_recon_s or 0.0))
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
        print("[載入自檢] 未提供 --ref-recon-a，跳過")

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
        print("[段集指紋自檢] 縮小規模跑，不強制比對")

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
        rr = run_reset_chunks(env, u, model, obs_mu, obs_sd, raw, task, code_rng=code_rng, garbage_k0=True)
        for k in range(len(rr["xy_err"])):
            reset_xy_by_k[k].append(float(rr["xy_err"][k]))
            reset_a_by_k[k].append(float(rr["recon_a_mse"][k]))
            reset_s_by_k[k].append(float(rr["recon_s_mse"][k]))
        garbage_k0_xy.append(rr["xy_err_garbage_k0"])
    print(f"done in {time.time()-ta:.1f}s")

    pooled_xy = np.concatenate([np.asarray(v) for v in reset_xy_by_k.values()])
    pooled_a = np.concatenate([np.asarray(v) for v in reset_a_by_k.values()])
    pooled_s = np.concatenate([np.asarray(v) for v in reset_s_by_k.values()])
    finite_xy = pooled_xy[np.isfinite(pooled_xy)]
    finite_a = pooled_a[np.isfinite(pooled_a)]
    finite_s = pooled_s[np.isfinite(pooled_s)]
    e_indep_q = q(finite_xy) if len(finite_xy) else dict.fromkeys(('25', '50', '75'), float('nan'))
    a_q = q(finite_a) if len(finite_a) else dict.fromkeys(('25', '50', '75'), float('nan'))
    s_q = q(finite_s) if len(finite_s) else dict.fromkeys(('25', '50', '75'), float('nan'))
    xy_mean = float(finite_xy.mean()) if len(finite_xy) else float('nan')
    a_mean = float(finite_a.mean()) if len(finite_a) else float('nan')
    s_mean = float(finite_s.mean()) if len(finite_s) else float('nan')
    print(f"\nE_indep（n_total={len(pooled_xy)}, n_finite={len(finite_xy)}）：xy p25/p50/p75/mean = "
          f"{e_indep_q['25']:.4f}/{e_indep_q['50']:.4f}/{e_indep_q['75']:.4f}/{xy_mean:.4f}")
    print(f"recon_a（n_total={len(pooled_a)}, n_finite={len(finite_a)}）：p25/p50/p75/mean = {a_q['25']:.5f}/{a_q['50']:.5f}/{a_q['75']:.5f}/{a_mean:.5f}")
    print(f"recon_s（n_total={len(pooled_s)}, n_finite={len(finite_s)}）：p25/p50/p75/mean = {s_q['25']:.5f}/{s_q['50']:.5f}/{s_q['75']:.5f}/{s_mean:.5f}")
    if any(len(finite) < len(total) for finite, total in ((finite_xy, pooled_xy), (finite_a, pooled_a), (finite_s, pooled_s))):
        print("⚠️ 非有限值已從統計中排除；本次結果標為 SUSPECT")

    k0_xy = np.asarray(reset_xy_by_k[0])
    garbage_k0_xy = np.asarray(garbage_k0_xy, dtype=float)
    finite_k0_xy = k0_xy[np.isfinite(k0_xy)]
    finite_garbage_k0_xy = garbage_k0_xy[np.isfinite(garbage_k0_xy)]
    k0_median = float(np.median(finite_k0_xy)) if len(finite_k0_xy) else float('nan')
    garbage_median = float(np.median(finite_garbage_k0_xy)) if len(finite_garbage_k0_xy) else float('nan')
    ratio = garbage_median / max(k0_median, 1e-12)
    print(f"\n[爛錨對照@k=0] 真字 median={k0_median:.4f}m vs 隨機字 median={garbage_median:.4f}m  "
          f"比值={ratio:.2f}")
    if len(finite_k0_xy) < len(k0_xy) or len(finite_garbage_k0_xy) < len(garbage_k0_xy):
        print("⚠️ 爛錨對照含非有限值，已從統計中排除")

    p50 = e_indep_q["50"]
    all_finite = all(len(finite) == len(total) for finite, total in ((finite_xy, pooled_xy), (finite_a, pooled_a), (finite_s, pooled_s), (finite_k0_xy, k0_xy), (finite_garbage_k0_xy, garbage_k0_xy)))
    if not all_finite:
        verdict = "SUSPECT（含非有限值）"
    elif p50 <= 0.027:
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
                              xy_mean=xy_mean, n=int(len(pooled_xy)), n_total=int(len(pooled_xy)), n_finite=int(len(finite_xy))),
        recon_a_baseline=dict(p25=a_q["25"], p50=a_q["50"], p75=a_q["75"], mean=a_mean, n=int(len(pooled_a)), n_total=int(len(pooled_a)), n_finite=int(len(finite_a))),
        recon_s_baseline=dict(p25=s_q["25"], p50=s_q["50"], p75=s_q["75"], mean=s_mean, n=int(len(pooled_s)), n_total=int(len(pooled_s)), n_finite=int(len(finite_s))),
        garbage_code_control_k0=dict(true_code_xy_p50=k0_median,
                                     garbage_code_xy_p50=garbage_median,
                                     ratio=float(ratio), n=int(len(k0_xy)), n_total=int(len(k0_xy)), n_finite=int(len(finite_k0_xy)), seed=args.garbage_seed),
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
