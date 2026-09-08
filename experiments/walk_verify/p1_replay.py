#!/usr/bin/env python
"""P1：重放測試 —— 四組對照，每組都在 MuJoCo 裡真的執行（set_state 到段起點後開環執行）。

視窗（window）定義：從官方 held-out val 檔（antmaze-medium-stitch-v0-val.npz、
完全沒進過訓練）抽 episode 與對齊 chunk 的起點 i0，模擬 horizon H 步（預設 80）。
  goal    = 真螞蟻在 i0+GOAL_STEPS（預設 40）時的 xy；d = |goal - xy_i0|
  N_data  = GOAL_STEPS（真螞蟻自己走完這一段用掉的步數）      ← 最緊、自我校準的尺
  N_table = 量尺包 distance->steps 表的 p50（同距離、全資料）  ← P2 要用的那把尺
  限時到達 = 在 ceil(mult*N) 步內、任一步 |pos_t - goal| <= goal_tol(0.5)
  ⚠️ budget > H 的視窗標成 not_evaluable（不填預設值、不當成失敗）

四組（arm）：
  true          真動作直接重放（天花板＋判定尺自檢）
  true_chunkcmp 真動作但每 chunk 從真起點重置、位移累加（dict_indep 合成法的對照組）
  dict_indep    字典段獨立重放：每 chunk 重置到真起點，壓真動作段→字→展開→執行、位移累加
  dict_cont     字典連續重放：只在視窗開頭重置一次，之後連續壓字、用當下 obs 展開執行

⛔ CPU-only（MuJoCo 物理，不 render、不碰 GPU）。
"""
import argparse
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["true", "true_chunkcmp", "dict_indep", "dict_cont", "dict_randcode"]
RESET_ARMS = ("true_chunkcmp", "dict_indep", "dict_randcode")
DICT_ARMS = ("dict_indep", "dict_cont", "dict_randcode")
# dict_randcode ＝ 打假球對照：同一台還原機、同一個重置協定，但【字是亂挑的】。
# ⛔ 沒有它就分不清「到達率高」是字典準、還是這個量測根本量不出差別。
# ⇒ 它必須明顯崩掉；沒崩＝這一關的數字不能用。


def sample_windows(raw, seg_len, horizon, n_windows, seed):
    starts, ends = wv.episode_bounds(raw["terminals"])
    rng = np.random.default_rng(seed)
    cands = []
    for s0, e0 in zip(starts, ends):
        T = int(e0 - s0 + 1)
        max_off = T - horizon - 1
        for off in range(0, max_off + 1, seg_len):
            cands.append(int(s0 + off))
    cands = np.array(cands, dtype=np.int64)
    if n_windows >= len(cands):
        return cands, len(cands)
    return np.sort(rng.choice(cands, size=n_windows, replace=False)), len(cands)


def sim_obs(u):
    return np.concatenate([np.asarray(u.data.qpos)[:15],
                           np.asarray(u.data.qvel)[:14]]).astype(np.float32)


def run_arm(arm, env, u, raw, model, wins, seg_len, horizon, fall_line):
    """模擬一次、把整條（合成或真實）xy 軌跡存下來，到達率之後對任意 goal_steps 後算。"""
    qpos, act, obs, qvel = raw["qpos"], raw["actions"], raw["observations"], raw["qvel"]
    lo, hi = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
    n_chunks = horizon // seg_len
    W = len(wins)

    pos_traj = np.zeros((W, horizon, 2))      # 每步之後的（合成）位置
    fell_any = np.zeros(W, dtype=bool)
    speeds = np.zeros((W, horizon), dtype=np.float32)
    zmin = np.zeros(W, dtype=np.float32)
    zs_all = np.zeros((W, horizon), dtype=np.float32)
    adiff_all = np.zeros((W, horizon - 1), dtype=np.float32)
    codes_used = []
    chunk_disp_err = []
    n_clip, n_elem = 0, 0
    rand_rng = np.random.default_rng(777)   # dict_randcode 專用、固定可重現

    for wi, i0 in enumerate(wins):
        i0 = int(i0)
        pos = qpos[i0, :2].copy()
        acts_win = np.zeros((horizon, wv.ACT_DIM))

        env.reset()
        if arm not in RESET_ARMS:
            u.set_state(qpos[i0].copy(), qvel[i0].copy())

        for k in range(n_chunks):
            s = i0 + k * seg_len
            if arm in RESET_ARMS:
                env.reset()
                u.set_state(qpos[s].copy(), qvel[s].copy())
                pos_chunk_base = pos.copy()

            if arm in ("true", "true_chunkcmp"):
                a_chunk = act[s:s + seg_len].astype(np.float64)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(act[s:s + seg_len].reshape(1, -1).astype(np.float32))
                    idx = model.encode_idx(x)
                    if arm == "dict_randcode":       # 打假球對照：字亂挑（固定 seed）
                        idx = torch.as_tensor([int(rand_rng.integers(0, model.K))])
                    o_in = (torch.from_numpy(sim_obs(u)[None, :]) if arm == "dict_cont"
                            else torch.from_numpy(obs[s:s + 1].astype(np.float32)))
                    raw_a = model.decode_from_idx(idx, o_in).numpy().reshape(seg_len, wv.ACT_DIM)
                codes_used.append(int(idx.item()))
                a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
                n_clip += int(np.sum(raw_a != a_chunk.astype(np.float32)))
                n_elem += a_chunk.size

            prev_xy = np.asarray(u.data.qpos)[:2].copy()
            for t in range(seg_len):
                gstep = k * seg_len + t
                env.step(a_chunk[t])
                cur = np.asarray(u.data.qpos)[:2].copy()
                dstep = cur - prev_xy
                pos = pos + dstep
                prev_xy = cur
                zs_all[wi, gstep] = float(np.asarray(u.data.qpos)[2])
                speeds[wi, gstep] = float(np.linalg.norm(dstep))
                acts_win[gstep] = a_chunk[t]
                pos_traj[wi, gstep] = pos

            if arm in RESET_ARMS:
                disp = np.asarray(u.data.qpos)[:2] - qpos[s, :2]
                true_disp = qpos[s + seg_len, :2] - qpos[s, :2]
                if arm in ("dict_indep", "dict_randcode"):
                    chunk_disp_err.append(float(np.linalg.norm(disp - true_disp)))
                assert np.allclose(pos, pos_chunk_base + disp, atol=1e-8)

        zmin[wi] = zs_all[wi].min()
        fell_any[wi] = bool(zmin[wi] < fall_line)
        adiff_all[wi] = np.linalg.norm(np.diff(acts_win, axis=0), axis=1)

    return dict(arm=arm, pos_traj=pos_traj, fell_any=fell_any,
                zs=zs_all, adiff=adiff_all, zmin=zmin, speeds=speeds,
                chunk_disp_err=np.asarray(chunk_disp_err, dtype=np.float64),
                codes_used=np.asarray(codes_used, dtype=np.int64),
                clip_frac=float(n_clip / max(n_elem, 1)), n_chunks_total=W * n_chunks)


def timed_reach(first_hit, N, mult, horizon):
    """回傳 (reached, evaluable) 兩個 bool 陣列。budget>horizon 的視窗算 not evaluable。"""
    budget = np.ceil(np.asarray(N, dtype=float) * mult)
    evaluable = budget <= horizon
    reached = (first_hit > 0) & (first_hit <= budget)
    return reached, evaluable


def q(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {str(x): None for x in qs}
    return {str(x): float(np.percentile(a, x)) for x in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ruler", type=str, default=os.path.join(HERE, "results", "ruler_pack.json"))
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--goal-steps", type=str, default="8,12,20,40",
                    help="逗號分隔的 goal horizon（同一次模擬、多把尺一起算）")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--tag", type=str, default="p1")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== P1 replay test  ckpt={os.path.basename(args.ckpt)} "
          f"n_windows={args.n_windows} H={args.horizon} goal_steps={args.goal_steps} "
          f"seed={args.seed} ===")

    model, cfg = wv.load_dict_v1(args.ckpt)
    seg_len = cfg["seg_len"]
    goal_steps_list = [int(x) for x in args.goal_steps.split(",")]
    assert args.horizon % seg_len == 0
    assert all(g % seg_len == 0 and g <= args.horizon for g in goal_steps_list)
    with open(args.ruler) as f:
        import json
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: seg_len={seg_len} K={cfg['k']} conditional_decoder={cfg['conditional_decoder']}")
    print(f"ruler: fall_line(p1 of torso z)={fall_line:.4f} goal_tol={wv.GOAL_TOL}")

    raw = wv.load_npz(args.data_dir, DATASET, "val")   # 官方 held-out val
    wins, n_cand = sample_windows(raw, seg_len, args.horizon, args.n_windows, args.seed)
    n_chunks = args.horizon // seg_len
    print(f"windows: {len(wins)} sampled from {n_cand} candidates in {raw['_path']}")
    print(f"segments per dict arm = {len(wins)*n_chunks}  (要求 >= 10000)")

    env = wv.make_env(args.data_dir, DATASET)
    u = env.unwrapped

    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    out = dict(config=vars(args), ruler_file=args.ruler, dict_cfg=cfg,
               fall_line=fall_line, goal_tol=wv.GOAL_TOL,
               n_windows=int(len(wins)), n_chunks_per_window=int(n_chunks),
               n_segments_per_dict_arm=int(len(wins) * n_chunks),
               goal_steps_list=goal_steps_list, arms={}, gates={})
    per_arm_raw = {}

    # 每個 goal horizon 的距離與兩種 N
    gdef = {}
    for gs in goal_steps_list:
        gxy = raw["qpos"][wins + gs, :2]
        gd = np.linalg.norm(gxy - raw["qpos"][wins, :2], axis=1)
        Nt, tags = [], []
        for d in gd:
            n, tg = wv.ruler_steps_for_distance(ruler, float(d), "50")
            Nt.append(n)
            tags.append(tg)
        gdef[gs] = dict(goal_xy=gxy, goal_d=gd, N_table=np.asarray(Nt),
                        N_data=np.full(len(wins), float(gs)), tags=tags)
        print(f"goal_steps={gs:3d}: d p50={np.percentile(gd,50):.2f} "
              f"N_table p50={np.percentile(Nt,50):.1f} "
              f"grid={dict(zip(*np.unique(tags, return_counts=True)))}")

    for arm in ARMS:
        ta = time.time()
        r = run_arm(arm, env, u, raw, model, wins, seg_len, args.horizon, fall_line)
        per_arm_raw[arm] = r
        rec = dict(seconds=time.time() - ta)
        # ---- 第一關：限時到達率（每個 goal horizon × 兩種 N × 兩個倍率）----
        rec["reach"] = {}
        for gs in goal_steps_list:
            g = gdef[gs]
            dist = np.linalg.norm(r["pos_traj"] - g["goal_xy"][:, None, :], axis=2)  # (W,H)
            in_tol = dist <= wv.GOAL_TOL
            first_hit = np.where(in_tol.any(1), in_tol.argmax(1) + 1, -1)
            sub = dict(end_err_mean=float(dist[:, gs - 1].mean()),
                       end_err_q=q(dist[:, gs - 1]),
                       traj_dev_final_q=q(np.linalg.norm(
                           r["pos_traj"][:, gs - 1] - raw["qpos"][wins + gs, :2], axis=1)),
                       reach_any_within_H=float((first_hit > 0).mean()),
                       first_hit_p50_of_reached=(float(np.median(first_hit[first_hit > 0]))
                                                 if (first_hit > 0).any() else None))
            for nm in ("N_data", "N_table"):
                for mult in (1.25, 1.5):
                    reached, ev = timed_reach(first_hit, g[nm], mult, args.horizon)
                    nev = int(ev.sum())
                    sub[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
                    sub[f"n_evaluable_{nm}_{mult:g}N"] = nev
                    sub[f"n_not_evaluable_{nm}_{mult:g}N"] = int((~ev).sum())
            sub["n_reached_windows"] = int((first_hit > 0).sum())
            sub["step_speed_reached_only_q"] = (q(r["speeds"][first_hit > 0].ravel())
                                                if (first_hit > 0).any() else None)
            rec["reach"][str(gs)] = sub
        rec["fall_rate_window"] = float(r["fell_any"].mean())
        rec["fall_rate_step"] = float((r["zs"] < fall_line).mean())
        rec["torso_z_q"] = q(r["zs"].ravel())
        rec["action_diff_q"] = q(r["adiff"].ravel())
        rec["action_diff_mean"] = float(r["adiff"].mean())
        rec["clip_frac"] = r["clip_frac"]
        rec["step_speed_mean"] = float(r["speeds"].mean())
        rec["step_speed_q"] = q(r["speeds"].ravel())
        if arm in DICT_ARMS:
            cu = r["codes_used"]
            cnt = np.bincount(cu, minlength=cfg["k"])
            fr = cnt / max(cnt.sum(), 1)
            nz = fr[fr > 0]
            rec["codes_active"] = int((cnt > 0).sum())
            rec["codes_perplexity"] = float(np.exp(-np.sum(nz * np.log(nz))))
            rec["codes_top1_frac"] = float(fr.max())
        if len(r["chunk_disp_err"]):
            rec["chunk_disp_err_mean"] = float(r["chunk_disp_err"].mean())
            rec["chunk_disp_err_q"] = q(r["chunk_disp_err"])
        out["arms"][arm] = rec
        print(f"[{arm:14s}] done in {rec['seconds']:.0f}s  fall(win)={rec['fall_rate_window']:.3f} "
              f"speed_p50={rec['step_speed_q']['50']:.4f} "
              f"adiff_mean={rec['action_diff_mean']:.3f} clip={rec['clip_frac']*100:.1f}%")

    # ---- 第一關全表 ----
    print("\n=== 第一關｜限時到達率（%），goal_tol=0.5，budget>H 的視窗標 n/a 不算 ===")
    for nm in ("N_data", "N_table"):
        print(f"\n-- N = {nm} --")
        hdr = f"{'arm':<15}" + "".join(f"{'gs=%d 1.25N' % g:>13}{'1.5N':>8}" for g in goal_steps_list)
        print(hdr)
        for arm in ARMS:
            row = f"{arm:<15}"
            for gs in goal_steps_list:
                s = out["arms"][arm]["reach"][str(gs)]
                for mult in (1.25, 1.5):
                    v = s[f"reach_{nm}_{mult:g}N"]
                    nev = s[f"n_not_evaluable_{nm}_{mult:g}N"]
                    cell = "n/a" if v is None else f"{v*100:.1f}" + ("*" if nev else "")
                    row += f"{cell:>13}" if mult == 1.25 else f"{cell:>8}"
            print(row)
    print("  (* = 有視窗因 budget>H 未納入；n 值在 json 的 n_not_evaluable_*)")
    print("\n=== 末端誤差（第 gs 步時離 goal 的距離，公尺）===")
    print(f"{'arm':<15}" + "".join(f"{'gs=%d' % g:>10}" for g in goal_steps_list))
    for arm in ARMS:
        print(f"{arm:<15}" + "".join(
            f"{out['arms'][arm]['reach'][str(g)]['end_err_mean']:>10.3f}" for g in goal_steps_list))

    # ---- 三張疊圖（vs 量尺包）----
    os.makedirs(args.out_dir, exist_ok=True)
    cap = 300000

    def sub(a):
        a = np.asarray(a, dtype=float).ravel()
        if len(a) <= cap:
            return a
        return a[np.random.default_rng(0).choice(len(a), cap, replace=False)]

    labels = {"true": "true actions", "true_chunkcmp": "true (chunk-reset)",
              "dict_indep": "dict independent", "dict_cont": "dict continuous",
              "dict_randcode": "RANDOM code (control)"}
    pngs = {}
    for key, getter, title, xlab, vl in (
        ("speed", lambda r: r["speeds"], "P1 gate2: step speed vs real ant",
         "|dxy| per step", None),
        ("z", lambda r: r["zs"], "P1 gate3a: torso height vs real ant", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", lambda r: r["adiff"], "P1 gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        ref = {"speed": "step_speed", "z": "torso_z", "adiff": "action_diff"}[key]
        series = [dict(label="REAL ant (ruler)", values=sub(ruler_np[ref]))]
        series += [dict(label=labels[a], values=sub(per_arm_raw[a][
            {"speed": "speeds", "z": "zs", "adiff": "adiff"}[key]])) for a in ARMS]
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        wv.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")
    out["artifacts"] = pngs

    js = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(out, js)
    print(f"saved: {js}")
    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        wins=wins,
                        **{f"goal_d_{g}": gdef[g]["goal_d"] for g in goal_steps_list},
                        **{f"N_table_{g}": gdef[g]["N_table"] for g in goal_steps_list},
                        **{f"{a}_pos_traj": per_arm_raw[a]["pos_traj"].astype(np.float32)
                           for a in ARMS},
                        **{f"{a}_{k}": per_arm_raw[a][k] for a in ARMS
                           for k in ("zmin", "fell_any")})
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
