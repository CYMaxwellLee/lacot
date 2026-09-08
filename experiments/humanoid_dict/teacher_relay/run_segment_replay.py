#!/usr/bin/env python
"""段獨立重放（block 1，「ant P1 形狀」）—— humanoidmaze-medium-stitch-v0。

對每個抽到的視窗，每個 4 步 chunk 都「從真起點 set_state」重置一次（跟 ant P1 的
dict_indep／true_chunkcmp／dict_randcode 三組協定同款，見
experiments/walk_verify/p1_replay.py 的 RESET_ARMS 分支）：

  true_chunkcmp   真動作直接播（每 chunk 仍重置到真起點）—— 天花板／對照組
  dict_indep      字典段獨立重放：encoder 壓真動作段 -> decoder（用該段真實錄製 obs
                   條件化）展開 -> 開環執行 4 步
  dict_randcode   打假球對照：同一個還原機、同一個重置協定，但字（每組）是亂挑的
                   （固定 seed=777，跟 ant 版同一個常數）—— 沒有它分不出「到達率高」
                   是字典準還是這個量測量不出差別（ant note 已證實限時到達率在這個
                   協定下會飽和，本檔案主判準跟 ant 一樣是【末端誤差】與【每段位移誤差】）

量測：末端 xy 誤差（視窗開頭累積位移到 gs 步時，離「真人形在 i0+gs 步時的 xy」多遠）
＋ 每段位移誤差（dict_indep / dict_randcode 專有：一個 chunk 展開後的位移 vs 真位移）
＋ 限時到達率／步速／翻倒率／平滑度（跟 ant P1 同款全表，附錄用；ant note 已指出這個
協定下限時到達率是鈍尺，主判準仍是末端誤差＋位移誤差，見 docs/NOTE-2026-09-08-humanoid-relay.md）。

⛔ CPU-only。⛔ 不訓練、只載入 experiments/humanoid_dict/ckpt/ 既有 ckpt。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import hr_common as hr  # noqa: E402
import hd_common as hc  # noqa: E402 (hr_common 已把上一層路徑塞進 sys.path)

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["true_chunkcmp", "dict_indep", "dict_randcode"]
RAND_SEED = 777  # 亂字對照專用，跟 ant 版同一個常數（p1_replay.py 的 rand_rng）


def sample_windows(raw, seg_len, horizon, n_windows, seed):
    starts, ends = hr.episode_bounds(raw["terminals"])
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


def run_arm(arm, env, u, raw, model, obs_mu, obs_sd, G, K, wins, seg_len, horizon, fall_line):
    qpos, act, obs, qvel = raw["qpos"], raw["actions"], raw["observations"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    n_chunks = horizon // seg_len
    W = len(wins)

    pos_traj = np.zeros((W, horizon, 2))
    fell_any = np.zeros(W, dtype=bool)
    speeds = np.zeros((W, horizon), dtype=np.float32)
    zs_all = np.zeros((W, horizon), dtype=np.float32)
    adiff_all = np.zeros((W, horizon - 1), dtype=np.float32)
    codes_used = []  # list of (G,) arrays, dict arms only
    chunk_disp_err = []
    n_clip, n_elem = 0, 0
    rand_rng = np.random.default_rng(RAND_SEED)

    for wi, i0 in enumerate(wins):
        i0 = int(i0)
        pos = qpos[i0, :2].copy()
        acts_win = np.zeros((horizon, hc.ACT_DIM))

        for k in range(n_chunks):
            s = i0 + k * seg_len
            env.reset()
            u.set_state(qpos[s].copy(), qvel[s].copy())
            pos_chunk_base = pos.copy()

            if arm == "true_chunkcmp":
                a_chunk = act[s:s + seg_len].astype(np.float64)
            else:
                with torch.no_grad():
                    x = torch.from_numpy(act[s:s + seg_len].reshape(1, -1).astype(np.float32))
                    idx = hr.encode_idx(model, x)  # (1,G)
                    if arm == "dict_randcode":
                        idx = torch.as_tensor(
                            [[int(rand_rng.integers(0, K)) for _ in range(G)]], dtype=torch.long)
                    o_in = torch.from_numpy(obs[s:s + 1].astype(np.float32))
                    raw_a = hr.decode_from_idx(model, idx, o_in, obs_mu, obs_sd).numpy().reshape(
                        seg_len, hc.ACT_DIM)
                codes_used.append(idx.numpy().reshape(-1).copy())
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

            disp = np.asarray(u.data.qpos)[:2] - qpos[s, :2]
            true_disp = qpos[s + seg_len, :2] - qpos[s, :2]
            if arm in ("dict_indep", "dict_randcode"):
                chunk_disp_err.append(float(np.linalg.norm(disp - true_disp)))
            assert np.allclose(pos, pos_chunk_base + disp, atol=1e-6)

        zmin = zs_all[wi].min()
        fell_any[wi] = bool(zmin < fall_line)
        adiff_all[wi] = np.linalg.norm(np.diff(acts_win, axis=0), axis=1)

    return dict(arm=arm, pos_traj=pos_traj, fell_any=fell_any,
                zs=zs_all, adiff=adiff_all, speeds=speeds,
                chunk_disp_err=np.asarray(chunk_disp_err, dtype=np.float64),
                codes_used=(np.stack(codes_used) if codes_used else np.zeros((0, G), dtype=np.int64)),
                clip_frac=float(n_clip / max(n_elem, 1)), n_chunks_total=W * n_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ruler", type=str, default=hr.RULER_DEFAULT)
    ap.add_argument("--data-dir", type=str, default=hr.DD_DEFAULT)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--goal-steps", type=str, default="8,12,20,40")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== humanoid segment-replay(P1形狀)  ckpt={os.path.basename(args.ckpt)} "
          f"n_windows={args.n_windows} H={args.horizon} goal_steps={args.goal_steps} "
          f"seed={args.seed} tag={args.tag} ===", flush=True)

    model, cfg, obs_mu, obs_sd = hr.load_dict(args.ckpt)
    G, K, seg_len = cfg["G"], cfg["K"], cfg["seg_len"]
    assert seg_len == 4, f"預期 seg_len=4，實際 {seg_len}"
    goal_steps_list = [int(x) for x in args.goal_steps.split(",")]
    assert args.horizon % seg_len == 0
    assert all(g % seg_len == 0 and g <= args.horizon for g in goal_steps_list)
    ruler = hr.load_ruler(args.ruler)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: G={G} K={K} seg_len={seg_len} capacity_bits={G*np.log2(K):.0f}  "
          f"ruler fall_line={fall_line:.4f} goal_tol={hc.GOAL_TOL}", flush=True)

    raw = hc.load_raw(args.data_dir, hc.DATASET_NAME, "val")   # 官方 held-out val
    assert raw["actions"].shape[1] == hc.ACT_DIM and raw["observations"].shape[1] == hc.OBS_DIM
    wins, n_cand = sample_windows(raw, seg_len, args.horizon, args.n_windows, args.seed)
    n_chunks = args.horizon // seg_len
    print(f"windows: {len(wins)} sampled from {n_cand} candidates in {raw['_path']}", flush=True)
    print(f"segments per dict arm = {len(wins)*n_chunks}", flush=True)

    env = hr.make_env(args.data_dir)
    u = env.unwrapped

    out = dict(config=vars(args), dict_cfg=cfg, fall_line=fall_line, goal_tol=hc.GOAL_TOL,
               n_windows=int(len(wins)), n_chunks_per_window=int(n_chunks),
               n_segments_per_dict_arm=int(len(wins) * n_chunks),
               goal_steps_list=goal_steps_list, arms={})
    per_arm_raw = {}

    gdef = {}
    for gs in goal_steps_list:
        gxy = raw["qpos"][wins + gs, :2]
        gd = np.linalg.norm(gxy - raw["qpos"][wins, :2], axis=1)
        Nt, tags = [], []
        for d in gd:
            n, tg = hr.ruler_steps_for_distance(ruler, float(d), "50")
            Nt.append(n)
            tags.append(tg)
        gdef[gs] = dict(goal_xy=gxy, goal_d=gd, N_table=np.asarray(Nt),
                        N_data=np.full(len(wins), float(gs)), tags=tags)
        print(f"goal_steps={gs:3d}: d p50={np.percentile(gd,50):.3f} "
              f"N_table p50={np.percentile(Nt,50):.1f} "
              f"grid={dict(zip(*np.unique(tags, return_counts=True)))}", flush=True)

    for arm in ARMS:
        ta = time.time()
        r = run_arm(arm, env, u, raw, model, obs_mu, obs_sd, G, K, wins, seg_len, args.horizon, fall_line)
        per_arm_raw[arm] = r
        rec = dict(seconds=time.time() - ta)
        rec["reach"] = {}
        for gs in goal_steps_list:
            g = gdef[gs]
            dist = np.linalg.norm(r["pos_traj"] - g["goal_xy"][:, None, :], axis=2)
            in_tol = dist <= hc.GOAL_TOL
            first_hit = np.where(in_tol.any(1), in_tol.argmax(1) + 1, -1)
            sub = dict(end_err_mean=float(dist[:, gs - 1].mean()), end_err_q=hr.q(dist[:, gs - 1]),
                       reach_any_within_H=float((first_hit > 0).mean()))
            for nm in ("N_data", "N_table"):
                for mult in (1.25, 1.5):
                    reached, ev = hr.timed_reach(first_hit, g[nm], mult, args.horizon)
                    nev = int(ev.sum())
                    sub[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
                    sub[f"n_evaluable_{nm}_{mult:g}N"] = nev
            rec["reach"][str(gs)] = sub
        rec["fall_rate_window"] = float(r["fell_any"].mean())
        rec["fall_rate_step"] = float((r["zs"] < fall_line).mean())
        rec["action_diff_q"] = hr.q(r["adiff"].ravel())
        rec["action_diff_mean"] = float(r["adiff"].mean())
        rec["clip_frac"] = r["clip_frac"]
        rec["step_speed_mean"] = float(r["speeds"].mean())
        rec["step_speed_q"] = hr.q(r["speeds"].ravel())
        if arm in ("dict_indep", "dict_randcode"):
            cu = r["codes_used"]  # (n_chunks_total, G)
            per_group = []
            for g in range(G):
                cnt = np.bincount(cu[:, g], minlength=K)
                fr = cnt / max(cnt.sum(), 1)
                nz = fr[fr > 0]
                per_group.append(dict(active=int((cnt > 0).sum()),
                                      perplexity=float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
                                      top1_frac=float(fr.max()) if fr.size else 0.0))
            rec["codes_per_group"] = per_group
            rec["codes_active_min_over_groups"] = int(min(p["active"] for p in per_group))
            rec["codes_perplexity_mean_over_groups"] = float(np.mean([p["perplexity"] for p in per_group]))
        if len(r["chunk_disp_err"]):
            rec["chunk_disp_err_mean"] = float(r["chunk_disp_err"].mean())
            rec["chunk_disp_err_q"] = hr.q(r["chunk_disp_err"])
        out["arms"][arm] = rec
        print(f"[{arm:14s}] done in {rec['seconds']:.0f}s  fall(win)={rec['fall_rate_window']:.3f} "
              f"speed_p50={rec['step_speed_q']['50']:.4f} "
              f"adiff_mean={rec['action_diff_mean']:.3f} clip={rec['clip_frac']*100:.1f}%", flush=True)

    print("\n=== 末端誤差（第 gs 步時離 goal 的距離，公尺）===", flush=True)
    hdr = f"{'arm':<15}" + "".join(f"{'gs=%d' % g:>10}" for g in goal_steps_list)
    print(hdr, flush=True)
    for arm in ARMS:
        print(f"{arm:<15}" + "".join(
            f"{out['arms'][arm]['reach'][str(g)]['end_err_mean']:>10.3f}" for g in goal_steps_list), flush=True)

    print("\n=== 每 chunk 位移誤差（公尺，僅 dict 組）===", flush=True)
    for arm in ("dict_indep", "dict_randcode"):
        rec = out["arms"][arm]
        if "chunk_disp_err_mean" in rec:
            print(f"{arm:<15} mean={rec['chunk_disp_err_mean']:.4f} p50={rec['chunk_disp_err_q']['50']:.4f}",
                  flush=True)

    print("\n=== 對照（一手引用自 docs/NOTE-2026-09-08-teacher-relay.md 前身 walk-verify §2.3，gs=40 那欄）===")
    print("ant: true_chunkcmp(真)=0.121  dict_indep(字典)=0.199  dict_randcode(亂字)=0.655", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    cap = 300000

    def sub(a):
        a = np.asarray(a, dtype=float).ravel()
        if len(a) <= cap:
            return a
        return a[np.random.default_rng(0).choice(len(a), cap, replace=False)]

    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    labels = {"true_chunkcmp": "true (chunk-reset)", "dict_indep": "dict independent",
              "dict_randcode": "RANDOM code (control)"}
    pngs = {}
    for key, title, xlab, vl in (
        ("speed", "P1(humanoid) gate2: step speed vs real humanoid", "|dxy| per step", None),
        ("z", "P1(humanoid) gate3a: torso height vs real humanoid", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "P1(humanoid) gate3b: action smoothness vs real humanoid",
         "L2 |a_t - a_(t-1)|", None),
    ):
        ref = {"speed": "step_speed", "z": "torso_z", "adiff": "action_diff"}[key]
        series = [dict(label="REAL humanoid (ruler)", values=sub(ruler_np[ref]))]
        series += [dict(label=labels[a], values=sub(per_arm_raw[a][
            {"speed": "speeds", "z": "zs", "adiff": "adiff"}[key]])) for a in ARMS]
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        hc.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}", flush=True)
    out["artifacts"] = pngs

    js = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    hr.save_json(out, js)
    print(f"saved: {js}", flush=True)
    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        wins=wins,
                        **{f"{a}_fell_any": per_arm_raw[a]["fell_any"] for a in ARMS})
    print(f"=== done wall={time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
