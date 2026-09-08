#!/usr/bin/env python
"""P2 分析：從 vq_oracle eval 落下的軌跡算四關 ＋ render 接力實走 gif。

輸入 = LACOT_VQO_TRACE_DIR 裡的 trace_t{task}_s{seed}.npz
  每個 chunk 一列：段起點 qpos(15)/qvel(14)、實際執行的 4 步動作、選到的字、當時路標
  另存 goal_xy / success / steps
⇒ 在 MuJoCo 裡把 trace 重跑一次（bit-exact：qpos/qvel 是 float64 直接存的）
  就能拿到每一步的 xy / z，四關全部量得出來。

四關：
  ① 限時到達率：全集（goal，goal_tol=0.5）＋ per-leg（路標，rho）各兩版（1.25N/1.5N）
  ② 步速分佈 vs 量尺包
  ③ 翻倒率 ＋ 平滑度分佈 vs 量尺包
  ④ 人眼關：接力實走 gif（成功與失敗都出）
⛔ CPU-only（MUJOCO_GL=osmesa 軟體算圖）。
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import wv_common as wv  # noqa: E402

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
HERE = os.path.dirname(os.path.abspath(__file__))
RHO_DEFAULT = 1.875   # planner 自己的到達半徑 = 0.25*DELTA_SUB（lacot/subgoal.py:230）


MAZE_CENTER = (9.42, 10.24)   # 從資料 xy 範圍量出（x -1.74~21.43, y -1.43~21.24）


def replay_trace(env, u, tr, render=False, renderer=None, cams=None, frame_every=1,
                 max_frames=200):
    """把一條 trace 重跑：回傳每步 xy / z / action，選擇性收 render 幀。

    render 時出【兩格】並排：左＝跟拍（看步態），右＝全迷宮固定機位（看走到哪）——
    只有跟拍的話螞蟻永遠在畫面正中，⛔ 看不出它到底有沒有前進。
    """
    qpos, qvel, acts = tr["qpos"], tr["qvel"], tr["actions"]
    n_chunk, seg_len, adim = acts.shape
    xs = np.zeros((n_chunk * seg_len, 2))
    zs = np.zeros(n_chunk * seg_len)
    frames = []
    env.reset()
    for k in range(n_chunk):
        u.set_state(qpos[k].copy(), qvel[k].copy())   # 每 chunk 對齊實際 eval 的狀態
        for t in range(seg_len):
            u.step(acts[k, t])
            g = k * seg_len + t
            xs[g] = np.asarray(u.data.qpos)[:2]
            zs[g] = float(np.asarray(u.data.qpos)[2])
            if render and (g % frame_every == 0) and len(frames) < max_frames:
                follow, wide = cams
                follow.lookat[0], follow.lookat[1] = float(xs[g][0]), float(xs[g][1])
                renderer.update_scene(u.data, camera=follow)
                fa = renderer.render().copy()
                renderer.update_scene(u.data, camera=wide)
                fb = renderer.render().copy()
                frames.append(np.concatenate([fa, fb], axis=1))
    a_flat = acts.reshape(-1, adim)
    return xs, zs, a_flat, frames


def build_legs(subs, seg_len):
    """從每 chunk 的路標序列切出 leg：路標換了就是新的一段。回傳 [(k0, k1, sub)]。"""
    legs, k0 = [], 0
    for k in range(1, len(subs)):
        if not np.allclose(subs[k], subs[k0]):
            legs.append((k0, k - 1, subs[k0]))
            k0 = k
    legs.append((k0, len(subs) - 1, subs[k0]))
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=str, default=os.path.join(HERE, "results", "p2_traces"))
    ap.add_argument("--ruler", type=str, default=os.path.join(HERE, "results", "ruler_pack.json"))
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--gif-dir", type=str, default=os.path.join(HERE, "results", "p2_gifs"))
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--n-gif-success", type=int, default=4)
    ap.add_argument("--n-gif-fail", type=int, default=4)
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--tag", type=str, default="p2")
    ap.add_argument("--max-traces", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.trace_dir, "trace_t*_s*.npz")))
    if not files:
        print(f"⛔ 找不到任何 trace：{args.trace_dir} —— 停手，不用預設值填")
        sys.exit(2)
    if args.max_traces:
        files = files[:args.max_traces]
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"=== P2 分析：{len(files)} 條 trace，rho={args.rho} fall_line={fall_line:.4f} "
          f"goal_tol={wv.GOAL_TOL} ===")

    env = wv.make_env(DD, DATASET)
    u = env.unwrapped

    ep_rows, all_speed, all_z, all_adiff, leg_rows = [], [], [], [], []
    code_counts = None
    for fp in files:
        tr = np.load(fp)
        xs, zs, a_flat, _ = replay_trace(env, u, tr)
        goal = tr["goal_xy"]
        succ = bool(tr["success"][0])
        start_xy = tr["qpos"][0][:2]
        d_goal = float(np.linalg.norm(goal - start_xy))
        dist = np.linalg.norm(xs - goal, axis=1)
        in_tol = dist <= wv.GOAL_TOL
        first_hit = int(np.argmax(in_tol) + 1) if in_tol.any() else -1
        Nt, tag = wv.ruler_steps_for_distance(ruler, d_goal, "50")
        ep_rows.append(dict(file=os.path.basename(fp), success=succ,
                            steps=int(tr["steps"][0]), d_goal=d_goal, N_table=Nt,
                            grid_tag=tag, first_hit=first_hit, n_steps_traced=len(xs),
                            final_dist=float(dist[-1])))
        all_speed.append(np.linalg.norm(np.diff(xs, axis=0), axis=1))
        all_z.append(zs)
        all_adiff.append(np.linalg.norm(np.diff(a_flat, axis=0), axis=1))
        c = np.bincount(tr["codes"], minlength=32)
        code_counts = c if code_counts is None else code_counts + c

        # per-leg
        for (k0, k1, sub) in build_legs(tr["subs"], tr["actions"].shape[1]):
            leg_start = tr["qpos"][k0][:2]
            d_leg = float(np.linalg.norm(sub - leg_start))
            span = slice(k0 * 4, (k1 + 1) * 4)
            dl = np.linalg.norm(xs[span] - sub, axis=1)
            reached_at = int(np.argmax(dl <= args.rho) + 1) if (dl <= args.rho).any() else -1
            # 真螞蟻把「離路標 d 縮到 rho 以內」要幾步 ⇒ 查表用 max(d-rho,0)
            Nl, ltag = wv.ruler_steps_for_distance(ruler, max(d_leg - args.rho, 1e-6), "50")
            leg_rows.append(dict(d=d_leg, N=Nl, grid_tag=ltag, steps_used=(k1 - k0 + 1) * 4,
                                 reached_at=reached_at, is_last=(k1 == len(tr["subs"]) - 1)))

    speed = np.concatenate(all_speed)
    zall = np.concatenate(all_z)
    adiff = np.concatenate(all_adiff)

    # ---- 第一關 ----
    def rate(rows, key_hit, key_N, mult):
        ok, ev = 0, 0
        for r in rows:
            b = np.ceil(r[key_N] * mult)
            ev += 1
            if 0 < r[key_hit] <= b:
                ok += 1
        return ok / max(ev, 1), ev

    n_succ = sum(r["success"] for r in ep_rows)
    print(f"\n--- 全鏈（參考錶）---")
    print(f"episodes={len(ep_rows)}  官方 success={n_succ/len(ep_rows):.3f}  "
          f"(進 goal_tol 過的集數 {sum(r['first_hit']>0 for r in ep_rows)}/{len(ep_rows)})")
    print("\n=== ① 限時到達率 ===")
    print(f"{'尺':<26}{'1.25N':>9}{'1.5N':>9}{'n':>8}")
    for nm, rows, kh, kN in (("全集 goal（N_table）", ep_rows, "first_hit", "N_table"),
                             ("per-leg 路標（N_table）", [r for r in leg_rows if not r["is_last"]],
                              "reached_at", "N")):
        r125, n = rate(rows, kh, kN, 1.25)
        r150, _ = rate(rows, kh, kN, 1.5)
        print(f"{nm:<26}{r125*100:>8.1f}%{r150*100:>8.1f}%{n:>8d}")
    legs_ok = [r for r in leg_rows if not r["is_last"]]
    print(f"per-leg 不限時到達率（對照 25987 主頭 .320）: "
          f"{np.mean([r['reached_at']>0 for r in legs_ok]):.3f}  "
          f"median steps/leg = {np.median([r['steps_used'] for r in legs_ok]):.0f}  "
          f"legs={len(legs_ok)}（末段未結算 {len(leg_rows)-len(legs_ok)} 段，同官方口徑排除）")

    # ---- 二、三關 ----
    ss, tz, ad = ruler["step_speed"], ruler["torso_z"], ruler["action_smoothness"]
    print("\n=== ② 步速（vs 量尺包）===")
    print(f"{'':<12}{'p25':>9}{'p50':>9}{'p75':>9}{'mean':>9}")
    print(f"{'real ant':<12}{ss['quantiles']['25']:>9.4f}{ss['quantiles']['50']:>9.4f}"
          f"{ss['quantiles']['75']:>9.4f}{ss['mean']:>9.4f}")
    print(f"{'vq_oracle':<12}{np.percentile(speed,25):>9.4f}{np.percentile(speed,50):>9.4f}"
          f"{np.percentile(speed,75):>9.4f}{speed.mean():>9.4f}")
    print("\n=== ③ 翻倒率＋平滑度 ===")
    print(f"翻倒（z<{fall_line:.4f}）：逐步 {np.mean(zall<fall_line)*100:.2f}% / "
          f"逐集 {np.mean([np.min(z)<fall_line for z in all_z])*100:.1f}%  "
          f"（真資料定義上逐步 1.00%）")
    print(f"平滑度 |a_t-a_(t-1)|：real p50={ad['quantiles']['50']:.3f} mean={ad['mean']:.3f}  |  "
          f"vq_oracle p50={np.percentile(adiff,50):.3f} mean={adiff.mean():.3f}")
    fr = code_counts / max(code_counts.sum(), 1)
    nz = fr[fr > 0]
    print(f"選字分佈：active {int((code_counts>0).sum())}/32  "
          f"perplexity {np.exp(-np.sum(nz*np.log(nz))):.2f}  top1 {fr.max()*100:.1f}%")

    # ---- 疊圖 ----
    rp = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    os.makedirs(args.out_dir, exist_ok=True)
    pngs = {}
    for key, vals, ref, title, xlab, vl in (
        ("speed", speed, "step_speed", "P2 gate2: step speed vs real ant", "|dxy| per step", None),
        ("z", zall, "torso_z", "P2 gate3a: torso height vs real ant", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", adiff, "action_diff", "P2 gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        wv.draw_hist_overlay([dict(label="REAL ant (ruler)", values=rp[ref]),
                              dict(label="vq_oracle relay", values=vals)],
                             p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")

    # ---- ④ gif ----
    import mujoco
    import imageio.v2 as imageio
    os.makedirs(args.gif_dir, exist_ok=True)
    env.reset()
    u.set_state(np.asarray(u.data.qpos).copy(), np.asarray(u.data.qvel).copy())
    _ = u.render()
    renderer = u.custom_renderer
    cam = mujoco.MjvCamera()          # 跟拍：看步態
    cam.lookat[2] = 0.3
    cam.distance = 6.0
    cam.elevation = -35
    cam.azimuth = 135
    cam_wide = mujoco.MjvCamera()     # 全迷宮固定機位：看有沒有真的前進
    cam_wide.lookat[0], cam_wide.lookat[1] = MAZE_CENTER
    cam_wide.lookat[2] = 0.0
    cam_wide.distance = 38.0
    cam_wide.elevation = -75
    cam_wide.azimuth = 90
    cams = (cam, cam_wide)

    def spread(rows, n):
        """跨 task 取樣，⛔ 不要 8 支都來自 task 1（主人要看的是不同題目長什麼樣）。"""
        by_task = {}
        for r in rows:
            t = r["file"].split("_")[1]     # trace_t{task}_s{seed}.npz
            by_task.setdefault(t, []).append(r["file"])
        out, i = [], 0
        while len(out) < n and any(by_task.values()):
            for t in sorted(by_task):
                if by_task[t] and len(out) < n:
                    out.append(by_task[t].pop(0))
            i += 1
            if i > n + 5:
                break
        return out

    succ_f = spread([r for r in ep_rows if r["success"]], args.n_gif_success)
    fail_f = spread([r for r in ep_rows if not r["success"]], args.n_gif_fail)
    made = []
    for kind, flist in (("success", succ_f), ("fail", fail_f)):
        for fn in flist:
            tr = np.load(os.path.join(args.trace_dir, fn))
            _, _, _, frames = replay_trace(env, u, tr, render=True, renderer=renderer, cams=cams,
                                           frame_every=args.frame_every, max_frames=args.max_frames)
            if not frames:
                print(f"⛔ render 失敗（0 幀）：{fn}")
                continue
            out = os.path.join(args.gif_dir, f"{kind}_{fn.replace('.npz','')}.gif")
            imageio.mimsave(out, frames, duration=0.08, loop=0)
            made.append(out)
            print(f"saved gif: {out}  ({len(frames)} 幀)")
    print(f"\ngif 總數 {len(made)}（success {len(succ_f)} / fail {len(fail_f)}）")

    wv.save_json(dict(
        n_traces=len(files), trace_dir=args.trace_dir, rho=args.rho, fall_line=fall_line,
        chain_success=float(n_succ / len(ep_rows)),
        gate1=dict(
            episode=dict(**{f"reach_N_table_{m:g}N": rate(ep_rows, "first_hit", "N_table", m)[0]
                            for m in (1.25, 1.5)}, n=len(ep_rows)),
            per_leg=dict(**{f"reach_N_table_{m:g}N": rate(legs_ok, "reached_at", "N", m)[0]
                            for m in (1.25, 1.5)},
                         untimed_reach=float(np.mean([r["reached_at"] > 0 for r in legs_ok])),
                         median_steps=float(np.median([r["steps_used"] for r in legs_ok])),
                         n_legs=len(legs_ok), n_open=len(leg_rows) - len(legs_ok))),
        gate2=dict(real=ss, vq_oracle=dict(mean=float(speed.mean()),
                                           quantiles={str(q): float(np.percentile(speed, q))
                                                      for q in (1, 5, 25, 50, 75, 95, 99)})),
        gate3=dict(fall_rate_step=float(np.mean(zall < fall_line)),
                   fall_rate_episode=float(np.mean([np.min(z) < fall_line for z in all_z])),
                   action_diff_real=ad,
                   action_diff_vq=dict(mean=float(adiff.mean()),
                                       quantiles={str(q): float(np.percentile(adiff, q))
                                                  for q in (1, 5, 25, 50, 75, 95, 99)})),
        codes=dict(counts=code_counts.tolist(), active=int((code_counts > 0).sum()),
                   perplexity=float(np.exp(-np.sum(nz * np.log(nz))))),
        episodes=ep_rows, artifacts=dict(pngs=pngs, gifs=made),
    ), os.path.join(args.out_dir, f"{args.tag}_gates.json"))
    print(f"saved: {os.path.join(args.out_dir, f'{args.tag}_gates.json')}")


if __name__ == "__main__":
    main()
