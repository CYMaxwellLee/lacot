#!/usr/bin/env python
"""教材接力（block 2，「照譜跳」）—— humanoidmaze-medium-stitch-v0。

協定「形狀」逐項照抄 experiments/walk_verify/teacher_relay/run_teacher_relay.py（ant 版，
見 docs/NOTE-2026-09-08-teacher-relay.md）：200 條官方 held-out val episode 各自在真實
xy 路徑上、每 DELTA_SUB 弧長取一個路標，兩版「hindsight 真字」播放方式：

  schedule（純開環）：第 k 個 chunk 一律用這條 episode 自己第 k 段的真字（encoder 對
                      act[s+4k:s+4k+4] 算出的 code），跟模擬中人形現在在哪裡無關。
  nn（閉環修正）     ：每個 chunk 先看模擬中人形現在的 xy，在同一條 episode 的 chunk
                      起點格點裡找歐氏距離最近的一點，用那一點的真字。

兩版的解碼／執行都用「模擬中人形的當下 obs」條件化 decoder（連續多 chunk 接力，不是
segment-replay 那種每 chunk 重置回真起點）。到達判定：rho（=0.25*DELTA_SUB）、leg 依序、
限時 1.25N/1.5N 兩把尺（N_data／N_table）都印。

DELTA_SUB／rho 數值與「要不要重新校」的理由見 hr_common.py 檔頭長註解（結論：跟 ant
完全同值 7.5／1.875，因為 humanoidmaze-medium 跟 antmaze-medium 是同一個迷宮座標尺度，
maze_unit 都是預設值 4.0，humanoid 較慢是時間尺度不是空間尺度）。

⛔ CPU-only。⛔ 不訓練，只載入既有 ckpt。
"""
import argparse
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
import hd_common as hc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = ("schedule", "nn")
# 對照行（一手引用自 docs/NOTE-2026-09-08-teacher-relay.md §二，ant 版數字，非本次量測）
COMPARE_ANT = dict(schedule=0.554, nn=0.555, greedy=0.063, main_head=0.320, lo_head=0.124)


def build_tasks(raw, n_traj, seed, ruler, rho, delta_sub, seg_len=4):
    starts, ends = hr.episode_bounds(raw["terminals"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(starts))
    tasks, skipped = [], []
    for ei in order:
        s0, e0 = int(starts[ei]), int(ends[ei])
        T = e0 - s0 + 1
        xy = raw["qpos"][s0:e0 + 1, :2].copy()
        al = hr.arclength(xy)
        total_L = float(al[-1])
        M = int(total_L // delta_sub)
        if M < 1:
            skipped.append(dict(episode=int(ei), arclen=total_L, reason="arclen<DELTA_SUB"))
            continue
        n_chunks = T // seg_len
        wp_local_idx, wp_xy, n_data = [], [], []
        prev_t = 0
        for m in range(1, M + 1):
            t_m = int(np.argmax(al >= m * delta_sub))
            wp_local_idx.append(t_m)
            wp_xy.append(xy[t_m].copy())
            n_data.append(t_m - prev_t)
            prev_t = t_m
        wp_xy = np.asarray(wp_xy)
        n_data = np.asarray(n_data, dtype=float)
        n_table, tags = [], []
        prev_xy = xy[0]
        for m in range(M):
            d_leg = float(np.linalg.norm(wp_xy[m] - prev_xy))
            nt, tg = hr.ruler_steps_for_distance(ruler, max(d_leg - rho, 1e-6), "50")
            n_table.append(nt)
            tags.append(tg)
            prev_xy = wp_xy[m]
        tasks.append(dict(episode=int(ei), s0=s0, e0=e0, T=T, n_chunks=n_chunks,
                          M=M, total_L=total_L, wp_xy=wp_xy, wp_local_idx=np.asarray(wp_local_idx),
                          n_data=n_data, n_table=np.asarray(n_table), n_table_tag=tags))
        if len(tasks) >= n_traj:
            break
    return tasks, skipped


def run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho, seg_len=4,
           render=False, renderer=None, cams=None, frame_every=4, max_frames=200,
           track_nn_idx=False):
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act = raw["actions"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    grid_local = np.arange(n_chunks) * seg_len
    grid_xy = raw["qpos"][s0 + grid_local, :2]

    env.reset()
    u.set_state(raw["qpos"][s0].copy(), raw["qvel"][s0].copy())

    active_leg = 1
    leg_reach_step, leg_active_step = {}, {1: 0}
    zs, speeds, acts_flat, xy_hist = [], [], [], []
    codes_used = []
    n_clip, n_elem = 0, 0
    global_step = 0
    frames = []
    if render:
        u.set_goal(goal_xy=np.asarray(wp_xy[0], dtype=np.float64))

    nn_j_hist = [] if track_nn_idx else None
    prev_xy = raw["qpos"][s0, :2].copy()
    for k in range(n_chunks):
        if version == "schedule":
            s_idx = s0 + seg_len * k
        else:
            cur_xy = np.asarray(u.data.qpos)[:2]
            d2 = np.sum((grid_xy - cur_xy[None, :]) ** 2, axis=1)
            j_star = int(np.argmin(d2))
            s_idx = s0 + grid_local[j_star]
            if track_nn_idx:
                nn_j_hist.append(j_star)

        with torch.no_grad():
            x = torch.from_numpy(act[s_idx:s_idx + seg_len].reshape(1, -1).astype(np.float32))
            idx = hr.encode_idx(model, x)  # (1,G)
            o_in = torch.from_numpy(hr.sim_obs(u)[None, :])
            raw_a = hr.decode_from_idx(model, idx, o_in, obs_mu, obs_sd).numpy().reshape(
                seg_len, hc.ACT_DIM)
        codes_used.append(idx.numpy().reshape(-1).copy())
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
        n_clip += int(np.sum(raw_a != a_chunk.astype(np.float32)))
        n_elem += a_chunk.size

        for t in range(seg_len):
            env.step(a_chunk[t])
            cur = np.asarray(u.data.qpos)[:2].copy()
            zs.append(float(np.asarray(u.data.qpos)[2]))
            speeds.append(float(np.linalg.norm(cur - prev_xy)))
            acts_flat.append(a_chunk[t].copy())
            xy_hist.append(cur.copy())
            prev_xy = cur

            while active_leg <= M and np.linalg.norm(cur - wp_xy[active_leg - 1]) <= rho:
                leg_reach_step[active_leg] = global_step
                active_leg += 1
                if active_leg <= M:
                    leg_active_step[active_leg] = global_step + 1
                if render:
                    tgt = wp_xy[active_leg - 1] if active_leg <= M else wp_xy[-1]
                    u.set_goal(goal_xy=np.asarray(tgt, dtype=np.float64))

            if render and (global_step % frame_every == 0) and len(frames) < max_frames:
                follow, wide = cams
                follow.lookat[0], follow.lookat[1] = float(cur[0]), float(cur[1])
                renderer.update_scene(u.data, camera=follow)
                fa = renderer.render().copy()
                renderer.update_scene(u.data, camera=wide)
                fb = renderer.render().copy()
                frames.append(np.concatenate([fa, fb], axis=1))

            global_step += 1
        if active_leg > M:
            break

    completed = active_leg > M
    codes_arr = np.stack(codes_used) if codes_used else np.zeros((0, G), dtype=np.int64)
    out = dict(completed=completed, legs_attempted=min(active_leg, M),
              legs_reached=active_leg - 1, n_steps_run=global_step,
              zs=np.asarray(zs), speeds=np.asarray(speeds), acts=np.asarray(acts_flat),
              xy=np.asarray(xy_hist), codes_used=codes_arr,
              clip_frac=float(n_clip / max(n_elem, 1)),
              leg_reach_step=dict(leg_reach_step), leg_active_step=dict(leg_active_step),
              frames=frames if render else None,
              nn_j_hist=np.asarray(nn_j_hist) if nn_j_hist is not None else None)
    return out


def measure_version(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, ruler, version, rho, fall_line,
                    seg_len=4):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    n_jump_back, n_jump_total = 0, 0
    for ti, task in enumerate(tasks):
        r = run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho, seg_len=seg_len,
                   track_nn_idx=(version == "nn"))
        if r["nn_j_hist"] is not None and len(r["nn_j_hist"]) > 1:
            d = np.diff(r["nn_j_hist"])
            n_jump_back += int((d < 0).sum())
            n_jump_total += len(d)
        per_task.append(dict(ti=ti, episode=task["episode"], completed=r["completed"],
                             legs_attempted=r["legs_attempted"], legs_reached=r["legs_reached"],
                             n_steps_run=r["n_steps_run"]))
        all_speed.append(r["speeds"])
        all_z.append(r["zs"])
        if len(r["acts"]) > 1:
            all_adiff.append(np.linalg.norm(np.diff(r["acts"], axis=0), axis=1))
        all_codes.append(r["codes_used"])
        H = task["n_chunks"] * seg_len
        for m in range(1, r["legs_attempted"] + 1):
            act_step = r["leg_active_step"][m]
            rs = r["leg_reach_step"].get(m, None)
            fh = (rs - act_step + 1) if rs is not None else -1
            first_hit_data.append(fh)
            N_data_arr.append(task["n_data"][m - 1])
            first_hit_table.append(fh)
            N_table_arr.append(task["n_table"][m - 1])
            horizon_arr.append(H - act_step)

    speed = np.concatenate(all_speed)
    zall = np.concatenate(all_z)
    adiff = np.concatenate(all_adiff) if all_adiff else np.array([])
    codes_all = np.concatenate(all_codes, axis=0) if all_codes and len(all_codes[0]) else np.zeros((0, G))
    first_hit_data = np.asarray(first_hit_data, dtype=float)
    first_hit_table = np.asarray(first_hit_table, dtype=float)
    N_data_arr = np.asarray(N_data_arr, dtype=float)
    N_table_arr = np.asarray(N_table_arr, dtype=float)
    horizon_arr = np.asarray(horizon_arr, dtype=float)

    legs_attempted_total = int(sum(p["legs_attempted"] for p in per_task))
    legs_reached_total = int(sum(p["legs_reached"] for p in per_task))
    per_leg_untimed = legs_reached_total / max(legs_attempted_total, 1)

    timed = {}
    for nm, N_arr, fh in (("N_data", N_data_arr, first_hit_data),
                          ("N_table", N_table_arr, first_hit_table)):
        for mult in (1.25, 1.5):
            reached, ev = hr.timed_reach(fh, N_arr, mult, horizon_arr)
            nev = int(ev.sum())
            timed[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
            timed[f"n_evaluable_{nm}_{mult:g}N"] = nev
            timed[f"n_not_evaluable_{nm}_{mult:g}N"] = int((~ev).sum())

    per_group_stats = []
    if codes_all.shape[0] > 0:
        for g in range(G):
            cnt = np.bincount(codes_all[:, g].astype(np.int64), minlength=K)
            fr = cnt / max(cnt.sum(), 1)
            nz = fr[fr > 0]
            per_group_stats.append(dict(active=int((cnt > 0).sum()),
                                        perplexity=float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
                                        top1_frac=float(fr.max()) if fr.size else 0.0))
    fell_any = np.array([bool(np.min(z) < fall_line) for z in all_z])
    n_completed = sum(p["completed"] for p in per_task)

    rec = dict(
        seconds=time.time() - ta, n_tasks=len(tasks),
        completion_ratio=n_completed / len(tasks),
        legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
        per_leg_untimed_reach=per_leg_untimed, timed_reach=timed,
        step_speed_q=hr.q(speed), step_speed_mean=float(speed.mean()),
        fall_rate_step=float((zall < fall_line).mean()), fall_rate_task=float(fell_any.mean()),
        action_diff_q=hr.q(adiff) if len(adiff) else None,
        action_diff_mean=float(adiff.mean()) if len(adiff) else None,
        codes_per_group=per_group_stats,
        codes_active_min_over_groups=(int(min(p["active"] for p in per_group_stats))
                                      if per_group_stats else None),
        codes_perplexity_mean_over_groups=(float(np.mean([p["perplexity"] for p in per_group_stats]))
                                           if per_group_stats else None),
        per_task=per_task,
        nn_backward_jump_frac=(n_jump_back / n_jump_total) if n_jump_total else None,
        nn_jump_n=n_jump_total,
    )
    raw_arrays = dict(speed=speed, z=zall, adiff=adiff)
    return rec, raw_arrays


def render_exemplars(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, results, rho, gif_dir,
                     seg_len=4, frame_every=4, max_frames=200):
    import mujoco
    env.reset()
    u.set_state(raw["qpos"][tasks[0]["s0"]].copy(), raw["qvel"][tasks[0]["s0"]].copy())
    _ = u.render()
    renderer = u.custom_renderer
    # 跟拍鏡頭：humanoid 站得比 ant 高（軀幹 z 中位數 ~1.1 vs ant ~0.55），
    # lookat/distance 依 experiments/humanoid_dict/hd_common.py 既有 render 函式的取景
    # 經驗值調整（該檔已跟主人核過視覺化，看到的畫面跟資料事實吻合）。
    cam = mujoco.MjvCamera()
    cam.lookat[2] = 0.9
    cam.distance = 5.0
    cam.elevation = -20
    cam.azimuth = 135
    cam_wide = mujoco.MjvCamera()
    cam_wide.lookat[0], cam_wide.lookat[1] = hr.MAZE_CENTER
    cam_wide.lookat[2] = 0.0
    cam_wide.distance = 38.0
    cam_wide.elevation = -75
    cam_wide.azimuth = 90
    cams = (cam, cam_wide)

    made = {}
    for version in VERSIONS:
        per_task = results[version]["per_task"]
        succ = next((p for p in per_task if p["completed"]), None)
        fail = next((p for p in per_task if not p["completed"]), None)
        for kind, rec in (("success", succ), ("fail", fail)):
            if rec is None:
                made[f"{version}_{kind}"] = None
                print(f"⛔ {version}/{kind}：這批題目裡沒有這種結果，無法產生這支 gif", flush=True)
                continue
            task = tasks[rec["ti"]]
            r = run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho, seg_len=seg_len,
                       render=True, renderer=renderer, cams=cams, frame_every=frame_every,
                       max_frames=max_frames)
            assert r["completed"] == rec["completed"] and r["n_steps_run"] == rec["n_steps_run"], (
                f"⛔ 重跑不一致（應為 bit-exact 決定性）：version={version} kind={kind}")
            fp = os.path.join(gif_dir, f"{version}_{kind}_ep{task['episode']}.gif")
            if r["frames"]:
                import imageio.v2 as imageio
                imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
                made[f"{version}_{kind}"] = dict(path=fp, n_frames=len(r["frames"]),
                                                 episode=task["episode"],
                                                 legs_reached=r["legs_reached"], M=task["M"],
                                                 n_steps_run=r["n_steps_run"])
                print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                      f"legs {r['legs_reached']}/{task['M']})", flush=True)
            else:
                made[f"{version}_{kind}"] = None
                print(f"⛔ render 失敗（0 幀）：{version}/{kind} episode={task['episode']}", flush=True)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ruler", type=str, default=hr.RULER_DEFAULT)
    ap.add_argument("--data-dir", type=str, default=hr.DD_DEFAULT)
    ap.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--gif-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "gifs"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=hr.RHO_DEFAULT)
    ap.add_argument("--delta-sub", type=float, default=hr.DELTA_SUB)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-gifs", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== humanoid teacher-relay  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} rho={args.rho} DELTA_SUB={args.delta_sub} ===", flush=True)

    model, cfg, obs_mu, obs_sd = hr.load_dict(args.ckpt)
    G, K, seg_len = cfg["G"], cfg["K"], cfg["seg_len"]
    assert seg_len == 4, f"預期 seg_len=4，實際 {seg_len}"
    ruler = hr.load_ruler(args.ruler)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: G={G} K={K} seg_len={seg_len} ruler fall_line={fall_line:.4f} "
          f"goal_tol={hc.GOAL_TOL}(未用於 leg 判定，leg 一律用 rho)", flush=True)

    raw = hc.load_raw(args.data_dir, hc.DATASET_NAME, "val")
    tasks, skipped = build_tasks(raw, args.n_traj, args.seed, ruler, args.rho, args.delta_sub, seg_len)
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}", flush=True)
    for s in skipped:
        print(f"  ⛔ skip episode={s['episode']} arclen={s['arclen']:.3f} reason={s['reason']}", flush=True)
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報（不用預設值填）"
    Ms = np.array([t["M"] for t in tasks])
    print(f"waypoints per task: min={Ms.min()} p50={np.median(Ms):.0f} max={Ms.max()} "
          f"total_legs_available={Ms.sum()}", flush=True)

    env = hr.make_env(args.data_dir)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    results, raw_arrays = {}, {}
    for version in VERSIONS:
        rec, ra = measure_version(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, ruler, version,
                                  args.rho, fall_line, seg_len=seg_len)
        results[version] = rec
        raw_arrays[version] = ra
        print(f"\n[{version:8s}] done in {rec['seconds']:.1f}s  "
              f"completion_ratio={rec['completion_ratio']:.3f}  "
              f"per-leg untimed={rec['per_leg_untimed_reach']:.3f} "
              f"(reached {rec['legs_reached_total']}/{rec['legs_attempted_total']})  "
              f"speed_p50={rec['step_speed_q']['50']:.4f}  fall(step)={rec['fall_rate_step']*100:.2f}%  "
              f"adiff_p50={(rec['action_diff_q']['50'] if rec['action_diff_q'] else float('nan')):.3f}",
              flush=True)
        if rec.get("nn_backward_jump_frac") is not None:
            print(f"           nn 最近鄰回頭跳字比例 = {rec['nn_backward_jump_frac']*100:.1f}% "
                  f"(n={rec['nn_jump_n']} 次 chunk-to-chunk 轉換)", flush=True)

    print("\n=== per-leg 限時到達率 ===", flush=True)
    for version in VERSIONS:
        tr = results[version]["timed_reach"]
        print(f"[{version}]", flush=True)
        for nm in ("N_data", "N_table"):
            r125 = tr[f"reach_{nm}_1.25N"]
            r150 = tr[f"reach_{nm}_1.5N"]
            print(f"  N={nm:8s} 1.25N={r125*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.25N']})  "
                  f"1.5N={r150*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.5N']})  "
                  f"not_evaluable={tr[f'n_not_evaluable_{nm}_1.5N']}", flush=True)

    print("\n=== 對照行（一手引用自 NOTE-2026-09-08-teacher-relay.md，ant 版）===", flush=True)
    print(f"ant: schedule={COMPARE_ANT['schedule']:.3f}  nn={COMPARE_ANT['nn']:.3f}  "
          f"greedy={COMPARE_ANT['greedy']:.3f}  main_head={COMPARE_ANT['main_head']:.3f}  "
          f"lo_head={COMPARE_ANT['lo_head']:.3f}", flush=True)
    for v in VERSIONS:
        print(f"humanoid teacher_relay[{v}] per-leg 不限時到達率 = {results[v]['per_leg_untimed_reach']:.3f}  "
              f"完整走完比例 = {results[v]['completion_ratio']:.3f}", flush=True)

    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "teacher-relay(humanoid) gate2: step speed vs real humanoid",
         "|dxy| per step", None),
        ("z", "torso_z", "teacher-relay(humanoid) gate3a: torso height vs real humanoid",
         "qpos[2] (m)", [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "teacher-relay(humanoid) gate3b: action smoothness vs real humanoid",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL humanoid (ruler)", values=ruler_np[ref])]
        for v in VERSIONS:
            arr = raw_arrays[v][key]
            if len(arr):
                series.append(dict(label=f"teacher_relay[{v}]", values=arr))
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        hc.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}", flush=True)

    gifs = {}
    if not args.skip_gifs:
        gifs = render_exemplars(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, results, args.rho,
                                args.gif_dir, seg_len=seg_len, frame_every=args.frame_every,
                                max_frames=args.max_frames)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif", flush=True)

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    hr.save_json(dict(config={k: v for k, v in vars(args).items()}, delta_sub=args.delta_sub,
                      dict_cfg=cfg, compare_ant=COMPARE_ANT, n_tasks_requested=args.n_traj,
                      n_tasks_used=len(tasks), skipped=skipped,
                      waypoints_per_task=dict(min=int(Ms.min()), p50=float(np.median(Ms)),
                                              max=int(Ms.max()), total=int(Ms.sum())),
                      results=results, artifacts=dict(pngs=pngs, gifs=gifs)), js_path)
    print(f"saved: {js_path}", flush=True)

    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        **{f"{v}_speed": raw_arrays[v]["speed"] for v in VERSIONS},
                        **{f"{v}_z": raw_arrays[v]["z"] for v in VERSIONS},
                        **{f"{v}_adiff": raw_arrays[v]["adiff"] for v in VERSIONS})
    print(f"saved: {os.path.join(args.out_dir, f'{args.tag}_raw.npz')}", flush=True)
    print(f"=== done wall={time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
