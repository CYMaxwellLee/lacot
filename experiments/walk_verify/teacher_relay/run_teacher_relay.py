#!/usr/bin/env python
"""教材上界測試（teacher relay）—— P2 貪心選字 .063 之後的修正實驗。

問題背景（見 docs/NOTE-2026-09-08-walk-verify.md P1/P2）：字典本身好（P1：照真字串重放
步速貼合真螞蟻 0.1356 vs 0.1325），但 P2「每 chunk 貪心挑最靠近路標的字」把接力戰績
打到 per-leg .063（主頭 .320／lo 頭 .124）。本檔不碰貪心，改成【直接在接力協定下
播真字串（hindsight teacher code）】，量它能不能走到 —— 這是之後 M2 選字頭的教材，
「教材本身能不能走到」是它值不值得拿去訓頭的前提。

⛔ 不修改任何既有檔案；只 import 同目錄上一層 walk_verify 的 wv_common / p1_replay /
   p2_analyze_traces（純函式，import 無副作用，見各檔的 if __name__=='__main__' guard）。
⛔ CPU-only（MuJoCo 物理 + 小 MLP encode/decode，不碰 GPU、不用 slurm）。

題目構造（自己設計 —— 這是資料軌跡上的新任務，不是既有 P1 窗口、也不是 P2 的
planner 接力）：
  對每條抽到的 val episode（固定長度 201 步）：
    - 起點 = episode 開頭的 qpos/qvel（真實 set_state）。
    - 沿 episode 真實 xy 路徑算累積弧長，每 DELTA_SUB=7.5 單位取一個路標
      （M = floor(總弧長/7.5)，實測 500 集裡 498 集弧長 >= 7.5、2 集太短跳過並點名）。
    - chunk 格點固定在 episode 起點對齊、每 4 步一個 chunk（跟 P0 訓練切法一致）。
    - 模擬步數上限 = 這條 episode 自己的 chunk 數（n_chunks=T//4，通常 50）
      —— 用真實資料自己的長度當公平上限（「有沒有跟上教材自己的步調」），
      一旦全部路標依序到達就提早結束。

兩版「真字」定義（教材播放方式，關鍵設計）：
  (i) schedule：第 k 個 chunk 一律用【這條 episode 自己】第 k 段的 hindsight 真字
      （encoder 對 act[s+4k:s+4k+4] 算出的 code）—— 跟模擬中的 ant 實際在哪裡無關，
      純開環照表播，會被 float32 混沌咬（見 walk-verify §2.2）。
  (ii) nn：每個 chunk 先看模擬中 ant 現在的 xy，在同一條 episode 的 chunk 起點格點
       （0,4,8,...）裡找最近的一點，用那一點的 hindsight 真字 —— 允許漂移後重新
       校準要播哪個字，是「教材上界」該有的語義（貼著參考路徑走，不是照錶宣讀）。
       ⚠️ 搜尋不限制方向（不強制單調前進），忠實照字面「找最近點」；若觀察到回頭
       跳字，那是量到的現象，不預先工程掉。
  兩版的解碼/執行方式相同：decoder 的 obs 條件輸入用「模擬中 ant 的當下 obs」
  （對應 P1 的 dict_cont 連續執行風格 —— 這裡本來就是連續多 chunk 的接力，不是
  P1 dict_indep 那種每 chunk reset 回真起點的獨立重放）。

到達判定（同接力協定）：rho=1.875（=0.25*DELTA_SUB，跟 P2 的 lacot/subgoal.py 同值，
見 p2_analyze_traces.py 的 RHO_DEFAULT）。leg 依序：leg m 要等 leg m-1 到達才「開始」，
沒被輪到的 leg 不算「試過」（跟官方 per-leg 統計一樣，只有被輪到的才進分母）。
限時 1.25N/1.5N 兩把尺都印：N_data（這條真實 episode 自己從上一個路標走到這個路標
用的步數）與 N_table（量尺包 distance→steps p50，距離＝waypoint 間直線距離 - rho，
跟 p2_analyze_traces.py 算 per-leg N_table 的方式一致）。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.dirname(HERE)  # experiments/walk_verify
sys.path.insert(0, WV_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q, timed_reach  # noqa: E402 - 純函式，重用
from p2_analyze_traces import MAZE_CENTER, RHO_DEFAULT  # noqa: E402 - 常數重用

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
DEFAULT_CKPT = os.path.join(WV_DIR, "results", "p0_dict_v1_L4K32_50k.pt")
DEFAULT_RULER = os.path.join(WV_DIR, "results", "ruler_pack.json")
DELTA_SUB = 7.5     # 同現行 DELTA_SUB 尺度（P2 report §3.4：路標目標間距 DELTA_SUB 7.5）
VERSIONS = ("schedule", "nn")
# 對照行（照抄 NOTE-2026-09-08-walk-verify.md §3.4，一手數字，非本次量測）
COMPARE = dict(greedy=0.063, main_head=0.320, lo_head=0.124)


def arclength(xy):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def build_tasks(raw, n_traj, seed, ruler, rho):
    """抽 >=n_traj 條可用 episode，算每條的路標序列 + N_data/N_table。回傳 (tasks, skipped)。"""
    starts, ends = wv.episode_bounds(raw["terminals"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(starts))
    tasks, skipped = [], []
    for ei in order:
        s0, e0 = int(starts[ei]), int(ends[ei])
        T = e0 - s0 + 1
        xy = raw["qpos"][s0:e0 + 1, :2].copy()
        al = arclength(xy)
        total_L = float(al[-1])
        M = int(total_L // DELTA_SUB)
        if M < 1:
            skipped.append(dict(episode=int(ei), arclen=total_L, reason="arclen<DELTA_SUB"))
            continue
        n_chunks = T // 4
        wp_local_idx, wp_xy, n_data = [], [], []
        prev_t = 0
        for m in range(1, M + 1):
            t_m = int(np.argmax(al >= m * DELTA_SUB))  # 第一個 >= 門檻的 local index
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
            nt, tg = wv.ruler_steps_for_distance(ruler, max(d_leg - rho, 1e-6), "50")
            n_table.append(nt)
            tags.append(tg)
            prev_xy = wp_xy[m]
        tasks.append(dict(episode=int(ei), s0=s0, e0=e0, T=T, n_chunks=n_chunks,
                          M=M, total_L=total_L, wp_xy=wp_xy, wp_local_idx=np.asarray(wp_local_idx),
                          n_data=n_data, n_table=np.asarray(n_table), n_table_tag=tags))
        if len(tasks) >= n_traj:
            break
    return tasks, skipped


def run_one(env, u, model, raw, task, version, rho, render=False, renderer=None,
           cams=None, frame_every=4, max_frames=200, track_nn_idx=False):
    """跑一條 task 的一次接力（schedule 或 nn）。回傳量測用的逐步資料 + leg 結果。"""
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act = raw["actions"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    grid_local = np.arange(n_chunks) * 4
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
            s_idx = s0 + 4 * k
        else:
            cur_xy = np.asarray(u.data.qpos)[:2]
            d2 = np.sum((grid_xy - cur_xy[None, :]) ** 2, axis=1)
            j_star = int(np.argmin(d2))
            s_idx = s0 + grid_local[j_star]
            if track_nn_idx:
                nn_j_hist.append(j_star)

        with torch.no_grad():
            x = torch.from_numpy(act[s_idx:s_idx + 4].reshape(1, -1).astype(np.float32))
            idx = model.encode_idx(x)
            o_in = torch.from_numpy(sim_obs(u)[None, :])
            raw_a = model.decode_from_idx(idx, o_in).numpy().reshape(4, wv.ACT_DIM)
        codes_used.append(int(idx.item()))
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
        n_clip += int(np.sum(raw_a != a_chunk.astype(np.float32)))
        n_elem += a_chunk.size

        for t in range(4):
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
    out = dict(completed=completed, legs_attempted=min(active_leg, M),
              legs_reached=active_leg - 1, n_steps_run=global_step,
              zs=np.asarray(zs), speeds=np.asarray(speeds), acts=np.asarray(acts_flat),
              xy=np.asarray(xy_hist), codes_used=np.asarray(codes_used, dtype=np.int64),
              clip_frac=float(n_clip / max(n_elem, 1)),
              leg_reach_step=dict(leg_reach_step), leg_active_step=dict(leg_active_step),
              frames=frames if render else None,
              nn_j_hist=np.asarray(nn_j_hist) if nn_j_hist is not None else None)
    return out


def measure_version(env, u, model, raw, tasks, ruler, version, rho, fall_line):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    n_jump_back, n_jump_total = 0, 0
    for ti, task in enumerate(tasks):
        r = run_one(env, u, model, raw, task, version, rho, track_nn_idx=(version == "nn"))
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
        H = task["n_chunks"] * 4
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
    codes_all = np.concatenate(all_codes)
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
            reached, ev = timed_reach(fh, N_arr, mult, horizon_arr)
            nev = int(ev.sum())
            timed[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
            timed[f"n_evaluable_{nm}_{mult:g}N"] = nev
            timed[f"n_not_evaluable_{nm}_{mult:g}N"] = int((~ev).sum())

    cnt = np.bincount(codes_all, minlength=int(codes_all.max()) + 1 if len(codes_all) else 1)
    fr = cnt / max(cnt.sum(), 1)
    nz = fr[fr > 0]
    fell_any = np.array([bool(np.min(z) < fall_line) for z in all_z])
    n_completed = sum(p["completed"] for p in per_task)

    rec = dict(
        seconds=time.time() - ta, n_tasks=len(tasks),
        completion_ratio=n_completed / len(tasks),
        legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
        per_leg_untimed_reach=per_leg_untimed, timed_reach=timed,
        step_speed_q=q(speed), step_speed_mean=float(speed.mean()),
        fall_rate_step=float((zall < fall_line).mean()), fall_rate_task=float(fell_any.mean()),
        action_diff_q=q(adiff) if len(adiff) else None,
        action_diff_mean=float(adiff.mean()) if len(adiff) else None,
        codes_active=int((cnt > 0).sum()),
        codes_perplexity=float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
        codes_top1_frac=float(fr.max()) if fr.size else 0.0,
        per_task=per_task,
        nn_backward_jump_frac=(n_jump_back / n_jump_total) if n_jump_total else None,
        nn_jump_n=n_jump_total,
    )
    raw_arrays = dict(speed=speed, z=zall, adiff=adiff)
    return rec, raw_arrays


def render_exemplars(env, u, model, raw, tasks, results, rho, gif_dir, frame_every=4, max_frames=200):
    """對每版各挑 1 條 completed=True、1 條 completed=False，重跑一次（render=True）存 gif。"""
    import mujoco
    env.reset()
    u.set_state(raw["qpos"][tasks[0]["s0"]].copy(), raw["qvel"][tasks[0]["s0"]].copy())
    _ = u.render()
    renderer = u.custom_renderer
    cam = mujoco.MjvCamera()
    cam.lookat[2] = 0.3
    cam.distance = 6.0
    cam.elevation = -35
    cam.azimuth = 135
    cam_wide = mujoco.MjvCamera()
    cam_wide.lookat[0], cam_wide.lookat[1] = MAZE_CENTER
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
                print(f"⛔ {version}/{kind}：200 條裡沒有這種結果，無法產生這支 gif")
                continue
            task = tasks[rec["ti"]]
            r = render_run = run_one(env, u, model, raw, task, version, rho, render=True,
                                     renderer=renderer, cams=cams, frame_every=frame_every,
                                     max_frames=max_frames)
            assert r["completed"] == rec["completed"] and r["n_steps_run"] == rec["n_steps_run"], (
                f"⛔ 重跑不一致（應為 bit-exact 決定性）：version={version} kind={kind} "
                f"first_pass=(completed={rec['completed']},steps={rec['n_steps_run']}) "
                f"second_pass=(completed={r['completed']},steps={r['n_steps_run']})")
            fp = os.path.join(gif_dir, f"{version}_{kind}_ep{task['episode']}.gif")
            if r["frames"]:
                import imageio.v2 as imageio
                imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
                made[f"{version}_{kind}"] = dict(path=fp, n_frames=len(r["frames"]),
                                                 episode=task["episode"],
                                                 legs_reached=r["legs_reached"], M=task["M"],
                                                 n_steps_run=r["n_steps_run"])
                print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                      f"legs {r['legs_reached']}/{task['M']})")
            else:
                made[f"{version}_{kind}"] = None
                print(f"⛔ render 失敗（0 幀）：{version}/{kind} episode={task['episode']}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--gif-dir", type=str, default=os.path.join(HERE, "results", "gifs"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, default="teacher_relay")
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-gifs", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== teacher-relay  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg = wv.load_dict_v1(args.ckpt)
    assert cfg["seg_len"] == 4, f"預期 seg_len=4，實際 {cfg['seg_len']}"
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}  ruler fall_line={fall_line:.4f} "
          f"goal_tol={wv.GOAL_TOL}(未用於 leg 判定，leg 一律用 rho)")

    raw = wv.load_npz(args.data_dir, DATASET, "val")
    tasks, skipped = build_tasks(raw, args.n_traj, args.seed, ruler, args.rho)
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}")
    for s in skipped:
        print(f"  ⛔ skip episode={s['episode']} arclen={s['arclen']:.3f} reason={s['reason']}")
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報（不用預設值填）"
    Ms = np.array([t["M"] for t in tasks])
    print(f"waypoints per task: min={Ms.min()} p50={np.median(Ms):.0f} max={Ms.max()} "
          f"total_legs_available={Ms.sum()}")

    env = wv.make_env(args.data_dir, DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    results, raw_arrays = {}, {}
    for version in VERSIONS:
        rec, ra = measure_version(env, u, model, raw, tasks, ruler, version, args.rho, fall_line)
        results[version] = rec
        raw_arrays[version] = ra
        print(f"\n[{version:8s}] done in {rec['seconds']:.1f}s  "
              f"completion_ratio={rec['completion_ratio']:.3f}  "
              f"per-leg untimed={rec['per_leg_untimed_reach']:.3f} "
              f"(reached {rec['legs_reached_total']}/{rec['legs_attempted_total']})  "
              f"speed_p50={rec['step_speed_q']['50']:.4f}  fall(step)={rec['fall_rate_step']*100:.2f}%  "
              f"adiff_p50={(rec['action_diff_q']['50'] if rec['action_diff_q'] else float('nan')):.3f}")
        if rec.get("nn_backward_jump_frac") is not None:
            print(f"           nn 最近鄰回頭跳字比例 = {rec['nn_backward_jump_frac']*100:.1f}% "
                  f"(n={rec['nn_jump_n']} 次 chunk-to-chunk 轉換)")

    print("\n=== per-leg 限時到達率 ===")
    for version in VERSIONS:
        tr = results[version]["timed_reach"]
        print(f"[{version}]")
        for nm in ("N_data", "N_table"):
            r125 = tr[f"reach_{nm}_1.25N"]
            r150 = tr[f"reach_{nm}_1.5N"]
            print(f"  N={nm:8s} 1.25N={r125*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.25N']})  "
                  f"1.5N={r150*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.5N']})  "
                  f"not_evaluable={tr[f'n_not_evaluable_{nm}_1.5N']}")

    print("\n=== 對照行（一手引用自 NOTE-2026-09-08-walk-verify.md §3.4）===")
    print(f"貪心(greedy)={COMPARE['greedy']:.3f}  主頭(main_head)={COMPARE['main_head']:.3f}  "
          f"lo頭(lo_head)={COMPARE['lo_head']:.3f}")
    for v in VERSIONS:
        print(f"teacher_relay[{v}] per-leg 不限時到達率 = {results[v]['per_leg_untimed_reach']:.3f}  "
              f"完整走完比例 = {results[v]['completion_ratio']:.3f}")

    # ---- 疊圖：真螞蟻 vs schedule vs nn ----
    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "teacher-relay gate2: step speed vs real ant", "|dxy| per step", None),
        ("z", "torso_z", "teacher-relay gate3a: torso height vs real ant", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "teacher-relay gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL ant (ruler)", values=ruler_np[ref])]
        for v in VERSIONS:
            arr = raw_arrays[v][key]
            if len(arr):
                series.append(dict(label=f"teacher_relay[{v}]", values=arr))
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        wv.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")

    # ---- gif ----
    gifs = {}
    if not args.skip_gifs:
        gifs = render_exemplars(env, u, model, raw, tasks, results, args.rho, args.gif_dir,
                                args.frame_every, args.max_frames)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    for v in VERSIONS:
        results[v].pop("per_task_full", None)
    wv.save_json(dict(config={k: v for k, v in vars(args).items()}, delta_sub=DELTA_SUB,
                      compare=COMPARE, n_tasks_requested=args.n_traj, n_tasks_used=len(tasks),
                      skipped=skipped,
                      waypoints_per_task=dict(min=int(Ms.min()), p50=float(np.median(Ms)),
                                              max=int(Ms.max()), total=int(Ms.sum())),
                      results=results, artifacts=dict(pngs=pngs, gifs=gifs)), js_path)
    print(f"saved: {js_path}")

    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        **{f"{v}_speed": raw_arrays[v]["speed"] for v in VERSIONS},
                        **{f"{v}_z": raw_arrays[v]["z"] for v in VERSIONS},
                        **{f"{v}_adiff": raw_arrays[v]["adiff"] for v in VERSIONS})
    print(f"saved: {os.path.join(args.out_dir, f'{args.tag}_raw.npz')}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
