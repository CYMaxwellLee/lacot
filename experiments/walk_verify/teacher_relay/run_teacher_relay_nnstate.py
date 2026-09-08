#!/usr/bin/env python
"""教材接力—— nn_state 變體（姿態感知選字，2026-09-08 追加驗證）。

背景（見 docs/NOTE-2026-09-08-teacher-relay.md）：既有 `nn`（閉環最近鄰）版只用
xy 位置找同一條 episode 上「離模擬中 ant 現在最近的參考點」，播那一點的 hindsight
真字。實測 nn 跟純開環 schedule 幾乎打平（.554 vs .555），沒有像預期那樣明顯更好。
主人裁示：換一個假說——「壞的不是找最近點這件事，是找最近點『只看位置』」，
如果參考點位置像但姿態（軀幹朝向/8 個關節角）不像，decoder 收到跟當下姿態不相容
的字，反而幫倒忙。本檔驗證：把最近鄰的距離度量從「xy 歐氏」換成「完整 normalize
state 的歐氏距離」，看 per-leg 到達率能不能救回來。

⛔ 不修改任何既有檔案；只 import 同目錄的 run_teacher_relay.py（重用 build_tasks／
   COMPARE，未修改該檔一行）與上一層 wv_common / p1_replay / p2_analyze_traces
   （純函式，import 無副作用）。
⛔ CPU-only。

兩個子版（設計決策，本檔自己拍板，理由如下）：
  nn_state_full ：距離 = 全 29 維 normalize state 的歐氏距離（含 root xy）。
  nn_state_pose ：距離 = 排除 root xy（obs 索引 0,1）之後剩下 27 維的歐氏距離——
                  qpos[2]=軀幹高度、qpos[3:7]=軀幹朝向 quaternion、qpos[7:15]=8個
                  關節角、qvel 全部 14 維（皆為速度，不編碼「在迷宮哪裡」）。
                  這才是「姿態」該有的定義：拿掉的只有「全域位置」，其餘（含高度／
                  朝向／速度）照樣算進去，因為它們共同決定「這個參考點的動作字
                  適不適合套用到當下身體姿態」。
  兩版都做的理由：原工單的疑慮「位置像但姿態不像」在 (a) 全 state 版裡其實還是有
  可能被 root xy 的大 scale 主導（xy 的原始跨度整個迷宮 ~24 個單位，正規化後跟其他
  維度同量級但仍是搜尋目標之一），所以另外做 (b) 把 xy 完全排除、逼搜尋只看姿態，
  才是最直接對症的版本；兩版一起跑才看得出「排除位置」這個動作本身有沒有差異。

normalize 用的 mean/std：直接重用字典 ckpt 裡已有的 `model.obs_mean` / `obs_std`
（buffer，訓練時對 train split 的 29 維 obs 算出、decoder 自己也用這份正規化），
不另外對 val split 重算——這是「該環境資料的 mean/std」最直接可得的來源，也讓
nn_state 搜尋用的正規化空間跟 decoder 內部 `norm_obs()` 完全一致（見
`wv_common.CondVQVAE.decode_from_idx`）。已用 obs_mean[:2]=(9.43,10.24) 跟
`p2_analyze_traces.MAZE_CENTER`=(9.42,10.24) 核對一致，確認索引 0,1 就是 root xy。

參考點的 state 向量直接取資料集的 `raw["observations"][s0+grid_local]`（不用重新
呼叫 sim_obs 在歷史時刻算一次——這兩者定義相同，wv_common.load_npz 已驗證
observations 逐位元對應 concat(qpos[:15],qvel[:14])，也是 wv_common.build_seg_tensors
訓練時取 decoder 條件輸入的同一個慣例）。
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
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q, timed_reach  # noqa: E402 - 純函式，重用
from p2_analyze_traces import MAZE_CENTER, RHO_DEFAULT  # noqa: E402 - 常數重用
import run_teacher_relay as rtr  # noqa: E402 - 重用 build_tasks / COMPARE，未改該檔

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = rtr.DATASET
DEFAULT_CKPT = rtr.DEFAULT_CKPT
DEFAULT_RULER = rtr.DEFAULT_RULER
DELTA_SUB = rtr.DELTA_SUB
build_tasks = rtr.build_tasks  # 未修改 run_teacher_relay.py，直接重用同一支函式

VERSIONS = ("schedule", "nn", "nn_state_full", "nn_state_pose")
NN_LIKE = ("nn", "nn_state_full", "nn_state_pose")
STATE_VERSIONS = ("nn_state_full", "nn_state_pose")
# root xy 在 29 維 obs 裡的索引（obs = concat(qpos[:15], qvel[:14])，qpos[:2]=xy）
POS_IDX = (0, 1)
# 對照行：既有量測值（一手引用，非本次量測）
COMPARE = dict(rtr.COMPARE)                 # greedy / main_head / lo_head
HIST_SCHEDULE = 0.554                       # NOTE-2026-09-08-teacher-relay.md §二
HIST_NN = 0.555
HIST_NN_BACKJUMP = 0.060                    # 同上，§二④


def pose_mask(dim):
    m = np.ones(dim, dtype=bool)
    for i in POS_IDX:
        m[i] = False
    return m


def run_one(env, u, model, raw, task, version, rho, obs_mean=None, obs_std=None,
           render=False, renderer=None, cams=None, frame_every=4, max_frames=200,
           track_nn_idx=False):
    """跑一條 task 的一次接力。version 多兩種：nn_state_full / nn_state_pose。

    跟 run_teacher_relay.run_one 逐行同款，只有「怎麼選 s_idx」那段（nn 系列）
    多了 state-based 分支；schedule／nn 的邏輯完全沒動，維持可比性。
    """
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act = raw["actions"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    grid_local = np.arange(n_chunks) * 4
    grid_xy = raw["qpos"][s0 + grid_local, :2]

    grid_obs_n_full = grid_obs_n_pose = None
    if version in STATE_VERSIONS:
        grid_obs = raw["observations"][s0 + grid_local].astype(np.float64)
        grid_obs_n_full = (grid_obs - obs_mean) / obs_std
        grid_obs_n_pose = grid_obs_n_full[:, pose_mask(grid_obs_n_full.shape[1])]

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
        elif version == "nn":
            cur_xy = np.asarray(u.data.qpos)[:2]
            d2 = np.sum((grid_xy - cur_xy[None, :]) ** 2, axis=1)
            j_star = int(np.argmin(d2))
            s_idx = s0 + grid_local[j_star]
            if track_nn_idx:
                nn_j_hist.append(j_star)
        elif version in STATE_VERSIONS:
            cur_obs = sim_obs(u).astype(np.float64)
            cur_obs_n = (cur_obs - obs_mean) / obs_std
            if version == "nn_state_pose":
                cur_n = cur_obs_n[pose_mask(len(cur_obs_n))]
                ref_n = grid_obs_n_pose
            else:
                cur_n = cur_obs_n
                ref_n = grid_obs_n_full
            d2 = np.sum((ref_n - cur_n[None, :]) ** 2, axis=1)
            j_star = int(np.argmin(d2))
            s_idx = s0 + grid_local[j_star]
            if track_nn_idx:
                nn_j_hist.append(j_star)
        else:
            raise ValueError(f"⛔ 未知 version={version}")

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


def measure_version(env, u, model, raw, tasks, ruler, version, rho, fall_line,
                    obs_mean=None, obs_std=None):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    n_jump_back, n_jump_total = 0, 0
    for ti, task in enumerate(tasks):
        r = run_one(env, u, model, raw, task, version, rho, obs_mean=obs_mean, obs_std=obs_std,
                   track_nn_idx=(version in NN_LIKE))
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


def render_exemplars(env, u, model, raw, tasks, results, rho, gif_dir, versions,
                     obs_mean=None, obs_std=None, frame_every=4, max_frames=200):
    """對指定 versions 各挑 1 條 completed=True、1 條 completed=False 重跑存 gif。"""
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
    for version in versions:
        per_task = results[version]["per_task"]
        succ = next((p for p in per_task if p["completed"]), None)
        fail = next((p for p in per_task if not p["completed"]), None)
        for kind, rec in (("success", succ), ("fail", fail)):
            if rec is None:
                made[f"{version}_{kind}"] = None
                print(f"⛔ {version}/{kind}：這批題目裡沒有這種結果，無法產生這支 gif")
                continue
            task = tasks[rec["ti"]]
            r = run_one(env, u, model, raw, task, version, rho, obs_mean=obs_mean,
                       obs_std=obs_std, render=True, renderer=renderer, cams=cams,
                       frame_every=frame_every, max_frames=max_frames)
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
    ap.add_argument("--tag", type=str, default="teacher_relay_nnstate")
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-gifs", action="store_true")
    ap.add_argument("--gif-versions", type=str, default="nn_state_pose",
                    help="逗號分隔，只對這些 version 出 gif（總預算：每個環境 2 支＝1 版本）")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== teacher-relay[nn_state]  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} rho={args.rho} DELTA_SUB={DELTA_SUB} versions={VERSIONS} ===")

    model, cfg = wv.load_dict_v1(args.ckpt)
    assert cfg["seg_len"] == 4, f"預期 seg_len=4，實際 {cfg['seg_len']}"
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}  ruler fall_line={fall_line:.4f}")

    obs_mean = model.obs_mean.detach().cpu().numpy().astype(np.float64)
    obs_std = np.maximum(model.obs_std.detach().cpu().numpy().astype(np.float64), 1e-6)
    assert obs_mean.shape[0] == wv.OBS_DIM, f"⛔ obs_mean 維度={obs_mean.shape[0]} 預期 {wv.OBS_DIM}"
    print(f"obs_mean[:2]={obs_mean[:2]} (核對 MAZE_CENTER={MAZE_CENTER})  "
          f"pose_mask 排除索引={POS_IDX}（root xy），保留 {wv.OBS_DIM - len(POS_IDX)}/{wv.OBS_DIM} 維")

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
        rec, ra = measure_version(env, u, model, raw, tasks, ruler, version, args.rho, fall_line,
                                  obs_mean=obs_mean, obs_std=obs_std)
        results[version] = rec
        raw_arrays[version] = ra
        print(f"\n[{version:14s}] done in {rec['seconds']:.1f}s  "
              f"completion_ratio={rec['completion_ratio']:.3f}  "
              f"per-leg untimed={rec['per_leg_untimed_reach']:.3f} "
              f"(reached {rec['legs_reached_total']}/{rec['legs_attempted_total']})  "
              f"speed_p50={rec['step_speed_q']['50']:.4f}  fall(step)={rec['fall_rate_step']*100:.2f}%  "
              f"fall(task)={rec['fall_rate_task']*100:.2f}%  "
              f"adiff_p50={(rec['action_diff_q']['50'] if rec['action_diff_q'] else float('nan')):.3f}")
        if rec.get("nn_backward_jump_frac") is not None:
            print(f"                nn 回頭跳字比例 = {rec['nn_backward_jump_frac']*100:.1f}% "
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

    print("\n=== 對照行（一手引用：既有 COMPARE 常數 ＋ NOTE-2026-09-08-teacher-relay.md §二）===")
    print(f"貪心(greedy)={COMPARE['greedy']:.3f}  主頭(main_head)={COMPARE['main_head']:.3f}  "
          f"lo頭(lo_head)={COMPARE['lo_head']:.3f}")
    print(f"歷史 schedule={HIST_SCHEDULE:.3f}  歷史 nn={HIST_NN:.3f}  "
          f"歷史 nn回頭跳字={HIST_NN_BACKJUMP*100:.1f}%")
    for v in VERSIONS:
        print(f"teacher_relay[{v}] per-leg 不限時到達率 = {results[v]['per_leg_untimed_reach']:.3f}  "
              f"完整走完比例 = {results[v]['completion_ratio']:.3f}")
    d_sched = results["schedule"]["per_leg_untimed_reach"] - HIST_SCHEDULE
    d_nn = results["nn"]["per_leg_untimed_reach"] - HIST_NN
    print(f"重現性檢查（應接近 0，流程全確定性）：schedule 差={d_sched:+.4f}  nn 差={d_nn:+.4f}")

    # ---- 疊圖：真螞蟻 vs 四版 ----
    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "teacher-relay[nn_state] gate2: step speed vs real ant",
         "|dxy| per step", None),
        ("z", "torso_z", "teacher-relay[nn_state] gate3a: torso height vs real ant",
         "qpos[2] (m)", [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "teacher-relay[nn_state] gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL ant (ruler)", values=ruler_np[ref])]
        for v in VERSIONS:
            arr = raw_arrays[v][key]
            if len(arr):
                series.append(dict(label=f"[{v}]", values=arr))
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        wv.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")

    # ---- gif：只對 --gif-versions 指定的版本出（總預算：本環境 2 支）----
    gifs = {}
    if not args.skip_gifs:
        gif_versions = tuple(v.strip() for v in args.gif_versions.split(",") if v.strip())
        for v in gif_versions:
            assert v in VERSIONS, f"⛔ --gif-versions 裡的 {v} 不在 {VERSIONS}"
        gifs = render_exemplars(env, u, model, raw, tasks, results, args.rho, args.gif_dir,
                                gif_versions, obs_mean=obs_mean, obs_std=obs_std,
                                frame_every=args.frame_every, max_frames=args.max_frames)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    for v in VERSIONS:
        results[v].pop("per_task_full", None)
    wv.save_json(dict(config={k: v for k, v in vars(args).items()}, delta_sub=DELTA_SUB,
                      pos_idx=list(POS_IDX), obs_mean=obs_mean, obs_std=obs_std,
                      compare=COMPARE, hist_schedule=HIST_SCHEDULE, hist_nn=HIST_NN,
                      hist_nn_backjump=HIST_NN_BACKJUMP,
                      n_tasks_requested=args.n_traj, n_tasks_used=len(tasks), skipped=skipped,
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
