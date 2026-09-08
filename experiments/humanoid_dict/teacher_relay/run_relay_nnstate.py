#!/usr/bin/env python
"""humanoid 教材接力—— nn_state 變體（姿態感知選字，2026-09-08 追加驗證）。

背景（docs/NOTE-2026-09-08-humanoid-relay.md §四）：既有 `nn`（閉環最近鄰，只看
xy）在 humanoid 上反而比純開環 `schedule` 更差（G1K32: .048<.122；G16K16:
.029<.052）——跟 ant 的「schedule≈nn」方向不同。該筆記的推論（未證實）是：只用 xy
位置找參考點，不管當下人形的姿態/朝向/相位是否吻合，對需要連續相位才能維持平衡的
雙足步態干擾更大。本檔驗證：把最近鄰距離換成「完整 normalize state」，看能不能把
這個劣化救回來、甚至讓 nn 系列贏過 schedule。

協定「形狀」照抄 experiments/walk_verify/teacher_relay/run_teacher_relay_nnstate.py
（ant 版，同一份工單、同一個假說），換成 humanoidmaze 的 hr_common / hd_common。

⛔ 不修改任何既有檔案；只 import 同目錄的 run_relay.py（重用 build_tasks／
   COMPARE_ANT，未修改該檔一行）與 hr_common.py / hd_common.py（純函式）。
⛔ CPU-only。⛔ 不訓練，只載入既有 ckpt。

兩個子版（設計與 ant 版同一套理由，見 run_teacher_relay_nnstate.py 檔頭）：
  nn_state_full ：距離 = 全 69 維 normalize state 的歐氏距離（含 root xy）。
  nn_state_pose ：距離 = 排除 root xy（obs 索引 0,1）之後剩下 67 維——
                  21 維關節角、頭高、四肢相對軀幹位置、軀幹朝向、質心速度、
                  27 維 qvel，皆不編碼「在迷宮哪裡」，只編碼姿態與運動狀態。
                  （obs 組成順序：xy(2)+joint_angles(21)+head_height(1)+
                  extremities(12)+torso_vert(3)+com_vel(3)+qvel(27)=69，一手核對
                  ogbench/locomaze/humanoid.py:get_ob，見下方查證註解。）

normalize 用 `hr.load_dict` 回傳的 `obs_mu`/`obs_sd`（ckpt 內建，訓練時對 train
split 的 69 維 obs 算出、decoder 自己也用這份正規化）。已核對 mu[:2]=(9.61,10.26)
跟 `hr_common.MAZE_CENTER`=(10.00,10.01)（val split 量出）同量級，確認索引 0,1
就是 root xy（跟 ant 索引位置相同，obs 定義都是「xy 開頭」）。

參考點 state 向量取 `raw["observations"][s0+grid_local]`——hr_common.sim_obs() 的
docstring 已寫明這逐位元對應資料集 observations 欄位，不用重新呼叫模擬取值。
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import hr_common as hr  # noqa: E402
import hd_common as hc  # noqa: E402
import run_relay as rr  # noqa: E402 - 重用 build_tasks / COMPARE_ANT，未改該檔

VERSIONS = ("schedule", "nn", "nn_state_full", "nn_state_pose")
NN_LIKE = ("nn", "nn_state_full", "nn_state_pose")
STATE_VERSIONS = ("nn_state_full", "nn_state_pose")
# root xy 在 69 維 obs 裡的索引（obs = concat(xy(2), joint_angles(21), head_height(1),
# extremities(12), torso_vert(3), com_vel(3), qvel(27))；xy 是最前 2 維）
POS_IDX = (0, 1)
build_tasks = rr.build_tasks  # 未修改 run_relay.py，直接重用同一支函式
# 對照行（一手引用，非本次量測）：ant 版 + humanoid 既有 schedule/nn 兩顆字典數字
COMPARE_ANT = dict(rr.COMPARE_ANT)
HIST_HUMANOID = dict(
    G16K16=dict(schedule=0.0524, nn=0.0293, nn_backjump=0.030),
    G1K32=dict(schedule=0.1216, nn=0.0481, nn_backjump=0.031),
)


def pose_mask(dim):
    m = np.ones(dim, dtype=bool)
    for i in POS_IDX:
        m[i] = False
    return m


def run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho, seg_len=4,
           obs_mean_state=None, obs_std_state=None, render=False, renderer=None,
           cams=None, frame_every=4, max_frames=200, track_nn_idx=False):
    """跟 run_relay.run_one 逐行同款，只有「怎麼選 s_idx」多了 state-based 分支。

    注意：obs_mu/obs_sd（decoder 正規化用，torch tensor）跟 obs_mean_state/
    obs_std_state（nn_state 距離度量用，numpy array）是同一組數字的兩份拷貝——
    分開命名是因為型別與用途不同（一個要喂進 torch decode，一個要拿來算 numpy
    距離），數值上是同一份 ckpt 內建統計，不是重算了兩次。
    """
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act = raw["actions"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    grid_local = np.arange(n_chunks) * seg_len
    grid_xy = raw["qpos"][s0 + grid_local, :2]

    grid_obs_n_full = grid_obs_n_pose = None
    if version in STATE_VERSIONS:
        grid_obs = raw["observations"][s0 + grid_local].astype(np.float64)
        grid_obs_n_full = (grid_obs - obs_mean_state) / obs_std_state
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
            s_idx = s0 + seg_len * k
        elif version == "nn":
            cur_xy = np.asarray(u.data.qpos)[:2]
            d2 = np.sum((grid_xy - cur_xy[None, :]) ** 2, axis=1)
            j_star = int(np.argmin(d2))
            s_idx = s0 + grid_local[j_star]
            if track_nn_idx:
                nn_j_hist.append(j_star)
        elif version in STATE_VERSIONS:
            cur_obs = hr.sim_obs(u).astype(np.float64)
            cur_obs_n = (cur_obs - obs_mean_state) / obs_std_state
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


def measure_version(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, ruler, version, rho,
                    fall_line, seg_len=4, obs_mean_state=None, obs_std_state=None):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    n_jump_back, n_jump_total = 0, 0
    for ti, task in enumerate(tasks):
        r = run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho, seg_len=seg_len,
                   obs_mean_state=obs_mean_state, obs_std_state=obs_std_state,
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
                     versions, seg_len=4, obs_mean_state=None, obs_std_state=None,
                     frame_every=4, max_frames=200):
    import mujoco
    env.reset()
    u.set_state(raw["qpos"][tasks[0]["s0"]].copy(), raw["qvel"][tasks[0]["s0"]].copy())
    _ = u.render()
    renderer = u.custom_renderer
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
    for version in versions:
        per_task = results[version]["per_task"]
        succ = next((p for p in per_task if p["completed"]), None)
        fail = next((p for p in per_task if not p["completed"]), None)
        for kind, rec in (("success", succ), ("fail", fail)):
            if rec is None:
                made[f"{version}_{kind}"] = None
                print(f"⛔ {version}/{kind}：這批題目裡沒有這種結果，無法產生這支 gif", flush=True)
                continue
            task = tasks[rec["ti"]]
            r = run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, rho,
                       seg_len=seg_len, obs_mean_state=obs_mean_state, obs_std_state=obs_std_state,
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
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--gif-dir", type=str, default=os.path.join(HERE, "results", "gifs"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=hr.RHO_DEFAULT)
    ap.add_argument("--delta-sub", type=float, default=hr.DELTA_SUB)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--hist-key", type=str, required=True, choices=list(HIST_HUMANOID.keys()))
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-gifs", action="store_true")
    ap.add_argument("--gif-versions", type=str, default="",
                    help="逗號分隔；預設空＝這次不出 gif（gif 只在 G16K16 那次呼叫時開）")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== humanoid teacher-relay[nn_state]  ckpt={os.path.basename(args.ckpt)} "
          f"n_traj={args.n_traj} seed={args.seed} rho={args.rho} DELTA_SUB={args.delta_sub} "
          f"versions={VERSIONS} ===", flush=True)

    model, cfg, obs_mu, obs_sd = hr.load_dict(args.ckpt)
    G, K, seg_len = cfg["G"], cfg["K"], cfg["seg_len"]
    assert seg_len == 4, f"預期 seg_len=4，實際 {seg_len}"
    ruler = hr.load_ruler(args.ruler)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: G={G} K={K} seg_len={seg_len} ruler fall_line={fall_line:.4f}", flush=True)

    obs_mean_state = obs_mu.detach().cpu().numpy().reshape(-1).astype(np.float64)
    obs_std_state = np.maximum(obs_sd.detach().cpu().numpy().reshape(-1).astype(np.float64), 1e-6)
    assert obs_mean_state.shape[0] == hc.OBS_DIM, (
        f"⛔ obs_mu 維度={obs_mean_state.shape[0]} 預期 {hc.OBS_DIM}")
    print(f"obs_mean_state[:2]={obs_mean_state[:2]} (核對 hr.MAZE_CENTER={hr.MAZE_CENTER})  "
          f"pose_mask 排除索引={POS_IDX}（root xy），保留 {hc.OBS_DIM - len(POS_IDX)}/{hc.OBS_DIM} 維",
          flush=True)

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
                                  args.rho, fall_line, seg_len=seg_len,
                                  obs_mean_state=obs_mean_state, obs_std_state=obs_std_state)
        results[version] = rec
        raw_arrays[version] = ra
        print(f"\n[{version:14s}] done in {rec['seconds']:.1f}s  "
              f"completion_ratio={rec['completion_ratio']:.3f}  "
              f"per-leg untimed={rec['per_leg_untimed_reach']:.3f} "
              f"(reached {rec['legs_reached_total']}/{rec['legs_attempted_total']})  "
              f"speed_p50={rec['step_speed_q']['50']:.4f}  fall(step)={rec['fall_rate_step']*100:.2f}%  "
              f"fall(task)={rec['fall_rate_task']*100:.2f}%  "
              f"adiff_p50={(rec['action_diff_q']['50'] if rec['action_diff_q'] else float('nan')):.3f}",
              flush=True)
        if rec.get("nn_backward_jump_frac") is not None:
            print(f"                nn 回頭跳字比例 = {rec['nn_backward_jump_frac']*100:.1f}% "
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

    hist = HIST_HUMANOID[args.hist_key]
    print("\n=== 對照行（一手引用：ant 版 COMPARE_ANT ＋ NOTE-2026-09-08-humanoid-relay.md §四）===",
          flush=True)
    print(f"ant: schedule={COMPARE_ANT['schedule']:.3f}  nn={COMPARE_ANT['nn']:.3f}  "
          f"greedy={COMPARE_ANT['greedy']:.3f}", flush=True)
    print(f"歷史[{args.hist_key}]: schedule={hist['schedule']:.4f}  nn={hist['nn']:.4f}  "
          f"nn回頭跳字={hist['nn_backjump']*100:.1f}%", flush=True)
    for v in VERSIONS:
        print(f"humanoid[{args.hist_key}] teacher_relay[{v}] per-leg 不限時到達率 = "
              f"{results[v]['per_leg_untimed_reach']:.4f}  完整走完比例 = "
              f"{results[v]['completion_ratio']:.3f}", flush=True)
    d_sched = results["schedule"]["per_leg_untimed_reach"] - hist["schedule"]
    d_nn = results["nn"]["per_leg_untimed_reach"] - hist["nn"]
    print(f"重現性檢查（應接近 0，流程全確定性）：schedule 差={d_sched:+.5f}  nn 差={d_nn:+.5f}",
          flush=True)

    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "teacher-relay[nn_state](humanoid) gate2: step speed vs real humanoid",
         "|dxy| per step", None),
        ("z", "torso_z", "teacher-relay[nn_state](humanoid) gate3a: torso height vs real humanoid",
         "qpos[2] (m)", [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "teacher-relay[nn_state](humanoid) gate3b: action smoothness vs real humanoid",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL humanoid (ruler)", values=ruler_np[ref])]
        for v in VERSIONS:
            arr = raw_arrays[v][key]
            if len(arr):
                series.append(dict(label=f"[{v}]", values=arr))
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        hc.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}", flush=True)

    gifs = {}
    gif_versions = tuple(v.strip() for v in args.gif_versions.split(",") if v.strip())
    if not args.skip_gifs and gif_versions:
        for v in gif_versions:
            assert v in VERSIONS, f"⛔ --gif-versions 裡的 {v} 不在 {VERSIONS}"
        gifs = render_exemplars(env, u, model, obs_mu, obs_sd, G, K, raw, tasks, results, args.rho,
                                args.gif_dir, gif_versions, seg_len=seg_len,
                                obs_mean_state=obs_mean_state, obs_std_state=obs_std_state,
                                frame_every=args.frame_every, max_frames=args.max_frames)
    else:
        print("⛔ 本次沒有產生 gif（--skip-gifs 或 --gif-versions 空）", flush=True)

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    hr.save_json(dict(config={k: v for k, v in vars(args).items()}, delta_sub=args.delta_sub,
                      pos_idx=list(POS_IDX), obs_mean_state=obs_mean_state, obs_std_state=obs_std_state,
                      dict_cfg=cfg, compare_ant=COMPARE_ANT, hist_humanoid=hist,
                      n_tasks_requested=args.n_traj, n_tasks_used=len(tasks), skipped=skipped,
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
