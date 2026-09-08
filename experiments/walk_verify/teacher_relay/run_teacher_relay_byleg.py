#!/usr/bin/env python
"""教材上界測試（teacher relay）—— 逐段（per-leg index）分解版。

⛔ 這是 run_teacher_relay.py 的複製衍生檔，不修改原檔一行。
背景：docs/NOTE-2026-09-08-teacher-relay.md 量出兩版 per-leg 不限時到達率 .554/.555，
但當時沒記「這是第幾段」，所以「累積型失敗（越後面的段越容易失敗：慢＋開環漂移
累積）」這個歸因假設驗不了。本檔在完全相同的任務構造（同 seed=20260908、同 200 條
val episode、同兩版真字播放協定）下，額外記錄每個 leg 的：
  - index（第幾段，m=1..M，M 分布見原 note：min=1 p50=3 max=4）
  - reached（有沒有在該集自己的步數上限內到達）
  - 失敗時「離路標的距離」（d_start＝該段開始時離路標多遠、d_end＝該段結束
    〔可能是到達、也可能是整集步數用完〕時離路標多遠）
  - 該段步速（該段步數窗口內 |dxy| 每步位移的平均）
供之後判讀「差一點沒到 vs 差很遠/繞遠了」「慢到超時 vs 走歪走丟」。

只新增：run_one() 多回傳 leg_start_xy（每段開始那一刻的 xy）；
        measure_version() 多蒐集 leg_rows（逐段明細）＋ by_leg（依 index m 分組彙總）。
其餘（build_tasks / 到達判定 rho / N_data·N_table 定義 / timed_reach 邏輯）
全部原封不動沿用 run_teacher_relay.py 與其 import 的 wv_common / p1_replay /
p2_analyze_traces —— 這樣兩版的「不限時到達率」加總後應該要重現原 note 的
.554／.555，可以拿來自我核對本檔沒有量錯。

⛔ CPU-only；不 render gif、不畫疊圖（這次工單不要求，省時間也少留檔案）。
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

import wv_common as wv  # noqa: E402 - 純函式，重用，未修改
from p1_replay import sim_obs, q, timed_reach  # noqa: E402 - 純函式，重用
from p2_analyze_traces import MAZE_CENTER, RHO_DEFAULT  # noqa: E402 - 常數重用（MAZE_CENTER 本檔未用，保留以示同源）

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
DEFAULT_CKPT = os.path.join(WV_DIR, "results", "p0_dict_v1_L4K32_50k.pt")
DEFAULT_RULER = os.path.join(WV_DIR, "results", "ruler_pack.json")
DELTA_SUB = 7.5
VERSIONS = ("schedule", "nn")
COMPARE = dict(greedy=0.063, main_head=0.320, lo_head=0.124)


def arclength(xy):
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def build_tasks(raw, n_traj, seed, ruler, rho):
    """跟 run_teacher_relay.py 一字不差 —— 必須完全一致才能保證抽到同 200 條題目。"""
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
            t_m = int(np.argmax(al >= m * DELTA_SUB))
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


def run_one(env, u, model, raw, task, version, rho, track_nn_idx=False):
    """跑一條 task 的一次接力。跟原檔同邏輯，多回傳 leg_start_xy（每段開始那一刻的 xy）。"""
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
    leg_start_xy = {1: raw["qpos"][s0, :2].copy()}  # <-- 新增：段開始那一刻的位置
    zs, speeds, acts_flat, xy_hist = [], [], [], []
    codes_used = []
    n_clip, n_elem = 0, 0
    global_step = 0

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
                    leg_start_xy[active_leg] = cur.copy()  # <-- 新增

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
              leg_start_xy=dict(leg_start_xy),
              nn_j_hist=np.asarray(nn_j_hist) if nn_j_hist is not None else None)
    return out


def summarize_by_leg(rows, rho):
    """對一組 leg 明細（同一個 m，或全部合併）算彙總：到達率、限時到達率、失敗形態。"""
    n_att = len(rows)
    if n_att == 0:
        return dict(n_attempted=0)
    n_reach = sum(1 for r in rows if r["reached"])
    fh = np.array([r["first_hit"] for r in rows], dtype=float)
    horizon = np.array([r["horizon"] for r in rows], dtype=float)
    timed = {}
    for nm in ("N_data", "N_table"):
        N = np.array([r[nm] for r in rows], dtype=float)
        reached, ev = timed_reach(fh, N, 1.25, horizon)
        nev = int(ev.sum())
        timed[f"reach_1.25{nm}"] = float(reached[ev].mean()) if nev else None
        timed[f"n_eval_1.25{nm}"] = nev
        timed[f"n_not_eval_1.25{nm}"] = int((~ev).sum())

    fails = [r for r in rows if not r["reached"]]
    succs = [r for r in rows if r["reached"]]
    fail_speed = np.array([r["speed_mean"] for r in fails if np.isfinite(r["speed_mean"])])
    succ_speed = np.array([r["speed_mean"] for r in succs if np.isfinite(r["speed_mean"])])
    fail_dend = np.array([r["d_end"] for r in fails])
    fail_dstart = np.array([r["d_start"] for r in fails])
    fail_prog = np.array([r["progress_frac"] for r in fails if r["progress_frac"] is not None])
    n_close = int((fail_dend <= 2 * rho).sum()) if len(fail_dend) else 0
    n_far = len(fails) - n_close
    n_no_progress = int((fail_dend >= fail_dstart).sum()) if len(fail_dend) else 0

    return dict(
        n_attempted=n_att, n_reached=n_reach, reach_untimed=n_reach / n_att,
        **timed,
        n_fail=len(fails),
        fail_close_2rho=n_close, fail_far_2rho=n_far,
        fail_no_net_progress=n_no_progress,
        fail_dend_q=q(fail_dend) if len(fail_dend) else None,
        fail_dstart_q=q(fail_dstart) if len(fail_dstart) else None,
        fail_progress_frac_q=q(fail_prog) if len(fail_prog) else None,
        fail_speed_mean=float(fail_speed.mean()) if len(fail_speed) else None,
        fail_speed_median=float(np.median(fail_speed)) if len(fail_speed) else None,
        succ_speed_mean=float(succ_speed.mean()) if len(succ_speed) else None,
        succ_speed_median=float(np.median(succ_speed)) if len(succ_speed) else None,
        speed_ratio_fail_over_succ=(float(fail_speed.mean() / succ_speed.mean())
                                    if len(fail_speed) and len(succ_speed) and succ_speed.mean() > 0
                                    else None),
    )


def measure_version(env, u, model, raw, tasks, ruler, version, rho, fall_line):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    per_task, all_leg_rows = [], []
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
            reached_m = rs is not None
            fh = (rs - act_step + 1) if reached_m else -1
            end_step = rs if reached_m else (r["n_steps_run"] - 1)
            start_xy = r["leg_start_xy"][m]
            target_xy = task["wp_xy"][m - 1]
            d_start = float(np.linalg.norm(start_xy - target_xy))
            if end_step >= act_step:
                end_xy = r["xy"][end_step]
                speed_mean = float(r["speeds"][act_step:end_step + 1].mean())
                steps_taken = end_step - act_step + 1
            else:
                # 極端邊界：同一物理步同時到達連續兩個路標（rho 內兩點相鄰）——沒有新步數可算
                end_xy = start_xy
                speed_mean = float("nan")
                steps_taken = 0
            d_end = float(np.linalg.norm(end_xy - target_xy))
            all_leg_rows.append(dict(
                version=version, ti=int(ti), episode=int(task["episode"]),
                m=int(m), M=int(task["M"]), reached=bool(reached_m),
                completed_task=bool(r["completed"]),
                active_step=int(act_step), end_step=int(end_step), steps_taken=int(steps_taken),
                first_hit=float(fh), d_start=d_start, d_end=d_end,
                progress_frac=(float((d_start - d_end) / d_start) if d_start > 1e-9 else None),
                speed_mean=speed_mean,
                N_data=float(task["n_data"][m - 1]), N_table=float(task["n_table"][m - 1]),
                horizon=float(H - act_step)))

    speed = np.concatenate(all_speed)
    zall = np.concatenate(all_z)
    adiff = np.concatenate(all_adiff) if all_adiff else np.array([])
    codes_all = np.concatenate(all_codes)

    legs_attempted_total = int(sum(p["legs_attempted"] for p in per_task))
    legs_reached_total = int(sum(p["legs_reached"] for p in per_task))
    per_leg_untimed = legs_reached_total / max(legs_attempted_total, 1)

    Ms_present = sorted(set(r["m"] for r in all_leg_rows))
    by_leg = {str(m): summarize_by_leg([r for r in all_leg_rows if r["m"] == m], rho)
              for m in Ms_present}
    pooled_leg_stats = summarize_by_leg(all_leg_rows, rho)

    cnt = np.bincount(codes_all, minlength=int(codes_all.max()) + 1 if len(codes_all) else 1)
    fr = cnt / max(cnt.sum(), 1)
    nz = fr[fr > 0]
    fell_any = np.array([bool(np.min(z) < fall_line) for z in all_z])
    n_completed = sum(p["completed"] for p in per_task)

    rec = dict(
        seconds=time.time() - ta, n_tasks=len(tasks),
        completion_ratio=n_completed / len(tasks),
        legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
        per_leg_untimed_reach=per_leg_untimed,
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
        by_leg=by_leg, pooled_leg_stats=pooled_leg_stats,
    )
    return rec, all_leg_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, default="teacher_relay_byleg")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== teacher-relay BYLEG  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} rho={args.rho} DELTA_SUB={DELTA_SUB} ===")

    model, cfg = wv.load_dict_v1(args.ckpt)
    assert cfg["seg_len"] == 4, f"預期 seg_len=4，實際 {cfg['seg_len']}"
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}  ruler fall_line={fall_line:.4f}")

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

    results, all_rows = {}, {}
    for version in VERSIONS:
        rec, rows = measure_version(env, u, model, raw, tasks, ruler, version, args.rho, fall_line)
        results[version] = rec
        all_rows[version] = rows
        print(f"\n[{version:8s}] done in {rec['seconds']:.1f}s  "
              f"completion_ratio={rec['completion_ratio']:.3f}  "
              f"per-leg untimed(全體彙總，核對用，應重現原 note .554/.555)="
              f"{rec['per_leg_untimed_reach']:.3f} "
              f"(reached {rec['legs_reached_total']}/{rec['legs_attempted_total']})")
        if rec.get("nn_backward_jump_frac") is not None:
            print(f"           nn 最近鄰回頭跳字比例 = {rec['nn_backward_jump_frac']*100:.1f}% "
                  f"(n={rec['nn_jump_n']})")

    print("\n=== 逐段（leg index）到達率 —— 不限時 ===")
    print(f"{'leg m':<8}" + "".join(f"{v+' n_att':>14}{v+' reach':>12}" for v in VERSIONS))
    max_m = max(int(m) for v in VERSIONS for m in results[v]["by_leg"])
    for m in range(1, max_m + 1):
        row = f"{m:<8}"
        for v in VERSIONS:
            bl = results[v]["by_leg"].get(str(m))
            if bl is None or bl["n_attempted"] == 0:
                row += f"{'—':>14}{'—':>12}"
            else:
                row += f"{bl['n_attempted']:>14d}{bl['reach_untimed']*100:>11.1f}%"
        print(row)

    print("\n=== 逐段（leg index）限時 1.25N 到達率（N_data；括號＝N_table）===")
    for m in range(1, max_m + 1):
        row = f"leg {m}: "
        for v in VERSIONS:
            bl = results[v]["by_leg"].get(str(m))
            if bl is None or bl["n_attempted"] == 0:
                row += f"{v}=— "
                continue
            d = bl["reach_1.25N_data"]
            t = bl["reach_1.25N_table"]
            ds = f"{d*100:.1f}%" if d is not None else "n/a"
            ts = f"{t*100:.1f}%" if t is not None else "n/a"
            row += f"{v}={ds}(N_table {ts}, n={bl['n_eval_1.25N_data']})  "
        print(row)

    print("\n=== 逐段失敗形態（d_end<=2rho 視為「差一點」，否則「差很遠」）===")
    rho = args.rho
    for m in range(1, max_m + 1):
        for v in VERSIONS:
            bl = results[v]["by_leg"].get(str(m))
            if bl is None or bl["n_fail"] == 0:
                print(f"  m={m} [{v}] 無失敗樣本（n_fail=0）")
                continue
            print(f"  m={m} [{v}] n_fail={bl['n_fail']}  "
                  f"close(<=2rho={2*rho:.2f})={bl['fail_close_2rho']}  "
                  f"far={bl['fail_far_2rho']}  "
                  f"no_net_progress(d_end>=d_start)={bl['fail_no_net_progress']}  "
                  f"speed fail_mean={bl['fail_speed_mean']}  succ_mean={bl['succ_speed_mean']}  "
                  f"ratio={bl['speed_ratio_fail_over_succ']}")

    pooled_all = {v: results[v]["pooled_leg_stats"] for v in VERSIONS}
    combined_rows = all_rows["schedule"] + all_rows["nn"]
    pooled_combined = summarize_by_leg(combined_rows, rho)
    print("\n=== 兩版合併（全部 leg，不分 index）失敗形態彙總 ===")
    print(f"n_fail={pooled_combined['n_fail']}  "
          f"close(<=2rho)={pooled_combined['fail_close_2rho']}  "
          f"far={pooled_combined['fail_far_2rho']}  "
          f"no_net_progress={pooled_combined['fail_no_net_progress']}")
    print(f"fail_dend_q(p25/50/75)="
          f"{pooled_combined['fail_dend_q']['25']:.2f}/"
          f"{pooled_combined['fail_dend_q']['50']:.2f}/"
          f"{pooled_combined['fail_dend_q']['75']:.2f}" if pooled_combined['fail_dend_q'] else "n/a")
    print(f"speed fail_mean={pooled_combined['fail_speed_mean']:.4f}  "
          f"succ_mean={pooled_combined['succ_speed_mean']:.4f}  "
          f"ratio={pooled_combined['speed_ratio_fail_over_succ']:.3f}"
          if pooled_combined['fail_speed_mean'] is not None else "speed: n/a")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(config={k: v for k, v in vars(args).items()}, delta_sub=DELTA_SUB,
                      compare=COMPARE, n_tasks_requested=args.n_traj, n_tasks_used=len(tasks),
                      skipped=skipped,
                      waypoints_per_task=dict(min=int(Ms.min()), p50=float(np.median(Ms)),
                                              max=int(Ms.max()), total=int(Ms.sum())),
                      results=results,
                      pooled_across_versions=pooled_combined,
                      leg_rows={v: all_rows[v] for v in VERSIONS}),
                 js_path)
    print(f"\nsaved: {js_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
