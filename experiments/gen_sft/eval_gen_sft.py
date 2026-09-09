#!/usr/bin/env python
"""驗收準則③④⑤：eval u —— 200 條接力題目（同 teacher_relay 構造）換成 u 生成的字串，
其餘（schedule 開環播放、decoder 執行、per-leg 判定、模擬步數上限=該集 n_chunks、
四關量測）全部沿用 run_teacher_relay.py。量兩種：貪心 1 條、BoN(N) 採樣上界。
另跑隨機字串對照（20 題），且【先跑、先過閘門】才准繼續（design 第 4 點）。

⛔ CPU-only、CPU-heavy（mujoco 物理 + 上千次 rollout），走 sbatch，用
   multiprocessing 在 --cpus-per-task 個 worker 間並行 rollout（生成字串本身用小
   transformer、很快，單行程做；只有「照字串走一遍」的物理模拟平行化)。
⛔ 不改動任何既有檔案；run_one_generated()/aggregate_runs() 是仿 run_teacher_relay.py
   的 run_one()/measure_version() 改寫（code 來源換成預先生成好的陣列），不是 import
   它們本身（那兩支內部綁死 schedule/nn 兩種 hindsight code 來源，沒辦法不改就套用
   生成字串），但 leg 判定/rho/decode 呼叫方式逐行對齊，屬「同構」而非「同一份程式」。
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402
from model import GenSFT, BOS_ID, EOS_ID  # noqa: E402

CONTROL_N = 20
CONTROL_SEED = 99
BON_SEED = 20260909
GATE_MAX_PER_LEG = 0.30   # design 第 4 點：隨機字串 per-leg >= .30 表示管線壞了，停手
MAX_NEW_TOKENS = 60       # 訓練語料 L 恆為 50，60 留安全邊界

# 對照行（一手引用；greedy/main_head/lo_head 抄自 run_teacher_relay.py 的 COMPARE，
# teacher_relay 兩版數字抄自 docs/NOTE-2026-09-08-teacher-relay.md §二，皆非本次量測）
ANCHORS = dict(
    greedy=0.063, main_head=0.320, lo_head=0.124,
    teacher_relay_schedule=0.554, teacher_relay_nn=0.555,
)
# 驗收準則字面門檻：per-leg >= .30 算「學會寫譜」（.320 只是括號裡的參照——已訓練
# 主頭的量級，不是門檻本身，這裡故意分開兩個常數避免自己讀錯）；>= .45 算貼近教材
# 上界（教材本身 = .554，單次 eval 抖動尺 ±.03~.07）。
LEARNED_GAIT_THRESHOLD = 0.30
MAIN_HEAD_REFERENCE_LEVEL = 0.320   # 僅供對照印出，不是門檻
NEAR_TEACHER_THRESHOLD = 0.45
TEACHER_UPPER = 0.554


def tasks_to_cond(tasks, raw):
    s0_obs = np.stack([raw["observations"][t["s0"]] for t in tasks]).astype(np.float32)
    Mmax = max(t["M"] for t in tasks)
    wp_xy = np.zeros((len(tasks), Mmax, 2), dtype=np.float32)
    wp_mask = np.zeros((len(tasks), Mmax), dtype=bool)
    for i, t in enumerate(tasks):
        M = t["M"]
        wp_xy[i, :M] = t["wp_xy"]
        wp_mask[i, :M] = True
    return torch.from_numpy(s0_obs), torch.from_numpy(wp_xy), torch.from_numpy(wp_mask)


def run_one_generated(env, u, dict_model, raw, task, codes, rho, render=False, renderer=None,
                      cams=None, frame_every=4, max_frames=200):
    """跑一條 task、照給定的 codes 陣列開環執行一遍。逐行對齊
    run_teacher_relay.run_one()：leg 判定／rho／decode_from_idx／clip 統計完全同款，
    唯一差異＝每個 chunk 要播哪個字直接查 codes[k]（不是 hindsight encoder 現算）。

    設計決定（design 沒明講、這裡明訂並在 NOTE 標註)：若生成字串比 n_chunks 短
    （EOS 提早出現），k>=len(codes) 時直接停止模擬（不補字、不重複播最後一字）——
    語意上等於「模型自己判斷字串已經講完」，跟『照譜開環播放』一致：沒有更多譜可照。
    若生成字串沒有更短（沒觸發 EOS 就撞到 max_new_tokens），多出來的部份本來就不會
    被用到，因為模擬步數上限固定是 n_chunks（design 第 3 點明訂）。
    """
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)

    env.reset()
    u.set_state(raw["qpos"][s0].copy(), raw["qvel"][s0].copy())

    active_leg = 1
    leg_reach_step, leg_active_step = {}, {1: 0}
    zs, speeds, acts_flat, xy_hist = [], [], [], []
    codes_used = []
    n_clip, n_elem = 0, 0
    global_step = 0
    frames = []
    n_codes = len(codes)
    ran_out = False
    if render:
        u.set_goal(goal_xy=np.asarray(wp_xy[0], dtype=np.float64))

    prev_xy = raw["qpos"][s0, :2].copy()
    for k in range(n_chunks):
        if k >= n_codes:
            ran_out = True
            break
        code_id = int(codes[k])
        with torch.no_grad():
            idx = torch.tensor([code_id], dtype=torch.long)
            o_in = torch.from_numpy(gc.sim_obs(u)[None, :])
            raw_a = dict_model.decode_from_idx(idx, o_in).numpy().reshape(4, gc.wv.ACT_DIM)
        codes_used.append(code_id)
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
    return dict(completed=completed, legs_attempted=min(active_leg, M),
               legs_reached=active_leg - 1, n_steps_run=global_step,
               zs=np.asarray(zs), speeds=np.asarray(speeds), acts=np.asarray(acts_flat),
               xy=np.asarray(xy_hist), codes_used=np.asarray(codes_used, dtype=np.int64),
               clip_frac=float(n_clip / max(n_elem, 1)),
               leg_reach_step=dict(leg_reach_step), leg_active_step=dict(leg_active_step),
               frames=frames if render else None,
               n_codes_available=int(n_codes), ran_out_of_codes=bool(ran_out))


def aggregate_runs(tasks, runs, fall_line):
    """逐行對齊 run_teacher_relay.measure_version() 的聚合邏輯（拿掉 nn 專用的
    backward-jump 診斷，加上 ran_out_of_codes_frac）。"""
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    for ti, (task, r) in enumerate(zip(tasks, runs)):
        per_task.append(dict(ti=ti, episode=task["episode"], completed=r["completed"],
                             legs_attempted=r["legs_attempted"], legs_reached=r["legs_reached"],
                             n_steps_run=r["n_steps_run"], ran_out_of_codes=r["ran_out_of_codes"],
                             n_codes_available=r["n_codes_available"]))
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

    speed = np.concatenate(all_speed) if all_speed else np.array([])
    zall = np.concatenate(all_z) if all_z else np.array([])
    adiff = np.concatenate(all_adiff) if all_adiff else np.array([])
    codes_all = (np.concatenate(all_codes) if all_codes and sum(len(c) for c in all_codes)
                else np.array([], dtype=np.int64))
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
            reached, ev = gc.timed_reach(fh, N_arr, mult, horizon_arr)
            nev = int(ev.sum())
            timed[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
            timed[f"n_evaluable_{nm}_{mult:g}N"] = nev
            timed[f"n_not_evaluable_{nm}_{mult:g}N"] = int((~ev).sum())

    if len(codes_all):
        cnt = np.bincount(codes_all, minlength=int(codes_all.max()) + 1)
        fr = cnt / max(cnt.sum(), 1)
        nz = fr[fr > 0]
        codes_active = int((cnt > 0).sum())
        codes_perplexity = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0
        codes_top1_frac = float(fr.max()) if fr.size else 0.0
    else:
        codes_active, codes_perplexity, codes_top1_frac = 0, 0.0, 0.0
    fell_any = np.array([bool(np.min(z) < fall_line) if len(z) else False for z in all_z])
    n_completed = sum(p["completed"] for p in per_task)
    ran_out_frac = float(np.mean([p["ran_out_of_codes"] for p in per_task])) if per_task else None

    rec = dict(
        n_tasks=len(tasks), completion_ratio=n_completed / max(len(tasks), 1),
        legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
        per_leg_untimed_reach=per_leg_untimed, timed_reach=timed,
        step_speed_q=gc.q(speed) if len(speed) else None,
        step_speed_mean=float(speed.mean()) if len(speed) else None,
        fall_rate_step=float((zall < fall_line).mean()) if len(zall) else None,
        fall_rate_task=float(fell_any.mean()),
        action_diff_q=gc.q(adiff) if len(adiff) else None,
        action_diff_mean=float(adiff.mean()) if len(adiff) else None,
        codes_active=codes_active, codes_perplexity=codes_perplexity, codes_top1_frac=codes_top1_frac,
        per_task=per_task, ran_out_of_codes_frac=ran_out_frac,
    )
    raw_arrays = dict(speed=speed, z=zall, adiff=adiff)
    return rec, raw_arrays


# ---------------------------------------------------------------------
# multiprocessing worker：每個 worker 各自建一次 env/u/dict_model/raw，重複使用
# ---------------------------------------------------------------------
_W = {}


def _worker_init(dict_ckpt, data_dir, dataset):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.set_num_threads(1)
    import gen_sft_common as gc2
    model, _ = gc2.load_dict_model(dict_ckpt)
    raw = gc2.wv.load_npz(data_dir, dataset, "val")
    env = gc2.wv.make_env(data_dir, dataset)
    _W["model"] = model
    _W["raw"] = raw
    _W["env"] = env
    _W["u"] = env.unwrapped


def _worker_run(item):
    task, codes, rho = item["task"], item["codes"], item["rho"]
    r = run_one_generated(_W["env"], _W["u"], _W["model"], _W["raw"], task, codes, rho)
    r.pop("frames", None)
    return dict(group=item["group"], i=item["i"], j=item.get("j"), r=r)


def parallel_walk(work_items, dict_ckpt, data_dir, dataset, n_workers):
    t0 = time.time()
    if n_workers <= 1:
        _worker_init(dict_ckpt, data_dir, dataset)
        out = [_worker_run(it) for it in work_items]
    else:
        with mp.Pool(processes=n_workers, initializer=_worker_init,
                    initargs=(dict_ckpt, data_dir, dataset)) as pool:
            out = pool.map(_worker_run, work_items, chunksize=4)
    print(f"  parallel_walk: {len(work_items)} rollouts, {n_workers} workers, "
          f"{time.time()-t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gensft-ckpt", type=str, required=True)
    ap.add_argument("--dict-ckpt", type=str, default=gc.DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=gc.DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=gc.DD)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--bon-n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260908)  # 200 題構造 seed，同 teacher_relay
    ap.add_argument("--rho", type=float, default=gc.RHO)
    ap.add_argument("--tag", type=str, default="gensft_eval")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    ap.add_argument("--gif-dir", type=str, default=os.path.join(gc.RESULTS_DIR, "gifs"))
    ap.add_argument("--skip-gifs", action="store_true")
    ap.add_argument("--force-past-gate", action="store_true",
                    help="⛔ 除錯用；正常執行不准開，對照組沒過閘門就是要停手")
    args = ap.parse_args()

    torch.set_num_threads(8)
    t0 = time.time()
    print(f"=== gen_sft eval  gensft_ckpt={os.path.basename(args.gensft_ckpt)}  "
          f"n_traj={args.n_traj}  bon_n={args.bon_n}  seed={args.seed}  rho={args.rho} ===")

    raw = gc.wv.load_npz(args.data_dir, gc.DATASET, "val")
    ruler = gc.load_ruler(args.ruler)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    dict_model, dict_cfg = gc.load_dict_model(args.dict_ckpt)
    print(f"dict: seg_len={dict_cfg['seg_len']} K={dict_cfg['k']}  fall_line={fall_line:.4f}")

    ck = torch.load(args.gensft_ckpt, map_location="cpu", weights_only=False)
    gcfg = ck["config"]
    gmodel = GenSFT(**gcfg)
    gmodel.load_state_dict(ck["state_dict"])
    gmodel.eval()
    print(f"gensft model: {gcfg}  best_step={ck.get('best_step')}  "
          f"held_out_at_best={ck.get('held_out_at_best')}")

    full_tasks, skipped = gc.rt.build_tasks(raw, n_traj=200, seed=args.seed, ruler=ruler, rho=args.rho)
    print(f"tasks: {len(full_tasks)} usable (要求>=200)  skipped={len(skipped)}")
    assert len(full_tasks) >= 200, "⛔ 可用軌跡不足 200，停手回報"
    eval_tasks = full_tasks[:args.n_traj]
    control_tasks = full_tasks[:CONTROL_N]
    Ms = np.array([t["M"] for t in eval_tasks])
    print(f"eval_tasks={len(eval_tasks)}  waypoints M: min={Ms.min()} p50={np.median(Ms):.0f} "
          f"max={Ms.max()}  total_legs={Ms.sum()}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 步驟一：對照組（隨機字串，均勻取 code），先跑、先過閘門（design 第 4 點）
    # ---------------------------------------------------------------
    print(f"\n=== 對照組：隨機字串 x {CONTROL_N} 題（control_seed={CONTROL_SEED}）===")
    crng = np.random.default_rng(CONTROL_SEED)
    control_codes = [crng.integers(0, 32, size=t["n_chunks"]).astype(np.int64) for t in control_tasks]
    control_items = [dict(group="control", i=i, task=t, codes=control_codes[i], rho=args.rho)
                     for i, t in enumerate(control_tasks)]
    control_raw = parallel_walk(control_items, args.dict_ckpt, args.data_dir, gc.DATASET,
                                min(args.workers, CONTROL_N))
    control_runs = [None] * CONTROL_N
    for o in control_raw:
        control_runs[o["i"]] = o["r"]
    control_rec, control_arrays = aggregate_runs(control_tasks, control_runs, fall_line)
    print(f"control per-leg 不限時到達率 = {control_rec['per_leg_untimed_reach']:.4f}  "
          f"(reached {control_rec['legs_reached_total']}/{control_rec['legs_attempted_total']})  "
          f"閘門門檻 < {GATE_MAX_PER_LEG}")

    gate_pass = control_rec["per_leg_untimed_reach"] < GATE_MAX_PER_LEG
    if not gate_pass and not args.force_past_gate:
        print(f"\n⛔⛔⛔ 對照組 per-leg={control_rec['per_leg_untimed_reach']:.4f} >= "
              f"{GATE_MAX_PER_LEG} —— 管線疑似壞了（隨機字串不該走得動），依 design 第 4 點"
              f"停手，不准繼續 greedy/BoN。")
        gc.wv.save_json(dict(gate_pass=False, control=control_rec,
                            config={k: v for k, v in vars(args).items()}),
                        os.path.join(args.out_dir, f"{args.tag}_GATE_FAIL.json"))
        sys.exit(1)
    print(f"閘門通過（{'force_past_gate 強制略過檢查' if not gate_pass else 'PASS'}），繼續 greedy/BoN。")

    # ---------------------------------------------------------------
    # 步驟二：u 生成字串 —— 貪心 1 條 + BoN(N) 條（都在單一行程內做，模型很小很快）
    # ---------------------------------------------------------------
    s0_t, wp_t, wpm_t = tasks_to_cond(eval_tasks, raw)
    tg0 = time.time()
    greedy_out = gmodel.generate(s0_t, wp_t, wpm_t, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
    greedy_codes = [g for g, _ in greedy_out]
    greedy_hit_cap = sum(1 for _, ended_eos in greedy_out if not ended_eos)  # 撞 cap=沒吐 EOS
    Lg = np.array([len(g) for g in greedy_codes])
    print(f"\ngreedy 生成：{len(greedy_codes)} 條，長度 min/p50/max="
          f"{Lg.min()}/{np.median(Lg):.0f}/{Lg.max()}  撞 max_new_tokens 未出 EOS 的條數="
          f"{greedy_hit_cap}  ({time.time()-tg0:.1f}s)")

    tb0 = time.time()
    N = args.bon_n
    s0_rep = s0_t.repeat_interleave(N, dim=0)
    wp_rep = wp_t.repeat_interleave(N, dim=0)
    wpm_rep = wpm_t.repeat_interleave(N, dim=0)
    bon_gen = torch.Generator().manual_seed(BON_SEED)
    bon_out = gmodel.generate(s0_rep, wp_rep, wpm_rep, max_new_tokens=MAX_NEW_TOKENS,
                              temperature=1.0, rng=bon_gen)
    bon_codes = [[bon_out[i * N + j][0] for j in range(N)] for i in range(len(eval_tasks))]
    bon_hit_cap = sum(1 for _, ended_eos in bon_out if not ended_eos)  # 撞 cap=沒吐 EOS
    Lb = np.array([len(g) for g, _ in bon_out])
    print(f"BoN 生成：{len(eval_tasks)}x{N}={len(bon_out)} 條（temperature=1.0, bon_seed={BON_SEED}），"
          f"長度 min/p50/max={Lb.min()}/{np.median(Lb):.0f}/{Lb.max()}  撞 cap 條數={bon_hit_cap}  "
          f"({time.time()-tb0:.1f}s)")

    # ---------------------------------------------------------------
    # 步驟三：照生成字串走一遍（mujoco，平行化）
    # ---------------------------------------------------------------
    work_items = []
    for i, t in enumerate(eval_tasks):
        work_items.append(dict(group="greedy", i=i, j=None, task=t, codes=greedy_codes[i], rho=args.rho))
    for i, t in enumerate(eval_tasks):
        for j in range(N):
            work_items.append(dict(group="bon", i=i, j=j, task=t, codes=bon_codes[i][j], rho=args.rho))
    print(f"\n=== 開始 rollout：greedy {len(eval_tasks)} + BoN {len(eval_tasks)}x{N} = "
          f"{len(work_items)} 條 ===")
    raw_out = parallel_walk(work_items, args.dict_ckpt, args.data_dir, gc.DATASET, args.workers)

    greedy_runs = [None] * len(eval_tasks)
    bon_all_runs = [[None] * N for _ in range(len(eval_tasks))]
    for o in raw_out:
        if o["group"] == "greedy":
            greedy_runs[o["i"]] = o["r"]
        else:
            bon_all_runs[o["i"]][o["j"]] = o["r"]

    greedy_rec, greedy_arrays = aggregate_runs(eval_tasks, greedy_runs, fall_line)

    # BoN：每題在 N 條裡挑 legs_reached 最大的那條當代表（legs_reached 越大，
    # per-task per-leg 比率 (active_leg-1)/active_leg 也越大，兩個判準等價，見
    # NOTE §一設計說明），用同一支 aggregate_runs 聚合這些「winner」。
    bon_winner_runs = []
    bon_winner_idx = []
    for i in range(len(eval_tasks)):
        runs_i = bon_all_runs[i]
        best_j = int(np.argmax([r["legs_reached"] for r in runs_i]))
        bon_winner_runs.append(runs_i[best_j])
        bon_winner_idx.append(best_j)
    bon_rec, bon_arrays = aggregate_runs(eval_tasks, bon_winner_runs, fall_line)

    print(f"\n[greedy] per-leg={greedy_rec['per_leg_untimed_reach']:.4f}  "
          f"completion={greedy_rec['completion_ratio']:.4f}  "
          f"speed_p50={greedy_rec['step_speed_q']['50']:.4f}  "
          f"fall(step)={greedy_rec['fall_rate_step']*100:.2f}%  "
          f"adiff_p50={greedy_rec['action_diff_q']['50']:.3f}")
    print(f"[BoN N={N}] per-leg={bon_rec['per_leg_untimed_reach']:.4f}  "
          f"completion={bon_rec['completion_ratio']:.4f}  "
          f"speed_p50={bon_rec['step_speed_q']['50']:.4f}  "
          f"fall(step)={bon_rec['fall_rate_step']*100:.2f}%  "
          f"adiff_p50={bon_rec['action_diff_q']['50']:.3f}")
    print(f"[control random] per-leg={control_rec['per_leg_untimed_reach']:.4f}  "
          f"completion={control_rec['completion_ratio']:.4f}")

    print(f"\n=== 對照行（一手引用，非本次量測）===")
    for k, v in ANCHORS.items():
        print(f"  {k:26s} = {v:.3f}")
    print(f"（參照：已訓練主頭 closed-loop per-leg 量級 = {MAIN_HEAD_REFERENCE_LEVEL}"
          f"——跟本次 u 生成開環播放是不同機制，不嚴格同分佈，見 NOTE 可比性註記）")
    print(f"驗收①「學會寫譜」門檻(>={LEARNED_GAIT_THRESHOLD}): greedy "
          f"{'PASS' if greedy_rec['per_leg_untimed_reach'] >= LEARNED_GAIT_THRESHOLD else 'FAIL'}  "
          f"(greedy={greedy_rec['per_leg_untimed_reach']:.4f})")
    print(f"驗收②「貼近教材上界」門檻(>={NEAR_TEACHER_THRESHOLD}): greedy "
          f"{'PASS' if greedy_rec['per_leg_untimed_reach'] >= NEAR_TEACHER_THRESHOLD else 'FAIL'}  BoN "
          f"{'PASS' if bon_rec['per_leg_untimed_reach'] >= NEAR_TEACHER_THRESHOLD else 'FAIL'}")

    # ---------------------------------------------------------------
    # 疊圖（真螞蟻 vs greedy vs BoN vs control）
    # ---------------------------------------------------------------
    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "gen_sft eval gate2: step speed vs real ant", "|dxy| per step", None),
        ("z", "torso_z", "gen_sft eval gate3a: torso height vs real ant", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "gen_sft eval gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL ant (ruler)", values=ruler_np[ref])]
        for name, arrs in (("greedy", greedy_arrays), ("BoN", bon_arrays), ("control", control_arrays)):
            arr = arrs[key]
            if len(arr):
                series.append(dict(label=f"gensft[{name}]", values=arr))
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        gc.wv.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")

    # ---------------------------------------------------------------
    # gif：成功/失敗各 2 支，取自 greedy 結果（沿 teacher_relay 的兩格並排格式）
    # ---------------------------------------------------------------
    gifs = {}
    if not args.skip_gifs:
        gifs = render_gif_exemplars(raw, dict_model, eval_tasks, greedy_rec, greedy_codes,
                                    args.rho, args.gif_dir, args.data_dir)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    gc.wv.save_json(dict(
        config={k: v for k, v in vars(args).items()}, anchors=ANCHORS,
        control_seed=CONTROL_SEED, bon_seed=BON_SEED, max_new_tokens=MAX_NEW_TOKENS,
        gate=dict(pass_=gate_pass, threshold=GATE_MAX_PER_LEG,
                  control_per_leg=control_rec["per_leg_untimed_reach"]),
        n_tasks_eval=len(eval_tasks), n_tasks_control=CONTROL_N,
        gensft_ckpt_meta=dict(config=gcfg, best_step=ck.get("best_step"),
                              held_out_at_best=ck.get("held_out_at_best")),
        generation=dict(greedy_len=dict(min=int(Lg.min()), p50=float(np.median(Lg)), max=int(Lg.max())),
                        greedy_hit_cap=int(greedy_hit_cap),
                        bon_len=dict(min=int(Lb.min()), p50=float(np.median(Lb)), max=int(Lb.max())),
                        bon_hit_cap=int(bon_hit_cap), bon_winner_idx=bon_winner_idx),
        results=dict(greedy=greedy_rec, bon=bon_rec, control=control_rec),
        artifacts=dict(pngs=pngs, gifs=gifs),
    ), js_path)
    print(f"saved: {js_path}")

    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        greedy_speed=greedy_arrays["speed"], greedy_z=greedy_arrays["z"],
                        greedy_adiff=greedy_arrays["adiff"], bon_speed=bon_arrays["speed"],
                        bon_z=bon_arrays["z"], bon_adiff=bon_arrays["adiff"],
                        control_speed=control_arrays["speed"], control_z=control_arrays["z"],
                        control_adiff=control_arrays["adiff"])
    print(f"=== done wall={time.time()-t0:.1f}s ===")


def render_gif_exemplars(raw, dict_model, eval_tasks, greedy_rec, greedy_codes, rho, gif_dir, data_dir):
    """成功/失敗各 2 支，取自 greedy 結果。相機/兩格並排/frame_every/max_frames 設定
    逐項照抄 run_teacher_relay.render_exemplars()（同檔案不可 import 復用，因為它綁死
    hindsight schedule/nn 兩種 code 來源，這裡改呼叫 run_one_generated 並傳生成字串）。
    """
    import mujoco
    env = gc.wv.make_env(data_dir, gc.DATASET)
    u = env.unwrapped
    env.reset()
    u.set_state(raw["qpos"][eval_tasks[0]["s0"]].copy(), raw["qvel"][eval_tasks[0]["s0"]].copy())
    _ = u.render()
    renderer = u.custom_renderer
    cam = mujoco.MjvCamera()
    cam.lookat[2] = 0.3
    cam.distance = 6.0
    cam.elevation = -35
    cam.azimuth = 135
    cam_wide = mujoco.MjvCamera()
    cam_wide.lookat[0], cam_wide.lookat[1] = gc.MAZE_CENTER
    cam_wide.lookat[2] = 0.0
    cam_wide.distance = 38.0
    cam_wide.elevation = -75
    cam_wide.azimuth = 90
    cams = (cam, cam_wide)

    per_task = greedy_rec["per_task"]
    succ = [p for p in per_task if p["completed"]][:2]
    fail = [p for p in per_task if not p["completed"]][:2]
    if len(succ) < 2:
        print(f"⚠️ greedy 只有 {len(succ)} 條 completed=True，成功 gif 數量不足 2 支")
    if len(fail) < 2:
        print(f"⚠️ greedy 只有 {len(fail)} 條 completed=False，失敗 gif 數量不足 2 支")

    made = {}
    for kind, recs in (("success", succ), ("fail", fail)):
        for n, p in enumerate(recs):
            task = eval_tasks[p["ti"]]
            codes = greedy_codes[p["ti"]]
            r = run_one_generated(env, u, dict_model, raw, task, codes, rho, render=True,
                                  renderer=renderer, cams=cams, frame_every=4, max_frames=200)
            assert r["completed"] == p["completed"] and r["n_steps_run"] == p["n_steps_run"], (
                f"⛔ 重跑不一致（應為 bit-exact 決定性）：kind={kind} n={n} "
                f"first_pass=(completed={p['completed']},steps={p['n_steps_run']}) "
                f"second_pass=(completed={r['completed']},steps={r['n_steps_run']})")
            fp = os.path.join(gif_dir, f"greedy_{kind}_{n}_ep{task['episode']}.gif")
            if r["frames"]:
                import imageio.v2 as imageio
                imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
                made[f"{kind}_{n}"] = dict(path=fp, n_frames=len(r["frames"]), episode=task["episode"],
                                           legs_reached=r["legs_reached"], M=task["M"],
                                           n_steps_run=r["n_steps_run"])
                print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                      f"legs {r['legs_reached']}/{task['M']})")
            else:
                made[f"{kind}_{n}"] = None
                print(f"⛔ render 失敗（0 幀）：{kind}/{n} episode={task['episode']}")
    return made


if __name__ == "__main__":
    main()
