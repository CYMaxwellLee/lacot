#!/usr/bin/env python
"""定量拆解「完美教材開環播放為什麼還是只有 .554」——量化損失 vs 執行漂移。

背景：docs/NOTE-2026-09-08-teacher-relay.md 量到 hindsight 真字串（完美教材譜）在
接力協定下 schedule 版（純開環照表播）per-leg 不限時到達率只有 .554，
docs/NOTE-2026-09-08-relay-byleg.md 進一步拆出失敗形態 77% 是「走歪」不是「走慢」。
兩篇都沒回答「這個病是【字典量化本身有損】還是【開環執行漸漸脫離教材軌跡】」——
本檔補這一刀，只量 schedule 版（「完美譜開環播放」就是這一版；nn 版是閉環修正，
不在本次範圍內）。

【在證明什麼、判準是什麼】
證明：schedule 版 per-leg .554 的失敗，量化損失與開環執行漂移各佔多少、失敗能不能
由「開始那一刻已經漂了多遠」預測。
判準：
  1. 復現斷言 —— 本檔重放的 3 條題目 per-leg 事件序列必須與已存檔的原 run
     （teacher_relay_byleg_summary.json 的 leg_rows）逐欄位一致；額外用同一次重放
     的 200 條題目彙總（completed/legs_attempted/legs_reached/n_steps_run）跟
     teacher_relay_summary.json 的 per_task 做全量對帳（比 3 條題目更嚴，免費多做）。
     ⛔ 任何一項不一致就印出差異、hard exit(1)，不往下跑 200 題全量測。
  2. 臂 1（cont，本檔新寫、逐 chunk 邊界加測漂移）跟臂 2（indep，本檔新寫、
     每 chunk 重置回真起點）的「首 chunk 誤差」必須同量級 —— 兩者在 j=1／k=0
     那一點理論上量的是同一件事（剛好都是「從真起點執行 1 個 chunk 後的誤差」），
     算出比值印出來，不預設一定要幾乎相等，只要求同量級（見下方程式碼裡的推導
     註解）。
  3. 拆帳表：臂 1 漂移隨 chunk index 的成長 減去 臂 2 的單 chunk 基線 = 執行漂移的
     淨貢獻；同時報「失敗 leg 開始時漂移」vs「成功 leg 開始時漂移」兩個分佈的
     分離度（AUC 式指標 + 門檻搜尋），分不開也要報。

【不修改任何既有檔案】只 import 同目錄上一層 walk_verify 的 wv_common / p1_replay，
以及同目錄的 teacher_relay/run_teacher_relay_byleg.py（import，不改）。
task 構造（build_tasks）直接 import 沿用，保證跟原 run 的 200 題 100% 同源；
run_one() 因為要新增逐 chunk 記錄，無法只 import，改成本檔複本＋擴充（複本邏輯
逐行對齊 run_teacher_relay_byleg.run_one 的 schedule 分支，對帳見上）。

⛔ CPU-only（MuJoCo 物理 + 小 MLP encode/decode）。全程跑 sbatch，見
   run_drift_analysis.sbatch。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))                       # drift_analysis/
WV_DIR = os.path.dirname(HERE)                                          # walk_verify/
TR_DIR = os.path.join(WV_DIR, "teacher_relay")                          # walk_verify/teacher_relay/
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402 - 既有共用模組，未修改
from p1_replay import sim_obs, q  # noqa: E402 - 純函式重用，未修改
import run_teacher_relay_byleg as trb  # noqa: E402 - 既有腳本當模組 import，未修改一行
from drift_plots import draw_percentile_band  # noqa: E402 - 本檔新寫的畫圖工具

DD = trb.DD
DATASET = trb.DATASET
DEFAULT_CKPT = trb.DEFAULT_CKPT
DEFAULT_RULER = trb.DEFAULT_RULER
DELTA_SUB = trb.DELTA_SUB
RHO_DEFAULT = trb.RHO_DEFAULT

GT_SUMMARY_JSON = os.path.join(TR_DIR, "results", "teacher_relay_summary.json")
GT_BYLEG_JSON = os.path.join(TR_DIR, "results", "teacher_relay_byleg_summary.json")

# 拆帳表要印出的 chunk-index 格點（不是全部 0..50 都印在表裡，圖裡是連續的）。
J_GRID_PRINT = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50]


# ------------------------------------------------------------------
# 臂 1（cont）：schedule 版接力，逐 chunk 邊界記錄漂移 + 逐 leg 記錄 leg-start 漂移。
# 複本＋擴充自 run_teacher_relay_byleg.run_one() 的 schedule 分支（未改一行控制流程，
# 只新增讀取式的記錄呼叫 —— 不寫 sim 狀態、不影響 env.step/set_state 的呼叫序列，
# 所以理論上不會改變 completed/leg_reach_step/leg_active_step/leg_start_xy 的值；
# 這個「理論上」就是為什麼要做復現斷言，不能只靠讀 code 保證。
# ------------------------------------------------------------------
def run_one_drift(env, u, model, raw, task, rho):
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act, obs_arr, qpos_arr, qvel_arr = raw["actions"], raw["observations"], raw["qpos"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)

    env.reset()
    u.set_state(qpos_arr[s0].copy(), qvel_arr[s0].copy())

    active_leg = 1
    leg_reach_step, leg_active_step = {}, {1: 0}
    leg_start_xy = {1: qpos_arr[s0, :2].copy()}
    leg_start_drift = {1: 0.0}  # leg 1 從 reset 點開始 —— 定義上漂移=0，不是量出來的
    xy_hist = []
    codes_used = []
    global_step = 0

    boundary_j, boundary_xy_drift, boundary_state_drift, boundary_active_leg = [], [], [], []

    def record_boundary(j):
        """在『第 j 個 chunk 即將開始』的那一刻記錄：跟教材軌跡第 j 個 chunk 起點
        （真實時刻索引 s0+4j）比，模擬狀態偏了多少。j=0 時剛 set_state 完，定義上=0。
        """
        cur_xy = np.asarray(u.data.qpos)[:2].copy()
        true_idx = s0 + 4 * j
        xy_drift = float(np.linalg.norm(cur_xy - qpos_arr[true_idx, :2]))
        sim_o = sim_obs(u)
        true_o = obs_arr[true_idx]
        with torch.no_grad():
            a_n = model.norm_obs(torch.from_numpy(sim_o[None, :].astype(np.float32)))
            b_n = model.norm_obs(torch.from_numpy(true_o[None, :].astype(np.float32)))
            state_drift = float(torch.norm(a_n - b_n).item())
        boundary_j.append(j)
        boundary_xy_drift.append(xy_drift)
        boundary_state_drift.append(state_drift)
        boundary_active_leg.append(active_leg)

    record_boundary(0)

    for k in range(n_chunks):
        s_idx = s0 + 4 * k  # schedule：一律用第 k 段自己的 hindsight 真字，跟模擬中 ant 在哪無關
        with torch.no_grad():
            x = torch.from_numpy(act[s_idx:s_idx + 4].reshape(1, -1).astype(np.float32))
            idx = model.encode_idx(x)
            o_in = torch.from_numpy(sim_obs(u)[None, :])
            raw_a = model.decode_from_idx(idx, o_in).numpy().reshape(4, wv.ACT_DIM)
        codes_used.append(int(idx.item()))
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)

        for t in range(4):
            env.step(a_chunk[t])
            cur = np.asarray(u.data.qpos)[:2].copy()
            xy_hist.append(cur.copy())
            while active_leg <= M and np.linalg.norm(cur - wp_xy[active_leg - 1]) <= rho:
                leg_reach_step[active_leg] = global_step
                active_leg += 1
                if active_leg <= M:
                    leg_active_step[active_leg] = global_step + 1
                    leg_start_xy[active_leg] = cur.copy()
                    true_idx_leg = s0 + leg_active_step[active_leg]
                    leg_start_drift[active_leg] = float(
                        np.linalg.norm(cur - qpos_arr[true_idx_leg, :2]))
            global_step += 1
        record_boundary(k + 1)
        if active_leg > M:
            break

    completed = active_leg > M
    return dict(
        completed=completed, legs_attempted=min(active_leg, M),
        legs_reached=active_leg - 1, n_steps_run=global_step,
        xy=np.asarray(xy_hist), codes_used=np.asarray(codes_used, dtype=np.int64),
        leg_reach_step=dict(leg_reach_step), leg_active_step=dict(leg_active_step),
        leg_start_xy=dict(leg_start_xy), leg_start_drift=dict(leg_start_drift),
        boundary_j=np.asarray(boundary_j), boundary_xy_drift=np.asarray(boundary_xy_drift),
        boundary_state_drift=np.asarray(boundary_state_drift),
        boundary_active_leg=np.asarray(boundary_active_leg))


def leg_rows_from_run(r, task):
    """把 run_one_drift() 的回傳轉成跟 teacher_relay_byleg leg_rows 同一組欄位，供對帳。
    這段邏輯逐行照抄 run_teacher_relay_byleg.measure_version() 裡的 leg-row 區塊
    （該檔未被本檔匯入這個區塊 —— 它埋在 measure_version 函式內部不是獨立函式，
    所以無法只 import；照抄＋在復現斷言裡逐欄位比對，是能驗證「照抄沒抄錯」的辦法）。
    """
    rows = []
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
        else:
            end_xy = start_xy
        d_end = float(np.linalg.norm(end_xy - target_xy))
        rows.append(dict(m=int(m), reached=bool(reached_m), active_step=int(act_step),
                          end_step=int(end_step), first_hit=float(fh),
                          d_start=d_start, d_end=d_end))
    return rows


# ------------------------------------------------------------------
# 臂 2（indep / 重置對照）：每個 chunk 開始前 set_state 回教材真起點，執行 1 chunk，
# 量純字典量化＋單步展開誤差。不追蹤 leg／到達（重置後軌跡鎖回真路徑，到達率無意義，
# 見工單設計 §2）。固定跑滿 n_chunks（不像臂 1 會提早結束）—— 每個 chunk index 都要
# 有全部 200 條題目的基線樣本。
#
# 跟 p1_replay.py 的 dict_indep 定義等價：那邊算 chunk_disp_err = norm(disp - true_disp)，
# disp = 執行後位移、true_disp = 真實同段位移，兩者都是相對同一個 base_xy=qpos[s,:2]
# （reset 到的真起點）算的位移，所以 disp - true_disp = end_xy - true_end_xy —— 跟本檔
# 直接比較「段尾絕對座標」代數上是同一件事，這裡走比較白話的絕對座標版本。
# ------------------------------------------------------------------
def run_reset_chunks(env, u, model, raw, task):
    s0, n_chunks = task["s0"], task["n_chunks"]
    act, obs_arr, qpos_arr, qvel_arr = raw["actions"], raw["observations"], raw["qpos"], raw["qvel"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    xy_err, state_err, codes = [], [], []
    for k in range(n_chunks):
        s = s0 + 4 * k
        env.reset()
        u.set_state(qpos_arr[s].copy(), qvel_arr[s].copy())
        with torch.no_grad():
            x = torch.from_numpy(act[s:s + 4].reshape(1, -1).astype(np.float32))
            idx = model.encode_idx(x)
            o_in = torch.from_numpy(sim_obs(u)[None, :])  # 剛 reset，等於真 obs
            raw_a = model.decode_from_idx(idx, o_in).numpy().reshape(4, wv.ACT_DIM)
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)
        for t in range(4):
            env.step(a_chunk[t])
        end_xy = np.asarray(u.data.qpos)[:2].copy()
        true_end_xy = qpos_arr[s + 4, :2]
        xy_err.append(float(np.linalg.norm(end_xy - true_end_xy)))
        sim_o = sim_obs(u)
        true_o = obs_arr[s + 4]
        with torch.no_grad():
            a_n = model.norm_obs(torch.from_numpy(sim_o[None, :].astype(np.float32)))
            b_n = model.norm_obs(torch.from_numpy(true_o[None, :].astype(np.float32)))
            state_err.append(float(torch.norm(a_n - b_n).item()))
        codes.append(int(idx.item()))
    return dict(xy_err=np.asarray(xy_err), state_err=np.asarray(state_err),
                codes=np.asarray(codes, dtype=np.int64))


def band_stats(values_by_x, xs_sorted):
    p25, p50, p75, n = [], [], [], []
    for x in xs_sorted:
        v = np.asarray(values_by_x.get(x, []), dtype=float)
        if len(v) == 0:
            p25.append(np.nan); p50.append(np.nan); p75.append(np.nan); n.append(0)
        else:
            p25.append(float(np.percentile(v, 25)))
            p50.append(float(np.percentile(v, 50)))
            p75.append(float(np.percentile(v, 75)))
            n.append(int(len(v)))
    return dict(xs=xs_sorted, p25=p25, p50=p50, p75=p75, n=n)


def auc_separation(fail_vals, succ_vals):
    """P(隨機一個失敗樣本的漂移 > 隨機一個成功樣本的漂移)，等值算 0.5。0.5=完全分不開。"""
    f = np.asarray(fail_vals, dtype=float)[:, None]
    s = np.asarray(succ_vals, dtype=float)[None, :]
    gt = (f > s).mean()
    eq = (f == s).mean()
    return float(gt + 0.5 * eq)


def best_threshold(fail_vals, succ_vals):
    """掃過所有觀察值當候選門檻（predict fail if drift>=thr），回傳最佳 accuracy 那個。"""
    f = np.asarray(fail_vals, dtype=float)
    s = np.asarray(succ_vals, dtype=float)
    cands = np.unique(np.concatenate([f, s]))
    n = len(f) + len(s)
    best = (-1.0, None)
    for thr in cands:
        acc = ((f >= thr).sum() + (s < thr).sum()) / n
        if acc > best[0]:
            best = (float(acc), float(thr))
    baseline = max(len(f), len(s)) / n
    return dict(best_acc=best[0], best_thr=best[1], majority_baseline_acc=float(baseline),
                n_fail=int(len(f)), n_succ=int(len(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    # ⚠️ 預設跟 run_teacher_relay_byleg.py 的 --threads 預設值（4）對齊，不是隨便挑的：
    # 這條 pipeline 的 schedule 版本身會被 float32 精度誤差放大成軌跡級差異
    # （見 docs/NOTE-2026-09-08-walk-verify.md §2.2「float32 混沌」），thread 數不同
    # 可能讓 CPU matmul 的加總順序跟著變、放大成復現斷言對不上——保守起見跟原 run
    # 用同一個 thread 數，把這個混淆變因先排除掉。sbatch 仍分配 --cpus-per-task=8，
    # 只是 torch intra-op 不用滿，不衝突。
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, default="drift")
    ap.add_argument("--repro-ti", type=str, default="0,2,3",
                     help="復現斷言要核對的 3 條題目索引（逗號分隔，對應 build_tasks 的 ti 順序）")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    repro_tis = [int(x) for x in args.repro_ti.split(",")]
    assert len(repro_tis) == 3, f"⛔ 復現斷言要求恰好 3 條題目，收到 {len(repro_tis)} 條"

    print(f"=== drift-analysis  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"seed={args.seed} rho={args.rho} DELTA_SUB={DELTA_SUB} repro_ti={repro_tis} ===")

    model, cfg = wv.load_dict_v1(args.ckpt)
    assert cfg["seg_len"] == 4, f"預期 seg_len=4，實際 {cfg['seg_len']}"
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}  ruler fall_line={fall_line:.4f}")

    raw = wv.load_npz(args.data_dir, DATASET, "val")
    tasks, skipped = trb.build_tasks(raw, args.n_traj, args.seed, ruler, args.rho)  # 既有函式，import 未改
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}")
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報（不用預設值填）"
    Ms = np.array([t["M"] for t in tasks])
    print(f"waypoints per task: min={Ms.min()} p50={np.median(Ms):.0f} max={Ms.max()} "
          f"total_legs_available={Ms.sum()}")
    n_chunks_all = np.array([t["n_chunks"] for t in tasks])
    assert (n_chunks_all == n_chunks_all[0]).all(), (
        f"⛔ 預期所有 task 的 n_chunks 一致（EP_LEN 固定 201），實際看到 {np.unique(n_chunks_all)}")
    N_CHUNKS = int(n_chunks_all[0])
    print(f"n_chunks per task = {N_CHUNKS}（全部一致，val episode 長度固定 201 步）")

    env = wv.make_env(args.data_dir, DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================
    # STEP 1：復現斷言（先做，做不到就停）
    # ================================================================
    print("\n=== STEP 1：復現斷言 ===")
    if not (os.path.exists(GT_SUMMARY_JSON) and os.path.exists(GT_BYLEG_JSON)):
        print(f"⛔ 找不到既有結果檔，無法核對：{GT_SUMMARY_JSON} / {GT_BYLEG_JSON}")
        sys.exit(1)
    gt_summary = json.load(open(GT_SUMMARY_JSON))
    gt_byleg = json.load(open(GT_BYLEG_JSON))
    gt_per_task = {p["ti"]: p for p in gt_summary["results"]["schedule"]["per_task"]}
    gt_leg_rows_by_ti = defaultdict(list)
    for row in gt_byleg["leg_rows"]["schedule"]:
        gt_leg_rows_by_ti[row["ti"]].append(row)

    repro_fail = []
    for ti in repro_tis:
        task = tasks[ti]
        gpt = gt_per_task.get(ti)
        if gpt is None or gpt["episode"] != task["episode"]:
            repro_fail.append(f"ti={ti}: 題目抽樣對不上（gt episode="
                               f"{gpt['episode'] if gpt else None} vs 本次 {task['episode']}）")
            continue
        r = run_one_drift(env, u, model, raw, task, args.rho)
        for field, gt_val in (("completed", gpt["completed"]),
                               ("legs_attempted", gpt["legs_attempted"]),
                               ("legs_reached", gpt["legs_reached"]),
                               ("n_steps_run", gpt["n_steps_run"])):
            mine = r[field]
            if mine != gt_val:
                repro_fail.append(f"ti={ti} task-level {field}: gt={gt_val} mine={mine}")

        mine_rows = leg_rows_from_run(r, task)
        gt_rows = sorted(gt_leg_rows_by_ti[ti], key=lambda x: x["m"])
        if len(mine_rows) != len(gt_rows):
            repro_fail.append(f"ti={ti}: leg 數不對，gt={len(gt_rows)} mine={len(mine_rows)}")
        for gr, mr in zip(gt_rows, mine_rows):
            for field in ("m", "reached", "active_step", "end_step", "first_hit", "d_start", "d_end"):
                gv, mv = gr[field], mr[field]
                ok = (gv == mv) if isinstance(gv, (bool, int)) else abs(gv - mv) < 1e-6
                if not ok:
                    repro_fail.append(f"ti={ti} leg m={gr['m']} {field}: gt={gv} mine={mv}")
        status = "PASS" if not any(f.startswith(f"ti={ti}") for f in repro_fail) else "FAIL"
        print(f"  ti={ti} episode={task['episode']} M={task['M']} legs={len(gt_rows)} "
              f"completed(gt/mine)={gpt['completed']}/{r['completed']}  對帳={status}")

    if repro_fail:
        print("\n⛔ 復現斷言失敗，明細：")
        for f in repro_fail:
            print("   -", f)
        print("⛔ 復現不了，依工單規則停手，不往下跑 200 題全量測。")
        sys.exit(1)
    print(f"復現斷言 PASS：{len(repro_tis)} 條題目（ti={repro_tis}）per-leg 事件序列與"
          f" teacher_relay_byleg_summary.json 逐欄位一致（m/reached/active_step/end_step/"
          f"first_hit/d_start/d_end），task 級欄位與 teacher_relay_summary.json 一致。")

    # ================================================================
    # STEP 2：臂 1（cont）—— 200 題全量，逐 chunk 邊界 + 逐 leg 記錄
    # ================================================================
    print("\n=== STEP 2：臂 1（cont，schedule 開環）200 題全量 ===")
    ta = time.time()
    boundary_rows = []   # (ti, j, completed, xy_drift, state_drift, active_leg)
    leg_rows_all = []    # (ti, m, M, reached, leg_start_drift)
    arm1_codes_by_ti = {}
    coarse_mismatch = []
    n_completed = 0
    for ti, task in enumerate(tasks):
        r = run_one_drift(env, u, model, raw, task, args.rho)
        arm1_codes_by_ti[ti] = r["codes_used"]
        if r["completed"]:
            n_completed += 1
        for j, xd, sd, al in zip(r["boundary_j"], r["boundary_xy_drift"],
                                  r["boundary_state_drift"], r["boundary_active_leg"]):
            boundary_rows.append((ti, int(j), bool(r["completed"]), float(xd), float(sd), int(al)))
        for m in range(1, r["legs_attempted"] + 1):
            reached_m = r["leg_reach_step"].get(m, None) is not None
            leg_rows_all.append(dict(ti=ti, m=m, M=task["M"], reached=reached_m,
                                     leg_start_drift=r["leg_start_drift"][m]))
        gpt = gt_per_task.get(ti)
        if gpt is not None:
            for field in ("completed", "legs_attempted", "legs_reached", "n_steps_run"):
                if r[field] != gpt[field]:
                    coarse_mismatch.append((ti, field, gpt[field], r[field]))
    completion_ratio = n_completed / len(tasks)
    print(f"done in {time.time()-ta:.1f}s  completion_ratio={completion_ratio:.3f}  "
          f"(gt teacher_relay 為 0.125，供對照)")
    print(f"200 題粗對帳（vs teacher_relay_summary.json per_task，4 欄位 x 200 題 = 800 項）："
          f"{'全部一致' if not coarse_mismatch else f'{len(coarse_mismatch)} 項不一致'}")
    if coarse_mismatch:
        print("⛔ 200 題粗對帳出現不一致，明細（最多印 10 項）：")
        for m in coarse_mismatch[:10]:
            print("   -", m)
        print("⛔ 停手回報，不往下跑臂 2 與後續分析。")
        sys.exit(1)

    # ================================================================
    # STEP 3：臂 2（indep 重置對照）—— 200 題 x 50 chunk 全量
    # ================================================================
    print("\n=== STEP 3：臂 2（indep 重置對照）200 題 x 每題全部 chunk ===")
    ta = time.time()
    reset_xy_err_by_k = defaultdict(list)
    reset_state_err_by_k = defaultdict(list)
    code_mismatch = 0
    code_compare_n = 0
    for ti, task in enumerate(tasks):
        rr = run_reset_chunks(env, u, model, raw, task)
        for k in range(len(rr["xy_err"])):
            reset_xy_err_by_k[k].append(float(rr["xy_err"][k]))
            reset_state_err_by_k[k].append(float(rr["state_err"][k]))
        # 自檢：encode 只吃真動作段，不吃 obs，理論上臂1／臂2 在同一個 k 選到的字必須一樣
        # （臂1 早停時只比對它實際跑過的那些 k）。
        n_common = len(arm1_codes_by_ti[ti])
        if n_common:
            code_compare_n += n_common
            code_mismatch += int((rr["codes"][:n_common] != arm1_codes_by_ti[ti]).sum())
    print(f"done in {time.time()-ta:.1f}s")
    print(f"字選擇自檢（臂1 vs 臂2，同 task 同 k 應選同一個字，因為 encode 不吃 obs）："
          f"{code_compare_n} 次比較中 {code_mismatch} 次不一致")
    if code_mismatch:
        print("⛔ 字選擇自檢失敗——代表臂1/臂2 的『只有 obs 輸入不同』這個受控變因假設不成立，"
              "拆帳的因果解讀站不住，停手回報。")
        sys.exit(1)

    # ================================================================
    # STEP 4：分析 —— 漂移曲線、拆帳表、失敗前兆
    # ================================================================
    print("\n=== STEP 4：分析 ===")
    xy_drift_by_j_completed = defaultdict(list)
    xy_drift_by_j_failed = defaultdict(list)
    state_drift_by_j_completed = defaultdict(list)
    state_drift_by_j_failed = defaultdict(list)
    n_active_by_j = defaultdict(int)
    for ti, j, completed, xd, sd, al in boundary_rows:
        n_active_by_j[j] += 1
        (xy_drift_by_j_completed if completed else xy_drift_by_j_failed)[j].append(xd)
        (state_drift_by_j_completed if completed else state_drift_by_j_failed)[j].append(sd)

    j_max = int(max(n_active_by_j.keys()))
    xs_all = list(range(0, j_max + 1))
    band_xy_completed = band_stats(xy_drift_by_j_completed, xs_all)
    band_xy_failed = band_stats(xy_drift_by_j_failed, xs_all)
    band_state_completed = band_stats(state_drift_by_j_completed, xs_all)
    band_state_failed = band_stats(state_drift_by_j_failed, xs_all)

    # ---- 自檢 #2：臂1 boundary j=1 vs 臂2 k=0，首 chunk 誤差同量級 ----
    arm1_j1_xy = np.asarray(xy_drift_by_j_completed[1] + xy_drift_by_j_failed[1], dtype=float)
    arm1_j1_state = np.asarray(state_drift_by_j_completed[1] + state_drift_by_j_failed[1], dtype=float)
    arm2_k0_xy = np.asarray(reset_xy_err_by_k[0], dtype=float)
    arm2_k0_state = np.asarray(reset_state_err_by_k[0], dtype=float)
    selfcheck = dict(
        arm1_j1_xy_median=float(np.median(arm1_j1_xy)), arm1_j1_xy_n=int(len(arm1_j1_xy)),
        arm2_k0_xy_median=float(np.median(arm2_k0_xy)), arm2_k0_xy_n=int(len(arm2_k0_xy)),
        ratio_xy=float(np.median(arm1_j1_xy) / max(np.median(arm2_k0_xy), 1e-12)),
        arm1_j1_state_median=float(np.median(arm1_j1_state)),
        arm2_k0_state_median=float(np.median(arm2_k0_state)),
        ratio_state=float(np.median(arm1_j1_state) / max(np.median(arm2_k0_state), 1e-12)),
    )
    print(f"自檢#2（首 chunk 誤差同量級）：xy drift 臂1(j=1) median={selfcheck['arm1_j1_xy_median']:.4f}m "
          f"(n={selfcheck['arm1_j1_xy_n']})  vs 臂2(k=0) median={selfcheck['arm2_k0_xy_median']:.4f}m "
          f"(n={selfcheck['arm2_k0_xy_n']})  比值={selfcheck['ratio_xy']:.3f}")
    print(f"                          state drift 臂1(j=1) median={selfcheck['arm1_j1_state_median']:.4f} "
          f"vs 臂2(k=0) median={selfcheck['arm2_k0_state_median']:.4f}  比值={selfcheck['ratio_state']:.3f}")

    # ---- 拆帳表 ----
    e_indep_xy_overall = float(np.median(np.concatenate(
        [np.asarray(v) for v in reset_xy_err_by_k.values()])))
    e_indep_state_overall = float(np.median(np.concatenate(
        [np.asarray(v) for v in reset_state_err_by_k.values()])))
    print(f"\n拆帳基線 E_indep（臂2 全部 chunk 合併中位數）：xy={e_indep_xy_overall:.4f}m  "
          f"state={e_indep_state_overall:.4f}")
    print(f"\n{'j':>4} {'n_active':>9} {'D_cont_xy_p50':>14} {'E_indep_xy_p50(k=j-1)':>22} "
          f"{'net_xy=Dcont-Eoverall':>22} {'D_cont_state_p50':>18}")
    accounting_rows = []
    for j in J_GRID_PRINT:
        if j > j_max:
            continue
        d_cont_xy = np.asarray(xy_drift_by_j_completed[j] + xy_drift_by_j_failed[j], dtype=float)
        d_cont_state = np.asarray(state_drift_by_j_completed[j] + state_drift_by_j_failed[j], dtype=float)
        if len(d_cont_xy) == 0:
            continue
        e_local = reset_xy_err_by_k.get(j - 1, [])
        e_local_med = float(np.median(e_local)) if e_local else float("nan")
        d_med = float(np.median(d_cont_xy))
        net = d_med - e_indep_xy_overall
        accounting_rows.append(dict(j=j, n_active=int(len(d_cont_xy)), d_cont_xy_p50=d_med,
                                    e_indep_xy_p50_local=e_local_med, net_xy=net,
                                    d_cont_state_p50=float(np.median(d_cont_state))))
        print(f"{j:>4} {len(d_cont_xy):>9} {d_med:>14.4f} {e_local_med:>22.4f} {net:>22.4f} "
              f"{float(np.median(d_cont_state)):>18.4f}")

    # ---- 失敗前兆分佈 ----
    fail_start_drift = np.asarray([r["leg_start_drift"] for r in leg_rows_all if not r["reached"]])
    succ_start_drift = np.asarray([r["leg_start_drift"] for r in leg_rows_all if r["reached"]])
    auc = auc_separation(fail_start_drift, succ_start_drift)
    thr = best_threshold(fail_start_drift, succ_start_drift)
    fail_q = q(fail_start_drift)
    succ_q = q(succ_start_drift)
    # leg-1 專門子群：leg_start_drift 定義上恆為 0，失敗率單獨報一次
    leg1_rows = [r for r in leg_rows_all if r["m"] == 1]
    leg1_fail_rate = float(np.mean([not r["reached"] for r in leg1_rows])) if leg1_rows else None
    overall_fail_rate = float(len(fail_start_drift) / max(len(fail_start_drift) + len(succ_start_drift), 1))

    print(f"\n失敗前兆：失敗 leg n={len(fail_start_drift)}  成功 leg n={len(succ_start_drift)}")
    print(f"  失敗 leg 開始時漂移 p25/50/75 = {fail_q['25']:.3f}/{fail_q['50']:.3f}/{fail_q['75']:.3f}")
    print(f"  成功 leg 開始時漂移 p25/50/75 = {succ_q['25']:.3f}/{succ_q['50']:.3f}/{succ_q['75']:.3f}")
    print(f"  AUC 分離度（0.5=分不開，1.0=完全分開，失敗 leg 漂移系統性更大）= {auc:.3f}")
    print(f"  最佳單一門檻：thr={thr['best_thr']}  accuracy={thr['best_acc']:.3f}  "
          f"多數類 baseline accuracy={thr['majority_baseline_acc']:.3f}")
    print(f"  leg-1 專門子群（leg_start_drift 定義上恆為 0，n={len(leg1_rows)}）："
          f"失敗率={leg1_fail_rate:.3f}  vs 全體失敗率={overall_fail_rate:.3f}")

    # ================================================================
    # STEP 5：存檔（json + npz + 2 張 png）
    # ================================================================
    print("\n=== STEP 5：存檔 ===")
    png_curve = os.path.join(args.out_dir, f"{args.tag}_curve_xy.png")
    draw_percentile_band(
        [dict(label="completed task (schedule)", color_idx=2, xs=band_xy_completed["xs"],
              p25=band_xy_completed["p25"], p50=band_xy_completed["p50"], p75=band_xy_completed["p75"]),
         dict(label="stuck/failed task (schedule)", color_idx=7, xs=band_xy_failed["xs"],
              p25=band_xy_failed["p25"], p50=band_xy_failed["p50"], p75=band_xy_failed["p75"])],
        png_curve,
        "drift-analysis: xy drift vs chunk index (schedule, cont arm)",
        "chunk index j (state right before chunk j starts)", "xy drift vs teacher xy (m)")
    print(f"saved: {png_curve}")

    png_precursor = os.path.join(args.out_dir, f"{args.tag}_failure_precursor.png")
    # xrange 手動縮到 0~5m（不是自動 0.2~99.8 百分位）——兩組都在 0 附近有尖峰（succ 尤其
    # 陡，因為 leg1 定義上漂移恆為 0），縮小範圍是為了看清楚尖峰之後兩條曲線怎麼分岔，
    # 不是為了藏尾巴：完整分位數（含 fail 一路到 p99）已經在 json/summary 的
    # failure_precursor.fail_quantiles 裡逐項留著，沒有因為畫圖砍掉。
    wv.draw_hist_overlay(
        [dict(label=f"failed leg (n={len(fail_start_drift)})", values=fail_start_drift),
         dict(label=f"succeeded leg (n={len(succ_start_drift)})", values=succ_start_drift)],
        png_precursor,
        "drift-analysis: leg-start xy drift, failed vs succeeded legs (schedule)",
        "xy drift at leg start (m); full quantiles incl. tail are in the json, not just this 0-5m window",
        bins=60, xrange=(0.0, 5.0))
    print(f"saved: {png_precursor}")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(
        config={k: v for k, v in vars(args).items()}, delta_sub=DELTA_SUB,
        n_tasks_used=len(tasks), n_chunks_per_task=N_CHUNKS,
        reproduction_check=dict(repro_ti=repro_tis, status="PASS"),
        coarse_check_200=dict(n_tasks=len(tasks), n_mismatch=len(coarse_mismatch)),
        arm2_code_selfcheck=dict(n_compared=code_compare_n, n_mismatch=code_mismatch),
        completion_ratio_arm1=completion_ratio,
        selfcheck_first_chunk=selfcheck,
        e_indep_baseline=dict(xy=e_indep_xy_overall, state=e_indep_state_overall),
        accounting_table=accounting_rows,
        drift_curve=dict(xy_completed=band_xy_completed, xy_failed=band_xy_failed,
                         state_completed=band_state_completed, state_failed=band_state_failed),
        failure_precursor=dict(
            n_fail=int(len(fail_start_drift)), n_succ=int(len(succ_start_drift)),
            fail_quantiles=fail_q, succ_quantiles=succ_q, auc=auc, threshold=thr,
            leg1_fail_rate=leg1_fail_rate, overall_fail_rate=overall_fail_rate),
        artifacts=dict(curve_png=png_curve, precursor_png=png_precursor),
    ), js_path)
    print(f"saved: {js_path}")

    npz_path = os.path.join(args.out_dir, f"{args.tag}_raw.npz")
    b_ti = np.array([r[0] for r in boundary_rows], dtype=np.int32)
    b_j = np.array([r[1] for r in boundary_rows], dtype=np.int32)
    b_completed = np.array([r[2] for r in boundary_rows], dtype=bool)
    b_xy = np.array([r[3] for r in boundary_rows], dtype=np.float64)
    b_state = np.array([r[4] for r in boundary_rows], dtype=np.float64)
    b_leg = np.array([r[5] for r in boundary_rows], dtype=np.int32)
    leg_ti = np.array([r["ti"] for r in leg_rows_all], dtype=np.int32)
    leg_m = np.array([r["m"] for r in leg_rows_all], dtype=np.int32)
    leg_M = np.array([r["M"] for r in leg_rows_all], dtype=np.int32)
    leg_reached = np.array([r["reached"] for r in leg_rows_all], dtype=bool)
    leg_drift = np.array([r["leg_start_drift"] for r in leg_rows_all], dtype=np.float64)
    max_k = max(reset_xy_err_by_k.keys()) + 1
    reset_xy_grid = np.full((len(tasks), max_k), np.nan)
    reset_state_grid = np.full((len(tasks), max_k), np.nan)
    for k, vals in reset_xy_err_by_k.items():
        reset_xy_grid[:len(vals), k] = vals
    for k, vals in reset_state_err_by_k.items():
        reset_state_grid[:len(vals), k] = vals
    np.savez_compressed(npz_path,
                        boundary_ti=b_ti, boundary_j=b_j, boundary_completed=b_completed,
                        boundary_xy_drift=b_xy, boundary_state_drift=b_state,
                        boundary_active_leg=b_leg,
                        leg_ti=leg_ti, leg_m=leg_m, leg_M=leg_M, leg_reached=leg_reached,
                        leg_start_drift=leg_drift,
                        reset_xy_err_grid=reset_xy_grid, reset_state_err_grid=reset_state_grid)
    print(f"saved: {npz_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
