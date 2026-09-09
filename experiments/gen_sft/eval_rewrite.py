#!/usr/bin/env python
"""rewrite-v1 步1：純 eval 掃 R —— 證明『定期重寫譜』（receding horizon）能不能吃掉開環
漂移的損失、把單條執行成績從 .5875 往上推。

機制（design 定死）：沿用 eval_gen_sft.py 的 200 題接力協定（build_tasks(seed=20260908)、
schedule 開環播放、per-leg 判定、模擬步數上限=n_chunks）完全不變；chunk [0,R) 用跟
baseline 完全相同的初始生成字串（同一次 gmodel.generate() 呼叫、同輸入、不重算——保證
R=2/4/8/16/∞ 在第一次重寫邊界之前的行為逐位元相同)。之後每滿 R 個 chunk 且尚未完成，
用『當下模擬 obs（gc.sim_obs(u)，活的替身）+ 尚未到達的路標』重新貪心生成剩餘字串，
取代目前 block，重複到 n_chunks 上限。R=∞ 這個 arm 不重寫，直接呼叫
eval_gen_sft.run_one_generated()（原封不動 import，不是重新實作）——這是驗收①『管線
斷言』的可信度來源：不是『重新寫一份邏輯剛好湊出同樣的數字』，是『同一份程式碼路徑』。

⛔ 新檔，不改動 eval_gen_sft.py / gen_sft_common.py / model.py / rewrite_common.py
   以外任何既有檔案。
⛔ CPU-only、CPU-heavy（mujoco 物理 x 5 個 R x 200 題 rollout + 重寫時的小 transformer
   重新生成），走 sbatch，不裸跑。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402
import eval_gen_sft as eg  # noqa: E402
import rewrite_common as rc  # noqa: E402
from model import GenSFT  # noqa: E402

R_GRID = [2, 4, 8, 16, None]          # None = ∞（baseline，不重寫）
R_LABELS = ["2", "4", "8", "16", "inf"]
BASELINE_ANCHOR = 0.5875              # gensft-v1 NOTE §2.4 錨點（同協定同 200 題）
PIPELINE_TOL = 0.01                   # 驗收①容忍帶（design 給的量級）
CLEAR_WIN_THRESHOLD = 0.66            # 驗收②：任一 R 的 per-leg >= 這個算清楚贏
GAIN_TRIGGER_STEP2 = 0.04             # design 步2 觸發門檻：最好 R 的增益 < 這個就觸發


def r_label(R):
    return "inf" if R is None else str(R)


def obs_proxy_check(raw, eval_tasks, data_dir, n=5):
    """假說 H 要對照的『分佈差』有多大，先量一個地板：即使『沒有任何漂移』（重寫發生在
    chunk 0、剛 reset+set_state 完那一刻），sim_obs(u) 這個『活的替身』跟訓練語料真正
    用的 raw["observations"][s0] 本身差多少？這個數字是『重�regenerate 時餵給 u 的條件
    分佈差』的下界（下游 mid-episode 的漂移只會疊加更多，不會更少)。"""
    env = gc.wv.make_env(data_dir, gc.DATASET)
    u = env.unwrapped
    diffs = []
    for t in eval_tasks[:n]:
        env.reset()
        u.set_state(raw["qpos"][t["s0"]].copy(), raw["qvel"][t["s0"]].copy())
        proxy = gc.sim_obs(u)
        real = raw["observations"][t["s0"]].astype(np.float32)
        d = np.abs(proxy - real)
        diffs.append(dict(episode=t["episode"], l2=float(np.linalg.norm(proxy - real)),
                          max_abs=float(d.max())))
    l2s = [d["l2"] for d in diffs]
    print(f"obs_proxy_check（sim_obs(u) vs raw['observations'][s0]，reset 剛完、零漂移）: "
          f"n={n}  L2 min/p50/max={min(l2s):.6f}/{np.median(l2s):.6f}/{max(l2s):.6f}")
    for d in diffs:
        print(f"    episode={d['episode']}  L2={d['l2']:.6f}  max_abs={d['max_abs']:.6f}")
    return diffs


def render_gif_exemplars_rewrite(raw, dict_model, gmodel, eval_tasks, rec, codes_map, R,
                                 rho, gif_dir, data_dir):
    """最佳設定成功/失敗各 1 支（驗收④，只要 1+1，不是 baseline NOTE 的 2+2)。R=None
    時退回呼叫 eg.run_one_generated；R 有限時呼叫 rc.run_one_rewrite。相機/兩格並排/
    frame_every/max_frames 沿用 eg.render_gif_exemplars 的設定值，不能 import 它本身
    （它綁死呼叫 run_one_generated，沒辦法不改就套用 rewrite 版本)。
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

    per_task = rec["per_task"]
    succ = [p for p in per_task if p["completed"]][:1]
    fail = [p for p in per_task if not p["completed"]][:1]
    if len(succ) < 1:
        print(f"⚠️ R={r_label(R)} 沒有 completed=True 的題目，成功 gif 缺")
    if len(fail) < 1:
        print(f"⚠️ R={r_label(R)} 沒有 completed=False 的題目，失敗 gif 缺")

    made = {}
    for kind, recs_ in (("success", succ), ("fail", fail)):
        for n, p in enumerate(recs_):
            task = eval_tasks[p["ti"]]
            codes0 = codes_map[p["ti"]]
            if R is None:
                r = eg.run_one_generated(env, u, dict_model, raw, task, codes0, rho, render=True,
                                        renderer=renderer, cams=cams, frame_every=4, max_frames=200)
            else:
                r = rc.run_one_rewrite(env, u, dict_model, gmodel, raw, task, codes0, rho, R,
                                       render=True, renderer=renderer, cams=cams,
                                       frame_every=4, max_frames=200)
            assert r["completed"] == p["completed"] and r["n_steps_run"] == p["n_steps_run"], (
                f"⛔ 重跑不一致（應為 bit-exact 決定性）：R={r_label(R)} kind={kind} n={n} "
                f"first_pass=(completed={p['completed']},steps={p['n_steps_run']}) "
                f"second_pass=(completed={r['completed']},steps={r['n_steps_run']})")
            fp = os.path.join(gif_dir, f"R{r_label(R)}_{kind}_{n}_ep{task['episode']}.gif")
            if r["frames"]:
                import imageio.v2 as imageio
                imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
                made[f"{kind}_{n}"] = dict(path=fp, n_frames=len(r["frames"]), episode=task["episode"],
                                           legs_reached=r["legs_reached"], M=task["M"],
                                           n_steps_run=r["n_steps_run"],
                                           n_rewrites=r.get("n_rewrites"))
                print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                      f"legs {r['legs_reached']}/{task['M']}, n_rewrites={r.get('n_rewrites')})")
            else:
                made[f"{kind}_{n}"] = None
                print(f"⛔ render 失敗（0 幀）：{kind}/{n} episode={task['episode']}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gensft-ckpt", type=str,
                    default=os.path.join(gc.RESULTS_DIR, "gensft_v1_best.pt"))
    ap.add_argument("--dict-ckpt", type=str, default=gc.DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=gc.DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=gc.DD)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260908)  # 200 題構造 seed，同 gen-sft-v1
    ap.add_argument("--rho", type=float, default=gc.RHO)
    ap.add_argument("--tag", type=str, default="rewrite_eval")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    ap.add_argument("--gif-dir", type=str, default=os.path.join(gc.RESULTS_DIR, "gifs_rewrite"))
    ap.add_argument("--skip-gifs", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(8)
    t0 = time.time()
    print(f"=== rewrite eval  gensft_ckpt={os.path.basename(args.gensft_ckpt)}  "
          f"n_traj={args.n_traj}  seed={args.seed}  rho={args.rho}  R_grid={R_LABELS} ===")

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
    Ms = np.array([t["M"] for t in eval_tasks])
    print(f"eval_tasks={len(eval_tasks)}  waypoints M: min={Ms.min()} p50={np.median(Ms):.0f} "
          f"max={Ms.max()}  total_legs={Ms.sum()}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # 假說 H 地板：sim_obs(u) 這個『活的替身』跟訓練語料真正用的 raw obs，即使零漂移
    # （reset 剛完那一刻）差多少
    # -----------------------------------------------------------------
    print()
    obs_proxy_diffs = obs_proxy_check(raw, eval_tasks, args.data_dir, n=5)

    # -----------------------------------------------------------------
    # 初始生成：跟 eval_gen_sft.py 的 greedy 完全同一次呼叫（同輸入、同 batch、
    # temperature=0），所有 R（含 ∞）的 chunk[0,R) 共用這個結果
    # -----------------------------------------------------------------
    s0_t, wp_t, wpm_t = eg.tasks_to_cond(eval_tasks, raw)
    tg0 = time.time()
    greedy_out = gmodel.generate(s0_t, wp_t, wpm_t, max_new_tokens=eg.MAX_NEW_TOKENS, temperature=0.0)
    initial_codes = [g for g, _ in greedy_out]
    Lg = np.array([len(g) for g in initial_codes])
    print(f"\n初始生成（貪心，全部 R 共用第一個 block）：{len(initial_codes)} 條，長度 "
          f"min/p50/max={Lg.min()}/{np.median(Lg):.0f}/{Lg.max()}  ({time.time()-tg0:.1f}s)")

    # -----------------------------------------------------------------
    # rollout：5 個 R x 200 題 = 1000 條，同一個 worker pool 一次做完
    # -----------------------------------------------------------------
    work_items = []
    for R in R_GRID:
        lbl = r_label(R)
        for i, t in enumerate(eval_tasks):
            work_items.append(dict(task=t, codes0=initial_codes[i], rho=args.rho, R=R,
                                   R_label=lbl, i=i))
    print(f"\n=== 開始 rollout：{len(R_GRID)} 個 R x {len(eval_tasks)} 題 = "
          f"{len(work_items)} 條 ===")
    raw_out = rc.parallel_walk_rewrite(work_items, args.dict_ckpt, args.gensft_ckpt,
                                       args.data_dir, gc.DATASET, args.workers)

    runs_by_R = {lbl: [None] * len(eval_tasks) for lbl in R_LABELS}
    for o in raw_out:
        runs_by_R[o["R_label"]][o["i"]] = o["r"]

    recs, arrays_by_R = {}, {}
    for lbl in R_LABELS:
        rec, arr = eg.aggregate_runs(eval_tasks, runs_by_R[lbl], fall_line)
        recs[lbl] = rec
        arrays_by_R[lbl] = arr

    # -----------------------------------------------------------------
    # 驗收①：管線斷言 —— R=∞ 必須復現 .5875±.01
    # -----------------------------------------------------------------
    inf_per_leg = recs["inf"]["per_leg_untimed_reach"]
    pipeline_ok = abs(inf_per_leg - BASELINE_ANCHOR) <= PIPELINE_TOL
    print(f"\n=== 驗收①管線斷言：R=∞ per-leg={inf_per_leg:.4f}  錨點={BASELINE_ANCHOR}  "
          f"容忍=±{PIPELINE_TOL}  {'PASS' if pipeline_ok else '⛔⛔⛔ FAIL'} ===")
    if not pipeline_ok:
        print(f"⛔⛔⛔ R=∞ 沒有復現錨點，管線疑似壞了，依 design 停手回報，不做判讀。")
        gc.wv.save_json(dict(pipeline_assertion_pass=False, inf_per_leg=inf_per_leg,
                            anchor=BASELINE_ANCHOR, tol=PIPELINE_TOL,
                            config={k: v for k, v in vars(args).items()}),
                        os.path.join(args.out_dir, f"{args.tag}_PIPELINE_FAIL.json"))
        sys.exit(1)

    # -----------------------------------------------------------------
    # 主表 + 甜蜜點判讀
    # -----------------------------------------------------------------
    avg_rewrites = {}
    for lbl in R_LABELS:
        rs = runs_by_R[lbl]
        avg_rewrites[lbl] = float(np.mean([r.get("n_rewrites", 0) for r in rs])) if rs else None

    print(f"\n=== 主表（{len(eval_tasks)} 題，貪心解碼，錨點 R=∞={BASELINE_ANCHOR}，"
          f"清楚贏門檻>={CLEAR_WIN_THRESHOLD}）===")
    header = (f"{'R':>6} | {'per-leg':>8} | {'completion':>10} | {'fall(step)':>10} | "
             f"{'adiff_p50':>9} | {'avg_rewrites':>12} | {'clear_win':>9}")
    print(header)
    print("-" * len(header))
    for lbl in R_LABELS:
        rec = recs[lbl]
        pl = rec["per_leg_untimed_reach"]
        cw = "PASS" if pl >= CLEAR_WIN_THRESHOLD else ""
        print(f"{lbl:>6} | {pl:8.4f} | {rec['completion_ratio']:10.4f} | "
              f"{rec['fall_rate_step']*100:9.2f}% | {rec['action_diff_q']['50']:9.3f} | "
              f"{avg_rewrites[lbl]:12.2f} | {cw:>9}")

    finite_labels = ["2", "4", "8", "16"]
    best_lbl = max(finite_labels, key=lambda l: recs[l]["per_leg_untimed_reach"])
    best_per_leg = recs[best_lbl]["per_leg_untimed_reach"]
    gain_over_inf = best_per_leg - inf_per_leg
    gain_over_anchor = best_per_leg - BASELINE_ANCHOR
    any_clear_win = any(recs[l]["per_leg_untimed_reach"] >= CLEAR_WIN_THRESHOLD for l in finite_labels)
    step2_triggered = gain_over_anchor < GAIN_TRIGGER_STEP2

    print(f"\n甜蜜點候選：R={best_lbl}  per-leg={best_per_leg:.4f}  "
          f"增益(對 R=∞ 本次量測)={gain_over_inf:+.4f}  增益(對錨點 {BASELINE_ANCHOR})="
          f"{gain_over_anchor:+.4f}")
    print(f"驗收②「任一 R 清楚贏」(>={CLEAR_WIN_THRESHOLD}): "
          f"{'PASS —— ' if any_clear_win else 'FAIL —— 沒有任何 R 清楚贏，'}"
          f"{'R='+best_lbl+' 達標' if any_clear_win else '重寫在本設定下無效（合法答案）'}")
    print(f"步2 觸發判準（最佳 R 對錨點增益 < {GAIN_TRIGGER_STEP2}）: "
          f"{'觸發' if step2_triggered else '不觸發'}")

    # -----------------------------------------------------------------
    # gif：最佳設定（finite R 裡最好的那個）成功/失敗各 1 支
    # -----------------------------------------------------------------
    gifs = {}
    if not args.skip_gifs:
        best_R = None if best_lbl == "inf" else int(best_lbl)
        gifs = render_gif_exemplars_rewrite(raw, dict_model, gmodel, eval_tasks, recs[best_lbl],
                                            initial_codes, best_R, args.rho, args.gif_dir,
                                            args.data_dir)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    gc.wv.save_json(dict(
        config={k: v for k, v in vars(args).items()},
        r_grid=R_LABELS, seed_note=dict(build_tasks_seed=args.seed, generation="temperature=0 貪心，決定性，無需 seed"),
        obs_proxy_check=obs_proxy_diffs,
        pipeline_assertion=dict(pass_=pipeline_ok, inf_per_leg=inf_per_leg,
                                anchor=BASELINE_ANCHOR, tol=PIPELINE_TOL),
        gensft_ckpt_meta=dict(config=gcfg, best_step=ck.get("best_step")),
        results={lbl: recs[lbl] for lbl in R_LABELS},
        avg_rewrites=avg_rewrites,
        sweet_spot=dict(best_R=best_lbl, best_per_leg=best_per_leg, gain_over_inf=gain_over_inf,
                        gain_over_anchor=gain_over_anchor, clear_win_threshold=CLEAR_WIN_THRESHOLD,
                        any_clear_win=any_clear_win, step2_gain_trigger=GAIN_TRIGGER_STEP2,
                        step2_triggered=step2_triggered),
        artifacts=dict(gifs=gifs),
    ), js_path)
    print(f"\nsaved: {js_path}")

    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        **{f"{lbl}_speed": arrays_by_R[lbl]["speed"] for lbl in R_LABELS},
                        **{f"{lbl}_z": arrays_by_R[lbl]["z"] for lbl in R_LABELS},
                        **{f"{lbl}_adiff": arrays_by_R[lbl]["adiff"] for lbl in R_LABELS})
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
