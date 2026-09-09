#!/usr/bin/env python
"""experiments/gen_sft/eval_rewrite_aug.py —— rewrite-v1 步2：鐘形定讞 + 半路起點增強
評測腳本。跟 eval_rewrite.py 同一套機制（100% 沿用 rewrite_common.py /
eval_gen_sft.run_one_generated，逐位元同構），差異只有兩點：

  (a) R_GRID 可由 --r-grid 指定（eval_rewrite.py 寫死 [2,4,8,16,None]）——本工單
      六顆 ckpt 各自只需要一部分 R（原配方 3 顆只要 {8,inf}；增強 3 顆要
      {4,8,16,inf}），不需要每次都跑滿五格。
  (b) 拿掉 eval_rewrite.py 的『管線斷言』硬 sys.exit（那個檢查是驗證『R=∞ 這條路徑
      在 rewrite 機制底下沒被破壞』，用的錨點 BASELINE_ANCHOR=.5875 專屬於 v1 那顆
      ckpt——本檔評的是六顆全新訓練的 ckpt，各自的 R=∞ 本來就不該假設等於 .5875
      （那正是本工單鐘形定讞 §1 要量的訓練抖動本身）。改成純描述性列印，不 assert、
      不 sys.exit；R=∞ 是否『管線可信』改用另一個跟 ckpt 無關的檢查：跟
      eval_gen_sft.run_one_generated() 原封不動 import、同一次初始生成共用，這件事
      本身沒有變過，不需要重新驗證。

⛔ 新檔（v3/aug 系列），不改動 eval_gen_sft.py / gen_sft_common.py / model.py /
   rewrite_common.py / eval_rewrite.py 任何一行。
⛔ CPU-only、CPU-heavy（mujoco 物理 x len(R_GRID) x 200 題 rollout + 重寫時的小
   transformer 重新生成），走 sbatch，不裸跑。
"""
import argparse
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

BASELINE_ANCHOR = 0.5875              # gensft-v1 NOTE §2.4 錨點（僅供描述性參考，非本檔 ckpt 的必然值）
PIPELINE_TOL = 0.01


def r_label(R):
    return "inf" if R is None else str(R)


def parse_r_grid(s):
    toks = [t.strip() for t in s.split(",") if t.strip()]
    Rs = [None if t.lower() == "inf" else int(t) for t in toks]
    labels = [r_label(R) for R in Rs]
    return Rs, labels


def obs_proxy_check(raw, eval_tasks, data_dir, n=5):
    """假說 H 地板檢查（跟 eval_rewrite.py 同款）：sim_obs(u) 這個『活的替身』跟訓練
    語料真正用的 raw["observations"][s0] 本身差多少（零漂移地板）。跟 gensft ckpt
    無關，每次呼叫理論上都應該是同一個結果（同一個 env/資料），這裡重印是為了讓
    本檔的輸出自成一份完整證據，不必回頭翻上一份 NOTE。"""
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
    return diffs


def render_gif_exemplars_rewrite(raw, dict_model, gmodel, eval_tasks, rec, codes_map, R,
                                 rho, gif_dir, data_dir):
    """最佳設定成功/失敗各 1 支（跟 eval_rewrite.py 同款函式，逐行同構）。"""
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
    ap.add_argument("--gensft-ckpt", type=str, required=True)
    ap.add_argument("--dict-ckpt", type=str, default=gc.DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=gc.DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=gc.DD)
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260908)  # 200 題構造 seed，同 gen-sft-v1/eval_rewrite
    ap.add_argument("--rho", type=float, default=gc.RHO)
    ap.add_argument("--r-grid", type=str, default="8,inf", help="逗號分隔，inf=不重寫。例：'4,8,16,inf'")
    ap.add_argument("--tag", type=str, default="rewrite_eval_aug")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    ap.add_argument("--gif-dir", type=str, default=os.path.join(gc.RESULTS_DIR, "gifs_rewrite_aug"))
    ap.add_argument("--render-gifs", action="store_true", help="預設不 render；只有明確要這格的 gif 才開")
    ap.add_argument("--gif-r", type=str, default=None, help="要 render gif 的那個 R（'8'/'inf'/...）；--render-gifs 時必填")
    args = ap.parse_args()
    if args.render_gifs:
        assert args.gif_r is not None, "⛔ --render-gifs 需要搭配 --gif-r 指定要哪一格"

    R_GRID, R_LABELS = parse_r_grid(args.r_grid)

    torch.set_num_threads(8)
    t0 = time.time()
    print(f"=== rewrite eval (aug)  gensft_ckpt={os.path.basename(args.gensft_ckpt)}  "
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
          f"held_out_at_best={ck.get('held_out_at_best')}  "
          f"train_args.corpus={ck.get('train_args', {}).get('corpus')}  "
          f"train_args.seed={ck.get('train_args', {}).get('seed')}")

    full_tasks, skipped = gc.rt.build_tasks(raw, n_traj=200, seed=args.seed, ruler=ruler, rho=args.rho)
    print(f"tasks: {len(full_tasks)} usable (要求>=200)  skipped={len(skipped)}")
    assert len(full_tasks) >= 200, "⛔ 可用軌跡不足 200，停手回報"
    eval_tasks = full_tasks[:args.n_traj]
    Ms = np.array([t["M"] for t in eval_tasks])
    print(f"eval_tasks={len(eval_tasks)}  waypoints M: min={Ms.min()} p50={np.median(Ms):.0f} "
          f"max={Ms.max()}  total_legs={Ms.sum()}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    print()
    obs_proxy_diffs = obs_proxy_check(raw, eval_tasks, args.data_dir, n=5)

    s0_t, wp_t, wpm_t = eg.tasks_to_cond(eval_tasks, raw)
    tg0 = time.time()
    greedy_out = gmodel.generate(s0_t, wp_t, wpm_t, max_new_tokens=eg.MAX_NEW_TOKENS, temperature=0.0)
    initial_codes = [g for g, _ in greedy_out]
    Lg = np.array([len(g) for g in initial_codes])
    print(f"\n初始生成（貪心，全部 R 共用第一個 block）：{len(initial_codes)} 條，長度 "
          f"min/p50/max={Lg.min()}/{np.median(Lg):.0f}/{Lg.max()}  ({time.time()-tg0:.1f}s)")

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

    # ---- R=∞ 描述性參考（不是硬 assert，見檔頭 (b)）----
    if "inf" in R_LABELS:
        inf_per_leg = recs["inf"]["per_leg_untimed_reach"]
        close_to_anchor = abs(inf_per_leg - BASELINE_ANCHOR) <= PIPELINE_TOL
        print(f"\n=== R=∞ 參考（描述性，非硬性判準）：per-leg={inf_per_leg:.4f}  "
              f"v1 錨點={BASELINE_ANCHOR}  差={inf_per_leg-BASELINE_ANCHOR:+.4f}  "
              f"{'落在 v1 錨點 ±.01 內' if close_to_anchor else '落在 v1 錨點 ±.01 外（不是異常——這是本工單要量的訓練抖動本身）'} ===")

    avg_rewrites = {}
    for lbl in R_LABELS:
        rs = runs_by_R[lbl]
        avg_rewrites[lbl] = float(np.mean([r.get("n_rewrites", 0) for r in rs])) if rs else None

    print(f"\n=== 主表（{len(eval_tasks)} 題，貪心解碼）===")
    header = (f"{'R':>6} | {'per-leg':>8} | {'completion':>10} | {'fall(step)':>10} | "
             f"{'adiff_p50':>9} | {'avg_rewrites':>12}")
    print(header)
    print("-" * len(header))
    for lbl in R_LABELS:
        rec = recs[lbl]
        pl = rec["per_leg_untimed_reach"]
        print(f"{lbl:>6} | {pl:8.4f} | {rec['completion_ratio']:10.4f} | "
              f"{rec['fall_rate_step']*100:9.2f}% | {rec['action_diff_q']['50']:9.3f} | "
              f"{avg_rewrites[lbl]:12.2f}")

    best_lbl = max(R_LABELS, key=lambda l: recs[l]["per_leg_untimed_reach"])
    print(f"\n本次 R_grid 內最佳：R={best_lbl}  per-leg={recs[best_lbl]['per_leg_untimed_reach']:.4f}")

    gifs = {}
    if args.render_gifs:
        gif_R = None if args.gif_r.lower() == "inf" else int(args.gif_r)
        gif_lbl = r_label(gif_R)
        assert gif_lbl in recs, f"⛔ --gif-r={args.gif_r} 不在本次 R_grid={R_LABELS} 裡"
        gifs = render_gif_exemplars_rewrite(raw, dict_model, gmodel, eval_tasks, recs[gif_lbl],
                                            initial_codes, gif_R, args.rho, args.gif_dir,
                                            args.data_dir)
    else:
        print("⛔ --render-gifs 未指定：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    gc.wv.save_json(dict(
        config={k: v for k, v in vars(args).items()},
        r_grid=R_LABELS, seed_note=dict(build_tasks_seed=args.seed, generation="temperature=0 貪心，決定性，無需 seed"),
        gensft_ckpt_meta=dict(config=gcfg, best_step=ck.get("best_step"),
                              held_out_at_best=ck.get("held_out_at_best"),
                              train_args=ck.get("train_args")),
        obs_proxy_check=obs_proxy_diffs,
        baseline_anchor_ref=BASELINE_ANCHOR,
        results={lbl: recs[lbl] for lbl in R_LABELS},
        avg_rewrites=avg_rewrites,
        best_in_grid=dict(label=best_lbl, per_leg=recs[best_lbl]["per_leg_untimed_reach"]),
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
