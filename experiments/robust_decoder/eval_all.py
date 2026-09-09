#!/usr/bin/env python
"""驗收標準 #1 sanity、#3 主結果表、#5 gif —— schedule 版教材接力，同一批 200 題，
5 顆並排（原版 reproduction-check + sigma0/0.1/0.35/1.0）。

【在證明什麼、判準是什麼】
證明：噪音 fine-tune 過的 decoder，在同一批 200 條 schedule 版教材接力題目上，
per-leg 到達率有沒有比基線 .5536（experiments/walk_verify/teacher_relay/results/
teacher_relay_summary.json 的 results.schedule.per_leg_untimed_reach）好。
判準（工單定死，逐字照搬）：
  x >= 0.62            -> 清楚贏（win）
  |x - baseline| <= .07 -> 無效（ineffective）
  否則                  -> 劑量/方向錯（dose_or_direction_wrong）
  判準優先序：先查 win（>=0.62），因為 baseline+.07=.6236，.62 技術上落在這個
  區間裡面 0.0036 —— 工單原文用「抖動尺 ±.03~.07 之外」形容 .62 這個門檻，用
  更緊的 ±.03（baseline±.03=[.524,.584]）看才是乾淨地在帶外；本檔忠實照工單
  給的兩個數字實作，win 優先判，這裡把這個邊界處理寫明，不是自己發明的規則。

流程分三段，中間有硬 gate（sanity 對照，工單規則：明顯掉就停手，不准繼續
往噪音檔位跑）：
  A. REPRO CHECK（不是工單要求的驗收項，是本檔自己的可信度前提）：用本檔的
     measure_version 呼叫（trr.measure_version，import 未改一行）重放「原始
     未動過」的 ckpt，per-leg 應該跟 teacher_relay_summary.json 存的 .5536
     位元級一致（同代碼路徑、同資料、同 200 題、同 seed，沒有理由不一致；
     不一致代表這支腳本本身有 bug，不能拿它評分任何一顆 fine-tune）。
  B. sigma=0（對照顆）：驗收 #1 sanity —— 應落在 baseline±.07 內。FAIL 就存
     部分結果、印清楚、exit(1)，不跑 C。
  C. sigma=0.1 / 0.35 / 1.0：三顆噪音檔位，同一組 200 題、同 env、同 tasks()。

⛔ 不改動任何既有檔案；只 import 上一層 walk_verify/teacher_relay 的
   run_teacher_relay（build_tasks / measure_version / run_one，純函式重用）
   與 wv_common。
⛔ CPU-only（MuJoCo 物理 + 小 MLP encode/decode），跑在 CPU sbatch
   （eval_all.sbatch，不掛 --gres）。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.normpath(os.path.join(HERE, "..", "walk_verify"))
TR_DIR = os.path.join(WV_DIR, "teacher_relay")
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402

import wv_common as wv  # noqa: E402 - 既有共用模組，未修改
import run_teacher_relay as trr  # noqa: E402 - 既有腳本當模組 import，未修改一行

RESULTS_DIR = os.path.join(HERE, "results")
GIF_DIR = os.path.join(RESULTS_DIR, "gifs")
GT_SUMMARY_JSON = os.path.join(TR_DIR, "results", "teacher_relay_summary.json")
NOISE_TAGS = ["sigma0", "sigma0.1", "sigma0.35", "sigma1.0"]
WIN_THRESHOLD = 0.62
SANITY_BAND = 0.07


def ckpt_path(tag):
    return os.path.join(RESULTS_DIR, f"ckpt_{tag}.pt")


def judge(x, baseline):
    if x >= WIN_THRESHOLD:
        return "win"
    if abs(x - baseline) <= SANITY_BAND:
        return "ineffective"
    return "dose_or_direction_wrong"


def gate_row(rec):
    tr = rec["timed_reach"]
    return dict(
        per_leg_untimed_reach=rec["per_leg_untimed_reach"],
        completion_ratio=rec["completion_ratio"],
        legs_reached_total=rec["legs_reached_total"], legs_attempted_total=rec["legs_attempted_total"],
        step_speed_p50=rec["step_speed_q"]["50"], step_speed_mean=rec["step_speed_mean"],
        fall_rate_task=rec["fall_rate_task"], fall_rate_step=rec["fall_rate_step"],
        action_diff_p50=(rec["action_diff_q"]["50"] if rec["action_diff_q"] else None),
        codes_active=rec["codes_active"], codes_perplexity=rec["codes_perplexity"],
        reach_1_25N_data=tr["reach_N_data_1.25N"], reach_1_5N_data=tr["reach_N_data_1.5N"])


def print_table(rows_ordered):
    hdr = f"{'tag':<14}{'per_leg':>9}{'completion':>12}{'speed_p50':>11}{'fall_task':>11}{'adiff_p50':>11}  judge"
    print(hdr)
    print("-" * len(hdr))
    for tag, g, j in rows_ordered:
        adiff = f"{g['action_diff_p50']:.3f}" if g["action_diff_p50"] is not None else "n/a"
        print(f"{tag:<14}{g['per_leg_untimed_reach']:>9.4f}{g['completion_ratio']:>12.3f}"
              f"{g['step_speed_p50']:>11.4f}{g['fall_rate_task']*100:>10.2f}%{adiff:>11}  {j}")


def render_success_fail(env, u, model, raw, tasks, per_task, rho, gif_tag):
    """複刻 run_teacher_relay.render_exemplars() 的單版本版（那支函式寫死
    loop VERSIONS=('schedule','nn') 兩版，這裡只要 schedule、只要 1 顆 model，
    不能直接呼叫，改寫等價的精簡版，run_one() 本身原封 import 不改）。"""
    import mujoco
    os.makedirs(GIF_DIR, exist_ok=True)
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
    cam_wide.lookat[0], cam_wide.lookat[1] = trr.MAZE_CENTER
    cam_wide.lookat[2] = 0.0
    cam_wide.distance = 38.0
    cam_wide.elevation = -75
    cam_wide.azimuth = 90
    cams = (cam, cam_wide)

    succ = next((p for p in per_task if p["completed"]), None)
    fail = next((p for p in per_task if not p["completed"]), None)
    made = {}
    for kind, rec_entry in (("success", succ), ("fail", fail)):
        if rec_entry is None:
            made[kind] = None
            print(f"⛔ {gif_tag}/{kind}：200 條裡沒有這種結果，無法產生這支 gif")
            continue
        task = tasks[rec_entry["ti"]]
        r = trr.run_one(env, u, model, raw, task, "schedule", rho, render=True, renderer=renderer,
                        cams=cams, frame_every=4, max_frames=200)
        assert r["completed"] == rec_entry["completed"] and r["n_steps_run"] == rec_entry["n_steps_run"], (
            f"⛔ 重跑不一致（應為 bit-exact 決定性）：{gif_tag}/{kind} "
            f"first_pass=(completed={rec_entry['completed']},steps={rec_entry['n_steps_run']}) "
            f"second_pass=(completed={r['completed']},steps={r['n_steps_run']})")
        fp = os.path.join(GIF_DIR, f"{gif_tag}_schedule_{kind}_ep{task['episode']}.gif")
        if r["frames"]:
            import imageio.v2 as imageio
            imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
            made[kind] = dict(path=fp, n_frames=len(r["frames"]), episode=task["episode"],
                              legs_reached=r["legs_reached"], M=task["M"], n_steps_run=r["n_steps_run"])
            print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                  f"legs {r['legs_reached']}/{task['M']})")
        else:
            made[kind] = None
            print(f"⛔ render 失敗（0 幀）：{gif_tag}/{kind} episode={task['episode']}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=trr.RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-gifs", action="store_true")
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "eval_summary.json"))
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, args.threads))
    rho = args.rho

    t0 = time.time()
    print(f"=== eval_all（schedule, 200 題, seed={args.seed}, rho={rho}）===")

    # ---- baseline 一手來源自檢 ----
    gt = json.load(open(GT_SUMMARY_JSON))
    baseline = gt["results"]["schedule"]["per_leg_untimed_reach"]
    assert abs(baseline - 0.5535714285714286) < 1e-9, (
        f"⛔ teacher_relay_summary.json 的 baseline 跟工單引用的 .554 對不上：{baseline}")
    print(f"baseline（schedule, 一手來源 {GT_SUMMARY_JSON}）= {baseline:.10f}")
    print(f"判準：win>={WIN_THRESHOLD}  ineffective=baseline±{SANITY_BAND}=[{baseline-SANITY_BAND:.4f},"
          f"{baseline+SANITY_BAND:.4f}]  否則=dose_or_direction_wrong")
    print(f"對照行（一手引用自 teacher_relay COMPARE 常數）：{trr.COMPARE}")

    ruler = json.load(open(trr.DEFAULT_RULER))
    fall_line = ruler["torso_z"]["fall_line_p1"]
    raw = wv.load_npz(trr.DD, trr.DATASET, "val")
    tasks, skipped = trr.build_tasks(raw, args.n_traj, args.seed, ruler, rho)
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報"
    print(f"tasks: {len(tasks)} usable, skipped={len(skipped)}")

    env = wv.make_env(trr.DD, trr.DATASET)
    u = env.unwrapped

    rows = []          # (tag, gate_row, judge_label) 給印表
    all_results = {}   # tag -> full rec (存 json 用，去掉 per_task_full 之類大欄位保持精簡)
    model_cache = {}   # tag -> (model_obj, cfg)  給後面 gif 用

    # ================= STEP A: REPRO CHECK（原始未動 ckpt）=================
    print("\n=== STEP A：REPRO CHECK（原始 ckpt，本檔 harness 可信度前提）===")
    orig_model, orig_cfg = wv.load_dict_v1(trr.DEFAULT_CKPT)
    rec0, _ = trr.measure_version(env, u, orig_model, raw, tasks, ruler, "schedule", rho, fall_line)
    diff0 = abs(rec0["per_leg_untimed_reach"] - baseline)
    repro_ok = diff0 < 1e-9
    print(f"my harness on ORIGINAL ckpt: per_leg={rec0['per_leg_untimed_reach']:.10f} "
          f"vs stored {baseline:.10f}  diff={diff0:.2e}  -> {'PASS' if repro_ok else 'FAIL'}")
    if not repro_ok:
        print("⛔ REPRO CHECK 失敗：本檔重放原始 ckpt 都對不上已存檔的 .554，harness 本身有 bug，"
              "停手回報，不評分任何 fine-tune 顆。")
        wv.save_json(dict(status="REPRO_CHECK_FAILED", baseline=baseline,
                          repro_check=dict(per_leg=rec0["per_leg_untimed_reach"], diff=diff0)), args.out)
        sys.exit(1)

    # ================= STEP B: sigma=0（sanity gate）=================
    print("\n=== STEP B：sigma=0（對照顆）—— 驗收 #1 sanity ===")
    tag0 = "sigma0"
    if not os.path.exists(ckpt_path(tag0)):
        print(f"⛔ 找不到 {ckpt_path(tag0)}，這顆還沒訓練完，停手回報。")
        sys.exit(1)
    m0, cfg0 = wv.load_dict_v1(ckpt_path(tag0))
    model_cache[tag0] = (m0, cfg0)
    rec, ra = trr.measure_version(env, u, m0, raw, tasks, ruler, "schedule", rho, fall_line)
    all_results[tag0] = rec
    x0 = rec["per_leg_untimed_reach"]
    j0 = judge(x0, baseline)
    sanity_pass = abs(x0 - baseline) <= SANITY_BAND
    g0 = gate_row(rec)
    rows.append((tag0, g0, j0))
    print(f"sigma=0 per_leg={x0:.4f}  |Δ from baseline|={abs(x0-baseline):.4f}  "
          f"sanity(<= {SANITY_BAND})={'PASS' if sanity_pass else 'FAIL'}  judge={j0}")
    if not sanity_pass:
        print("⛔ 驗收 #1 sanity 失敗：fine-tune 管線本身傷分（不是噪音的問題，是管線本身），"
              "依工單規則停手，不繼續往噪音檔位跑。")
        print_table(rows)
        wv.save_json(dict(status="SANITY_FAILED", baseline=baseline, thresholds=dict(
            win=WIN_THRESHOLD, sanity_band=SANITY_BAND),
            repro_check=dict(per_leg=rec0["per_leg_untimed_reach"], diff=diff0, status="PASS"),
            results={tag0: gate_row(rec)}, judge={tag0: j0}), args.out)
        sys.exit(1)

    # ================= STEP C: sigma>0 三檔 =================
    print("\n=== STEP C：sigma=0.1 / 0.35 / 1.0 ===")
    for tag in ["sigma0.1", "sigma0.35", "sigma1.0"]:
        if not os.path.exists(ckpt_path(tag)):
            print(f"⛔ 找不到 {ckpt_path(tag)}，這顆還沒訓練完，停手回報（不用預設值填）。")
            sys.exit(1)
        m, cfgm = wv.load_dict_v1(ckpt_path(tag))
        model_cache[tag] = (m, cfgm)
        rec, ra = trr.measure_version(env, u, m, raw, tasks, ruler, "schedule", rho, fall_line)
        all_results[tag] = rec
        x = rec["per_leg_untimed_reach"]
        j = judge(x, baseline)
        rows.append((tag, gate_row(rec), j))
        print(f"[{tag}] per_leg={x:.4f}  completion={rec['completion_ratio']:.3f}  judge={j}")

    print("\n=== 主結果表（schedule, 200 題, 四關）===")
    print(f"{'baseline(orig ckpt)':<14}{baseline:>9.4f}   .125*      (見一手 json，本檔沒重列四關)")
    print_table(rows)
    print("* baseline completion 取自 teacher_relay_summary.json（非本檔重算）")

    best_tag = max(all_results, key=lambda t: all_results[t]["per_leg_untimed_reach"])
    nonzero_tags = [t for t in all_results if t != tag0]
    best_nonzero_tag = max(nonzero_tags, key=lambda t: all_results[t]["per_leg_untimed_reach"])
    gif_tag = best_tag if best_tag != tag0 else best_nonzero_tag
    note = ("best_tag 本身就是 sigma0（噪音沒有比對照組好），drift 加測改用 best_nonzero_tag 對照，"
            "已標明是替代選擇" if best_tag == tag0 else "best_tag 是噪音變體，drift 加測直接用它")
    print(f"\nbest_tag(全體 argmax)={best_tag}  best_nonzero_tag={best_nonzero_tag}  "
          f"gif/drift 用的 tag={gif_tag}  ({note})")

    # ================= gif（最好顆，schedule，成功+失敗各 1）=================
    gifs = {}
    if not args.skip_gifs:
        print(f"\n=== gif：{gif_tag}（schedule）成功/失敗各 1 ===")
        gm, _ = model_cache[gif_tag]
        gifs = render_success_fail(env, u, gm, raw, tasks, all_results[gif_tag]["per_task"], rho, gif_tag)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    # ================= 存檔 =================
    table_rows = [dict(tag=tag0, **rows[0][1], judge=rows[0][2])] + \
                 [dict(tag=t, **g, judge=j) for t, g, j in rows[1:]]
    out = dict(
        status="OK", baseline=baseline, baseline_source=GT_SUMMARY_JSON,
        thresholds=dict(win=WIN_THRESHOLD, sanity_band=SANITY_BAND),
        compare_reference=trr.COMPARE,
        config=dict(n_traj=args.n_traj, seed=args.seed, rho=rho, n_tasks_used=len(tasks),
                   skipped=skipped),
        repro_check=dict(per_leg=rec0["per_leg_untimed_reach"], diff=diff0, status="PASS"),
        sanity_check=dict(tag=tag0, per_leg=x0, diff=abs(x0 - baseline), band=SANITY_BAND, status="PASS"),
        table_rows=table_rows,
        best_tag=best_tag, best_nonzero_tag=best_nonzero_tag, gif_tag=gif_tag, gif_tag_note=note,
        gifs=gifs,
        wall_seconds=time.time() - t0,
    )
    wv.save_json(out, args.out)
    print(f"\nsaved: {args.out}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
