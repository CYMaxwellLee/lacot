#!/usr/bin/env python
"""experiments/gen_sft/pose_permute_probe.py —— 姿態置換探針（主人 2026-09-09 追加驗收）。

在證明什麼、判準是什麼：證明「u 的鐘形/重寫收益，是不是其實只靠 waypoint 條件撐起來、
u 根本沒在用姿態（obs）token」。判準（主人原話，非精確數字門檻——見下方 verdict 的
明標）：掉幅大＝u 有在用姿態；幾乎不掉＝u 忽略姿態 token，半路重寫『餵當下真實姿態』
這件事本身不帶資訊增益，鐘形/增強的解讀都要重新想成純 waypoint 驅動。

探針設計：對指定的一顆 ckpt，取 N 題（20~50，不用全 200，來自標準 200 題的固定前綴，
build_tasks(seed=20260908) 同一個母體）。固定 seed 亂配對出一個 derangement
partner(i)（i 自己配不到自己）。『置換』只動 u 的『姿態條件輸入』——初始
gmodel.generate() 的 s0、以及 R 重寫時 regenerate 的姿態輸入——一律換成 partner(i)
的『真實資料 s0 obs』（raw["observations"][tasks[partner(i)]["s0"]]，固定向量，不隨
模擬時間變化）。其餘全部不變：物理模擬仍從 task i 自己的真實 s0 起跑、waypoint 目標
仍是 task i 自己的 wp_xy、leg 判定仍用真實模擬 xy、字典 decode_from_idx 的 obs 條件
（o_in）仍用真實 gc.sim_obs(u)（探針只動『u 決定寫哪個字』這一步，不動『那個字怎麼
被解碼成動作』——這是精確隔離「u 用不用姿態」跟「decoder 用不用姿態」兩件事的關鍵）。

⛔ 新檔（追加驗收，主人點名）；不改動 eval_gen_sft.py / gen_sft_common.py / model.py /
   rewrite_common.py / eval_rewrite.py / eval_rewrite_aug.py 任何一行。run_one_rewrite
   的重寫迴圈在本檔內仿寫一份加姿態覆寫分支（跟 rewrite_common.py 自己「同構非 import
   復用」的既有慣例一致——原函式沒有掛鉤可以不改一行插入這個分支)。
⛔ CPU-only，走 sbatch。
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

PAIR_SEED = 20260909314  # 固定亂配對 seed


def make_derangement(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    for i in fixed:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    assert np.all(perm != np.arange(n)), "⛔ derangement 構造失敗，仍有 fixed point"
    assert len(set(perm.tolist())) == n, "⛔ 不是排列"
    return perm


def run_one_rewrite_posepermuted(env, u, dict_model, gmodel, raw, task, initial_codes, rho, R,
                                 pose_override, max_new_tokens=eg.MAX_NEW_TOKENS):
    """跟 rewrite_common.run_one_rewrite 逐行同構，唯一差異：regenerate 時餵給 u 的姿態
    條件用固定的 pose_override（partner 的真實 obs），不是 gc.sim_obs(u)；decode_from_idx
    的 obs 條件（o_in）不動，仍是真實 gc.sim_obs(u)——探針只動『u 決定寫哪個字』。"""
    assert R is not None and R >= 1
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)

    env.reset()
    u.set_state(raw["qpos"][s0].copy(), raw["qvel"][s0].copy())

    active_leg = 1
    leg_reach_step = {}
    global_step = 0
    n_rewrites = 0
    codes = initial_codes
    block_start = 0
    ran_out = False

    for k in range(n_chunks):
        if (k - block_start) == R:
            remaining_wp = wp_xy[active_leg - 1:]
            s0_perm_t = torch.from_numpy(pose_override[None, :].astype(np.float32))
            wp_t = torch.from_numpy(np.asarray(remaining_wp, dtype=np.float32)[None, :, :])
            wp_mask = torch.ones(1, wp_t.shape[1], dtype=torch.bool)
            out = gmodel.generate(s0_perm_t, wp_t, wp_mask, max_new_tokens=max_new_tokens, temperature=0.0)
            codes, _ = out[0]
            block_start = k
            n_rewrites += 1

        local_idx = k - block_start
        if local_idx >= len(codes):
            ran_out = True
            break
        code_id = int(codes[local_idx])
        with torch.no_grad():
            idx = torch.tensor([code_id], dtype=torch.long)
            o_in = torch.from_numpy(gc.sim_obs(u)[None, :])  # ⛔ 真實 obs，不置換（見檔頭）
            raw_a = dict_model.decode_from_idx(idx, o_in).numpy().reshape(4, gc.wv.ACT_DIM)
        a_chunk = np.clip(raw_a, lo, hi).astype(np.float64)

        for t in range(4):
            env.step(a_chunk[t])
            cur = np.asarray(u.data.qpos)[:2].copy()
            while active_leg <= M and np.linalg.norm(cur - wp_xy[active_leg - 1]) <= rho:
                leg_reach_step[active_leg] = global_step
                active_leg += 1
            global_step += 1
        if active_leg > M:
            break

    completed = active_leg > M
    return dict(completed=completed, legs_attempted=min(active_leg, M),
               legs_reached=active_leg - 1, n_steps_run=global_step,
               n_rewrites=n_rewrites, ran_out_of_codes=bool(ran_out))


def aggregate_simple(results):
    legs_attempted_total = sum(r["legs_attempted"] for r in results)
    legs_reached_total = sum(r["legs_reached"] for r in results)
    per_leg = legs_reached_total / max(legs_attempted_total, 1)
    completion = sum(int(r["completed"]) for r in results) / len(results)
    return dict(per_leg_untimed_reach=per_leg, completion_ratio=completion,
               legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
               n_tasks=len(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gensft-ckpt", type=str, required=True)
    ap.add_argument("--dict-ckpt", type=str, default=gc.DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=gc.DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=gc.DD)
    ap.add_argument("--n-probe", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--rho", type=float, default=gc.RHO)
    ap.add_argument("--r-rewrite", type=int, default=8)
    ap.add_argument("--tag", type=str, default="pose_permute")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    args = ap.parse_args()

    torch.set_num_threads(8)
    t0 = time.time()
    print(f"=== pose permute probe  ckpt={os.path.basename(args.gensft_ckpt)}  n_probe={args.n_probe}  "
          f"seed={args.seed}  pair_seed={PAIR_SEED}  R={args.r_rewrite} ===")

    raw = gc.wv.load_npz(args.data_dir, gc.DATASET, "val")
    ruler = gc.load_ruler(args.ruler)
    dict_model, dict_cfg = gc.load_dict_model(args.dict_ckpt)
    ck = torch.load(args.gensft_ckpt, map_location="cpu", weights_only=False)
    gmodel = GenSFT(**ck["config"])
    gmodel.load_state_dict(ck["state_dict"])
    gmodel.eval()
    print(f"gensft model best_step={ck.get('best_step')}  train_args.corpus={ck.get('train_args', {}).get('corpus')}  "
          f"train_args.seed={ck.get('train_args', {}).get('seed')}")

    full_tasks, skipped = gc.rt.build_tasks(raw, n_traj=200, seed=args.seed, ruler=ruler, rho=args.rho)
    tasks = full_tasks[:args.n_probe]
    print(f"probe tasks: {len(tasks)}（來自標準 200 題子集，同一個 build_tasks(seed={args.seed})）")

    partner_idx = make_derangement(len(tasks), PAIR_SEED)
    print(f"derangement pairing (fixed seed={PAIR_SEED})，前 5 對：" +
          ", ".join(f"{i}->{int(partner_idx[i])}" for i in range(min(5, len(tasks)))))

    env = gc.wv.make_env(args.data_dir, gc.DATASET)
    u = env.unwrapped

    # ---- baseline（未置換，跟主 eval 同款函式呼叫，證明本檔重跑的是同一個機制）----
    s0_t, wp_t, wpm_t = eg.tasks_to_cond(tasks, raw)
    codes0_base = [g for g, _ in gmodel.generate(s0_t, wp_t, wpm_t, max_new_tokens=eg.MAX_NEW_TOKENS, temperature=0.0)]

    base_inf = [eg.run_one_generated(env, u, dict_model, raw, tasks[i], codes0_base[i], args.rho)
               for i in range(len(tasks))]
    base_r = [rc.run_one_rewrite(env, u, dict_model, gmodel, raw, tasks[i], codes0_base[i], args.rho, args.r_rewrite)
             for i in range(len(tasks))]

    # ---- 置換（u 的姿態條件輸入換成 partner 的真實 obs；decode 條件不動）----
    partner_s0_idx = np.array([tasks[int(partner_idx[i])]["s0"] for i in range(len(tasks))])
    permuted_obs = raw["observations"][partner_s0_idx].astype(np.float32)
    s0_perm_t = torch.from_numpy(permuted_obs)
    assert s0_perm_t.shape == s0_t.shape
    codes0_perm = [g for g, _ in gmodel.generate(s0_perm_t, wp_t, wpm_t, max_new_tokens=eg.MAX_NEW_TOKENS, temperature=0.0)]

    perm_inf = [eg.run_one_generated(env, u, dict_model, raw, tasks[i], codes0_perm[i], args.rho)
               for i in range(len(tasks))]
    perm_r = [run_one_rewrite_posepermuted(env, u, dict_model, gmodel, raw, tasks[i], codes0_perm[i], args.rho,
                                           args.r_rewrite, permuted_obs[i]) for i in range(len(tasks))]

    rec_base_inf = aggregate_simple(base_inf)
    rec_base_r = aggregate_simple(base_r)
    rec_perm_inf = aggregate_simple(perm_inf)
    rec_perm_r = aggregate_simple(perm_r)

    print(f"\n=== 結果（n={len(tasks)} 題子集，ckpt={os.path.basename(args.gensft_ckpt)}）===")
    print(f"{'arm':>20} | {'R=inf per-leg':>14} | {'R='+str(args.r_rewrite)+' per-leg':>14}")
    print(f"{'baseline(真姿態)':>20} | {rec_base_inf['per_leg_untimed_reach']:14.4f} | {rec_base_r['per_leg_untimed_reach']:14.4f}")
    print(f"{'permuted(換別題姿態)':>20} | {rec_perm_inf['per_leg_untimed_reach']:14.4f} | {rec_perm_r['per_leg_untimed_reach']:14.4f}")
    drop_inf = rec_base_inf['per_leg_untimed_reach'] - rec_perm_inf['per_leg_untimed_reach']
    drop_r = rec_base_r['per_leg_untimed_reach'] - rec_perm_r['per_leg_untimed_reach']
    print(f"\n掉幅：R=inf {drop_inf:+.4f}   R={args.r_rewrite} {drop_r:+.4f}")
    big_drop_threshold = 0.10  # ⛔ 拍的，本探針自訂門檻，非工單既有判準，明標於此
    verdict = ("u 有在用姿態 token（掉幅明顯 >= 自訂門檻 .10）"
              if max(drop_inf, drop_r) >= big_drop_threshold else
              "u 幾乎沒在用姿態 token（掉幅 < 自訂門檻 .10，重寫收益的解讀要重新想成偏向純 waypoint 驅動）")
    print(f"判讀（門檻={big_drop_threshold}，⛔ 本探針自訂、非工單既有精確判準，質性判讀以掉幅數字本身為準）：{verdict}")

    os.makedirs(args.out_dir, exist_ok=True)
    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    gc.wv.save_json(dict(
        config={k: v for k, v in vars(args).items()},
        pair_seed=PAIR_SEED, n_probe=len(tasks),
        gensft_ckpt_meta=dict(best_step=ck.get("best_step"), train_args=ck.get("train_args")),
        baseline=dict(inf=rec_base_inf, r=rec_base_r),
        permuted=dict(inf=rec_perm_inf, r=rec_perm_r),
        drop=dict(inf=drop_inf, r=drop_r),
        big_drop_threshold_self_set=big_drop_threshold,
        verdict=verdict,
    ), js_path)
    print(f"\nsaved: {js_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
