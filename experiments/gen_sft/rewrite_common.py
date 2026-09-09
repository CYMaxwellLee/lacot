"""experiments/gen_sft/rewrite_common.py —— receding-horizon 重寫機制本體。

工單問題：u 一次寫完整段字串、開環播放，位置漂移隨時間自我放大（第 10 段已是量化基線
27 倍）。藥＝執行到一半用真實狀態重寫。本檔實作『定期重寫譜』：每滿 R 個 chunk 就
用『當下模擬 obs（漂移後的真實姿態）＋尚未到達的路標』重新貪心生成剩餘字串。

⛔ 新檔，不改動 eval_gen_sft.py / gen_sft_common.py / model.py 任何一行。
`run_one_rewrite()` 是仿 eval_gen_sft.run_one_generated() 改寫（同構、非 import 復用，
理由跟 eval_gen_sft.py 檔頭講的一樣：那支函式的迴圈主體要插入『滿 R 步就重寫』的
分支，沒辦法不改一行就套用）——物理/leg 判定/clip 統計/render 逐行對齊，唯一差異
在『codes 從哪個 block 來、多久換一次 block』。

R=∞（不重寫）本檔不重新實作：正式量表的 R=∞ arm 直接呼叫
eval_gen_sft.run_one_generated()（原封不動 import），bit-exact 保證見
docs/NOTE-2026-09-09-rewrite-receding.md §一管線斷言。本檔的 run_one_rewrite 只服務
R∈{2,4,8,16} 這四個真正會觸發重寫的 arm，以及 gif 例外情境（見 eval_rewrite.py）。

重寫時的條件建構（呼應 design 假說 H）：s0_obs 用 gc.sim_obs(u)（qpos[:15]+qvel[:14],
29 維)——這是『活的』替身，語意上等同 raw["observations"]（eval_gen_sft.py 的
run_one_generated 本來就已經用同一個函式當 decode_from_idx 的 obs 條件，見其 §1.3
的 dict_cont 對照邏輯），但數值上是『走到一半、帶漂移』的姿態，不是訓練語料裡『教材
軌跡的乾淨開頭』——這正是假說 H 要對照的分佈差。剩餘路標＝wp_xy[active_leg-1:]
（1-indexed active_leg：leg 1..active_leg-1 已到達，剩下 active_leg..M 尚未到達）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402
import eval_gen_sft as eg  # noqa: E402  -- run_one_generated / aggregate_runs / tasks_to_cond 等原封不動重用
from model import GenSFT  # noqa: E402


def regenerate_codes(gmodel, u, remaining_wp, max_new_tokens):
    """重寫邊界：用『當下模擬 obs + 尚未到達的路標』貪心（temperature=0，決定性）重新生成
    剩餘字串。單樣本 batch（B=1），wp_mask 全 True（沒有 padding 需求）。
    """
    s0_obs = gc.sim_obs(u)  # (29,) float32 —— 活的替身，見檔頭說明
    s0_t = torch.from_numpy(s0_obs[None, :])
    wp_t = torch.from_numpy(np.asarray(remaining_wp, dtype=np.float32)[None, :, :])
    wp_mask = torch.ones(1, wp_t.shape[1], dtype=torch.bool)
    out = gmodel.generate(s0_t, wp_t, wp_mask, max_new_tokens=max_new_tokens, temperature=0.0)
    codes, _ended_eos = out[0]
    return codes


def run_one_rewrite(env, u, dict_model, gmodel, raw, task, initial_codes, rho, R,
                    max_new_tokens=eg.MAX_NEW_TOKENS, render=False, renderer=None,
                    cams=None, frame_every=4, max_frames=200):
    """走一條 task，每滿 R 個 chunk（且尚未完成）重寫一次剩餘字串。R 必須是正整數
    （R=∞ 不走這條路徑，見檔頭）。chunk [0,R) 一律用 initial_codes（跟 baseline/其他 R
    共用同一次『整段初始生成』的輸出，保證第一次重寫邊界之前，所有 R 的行為完全相同，
    只有邊界之後才分岔——這是本次 R 掃描彼此可比的結構保證，不是巧合)。

    逐行對齊 eg.run_one_generated：物理 step／leg 判定／clip 統計／render 完全同款，
    差異只在『codes 現在指向哪個 block、block 何時換』。
    """
    assert R is not None and R >= 1, f"⛔ run_one_rewrite 只服務有限 R，收到 R={R}"
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
    ran_out = False
    n_rewrites = 0
    rewrite_at_chunk = []
    codes = initial_codes
    block_start = 0
    if render:
        u.set_goal(goal_xy=np.asarray(wp_xy[0], dtype=np.float64))

    prev_xy = raw["qpos"][s0, :2].copy()
    for k in range(n_chunks):
        if (k - block_start) == R:
            # 重寫邊界：active_leg<=M 保證成立（若 active_leg>M 上一輪已經 break 出迴圈）
            remaining_wp = wp_xy[active_leg - 1:]
            codes = regenerate_codes(gmodel, u, remaining_wp, max_new_tokens)
            block_start = k
            n_rewrites += 1
            rewrite_at_chunk.append(k)

        local_idx = k - block_start
        if local_idx >= len(codes):
            ran_out = True
            break
        code_id = int(codes[local_idx])
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
               n_codes_available=int(len(codes)), ran_out_of_codes=bool(ran_out),
               n_rewrites=int(n_rewrites), rewrite_at_chunk=rewrite_at_chunk)


# ---------------------------------------------------------------------
# multiprocessing worker：每個 worker 各自建一次 env/dict_model/gmodel/raw，重複使用。
# 獨立於 eval_gen_sft.py 的 _W/_worker_init/_worker_run（各自 mp.Pool 行程空間不共用，
# 這裡另開一份全域字典只是避免同檔案內名稱混淆，不是真的有 collision 風險)。
# ---------------------------------------------------------------------
_WR = {}


def _worker_init_rewrite(dict_ckpt, gensft_ckpt, data_dir, dataset):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    torch.set_num_threads(1)
    import gen_sft_common as gc2
    dict_model, _ = gc2.load_dict_model(dict_ckpt)
    raw = gc2.wv.load_npz(data_dir, dataset, "val")
    env = gc2.wv.make_env(data_dir, dataset)
    ck = torch.load(gensft_ckpt, map_location="cpu", weights_only=False)
    gmodel = GenSFT(**ck["config"])
    gmodel.load_state_dict(ck["state_dict"])
    gmodel.eval()
    _WR["dict_model"] = dict_model
    _WR["raw"] = raw
    _WR["env"] = env
    _WR["u"] = env.unwrapped
    _WR["gmodel"] = gmodel


def _worker_run_rewrite(item):
    task, codes0, rho, R = item["task"], item["codes0"], item["rho"], item["R"]
    if R is None:
        r = eg.run_one_generated(_WR["env"], _WR["u"], _WR["dict_model"], _WR["raw"], task, codes0, rho)
    else:
        r = run_one_rewrite(_WR["env"], _WR["u"], _WR["dict_model"], _WR["gmodel"], _WR["raw"],
                            task, codes0, rho, R)
    r.pop("frames", None)
    return dict(R_label=item["R_label"], i=item["i"], r=r)


def parallel_walk_rewrite(work_items, dict_ckpt, gensft_ckpt, data_dir, dataset, n_workers):
    import time
    import multiprocessing as mp
    t0 = time.time()
    if n_workers <= 1:
        _worker_init_rewrite(dict_ckpt, gensft_ckpt, data_dir, dataset)
        out = [_worker_run_rewrite(it) for it in work_items]
    else:
        with mp.Pool(processes=n_workers, initializer=_worker_init_rewrite,
                    initargs=(dict_ckpt, gensft_ckpt, data_dir, dataset)) as pool:
            out = pool.map(_worker_run_rewrite, work_items, chunksize=4)
    print(f"  parallel_walk_rewrite: {len(work_items)} rollouts, {n_workers} workers, "
          f"{time.time()-t0:.1f}s")
    return out
