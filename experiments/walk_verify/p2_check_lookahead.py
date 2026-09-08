#!/usr/bin/env python
"""P2 正確性檢查：前瞻預測 vs 實走結果一致嗎？（CPU-only，不碰 GPU）

vq_oracle 的整個主張建立在「前瞻算出的終點＝實際走完的終點」。若 rewind 沒還原乾淨，
它挑的字就不是它以為的那個 —— 而這種錯【不會報錯】，成績照樣算得出來。
⇒ 這支直接量：32 個字各前瞻一次、還原、再用 wrapped env.step 實走選中的字，
   比對實走終點 vs 前瞻預測終點。
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "/home/cymaxwelllee/Projects/lacot/experiments/walk_verify")
sys.path.insert(0, "/home/cymaxwelllee/Projects/lacot-vqoracle/experiments")
import numpy as np, torch
import wv_common as wv
import lacot_vqo as vqo

DD = "/home/cymaxwelllee/.ogbench/data"; DS = "antmaze-medium-stitch-v0"
CK = "/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/p0_dict_v1_L4K32_50k.pt"
env = wv.make_env(DD, DS); u = env.unwrapped
pol = vqo.VQOraclePolicy(env, CK, 4, 8, verbose=True)
raw = wv.load_npz(DD, DS, "val")
rng = np.random.default_rng(3)
starts = rng.choice(len(raw["qpos"]) - 10, 200, replace=False)
errs, chosen, nsteps_ok = [], [], 0
for s in starts:
    env.reset(); u.set_state(raw["qpos"][s].copy(), raw["qvel"][s].copy())
    obs = np.concatenate([np.asarray(u.data.qpos)[:15], np.asarray(u.data.qvel)[:14]])
    w = np.asarray(u.data.qpos)[:2] + rng.normal(0, 2.0, 2)          # 假路標
    # 先自己算一次「若選 k 會走到哪」（前瞻），記下每個 k 的終點
    st = vqo.save_state(u)
    with torch.no_grad():
        cand = pol.model.decode_all_codes(torch.as_tensor(obs[None, :], dtype=torch.float32))
    cand = np.clip(cand.numpy().reshape(32, 4, 8), -1, 1).astype(np.float64)
    ends = []
    for k in range(32):
        vqo.restore_state(u, st)
        for t in range(4): u.step(cand[k, t])
        ends.append(np.asarray(u.data.qpos)[:2].copy())
    vqo.restore_state(u, st)
    best = int(np.argmin([np.linalg.norm(e - w) for e in ends]))
    # 再走一次 policy（它內部也會前瞻＋還原），拿它回傳的動作用 wrapped env.step 實走
    a = pol(obs, w)
    for t in range(4):
        env.step(a[t].astype(np.float64)); nsteps_ok += 1
    real_end = np.asarray(u.data.qpos)[:2].copy()
    errs.append(float(np.linalg.norm(real_end - ends[best])))
    chosen.append(best)
errs = np.asarray(errs)
print(f"\nn={len(errs)}  實走終點 vs 前瞻預測終點 的距離（公尺）")
print(f"  max={errs.max():.3e}  p99={np.percentile(errs,99):.3e}  p50={np.percentile(errs,50):.3e}")
print(f"  超過 1e-9 的比例 {np.mean(errs>1e-9)*100:.1f}%   超過 1e-6 的比例 {np.mean(errs>1e-6)*100:.1f}%")
print(f"選到的字分佈：{len(set(chosen))} 種不同的字（共 {len(chosen)} 次）")
print("判讀：max 誤差若 ~1e-15 ⇒ rewind 逐位元乾淨、oracle 挑的就是它以為的那個。")
