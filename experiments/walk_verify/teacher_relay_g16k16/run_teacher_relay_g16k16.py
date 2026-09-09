#!/usr/bin/env python
"""G16K16 教材上界測試（teacher relay）—— K32 單碼本 .554 的換裝候選驗證。

工單背景（見 docs/NOTE-2026-09-09-teacher-g16k16.md）：G×K 因子化字典掃描
（docs/NOTE-2026-09-08-gk-scan-route.md）膝點落在 G=16,K=16（64 bits/chunk，
val 重建 MSE .01529），但「教材能不能走」只驗過 K32 單碼本
（docs/NOTE-2026-09-08-teacher-relay.md：schedule 版 per-leg .5536）。本檔補
G16K16 的同款驗證，判準：同一批 200 條接力題目、schedule 開環版協定，per-leg
到達率與 K32 .554 比較，|Δ|<=.07 算打平（續留換裝候選）。

⛔ 不修改任何既有檔案；只 import：
  - experiments/walk_verify/wv_common.py（資料/env/畫圖，純函式）
  - experiments/walk_verify/p1_replay.py（sim_obs/q/timed_reach，純函式）
  - experiments/walk_verify/p2_analyze_traces.py（MAZE_CENTER/RHO_DEFAULT，常數）
  - experiments/walk_verify/teacher_relay/run_teacher_relay.py（build_tasks/DELTA_SUB/
    DATASET，直接重用同一份題目構造函式 —— 保證跟 K32 版是【同一批】200 題，
    不是重新實作一份可能subtly不同的版本）
  - experiments/gk_scan/gk_common.py（G16K16 的模型類別 FactorizedCondVQVAE 與
    資料切分工具，直接重用訓練時的同一套程式碼與 split_seed，保證 train/val
    切分與訓練當時完全一致）

跟 K32 版唯一的協定差異 ＝【encode/decode 用哪本字典】：
  - K32（CondVQVAE，wv_common）：encode_idx(x)->單一 scalar code（K=32 選 1）、
    decode_from_idx(idx, obs_raw)（obs 正規化在模型內部做）。
  - G16K16（FactorizedCondVQVAE，gk_common）：encoder(x)->128 維連續 -> 16 組
    各自的 VQEMA(K=16) 各選 1 字 -> idx 是長度 16 的 code 元組；decode_codes(idx,
    obs_norm) 需要【呼叫端自己先正規化】obs（obs_mu/obs_sd 存在 ckpt 裡，不是
    模型的 buffer）——這是本檔跟 K32 版介面上最大的不同，已用 ckpt 內建的
    obs_mu/obs_sd 處理，沒有重算任何統計量。

範圍縮減（相對 K32 版 run_teacher_relay.py 的兩個版本 schedule/nn）：本檔只做
schedule（純開環，第 k 個 chunk 播該集自己第 k 段的 hindsight 真字），不做 nn
（最近鄰閉環修正）——工單§主結果只要求 schedule 版跟 K32 的 .554 比。

新增（K32 版沒有的）：爛錨對照（random 模式）——每個 chunk 不查真字，改成 16
組各自 uniform 隨機取字，同協定（相同題目子集、相同 leg 判定、相同 obs
conditioning）跑一遍。用來驗證量測管線本身沒有毛病（如果隨機字都能有高
per-leg，代表 leg 判定或環境互動某處壞了，不是字典的功勞）。

⛔ CPU-only（MuJoCo 物理 + 小 MLP encode/decode，不碰 GPU、不用 GPU 相關套件）。
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

HERE = os.path.dirname(os.path.abspath(__file__))               # teacher_relay_g16k16/
WV_DIR = os.path.dirname(HERE)                                   # walk_verify/
EXP_DIR = os.path.dirname(WV_DIR)                                 # experiments/
TR_DIR = os.path.join(WV_DIR, "teacher_relay")                    # walk_verify/teacher_relay/
GK_DIR = os.path.join(EXP_DIR, "gk_scan")                         # experiments/gk_scan/

sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)
sys.path.insert(0, GK_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q, timed_reach  # noqa: E402 - 純函式，重用
from p2_analyze_traces import MAZE_CENTER, RHO_DEFAULT  # noqa: E402 - 常數重用
import run_teacher_relay as trk32  # noqa: E402 - 重用 build_tasks/DELTA_SUB/DATASET/COMPARE（不改動）
import gk_common as gc  # noqa: E402 - G16K16 模型類別與資料切分工具（不改動）

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DEFAULT_CKPT = os.path.join(GK_DIR, "ckpt", "G16_K16.pt")
DEFAULT_RULER = os.path.join(WV_DIR, "results", "ruler_pack.json")
K32_SUMMARY_JSON = os.path.join(TR_DIR, "results", "teacher_relay_summary.json")

# NOTE-2026-09-08-gk-scan-route.md §1.2 一列 / experiments/gk_scan/results/G16_K16.json（一手引用，非本次量測）
REF_VAL_MSE_NOTE = 0.01529
REF_VAL_MSE_JSON = 0.01528940163552761
REF_TRAIN_MSE_JSON = 0.014366136863827705
GK_JSON_REF = os.path.join(GK_DIR, "results", "G16_K16.json")

TIE_BAND = 0.07  # 工單判準：|Δ| <= .07 算打平


# ----------------------------------------------------------------------
# G16K16 模型載入（obs_mu/obs_sd 從 ckpt 讀，⛔ 不重算）
# ----------------------------------------------------------------------
def load_g16k16(ckpt_path, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    if "obs_mu" not in ck or "obs_sd" not in ck:
        raise RuntimeError(f"⛔ ckpt 裡沒有 obs_mu/obs_sd：{ckpt_path} —— 不准自己重算充數，停手回報")
    model = gc.FactorizedCondVQVAE(seg_dim=cfg["seg_len"] * gc.ACT_DIM, obs_dim=gc.OBS_DIM,
                                   G=cfg["G"], K=cfg["K"], D=cfg["group_dim"], hidden=cfg["hidden"],
                                   decay=cfg["decay"], dead_steps=cfg["dead_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    obs_mu = torch.from_numpy(np.asarray(ck["obs_mu"], dtype=np.float32))
    obs_sd = torch.from_numpy(np.asarray(ck["obs_sd"], dtype=np.float32))
    return model, cfg, obs_mu, obs_sd


def load_k32_reference():
    if not os.path.exists(K32_SUMMARY_JSON):
        raise RuntimeError(f"⛔ 找不到 K32 對照來源：{K32_SUMMARY_JSON} —— 停手回報")
    with open(K32_SUMMARY_JSON) as f:
        d = json.load(f)
    return d["results"]["schedule"], d


# ----------------------------------------------------------------------
# 選字使用率：G 組各自 K-way，回報每組 + 跨組彙總（K32 版是單一 K=32 codebook，
# 這裡沒有直接對應的單一數字，用跨組平均當最接近的類比健康度指標）
# ----------------------------------------------------------------------
def group_usage_stats(codes_all, K):
    n, gdim = codes_all.shape
    per_group = []
    for g in range(gdim):
        cnt = np.bincount(codes_all[:, g], minlength=K)
        fr = cnt / max(cnt.sum(), 1)
        nz = fr[fr > 0]
        perplexity = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0
        per_group.append(dict(active=int((cnt > 0).sum()), perplexity=perplexity,
                              top1=float(fr.max()) if fr.size else 0.0))
    active = np.array([p["active"] for p in per_group], dtype=float)
    perp = np.array([p["perplexity"] for p in per_group], dtype=float)
    top1 = np.array([p["top1"] for p in per_group], dtype=float)
    return dict(G=int(gdim), K=int(K), n_samples=int(n), per_group=per_group,
                mean_active=float(active.mean()), min_active=int(active.min()), max_active=int(active.max()),
                mean_perplexity=float(perp.mean()), mean_top1=float(top1.mean()))


# ----------------------------------------------------------------------
# 一次接力（schedule=hindsight 真字 / random=爛錨對照）
# ----------------------------------------------------------------------
def run_one_g16k16(env, u, model, obs_mu, obs_sd, raw, task, mode, rho, code_rng=None,
                   render=False, renderer=None, cams=None, frame_every=4, max_frames=200):
    s0, n_chunks, M, wp_xy = task["s0"], task["n_chunks"], task["M"], task["wp_xy"]
    act = raw["actions"]
    lo = env.action_space.low.astype(np.float64)
    hi = env.action_space.high.astype(np.float64)
    Gd, Kd = model.G, model.K

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

    prev_xy = raw["qpos"][s0, :2].copy()
    for k in range(n_chunks):
        with torch.no_grad():
            if mode == "schedule":
                s_idx = s0 + 4 * k
                x = torch.from_numpy(act[s_idx:s_idx + 4].reshape(1, -1).astype(np.float32))
                z_e = model.encoder(x)
                _, idx, _ = model.vq(z_e, training=False)
            elif mode == "random":
                assert code_rng is not None, "⛔ random 模式需要 code_rng"
                idx = torch.from_numpy(code_rng.integers(0, Kd, size=(1, Gd), dtype=np.int64))
            else:
                raise ValueError(f"⛔ 不支援的 mode={mode}")
            o_raw = torch.from_numpy(sim_obs(u)[None, :])
            o_norm = (o_raw - obs_mu) / obs_sd
            raw_a = model.decode_codes(idx, o_norm).numpy().reshape(4, gc.ACT_DIM)
        codes_used.append(idx.numpy().reshape(-1).copy())
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
    out = dict(completed=completed, legs_attempted=min(active_leg, M),
              legs_reached=active_leg - 1, n_steps_run=global_step,
              zs=np.asarray(zs), speeds=np.asarray(speeds), acts=np.asarray(acts_flat),
              xy=np.asarray(xy_hist),
              codes_used=(np.stack(codes_used, axis=0) if codes_used else np.zeros((0, Gd), dtype=np.int64)),
              clip_frac=float(n_clip / max(n_elem, 1)),
              leg_reach_step=dict(leg_reach_step), leg_active_step=dict(leg_active_step),
              frames=frames if render else None)
    return out


def measure_mode_g16k16(env, u, model, obs_mu, obs_sd, raw, tasks, ruler, mode, rho, fall_line,
                        code_rng=None):
    ta = time.time()
    all_speed, all_z, all_adiff, all_codes = [], [], [], []
    first_hit_data, N_data_arr, first_hit_table, N_table_arr, horizon_arr = [], [], [], [], []
    per_task = []
    for ti, task in enumerate(tasks):
        r = run_one_g16k16(env, u, model, obs_mu, obs_sd, raw, task, mode, rho, code_rng=code_rng)
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
            fh = (rs - act_step + 1) if rs is not None else -1
            first_hit_data.append(fh)
            N_data_arr.append(task["n_data"][m - 1])
            first_hit_table.append(fh)
            N_table_arr.append(task["n_table"][m - 1])
            horizon_arr.append(H - act_step)

    speed = np.concatenate(all_speed)
    zall = np.concatenate(all_z)
    adiff = np.concatenate(all_adiff) if all_adiff else np.array([])
    non_empty_codes = [c for c in all_codes if len(c)]
    codes_all = np.concatenate(non_empty_codes, axis=0) if non_empty_codes else np.zeros((0, model.G), dtype=np.int64)
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
            reached, ev = timed_reach(fh, N_arr, mult, horizon_arr)
            nev = int(ev.sum())
            timed[f"reach_{nm}_{mult:g}N"] = float(reached[ev].mean()) if nev else None
            timed[f"n_evaluable_{nm}_{mult:g}N"] = nev
            timed[f"n_not_evaluable_{nm}_{mult:g}N"] = int((~ev).sum())

    usage = group_usage_stats(codes_all, model.K) if len(codes_all) else None
    fell_any = np.array([bool(np.min(z) < fall_line) for z in all_z])
    n_completed = sum(p["completed"] for p in per_task)

    rec = dict(
        mode=mode, seconds=time.time() - ta, n_tasks=len(tasks),
        completion_ratio=n_completed / len(tasks),
        legs_attempted_total=legs_attempted_total, legs_reached_total=legs_reached_total,
        per_leg_untimed_reach=per_leg_untimed, timed_reach=timed,
        step_speed_q=q(speed), step_speed_mean=float(speed.mean()),
        fall_rate_step=float((zall < fall_line).mean()), fall_rate_task=float(fell_any.mean()),
        action_diff_q=q(adiff) if len(adiff) else None,
        action_diff_mean=float(adiff.mean()) if len(adiff) else None,
        code_usage=usage,
        per_task=per_task,
    )
    raw_arrays = dict(speed=speed, z=zall, adiff=adiff)
    return rec, raw_arrays


def render_exemplars_g16k16(env, u, model, obs_mu, obs_sd, raw, tasks, per_task, rho, gif_dir,
                            frame_every=4, max_frames=200):
    """對 schedule 版挑 1 條 completed=True、1 條 completed=False，重跑一次（render=True）存 gif。"""
    import mujoco
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
    cam_wide.lookat[0], cam_wide.lookat[1] = MAZE_CENTER
    cam_wide.lookat[2] = 0.0
    cam_wide.distance = 38.0
    cam_wide.elevation = -75
    cam_wide.azimuth = 90
    cams = (cam, cam_wide)

    made = {}
    succ = next((p for p in per_task if p["completed"]), None)
    fail = next((p for p in per_task if not p["completed"]), None)
    for kind, rec in (("success", succ), ("fail", fail)):
        if rec is None:
            made[kind] = None
            print(f"⛔ schedule/{kind}：200 條裡沒有這種結果，無法產生這支 gif")
            continue
        task = tasks[rec["ti"]]
        r = run_one_g16k16(env, u, model, obs_mu, obs_sd, raw, task, "schedule", rho, render=True,
                           renderer=renderer, cams=cams, frame_every=frame_every, max_frames=max_frames)
        assert r["completed"] == rec["completed"] and r["n_steps_run"] == rec["n_steps_run"], (
            f"⛔ 重跑不一致（應為 bit-exact 決定性）：kind={kind} "
            f"first_pass=(completed={rec['completed']},steps={rec['n_steps_run']}) "
            f"second_pass=(completed={r['completed']},steps={r['n_steps_run']})")
        fp = os.path.join(gif_dir, f"schedule_{kind}_ep{task['episode']}.gif")
        if r["frames"]:
            import imageio.v2 as imageio
            imageio.mimsave(fp, r["frames"], duration=0.08, loop=0)
            made[kind] = dict(path=fp, n_frames=len(r["frames"]), episode=task["episode"],
                              legs_reached=r["legs_reached"], M=task["M"], n_steps_run=r["n_steps_run"])
            print(f"saved gif: {fp} ({len(r['frames'])} 幀, episode={task['episode']}, "
                  f"legs {r['legs_reached']}/{task['M']})")
        else:
            made[kind] = None
            print(f"⛔ render 失敗（0 幀）：schedule/{kind} episode={task['episode']}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--ruler", type=str, default=DEFAULT_RULER)
    ap.add_argument("--data-dir", type=str, default=DD)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--gif-dir", type=str, default=os.path.join(HERE, "results", "gifs"))
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--n-anchor", type=int, default=20)
    ap.add_argument("--anchor-threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260908)          # 同 K32 版：保證同一批 200 題
    ap.add_argument("--anchor-seed", type=int, default=20260909)   # 只驅動爛錨的隨機碼，跟題目抽樣種子分開
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", type=str, default="teacher_relay_g16k16")
    ap.add_argument("--frame-every", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--skip-gifs", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print(f"=== teacher-relay-g16k16  ckpt={os.path.basename(args.ckpt)} n_traj={args.n_traj} "
          f"n_anchor={args.n_anchor} seed={args.seed} anchor_seed={args.anchor_seed} rho={args.rho} "
          f"DELTA_SUB={trk32.DELTA_SUB} ===")

    model, cfg, obs_mu, obs_sd = load_g16k16(args.ckpt)
    assert cfg["seg_len"] == 4 and cfg["G"] == 16 and cfg["K"] == 16, (
        f"⛔ 預期 G16K16 seg_len=4 G=16 K=16，實際 cfg={cfg}")
    print(f"dict: G={cfg['G']} K={cfg['K']} group_dim={cfg['group_dim']} seg_len={cfg['seg_len']} "
          f"hidden={cfg['hidden']} split_seed={cfg['split_seed']} val_frac={cfg['val_frac']} "
          f"ckpt_train_seed={cfg['seed']}")
    print(f"obs_mu[:2]={obs_mu.numpy().flatten()[:2]}（核對 MAZE_CENTER={MAZE_CENTER}）")

    # ======================================================================
    # 驗收①：載入自檢 —— 3 個 train 段 encode->decode，量重建誤差量級
    # ======================================================================
    train_npz = os.path.join(args.data_dir, f"{trk32.DATASET}.npz")
    data = gc.load_segments_with_obs(train_npz, seg_len=cfg["seg_len"],
                                     split_seed=cfg["split_seed"], val_frac=cfg["val_frac"])
    print(f"[驗收①] train/val 切分重建：n_train={len(data['train_idx'])} n_val={len(data['val_idx'])} "
          f"（G16_K16.json 記錄 n_train=225000 n_val=25000，來源={GK_JSON_REF}）")

    pick3 = data["train_idx"][:3]
    x3 = torch.from_numpy(data["segs"][pick3])
    obs3_raw = torch.from_numpy(data["obs_start"][pick3])
    obs3_norm = (obs3_raw - obs_mu) / obs_sd
    with torch.no_grad():
        recon3, idx3, _ = model(x3, obs3_norm, training=False)
        per_sample_mse3 = ((recon3 - x3) ** 2).mean(dim=1).numpy()
        mse3 = float(F.mse_loss(recon3, x3).item())
    seg_starts3 = data["seg_starts"][pick3].tolist()
    print(f"[驗收①] 3 個 train 段（seg_starts={seg_starts3}）：per-sample MSE={per_sample_mse3.tolist()}  "
          f"mean={mse3:.5f}")
    same_order = 0.001 < mse3 < 0.1
    print(f"[驗收①] 對照 {GK_JSON_REF} 記錄 val MSE={REF_VAL_MSE_JSON:.6f}（NOTE 表列 {REF_VAL_MSE_NOTE}），"
          f"train MSE={REF_TRAIN_MSE_JSON:.6f} —— {'同量級' if same_order else '⚠️ 量級不符，請檢查'}")

    # bonus：整個 val 切分（理論上 25000 段）重建 MSE，直接比對 JSON 精確值（更強的載入正確性證據，
    # 非工單硬性要求，但成本近乎 0 且是更嚴格的驗證，一併做）
    val_x = torch.from_numpy(data["segs"][data["val_idx"]])
    val_o_raw = torch.from_numpy(data["obs_start"][data["val_idx"]])
    val_o_norm = (val_o_raw - obs_mu) / obs_sd
    with torch.no_grad():
        val_recon, _, _ = model(val_x, val_o_norm, training=False)
        recon_mse_val_check = float(F.mse_loss(val_recon, val_x).item())
    val_mse_diff = abs(recon_mse_val_check - REF_VAL_MSE_JSON)
    val_mse_match = val_mse_diff < 1e-4
    print(f"[驗收①-加驗，非工單硬性項] 整個 val 切分（{len(data['val_idx'])} 段）重建 MSE="
          f"{recon_mse_val_check:.6f} vs JSON 記錄 {REF_VAL_MSE_JSON:.6f}（差={val_mse_diff:.2e}）—— "
          f"{'吻合，載入正確' if val_mse_match else '⚠️ 對不起來，需要查'}")

    # ======================================================================
    # 接力題目（沿用 run_teacher_relay.build_tasks；同一顆 seed=20260908 => 同一批 200 題）
    # ======================================================================
    with open(args.ruler) as f:
        ruler = json.load(f)
    fall_line = ruler["torso_z"]["fall_line_p1"]
    raw = wv.load_npz(args.data_dir, trk32.DATASET, "val")
    tasks, skipped = trk32.build_tasks(raw, args.n_traj, args.seed, ruler, args.rho)
    print(f"tasks: {len(tasks)} usable (要求 >= {args.n_traj})  skipped={len(skipped)}")
    for s in skipped:
        print(f"  ⛔ skip episode={s['episode']} arclen={s['arclen']:.3f} reason={s['reason']}")
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報（不用預設值填）"
    Ms = np.array([t["M"] for t in tasks])
    print(f"waypoints per task: min={Ms.min()} p50={np.median(Ms):.0f} max={Ms.max()} "
          f"total_legs_available={Ms.sum()}")

    env = wv.make_env(args.data_dir, trk32.DATASET)
    u = env.unwrapped
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    # ======================================================================
    # 驗收②：爛錨對照（管線自檢，先跑；20 題，16 組各自 uniform 隨機取字）
    # ======================================================================
    anchor_tasks = tasks[:args.n_anchor]
    anchor_rng = np.random.default_rng(args.anchor_seed)
    anchor_rec, anchor_ra = measure_mode_g16k16(env, u, model, obs_mu, obs_sd, raw, anchor_tasks,
                                                ruler, "random", args.rho, fall_line, code_rng=anchor_rng)
    print(f"\n[驗收②-爛錨] n_tasks={len(anchor_tasks)}  per-leg 不限時到達率="
          f"{anchor_rec['per_leg_untimed_reach']:.4f} (reached {anchor_rec['legs_reached_total']}/"
          f"{anchor_rec['legs_attempted_total']})  completion_ratio={anchor_rec['completion_ratio']:.3f}")
    if anchor_rec["per_leg_untimed_reach"] >= args.anchor_threshold:
        print(f"⛔⛔⛔ 爛錨對照 per-leg={anchor_rec['per_leg_untimed_reach']:.4f} >= "
              f"門檻 {args.anchor_threshold} —— 管線可能壞了，停下來查，不繼續跑主結果 ⛔⛔⛔")
        js_path = os.path.join(args.out_dir, f"{args.tag}_ABORTED_anchor_summary.json")
        wv.save_json(dict(reason="anchor_control_failed", anchor_control=anchor_rec,
                          anchor_threshold=args.anchor_threshold), js_path)
        print(f"saved (中止證據): {js_path}")
        sys.exit(1)
    print(f"[驗收②] 爛錨 per-leg={anchor_rec['per_leg_untimed_reach']:.4f} < {args.anchor_threshold} —— "
          f"管線自檢通過，繼續主結果")

    # ======================================================================
    # 驗收③：主結果 —— 200 題 schedule 版、hindsight 真字
    # ======================================================================
    main_rec, main_ra = measure_mode_g16k16(env, u, model, obs_mu, obs_sd, raw, tasks, ruler,
                                            "schedule", args.rho, fall_line)
    print(f"\n[schedule/g16k16] done in {main_rec['seconds']:.1f}s  "
          f"completion_ratio={main_rec['completion_ratio']:.3f}  "
          f"per-leg untimed={main_rec['per_leg_untimed_reach']:.4f} "
          f"(reached {main_rec['legs_reached_total']}/{main_rec['legs_attempted_total']})  "
          f"speed_p50={main_rec['step_speed_q']['50']:.4f}  fall(step)={main_rec['fall_rate_step']*100:.3f}%  "
          f"adiff_p50={(main_rec['action_diff_q']['50'] if main_rec['action_diff_q'] else float('nan')):.3f}")
    cu = main_rec["code_usage"]
    print(f"           選字使用率（G={cu['G']} 組 K={cu['K']}，跨組平均）：mean active={cu['mean_active']:.1f}/"
          f"{cu['K']} (min={cu['min_active']} max={cu['max_active']})  "
          f"mean perplexity={cu['mean_perplexity']:.2f}/{cu['K']}  mean top1={cu['mean_top1']*100:.1f}%")

    print("\n=== per-leg 限時到達率（schedule/g16k16）===")
    tr = main_rec["timed_reach"]
    for nm in ("N_data", "N_table"):
        r125 = tr[f"reach_{nm}_1.25N"]
        r150 = tr[f"reach_{nm}_1.5N"]
        print(f"  N={nm:8s} 1.25N={r125*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.25N']})  "
              f"1.5N={r150*100:.1f}% (n={tr[f'n_evaluable_{nm}_1.5N']})  "
              f"not_evaluable={tr[f'n_not_evaluable_{nm}_1.5N']}")

    # ---- 判讀：跟 K32 .554（一手來源：teacher_relay_summary.json，不手key）----
    k32_sched, k32_full = load_k32_reference()
    compare_k32 = k32_sched["per_leg_untimed_reach"]
    delta = main_rec["per_leg_untimed_reach"] - compare_k32
    if abs(delta) <= TIE_BAND:
        verdict = "打平（G16K16 以更高容量打平，續留換裝候選）"
    elif delta > TIE_BAND:
        verdict = "明顯更高（換裝候選）"
    else:
        verdict = "明顯更低（G16K16 教材出局）"
    print(f"\n=== 對照 K32（一手來源：{K32_SUMMARY_JSON}）===")
    print(f"K32[schedule]     per-leg 不限時到達率 = {compare_k32:.4f}  完整走完比例 = "
          f"{k32_sched['completion_ratio']:.3f}")
    print(f"G16K16[schedule]  per-leg 不限時到達率 = {main_rec['per_leg_untimed_reach']:.4f}  完整走完比例 = "
          f"{main_rec['completion_ratio']:.3f}")
    print(f"Δ = {delta:+.4f}  判準 |Δ|<={TIE_BAND} => {verdict}")
    print(f"對照行（沿用 K32 對照，一手引用非本次量測）：{trk32.COMPARE}")

    # ---- 疊圖：真螞蟻 vs G16K16 schedule ----
    ruler_np = np.load(os.path.join(os.path.dirname(args.ruler), "ruler_pack.npz"))
    pngs = {}
    for key, ref, title, xlab, vl in (
        ("speed", "step_speed", "teacher-relay-g16k16 gate2: step speed vs real ant", "|dxy| per step", None),
        ("z", "torso_z", "teacher-relay-g16k16 gate3a: torso height vs real ant", "qpos[2] (m)",
         [dict(x=fall_line, label=f"fall line p1={fall_line:.3f}", color_idx=7)]),
        ("adiff", "action_diff", "teacher-relay-g16k16 gate3b: action smoothness vs real ant",
         "L2 |a_t - a_(t-1)|", None),
    ):
        series = [dict(label="REAL ant (ruler)", values=ruler_np[ref]),
                 dict(label="teacher_relay_g16k16[schedule]", values=main_ra[key])]
        p = os.path.join(args.out_dir, f"{args.tag}_{key}_overlay.png")
        wv.draw_hist_overlay(series, p, title, xlab, vlines=vl)
        pngs[key] = p
        print(f"saved: {p}")

    # ======================================================================
    # 驗收④：gif（成功／失敗各 1 支）
    # ======================================================================
    gifs = {}
    if not args.skip_gifs:
        gifs = render_exemplars_g16k16(env, u, model, obs_mu, obs_sd, raw, tasks,
                                       main_rec["per_task"], args.rho, args.gif_dir,
                                       args.frame_every, args.max_frames)
    else:
        print("⛔ --skip-gifs：本次沒有產生 gif")

    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    wv.save_json(dict(
        config=dict(vars(args)),
        delta_sub=trk32.DELTA_SUB, compare_legacy=trk32.COMPARE, tie_band=TIE_BAND,
        n_tasks_requested=args.n_traj, n_tasks_used=len(tasks), skipped=skipped,
        waypoints_per_task=dict(min=int(Ms.min()), p50=float(np.median(Ms)),
                                max=int(Ms.max()), total=int(Ms.sum())),
        load_self_check=dict(
            train_npz=train_npz, split_seed=cfg["split_seed"], val_frac=cfg["val_frac"],
            n_train=len(data["train_idx"]), n_val=len(data["val_idx"]),
            seg_starts_3=seg_starts3, per_sample_mse_3=per_sample_mse3.tolist(), mean_mse_3=mse3,
            ref_val_mse_note=REF_VAL_MSE_NOTE, ref_val_mse_json=REF_VAL_MSE_JSON,
            ref_train_mse_json=REF_TRAIN_MSE_JSON, ref_source=GK_JSON_REF,
            same_order_of_magnitude=same_order,
            full_val_recon_mse_check=recon_mse_val_check, full_val_recon_mse_diff=val_mse_diff,
            full_val_recon_mse_match=val_mse_match,
        ),
        anchor_control=dict(anchor_rec, threshold=args.anchor_threshold, seed=args.anchor_seed,
                            n_tasks=len(anchor_tasks)),
        results=dict(schedule=main_rec),
        k32_reference=dict(source=K32_SUMMARY_JSON, schedule=k32_sched),
        verdict=dict(delta=delta, tie_band=TIE_BAND, label=verdict, compare_k32=compare_k32),
        artifacts=dict(pngs=pngs, gifs=gifs),
    ), js_path)
    print(f"saved: {js_path}")

    np.savez_compressed(os.path.join(args.out_dir, f"{args.tag}_raw.npz"),
                        schedule_speed=main_ra["speed"], schedule_z=main_ra["z"],
                        schedule_adiff=main_ra["adiff"],
                        anchor_speed=anchor_ra["speed"], anchor_z=anchor_ra["z"],
                        anchor_adiff=anchor_ra["adiff"])
    print(f"saved: {os.path.join(args.out_dir, f'{args.tag}_raw.npz')}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
