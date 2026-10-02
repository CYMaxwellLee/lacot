#!/usr/bin/env python
"""K32 vs G16K16 decoder重建 MSE 對照（同一批 200 段，同 E_indep 的任務集）。

工單§4 附加欄：decoder 重建 MSE 對照——已知 G16K16 重建遠優（gk_scan 的 val 集，
與同目錄 run_drift_analysis_g16k16.py 用 sim_obs() 重跑過一次，見其
recon_mse_baseline），K32 這支字典的重建 MSE 從未在任何既有 NOTE 跟 gk_scan
同條件比較過（見 docs/NOTE-2026-09-09-teacher-g16k16.md §五#3 明確點名的缺口）。
本檔補上：用【跟 E_indep 完全同一批 200 task × 50 chunk】、【跟
gk_common.load_segments_with_obs 同一種 obs 取法（raw["observations"][s]，
不經 MuJoCo 重模擬，見該函式第 97 行 obs_start = observations[seg_starts]）】，
兩本字典各自算一次 decode(encode(真動作段), 真obs) vs 真動作段 的 MSE（clip
前，跟 gk_scan 的 MSE 定義一致），兩邊同一份 task/chunk 集合、同一種 obs 來源，
是嚴格意義上的 apples-to-apples。

⛔ 不修改任何既有檔案，只 import wv_common / teacher_relay/run_teacher_relay.py /
teacher_relay_g16k16/run_teacher_relay_g16k16.py / gk_scan/gk_common.py（皆未改
一行）。不需要 MuJoCo/env（只算 decoder 重建，不執行物理），純 CPU tensor 運算，
前景直接跑（不需要 slurm）。
"""
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.dirname(HERE)
EXP_DIR = os.path.dirname(WV_DIR)
TR_DIR = os.path.join(WV_DIR, "teacher_relay")
TRG16_DIR = os.path.join(WV_DIR, "teacher_relay_g16k16")
GK_DIR = os.path.join(EXP_DIR, "gk_scan")
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)
sys.path.insert(0, TRG16_DIR)
sys.path.insert(0, GK_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import q  # noqa: E402
import run_teacher_relay as trk32  # noqa: E402
import run_teacher_relay_g16k16 as trg16  # noqa: E402
import gk_common as gc  # noqa: E402

DD = trg16.DD
K32_CKPT = trk32.DEFAULT_CKPT
G16K16_CKPT = trg16.DEFAULT_CKPT
RULER = trg16.DEFAULT_RULER
SEED = 20260908
N_TRAJ = 200
RHO = trg16.RHO_DEFAULT

t0 = time.time()
print(f"=== measure_recon_mse_both  K32_ckpt={os.path.basename(K32_CKPT)}  "
      f"G16K16_ckpt={os.path.basename(G16K16_CKPT)}  seed={SEED} n_traj={N_TRAJ} ===")

k32_model, k32_cfg = wv.load_dict_v1(K32_CKPT)
g16_model, g16_cfg, obs_mu, obs_sd = trg16.load_g16k16(G16K16_CKPT)
print(f"K32: seg_len={k32_cfg['seg_len']} K={k32_cfg['k']}")
print(f"G16K16: G={g16_cfg['G']} K={g16_cfg['K']} seg_len={g16_cfg['seg_len']}")

with open(RULER) as f:
    ruler = json.load(f)
raw = wv.load_npz(DD, trk32.DATASET, "val")
tasks, skipped = trk32.build_tasks(raw, N_TRAJ, SEED, ruler, RHO)
assert len(tasks) >= N_TRAJ, "⛔ 可用軌跡不足，停手回報"
print(f"tasks: {len(tasks)} usable  skipped={len(skipped)}")

act = raw["actions"]
obs_all = raw["observations"]

k32_mse, g16_mse = [], []
with torch.no_grad():
    for task in tasks:
        s0, n_chunks = task["s0"], task["n_chunks"]
        for k in range(n_chunks):
            s = s0 + 4 * k
            x = torch.from_numpy(act[s:s + 4].reshape(1, -1).astype(np.float32))
            o_raw = torch.from_numpy(obs_all[s:s + 1].astype(np.float32))

            idx_k32 = k32_model.encode_idx(x)
            recon_k32 = k32_model.decode_from_idx(idx_k32, o_raw)
            k32_mse.append(float(torch.mean((recon_k32 - x) ** 2).item()))

            z_e = g16_model.encoder(x)
            _, idx_g, _ = g16_model.vq(z_e, training=False)
            o_norm = (o_raw - obs_mu) / obs_sd
            recon_g16 = g16_model.decode_codes(idx_g, o_norm)
            g16_mse.append(float(torch.mean((recon_g16 - x) ** 2).item()))

k32_mse = np.asarray(k32_mse)
g16_mse = np.asarray(g16_mse)
assert len(k32_mse) == len(g16_mse) == 10000 or len(k32_mse) == len(g16_mse), (
    f"⛔ n 不對: k32={len(k32_mse)} g16={len(g16_mse)}")

k32_q = q(k32_mse)
g16_q = q(g16_mse)
print(f"\nK32    recon MSE（n={len(k32_mse)}）p25/p50/p75/mean = "
      f"{k32_q['25']:.6f}/{k32_q['50']:.6f}/{k32_q['75']:.6f}/{k32_mse.mean():.6f}")
print(f"G16K16 recon MSE（n={len(g16_mse)}）p25/p50/p75/mean = "
      f"{g16_q['25']:.6f}/{g16_q['50']:.6f}/{g16_q['75']:.6f}/{g16_mse.mean():.6f}")
print(f"G16K16/K32 p50 比值 = {g16_q['50']/k32_q['50']:.4f}  "
      f"（G16K16 相對 K32 降幅 = {(1 - g16_q['50']/k32_q['50'])*100:.1f}%）")
print(f"\n外部參照（非本次量測，一手引用）：gk_scan G16_K16.json val MSE = "
      f"{trg16.REF_VAL_MSE_JSON:.6f}（不同母體：25000 段官方 val split vs 這裡 10000 段"
      f"relay-task chunk，數量級核對用，不是同一統計量）")

out = dict(config=dict(k32_ckpt=K32_CKPT, g16k16_ckpt=G16K16_CKPT, seed=SEED, n_traj=N_TRAJ,
                       rho=RHO, obs_source="raw['observations'][s]（同 gk_common.load_segments_with_obs "
                                          "第 97 行 obs_start 取法，未經 MuJoCo 重模擬）"),
           n_tasks_used=len(tasks), n_segments=len(k32_mse),
           k32_recon_mse=dict(p25=k32_q["25"], p50=k32_q["50"], p75=k32_q["75"],
                              mean=float(k32_mse.mean()), n=int(len(k32_mse))),
           g16k16_recon_mse=dict(p25=g16_q["25"], p50=g16_q["50"], p75=g16_q["75"],
                                 mean=float(g16_mse.mean()), n=int(len(g16_mse))),
           g16k16_vs_k32_p50_ratio=float(g16_q["50"] / k32_q["50"]),
           external_ref_g16k16_val_mse=trg16.REF_VAL_MSE_JSON,
           external_ref_source=trg16.GK_JSON_REF)
out_path = os.path.join(HERE, "results", "recon_mse_both_20260911_summary.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsaved: {out_path}")
print(f"=== done wall={time.time()-t0:.1f}s ===")
