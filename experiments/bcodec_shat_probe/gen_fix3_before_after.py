"""FIX3 第 2 項留痕檔生成器（可重生、自洽自證）。

修前（before）：ŝ 目標＝正規化未來絕對態 norm(s_future)。
修後（after）：ŝ 目標＝正規化位移 norm(s_future) − norm(s_current)。
兩欄同在正規化空間 —— 檔內 assert：after == before − tile(current_norm)。

初版留痕檔的病（檢察 2026-09-27 抓到）：before 欄存了 raw、after 欄除了 sd，
兩欄不同空間、無法互推；且合成位移恆定使 8 筆 after 同值。本腳本重造：
非平凡 mu/sd、8 筆各異位移、mu/sd 一併存檔。
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bcodec_v0"))
import bcodec_common as bc

SEG_LEN, OBS_DIM, N = 4, 29, 8
rng = np.random.default_rng(20260927)

obs_mu = rng.normal(0.0, 0.5, size=OBS_DIM).astype(np.float32)
obs_sd = rng.uniform(0.3, 1.7, size=OBS_DIM).astype(np.float32)

current = rng.normal(0.0, 1.0, size=(N, OBS_DIM)).astype(np.float32)
# 8 筆各異的位移（初版恆定 +0.02 導致 after 全同值 —— 這裡逐筆逐步不同）
disp = rng.normal(0.0, 0.15, size=(N, SEG_LEN, OBS_DIM)).astype(np.float32)
future = current[:, None, :] + np.cumsum(disp, axis=1)

# 走 shat_probe.py 同一條正規化路徑（bc._norm_seg 對 (N, L*D) 攤平段做逐步正規化）
future_norm = bc._norm_seg(
    future.reshape(N, SEG_LEN * OBS_DIM), obs_mu, obs_sd, SEG_LEN, OBS_DIM
).reshape(N, SEG_LEN * OBS_DIM)
current_norm = ((current - obs_mu) / obs_sd).astype(np.float32)

before_target = future_norm                                   # 修前語義
after_target = (
    future_norm.reshape(N, SEG_LEN, OBS_DIM) - current_norm[:, None, :]
).reshape(N, SEG_LEN * OBS_DIM)                               # 修後語義（shat_probe.py:159-160 同式）

# ---- 自洽 assert（檢察的重放檢查內建化）----
recon = before_target.reshape(N, SEG_LEN, OBS_DIM) - current_norm[:, None, :]
assert np.allclose(recon.reshape(N, -1), after_target, atol=1e-6), "after != before - current_norm"
manual = ((future - obs_mu) / obs_sd - ((current - obs_mu) / obs_sd)[:, None, :])
assert np.allclose(manual.reshape(N, -1), after_target, atol=1e-5), "位移≠正規化差（二次除 sd 病）"
assert len({a.tobytes() for a in after_target}) == N, "8 筆 after 必須各異"

out = Path(__file__).resolve().parent / "results" / "fix3_before_after.npz"
np.savez(
    out,
    sample_current=current,
    sample_current_normalized=current_norm,
    sample_future_absolute=future,
    before_target_absolute_normalized=before_target,
    after_target_displacement_normalized=after_target,
    obs_mu=obs_mu,
    obs_sd=obs_sd,
)
print(f"OK wrote {out}  n={N} seg_len={SEG_LEN} obs_dim={OBS_DIM}  "
      f"after_distinct={len({a.tobytes() for a in after_target})}/8  asserts=3 passed")
