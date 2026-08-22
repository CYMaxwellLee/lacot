"""資料集跟 env 對得上嗎？—— 官方 GCBC 只跑出 0.15 而論文是 0.99，先驗地基。

`[實測]` 官方 GCBC（用 hyperparameters.sh 指定的預設 flags、1M 步、batch 1024）
在 pointmaze-medium-navigate-v0 上只拿到 overall 0.15（best 0.18），
而論文 Table 2 是 99±6。訓練 MSE 也只從 0.421 降到 0.362（「猜平均」是 0.494）。
=> 在檢討方法之前，要先確定【資料集與環境是匹配的】。

檢查方式（決定性）：
  把 env 用資料集的 qpos/qvel 設回某個時間點，然後餵資料集記錄的 action，
  逐步比對 env 吐出的 observation 與資料集記錄的 observation。
  對得上 => 資料與 env 匹配，問題在別處。
  對不上 => 地基就是壞的，所有實驗數字都要重來。

⛔ 同時做一個【負控制】：故意餵打亂的 action，偏差必須明顯變大。
   （否則這個檢查可能是「怎樣都會過」的假綠燈。）
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ogbench

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
ENV = "pointmaze-medium-navigate-v0"

d = np.load(f"{OGB_DATA}/{ENV}.npz")
print("keys:", list(d.keys()))
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
QPOS = np.asarray(d["qpos"], np.float64) if "qpos" in d else None
QVEL = np.asarray(d["qvel"], np.float64) if "qvel" in d else None
print("observations", OBS.shape, "| actions", ACT.shape, "| qpos",
      None if QPOS is None else QPOS.shape, "| qvel", None if QVEL is None else QVEL.shape)

env, train_ds, val_ds = ogbench.make_env_and_datasets(ENV)
print("\n=== 官方 loader 回的 dataset 跟我們讀的 .npz 一樣嗎 ===")
for k in ("observations", "actions", "terminals"):
    a = np.asarray(train_ds[k])
    b = {"observations": OBS, "actions": ACT, "terminals": TERM}[k]
    same = a.shape == b.shape and np.allclose(np.asarray(a, np.float64)[:1000],
                                              np.asarray(b, np.float64)[:1000], atol=1e-5)
    print(f"  {k:<14} loader {str(a.shape):<16} npz {str(b.shape):<16} 前 1000 筆相同: {same}")

ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])


def replay(start_idx, n_steps, shuffle_actions=False):
    """把 env 設到 start_idx 的狀態，餵資料集的 action，比對 observation。"""
    env.reset()
    u = env.unwrapped
    u.set_state(QPOS[start_idx].copy(), QVEL[start_idx].copy())
    errs = []
    acts = ACT[start_idx:start_idx + n_steps].copy()
    if shuffle_actions:
        rng = np.random.default_rng(0)
        acts = acts[rng.permutation(len(acts))]
    for t in range(n_steps):
        ob, _, _, _, _ = env.step(np.clip(acts[t], -1, 1))
        rec = OBS[start_idx + t + 1]
        errs.append(float(np.linalg.norm(np.asarray(ob, np.float64)[:2] - rec[:2])))
    return np.array(errs)


if QPOS is None:
    print("\n⛔ 資料集沒有 qpos/qvel，無法把 env 設回同一狀態 —— 這個檢查做不了")
    sys.exit(2)

print("\n=== replay 檢查（env 設回資料集狀態，餵資料集的 action）===")
print(f"  {'起點':>10} | {'真 action 累積誤差':>20} | {'打亂 action（負控制）':>22}")
print("  " + "-" * 60)
N = 50
real_all, shuf_all = [], []
for s0 in starts[:5]:
    s0 = int(s0) + 10
    e_real = replay(s0, N)
    e_shuf = replay(s0, N, shuffle_actions=True)
    real_all.append(e_real[-1]); shuf_all.append(e_shuf[-1])
    print(f"  {s0:>10} | {e_real[-1]:>20.4f} | {e_shuf[-1]:>22.4f}")

r, sh = float(np.mean(real_all)), float(np.mean(shuf_all))
print("  " + "-" * 60)
print(f"  {'平均':>10} | {r:>20.4f} | {sh:>22.4f}")
print()
if r < 0.05 and sh > r * 5:
    print("✅ 資料集與 env 匹配（真 action 幾乎零誤差，打亂後明顯變大）=> 地基沒問題，問題在別處")
elif r < 0.05:
    print("⚠️ 真 action 誤差很小，但【負控制沒有拉開】—— 這個檢查可能無效，別採信")
else:
    print("🚨 真 action 也對不上 => 資料集與 env 不匹配，所有實驗數字都要重來")
