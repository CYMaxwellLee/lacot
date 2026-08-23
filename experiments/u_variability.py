"""只載 checkpoint，量『u 在不同題目之間到底變不變』。⛔ 不跑環境、不重訓，幾秒鐘。

主人 2026-08-23 問「u 填 random 呢」，探針跑出來的第一行就露餡了：
真 u 的逐維標準差是 0.0002，而它自己的量值是 0.235 —— 相差一千倍。
⇒ u 對所有題目幾乎是【同一個向量】。這支就是把那件事量清楚。
"""
import os
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.nf_head import Flow
from lacot.model import RefineOperator

OGB = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]; p = h
    return nn.Sequential(*L, nn.Linear(p, o))


print(f"{'checkpoint':<46} {'|u|':>8} {'逐維std':>9} {'std/|u|':>9} {'兩兩cos':>9} {'R0的cos':>9}")
print("-" * 96)
for fn in sorted(os.listdir("results")):
    if not (fn.startswith("ckpt_") and fn.endswith(".pt")):
        continue
    ck = torch.load(os.path.join("results", fn), map_location=device, weights_only=False)
    cfg = ck["cfg"]; K, COND, D = cfg["K"], cfg["COND"], cfg["D_MODEL"]
    # ⚠️ 早期的 checkpoint 檔名沒有環境那一段（那時只跑 medium-navigate）⇒ 認不出來就當它是 medium-navigate。
    env_name = fn.split("_")[1]
    known = {"medium-navigate": "pointmaze-medium-navigate-v0", "large-navigate": "pointmaze-large-navigate-v0",
             "medium-stitch": "pointmaze-medium-stitch-v0", "large-stitch": "pointmaze-large-stitch-v0"}
    full = known.get(env_name, "pointmaze-medium-navigate-v0")
    d = np.load(f"{OGB}/{full}.npz"); OBS = np.asarray(d["observations"], np.float32)
    mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
    ce = sota_mlp(2, 512, 512).to(device); chd = sota_mlp(1024, 512, COND).to(device)
    fl = Flow(token_dim=D, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
    rf = RefineOperator(COND, K, D, hidden=256).to(device)
    for m, k in [(ce, "cond_enc"), (chd, "cond_head"), (fl, "flow"), (rf, "refine")]:
        m.load_state_dict(ck[k]); m.eval()
    with torch.no_grad():
        r = np.random.default_rng(7); i1 = r.integers(0, len(OBS), 512); i2 = r.integers(0, len(OBS), 512)
        s = torch.tensor((OBS[i1] - mu) / sd, device=device); g = torch.tensor((OBS[i2] - mu) / sd, device=device)
        c = chd(torch.cat([ce(s), ce(g)], 1))
        u0 = fl.sample(512, c); u = u0
        for _ in range(cfg.get("R", 3) if False else 3):
            u = rf(c, u)
        flat = lambda x: x.reshape(x.shape[0], -1)
        cosm = lambda x: (lambda z: ((z @ z.t()).sum() - (z @ z.t()).diag().sum()) / (512 * 511))(F.normalize(flat(x), dim=1)).item()
        mag = u.abs().mean().item(); st = u.std(0).mean().item()
        print(f"{fn[5:-3]:<46} {mag:>8.4f} {st:>9.5f} {st/max(mag,1e-9):>9.4f} {cosm(u):>9.4f} {cosm(u0):>9.4f}")
print("-" * 96)
print("兩兩cos 接近 1 ＝ 所有題目的 u 幾乎同一個方向（塌了）。R0 那欄是 refine 之前的 u。")
