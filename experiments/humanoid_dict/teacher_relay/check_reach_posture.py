#!/usr/bin/env python
"""補充診斷（非主流程，回答一個具體疑點）：teacher-relay 裡「到達路標」的當下，人形是不是
已經跌倒了？（gif 抽查發現 schedule_success_ep273 中途已經倒地——這支腳本把它變成
一個跨全部題目的量測，不是只看 1~2 支 gif 的個案。）

做法：對每個 task 重跑 run_one（跟 run_relay.py 的量測完全同一份函式，只是額外在每個
leg 被判定「到達」的那個 global_step 記下當時的軀幹 z，跟量尺包的翻倒線 fall_line_p1
比較。⛔ 不改變任何量測邏輯、不重跑亂數、fully deterministic，純粹是多印一個統計量。
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import hr_common as hr  # noqa: E402
import hd_common as hc  # noqa: E402
from run_relay import build_tasks, run_one, VERSIONS  # noqa: E402

torch.set_num_threads(4)

ruler = hr.load_ruler()
fall_line = ruler["torso_z"]["fall_line_p1"]
raw = hc.load_raw(hr.DD_DEFAULT, hc.DATASET_NAME, "val")

for tag, ckpt in [("G1K32", "../ckpt/G1_K32.pt"), ("G16K16", "../ckpt/G16_K16.pt")]:
    model, cfg, obs_mu, obs_sd = hr.load_dict(ckpt)
    G, K, seg_len = cfg["G"], cfg["K"], cfg["seg_len"]
    tasks, skipped = build_tasks(raw, 200, 20260908, ruler, hr.RHO_DEFAULT, hr.DELTA_SUB, seg_len)
    env = hr.make_env(hr.DD_DEFAULT)
    u = env.unwrapped
    print(f"\n=== {tag} ===")
    for version in VERSIONS:
        t0 = time.time()
        n_reach_total, n_reach_fallen = 0, 0
        z_at_reach = []
        for task in tasks:
            r = run_one(env, u, model, obs_mu, obs_sd, G, K, raw, task, version, hr.RHO_DEFAULT,
                       seg_len=seg_len)
            for m, gstep in r["leg_reach_step"].items():
                z = float(r["zs"][gstep])
                z_at_reach.append(z)
                n_reach_total += 1
                if z < fall_line:
                    n_reach_fallen += 1
        z_at_reach = np.asarray(z_at_reach)
        frac_fallen = n_reach_fallen / max(n_reach_total, 1)
        print(f"[{version:8s}] legs_reached={n_reach_total}  reached_while_z<fall_line={n_reach_fallen} "
              f"({frac_fallen*100:.1f}%)  z_at_reach: mean={z_at_reach.mean():.3f} "
              f"p10={np.percentile(z_at_reach,10):.3f} p50={np.percentile(z_at_reach,50):.3f} "
              f"p90={np.percentile(z_at_reach,90):.3f}  (fall_line={fall_line:.3f}, real p50 z=1.117)  "
              f"[{time.time()-t0:.1f}s]")
