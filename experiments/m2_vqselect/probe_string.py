#!/usr/bin/env python
"""M2 定位探針：寫譜頭寫出來的【字串】長得像不像真的步法譜。

⭐ 為什麼要它：eval 量到寫譜頭實走的字串「連續 chunk 同字率 30.9%」，而教材的真字串
   只有 2.8% —— 真步態幾乎每個 chunk 都換字。這支把「是解碼法造成的」跟
   「是頭學壞了」分開：**在同一批資料狀態上**（頭是 in-distribution 的，沒有
   covariate shift），比 argmax 解碼 vs 依 p(code|s,w) 抽樣 vs 真字串 三者的字串統計。
⛔ 這支【不是】rollout —— 它不回答到達率，只回答「argmax 是不是把譜壓平了」。

⛔ CPU-only、⛔ 不碰 GPU、⛔ 不寫既有檔案。
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import lacot_vqsel as vqsel  # noqa: E402
from probe_headout import load_split, sota_mlp  # noqa: E402

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
ENV_NAME = "antmaze-medium-stitch-v0"
CHUNK = 4


def stats(strings, K, name):
    same = float(np.mean([np.mean(c[1:] == c[:-1]) for c in strings if len(c) > 1]))
    flat = np.concatenate(strings)
    fr = np.bincount(flat, minlength=K) / len(flat)
    nz = fr[fr > 0]
    px = float(np.exp(-np.sum(nz * np.log(nz))))
    print(f"  {name:<28s} 連續同字率 {same * 100:5.1f}%   perplexity {px:5.2f}/{K}"
          f"   top1 {fr.max() * 100:4.1f}%   用到 {int((fr > 0).sum())}/{K}")
    return dict(same_code_rate=same, perplexity=px, top1=float(fr.max()),
                active=int((fr > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-eps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "results", "m2_string.json"))
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    vs, cfg = ck["vqsel"], ck["cfg"]
    dic, dcfg = vqsel.load_dict(vs["dict_ckpt"], device="cpu", verbose=False)
    K = int(vs["K"])
    arc = float(cfg.get("VQSEL_ARC", 7.5))

    s_enc = sota_mlp(dcfg["obs_dim"], 512, 512); w_enc = sota_mlp(2, 512, 512)
    chn = sota_mlp(1024, 512, int(cfg["COND"])); hd = sota_mlp(int(cfg["COND"]), 512, K, n=3)
    s_enc.load_state_dict(vs["s"]); w_enc.load_state_dict(vs["w"])
    chn.load_state_dict(vs["ch"]); hd.load_state_dict(vs["head"])
    for m in (s_enc, w_enc, chn, hd):
        m.eval()

    tr_obs, _, _ = load_split(os.path.join(DD, f"{ENV_NAME}.npz"))
    mu, sd = tr_obs.mean(0), tr_obs.std(0) + 1e-6
    del tr_obs
    obs, act, te = load_split(os.path.join(DD, f"{ENV_NAME}-val.npz"))
    arccum = vqsel.build_arc_cum(obs[:, :2], te)
    ends = np.flatnonzero(np.diff(te) != 0).tolist()
    starts = np.r_[0, np.array(te)[:-1][np.diff(te) != 0] + 1]

    gen = torch.Generator().manual_seed(a.seed)
    S_true, S_arg, S_smp = [], [], []
    with torch.no_grad():
        for s0 in starts[:a.n_eps]:
            e0 = int(te[s0]); nc = (e0 - s0 + 1) // CHUNK
            if nc < 2:
                continue
            rows = s0 + np.arange(nc) * CHUNK
            segs = act[rows[:, None] + np.arange(CHUNK)[None, :]].reshape(nc, CHUNK * 8)
            S_true.append(dic.encode_idx(torch.from_numpy(np.ascontiguousarray(segs, np.float32))).numpy())
            # 路標＝同一條軌跡往前 arc 單位弧長的點（訓練時的定義；⛔ 這裡不是 conf2 subgoal）
            tg = vqsel.arc_waypoint(arccum, te, rows, np.full(nc, arc))
            s = torch.from_numpy(np.ascontiguousarray((obs[rows] - mu) / sd, np.float32))
            w = torch.from_numpy(np.ascontiguousarray(
                (obs[tg, :2].astype(np.float64) - obs[rows, :2].astype(np.float64)) / arc, np.float32))
            lg = hd(chn(torch.cat([s_enc(s), w_enc(w)], 1)))
            S_arg.append(lg.argmax(1).numpy())
            S_smp.append(torch.multinomial(torch.softmax(lg, 1), 1, generator=gen)[:, 0].numpy())

    print(f"=== 字串統計（官方 val {len(S_true)} 集、{sum(len(c) for c in S_true)} 個 chunk）===")
    print("  ⚠️ 狀態全部來自【資料軌跡】⇒ 頭是 in-distribution 的，"
          "⛔ 這裡量不到 rollout 的 covariate shift，只量解碼法的效果。")
    out = {"n_eps": len(S_true), "n_chunks": int(sum(len(c) for c in S_true)), "arc": arc}
    out["true"] = stats(S_true, K, "教材真字串（上界）")
    out["argmax"] = stats(S_arg, K, "寫譜頭 argmax（eval 用的）")
    out["sample"] = stats(S_smp, K, "寫譜頭 依 p 抽樣")
    m = float(np.mean([np.mean(x == y) for x, y in zip(S_true, S_arg)]))
    print(f"\n  argmax 對教材的 top-1 一致率 {m * 100:.1f}%（同一批 chunk）")
    out["argmax_vs_true_top1"] = m
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
