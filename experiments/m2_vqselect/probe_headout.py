#!/usr/bin/env python
"""M2 補洞探針：寫譜頭在【沒進過訓練的 held-out 集】上的 top-1 accuracy。

⭐ 為什麼要它：訓練 log 的 top1 量的是【訓練分佈上的】—— 它不回答「頭是背下來的
   還是學到規律」。這支拿官方 held-out val 檔（antmaze-medium-stitch-v0-val.npz，
   完全沒進過主模型或字典的訓練）重量一次，兩個數字並排才講得清楚。
⚠️ 亂猜線【不是】1/K：教材的類別分佈不均勻 ⇒ 最強的常數預測器＝最大類的比例。
   本支把「最大類」與「1/K」兩條線都印出來，⛔ 不讓判讀偷用比較低的那條。

⛔ CPU-only、⛔ 不碰 GPU、⛔ 不寫任何既有檔案。
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))       # experiments/
import lacot_vqsel as vqsel  # noqa: E402

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
ENV_NAME = "antmaze-medium-stitch-v0"
CHUNK = 4


def sota_mlp(i, h, o, n=2):
    """⚠️ 逐行照 scratch_lacot_rollout.py 的同名函式 —— 形狀不同 load_state_dict 會炸。"""
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]
        p = h
    return nn.Sequential(*L, nn.Linear(p, o))


def load_split(path):
    d = np.load(path)
    obs = np.asarray(d["observations"], np.float32)
    act = np.asarray(d["actions"], np.float32)
    term = np.asarray(d["terminals"], bool)
    n = obs.shape[0]
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])
    te = np.empty(n, np.int64)
    for s0, e0 in zip(starts, ends):
        te[s0:e0 + 1] = e0
    assert ends[-1] == n - 1, f"⛔ {path} 最後一筆不是 terminal"
    return obs, act, te


def eval_split(name, obs, act, te, dic, head_fn, mu, sd, arcmin, arc, K, n_pairs, seed):
    arccum = vqsel.build_arc_cum(obs[:, :2], te)
    valid = (te - np.arange(len(obs))) >= CHUNK
    rng = np.random.default_rng(seed)
    rows = np.empty(n_pairs, np.int64)
    f = 0
    while f < n_pairs:
        r = rng.integers(0, len(obs), n_pairs - f)
        r = r[valid[r]]
        rows[f:f + len(r)] = r
        f += len(r)
    arcs = rng.uniform(arcmin, arc, n_pairs)
    tgts = vqsel.arc_waypoint(arccum, te, rows, arcs)
    with torch.no_grad():
        segs = act[rows[:, None] + np.arange(CHUNK)[None, :]].reshape(n_pairs, -1)
        y = dic.encode_idx(torch.from_numpy(np.ascontiguousarray(segs, np.float32)))
        s = torch.from_numpy(np.ascontiguousarray((obs[rows] - mu) / sd, np.float32))
        w = torch.from_numpy(np.ascontiguousarray(
            (obs[tgts, :2].astype(np.float64) - obs[rows, :2].astype(np.float64)) / arc, np.float32))
        lg = head_fn(s, w)
        ce = float(torch.nn.functional.cross_entropy(lg, y))
        top1 = float((lg.argmax(1) == y).float().mean())
        top5 = float((lg.topk(5, 1).indices == y[:, None]).any(1).float().mean())
        pred = lg.argmax(1).cpu().numpy()
    hist = np.bincount(y.cpu().numpy(), minlength=K) / n_pairs
    ph = np.bincount(pred, minlength=K) / n_pairs
    nz = ph[ph > 0]
    print(f"\n=== {name}（n={n_pairs}）===")
    print(f"  cross-entropy      {ce:.4f}        （均勻亂猜 ln{K} = {np.log(K):.4f}）")
    print(f"  top-1 accuracy     {top1 * 100:.2f}%")
    print(f"  top-5 accuracy     {top5 * 100:.2f}%")
    print(f"  ⭐ 亂猜線（最大類）  {hist.max() * 100:.2f}%   ⛔ 不是 1/{K}={100 / K:.2f}%"
          f"  ⇒ 倍率 {top1 / max(hist.max(), 1e-9):.2f}×")
    print(f"  頭的輸出分佈：用到 {int((ph > 0).sum())}/{K} 個字、"
          f"perplexity {np.exp(-np.sum(nz * np.log(nz))):.2f}、top1 佔 {ph.max() * 100:.1f}%"
          f"   （教材側 perplexity {np.exp(-np.sum(hist[hist > 0] * np.log(hist[hist > 0]))):.2f}）")
    return dict(n=n_pairs, ce=ce, top1=top1, top5=top5, majority=float(hist.max()),
                pred_active=int((ph > 0).sum()),
                pred_perplexity=float(np.exp(-np.sum(nz * np.log(nz)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pairs", type=int, default=200000)
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "results", "m2_headout.json"))
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert "vqsel" in ck, f"⛔ 這顆 ckpt 沒有 vqsel 段：{a.ckpt}"
    vs, cfg = ck["vqsel"], ck["cfg"]
    dic, dcfg = vqsel.load_dict(vs["dict_ckpt"], device="cpu")
    K = int(vs["K"])
    arc = float(cfg.get("VQSEL_ARC", 7.5))
    arcmin = float(cfg.get("VQSEL_ARCMIN", 1.875))
    print(f"ckpt   : {os.path.basename(a.ckpt)}")
    print(f"字典   : {os.path.basename(vs['dict_ckpt'])}  K={K}")
    print(f"弧長窗 : [{arcmin:g}, {arc:g}]   訓練末 EMA: ce {cfg.get('VQSEL_CE_EMA')}"
          f"  top1 {cfg.get('VQSEL_ACC_EMA')}")

    obs_dim = dcfg["obs_dim"]
    s_enc = sota_mlp(obs_dim, 512, 512)
    w_enc = sota_mlp(2, 512, 512)
    ch = sota_mlp(1024, 512, int(cfg["COND"]))
    hd = sota_mlp(int(cfg["COND"]), 512, K, n=3)
    s_enc.load_state_dict(vs["s"]); w_enc.load_state_dict(vs["w"])
    ch.load_state_dict(vs["ch"]); hd.load_state_dict(vs["head"])
    for m in (s_enc, w_enc, ch, hd):
        m.eval()

    def head_fn(s, w):
        return hd(ch(torch.cat([s_enc(s), w_enc(w)], 1)))

    tr_obs, tr_act, tr_te = load_split(os.path.join(DD, f"{ENV_NAME}.npz"))
    mu, sd = tr_obs.mean(0), tr_obs.std(0) + 1e-6      # ⚠️ 一定用【訓練切分】的統計
    va_obs, va_act, va_te = load_split(os.path.join(DD, f"{ENV_NAME}-val.npz"))
    print(f"train {tr_obs.shape}  val {va_obs.shape}（官方 held-out，⛔ 沒進過任何訓練）")

    out = {"ckpt": a.ckpt, "dict": vs["dict_ckpt"], "K": K, "arc": [arcmin, arc]}
    out["train"] = eval_split("訓練切分（跟訓練同分佈）", tr_obs, tr_act, tr_te, dic, head_fn,
                              mu, sd, arcmin, arc, K, a.n_pairs, 20260908)
    out["val"] = eval_split("官方 held-out val（沒進過訓練）", va_obs, va_act, va_te, dic, head_fn,
                            mu, sd, arcmin, arc, K, a.n_pairs, 20260909)
    d = out["train"]["top1"] - out["val"]["top1"]
    print(f"\n⇒ train − val 的 top-1 差距 {d * 100:+.2f} 個百分點"
          f"（{'沒有過擬合的跡象' if abs(d) < 0.02 else '⚠️ 差距不小，判讀要打折'}）")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import json
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
