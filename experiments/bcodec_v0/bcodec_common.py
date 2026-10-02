"""行為字典 v0 共用模組（DESIGN-2026-09-11-behavior-codec.md §三 零件1）。

在證明什麼：壓 (s,a) 整段的行為字典，物理保真 ≥ 純 action 字典（見 docs/DESIGN
2026-09-11 §四 關1）。本檔只負責「字典本身」（資料切段＋模型＋訓練迴圈），量測
（關1 E_indep）在同目錄 bcodec_drift_eval.py，刻意分開＝訓練跟量測互不依賴。

改動最小化、能復用就復用：
  - 常數 ACT_DIM/OBS_DIM/EP_LEN/DATASET_NAME、段切法、_usage_stats、save_json
    ⇐ 直接 import experiments/gk_scan/gk_common.py（一行未改）。
  - MLP / VQEMA（單一 codebook，EMA + dead-code restart）
    ⇐ 直接 import experiments/walk_verify/wv_common.py（這是「現行 K32 單本」實際
    使用的實作——task 交代「單本 VQEMA 優先復用已存在的實作」指的就是這份，
    不是 gk_common 的 MultiVQEMA(G×K 平行頭，本輪明確不用)）。一行未改。

跟現行 K32（wv_common.CondVQVAE）的差異，只有這三處（DESIGN §三 零件1）：
  1. encoder 輸入：現行只吃 a_{t:t+4}（32 維）→ 這裡吃 concat(s_{t:t+4} 正規化,
     a_{t:t+4})（116+32=148 維）。
  2. decoder 輸出：現行只吐 â_{t:t+4}（32 維）→ 這裡吐 concat(â_{t:t+4},
     ŝ_{t+1:t+5} 正規化)（32+116=148 維）。
  3. loss：現行 recon + beta*commit → 這裡 recon_a + λ_s*recon_s + beta*commit。
  decoder 條件輸入维持不變＝正規化 s_t（段起點狀態），跟現行 K32/G16K16/G32K32
  同一慣例。

正規化：obs_mu/obs_sd 用段起點 obs_start（跟現行 gk_common.compute_obs_norm 同
一種算法、只用 train 段），套用到：decoder 條件 s_t、encoder 的狀態段 s_{t:t+4}
（逐時間步套用同一組 mu/sd）、以及 decoder 重建目標 ŝ_{t+1:t+5}（同一組 mu/sd）。
動作維持在原始尺度（antmaze 動作本來就在 [-1,1] 附近，不特別正規化，跟現行慣例
一致）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_HERE)
_GK_DIR = os.path.join(_EXP_DIR, "gk_scan")
_WV_DIR = os.path.join(_EXP_DIR, "walk_verify")
for _p in (_GK_DIR, _WV_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gk_common as gc  # noqa: E402 - 常數 / 段切法 / _usage_stats / save_json，未改動
import wv_common as wvc  # noqa: E402 - MLP / VQEMA（單本），未改動

ACT_DIM = gc.ACT_DIM
OBS_DIM = gc.OBS_DIM
EP_LEN = gc.EP_LEN
DATASET_NAME = gc.DATASET_NAME
save_json = gc.save_json


# ------------------------------------------------------------------
# 資料：沿用 gk_common.load_segments_with_obs 切 a_{t:t+4} + seg_starts/split，
# 只【追加】s_{t:t+4}（encoder 用）與 s_{t+1:t+5}（decoder 重建目標）。
# ------------------------------------------------------------------
def load_bcodec_segments(data_path, seg_len=4, split_seed=42, val_frac=0.1, split_by="trajectory"):
    base = gc.load_segments_with_obs(data_path, seg_len=seg_len, split_seed=split_seed, val_frac=val_frac,
                                     split_by=split_by)
    if base["fallback_segmentation_used"]:
        raise RuntimeError(
            "⛔ fallback_segmentation_used=True：非固定 grid episode，本檔 s_next 的邊界算法"
            "（idx0+seg_len 可能跨出該集）在這個資料集上未驗證過，停手回報，不要假裝能用。"
        )
    d = np.load(data_path)
    observations = np.asarray(d["observations"], dtype=np.float32)
    seg_starts = base["seg_starts"]
    idx = seg_starts[:, None] + np.arange(seg_len)[None, :]      # s_{t:t+4}
    idx_next = idx + 1                                            # s_{t+1:t+5}
    assert idx_next.max() < len(observations), (
        f"⛔ s_next 索引越界：max idx_next={idx_next.max()} len(observations)={len(observations)}"
    )
    s_seg = observations[idx].reshape(len(seg_starts), seg_len * OBS_DIM).astype(np.float32)
    s_next = observations[idx_next].reshape(len(seg_starts), seg_len * OBS_DIM).astype(np.float32)

    out = dict(base)
    out["a_seg"] = base["segs"]           # 現成的 a_{t:t+4}，改個名字比較直白
    out["s_seg"] = s_seg
    out["s_next"] = s_next
    return out


def _norm_seg(x_flat, obs_mu, obs_sd, seg_len, obs_dim):
    n = x_flat.shape[0]
    xr = x_flat.reshape(n, seg_len, obs_dim)
    xn = (xr - obs_mu.reshape(1, 1, obs_dim)) / obs_sd.reshape(1, 1, obs_dim)
    return xn.reshape(n, seg_len * obs_dim).astype(np.float32)


# ------------------------------------------------------------------
# 模型：encoder(s_seg_norm ‖ a_seg) -> 單本 VQEMA(K,D) -> decoder(z_q ‖ s_t_norm)
#       -> (â_{t:t+4}, ŝ_{t+1:t+5}_norm)
# ------------------------------------------------------------------
class BehaviorCodecVQVAE(nn.Module):
    def __init__(self, act_dim, obs_dim, seg_len, K, D=16, hidden=512, decay=0.99, dead_steps=500):
        super().__init__()
        self.act_dim, self.obs_dim, self.seg_len = act_dim, obs_dim, seg_len
        self.K, self.D = K, D
        self.a_out_dim = seg_len * act_dim
        self.s_out_dim = seg_len * obs_dim
        enc_in = seg_len * obs_dim + seg_len * act_dim
        self.encoder = wvc.MLP(enc_in, D, hidden)
        self.decoder = wvc.MLP(D + obs_dim, self.a_out_dim + self.s_out_dim, hidden)
        self.vq = wvc.VQEMA(K, D, decay=decay, dead_steps=dead_steps)

    def encode(self, s_seg_norm, a_seg):
        return self.encoder(torch.cat([s_seg_norm, a_seg], dim=1))

    def split_out(self, out):
        return out[:, : self.a_out_dim], out[:, self.a_out_dim :]

    def decode_codes(self, idx, obs_t_norm):
        z_q = self.vq.embed[idx]
        out = self.decoder(torch.cat([z_q, obs_t_norm], dim=1))
        return self.split_out(out)

    def forward(self, s_seg_norm, a_seg, obs_t_norm, training):
        z_e = self.encode(s_seg_norm, a_seg)
        z_q_st, idx, commit = self.vq(z_e, training=training)
        out = self.decoder(torch.cat([z_q_st, obs_t_norm], dim=1))
        a_hat, s_hat = self.split_out(out)
        return a_hat, s_hat, idx, commit


def chunked_forward(model, s_in, a_in, o_in, chunk=8192):
    """⛔ 踩過的坑（K8192 兩顆訓練 job OOM，見 handoff）：VQEMA.forward 內部
    dist/onehot 都是 (N, K) 矩陣；K=8192、N=225000（整個 train 切分一次做完
    forward）會炸到 ~7GB 一份、兩份疊起來單一 job 就頂到記憶體上限。改成分塊
    forward，一次只吃 chunk 筆，(chunk, K) 矩陣頂多幾百 MB，數值上跟一次做完
    完全等價（VQ 是逐樣本獨立運算，沒有跨樣本的 batch norm 之類東西）。訓練迴圈
    本身（minibatch=batch=512）不受影響，這裡只管「一次評完整個 split」的路徑。
    """
    a_hats, s_hats, idxs = [], [], []
    n = s_in.shape[0]
    with torch.no_grad():
        for i0 in range(0, n, chunk):
            sl = slice(i0, min(i0 + chunk, n))
            a_hat_c, s_hat_c, idx_c, _ = model(s_in[sl], a_in[sl], o_in[sl], training=False)
            a_hats.append(a_hat_c)
            s_hats.append(s_hat_c)
            idxs.append(idx_c)
    return torch.cat(a_hats), torch.cat(s_hats), torch.cat(idxs)


# ------------------------------------------------------------------
# 訓練＋量測（單一 (K, lam_s) 顆；沒有 plateau 自動延長——DESIGN 明講 50k 步/顆
# 分鐘級，不像 gk_scan 24 格那樣需要自動延長）
# ------------------------------------------------------------------
def train_and_eval(data, K, D=16, lam_s=1.0, steps=50000, batch=512, lr=3e-4, beta=0.25,
                    hidden=512, decay=0.99, dead_steps=500, seed=0, log_every=1000,
                    collapse_top1_thresh=0.5):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    a_seg, s_seg, s_next = data["a_seg"], data["s_seg"], data["s_next"]
    obs_start = data["obs_start"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    act_dim, obs_dim, seg_len = data["act_dim"], data["obs_dim"], data["seg_len"] if "seg_len" in data else 4

    obs_mu, obs_sd = gc.compute_obs_norm(obs_start, train_idx)  # (1,obs_dim)，只用 train 段算，跟現行慣例一致
    s_seg_norm_all = _norm_seg(s_seg, obs_mu, obs_sd, seg_len, obs_dim)
    s_next_norm_all = _norm_seg(s_next, obs_mu, obs_sd, seg_len, obs_dim)
    obs_start_norm_all = ((obs_start - obs_mu) / obs_sd).astype(np.float32)

    model = BehaviorCodecVQVAE(act_dim, obs_dim, seg_len, K, D=D, hidden=hidden,
                                decay=decay, dead_steps=dead_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def T(x):
        return torch.from_numpy(x).to(device)

    train_a, train_s, train_snext, train_o = T(a_seg[train_idx]), T(s_seg_norm_all[train_idx]), \
        T(s_next_norm_all[train_idx]), T(obs_start_norm_all[train_idx])
    val_a, val_s, val_snext, val_o = T(a_seg[val_idx]), T(s_seg_norm_all[val_idx]), \
        T(s_next_norm_all[val_idx]), T(obs_start_norm_all[val_idx])
    n_train = len(train_idx)

    batch_rng = np.random.default_rng(seed + 1)
    loss_curve = []  # [step, recon_a, recon_s, commit]
    t0 = time.time()

    for i in range(steps):
        bidx = batch_rng.integers(0, n_train, size=batch)
        a_hat, s_hat, _idx, commit = model(train_s[bidx], train_a[bidx], train_o[bidx], training=True)
        recon_a = F.mse_loss(a_hat, train_a[bidx])
        recon_s = F.mse_loss(s_hat, train_snext[bidx])
        loss = recon_a + lam_s * recon_s + beta * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % log_every == 0 or i == steps - 1:
            loss_curve.append([i, float(recon_a.item()), float(recon_s.item()), float(commit.item())])

    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        a_hat_tr, s_hat_tr, idx_tr = chunked_forward(model, train_s, train_a, train_o)
        recon_a_train = float(F.mse_loss(a_hat_tr, train_a).item())
        recon_s_train = float(F.mse_loss(s_hat_tr, train_snext).item())
        a_hat_val, s_hat_val, idx_val = chunked_forward(model, val_s, val_a, val_o)
        recon_a_val = float(F.mse_loss(a_hat_val, val_a).item())
        recon_s_val = float(F.mse_loss(s_hat_val, val_snext).item())
        all_codes = torch.cat([idx_tr, idx_val]).numpy()

    usage = gc._usage_stats(all_codes, K)
    collapsed = bool(usage["n_active_codes"] < K or usage["top1_usage_frac"] > collapse_top1_thresh)

    baseline_a_vec = a_seg[train_idx].mean(0, keepdims=True)
    baseline_mse_a_val = float(np.mean((a_seg[val_idx] - baseline_a_vec) ** 2))
    baseline_s_vec = s_next_norm_all[train_idx].mean(0, keepdims=True)
    baseline_mse_s_val = float(np.mean((s_next_norm_all[val_idx] - baseline_s_vec) ** 2))

    metrics = dict(
        K=K, D=D, lam_s=lam_s, steps=steps, train_seconds=train_seconds,
        recon_a_train=recon_a_train, recon_a_val=recon_a_val,
        recon_s_train=recon_s_train, recon_s_val=recon_s_val,
        baseline_mse_a_val=baseline_mse_a_val, baseline_mse_s_val=baseline_mse_s_val,
        val_improve_a_pct=100.0 * (1.0 - recon_a_val / baseline_mse_a_val),
        val_improve_s_pct=(100.0 * (1.0 - recon_s_val / baseline_mse_s_val)) if lam_s > 0 else None,
        commit_final=loss_curve[-1][3] if loss_curve else None,
        perplexity=usage["perplexity"], n_active_codes=usage["n_active_codes"],
        top1_usage_frac=usage["top1_usage_frac"], top3_usage_frac=usage["top3_usage_frac"],
        collapsed=collapsed, collapse_top1_thresh=collapse_top1_thresh,
    )
    results = dict(metrics=metrics, loss_curve=loss_curve, usage=usage)
    extra = dict(obs_mu=obs_mu, obs_sd=obs_sd)
    return model, results, extra


def load_bcodec_ckpt(ckpt_path, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = BehaviorCodecVQVAE(cfg["act_dim"], cfg["obs_dim"], cfg["seg_len"], cfg["K"], D=cfg["D"],
                                hidden=cfg["hidden"], decay=cfg["decay"], dead_steps=cfg["dead_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    obs_mu = torch.from_numpy(np.asarray(ck["obs_mu"], dtype=np.float32))
    obs_sd = torch.from_numpy(np.asarray(ck["obs_sd"], dtype=np.float32))
    return model, cfg, obs_mu, obs_sd


# ====================================================================
# 補跑（主人 2026-09-12 裁）：平行頭版（G×K，跟昨天 G16K16/G32K32 完全同容量）
# ×(s,a) 輸入／雙頭輸出——「同容量下，行為詞 (s,a) 是否 >= 動作詞」。
# G×K 本體（MultiVQEMA／單組 VQEMA）一行未改，直接 import gk_common；跟
# BehaviorCodecVQVAE 唯一差異＝把單本 wvc.VQEMA 換成 gc.MultiVQEMA(G,K,D)，
# encoder/decoder 的 (s,a)輸入／雙頭輸出改法跟 v0 一模一樣。
# ====================================================================
class BehaviorCodecFactorizedVQVAE(nn.Module):
    def __init__(self, act_dim, obs_dim, seg_len, G, K, D=None, hidden=512, decay=0.99, dead_steps=500):
        super().__init__()
        D = gc.GROUP_DIM if D is None else D  # 預設跟昨天 G16K16/G32K32 一樣＝8，容量才對得齊
        self.act_dim, self.obs_dim, self.seg_len = act_dim, obs_dim, seg_len
        self.G, self.K, self.D = G, K, D
        self.a_out_dim = seg_len * act_dim
        self.s_out_dim = seg_len * obs_dim
        enc_in = seg_len * obs_dim + seg_len * act_dim
        self.encoder = gc.MLP(enc_in, G * D, hidden)
        self.decoder = gc.MLP(G * D + obs_dim, self.a_out_dim + self.s_out_dim, hidden)
        self.vq = gc.MultiVQEMA(G, K, D, decay=decay, dead_steps=dead_steps)  # 跟昨天完全同檔，未改

    def encode(self, s_seg_norm, a_seg):
        return self.encoder(torch.cat([s_seg_norm, a_seg], dim=1))

    def split_out(self, out):
        return out[:, : self.a_out_dim], out[:, self.a_out_dim :]

    def decode_codes(self, idx_full, obs_t_norm):
        """idx_full: (B, G)，跟 gk_common.FactorizedCondVQVAE.decode_codes 同一種查表法。"""
        z_q_chunks = [self.vq.groups[g].embed[idx_full[:, g]] for g in range(self.G)]
        z_q = torch.cat(z_q_chunks, dim=1)
        out = self.decoder(torch.cat([z_q, obs_t_norm], dim=1))
        return self.split_out(out)

    def forward(self, s_seg_norm, a_seg, obs_t_norm, training):
        z_e = self.encode(s_seg_norm, a_seg)
        z_q_full, idx_full, commit = self.vq(z_e, training=training)
        out = self.decoder(torch.cat([z_q_full, obs_t_norm], dim=1))
        a_hat, s_hat = self.split_out(out)
        return a_hat, s_hat, idx_full, commit


def train_and_eval_gk(data, G, K, D=None, lam_s=1.0, steps=50000, batch=512, lr=3e-4, beta=0.25,
                       hidden=512, decay=0.99, dead_steps=500, seed=0, log_every=1000,
                       collapse_top1_thresh=0.5):
    """跟 train_and_eval 幾乎同款迴圈，只換模型類別＋usage stats 要逐 group 算
    （照抄 gk_common.train_and_eval 的 per_group 迴圈，同一種算法）。"""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    D = gc.GROUP_DIM if D is None else D

    a_seg, s_seg, s_next = data["a_seg"], data["s_seg"], data["s_next"]
    obs_start = data["obs_start"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    act_dim, obs_dim, seg_len = data["act_dim"], data["obs_dim"], data.get("seg_len", 4)

    obs_mu, obs_sd = gc.compute_obs_norm(obs_start, train_idx)
    s_seg_norm_all = _norm_seg(s_seg, obs_mu, obs_sd, seg_len, obs_dim)
    s_next_norm_all = _norm_seg(s_next, obs_mu, obs_sd, seg_len, obs_dim)
    obs_start_norm_all = ((obs_start - obs_mu) / obs_sd).astype(np.float32)

    model = BehaviorCodecFactorizedVQVAE(act_dim, obs_dim, seg_len, G, K, D=D, hidden=hidden,
                                          decay=decay, dead_steps=dead_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def T(x):
        return torch.from_numpy(x).to(device)

    train_a, train_s, train_snext, train_o = T(a_seg[train_idx]), T(s_seg_norm_all[train_idx]), \
        T(s_next_norm_all[train_idx]), T(obs_start_norm_all[train_idx])
    val_a, val_s, val_snext, val_o = T(a_seg[val_idx]), T(s_seg_norm_all[val_idx]), \
        T(s_next_norm_all[val_idx]), T(obs_start_norm_all[val_idx])
    n_train = len(train_idx)

    batch_rng = np.random.default_rng(seed + 1)
    loss_curve = []
    t0 = time.time()

    for i in range(steps):
        bidx = batch_rng.integers(0, n_train, size=batch)
        a_hat, s_hat, _idx, commit = model(train_s[bidx], train_a[bidx], train_o[bidx], training=True)
        recon_a = F.mse_loss(a_hat, train_a[bidx])
        recon_s = F.mse_loss(s_hat, train_snext[bidx])
        loss = recon_a + lam_s * recon_s + beta * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % log_every == 0 or i == steps - 1:
            loss_curve.append([i, float(recon_a.item()), float(recon_s.item()), float(commit.item())])

    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        a_hat_tr, s_hat_tr, idx_tr = chunked_forward(model, train_s, train_a, train_o)
        recon_a_train = float(F.mse_loss(a_hat_tr, train_a).item())
        recon_s_train = float(F.mse_loss(s_hat_tr, train_snext).item())
        a_hat_val, s_hat_val, idx_val = chunked_forward(model, val_s, val_a, val_o)
        recon_a_val = float(F.mse_loss(a_hat_val, val_a).item())
        recon_s_val = float(F.mse_loss(s_hat_val, val_snext).item())
        all_codes = torch.cat([idx_tr, idx_val]).numpy()  # (N, G)

    per_group = []
    any_collapsed = False
    for g in range(G):
        stats = gc._usage_stats(all_codes[:, g], K)
        collapsed_g = (stats["n_active_codes"] < K) or (stats["top1_usage_frac"] > collapse_top1_thresh)
        stats["collapsed"] = bool(collapsed_g)
        any_collapsed = any_collapsed or collapsed_g
        per_group.append(stats)
    mean_perplexity = float(np.mean([s["perplexity"] for s in per_group]))
    mean_active = float(np.mean([s["n_active_codes"] for s in per_group]))
    mean_top1 = float(np.mean([s["top1_usage_frac"] for s in per_group]))

    baseline_a_vec = a_seg[train_idx].mean(0, keepdims=True)
    baseline_mse_a_val = float(np.mean((a_seg[val_idx] - baseline_a_vec) ** 2))
    baseline_s_vec = s_next_norm_all[train_idx].mean(0, keepdims=True)
    baseline_mse_s_val = float(np.mean((s_next_norm_all[val_idx] - baseline_s_vec) ** 2))

    metrics = dict(
        G=G, K=K, D=D, lam_s=lam_s, steps=steps, train_seconds=train_seconds,
        recon_a_train=recon_a_train, recon_a_val=recon_a_val,
        recon_s_train=recon_s_train, recon_s_val=recon_s_val,
        baseline_mse_a_val=baseline_mse_a_val, baseline_mse_s_val=baseline_mse_s_val,
        val_improve_a_pct=100.0 * (1.0 - recon_a_val / baseline_mse_a_val),
        val_improve_s_pct=(100.0 * (1.0 - recon_s_val / baseline_mse_s_val)) if lam_s > 0 else None,
        commit_final=loss_curve[-1][3] if loss_curve else None,
        mean_perplexity=mean_perplexity, mean_active_codes=mean_active, mean_top1_usage_frac=mean_top1,
        any_group_collapsed=bool(any_collapsed), collapse_top1_thresh=collapse_top1_thresh,
        bits_per_chunk=float(G * np.log2(K)),
    )
    results = dict(metrics=metrics, loss_curve=loss_curve, per_group_usage=per_group)
    extra = dict(obs_mu=obs_mu, obs_sd=obs_sd)
    return model, results, extra


def load_bcodec_gk_ckpt(ckpt_path, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = BehaviorCodecFactorizedVQVAE(cfg["act_dim"], cfg["obs_dim"], cfg["seg_len"], cfg["G"], cfg["K"],
                                          D=cfg["D"], hidden=cfg["hidden"], decay=cfg["decay"],
                                          dead_steps=cfg["dead_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    obs_mu = torch.from_numpy(np.asarray(ck["obs_mu"], dtype=np.float32))
    obs_sd = torch.from_numpy(np.asarray(ck["obs_sd"], dtype=np.float32))
    return model, cfg, obs_mu, obs_sd
