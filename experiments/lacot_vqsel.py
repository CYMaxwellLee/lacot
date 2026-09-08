"""vq_select 寫譜頭：M2 —— 一顆小 head 臨場選步法字（B2 步法字典的可部署選字端）。

⭐ 它回答什麼：P2 證明了「每 chunk 貪心挑最靠近路標的字」把字典毀了（per-leg .063），
   teacher-relay 證明了「照 hindsight 真字串接力」走得到（per-leg .554）。
   ⇒ 中間缺的那一塊＝【臨場能不能自己寫出那條字串】。本檔提供該頭要用的字典側工具：
     ① 訓練標籤：encoder 對 act[t:t+seg_len] 算出的 hindsight 真字（cross-entropy 的 y）
     ② 執行端  ：decoder(字 ‖ 當下 obs) 展開 4 步動作
⛔ 本檔【不含】頭本身 —— 頭建在 scratch_lacot_rollout.py 裡（同 π_lo 紀律：
   sota_mlp 同容量、fork_rng 建構、獨立 optimizer），這樣它跟主模型的隔離證據
   才跟 lo 頭 v1 是同一份。

⛔ 旗預設 off：只有 LACOT_VQSEL_W>0 或 LACOT_SUB_POLICY=vq_select 才會被 import
   與初始化，關掉時本檔一行都不執行、既有檔名與 stdout 一字不變。

字典＝experiments/walk_verify/p0_train_dict_v1.py 訓出來的條件化 VQ-VAE
（decoder 吃「字 embedding ＋ 段起點完整 obs(29)」）；⭐ 教材驗證過的就是這一本
（P0 v1 50k、K=32 單碼本），⛔ 不是 gk_scan 的 G16K16。路徑由 LACOT_VQSEL_CKPT 指定。

⚠️ 與 wv_common.CondVQVAE 的關係：state_dict 完全相容（同 module 名、同形狀），
   這裡只保留推論路徑（encode_idx / decode_from_idx / decode_all_codes），
   刻意讓 worktree 的施工檔不依賴主 repo 的未追蹤檔案（同 lacot_vqo.py 的做法）。
"""
import os

import numpy as np
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class _VQEMA(nn.Module):
    """只有 buffer（推論不需要 EMA 更新）；buffer 名稱與形狀同 wv_common.VQEMA。"""

    def __init__(self, K, D):
        super().__init__()
        self.K, self.D = K, D
        self.register_buffer("embed", torch.randn(K, D) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", torch.zeros(K, D))
        self.register_buffer("unused_steps", torch.zeros(K, dtype=torch.long))
        self.register_buffer("restarts_total", torch.zeros(1, dtype=torch.long))

    def quantize_idx(self, z_e):
        """⚠️ 逐項照 wv_common.VQEMA.quantize_idx —— 這是【標籤的定義】，
        算式不一樣就等於換了一本字典，而它不會報錯。"""
        dist = (z_e.pow(2).sum(1, keepdim=True) - 2 * z_e @ self.embed.t()
                + self.embed.pow(2).sum(1))
        return dist.argmin(1)


class CondVQVAE(nn.Module):
    """字典 v1 的推論面：encoder(動作段)->z->VQ->字；decoder(字 ‖ 正規化 obs)->動作段。"""

    def __init__(self, seg_dim, latent_dim, K, obs_dim=29, hidden=512, **_):
        super().__init__()
        self.seg_dim, self.latent_dim, self.K, self.obs_dim = seg_dim, latent_dim, K, obs_dim
        self.encoder = _MLP(seg_dim, latent_dim, hidden)
        self.decoder = _MLP(latent_dim + obs_dim, seg_dim, hidden)
        self.vq = _VQEMA(K, latent_dim)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def norm_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def encode_idx(self, segs):
        """動作段 (B, seg_dim) → hindsight 真字 (B,)。⭐ 訓練標籤就是這支算的。"""
        return self.vq.quantize_idx(self.encoder(segs))

    def decode_from_idx(self, idx, obs):
        """字 index (B,) ＋ 原始 obs (B, obs_dim) → 動作段 (B, seg_dim)。執行端用這支。"""
        return self.decoder(torch.cat([self.vq.embed[idx], self.norm_obs(obs)], dim=1))

    def decode_all_codes(self, obs):
        """同一個 obs 展開全部 K 個字 → (K, seg_dim)（診斷用；⛔ M2 不做前瞻）。"""
        o = self.norm_obs(obs).expand(self.K, -1)
        return self.decoder(torch.cat([self.vq.embed, o], dim=1))


def load_dict(ckpt_path, device="cpu", verbose=True):
    """回 (model, cfg)。⚠️ strict load ⇒ 拿錯字典（K 不同／無條件版）會【當場炸】，
    ⛔ 不會靜默地用一本別的字典把標籤算成另一套。"""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = CondVQVAE(cfg["seg_dim"], cfg["latent_dim"], cfg["k"], obs_dim=cfg["obs_dim"],
                  hidden=cfg["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)          # ⛔ 字典是凍的：任何梯度都不准流回去
    if verbose:
        print(f"  🔮 vq_select 字典：{os.path.basename(ckpt_path)} "
              f"K={cfg['k']} seg_len={cfg['seg_len']} latent={cfg['latent_dim']} "
              f"obs_dim={cfg['obs_dim']} conditional_decoder={cfg.get('conditional_decoder')}",
              flush=True)
    return m, cfg


# ------------------------------------------------------------------ 弧長路標
def build_arc_cum(obs_xy, traj_end):
    """全域累積弧長（episode 邊界處增量歸零 ⇒ 全域非遞減、集內差值正確）。

    ⭐ 為什麼要全域單調：這樣 `np.searchsorted` 一次就能查「往前 a 單位弧長落在哪一格」，
       ⛔ 不用逐條 episode 迴圈（N≈1e6）。集外的答案由呼叫端 clamp 到 traj_end 擋掉。
    """
    xy = np.asarray(obs_xy, np.float64)
    step = np.zeros(len(xy), np.float64)
    step[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    step[1:][traj_end[:-1] != traj_end[1:]] = 0.0      # ⭐ 跨 episode 的那一步不算弧長
    return np.cumsum(step)


def arc_waypoint(arc_cum, traj_end, rows, arcs):
    """rows 各自往前 arcs 單位弧長的那個 index（超出 episode ⇒ clamp 到 traj_end）。

    ⚠️ clamp ⛔ 不重抽 —— 同 _lo_pairs 的 F6 教訓：重抽會把靠近軌跡結尾的起點
       系統性丟掉，而它不會報錯。
    """
    j = np.searchsorted(arc_cum, arc_cum[rows] + arcs, side="left")
    return np.minimum(j, traj_end[rows])


# ------------------------------------------------------------------ 選字統計
class CodeStats:
    """選了哪些字（perplexity / top1 / 全活）—— 同 P2 vq_oracle 的那張表，才比得起來。"""

    def __init__(self, K):
        self.K = int(K)
        self.counts = np.zeros(self.K, np.int64)
        self.n_chunks = 0

    def add(self, k):
        self.counts[int(k)] += 1
        self.n_chunks += 1

    def stats(self):
        c = self.counts
        fr = c / max(c.sum(), 1)
        nz = fr[fr > 0]
        return dict(n_chunks=int(self.n_chunks),
                    codes_active=int((c > 0).sum()), codes_K=self.K,
                    code_perplexity=float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
                    code_top1_frac=float(fr.max()) if len(fr) else 0.0,
                    code_counts=c.tolist())
