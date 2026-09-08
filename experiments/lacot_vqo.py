"""vq_oracle 執行端：B2 步法字典的「作弊選字上界」（P2）。

⭐ 它回答什麼：conf2 的【選路標】一行不動，只把【開車的頭】換成
   「枚舉 32 個步法字 → 各用還原機展開 4 步 → 在模擬器裡前瞻執行 →
     選終點離當前路標最近的字」實際執行。
⇒ 這是【字典＋完美選字】的實戰天花板，⛔ 不是可部署方法（用了模擬器前瞻）。

⛔ 旗預設 off：只有 LACOT_SUB_POLICY=vq_oracle 才會被 import 與初始化，
   關掉時本檔一行都不執行、既有檔名與 stdout 一字不變。

字典 = experiments/walk_verify/p0_train_dict_v1.py 訓出來的條件化 VQ-VAE
（decoder 吃「字 embedding ＋ 段起點完整 obs(29)」）。ckpt 路徑由
LACOT_VQO_CKPT 指定。

⚠️ 前瞻的紀律（實作時逐條擋掉的雷）：
  1. 前瞻走 env.unwrapped.step ⇒ 不碰 TimeLimit._elapsed_steps（wrapper 的步數計）。
  2. 前瞻不消耗任何 torch/numpy RNG（純查表＋物理），配對臂的隨機流不受影響。
  3. 存檔/還原 qpos, qvel, act, time, qacc_warmstart —— warmstart 不還原的話
     真正執行那一步會跟前瞻預測有微小差異。
  4. medium-stitch 沒有 teleport（maze.py:89 _teleport_info=None）⇒ step 無隨機性。
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
    def __init__(self, K, D, decay=0.99, eps=1e-5, dead_steps=500):
        super().__init__()
        self.K, self.D = K, D
        self.register_buffer("embed", torch.randn(K, D) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", torch.zeros(K, D))
        self.register_buffer("unused_steps", torch.zeros(K, dtype=torch.long))
        self.register_buffer("restarts_total", torch.zeros(1, dtype=torch.long))


class CondVQVAE(nn.Module):
    """與 experiments/walk_verify/wv_common.py 的 CondVQVAE 同構（state_dict 相容）。

    這裡只保留 P2 需要的推論路徑（decode_all_codes），不含訓練用的 EMA 更新，
    刻意讓 worktree 的施工檔不依賴主 repo 的未追蹤檔案。
    """

    def __init__(self, seg_dim, latent_dim, K, obs_dim=29, hidden=512, **_):
        super().__init__()
        self.seg_dim, self.latent_dim, self.K, self.obs_dim = seg_dim, latent_dim, K, obs_dim
        self.encoder = _MLP(seg_dim, latent_dim, hidden)
        self.decoder = _MLP(latent_dim + obs_dim, seg_dim, hidden)
        self.vq = _VQEMA(K, latent_dim)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def decode_all_codes(self, obs):
        """obs (1, obs_dim) → (K, seg_dim)：同一個狀態展開全部 K 個字。"""
        o = ((obs - self.obs_mean) / self.obs_std).expand(self.K, -1)
        return self.decoder(torch.cat([self.vq.embed, o], dim=1))


def load_dict(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = CondVQVAE(cfg["seg_dim"], cfg["latent_dim"], cfg["k"], obs_dim=cfg["obs_dim"],
                  hidden=cfg["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    return m, cfg


# ---------------------------------------------------------------- 模擬器狀態
def save_state(u):
    d = u.data
    return (np.asarray(d.qpos).copy(), np.asarray(d.qvel).copy(),
            np.asarray(d.act).copy() if d.act.size else None,
            float(d.time), np.asarray(d.qacc_warmstart).copy())


def restore_state(u, st):
    qpos, qvel, act, t, ws = st
    u.set_state(qpos, qvel)          # 內含 mj_forward
    if act is not None and u.data.act.size:
        u.data.act[:] = act
    u.data.time = t
    u.data.qacc_warmstart[:] = ws


class VQOraclePolicy:
    """每個 chunk：枚舉 K 個字 → 展開 → 前瞻執行 → 選終點離路標最近的字。"""

    def __init__(self, env, ckpt_path, chunk, act_dim, device="cpu", verbose=True):
        self.env = env
        self.u = env.unwrapped
        self.model, self.cfg = load_dict(ckpt_path, device)
        self.device = device
        self.K = int(self.cfg["k"])
        self.seg_len = int(self.cfg["seg_len"])
        self.act_dim = int(act_dim)
        assert self.seg_len == int(chunk), (
            f"⛔ 字典段長 {self.seg_len} != eval CHUNK {chunk} —— 前瞻與執行必須對齊")
        assert self.cfg["seg_dim"] == self.seg_len * self.act_dim
        self.n_calls = 0
        self.n_lookahead_steps = 0
        self.code_counts = np.zeros(self.K, dtype=np.int64)
        self.trace = None
        if verbose:
            print(f"  🔮 vq_oracle：字典 {os.path.basename(ckpt_path)} "
                  f"K={self.K} seg_len={self.seg_len} obs_dim={self.cfg['obs_dim']} "
                  f"conditional_decoder={self.cfg.get('conditional_decoder')}", flush=True)

    def start_trace(self):
        self.trace = dict(qpos=[], qvel=[], actions=[], codes=[], subs=[])

    def pop_trace(self):
        t, self.trace = self.trace, None
        return t

    @torch.no_grad()
    def __call__(self, obs, w_xy):
        u = self.u
        obs = np.asarray(obs, np.float64)
        w = np.asarray(w_xy, np.float64)[:2]
        cand = self.model.decode_all_codes(
            torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
        ).cpu().numpy().reshape(self.K, self.seg_len, self.act_dim)
        cand = np.clip(cand, -1.0, 1.0).astype(np.float64)

        st = save_state(u)
        best_k, best_d = 0, np.inf
        for k in range(self.K):
            if k:
                restore_state(u, st)
            for t in range(self.seg_len):
                u.step(cand[k, t])       # ⛔ unwrapped：不動 TimeLimit 的步數計
            self.n_lookahead_steps += self.seg_len
            d = float(np.linalg.norm(np.asarray(u.data.qpos)[:2] - w))
            if d < best_d:
                best_d, best_k = d, k
        restore_state(u, st)

        self.n_calls += 1
        self.code_counts[best_k] += 1
        if self.trace is not None:
            self.trace["qpos"].append(st[0][:15].copy())
            self.trace["qvel"].append(st[1][:14].copy())
            self.trace["actions"].append(cand[best_k].copy())
            self.trace["codes"].append(int(best_k))
            self.trace["subs"].append(w.copy())
        return cand[best_k].astype(np.float32)

    def stats(self):
        c = self.code_counts
        fr = c / max(c.sum(), 1)
        nz = fr[fr > 0]
        return dict(n_chunks=int(self.n_calls),
                    n_lookahead_steps=int(self.n_lookahead_steps),
                    codes_active=int((c > 0).sum()), codes_K=int(self.K),
                    code_perplexity=float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0,
                    code_top1_frac=float(fr.max()) if len(fr) else 0.0,
                    code_counts=c.tolist())
