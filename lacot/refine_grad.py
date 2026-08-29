"""主人的更新式 —— 在 latent 上做 energy-guided 梯度修正。

    u ← u + η · [ −clip(∇u E_geo(u)) + λ · clip(∇u log p(u | s,g)) ]
        羅盤（往 energy 低的好路走）   結界（別走出 flow 認得的地方）

⭐ 2026-08-29 主人裁：叫 energy、不叫 value —— (a) 不跟 RL 的 value（expected return）
   混淆；(b) energy 框架自然接 score-based generative 一族（引擎未必永遠是 NF）。
   E 越小越好、沿 −∇E 下坡；與舊的 V=−E 版逐位等價。

🚨 出處：主人 2026-08-22 提的 value-directed refine。到 2026-08-28 之前一行都沒寫，
   而主線的 `refine` 是一顆【學出來的網路】—— 裡面沒有打分項、不是梯度修正，
   ⛔ 那不是這條式子。（8/26 實測：那顆網路反向跑反而追平 bc 地板 ⇒ 它是主動有害的。）

## 為什麼 energy 用【算的】不用【學的】

交接記著一條文獻結論：裸的「梯度優化一個學出來的評分器」至今沒有一篇成功 ——
學出來的評分一定有破綻，優化會去找那個破綻（找到分數好看、實際很爛的點）。

⇒ ⭐ 幾何算出來的 energy 沒有破綻可找：想騙它，就得真的走出一條不穿牆、真的到得了目標的短路。
⇒ 而這正是 pointmaze 給我們、而做語言模型的那批人沒有的牌 —— 路好不好是幾何可驗的。

## 佔據圖從【資料】蓋，⛔ 不從 env.maze_map 讀

理由不是潔癖：對標 SOTA 的時候，「你偷看了模擬器的地圖」是會被問的。
從 20k 條軌跡走過的位置蓋出來的圖，是 offline 資料本身就有的資訊。
（`debug_maze_map=True` 留給對照用，⛔ 不進正式結果。）
"""
import numpy as np
import torch
import torch.nn.functional as F


class GeoEnergy:
    """幾何 energy —— 四項全部對【座標點】可微，因此對 u 可微（經過 decoder）。

        E_geo = w_wall·穿牆 + w_goal·終點沒到 + w_start·起點不對 + w_len·路徑長度（越小越好）

    ⭐ 權重排序是主人 2026-08-26 的原話：「穿牆應該要有個大懲罰」⇒ w_wall 最大。
    ⚠️ w_len 最小是刻意的：先要求「走得通」，再要求「走得短」。
       （交接記著一條：validity 比 length 更該當第一個訊號。）
    """

    def __init__(self, obs_xy, mu, sd, res=8, device="cpu",
                 w_wall=10.0, w_goal=3.0, w_start=3.0, w_len=0.3):
        """obs_xy: [N,2] 資料集裡走過的所有位置（原始座標）。res: 每格切幾份。"""
        from scipy.ndimage import distance_transform_edt
        self.w = dict(wall=w_wall, goal=w_goal, start=w_start, length=w_len)
        self.device = device
        mu = np.asarray(mu, np.float64); sd = np.asarray(sd, np.float64)
        z = (np.asarray(obs_xy, np.float64) - mu) / sd          # 正規化座標
        self.lo = z.min(0) - 0.5
        self.hi = z.max(0) + 0.5
        span = self.hi - self.lo
        # ⭐ 網格解析度綁在資料的實際跨度上，⛔ 不寫死格數 —— medium 與 large 的跨度差很多
        self.shape = np.maximum((span * res).round().astype(int), 8)
        idx = ((z - self.lo) / span * (self.shape - 1)).round().astype(int)
        idx = np.clip(idx, 0, self.shape - 1)
        occ = np.zeros(self.shape, bool)
        occ[idx[:, 0], idx[:, 1]] = True                        # 走過的地方 ＝ 自由空間
        self.coverage = float(occ.mean())
        # 到「自由空間」的距離：自由格是 0，牆裡面越深越大
        dist = distance_transform_edt(~occ) / res               # 換算成「格」為單位
        self.dist = torch.tensor(dist, dtype=torch.float32, device=device)[None, None]
        self.lo_t = torch.tensor(self.lo, dtype=torch.float32, device=device)
        self.span_t = torch.tensor(span, dtype=torch.float32, device=device)

    def _sample(self, field, pts):
        """把 [1,1,H,W] 的場，用【同一條映射】採樣在 pts [B,T,2] 上 → [B,T]。

        ⛔ 映射邏輯逐字沿用原本的 wall_depth，⛔ 一個字都沒動
           （2026-08-28 reviewer 已驗證它是對的：格心 round-trip 誤差 5.4e-07）。
        ⭐ 抽出來只為了讓 `mapping_error()` 能拿【編號場】走同一條路 ——
           自己驗自己的映射，⛔ 不是另寫一份會分岔的複本。
        """
        lo = self.lo_t.to(pts.dtype); span = self.span_t.to(pts.dtype)
        uv = (pts - lo) / span                                   # → [0,1]
        grid = (uv * 2.0 - 1.0).flip(-1)[:, None]                # grid_sample 要 (x,y) 且 [-1,1]
        d = F.grid_sample(field.to(pts.dtype).expand(len(pts), -1, -1, -1), grid,
                          mode="bilinear", padding_mode="border", align_corners=True)
        return d[:, 0, 0]                                        # [B,T]

    def wall_depth(self, pts):
        """pts [B,T,2]（正規化座標）→ [B,T] 陷進牆裡多深（0 ＝ 在資料走過的地方）。"""
        return self._sample(self.dist, pts)

    # ─────────────────────────────────────────────────────────────
    # 健康檢查（2026-08-28 加）
    #
    # 🚨 為什麼要加：主線舊有的 sanity 是「真軌跡的穿牆深度中位 < 0.15」，而它
    #    【結構上必然通過】—— occ 是用 OBS 蓋的，探針的點又是 make_batch 從
    #    【同一批 OBS】內插出來的 ⇒ 穿牆深度恆為 0，跟映射對不對無關。
    #    `[實測]` 拿「覆蓋整個盒子」的資料建 GeoEnergy ⇒ 真軌跡穿牆中位 0.0000、
    #    舊 assert 照樣通過，但 ‖∂wall/∂pts‖ = 1.2e-03 ⇒ 穿牆項對爬坡毫無貢獻，
    #    E_geo 安靜地退化成只有 goal/start/length。
    # ⇒ 下面兩格才是真的擋得住的：
    #     mapping_error()  抓「映射寫歪」  —— 格心 round-trip
    #     health()         抓「牆這項是空的」—— 盒內隨機點必須真的被判成牆
    # ─────────────────────────────────────────────────────────────
    def mapping_error(self):
        """格心 round-trip：每一格的中心座標丟回採樣映射，看它落回原格（單位＝格）。

        ⭐ 用【編號場】當探針 ⇒ flip / align_corners / lo / span 任何一項寫錯都會叫。
        正確時 < 1e-4。
        """
        H, W = int(self.shape[0]), int(self.shape[1])
        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel()], 1).astype(np.float64)
        denom = np.maximum(np.asarray(self.shape, np.float64) - 1.0, 1.0)
        z = self.lo + idx / denom * (self.hi - self.lo)          # 格心的正規化座標
        pts = torch.tensor(z[None], dtype=torch.float64, device=self.device)   # [1, H*W, 2]
        err = 0.0
        for axis, want_np in ((0, ii), (1, jj)):
            field = torch.tensor(want_np.astype(np.float64), device=self.device)[None, None]
            got = self._sample(field, pts)[0]
            want = torch.tensor(want_np.ravel().astype(np.float64), device=self.device)
            err = max(err, float((got - want).abs().max()))
        return err

    def health(self, n=4096, seed=0, wall_min=0.05, cov_lo=0.05, cov_hi=0.80,
               map_tol=1e-4):
        """兩格結構性檢查 —— ⛔ 跟「真軌跡穿牆 ≈0」不同，這兩格【不是】恆真的。

        (a) 格心 round-trip 誤差 < map_tol      ⇒ 抓映射寫歪
        (b) 盒內【隨機】點的穿牆中位 > wall_min，且 cov_lo < coverage < cov_hi
            ⇒ 抓「全部判成自由空間」。`[實測]` real L-maze 0.61；全覆蓋的盒子 0.0000
        回 dict（含 ok 與 reasons），⛔ 由呼叫端決定要不要 assert。
        """
        rng = np.random.default_rng(seed)
        z = self.lo + rng.random((n, 2)) * (self.hi - self.lo)
        pts = torch.tensor(z[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            wd = float(self.wall_depth(pts).median())
        me = self.mapping_error()
        reasons = []
        if not (me < map_tol):
            reasons.append(f"格心 round-trip 誤差 {me:.2e} ≥ {map_tol:g} ⇒ 座標映射寫歪了")
        if not (wd > wall_min):
            reasons.append(f"盒內隨機點的穿牆中位 {wd:.4f} ≤ {wall_min:g} ⇒ 幾乎每個點都被判成"
                           f"自由空間 ⇒ 穿牆這一項是【空的】，E_geo 退化成 goal/start/length")
        if not (cov_lo < self.coverage < cov_hi):
            reasons.append(f"資料覆蓋 {self.coverage:.1%} 不在 ({cov_lo:.0%}, {cov_hi:.0%}) 之間"
                           f" ⇒ 佔據圖不是一張迷宮（太空或太滿）")
        return dict(ok=not reasons, mapping_err=me, wall_median_random=wd,
                    coverage=float(self.coverage), reasons=reasons)

    def __call__(self, pts, s, g, per_term=False):
        """pts [B,T,2]、s/g [B,2]（皆正規化座標）→ E [B]（energy，越【小】越好）。

        ⭐ 2026-08-29 主人裁：改稱 energy、符號翻正 —— 不跟 value（expected return）混淆，
           energy 框架也自然接 score-based generative 那一族（引擎未必永遠是 NF）。
           更新式同步翻成下坡：u ← u + η[−clip(∇E) + λ clip(∇log p)]，行為逐位等價。
        """
        wall = self.wall_depth(pts).mean(1)
        goal = (pts[:, -1] - g).norm(dim=-1)
        start = (pts[:, 0] - s).norm(dim=-1)
        length = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(1)
        e = (self.w["wall"] * wall + self.w["goal"] * goal
             + self.w["start"] * start + self.w["length"] * length)
        if per_term:
            return e, dict(wall=wall, goal=goal, start=start, length=length)
        return e


def grad_steps(R, has_warm, grad_r, grad_r_warm):
    """回 (要不要接上一個 chunk 的 u, 這個 chunk 爬幾步)。

    🚨 2026-08-28 修（整集凍住）：舊版 warm 分支只看 `_warm is not None`，⛔ 沒看 R。
       ⇒ R=0 時 `_steps = R*GRAD_R = 0`，而第一個 chunk 仍把【沒爬過的】u 寫進快取；
         之後每個 chunk 都走 warm 分支、也爬 0 步 ⇒ 每次 flow.sample 抽的新 u 全被丟掉
         ⇒ 整集用第 0 步那個 u，而 cond 已經換過幾十次。註解寫「R=0 ⇒ flow 直接用」，
         行為跟註解相反，⛔ 而且不會報錯。
    ⇒ R=0 的語意釘死成：不爬、不讀快取、⛔ 也不寫快取（呼叫端必須跳過 _GRAD_CACHE）。
    """
    if R <= 0:
        return False, 0
    if has_warm and grad_r_warm > 0:
        return True, int(R) * int(grad_r_warm)      # 接續上一個 chunk 的計畫
    return False, int(R) * int(grad_r)              # 第一個 chunk：從 flow 樣本起爬


def _clip(grad, mode="normalize", dims=(1, 2)):
    """把一項梯度限制成單位長度。

    ⭐ 主人 2026-08-26 的原話是「value guided trust region」。兩種讀法都合理，這裡預設 normalize：
       normalize  兩項【等權】相加 —— 解掉「E 的梯度比 log p 大兩個數量級」這種問題
       clamp      只在超過 1 時縮放 —— 純粹限制步長，兩項的相對量級保持原樣
    ⚠️ 這是一個【選擇】，⛔ 不是唯一正解。要換就換，但要記得結果不可跨模式比較。
    """
    n = grad.norm(dim=dims, keepdim=True).clamp_min(1e-8)
    return grad / n if mode == "normalize" else grad / n.clamp_min(1.0)


def grad_refine(u0, cond, decoder, flow, geo, s, g, steps=50, eta=0.1, lam=0.3,
                clip_mode="normalize", trace=False):
    """主人的更新式。u0 [B,K,D] → 爬完的 u [B,K,D]。

    ⚠️ 呼叫端如果在 `@torch.no_grad()` 裡面，這裡要自己開 enable_grad ——
       ⛔ 忘了會直接炸（⭐ 炸是好事，不會靜默地給零梯度）。
    """
    hist = []
    with torch.enable_grad():
        u = u0.detach().clone().requires_grad_(True)
        for i in range(steps):
            pts = decoder(u)
            e, terms = geo(pts, s, g, per_term=True)
            ge = torch.autograd.grad(e.sum(), u, retain_graph=True)[0]
            lp = flow.log_prob(u, cond)
            gp = torch.autograd.grad(lp.sum(), u)[0]
            # ⭐ 兩項【各自】限長再相加 —— 主人的 trust region。
            #    energy 走下坡（−∇E）、log p 走上坡（+∇log p）；normalize 下與舊 V 版逐位等價。
            step = -_clip(ge, clip_mode) + lam * _clip(gp, clip_mode)
            u = (u + eta * step).detach().requires_grad_(True)
            if trace:
                hist.append(dict(i=i, e=float(e.mean()), logp=float(lp.mean()),
                                 **{k: float(t.mean()) for k, t in terms.items()}))
    return (u.detach(), hist) if trace else u.detach()
