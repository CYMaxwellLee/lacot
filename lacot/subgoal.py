"""兩層規劃：長程想幾何、短程走路。

🚨 起因：主人 2026-08-28 的質疑 ——
   「我們目前都是要一次就把整條 path 根據起點終點一次規劃好，⋯⋯
     如果沒搞頭，那可能就是走一段 action chunk 或者到 subgoal，然後繼續 think again」

`[實測]` 他指到的東西是真的，而且形狀比「一次規劃太難」更具體：

    訓練時 goal 在【同一條軌跡內】均勻抽（scratch_lacot_rollout.py:78-92，官方 GCBC 抽法）
    ⇒ u 被要求表示的路，上界就是單條軌跡
    eval 時 goal 是【最終目標】，整集固定（:352-358）
    ⇒ 而 stitch 最難那層的題比任何一條訓練軌跡都長
    ⇒ ⭐ 「一次規劃整條路」這件事，訓練資料裡從來沒有出現過任何示範

## 為什麼分兩層就解得掉，而且不會咬到自己的尾巴

ルナ第一版想的是「從 decode(u) 上取 subgoal」，但馬上看到一個循環：
u 要靠 cond=(現在, 目標) 生出來，若要它只想一小段，cond 就得換成 (現在, subgoal)，
⇒ 而 subgoal 又要先有 u 才解得出來。

⇒ ⭐ 解法是把它拆成【階層】而不是遞迴（Fable 5 的形狀，ルナ複驗後同意）：

    長程層  cond = (現在, 最終目標)。u_long 由 flow 起手、V_geo 爬。
            ⛔ 它不出動作、head 不吃它 —— 唯一的產出是 decode(u_long) 的幾何形狀。
    短程層  cond = (現在, subgoal)。flow → 爬 → head 出動作，全部落回訓練分布內。

⇒ 沒有循環：上層只供幾何、下層只管執行。
⇒ 而且對 u_long 的要求，從「處處準且 head 讀得動」降到「解出來的路大致合法」——
   ⭐ 而「大致合法」恰好就是 V_geo 在最佳化的東西（牆、端點、連續）。
"""
import numpy as np
import torch


def arc_subgoal(pts, delta, ret_index=False):
    """沿 decode 出來的路徑取【弧長 ≈ delta】的那個點當 subgoal。

    🚨 ⛔ 不能取「第 k 個點」—— 那 128 個點是按【時間】均勻攤的，
       同一個 k 在長路與短路上代表完全不同的距離 ⇒ 短程 cond 的分布會跟著路長飄。
    ⭐ 取弧長就穩：不管整條路多長，subgoal 永遠落在「離現在約 delta」的地方
       ⇒ 短程層永遠坐在訓練分布最肥的位置。

    pts   [B, T, 2] decode 出來的座標序列
    delta 目標弧長（與 pts 同單位）
    回傳  [B, 2]（ret_index=True 時另回傳選中的索引 [B]）
    """
    seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)          # [B, T-1]
    cum = torch.cumsum(seg, dim=1)                          # [B, T-1] 累積弧長
    # 第一個累積弧長 ≥ delta 的點；整條路都比 delta 短就取終點
    reach = (cum >= delta)
    idx = torch.where(reach.any(1), reach.float().argmax(1) + 1,
                      torch.full((len(pts),), pts.shape[1] - 1, device=pts.device))
    sg = pts[torch.arange(len(pts), device=pts.device), idx]
    return (sg, idx) if ret_index else sg


class SubgoalPlanner:
    """管「什麼時候該重想」與「subgoal 在哪」。⛔ 不管動作 —— 那是短程層的事。

    三個觸發（缺一不可，理由各自不同）：
      到了      離 subgoal < rho ⇒ 換下一個。這是正常節奏。
      走太久    用滿 cap 個 chunk ⇒ 強制重想。防「追一個永遠到不了的 subgoal」。
      卡住      連續 stuck_m 個 chunk 沒有更靠近 ⇒ 強制重想。
                ⭐ 這格擋的是貪心追 subgoal 造成的來回震盪 —— ⛔ 上面兩格都擋不到它。

    🚨 2026-08-28 修（單位錯）：`observe()` 是【每個 chunk】被呼叫一次，⛔ 不是每個 env step
       —— 呼叫端是 policy，而 policy 一次回 CHUNK 步。舊 docstring 三處都寫「步」，
       於是 cap=40 讀起來像 40 步、實際是 40×CHUNK＝160 個 env step（CHUNK=4）
       ⇒ 兩個「強制重想」的觸發根本按不下去，而它【不會報錯】。
       ⇒ 現在單位明確定義成【chunk 數】，預設同步改成 cap 10（＝40 步）、stuck_m 3（＝12 步）。
       ⚠️ 建構時傳 chunk= 只為了把換算印得出來，⛔ 它不參與任何判斷。
    """

    def __init__(self, delta_sub, rho=None, cap=10, stuck_m=3, stuck_eps=1e-3, chunk=1):
        self.delta_sub = float(delta_sub)
        self.rho = float(rho) if rho is not None else 0.25 * float(delta_sub)
        self.cap, self.stuck_m, self.stuck_eps = int(cap), int(stuck_m), float(stuck_eps)
        self.chunk = int(chunk)          # ⭐ 只用來換算成 env step，⛔ 不進判斷
        self.reset()

    @property
    def cap_steps(self):
        """cap 換算成 env step ⇒ 報告用。⛔ 判斷一律用 chunk 數。"""
        return self.cap * self.chunk

    @property
    def stuck_steps(self):
        return self.stuck_m * self.chunk

    def reset(self):
        self.sub = None
        self.since = 0          # 這個 subgoal 用了幾個 chunk
        self.best = float("inf")   # 這個 subgoal 期間內最近的距離
        self.stuck = 0
        self.n_set = 0          # 設過幾次 subgoal（含第一次）
        self.n_replan = 0       # ⭐ 診斷用：整集【重想】了幾次（⛔ 第一次不算重想）

    def observe(self, s_xy):
        """走完一個 chunk 之後餵當前位置。回傳 True 代表「該重想了」。"""
        if self.sub is None:
            return True
        d = float(np.linalg.norm(np.asarray(s_xy) - self.sub))
        self.since += 1
        if d < self.best - self.stuck_eps:
            self.best, self.stuck = d, 0
        else:
            self.stuck += 1
        return (d < self.rho) or (self.since >= self.cap) or (self.stuck >= self.stuck_m)

    def set(self, sub_xy):
        # 🚨 舊版把「第一次設 subgoal」也算成重想 ⇒ n_replan 永遠多 1。
        #    ⛔ 那個 +1 不會報錯，只會讓「重想頻率」這個診斷數字系統性偏高。
        if self.sub is not None:
            self.n_replan += 1
        self.n_set += 1
        self.sub = np.asarray(sub_xy, np.float32)
        self.since, self.best, self.stuck = 0, float("inf"), 0
        return self.sub


def bfs_subgoal(env, s_ij, g_ij, delta_cells, bfs_from):
    """S0 對照：subgoal 改由 BFS 在格圖上生，⛔ 不用 latent。

    ⭐ 為什麼一定要有這支：S1（latent 生 subgoal）如果贏了，我們必須分得出
       贏的是「階層化這個結構」還是「長程 latent 真的在推理」。
       ⛔ 少了它，S1 贏了我們會把功勞記錯人。

    🚨 2026-08-28 修（S0 被系統性弄弱）：舊版算了 `cur = dist[s_ij]` 之後【一次都沒用到】，
       挑點的 key 是 `(abs(dc − delta_cells), dist[c])` ＝「離【起點】約 delta 的格」
       ⇒ ⛔ 完全沒有要求它比現在更靠近目標。
       實測（fake maze，走廊 0..6）：agent 距目標 1 格 ⇒ 選出的 subgoal 仍距目標 1 格；
       agent 就站在目標上 ⇒ 選出距目標 2 格的點（三角不等式：dist[c] ≥ |cur − dc|
       ⇒ cur 小於 delta_cells 時，選出來的必定【更遠】）。
       ⚠️ S0 是拆功勞用的對照組 ⇒ 它被弄弱的話，S1 贏了我們會把功勞記錯人。

    ⇒ 修法兩格：
       ① 已經在半徑內（cur ≤ delta_cells）⇒ 直接指最終目標，⛔ 不要繞路
       ② 否則只在「嚴格更靠近目標」的格子裡挑（dist[c] ≤ cur − 1），再用原本的 key
    """
    dist = bfs_from(env, g_ij)                 # 每格到【目標】的距離
    if s_ij not in dist:
        return None
    cur = dist[s_ij]
    if cur <= delta_cells:                     # ① 已在半徑內 ⇒ 直接指目標
        return tuple(g_ij)
    # ② 只考慮「比現在更靠近目標」且從起點可達的格
    reach = bfs_from(env, s_ij)
    cand = [(c, dc) for c, dc in reach.items() if c in dist and dist[c] <= cur - 1]
    if not cand:                               # ⛔ 理論上不會發生（最短路上的鄰格必在裡面）
        return tuple(g_ij)
    best, best_key = None, None
    for c, dc in cand:
        key = (abs(dc - delta_cells), dist[c])   # 先挑「離起點約 delta」，再挑「離目標近」
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best
