"""開發尺：把 eval 從「5 個 task 各抖 50 次」換成「幾百個真正獨立的題」。

🚨 為什麼非換不可（2026-08-26 稽核抓到，ルナ親眼複驗）
    官方 eval ＝ 5 task × 50 seed ＝ 250 集，但 50 個 seed 共用同一組 task_info
    （同 goal、init 只在同一格內抖）⇒ **成敗是 task 決定的，不是 policy 決定的**。
    實測：同一輪裡三個結構完全不同的 policy（BC 地板 / u 歸零 / 別人的 u）
    全部拿到 150/250 ＝ 0.600 —— 三個獨立的東西剛好同分，機率約千分之六。
    ⇒ 獨立樣本數是 **5**，不是 250。實測地板變異 0.144 ≈ 一個 task 翻面。
    ⇒ ⛔ 在這把尺上，任何小於 0.2 的效應都看不見；今天所有 rollout 結論都建在它上面。

⭐ 這支提供三件事，⛔ 都不要再複製到別的腳本裡（今天已經因為複製吃過虧）：
    build_dev_tasks()  從迷宮地圖列舉 (起點格, 終點格)，用 env 內建的 BFS 分難度層
    dev_eval()         吃一個 policy callable，回【每一集】的明細
    sanity_check()     驗收這把尺本身：三個已知不同的 policy 要分得開

⭐ 兩個關鍵設計
    ① 回 per-episode 明細而不是總分 —— 配對比較（同一批題比 R=0 vs R=3）需要它，
       而今天的 rollout 只回總分，把配對設計白白浪費掉了。
    ② 除了 binary success，一律記【連續指標】（走了幾步、最後離目標多遠）。
       ⇒ 不用改 env 就能讓「整個 task 一起翻面」這種量子化消失。
"""
import numpy as np

__all__ = ["build_dev_tasks", "dev_eval", "sanity_check", "summarize", "cell_width"]


def _passable_cells(env):
    """迷宮裡所有不是牆的格子。⚠️ 讀 env 自己的 maze_map，⛔ 不自己重畫一份。"""
    M = np.asarray(env.unwrapped.maze_map)
    return [(int(i), int(j)) for i in range(M.shape[0]) for j in range(M.shape[1]) if M[i, j] == 0]


def cell_width(env):
    """相鄰兩格的間距（原始座標單位）。

    🚨 2026-08-28 抽進來共用。舊版 `exp_span_gap.py` 取的是【頭兩個可通行格】的距離：
         cells = _passable_cells(env)  # row-major 列舉
         cell_w = ‖ij_to_xy(cells[0]) − ij_to_xy(cells[1])‖
       ⇒ 只有在「同一列相鄰兩行都可通行」時才等於一格寬。
       `[實測]` 把某一列換成 [1,0,1,0,1,0,1] ⇒ cells[0] 與 cells[1] 隔了兩格 ⇒ cell_w 變兩倍，
       ⛔ 而且不會報錯。而 `≈路長 = bfs_dist × cell_w` 正是「最難那層 100% 超出訓練 p99」
       與 DELTA_SUB=7.5 的來源 ⇒ 這個常數錯掉，那兩個結論一起錯。
    ⇒ 正解是【全對距離取最小】（scratch_lacot_rollout.py 原本就是這樣算的，兩份不一致）。
    """
    cells = _passable_cells(env)
    if len(cells) < 2:
        raise SystemExit("⛔ 這張迷宮圖不到兩個可通行格 ⇒ 算不出格寬")
    xy = np.array([env.unwrapped.ij_to_xy(c) for c in cells], np.float64)
    d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    return float(np.sqrt(d2[d2 > 1e-9].min()))


def _bfs_from(env, src):
    """從 src 格出發的 BFS 步距。回 dict[(i,j)] = 步數；到不了的不在 dict 裡。"""
    M = np.asarray(env.unwrapped.maze_map)
    H, W = M.shape
    dist = {tuple(src): 0}
    frontier = [tuple(src)]
    while frontier:
        nxt = []
        for (i, j) in frontier:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and M[ni, nj] == 0 and (ni, nj) not in dist:
                    dist[(ni, nj)] = dist[(i, j)] + 1
                    nxt.append((ni, nj))
        frontier = nxt
    return dist


def build_dev_tasks(env, n_per_tier=80, n_tiers=3, seed=0, min_dist=2):
    """列舉所有可達的 (起點格, 終點格)，按 BFS 距離切成 n_tiers 層，每層抽 n_per_tier 個。

    ⭐ 為什麼分層：遠的那一層很可能【資料裡沒有整條走過】，而 TTGS 自承
       「目標落在資料流形外時收益只有 0~2 個百分點」⇒ 那層正是我們該展示優勢的地方。
       ⇒ 所以分層不只是控制難度，它同時把「我們贏在哪」的舞台切出來。
    ⚠️ min_dist：太近的題連 BC 都會過，沒有區辨力。
    """
    cells = _passable_cells(env)
    pairs = []
    for src in cells:
        dist = _bfs_from(env, src)
        for dst, d in dist.items():
            if d >= min_dist:
                pairs.append((src, dst, d))
    if not pairs:
        raise SystemExit("⛔ 這張迷宮圖抽不出任何題 —— 檢查 maze_map 是不是讀對了")
    ds = np.array([p[2] for p in pairs])
    # 用分位數切，⛔ 不用等寬切（等寬會讓某層空掉）
    edges = np.quantile(ds, np.linspace(0, 1, n_tiers + 1))
    rng = np.random.default_rng(seed)
    tasks = []
    for t in range(n_tiers):
        lo, hi = edges[t], edges[t + 1]
        sel = [p for p in pairs if (lo <= p[2] <= hi if t == n_tiers - 1 else lo <= p[2] < hi)]
        if not sel:
            continue
        take = rng.choice(len(sel), size=min(n_per_tier, len(sel)), replace=False)
        for k in take:
            src, dst, d = sel[int(k)]
            tasks.append(dict(
                task_name=f"dev-t{t}-{src[0]}{src[1]}-{dst[0]}{dst[1]}",
                init_ij=src, goal_ij=dst, tier=t, bfs_dist=int(d),
                init_xy=tuple(float(x) for x in env.unwrapped.ij_to_xy(src)),
                goal_xy=tuple(float(x) for x in env.unwrapped.ij_to_xy(dst)),
            ))
    return tasks


def dev_eval(env, tasks, policy_chunk_fn, maxh, seed0=0, torch_seed_fn=None,
             on_episode_start=None):
    """跑完所有題，回【每一集】的明細。

    policy_chunk_fn(obs, goal) -> 一段動作 (chunk, adim)
    torch_seed_fn(idx)         -> 每題開始前呼叫，讓不同 arm 共用同一條 action-noise stream
                                  ⭐ 這是配對比較的關鍵：⛔ 少了它，兩個 arm 的差就混進了取樣噪聲
    on_episode_start(obs, goal, task) -> 每題開始前呼叫（2026-08-28 加，預設 None ⇒ 行為不變）
                                  ⭐ 給【有狀態】的 policy 用 —— 例如兩層 subgoal 規劃，
                                  ⛔ 沒有它的話上一題的 subgoal 會漏到下一題

    ⚠️ 回的是 list of dict，⛔ 不是總分。總分由 summarize() 算，
       而配對比較必須拿明細去做（今天的教訓：rollout 只回總分 ⇒ 配對統計算不出來）。
    """
    rows = []
    for idx, t in enumerate(tasks):
        # 🚨 2026-08-26 稽核抓到：ogbench 的 maze env 對 init/goal 加的噪聲走的是
        #    【全域 np.random】(`locomaze/maze.py:565`)，⛔ 不是 reset(seed=) 那條 RNG。
        #    ⇒ 各 arm 順序跑的話，同一題在不同 arm 的起點/終點抖動【不一樣】
        #      ⇒ 配對被稀釋成半配對，差值裡混進了題目本身的差異。
        #    ⇒ 每題 reset 之前把三條 stream 全部釘死，配對才是真的。
        np.random.seed(seed0 + idx)
        try:
            env.action_space.seed(seed0 + idx)
        except Exception:
            pass
        obs, info = env.reset(seed=seed0 + idx,
                              options={"task_info": t, "render_goal": False})
        goal = info["goal"]
        if torch_seed_fn is not None:
            torch_seed_fn(idx)
        if on_episode_start is not None:
            on_episode_start(obs, goal, t)
        success, steps = False, 0
        best_d = float(np.linalg.norm(np.asarray(obs[:2]) - np.asarray(goal[:2])))
        while steps < maxh and not success:
            for a in policy_chunk_fn(obs, goal):
                obs, rew, term, trunc, info = env.step(a)
                steps += 1
                d = float(np.linalg.norm(np.asarray(obs[:2]) - np.asarray(goal[:2])))
                best_d = min(best_d, d)
                if info.get("success"):
                    success = True
                if success or term or trunc or steps >= maxh:
                    break
        rows.append(dict(idx=idx, tier=t["tier"], bfs_dist=t["bfs_dist"],
                         success=bool(success), steps=int(steps),
                         final_dist=float(d), best_dist=best_d))
    return rows


def summarize(rows):
    """總分 ＋ 連續指標 ＋ per-tier 明細。⛔ 永遠不要只看第一個數字。"""
    a = np.array([r["success"] for r in rows], float)
    out = dict(n=len(rows), success=float(a.mean()),
               se=float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan"),
               steps_med=float(np.median([r["steps"] for r in rows])),
               best_dist_med=float(np.median([r["best_dist"] for r in rows])))
    out["per_tier"] = {}
    for t in sorted({r["tier"] for r in rows}):
        sub = [r for r in rows if r["tier"] == t]
        sa = np.array([r["success"] for r in sub], float)
        out["per_tier"][str(t)] = dict(n=len(sub), success=float(sa.mean()),
                                       bfs_med=float(np.median([r["bfs_dist"] for r in sub])),
                                       best_dist_med=float(np.median([r["best_dist"] for r in sub])))
    return out


def paired_diff(rows_a, rows_b, boot=2000, seed=0):
    """配對差值 ＋ bootstrap CI。⚠️ 兩邊必須是同一批題、同樣的順序。"""
    assert len(rows_a) == len(rows_b), "⛔ 兩個 arm 的題數不一樣 ⇒ 不是配對的"
    assert all(x["idx"] == y["idx"] for x, y in zip(rows_a, rows_b)), "⛔ 題目順序對不上"
    d = np.array([float(x["success"]) - float(y["success"]) for x, y in zip(rows_a, rows_b)])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)]
    # ⭐ McNemar：只看不一致的配對，是同一件事的銳利版。bootstrap 給效應量、它給 p。
    nb, nc = int((d > 0).sum()), int((d < 0).sum())
    p = _mcnemar_p(nb, nc)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n=len(d), n_a_only=nb, n_b_only=nc, mcnemar_p=p)


def _mcnemar_p(nb, nc):
    """discordant pairs 的 exact binomial 雙尾 p。⛔ 不用 scipy，避免多一層依賴。"""
    n = nb + nc
    if n == 0:
        return 1.0
    from math import comb
    k = min(nb, nc)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def sanity_check(named_rows, sens_pair=("random", "bc"), spec_pair=("bc", "bc_rerun"),
                 sens_min=0.30, spec_max=0.03, sep_alpha=0.05, report_pairs=()):
    """⭐ 驗收這把尺【本身】。

    🚨 2026-08-26 稽核修正 —— 舊版的 gate 是「任一對 policy 分不開就判尺壞」，那是錯的組成：
       `bc vs lacot` 的【真答案】就是差 ≈0（歷史三個 seed 都是 0.80 對 0.79）
       ⇒ 儀器會在「實驗答案剛好是零」的時候被判成故障 ⇒ 永遠 deadlock。
       ⛔ 把【實驗問題】放進【儀器驗收】是設計錯誤，不是門檻鬆緊的問題。

    ⇒ gate 只用「已知排序」的對子：
        靈敏度  random-action vs 訓好的 bc   ⇒ 必須分得開，而且差 ≥ sens_min
        特異度  同一顆 bc 換 action-seed 重跑 ⇒ 必須【分不開】，而且 |差| < spec_max
                （⚠️ 這格會抓到配對沒釘好 —— 沒釘的話同一顆模型自己跟自己都會有差）
    ⇒ 其他對子（lacot / null_u / oracle）照跑照報，但 ⛔ 不進 gate。

    🚨 2026-08-28 修正 —— 「分得開」舊版只看 bootstrap CI，而 CI 在 discordant pairs
       只有個位數時會過度自信：seed 1 只有 4 題不一致、且全同向 ⇒ CI 排除 0 判「分得開」，
       但那等於丟四次銅板全同面（exact p=0.125）⇒ 什麼都沒證明。
    ⇒ 現在 CI 與 McNemar p 兩個都要過才算分得開（sep_alpha，預設 0.05）。
    ⚠️ 配對比較的有效樣本數是 discordant pairs 的個數，⛔ 不是題數。
    ⚠️ 印出來的差值一律是【前者 − 後者】，⛔ 舊版靈敏度那行的標籤跟數值是反的。

    ⏳ 未修（設計問題，留給主人裁）：特異度那格是反向判準（要求「分不開」），
       嚴格講也該加上「p 不顯著」，但那會讓已經通過的驗收變嚴 ⇒ ⛔ 沒自己動。

    🚨 2026-08-28 —— 特異度這格真正的洞【不是】門檻，是【受測對象選錯】：
       主線拿 `bc` 當它的兩臂，而 bc 這條路徑（bc_head(cond)）⛔ 不消耗 torch 亂數
       ⇒ 換 tseed 是 no-op ⇒ 兩臂逐位元相同 ⇒ 這格恆過而且什麼都沒驗到。
       ⭐ 真正有風險的是【會抽樣】的 arm（`lacot` 走 flow.sample、`shuf` 走 _foreign_u）——
       配對沒釘好的話，破綻只會出現在它們身上。
       ⇒ 下面新增的 discordant==0 判 False 只是【讓它叫】；⛔ 要真的驗到配對，
         呼叫端必須把 spec_pair 換成一個會抽樣的 arm 的兩次重跑。
    """
    summ = {k: summarize(v) for k, v in named_rows.items()}
    notes, gates = [], {}

    def _pair(a, b):
        if a not in named_rows or b not in named_rows:
            return None
        return paired_diff(named_rows[a], named_rows[b])

    sp = _pair(*sens_pair)
    if sp is None:
        notes.append(f"🚨 靈敏度 gate 缺 arm：{sens_pair} ⇒ 尺沒有被驗過")
        gates["sensitivity"] = False
    else:
        ok = ((sp["ci95"][0] > 0 or sp["ci95"][1] < 0) and abs(sp["mean"]) >= sens_min
              and sp["mcnemar_p"] < sep_alpha)
        gates["sensitivity"] = bool(ok)
        notes.append(f"[靈敏度] {sens_pair[0]} − {sens_pair[1]} = {sp['mean']:+.3f} "
                     f"CI [{sp['ci95'][0]:+.3f},{sp['ci95'][1]:+.3f}] p={sp['mcnemar_p']:.4f}"
                     f"  {'✓ 尺看得見已知的大差' if ok else '🚨 連保證不同的東西都分不開'}")

    kp = _pair(*spec_pair)
    if kp is None:
        notes.append(f"🚨 特異度 gate 缺 arm：{spec_pair} ⇒ 配對有沒有釘好無從得知")
        gates["specificity"] = False
    elif kp["n_a_only"] + kp["n_b_only"] == 0:
        # 🚨 2026-08-28：兩臂【逐位元相同】⇒ 差恆為 0、CI 恆含 0 ⇒ 舊版判 ✓ 滿分。
        #    ⛔ 但那不是「沒有假訊號」，是【這格根本沒有驗到配對】——
        #    退化的輸入拿到滿分，而一把不會叫的尺跟壞掉的尺長得一模一樣。
        #    ⚠️ 加 p 值救不了：nb+nc=0 ⇒ McNemar p 恆為 1.0 ⇒ 永遠不顯著。
        gates["specificity"] = False
        notes.append("[特異度] 🚨 兩臂逐位元相同（discordant = 0）⇒ 這格沒有驗到配對 ——"
                     " 換一條 action-noise stream 重跑才算數（例如 tseed 換一個值）")
    else:
        ok = (kp["ci95"][0] <= 0 <= kp["ci95"][1]) and abs(kp["mean"]) < spec_max
        gates["specificity"] = bool(ok)
        notes.append(f"[特異度] 同一顆模型重跑 = {kp['mean']:+.3f} "
                     f"CI [{kp['ci95'][0]:+.3f},{kp['ci95'][1]:+.3f}]"
                     f"  (discordant {kp['n_a_only']}+{kp['n_b_only']})"
                     f"  {'✓ 沒有假訊號' if ok else '🚨 自己跟自己都有差 ⇒ 配對沒釘好'}")

    for a, b in report_pairs:
        pd = _pair(a, b)
        if pd is None:
            continue
        sep = ((pd["ci95"][0] > 0) or (pd["ci95"][1] < 0)) and pd["mcnemar_p"] < sep_alpha
        notes.append(f"[參考] {a} − {b} = {pd['mean']:+.3f} "
                     f"CI [{pd['ci95'][0]:+.3f},{pd['ci95'][1]:+.3f}] p={pd['mcnemar_p']:.4f}"
                     f"  {'分得開' if sep else '分不開'}   ⛔ 不進 gate")

    return dict(passed=bool(all(gates.values())) if gates else False,
                gates=gates, summaries=summ, notes=notes)
