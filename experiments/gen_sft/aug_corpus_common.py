"""experiments/gen_sft/aug_corpus_common.py —— 中途起點增強語料共用模組（rewrite-v1 步2）。

工單問題：R=2/4（密集重寫）淨損失於錨點；已定讞假說＝u 的訓練教材全部『從乾淨開頭寫到
底』，從沒學過『從半路、帶偏移姿態接著寫』（見 docs/NOTE-2026-09-09-rewrite-receding.md）。
本檔實作『增強教材』：每條 episode 除原樣本（從頭起筆）外，從中途 chunk 邊界抽 2 個
切點，各造一個子樣本：條件＝該中途點的真實 obs（資料裡的，非模擬）＋其後尚未經過的
路標序列，目標＝其後字串——直接教模型半路接寫這件事。

⛔ 新檔，不改動 build_corpus.py / gen_sft_common.py 任何一行。

⛔ 重要：gen_sft_common.split_corpus() 假設『每個 episode 只有一個樣本』——它用
   `s["episode"] for s in samples` 直接建 eps 陣列（長度=樣本數，不是 episode 數）、
   用這個陣列長度算 n_val。增強後同一個 episode 有 3 個樣本（1 原 + 2 子），這個陣列
   會有重複值，`rng.permutation` 打散的是『陣列位置』不是『episode 身份』——同一個
   episode 的不同副本很可能被分到不同位置，導致同一個 episode id 同時出現在
   val_eps 和 train_eps 集合裡，`assert train_eps.isdisjoint(val_eps)` 會直接炸掉。
   本檔的 split_corpus_by_episode() 是另一份實作：先對『唯一 episode id』集合做
   permutation，再依集合成員關係分樣本——同一個 episode 的原樣本＋全部子樣本永遠
   同側，這是防洩漏的結構保證，不是事後檢查。當輸入『每 episode 恰一個樣本』（沒有
   增強）時，兩份函式的 rng 消耗方式與索引邏輯逐位元相同（sorted unique eps 在無重複
   時就是 sorted eps 本身）——build_corpus_aug.py 的自檢②直接驗證這件事:增強後的
   train/val episode 集合必須跟 corpus_v1.pt 原本的切分完全相同。

切點選法（design 第 1 點逐字照做）：
  - 候選＝chunk 邊界 k ∈ {1, ..., n_chunks-1}（k=0 是原樣本本身，不算子樣本候選）。
  - 『避開最後 4 個 chunk』＝候選再排除 k ∈ {n_chunks-4, ..., n_chunks-1}，
    剩 k ∈ {1, ..., n_chunks-5}。
  - 均勻隨機不重複抽 2 個（候選不足 2 個就有幾個抽幾個、候選=0 就跳過該 episode，
    都記進 ledger，不是靜默漏做）。
  - 抽樣用單一 rng（seed=CUT_SEED）、按 episode id 排序後的固定走訪順序逐集抽 2 個，
    可重現、不依賴 dict 走訪順序（Python dict 保序但不該依賴這個當隨機性來源）。

『尚未到達的路標』判定：wp_local_idx[m-1]＝路標 m 第一次被『這條 episode 的真實歷史
軌跡』到達的 local timestep（來自 run_teacher_relay.build_tasks，逐 m 遞增/不遞減）。
切點在 local time cut_local_t=4k 時，路標 m 若 wp_local_idx[m-1] <= cut_local_t 算
『已到達』（此刻或更早就已經在那個位置)，> cut_local_t 才算『尚未到達』——用
np.searchsorted(..., side='right') 取第一個 > cut_local_t 的位置，之後全部路標留給
子樣本。
"""
import numpy as np

CUT_SEED = 20260909          # 切點抽樣固定種子（design 第1點「切點抽樣固定 seed」）
N_CUTS_PER_EPISODE = 2
AVOID_LAST_CHUNKS = 4
MIN_REMAINING_WAYPOINTS = 1  # 子樣本必須至少留 1 個尚未到達的路標，否則跳過（見檔頭）


def valid_cut_points(n_chunks, avoid_last=AVOID_LAST_CHUNKS):
    """候選切點 k：排除 k=0（=原樣本)，排除最後 avoid_last 個 chunk（k>=n_chunks-avoid_last
    不可選)。回傳排序好的 list[int]，候選為空回傳 []。"""
    hi = n_chunks - avoid_last  # k 的 exclusive 上界
    return list(range(1, hi)) if hi > 1 else []


def remaining_waypoints_idx(wp_local_idx, cut_local_t):
    """回傳『尚未到達』路標的起始 index（wp_local_idx 非遞減，side='right' 保證
    wp_local_idx[idx-1] <= cut_local_t < wp_local_idx[idx] 語意下 idx 之前全部算
    已到達，idx 起（含）才是『其後尚未經過的路標』)。"""
    return int(np.searchsorted(np.asarray(wp_local_idx), cut_local_t, side="right"))


def build_augmented_samples(orig_by_ep, tasks_by_ep, raw, seed=CUT_SEED,
                            n_cuts=N_CUTS_PER_EPISODE, avoid_last=AVOID_LAST_CHUNKS):
    """對每個 episode（用 orig_by_ep 的 key 集合＝build_corpus() 收錄的全部可用 episode）
    造 <=n_cuts 個中途起點子樣本。回傳 (sub_samples: list[dict], ledger: dict)。

    orig_by_ep:  {episode_id: sample_dict}（來自既有 corpus_v1.pt 的 train+val 合併，
                 每個 sample 有 s0_idx/s0_obs/wp_xy/code_seq/M/n_chunks）
    tasks_by_ep: {episode_id: task_dict}（來自重新呼叫 rt.build_tasks() 的回傳，含
                 wp_local_idx——corpus_v1.pt 既有 sample 格式沒有存這個欄位，這是唯一
                 需要重新呼叫 build_tasks 的原因）
    raw:         原始資料 dict（需要 raw["observations"]，取切點當下的『真實』obs）
    """
    eps_sorted = sorted(orig_by_ep.keys())
    rng = np.random.default_rng(seed)
    sub_samples = []
    ledger = dict(n_episodes=len(eps_sorted), n_episodes_zero_candidates=0,
                  n_episodes_lt_n_cuts=0, n_cuts_drawn_total=0, n_subsamples_created=0,
                  n_subsamples_skipped_zero_remaining_wp=0)
    obs_arr = raw["observations"]
    for ei in eps_sorted:
        orig = orig_by_ep[ei]
        task = tasks_by_ep[ei]
        n_chunks = orig["n_chunks"]
        assert task["n_chunks"] == n_chunks, (
            f"⛔ episode={ei} n_chunks 不一致：orig(corpus_v1)={n_chunks} "
            f"task(重建)={task['n_chunks']}——停手回報，不能假設兩邊同構")
        cands = valid_cut_points(n_chunks, avoid_last)
        if len(cands) == 0:
            ledger["n_episodes_zero_candidates"] += 1
            continue
        if len(cands) < n_cuts:
            ledger["n_episodes_lt_n_cuts"] += 1
        k_take = min(n_cuts, len(cands))
        ks = rng.choice(np.asarray(cands, dtype=np.int64), size=k_take, replace=False)
        ledger["n_cuts_drawn_total"] += int(k_take)
        wp_local_idx = np.asarray(task["wp_local_idx"])
        for k in sorted(int(x) for x in ks):
            cut_local_t = 4 * k
            idx = remaining_waypoints_idx(wp_local_idx, cut_local_t)
            M_sub = orig["M"] - idx
            if M_sub < MIN_REMAINING_WAYPOINTS:
                ledger["n_subsamples_skipped_zero_remaining_wp"] += 1
                continue
            s0_idx_sub = int(orig["s0_idx"] + 4 * k)
            sub_samples.append(dict(
                episode=int(ei), s0_idx=s0_idx_sub,
                s0_obs=obs_arr[s0_idx_sub].astype(np.float32).copy(),
                wp_xy=orig["wp_xy"][idx:].copy(), code_seq=orig["code_seq"][k:].copy(),
                M=int(M_sub), n_chunks=int(n_chunks - k),
                is_aug=True, cut_chunk=int(k), parent_n_chunks=int(n_chunks),
            ))
            ledger["n_subsamples_created"] += 1
    return sub_samples, ledger


def split_corpus_by_episode(samples, split_seed=42, val_frac=0.1):
    """Episode 級 9:1 切分，對『唯一 episode id』集合做切分（不是對樣本數）——見檔頭
    ⛔ 段落。無重複（無增強）情況下跟 gen_sft_common.split_corpus() 逐位元同構。"""
    eps = np.array(sorted(set(int(s["episode"]) for s in samples)), dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(eps))
    n_val = int(len(eps) * val_frac)
    val_eps = set(eps[perm[:n_val]].tolist())
    train_eps = set(eps[perm[n_val:]].tolist())
    assert train_eps.isdisjoint(val_eps)
    train_samples = [s for s in samples if s["episode"] in train_eps]
    val_samples = [s for s in samples if s["episode"] in val_eps]
    assert len(train_samples) + len(val_samples) == len(samples)
    return train_samples, val_samples, dict(
        split_seed=split_seed, val_frac=val_frac,
        n_train_episodes=len(train_eps), n_val_episodes=len(val_eps),
        n_train=len(train_samples), n_val=len(val_samples))
