"""experiments/gen_sft/dyn_corpus_common.py —— 主人裁示①②的共用模組（動態隨機起點 +
漂移型擾動，rewrite-v1 步2 的升級版）。

跟 aug_corpus_common.py（固定 2 刀、corpus-build 時就把子樣本切死存進 .pt）的差異：
本檔不預先切死任何子樣本，而是把每條 episode 存成『軌跡級原料』——
  obs_chunks:  每個 chunk 邊界 k=0..n_chunks-1 的『真實』obs（raw["observations"] 直接
               索引，不模擬）
  code_seq:    完整字串（母 episode 全長）
  wp_xy / wp_local_idx: 完整路標序列 + 每個路標第一次被真實軌跡到達的 local timestep
  usable_cuts / usable_cuts_idx: 預先算好『哪些中途 chunk 邊界 k 可用』（避開最後 4 個
               chunk、且切下去之後至少留 1 個尚未到達的路標）＋對應的『尚未到達路標』
               起始 index——這就是工單設計點1說的『路標弧長表 —— 或等價可即時切的形式』：
               wp_local_idx 本身已經是 aug_corpus_common 用來判定『尚未到達』的機制，
               這裡只是預先把每個 episode 的可用切點過濾好存起來，訓練時 O(1) 查表，
               不必每步重算 searchsorted。

真正的『動態』發生在訓練迴圈（train_gen_sft_dyn.py 的 iterate_batches_dyn）：每次
取樣，對抽到的 episode 丟一枚硬幣（cut_rng，跟權重初始化/batch-order 的 seed
彼此獨立的另一個 rng 流）——50% 用原起點（k=0，從頭）、50% 從該 episode 的
usable_cuts 均勻隨機挑一個 k，即時建出 (條件, 目標) pair。教材本身『連續覆蓋整條
軌跡』（任何一個 usable 的中途點都可能被抽到，不像 aug 版死死只有 2 個固定點）。

邊界安全性（正確性自檢會驗證，不是純推導）：obs_chunks 只存 k=0..n_chunks-1（不含
k=n_chunks 那個『最後一個 chunk 之後』的邊界）——因為 n_chunks=floor(T/4)，
4*(n_chunks-1) <= T-4 < T，保證每一列都落在該集自己的 raw observation 範圍內，
不會讀到下一集的資料或越界。usable_cuts 的候選本來就被 avoid_last_chunks=4 限制在
k<=n_chunks-5，遠低於 n_chunks-1，所以這個儲存範圍對目前設計綽綽有餘。

漂移型擾動（僅擾動組，perturb_obs()）：xy 兩維——方向均勻、幅度取自半常態分佈
scale=xy_sigma（校準見 build_corpus_dyn.py，對齊 drift_summary.json 的
succ_quantiles.p75）；其餘 27 維——按『train 資料各維 std』乘一個自訂小係數
（OTHER_DIM_COEF，量級遠小於 xy 主擾動，見 build_corpus_dyn.py 的稽核輸出）同步擾動。
只加在條件 obs，目標字串（code_seq）完全不動。

⛔ 新檔（dyn 系列），不改動 aug_corpus_common.py / gen_sft_common.py / build_corpus.py /
   build_corpus_aug.py / train_gen_sft.py / model.py 任何一行——只 import
   aug_corpus_common 的 valid_cut_points() / remaining_waypoints_idx()（純函式，避免
   重寫同一份切點判定邏輯出現兩份可能漂移的版本）。
"""
import numpy as np

import aug_corpus_common as ac

USE_FULL_PROB = 0.5          # 工單設計點1：每次取樣 50% 從頭、50% 中途切點
DYN_CUT_SEED = 20260909      # 切樣隨機流固定 seed——⛔ 跟權重初始化/batch-order 的
                              # --seed（20260909/10/11）分開記錄；本檔 6 顆 dyn 訓練
                              # 全部共用同一個 DYN_CUT_SEED（不隨訓練 seed 變動），
                              # 這樣「哪個點被切到」的隨機性不會混進 3-seed 的變異量測。
PERTURB_SEED = 20260912      # 擾動隨機流固定 seed，同樣跨 3 個訓練 seed 共用一個值，
                              # 理由同上；刻意選一個不等於任何訓練 seed 的數字避免誤讀。
OTHER_DIM_COEF = 0.005       # 其餘 27 維擾動係數（本檔自訂，非工單給定數字）。⚠️ 校準記錄：
                              # 第一次嘗試 0.05 時，27 維各自看起來很小，但合成 L2 norm
                              # （27 個獨立 Gaussian 疊起來，量級隨 sqrt(27)≈5.2 放大）
                              # p75=0.4238 反而「大於」xy 主擾動 p75=0.3659（比例1.158）——
                              # build_corpus_dyn.py 的稽核印出來當場抓到，不是事後才發現；
                              # 改成 0.005（縮小 10 倍）後 p75≈0.042，明顯小於 xy 主擾動，
                              # 見 build_corpus_dyn.py 稽核輸出/corpus_dyn_v1.pt meta。
N_SELFCHECK_DYN = 20         # 驗收標準①：動態切 20 個樣本位元自檢
N_PERTURB_AUDIT = 1000       # 驗收標準④：擾動幅度分佈稽核樣本數


def build_trajectory_record(ei, orig_sample, wp_local_idx, obs_arr):
    """把一條 episode 轉成『軌跡級原料』record。orig_sample 來自 corpus_v1.pt（含
    s0_idx/wp_xy/code_seq/M/n_chunks，皆為母 episode 全長，不是任何切過的版本）。
    wp_local_idx 來自重新呼叫 rt.build_tasks() 的回傳（corpus_v1.pt 既有樣本格式沒存
    這個欄位，跟 aug_corpus_common 取得方式完全相同）。obs_arr=raw["observations"]。
    """
    s0_idx = int(orig_sample["s0_idx"])
    n_chunks = int(orig_sample["n_chunks"])
    M = int(orig_sample["M"])
    # obs_chunks[k] = 該集第 k 個 chunk 邊界的真實 obs，k=0..n_chunks-1（見檔頭邊界說明）
    obs_chunks = obs_arr[s0_idx: s0_idx + 4 * n_chunks: 4].astype(np.float32).copy()
    assert obs_chunks.shape[0] == n_chunks, (
        f"⛔ episode={ei} obs_chunks 列數={obs_chunks.shape[0]} != n_chunks={n_chunks}——"
        f"越界或切片算錯，停手回報")

    cands = ac.valid_cut_points(n_chunks)
    usable_k, usable_idx = [], []
    for k in cands:
        idx = ac.remaining_waypoints_idx(wp_local_idx, 4 * k)
        if (M - idx) >= 1:
            usable_k.append(k)
            usable_idx.append(idx)

    return dict(
        episode=int(ei), s0_idx=s0_idx, n_chunks=n_chunks, M=M,
        obs_chunks=obs_chunks,
        code_seq=orig_sample["code_seq"].copy(),
        wp_xy=orig_sample["wp_xy"].copy(),
        wp_local_idx=np.asarray(wp_local_idx, dtype=np.int64).copy(),
        usable_cuts=np.asarray(usable_k, dtype=np.int64),
        usable_cuts_idx=np.asarray(usable_idx, dtype=np.int64),
        n_candidates=len(cands),
    ), dict(n_candidates=len(cands), n_usable=len(usable_k))


def perturb_obs(obs, rng, xy_sigma, other_coef, other_std):
    """漂移型擾動，只用在『條件』obs，回傳新陣列（不 in-place 改輸入）。
    xy（obs[0:2]，見 wv_common 的 xy=qpos[:,:2] 慣例、sim_obs=qpos[:15]+qvel[:14]
    故 obs[0:2] 即 x,y）：方向 uniform[0,2π)，幅度 = |N(0, xy_sigma)|（半常態）。
    其餘 27 維（obs[2:29]）：獨立 N(0, (other_coef*other_std_d)^2) 逐維相加，
    other_std 是『train 資料各維 std』（build_corpus_dyn.py 算好傳入，非本檔重算）。
    """
    obs = np.asarray(obs, dtype=np.float32).copy()
    theta = rng.uniform(0.0, 2.0 * np.pi)
    mag = abs(rng.normal(0.0, xy_sigma))
    obs[0] += mag * np.cos(theta)
    obs[1] += mag * np.sin(theta)
    noise27 = rng.normal(0.0, other_coef * np.asarray(other_std, dtype=np.float64))
    obs[2:29] += noise27.astype(np.float32)
    return obs


def draw_dynamic_sample(traj, cut_rng, use_full_prob=USE_FULL_PROB,
                        perturb_rng=None, perturb_cfg=None, stats=None):
    """核心動態抽樣：50/50 決定用原起點還是中途切點；中途切點時若 perturb_cfg 給了
    （dict(xy_sigma=.., other_coef=.., other_std=..)）就對條件 obs 加擾動——只有
    perturb_rng 不是 None 且 perturb_cfg 不是 None 時才會擾動（no-pert 組兩者都傳 None）。
    回傳 dict(s0_obs, wp_xy, code_seq, M, n_chunks)，跟 train_gen_sft.collate_batch
    要求的樣本 dict schema 完全相容。stats（可選 dict）用來累計 n_full/n_cut/n_perturbed
    計數，供訓練結束後寫進 summary（透明度，不是事後才想到)。
    """
    n_usable = len(traj["usable_cuts"])
    use_full = (cut_rng.random() < use_full_prob) or (n_usable == 0)
    if use_full:
        if stats is not None:
            stats["n_full"] = stats.get("n_full", 0) + 1
            if n_usable == 0:
                stats["n_forced_full_no_usable_cuts"] = stats.get("n_forced_full_no_usable_cuts", 0) + 1
        return dict(s0_obs=traj["obs_chunks"][0].copy(), wp_xy=traj["wp_xy"],
                    code_seq=traj["code_seq"], M=traj["M"], n_chunks=traj["n_chunks"])

    pos = int(cut_rng.integers(0, n_usable))
    k = int(traj["usable_cuts"][pos])
    idx = int(traj["usable_cuts_idx"][pos])
    obs = traj["obs_chunks"][k].copy()
    if perturb_cfg is not None:
        assert perturb_rng is not None, "⛔ perturb_cfg 給了但 perturb_rng=None——停手回報"
        obs = perturb_obs(obs, perturb_rng, perturb_cfg["xy_sigma"], perturb_cfg["other_coef"],
                          perturb_cfg["other_std"])
        if stats is not None:
            stats["n_perturbed"] = stats.get("n_perturbed", 0) + 1
    if stats is not None:
        stats["n_cut"] = stats.get("n_cut", 0) + 1

    M_sub = traj["M"] - idx
    return dict(s0_obs=obs, wp_xy=traj["wp_xy"][idx:].copy(), code_seq=traj["code_seq"][k:].copy(),
                M=int(M_sub), n_chunks=int(traj["n_chunks"] - k))
