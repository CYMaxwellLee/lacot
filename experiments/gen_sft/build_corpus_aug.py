#!/usr/bin/env python
"""experiments/gen_sft/build_corpus_aug.py —— rewrite-v1 步2：建『中途起點增強』教材語料。

工單 design 第1點：每條（build_corpus 收錄的）episode 除原樣本（從頭起筆）外，從中途
chunk 邊界均勻隨機抽 2 個切點（避開最後 4 個 chunk），各造一個子樣本：條件＝該中途點
的真實 obs（資料裡的，非模擬）＋其後尚未經過的路標序列，目標＝其後字串。原樣本＋子
樣本混合＝增強語料（樣本數帳記錄：原 vs 增強後）。held-out 切分沿用 episode 級 9:1
（split_seed=42），子樣本跟母 episode 同側（見 aug_corpus_common.split_corpus_by_episode
檔頭，防洩漏)。

⛔ CPU-only、輕量（複用 corpus_v1.pt 已編碼好的 code_seq，不重新跑 dict encoder；唯一
   新算的是每集『路標第一次被真實軌跡到達的 local timestep』——重新呼叫一次
   rt.build_tasks()，跟 gc.build_corpus() 內部呼叫的參數（seed=BUILD_TASKS_SEED=0、
   同 raw/ruler/rho）完全相同，數學上保證是同一組 task/wp_xy——本檔自檢①逐位元比對
   wp_xy 證明，不是假設。
⛔ 不改動任何既有檔案（build_corpus.py / gen_sft_common.py 原封不動，只 import）。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402
import aug_corpus_common as ac  # noqa: E402

BUILD_TASKS_SEED = 0   # ⛔ 必須跟 build_corpus.py 的常數一致，否則重新呼叫 build_tasks
                       # 得到的 task 走訪順序/wp_local_idx 對不上 corpus_v1.pt 的內容
SPLIT_SEED = 42
VAL_FRAC = 0.1
N_SELF_CHECK_SAMPLES = 5


def main():
    torch.set_num_threads(4)
    t0 = time.time()

    src_path = os.path.join(gc.RESULTS_DIR, "corpus_v1.pt")
    print(f"=== build_corpus_aug: 讀既有語料 {src_path} ===")
    blob = torch.load(src_path, weights_only=False)
    orig_train, orig_val, orig_meta = blob["train"], blob["val"], blob["meta"]
    orig_all = orig_train + orig_val
    print(f"loaded: n_train={len(orig_train)} n_val={len(orig_val)} n_total={len(orig_all)}  "
          f"build_tasks_seed(meta)={orig_meta['build_tasks_seed']}  split_seed(meta)={orig_meta['split_seed']}")
    assert orig_meta["build_tasks_seed"] == BUILD_TASKS_SEED, (
        f"⛔ corpus_v1.pt 的 build_tasks_seed={orig_meta['build_tasks_seed']} 跟本檔常數 "
        f"{BUILD_TASKS_SEED} 不一致，重新呼叫 build_tasks 對不上——停手回報")
    assert orig_meta["split_seed"] == SPLIT_SEED

    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()

    # 重新呼叫 build_tasks 只為了拿 wp_local_idx（corpus_v1.pt 的既有 sample 格式沒存
    # 這個欄位）——同 seed/raw/ruler/rho，數學上保證跟 gc.build_corpus() 內部呼叫的是
    # 同一組 task。自檢①：wp_xy 逐位元比對，證明真的是同一組，不是巧合對上。
    starts, ends = gc.wv.episode_bounds(raw["terminals"])
    n_ep = len(starts)
    tasks, skipped_tasks = gc.rt.build_tasks(raw, n_traj=n_ep + 1, seed=BUILD_TASKS_SEED,
                                             ruler=ruler, rho=gc.RHO)
    tasks_by_ep = {int(t["episode"]): t for t in tasks}
    print(f"rebuilt tasks（僅為取 wp_local_idx）: {len(tasks)} usable  skipped={len(skipped_tasks)}  "
          f"n_ep(raw)={n_ep}")

    orig_by_ep = {int(s["episode"]): s for s in orig_all}
    assert set(orig_by_ep.keys()) == set(tasks_by_ep.keys()), (
        f"⛔ episode 集合對不上：corpus_v1 有 {len(orig_by_ep)} 集，重建 tasks 有 "
        f"{len(tasks_by_ep)} 集——停手回報，不能假設兩邊同構")

    # ---- 自檢①：wp_xy 逐位元比對（全部集，證明重建的 task 真的跟 corpus_v1.pt 用的 ----
    # ---- 是同一組，不是「seed 一樣就假設一樣」）----
    mismatches = []
    for ei in orig_by_ep:
        a, b = orig_by_ep[ei]["wp_xy"], tasks_by_ep[ei]["wp_xy"]
        if a.shape != b.shape or not np.array_equal(a, b):
            mismatches.append(ei)
    print(f"\n自檢①（wp_xy 逐位元比對，全部 {len(orig_by_ep)} 集）："
          f"{'PASS 0 mismatches' if not mismatches else f'⛔ FAIL {len(mismatches)} mismatches'}")
    assert not mismatches, f"⛔ wp_xy 對不上的 episode（前10）: {mismatches[:10]}——停手回報"

    # ---- 建子樣本 ----
    sub_samples, ledger = ac.build_augmented_samples(orig_by_ep, tasks_by_ep, raw, seed=ac.CUT_SEED,
                                                      n_cuts=ac.N_CUTS_PER_EPISODE,
                                                      avoid_last=ac.AVOID_LAST_CHUNKS)
    print(f"\n增強子樣本 ledger（cut_seed={ac.CUT_SEED} n_cuts={ac.N_CUTS_PER_EPISODE} "
          f"avoid_last={ac.AVOID_LAST_CHUNKS}）:")
    for k, v in ledger.items():
        print(f"  {k} = {v}")

    for s in orig_all:
        s["is_aug"] = False
        s["cut_chunk"] = None
    all_samples = orig_all + sub_samples
    print(f"\n樣本數帳：原始樣本={len(orig_all)}  子樣本={len(sub_samples)}  混合後總計={len(all_samples)}")

    # ---- split（自己的 by-episode 版本，防洩漏；子樣本跟母 episode 同側）----
    train_aug, val_aug, split_meta = ac.split_corpus_by_episode(all_samples, split_seed=SPLIT_SEED,
                                                                 val_frac=VAL_FRAC)
    print(f"\nsplit(episode-level, split_seed={SPLIT_SEED}): {split_meta}")

    # ---- 自檢②：episode 級切分要跟 corpus_v1.pt 原本的切分完全相同（增強只加樣本，
    # ---- 不該改變哪個 episode 在哪一側；這是防洩漏保證的直接驗證，不是信任 rng 邏輯）----
    orig_train_eps = set(int(s["episode"]) for s in orig_train)
    orig_val_eps = set(int(s["episode"]) for s in orig_val)
    aug_train_eps = set(int(s["episode"]) for s in train_aug)
    aug_val_eps = set(int(s["episode"]) for s in val_aug)
    same_split = (orig_train_eps == aug_train_eps) and (orig_val_eps == aug_val_eps)
    print(f"\n自檢②（episode 級切分跟 corpus_v1.pt 原本切分逐集比對）："
          f"{'PASS 完全相同' if same_split else '⛔ FAIL 不同'}")
    assert same_split, "⛔ 增強後的 episode 級切分跟原始切分不一致——防洩漏保證破了，停手回報"

    # ---- 驗收①（工單 acceptance_criteria 第1條）：抽 5 個子樣本，字串=母 episode 對應 ----
    # ---- 後綴、條件 obs 與資料原值位元一致；對不上就 assert 炸掉，不產出可疑語料 ----
    rng_check = np.random.default_rng(777)
    aug_only = [s for s in (train_aug + val_aug) if s["is_aug"]]
    picks = rng_check.choice(len(aug_only), size=N_SELF_CHECK_SAMPLES, replace=False)
    print(f"\n=== 驗收①自檢：{N_SELF_CHECK_SAMPLES} 個子樣本 vs 母 episode 完整字串後綴 + 條件 obs 位元一致 ===")
    all_ok = True
    for pi in picks:
        s = aug_only[int(pi)]
        ei, k = s["episode"], s["cut_chunk"]
        parent = orig_by_ep[ei]
        expect_codes = parent["code_seq"][k:]
        codes_ok = np.array_equal(s["code_seq"], expect_codes)
        expect_obs = raw["observations"][s["s0_idx"]].astype(np.float32)
        obs_ok = np.array_equal(s["s0_obs"], expect_obs)
        obs_l2 = float(np.linalg.norm(s["s0_obs"].astype(np.float64) - expect_obs.astype(np.float64)))
        print(f"  episode={ei:5d} cut_chunk={k:2d}/{s['parent_n_chunks']}  "
              f"code_seq len={len(s['code_seq']):2d} (parent len={len(parent['code_seq'])}): "
              f"{'PASS bit-exact suffix' if codes_ok else '⛔ FAIL NOT suffix'}   "
              f"s0_obs: {'PASS bit-exact' if obs_ok else '⛔ FAIL'} (L2={obs_l2:.8f})   "
              f"M={s['M']} (parent M={parent['M']})")
        all_ok = all_ok and codes_ok and obs_ok
    print(f"驗收①自檢：{'5/5 PASS' if all_ok else '⛔ 有 FAIL，見上'}")
    assert all_ok, "⛔ 驗收①自檢沒有全過——構造邏輯有錯，停手修，不准帶著錯的語料往下訓練"

    os.makedirs(gc.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(gc.RESULTS_DIR, "corpus_aug_v1.pt")
    torch.save(dict(
        train=train_aug, val=val_aug,
        meta=dict(**{k: v for k, v in orig_meta.items()},
                  aug_source_corpus=src_path,
                  aug_ledger=ledger, aug_cut_seed=ac.CUT_SEED,
                  aug_n_cuts_per_episode=ac.N_CUTS_PER_EPISODE,
                  aug_avoid_last_chunks=ac.AVOID_LAST_CHUNKS,
                  aug_split_meta=split_meta,
                  aug_sample_ledger=dict(n_original=len(orig_all), n_subsamples=len(sub_samples),
                                        n_total=len(all_samples), n_train=len(train_aug),
                                        n_val=len(val_aug)),
                  aug_self_check=dict(wp_xy_mismatches=len(mismatches), split_matches_original=bool(same_split),
                                      five_sample_check_pass=bool(all_ok),
                                      n_self_check_samples=N_SELF_CHECK_SAMPLES),
                  aug_built_at=time.strftime("%Y-%m-%d %H:%M:%S")),
    ), out_path)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"\nsaved: {out_path} ({sz_mb:.2f} MB)")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
