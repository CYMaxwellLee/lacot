#!/usr/bin/env python
"""experiments/gen_sft/build_corpus_aug_v2.py —— rewrite-v1 步3（刀數加碼）：把「每條軌跡
切點數」參數化成 CLI 引數（本工單跑 {4, 8}），沿用 build_corpus_aug.py 的全部邏輯與自檢。

工單：「半路起點增強」的刀數加碼（2→4→8）能不能讓寫譜模型更好、更穩。本檔只負責建語料
+ 自檢＋長度分佈記帳；訓練/eval 走既有的 train_gen_sft_aug.sbatch / eval_rewrite_aug.sbatch
（兩者本來就用 env var 傳 CORPUS/GENSFT_CKPT，不需要新 sbatch）。2 刀版語料
（results/corpus_aug_v1.pt）直接引用既有檔案，本檔不重跑。

⛔ 新檔，不改動 aug_corpus_common.py / build_corpus_aug.py / gen_sft_common.py 任何一行——
   核心構造邏輯 100% reuse aug_corpus_common.build_augmented_samples()（它本來就接受任意
   n_cuts 參數，build_corpus_aug.py 只是沒把它暴露成 CLI，本檔把它暴露出來)。
⛔ CPU-only、輕量（同 build_corpus_aug.py：複用 corpus_v1.pt 已編碼好的 code_seq，唯一
   新算的是重新呼叫一次 rt.build_tasks() 拿 wp_local_idx，自檢①逐位元比對證明不是假設)。

用法：
  python build_corpus_aug_v2.py --n-cuts 4 --cut-seed 20260912
  python build_corpus_aug_v2.py --n-cuts 8 --cut-seed 20260913
輸出：results/corpus_aug_v2_c{n_cuts}.pt

⚠️ 若一條軌跡的可用切點候選不足 n_cuts 個：build_augmented_samples() 既有邏輯
   （k_take = min(n_cuts, len(cands))）已經處理——取到不重複的上限即可，並記進
   ledger['n_episodes_lt_n_cuts']，不是靜默漏做。本檔不需要新寫這段。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402
import aug_corpus_common as ac  # noqa: E402

BUILD_TASKS_SEED = 0   # ⛔ 必須跟 build_corpus.py 的常數一致
SPLIT_SEED = 42
VAL_FRAC = 0.1
N_SELF_CHECK_SAMPLES = 3   # 工單 acceptance_criteria 第2點指定「各抽 3 個」（v1 用 5，本檔照工單改 3）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cuts", type=int, required=True, choices=[4, 8],
                    help="每條 episode 的切點數；2 刀沿用既有 corpus_aug_v1.pt，本檔不接受 2")
    ap.add_argument("--cut-seed", type=int, required=True,
                    help="切點抽樣固定 seed——⛔ 工單要求 4/8 刀用不同的固定 seed，"
                         "呼叫端必須顯式傳入，刻意不給預設值以避免誤用同一顆種子")
    ap.add_argument("--avoid-last", type=int, default=ac.AVOID_LAST_CHUNKS)
    args = ap.parse_args()

    torch.set_num_threads(4)
    t0 = time.time()

    src_path = os.path.join(gc.RESULTS_DIR, "corpus_v1.pt")
    print(f"=== build_corpus_aug_v2: n_cuts={args.n_cuts} cut_seed={args.cut_seed} "
          f"讀既有語料 {src_path} ===")
    blob = torch.load(src_path, weights_only=False)
    orig_train, orig_val, orig_meta = blob["train"], blob["val"], blob["meta"]
    orig_all = orig_train + orig_val
    print(f"loaded: n_train={len(orig_train)} n_val={len(orig_val)} n_total={len(orig_all)}  "
          f"build_tasks_seed(meta)={orig_meta['build_tasks_seed']}  split_seed(meta)={orig_meta['split_seed']}")
    assert orig_meta["build_tasks_seed"] == BUILD_TASKS_SEED, (
        f"⛔ corpus_v1.pt 的 build_tasks_seed={orig_meta['build_tasks_seed']} 跟本檔常數 "
        f"{BUILD_TASKS_SEED} 不一致——停手回報")
    assert orig_meta["split_seed"] == SPLIT_SEED

    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()

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
        f"{len(tasks_by_ep)} 集——停手回報")

    # ---- 自檢①（同 build_corpus_aug.py 逐行同款）----
    mismatches = []
    for ei in orig_by_ep:
        a, b = orig_by_ep[ei]["wp_xy"], tasks_by_ep[ei]["wp_xy"]
        if a.shape != b.shape or not np.array_equal(a, b):
            mismatches.append(ei)
    print(f"\n自檢①（wp_xy 逐位元比對，全部 {len(orig_by_ep)} 集）："
          f"{'PASS 0 mismatches' if not mismatches else f'⛔ FAIL {len(mismatches)} mismatches'}")
    assert not mismatches, f"⛔ wp_xy 對不上的 episode（前10）: {mismatches[:10]}——停手回報"

    # ---- 建子樣本：唯一差異＝n_cuts/seed 從 CLI 來，函式本體 100% reuse ----
    sub_samples, ledger = ac.build_augmented_samples(orig_by_ep, tasks_by_ep, raw, seed=args.cut_seed,
                                                      n_cuts=args.n_cuts, avoid_last=args.avoid_last)
    print(f"\n增強子樣本 ledger（cut_seed={args.cut_seed} n_cuts={args.n_cuts} "
          f"avoid_last={args.avoid_last}）:")
    for k, v in ledger.items():
        print(f"  {k} = {v}")

    for s in orig_all:
        s["is_aug"] = False
        s["cut_chunk"] = None
    all_samples = orig_all + sub_samples
    print(f"\n樣本數帳：原始樣本={len(orig_all)}  子樣本={len(sub_samples)}  混合後總計={len(all_samples)}")

    # ---- split（沿用 aug_corpus_common.split_corpus_by_episode，同 split_seed=42）----
    train_aug, val_aug, split_meta = ac.split_corpus_by_episode(all_samples, split_seed=SPLIT_SEED,
                                                                 val_frac=VAL_FRAC)
    print(f"\nsplit(episode-level, split_seed={SPLIT_SEED}): {split_meta}")

    # ---- 自檢②：episode 級切分要跟 corpus_v1.pt 原本的切分完全相同 ----
    orig_train_eps = set(int(s["episode"]) for s in orig_train)
    orig_val_eps = set(int(s["episode"]) for s in orig_val)
    aug_train_eps = set(int(s["episode"]) for s in train_aug)
    aug_val_eps = set(int(s["episode"]) for s in val_aug)
    same_split = (orig_train_eps == aug_train_eps) and (orig_val_eps == aug_val_eps)
    print(f"\n自檢②（episode 級切分跟 corpus_v1.pt 原本切分逐集比對）："
          f"{'PASS 完全相同' if same_split else '⛔ FAIL 不同'}")
    assert same_split, "⛔ 增強後的 episode 級切分跟原始切分不一致——防洩漏保證破了，停手回報"

    # ---- 驗收①（工單 acceptance_criteria 第2點：各抽 3 個子樣本核對後綴+obs 位元一致）----
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
    print(f"驗收①自檢：{f'{N_SELF_CHECK_SAMPLES}/{N_SELF_CHECK_SAMPLES} PASS' if all_ok else '⛔ 有 FAIL，見上'}")
    assert all_ok, "⛔ 驗收①自檢沒有全過——構造邏輯有錯，停手修，不准帶著錯的語料往下訓練"

    # ---- 長度分佈記帳（acceptance_criteria 第1點：p25/50/75）----
    all_lens = np.array([len(s["code_seq"]) for s in all_samples])
    sub_lens = np.array([len(s["code_seq"]) for s in sub_samples])
    print(f"\n語料整體長度分佈（全部 {len(all_samples)} 樣本，code_seq len，含原樣本+子樣本）: "
          f"p25={np.percentile(all_lens, 25):.1f} p50={np.percentile(all_lens, 50):.1f} "
          f"p75={np.percentile(all_lens, 75):.1f}  min={all_lens.min()} max={all_lens.max()}")
    print(f"子樣本長度分佈（僅 is_aug=True 的 {len(sub_lens)} 個, code_seq len）: "
          f"p25={np.percentile(sub_lens, 25):.1f} p50={np.percentile(sub_lens, 50):.1f} "
          f"p75={np.percentile(sub_lens, 75):.1f}  min={sub_lens.min()} max={sub_lens.max()}")

    os.makedirs(gc.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(gc.RESULTS_DIR, f"corpus_aug_v2_c{args.n_cuts}.pt")
    torch.save(dict(
        train=train_aug, val=val_aug,
        meta=dict(**{k: v for k, v in orig_meta.items()},
                  aug_source_corpus=src_path,
                  aug_ledger=ledger, aug_cut_seed=args.cut_seed,
                  aug_n_cuts_per_episode=args.n_cuts,
                  aug_avoid_last_chunks=args.avoid_last,
                  aug_split_meta=split_meta,
                  aug_sample_ledger=dict(n_original=len(orig_all), n_subsamples=len(sub_samples),
                                        n_total=len(all_samples), n_train=len(train_aug),
                                        n_val=len(val_aug)),
                  aug_length_dist=dict(
                      all=dict(p25=float(np.percentile(all_lens, 25)), p50=float(np.percentile(all_lens, 50)),
                               p75=float(np.percentile(all_lens, 75)),
                               min=int(all_lens.min()), max=int(all_lens.max())),
                      sub_only=dict(p25=float(np.percentile(sub_lens, 25)), p50=float(np.percentile(sub_lens, 50)),
                                    p75=float(np.percentile(sub_lens, 75)),
                                    min=int(sub_lens.min()), max=int(sub_lens.max()))),
                  aug_self_check=dict(wp_xy_mismatches=len(mismatches), split_matches_original=bool(same_split),
                                      self_check_pass=bool(all_ok),
                                      n_self_check_samples=N_SELF_CHECK_SAMPLES),
                  aug_built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  aug_script="build_corpus_aug_v2.py（刀數加碼 rewrite-v1 步3，n_cuts 參數化）"),
    ), out_path)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"\nsaved: {out_path} ({sz_mb:.2f} MB)")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
