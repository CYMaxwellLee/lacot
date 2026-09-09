#!/usr/bin/env python
"""experiments/gen_sft/build_corpus_dyn.py —— 主人裁示①②：建『動態隨機起點 + 漂移擾動
校準』所需的軌跡級語料 corpus_dyn_v1.pt，並跑驗收標準①（動態切 20 個樣本位元自檢）
與④（擾動幅度分佈稽核）。

⛔ CPU-only、輕量，直接在本機跑、不經 slurm（同 build_corpus.py / build_corpus_aug.py
   慣例）。
⛔ 不改動任何既有檔案（含 aug_corpus_common.py / build_corpus_aug.py）——只 import。
⛔ drift_summary.json 唯讀引用（main repo lacot，不是這個 worktree）。
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import halfnorm  # noqa: E402

import gen_sft_common as gc  # noqa: E402
import aug_corpus_common as ac  # noqa: E402
import dyn_corpus_common as dc  # noqa: E402

BUILD_TASKS_SEED = 0     # ⛔ 必須跟 build_corpus.py 常數一致（同 build_corpus_aug.py 的理由）
SPLIT_SEED = 42
DRIFT_SUMMARY_PATH = "/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis/results/drift_summary.json"
SELFCHECK_SEED = 20260909777   # 只決定自檢抽哪 20 個 (episode,cut)，跟訓練用的 DYN_CUT_SEED/PERTURB_SEED 分開


def main():
    torch.set_num_threads(4)
    t0 = time.time()

    src_path = os.path.join(gc.RESULTS_DIR, "corpus_v1.pt")
    print(f"=== build_corpus_dyn: 讀既有語料 {src_path} ===")
    blob = torch.load(src_path, weights_only=False)
    orig_train, orig_val, orig_meta = blob["train"], blob["val"], blob["meta"]
    orig_all = orig_train + orig_val
    orig_by_ep = {int(s["episode"]): s for s in orig_all}
    train_eps = set(int(s["episode"]) for s in orig_train)
    val_eps = set(int(s["episode"]) for s in orig_val)
    print(f"loaded: n_train_eps={len(train_eps)} n_val_eps={len(val_eps)}  "
          f"build_tasks_seed(meta)={orig_meta['build_tasks_seed']}  split_seed(meta)={orig_meta['split_seed']}")
    assert orig_meta["build_tasks_seed"] == BUILD_TASKS_SEED
    assert orig_meta["split_seed"] == SPLIT_SEED

    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()

    # 重新呼叫 build_tasks 只為了拿 wp_local_idx（跟 build_corpus_aug.py 完全同一招）。
    starts, ends = gc.wv.episode_bounds(raw["terminals"])
    n_ep = len(starts)
    tasks, skipped_tasks = gc.rt.build_tasks(raw, n_traj=n_ep + 1, seed=BUILD_TASKS_SEED,
                                             ruler=ruler, rho=gc.RHO)
    tasks_by_ep = {int(t["episode"]): t for t in tasks}
    print(f"rebuilt tasks（僅為取 wp_local_idx）: {len(tasks)} usable  skipped={len(skipped_tasks)}  n_ep(raw)={n_ep}")
    assert set(orig_by_ep.keys()) == set(tasks_by_ep.keys()), (
        f"⛔ episode 集合對不上：corpus_v1 有 {len(orig_by_ep)} 集，重建 tasks 有 "
        f"{len(tasks_by_ep)} 集——停手回報，不能假設兩邊同構")

    # ---- 自檢①（沿用 build_corpus_aug.py 同款）：wp_xy 逐位元比對，全部集 ----
    mismatches = []
    for ei in orig_by_ep:
        a, b = orig_by_ep[ei]["wp_xy"], tasks_by_ep[ei]["wp_xy"]
        if a.shape != b.shape or not np.array_equal(a, b):
            mismatches.append(ei)
    print(f"\n自檢①（wp_xy 逐位元比對，全部 {len(orig_by_ep)} 集）："
          f"{'PASS 0 mismatches' if not mismatches else f'⛔ FAIL {len(mismatches)} mismatches'}")
    assert not mismatches, f"⛔ wp_xy 對不上的 episode（前10）: {mismatches[:10]}——停手回報"

    # ---- 建軌跡級 record（全部 episode）----
    obs_arr = raw["observations"]
    records = {}
    ledger = dict(n_episodes=0, n_episodes_zero_usable_cuts=0, n_candidates_total=0, n_usable_total=0)
    for ei in sorted(orig_by_ep.keys()):
        rec, stat = dc.build_trajectory_record(ei, orig_by_ep[ei], tasks_by_ep[ei]["wp_local_idx"], obs_arr)
        records[ei] = rec
        ledger["n_episodes"] += 1
        ledger["n_candidates_total"] += stat["n_candidates"]
        ledger["n_usable_total"] += stat["n_usable"]
        if stat["n_usable"] == 0:
            ledger["n_episodes_zero_usable_cuts"] += 1
    print(f"\n軌跡級 record ledger: {ledger}")

    train_records = [records[ei] for ei in sorted(train_eps)]
    val_records = [records[ei] for ei in sorted(val_eps)]
    print(f"split（沿用 corpus_v1.pt 原本 episode 級切分，split_seed={SPLIT_SEED}）: "
          f"n_train_episodes={len(train_records)}  n_val_episodes={len(val_records)}")

    # ---- 邊界安全性自檢：obs_chunks 形狀跟 usable_cuts 的 k 都在範圍內 ----
    for ei, rec in records.items():
        assert rec["obs_chunks"].shape == (rec["n_chunks"], 29)
        if len(rec["usable_cuts"]) > 0:
            assert rec["usable_cuts"].max() < rec["n_chunks"], f"⛔ episode={ei} usable_cuts 越界"
            assert rec["usable_cuts"].min() >= 1

    # ================= 擾動校準 =================
    print(f"\n=== 擾動校準（drift_summary.json 唯讀引用 {DRIFT_SUMMARY_PATH}）===")
    with open(DRIFT_SUMMARY_PATH) as f:
        drift = json.load(f)
    succ_p75 = float(drift["failure_precursor"]["succ_quantiles"]["75"])
    halfnorm_mult_p75 = float(halfnorm.ppf(0.75))  # standard half-normal(scale=1) 的 p75 分位數
    xy_sigma = succ_p75 / halfnorm_mult_p75
    print(f"drift succ_quantiles.p75(xy)={succ_p75:.6f}m（一手引用，failure_precursor.succ_quantiles）")
    print(f"halfnorm.ppf(0.75)(scale=1)={halfnorm_mult_p75:.6f}  =>  xy_sigma = {succ_p75:.6f}/{halfnorm_mult_p75:.6f} = {xy_sigma:.6f}")

    # 其餘 27 維 std：用 train episodes 的『全部 timestep』raw obs 算（不只 chunk 邊界）
    mask = np.zeros(obs_arr.shape[0], dtype=bool)
    for ei in train_eps:
        s0, e0 = int(starts[ei]), int(ends[ei])
        mask[s0:e0 + 1] = True
    train_obs = obs_arr[mask].astype(np.float64)
    other_std = train_obs[:, 2:29].std(axis=0)
    print(f"train 資料其餘 27 維 std：n_timesteps={train_obs.shape[0]}  "
          f"min={other_std.min():.4f}  median={np.median(other_std):.4f}  max={other_std.max():.4f}")
    print(f"OTHER_DIM_COEF（自訂，見 dyn_corpus_common.py 檔頭說明）= {dc.OTHER_DIM_COEF}")

    perturb_cfg = dict(xy_sigma=float(xy_sigma), other_coef=float(dc.OTHER_DIM_COEF),
                       other_std=other_std.astype(np.float64))

    # ---- 驗收標準④：擾動幅度分佈稽核（xy，1000 樣本）----
    audit_rng = np.random.default_rng(dc.PERTURB_SEED)
    mags = []
    for _ in range(dc.N_PERTURB_AUDIT):
        theta = audit_rng.uniform(0.0, 2.0 * np.pi)
        mag = abs(audit_rng.normal(0.0, xy_sigma))
        mags.append(mag)
    mags = np.asarray(mags)
    audit = dict(n=int(dc.N_PERTURB_AUDIT), seed=int(dc.PERTURB_SEED),
                p50=float(np.percentile(mags, 50)), p75=float(np.percentile(mags, 75)),
                p95=float(np.percentile(mags, 95)), mean=float(mags.mean()),
                target_p75=succ_p75, xy_sigma=float(xy_sigma))
    print(f"\n=== 驗收④：擾動幅度分佈稽核（n={dc.N_PERTURB_AUDIT}，PERTURB_SEED={dc.PERTURB_SEED}）===")
    print(f"  xy 擾動幅度  p50={audit['p50']:.4f}  p75={audit['p75']:.4f}  p95={audit['p95']:.4f}  "
          f"mean={audit['mean']:.4f}  (目標 p75≈{succ_p75:.4f}，量出來的 p75={audit['p75']:.4f}，"
          f"差={audit['p75']-succ_p75:+.4f})")

    # 其餘 27 維擾動量級檢查（附帶稽核，非驗收標準硬性項目，但工單設計點2要求『量級明顯
    # 小於 xy 主擾動』，這裡量出來備查）
    other_rng = np.random.default_rng(dc.PERTURB_SEED + 1)  # 獨立於上面 xy 稽核用的流，避免混淆
    other_norms = []
    for _ in range(dc.N_PERTURB_AUDIT):
        noise27 = other_rng.normal(0.0, dc.OTHER_DIM_COEF * other_std)
        other_norms.append(np.linalg.norm(noise27))
    other_norms = np.asarray(other_norms)
    print(f"  其餘27維擾動 L2 norm  p50={np.percentile(other_norms,50):.4f}  "
          f"p75={np.percentile(other_norms,75):.4f}  (vs xy 擾動 p75={audit['p75']:.4f}，"
          f"比例={np.percentile(other_norms,75)/audit['p75']:.3f}——{'明顯小於' if np.percentile(other_norms,75) < 0.5*audit['p75'] else '⚠️ 未明顯小於'} xy 主擾動)")

    # ---- 驗收標準①：動態切 20 個樣本，位元自檢（code_seq 後綴 + obs 位元一致）----
    print(f"\n=== 驗收①：動態切 {dc.N_SELFCHECK_DYN} 個樣本，位元自檢（強制 cut 分支，perturb=None）===")
    sc_rng = np.random.default_rng(SELFCHECK_SEED)
    train_eps_sorted = sorted(train_eps)
    all_ok = True
    rows = []
    n_checked = 0
    tries = 0
    while n_checked < dc.N_SELFCHECK_DYN and tries < dc.N_SELFCHECK_DYN * 20:
        tries += 1
        ei = int(sc_rng.choice(train_eps_sorted))
        rec = records[ei]
        if len(rec["usable_cuts"]) == 0:
            continue
        sample = dc.draw_dynamic_sample(rec, sc_rng, use_full_prob=0.0, perturb_rng=None, perturb_cfg=None)
        # sample 是強制 cut（use_full_prob=0 且 rec 有 usable_cuts），反推它用的是哪個 k：
        # n_chunks_sub = n_chunks - k  =>  k = n_chunks - n_chunks_sub
        k = rec["n_chunks"] - sample["n_chunks"]
        expect_codes = rec["code_seq"][k:]
        codes_ok = np.array_equal(sample["code_seq"], expect_codes)
        expect_obs = obs_arr[rec["s0_idx"] + 4 * k].astype(np.float32)
        obs_ok = np.array_equal(sample["s0_obs"], expect_obs)
        obs_l2 = float(np.linalg.norm(sample["s0_obs"].astype(np.float64) - expect_obs.astype(np.float64)))
        rows.append(dict(episode=ei, cut_k=int(k), parent_n_chunks=rec["n_chunks"],
                         sub_n_chunks=int(sample["n_chunks"]), codes_ok=bool(codes_ok),
                         obs_ok=bool(obs_ok), obs_l2=obs_l2))
        print(f"  episode={ei:5d} cut_k={k:2d}/{rec['n_chunks']}  code_seq len={len(sample['code_seq']):2d} "
              f"(parent len={len(rec['code_seq'])}): "
              f"{'PASS bit-exact suffix' if codes_ok else '⛔ FAIL NOT suffix'}   "
              f"s0_obs: {'PASS bit-exact' if obs_ok else '⛔ FAIL'} (L2={obs_l2:.8f})   "
              f"M={sample['M']} (parent M={rec['M']})")
        all_ok = all_ok and codes_ok and obs_ok
        n_checked += 1
    assert n_checked == dc.N_SELFCHECK_DYN, f"⛔ 自檢只抽到 {n_checked}/{dc.N_SELFCHECK_DYN} 個可用樣本，停手回報"
    print(f"驗收①自檢：{f'{n_checked}/{n_checked} PASS' if all_ok else '⛔ 有 FAIL，見上'}")
    assert all_ok, "⛔ 驗收①自檢沒有全過——構造邏輯有錯，停手修，不准帶著錯的語料往下訓練"

    self_check = dict(n_checked=n_checked, all_ok=bool(all_ok), seed=int(SELFCHECK_SEED), rows=rows)

    # ---- 存檔 ----
    os.makedirs(gc.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(gc.RESULTS_DIR, "corpus_dyn_v1.pt")
    meta = dict(
        source_corpus=src_path, build_tasks_seed=BUILD_TASKS_SEED, split_seed=SPLIT_SEED,
        n_train_episodes=len(train_records), n_val_episodes=len(val_records),
        avoid_last_chunks=ac.AVOID_LAST_CHUNKS, use_full_prob=dc.USE_FULL_PROB,
        dyn_cut_seed=dc.DYN_CUT_SEED, perturb_seed=dc.PERTURB_SEED,
        other_dim_coef=dc.OTHER_DIM_COEF,
        obs_dim=29, xy_dims=[0, 1], other_dims=list(range(2, 29)),
        ledger=ledger,
        perturb_calibration=dict(
            drift_source=DRIFT_SUMMARY_PATH, drift_succ_p75_xy_target=succ_p75,
            halfnorm_ppf_0p75=halfnorm_mult_p75, xy_sigma=float(xy_sigma),
            other_dim_coef=float(dc.OTHER_DIM_COEF),
            other_std_train=other_std.tolist(),
            other_norm_audit_p50=float(np.percentile(other_norms, 50)),
            other_norm_audit_p75=float(np.percentile(other_norms, 75)),
        ),
        perturb_audit_xy=audit,
        self_check_dyn20=dict(n_checked=n_checked, all_ok=bool(all_ok), seed=int(SELFCHECK_SEED)),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    torch.save(dict(train_episodes=train_records, val_episodes=val_records, meta=meta), out_path)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"\nsaved: {out_path} ({sz_mb:.2f} MB)")

    # self_check 的完整 rows 另外存一份 json（含 20 條逐條紀錄，供 NOTE 附錄引用一手 log）
    sc_path = os.path.join(gc.RESULTS_DIR, "dyn_selfcheck20.json")
    gc.wv.save_json(self_check, sc_path)
    print(f"saved: {sc_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
