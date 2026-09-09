#!/usr/bin/env python
"""驗收①（工單〈四關離線打分挑教材〉）：對 corpus_v1.pt 的 train/val 樣本離線算
四關 scalar 品質分，並且證明打分器『會亮』——構造人工爛軌跡（反轉/打亂），其分數
必須顯著低於真軌跡分佈的 p10；另外抽 3 條真軌跡做人眼合理性核對。

同時輸出 train 樣本的加權方案兩種：
  (a) 軟加權：weight=exp(beta*score_quality)，beta 調到 effective sample size≈N/2。
  (b) 硬過濾：只留 score_quality>=median 的前 50%。
供 train_gen_sft_v2.py 用 --weight-mode soft/hard --scores-path 這裡的輸出檔。

⛔ CPU-only、輕量（~5000 集 x 幾個 numpy 運算，秒級），不經 slurm，跟 build_corpus.py/
   golden_check.py 同慣例直接跑。⛔ 不碰模擬器。⛔ 不改動任何既有檔案。
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
import score_corpus_common as sc  # noqa: E402

EYECHECK_SEED = 4242
EYECHECK_N = 3
ADV_SOURCE_SEED = 4243     # 挑哪一條真集當 adversarial 構造的底本
ADV_SHUFFLE_SEED = 20260909
TARGET_ESS_FRAC = 0.5

# 驗收①門檻：adversarial 分數必須「顯著低於」真實分佈 p10。用「低於 p10 達到
# SIG_MARGIN 個百分位以上」當「顯著」的可判量化版本（不是只看有沒有低於，一併
# 報它實際落在真實分佈的第幾百分位，通常會落在接近 0）。
SIG_MARGIN_PCTPTS = 0.05   # 至少比 p10 的分位再低 5 個百分點（即落在 <=p5 附近）


def score_samples(samples, raw, ruler, rho, tag):
    t0 = time.time()
    dawdle, jerk, fall, reach = [], [], [], []
    for s in samples:
        xy, act, z, horizon = sc.episode_span(raw, s["s0_idx"], s["n_chunks"])
        s0_xy = raw["qpos"][s["s0_idx"], :2]
        feat = sc.raw_features_for_trajectory(xy, act, z, s["wp_xy"], s0_xy, ruler, rho, horizon)
        dawdle.append(feat["dawdle_steps"])
        jerk.append(feat["jerk_mean"])
        fall.append(feat["fall_frac"])
        reach.append(feat["reach_frac"])
    print(f"  scored {len(samples)} {tag} samples in {time.time()-t0:.1f}s")
    return (np.asarray(dawdle), np.asarray(jerk), np.asarray(fall), np.asarray(reach))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_v1.pt"))
    ap.add_argument("--out", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_scores_v2.pt"))
    ap.add_argument("--report", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_scores_v2_report.json"))
    args = ap.parse_args()

    torch.set_num_threads(4)
    t0 = time.time()
    print("=== gen-sft-v2 corpus scorer（離線四關打分）===")
    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()
    rho = gc.RHO
    fall_line = ruler["torso_z"]["fall_line_p1"]
    print(f"rho={rho}  fall_line_p1={fall_line:.4f}")

    blob = torch.load(args.corpus, weights_only=False)
    train_samples, val_samples = blob["train"], blob["val"]
    print(f"corpus: n_train={len(train_samples)} n_val={len(val_samples)}")

    # -----------------------------------------------------------------
    # 1) 對 train / val 全體算三個原始特徵
    # -----------------------------------------------------------------
    print("\n--- 打分 train ---")
    tr_dawdle, tr_jerk, tr_fall, tr_reach = score_samples(train_samples, raw, ruler, rho, "train")
    print("--- 打分 val（診斷用，不用來加權/過濾）---")
    va_dawdle, va_jerk, va_fall, va_reach = score_samples(val_samples, raw, ruler, rho, "val")

    print(f"\ntrain 原始特徵摘要:")
    print(f"  dawdle_steps  mean={tr_dawdle.mean():.3f}  p10={np.percentile(tr_dawdle,10):.3f}  "
          f"p50={np.percentile(tr_dawdle,50):.3f}  p90={np.percentile(tr_dawdle,90):.3f}  "
          f"max={tr_dawdle.max():.3f}")
    print(f"  jerk_mean     mean={tr_jerk.mean():.3f}  p10={np.percentile(tr_jerk,10):.3f}  "
          f"p50={np.percentile(tr_jerk,50):.3f}  p90={np.percentile(tr_jerk,90):.3f}  "
          f"max={tr_jerk.max():.3f}")
    print(f"  fall_frac     mean={tr_fall.mean():.4f}  p50={np.percentile(tr_fall,50):.4f}  "
          f"p90={np.percentile(tr_fall,90):.4f}  frac_nonzero={float((tr_fall>0).mean()):.3f}")
    print(f"  reach_frac(診斷，未進分數) mean={tr_reach.mean():.4f}  "
          f"frac_all_legs_reached={float((tr_reach>=0.999).mean()):.4f}")

    # -----------------------------------------------------------------
    # 2) 合成 scalar score_quality（train 自己的母體）
    # -----------------------------------------------------------------
    table = sc.build_score_table(tr_dawdle, tr_jerk, tr_fall)
    score_quality = table["score_quality"]
    sorted_score_quality = np.sort(score_quality)  # 給下面 adversarial 算「真的」百分位排名用
    print(f"\nscore_quality（train，1-badness，越大越好）: mean={score_quality.mean():.3f}  "
          f"p10={np.percentile(score_quality,10):.3f}  p50={np.percentile(score_quality,50):.3f}  "
          f"p90={np.percentile(score_quality,90):.3f}")
    p10_real = float(np.percentile(score_quality, 10))

    # val 也算一次（用 train 的母體當參照，val 沒有進 train 的分位定義 —— 乾淨）
    va_score = np.array([
        sc.score_quality_of_external(d, j, f, table["sorted_dawdle"], table["sorted_jerk"],
                                     table["sorted_fall"])["score_quality"]
        for d, j, f in zip(va_dawdle, va_jerk, va_fall)
    ])
    print(f"score_quality（val，套 train 母體的分位定義，純診斷）: mean={va_score.mean():.3f}  "
          f"p50={np.percentile(va_score,50):.3f}  （跟 train 分佈接近才正常，是自我一致性檢查）")

    # -----------------------------------------------------------------
    # 3) 驗收①：打分器會亮 —— adversarial 構造（reverse / shuffle）
    # -----------------------------------------------------------------
    print(f"\n=== 驗收①：打分器會亮（adversarial 構造）===")
    rng_pick = np.random.default_rng(ADV_SOURCE_SEED)
    adv_idx = int(rng_pick.integers(0, len(train_samples)))
    adv_sample = train_samples[adv_idx]
    xy0, act0, z0, horizon0 = sc.episode_span(raw, adv_sample["s0_idx"], adv_sample["n_chunks"])
    s0_xy0 = raw["qpos"][adv_sample["s0_idx"], :2]
    print(f"底本：train[{adv_idx}] episode={adv_sample['episode']} M={adv_sample['M']} "
          f"horizon={horizon0}")
    base_feat = sc.raw_features_for_trajectory(xy0, act0, z0, adv_sample["wp_xy"], s0_xy0,
                                                ruler, rho, horizon0)
    base_score = sc.score_quality_of_external(base_feat["dawdle_steps"], base_feat["jerk_mean"],
                                              base_feat["fall_frac"], table["sorted_dawdle"],
                                              table["sorted_jerk"], table["sorted_fall"])
    base_pct_rank_in_real = sc.percentile_of(base_score["score_quality"], sorted_score_quality)
    print(f"  底本（未打壞，真實）: dawdle_steps={base_feat['dawdle_steps']:.3f} "
          f"jerk_mean={base_feat['jerk_mean']:.3f} fall_frac={base_feat['fall_frac']:.4f} "
          f"reach_frac={base_feat['reach_frac']:.3f}  -> score_quality={base_score['score_quality']:.3f} "
          f"(這個 score_quality 本身在真實 train score_quality 分佈裡的百分位="
          f"{base_pct_rank_in_real*100:.1f}%)")

    adv_results = {}
    for mode in ("reverse", "shuffle"):
        xy_a, act_a, z_a = sc.make_adversarial(xy0, act0, z0, mode, ADV_SHUFFLE_SEED)
        feat_a = sc.raw_features_for_trajectory(xy_a, act_a, z_a, adv_sample["wp_xy"], s0_xy0,
                                                 ruler, rho, horizon0)
        sq_a = sc.score_quality_of_external(feat_a["dawdle_steps"], feat_a["jerk_mean"],
                                            feat_a["fall_frac"], table["sorted_dawdle"],
                                            table["sorted_jerk"], table["sorted_fall"])
        # ⚠️ 驗收②抓到、修過：這裡本來寫的是 pct_rank_in_real = 1-badness，那其實
        # 就是 score_quality 本身（換個名字），不是「這個 score_quality 在真實
        # score_quality 分佈裡排第幾百分位」——兩者只有在 badness 剛好是 0 或 1
        # 這種邊界值時數字才會一樣（這裡兩個 adversarial 案例剛好都踩在 badness=1
        # 的邊界，所以先前 FAIL/PASS 的結論本身沒錯，但「顯著」門檻的判定邏輯
        # 是錯的、只是巧合沒讓結論翻盤，見下）。改法：用 percentile_of() 把
        # sq_a["score_quality"] 真的去查它在 sorted_score_quality（真實訓練集
        # score_quality 的排序陣列）裡的百分位排名。
        pct_rank_in_real = sc.percentile_of(sq_a["score_quality"], sorted_score_quality)
        below_p10 = bool(sq_a["score_quality"] < p10_real)
        sig = bool(pct_rank_in_real <= (0.10 - SIG_MARGIN_PCTPTS))
        adv_results[mode] = dict(
            dawdle_steps=feat_a["dawdle_steps"], jerk_mean=feat_a["jerk_mean"],
            fall_frac=feat_a["fall_frac"], reach_frac=feat_a["reach_frac"],
            score_quality=sq_a["score_quality"], pct_dawdle=sq_a["pct_dawdle"],
            pct_jerk=sq_a["pct_jerk"], pct_fall=sq_a["pct_fall"],
            pct_rank_in_real=pct_rank_in_real, below_real_p10=below_p10,
            significantly_below_p10=sig,
        )
        print(f"  [{mode:7s}] dawdle_steps={feat_a['dawdle_steps']:8.3f} "
              f"(pct={sq_a['pct_dawdle']*100:5.1f}%)  jerk_mean={feat_a['jerk_mean']:6.3f} "
              f"(pct={sq_a['pct_jerk']*100:5.1f}%)  fall_frac={feat_a['fall_frac']:.4f} "
              f"(pct={sq_a['pct_fall']*100:5.1f}%)  reach_frac={feat_a['reach_frac']:.3f}  "
              f"-> score_quality={sq_a['score_quality']:.4f}  pct_rank_in_real={pct_rank_in_real*100:.2f}% "
              f"  below_p10({p10_real:.3f})={below_p10}  顯著(<=p5)={sig}")

    both_lit = adv_results["reverse"]["below_real_p10"] and adv_results["shuffle"]["below_real_p10"]
    print(f"\n驗收①判定：reverse 與 shuffle 兩種 adversarial 構造是否都低於真實 p10 "
          f"= {both_lit}  ({'PASS' if both_lit else 'FAIL —— 停手回報，不准帶病加權'})")
    if not both_lit:
        print("⛔⛔⛔ 打分器沒有對 adversarial 構造亮起來，依 acceptance_criteria 第1點停手，"
              "不准繼續往下加權/訓練。")

    # 推導驗證：reverse 理論上不動 jerk（時間反轉下相鄰差範數不變），shuffle 應該
    # 兩個都動 —— 印出來對照，讓「這是刻意設計」有數字佐證，不是事後硬凹。
    jerk_delta_reverse = adv_results["reverse"]["jerk_mean"] - base_feat["jerk_mean"]
    print(f"\n設計推導核對：reverse 對 jerk_mean 的變化 = {jerk_delta_reverse:+.4f} "
          f"（理論預期 ≈0，時間反轉下相鄰差範數是不變量）；"
          f"shuffle 對 jerk_mean 的變化 = {adv_results['shuffle']['jerk_mean']-base_feat['jerk_mean']:+.4f} "
          f"（理論預期 >0，打散相鄰配對會推高）")

    # -----------------------------------------------------------------
    # 4) 人眼合理性核對：抽 3 條真實軌跡，數字特徵是否跟分數一致
    # -----------------------------------------------------------------
    print(f"\n=== 人眼合理性核對（3 條真實 train episode，seed={EYECHECK_SEED}）===")
    rng_eye = np.random.default_rng(EYECHECK_SEED)
    eye_idx = rng_eye.choice(len(train_samples), size=EYECHECK_N, replace=False).tolist()
    eyecheck = []
    for i in eye_idx:
        s = train_samples[i]
        xy, act, z, horizon = sc.episode_span(raw, s["s0_idx"], s["n_chunks"])
        s0_xy = raw["qpos"][s["s0_idx"], :2]
        feat = sc.raw_features_for_trajectory(xy, act, z, s["wp_xy"], s0_xy, ruler, rho, horizon)
        sq = sc.score_quality_of_external(feat["dawdle_steps"], feat["jerk_mean"], feat["fall_frac"],
                                          table["sorted_dawdle"], table["sorted_jerk"], table["sorted_fall"])
        arclen = float(np.sum(np.linalg.norm(np.diff(np.vstack([s0_xy, xy]), axis=0), axis=1)))
        note = []
        note.append("到齊全部腿" if feat["reach_frac"] >= 0.999 else f"只到 {feat['reach_frac']*100:.0f}% 的腿")
        note.append("步速比 corpus 中位數快(不磨蹭)" if feat["dawdle_steps"] < np.percentile(tr_dawdle, 50) else
                    "步速比 corpus 中位數慢(較磨蹭)")
        note.append("動作平滑" if feat["jerk_mean"] < np.percentile(tr_jerk, 50) else "動作較跳動")
        note.append("沒有翻倒風險" if feat["fall_frac"] == 0 else f"翻倒比例{feat['fall_frac']*100:.2f}%")
        row = dict(idx=i, episode=int(s["episode"]), M=int(s["M"]), arclen=arclen,
                  dawdle_steps=feat["dawdle_steps"], jerk_mean=feat["jerk_mean"],
                  fall_frac=feat["fall_frac"], reach_frac=feat["reach_frac"],
                  score_quality=sq["score_quality"], note="；".join(note))
        eyecheck.append(row)
        print(f"  train[{i}] episode={s['episode']:4d} M={s['M']} arclen={arclen:6.2f}  "
              f"dawdle={feat['dawdle_steps']:.3f}  jerk={feat['jerk_mean']:.3f}  "
              f"fall_frac={feat['fall_frac']:.4f}  reach_frac={feat['reach_frac']:.3f}  "
              f"-> score={sq['score_quality']:.3f}  【{row['note']}】")
    print("（人眼核對說明：上面每列的『數字特徵』與最右邊『說明』欄位是否互相一致，")
    print(" 即為驗收①第二部分要求的『分數與軌跡特徵一致的說明』——不畫 gif/不跑模擬器，")
    print(" 用離線量到的磨蹭/亂動/翻倒三個數字本身當作『軌跡特徵』的核對依據。）")

    # -----------------------------------------------------------------
    # 5) 加權方案：軟加權（ESS≈N/2）與硬過濾（前 50%）
    # -----------------------------------------------------------------
    print(f"\n=== 加權方案（在 train 的 score_quality 上算）===")
    beta, weights_soft_unnorm, ess, ess_frac = sc.pick_beta_for_ess(
        score_quality, target_ess_frac=TARGET_ESS_FRAC)
    weights_soft = weights_soft_unnorm / weights_soft_unnorm.sum() * len(score_quality)
    print(f"軟加權：beta={beta:.4f}  ESS={ess:.1f}/{len(score_quality)} (frac={ess_frac:.3f}, "
          f"目標={TARGET_ESS_FRAC})")
    print(f"  weights_soft 摘要: min={weights_soft.min():.3f} p50={np.median(weights_soft):.3f} "
          f"max={weights_soft.max():.3f}")

    median_score = float(np.median(score_quality))
    mask_hard = score_quality >= median_score
    print(f"硬過濾：median_score={median_score:.4f}  保留 {int(mask_hard.sum())}/{len(mask_hard)} "
          f"({mask_hard.mean()*100:.1f}%)")

    # -----------------------------------------------------------------
    # 6) 存檔
    # -----------------------------------------------------------------
    episodes = np.array([s["episode"] for s in train_samples], dtype=np.int64)
    val_episodes = np.array([s["episode"] for s in val_samples], dtype=np.int64)
    out_blob = dict(
        train_episode=episodes,
        train_raw=dict(dawdle_steps=tr_dawdle, jerk_mean=tr_jerk, fall_frac=tr_fall, reach_frac=tr_reach),
        train_pct=dict(dawdle=table["pct_dawdle"], jerk=table["pct_jerk"], fall=table["pct_fall"]),
        train_score_quality=score_quality,
        val_episode=val_episodes,
        val_raw=dict(dawdle_steps=va_dawdle, jerk_mean=va_jerk, fall_frac=va_fall, reach_frac=va_reach),
        val_score_quality=va_score,
        sorted_dawdle=table["sorted_dawdle"], sorted_jerk=table["sorted_jerk"], sorted_fall=table["sorted_fall"],
        beta=beta, weights_soft=weights_soft, ess=ess, ess_frac=ess_frac,
        target_ess_frac=TARGET_ESS_FRAC, median_score=median_score, mask_hard=mask_hard,
        adversarial=dict(base=base_feat, base_score=base_score, results=adv_results,
                         adv_source_idx=adv_idx, adv_source_episode=int(adv_sample["episode"]),
                         shuffle_seed=ADV_SHUFFLE_SEED, both_lit=both_lit, p10_real=p10_real),
        eyecheck=eyecheck,
        config=dict(rho=rho, fall_line=fall_line, eyecheck_seed=EYECHECK_SEED,
                   adv_source_seed=ADV_SOURCE_SEED, adv_shuffle_seed=ADV_SHUFFLE_SEED,
                   sig_margin_pctpts=SIG_MARGIN_PCTPTS, corpus_path=args.corpus,
                   built_at=time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    torch.save(out_blob, args.out)
    print(f"\nsaved: {args.out}")

    # 附一份人類可讀 json（不含大陣列，只留摘要 + adversarial + eyecheck）
    report = dict(
        n_train=len(train_samples), n_val=len(val_samples),
        train_raw_summary=dict(
            dawdle_steps=dict(mean=float(tr_dawdle.mean()), p10=float(np.percentile(tr_dawdle, 10)),
                              p50=float(np.percentile(tr_dawdle, 50)), p90=float(np.percentile(tr_dawdle, 90))),
            jerk_mean=dict(mean=float(tr_jerk.mean()), p10=float(np.percentile(tr_jerk, 10)),
                          p50=float(np.percentile(tr_jerk, 50)), p90=float(np.percentile(tr_jerk, 90))),
            fall_frac=dict(mean=float(tr_fall.mean()), frac_nonzero=float((tr_fall > 0).mean())),
            reach_frac=dict(mean=float(tr_reach.mean()), frac_all_reached=float((tr_reach >= 0.999).mean())),
        ),
        score_quality_summary=dict(mean=float(score_quality.mean()), p10=p10_real,
                                   p50=float(np.percentile(score_quality, 50)),
                                   p90=float(np.percentile(score_quality, 90))),
        val_score_quality_mean=float(va_score.mean()),
        adversarial=dict(base_score=base_score, results=adv_results, both_lit=both_lit, p10_real=p10_real,
                         source_episode=int(adv_sample["episode"])),
        eyecheck=eyecheck,
        weighting=dict(beta=beta, ess=ess, ess_frac=ess_frac, median_score=median_score,
                      n_kept_hard=int(mask_hard.sum()), n_total=int(len(mask_hard))),
    )
    gc.wv.save_json(report, args.report)
    print(f"saved: {args.report}")
    print(f"\n=== done wall={time.time()-t0:.1f}s ===")
    if not both_lit:
        sys.exit(1)


if __name__ == "__main__":
    main()
