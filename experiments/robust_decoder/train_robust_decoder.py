#!/usr/bin/env python
"""Robust decoder fine-tune —— decoder 吃「偏移的姿態」（obs 條件加噪）。

【在證明什麼、判準是什麼】
證明：字典 decoder 在訓練時見過「偏移的姿態」之後，開環接力執行會變穩。
本檔只負責「訓練」這一步；判準（per-leg 到達率 vs .554）由同目錄
eval_all.py 在 200 題教材接力上量，這裡只保證：
  (a) encoder 與 codebook 真的 frozen（不被本檔任何一行改動）；
  (b) decoder 的重建 loss 用跟原配方一樣的 F.mse_loss(recon, x)；
  (c) obs 噪音的校準有出處（見 --sigma-xy 說明），不是拍腦袋。

背景：docs/NOTE-2026-09-09-drift-analysis.md 定讞 —— schedule 版教材接力
per-leg 只有 .554，失敗的大頭是「開環執行漂移」（decoder 執行時吃到的 obs
離訓練時看過的乾淨姿態越來越遠，decoder 對這種偏移沒看過、外推失效，
而且自我放大：吃到越偏的 obs → 下一段動作更歪 → obs 更偏）。
病理指向：decoder 訓練時只見過資料裡的「乾淨」姿態（段起點 obs 沒有加過
任何擾動）。藥 = fine-tune 時把「執行時真的會遇到的偏移」餵給 decoder 看過
（obs 條件輸入加噪），其餘都不動：
  - encoder 定義「字」（code）：完全 frozen，一行 grad 都不流過去。
  - VQ codebook（vq.embed 等 EMA buffer）：frozen —— 本檔繞過
    CondVQVAE.forward()／VQEMA.forward() 整段（那段函式一被呼叫且 training=True
    就會用 no_grad 區塊動 EMA buffer），改成手刻三行：encoder(x) -> quantize_idx
    -> embed[idx]，且整段包在 torch.no_grad() 裡，VQEMA.forward 完全不會被呼叫
    到，EMA/dead-code-restart 邏輯沒有機會執行 —— 不是「大機率不動」，是
    「這條路徑上不存在能動它的程式碼」。
  - obs_mean/obs_std（decoder 正規化用的常數）：frozen，不重算、不 set_obs_stats。
    噪音加在「正規化之前」的原始 obs 空間（obs' = obs + eps），跟原本
    norm_obs() 的定義完全相容，decoder 看到的仍是「(obs'-mean)/std」。
  ⇒ 教材字串（hindsight code 的定義）不變，「唯一變因＝decoder」的對照
    邏輯成立。

噪音校準（出處＝experiments/walk_verify/drift_analysis/results/drift_summary.json
的 failure_precursor 欄，2026-09-09 一手量測）：
  - 失敗門檻（最佳單一 accuracy 門檻）        thr = 0.2352 m
  - 成功 leg 開始時漂移 p75                   succ.q75 = 0.3580 m
  - 失敗 leg 開始時漂移中位數                 fail.q50 = 1.1831 m
  --sigma-xy 三檔對齊（工單定死的三個值，這裡標明對齊到哪個參考點）：
    0.10 ≈ 門檻(.235) 之下的次門檻劑量（decoder 看過「還沒壞」等級的偏移）
    0.35 ≈ 成功 leg p75（.358，幾乎貼合）—— decoder 看過「還走得到」的漂移上緣
    1.00 ≈ 失敗 leg 中位數（1.183）的保守整數近似（不是精確貼合，是同量級、
           取一個好記的整數；1.0 本身也還在 fail_quantiles 的 25~50 百分位之間，
           不是外插值）
    0.00 ＝ 對照顆（sanity：fine-tune 管線本身不該傷分，見 eval_all.py 判準①）
  其餘 27 維（姿態/速度，obs dims 2..28）：沒有一個「漂移公尺數」可以直接套，
  改用「跟 xy 同一個相對劑量」——
    rho = sigma_xy / mean(obs_std[xy])         （xy 在 train 資料上的自然展幅）
    noise_std[other_dim] = rho * obs_std[other_dim]
  這樣 sigma_xy=0 時其餘維度噪音也是 0（對照顆乾淨），且劑量隨 xy 檔位等比放大，
  不是另外拍一組獨立係數。obs_std 直接讀 ckpt 已存的 buffer（P0 訓練時用
  train split 算的），不重算——重算版在下面 REPRO CHECK 會另外核對一致，
  抓「資料是不是被誰動過」這種沉默失敗。

⛔ 不改動任何既有檔案；只 import 同目錄上一層 walk_verify 的 wv_common
   （純函式/模型定義，import 無副作用）。
⛔ 本檔跑在 sbatch（GPU，見 train_robust_decoder.sbatch）；device 預設 cuda，
   找不到 GPU 就印警告退回 cpu（不 hard fail —— 這顆模型很小，cpu 也能跑，
   但正常情況不應該發生，發生了要讓人看得到）。

用法：
  python train_robust_decoder.py --sigma-xy 0.0   --tag sigma0
  python train_robust_decoder.py --sigma-xy 0.1   --tag sigma0.1
  python train_robust_decoder.py --sigma-xy 0.35  --tag sigma0.35
  python train_robust_decoder.py --sigma-xy 1.0   --tag sigma1.0
"""
import argparse
import os
import sys
import time

import torch  # noqa: E402
_GPU_VISIBLE_AT_IMPORT = torch.cuda.is_available()  # 快取住 GPU 可見性（已知雷，見工單 context）

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.normpath(os.path.join(HERE, "..", "walk_verify"))
sys.path.insert(0, WV_DIR)

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wv_common as wv  # noqa: E402 - 既有共用模組，未修改

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
DEFAULT_BASE_CKPT = os.path.join(WV_DIR, "results", "p0_dict_v1_L4K32_50k.pt")

# 噪音校準的一手來源（供 NOTE 引用；本檔不讀這個 json，數字已手抄進上面 docstring
# 並在下面 print 出來以便跟 json 對帳）。
DRIFT_SUMMARY_JSON = os.path.join(WV_DIR, "drift_analysis", "results", "drift_summary.json")
CALIB_REF = dict(fail_threshold_m=0.23523661929341885,
                 succ_leg_start_p75_m=0.357982619350346,
                 fail_leg_start_median_m=1.1831176980018214)


def evaluate_recon(model, x_all, o_all, noise_std_t, device, repeats, nrng, chunk=65536):
    """回傳 (mse_noisy_avg_over_repeats, mse_clean)。eval mode 概念上等價（這個架構沒有
    dropout/batchnorm，train()/eval() 對前向計算沒有數值差異，這裡仍用 no_grad 關掉梯度）。
    """
    n = len(x_all)
    with torch.no_grad():
        # clean（noise=0）
        se_clean, ntot = 0.0, 0
        for i in range(0, n, chunk):
            x, o = x_all[i:i + chunk].to(device), o_all[i:i + chunk].to(device)
            idx = model.encode_idx(x)
            recon = model.decode_from_idx(idx, o)
            se_clean += float(((recon - x) ** 2).sum().item())
            ntot += x.numel()
        mse_clean = se_clean / max(ntot, 1)

        # noisy（repeats 次獨立噪音抽樣取平均，降低 eval 曲線本身的抽樣雜訊）
        mse_noisy_list = []
        for _ in range(repeats):
            se, ntot2 = 0.0, 0
            for i in range(0, n, chunk):
                x, o = x_all[i:i + chunk], o_all[i:i + chunk]
                eps = nrng.normal(0.0, 1.0, size=o.shape).astype(np.float32) * noise_std_t.numpy()[None, :]
                o_noisy = (o.numpy() + eps)
                x_d, o_d = x.to(device), torch.from_numpy(o_noisy).to(device)
                idx = model.encode_idx(x_d)
                recon = model.decode_from_idx(idx, o_d)
                se += float(((recon - x_d) ** 2).sum().item())
                ntot2 += x_d.numel()
            mse_noisy_list.append(se / max(ntot2, 1))
    return float(np.mean(mse_noisy_list)), float(mse_clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=str, default=DEFAULT_BASE_CKPT)
    ap.add_argument("--sigma-xy", type=float, required=True,
                     help="obs dims[0,1]（xy，公尺）加噪的 std；0=對照顆")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 20260909 + int(round(args.sigma_xy * 1000))
    tag = args.tag or f"sigma{args.sigma_xy:g}"
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ --device cuda 但 torch.cuda.is_available()=False（GPU_at_import="
              f"{_GPU_VISIBLE_AT_IMPORT}），退回 cpu。這顆模型很小 cpu 也能跑，"
              "但正常情況（sbatch --gres=gpu:1）不應該發生，如實印出。")
        device = "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")

    print(f"=== robust-decoder fine-tune  tag={tag}  sigma_xy={args.sigma_xy}  "
          f"steps={args.steps}  seed={args.seed}  device={device} ===")
    print(f"噪音校準參考點（一手來源：{DRIFT_SUMMARY_JSON}）：{CALIB_REF}")
    t_all = time.time()

    # ---------------- 資料（跟 P0 同配方：train split 用來源同一份、split_seed=42）
    raw = wv.load_npz(args.data_dir, DATASET, "train")
    seg_starts = wv.cut_segments(raw, 4)
    segs, cond = wv.build_seg_tensors(raw, seg_starts, 4)
    tr_idx, va_idx = wv.split_segments(len(seg_starts), args.split_seed, args.val_frac)
    print(f"segments total={len(seg_starts)} train={len(tr_idx)} val={len(va_idx)}")

    # ---------------- 載入 base ckpt（encoder/decoder/vq/obs_stats 全部原封load進來）
    model, cfg = wv.load_dict_v1(args.base_ckpt)
    assert cfg["seg_len"] == 4 and cfg["k"] == 32, f"預期 L4K32，實際 {cfg}"
    model = model.to(device)

    # ---------------- REPRO CHECK：obs_std/obs_mean 重算 vs ckpt buffer 一致
    # （抓「資料被誰動過」或「split 對不上」這種會讓整個噪音校準基礎垮掉的沉默失敗）
    obs_mean_recalc = cond[tr_idx].mean(0)
    obs_std_recalc = cond[tr_idx].std(0)
    ckpt_obs_mean = model.obs_mean.detach().cpu().numpy()
    ckpt_obs_std = model.obs_std.detach().cpu().numpy()
    mean_ok = np.allclose(obs_mean_recalc, ckpt_obs_mean, rtol=1e-4, atol=1e-5)
    std_ok = np.allclose(obs_std_recalc, ckpt_obs_std, rtol=1e-4, atol=1e-5)
    print(f"REPRO CHECK obs stats（重算 vs ckpt buffer）：mean {'PASS' if mean_ok else 'FAIL'}  "
          f"std {'PASS' if std_ok else 'FAIL'}  "
          f"(max|Δmean|={np.max(np.abs(obs_mean_recalc-ckpt_obs_mean)):.2e}  "
          f"max|Δstd|={np.max(np.abs(obs_std_recalc-ckpt_obs_std)):.2e})")
    if not (mean_ok and std_ok):
        print("⛔ obs 統計量重算對不上 ckpt buffer —— 資料或 split 跟原訓練不一致，"
              "噪音校準的基礎（obs_std）不可信，停手回報，不往下訓練。")
        sys.exit(1)

    # ---------------- freeze encoder + codebook（見檔頭說明：手刻 forward 繞過
    # VQEMA.forward，這條路徑上不存在能動 EMA buffer 的程式碼；encoder 參數額外
    # 也 requires_grad_(False) 雙重保險，即使未來有人不小心把它塞進 optimizer）
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    n_dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"decoder trainable params = {n_dec_params}（encoder frozen, vq codebook frozen）")
    opt = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    # ---------------- 噪音 std 向量（見檔頭校準說明）
    xy_std_ref = float((ckpt_obs_std[0] + ckpt_obs_std[1]) / 2.0)
    rho = args.sigma_xy / xy_std_ref
    noise_std_np = np.zeros(29, dtype=np.float32)
    noise_std_np[0] = args.sigma_xy
    noise_std_np[1] = args.sigma_xy
    noise_std_np[2:] = rho * ckpt_obs_std[2:]
    noise_std_t = torch.from_numpy(noise_std_np)
    print(f"xy_std_ref(train, mean of obs_std[0:2])={xy_std_ref:.4f}  rho=sigma_xy/xy_std_ref={rho:.5f}")
    print(f"noise_std[xy]={noise_std_np[:2].tolist()}")
    print(f"noise_std[other 27 dims] p25/50/75="
          f"{np.percentile(noise_std_np[2:], [25, 50, 75]).tolist()}")

    tr_x = torch.from_numpy(segs[tr_idx])
    tr_o = torch.from_numpy(cond[tr_idx])
    va_x = torch.from_numpy(segs[va_idx])
    va_o = torch.from_numpy(cond[va_idx])

    brng = np.random.default_rng(args.seed + 1)   # 批次抽樣（跟 P0 同公式風格）
    nrng = np.random.default_rng(args.seed + 2)   # 訓練噪音（獨立 stream）
    nrng_eval = np.random.default_rng(args.seed + 3)  # eval 噪音（獨立 stream，跟訓練噪音不共用抽樣序）
    n_train = len(tr_idx)

    curve = []  # [step, recon_loss(batch, noisy), val_mse_noisy, val_mse_clean]
    t0 = time.time()
    for step in range(args.steps):
        b = brng.integers(0, n_train, size=args.batch)
        x, o = tr_x[b], tr_o[b]
        eps = nrng.normal(0.0, 1.0, size=o.shape).astype(np.float32) * noise_std_np[None, :]
        o_noisy = o.numpy() + eps
        x_d = x.to(device)
        o_d = torch.from_numpy(o_noisy).to(device)

        with torch.no_grad():
            z_e = model.encoder(x_d)
            idx = model.vq.quantize_idx(z_e)
            z_q = model.vq.embed[idx]
        recon = model.decoder(torch.cat([z_q, model.norm_obs(o_d)], dim=1))
        loss = F.mse_loss(recon, x_d)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            vm_noisy, vm_clean = evaluate_recon(model, va_x, va_o, noise_std_t, device,
                                                args.eval_repeats, nrng_eval)
            curve.append([step, float(loss.item()), vm_noisy, vm_clean])
            if step % (args.eval_every * 5) == 0 or step == args.steps - 1:
                print(f"  step {step:6d} train_recon(noisy,batch)={loss.item():.5f} "
                      f"val_mse_noisy={vm_noisy:.5f} val_mse_clean={vm_clean:.5f} "
                      f"({time.time()-t0:.0f}s)")
    train_seconds = time.time() - t0

    # ---------------- 走平判定（跟 P0 同一種算法：最後 20% vs 前一段 20%）
    cv = np.array([c[2] for c in curve], dtype=float)  # val_mse_noisy 曲線
    n_tail = max(2, len(cv) // 5)
    early_tail = float(cv[-2 * n_tail:-n_tail].mean()) if len(cv) >= 2 * n_tail else float("nan")
    late_tail = float(cv[-n_tail:].mean())
    rel_improve = (early_tail - late_tail) / max(early_tail, 1e-12) if np.isfinite(early_tail) else float("nan")
    is_flat = bool(np.isfinite(rel_improve) and abs(rel_improve) < 0.02)  # <2% 波動視為走平
    print(f"loss-flatness（val_mse_noisy）：late-{n_tail}pt mean={late_tail:.5f} vs "
          f"prev-{n_tail}pt mean={early_tail:.5f}  rel_improve={rel_improve*100:.2f}%  "
          f"flat(<2%)={is_flat}")

    # ---------------- 存檔
    res_dir = args.out_dir
    os.makedirs(res_dir, exist_ok=True)
    png = os.path.join(res_dir, f"loss_{tag}.png")
    wv.draw_loss_curve(
        [dict(label="train recon (noisy,batch)", xs=[c[0] for c in curve], ys=[c[1] for c in curve]),
         dict(label="val_mse noisy (repeat-avg)", xs=[c[0] for c in curve], ys=[c[2] for c in curve]),
         dict(label="val_mse clean (sigma=0)", xs=[c[0] for c in curve], ys=[c[3] for c in curve])],
        png, title=f"robust-decoder fine-tune {tag} (sigma_xy={args.sigma_xy}, seed={args.seed})")

    ckpt_out = os.path.join(res_dir, f"ckpt_{tag}.pt")
    new_cfg = dict(cfg)  # 保留原架構 cfg（wv.load_dict_v1 要吃這些 key），新增 finetune 子欄位
    new_cfg["finetune"] = dict(
        base_ckpt=os.path.abspath(args.base_ckpt), sigma_xy=args.sigma_xy,
        noise_std=noise_std_np.tolist(), xy_std_ref=xy_std_ref, rho=rho,
        steps=args.steps, seed=args.seed, lr=args.lr, batch=args.batch,
        split_seed=args.split_seed, val_frac=args.val_frac, eval_every=args.eval_every,
        eval_repeats=args.eval_repeats, tag=tag, device=device,
        calibration_ref=CALIB_REF, calibration_ref_source=DRIFT_SUMMARY_JSON,
        encoder_codebook_frozen=True, reconstruction_loss="F.mse_loss(recon, x) only "
        "(commit loss dropped: encoder/vq frozen so it is a constant w.r.t. decoder params)")
    torch.save(dict(state_dict=model.state_dict(), config=new_cfg), ckpt_out)

    out = dict(
        config=dict(vars(args)),
        noise=dict(xy_std_ref=xy_std_ref, rho=rho, noise_std=noise_std_np.tolist(),
                   calibration_ref=CALIB_REF, calibration_ref_source=DRIFT_SUMMARY_JSON),
        repro_check_obs_stats=dict(mean_ok=bool(mean_ok), std_ok=bool(std_ok),
                                   max_abs_diff_mean=float(np.max(np.abs(obs_mean_recalc - ckpt_obs_mean))),
                                   max_abs_diff_std=float(np.max(np.abs(obs_std_recalc - ckpt_obs_std)))),
        metrics=dict(final_val_mse_noisy=curve[-1][2], final_val_mse_clean=curve[-1][3],
                     final_train_recon_noisy_batch=curve[-1][1]),
        flatness=dict(late_tail=late_tail, prev_tail=early_tail, rel_improve=rel_improve,
                      n_tail_points=int(n_tail), is_flat_lt2pct=is_flat),
        loss_curve=curve,
        data=dict(n_segments=int(len(seg_starts)), n_train=int(len(tr_idx)), n_val=int(len(va_idx))),
        timing=dict(train_seconds=train_seconds, wall_seconds=time.time() - t_all),
        artifacts=dict(loss_png=png, ckpt=ckpt_out),
        n_decoder_trainable_params=int(n_dec_params),
    )
    js = os.path.join(res_dir, f"train_{tag}.json")
    wv.save_json(out, js)
    print(f"saved: {js}\nsaved: {png}\nsaved: {ckpt_out}")
    print(f"=== done {tag} wall={time.time()-t_all:.1f}s ===")


if __name__ == "__main__":
    main()
