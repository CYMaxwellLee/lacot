#!/usr/bin/env python
"""工單2（主人核准施工計畫，2026-09-13）：ŝ 外掛頭。

在證明什麼：λ0（訓練時沒有 state 重建頭）練好的行為字典，其碼裡已藏有「該段
預期到達狀態 ŝ」的資訊——外掛一個小 decoder（讀碼→ŝ，梯度不回字典，字典凍住）
能達到堪用的 ŝ 預測。

判準（相對比較，沒有絕對線）：外掛頭的 ŝ 預測誤差（同段集）對比
  (a) λ1 版字典自帶頭的 ŝ 誤差（它是被逼著記的、當參照上界）——不重跑，直接讀
      experiments/bcodec_v0/results/G{16,32}K{16,32}_lam1p0.json 的
      metrics.recon_s_val（一手數字，主人施工計畫核可時已核對過）。
  (b) 爛錨 control（打亂碼、保留配對 obs_t_norm 的誤差）：只檢查碼與狀態的
      配對依賴，不能單獨證明碼提供超出狀態的資訊。
  (c) obs-only probe（同架構、hidden、steps、batch、lr 與切分，獨立訓練）：
      只有 err_real < err_obs_only × margin 才判碼有增量資訊；margin 可配置，
      預設 0.9，要求 full probe 的誤差至少比 obs-only 低 10%。

字典凍住：gk_model 全部 parameters() 設 requires_grad_(False)，優化器只吃 probe
頭自己的參數。

⛔ 不改動任何既有檔案（bcodec_common.py / gk_common.py 一行未改，本檔只 import）。
⛔ CPU-only。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_HERE)  # .../lacot/experiments
_BCODEC_DIR = os.path.join(_EXP_DIR, "bcodec_v0")
_GK_DIR = os.path.join(_EXP_DIR, "gk_scan")
for _p in (_BCODEC_DIR, _GK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bcodec_common as bc  # noqa: E402 -- 跨樹唯讀 import，不改動
import gk_common as gkc  # noqa: E402 -- 跨樹唯讀 import，不改動（拿 MLP 類別）

DATA_PATH_DEFAULT = os.path.join(
    os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data"),
    f"{bc.DATASET_NAME}.npz")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def get_code_embed(gk_model, idx):
    """跟 model.decode_codes 內部組 z_q 的邏輯一模一樣（逐行照抄
    BehaviorCodecFactorizedVQVAE.decode_codes 的前兩行），單獨拉出來給 probe 用。
    idx: (N,G) long -> z_q: (N, G*D) float，來自凍結字典自己的 codebook embedding，
    不含梯度（torch.no_grad 包住，字典凍住的具體實作）。"""
    with torch.no_grad():
        z_q_chunks = [gk_model.vq.groups[g].embed[idx[:, g]] for g in range(gk_model.G)]
        z_q = torch.cat(z_q_chunks, dim=1)
    return z_q.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="bcodec_v0 λ0 ckpt 路徑（G16K16_lam0p0.pt 或 G32K32_lam0p0.pt）")
    ap.add_argument("--lam1-ref-json", type=str, required=True,
                    help="對應的 λ1 結果 json（讀 metrics.recon_s_val 當參照(a)，不重跑）")
    ap.add_argument("--data-path", type=str, default=DATA_PATH_DEFAULT)
    ap.add_argument("--split-seed", type=int, default=42, help="⛔ 必須跟 bcodec_v0 訓練同一顆，保證同段集")
    ap.add_argument("--val-frac", type=float, default=0.1, help="⛔ 必須跟 bcodec_v0 訓練同一顆，保證同段集")
    ap.add_argument("--hidden", type=int, default=256, help="[拍] 比字典自己的 512 小，任務更簡單")
    ap.add_argument("--steps", type=int, default=20000, help="[拍] 分鐘級，跟 bcodec_v0 訓練同數量級的一半")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bad-anchor-seed", type=int, default=20260913)
    ap.add_argument("--suspect-ratio-threshold", type=float, default=2.0,
                    help="[拍] err_bad/err_real 低於此值＝量測疑似壞了，這是主人裁示第6點要求的自覺檢查")
    ap.add_argument("--code-info-margin", type=float, default=0.9,
                    help="err_real < err_obs_only × margin 才判碼有增量資訊（預設 0.9）")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=os.path.join(_HERE, "results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    print(f"=== shat_probe  ckpt={args.ckpt}  tag={args.tag} ===")
    gk_model, cfg, obs_mu_t, obs_sd_t = bc.load_bcodec_gk_ckpt(args.ckpt)
    for p in gk_model.parameters():
        p.requires_grad_(False)
    gk_model.eval()
    G, K, D = cfg["G"], cfg["K"], cfg["D"]
    obs_dim, act_dim, seg_len = cfg["obs_dim"], cfg["act_dim"], cfg["seg_len"]
    print(f"dict: G={G} K={K} D={D} lam_s={cfg['lam_s']}（本工單只用 λ0 顆，字典凍住）"
          f"  obs_dim={obs_dim} act_dim={act_dim} seg_len={seg_len}")

    # ------------------------------------------------------------
    # 資料：跟 bcodec_v0 訓練同一個函式、同一顆 split_seed/val_frac —— 這就是
    # 「同段集」的保證來源（呼叫同一段程式碼，不是另外對表）。
    # ------------------------------------------------------------
    data = bc.load_bcodec_segments(args.data_path, seg_len=seg_len,
                                    split_seed=args.split_seed, val_frac=args.val_frac)
    a_seg, s_seg, s_next = data["a_seg"], data["s_seg"], data["s_next"]
    obs_start = data["obs_start"]
    train_idx, val_idx = data["train_idx"], data["val_idx"]
    print(f"segments: n_total={len(a_seg)}  n_train={len(train_idx)}  n_val={len(val_idx)}  "
          f"split_seed={args.split_seed}  val_frac={args.val_frac}")

    s_seg_norm_all = bc._norm_seg(s_seg, obs_mu_t.numpy(), obs_sd_t.numpy(), seg_len, obs_dim)
    s_next_norm_all = bc._norm_seg(s_next, obs_mu_t.numpy(), obs_sd_t.numpy(), seg_len, obs_dim)
    obs_start_norm_all = ((obs_start - obs_mu_t.numpy()) / obs_sd_t.numpy()).astype(np.float32)

    def T(x):
        return torch.from_numpy(x)

    # 用凍結字典自己的 encoder+vq 把每段的碼算出來（一次性、no_grad，字典凍住）
    with torch.no_grad():
        z_e_all = gk_model.encode(T(s_seg_norm_all), T(a_seg))
        _, idx_all, _ = gk_model.vq(z_e_all, training=False)  # (N, G) long

    train_z = get_code_embed(gk_model, idx_all[train_idx])       # (n_train, G*D)
    val_z = get_code_embed(gk_model, idx_all[val_idx])           # (n_val, G*D)
    train_obs = T(obs_start_norm_all[train_idx])                 # (n_train, obs_dim)
    val_obs = T(obs_start_norm_all[val_idx])
    train_target = T(s_next_norm_all[train_idx])                 # (n_train, seg_len*obs_dim)
    val_target = T(s_next_norm_all[val_idx])
    n_train = len(train_idx)

    # ------------------------------------------------------------
    # 外掛 probe 頭：MLP(碼嵌入 concat obs_t_norm -> hidden -> hidden -> ŝ_norm)。
    # 跟字典自己的 decode_codes 用同一種輸入配方（碼+obs_t_norm）——這樣「碼裡有
    # 沒有多的資訊」才是乾淨對照：如果只給碼不給 obs_t_norm，跟字典自己的頭不是
    # 同條件比較。
    # ------------------------------------------------------------
    probe = gkc.MLP(G * D + obs_dim, seg_len * obs_dim, hidden=args.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    batch_rng = np.random.default_rng(args.seed + 1)

    loss_curve = []
    for i in range(args.steps):
        bidx = batch_rng.integers(0, n_train, size=args.batch)
        pred = probe(torch.cat([train_z[bidx], train_obs[bidx]], dim=1))
        loss = F.mse_loss(pred, train_target[bidx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0 or i == args.steps - 1:
            loss_curve.append([i, float(loss.item())])
            print(f"  step={i:6d}  train_mse={loss.item():.6f}")

    # obs-only 對照：同架構與訓練預算，只拿掉碼；用同一批次序列獨立訓練。
    obs_only_probe = gkc.MLP(obs_dim, seg_len * obs_dim, hidden=args.hidden)
    obs_only_opt = torch.optim.Adam(obs_only_probe.parameters(), lr=args.lr)
    obs_only_batch_rng = np.random.default_rng(args.seed + 1)
    for i in range(args.steps):
        bidx = obs_only_batch_rng.integers(0, n_train, size=args.batch)
        pred = obs_only_probe(train_obs[bidx])
        loss = F.mse_loss(pred, train_target[bidx])
        obs_only_opt.zero_grad()
        loss.backward()
        obs_only_opt.step()

    train_seconds = time.time() - t0

    # ------------------------------------------------------------
    # 評測 (real)：val 集，碼-obs 正確配對
    # ------------------------------------------------------------
    probe.eval()
    obs_only_probe.eval()
    with torch.no_grad():
        pred_val_real = probe(torch.cat([val_z, val_obs], dim=1))
        err_real = float(F.mse_loss(pred_val_real, val_target).item())
        err_obs_only = float(F.mse_loss(obs_only_probe(val_obs), val_target).item())

        # ------------------------------------------------------------
        # 爛錨 control (b)：碼打亂（跟別的 val 樣本配對），obs_t_norm 保留自己
        # 真正的值——此臂只檢查 full probe 是否依賴碼與 obs_t_norm 的配對；
        # 碼是否提供超出狀態的資訊由 obs-only 對照判定。
        # ------------------------------------------------------------
        bad_rng = np.random.default_rng(args.bad_anchor_seed)
        perm = bad_rng.permutation(len(val_idx))
        pred_val_bad = probe(torch.cat([val_z[perm], val_obs], dim=1))
        err_bad = float(F.mse_loss(pred_val_bad, val_target).item())

    # ------------------------------------------------------------
    # 參照 (a)：λ1 自帶頭的 ŝ 誤差，直接讀 json，不重跑
    # ------------------------------------------------------------
    with open(args.lam1_ref_json) as f:
        lam1_doc = json.load(f)
    lam1_metrics = lam1_doc.get("metrics", lam1_doc)
    ref_a_recon_s_val = lam1_metrics["recon_s_val"]
    assert lam1_metrics["G"] == G and lam1_metrics["K"] == K, (
        f"⛔ λ1 參照 json 的 G/K={lam1_metrics['G']}/{lam1_metrics['K']} 跟本次探測的 λ0 ckpt "
        f"G/K={G}/{K} 對不上，停手回報，不准硬比")

    ratio_bad_to_real = err_bad / max(err_real, 1e-12)
    measurement_suspect = ratio_bad_to_real < args.suspect_ratio_threshold
    ratio_to_lambda1 = err_real / max(ref_a_recon_s_val, 1e-12)
    ratio_full_vs_obsonly = err_obs_only / max(err_real, 1e-12)
    code_adds_information = err_real < err_obs_only * args.code_info_margin

    if measurement_suspect:
        verdict = "shuffle 未顯示足夠配對依賴；碼的增量資訊另見 obs-only 判準"
        print(f"\n⛔⛔⛔ err_bad/err_real = {ratio_bad_to_real:.3f} < 門檻 {args.suspect_ratio_threshold} "
              f"—— 打亂碼後誤差沒有遠高於正確配對；shuffle 臂未證實配對依賴。")
    elif not code_adds_information:
        verdict = "碼未顯示超出狀態的增量資訊（full 未顯著優於 obs-only）"
    elif ratio_to_lambda1 <= 1.5:
        verdict = "假說成立（外掛頭接近λ1自帶頭，且顯著優於 obs-only）"
    else:
        verdict = "碼有增量資訊，但離λ1自帶頭有距離，見數字，人工判讀"

    print(f"\n=== 結果 ===")
    print(f"(real) val ŝ MSE (外掛頭，正確配對)     = {err_real:.6f}")
    print(f"(a)    λ1 自帶頭 ŝ MSE（參照上界，讀檔） = {ref_a_recon_s_val:.6f}  "
          f"(ratio real/lambda1 = {ratio_to_lambda1:.3f}, 越接近1越好)")
    print(f"(b)    爛錨 control ŝ MSE（碼打亂）       = {err_bad:.6f}  "
          f"(ratio bad/real = {ratio_bad_to_real:.3f}, [拍]門檻={args.suspect_ratio_threshold}；只檢查碼×狀態配對依賴)")
    print(f"(c)    obs-only ŝ MSE（只給狀態）         = {err_obs_only:.6f}  "
          f"(ratio obs-only/full = {ratio_full_vs_obsonly:.3f}；碼有增量資訊={code_adds_information}，"
          f"判準 err_real < err_obs_only × {args.code_info_margin})")
    print(f"判定：{verdict}")

    summary = dict(
        tag=args.tag, ckpt=args.ckpt, lam1_ref_json=args.lam1_ref_json,
        dict_cfg=dict(G=G, K=K, D=D, obs_dim=obs_dim, act_dim=act_dim, seg_len=seg_len, lam_s=cfg["lam_s"]),
        data=dict(data_path=args.data_path, split_seed=args.split_seed, val_frac=args.val_frac,
                  n_total=len(a_seg), n_train=len(train_idx), n_val=len(val_idx)),
        probe=dict(hidden=args.hidden, steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed,
                  input_dim=G * D + obs_dim, output_dim=seg_len * obs_dim,
                  input_recipe="concat(code_embed_from_frozen_dict, obs_t_norm)"),
        bad_anchor=dict(seed=args.bad_anchor_seed, method="permute code embedding pairing only, keep obs_t_norm true"),
        results=dict(err_real_val=err_real, err_bad_val=err_bad, err_obs_only=err_obs_only,
                    ref_a_lambda1_recon_s_val=ref_a_recon_s_val,
                    ratio_bad_to_real=ratio_bad_to_real, ratio_real_to_lambda1=ratio_to_lambda1,
                    ratio_full_vs_obsonly=ratio_full_vs_obsonly,
                    code_info_margin=args.code_info_margin, code_adds_information=code_adds_information,
                    suspect_ratio_threshold=args.suspect_ratio_threshold,
                    measurement_suspect=measurement_suspect, verdict=verdict),
        loss_curve=loss_curve, train_seconds=train_seconds,
        wall_seconds=time.time() - t0,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    js_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    save_json(summary, js_path)
    print(f"saved: {js_path}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
