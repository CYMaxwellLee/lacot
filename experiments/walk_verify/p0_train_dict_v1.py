#!/usr/bin/env python
"""P0：字典 v1 —— 還原機條件化重訓（decoder 吃「字 embedding ＋ 段起點完整 obs(29)」）。

對照 M0 無條件版（experiments/m0_gait_dict，L4K32 val MSE .21226）：
  * encoder / VQ-EMA / dead-code restart / 超參數 / 資料切法 / split_seed 全部照 M0
  * 唯一差別 = decoder 的輸入多了正規化的段起點 obs
⇒ val MSE 的差就是「條件化」買到的東西，其他變因固定。

⛔ CPU-only（不碰 GPU、不經 slurm）。

用法：
  python p0_train_dict_v1.py --seg-len 4 --k 32 --steps 20000
  python p0_train_dict_v1.py --seg-len 4 --k 32 --steps 50000 --tag L4K32_50k
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wv_common as wv  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
HERE = os.path.dirname(os.path.abspath(__file__))
M0_UNCOND_VAL_MSE = 0.21226  # experiments/m0_gait_dict/results/L4_K32.json（一手來源）


def evaluate(model, segs_t, cond_t, chunk=65536):
    """分塊算重建 MSE 與 code 指派（eval mode，不更新 EMA）。"""
    tot_se, tot_n, codes = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(segs_t), chunk):
            x, o = segs_t[i:i + chunk], cond_t[i:i + chunk]
            recon, idx, _ = model(x, o, training=False)
            tot_se += float(((recon - x) ** 2).sum().item())
            tot_n += x.numel()
            codes.append(idx.numpy())
    return tot_se / max(tot_n, 1), np.concatenate(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-len", type=int, default=4)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=HERE)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 10000 + args.seg_len * 100 + args.k  # 與 M0 同公式，可重現
    tag = args.tag or f"L{args.seg_len}K{args.k}"
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(args.seed)

    print(f"=== P0 dict-v1 (conditional decoder) {tag} seed={args.seed} steps={args.steps} ===")
    t_all = time.time()

    raw = wv.load_npz(args.data_dir, DATASET, "train")
    seg_starts = wv.cut_segments(raw, args.seg_len)
    segs, cond = wv.build_seg_tensors(raw, seg_starts, args.seg_len)
    tr_idx, va_idx = wv.split_segments(len(seg_starts), args.split_seed, args.val_frac)
    print(f"segments total={len(seg_starts)} train={len(tr_idx)} val={len(va_idx)} "
          f"seg_dim={segs.shape[1]} obs_dim={cond.shape[1]}")

    # 官方 held-out val 檔（完全沒看過的 episode）—— 比段級切分更嚴的一把尺
    raw_ho = wv.load_npz(args.data_dir, DATASET, "val")
    ho_starts = wv.cut_segments(raw_ho, args.seg_len)
    ho_segs, ho_cond = wv.build_seg_tensors(raw_ho, ho_starts, args.seg_len)
    print(f"official held-out val segments={len(ho_starts)} ({raw_ho['_path']})")

    obs_mean = cond[tr_idx].mean(0)
    obs_std = cond[tr_idx].std(0)
    n_low_var = int((obs_std < 1e-6).sum())
    print(f"obs normalizer from TRAIN split: dims with std<1e-6 = {n_low_var}")

    model = wv.CondVQVAE(segs.shape[1], args.latent_dim, args.k, obs_dim=cond.shape[1],
                         hidden=args.hidden, decay=args.decay, dead_steps=args.dead_steps)
    model.set_obs_stats(obs_mean, obs_std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_x = torch.from_numpy(segs[tr_idx])
    tr_o = torch.from_numpy(cond[tr_idx])
    va_x = torch.from_numpy(segs[va_idx])
    va_o = torch.from_numpy(cond[va_idx])
    ho_x = torch.from_numpy(ho_segs)
    ho_o = torch.from_numpy(ho_cond)
    n_train = len(tr_idx)

    brng = np.random.default_rng(args.seed + 1)
    curve = []  # [step, recon_loss(batch), commit_loss(batch), val_mse]
    t0 = time.time()
    for step in range(args.steps):
        b = brng.integers(0, n_train, size=args.batch)
        x, o = tr_x[b], tr_o[b]
        recon, _idx, commit = model(x, o, training=True)
        rl = F.mse_loss(recon, x)
        loss = rl + args.beta * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            vm, _ = evaluate(model, va_x, va_o)
            model.train()
            curve.append([step, float(rl.item()), float(commit.item()), float(vm)])
            if step % (args.eval_every * 5) == 0 or step == args.steps - 1:
                print(f"  step {step:6d} recon={rl.item():.5f} commit={commit.item():.5f} "
                      f"val_mse={vm:.5f} ({time.time()-t0:.0f}s)")
    train_seconds = time.time() - t0

    model.eval()
    train_mse, train_codes = evaluate(model, tr_x, tr_o)
    val_mse, val_codes = evaluate(model, va_x, va_o)
    ho_mse, ho_codes = evaluate(model, ho_x, ho_o)

    # baseline：猜訓練切分的平均動作段（同 M0 定義）
    base_vec = segs[tr_idx].mean(0, keepdims=True)
    base_tr = float(np.mean((segs[tr_idx] - base_vec) ** 2))
    base_va = float(np.mean((segs[va_idx] - base_vec) ** 2))
    base_ho = float(np.mean((ho_segs - base_vec) ** 2))

    all_codes = np.concatenate([train_codes, val_codes])
    counts = np.bincount(all_codes, minlength=args.k).astype(np.int64)
    frac = counts / max(counts.sum(), 1)
    nz = frac[frac > 0]
    perpl = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0
    order = np.argsort(-frac)

    # 走平判定：最後 20% 的 val_mse 相對前一段的改善幅度
    cv = np.array([c[3] for c in curve], dtype=float)
    n_tail = max(2, len(cv) // 5)
    early_tail = float(cv[-2 * n_tail:-n_tail].mean()) if len(cv) >= 2 * n_tail else float("nan")
    late_tail = float(cv[-n_tail:].mean())
    rel_improve = (early_tail - late_tail) / max(early_tail, 1e-12) if np.isfinite(early_tail) else float("nan")

    print("--- RESULT ---")
    print(f"train_mse={train_mse:.5f} val_mse={val_mse:.5f} heldout_val_mse={ho_mse:.5f}")
    print(f"baseline(mean-seg): train={base_tr:.5f} val={base_va:.5f} heldout={base_ho:.5f}")
    print(f"M0 uncond L4K32 val_mse={M0_UNCOND_VAL_MSE:.5f} -> v1 val_mse={val_mse:.5f} "
          f"({(1-val_mse/M0_UNCOND_VAL_MSE)*100:+.1f}% vs M0)")
    print(f"perplexity={perpl:.2f}/{args.k} active_codes={int((counts>0).sum())}/{args.k} "
          f"top1={frac[order[0]]*100:.1f}% top3={frac[order[:3]].sum()*100:.1f}% "
          f"restarts={int(model.vq.restarts_total.item())}")
    print(f"loss-flatness: val_mse mean last-{n_tail}pts={late_tail:.5f} vs prev-{n_tail}pts="
          f"{early_tail:.5f} rel_improve={rel_improve*100:.2f}%")
    print(f"train_seconds={train_seconds:.1f}")

    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    png = os.path.join(res_dir, f"p0_loss_{tag}.png")
    wv.draw_loss_curve(
        [dict(label="train recon (batch)", xs=[c[0] for c in curve], ys=[c[1] for c in curve]),
         dict(label="val recon MSE", xs=[c[0] for c in curve], ys=[c[3] for c in curve]),
         dict(label="commit (batch)", xs=[c[0] for c in curve], ys=[c[2] for c in curve]),
         dict(label=f"M0 uncond val {M0_UNCOND_VAL_MSE:.3f}",
              xs=[curve[0][0], curve[-1][0]], ys=[M0_UNCOND_VAL_MSE] * 2)],
        png, title=f"P0 dict-v1 conditional decoder {tag} (seed={args.seed})")
    hist = os.path.join(res_dir, f"p0_usage_{tag}.png")
    wv.draw_usage_histogram(frac.tolist(), hist, f"code usage - dict v1 {tag}")

    ckpt = os.path.join(res_dir, f"p0_dict_v1_{tag}.pt")
    torch.save(dict(state_dict=model.state_dict(),
                    config=dict(seg_len=args.seg_len, k=args.k, latent_dim=args.latent_dim,
                                hidden=args.hidden, decay=args.decay, dead_steps=args.dead_steps,
                                seg_dim=int(segs.shape[1]), obs_dim=int(cond.shape[1]),
                                beta=args.beta, lr=args.lr, batch=args.batch, steps=args.steps,
                                seed=args.seed, split_seed=args.split_seed, val_frac=args.val_frac,
                                dataset=DATASET, conditional_decoder=True)), ckpt)

    out = dict(
        config=dict(vars(args)),
        metrics=dict(recon_mse_train=train_mse, recon_mse_val=val_mse,
                     recon_mse_heldout_official_val=ho_mse,
                     baseline_mse_train=base_tr, baseline_mse_val=base_va,
                     baseline_mse_heldout=base_ho,
                     m0_uncond_val_mse=M0_UNCOND_VAL_MSE,
                     rel_vs_m0_val=float(1 - val_mse / M0_UNCOND_VAL_MSE),
                     perplexity=perpl, perplexity_max=args.k,
                     n_active_codes=int((counts > 0).sum()),
                     usage_counts=counts.tolist(), usage_frac=frac.tolist(),
                     top1_usage_frac=float(frac[order[0]]),
                     top3_usage_frac=float(frac[order[:3]].sum()),
                     restarts_total=int(model.vq.restarts_total.item()),
                     obs_norm_dims_low_var=n_low_var),
        flatness=dict(late_tail_val_mse=late_tail, prev_tail_val_mse=early_tail,
                      rel_improve=rel_improve, n_tail_points=int(n_tail)),
        loss_curve=curve,
        data=dict(n_segments=int(len(seg_starts)), n_train=int(len(tr_idx)),
                  n_val=int(len(va_idx)), n_heldout=int(len(ho_starts))),
        timing=dict(train_seconds=train_seconds, wall_seconds=time.time() - t_all),
        artifacts=dict(loss_png=png, usage_png=hist, ckpt=ckpt),
    )
    js = os.path.join(res_dir, f"p0_{tag}.json")
    wv.save_json(out, js)
    print(f"saved: {js}\nsaved: {png}\nsaved: {hist}\nsaved: {ckpt}")
    print(f"=== done {tag} wall={time.time()-t_all:.1f}s ===")


if __name__ == "__main__":
    main()
