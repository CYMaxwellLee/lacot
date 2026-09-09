#!/usr/bin/env python
"""experiments/gen_sft/train_gen_sft_dyn.py —— 主人裁示①②：動態隨機起點（每步即時切）
+ 可選漂移擾動 訓練腳本。GPU only（走 sbatch）。

跟既有 train_gen_sft.py 的關係：訓練迴圈骨架（optimizer/schedule/early-stop/smoke
判讀/存檔格式）逐項照抄同一份配方（工單「走平早停（沿用配方）」），唯一差異是
batch 的來源——train_gen_sft.py 的 iterate_batches() 對『corpus-build 時就切死』的
固定樣本清單做 index；本檔的 iterate_batches_dyn() 對『軌跡級原料』
（corpus_dyn_v1.pt，見 dyn_corpus_common.py）每次取樣即時構造：50% 原起點、50%
均勻隨機中途切點，切點/擾動走各自獨立的固定 rng 流（跟 --seed 分開，--seed 只管權重
初始化與 batch-order，同 collate_batch/evaluate 直接 import 沿用，不重寫）。

held-out 評測沿用 corpus_aug_v1.pt 的 val split（跟 aug 六顆訓練共用同一組 held-out
樣本，方便三個 recipe 的 held-out 數字直接互比）——這個 val 集本身跟本次訓練的
train 集是同一批 episode 級切分（split_seed=42 一路沿用到底），不會有 held-out
episode 洩漏進本次的動態訓練集。

⛔ 不改動任何既有檔案（train_gen_sft.py / model.py / gen_sft_common.py /
   aug_corpus_common.py 皆原封不動，只 import）。
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ⛔ GPU-visibility-guard：逐字沿用 train_gen_sft.py 檔頭的除錯結論（job 26819 系列
# FAILED 的根因）——torch.cuda.is_available() 必須在 import gen_sft_common（連帶
# import run_teacher_relay，它會把 HIP_VISIBLE_DEVICES setdefault 成空字串）之前
# 呼叫過一次，把偵測結果快取住，否則 ROCm 會把「HIP 可見裝置=空字串」誤讀成「沒卡」。
_cuda_ok_before = torch.cuda.is_available()
print(f"[gpu-visibility guard] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
      f"is_available(before importing gen_sft_common)={_cuda_ok_before}", flush=True)

import gen_sft_common as gc  # noqa: E402
from model import GenSFT, BOS_ID, EOS_ID  # noqa: E402
import dyn_corpus_common as dc  # noqa: E402
from train_gen_sft import collate_batch, evaluate  # noqa: E402  -- 原封不動 import 重用

print(f"[gpu-visibility guard] HIP_VISIBLE_DEVICES(after importing gen_sft_common)="
      f"{os.environ.get('HIP_VISIBLE_DEVICES')!r}  is_available(cached)={torch.cuda.is_available()}",
      flush=True)


def iterate_batches_dyn(episodes, batch_size, order_rng, cut_rng, use_full_prob,
                        perturb_rng, perturb_cfg, stats):
    """跟 train_gen_sft.iterate_batches 同角色：無限 batch generator，每滿一輪 episode
    重新洗牌（order_rng，跟 --seed 綁定，這是唯一隨 3 個訓練 seed 變動的隨機源）。
    每次抽到一個 episode，用 dc.draw_dynamic_sample()（cut_rng/perturb_rng 為固定、
    跨 3 個訓練 seed 共用的獨立流）即時決定原起點或中途切點、要不要擾動。
    """
    n = len(episodes)
    order = order_rng.permutation(n)
    pos = 0
    while True:
        if pos + batch_size > n:
            order = order_rng.permutation(n)
            pos = 0
        idx = order[pos:pos + batch_size]
        pos += batch_size
        batch = [dc.draw_dynamic_sample(episodes[i], cut_rng, use_full_prob=use_full_prob,
                                        perturb_rng=perturb_rng, perturb_cfg=perturb_cfg,
                                        stats=stats) for i in idx]
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_dyn_v1.pt"))
    ap.add_argument("--heldout-corpus", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_aug_v1.pt"),
                    help="held-out eval 樣本來源，沿用 aug 六顆訓練的 val split（同一批 episode，方便互比）")
    ap.add_argument("--perturb", action="store_true", help="開漂移型擾動（條件 obs），關=無擾動組")
    ap.add_argument("--use-full-prob", type=float, default=dc.USE_FULL_PROB)
    ap.add_argument("--dyn-cut-seed", type=int, default=dc.DYN_CUT_SEED)
    ap.add_argument("--perturb-seed", type=int, default=dc.PERTURB_SEED)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--patience-evals", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234, help="權重初始化 + batch-order（跟 dyn-cut-seed/perturb-seed 分開）")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="gensft_dyn")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    args = ap.parse_args()

    assert args.device == "cpu" or torch.cuda.is_available(), (
        "⛔ 要求 --device cuda 但 torch.cuda.is_available()=False —— 停手回報，不准偷偷退回 CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    order_rng = np.random.default_rng(args.seed)
    cut_rng = np.random.default_rng(args.dyn_cut_seed)
    perturb_rng = np.random.default_rng(args.perturb_seed) if args.perturb else None

    t0 = time.time()
    blob = torch.load(args.corpus, weights_only=False)
    train_episodes, meta = blob["train_episodes"], blob["meta"]
    print(f"corpus(dyn): n_train_episodes={len(train_episodes)}  use_full_prob={args.use_full_prob}  "
          f"dyn_cut_seed={args.dyn_cut_seed}  perturb={args.perturb}  perturb_seed={args.perturb_seed if args.perturb else 'N/A'}  "
          f"(訓練 seed={args.seed} 只管權重初始化/batch-order)")

    perturb_cfg = None
    if args.perturb:
        pc = meta["perturb_calibration"]
        perturb_cfg = dict(xy_sigma=float(pc["xy_sigma"]), other_coef=float(pc["other_dim_coef"]),
                           other_std=np.asarray(pc["other_std_train"], dtype=np.float64))
        print(f"perturb_cfg: xy_sigma={perturb_cfg['xy_sigma']:.6f}  other_coef={perturb_cfg['other_coef']}  "
              f"(drift_source={pc['drift_source']}  target_p75={pc['drift_succ_p75_xy_target']:.6f})")

    heldout_blob = torch.load(args.heldout_corpus, weights_only=False)
    val_samples = heldout_blob["val"]
    print(f"held-out（沿用 {os.path.basename(args.heldout_corpus)} 的 val split）: n_val={len(val_samples)}  "
          f"⚠️ 跟 antmaze -val.npz 的 200 題接力 eval 是兩件事")

    model = GenSFT(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                   dropout=args.dropout, max_len=96).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: d_model={args.d_model} n_layers={args.n_layers} n_heads={args.n_heads} n_params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(prog, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    cut_stats = dict()
    batches = iterate_batches_dyn(train_episodes, args.batch_size, order_rng, cut_rng,
                                  args.use_full_prob, perturb_rng, perturb_cfg, cut_stats)
    train_loss_hist, held_out_hist = [], []
    smoke_hist = []
    best_held_loss = float("inf")
    best_state = None
    best_step = -1
    evals_since_improve = 0
    stopped_early_at = None

    ema_loss = None
    for step in range(args.steps):
        batch = next(batches)
        s0_obs, wp_xy, wp_mask, out_tokens, targets = collate_batch(batch, device)
        logits = model(s0_obs, wp_xy, wp_mask, out_tokens)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        lv = float(loss.item())
        ema_loss = lv if ema_loss is None else (0.98 * ema_loss + 0.02 * lv)
        if step % args.log_every == 0 or step == args.steps - 1:
            smoke_hist.append((step, lv, ema_loss))
            train_loss_hist.append((step, lv))
        if step < 500 and step % 20 == 0:
            print(f"  step={step:5d}  loss={lv:.4f}  ema={ema_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ev = evaluate(model, val_samples, device)
            held_out_hist.append((step + 1, ev["loss"], ev["top1_codes_only"], ev["top1_all"], ev["top1_eos"]))
            print(f"[eval step={step+1:6d}] train_loss(ema)={ema_loss:.4f}  "
                  f"held_out_loss={ev['loss']:.4f}  top1_codes={ev['top1_codes_only']*100:.2f}%  "
                  f"top1_all={ev['top1_all']*100:.2f}%  top1_eos={ev['top1_eos']*100:.2f}%  "
                  f"cut_stats={cut_stats}")
            if ev["loss"] < best_held_loss - args.min_delta:
                best_held_loss = ev["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step + 1
                evals_since_improve = 0
            else:
                evals_since_improve += 1
            if evals_since_improve >= args.patience_evals:
                print(f"⏹ held-out loss 連續 {args.patience_evals} 次 eval（{args.patience_evals*args.eval_every} "
                      f"steps）沒有改善 > {args.min_delta}，提早停在 step={step+1}（plateau）")
                stopped_early_at = step + 1
                break

    total_steps_run = stopped_early_at if stopped_early_at else args.steps
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = total_steps_run

    smoke_pre500 = [(s, l) for s, l, _ in smoke_hist if s <= 500]
    smoke_loss0 = smoke_pre500[0][1] if smoke_pre500 else None
    smoke_loss500 = smoke_pre500[-1][1] if smoke_pre500 else None
    smoke_drop = (smoke_loss0 - smoke_loss500) if (smoke_loss0 is not None and smoke_loss500 is not None) else None
    rand_baseline_ce = float(np.log(32))
    print(f"\n=== smoke（前 500 steps）: loss[0]={smoke_loss0:.4f} -> loss[~500]={smoke_loss500:.4f}  "
          f"drop={smoke_drop:.4f}  (亂猜 CE 基準=ln(32)={rand_baseline_ce:.4f}) ===")

    os.makedirs(args.out_dir, exist_ok=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_ev = evaluate(model, val_samples, device)

    ckpt_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
    torch.save(dict(state_dict=best_state, config=dict(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, max_len=96), best_step=best_step,
        total_steps_run=total_steps_run, held_out_at_best=final_ev,
        train_args=vars(args)), ckpt_path)
    print(f"saved checkpoint (best held-out loss @ step={best_step}): {ckpt_path}")

    png_path = os.path.join(args.out_dir, f"{args.tag}_loss_curve.png")
    curves = [
        dict(label="train loss (raw, log-every)", xs=[s for s, _ in train_loss_hist],
             ys=[l for _, l in train_loss_hist]),
        dict(label="held-out loss (aug corpus val)", xs=[s for s, _, _, _, _ in held_out_hist],
             ys=[l for _, l, _, _, _ in held_out_hist]),
    ]
    gc.wv.draw_loss_curve(curves, png_path, f"gen_sft_dyn {args.tag}: train vs held-out loss", ylog=True)
    print(f"saved: {png_path}")

    summary = dict(
        args=vars(args), n_params=n_params,
        dyn_meta=dict(source_corpus=args.corpus, n_train_episodes=len(train_episodes),
                     use_full_prob=args.use_full_prob, dyn_cut_seed=args.dyn_cut_seed,
                     perturb=args.perturb, perturb_seed=args.perturb_seed if args.perturb else None,
                     perturb_cfg=(dict(xy_sigma=perturb_cfg["xy_sigma"], other_coef=perturb_cfg["other_coef"])
                                  if perturb_cfg is not None else None),
                     cut_stats=cut_stats, heldout_corpus=args.heldout_corpus),
        total_steps_run=total_steps_run, stopped_early=(stopped_early_at is not None),
        best_step=best_step, best_held_out=final_ev,
        smoke=dict(loss_at_0=smoke_loss0, loss_at_500=smoke_loss500, drop=smoke_drop,
                   random_baseline_ce_ln32=rand_baseline_ce),
        random_baseline_top1=dict(codes_only_1_over_32=1.0 / 32, all_1_over_33=1.0 / 33),
        held_out_curve=[dict(step=s, loss=l, top1_codes=tc, top1_all=ta, top1_eos=te)
                        for s, l, tc, ta, te in held_out_hist],
        wall_seconds=time.time() - t0,
        ckpt_path=ckpt_path, loss_curve_png=png_path,
    )
    js_path = os.path.join(args.out_dir, f"{args.tag}_train_summary.json")
    gc.wv.save_json(summary, js_path)
    print(f"saved: {js_path}")
    print(f"\n=== FINAL（best checkpoint @ step={best_step}）===")
    print(f"held-out loss={final_ev['loss']:.4f}  top1_codes_only={final_ev['top1_codes_only']*100:.2f}%  "
          f"(baseline 3.125%)  top1_all={final_ev['top1_all']*100:.2f}%  top1_eos={final_ev['top1_eos']*100:.2f}%")
    print(f"cut_stats(全訓練累計)={cut_stats}")
    print(f"=== done wall={time.time()-t0:.1f}s  total_steps_run={total_steps_run} "
          f"stopped_early={stopped_early_at is not None} ===")


if __name__ == "__main__":
    main()
