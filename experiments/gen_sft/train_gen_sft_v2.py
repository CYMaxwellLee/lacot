#!/usr/bin/env python
"""experiments/gen_sft/train_gen_sft_v2.py —— gen-sft-v2：容量掃 + 加權教材，兩用一支。

跟 train_gen_sft.py 的關係：⛔ 不改動既有檔案，這支是新檔，import 既有
train_gen_sft.py 的 collate_batch()/evaluate()（模組層函式，import 不會觸發它的
main()，因為它有 `if __name__=="__main__"` guard），訓練迴圈本身重寫一份 ——
唯一新增的行為是「怎麼從 train_samples 抽 batch」（加權/過濾），其餘（模型建構、
optimizer/schedule、smoke 判讀、checkpoint schema、loss curve 存圖、summary json）
逐項照抄 v1，刻意保持 checkpoint schema 跟 v1 位元對位相容，這樣 eval_gen_sft.py
（讀 ckpt 內的 config 字典建模型）不必修改就能吃任何 v2 checkpoint。

兩個用途：
  1) 容量掃（design 第一步）：--weight-mode none，只改 --d-model/--n-layers/--n-heads，
     教材維持均勻抽樣（跟 v1 完全同語意），只是換架構。
  2) 加權教材（design 第二步，在容量掃選出的 C* 架構上跑）：--weight-mode soft 用
     score_corpus.py 存好的 weights_soft 做『有放回、依權重』抽樣；--weight-mode
     hard 用 mask_hard 篩選 train_samples 後、抽樣方式退回跟 v1 一樣的均勻無放回
     epoch 循環（只是資料集變小了，抽法不變）。

⛔ 新檔，不改動任何既有檔案。GPU only（走 sbatch）。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ⛔ 同 train_gen_sft.py §1.3 的雷（run_teacher_relay.py 的
# os.environ.setdefault("HIP_VISIBLE_DEVICES","") 會在 slurm 從沒設過這個變數的
# 前提下把它釘成空字串，蓋過 CUDA_VISIBLE_DEVICES）——修法一樣：在任何會 import
# gen_sft_common/run_teacher_relay/train_gen_sft 的敘述之前，先呼叫一次
# torch.cuda.is_available() 把偵測結果快取住（torch 對這個呼叫的結果是 process
# 全域快取，不管是哪個模組觸發的都算數）。這裡直接照抄，不依賴「import
# train_gen_sft 會連帶觸發它自己那份 guard」這種間接路徑。
_cuda_ok_before = torch.cuda.is_available()
print(f"[gpu-visibility guard] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
      f"is_available(before importing gen_sft_common)={_cuda_ok_before}", flush=True)

import gen_sft_common as gc  # noqa: E402
from model import GenSFT, BOS_ID, EOS_ID  # noqa: E402
import train_gen_sft as base  # noqa: E402  -- 重用 collate_batch()/evaluate()

print(f"[gpu-visibility guard] HIP_VISIBLE_DEVICES(after importing)="
      f"{os.environ.get('HIP_VISIBLE_DEVICES')!r}  is_available(cached)={torch.cuda.is_available()}",
      flush=True)


def load_weighting(scores_path, train_samples, weight_mode, corpus_path=None):
    """從 score_corpus.py 的輸出檔取出跟 train_samples 對齊的 weights 或 mask。
    用 episode id 對齊（不假設 corpus_v1.pt 跟 corpus_scores_v2.pt 兩邊的樣本順序
    相同），並且 assert 每一個 train_samples 的 episode 都能在 scores 檔裡找到——
    找不到就是兩份檔案不是同一批 corpus 建的，停手，不准悄悄跳過。

    ⚠️ 補強（opus 抽驗抓到原本的 assert 沒檢查兩件事）：① scores 檔內部
    `weights_soft`/`mask_hard` 長度跟 `train_episode` 長度是否一致（不一致代表
    這份分數檔本身壞掉，之後用 order 去 index 會拿到錯位或直接 IndexError，訊息
    不明顯，先在這裡明確擋）；② scores 檔記錄的 `config.corpus_path`（見
    score_corpus.py）是否跟這次訓練實際讀的 corpus 同一份——不同不一定是錯（例如
    刻意換了一份新 corpus 但事後才發現忘記重算分數），但至少要印警告，不能安靜
    地拿舊分數配新語料。
    """
    blob = torch.load(scores_path, weights_only=False)
    score_ep = blob["train_episode"]
    for key in ("weights_soft", "mask_hard"):
        assert len(blob[key]) == len(score_ep), (
            f"⛔ {scores_path} 內部長度不一致：train_episode={len(score_ep)} "
            f"{key}={len(blob[key])}——這份分數檔本身壞了，停手")
    scores_corpus_path = blob.get("config", {}).get("corpus_path")
    if corpus_path is not None and scores_corpus_path is not None and (
            os.path.abspath(scores_corpus_path) != os.path.abspath(corpus_path)):
        print(f"⚠️ 警告：{scores_path} 是對 {scores_corpus_path} 算的分數，但這次訓練讀的 corpus 是 "
              f"{corpus_path}——路徑不同，可能不是同一批建的，continue 但請自己確認")
    ep_to_idx = {int(e): i for i, e in enumerate(score_ep)}
    my_ep = [int(s["episode"]) for s in train_samples]
    missing = [e for e in my_ep if e not in ep_to_idx]
    assert not missing, (
        f"⛔ {len(missing)} 個 train episode 在 {scores_path} 裡找不到分數"
        f"（前 5 個：{missing[:5]}）—— corpus_v1.pt 跟 corpus_scores_v2.pt 疑似不是同一批，停手")
    order = np.array([ep_to_idx[e] for e in my_ep], dtype=np.int64)
    if weight_mode == "soft":
        weights_soft = np.asarray(blob["weights_soft"])[order]
        return dict(weights=weights_soft, beta=float(blob["beta"]), ess_frac=float(blob["ess_frac"]))
    if weight_mode == "hard":
        mask_hard = np.asarray(blob["mask_hard"])[order]
        return dict(mask=mask_hard, median_score=float(blob["median_score"]))
    raise ValueError(f"⛔ 不認得的 weight_mode={weight_mode!r}")


def iterate_batches_weighted(samples, batch_size, rng, weights):
    """有放回、依權重抽樣（multinomial importance sampling 的標準做法）——跟 v1
    iterate_batches() 的『均勻、無放回、epoch 循環』不同語意，這是刻意的改變：
    加權抽樣如果還要維持『無放回直到用完一輪』會讓權重在一個 epoch 內被攤平
    （每個樣本一個 epoch 內固定只出現一次，權重就只影響『順序』不影響『頻率』），
    真正讓權重影響『看到的頻率』要用有放回抽樣，這是這個模式存在的意義本身。
    """
    n = len(samples)
    p = weights / weights.sum()
    while True:
        idx = rng.choice(n, size=batch_size, replace=True, p=p)
        yield [samples[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_v1.pt"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--patience-evals", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="gensft_v2")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
    ap.add_argument("--weight-mode", type=str, default="none", choices=["none", "soft", "hard"])
    ap.add_argument("--scores-path", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_scores_v2.pt"))
    args = ap.parse_args()

    assert args.device == "cpu" or torch.cuda.is_available(), (
        "⛔ 要求 --device cuda 但 torch.cuda.is_available()=False —— 停手回報，不准偷偷退回 CPU")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    blob = torch.load(args.corpus, weights_only=False)
    train_samples, val_samples, meta = blob["train"], blob["val"], blob["meta"]
    print(f"corpus: n_train={len(train_samples)} n_val={len(val_samples)} "
          f"split_seed={meta['split_seed']}  weight_mode={args.weight_mode}")

    weighting_info = dict(weight_mode=args.weight_mode)
    sample_weights = None
    if args.weight_mode == "none":
        train_iter_samples = train_samples
        batches = base.iterate_batches(train_iter_samples, args.batch_size, rng)
    elif args.weight_mode == "hard":
        w = load_weighting(args.scores_path, train_samples, "hard", corpus_path=args.corpus)
        mask = w["mask"]
        train_iter_samples = [s for s, keep in zip(train_samples, mask) if keep]
        print(f"硬過濾：{len(train_iter_samples)}/{len(train_samples)} 保留 "
              f"(median_score={w['median_score']:.4f}，來自 {args.scores_path})")
        weighting_info.update(median_score=w["median_score"], n_kept=len(train_iter_samples),
                              n_total=len(train_samples))
        batches = base.iterate_batches(train_iter_samples, args.batch_size, rng)
    else:  # soft
        w = load_weighting(args.scores_path, train_samples, "soft", corpus_path=args.corpus)
        sample_weights = w["weights"]
        train_iter_samples = train_samples
        print(f"軟加權：beta={w['beta']:.4f} ess_frac={w['ess_frac']:.3f} "
              f"weights min/p50/max={sample_weights.min():.3f}/{np.median(sample_weights):.3f}/"
              f"{sample_weights.max():.3f}（來自 {args.scores_path}）")
        weighting_info.update(beta=w["beta"], ess_frac=w["ess_frac"])
        batches = iterate_batches_weighted(train_iter_samples, args.batch_size, rng, sample_weights)

    model = GenSFT(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                   dropout=args.dropout, max_len=96).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: d_model={args.d_model} n_layers={args.n_layers} n_heads={args.n_heads} "
          f"n_params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(prog, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

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
        s0_obs, wp_xy, wp_mask, out_tokens, targets = base.collate_batch(batch, device)
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
            ev = base.evaluate(model, val_samples, device)
            held_out_hist.append((step + 1, ev["loss"], ev["top1_codes_only"], ev["top1_all"], ev["top1_eos"]))
            print(f"[eval step={step+1:6d}] train_loss(ema)={ema_loss:.4f}  "
                  f"held_out_loss={ev['loss']:.4f}  top1_codes={ev['top1_codes_only']*100:.2f}%  "
                  f"top1_all={ev['top1_all']*100:.2f}%  top1_eos={ev['top1_eos']*100:.2f}%")
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
    final_ev = base.evaluate(model, val_samples, device)

    ckpt_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
    torch.save(dict(state_dict=best_state, config=dict(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, max_len=96), best_step=best_step,
        total_steps_run=total_steps_run, held_out_at_best=final_ev,
        train_args=dict(vars(args), weighting_info=weighting_info)), ckpt_path)
    print(f"saved checkpoint (best held-out loss @ step={best_step}): {ckpt_path}")

    png_path = os.path.join(args.out_dir, f"{args.tag}_loss_curve.png")
    curves = [
        dict(label="train loss (raw, log-every)", xs=[s for s, _ in train_loss_hist],
             ys=[l for _, l in train_loss_hist]),
        dict(label="held-out loss (corpus val)", xs=[s for s, _, _, _, _ in held_out_hist],
             ys=[l for _, l, _, _, _ in held_out_hist]),
    ]
    gc.wv.draw_loss_curve(curves, png_path, f"gen_sft {args.tag}: train vs held-out loss", ylog=True)
    print(f"saved: {png_path}")

    summary = dict(
        args=vars(args), weighting_info=weighting_info, n_params=n_params,
        corpus_meta={k: v for k, v in meta.items() if k != "dict_cfg"},
        n_train_used=len(train_iter_samples), n_train_total=len(train_samples),
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
    print(f"=== done wall={time.time()-t0:.1f}s  total_steps_run={total_steps_run} "
          f"stopped_early={stopped_early_at is not None} ===")


if __name__ == "__main__":
    main()
