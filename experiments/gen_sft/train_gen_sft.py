#!/usr/bin/env python
"""訓練 u：teacher forcing CE，(s0,路標序列)->code 序列自回歸。GPU only（走 sbatch）。

驗收準則②：
  - smoke：前 500 steps 內 loss 要明顯下降（本檔用細顆粒度 log，事後從同一輪 log
    的前 500 steps 直接讀，不另外跑一輪省時間 GPU）。
  - 訓完（30k~60k steps 或 held-out loss 走平提早停）：held-out（本檔的 corpus 9:1
    切分之 val，⚠️ 不是 antmaze -val.npz 那個 200 題接力用的 val，兩者不同一件事）
    teacher-forcing next-token top-1（只算「猜哪個 code」的位置，不含 EOS 位置，
    baseline=1/32=3.125%，跟驗收準則的字面基準對齊）要遠超過亂猜線，如實報數字。

⛔ 不改動任何既有檔案。
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

# ⛔ 除錯記錄（2026-09-09，job 26819/21/23/25 連續 FAILED，job 26827 起加這段才過）：
# run_teacher_relay.py 檔頭有 `os.environ.setdefault("HIP_VISIBLE_DEVICES", "")`（它
# 自己是 CPU-only 腳本，這樣寫是防自己誤連 GPU）。slurm 的 gres 只設了
# CUDA_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES，從沒設過 HIP_VISIBLE_DEVICES —— 所以
# 這個 setdefault 在「這個變數本來就不存在」時真的會生效，把它釘成空字串。gen_sft
# 這邊經 `import gen_sft_common` 會連帶 `import run_teacher_relay`，一旦這個 import
# 發生在本行程第一次呼叫 torch.cuda.is_available() 之前，ROCm 就會把「HIP 可見裝置
# =空字串」讀成「看不到卡」，即使 CUDA_VISIBLE_DEVICES 明明是 "0"。torch 對
# is_available() 的裝置偵測結果會在第一次呼叫後快取住，所以只要在 import
# gen_sft_common 之前先叫一次就能把「有卡」鎖住，後面 setdefault 再怎麼設都不影響
# 這個行程。故意在這裡先呼叫一次、印出來當證據，不是留下的除錯殘骸。
_cuda_ok_before = torch.cuda.is_available()
print(f"[gpu-visibility guard] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
      f"is_available(before importing gen_sft_common)={_cuda_ok_before}", flush=True)

import gen_sft_common as gc  # noqa: E402
from model import GenSFT, BOS_ID, EOS_ID  # noqa: E402

print(f"[gpu-visibility guard] HIP_VISIBLE_DEVICES(after importing gen_sft_common)="
      f"{os.environ.get('HIP_VISIBLE_DEVICES')!r}  is_available(cached)={torch.cuda.is_available()}",
      flush=True)


def collate_batch(samples, device):
    B = len(samples)
    Mmax = max(s["M"] for s in samples)
    Lseq = max(s["n_chunks"] + 1 for s in samples)
    s0_obs = torch.zeros(B, 29)
    wp_xy = torch.zeros(B, Mmax, 2)
    wp_mask = torch.zeros(B, Mmax, dtype=torch.bool)
    out_tokens = torch.zeros(B, Lseq, dtype=torch.long)
    targets = torch.full((B, Lseq), -100, dtype=torch.long)
    for i, s in enumerate(samples):
        M, L = s["M"], s["n_chunks"]
        s0_obs[i] = torch.from_numpy(s["s0_obs"])
        wp_xy[i, :M] = torch.from_numpy(s["wp_xy"])
        wp_mask[i, :M] = True
        codes = torch.from_numpy(s["code_seq"])
        out_tokens[i, 0] = BOS_ID
        out_tokens[i, 1:1 + L] = codes
        targets[i, 0:L] = codes
        targets[i, L] = EOS_ID
    return (s0_obs.to(device), wp_xy.to(device), wp_mask.to(device),
            out_tokens.to(device), targets.to(device))


def iterate_batches(samples, batch_size, rng):
    n = len(samples)
    order = rng.permutation(n)
    pos = 0
    while True:
        if pos + batch_size > n:
            order = rng.permutation(n)
            pos = 0
        idx = order[pos:pos + batch_size]
        pos += batch_size
        yield [samples[i] for i in idx]


@torch.no_grad()
def evaluate(model, samples, device, batch_size=256):
    model.eval()
    tot_loss, tot_n = 0.0, 0
    correct_codes, n_codes = 0, 0
    correct_all, n_all = 0, 0
    correct_eos, n_eos = 0, 0
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        s0_obs, wp_xy, wp_mask, out_tokens, targets = collate_batch(batch, device)
        logits = model(s0_obs, wp_xy, wp_mask, out_tokens)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                               ignore_index=-100, reduction="sum")
        mask = targets != -100
        tot_loss += float(loss.item())
        tot_n += int(mask.sum().item())
        pred = logits.argmax(dim=-1)
        correct_all += int(((pred == targets) & mask).sum().item())
        n_all += int(mask.sum().item())
        code_mask = mask & (targets != EOS_ID)
        correct_codes += int(((pred == targets) & code_mask).sum().item())
        n_codes += int(code_mask.sum().item())
        eos_mask = mask & (targets == EOS_ID)
        correct_eos += int(((pred == targets) & eos_mask).sum().item())
        n_eos += int(eos_mask.sum().item())
    model.train()
    return dict(
        loss=tot_loss / max(tot_n, 1),
        top1_all=correct_all / max(n_all, 1), n_all=n_all,
        top1_codes_only=correct_codes / max(n_codes, 1), n_codes=n_codes,
        top1_eos=correct_eos / max(n_eos, 1), n_eos=n_eos,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=os.path.join(gc.RESULTS_DIR, "corpus_v1.pt"))
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--patience-evals", type=int, default=8,
                    help="held-out loss 連續這麼多次 eval 沒有改善(> min_delta)就提早停")
    ap.add_argument("--min-delta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tag", type=str, default="gensft_v1")
    ap.add_argument("--out-dir", type=str, default=gc.RESULTS_DIR)
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
          f"(corpus 內部 9:1 held-out，split_seed={meta['split_seed']} —— "
          f"⚠️ 不是 antmaze -val.npz 那個 200 題接力用的 val)")
    print(f"waypoints_M={meta['waypoints_M']}  code_len_L={meta['code_len_L']}")

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

    batches = iterate_batches(train_samples, args.batch_size, rng)
    train_loss_hist, held_out_hist = [], []
    smoke_hist = []  # (step, loss) 逐 log_every 紀錄，供事後抽前 500 steps 檢查
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

    # ---- smoke 判讀：前 500 steps 內 loss 是否明顯下降 ----
    smoke_pre500 = [(s, l) for s, l, _ in smoke_hist if s <= 500]
    smoke_loss0 = smoke_pre500[0][1] if smoke_pre500 else None
    smoke_loss500 = smoke_pre500[-1][1] if smoke_pre500 else None
    smoke_drop = (smoke_loss0 - smoke_loss500) if (smoke_loss0 is not None and smoke_loss500 is not None) else None
    rand_baseline_ce = float(np.log(32))  # 均勻猜 32 類的 CE
    print(f"\n=== smoke（前 500 steps）: loss[0]={smoke_loss0:.4f} -> loss[~500]={smoke_loss500:.4f}  "
          f"drop={smoke_drop:.4f}  (亂猜 CE 基準=ln(32)={rand_baseline_ce:.4f}) ===")

    os.makedirs(args.out_dir, exist_ok=True)
    # 最終評估：用 best checkpoint（held-out loss 最低）
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_ev = evaluate(model, val_samples, device)

    ckpt_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")
    torch.save(dict(state_dict=best_state, config=dict(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, max_len=96), best_step=best_step,
        total_steps_run=total_steps_run, held_out_at_best=final_ev,
        train_args=vars(args)), ckpt_path)
    print(f"saved checkpoint (best held-out loss @ step={best_step}): {ckpt_path}")

    # loss curve（Pillow，沿用 wv_common）
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
        args=vars(args), n_params=n_params, corpus_meta={k: v for k, v in meta.items() if k != "dict_cfg"},
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
