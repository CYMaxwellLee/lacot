#!/usr/bin/env python
"""建教材語料：train split 全部 episodes -> (s0_obs, wp_xy, code_seq) 樣本，
9:1 episode 級切分（split_seed=42），存成 results/corpus_v1.pt。

⛔ CPU-only、輕量（batched MLP forward over ~25萬段，秒級），不經 slurm。
⛔ 不改動任何既有檔案。
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402

BUILD_TASKS_SEED = 0   # 見 gen_sft_common.build_corpus docstring：只影響走訪順序，不影響內容
SPLIT_SEED = 42
VAL_FRAC = 0.1


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()
    model, cfg = gc.load_dict_model()
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}")

    samples, skipped, meta = gc.build_corpus(raw, ruler, model, seed=BUILD_TASKS_SEED)
    print(f"build_corpus: usable={meta['n_usable']} skipped={meta['n_skipped']} "
          f"n_episodes={meta['n_episodes']}  seed={BUILD_TASKS_SEED}")
    for s in skipped[:10]:
        print(f"  skip episode={s['episode']} arclen={s['arclen']:.3f} reason={s['reason']}")
    if len(skipped) > 10:
        print(f"  ...(共 {len(skipped)} 條跳過，只列前 10)")

    Ms = np.array([s["M"] for s in samples])
    Ls = np.array([s["n_chunks"] for s in samples])
    print(f"waypoints M: min={Ms.min()} p50={np.median(Ms):.0f} max={Ms.max()}")
    print(f"code_seq 長度 L(=n_chunks): min={Ls.min()} p50={np.median(Ls):.0f} max={Ls.max()}")

    train_samples, val_samples, split_meta = gc.split_corpus(
        samples, split_seed=SPLIT_SEED, val_frac=VAL_FRAC)
    print(f"split(episode-level, split_seed={SPLIT_SEED}): "
          f"n_train={split_meta['n_train']} n_val={split_meta['n_val']}")

    os.makedirs(gc.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(gc.RESULTS_DIR, "corpus_v1.pt")
    torch.save(dict(
        train=train_samples, val=val_samples,
        meta=dict(**meta, **split_meta,
                  delta_sub=gc.DELTA_SUB, rho=gc.RHO, dict_ckpt=gc.DEFAULT_CKPT,
                  dict_cfg=cfg, dataset=gc.DATASET, data_dir=gc.DD,
                  waypoints_M=dict(min=int(Ms.min()), p50=float(np.median(Ms)), max=int(Ms.max())),
                  code_len_L=dict(min=int(Ls.min()), p50=float(np.median(Ls)), max=int(Ls.max())),
                  built_at=time.strftime("%Y-%m-%d %H:%M:%S")),
    ), out_path)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"saved: {out_path} ({sz_mb:.2f} MB)")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
