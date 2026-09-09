#!/usr/bin/env python
"""驗收準則①：golden 一致性 —— 抽 3 條 train episodes 各 4 段，資料管線（build_corpus，
走 wv.cut_segments + wv.build_seg_tensors 批次切）轉出的 code 要跟「直接對同一段動作
呼叫 ckpt encoder」逐位元一致；並附一個故意打亂動作段時間順序的反例，證明比對機制
真的會亮（code 必須不同，不是恆等的空比對）。

⛔ CPU-only、輕量（幾秒等級：25 萬段的一次 batched MLP forward + 12 段直接比對），
   不經 slurm（跟其他同量級 walk_verify 診斷腳本同慣例）。
⛔ 不改動任何既有檔案。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402

EP_PICK_SEED = 1234   # 只決定「抽哪 3 條 episode 來示範」，不影響比對邏輯本身
N_EP = 3
N_SEG_PER_EP = 4


def main():
    torch.set_num_threads(4)
    raw = gc.wv.load_npz(gc.DD, gc.DATASET, "train")
    ruler = gc.load_ruler()
    model, cfg = gc.load_dict_model()
    print(f"dict: seg_len={cfg['seg_len']} K={cfg['k']}  train episodes(steps)={len(raw['terminals'])}")

    samples, skipped, meta = gc.build_corpus(raw, ruler, model, seed=0)
    print(f"build_corpus: usable={meta['n_usable']} skipped={meta['n_skipped']} "
          f"n_episodes={meta['n_episodes']}")
    by_ep = {s["episode"]: s for s in samples}

    starts, ends = gc.wv.episode_bounds(raw["terminals"])
    rng = np.random.default_rng(EP_PICK_SEED)
    # 只從「有被 build_corpus 收錄」的 episode 裡抽（M>=1），且要有 >=N_SEG_PER_EP 段
    eligible = [ei for ei in by_ep if by_ep[ei]["n_chunks"] >= N_SEG_PER_EP]
    picked_eps = sorted(rng.choice(eligible, size=N_EP, replace=False).tolist())
    print(f"picked episodes (seed={EP_PICK_SEED}): {picked_eps}")

    act = raw["actions"]
    n_total, n_match = 0, 0
    rows = []
    for ei in picked_eps:
        s0 = int(starts[ei])
        pipeline_codes = by_ep[ei]["code_seq"]
        for c in range(N_SEG_PER_EP):
            seg = act[s0 + 4 * c: s0 + 4 * c + 4].reshape(1, -1).astype(np.float32)
            with torch.no_grad():
                direct_code = int(model.encode_idx(torch.from_numpy(seg)).item())
            pipe_code = int(pipeline_codes[c])
            ok = (direct_code == pipe_code)
            n_total += 1
            n_match += int(ok)
            rows.append(dict(episode=int(ei), chunk=c, pipeline_code=pipe_code,
                             direct_code=direct_code, match=ok))
            print(f"  ep={ei:4d} chunk={c}  pipeline_code={pipe_code:2d}  "
                  f"direct_code={direct_code:2d}  match={ok}")

    all_match = (n_match == n_total)
    print(f"\n=== golden 一致性：{n_match}/{n_total} match  ALL_MATCH={all_match} ===")

    # ---- 反例：打亂該段 4 步的時間順序（reverse），確認 code 會變 ----
    print("\n=== 反例（時間順序打亂，預期 code 改變）===")
    n_neg_total, n_neg_changed = 0, 0
    neg_rows = []
    for ei in picked_eps:
        s0 = int(starts[ei])
        for c in range(N_SEG_PER_EP):
            seg = act[s0 + 4 * c: s0 + 4 * c + 4].copy()
            with torch.no_grad():
                orig_code = int(model.encode_idx(
                    torch.from_numpy(seg.reshape(1, -1).astype(np.float32))).item())
            shuffled = seg[::-1].copy()  # 反轉 4 步順序
            with torch.no_grad():
                shuf_code = int(model.encode_idx(
                    torch.from_numpy(shuffled.reshape(1, -1).astype(np.float32))).item())
            changed = (shuf_code != orig_code)
            n_neg_total += 1
            n_neg_changed += int(changed)
            neg_rows.append(dict(episode=int(ei), chunk=c, orig_code=orig_code,
                                 reversed_code=shuf_code, changed=changed))
            print(f"  ep={ei:4d} chunk={c}  orig_code={orig_code:2d}  "
                  f"reversed_code={shuf_code:2d}  changed={changed}")

    fallback_rows = []
    if n_neg_changed < n_neg_total:
        print("\n  某些反轉沒有改變 code，升級反例：改用『換成另一條完全不同 episode 的段』")
        other_rng = np.random.default_rng(EP_PICK_SEED + 1)
        for r in neg_rows:
            if r["changed"]:
                continue
            ei, c = r["episode"], r["chunk"]
            s0 = int(starts[ei])
            orig_code = r["orig_code"]
            other_ei = int(other_rng.choice([e for e in range(len(starts)) if e != ei]))
            other_s0 = int(starts[other_ei])
            other_seg = act[other_s0:other_s0 + 4].reshape(1, -1).astype(np.float32)
            with torch.no_grad():
                other_code = int(model.encode_idx(torch.from_numpy(other_seg)).item())
            changed2 = (other_code != orig_code)
            fallback_rows.append(dict(episode=ei, chunk=c, orig_code=orig_code,
                                      other_episode=other_ei, other_code=other_code,
                                      changed=changed2))
            print(f"    ep={ei} chunk={c}: orig={orig_code} vs other_ep={other_ei}_chunk0="
                  f"{other_code}  changed={changed2}")
            n_neg_changed += int(changed2)

    neg_pass = (n_neg_changed >= 1)  # 至少要能亮：不是恆等比對
    all_neg_ok = all(r["changed"] for r in neg_rows) or (
        len(fallback_rows) > 0 and all(r["changed"] for r in fallback_rows))

    out = dict(
        golden=dict(n_total=n_total, n_match=n_match, all_match=all_match, rows=rows,
                   picked_episodes=picked_eps, ep_pick_seed=EP_PICK_SEED),
        adversarial_reverse=dict(n_total=n_neg_total, n_changed=n_neg_changed, rows=neg_rows),
        adversarial_fallback_cross_episode=fallback_rows,
        neg_pass_at_least_one_changed=neg_pass,
        all_reverse_or_fallback_changed=all_neg_ok,
    )
    os.makedirs(gc.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(gc.RESULTS_DIR, "golden_check.json")
    gc.wv.save_json(out, out_path)
    print(f"\nsaved: {out_path}")
    print(f"\n=== PASS/FAIL ===")
    print(f"golden 一致性 (12/12 bit-exact 必須全過): {'PASS' if all_match else 'FAIL'}")
    print(f"反例會亮 (至少 1 個 changed): {'PASS' if neg_pass else 'FAIL'}")
    print(f"反例全部亮 (reverse 或 fallback 後 12/12 changed): "
          f"{'PASS' if all_neg_ok else 'FAIL(見 rows 細節)'}")
    if not all_match:
        print("⛔ golden 不一致，停手回報，不准繼續訓練/eval。")
        sys.exit(1)


if __name__ == "__main__":
    main()
