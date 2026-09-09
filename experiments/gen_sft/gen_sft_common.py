"""experiments/gen_sft/gen_sft_common.py —— gen-sft-v1 共用模組。

負責：把 experiments/walk_verify 與 experiments/walk_verify/teacher_relay 掛進
sys.path，然後把它們既有的函式（wv_common / p1_replay / p2_analyze_traces /
run_teacher_relay）原樣 re-export，供 gen_sft/ 下的腳本 import。

⛔ 不改動任何既有檔案；只 import（純函式呼叫，run_teacher_relay.py 的
   if __name__ == "__main__" guard 讓 import 它沒有副作用）。
⛔ 本檔與 gen_sft/ 下所有新檔一律不碰 GPU（GPU 只在 train_gen_sft.py 內、且只透過
   sbatch 啟動）；本檔自己拿到的 model 走 CPU。

任務構造嚴格同構於 run_teacher_relay.py 的 build_tasks()：直接 import 它來用，
不重寫一份 —— 這樣「同構」是結構保證，不是靠人工比對兩份程式碼。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                       # .../experiments/gen_sft
EXPERIMENTS_DIR = os.path.dirname(HERE)                                  # .../experiments
WV_DIR = os.path.join(EXPERIMENTS_DIR, "walk_verify")                    # .../experiments/walk_verify
TR_DIR = os.path.join(WV_DIR, "teacher_relay")                           # .../teacher_relay
for p in (WV_DIR, TR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import wv_common as wv  # noqa: E402
from p1_replay import sim_obs, q, timed_reach  # noqa: E402,F401
from p2_analyze_traces import MAZE_CENTER, RHO_DEFAULT  # noqa: E402,F401
import run_teacher_relay as rt  # noqa: E402  -- build_tasks / run_one / measure_version / render_exemplars 全部重用

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
DEFAULT_CKPT = os.path.join(WV_DIR, "results", "p0_dict_v1_L4K32_50k.pt")
DEFAULT_RULER = os.path.join(WV_DIR, "results", "ruler_pack.json")
DELTA_SUB = rt.DELTA_SUB
assert DELTA_SUB == 7.5
RHO = RHO_DEFAULT
assert abs(RHO - 1.875) < 1e-9

GEN_SFT_DIR = HERE
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(HERE, "logs")


def load_ruler(path=DEFAULT_RULER):
    import json
    with open(path) as f:
        return json.load(f)


def load_dict_model(ckpt=DEFAULT_CKPT):
    model, cfg = wv.load_dict_v1(ckpt, map_location="cpu")
    assert cfg["seg_len"] == 4
    assert cfg["k"] == 32
    model.eval()
    return model, cfg


def build_corpus(raw, ruler, dict_model, seed=0, delta_sub=DELTA_SUB, rho=RHO):
    """把 train split 的『全部』episodes 轉成 (s0_obs, wp_xy, code_seq) 樣本清單。

    設計對應任務 <design> 第 1 點：
      - 路標序列（M、wp_xy、skip-if-arclen<DELTA_SUB）：直接呼叫 rt.build_tasks()，
        跟 teacher_relay 的 200 條接力題目用【同一個函式】構造 —— 嚴格同構是結構保證。
        n_traj 設成「總 episode 數 + 1」讓它把每一條可用的都收進來，不因 n_traj 提早
        stop；seed 只決定 build_tasks 內部走訪 episode 的順序，不影響哪些 episode
        被收錄（只要 n_traj >= 總數就是全收）、也不影響任何 wp_xy/s0 的數值 ——
        訓練用的 train/val 切分另外用 split_seed=42 對 episode id 做（見
        split_corpus()），不依賴這個順序，所以這個 seed 對最終資料集內容無影響，
        仍然記錄下來（seed=0）以昭固定。
      - code 序列：不逐段迴圈呼叫 encoder（慢），改用 wv.cut_segments +
        wv.build_seg_tensors 一次切出【整個資料集】的所有不跨 episode 邊界的 4 步
        動作段（跟 P0 訓練字典時的切法完全同一套函式），對齊 episode_bounds 的
        走訪順序做逐集切片，得到跟 rt.build_tasks 回傳的 task["n_chunks"] 完全對得上
        的每集 code 序列（都是 T//4，來源同一份 cut_segments 保證不會兩邊算出不同的
        chunk 數）。encode 全部用 model.encode_idx 一次 batched 呼叫（eval mode，
        MLP 無 dropout/batchnorm，train/eval 對 forward 無影響，但仍呼叫過
        model.eval() 以求明確）。

    回傳 list[dict(episode, s0_idx, s0_obs(29,), wp_xy(M,2), code_seq(n_chunks,), M, n_chunks)]
    以及 skipped（同 build_tasks 回傳格式）。
    """
    starts, ends = wv.episode_bounds(raw["terminals"])
    n_ep = len(starts)
    tasks, skipped = rt.build_tasks(raw, n_traj=n_ep + 1, seed=seed, ruler=ruler, rho=rho)
    assert len(tasks) + len(skipped) == n_ep, (
        f"⛔ build_tasks 沒有走訪到全部 episode：tasks={len(tasks)} skipped={len(skipped)} "
        f"n_ep={n_ep}")

    seg_starts_all = wv.cut_segments(raw, 4)
    segs_all, _cond_all = wv.build_seg_tensors(raw, seg_starts_all, 4)
    with torch.no_grad():
        idx_all = dict_model.encode_idx(torch.from_numpy(segs_all)).numpy().astype(np.int64)

    # 逐集切片出 code_seq：cut_segments 內部就是照 episode_bounds 的順序、每集切
    # T//4 段（不跨界），offset 累加就能精確對齊。
    code_seq_by_ep = {}
    offset = 0
    for ei, (s0, e0) in enumerate(zip(starts, ends)):
        T = int(e0 - s0 + 1)
        n_chunks = T // 4
        code_seq_by_ep[ei] = idx_all[offset:offset + n_chunks].copy()
        offset += n_chunks
    assert offset == len(idx_all), f"⛔ 切片沒有用完全部段：offset={offset} len={len(idx_all)}"

    samples = []
    for t in tasks:
        ei = t["episode"]
        code_seq = code_seq_by_ep[ei]
        assert len(code_seq) == t["n_chunks"], (
            f"⛔ code_seq 長度與 task n_chunks 不一致：episode={ei} "
            f"len(code_seq)={len(code_seq)} n_chunks={t['n_chunks']}")
        s0_obs = raw["observations"][t["s0"]].astype(np.float32).copy()
        samples.append(dict(
            episode=int(ei), s0_idx=int(t["s0"]), s0_obs=s0_obs,
            wp_xy=np.asarray(t["wp_xy"], dtype=np.float32).copy(),
            code_seq=code_seq.astype(np.int64), M=int(t["M"]), n_chunks=int(t["n_chunks"]),
        ))
    return samples, skipped, dict(build_tasks_seed=seed, n_episodes=n_ep,
                                  n_usable=len(tasks), n_skipped=len(skipped))


def split_corpus(samples, split_seed=42, val_frac=0.1):
    """episode 級 9:1 切分（design 第 2 點）。用 episode id 排序後打亂，避免依賴
    build_corpus 回傳順序（該順序來自 build_tasks 的隨機走訪序，不應該混進切分）。
    """
    eps = np.array(sorted(s["episode"] for s in samples), dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(eps))
    n_val = int(len(eps) * val_frac)
    val_eps = set(eps[perm[:n_val]].tolist())
    train_eps = set(eps[perm[n_val:]].tolist())
    assert train_eps.isdisjoint(val_eps)
    train_samples = [s for s in samples if s["episode"] in train_eps]
    val_samples = [s for s in samples if s["episode"] in val_eps]
    assert len(train_samples) + len(val_samples) == len(samples)
    return train_samples, val_samples, dict(split_seed=split_seed, val_frac=val_frac,
                                            n_train=len(train_samples), n_val=len(val_samples))
