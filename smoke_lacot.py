"""m4 獨立性 smoke test — 證明它真的脫離了 wpm/fpo。"""
import sys, os, importlib, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FAIL = []

print("=== ① lacot 每個模組 import ===")
for m in ["lacot.backbone", "lacot.e_target", "lacot.heads", "lacot.model", "lacot.nf_head"]:
    try:
        importlib.import_module(m)
        print("  OK  " + m)
    except Exception as e:
        FAIL.append((m, repr(e)))
        print("  FAIL " + m + ": " + str(e))

print("\n=== \u2461 核心 class 真的跑一次 forward（不只建得起來）===")
import torch
try:
    from lacot.e_target import PerceiverPooler
    from lacot.nf_head import Flow
    from lacot.model import RefineOperator
    from lacot.heads import DiscretizedActionHead
    torch.manual_seed(0)
    B, T, D_IN, D_MODEL, K = 4, 12, 16, 64, 8

    pool = PerceiverPooler(d_in=D_IN, d_model=D_MODEL, k=K, num_layers=1, num_heads=2)
    feats = torch.randn(B, T, D_IN)
    et = pool(feats)
    assert et.shape == (B, K, D_MODEL), et.shape
    print("  OK  PerceiverPooler  {} -> {}  params={:,}".format(
        tuple(feats.shape), tuple(et.shape), sum(x.numel() for x in pool.parameters())))

    flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=2, d_hidden=64, cond_dim=D_MODEL)
    cond = torch.randn(B, D_MODEL)
    lp = flow.log_prob(et, cond)
    assert lp.shape == (B,), lp.shape
    u = flow.sample(B, cond)
    assert u.shape == (B, K, D_MODEL), u.shape
    print("  OK  Flow.log_prob   -> {}  mean={:.3f}".format(tuple(lp.shape), lp.mean().item()))
    print("  OK  Flow.sample     -> {}".format(tuple(u.shape)))
except Exception as e:
    FAIL.append(("forward", repr(e)))
    traceback.print_exc()

print("\n=== ③ 資料讀得到（走官方 OGBENCH_DATA_DIR）===")
import numpy as np
DATA = os.environ.get("OGBENCH_DATA_DIR")
print("  OGBENCH_DATA_DIR = " + str(DATA))
try:
    d = np.load(os.path.join(DATA, "pointmaze-medium-navigate-v0.npz"))
    keys = list(d.keys())
    shape = d["observations"].shape
    print("  OK  keys={} observations.shape={}".format(keys[:5], shape))
except Exception as e:
    FAIL.append(("data", repr(e)))
    print("  FAIL " + str(e))

print("\n=== ④ 決定性檢查：wpm/fpo 有沒有偷偷被載進來 ===")
polluted = [m for m in sys.modules if m.split(".")[0] in ("wpm", "fpo")]
print("  sys.modules 裡的 wpm/fpo: " + (str(polluted) if polluted else "無 (乾淨)"))
if polluted: FAIL.append(("pollution", str(polluted)))
bad_path = [p for p in sys.path if "/fpo" in p]
print("  sys.path 指向 fpo 的:    " + (str(bad_path) if bad_path else "無 (乾淨)"))
if bad_path: FAIL.append(("syspath", str(bad_path)))

print("\n" + "=" * 52)
if FAIL:
    print("FAILED {} 項: {}".format(len(FAIL), FAIL))
    sys.exit(1)
print("ALL PASS — LaCoT 已完全脫離 wpm/fpo，可獨立執行")
