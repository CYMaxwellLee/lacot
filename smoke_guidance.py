"""smoke：eval-time CFG 式 intent 引導（LACOT_INTENT_GUID_W）。

⛔ 本機沒 GPU、也不用真 ckpt —— 把主線那三支【頂層純函式】用 AST 挖出來 exec，
   配一顆隨機小權重的真 Flow（lacot/nf_head.py），驗的就是真的跑在主線上的那一份。

驗三件事（全 PASS 才算交）：
  ① GUID_W=0／env 缺席 ⇒ 取樣輸出與未開 guidance 的路徑【逐位元相同】
  ② w=1 ⇒ 與正常帶 intent 取樣 allclose(atol=1e-5)（本實作用重排式 ⇒ 實際是逐位元）
  ③ w=2 ⇒ 輸出確實改變（非恆等）且全部有限

    $HOME/venvs/lacot-rocm/bin/python smoke_guidance.py
"""
import ast
import os
import pathlib
import sys

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lacot.nf_head import Flow

ROLLOUT = ROOT / "experiments" / "scratch_lacot_rollout.py"
SRC = ROLLOUT.read_text()
TREE = ast.parse(SRC)
fails = []


def bad(msg):
    fails.append(msg)
    print(f"       🚨 {msg}")


def load_funcs(*names):
    """把指定的頂層函式挖出來 exec 進【同一個】命名空間 ⇒ 它們彼此看得見、也看得見我塞的全域。"""
    ns = {"torch": torch}
    got = {n.name: n for n in ast.walk(TREE)
           if isinstance(n, ast.FunctionDef) and n.name in names}
    for nm in names:
        assert nm in got, f"⛔ 主線裡找不到 {nm}() —— 是不是被改名了"
        mod = ast.Module(body=[got[nm]], type_ignores=[])
        exec(compile(ast.fix_missing_locations(mod), str(ROLLOUT), "exec"), ns)
    return ns


# ═══════════════════════════════════════════════════════════════════
# #0  env 預設：float 字面值 0.0（⛔ 對不齊就等於「env 缺席時不是關的」）
# ═══════════════════════════════════════════════════════════════════
print("#0  LACOT_INTENT_GUID_W 的 env 預設")
assert "LACOT_INTENT_GUID_W" not in os.environ, "⛔ 這支 smoke 要在 env 缺席下跑"
_envd = {}
for n in ast.walk(TREE):
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get" and len(n.args) == 2):
        try:
            _envd[ast.literal_eval(n.args[0])] = ast.literal_eval(n.args[1])
        except Exception:
            pass
_d = _envd.get("LACOT_INTENT_GUID_W", "⛔缺")
if _d != 0.0 or not isinstance(_d, float):
    bad(f"#0 env 預設要是 float 字面值 0.0（照 repo 慣例），實際 {_d!r}")
else:
    print(f"    預設 {_d!r}（float）✓ ⇒ env 缺席＝完全關閉")
# ⛔ 診斷旋鈕不進檔名，但一定要進 json ⇒ 不然收表分不出哪個 w
_out = [n for n in ast.walk(TREE)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "out"]
if "guid_w" not in {k.arg for k in _out[0].value.keywords}:
    bad("#0 out=dict(...) 少了 guid_w ⇒ 從 json 讀不出這輪的引導強度")
else:
    print("    out=dict(...) 有 guid_w ✓（⛔ 不進檔名 ⇒ 產物請用 OUT_DIR 分目錄）")
# ⛔ Permutation 的 flip 只准交給 Flow 自己處理
_gsrc = ast.get_source_segment(SRC, [n for n in ast.walk(TREE)
                                     if isinstance(n, ast.FunctionDef)
                                     and n.name == "_flow_sample_guided"][0])
if "_cond_for_block" not in _gsrc or "torch.flip" in _gsrc:
    bad("#0 引導取樣必須用 flow._cond_for_block 同步 Permutation，⛔ 不准自己 flip cond")
else:
    print("    flip 同步走 flow._cond_for_block ✓（⛔ 沒有自己 flip）")

# ═══════════════════════════════════════════════════════════════════
# 假但真：一顆真的 Flow ＋ 照抄 embed 接法語義的 condvec
# ═══════════════════════════════════════════════════════════════════
torch.manual_seed(20260905)
COND, IXD, K, D = 32, 8, 6, 4
flow = Flow(token_dim=D, seq_len=K, n_blocks=4, d_hidden=32, n_layers=2, n_heads=4, cond_dim=COND)
# ⚠️ 必須打破 zero-init：to_params 全零時 block 是恆等映射 ⇒ cond 進不到輸出，
#    ①②③ 會變成 0=0 的假過（smoke_nf_pertoken 9/4 踩過同一個坑）。
with torch.no_grad():
    for b in flow.blocks:
        b.to_params.weight.normal_(0, 0.05)
        b.to_params.bias.normal_(0, 0.02)
flow.eval()
device = torch.device("cpu")

_enc = nn.Linear(2, 16)
_head = nn.Linear(32 + IXD, COND)


def condvec(s, g, ix=None):
    """⛔ 語義逐行照抄主線 condvec 的 embed 分支：ix=None ⇒ 尾巴拼零。"""
    x = torch.cat([_enc(s), _enc(g)], 1)
    if ix is None:
        ix = x.new_zeros(x.shape[0], IXD)
    elif ix.shape[0] == 1 and x.shape[0] != 1:
        ix = ix.expand(x.shape[0], -1)
    return _head(torch.cat([x, ix], 1))


def flow_cond(cv, anchors_t=None):
    return cv                       # embed 接法 ⇒ 恆等（主線 INTENT != "anchor" 那一支）


ns = load_funcs("_block_inverse_guided", "_flow_sample_guided", "sample_plan")
ns.update(flow=flow, device=device, condvec=condvec, flow_cond=flow_cond,
          _GUID_A_CAP=60.0, INTENT_GUID_W=0.0)
sample_plan = ns["sample_plan"]

s = torch.randn(1, 2)
g = torch.randn(1, 2)
ix = torch.randn(1, IXD) * 1.5                       # 「有路線」的 intent 尾巴
cond_int = condvec(s, g, ix)
cond_zero = condvec(s, g, None)
assert not torch.allclose(cond_int, cond_zero), \
    "⛔ 帶 intent 與拼零的 cond 一樣 ⇒ 引導方向恆為零，②③ 都會變假過"
print(f"    對照組健全：‖cond_int − cond_zero‖ = {(cond_int - cond_zero).norm():.4f} > 0 ✓")


def draw(w, n=3, cond=None, seed=777):
    """同 seed 同輸入 ⇒ z 完全一樣，差別只在每步的 (mu, alpha)。"""
    ns["INTENT_GUID_W"] = w
    torch.manual_seed(seed)
    return sample_plan(n, (cond_int if cond is None else cond).expand(n, -1), None, s, g)


# ═══════════════════════════════════════════════════════════════════
# ①  GUID_W=0 ⇒ 與未開 guidance 的路徑逐位元相同
# ═══════════════════════════════════════════════════════════════════
print("\n#1  GUID_W=0／env 缺席 ⇒ 逐位元不變")
torch.manual_seed(777)
u_ref = flow.sample(3, cond_int.expand(3, -1))       # ⛔ 這就是「沒有這次改動」時跑的那一行
u_off = draw(0.0)
if not torch.equal(u_ref, u_off):
    bad(f"#1 GUID_W=0 不是逐位元相同（max diff {(u_ref-u_off).abs().max():.3e}）")
else:
    print("    flow.sample(...) vs sample_plan(w=0)：torch.equal ✓ 逐位元相同")
# ⭐ 更強的一格：w=0 時連 (s,g) 都不該被碰 ⇒ 傳 None 也要能跑（證明 cond0 根本沒算）
ns["INTENT_GUID_W"] = 0.0
torch.manual_seed(777)
u_none = sample_plan(3, cond_int.expand(3, -1), None, None, None)
if not torch.equal(u_ref, u_none):
    bad("#1 w=0 時 s/g=None 走不通 ⇒ 關掉的路徑仍在做多餘的事")
else:
    print("    w=0 且 s=g=None 照樣逐位元相同 ✓（⛔ 關掉＝一行都沒多做）")

# ═══════════════════════════════════════════════════════════════════
# ②  w=1 ⇒ 回到正常帶 intent 取樣
# ═══════════════════════════════════════════════════════════════════
print("\n#2  w=1 ⇒ v = v_int")
u_w1 = draw(1.0)
err = float((u_ref - u_w1).abs().max())
if not torch.allclose(u_ref, u_w1, atol=1e-5):
    bad(f"#2 w=1 沒回到正常取樣（max err {err:.3e} > 1e-5）")
else:
    print(f"    max err {err:.3e} ≤ 1e-5 ✓" + ("（重排式 ⇒ 逐位元相同）" if err == 0.0 else ""))
# ⭐ 引導方向為零（兩份 cond 相同）⇒ 任何 w 都要退回正常取樣
u_flat = draw(3.0, cond=cond_int)                    # cond_int == cond_int ⇒ 差為零
ns["INTENT_GUID_W"] = 3.0
torch.manual_seed(777)
_saved, ns["condvec"] = ns["condvec"], (lambda s_, g_, ix_=None: cond_int)   # 兩邊都給同一份
u_flat = sample_plan(3, cond_int.expand(3, -1), None, s, g)
ns["condvec"] = _saved
if not torch.allclose(u_ref, u_flat, atol=1e-5):
    bad(f"#2 cond_int==cond_zero 時 w=3 應退回正常取樣，實際 max err {(u_ref-u_flat).abs().max():.3e}")
else:
    print("    cond_zero≡cond_int 時 w=3 仍等於正常取樣 ✓（外插的是【差】）")

# ═══════════════════════════════════════════════════════════════════
# ③  w=2 ⇒ 真的改變、且有限
# ═══════════════════════════════════════════════════════════════════
print("\n#3  w=2 ⇒ 非恆等且有限")
u_w2 = draw(2.0)
d2 = float((u_w2 - u_ref).abs().max())
if torch.allclose(u_w2, u_ref, atol=1e-6):
    bad("#3 w=2 跟正常取樣一樣 ⇒ 引導沒接上（恆等）")
elif not torch.isfinite(u_w2).all():
    bad("#3 w=2 出現 inf/nan")
else:
    print(f"    max |Δ| vs w=1 路徑 = {d2:.4f} ≠ 0、全 finite ✓")
# ⭐ 單調性（弱檢查）：w 越大離「帶 intent」越遠 —— 外插方向對不對的唯一便宜證據
d15 = float((draw(1.5) - u_ref).abs().max())
if not (d15 < d2):
    bad(f"#3 w=1.5 的偏移 {d15:.4f} 沒有小於 w=2 的 {d2:.4f} ⇒ 外插方向可疑")
else:
    print(f"    w=1.5 偏移 {d15:.4f} < w=2 偏移 {d2:.4f} ✓（沿同一方向放大）")
# ⭐ 反向（w=0.5，往 zero 側內插）也要動且有限 —— 證明 w 是連續旋鈕不是開關
u_h = draw(0.5)
if torch.allclose(u_h, u_ref, atol=1e-6) or not torch.isfinite(u_h).all():
    bad("#3 w=0.5（往零 intent 側內插）沒有作用或不有限")
else:
    print(f"    w=0.5 也動了（max |Δ| {float((u_h-u_ref).abs().max()):.4f}）且 finite ✓")

print(f"\n{'ALL PASS (3/3 + 護欄全綠)' if not fails else '🚨 FAIL ' + str(len(fails))}")
sys.exit(1 if fails else 0)
