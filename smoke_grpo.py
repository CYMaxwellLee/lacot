"""smoke：GRPO-on-thoughts rung 1 patch（LACOT_GRPO_*；設計卡 docs/DESIGN-2026-09-06-grpo-thoughts.md）。

⭐ 四格，全部【真的跑】或【AST 挖主線原始碼 exec】，⛔ 不抄一份會分岔的公式：
  #0 golden：HEAD 原版 vs patch 後 env 缺席 vs patch 後 GRPO_W=0 —— 三顆 ckpt 逐位元同
     （sha256＋逐 tensor 雙保險）、檔名零變化（無 _grpo 段）。
  #1 GRPO_W=1 小跑 20 步：不炸、檔名帶 _grpo1、grpolog 行數對、l_grpo 全有限；
     β 定標事件在 ⇒ ‖∇L_GRPO‖>0 且 flow 權重相對 golden 有動
     （data RNG 流兩臂成對 ⇒ 任何權重差＝GRPO 梯度；全退化群則權重必須逐位元同——兩種都判）。
  #2 AST 挖 _stage2_loop 的 `if GRPO_W > 0` 段 exec（受控 stub、⛔ 非退化 reward）：
     真 Flow 上 logπ 對參數的梯度 norm>0、advantage／L_GRPO 跟手算逐位對齊（含 /(K·d) 與
     group reshape 方向）、total 真的被加了 GRPO_W·β·L_GRPO。
  #3 reward 單測：挖主線 _grpo_reward＋格 helper、灌合成走廊 —— 好計畫拿高分；
     「塌短刷分」（legality=1 的常數計畫）被 N(z) arclen 閘掉＝0 分；teleport 被鄰步閘掉＝0 分。
  #4 既有 smoke_rollout_fixes ＋ smoke_nf_pertoken 全綠（golden 鎖沒被本 patch 打破）。

    OGBENCH_DATA_DIR=$HOME/data/ogbench $HOME/venvs/lacot-rocm/bin/python smoke_grpo.py
"""
import ast
import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
ROLLOUT = ROOT / "experiments" / "scratch_lacot_rollout.py"
PY = os.environ.get("LACOT_SMOKE_PY", f"{os.path.expanduser('~')}/venvs/lacot-rocm/bin/python")
DATA = os.environ.get("OGBENCH_DATA_DIR", f"{os.path.expanduser('~')}/data/ogbench")
fails = []


def bad(msg):
    fails.append(msg)
    print(f"       🚨 {msg}")


# ═══ 共用：跑一次主線（CPU、最小規模；BASE 抄 smoke_cont_div 的已驗配方、STEPS2=20）══════
BASE = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa",
    "OGBENCH_DATA_DIR": DATA,
    "LACOT_ENV": "pointmaze-medium-stitch-v0", "LACOT_K": "4", "LACOT_TCAP": "32",
    "LACOT_STEPS1": "5", "LACOT_STEPS2": "20",
    "LACOT_ENC_OBJ": "recon_ictr", "LACOT_LEARNED_REFINE": "0", "LACOT_BC_INDEP": "1",
    "LACOT_COND_DROP": "0.1", "LACOT_INTENT": "embed", "LACOT_INTENT_DROP": "0.3",
    "LACOT_EMA_W": "0.999",
    # ⛔ eval 縮到最小：這支驗的是訓練分支的管線，⛔ 不是成功率
    "LACOT_EVAL_MAXH": "5", "LACOT_EVAL_EPISODES": "1", "LACOT_EVAL_RS": "0",
    "LACOT_DEV_EVAL": "0",
}


def run(script, outdir, extra, label):
    env = {**os.environ, **BASE, "LACOT_OUT_DIR": str(outdir), **extra}
    p = subprocess.run([PY, "-u", str(script)], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        bad(f"{label}: returncode={p.returncode}\n"
            + "\n".join(p.stdout.splitlines()[-12:]) + "\n" + "\n".join(p.stderr.splitlines()[-12:]))
    return p


def files(d, pat):
    return sorted(x.name for x in pathlib.Path(d).glob(pat))


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def tree_of(path):
    return ast.parse(path.read_text())


def func_node(path, name):
    for n in ast.walk(tree_of(path)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


TMP = pathlib.Path(tempfile.mkdtemp(prefix="smoke_grpo_"))
print(f"⚙️  暫存目錄 {TMP}\n")

# ═══════════════════════════════════════════════════════════════════
# #0  golden：HEAD 原版 vs patch 後（env 缺席／GRPO_W=0 明寫）⇒ ckpt 逐位元同
# ═══════════════════════════════════════════════════════════════════
print("#0  golden：訓練預設路徑零行為差（HEAD 原版當對照、⛔ 不是 patch 自己對自己）")
HEAD_COPY = ROOT / "experiments" / ".smoke_grpo_head_baseline.py"
_head_src = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:experiments/scratch_lacot_rollout.py"],
                           capture_output=True, text=True)
assert _head_src.returncode == 0, "⛔ 拿不到 HEAD 版主檔（git show 失敗）"
HEAD_COPY.write_text(_head_src.stdout)
try:
    H = TMP / "head"; A = TMP / "absent"; B = TMP / "w0"
    for d in (H, A, B):
        d.mkdir()
    run(HEAD_COPY, H, {}, "#0 HEAD 原版")
    pA = run(ROLLOUT, A, {}, "#0 patch 後 env 缺席")
    run(ROLLOUT, B, {"LACOT_GRPO_W": "0.0", "LACOT_GRPO_G": "8", "LACOT_GRPO_EVERY": "1",
                     "LACOT_GRPO_BQ": "4"}, "#0 patch 後 GRPO_W=0 明寫")
finally:
    HEAD_COPY.unlink(missing_ok=True)
ckH, ckA, ckB = files(H, "ckpt_*.pt"), files(A, "ckpt_*.pt"), files(B, "ckpt_*.pt")
if not (len(ckH) == len(ckA) == len(ckB) == 1):
    bad(f"#0 三臂沒各產出唯一 ckpt：{ckH} {ckA} {ckB}")
    print("\n🚨 golden 基座就掛了，後面沒意義"); sys.exit(1)
print(f"    ckpt  {ckH[0]}")
if not (ckH[0] == ckA[0] == ckB[0]):
    bad(f"#0 檔名變了：HEAD {ckH[0]} / absent {ckA[0]} / w0 {ckB[0]} ⇒ ⛔ 舊索引會被打斷")
if "_grpo" in ckA[0]:
    bad(f"#0 預設檔名帶了 _grpo 段：{ckA[0]}")
hH, hA, hB = sha(H / ckH[0]), sha(A / ckA[0]), sha(B / ckB[0])
print(f"    sha256  HEAD   {hH[:16]}…\n            absent {hA[:16]}…\n            w0     {hB[:16]}…")
if hH == hA == hB:
    print("    ⇒ 三顆 ckpt sha256 逐位元同 ✓（GRPO_W=0/缺席 ⇒ 連一個 op 都不進 graph）")
else:
    # 雙保險：sha 不同時逐 tensor 對（zip 中繼資料 vs 真權重差要分開講）
    SH = torch.load(H / ckH[0], map_location="cpu", weights_only=False)
    for name, other in (("absent", A / ckA[0]), ("w0", B / ckB[0])):
        so = torch.load(other, map_location="cpu", weights_only=False)
        dif = [k for k in ("flow", "cond_head", "cond_enc", "ahead", "u_dec")
               if any(not torch.equal(SH[k][t], so[k][t]) for t in SH[k])]
        bad(f"#0 sha256 不同（{name}），逐 tensor 差異模組：{dif or '無（⚠️ 只差中繼資料？）'}")
# 預設臂 stdout 不該有任何 grpo 字樣（⛔ gate 住＝連 print 都不出現）
if "grpo" in pA.stdout.lower():
    bad("#0 env 缺席的 stdout 出現 grpo 字樣 ⇒ gate 沒關乾淨")
else:
    print("    ⇒ 預設臂 stdout 零 grpo 字樣 ✓（整段沒跑）")

# ═══════════════════════════════════════════════════════════════════
# #1  GRPO_W=1 小跑：不炸、grpolog、l_grpo 有限、梯度真的到了權重
# ═══════════════════════════════════════════════════════════════════
print("\n#1  GRPO_W=1（warm=0）小跑 20 步")
G = TMP / "grpo"
G.mkdir()
p1 = run(ROLLOUT, G, {"LACOT_GRPO_W": "1", "LACOT_GRPO_WARM": "0"}, "#1 GRPO 臂")
ckG = files(G, "ckpt_*.pt")
gl = files(G, "grpolog_*.jsonl")
print(f"    ckpt   {ckG}")
print(f"    grpolog {gl}")
if not ckG or "_grpo1" not in ckG[0]:
    bad(f"#1 檔名沒帶 _grpo1：{ckG} ⇒ ⛔ 會蓋掉純 FM 對照那顆")
if "⭐ GRPO：W=1" not in p1.stdout:
    bad("#1 啟動時沒印 GRPO 設定行")
if not gl:
    bad("#1 沒產出 grpolog jsonl")
else:
    rows = [json.loads(x) for x in (G / gl[0]).read_text().splitlines() if x.strip()]
    data_rows = [r for r in rows if "event" not in r]
    ev_calib = [r for r in rows if r.get("event") == "calib"]
    print(f"    {len(rows)} 筆（資料 {len(data_rows)}、calib {len(ev_calib)}、"
          f"halve {sum(1 for r in rows if r.get('event') == 'halve')}）")
    if len(data_rows) != 20:
        bad(f"#1 EVERY=1×20 步應有 20 筆資料列，實際 {len(data_rows)}")
    if not all(math.isfinite(r["l_grpo"]) for r in data_rows):
        bad("#1 l_grpo 出現非有限值")
    if not all(0.0 <= r["gate_rate"] <= 1.0 and 0.0 <= r["degen_frac"] <= 1.0 for r in data_rows):
        bad("#1 gate_rate／degen_frac 越界")
    rmeans = [round(r["r_mean"], 3) for r in data_rows]
    print(f"    r_mean 前 5 步 {rmeans[:5]} … 末 3 步 {rmeans[-3:]}   "
          f"gate_rate 均值 {np.mean([r['gate_rate'] for r in data_rows]):.2f}   "
          f"degen_frac 均值 {np.mean([r['degen_frac'] for r in data_rows]):.2f}")
    if gl[0][len("grpolog_"):-len(".jsonl")] + ".pt" != ckG[0][len("ckpt_"):]:
        bad("#1 grpolog 的 tag 跟 ckpt 對不上 ⇒ 收表配不起來")
    # 梯度到權重了嗎：data RNG 兩臂成對 ⇒ flow/cond_head 任何差＝GRPO 梯度（誠實分兩枝判）
    Wg = torch.load(G / ckG[0], map_location="cpu", weights_only=False)
    W0 = torch.load(A / ckA[0], map_location="cpu", weights_only=False)
    n_diff = sum(0 if torch.equal(Wg["flow"][k], W0["flow"][k]) else 1 for k in W0["flow"])
    if ev_calib:
        c = ev_calib[0]
        print(f"    β 定標：‖∇L_FM‖={c['gn_fm']:.3e} ‖∇L_GRPO‖={c['gn_grpo']:.3e} "
              f"β_target={c['beta_target']:.3e}")
        if not (c["gn_grpo"] > 0):
            bad("#1 calib 的 ‖∇L_GRPO‖=0 ⇒ logπ 對參數沒有梯度")
        print(f"    flow 權重 vs golden：{n_diff} 個 tensor 不同"
              f"   {'✓ GRPO 梯度真的推到了 flow' if n_diff > 0 else '🚨'}")
        if n_diff == 0:
            bad("#1 有 calib（β>0、群非退化）但 flow 權重跟純 FM 逐位元同 ⇒ L_GRPO 沒接進 backward")
    else:
        print("    ⚠️ 20 步內全退化群（β 一直定不了標）⇒ 這輪【必須】跟純 FM 逐位元同")
        if n_diff != 0:
            bad("#1 全退化群（Â≡0）卻動了 flow 權重 ⇒ 有別的東西漏進 loss")
        print("    （logπ 梯度的存在性由 #2 的受控 stub 證，⛔ 不靠這一枝）")

# ═══════════════════════════════════════════════════════════════════
# #2  AST 挖主線 loop 的 GRPO 段 exec：logπ 梯度／advantage／L_GRPO 手算對齊
# ═══════════════════════════════════════════════════════════════════
print("\n#2  主線 GRPO 段（AST 原樣 exec、受控非退化 reward）：形狀與手算對")
sys.path.insert(0, str(ROOT))
from lacot.nf_head import Flow                                    # noqa: E402

_loop = func_node(ROLLOUT, "_stage2_loop")
_cand = [n for n in ast.walk(_loop) if isinstance(n, ast.If) and ast.unparse(n.test) == "GRPO_W > 0"]
if len(_cand) != 1:
    bad(f"#2 _stage2_loop 裡 `if GRPO_W > 0` 有 {len(_cand)} 段（期望 1）")
if _cand:
    _blk = _cand[0]
    torch.manual_seed(0)
    _fl = Flow(token_dim=6, seq_len=3, n_blocks=2, cond_dim=16)
    with torch.no_grad():                                          # 打破 zero-init（nf smoke 的教訓）
        for b in _fl.blocks:
            b.to_params.weight.normal_(0, 0.05); b.to_params.bias.normal_(0, 0.02)
    _lin = torch.nn.Linear(4, 16)
    _rr = np.random.default_rng(7)
    _obs = np.stack([np.linspace(0, 5, 60), np.zeros(60)], 1).astype(np.float32)
    _u0 = torch.randn(2, 3, 6)
    _c0 = _lin(torch.cat([torch.zeros(2, 2), torch.ones(2, 2)], 1))
    DIMS = 18
    ns = dict(torch=torch, np=np, device="cpu", print=print, flow=_fl,
              condvec=lambda s, g, ix=None: _lin(torch.cat([s, g], 1)),
              flow_cond=lambda cv, a=None: cv, intent_ad=None, INTENT="",
              _q=lambda u: u, _dec=lambda u, s: u.reshape(len(u), -1)[:, :16].reshape(-1, 8, 2),
              _intent_inv=lambda p, a: p,
              _grpo_reward=lambda pts, s, g: (float(_rr.random()), 1.0),   # ⭐ 受控：必非退化
              OBS=_obs, traj_end=np.full(60, 59, np.int64), N=60, CHUNK=4,
              mu=np.zeros(2, np.float32), sd=np.ones(2, np.float32),
              GRPO_W=1.0, GRPO_G=4, GRPO_EVERY=1, GRPO_BQ=3, GRPO_WARM=0, GRPO_STDNORM=1,
              GRPO_RATIO=0.2, _GRPO_RATIO=0.2, DIM=DIMS, fsq=None, vq=None,
              _GRPO_TGEN=torch.Generator().manual_seed(3), _GRPO_QRNG=np.random.default_rng(4),
              _GRPO_ST={"t": 0, "beta": None, "ref": None, "ema": None, "cool": 0,
                        "fired": 0, "degen": 0, "groups": 0, "noroute": 0, "gate": 0.0},
              _GRPOLOG=[], _gstp=0, l_nf=_fl.nll(_u0, _c0) / DIMS, total=torch.zeros(()))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[_blk], type_ignores=[])),
                 "<grpo-block>", "exec"), ns)
    lg = ns["l_grpo"]
    print(f"    l_grpo={lg.item():+.5f}（有限 {'✓' if math.isfinite(lg.item()) else '🚨'}）"
          f"  β={ns['_GRPO_ST']['beta']:.3e}  fired={ns['_GRPO_ST']['fired']}")
    if not math.isfinite(lg.item()):
        bad("#2 l_grpo 非有限")
    if ns["_GRPO_ST"]["beta"] is None:
        bad("#2 受控非退化 reward 下 β 竟然沒定標")
    # 手算對齊：advantage（stdnorm）與 −mean(Â·logπ)/(K·d)，⛔ 用 ns 裡主線自己收的原料重算
    _rw = ns["_grw"].reshape(3, 4)
    _adv = (_rw - _rw.mean(1, keepdims=True)) / (_rw.std(1, keepdims=True) + 1e-6)
    _want = float(-(torch.tensor(_adv.reshape(-1), dtype=torch.float32)
                    * ns["_glogp"].detach()).mean() / DIMS)
    print(f"    手算 L_GRPO={_want:+.5f} vs 主線 {lg.item():+.5f}"
          f"   {'✓ 對齊（含 /(K·d) 與 reshape 方向）' if abs(_want - lg.item()) < 1e-6 else '🚨'}")
    if abs(_want - lg.item()) > 1e-6:
        bad(f"#2 L_GRPO 跟手算不合：主線 {lg.item()} vs 手算 {_want}")
    _texp = 1.0 * ns["_GRPO_ST"]["beta"] * 1.0 * lg.item()        # GRPO_W·β·bfrac(=1)·l_grpo
    if abs(ns["total"].item() - _texp) > 1e-6 * max(1.0, abs(_texp)):
        bad(f"#2 total 增量不對：{ns['total'].item()} vs 期望 {_texp}")
    print(f"    total={ns['total'].item():+.5f}＝GRPO_W·β·L_GRPO ✓")
    ns["total"].backward()
    gn = sum(float(p.grad.pow(2).sum()) for p in _fl.parameters() if p.grad is not None) ** 0.5
    print(f"    backward 後 flow 參數 grad norm {gn:.3e}   {'✓ logπ 對參數有梯度' if gn > 0 else '🚨'}")
    if not (gn > 0):
        bad("#2 flow 參數 grad norm=0 ⇒ logπ 沒把梯度帶回 policy")
    if any(bool(torch.isnan(p.grad).any()) for p in _fl.parameters() if p.grad is not None):
        bad("#2 grad 出現 NaN")

# ═══════════════════════════════════════════════════════════════════
# #3  reward 單測：挖主線 _grpo_reward、合成走廊 —— 刷分樣本被 N(z) 閘掉
# ═══════════════════════════════════════════════════════════════════
print("\n#3  reward（主線 _grpo_reward 原樣 exec、合成走廊佔據圖）")
from lacot.subgoal import grid_bfs as _real_bfs                    # noqa: E402（單一來源、⛔ 不另寫 BFS）

_occ = np.zeros((3, 11), bool); _occ[1, :] = True                  # 走廊：row1 全自由、上下是牆
_rns = dict(np=np, _G_LO=np.zeros(2), _G_SPAN=np.array([2.0, 10.0]),
            _G_SHAPE=np.array([3, 11]), _GOCC=_occ, _GFREE=np.argwhere(_occ),
            _G_CELL=1.0, _GRPO_DMAPS={}, _grpo_bfs=_real_bfs, _GRPO_RHO=0.5, _GRPO_DSTEP=1.25)
for _fn in ("_g_cells_of", "_g_cell_snap", "_g_dist_from", "_grpo_reward"):
    n = func_node(ROLLOUT, _fn)
    if n is None:
        bad(f"#3 主線找不到 {_fn}"); break
    exec(compile(ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])), "<rw>", "exec"), _rns)
else:
    R = _rns["_grpo_reward"]
    s0, g0 = np.array([1.0, 1.0]), np.array([1.0, 9.0])            # d_s=8 格
    good = np.stack([np.full(17, 1.0), np.linspace(1, 9, 17)], 1)  # 沿走廊直走（步距 0.5）
    lazy = np.tile(s0, (17, 1))                                    # 塌短刷分：貼在起點自由格
    tele = np.vstack([np.tile(s0, (16, 1)), g0])                   # teleport：末點瞬移貼 goal
    wallr = np.vstack([np.stack([np.zeros(16), np.linspace(1, 9, 16)], 1), g0[None]])  # 沿牆爬
    r_good, g_good = R(good, s0, g0)
    r_lazy, g_lazy = R(lazy, s0, g0)
    r_tele, g_tele = R(tele, s0, g0)
    r_wall, g_wall = R(wallr, s0, g0)
    print(f"    好計畫（走廊直走）      r={r_good:.3f} gate={g_good:g}   期望 r=1、gate=1")
    print(f"    塌短刷分（legality=1）  r={r_lazy:.3f} gate={g_lazy:g}   期望 0、0（arclen 閘）")
    print(f"    teleport（末點瞬移）    r={r_tele:.3f} gate={g_tele:g}   期望 0、0（鄰步閘）")
    print(f"    沿牆爬（中段全在牆）    r={r_wall:.3f} gate={g_wall:g}   期望 gate=1 但 r 明顯低")
    if not (g_good == 1.0 and abs(r_good - 1.0) < 1e-9):
        bad(f"#3 好計畫沒拿滿分：r={r_good} gate={g_good}")
    if not (r_lazy == 0.0 and g_lazy == 0.0):
        bad(f"#3 塌短刷分沒被 N(z) 閘掉：r={r_lazy} gate={g_lazy}（⛔ N3 的平凡極大解又回來了）")
    if not (r_tele == 0.0 and g_tele == 0.0):
        bad(f"#3 teleport 沒被鄰步閘掉：r={r_tele} gate={g_tele}")
    if not (g_wall == 1.0 and r_wall < r_good - 0.15):
        bad(f"#3 沿牆爬應 gate=1 但 r 顯著低於好計畫：r={r_wall} gate={g_wall}")
    # 退化群語義順手驗：全同 reward ⇒ std=0 ⇒（主線判退化）
    if float(np.std([R(lazy, s0, g0)[0] for _ in range(4)])) != 0.0:
        bad("#3 同一條計畫重複打分竟然不同 ⇒ reward 不是確定性的")

# ═══════════════════════════════════════════════════════════════════
# #4  既有 golden smoke 全綠
# ═══════════════════════════════════════════════════════════════════
print("\n#4  既有 smoke：rollout_fixes ＋ nf_pertoken")
for name in ("smoke_rollout_fixes.py", "smoke_nf_pertoken.py"):
    p = subprocess.run([PY, "-u", str(ROOT / name)], capture_output=True, text=True,
                       env={**os.environ, "OGBENCH_DATA_DIR": DATA})
    last = (p.stdout.strip().splitlines() or ["<no output>"])[-1]
    print(f"    {name:<26} rc={p.returncode}  尾行：{last}")
    if p.returncode != 0 or ("PASS" not in last):
        bad(f"#4 {name} 沒過（rc={p.returncode}）\n" + "\n".join(p.stdout.splitlines()[-8:])
            + "\n" + "\n".join(p.stderr.splitlines()[-8:]))

print()
shutil.rmtree(TMP, ignore_errors=True)
if fails:
    print(f"🚨 FAIL {len(fails)} 項")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("✅ ALL PASS")
