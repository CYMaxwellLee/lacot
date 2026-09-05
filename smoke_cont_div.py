"""smoke：Patch A（LACOT_CONT_TRAIN 續訓 ＋ LACOT_DIV_LOG_EVERY 散度時間序列）
　　　　　Patch B（LACOT_DIV_W／LACOT_DIV_M 的 L_div hinge）。

⭐ 這支【真的把主線跑起來】（CPU、pointmaze-medium-stitch、K=4、T_CAP=32、每格幾秒），
   ⛔ 不是字串比對 —— 「檔名有沒有變」「有沒有蓋掉來源」這種事只有真的跑一次才問得出來。
⭐ 梯度那一格用 AST 把主線【那幾行 hinge】原樣挖出來 exec ⇒ 驗的是真的跑在主線上的那一份，
   ⛔ 不是這支 smoke 自己抄一份會分岔的公式。

    OGBENCH_DATA_DIR=$HOME/data/ogbench $HOME/venvs/lacot-rocm/bin/python smoke_cont_div.py
"""
import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parent
ROLLOUT = ROOT / "experiments" / "scratch_lacot_rollout.py"
PY = os.environ.get("LACOT_SMOKE_PY", f"{os.path.expanduser('~')}/venvs/lacot-rocm/bin/python")
DATA = os.environ.get("OGBENCH_DATA_DIR", f"{os.path.expanduser('~')}/data/ogbench")
fails = []


def bad(msg):
    fails.append(msg)
    print(f"       🚨 {msg}")


# ═══ 共用：跑一次主線（CPU、最小規模）═══════════════════════════════════════
BASE = {
    "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa",
    "OGBENCH_DATA_DIR": DATA,
    "LACOT_ENV": "pointmaze-medium-stitch-v0", "LACOT_K": "4", "LACOT_TCAP": "32",
    "LACOT_STEPS1": "5", "LACOT_STEPS2": "10",
    "LACOT_ENC_OBJ": "recon_ictr", "LACOT_LEARNED_REFINE": "0", "LACOT_BC_INDEP": "1",
    "LACOT_COND_DROP": "0.1", "LACOT_INTENT": "embed", "LACOT_INTENT_DROP": "0.3",
    "LACOT_EMA_W": "0.999",
    # ⛔ eval 縮到最小：這支驗的是續訓／散度的管線，⛔ 不是成功率
    "LACOT_EVAL_MAXH": "5", "LACOT_EVAL_EPISODES": "1", "LACOT_EVAL_RS": "0",
    "LACOT_DEV_EVAL": "0",
}


def run(outdir, extra, label, expect_fail=False):
    env = {**os.environ, **BASE, "LACOT_OUT_DIR": str(outdir), **extra}
    p = subprocess.run([PY, "-u", str(ROLLOUT)], env=env, capture_output=True, text=True)
    okness = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    if not okness:
        bad(f"{label}: returncode={p.returncode}（期望{'非零' if expect_fail else '0'}）\n"
            + "\n".join(p.stdout.splitlines()[-12:]) + "\n" + "\n".join(p.stderr.splitlines()[-12:]))
    return p


def files(d, pat):
    return sorted(x.name for x in pathlib.Path(d).glob(pat))


TMP = pathlib.Path(tempfile.mkdtemp(prefix="smoke_cont_div_"))
print(f"⚙️  暫存目錄 {TMP}\n")

# ═══════════════════════════════════════════════════════════════════
# #0  baseline：兩個 env 都缺席 ⇒ 產一顆小 ckpt 當續訓的起點
# ═══════════════════════════════════════════════════════════════════
print("#0  baseline（兩個新 env 都缺席）—— 順便生出續訓要用的小 ckpt")
B = TMP / "base"
B.mkdir()
run(B, {}, "#0 baseline")
base_ck = files(B, "ckpt_*.pt")
base_js = files(B, "rollout_*.json")
if len(base_ck) != 1 or len(base_js) != 1:
    bad(f"#0 baseline 沒產出唯一的 ckpt/json：{base_ck} {base_js}")
    print("\n🚨 baseline 就掛了，後面全部沒意義")
    sys.exit(1)
CK0 = B / base_ck[0]
SHA0 = torch.load(CK0, map_location="cpu", weights_only=False)
print(f"    ckpt  {base_ck[0]}")
print(f"    json  {base_js[0]}")
if "_ct" in base_ck[0] or "_dvw" in base_ck[0]:
    bad("#0 全預設的檔名竟然帶了 _ct / _dvw ⇒ ⛔ 舊索引會被打斷")
print("    ⇒ 檔名不含 _ct / _dvw ✓（⛔ 預設檔名零變化）")
if "opt2" in SHA0:
    bad("#0 非續訓的 ckpt 竟然存了 optimizer 狀態 ⇒ ⛔ 預設路徑的 ckpt 內容變了")
print("    ⇒ 預設 ckpt 沒有 opt2 段 ✓（optimizer 狀態只在續訓模式存）")

# ═══════════════════════════════════════════════════════════════════
# #1  (a) CONT_TRAIN=1 續訓 20 步：不炸、存【新檔名】、⛔ 不覆蓋來源
# ═══════════════════════════════════════════════════════════════════
print("\n#1  (a) 續訓 20 步 —— 新檔名、⛔ 不覆蓋來源")
mt0 = CK0.stat().st_mtime_ns
sz0 = CK0.stat().st_size
C = TMP / "cont"
C.mkdir()
p1 = run(C, {"LACOT_CONT_TRAIN": "1", "LACOT_STEPS2": "20",
             "LACOT_LOAD_CKPT": str(CK0), "LACOT_SEED": "0"}, "#1 續訓")
cont_ck = files(C, "ckpt_*.pt")
print(f"    續訓存出 {cont_ck}")
if len(cont_ck) != 1:
    bad(f"#1 續訓沒存出唯一 ckpt：{cont_ck}")
elif "_ct20" not in cont_ck[0]:
    bad(f"#1 續訓檔名沒有 _ct20 段：{cont_ck[0]} ⇒ ⛔ 會跟同 seed 的別的產物撞名")
else:
    print("    ⇒ 檔名帶 _ct20 ✓，且 _st0（不是從頭訓的）"
          f" {'✓' if '_st0_' in cont_ck[0] else '🚨'}")
if CK0.stat().st_mtime_ns != mt0 or CK0.stat().st_size != sz0:
    bad("#1 來源 ckpt 的 mtime/size 變了 ⇒ 🚨 被覆蓋了")
print("    ⇒ 來源 ckpt mtime/size 未變 ✓（⛔ 沒被覆蓋）")
if "沒存 optimizer 狀態" not in p1.stdout:
    bad("#1 來源沒有 optimizer 狀態，卻沒印出 fresh Adam 的警告 ⇒ ⛔ 靜默換了訓練條件")
print("    ⇒ 印出「沒存 optimizer ⇒ fresh Adam」警告 ✓")
if "續訓：EMA 影子從 ckpt 的 ema 段接續" not in p1.stdout:
    bad("#1 ckpt 有 ema 段但沒接續影子 ⇒ ⛔ 存出來的影子會停在載入那一刻")
print("    ⇒ EMA 影子從 ckpt 接續 ✓")
if cont_ck:
    _cc = torch.load(C / cont_ck[0], map_location="cpu", weights_only=False)
    for _k in ("opt2", "ema"):
        if _k not in _cc:
            bad(f"#1 續訓存出的 ckpt 少了 {_k} 段")
    if _cc.get("cfg", {}).get("SRC_CKPT") != base_ck[0]:
        bad(f"#1 續訓 ckpt 的 cfg 沒記下來源：{_cc.get('cfg', {}).get('SRC_CKPT')}")
    print(f"    ⇒ 續訓 ckpt 有 opt2 + ema 段、cfg.SRC_CKPT={_cc.get('cfg', {}).get('SRC_CKPT')} ✓")
    # 權重真的動了（⛔ 「跑完不炸」不等於「有訓練」）
    _d = max(float((_cc["cond_head"][k].float() - SHA0["cond_head"][k].float()).abs().max())
             for k in SHA0["cond_head"])
    print(f"    ⇒ cond_head 權重最大變動 {_d:.3e} {'✓ 真的訓了' if _d > 0 else '🚨 一步都沒動'}")
    if _d <= 0:
        bad("#1 續訓後權重完全沒變 ⇒ 那 20 步沒跑到")
# ⛔ 背水線：把續訓產物再續訓同樣步數 ⇒ 會算出同一個檔名 ⇒ 必須【在訓練前】就擋下來
if cont_ck:
    p1b = run(C, {"LACOT_CONT_TRAIN": "1", "LACOT_STEPS2": "20",
                  "LACOT_LOAD_CKPT": str(C / cont_ck[0]), "LACOT_SEED": "0"},
              "#1 撞名守門", expect_fail=True)
    if "已經帶 _ct20" not in (p1b.stdout + p1b.stderr):
        bad("#1 撞名守門沒開火（來源已帶 _ct20 ⇒ 會蓋掉來源）")
    else:
        print("    ⇒ 來源已帶 _ct20 時，訓練【開始前】就 assert 擋下 ✓")

# ═══════════════════════════════════════════════════════════════════
# #2  (b) DIV_LOG_EVERY=5 ⇒ jsonl 行數正確
# ═══════════════════════════════════════════════════════════════════
print("\n#2  (b) DIV_LOG_EVERY=5 的散度時間序列")
D = TMP / "divlog"
D.mkdir()
p2 = run(D, {"LACOT_CONT_TRAIN": "1", "LACOT_STEPS2": "20", "LACOT_LOAD_CKPT": str(CK0),
             "LACOT_SEED": "0", "LACOT_DIV_LOG_EVERY": "5"}, "#2 divlog")
dv = files(D, "divlog_*.jsonl")
print(f"    產出 {dv}")
if len(dv) != 1:
    bad(f"#2 沒產出唯一的 divlog jsonl：{dv}")
else:
    rows = [json.loads(x) for x in (D / dv[0]).read_text().splitlines() if x.strip()]
    steps = [r["step"] for r in rows]
    print(f"    {len(rows)} 行，step={steps}，div_median={[round(r['div_median'], 4) for r in rows]}")
    if len(rows) != 4:                       # 20 步 / 每 5 步 = 4 筆
        bad(f"#2 期望 20/5=4 行，實際 {len(rows)} 行")
    if steps != [5, 10, 15, 20]:
        bad(f"#2 step 欄位期望 [5,10,15,20]，實際 {steps}")
    if not all(set(r) == {"step", "div_median"} for r in rows):
        bad(f"#2 每一行要剛好是 step/div_median 兩欄，實際 {sorted(rows[0])}")
    if not all(r["div_median"] > 0 for r in rows):
        bad("#2 div_median 全 0 ⇒ 這把尺量到的是常數，⛔ 不是散度")
    if dv[0][len("divlog_"):-len(".jsonl")] + ".json" != files(D, "rollout_*.json")[0][len("rollout_"):]:
        bad("#2 divlog 的 tag 跟 rollout json 的 tag 對不上 ⇒ ⛔ 收表時配不起來")
    print("    ⇒ 行數／step／欄位／tag 全對 ✓")
if "divlog step 5" not in p2.stdout:
    bad("#2 訓練當下沒有 print 出 divlog 行 ⇒ ⛔ 中途掛掉就沒有備份")
print("    ⇒ 訓練當下也 print 一行 ✓（中途掛掉時 log 是備份）")
# ⛔ 純觀測：開了 DIV_LOG_EVERY 不可以動到權重
c2 = files(D, "ckpt_*.pt")
if c2 and cont_ck:
    a = torch.load(C / cont_ck[0], map_location="cpu", weights_only=False)["cond_head"]
    b = torch.load(D / c2[0], map_location="cpu", weights_only=False)["cond_head"]
    same = all(torch.equal(a[k], b[k]) for k in a)
    print(f"    ⇒ 開 DIV_LOG_EVERY 後權重逐位元同 #1 {'✓' if same else '🚨'}（純觀測、⛔ 不動 RNG）")
    if not same:
        bad("#2 DIV_LOG_EVERY 改變了訓練結果 ⇒ 它不是純觀測")
if c2 and "_dvw" in c2[0]:
    bad("#2 DIV_LOG_EVERY 進了檔名（診斷旋鈕不該進）")

# ═══════════════════════════════════════════════════════════════════
# #3  (c) DIV_W=0 逐位元同 baseline；DIV_W>0 才進檔名
# ═══════════════════════════════════════════════════════════════════
print("\n#3  (c-1) DIV_W=0.0 明寫 ⇒ 逐位元同 baseline")
Z = TMP / "divw0"
Z.mkdir()
run(Z, {"LACOT_DIV_W": "0.0", "LACOT_DIV_M": "0.3"}, "#3 DIV_W=0")
z_ck = files(Z, "ckpt_*.pt")
if z_ck != base_ck:
    bad(f"#3 DIV_W=0 的檔名跟 baseline 不同：{z_ck} vs {base_ck}")
else:
    zz = torch.load(Z / z_ck[0], map_location="cpu", weights_only=False)
    diffs = [k for k in SHA0["cond_head"] if not torch.equal(SHA0["cond_head"][k], zz["cond_head"][k])]
    nfd = [k for k in SHA0["flow"] if not torch.equal(SHA0["flow"][k], zz["flow"][k])]
    print(f"    檔名同 baseline ✓；cond_head 不同的張量 {len(diffs)} 個、flow {len(nfd)} 個")
    if diffs or nfd:
        bad("#3 DIV_W=0 竟然改到權重 ⇒ ⛔ 預設路徑有行為差")
    else:
        print("    ⇒ 逐位元相同 ✓（DIV_W=0 連一個 op 都不進 graph）")

print("\n#3  (c-2) DIV_W>0：跑得動、進檔名、l_div 有印出來")
W = TMP / "divw"
W.mkdir()
p3 = run(W, {"LACOT_CONT_TRAIN": "1", "LACOT_STEPS2": "20", "LACOT_LOAD_CKPT": str(CK0),
             "LACOT_SEED": "0", "LACOT_DIV_W": "1.0", "LACOT_DIV_LOG_EVERY": "5"}, "#3 DIV_W>0")
w_ck = files(W, "ckpt_*.pt")
print(f"    {w_ck}")
if not w_ck or "_dvw1" not in w_ck[0]:
    bad(f"#3 DIV_W=1.0 沒進檔名：{w_ck} ⇒ ⛔ 會蓋掉同設定沒吃藥的那顆")
else:
    print("    ⇒ 檔名帶 _dvw1 ✓")
if "l_div" not in p3.stdout:
    bad("#3 DIV_W>0 但 log 裡看不到 l_div ⇒ ⛔ 開了藥卻不知道它在做什麼")
else:
    print("    ⇒ log 帶 l_div ✓")
# margin 也要分得開（⛔ 掃 margin 時不可互蓋）
if w_ck and cont_ck and w_ck[0] == cont_ck[0]:
    bad("#3 吃藥與不吃藥的 ckpt 同名")

# ═══════════════════════════════════════════════════════════════════
# #4  (c-3) hinge 的梯度：拿【主線那幾行】原樣 exec ⇒ 對 cond 生成端非零
# ═══════════════════════════════════════════════════════════════════
print("\n#4  (c-3) L_div hinge 的梯度（AST 挖主線原始碼、⛔ 不抄一份會分岔的公式）")
_tree = ast.parse(ROLLOUT.read_text())
_fn = [n for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef) and n.name == "_stage2_loop"]
if not _fn:
    bad("#4 找不到 _stage2_loop（續訓靠它跟主線共用同一份訓練迴圈）")
    _hinge = None
else:
    _cand = [n for n in ast.walk(_fn[0]) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "DIV_W > 0"]
    if len(_cand) != 1:
        bad(f"#4 _stage2_loop 裡 `if DIV_W > 0` 有 {len(_cand)} 段（期望 1）")
        _hinge = None
    else:
        _hinge = _cand[0]
        print("    挖到主線的 hinge 段：\n" + "\n".join(
            "        " + l for l in ast.unparse(_hinge).splitlines()))

if _hinge is not None:
    src_h = ast.unparse(_hinge)
    if "relu" not in src_h:
        bad("#4 主線的 L_div 不是 hinge（找不到 relu）")
    for banned in ("cosine_similarity", "cos_sim"):
        if banned in src_h:
            bad(f"#4 主線的 L_div 用了 {banned} ⇒ ⛔ 塌陷點梯度會消失")
    if "ix_full" not in src_h:
        bad("#4 c_int 沒有用 drop【前】的 ix_full ⇒ 被 drop 的樣本算不到散度")

    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.GELU(), torch.nn.Linear(32, 32))
    head = torch.nn.Sequential(torch.nn.Linear(64 + 8, 32), torch.nn.GELU(), torch.nn.Linear(32, 16))
    ad = torch.nn.Linear(8, 8)                     # 代替 intent adapter 的 cond_global

    def condvec(s, g, ix=None):
        x = torch.cat([enc(s), enc(g)], 1)
        if ix is None:
            ix = x.new_zeros(x.shape[0], 8)
        return head(torch.cat([x, ix], 1))

    def hinge(ix_raw, div_m, div_w=1.0):
        ns = dict(torch=torch, condvec=condvec, DIV_W=div_w, DIV_M=div_m,
                  s=S, g=G, ix_full=ad(ix_raw), total=torch.zeros(()))
        exec(compile(ast.Module(body=[_hinge], type_ignores=[]), "<hinge>", "exec"), ns)
        return ns["l_div"], ns["total"]

    S = torch.randn(16, 2)
    G = torch.randn(16, 2)

    # ① 一般情形：l_div>0、對 cond_enc/cond_head/adapter 三邊都要有非零梯度
    for m in (enc, head, ad):
        m.zero_grad(set_to_none=True)
    l, tot = hinge(torch.randn(16, 8), 0.3)
    tot.backward()
    gn = {n: sum(float(p.grad.pow(2).sum()) for p in m.parameters() if p.grad is not None) ** 0.5
          for n, m in (("cond_enc", enc), ("cond_head", head), ("intent_ad", ad))}
    print(f"    l_div={l.item():.4f}   grad norm {  {k: round(v, 5) for k, v in gn.items()} }")
    if l.item() <= 0:
        bad("#4 隨機初始化下 l_div 應該>0（散度遠小於 margin 0.3）")
    for k, v in gn.items():
        if not (v > 0):
            bad(f"#4 {k} 的 grad norm = {v} ⇒ ⛔ 藥推不到 cond 生成端")
    if any(v != v for v in gn.values()):
        bad("#4 grad 出現 NaN")
    print("    ⇒ 三個 cond 生成端模組 grad norm 全 > 0 ✓")

    # ② hinge 的定義：散度 ≥ margin ⇒ 精確為 0（⛔ 不再推）
    l0, _ = hinge(torch.randn(16, 8), 0.0)
    print(f"    margin=0 ⇒ l_div={l0.item():.6f} {'✓ 精確 0' if l0.item() == 0.0 else '🚨'}")
    if l0.item() != 0.0:
        bad("#4 margin=0 時 hinge 應該精確為 0（relu 的定義）")

    # ③ ⭐ 塌陷側：散度越小，梯度【不可以】跟著消失（這就是 ⛔ 不用 cos 的理由）
    gs = []
    for scale in (1.0, 1e-3, 1e-6):
        for m in (enc, head, ad):
            m.zero_grad(set_to_none=True)
        _, t = hinge(torch.randn(16, 8) * scale, 0.3)
        t.backward()
        gs.append(sum(float(p.grad.pow(2).sum()) for p in head.parameters()) ** 0.5)
    print(f"    ix 縮到 1 / 1e-3 / 1e-6 ⇒ cond_head grad norm "
          f"{[round(x, 5) for x in gs]}")
    if not all(x > 0 for x in gs):
        bad("#4 散度趨近 0 時梯度消失 ⇒ ⛔ 塌掉的顆爬不出來（hinge 的意義就在這裡）")
    if gs[-1] < gs[0] * 0.1:
        bad(f"#4 散度縮 1e6 倍，梯度掉了 {gs[0]/max(gs[-1],1e-30):.1f} 倍 ⇒ 有消失的傾向")
    print("    ⇒ 散度縮 1e6 倍，梯度量級不塌 ✓（⛔ cos 在這一格會歸零）")

    # ④ ⛔ NaN 護欄：c_int 與 c_zero 完全相同（真塌陷）也不可以吐 NaN
    for m in (enc, head, ad):
        m.zero_grad(set_to_none=True)
    l4, t4 = hinge(torch.zeros(16, 8), 0.3)
    t4.backward()
    nan = any(p.grad is not None and bool(torch.isnan(p.grad).any())
              for m in (enc, head, ad) for p in m.parameters())
    print(f"    完全塌陷（ix=0、ad 有 bias ⇒ c_int≈c_zero）l_div={l4.item():.4f}"
          f"  NaN={'🚨 有' if nan else '✓ 無'}")
    if nan:
        bad("#4 完全塌陷時反向吐 NaN ⇒ ⛔ 整輪訓練會死在最需要這帖藥的那一刻")

# ═══════════════════════════════════════════════════════════════════
print()
shutil.rmtree(TMP, ignore_errors=True)
if fails:
    print(f"🚨 FAIL {len(fails)} 項")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("✅ ALL PASS")
