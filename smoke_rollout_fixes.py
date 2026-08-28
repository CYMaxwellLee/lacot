"""smoke：2026-08-28 code review 那 16 條的驗證。⛔ 本機沒有 GPU／沒有 /archive 資料 ⇒
   能【跑真的】就跑真的（純函式），跑不了的就用 AST 驗【那一行真的長那樣】。

⭐ 每個 case 都寫明【為什麼期望是這個答案】—— 不然測試自己壞了也看不出來。
⚠️ AST 那幾格擋的是「改回去」，⛔ 不是「這次跑對了」—— 後者要上叢集，指令寫在報告裡。

    ./.venv/bin/python smoke_rollout_fixes.py
"""
import ast
import pathlib
import sys
import types

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lacot.dev_eval import _bfs_from, _passable_cells, cell_width
from lacot.refine_grad import GeoValue, grad_steps
from lacot.subgoal import SubgoalPlanner, arc_subgoal, bfs_subgoal

fails = []
ROLLOUT = ROOT / "experiments" / "scratch_lacot_rollout.py"
DECPROBE = ROOT / "experiments" / "exp_decode_probe.py"
SPANGAP = ROOT / "experiments" / "exp_span_gap.py"


def bad(msg):
    fails.append(msg)
    print(f"       🚨 {msg}")


def src(path):
    return path.read_text()


def tree(path):
    return ast.parse(src(path))


def func_node(path, name):
    for n in ast.walk(tree(path)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def load_func(path, name):
    """把某一支【頂層純函式】從原始碼裡挖出來 exec ⇒ 驗的是真的跑在主線上的那一份。

    ⭐ 主線整支腳本 import 就會去讀 /archive 的資料 ⇒ 本機 import 不了；
       挖單一個函式出來是唯一能【真的呼叫它】的辦法，⛔ 比字串比對強得多。
    """
    n = func_node(path, name)
    assert n is not None, f"⛔ {path.name} 裡找不到 {name}()"
    mod = ast.Module(body=[n], type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), ns)
    return ns[name]


def env_defaults(path):
    """掃出所有 os.environ.get("LACOT_X", <literal>) 的預設值。"""
    out = {}
    for n in ast.walk(tree(path)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and len(n.args) == 2):
            try:
                key = ast.literal_eval(n.args[0])
                val = ast.literal_eval(n.args[1])
            except Exception:
                continue
            if isinstance(key, str) and key.startswith("LACOT_"):
                out[key] = val
    return out


def mk_env(M, unit=4.0, off=4.0):
    """假 env：只提供 maze_map ＋ ij_to_xy，⛔ 不需要 mujoco。"""
    M = np.asarray(M)
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(
        maze_map=M, ij_to_xy=lambda c: (c[1] * unit - off, c[0] * unit - off)))


CORRIDOR = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]]
MEDIUM = [[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1],
          [1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1],
          [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
COMB = [[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]]

# ═══════════════════════════════════════════════════════════════════
# #1  R=0 ＋ GRAD_REFINE=1 ⇒ 整集只抽一次 flow，u 從此凍住
# ═══════════════════════════════════════════════════════════════════
print("#1  grad_steps：R=0 不爬、⛔ 也不碰快取")
# 期望：R=0 一律 (False, 0) ⇒ 呼叫端據此【跳過】_GRAD_CACHE
for R, warm, want in ((0, False, (False, 0)), (0, True, (False, 0)),
                      (1, False, (False, 50)), (1, True, (True, 10)),
                      (3, False, (False, 150)), (3, True, (True, 30))):
    got = grad_steps(R, warm, 50, 10)
    ok = got == want
    print(f"    R={R} warm={int(warm)} ⇒ {got}  {'✓' if ok else '🚨 期望 ' + str(want)}")
    if not ok:
        bad(f"#1 grad_steps(R={R}, warm={warm}) = {got}，期望 {want}")
# GRAD_R_WARM=0 ⇒ ⛔ 不接續，每個 chunk 從頭爬滿
if grad_steps(2, True, 50, 0) != (False, 100):
    bad("#1 GRAD_R_WARM=0 時應該不接續、爬滿 R*GRAD_R")


def _sim(steps_fn, n_chunks=6, R=0, grad_r=50, warm=10):
    """模擬一整集：每個 chunk 抽一個新 u，看最後手上是【第幾次抽的】。
    ⭐ 這才是 #1 真正的傷害 —— 不是「爬幾步」，是「整集只用第一個 u」。"""
    cache = {"u": None}
    used = []
    for c in range(n_chunks):
        u = f"flow_sample_{c}"                     # 每個 chunk 都重抽
        use_warm, st = steps_fn(R, cache["u"] is not None, grad_r, warm)
        if steps_fn is _steps_old:                 # 舊版：不管 st 是不是 0 都讀寫快取
            if use_warm:
                u = cache["u"]
            cache["u"] = u
        elif st > 0:
            if use_warm:
                u = cache["u"]
            cache["u"] = u
        used.append(u)
    return used


def _steps_old(R, has_warm, grad_r, grad_r_warm):
    """舊版邏輯（⛔ 原樣重現，用來證明那個病真的存在）。"""
    if has_warm and grad_r_warm > 0:
        return True, R * grad_r_warm
    return False, R * grad_r


old_used = _sim(_steps_old, R=0)
new_used = _sim(grad_steps, R=0)
print(f"    R=0 整集六個 chunk 實際用到的 u：")
print(f"      舊版 {old_used[0]} … {old_used[-1]}   （{len(set(old_used))} 個相異）")
print(f"      新版 {new_used[0]} … {new_used[-1]}   （{len(set(new_used))} 個相異）")
if len(set(old_used)) != 1:
    bad("#1 舊版應該【整集凍在同一個 u】⇒ 這個 case 沒重現到病，換一組數字")
if len(set(new_used)) != 6:
    bad(f"#1 新版 R=0 應該每個 chunk 都用新抽的 u，實際只有 {len(set(new_used))} 個相異")
print(f"    ⇒ 舊版凍住 ✓ 已重現、新版每 chunk 換新 u ✓")

# 主線真的改成用它了嗎（⛔ 不是我在這裡自己算一遍）
pc = func_node(ROLLOUT, "policy_chunk")
pc_src = ast.unparse(pc)
if "grad_steps(" not in pc_src:
    bad("#1 policy_chunk 沒有呼叫 grad_steps ⇒ 這個修法沒接上主線")
if "_steps > 0" not in pc_src or "_GRAD_CACHE" not in pc_src:
    bad("#1 policy_chunk 應該用 `if _steps > 0:` 包住 _GRAD_CACHE 的讀寫")

# ═══════════════════════════════════════════════════════════════════
# #2  分段 arm 用 R=0，它的對手 LaCoT 用 R=1
# ═══════════════════════════════════════════════════════════════════
print("\n#2  分段 arm 與 LaCoT 用同一個 R")
calls = [n for n in ast.walk(tree(ROLLOUT))
         if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "dev_rollout"]
sg_calls = [c for c in calls if any(k.arg == "subgoal" for k in c.keywords)]
lacot_calls = [c for c in calls if c.args and "LaCoT" in ast.unparse(c.args[2] if len(c.args) > 2 else c)]
if not sg_calls:
    bad("#2 找不到 subgoal=True 的 dev_rollout 呼叫")
for c in sg_calls:
    r_arg = ast.unparse(c.args[0])
    print(f"    分段 arm 的 R = {r_arg}")
    if r_arg == "0":
        bad("#2 分段 arm 還是寫死 R=0 ⇒ 它的短程層一步都不爬，而對手 LaCoT 有爬")
    if r_arg != "_R_ARM":
        bad(f"#2 分段 arm 的 R 應該是 _R_ARM（跟 LaCoT 同一個），實際 {r_arg}")
for c in lacot_calls:
    print(f"    LaCoT arm 的 R     = {ast.unparse(c.args[0])}")
    if ast.unparse(c.args[0]) != "_R_ARM":
        bad("#2 LaCoT arm 的 R 應該也是 _R_ARM ⇒ 兩邊共用同一個值才叫對照")

# ═══════════════════════════════════════════════════════════════════
# #3  官方 rollout() 從不重置 _GRAD_CACHE（dev 那條有）
# ═══════════════════════════════════════════════════════════════════
print("\n#3  官方 rollout() 每集重置爬坡快取")
ro = ast.unparse(func_node(ROLLOUT, "rollout"))
for want, why in (("_reset_grad_cache()", "跨集/跨 task/跨 arm 的爬坡快取汙染"),
                  ("_reseed_shuf(", "#16：shuf arm 的取樣流也要每集釘死")):
    ok = want in ro
    print(f"    rollout() 內有 {want:<22} {'✓' if ok else '🚨'}   ({why})")
    if not ok:
        bad(f"#3/#16 rollout() 少了 {want}")
# 順序：一定要在 env.reset 之後（⛔ 之前重置等於沒重置）
i_reset = ro.index("env.reset")
i_cache = ro.index("_reset_grad_cache()")
if not i_reset < i_cache:
    bad("#3 _reset_grad_cache() 必須在 env.reset 之後")

# ═══════════════════════════════════════════════════════════════════
# #4  要互相對打的 arm 寫進同一個檔名
# ═══════════════════════════════════════════════════════════════════
print("\n#4  三個 arm 的檔名必須互不相同")
tag_extra = load_func(ROLLOUT, "_tag_extra")
base = dict(ENC_OBJ="recon", LEARNED_REFINE=0, GRAD_REFINE=1)
arms = {
    "flat-grad": dict(base, SUBGOAL=""),
    "S1(latent)": dict(base, SUBGOAL="latent"),
    "S0(bfs)": dict(base, SUBGOAL="bfs"),
}
def _tag_extra_OLD(ENC_OBJ="sg_infonce", LEARNED_REFINE=1, COND_DROP=0.0, BC_INDEP=0, **_ig):
    """舊版（⛔ 原樣重現）：只帶 ENC_OBJ / LEARNED_REFINE / COND_DROP / BC_INDEP。"""
    x = ""
    if ENC_OBJ != "sg_infonce":
        x += f"_eo{ENC_OBJ}"
    if not LEARNED_REFINE:
        x += "_norf"
    if COND_DROP > 0:
        x += f"_cd{COND_DROP:g}"
    if BC_INDEP:
        x += "_bci"
    return x


names, old_names = {}, {}
for k, cfg in arms.items():
    names[k] = tag_extra(**cfg)
    old_names[k] = _tag_extra_OLD(**cfg)
    print(f"    {k:<11} 舊 '{old_names[k]}'   新 '{names[k]}'")
# ⭐ reviewer 講的情境：LOAD_CKPT 模式下 STEPS2=0、其他全同 ⇒ 舊版三個 arm 同一個檔名
print(f"    ⇒ 舊版 {len(set(old_names.values()))} 個相異檔名"
      f"（{'🚨 三個 arm 互相覆蓋' if len(set(old_names.values())) == 1 else '?'}）"
      f"；新版 {len(set(names.values()))} 個")
if len(set(old_names.values())) != 1:
    bad("#4 舊版在這組配置下沒有撞名 ⇒ 這個 case 沒重現到病，換一組")
if len(set(names.values())) != len(names):
    bad(f"#4 三個 arm 產生重複檔名：{names} ⇒ 後跑的會覆蓋先跑的")
# 預設值一律不進檔名 ⇒ ⛔ 舊結果的檔名不變
if tag_extra() != "":
    bad(f"#4 全預設應該產生空後綴（⛔ 否則舊檔名全變），實際 '{tag_extra()}'")
print(f"    全預設 ⇒ '' {'✓' if tag_extra() == '' else '🚨'}（⛔ 舊檔名不變）")
# 爬坡參數也要分得開
if tag_extra(GRAD_REFINE=1, GRAD_R=50) == tag_extra(GRAD_REFINE=1, GRAD_R=20):
    bad("#4 GRAD_R 不同卻同檔名")
if tag_extra(GRAD_REFINE=1, GRAD_ETA=0.1) == tag_extra(GRAD_REFINE=1, GRAD_ETA=0.5):
    bad("#4 GRAD_ETA 不同卻同檔名")
if tag_extra(DEV_TIERS="") == tag_extra(DEV_TIERS="2"):
    bad("#4 DEV_TIERS 不同卻同檔名（⛔ 單層結果不可跟全 tier 混）")
# ⭐ _tag_extra 的預設值必須跟 os.environ.get 的預設值逐項對齊，⛔ 對不齊就判錯「是不是預設」
import inspect

envd = env_defaults(ROLLOUT)
sig = inspect.signature(tag_extra)
mismatch = []
for pname, prm in sig.parameters.items():
    key = "LACOT_" + pname
    if key not in envd:
        mismatch.append(f"{pname}: 主線沒有 {key}")
    elif envd[key] != prm.default:
        mismatch.append(f"{pname}: env 預設 {envd[key]!r} ≠ _tag_extra 預設 {prm.default!r}")
print(f"    _tag_extra 的 {len(sig.parameters)} 個預設值 vs 主線 env 預設："
      f"{'✓ 全對齊' if not mismatch else '🚨 ' + '; '.join(mismatch)}")
for m in mismatch:
    bad(f"#4 {m}")
# 旋鈕也要落進 json 的頂層
out_assign = [n for n in ast.walk(tree(ROLLOUT))
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "out"]
out_keys = {k.arg for k in out_assign[0].value.keywords}
need = {"subgoal", "grad_refine", "grad_r", "grad_eta", "grad_lam", "grad_r_warm",
        "delta_sub", "sub_cap_chunks", "sub_stuck_chunks", "dev_tiers",
        "enc_obj", "learned_refine", "cond_drop", "bc_indep"}
miss = need - out_keys
print(f"    out 的頂層旋鈕欄位：{'✓ 齊了' if not miss else '🚨 少 ' + str(sorted(miss))}")
if miss:
    bad(f"#4 out 少了 {sorted(miss)} ⇒ 從 json 讀不出這輪是哪個配置")

# ═══════════════════════════════════════════════════════════════════
# #5  bfs_subgoal 不保證 subgoal 比現在更靠近目標
# ═══════════════════════════════════════════════════════════════════
print("\n#5  bfs_subgoal 一定要更靠近目標")


def bfs_subgoal_OLD(env, s_ij, g_ij, delta_cells, bfs_from):
    """舊版（⛔ 原樣重現）：cur 算了沒用，key 只挑「離【起點】約 delta」。"""
    dist = bfs_from(env, g_ij)
    if s_ij not in dist:
        return None
    cur = dist[s_ij]                                     # noqa: F841 ← 舊版就是沒用到
    reach = bfs_from(env, s_ij)
    best, best_key = None, None
    for c, dc in reach.items():
        if c not in dist:
            continue
        key = (abs(dc - delta_cells), dist[c])
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


env = mk_env(CORRIDOR)
G = (1, 7)
D = _bfs_from(env, G)
n_old_bad = n_new_bad = 0
print(f"    走廊 0..7、目標 (1,7)、delta_cells=2")
for s in [(1, 1), (1, 3), (1, 5), (1, 6), (1, 7)]:
    co = bfs_subgoal_OLD(env, s, G, 2, _bfs_from)
    cn = bfs_subgoal(env, s, G, 2, _bfs_from)
    o_ok = D[co] < D[s] or D[s] == 0
    n_ok = D[cn] < D[s] or D[s] == 0
    n_old_bad += (not o_ok)
    n_new_bad += (not n_ok)
    print(f"      agent{s} 距目標 {D[s]} ⇒ 舊 {co} 距 {D[co]} {'✓' if o_ok else '🚨'}"
          f"   新 {cn} 距 {D[cn]} {'✓' if n_ok else '🚨'}")
if n_old_bad == 0:
    bad("#5 舊版在這組輸入上沒犯錯 ⇒ 這個 case 沒重現到病，換一組")
if n_new_bad:
    bad(f"#5 新版仍有 {n_new_bad} 個 case 沒有更靠近目標")
# 迷宮版：每一個可達 (s, g) 都必須嚴格前進（⛔ 不是抽樣看看）
mz = mk_env(MEDIUM)
cells = _passable_cells(mz)
n_pairs = n_stall = 0
for g in cells:
    dg = _bfs_from(mz, g)
    for s in cells:
        if s not in dg or dg[s] == 0:
            continue
        c = bfs_subgoal(mz, s, g, 2, _bfs_from)
        n_pairs += 1
        if c is None or dg[c] >= dg[s]:
            n_stall += 1
print(f"    medium 迷宮全 {n_pairs} 組 (s,g)：沒有前進的有 {n_stall} 組"
      f"   {'✓' if n_stall == 0 else '🚨'}")
if n_stall:
    bad(f"#5 medium 迷宮上有 {n_stall}/{n_pairs} 組 subgoal 沒有更靠近目標")
# 已在半徑內 ⇒ 直接指目標
if bfs_subgoal(env, (1, 6), G, 3, _bfs_from) != G:
    bad("#5 cur ≤ delta_cells 時應該直接回目標")

# ═══════════════════════════════════════════════════════════════════
# #6  decoder 的健康檢查跑在 LOAD_CKPT 載入【之前】
# ═══════════════════════════════════════════════════════════════════
print("\n#6  decoder 檢查要在權重載入【之後】")
s_ro = src(ROLLOUT)
i_load = s_ro.index("_ck = torch.load(")
i_call = s_ro.index("_n, _sh, _gap = _decoder_health()")
print(f"    torch.load 在 {s_ro[:i_load].count(chr(10)) + 1} 行，"
      f"decoder 探針在 {s_ro[:i_call].count(chr(10)) + 1} 行"
      f"   {'✓ 探針在後' if i_call > i_load else '🚨 探針在前 ⇒ 只評估模式下量的是隨機初始化'}")
if i_call < i_load:
    bad("#6 decoder 探針仍在 LOAD_CKPT 之前 ⇒ 只評估模式下量的是隨機初始化的 u_dec")
dh = ast.unparse(func_node(ROLLOUT, "_decoder_health"))
for want, why in (("get_rng_state", "⛔ 探針不可以擾動主流程的取樣流（否則訓練不再逐位元重現）"),
                  ("default_rng(", "探針要有自己的 numpy RNG")):
    if want not in dh:
        bad(f"#6 _decoder_health 少了 {want}（{why}）")
print("    _decoder_health 用自己的 RNG ＋ 存還原 torch RNG 狀態 ✓（⇒ 訓練逐位元不變）")
if "_dgap >= 0.02" not in s_ro:
    bad("#6 前置檢查缺少 decoder ctx_usage 的 assert（跟穿牆 assert 同層級）")
else:
    print("    SUBGOAL=latent / GRAD_REFINE 的前置檢查有 `assert _dgap >= 0.02` ✓")

# ═══════════════════════════════════════════════════════════════════
# #7  cell_w 取「頭兩個可通行格」不保證相鄰
# ═══════════════════════════════════════════════════════════════════
print("\n#7  cell_width 用全對距離取最小")


def cell_w_OLD(env):
    c = _passable_cells(env)
    a, b = env.unwrapped.ij_to_xy(c[0]), env.unwrapped.ij_to_xy(c[1])
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


for name, M, want in (("medium（頭兩格剛好相鄰）", MEDIUM, 4.0),
                      ("梳齒列 [1,0,1,0,1,0,1]", COMB, 4.0)):
    e = mk_env(M)
    o, n = cell_w_OLD(e), cell_width(e)
    print(f"    {name:<26} 舊 {o:.2f}   新 {n:.2f}   （真值 {want:.2f}）"
          f"   {'✓' if abs(n - want) < 1e-9 else '🚨'}")
    if abs(n - want) > 1e-9:
        bad(f"#7 cell_width 在 {name} 上算出 {n}，期望 {want}")
if abs(cell_w_OLD(mk_env(COMB)) - 4.0) < 1e-9:
    bad("#7 梳齒列上舊版沒出錯 ⇒ 這個 case 沒重現到病")
print(f"    ⇒ 舊版在梳齒列上翻倍 ✓ 已重現")
# ⭐ 影響範圍：拿【真的】迷宮圖量，⛔ 不是只用我編的例子
try:
    import inspect
    import re

    from ogbench.locomaze import maze as _mz
    _src = inspect.getsource(_mz)
    _maps = {"medium": MEDIUM}
    for _n in ("large", "giant"):
        _m = re.search(r"elif self\._maze_type == '%s':\s*\n\s*maze_map = \[(.*?)\]\s*\n\s*elif"
                       % _n, _src, re.S)
        if _m:
            _maps[_n] = [[int(v) for v in r.split(",") if v.strip()]
                         for r in re.findall(r"\[([0-9,\s]+)\]", _m.group(1))]
    print("    真迷宮圖的影響範圍（⭐ 這決定既有結論有沒有被咬到）：")
    for _n, _M in _maps.items():
        _e = mk_env(_M)
        _o, _nw = cell_w_OLD(_e), cell_width(_e)
        print(f"      {_n:<7} 頭兩格 {_passable_cells(_e)[0]} {_passable_cells(_e)[1]}"
              f"   舊 {_o:.2f}   新 {_nw:.2f}"
              f"   {'✓ 舊版剛好也對 ⇒ 既有結論沒被咬到' if abs(_o - _nw) < 1e-9 else '🚨 舊版翻倍'}")
        if abs(_nw - 4.0) > 1e-9:
            bad(f"#7 {_n} 的格寬算出 {_nw}，期望 4.0")
except Exception as _e:                       # ⛔ 沒裝 ogbench 就跳過這一格，⛔ 不算通過
    print(f"    ⚠️ 讀不到 ogbench 的迷宮圖（{type(_e).__name__}）⇒ 影響範圍這格沒驗到")
for f in (ROLLOUT, SPANGAP):
    if "cell_width(env)" not in src(f):
        bad(f"#7 {f.name} 沒有改用共用的 DE.cell_width")
print(f"    兩支呼叫端都改用 DE.cell_width ✓（⛔ 不留兩份會分岔的實作）")

# ═══════════════════════════════════════════════════════════════════
# #8  wall_depth 的 sanity assert 結構上必然通過
# ═══════════════════════════════════════════════════════════════════
print("\n#8  幾何 value 的健康檢查要擋得住「牆是空的」")
rng = np.random.default_rng(0)
seg = []
for _ in range(400):
    t = rng.random(60)
    x = np.where(t < .5, t * 8, 4.) + rng.normal(0, .05, 60)
    y = np.where(t < .5, 0., (t - .5) * 8) + rng.normal(0, .05, 60)
    seg.append(np.stack([x, y], 1))
GOOD = np.concatenate(seg).astype(np.float32)                 # L 形走廊 ⇒ 有牆
BADBOX = rng.uniform(-1, 9, (24000, 2)).astype(np.float32)    # 整盒都走過 ⇒ 牆是空的
for name, OBS, want_ok in (("L 形走廊（真的有牆）", GOOD, True),
                           ("整盒都是自由空間", BADBOX, False)):
    mu, sd_ = OBS.mean(0), OBS.std(0) + 1e-6
    geo = GeoValue(OBS, mu, sd_, res=8)
    Z = torch.tensor((OBS[:64 * 32] - mu) / sd_, dtype=torch.float32).reshape(64, 32, 2)
    old_med = float(geo.wall_depth(Z).median())               # 舊 assert 看的東西
    h = geo.health()
    print(f"    {name}")
    print(f"      舊 assert（真軌跡穿牆中位 < 0.15）：{old_med:.4f} ⇒ "
          f"{'通過' if old_med < 0.15 else '擋下'}   ⚠️ 兩種情況都通過 ⇒ ⛔ 它是恆真的")
    print(f"      新 health()：ok={h['ok']}   格心 round-trip {h['mapping_err']:.2e}"
          f"   盒內隨機點穿牆中位 {h['wall_median_random']:.4f}   覆蓋 {h['coverage']:.1%}")
    for r in h["reasons"]:
        print(f"        - {r}")
    if old_med >= 0.15:
        bad(f"#8 舊 assert 在「{name}」上擋下了 ⇒ 它不是恆真的，這個 case 的立論要改")
    if h["ok"] != want_ok:
        bad(f"#8 health() 在「{name}」上判 ok={h['ok']}，期望 {want_ok}")
    if h["mapping_err"] >= 1e-4:
        bad(f"#8 格心 round-trip 誤差 {h['mapping_err']:.2e} ≥ 1e-4 ⇒ 座標映射壞了")
if "GEO.health()" not in s_ro:
    bad("#8 主線沒有呼叫 GEO.health()")

# ═══════════════════════════════════════════════════════════════════
# #10  planner 的三個觸發數的是 chunk 不是 step
# ═══════════════════════════════════════════════════════════════════
print("\n#10 planner 的 cap / stuck 單位是 chunk")
pl = SubgoalPlanner(delta_sub=7.5, cap=10, stuck_m=3, chunk=4)
print(f"    預設 cap={SubgoalPlanner(7.5).cap} chunk  stuck_m={SubgoalPlanner(7.5).stuck_m} chunk"
      f"   （CHUNK=4 ⇒ {pl.cap_steps} / {pl.stuck_steps} 個 env step）")
if SubgoalPlanner(7.5).cap != 10 or SubgoalPlanner(7.5).stuck_m != 3:
    bad("#10 預設應改成 cap=10 / stuck_m=3（chunk 數）")
# cap：追一個永遠到不了的 subgoal ⇒ 第 10 次 observe 必須喊重想
pl.set([100.0, 100.0])
fires = [i + 1 for i in range(12) if pl.observe([0.0, float(i) * 0.001])]
print(f"    追不到的 subgoal：第 {fires[0]} 次 observe 喊重想（cap=10）"
      f"   {'✓' if fires[0] <= 10 else '🚨'}")
if fires[0] > 10:
    bad(f"#10 cap=10 卻到第 {fires[0]} 次才觸發")
# stuck：完全不動 ⇒ 第 3 次就該喊（比 cap 早）
pl2 = SubgoalPlanner(delta_sub=7.5, cap=10, stuck_m=3, chunk=4)
pl2.set([100.0, 100.0])
f2 = [i + 1 for i in range(12) if pl2.observe([0.0, 0.0])]
# ⚠️ 期望是 4 不是 3：第一次 observe 把 best 從 inf 更新成當下距離（＝建立基準線），
#    「連續 3 個 chunk 沒有更靠近」要到第 2、3、4 次才數滿 ⇒ ⛔ 這是定義，不是差一錯誤。
print(f"    原地不動：第 {f2[0]} 次 observe 喊重想"
      f"（stuck_m=3，第 1 次是建立基準線 ⇒ 期望 4）   {'✓' if f2[0] == 4 else '🚨'}")
if f2[0] != 4:
    bad(f"#10 stuck_m=3 應在第 4 次 observe 觸發（第 1 次建立基準線），實際第 {f2[0]} 次")
if f2[0] * pl2.chunk > 20:
    bad(f"#10 卡住觸發要 {f2[0] * pl2.chunk} 個 env step ⇒ 太鬆，等於按不下去")
# n_replan：第一次設 subgoal ⛔ 不算重想
pl3 = SubgoalPlanner(7.5)
pl3.set([1.0, 1.0])
a1 = pl3.n_replan
pl3.set([2.0, 2.0])
pl3.set([3.0, 3.0])
print(f"    設 1 次 ⇒ n_replan={a1}（期望 0）   設 3 次 ⇒ n_replan={pl3.n_replan}（期望 2）"
      f"   {'✓' if (a1, pl3.n_replan) == (0, 2) else '🚨'}")
if (a1, pl3.n_replan) != (0, 2):
    bad(f"#10 n_replan 應該不算第一次，實際 {a1} / {pl3.n_replan}")
# docstring 不准再寫「步」
doc = (SubgoalPlanner.__doc__ or "") + (SubgoalPlanner.observe.__doc__ or "")
if "cap 步" in doc or "stuck_m 步" in doc:
    bad("#10 docstring 還寫著「步」⇒ 單位講錯的來源沒堵掉")
if "chunk" not in doc:
    bad("#10 docstring 沒有明確講單位是 chunk")
# 主線要把 CHUNK 與 SUB_STUCK 傳進去
if "stuck_m=SUB_STUCK" not in s_ro or "chunk=CHUNK" not in s_ro:
    bad("#10 主線建構 SubgoalPlanner 時沒傳 stuck_m / chunk")
print("    主線傳了 stuck_m=SUB_STUCK, chunk=CHUNK ✓")

# ═══════════════════════════════════════════════════════════════════
# #11 / #12  arc_subgoal 沒檢查 pts[0] ≈ 現在位置；S0/S1 的單位不同
# ═══════════════════════════════════════════════════════════════════
print("\n#11 路塌成一點時 arc_subgoal 會安靜地回終點")
flat = torch.zeros(1, 128, 2) + torch.tensor([9.0, 9.0])     # 整條路塌在 (9,9)
sub = arc_subgoal(flat, 7.5)
print(f"    整條路塌成一點 ⇒ subgoal {sub[0].tolist()}（＝路的終點，⛔ 不叫）")
if not torch.allclose(sub[0], torch.tensor([9.0, 9.0])):
    bad("#11 這個 case 沒重現到「塌掉就回終點」")
d0 = float(torch.linalg.norm(flat[0, 0] - torch.tensor([0.0, 0.0])))
print(f"    ⇒ 這時 d0 = ‖路的第0點 − 現在(0,0)‖ = {d0:.2f} ＞ 0.5*DELTA_SUB(3.75)"
      f" ⇒ 診斷欄位會記一筆 bad_d0")
for want in ("SUB_DIAG", '"d0"', '"dsub"', "n_bad_d0", "n_bad_dsub"):
    if want not in s_ro:
        bad(f"#11 主線少了診斷欄位 {want}")
if 'SUB_DIAG["dsub"].append' not in s_ro:
    bad("#12 主線沒有記錄 ‖sub − 現在‖（S0/S1 對照要用）")
if s_ro.count('SUB_DIAG["dsub"].append') != 1 or "return sub" not in s_ro:
    bad("#12 dsub 應該記在 _plan 的【共同出口】⇒ latent 與 bfs 兩條路都要進到同一格")
print("    主線記錄 d0 / dsub / n_replan，收工報一行 ✓（#12：兩個 arm 的落點分布可對照）")

# ═══════════════════════════════════════════════════════════════════
# #13 / #14  exp_decode_probe
# ═══════════════════════════════════════════════════════════════════
print("\n#13 exp_decode_probe 的 T_CAP 要跟主線同步")
t_main = env_defaults(ROLLOUT)["LACOT_TCAP"]
t_probe = env_defaults(DECPROBE)["LACOT_TCAP"]
print(f"    主線 {t_main}   decode probe {t_probe}   {'✓ 同步' if t_main == t_probe else '🚨 不同步'}")
if t_main != t_probe:
    bad(f"#13 T_CAP 預設不同步：主線 {t_main} vs probe {t_probe}"
        f" ⇒ decoder 的 pos_q 長度與每點間距都不一樣")
if t_probe != 128:
    bad(f"#13 probe 的 T_CAP 預設應為 128，實際 {t_probe}")
if "必須跟主線同步" not in src(DECPROBE):
    bad("#13 沒有在註解標明「必須跟主線同步」")

print("\n#14 ⛔ 不靠 closure 順序拿模組")
s_dp = src(DECPROBE)
# ⚠️ 看的是【真的屬性存取】，⛔ 不是字串比對 —— docstring 裡會提到 __closure__ 這個字
attr_hits = [n for n in ast.walk(tree(DECPROBE))
             if isinstance(n, ast.Attribute) and n.attr == "__closure__"]
print(f"    程式碼裡還有 .__closure__ 的存取嗎：{'🚨 有 ' + str(len(attr_hits)) if attr_hits else '✓ 沒有'}"
      f"（docstring 裡的說明文字不算）")
if attr_hits:
    bad("#14 還在用 __closure__ ⇒ 改名或多引用一個外層變數就會讓兩顆對調存進 ckpt")
bc = func_node(DECPROBE, "build_context_fn")
rets = [n for n in ast.walk(bc) if isinstance(n, ast.Return)]
sizes = {len(r.value.elts) for r in rets if isinstance(r.value, ast.Tuple)}
print(f"    build_context_fn 的 {len(rets)} 個 return 都是 {sizes} 元組"
      f"   {'✓' if sizes == {4} else '🚨'}")
if sizes != {4}:
    bad(f"#14 build_context_fn 的 return 元組長度不一致：{sizes}")
if 'ctx_mods["traj_enc"]' not in s_dp or 'ctx_mods["e_pooler"]' not in s_dp:
    bad("#14 存 ckpt 時沒有按名字從模組 dict 拿")
# ⭐ closure 順序【真的】是按名字排的 —— 這格證明那個 bug 不是臆測
_a_first, _z_last = object(), object()


def _mk():
    zzz, aaa = _z_last, _a_first
    return lambda: (zzz, aaa)


order = [c.cell_contents is _a_first for c in _mk().__closure__]
print(f"    freevar 順序是按【名字】排的：closure[0] 是 aaa ⇒ {order[0]}"
      f"   {'✓ 證實' if order[0] else '🚨'}")
if not order[0]:
    bad("#14 這個 python 上 closure 不按名字排 ⇒ 這條的立論要重寫")

# ═══════════════════════════════════════════════════════════════════
# #15  X 對照在 GRAD_REFINE=1 或 LEARNED_REFINE=0 下是 no-op
# ═══════════════════════════════════════════════════════════════════
print("\n#15 X 對照不適用時要說出來、⛔ 不要照樣印判決")
if "_REV_OK = bool(LEARNED_REFINE) and not GRAD_REFINE" not in s_ro:
    bad("#15 缺少 _REV_OK 的判斷（LEARNED_REFINE and not GRAD_REFINE）")
if "本配置沒有 learned refine" not in s_ro:
    bad("#15 不適用時沒有印一行說明")
if '"applicable": False' not in s_ro:
    bad("#15 json 沒有記下這個對照不適用")
print("    _REV_OK ＝ LEARNED_REFINE and not GRAD_REFINE ✓；不適用時印一行＋json 記 applicable=False ✓")
# _RDIR 只被 _apply_refine 讀 ⇒ 這條的立論
n_rdir = s_ro.count("_RDIR[0]")
where = [ast.unparse(f).count("_RDIR[0]") for f in [func_node(ROLLOUT, "_apply_refine")]]
print(f"    _RDIR[0] 全檔出現 {n_rdir} 次，其中 _apply_refine 裡 {where[0]} 次"
      f"（其餘是設定值的地方）")
if where[0] == 0:
    bad("#15 _RDIR 不在 _apply_refine 裡 ⇒ 這條的立論要重查")

# ═══════════════════════════════════════════════════════════════════
# #16  _shuf_rng 不隨集重置
# ═══════════════════════════════════════════════════════════════════
print("\n#16 shuf arm 的取樣流每集重置")
if "def _reseed_shuf(" not in s_ro:
    bad("#16 沒有 _reseed_shuf")
ep = [n for n in ast.walk(tree(ROLLOUT)) if isinstance(n, ast.FunctionDef) and n.name == "_ep_seed"]
if not ep:
    bad("#16 dev_rollout 裡沒有 _ep_seed（torch seed ＋ shuf seed 要一起釘）")
else:
    b = ast.unparse(ep[0])
    print(f"    dev 的每集 seed 函式：{b.splitlines()[1].strip()} / {b.splitlines()[2].strip()}")
    if "manual_seed" not in b or "_reseed_shuf" not in b:
        bad("#16 _ep_seed 要同時釘 torch 與 shuf 兩條流")
# 同一個 i ⇒ 同一組 u（可重現）；不同 i ⇒ 不同（有變化）
r1 = np.random.default_rng(20260823 + 5).integers(0, 1000, 3)
r2 = np.random.default_rng(20260823 + 5).integers(0, 1000, 3)
r3 = np.random.default_rng(20260823 + 6).integers(0, 1000, 3)
print(f"    ep=5 兩次 {r1.tolist()} / {r2.tolist()}（一樣 ⇒ 可重現、可配對）"
      f"   ep=6 {r3.tolist()}（不同 ⇒ 沒有退化成常數）")
if not (np.array_equal(r1, r2) and not np.array_equal(r1, r3)):
    bad("#16 per-episode reseed 的性質不對")

print()
if fails:
    print(f"🚨 FAIL（{len(fails)} 項）")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("✅ ALL PASS")
