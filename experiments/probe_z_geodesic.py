"""路線一（距離幾何）第一探針：凍結 stage-1 encoder 的 latent 空間，量「加 quasimetric 約束
之前」的基線幾何。這支腳本只讀（唯讀 ckpt + 資料 npz），不訓練、不改任何既有檔案。

量三件事：
  1. 兩條真軌跡片段的 latent et_A/et_B 之間，t∈{0.25,0.5,0.75} 線性插值，decode 出的路徑
     落在佔據圖自由格的比例（合法率）—— 跟「真軌跡 encode→decode 直接還原」「隨機 u decode」
     兩組控制組對比。
  2. latent 距離 ‖et_A−et_B‖ vs 兩軌跡起點的格圖 BFS 距離之 Spearman 相關。

⛔ 唯讀：不 import experiments/scratch_lacot_rollout.py（它 import 即執行 2438 行訓練/eval
   主流程，副作用太大）。所有需要的類別／權重載入方式，是讀懂該檔之後在這裡重建的 ——
   每個決定都在下面附 file:line 出處；引不到出處的地方明講是「本探針自己的選擇」。

═══ 材料與出處 ══════════════════════════════════════════════════════════════
- 凍結 ckpt：results/ckpt_large-stitch_..._s27.pt。cfg 直接存在 ckpt 裡
  （scratch_lacot_rollout.py:2433-2438 存檔那行），本腳本一律讀 ckpt["cfg"]，
  不猜檔名 token、不用腳本裡的 env var 預設值（那些預設值不保證等於這顆 ckpt 訓練時的值）。
- 資料切窗／正規化：跟訓練一致，見下面「切窗」與「正規化」兩段。
- decode 用法：跟 eval 時的 `_dec()` 一致，見下面「decode」段。
- 佔據圖／BFS：lacot/refine_grad.py 的 GeoEnergy + lacot/subgoal.py 的 grid_bfs
  （lacot/intent.py 本身不建圖、不做 BFS，它 import lacot.subgoal.grid_shortest_path
  再包一層 route_cells——lacot/intent.py:8,18,46-49 寫明「⛔ BFS 一律用 lacot.subgoal
  的單一來源」。任務材料說「機制在 lacot/intent.py」，實際單一來源在 lacot/subgoal.py，
  這裡兩份都讀了，用 subgoal.py 的 grid_bfs，跟 intent.py 的 route_cells 是同一份代碼。）

═══ 兩個儀器 gate（如實報，不修飾）═══════════════════════════════════════════
  gate 1: 真軌跡 direct-decode 合法率 ≥ 0.95
  gate 2: 真軌跡合法率 − 隨機 u 合法率 ≥ 0.2
  任一沒過 ⇒ 印 "INSTRUMENT INVALID" + 原因，數字照樣印，不隱藏。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench \
    /home/cymaxwelllee/venvs/lacot-rocm/bin/python experiments/probe_z_geodesic.py \
        2>&1 | tee experiments/probe_z_geodesic_report.txt
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 唯讀重用：只 import 純類別定義的模組（無頂層執行副作用），不 import 主檔。
from lacot.e_target import PerceiverPooler      # noqa: E402  (scratch_lacot_rollout.py:12)
from lacot.traj_decoder import TrajDecoder      # noqa: E402  (scratch_lacot_rollout.py:734)
from lacot.refine_grad import GeoEnergy         # noqa: E402  (佔據圖)
from lacot.subgoal import grid_bfs              # noqa: E402  (BFS 單一來源)

T0 = time.time()

# ─────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
CKPT_NAME = ("ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5"
             "_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt")
CKPT_PATH = os.environ.get("LACOT_CKPT", os.path.join(REPO_ROOT, "results", CKPT_NAME))
N_PAIRS = int(os.environ.get("LACOT_N_PAIRS", 100))
SEED = int(os.environ.get("LACOT_SEED", 0))          # 抽樣資料對用的主 rng seed
RAND_U_SEED = 20260905                                # 隨機 u 控制組獨立 torch RNG（今天日期，避免跟資料抽樣糾纏）
GEO_RES = 8   # 佔據圖解析度 —— 沿用全 repo 唯一慣例，見下方「occ 出處」print
GATE1_MIN_REAL_LEGAL = 0.95
GATE2_MIN_MARGIN = 0.20
device = "cpu"


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────
# 0. 資料載入 + 切窗 + 正規化（逐字對齊 experiments/scratch_lacot_rollout.py）
# ─────────────────────────────────────────────────────────────────
hr("0. 資料載入（切窗／正規化 file:line 出處）")
print(f"  env={ENV_NAME}  data_dir={OGB_DATA}  N_PAIRS={N_PAIRS}  seed={SEED}")
print("  直接讀 npz（不經 ogbench.make_env_and_datasets，不需要模擬器/渲染 —— 這支探針")
print("  只用 offline OBS 陣列＋凍結權重，不 step 環境）："
      "\n    出處＝experiments/scratch_lacot_rollout.py:29（同款 np.load 用法）")
_npz = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(_npz["observations"], np.float32)
TERM = np.asarray(_npz["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "⛔ 資料集最後一筆不是 terminal ⇒ traj_end 尾巴未初始化（同 rollout.py:35 的自檢）"
# 正規化：experiments/scratch_lacot_rollout.py:37-38（mu/sd 用【全資料集】的 OBS 算，
# ⛔ 不是用抽出來的片段算 —— 片段的 mu/sd 會跟訓練不一致）。
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
print(f"  OBS shape={OBS.shape}  episodes={len(ends)}（每集長度 min/median/max="
      f"{(ends - starts + 1).min()}/{int(np.median(ends - starts + 1))}/{(ends - starts + 1).max()}）")
print(f"  mu={mu.tolist()}  sd={sd.tolist()}   出處=scratch_lacot_rollout.py:37-38")


# ─────────────────────────────────────────────────────────────────
# 1. 讀 ckpt cfg，重建 traj_enc / e_pooler / u_dec，assert 權重鍵全對上
# ─────────────────────────────────────────────────────────────────
hr("1. 重建凍結模型（架構 file:line 出處 + state_dict 鍵驗證）")
print(f"  ckpt={CKPT_PATH}")
ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)  # weights_only=False：同
# scratch_lacot_rollout.py:1220 的用法 —— cfg 是 python dict，不是純 tensor。
cfg = ck.get("cfg", {})
print(f"  ckpt['cfg']={cfg}   出處＝存檔格式見 scratch_lacot_rollout.py:2433-2438")
K, COND, D_MODEL, T_CAP = cfg["K"], cfg["COND"], cfg["D_MODEL"], cfg["T_CAP"]
ENC_OBJ, CHUNK = cfg["ENC_OBJ"], cfg["CHUNK"]

# ⛔ 卡點檢查①：這支探針只驗過 ENC_OBJ=recon* 且 DEC_START≠hard（無 s_embed）、無 VQ/FSQ、
#    無 intent adapter 的 ckpt。任何一項不符，停手回報，不硬湊。
problems = []
if not ENC_OBJ.startswith("recon"):
    problems.append(f"ENC_OBJ={ENC_OBJ!r} 不是 recon* ⇒ 這顆 ckpt 可能沒有 u_dec（sg_infonce 目標不存 decoder）")
if "u_dec" not in ck:
    problems.append("ckpt 沒有 'u_dec' 鍵 ⇒ 無法 decode，探針做不下去")
if "s_embed" in ck:
    problems.append("ckpt 有 's_embed' 鍵 ⇒ 這顆是 DEC_START=hard 訓的，_dec() 需要起點 token 拼接"
                     "（scratch_lacot_rollout.py:797-806），本腳本沒實作這條分支，停手")
if "vq" in ck:
    problems.append("ckpt 有 'vq' 鍵 ⇒ decode 前需要先 VQ snap（_q()，scratch_lacot_rollout.py:783-789），"
                     "本腳本沒實作，停手")
if "intent_ad" in ck:
    problems.append("ckpt 有 'intent_ad' 鍵 ⇒ decode 前後需要 intent adapter 的 cond/inv 轉換，"
                     "本腳本沒實作，停手")
if problems:
    print("\n⛔⛔⛔ 卡點：ckpt 的訓練設定超出本腳本驗證過的範圍 ⛔⛔⛔")
    for p in problems:
        print("  - " + p)
    print("不硬湊、不塞假資料。停在這裡。")
    sys.exit(1)
print("  ✓ ENC_OBJ=recon_ictr / 無 s_embed / 無 vq / 無 intent_ad ⇒ decode = 純 u_dec(u)，"
      "不需要輔助輸入")
print("    出處：_dec() 定義於 scratch_lacot_rollout.py:792-807；DEC_START!='hard' 分支"
      "（:795-796）只回傳 u_dec(u)；載入端在 :1258-1262 對『ckpt 有無 s_embed』與"
      "『DEC_START 設定』做一致性 assert，本探針用『ckpt 沒有 s_embed』反推 DEC_START≠hard，"
      "邏輯與 :1261-1262 那條 assert 相同方向。")


def sota_mlp(i, h, o, n=2):
    """逐字抄自 experiments/scratch_lacot_rollout.py:507-513（traj_enc 的架構）。"""
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]
        p = h
    lin = nn.Linear(p, o)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


# 建構呼叫逐字對齊 scratch_lacot_rollout.py:559（traj_enc、e_pooler）與 :735（u_dec）。
traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
u_dec = TrajDecoder(D_MODEL, T_CAP).to(device)

for name, mod in (("traj_enc", traj_enc), ("e_pooler", e_pooler), ("u_dec", u_dec)):
    missing, unexpected = mod.load_state_dict(ck[name], strict=False)
    status = "✓ 全部對上" if (not missing and not unexpected) else "⛔ 鍵不對！"
    print(f"  {name:10s} load_state_dict  missing={missing}  unexpected={unexpected}  {status}")
    assert not missing and not unexpected, (
        f"⛔ 卡點：{name} 的 state_dict 鍵沒有完全對上（missing={missing}, unexpected={unexpected}）"
        f" ⇒ 重建的架構跟 ckpt 存的不是同一顆模型，停手。")
print("  出處：load_state_dict 呼叫方式對齊 scratch_lacot_rollout.py:1231-1232（traj_enc/e_pooler）"
      "與 :1257（u_dec）。")

for m in (traj_enc, e_pooler, u_dec):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def etarget(traj, mask):
    """逐字對齊 scratch_lacot_rollout.py:619-621。"""
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)


def decode(u):
    """對齊 _dec(u, s_n)（scratch_lacot_rollout.py:792-807）在 DEC_START != 'hard' 分支
    （本 ckpt 無 s_embed，已於上面驗證）：直接回傳 u_dec(u)，不需要 s_n。"""
    return u_dec(u)


# ─────────────────────────────────────────────────────────────────
# 2. 佔據圖 + BFS（沿用既有工具，不重寫 cell 定義的核心公式）
# ─────────────────────────────────────────────────────────────────
hr("2. 佔據圖 + BFS 工具（file:line 出處）")
print(f"  GeoEnergy(OBS, mu, sd, res={GEO_RES})   出處：lacot/refine_grad.py:43-65（建圖邏輯）；"
      f"res=8 是全 repo 唯一用法，precedent 見 scratch_lacot_rollout.py:294,328,1590")
geo = GeoEnergy(OBS, mu, sd, res=GEO_RES, device="cpu")
occ = (geo.dist[0, 0].numpy() == 0.0)              # 自由格判準：lacot/intent.py:329（同款寫法）
shape_arr = np.asarray(geo.shape, np.int64)
lo_np, span_np = geo.lo, (geo.hi - geo.lo)
print(f"  grid shape={tuple(int(s) for s in geo.shape)}  自由格覆蓋率={geo.coverage:.1%}"
      f"（free={int(occ.sum())}/{int(occ.size)}）")
gh = geo.health()
print(f"  GeoEnergy.health()（lacot/refine_grad.py:119-144，內建健康檢查，非本探針的兩個 gate）："
      f"ok={gh['ok']}  mapping_err={gh['mapping_err']:.2e}  wall_median_random={gh['wall_median_random']:.4f}")
if not gh["ok"]:
    print(f"  ⚠️ health() 沒過：{gh['reasons']}（照實報，不影響下面兩個正式 gate 的判定）")


def zn_to_cell_batch(pts):
    """[...,2] 正規化座標 → cell index（round + clip）。⚠️ 設計選擇：這裡【不做】snap-到-最近自由格
    （lacot/intent.py 的 _i_zn_to_cell 之類的用法在牆格會 snap；本探針量的正是『有沒有落在自由格』，
    snap 會讓合法率恆為 1，量不出任何東西，所以刻意拿掉 snap，只留 round+clip 這半段）。
    公式出處：scratch_lacot_rollout.py:338（_i_zn_to_cell 的 round/clip 那行，同一條公式）。
    clip-到邊界（而非另開一個『出界』分類）沿用 GeoEnergy._sample 的 padding_mode='border'
    這個既有慣例：lacot/refine_grad.py:76-79。"""
    idx = np.round((np.asarray(pts, np.float64) - lo_np) / span_np * (shape_arr - 1)).astype(np.int64)
    return np.clip(idx, 0, shape_arr - 1)


def legal_fraction(pts):
    """pts: [...,2] 正規化座標 → (合法率, n點數)。"""
    idx = zn_to_cell_batch(pts)
    ok = occ[idx[..., 0], idx[..., 1]]
    return float(ok.mean()), int(ok.size)


def zn_to_cell_one(z, allow_snap=False):
    """單點版本，給 BFS 起點用。allow_snap=True 時若落牆格才 snap（真實資料點理論上不該發生，
    發生的話用這個保底並且【計數回報】，⛔ 不靜默蓋過去）。回 (cell_tuple, was_snapped)。"""
    idx = zn_to_cell_batch(z[None])[0]
    c = tuple(int(v) for v in idx)
    if occ[c] or not allow_snap:
        return c, (not occ[c])
    free_cells = np.argwhere(occ)
    nn_idx = free_cells[np.abs(free_cells - idx).sum(1).argmin()]
    return tuple(int(v) for v in nn_idx), True


# ─────────────────────────────────────────────────────────────────
# 3. 抽 N_PAIRS 對真軌跡片段（切窗邏輯逐字對齊 make_batch）
# ─────────────────────────────────────────────────────────────────
hr(f"3. 抽 {N_PAIRS} 對真軌跡片段（T_CAP={T_CAP}，CHUNK={CHUNK}）")
print("  起點/終點抽法出處：scratch_lacot_rollout.py:452-465（make_batch，官方 GCBC 抽法：")
print("  同一條軌跡內、[r+1, 軌跡結尾] 均勻抽 goal；F6 修正＝越界 clamp 不重抽）。")
print("  固定 T_CAP 點的內插重採樣出處：scratch_lacot_rollout.py:484-489（F7 修正：一律線性內插")
print("  取 T_CAP 個點，⛔ 不是按索引取整數點 —— 避免長度資訊從點數/mask 洩漏）。")

rng = np.random.default_rng(SEED)


def sample_rows_goals(n):
    rows, goals = [], []
    n_retry = 0
    while len(rows) < n:
        r = int(rng.integers(0, N))            # 均勻抽（_sample_r 預設分支，rollout.py:438-440）
        te = int(traj_end[r])
        if te - r < CHUNK:
            n_retry += 1
            continue
        _d = rng.random()
        gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
        gr = max(gr, min(r + CHUNK, te))
        rows.append(r)
        goals.append(gr)
    return np.array(rows), np.array(goals), n_retry


def build_traj(rows, goals):
    """scratch_lacot_rollout.py:484-489 逐字對齊（改成單獨函式，行為相同）。"""
    n = len(rows)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                     T_CAP, axis=1).reshape(n, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    return traj


rows_A, goals_A, retry_A = sample_rows_goals(N_PAIRS)
rows_B, goals_B, retry_B = sample_rows_goals(N_PAIRS)
trajA = build_traj(rows_A, goals_A)   # [N_PAIRS, T_CAP, 2] 正規化
trajB = build_traj(rows_B, goals_B)
print(f"  A/B 各抽到 {N_PAIRS} 段（CHUNK 篩掉重試次數：A={retry_A}, B={retry_B}）")
print(f"  片段長度（原始索引步數，gr-r）：A p50={np.median(goals_A - rows_A):.0f}"
      f" B p50={np.median(goals_B - rows_B):.0f}")

trajA_t = torch.from_numpy(trajA)
trajB_t = torch.from_numpy(trajB)
mask_full = torch.zeros(N_PAIRS, T_CAP, dtype=torch.bool)   # 全 False＝全部真點，同 rollout.py:500

with torch.no_grad():
    etA = etarget(trajA_t, mask_full)     # [N_PAIRS, K, D_MODEL]
    etB = etarget(trajB_t, mask_full)
    pts_realA = decode(etA)               # 真軌跡 encode→decode 直接還原（roundtrip_gate 同款用法，
    pts_realB = decode(etB)               # 出處：scratch_lacot_rollout.py:704-722 roundtrip_gate()）


# ─────────────────────────────────────────────────────────────────
# 4. 三組合法率
# ─────────────────────────────────────────────────────────────────
hr("4. 三組合法率")
print("  設計選擇：『真軌跡直接 decode』＝真軌跡 encode(traj_enc→e_pooler)→decode(u_dec) 的")
print("  roundtrip 還原點（不是拿原始插值點本身 —— 那樣必然 100% 合法，測不出 decoder 的行為），")
print("  precedent＝roundtrip_gate()，scratch_lacot_rollout.py:704-722。")

real_pts = np.concatenate([pts_realA.numpy(), pts_realB.numpy()], axis=0).reshape(-1, 2)
real_legal, real_n = legal_fraction(real_pts)

interp_pts_by_t = {}
with torch.no_grad():
    for t in (0.25, 0.5, 0.75):
        et_t = (1.0 - t) * etA + t * etB     # 線性插值，任務規格直接指定的公式（非 file:line 出處）
        interp_pts_by_t[t] = decode(et_t).numpy()
interp_pts_all = np.concatenate([interp_pts_by_t[t] for t in (0.25, 0.5, 0.75)], axis=0).reshape(-1, 2)
interp_legal, interp_n = legal_fraction(interp_pts_all)

print("  設計選擇：隨機 u 控制組＝與 et_A/et_B 逐維（K×D_MODEL）mean/std 匹配的高斯雜訊")
print("  （保留邊際尺度、打散跨 K/維度的結構相關），precedent＝experiments/probe_u.py:92,110-111")
print("  （U_MEAN/U_STD + torch.randn 的 'matched' 控制組寫法，同一個 repo 既有慣例）。")
et_pool = torch.cat([etA, etB], 0)
U_MEAN, U_STD = et_pool.mean(0, keepdim=True), et_pool.std(0, keepdim=True)
g_rand = torch.Generator().manual_seed(RAND_U_SEED)
et_rand = U_MEAN + U_STD * torch.randn(N_PAIRS, K, D_MODEL, generator=g_rand)
with torch.no_grad():
    pts_rand = decode(et_rand).numpy()
rand_legal, rand_n = legal_fraction(pts_rand.reshape(-1, 2))

print(f"\n  {'組別':32s}{'n_waypoints':>12s}{'合法率':>10s}")
print(f"  {'真軌跡 direct-decode（控制組↑）':32s}{real_n:>12d}{real_legal:>10.1%}")
print(f"  {'插值 decode（量測值）':32s}{interp_n:>12d}{interp_legal:>10.1%}")
print(f"  {'隨機 u decode（控制組↓）':32s}{rand_n:>12d}{rand_legal:>10.1%}")
print("\n  插值合法率按 t 拆開（附加資訊，非驗收必要）：")
for t in (0.25, 0.5, 0.75):
    lf, ln = legal_fraction(interp_pts_by_t[t].reshape(-1, 2))
    print(f"    t={t:.2f}   n={ln:6d}   合法率={lf:.1%}")

# ⛔ 兩個儀器 gate —— 如實報，不修飾。
hr("儀器 GATE 判定")
gate1_ok = real_legal >= GATE1_MIN_REAL_LEGAL
gate2_margin = real_legal - rand_legal
gate2_ok = gate2_margin >= GATE2_MIN_MARGIN
print(f"  gate 1  真軌跡合法率 {real_legal:.1%} ≥ {GATE1_MIN_REAL_LEGAL:.0%} ？ "
      f"{'PASS' if gate1_ok else 'FAIL'}")
print(f"  gate 2  真軌跡−隨機u 合法率差 {gate2_margin:+.1%} ≥ {GATE2_MIN_MARGIN:.0%} ？ "
      f"{'PASS' if gate2_ok else 'FAIL'}")
INSTRUMENT_VALID = gate1_ok and gate2_ok
if not INSTRUMENT_VALID:
    print("\n⛔⛔⛔ INSTRUMENT INVALID ⛔⛔⛔")
    if not gate1_ok:
        print(f"  原因：真軌跡 direct-decode 合法率 {real_legal:.1%} 沒到 {GATE1_MIN_REAL_LEGAL:.0%}"
              f" ⇒ decoder 本身重建就有問題，下面的插值/BFS 數字不可信。")
    if not gate2_ok:
        print(f"  原因：合法率差只有 {gate2_margin:+.1%}，沒到 {GATE2_MIN_MARGIN:.0%} 門檻"
              f" ⇒ 佔據圖判準可能太寬鬆（隨機雜訊都能解出合法路），legal_fraction 這把尺不夠敏感。")
else:
    print("\n  ✓ 兩個 gate 都過，下面的插值合法率與相關數字視為可信。")


# ─────────────────────────────────────────────────────────────────
# 5. latent 距離 vs BFS 距離
# ─────────────────────────────────────────────────────────────────
hr("5. latent 距離 ‖et_A−et_B‖ vs 起點 BFS 距離")
print("  latent 距離：攤平 [K,D_MODEL] 成一個向量後取歐氏距離 —— 任務規格寫的就是 ‖et_A−et_B‖")
print("  這個記法本身（非 file:line 出處）；『攤平 K×D_MODEL 當一個向量』這個處理方式沿用")
print("  scratch_lacot_rollout.py:893 的 et.reshape(B,-1) 用法（precedent，⛔ 該行另外做了")
print("  F.normalize 算 cosine，本探針按規格【不】做 normalize，是純 L2 距離）。")
lat_dist = (etA.reshape(N_PAIRS, -1) - etB.reshape(N_PAIRS, -1)).norm(dim=1).numpy()

print("\n  BFS 距離：兩軌跡『起點』（片段第 0 點，即 traj[:,0]，正規化座標，數值上＝軌跡片段的")
print("  真實起始狀態 s＝(OBS[r]-mu)/sd）在佔據圖上的 4-鄰 BFS 步距。")
print("  出處：grid_bfs()＝lacot/subgoal.py:35-52；intent.py 對 BFS 的唯一來源聲明＝lacot/intent.py:8。")

bfs_dist = []
n_unreachable = 0
n_snap = 0
for i in range(N_PAIRS):
    cA, snapA = zn_to_cell_one(trajA[i, 0], allow_snap=True)
    cB, snapB = zn_to_cell_one(trajB[i, 0], allow_snap=True)
    n_snap += int(snapA) + int(snapB)
    dist_map = grid_bfs(occ, cA)
    d = dist_map.get(cB, None)
    bfs_dist.append(d)
    if d is None:
        n_unreachable += 1
bfs_dist = np.array([np.nan if d is None else d for d in bfs_dist])

print(f"  起點落牆格需要 snap 的次數：{n_snap}/{2 * N_PAIRS}"
      f"（理論上應為 0——起點是真實資料點，本來就在建圖時被算進自由格；非 0 要明講，不能沉默蓋過）")
print(f"  BFS 不連通的對數：{n_unreachable}/{N_PAIRS}"
      f"（{'無' if n_unreachable == 0 else '有，這些對已從相關係數計算中排除'}）")

valid_mask = ~np.isnan(bfs_dist)
n_valid = int(valid_mask.sum())
if n_valid < 3:
    print(f"\n  ⛔ 有效對數只有 {n_valid} < 3，Spearman 相關算不出有意義的數字，停在這裡明講，不硬算。")
    rho, pval = float("nan"), float("nan")
else:
    rho, pval = spearmanr(lat_dist[valid_mask], bfs_dist[valid_mask])
    print(f"\n  n={n_valid}   Spearman rho={rho:.3f}   p={pval:.4g}")
    print(f"  latent 距離：mean={lat_dist[valid_mask].mean():.3f} std={lat_dist[valid_mask].std():.3f}")
    print(f"  BFS 距離（格）：mean={np.nanmean(bfs_dist):.1f} p50={np.nanmedian(bfs_dist):.1f}"
          f" max={np.nanmax(bfs_dist):.0f}")


# ─────────────────────────────────────────────────────────────────
# 簡表彙總
# ─────────────────────────────────────────────────────────────────
hr("彙總簡表")
print(f"  INSTRUMENT_VALID = {INSTRUMENT_VALID}")
print(f"  {'組別':32s}{'n':>10s}{'合法率':>10s}")
print(f"  {'真軌跡 direct-decode':32s}{real_n:>10d}{real_legal:>10.1%}")
print(f"  {'插值 decode（t=.25/.5/.75 合併）':32s}{interp_n:>10d}{interp_legal:>10.1%}")
print(f"  {'隨機 u decode':32s}{rand_n:>10d}{rand_legal:>10.1%}")
print(f"  latent-vs-BFS Spearman rho={rho:.3f}  n={n_valid}  p={pval:.4g}")
print(f"\n耗時 {time.time() - T0:.1f}s")
