"""LaCoT SUCCESS-RATE eval (the real OGBench metric, not BC-MSE).
Train LaCoT (state: contrastive e_target frozen -> flow -> refine -> action MLP),
then ROLL OUT in the pointmaze env and measure success rate, comparing:
  * (s,g)-only floor  = ahead(cond, ZERO-u)      [GCBC / depth-0 floor]
  * LaCoT refine R = 0/1/3/5/8                        [test-time scaling]
Success = the env's own info['success']. Receding-horizon CHUNK execution.
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lacot repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.model import RefineOperator
from lacot import dev_eval as DE
import ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
# ⚠️ 環境改成變數（主人 2026-08-23：「跑個 large，跑個 stitch」）。
# ★ 為什麼要換環境：medium-navigate 上誠實 BC 地板已經 0.900、天花板 1.000
#   ⇒ 只剩 0.1 的空間，而 seed 噪聲就有 0.1 ⇒ ⛔ 這個任務量不出 u 的價值，
#   不管 u 好不好。stitch 是刻意設計成「資料裡沒有完整路徑、必須把片段接起來」的，
#   BC 在那上面本來就會爛 —— 而接片段正是 u 該做的事。
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-navigate-v0")
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "資料集最後一筆不是 terminal ⇒ traj_end 尾巴是未初始化的記憶體"
MAX_TRAIN_T = int((ends - starts + 1).max())   # 訓練時 T 到得了的上限＝最長的一條軌跡
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)
# 🚨 K 與 COND 改成環境變數，⛔ 不要再靠手改檔案跑不同設定 ——
#    2026-08-23 就踩到：檔案裡寫 K=64，但 FINDINGS 記的三 seed 成功率標的是 K=4，
#    對不起來，等於那組 0.85 沒辦法照原樣重跑（改了沒記＝下次的自己找不回來）。
#    預設用【文件記錄的那組】K=4；⚠️ K=4 是 u_dim sweep 量到 flow 對齊最好的（0.934 vs 0.523）。
K = int(os.environ.get("LACOT_K", 4))
COND = int(os.environ.get("LACOT_COND", 256))
# CHUNK 也做成環境變數：官方 GCBC 是【每步】重新決策（等於 CHUNK=1），
# LaCoT 是一次輸出 4 步 —— 而 pointmaze 的 observation 只有 (x,y)、沒有速度，
# 球又有慣性 ⇒ 對 Markovian policy 是 POMDP，分塊等於做了時間平滑。
# ⇒ 這可能就是「官方 0.15 vs 我們 0.85」的原因。主人 2026-08-23 核可查。
CHUNK = int(os.environ.get("LACOT_CHUNK", 4))
# 🚨 2026-08-24 對齊官方（主人：「缺口改掉，下次不要重新再被影響到」）。
#    ⛔ 此後的數字跟 08-23 之前的【不可直接比】—— goal 抽法、T_CAP、eval 集數三項都變了。
# ⭐ T_CAP = Perceiver 一次讀幾個軌跡點（⛔ 不是截斷，是整條路等距取樣成這麼多點）。
#    `[實測 2026-08-24]` eval 的 oracle 路徑 medium-navigate 平均 126 步、large-stitch 平均 255 步；
#    T_CAP=16 會讓 large-stitch 變成每 15.9 步一個點，而訓練時每 2.4 步一個 ⇒ 密度差 6.6 倍。
#    ⚠️ Perceiver 的 cross-attention 成本 K×T（K 固定）⇒ 線性，放大吃得下。
# 🚨 2026-08-28：預設 256 → 128。主人 8/24 就裁示 128，但一直只能靠 env var 記得帶，
#    忘了帶就會靜默跑成 256 ⇒ 把裁示放進預設。⛔ 這【會】改變不帶 env var 時的行為。
T_CAP_REQ = int(os.environ.get("LACOT_TCAP", 128))
# ⭐ 2026-08-26：開發尺 ＋ 兩個前置量測（預設關，⛔ 不影響既有行為）
#    DEV_EVAL  用 lacot/dev_eval.py 的幾百題取代官方 5 task × 50 seed
#    PREREQ    跑 D0（head 接不接得住更好的 u）與 D4（flow 有沒有多樣性）
DEV_EVAL = int(os.environ.get("LACOT_DEV_EVAL", 0))
DEV_PER_TIER = int(os.environ.get("LACOT_DEV_PER_TIER", 80))
PREREQ = int(os.environ.get("LACOT_PREREQ", 0))
PREREQ_N = int(os.environ.get("LACOT_PREREQ_N", 16))
# ⭐ 2026-08-28 三個新旋鈕。⛔ 預設一律等於【現行行為】—— 舊結果照樣重現得出來。
#
# ENC_OBJ ── stage 1 的訓練目標。這是 2026-08-28 那個直接測量的下游動作：
#   實測 u 的資訊回收率只有 18.5%（門檻 25%）⇒ (s,g)↔τ InfoNCE 的最優解不要求裝路徑。
#   （完整經過見 docs/EXPERIMENT-INDEX.md Q1，⛔ 那條已封盤，別再量第六次。）
#     sg_infonce  現行：讓 (s,g) 認得出 τ。⇒ 路徑資訊可有可無 ⇒ 它就沒有
#     recon       改成「解得回那 128 個座標點」⇒ 解不回就壓不下 loss ⇒ 路徑資訊【必要】
#     recon_ictr  recon ＋ 同一條 τ 兩個增強視角要互認（instance-level，⛔ 不是 (s,g)↔τ）
#   ⚠️ recon/recon_ictr 會【移除】(s,g)↔τ InfoNCE —— 關鍵在移除，不在加。
#      對抗一個目標不如拿掉它（8/26 實測：難負樣本疊在權重 1.0 的收縮項上，加了不動）。
ENC_OBJ = os.environ.get("LACOT_ENC_OBJ", "sg_infonce")
W_ICTR = float(os.environ.get("LACOT_W_ICTR", 0.2))
ICTR_SIGMA = float(os.environ.get("LACOT_ICTR_SIGMA", 0.05))
# LEARNED_REFINE=0 ── 刪掉 l_refine 與 l_cons，並停用 refine 網路。
#   🚨 l_refine 拿 flow 【隨機抽】的 u（很可能是另一條路）餵 head，卻要求輸出資料那條路的動作
#      ⇒ 明文在教 head「不管 u 說什麼都輸出 cond 的答案」，權重 1.0，跟 l_anchor 一樣重。
#   ⚠️ 現在它空轉（u 只編碼起終點 ⇒ 兩條路的 u 一樣），但 ENC_OBJ 一改好它會立刻變成
#      第二個 bypass ⇒ ⛔ 換目標【必須】同一輪把它拿掉，只做一半很可能白做。
#   ⭐ 佐證：devday-ms0 把 refine 反向跑是 0.488、正向只有 0.292（bc 地板 0.476）
#      ⇒ 學出來的 refine 是【主動有害】的，不是沒作用。
LEARNED_REFINE = int(os.environ.get("LACOT_LEARNED_REFINE", 1))
# COND_DROP ── l_anchor 那一路以 p 機率把整個 cond 歸零，逼 head 從 u 讀路徑。
#   ⛔ 不做 u-dropout —— 那會反過來教 head 繞開 u。
COND_DROP = float(os.environ.get("LACOT_COND_DROP", 0.0))
# BC_INDEP ── bc 地板改用獨立 optimizer ＋ 獨立 grad-clip。
#   主人 8/24 對 floor 的定義是「真正 BC 能到達的」⇒ 它現在跟主模型共用 opt2 與全域 clip
#   ⇒ ⛔ 不是獨立 baseline。⚠️ 開了之後 bc 的數字跟歷史結果【不可直接比】⇒ 預設仍是 0。
BC_INDEP = int(os.environ.get("LACOT_BC_INDEP", 0))
# ⭐ SUBGOAL ── 主人 2026-08-28 的分段規劃：「想一個 u、走一段或到 subgoal、再 think again」。
#   `[實測]` 他指到的病是真的（見 docs/EXPERIMENT-INDEX.md Q2）：
#     訓練時 u 要表示的路長中位數 7.5、p99 33.2；而 large-stitch 最難那層中位 48
#     ⇒ 那層【100%】的題超出訓練 p99 ⇒ u 被要求表示訓練時沒見過那麼長的東西。
#   ⇒ 修法是拆成兩層（⛔ 不是遞迴）：
#       長程層 cond=(現在, 最終目標)，只產出 decode(u_long) 的幾何 ⇒ 取 subgoal
#       短程層 cond=(現在, subgoal)，flow → head 出動作 ⇒ 回到訓練分布內
#     latent  subgoal 由長程 u 解碼出的路徑取（沿【弧長】，⛔ 不是第 k 個點）
#     bfs     subgoal 由 BFS 在格圖上生 ⇒ ⭐ 對照組：拆「贏的是階層化還是長程推理」
#             ⛔ 少了它，latent 贏了我們會把功勞記錯人
SUBGOAL = os.environ.get("LACOT_SUBGOAL", "")
assert SUBGOAL in ("", "latent", "bfs", "conf", "conf2"), \
    f"⛔ LACOT_SUBGOAL 只能是空/latent/bfs/conf/conf2，收到 {SUBGOAL}"
# ⭐ conf2 ── 主人 8/29 下午的統一版：「g 信心夠高就直接走到底，不夠就挑最遠但信心夠高的點」。
#   每次重想 fresh 抽 M 份（⛔ 不接續修 —— 治「計畫殘骸」病）、修完、判信心。門檻自校準。
# 間距用訓練分布的中位數（exp_span_gap.py 實測 7.5，原始座標單位）⇒ 短程層坐在資料最肥的地方
DELTA_SUB = float(os.environ.get("LACOT_DELTA_SUB", 7.5))
# ⭐ conf ── 信心選點（主人 2026-08-29）：抽 M 份長程計畫，subgoal 取「窗內共識最高」的點。
#   固定弧長 7.5 是它的固定近似；窗 [LO,HI]×DELTA_SUB 下限擋原地共識、上限擋短程 cond 出分布。
SUB_M = int(os.environ.get("LACOT_SUB_M", 4))
SUB_CONF_LO = float(os.environ.get("LACOT_SUB_CONF_LO", 0.5))
SUB_CONF_HI = float(os.environ.get("LACOT_SUB_CONF_HI", 1.5))
# ⭐ 歸因對照（主人 8/29 晚）：分段模式的【短程】改走 bc head ——「bc＋BFS 中繼點」
#   回答 0.750 的 +4 是短程 u 的功勞、還是分段結構本身的功勞。⛔ 只影響分段 arm。
SUB_POLICY = os.environ.get("LACOT_SUB_POLICY", "")
assert SUB_POLICY in ("", "bc"), f"⛔ LACOT_SUB_POLICY 只能是空/bc，收到 {SUB_POLICY}"
# 🚨 2026-08-28 修（單位錯）：SubgoalPlanner.observe() 是【每個 chunk】呼叫一次，
#    ⛔ 不是每個 env step —— 呼叫端是 policy，而 policy 一次回 CHUNK 步。
#    ⇒ 舊預設 cap=40 讀起來像「40 步」、實際是 160 個 env step（CHUNK=4）
#      ⇒ 兩個「強制重想」的觸發根本按不下去，而它【不會報錯】。
#    ⚠️ 這【會】改變 SUBGOAL 分段 arm 的行為（cap 40→10、stuck 12→3，皆為 chunk 數）。
#      ⛔ 但那個 arm 在今天之前一步都沒爬過（見 policy_chunk 的 R=0 凍結 bug）⇒ 沒有可重現的舊結果。
SUB_CAP = int(os.environ.get("LACOT_SUB_CAP", 10))        # chunk 數
SUB_STUCK = int(os.environ.get("LACOT_SUB_STUCK", 3))     # chunk 數
GRAD_R = int(os.environ.get("LACOT_GRAD_R", 50))       # 長程 u 爬幾步
GRAD_ETA = float(os.environ.get("LACOT_GRAD_ETA", 0.1))
GRAD_LAM = float(os.environ.get("LACOT_GRAD_LAM", 0.3))
# ⭐ 病一快篩（主人 8/29 開跑）：E_geo 的 length 項是唯一不飽和的梯度 ⇒ 接力爬幾千步
#   的漸近行為＝把路越磨越短（G3 的 S1 量到路蜷縮在 1.15 內）。w_len=0 直接驗這個假說。
W_LEN = float(os.environ.get("LACOT_W_LEN", 0.3))
# ⭐ 病二快篩（主人 8/29 開跑）：所有帶 u 的 arm 都「到附近進不了洞」而 bc 進得去
#   ⇒ 終局讓 u 退位、換獨立 bc head 收尾。>0 開啟（原始座標單位；半格＝2.0）。
FINISH_R = float(os.environ.get("LACOT_FINISH_R", 0.0))
# ⭐ 終局接管的模式（主人 8/29 下午）：
#   bc        換獨立 bc head 收尾（已實測 flat 0.390→0.520）
#   resample  fresh 重抽一份短計畫、不爬不碰快取（R=0 語義）——
#             驗「終局病是計畫殘骸、不是 head 權重」：這個過了，統一信心機制就不需要第二顆 head
FINISH_MODE = os.environ.get("LACOT_FINISH_MODE", "bc")
assert FINISH_MODE in ("bc", "resample"), f"⛔ LACOT_FINISH_MODE 只能是 bc/resample，收到 {FINISH_MODE}"
_FIN_COUNT = [0]   # ⭐ 終局接管的觸發次數 —— ⛔ 開關條件寫錯＝永不觸發＝快篩白跑，要能看見
# ⭐ LOAD_CKPT ── 載入已訓好的權重，跳過訓練，只跑評估。
#   🚨 起因：交接記著「今天為了換探針重訓了三輪」。換一個 arm、換一個判準就重訓一次，
#      是這個 repo 從 8/23 就有的浪費 —— 而且它讓「補一個對照」的成本高到我們不想補。
#   ⚠️ 載入時會逐項比對 ckpt 的 cfg，對不上就停 —— ⛔ 形狀對得上不代表是同一個模型。
LOAD_CKPT = os.environ.get("LACOT_LOAD_CKPT", "")
# ⭐ GRAD_REFINE ── 用主人的梯度爬坡取代 learned refine（⛔ 跟 SUBGOAL 正交，可單獨開）。
#   ⇒ 三種部署因此可以同輪對打，⛔ 而且它們共用同一顆 ckpt、同一批題、同一條噪聲流：
#       flat-grad  GRAD_REFINE=1 SUBGOAL=""       一次規劃整條路（主人問的那個做法）
#       S1         GRAD_REFINE=1 SUBGOAL=latent   分段，subgoal 從長程 u 解出來
#       S0         GRAD_REFINE=1 SUBGOAL=bfs      分段，subgoal 由 BFS 生 ⇒ 拆功勞用
GRAD_REFINE = int(os.environ.get("LACOT_GRAD_REFINE", 0))
# ⭐ warm-start：第一個 chunk 從 flow 樣本起爬 GRAD_R 步，之後【接續上一個 chunk 的 u】只爬這麼多。
#   🚨 為什麼一定要有：每 chunk 都從頭爬 50 步 ⇒ 300 題要三十幾小時，⛔ 跑不完。
#   ⭐ 而且它本來就是對的形狀 —— u 是一條【邊走邊修】的計畫，⛔ 不是每四步重想一次的東西。
GRAD_R_WARM = int(os.environ.get("LACOT_GRAD_R_WARM", 10))
# ⭐ 只跑指定的 tier（例：LACOT_DEV_TIERS=2）。⛔ 空＝全部。
#   ⚠️ 前兩層 bc 已經 0.93/0.87，⛔ 沒有空間；效應只可能出現在最難那層。
DEV_TIERS = os.environ.get("LACOT_DEV_TIERS", "")
T_CAP = min(T_CAP_REQ, MAX_TRAIN_T)
if T_CAP != T_CAP_REQ:
    print(f"⚠️ T_CAP {T_CAP_REQ} → {T_CAP}（夾到本資料集最長軌跡 {MAX_TRAIN_T}，"
          f"⛔ 否則 eval 會用到沒訓練過的 pos_emb）", flush=True)
# 🚨 F1（2026-08-24 subagent 稽核）：T_CAP 超過【訓練時到得了的長度】的話，多出來那段
#    positional embedding 從沒收過梯度（large-stitch 每條軌跡只有 201 步，而 eval 的
#    oracle 路徑 240~256 步 ⇒ pos_emb[201:256] 還是隨機初始化，且那是最靠近目標的一段）。
#    ⛔ 這支腳本的常數區在資料載入【之後】（跟 exp_etarget_ceiling.py 相反）
#    ⇒ 夾子必須寫在這一行下面，寫到上面去會 NameError（2026-08-24 smoke 19734 實測炸過）。
B, D_MODEL, TEMP, ADIM = 64, 256, 0.1, 2
DIM = K * D_MODEL

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        # 🚨 官方抽法（OGBench `impls/utils/datasets.py` GCDataset.sample_goals,
        #    geom_sample=False 分支）：在【現在的下一步 → 軌跡結尾】之間【均勻】抽。
        #    gcbc.py 的 actor_p_trajgoal=1.0 / actor_geom_sample=False，且 hyperparameters.sh
        #    裡 pointmaze 的 GCBC 沒有任何 override ⇒ 這就是官方值。⛔ 不要改回 geometric。
        _d = rng.random()
        gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
        # 🚨 F6：原本是 `if gr - r < CHUNK: continue`，而 continue 會【連起點一起重抽】
        #    ⇒ 越靠近軌跡結尾的起點越容易被丟掉。官方不重抽 ⇒ 改 clamp，起點分布不動。
        gr = max(gr, min(r + CHUNK, te))
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    # 🚨 F7（2026-08-26，主人核可修）：舊版是
    #      np.unique(np.linspace(r, g, min(T_CAP, g-r+1)).round())
    #    ⇒ 取樣點數 ＝ min(T_CAP, L+1)，而 T_CAP(預設 256) 永遠 ≥ 最長軌跡(201)
    #    ⇒ 點數【恆等於 L+1】；再加上 mask 上標著「這條路有幾個真點」
    #    ⇒ 長度是直接寫在輸入上的。`[實測]` 光數點數當分數 ⇒ 同題排序 1.000。
    #    ⛔ 這跟 8/25 subagent 在 exp_value_u.py 抓到的是同一行 code 的同一個病。
    #
    #    🚨 主線的後果比 Step 1 嚴重：
    #      訓練時 u ＝ e_target(真軌跡) ⇒ 帶著「這條路多長」，head 可以靠它決定動作
    #      eval 時 u ＝ flow.sample(cond) ⇒ 沒有 mask 這回事 ⇒ 那個特徵消失
    #      ⇒ head 依賴一個 eval 時不存在的特徵 ⇒ 我們自己造出來的分布不匹配。
    #
    #    修法：一律取【固定 T_CAP 個點】，而且用【插值】不取整。
    #      ⇒ 點數對所有樣本一樣、mask 全真 ⇒ 點數不洩漏
    #      ⇒ 插值後每個點的座標都不同 ⇒ 「數不重複的點」也不洩漏
    #      ⭐ 長度資訊還在，它藏在【相鄰點的間距】裡 —— 那本來就是該讀的東西。
    #    ⭐ 順便修掉 F1：現在 pos_emb[0:T_CAP] 每一格都會收到梯度。
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    T_CAP, axis=1).reshape(B, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])      # ⚠️ 夾在終點內，⛔ 不可以跨到下一條軌跡
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    mask = np.zeros((B, T_CAP), bool)                # 全 False ＝ 全部都是真點
    # ⚠️ 自檢：一響就代表洩漏又回來了（⛔ 別把它拿掉）
    assert mask.shape[1] == T_CAP and not mask.any(), "⛔ 取樣點數不再固定 ⇒ 長度會從 mask 洩漏"
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        x = torch.cat([cond, u.reshape(u.shape[0], -1)], -1)
        return self.net(x).reshape(-1, CHUNK, ADIM)

SEED = int(os.environ.get("LACOT_SEED", 0))
# refine 的 consistency 目標：self = 跟下一刻的自己比（原版）；ema = 跟 EMA 副本比。
# `[實測 2026-08-23]` anti-collapse 掃了 11 個變體，只有 ema 系列三項全過
#   （塌度 0.056 / 路徑資訊贏內插 / cos 真et +0.833），byol 那組塌度 0.48~0.69 全塌。
# ⚠️ 但那次掃描 **每格只有 1 個 seed**，⇒ ema 內部 m99/m996/m999 誰最好還不算數，
#    只有「ema 系列 vs byol 系列」那個差距大到不可能是 seed 噪聲。
CONS = os.environ.get("LACOT_CONS", "self")
EMA_M = float(os.environ.get("LACOT_EMA_M", 0.996))
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
print(f"設定：seed={SEED} cons={CONS} ema_m={EMA_M} K={K} COND={COND}", flush=True)
traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)
# ⭐ A1（2026-08-26）：VICReg 的 variance ＋ covariance 兩項。
#    出處：PLDM(arXiv 2502.14819) 的消融 —— 完整 98.0% / 拿掉 variance 13.4% / 拿掉 covariance 29.2%
#    ⇒ 這在 latent 世界模型上不是裝飾，是決定生死的。而我們的 encoder 一項都沒有。
#    ⛔ 不含 invariance 那一項 —— 那由既有的 contrastive 負責。
#    ⚠️ 8/26 實測：u 的有效維度只有 0.092（1024 維裡實際只用約 6 個方向）。
#    ⚠️ 但文獻也說「低有效秩本身不代表失敗」⇒ 所以判準是【維度升 + 成功率也升】，
#      ⛔ 只有維度升不算數。
W_VAR = float(os.environ.get("LACOT_W_VAR", 0.0))
W_COV = float(os.environ.get("LACOT_W_COV", 0.0))


def vicreg_terms(z, eps=1e-4):
    z = z.reshape(len(z), -1)
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    v = F.relu(1.0 - std).mean()                      # 每個維度的標準差要 ≥ 1
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return v, (off ** 2).sum() / d                    # 維度之間要去相關


@torch.no_grad()
def eff_dim(z):
    """有效維度（participation ratio，正規化到 0~1）⇒ 防塌的守門員。"""
    z = z.reshape(len(z), -1).float()
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z) ** 2
    return float((s.sum() ** 2) / (s ** 2).sum() / len(s))


def _decoder_health():
    """decoder 到底有沒有在讀 u，還是學成了「不管給什麼都吐同一條平均路」。回 (rmse, shuf, gap)。

    🚨 2026-08-28 修（探針跑在載入【之前】）：舊版這段緊接在 stage 1 後面，而 LOAD_CKPT 模式下
       _S1=0 ⇒ stage 1 整個跳過 ⇒ 它評的是【隨機初始化】的 u_dec，然後照樣印判決；
       真權重要到 LOAD_CKPT 那一段才進來 ⇒ ⛔ 只評估模式下，真正要當 E_geo 眼睛的那顆
       decoder【一個檢查都沒有】。
    ⚠️ 疊加：u_dec.eval() 之後，traj_decoder.py 裡的塌陷 assert 因 self.training=False 永不執行。
    ⇒ 搬到載入之後呼叫。⭐ 為了讓訓練【逐位元不變】，這裡用自己的 numpy RNG，
      並把 torch 的 RNG 狀態存起來再還原 —— ⛔ 探針不可以擾動主流程的取樣流。
    """
    from lacot.traj_decoder import ctx_usage_probe
    _cpu_state = torch.get_rng_state()
    _cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        _prng = np.random.default_rng(20260828)
        with torch.no_grad():
            _t, _m2, _s2, _g2, _ = make_batch(_prng)
            return ctx_usage_probe(u_dec, etarget(_t, _m2), _t)
    finally:
        torch.set_rng_state(_cpu_state)
        if _cuda_state is not None:
            torch.cuda.set_rng_state_all(_cuda_state)


def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
# ⭐ ENC_OBJ=recon* 要一顆 decoder：解得回 128 個座標點，才代表 u 真的裝了那條路。
#    ⛔ 它不是暫時的鷹架 —— E_geo（幾何 energy）也靠同一顆把 u 解成可微的座標點。
u_dec = None
if ENC_OBJ.startswith("recon"):
    from lacot.traj_decoder import TrajDecoder
    u_dec = TrajDecoder(D_MODEL, T_CAP).to(device)
    u_dec.check_p = 0.01
    # ⛔ sg_c / q_pooler 不進 optimizer —— (s,g)↔τ InfoNCE 整條移除，⛔ 不是降權重。
    opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, u_dec) for p in m.parameters()], lr=1e-3)
elif ENC_OBJ != "sg_infonce":
    raise ValueError(f"⛔ LACOT_ENC_OBJ 只能是 sg_infonce/recon/recon_ictr，收到 {ENC_OBJ}")

print(f"stage 1 e_target 目標={ENC_OBJ} ...  w_var={W_VAR} w_cov={W_COV}"
      + (f" w_ictr={W_ICTR} sigma={ICTR_SIGMA}" if ENC_OBJ == "recon_ictr" else ""), flush=True)
_S1 = 0 if LOAD_CKPT else int(os.environ.get("LACOT_STEPS1", 1500))
_ed_hist, logits = [], None
for stp in range(_S1):
    traj, mask, s, g, _ = make_batch(rng)
    et = etarget(traj, mask)
    if ENC_OBJ == "sg_infonce":
        q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        _main = loss
    else:
        _main = (u_dec(et) - traj).pow(2).mean()    # ⭐ 重建 128 個座標點
        loss = _main
        if ENC_OBJ == "recon_ictr":
            # instance-level：同一條 τ 的兩個【加噪視角】要互認。
            # ⛔ 這跟被移除的那項不同 —— 它認的是「同一條軌跡」，⛔ 不是「同一組 (s,g)」，
            #    所以它【不會】獎勵把路徑資訊丟掉。
            # ⛔ 不用時間反轉／空間翻轉當增強：方向與迷宮形狀正是要保的資訊。
            v1 = etarget(traj + ICTR_SIGMA * torch.randn_like(traj), mask)
            v2 = etarget(traj + ICTR_SIGMA * torch.randn_like(traj), mask)
            lg = (F.normalize(v1.reshape(B, -1), dim=1)
                  @ F.normalize(v2.reshape(B, -1), dim=1).t()) / TEMP
            loss = loss + W_ICTR * 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
            logits = lg
    if W_VAR > 0 or W_COV > 0:                      # ⭐ A1：防塌正則
        _v, _c = vicreg_terms(et)
        loss = loss + W_VAR * _v + W_COV * _c
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    if _S1 and ((stp + 1) % max(_S1 // 4, 1) == 0 or stp == 0):
        _ed_hist.append(eff_dim(et))                # ⚠️ 守門員：⛔ 別等結果出來才發現 u 塌了
        _lbl = "ctr" if ENC_OBJ == "sg_infonce" else "recon-mse"
        print(f"  step {stp+1}  {_lbl} {_main.item():.4f}  總 {loss.item():.4f}"
              f"  u 有效維度 {_ed_hist[-1]:.3f}", flush=True)
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
if u_dec is not None:
    # ⭐ decoder 也凍住：stage 2 之後它只被當成「u → 座標」的固定讀取器（E_geo 要用）。
    u_dec.eval()
    for p in u_dec.parameters():
        p.requires_grad_(False)
_ma = (float("nan") if (_S1 == 0 or logits is None)
       else (logits.argmax(1) == lab).float().mean().item())
print(f"  e_target match-acc(train batch) {_ma:.3f}", flush=True)   # ⚠️ 訓練批的數字，⛔ 不是驗證指標
if _ed_hist:
    print(f"  ⇒ u 有效維度 起 {_ed_hist[0]:.3f} → 末 {_ed_hist[-1]:.3f}", flush=True)

cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
import copy
refine_ema = copy.deepcopy(refine)
for _p in refine_ema.parameters():
    _p.requires_grad_(False)
ahead = ActionMLP().to(device)

# 🚨 誠實的 BC 地板（主人 2026-08-23 要求）。
# ⛔ 舊做法是 eval 時把 u 塞 0 當地板 —— 但訓練損失裡【沒有】u=0 的分支，
#    head 從沒看過零 ⇒ 那是分布外探針，不是 baseline，而且訓練愈久掉愈兇。
#    同一個壞探針先後撐起了「u 沒貢獻」與「u 貢獻 0.61」兩個相反的結論。
# ⇒ 這裡另外養一顆【從頭到尾只吃 cond、永遠不給 u】的 head，跟 ahead 同容量、同優化器。
class CondOnlyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND, 512, CHUNK * ADIM, n=3)
    def forward(self, cond):
        return self.net(cond).reshape(-1, CHUNK, ADIM)

bc_head = CondOnlyMLP().to(device)
# ⭐ BC_INDEP：bc 地板拆出去用自己的 optimizer ＋ 自己的 grad-clip。
#    主人 8/24 對 floor 的定義是「真正 BC 能到達的」—— 而共用 opt2 與全域 clip 的話，
#    它的更新會被主模型的梯度規模牽著走 ⇒ ⛔ 那不是獨立 baseline。
#    ⚠️ 開了之後 bc 的數字跟歷史結果不可直接比 ⇒ 預設仍是 0。
f_mods = ([cond_enc, cond_head, flow, refine, ahead] if BC_INDEP
          else [cond_enc, cond_head, flow, refine, ahead, bc_head])
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
opt_bc = torch.optim.Adam(bc_head.parameters(), lr=5e-4) if BC_INDEP else None
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()
print("stage 2 flow+refine+action ...", flush=True)
STEPS2 = 0 if LOAD_CKPT else int(os.environ.get("LACOT_STEPS2", 2000))
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    # ⭐ COND_DROP：l_anchor 這一路以 p 機率把整個 cond 歸零，逼 head 從 u 讀路徑。
    #    ⛔ 只丟 cond、不丟 u —— 丟 u 會反過來教 head 繞開 u。
    _ca = cond
    if COND_DROP > 0:
        _keep = (torch.rand(len(cond), 1, device=cond.device) >= COND_DROP).float()
        _ca = cond * _keep
    l_anchor = mse(ahead(_ca, et), act)
    if LEARNED_REFINE:
        u = flow.sample(B, cond).detach(); us = [u]
        for _ in range(3):
            u = refine(cond, u); us.append(u)
        if CONS == "ema":
            with torch.no_grad():
                tgts = [refine_ema(cond, us[r]) for r in range(3)]
            l_cons = sum((us[r + 1] - tgts[r]).pow(2).mean() for r in range(3)) / 3
        else:
            l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
        l_refine = sum(mse(ahead(cond, us[r + 1]), act) for r in range(3)) / 3
    else:
        # 🚨 l_refine 拿 flow 隨機抽的 u（很可能是另一條路），卻要求 head 輸出資料那條路的動作
        #    ⇒ 明文教 head 無視 u。ENC_OBJ 一改好它會立刻變成第二個 bypass ⇒ 同一輪拿掉。
        l_cons = l_refine = torch.zeros((), device=device)
    # ⭐ 誠實地板：只吃 cond，跟 u 完全無關。
    # ⚠️ cond 要 detach —— 否則 l_bc 的梯度會流進 cond_enc/cond_head，
    #    把 cond 訓練得更會單獨預測動作 ⇒ ① 主模型被這個 baseline 改動了、
    #    ② 比較會系統性地偏向「u 沒必要」。detach 之後量的才是乾淨的問題：
    #    「在【同一個】cond 表徵上，u 有沒有加值」。
    l_bc = mse(bc_head(cond.detach()), act)
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons + (0.0 * l_bc if BC_INDEP else l_bc)
    opt2.zero_grad(set_to_none=True)
    if opt_bc is not None:
        opt_bc.zero_grad(set_to_none=True)
        (total + l_bc).backward()               # ⭐ 一次 backward，但兩組參數各自 clip / step
        torch.nn.utils.clip_grad_norm_(bc_head.parameters(), 1.0)
    else:
        total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    if opt_bc is not None:
        opt_bc.step()
    if CONS == "ema" and LEARNED_REFINE:
        with torch.no_grad():
            for pe, pr in zip(refine_ema.parameters(), refine.parameters()):
                pe.mul_(EMA_M).add_(pr, alpha=1 - EMA_M)
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor {l_anchor.item():.4f} l_refine {l_refine.item():.4f}", flush=True)
if LOAD_CKPT:
    _lp = LOAD_CKPT if os.path.isabs(LOAD_CKPT) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), LOAD_CKPT)
    _ck = torch.load(_lp, map_location=device, weights_only=False)
    _cfg = _ck.get("cfg", {})
    # 🚨 逐項比對，⛔ 形狀對得上不代表是同一個模型 —— 這個 repo 已經被
    #    「同族東西混不同設定/架構」咬過三次（8/27 舊 code 版本、8/28 兩次不同 encoder 架構）。
    for _k, _v in (("K", K), ("COND", COND), ("T_CAP", T_CAP), ("CHUNK", CHUNK),
                   ("ENC_OBJ", ENC_OBJ)):
        _have = _cfg.get(_k)
        assert _have is None or _have == _v, (
            f"⛔ ckpt 的 {_k}={_have!r} 跟本次 {_v!r} 不一致 ⇒ 載進來的不是同一個模型")
    for _name, _mod in (("cond_enc", cond_enc), ("cond_head", cond_head), ("flow", flow),
                        ("refine", refine), ("ahead", ahead), ("bc_head", bc_head),
                        ("traj_enc", traj_enc), ("e_pooler", e_pooler)):
        _mod.load_state_dict(_ck[_name])
    if u_dec is not None:
        assert "u_dec" in _ck, (
            "⛔ ENC_OBJ=recon* 但 ckpt 裡沒有 u_dec ⇒ 那顆 ckpt 是舊版存的，"
            " 沒有 decoder 就沒有 E_geo 的眼睛")
        u_dec.load_state_dict(_ck["u_dec"])
    print(f"✅ 載入 {os.path.basename(_lp)}（跳過訓練，只跑評估）"
          f"  cfg={ {k: _cfg.get(k) for k in ('K','T_CAP','ENC_OBJ','LEARNED_REFINE','COND_DROP')} }",
          flush=True)

for m in f_mods:
    m.eval()
if BC_INDEP:
    bc_head.eval()      # ⚠️ BC_INDEP 時它不在 f_mods ⇒ 這一行漏了它會留在 train 模式進 rollout

if u_dec is not None:
    # ⭐ 這裡才是 decoder 的最終權重（訓練完 or ckpt 載完）⇒ 檢查要在這裡做，⛔ 不是在 stage 1 後面
    _n, _sh, _gap = _decoder_health()
    print(f"  decoder 內部點 RMSE {_n:.4f}   打亂 u 之後 {_sh:.4f}   用到的 u 值 {_gap:+.4f}"
          f"   {'🚨 幾乎沒在讀 u ⇒ 這一輪的 recon 數字不能拿來談 u' if _gap < 0.02 else '✓'}", flush=True)

# -------- SUCCESS-RATE ROLLOUT --------
def normstate(x):  # raw env position -> normalized torch [1,2]
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]

# ⭐ 「別人的 u」探針（主人 2026-08-23 核可）。
# ⛔ 零向量那個探針會製造 OOD，量到的是「head 沒看過零」的懲罰，不是 u 的價值。
# ★ 這個換法只換【內容】不換【分布】：從資料集隨機抽另一組 (s,g)，用它的 cond 生成並
#   refine 出 u，再把那個 u 配上【本題】的 cond 餵給 head。
#   成績不掉 ⇒ u 的內容根本沒被讀，head 只需要「那個位置有東西」。
#   成績掉了 ⇒ 內容有被讀，只是沒比 cond 多帶東西。
# 🚨 2026-08-28 修：舊版整支腳本只 seed 一次 ⇒ shuf arm 的 u 取決於「它是第幾個被跑的」
#    ⇒ ⛔ 不可重現，⛔ 而且跟其他 arm 沒有配對（別的 arm 都有 per-episode 的 torch seed）。
#    ⇒ 改成跟 torch_seed_fn 同一個做法：每集用 episode index 重新 seed。
#    ⚠️ 這【會】改變 shuf arm 的數字 ⇒ 舊的 shuf 結果不可直接比。
_SHUF_SEED0 = 20260823
_shuf_rng = np.random.default_rng(_SHUF_SEED0)


def _reseed_shuf(i):
    """每集把 shuf 的取樣流釘回同一個位置 ⇒ 各 arm 之間配對。"""
    global _shuf_rng
    _shuf_rng = np.random.default_rng(_SHUF_SEED0 + int(i))

# ⭐ X（2026-08-26 主人核可）：refine 的【位移方向】開關。
#    u ← u + dir·(refine(u) − u)   ⇒ dir=1 完全等價於原本的 u ← refine(u)
#    ⛔ 這不是裝飾，它是「這條鏈是活的嗎」的唯一直接證據：
#      反向跑（dir=−1）成功率【應該下降】；要是不動 ⇒ refine 根本沒在做事
#      ⇒ 那所有拿 refine 當賣點的比較都白比。
#    ⭐ 同一顆模型、同一批 episode、只差方向 ⇒ 配對比較，seed 噪聲咬不到。
_RDIR = [1.0]


# ⭐ 爬坡的 warm-start 快取。⛔ 每集一定要重置 —— 上一題的計畫漏到下一題就毀了配對。
_GRAD_CACHE = {"u": None}


def _reset_grad_cache(*_a):
    _GRAD_CACHE["u"] = None


def _apply_refine(cond, u, R):
    # 🚨 2026-08-28：LEARNED_REFINE=0 時 refine 網路【從來沒被訓練過】（隨機初始化）
    #    ⇒ 跑它等於把 u 推去一個隨機方向，結果比不 refine 還差，而且【不會報錯】。
    #    ⛔ 所以在這裡擋掉，⛔ 不是靠呼叫端記得把 R 設成 0。
    #    ⚠️ 於是 LEARNED_REFINE=0 的配置下「LaCoT R=k」對所有 k 都等於 flow R=0 ——
    #      那是誠實的：這個配置裡本來就沒有 refine，真正的 refine 是梯度爬坡（SUBGOAL/W2）。
    if not LEARNED_REFINE:
        return u
    for _ in range(R):
        u = u + _RDIR[0] * (refine(cond, u) - u)
    return u

@torch.no_grad()
def _foreign_u(R):
    """從資料集隨機抽一組 (s,g)，回傳它 refine R 輪之後的 u。"""
    while True:
        r = int(_shuf_rng.integers(0, N)); te = int(traj_end[r])
        if te - r >= CHUNK:
            break
    # 🚨 F5：⛔ 必須跟 make_batch 【完全】一致（含 clamp）。這支探針的全部立論是
    #    「分布對、只有內容錯」；少一個 clamp 就有幾 % 的樣本落在訓練從沒出現的近距離區間
    #    （實測 large-stitch 5.6%）⇒ 量到的一部分會是 OOD 懲罰，不是「內容沒被讀」。
    _d = _shuf_rng.random()
    gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
    gr = max(gr, min(r + CHUNK, te))
    s2 = torch.tensor((OBS[r] - mu) / sd, device=device)[None]
    g2 = torch.tensor((OBS[gr] - mu) / sd, device=device)[None]
    c2 = condvec(s2, g2)
    u = flow.sample(1, c2)
    return _apply_refine(c2, u, R)


@torch.no_grad()
def policy_chunk(obs, goal, R, use_u):
    # ⭐ 病二快篩（主人 8/29）：終局讓 u 退位 —— 離目標 < FINISH_R 就換獨立 bc head 收尾。
    #   ⛔ 只影響主 arm（use_u is True）；bc/shuf/null 對照不受影響。
    #   ⚠️ 分段模式另有一道在 make_subgoal_policy.policy()（那裡的 goal 是 subgoal，
    #      這裡判的必須是【最終目標】—— flat 的呼叫端傳的就是最終目標）。
    if FINISH_R > 0.0 and use_u is True and float(
            np.linalg.norm(np.asarray(obs[:2], np.float64)
                           - np.asarray(goal[:2], np.float64))) < FINISH_R:
        _FIN_COUNT[0] += 1
        if FINISH_MODE == "bc":
            use_u = "bc"
        else:               # resample：R=0 語義 ⇒ fresh flow 短計畫、不爬、不碰快取
            R = 0
    s = normstate(obs); g = normstate(goal); cond = condvec(s, g)
    if use_u == "bc":                              # ⭐ 誠實地板，走另一顆 head
        a = bc_head(cond)[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u == "shuf":                            # ⭐ 別人的 u，本題的 cond
        a = ahead(cond, _foreign_u(R))[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u:
        u = flow.sample(1, cond)
        if GRAD_REFINE:
            # ⭐ 主人 8/22 的更新式取代 learned refine。
            # ⚠️ 這一層是 @torch.no_grad()，而 grad_refine 內部自己開 enable_grad
            #    ⇒ ⛔ 忘了開會【炸】，而炸是好事：靜默的零梯度才是災難。
            # 🚨 2026-08-28 修：舊版寫 steps=R —— 而 R 是【refine 輪數】（dev eval 傳 1），
            #    GRAD_R 才是【每輪爬幾步】。混用的結果是整輪只爬 1 步 ⇒ 幾乎等於沒爬
            #    ⇒ 而它【不會報錯】，只會安靜地跑出「爬坡沒有用」的結論。
            #    ⭐ R × GRAD_R 才對：R=0 ⇒ 不爬（＝flow 直接用），R=3 ⇒ 150 步 ⇒ test-time scaling 也對。
            # 🚨 2026-08-28 修（整集凍住）：舊版的 warm 分支只看 `_warm is not None`，
            #    ⛔ 沒看 R。⇒ R=0 時 _steps = R*GRAD_R = 0，但第一個 chunk 照樣把
            #    【沒爬過的】u 寫進快取；之後每個 chunk 都走 warm 分支、也爬 0 步
            #    ⇒ 每次 flow.sample 抽的新 u 全被丟掉 ⇒ 整集都在用第 0 個 chunk 那個 u，
            #      而 cond 已經換過幾十次。註解寫「R=0 ⇒ flow 直接用」，行為跟它相反。
            #    ⇒ 決策抽成 lacot.refine_grad.grad_steps（純函式 ⇒ 沒有 GPU 也驗得了）。
            _use_warm, _steps = grad_steps(R, _GRAD_CACHE["u"] is not None, GRAD_R, GRAD_R_WARM)
            if _steps > 0:
                if _use_warm:
                    u = _GRAD_CACHE["u"]                 # 接續上一個 chunk 的計畫
                u = grad_refine(u, cond, u_dec, flow, GEO, s, g,
                                steps=_steps, eta=GRAD_ETA, lam=GRAD_LAM)
                _GRAD_CACHE["u"] = u
            # ⛔ _steps == 0（R=0）⇒ flow 抽的 u 直接用，⛔ 不碰 _GRAD_CACHE
        else:
            u = _apply_refine(cond, u, R)
    else:
        u = torch.zeros(1, K, D_MODEL, device=device)  # (s,g)-only floor
    a = ahead(cond, u)[0].cpu().numpy()  # [CHUNK,2]
    return np.clip(a, -1.0, 1.0).astype(np.float32)

# ⚠️ ogbench 不看 OGBENCH_DATA_DIR，它只看 dataset_dir 參數（預設 ~/.ogbench/data）。
#    2026-08-23 實測：不給 dataset_dir 它會【重新下載到 home】，而 home 是 NFS。
#    ⇒ 一定要明確傳本機 /archive 的路徑。
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))  # 官方標準，不自訂難度
N_TASKS = len(env.unwrapped.task_infos); SEEDS = int(os.environ.get("LACOT_EVAL_EPISODES", 50))
# 🚨 50，⛔ 不是 20。一手來源：OGBench `impls/hyperparameters.sh` —— pointmaze 的【每一行】
#    （navigate 與 stitch、六個 agent 全部）都寫 `--eval_episodes=50`。
#    ⚠️ 20 是 `impls/main.py` 的 flag 預設值，08-24 之前這裡把它誤當成官方值。

# ─────────────────────────────────────────────────────────────────────
# 兩層規劃（主人 2026-08-28 的分段做法）。⛔ SUBGOAL="" 時整段不執行。
# ─────────────────────────────────────────────────────────────────────
GEO = SUB_HELPERS = None
if SUBGOAL:
    from lacot.subgoal import (SubgoalPlanner, arc_subgoal, bfs_subgoal, consensus_subgoal,
                               farthest_confident_subgoal)
# 幾何 energy：SUBGOAL=latent/conf（長程層要它修）與 GRAD_REFINE（短程層要它修）都需要
if SUBGOAL in ("latent", "conf", "conf2") or GRAD_REFINE:
    assert u_dec is not None, (
        f"⛔ {'SUBGOAL=' + SUBGOAL if SUBGOAL else 'GRAD_REFINE=1'} 需要 decoder，"
        f" 而 ENC_OBJ={ENC_OBJ} 沒有訓 decoder。"
        " ⇒ 用 ENC_OBJ=recon/recon_ictr（或載一顆有 u_dec 的 ckpt），或改用 SUBGOAL=bfs")
    from lacot.refine_grad import GeoEnergy, grad_refine, grad_steps
    GEO = GeoEnergy(OBS, mu, sd, res=8, device=device, w_len=W_LEN)
    if W_LEN != 0.3:
        print(f"  ⚠️ 病一快篩：w_len={W_LEN:g}（預設 0.3）", flush=True)
    print(f"  幾何 energy：佔據圖 {tuple(GEO.shape)}，資料覆蓋 {GEO.coverage:.1%} 的格", flush=True)
    # 🚨 sanity：資料裡【真實走過】的路，穿牆懲罰必須 ≈0。不 ≈0 就是 SDF 蓋歪了。
    # ⚠️ 2026-08-28：⛔ 這一格【結構上必然通過】—— occ 是拿 OBS 蓋的，而 _t 又是
    #    make_batch 從【同一批 OBS】內插出來的 ⇒ 穿牆深度恆為 0，跟映射對不對無關。
    #    `[實測]` 用「覆蓋整個盒子」的資料建 GeoEnergy ⇒ 這行照樣是 0.0000、照樣通過，
    #    但盒內隨機點的穿牆中位也是 0.0000 ⇒ 牆這一項是【空的】，
    #    E_geo 安靜地退化成只有 goal/start/length。
    # ⇒ 留著它（便宜、而且真的壞掉時它會第一個叫），但真正的守門員是下面 GEO.health()。
    with torch.no_grad():
        _t, _m, _s, _g, _ = make_batch(rng)
        _wd = GEO.wall_depth(_t)
    assert float(_wd.median()) < 0.15, (
        f"⛔ 真軌跡的穿牆深度中位 {float(_wd.median()):.3f} 太大 ⇒ 佔據圖蓋歪了")
    _gh = GEO.health()
    print(f"  sanity：真軌跡穿牆中位 {float(_wd.median()):.4f}（⚠️ 恆真、參考用）"
          f"   格心 round-trip {_gh['mapping_err']:.2e}"
          f"   盒內隨機點穿牆中位 {_gh['wall_median_random']:.4f}", flush=True)
    assert _gh["ok"], "⛔ 幾何 energy 沒過健康檢查 ⇒ " + "；".join(_gh["reasons"])
    print("  ✓ 幾何 energy 健康檢查通過（映射對得上、牆這一項不是空的）", flush=True)
    # 🚨 decoder 是 E_geo 的【眼睛】—— 它若不讀 u，爬坡就是在對一條固定的平均路做最佳化，
    #    ⛔ 而且不會報錯：V 照樣會上升（它在改那條平均路），u 卻沒有任何意義。
    #    ⇒ 跟上面穿牆那格同層級：不過就停。
    _dn, _dsh, _dgap = _decoder_health()
    assert _dgap >= 0.02, (
        f"⛔ decoder 幾乎沒在讀 u（打亂 u 之後內部點 RMSE 只變 {_dgap:+.4f} < 0.02）"
        f" ⇒ 它吐的是「不管給什麼都一樣的平均路」⇒ E_geo 沒有眼睛，爬坡沒有意義")
    print(f"  ✓ decoder 讀得到 u（打亂後 RMSE {_dn:.4f} → {_dsh:.4f}，差 {_dgap:+.4f}）", flush=True)

if SUBGOAL == "bfs":
    _cells = DE._passable_cells(env)
    _cell_xy = np.array([env.unwrapped.ij_to_xy(c) for c in _cells], np.float64)
    # ⭐ 格寬從 env 算，⛔ 不寫死 —— 換環境就錯，而且錯了不會叫
    # 🚨 2026-08-28：這一份（全對距離取最小）是【對的】，而 exp_span_gap.py 另有一份
    #    取「頭兩個可通行格」的 —— 兩份不一致，而錯的那份餵出了 DELTA_SUB=7.5。
    #    ⇒ 抽進 lacot/dev_eval.py 共用，⛔ 不留兩份。
    _CELL_W = DE.cell_width(env)

    def _xy_to_ij(xy):
        return _cells[int(np.argmin(((_cell_xy - np.asarray(xy[:2])) ** 2).sum(1)))]
    SUB_HELPERS = (_cells, _xy_to_ij, _CELL_W)
    print(f"  BFS subgoal：格寬 {_CELL_W:.2f}，"
          f"subgoal 隔 {max(1, int(round(DELTA_SUB / _CELL_W)))} 格", flush=True)


# ⭐ #11/#12 診斷（2026-08-28）。⛔ 只記錄、不改行為。
#  d0   ‖decode 出來那條路的第 0 點 − 現在位置‖
#       🚨 arc_subgoal 從【路徑】上取點，卻【沒有】檢查那條路是不是從現在這裡出發。
#          路一塌成一點 ⇒ cum≈0 ⇒ 每次都回 pts[:, -1]，⛔ 而且不會叫。
#  dsub ‖subgoal − 現在位置‖（歐氏）
#       🚨 S1 沿【弧長】取 DELTA_SUB、S0 用【BFS 格數】—— 兩把尺不同單位，
#          而兩個 arm 的差正是歸因依據。⇒ 至少把實際落點的分布印出來對照。
#          ⛔ 沒有統一單位：弧長是「沿著計畫走的長度」、BFS 是「沿著最短路走的格數」，
#            各自都是自己那層的自然單位；硬換算會把 decode 路徑的彎曲度算進 S0 頭上。
SUB_DIAG = {"d0": [], "dsub": [], "n_bad_d0": 0, "n_bad_dsub": 0, "n_replan": [],
            "spread": [], "n_direct": 0, "n_fallback": 0}


def make_subgoal_policy(R, use_u):
    """回傳 (policy_chunk_fn, on_episode_start)。⭐ policy 有狀態 ⇒ 每題一定要重置。"""
    planner = SubgoalPlanner(delta_sub=DELTA_SUB, cap=SUB_CAP, stuck_m=SUB_STUCK, chunk=CHUNK)
    SUB_DIAG["_planner"] = planner      # ⭐ 收工時把【最後一題】的計數也收進來（on_start 收不到它）
    box = {"goal": None}

    def on_start(obs, goal, task):
        if planner.n_set:                       # ⭐ 上一題的重想次數收進診斷再重置
            SUB_DIAG["n_replan"].append(planner.n_replan)
        planner.reset()
        box["goal"] = np.asarray(goal[:2], np.float64)
        box["u_long"] = None
        _reset_grad_cache()

    def _plan(obs):
        s_n = normstate(obs); g_n = normstate(box["goal"])
        if SUBGOAL == "latent":
            # 長程層：flow 起手 → E_geo 爬 → 解碼 → 沿弧長取點
            # ⚠️ λ 調小：cond=(現在, 最終目標) 對 flow 來說是分布外的，結界本來就不太可信
            cond_l = condvec(s_n, g_n)
            if box.get("u_long") is not None and GRAD_R_WARM > 0:
                u_l, _st = box["u_long"], GRAD_R_WARM    # 長程計畫也是接續修，⛔ 不重想
            else:
                u_l, _st = flow.sample(1, cond_l).detach(), GRAD_R
            u_l = grad_refine(u_l, cond_l, u_dec, flow, GEO, s_n, g_n,
                              steps=_st, eta=GRAD_ETA, lam=GRAD_LAM)
            box["u_long"] = u_l
            with torch.no_grad():
                pts_n = u_dec(u_l)                       # [1, T_CAP, 2] 正規化座標
                pts_raw = pts_n * SD + MU                # ⭐ 換回原始座標 ⇒ 跟 DELTA_SUB 同單位
                sub = arc_subgoal(pts_raw, DELTA_SUB)[0].cpu().numpy()
            _d0 = float(np.linalg.norm(pts_raw[0, 0].cpu().numpy() - np.asarray(obs[:2])))
            SUB_DIAG["d0"].append(_d0)
            if _d0 > 0.5 * DELTA_SUB:                    # 路的起點根本不在現在這裡
                SUB_DIAG["n_bad_d0"] += 1
        elif SUBGOAL == "conf":
            # ⭐ 信心選點（主人 2026-08-29）：抽 M 份長程計畫、各自修完，
            #    subgoal 取「窗內 M 條共識最高（分散最小）」的點 —— 固定 7.5 的自適應版。
            cond_l = condvec(s_n, g_n).expand(SUB_M, -1)
            if box.get("u_long") is not None and GRAD_R_WARM > 0:
                u_l, _st = box["u_long"], GRAD_R_WARM    # M 份一起接續修
            else:
                u_l, _st = flow.sample(SUB_M, cond_l).detach(), GRAD_R
            u_l = grad_refine(u_l, cond_l, u_dec, flow, GEO,
                              s_n.expand(SUB_M, -1), g_n.expand(SUB_M, -1),
                              steps=_st, eta=GRAD_ETA, lam=GRAD_LAM)
            box["u_long"] = u_l
            with torch.no_grad():
                pts_raw = u_dec(u_l) * SD + MU           # [M, T_CAP, 2] 原始座標
            sub, _cs = consensus_subgoal(pts_raw, SUB_CONF_LO * DELTA_SUB,
                                         SUB_CONF_HI * DELTA_SUB, ret_stats=True)
            SUB_DIAG["spread"].append(_cs["spread"])
            _d0 = float(np.linalg.norm(
                pts_raw[:, 0].mean(0).cpu().numpy() - np.asarray(obs[:2])))
            SUB_DIAG["d0"].append(_d0)
            if _d0 > 0.5 * DELTA_SUB:
                SUB_DIAG["n_bad_d0"] += 1
        elif SUBGOAL == "conf2":
            # ⭐ 主人 8/29 統一版：fresh M 份（⛔ 不 warm，治計畫殘骸）→ 修 → 判信心。
            cond_l = condvec(s_n, g_n).expand(SUB_M, -1)
            u_l = flow.sample(SUB_M, cond_l).detach()
            u_l = grad_refine(u_l, cond_l, u_dec, flow, GEO,
                              s_n.expand(SUB_M, -1), g_n.expand(SUB_M, -1),
                              steps=GRAD_R, eta=GRAD_ETA, lam=GRAD_LAM)
            with torch.no_grad():
                pts_raw = u_dec(u_l) * SD + MU           # [M, T_CAP, 2] 原始座標
            sub, _fc = farthest_confident_subgoal(
                pts_raw, box["goal"], min_arc=0.25 * DELTA_SUB, ret_stats=True)
            SUB_DIAG["spread"].append(_fc.get("spread", _fc["g_spread"]))
            if _fc["direct"]:
                SUB_DIAG["n_direct"] += 1                # 走到底（g 信心夠）
            if sub is None:                              # ③ 整條發散 ⇒ 固定弧長保底
                SUB_DIAG["n_fallback"] += 1
                sub = arc_subgoal(pts_raw, DELTA_SUB)[0].cpu().numpy()
            _d0 = float(np.linalg.norm(
                pts_raw[:, 0].mean(0).cpu().numpy() - np.asarray(obs[:2])))
            SUB_DIAG["d0"].append(_d0)
            if _d0 > 0.5 * DELTA_SUB:
                SUB_DIAG["n_bad_d0"] += 1
        else:
            c = bfs_subgoal(env, SUB_HELPERS[1](obs), SUB_HELPERS[1](box["goal"]),
                            delta_cells=max(1, int(round(DELTA_SUB / SUB_HELPERS[2]))),
                            bfs_from=DE._bfs_from)
            sub = np.asarray(env.unwrapped.ij_to_xy(c), np.float64) if c is not None else box["goal"]
        sub = np.asarray(sub, np.float64)
        _ds = float(np.linalg.norm(sub - np.asarray(obs[:2])))
        SUB_DIAG["dsub"].append(_ds)
        if _ds > 2 * DELTA_SUB:                          # subgoal 遠到不像「一小段」
            SUB_DIAG["n_bad_dsub"] += 1
        return sub

    def policy(obs, goal):
        if box["goal"] is None:                          # ⚠️ 沒接 on_start 的呼叫端也要能跑
            box["goal"] = np.asarray(goal[:2], np.float64)
        # ⭐ 病二快篩：分段模式的終局判斷用【最終目標】（planner.sub 是中繼點，⛔ 不能拿來判）
        if FINISH_R > 0.0 and use_u is True and float(
                np.linalg.norm(np.asarray(obs[:2], np.float64) - box["goal"])) < FINISH_R:
            _FIN_COUNT[0] += 1
            if FINISH_MODE == "bc":
                return policy_chunk(obs, box["goal"], R, "bc")
            return policy_chunk(obs, box["goal"], 0, True)   # resample：fresh 短計畫直取 g
        if planner.observe(obs[:2]):
            planner.set(_plan(obs))
        return policy_chunk(obs, planner.sub, R,
                            "bc" if SUB_POLICY == "bc" else use_u)

    return policy, on_start


def rollout(R, use_u, tag):
    succ, ep = 0, 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(SEEDS):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)  # action-sampler stream
            # 🚨 2026-08-28 修：這條官方路徑【從來沒有】重置過爬坡快取 —— 而 dev 那條有
            #    （dev_rollout 掛了 on_episode_start=_reset_grad_cache）。
            #    ⇒ 上一集爬出來的 u 會被下一集當 warm 起點，跨集、跨 task、跨 arm 互相汙染，
            #      ⛔ 而且不會報錯：後面每一集都拿到「別題的計畫」，成績照樣算得出來。
            #    ⚠️ _GRAD_CACHE 定義處的註解自己就寫著「⛔ 每集一定要重置」。
            _reset_grad_cache()
            _reseed_shuf(1000 * task + sd_)    # #16：shuf arm 也要每集釘死 ⇒ 各 arm 配對
            while steps < MAXH and not success:
                for a in policy_chunk(obs, goal, R, use_u):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag}: success {succ}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep

print(f"\n==== SUCCESS RATE (env={ENV_NAME}, {N_TASKS} tasks x {SEEDS} seeds, MAXH {MAXH}) ====", flush=True)
out = dict(env=ENV_NAME, seed=SEED, cons=CONS, ema_m=EMA_M, K=K, cond=COND, chunk=CHUNK, steps2=STEPS2,
           tcap=T_CAP, tcap_requested=T_CAP_REQ, max_train_T=MAX_TRAIN_T, goal_sampling="uniform-official",
           episodes=N_TASKS * SEEDS, maxh=MAXH, rates={},
           # 🚨 2026-08-28：這一整排以前【沒有】落進 json ⇒ 拿到一個結果檔也讀不出
           #    它是 flat-grad / S1 / S0 哪一個、爬了幾步、用什麼 η。
           #    ⛔ 檔名帶得再全，也不能只靠檔名 —— 檔名會被人改、被人抄。
           enc_obj=ENC_OBJ, learned_refine=LEARNED_REFINE, cond_drop=COND_DROP,
           bc_indep=BC_INDEP, w_var=W_VAR, w_cov=W_COV,
           subgoal=SUBGOAL, grad_refine=GRAD_REFINE, grad_r=GRAD_R, grad_eta=GRAD_ETA,
           grad_lam=GRAD_LAM, grad_r_warm=GRAD_R_WARM, delta_sub=DELTA_SUB,
           sub_cap_chunks=SUB_CAP, sub_stuck_chunks=SUB_STUCK,
           dev_tiers=DEV_TIERS, dev_eval=None, load_ckpt=os.path.basename(LOAD_CKPT) or None)
# ⚠️ rollout 是整支腳本最貴的部分，而成本跟 CHUNK 成反比：CHUNK=1 每步都要重新決策，
#    比 CHUNK=4 多四倍的 policy 呼叫。⇒ 要跑 chunk 對照時用 LACOT_EVAL_RS 只留需要的輪數，
#    不然單格會超過叢集的時間上限（實測：CHUNK=1 一個變體就要 ~26 分）。
RS_PRE = [int(x) for x in os.environ.get("LACOT_EVAL_RS", "0,1,3,5,8").split(",") if x != ""]
# 🚨 2026-08-26 稽核抓到：RS 空的話，下面的 min()/max() 會丟 ValueError，
#    而炸點在 json 存檔【之前】⇒ 整輪訓練加 eval 的結果全部丟掉。
if not RS_PRE:
    raise SystemExit("⛔ LACOT_EVAL_RS 不能是空的 —— 至少要有一個輪數（例如 \"0\"）")

# ================= 開發尺（DEV_EVAL）=================
# 🚨 官方尺的獨立樣本數是 5 個 task，不是 250 集（2026-08-26 稽核＋實測複驗）。
#    ⇒ 任何小於 0.2 的效應都看不見 ⇒ 先驗尺，再談實驗。
if DEV_EVAL:
    # 🚨 題庫的 seed 固定成常數，⛔ 不能綁 LACOT_SEED —— 否則三個訓練 seed 拿到
    #    三份【不同的題目】⇒ 跨 seed、跨天的數字都不可比。
    DEV_TASKS = DE.build_dev_tasks(env.unwrapped, n_per_tier=DEV_PER_TIER,
                                   seed=int(os.environ.get("LACOT_DEV_TASK_SEED", 0)),
                                   min_dist=int(os.environ.get("LACOT_DEV_MIN_DIST", 3)))
    print(f"\n==== 開發尺：{len(DEV_TASKS)} 題（{env.unwrapped.maze_map.shape} 迷宮）====", flush=True)
    _tc = {}
    for t in DEV_TASKS:
        _tc[t["tier"]] = _tc.get(t["tier"], 0) + 1
    print(f"  分層 {dict(sorted(_tc.items()))}   BFS 距離 "
          f"{min(t['bfs_dist'] for t in DEV_TASKS)}~{max(t['bfs_dist'] for t in DEV_TASKS)}", flush=True)

    if DEV_TIERS:
        _keep = {int(x) for x in DEV_TIERS.split(",")}
        DEV_TASKS = [t for t in DEV_TASKS if t["tier"] in _keep]
        print(f"  ⚠️ 只跑 tier {sorted(_keep)} ⇒ {len(DEV_TASKS)} 題"
              f"（⛔ 跟全 tier 的結果不可直接比）", flush=True)

    def dev_rollout(R, use_u, tag, tseed=31337, subgoal=False):
        """subgoal=True ⇒ 走兩層規劃（長程想幾何、短程走路），⛔ 而且它有狀態、每題要重置。"""
        def _ep_seed(i):
            torch.manual_seed(tseed + i)
            _reseed_shuf(i)          # #16：shuf 的 numpy 流也要每集釘死，⛔ 否則沒有配對
        _on = None
        if subgoal:
            _pol, _on = make_subgoal_policy(R, use_u)
        elif use_u == "random":
            _pol = lambda o, gl: np.random.uniform(-1, 1, (CHUNK, ADIM)).astype(np.float32)
        else:
            _pol = lambda o, gl: policy_chunk(o, gl, R, use_u)
            _on = _reset_grad_cache if GRAD_REFINE else None
        rows = DE.dev_eval(
            env, DEV_TASKS, _pol, MAXH,
            seed0=10_000,
            # ⭐ 同一條 action-noise stream ⇒ 各 arm 之間【配對】，⛔ 少了它差值會混進取樣噪聲
            torch_seed_fn=_ep_seed,
            # 🚨 兩層 policy 有狀態 ⇒ ⛔ 沒有這個 hook 的話上一題的 subgoal 會漏到下一題
            on_episode_start=_on)
        s = DE.summarize(rows)
        print(f"  {tag}: {s['success']:.3f} ± {s['se']:.3f}  (n={s['n']})"
              f"   步數中位 {s['steps_med']:.0f}   最近距離中位 {s['best_dist_med']:.2f}", flush=True)
        print(f"      per tier " + "  ".join(
            f"t{k}:{v['success']:.3f}(n={v['n']},d={v['bfs_med']:.0f})"
            for k, v in s["per_tier"].items()), flush=True)
        return rows, s

    # ⭐ 驗收這把尺【本身】：三個已知不同的 policy 要分得開。⛔ 分不開就別拿它跑實驗。
    # 🚨 2026-08-26 稽核修正：gate 只用【已知排序】的對子。
    #    ⛔ 舊版把 bc vs lacot 放進 gate —— 但那一對的真答案就是差 ≈0
    #    ⇒ 儀器會在「實驗答案是零」的時候被判故障 ⇒ 永遠 deadlock。
    print("\n  --- 尺的驗收 ---", flush=True)
    _rr, _sr = dev_rollout(0, "random", "亂走       ")          # 靈敏度的下界
    _rb, _sb = dev_rollout(0, "bc", "bc 地板    ")
    _rb2, _sb2 = dev_rollout(0, "bc", "bc 重跑    ", tseed=71337)  # 特異度：同一顆模型
    _rz, _sz = dev_rollout(0, False, "u 歸零     ")
    _R_ARM = max(1, min(RS_PRE))
    _rm, _sm = dev_rollout(_R_ARM, True, f"LaCoT (R={_R_ARM})   ")
    # 🚨 2026-08-28 修：舊版分段 arm 寫死 R=0，而它的對手 LaCoT 用 R=1。
    #    ⇒ 短程層一步都不爬（疊上 policy_chunk 的 R=0 凍結 bug ⇒ 整集凍在第一個 u），
    #      而 ("subgoal","lacot") 這個 pair 照樣算得出 p 值 ⇒ 差值會被讀成「階層化沒幫助」。
    #    ⇒ 兩邊用【同一個 R】，差的才只有「有沒有分段」這一件事。
    _rsg, _ssg = (dev_rollout(_R_ARM, True, f"分段 {SUBGOAL:<6}", subgoal=True)
                  if SUBGOAL else (None, None))
    # ⏳ 2026-08-28 待主人裁（docs/2026-08-28-fable-plan-verification.md ④）：
    #    bc 這條路徑（bc_head(cond)）⛔ 不消耗 torch 亂數 ⇒ 換 tseed 是 no-op
    #    ⇒ bc 與 bc_rerun 位元相同 ⇒ 特異度那格【沒有驗到配對】。
    #    ⭐ dev_eval.sanity_check 現在會為此判 specificity=False 並說明原因
    #      ⇒ ⚠️ 在主人裁定之前，`尺的驗收` 會固定顯示沒過 —— 那是【誠實的紅燈】，
    #        ⛔ 不是新的故障。要讓它變綠，得把受測對象換成【會抽樣】的 arm
    #        （例如把 "bc 重跑" 改成 "LaCoT 重跑"，成本一樣是一個 arm）。
    #    ⛔ ルナ沒有自己換 —— 那份文件把它列成「二選一，待裁」。
    _named = {"random": _rr, "bc": _rb, "bc_rerun": _rb2, "null_u": _rz, "lacot": _rm}
    _pairs = [("bc", "lacot"), ("null_u", "lacot"), ("bc", "null_u")]
    if _rsg is not None:
        _named["subgoal"] = _rsg
        _pairs += [("subgoal", "bc"), ("subgoal", "lacot")]
    _pl = SUB_DIAG.pop("_planner", None)
    if _pl is not None and _pl.n_set:        # 最後一題沒有下一個 on_start 來收 ⇒ 這裡補
        SUB_DIAG["n_replan"].append(_pl.n_replan)
    if SUBGOAL and (SUB_DIAG["dsub"] or SUB_DIAG["d0"]):
        _q = lambda v: (f"中位 {np.median(v):.2f}  p10 {np.percentile(v, 10):.2f}"
                        f"  p90 {np.percentile(v, 90):.2f}") if v else "（沒有樣本）"
        print(f"\n  --- 分段 {SUBGOAL} 的 subgoal 診斷（⛔ 只報告、不進 gate）---", flush=True)
        print(f"    ‖sub − 現在‖  {_q(SUB_DIAG['dsub'])}"
              f"   （DELTA_SUB={DELTA_SUB:g}；> {2*DELTA_SUB:g} 的有 "
              f"{SUB_DIAG['n_bad_dsub']}/{len(SUB_DIAG['dsub'])}）", flush=True)
        if SUB_DIAG["d0"]:
            print(f"    ‖路的第0點 − 現在‖  {_q(SUB_DIAG['d0'])}"
                  f"   （> {0.5*DELTA_SUB:g} 的有 {SUB_DIAG['n_bad_d0']}/{len(SUB_DIAG['d0'])}"
                  f" ⇒ 🚨 arc_subgoal 是沿【那條路】取點的，路不從這裡出發的話取出來的就不是"
                  f" 'DELTA_SUB 遠的地方'）", flush=True)
        if SUB_DIAG["n_replan"]:
            print(f"    每題重想次數  中位 {np.median(SUB_DIAG['n_replan']):.0f}"
                  f"  最多 {max(SUB_DIAG['n_replan'])}"
                  f"   (cap {SUB_CAP} chunk = {SUB_CAP*CHUNK} step,"
                  f" stuck {SUB_STUCK} chunk = {SUB_STUCK*CHUNK} step)", flush=True)
        out["subgoal_diag"] = dict(
            mode=SUBGOAL, delta_sub=DELTA_SUB,
            dsub_med=float(np.median(SUB_DIAG["dsub"])) if SUB_DIAG["dsub"] else None,
            dsub_p10=float(np.percentile(SUB_DIAG["dsub"], 10)) if SUB_DIAG["dsub"] else None,
            dsub_p90=float(np.percentile(SUB_DIAG["dsub"], 90)) if SUB_DIAG["dsub"] else None,
            d0_med=float(np.median(SUB_DIAG["d0"])) if SUB_DIAG["d0"] else None,
            d0_p90=float(np.percentile(SUB_DIAG["d0"], 90)) if SUB_DIAG["d0"] else None,
            n_bad_d0=SUB_DIAG["n_bad_d0"], n_bad_dsub=SUB_DIAG["n_bad_dsub"],
            n_plans=len(SUB_DIAG["dsub"]),
            n_replan_med=float(np.median(SUB_DIAG["n_replan"])) if SUB_DIAG["n_replan"] else None,
            sub_cap_chunks=SUB_CAP, sub_stuck_chunks=SUB_STUCK, chunk=CHUNK)

    _chk = DE.sanity_check(_named, report_pairs=tuple(_pairs))
    for _n in _chk["notes"]:
        print("    " + _n, flush=True)
    print(f"  ⇒ 尺的驗收 {'✓ 通過' if _chk['passed'] else '🚨 沒過 ⇒ 這把尺還分不開已知不同的東西'}",
          flush=True)
    out["dev_eval"] = dict(n_tasks=len(DEV_TASKS), passed=bool(_chk["passed"]),
                           gates=_chk["gates"], random=_sr, bc=_sb, bc_rerun=_sb2,
                           null_u=_sz, lacot_r1=_sm, notes=_chk["notes"],
                           **({"subgoal": _ssg, "subgoal_mode": SUBGOAL} if _ssg else {}))
    # 🚨 2026-08-28 補：per-episode 明細一定要落 json。
    #    ⛔ 只存 summary 的話【跨 seed 合併 McNemar 算不出來】—— 8/27 就是卡在這裡，
    #    只能拿單一 seed 的 bootstrap CI 去推上界，而那個假設我自己十分鐘前才說過不成立。
    #    ⭐ 配對比較的有效樣本數是 discordant pairs（實測個位數），⛔ 不是題數
    #      ⇒ 沒有明細就沒有 discordant，沒有 discordant 就沒有合法的統計。
    out["dev_rows"] = {k: [{kk: vv for kk, vv in r.items()
                            if kk in ("idx", "tier", "bfs_dist", "success", "steps")} for r in v]
                       for k, v in ([("random", _rr), ("bc", _rb), ("bc_rerun", _rb2),
                                     ("null_u", _rz), ("lacot_r1", _rm)]
                                    + ([("subgoal", _rsg)] if _rsg is not None else []))}

# ================= 前置量測（D0 / D4）=================
# ⚠️ 這兩格任一個亮紅燈，爬坡實驗就不用做 —— 失敗會記在這一格頭上，
#    ⛔ 而不是記在「value 引導的爬坡不可行」頭上。
if PREREQ:
    print("\n==== 前置量測 ====", flush=True)

    # ---- D0：head 有沒有能力接受【更好的 u】？----
    # 做法：對同一個 (s,g)，比較三種 u 餵給 head 之後的動作預測誤差
    #   ① flow 生的 u        ← 實際 eval 走的路
    #   ② 真軌跡壓出來的 et  ← 「完美的 u」
    #   ⛔ 如果 ② 沒有明顯比 ① 好，代表 u 根本不在輸出的必經之路上（cond 已經夠用）
    #      ⇒ 那麼再怎麼改善 u 都不會反映到行為上 ⇒ 爬坡免談。
    with torch.no_grad():
        _d0 = {"flow": [], "oracle": [], "zero": []}
        _r0 = np.random.default_rng(999)
        for _ in range(20):
            traj, mask, s, g, act = make_batch(_r0)
            cond = condvec(s, g)
            et_true = etarget(traj, mask)
            _d0["oracle"].append(mse(ahead(cond, et_true), act).item())
            _d0["flow"].append(mse(ahead(cond, flow.sample(len(s), cond)), act).item())
            _d0["zero"].append(mse(ahead(cond, torch.zeros_like(et_true)), act).item())
    _m = {k: float(np.mean(v)) for k, v in _d0.items()}
    _gain = (_m["flow"] - _m["oracle"]) / max(_m["flow"], 1e-9)
    out["prereq_d0"] = dict(action_mse=_m, oracle_gain_frac=_gain)
    print(f"  D0 head 吃不同 u 的動作誤差：flow {_m['flow']:.4f} / "
          f"真軌跡 {_m['oracle']:.4f} / 全零 {_m['zero']:.4f}", flush=True)
    print(f"     ⇒ 換成完美的 u，誤差降 {100*_gain:.1f}%"
          f"{'   🚨 head 接不住更好的 u ⇒ 爬坡免談' if _gain < 0.05 else '   ✓ head 有能力接受更好的 u'}",
          flush=True)

    # ---- D4：flow 對固定的 (s,g) 生得出【不同】的 u 嗎？----
    # ⚠️ 若 p(u|s,g) 幾乎是一個點，λ·∇log p 這道結界就是把 u 鎖回原地的力場 ⇒ 爬坡無處可去。
    # ⭐ 8/25 我們證明了【資料裡】同題有多解，⛔ 但從沒證明【flow 學到了】。
    with torch.no_grad():
        _r4 = np.random.default_rng(777)
        _within, _act_spreads, _mus = [], [], []
        for _ in range(16):
            traj, mask, s, g, act = make_batch(_r4)
            c1 = condvec(s[:1], g[:1]).expand(PREREQ_N, -1)
            us = flow.sample(PREREQ_N, c1).reshape(PREREQ_N, -1)
            # ⭐ 自校準的量法（稽核建議）：同題內的散布 ÷ 跨題的散布。
            #    ⛔ 別用絕對 cos —— 它被「條件均值有多大」支配，門檻只能拍腦袋。
            _c = us - us.mean(0, keepdim=True)
            _within.append(float(torch.cdist(_c, _c).median()))
            _mus.append(us.mean(0))
            a1 = ahead(c1, us.reshape(PREREQ_N, K, D_MODEL)).reshape(PREREQ_N, -1)
            _act_spreads.append(float((a1.std(0)).mean()))
        _MU_ = torch.stack(_mus)
        _across = float(torch.cdist(_MU_ - _MU_.mean(0, keepdim=True),
                                    _MU_ - _MU_.mean(0, keepdim=True)).median())
    _w = float(np.mean(_within)); _ratio = _w / max(_across, 1e-9); _asd = float(np.mean(_act_spreads))
    out["prereq_d4"] = dict(within_med=_w, across_med=_across, ratio=_ratio,
                            action_std=_asd, n_samples=PREREQ_N)
    print(f"  D4 同一題抽 {PREREQ_N} 個 u：同題內散布 {_w:.3f} / 跨題散布 {_across:.3f}"
          f"  ⇒ 比值 {_ratio:.4f}", flush=True)
    # ⚠️ action_std 只印不判 —— head 若 bypass u，u 再多樣它也會是 0
    #    ⇒ 拿它進判準會把 D0 的病記到 D4 頭上（稽核指出）
    print(f"     解碼後動作的標準差 {_asd:.5f}（⛔ 診斷用，不進判準）", flush=True)
    print(f"     ⇒ {'🚨 同題內幾乎沒有散布 ⇒ flow 是點質量 ⇒ 爬坡無處可去' if _ratio < 0.1 else '✓ flow 生得出不同的 u'}",
          flush=True)

if out.get("dev_eval") is None:
    out.pop("dev_eval", None)       # ⛔ DEV_EVAL=0 時不要留一個誤導的 null

RS = RS_PRE
out["rates"]["bc"] = rollout(0, "bc", "誠實 BC 地板（獨立 head，只吃 cond）")
out["rates"]["null_u"] = rollout(0, False, "u 歸零（⚠️ OOD 探針，不是地板）")
out["rates"]["shuf"] = rollout(3, "shuf", "別人的 u（分布對、內容錯）")
for R in RS:
    out["rates"][f"R{R}"] = rollout(R, True, f"LaCoT refine R={R}")
# ⭐ X：反向 refine。⛔ 少了這格，「refine 有用」這個主張沒有任何直接證據。
# 🚨 2026-08-28 修：_RDIR 只被 _apply_refine 讀，而 _apply_refine 在 LEARNED_REFINE=0 時
#    直接 return、在 GRAD_REFINE=1 時根本不會被呼叫（policy_chunk 走爬坡那一支）
#    ⇒ 這兩個配置下「反向」跟「正向」是【同一件事】，gap 恆為 0（或被殘留快取汙染成垃圾非零值）
#      ⇒ 舊版照樣印判決 ⇒ 可能印成「refine 這條鏈是活的」，⛔ 而且白跑 250 集。
_REV_OK = bool(LEARNED_REFINE) and not GRAD_REFINE
if int(os.environ.get("LACOT_REV_ARM", 1)) and max(RS) > 0 and not _REV_OK:
    print(f"\n=== X 對照：⛔ 本配置沒有 learned refine"
          f"（LEARNED_REFINE={LEARNED_REFINE}, GRAD_REFINE={GRAD_REFINE}）"
          f" ⇒ 反向 refine 是 no-op ⇒ 這個對照不適用，跳過（省一輪 250 集 rollout）===",
          flush=True)
    out["x_refine_direction"] = {"applicable": False,
                                 "why": "LEARNED_REFINE=0 或 GRAD_REFINE=1 ⇒ _RDIR 沒有作用點"}
if int(os.environ.get("LACOT_REV_ARM", 1)) and max(RS) > 0 and _REV_OK:
    _rr = max(RS)
    _RDIR[0] = -1.0
    out["rates"][f"R{_rr}_reversed"] = rollout(_rr, True, f"⛔ 反向 refine R={_rr}（應該變差）")
    _RDIR[0] = 1.0
    _fwd = out["rates"].get(f"R{_rr}"); _rev = out["rates"].get(f"R{_rr}_reversed")
    if _fwd is not None and _rev is not None:
        _gap = _fwd - _rev
        print(f"\n=== X 對照：refine 方向 ===", flush=True)
        print(f"  正向 R={_rr}  {_fwd:.3f}   反向 {_rev:.3f}   差 {_gap:+.3f}", flush=True)
        print("  ⇒ 差 ≈ 0 代表 refine 根本沒在做事 ⇒ ⛔ 拿 refine 當賣點的比較全部不算數"
              if abs(_gap) < 0.02 else "  ⇒ 方向確實有影響 ⇒ refine 這條鏈是活的", flush=True)
        out["x_refine_direction"] = {"applicable": True, "forward": _fwd, "reversed": _rev,
                                     "gap": _gap, "R": _rr}
print("=> want: LaCoT > floor (reasoning helps on success), rate rises with R (test-time scaling).", flush=True)

# ⛔ 印到 stdout 不算存下來 —— log 會被覆蓋、也收不成表。
import json
# ⛔ 檔名要帶【所有會變的設定】—— 少一個就會跟別組互相覆蓋，
#    而覆蓋掉的舊檔在數值上完全合理、看不出來（2026-08-07 已經被這個咬過）。
# 🚨 F2：⛔ 檔名少一個會變的旋鈕，就會覆蓋掉舊設定跑出來的結果，
#    而被覆蓋的舊檔在數值上完全合理、看不出來（2026-08-07 已經被同一件事咬過一次）。
#    2026-08-24 新增 T_CAP / eval 集數 / goal 抽法三項 ⇒ 全部進檔名。
# 🚨 新旋鈕【必須】進檔名，否則不同設定會互相覆蓋 —— 這個 repo 已經被
#    「同族結果檔混著不同設定/架構的產物」咬過三次（8/27 引到舊 code 版本、
#    8/28 兩次引到不同 encoder 架構）。⇒ ⭐ 只有非預設值才加，預設跑出來的檔名不變。
# 🚨 2026-08-28 修（三個 arm 共用一個檔名）：_extra 沒帶 SUBGOAL / GRAD_* / DELTA_SUB /
#    SUB_CAP / DEV_TIERS ⇒ LOAD_CKPT 模式下 STEPS2=0、其他全同
#    ⇒ flat-grad（GRAD_REFINE=1 SUBGOAL=""）、S1（=latent）、S0（=bfs）
#      產生【完全相同的檔名】，後跑的直接覆蓋先跑的，⛔ 而被覆蓋的舊檔在數值上完全合理。
#    ⭐ 抽成純函式 ⇒ 沒有 GPU／資料也驗得了「三個 arm 的檔名互不相同」。
def _tag_extra(ENC_OBJ="sg_infonce", LEARNED_REFINE=1, COND_DROP=0.0, BC_INDEP=0,
               SUBGOAL="", GRAD_REFINE=0, GRAD_R=50, GRAD_ETA=0.1, GRAD_LAM=0.3,
               GRAD_R_WARM=10, DELTA_SUB=7.5, SUB_CAP=10, SUB_STUCK=3, DEV_TIERS="",
               W_LEN=0.3, FINISH_R=0.0, SUB_M=4, FINISH_MODE="bc", SUB_POLICY=""):
    """檔名後綴。⭐ 只有【非預設值】才進去 ⇒ 預設跑出來的檔名跟歷史一致（⛔ 不破壞舊索引）。

    ⚠️ 預設值必須跟上面那些 os.environ.get 的第二個參數逐一對齊 ——
       對不齊的話「非預設」就判錯，而它不會報錯。（smoke_rollout_fixes.py 逐項驗這件事。）
    """
    x = ""
    if ENC_OBJ != "sg_infonce":
        x += f"_eo{ENC_OBJ}"
    if not LEARNED_REFINE:
        x += "_norf"
    if COND_DROP > 0:
        x += f"_cd{COND_DROP:g}"
    if BC_INDEP:
        x += "_bci"
    if SUBGOAL:                                  # ⭐ S1 / S0 的分水嶺
        x += f"_sg{SUBGOAL}"
        if DELTA_SUB != 7.5:
            x += f"_ds{DELTA_SUB:g}"
        if SUB_CAP != 10:
            x += f"_sc{SUB_CAP}"
        if SUB_STUCK != 3:
            x += f"_sk{SUB_STUCK}"
        if SUBGOAL.startswith("conf") and SUB_M != 4:
            x += f"_m{SUB_M}"
        if SUB_POLICY:                           # 歸因對照：短程走 bc
            x += f"_sp{SUB_POLICY}"
    if GRAD_REFINE:                              # ⭐ flat-grad 的分水嶺
        x += f"_gr{GRAD_R}"
        if GRAD_ETA != 0.1:
            x += f"e{GRAD_ETA:g}"
        if GRAD_LAM != 0.3:
            x += f"l{GRAD_LAM:g}"
        if GRAD_R_WARM != 10:
            x += f"w{GRAD_R_WARM}"
        if W_LEN != 0.3:                         # 病一快篩（w_len 只在爬坡開著時有作用）
            x += f"_wl{W_LEN:g}"
    if FINISH_R > 0.0:                           # 病二快篩（rs＝resample 模式）
        x += f"_fin{FINISH_R:g}" + ("rs" if FINISH_MODE == "resample" else "")
    if DEV_TIERS:                                # 只跑部分 tier 的結果 ⛔ 不可跟全 tier 混
        x += "_dt" + DEV_TIERS.replace(",", "")
    return x


_extra = _tag_extra(ENC_OBJ=ENC_OBJ, LEARNED_REFINE=LEARNED_REFINE, COND_DROP=COND_DROP,
                    BC_INDEP=BC_INDEP, SUBGOAL=SUBGOAL, GRAD_REFINE=GRAD_REFINE,
                    GRAD_R=GRAD_R, GRAD_ETA=GRAD_ETA, GRAD_LAM=GRAD_LAM,
                    GRAD_R_WARM=GRAD_R_WARM, DELTA_SUB=DELTA_SUB, SUB_CAP=SUB_CAP,
                    SUB_STUCK=SUB_STUCK, DEV_TIERS=DEV_TIERS,
                    W_LEN=W_LEN, FINISH_R=FINISH_R, SUB_M=SUB_M, FINISH_MODE=FINISH_MODE,
                    SUB_POLICY=SUB_POLICY)
tag = (f"{ENV_NAME.replace('pointmaze-', '').replace('-v0', '')}_{CONS}_K{K}_c{COND}"
       f"_ch{CHUNK}_st{STEPS2}_T{T_CAP}_ep{SEEDS}_gu{_extra}_s{SEED}")   # gu = goal uniform(official)
# 🚨 smoke／假資料跑出來的檔【不准】落進 results/ —— 同族檔案混版本正是這個 repo 咬過
#    我們三次的病。exp_decode_probe.py 早就有 LACOT_DP_OUT 這個逃生口，主線一直沒有
#    ⇒ 補上。⭐ 預設仍是 "results" ⇒ ⛔ 既有行為不變。
_OUT_DIR = os.environ.get("LACOT_OUT_DIR", "")
dst = os.path.join(_OUT_DIR or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    f"rollout_{tag}.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print(f"寫入 {dst}", flush=True)

if FINISH_R > 0.0:
    print(f"  病二快篩：終局接管觸發 {_FIN_COUNT[0]} 次（⛔ 0 次＝開關沒作用，快篩白跑）", flush=True)
# ⭐ 存 checkpoint：以後要換探針就不必重訓一次（今天為了換探針重訓了三輪）。
ck = os.path.join(os.path.dirname(dst), f"ckpt_{tag}.pt")
if LOAD_CKPT:
    # ⛔ 只評估模式不存 ckpt —— 會把來源蓋掉，而且存的是同一份權重
    print(f"（只評估模式：⛔ 不存 ckpt，來源是 {os.path.basename(LOAD_CKPT)}）", flush=True)
    raise SystemExit(0)
torch.save({"cond_enc": cond_enc.state_dict(), "cond_head": cond_head.state_dict(),
            "flow": flow.state_dict(), "refine": refine.state_dict(),
            "ahead": ahead.state_dict(), "bc_head": bc_head.state_dict(),
            "traj_enc": traj_enc.state_dict(), "e_pooler": e_pooler.state_dict(),
            # ⭐ decoder 一定要跟著存：E_geo（幾何 energy）靠它把 u 解成可微的座標點，
            #    ⛔ 沒存的話下一步要重訓 encoder 才拿得回配得上的 decoder。
            **({"u_dec": u_dec.state_dict()} if u_dec is not None else {}),
            "cfg": dict(K=K, COND=COND, CHUNK=CHUNK, D_MODEL=D_MODEL, STEPS2=STEPS2, T_CAP=T_CAP,
                        GOAL_SAMPLING="uniform-official", EVAL_EPISODES=SEEDS,
                        CONS=CONS, EMA_M=EMA_M, SEED=SEED,
                        ENC_OBJ=ENC_OBJ, LEARNED_REFINE=LEARNED_REFINE,
                        COND_DROP=COND_DROP, BC_INDEP=BC_INDEP)}, ck)
print(f"存 checkpoint {ck}", flush=True)
