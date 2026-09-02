"""LaCoT SUCCESS-RATE eval (the real OGBench metric, not BC-MSE).
Train LaCoT (state: contrastive e_target frozen -> flow -> refine -> action MLP),
then ROLL OUT in the pointmaze env and measure success rate, comparing:
  * (s,g)-only floor  = ahead(cond, ZERO-u)      [GCBC / depth-0 floor]
  * LaCoT refine R = 0/1/3/5/8                        [test-time scaling]
Success = the env's own info['success']. Receding-horizon CHUNK execution.
"""
import os, sys, json, numpy as np, torch
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
assert SUBGOAL in ("", "latent", "bfs", "conf", "conf2", "ebfs"), \
    f"⛔ LACOT_SUBGOAL 只能是空/latent/bfs/conf/conf2/ebfs，收到 {SUBGOAL}"
# ⭐ conf2 ── 主人 8/29 下午的統一版：「g 信心夠高就直接走到底，不夠就挑最遠但信心夠高的點」。
#   每次重想 fresh 抽 M 份（⛔ 不接續修 —— 治「計畫殘骸」病）、修完、判信心。門檻自校準。
# 間距用訓練分布的中位數（exp_span_gap.py 實測 7.5，原始座標單位）⇒ 短程層坐在資料最肥的地方
DELTA_SUB = float(os.environ.get("LACOT_DELTA_SUB", 7.5))
# ⭐ conf ── 信心選點（主人 2026-08-29）：抽 M 份長程計畫，subgoal 取「窗內共識最高」的點。
#   固定弧長 7.5 是它的固定近似；窗 [LO,HI]×DELTA_SUB 下限擋原地共識、上限擋短程 cond 出分布。
SUB_M = int(os.environ.get("LACOT_SUB_M", 4))
# ⭐ E 選計畫（LACOT_SUB_ESEL、9/2）：0（預設）＝行為不變；N>0＝conf2 先抽 N 份、用 GEO 的 E 留最低的 SUB_M 份
#    再做共識選路標。動機：ebfs 分辨器證明執行通道 1.000、失敗全在計畫內容；而主打臂測試時 E 未出場
#    （GRAD_R=0、只看 M 份共識 ⇒「一致地錯」挑到遠錯點）。8/29 flat 家族 selection 0.600 > climb 0.520 為先例。
SUB_ESEL = int(os.environ.get("LACOT_SUB_ESEL", 0))
# ⭐ u 來源探針（LACOT_U_SOURCE、9/2 主人「好跑下去」）：flow（預設）＝原行為；oracle＝每次規劃把「當下→終點」
#    在 E 佔據圖上的 BFS 正確路徑餵進 encoder 得到 u。⇒ 分辨「u 沒練好」是 encoder/decoder 那段歪了（oracle 也差）、
#    還是 flow 生的 u 落錯地方（oracle 好、flow 差）。⛔ 只當診斷、不進配方（它偷看了圖）。
U_SOURCE = os.environ.get("LACOT_U_SOURCE", "flow")
assert U_SOURCE in ("flow", "oracle"), f"⛔ LACOT_U_SOURCE 只能是 flow/oracle，收到 {U_SOURCE}"
# ⭐ VQ 錨定（LACOT_VQ=V、9/2 離散化階梯第二層）：0（預設）＝行為不變；V>0＝pooler 出口每個 token 量化到 V 個 code。
#    訓練：decoder 與 action head 吃 u_q（straight-through）、flow 仍對連續 u 建模（NF 不能吃點質量）；
#    推論：flow.sample／refine 後 snap 到最近 code 再解碼／餵 head。動機：oracle 探針證明表示本身歪（每顆 init 各長一套字）。
VQ_V = int(os.environ.get("LACOT_VQ", 0))
VQ_BETA = float(os.environ.get("LACOT_VQ_BETA", 0.25))
VQ_NOISE_P = float(os.environ.get("LACOT_VQ_NOISE_P", 0.0))
# ⭐ 軟錨（LACOT_VQ_SOFT=1、9/2 VQ64 硬量化容量損失後的變體）：codebook 與 commitment 照舊（把 u 拉向字彙），
#    但 decoder／head／推論都吃【連續】u、不 snap ⇒ 只留「錨」、不留「瓶頸」。0（預設）＝硬量化。
VQ_SOFT = int(os.environ.get("LACOT_VQ_SOFT", 0))
# ⭐ 開頭綁定（LACOT_DEC_START、9/2 晚主人裁「三為主、二對照」）：""（預設）＝行為不變；
#    hard＝解碼後整條平移使第 0 點＝起點（訓練與推論全套同一個 helper ⇒ decoder 只學形狀、開頭結構上就是起點）；
#    soft＝解碼照舊、stage 1 加 W·‖第 0 點 − 起點‖²。動機：單集追蹤實證爛顆的計畫不從 s 出發、conf2 只看尾就直達。
DEC_START = os.environ.get("LACOT_DEC_START", "")
assert DEC_START in ("", "hard", "soft"), f"⛔ LACOT_DEC_START 只能是 空/hard/soft，收到 {DEC_START}"
DEC_START_W = float(os.environ.get("LACOT_DEC_START_W", 1.0))
# ⭐ flow 探針（LACOT_FLOW_PROBE=M、9/2 主人「好好看看問題出在哪」）：eval 載入時，對每個官方任務抽 M 份計畫、解碼，
#    逐份量「離 BFS 正確路徑的距離（Chamfer 一邊）／穿牆深度／末點距終點」⇒ 對路率與 M 份分散度。只診斷、不改任何行為。
FLOW_PROBE = int(os.environ.get("LACOT_FLOW_PROBE", 0))
# ⭐ 路標吸附（LACOT_SUB_SNAP=1、9/2 晚）：conf/conf2/latent 挑出的路標，吸附到 3×3 鄰域內「淨空最大」的自由 E 格心。
#    假說：計畫路線對（flow 探針進度 .8~.95）、路標對就全過（ebfs 1.0）、死在牆角與路口 ⇒ 差在路標貼牆／太遠。0＝行為不變。
SUB_SNAP = int(os.environ.get("LACOT_SUB_SNAP", 0))
# ⭐ 單集追蹤（LACOT_TRACE=task、9/2 晚）：該 task 的第一集，印每次重規劃的位置／路標／距離／路標 E 格與淨空／計畫開頭，
#    以及每 200 步的位置。只印、不改行為。0＝關。
TRACE_TASK = int(os.environ.get("LACOT_TRACE", 0))
# ⭐ 開頭守門（LACOT_SUB_HEADGUARD=τ 原始單位、9/2 晚追蹤後）：conf2 解出的計畫第 0 點離現在位置 > τ ⇒ 這份計畫不可信
#    （flow 沒學會「開頭在 s」）⇒ 不准直達、本次重規劃改用 E 圖 BFS 路標（ebfs 同套、資料重建圖、可部署）。0＝關。
SUB_HEADGUARD = float(os.environ.get("LACOT_SUB_HEADGUARD", 0.0))
SUB_CONF_LO = float(os.environ.get("LACOT_SUB_CONF_LO", 0.5))
SUB_CONF_HI = float(os.environ.get("LACOT_SUB_CONF_HI", 1.5))
# ⭐ 歸因對照（主人 8/29 晚）：分段模式的【短程】改走 bc head ——「bc＋BFS 中繼點」
#   回答 0.750 的 +4 是短程 u 的功勞、還是分段結構本身的功勞。⛔ 只影響分段 arm。
SUB_POLICY = os.environ.get("LACOT_SUB_POLICY", "")
assert SUB_POLICY in ("", "bc"), f"⛔ LACOT_SUB_POLICY 只能是空/bc，收到 {SUB_POLICY}"
# ⭐ DEC_ANCHOR ── eval-time 平移錨定（P1a 前哨、2026-08-30 調研引出）：把長程解碼路徑
#   整條平移到「第 0 點＝當前位置」再取 subgoal。⛔ 只影響 latent/conf/conf2 的供點解碼，
#   不碰爬坡的 E 評分、不碰 head 的 u。
#   目的＝把 93% 病拆層：cond ignoring 的「絕對位置錯」被平移修掉（Diffuser inpainting 的
#   零重訓近似），殘下的才是「形狀錯」。⇒ 錨定後 d0 診斷恆 0 是【生效指標】，不是尺壞了。
DEC_ANCHOR = int(os.environ.get("LACOT_DEC_ANCHOR", 0))
# ⭐ SUB_MAX_ARC ── conf2 選點的【上限】（單位＝DELTA_SUB 的倍數；0＝關＝歷史行為）。
#   L-tch 診斷（8/30）：路線內容錯而 M 份一致 ⇒「最遠但信心夠」挑到 18+ 遠的錯點。
#   上限把「一次只走一小段」落實成硬約束；direct 分支（直指 g）不受限。
SUB_MAX_ARC = float(os.environ.get("LACOT_SUB_MAX_ARC", 0.0))
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
S1_FROM = os.environ.get("LACOT_S1_FROM", "")     # ⭐ 凍同一個 stage 1（9/2 夜）：從這個 ckpt 載 stage 1 模組、跳過 stage 1 訓練；SEED 只抽 stage 2
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
# ⭐ 中間站（主人 8/29 晚核准，方言病的兩帖便宜藥）：
#   GRAD_MODE=climb（現行梯度爬坡）| select（抽 SEL_N 份、E 挑最低 —— 永遠是殼上的點）
#   GRAD_PROJ=1：爬完過 encoder 往返（decode→re-encode）拉回殼上再給 head
GRAD_MODE = os.environ.get("LACOT_GRAD_MODE", "climb")
assert GRAD_MODE in ("climb", "select"), f"⛔ LACOT_GRAD_MODE 只能是 climb/select，收到 {GRAD_MODE}"
SEL_N = int(os.environ.get("LACOT_SEL_N", 8))
GRAD_PROJ = int(os.environ.get("LACOT_GRAD_PROJ", 0))
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

# ═══ P1b：ebfs teacher 資料引擎（主人 2026-08-30「開始跑」）════════════════
# ⭐ 治 cond OOD 的根：訓練配對混入「資料圖搜出來的遠距 (s,g)＋最優拼接軌跡」。
#    圖＝GeoEnergy 的佔據圖（資訊 ⊆ D、跟 ebfs 供點同一張 ⇒ 公平性同一條論證）。
#    teacher 樣本【只餵幾何側】（e_target/flow/decoder）；action loss（l_anchor/l_bc）
#    用 real-mask 擋掉 —— ⛔ bc/head 只吃真資料的真動作，執行器不被合成動作污染。
# ⚠️ (s,g) 抽法＝全域均勻可達對：原資料（hindsight、全短）供「短」的密度，teacher 專供
#    遠距 ⇒ 混合本身就是重尾（調研配方）。mix 比例是超參（ablation #13）。
TEACHER_MIX = float(os.environ.get("LACOT_TEACHER_MIX", 0.0))
_TCH = None
if TEACHER_MIX > 0:
    assert not LEARNED_REFINE, "⛔ TEACHER_MIX 的 real-mask 尚未接進 l_refine 分支 —— 要開先接"
    from lacot.refine_grad import GeoEnergy as _TchGeo
    from lacot.subgoal import grid_shortest_path as _tch_sp
    _tg = _TchGeo(OBS, mu, sd, res=8, device="cpu")
    _tocc = (_tg.dist[0, 0].numpy() == 0.0)
    _tfree = np.argwhere(_tocc)
    _tlo = np.asarray(_tg.lo, np.float64)
    _tspan = np.asarray(_tg.hi - _tg.lo, np.float64)
    _tshape = np.asarray(_tg.shape, np.int64)
    _tcell_norm = _tspan / (_tshape - 1)                     # 一格的正規化尺寸（jitter 用）

    def _tch_cell_to_norm(c):
        return _tlo + np.asarray(c, np.float64) * _tcell_norm

    _trng = np.random.default_rng(20260830)                  # ⛔ 題庫 seed 固定，不綁 LACOT_SEED
    _pool = []
    _tries = 0
    while len(_pool) < 4096 and _tries < 100000:
        _tries += 1
        a = tuple(_tfree[int(_trng.integers(len(_tfree)))])
        b = tuple(_tfree[int(_trng.integers(len(_tfree)))])
        p = _tch_sp(_tocc, a, b)
        if p is not None and len(p) >= 4:                    # 太短的不進 teacher（原資料已滿是短的）
            _pool.append(np.array([_tch_cell_to_norm(c) for c in p]))   # [L,2] 正規化格心序列
    assert len(_pool) >= 1024, f"⛔ teacher 題庫只湊到 {len(_pool)} 條 —— 佔據圖連通性有問題"
    _TCH = _pool
    _tlens = np.array([len(p) for p in _pool])
    print(f"  teacher 題庫：{len(_pool)} 條（細格步數 p50 {np.median(_tlens):.0f} "
          f"p90 {np.percentile(_tlens, 90):.0f} max {_tlens.max()}），mix={TEACHER_MIX:g}", flush=True)

# ═══ P2：自舉蒸餾資料（LACOT_BOOT_DATA＝BOOT_GEN 產的 npz）═════════════════
# exp(−βE) 加權蒸餾的實作＝【按 w 加權抽樣】（期望上等價、⛔ 不動 loss）。
# 樣本走 teacher 通道 ⇒ real-mask 自動擋 action loss（bc/head 不吃合成動作，同 P1b）。
BOOT_DATA = os.environ.get("LACOT_BOOT_DATA", "")
BOOT_TAG = os.environ.get("LACOT_BOOT_TAG", "")
BOOT_BETA = float(os.environ.get("LACOT_BOOT_BETA", 4.0))    # 有限 β＝自舉檔位（β→∞＝argmax）
BOOT_FRAC = float(os.environ.get("LACOT_BOOT_FRAC", 0.5))    # teacher 樣本裡自舉佔比
_BOOT = None
if BOOT_DATA:
    assert TEACHER_MIX > 0, "⛔ BOOT_DATA 走 teacher 通道 —— LACOT_TEACHER_MIX 要 > 0 才有通道"
    assert BOOT_TAG, "⛔ 自舉訓練必須給 LACOT_BOOT_TAG（進檔名）—— 不給會跟非自舉 ckpt 互蓋"
    _bz = np.load(BOOT_DATA, allow_pickle=False)
    _bmeta = json.loads(str(_bz["meta"]))
    assert _bmeta.get("env") == ENV_NAME, (
        f"⛔ 自舉樣本是 {_bmeta.get('env')} 生的，本次訓練是 {ENV_NAME} —— 拿錯檔了")
    _bE = _bz["E"].astype(np.float64)
    _bw = np.exp(-BOOT_BETA * (_bE - _bE.min()))
    _BOOT = (_bz["trajs"].astype(np.float32), _bw / _bw.sum())
    print(f"  自舉蒸餾集：{len(_bE)} 條（{os.path.basename(BOOT_DATA)}，"
          f"pass@{_bmeta.get('M')} {_bmeta.get('pass_at_m'):.3f}）β={BOOT_BETA:g} "
          f"frac={BOOT_FRAC:g}   有效樣本數 {1.0 / float((_BOOT[1] ** 2).sum()):.0f}", flush=True)

# make_batch 的 real-mask side channel：最近一個 batch 裡哪些樣本是真資料（有真動作）。
_REAL_W = [None]


def _teacher_traj(rng, n):
    """從題庫抽 n 條 teacher 軌跡 → traj[n,T,2] 正規化。cell 內 jitter。
    ⭐ P2：_BOOT 有東西時，n 條裡 BOOT_FRAC 比例改抽自舉樣本（按 exp(−βE) 權重）。
    自舉樣本已是 T_CAP 點 ⇒ 不再內插、不 jitter（flow 生成自帶連續多樣性）。"""
    n_b = int(round(n * BOOT_FRAC)) if _BOOT is not None else 0
    trajs = np.empty((n, T_CAP, 2), np.float64)
    for i in range(n - n_b):
        p = _TCH[int(rng.integers(len(_TCH)))]
        p = p + rng.uniform(-0.5, 0.5, size=(1, 2)) * _tcell_norm   # 整條小平移（cell 內）
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        tt = np.linspace(0.0, cum[-1], T_CAP)
        for k in (0, 1):
            trajs[i, :, k] = np.interp(tt, cum, p[:, k])
    if n_b > 0:
        _r = _BOOT_RNG if _BOOT_RNG is not None else rng      # ⭐ 9/2：BOOT_SEED 設了就走獨立流
        idx = _r.choice(len(_BOOT[0]), size=n_b, p=_BOOT[1])
        trajs[n - n_b:] = _BOOT[0][idx]
    return trajs.astype(np.float32)


# ⭐ 荒漠重採樣（LACOT_DATA_RESAMPLE=1）：訓練 batch 的起點 transition 從「均勻」改成
#    「按位置資料密度反比」兩段式抽（先加權抽 4×4 格、再格內均勻）。病理依據 2026-09-01
#    驗屍：稀疏走廊＝BC 跨 seed 共通爛＝學習訊號弱。⛔ 預設關＝抽樣分布與歷史完全一致。
#    ⚠️ 只動 dataset transition 的抽法；teacher 題庫與 shuf 探針的抽樣不碰。
DATA_RESAMPLE = int(os.environ.get("LACOT_DATA_RESAMPLE", 0))
_RS_CELLS = None
if DATA_RESAMPLE:
    _rs_ij = np.floor((OBS[:, :2] + 2.0) / 4.0).astype(np.int64)   # 4×4 格、對齊迷宮牆格
    _rs_key = _rs_ij[:, 0] * 1000 + _rs_ij[:, 1]
    _rs_uniq, _rs_inv, _rs_cnt = np.unique(_rs_key, return_inverse=True, return_counts=True)
    _rs_w = 1.0 / (_rs_cnt + 0.1 * _rs_cnt.mean())
    _rs_w = _rs_w / _rs_w.sum()
    _RS_CELLS = [np.flatnonzero(_rs_inv == k) for k in range(len(_rs_uniq))]
    print(f"⭐ 荒漠重採樣：{len(_rs_uniq)} 格、最稀/最富抽中率比 "
          f"{float(_rs_w.max() / _rs_w.min()):.1f}x", flush=True)


def _sample_r(rng):
    if _RS_CELLS is None:
        return int(rng.integers(0, N))
    c = int(rng.choice(len(_RS_CELLS), p=_rs_w))
    idx = _RS_CELLS[c]
    return int(idx[int(rng.integers(len(idx)))])


def make_batch(rng, teacher_mix=0.0):
    """⚠️ teacher_mix 預設 0 ⇒ 行為與歷史完全一致；只有 stage1/stage2 的訓練呼叫傳 TEACHER_MIX。
    探針（D0/D4/GEO sanity）一律用預設 0 —— 它們的 act/穿牆語義不容 teacher 樣本混入。"""
    n_t = int(round(B * teacher_mix)) if (_TCH is not None and teacher_mix > 0) else 0
    n_r = B - n_t
    rows, goals = [], []
    while len(rows) < n_r:
        r = _sample_r(rng); te = int(traj_end[r])
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
                    T_CAP, axis=1).reshape(n_r, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])      # ⚠️ 夾在終點內，⛔ 不可以跨到下一條軌跡
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    if n_t > 0:
        # teacher 樣本：正規化格心內插軌跡；s/g＝軌跡端點；act＝零（⛔ 只佔位，loss 端用
        # real-mask 擋掉 —— head/bc 不吃合成動作）。
        tt = _teacher_traj(rng, n_t)                 # [n_t, T_CAP, 2] 已正規化
        traj = np.concatenate([traj, tt], 0)
        s = np.concatenate([s, tt[:, 0].astype(np.float64)], 0)
        g = np.concatenate([g, tt[:, -1].astype(np.float64)], 0)
        act = np.concatenate([act, np.zeros((n_t, CHUNK, ADIM), np.float32)], 0)
    mask = np.zeros((B, T_CAP), bool)                # 全 False ＝ 全部都是真點
    # ⚠️ 自檢：一響就代表洩漏又回來了（⛔ 別把它拿掉）
    assert mask.shape[1] == T_CAP and not mask.any(), "⛔ 取樣點數不再固定 ⇒ 長度會從 mask 洩漏"
    _REAL_W[0] = torch.cat([torch.ones(n_r), torch.zeros(n_t)]).to(device)
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
# ⭐ seed 拆分診斷（LACOT_DATA_SEED、9/1 seed 病因 2×2）：-1（預設）＝跟 SEED 走、行為不變；
#    ≥0＝資料抽樣流（batch 順序）改用此值，model init／torch 訓練噪聲仍吃 SEED。
#    ⇒ 2×2 交叉可拆「壞初始化 vs 壞資料順序」誰是爛 seed 元兇。
DATA_SEED = int(os.environ.get("LACOT_DATA_SEED", -1))
# ⭐ 自舉抽樣獨立 rng（LACOT_BOOT_SEED、9/2 dz2 重現後）：-1（預設）＝boot 樣本跟主 rng 同一條流、行為不變；
#    ≥0＝boot 集「抽到哪些條」改用此 seed，主資料順序仍吃 DATA_SEED/SEED。
#    ⇒ 可拆「主資料順序 vs boot 抽樣」誰是 s2 型方差的來源（9/2 四顆 0.35~0.90 的病根）。
BOOT_SEED = int(os.environ.get("LACOT_BOOT_SEED", -1))
# ⭐ lr 縮放診斷（LACOT_LR_SCALE、9/1）：全部 optimizer 的 lr 乘此係數。1.0＝行為不變。
LR_SCALE = float(os.environ.get("LACOT_LR_SCALE", 1.0))
# ⭐ 訓練期高頻診斷（LACOT_DIAG_TRAIN=1、9/1）：stage1 前 300 步每 10 步印
#    recon／總 loss／梯度範數／u 有效維度。只加記錄、⛔ 不改任何訓練行為。
DIAG_TRAIN = int(os.environ.get("LACOT_DIAG_TRAIN", 0))


def _apply_lr_scale(opt):
    if LR_SCALE != 1.0:
        for pg in opt.param_groups:
            pg["lr"] *= LR_SCALE
    return opt


torch.manual_seed(SEED); rng = np.random.default_rng(SEED if DATA_SEED < 0 else DATA_SEED)
_BOOT_RNG = np.random.default_rng(BOOT_SEED) if BOOT_SEED >= 0 else None   # ⭐ 9/2 boot 抽樣獨立流
print(f"設定：seed={SEED} cons={CONS} ema_m={EMA_M} K={K} COND={COND}"
      + (f" data_seed={DATA_SEED}" if DATA_SEED >= 0 else "")
      + (f" boot_seed={BOOT_SEED}" if BOOT_SEED >= 0 else "")
      + (f" lr_scale={LR_SCALE:g}" if LR_SCALE != 1.0 else ""), flush=True)
traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
opt1 = _apply_lr_scale(torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3))
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


def _route_traj(obs_xy, goal_xy):
    """obs→goal 在 E 佔據圖上的 BFS 正確路徑（teacher 同一張圖）→ 重採樣 T_CAP → [1,T_CAP,2] 正規化；無路回 None。"""
    assert _TCH is not None, "⛔ 需要 TEACHER_MIX>0（借 teacher 的 E 佔據圖）"
    def _cell(xy):
        z = (np.asarray(xy[:2], np.float64) - np.asarray(mu, np.float64)[:2]) / np.asarray(sd, np.float64)[:2]
        c = np.clip(np.rint((z - _tlo) / _tcell_norm).astype(int), 0, _tshape - 1)
        if not _tocc[tuple(c)]:                         # 牆格 ⇒ snap 到最近自由格
            c = _tfree[int(((_tfree - c) ** 2).sum(1).argmin())]
        return tuple(int(v) for v in c)
    p = _tch_sp(_tocc, _cell(obs_xy), _cell(goal_xy))
    if p is None or len(p) < 2:
        return None
    pts = np.array([_tch_cell_to_norm(c) for c in p], np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); cum = np.concatenate([[0.0], np.cumsum(seg)])
    tt = np.linspace(0.0, cum[-1], T_CAP)
    traj = np.stack([np.interp(tt, cum, pts[:, k]) for k in (0, 1)], 1).astype(np.float32)
    return torch.tensor(traj, device=device)[None]


def _oracle_u(obs_xy, goal_xy, n):
    """探針：正確路徑 → encoder → u [n,K,D]；無路回 None。"""
    tr = _route_traj(obs_xy, goal_xy)
    if tr is None:
        return None
    with torch.no_grad():
        u = etarget(tr, torch.zeros(1, T_CAP, dtype=torch.bool, device=device))
    return u.expand(n, -1, -1).contiguous()


def flow_probe(tasks_sg, M):
    """對每題：flow.sample(M | cond(s,g)) → (VQ snap) → decoder → 錨到 obs（同 conf2）→ 只看計畫【開頭一段】
    （弧長 ≤ SUB_MAX_ARC×DELTA_SUB 原始單位＝conf2 選路標的視窗），量它離 BFS 正確路徑多遠（原始座標、Chamfer 一邊）、
    穿牆深度、與 M 份的分散度。對路門檻＝正確路徑往返後同一段的距離 ×2（下限 0.5）。⛔ 只診斷。"""
    out = []
    arc_cap = (SUB_MAX_ARC * DELTA_SUB) if SUB_MAX_ARC > 0 else 2.0 * DELTA_SUB
    def _head_mask(pr):                                             # pr [B,T,2] 原始 → 開頭弧長 ≤ arc_cap 的點
        seg = (pr[:, 1:] - pr[:, :-1]).norm(dim=-1); cum = torch.cat([torch.zeros(pr.shape[0], 1, device=pr.device), seg.cumsum(1)], 1)
        m = cum <= arc_cap; m[:, :4] = True
        return m
    def _route_d(pr, route_raw):                                    # 開頭段各點到路徑折線最近距離的平均 [B]
        d = torch.cdist(pr, route_raw[None].expand(pr.shape[0], -1, -1)).min(2).values   # [B,T]
        m = _head_mask(pr).float()
        return (d * m).sum(1) / m.sum(1)
    for (sx, gx) in tasks_sg:
        tr = _route_traj(sx, gx)
        if tr is None or u_dec is None:
            out.append(None); continue
        with torch.no_grad():
            route_raw = tr[0] * SD + MU                                # [T,2] 原始
            s_n = normstate(np.asarray(sx, np.float64)); g_n = normstate(np.asarray(gx, np.float64))
            cond = condvec(s_n, g_n).expand(M, -1)
            u = flow.sample(M, cond)
            pts_n = _dec(_q(u), s_n); pts_raw = _anchor_pts(pts_n * SD + MU, np.asarray(sx, np.float64))   # [M,T,2]
            route_d = _route_d(pts_raw, route_raw)                     # [M]
            wall = (GEO.wall_depth(pts_n) * _head_mask(pts_raw).float()).sum(1) / _head_mask(pts_raw).float().sum(1)
            u_o = etarget(tr, torch.zeros(1, T_CAP, dtype=torch.bool, device=device))
            po = _anchor_pts(_dec(_q(u_o), tr[:, 0]) * SD + MU, np.asarray(sx, np.float64))
            ref = float(_route_d(po, route_raw)[0]); thr = max(2.0 * ref, 0.5)
            onroute = (route_d < thr).float()                          # 對路只看路徑距；穿牆分開報
            heads = [pr[_head_mask(pr[None])[0]] for pr in pts_raw]
            ends = torch.stack([h[-1] for h in heads])                 # 每份開頭段的終點 [M,2]
            spread = float(torch.cdist(ends, ends).mean())
            # ⭐ 進度（9/2 第二版）：整條計畫沿【正確路徑】走到多遠才偏掉。每個計畫點找最近的路徑點索引，
            #    只算距離 ≤ 走廊半寬（1.5 原始單位）的點，取最遠索引 / T ⇒ 走錯岔路的計畫會停在岔路口的索引。
            dfull = torch.cdist(pts_raw, route_raw[None].expand(M, -1, -1))     # [M,T,T]
            dmin, idx = dfull.min(2)                                             # [M,T]
            near = dmin <= 1.5
            prog = torch.where(near, idx.float(), torch.zeros_like(idx.float())).max(1).values / (route_raw.shape[0] - 1)
            po_d, po_i = torch.cdist(po, route_raw[None]).min(2); po_prog = float(torch.where(po_d <= 1.5, po_i.float(), torch.zeros_like(po_i.float())).max() / (route_raw.shape[0] - 1))
        out.append(dict(M=M, ref_route_d=ref, thr=thr, onroute=float(onroute.mean()),
                        route_d_med=float(route_d.median()), route_d_min=float(route_d.min()),
                        wall_med=float(wall.median()), head_end_spread=spread,
                        prog_med=float(prog.median()), prog_frac80=float((prog >= 0.8).float().mean()), oracle_prog=po_prog))
    return out


def roundtrip_gate(tasks_sg):
    """⭐ embedding 軟尺（9/2 主人「先確認 embedding 好」）：任務路徑 → encoder →（VQ snap）→ decoder，
    量往返 mse、解出路徑的穿牆深度（E 的 wall 項）、末點離終點距離。不跑模擬器、每顆 ckpt 幾毫秒。
    tasks_sg: [(start_xy, goal_xy), ...] 原始座標。回 list of dict。"""
    out = []
    for (sx, gx) in tasks_sg:
        tr = _route_traj(sx, gx)
        if tr is None or u_dec is None:
            out.append(None); continue
        with torch.no_grad():
            u = etarget(tr, torch.zeros(1, T_CAP, dtype=torch.bool, device=device))
            pts = _dec(_q(u), tr[:, 0])                                 # [1,T_CAP,2] 正規化
            mse = float((pts - tr).pow(2).mean())
            wall = float(GEO.wall_depth(pts).mean()) if "GEO" in globals() and GEO is not None else float("nan")
            gdist = float((pts[0, -1] - tr[0, -1]).norm())
        out.append(dict(mse=mse, wall=wall, gdist=gdist))
    return out


def encode_u(pts):
    """往返投影的後半：decode 出的（正規化）座標點重新編碼回 u —— 拉回 encoder 的殼上。"""
    m = torch.zeros(pts.shape[0], pts.shape[1], dtype=torch.bool, device=pts.device)
    return etarget(pts, m)
# ⭐ ENC_OBJ=recon* 要一顆 decoder：解得回 128 個座標點，才代表 u 真的裝了那條路。
#    ⛔ 它不是暫時的鷹架 —— E_geo（幾何 energy）也靠同一顆把 u 解成可微的座標點。
u_dec = None
s_embed = None
if ENC_OBJ.startswith("recon"):
    from lacot.traj_decoder import TrajDecoder
    u_dec = TrajDecoder(D_MODEL, T_CAP).to(device)
    s_embed = nn.Linear(2, D_MODEL).to(device) if DEC_START == "hard" else None   # ⭐ 起點 token（hard 綁定；進 opt1 與 ckpt）
    u_dec.check_p = 0.01
    # ⛔ sg_c / q_pooler 不進 optimizer —— (s,g)↔τ InfoNCE 整條移除，⛔ 不是降權重。
    opt1 = _apply_lr_scale(torch.optim.Adam([p for m in (traj_enc, e_pooler, u_dec) for p in m.parameters()]
                                            + ([p for p in s_embed.parameters()] if s_embed is not None else []), lr=1e-3))
elif ENC_OBJ != "sg_infonce":
    raise ValueError(f"⛔ LACOT_ENC_OBJ 只能是 sg_infonce/recon/recon_ictr，收到 {ENC_OBJ}")

vq = None
if VQ_V > 0:
    from lacot.vq import TokenVQ
    vq = TokenVQ(VQ_V, D_MODEL, beta=VQ_BETA, noise_p=VQ_NOISE_P).to(device)
    print(f"  ⭐ VQ 錨定：每 token {VQ_V} 個 code（β={VQ_BETA:g} noise_p={VQ_NOISE_P:g}）", flush=True)


def _q(u):
    """推論用：VQ 開著就 snap 到最近 code；關著＝恆等。"""
    return vq.snap(u) if (vq is not None and not VQ_SOFT) else u


def _dec(u, s_n):
    """u → 座標序列（正規化）。DEC_START=hard 時整條平移使第 0 點＝s_n（[1,2] 或 [B,2]）；其餘＝u_dec(u)。
    ⛔ 所有解碼點都走這裡（訓練 recon、boot 出題、select、投影、latent/conf/conf2、往返尺、flow 探針）——不要另寫。"""
    if DEC_START != "hard":
        return u_dec(u)
    s2 = torch.as_tensor(s_n, dtype=u.dtype, device=u.device).reshape(-1, 2)
    if s2.shape[0] == 1 and u.shape[0] != 1:
        s2 = s2.expand(u.shape[0], -1)
    ctx = torch.cat([u, s_embed(s2)[:, None, :]], 1)   # ⭐ 起點當第 K+1 個 token 餵 decoder（9/2 晚第四版）
    pts = u_dec(ctx)
    if True:
        # ⭐ 位移形式（9/2 晚第三版）：第 0 點＝起點，decoder 的第 1..T-1 個輸出＝相對起點的位移。
        #    ⛔ 前兩版「整條減掉自己的第 0 點再加 s」都卡在 recon 0.3~0.46（V8 0.04）：不 detach 時參考點吃到 127 點誤差總和，
        #    detach 後參考點又變成會漂的移動靶。位移形式沒有參考點耦合，是原目標的線性重參數化。
        pts = torch.cat([s2[:, None, :], s2[:, None, :] + pts[:, 1:]], 1)
    return pts

# ⭐ 起步暖身（LACOT_WARMUP=N）：兩段訓練各自的前 N 步 lr 線性 0→base。病理依據 2026-09-01
#    驗屍：s1/s4 型爛 seed＝encoder 前期一步走歪就卡死（現狀等於全油門起步）。⛔ 預設 0＝行為不變。
WARMUP = int(os.environ.get("LACOT_WARMUP", 0))


def _warm_lr(opt, stp):
    if WARMUP <= 0 or opt is None:
        return
    if not hasattr(opt, "_base_lrs"):
        opt._base_lrs = [pg["lr"] for pg in opt.param_groups]
    if stp <= WARMUP:
        f = min(1.0, (stp + 1) / WARMUP)
        for pg, b in zip(opt.param_groups, opt._base_lrs):
            pg["lr"] = b * f


print(f"stage 1 e_target 目標={ENC_OBJ} ...  w_var={W_VAR} w_cov={W_COV}"
      + (f" w_ictr={W_ICTR} sigma={ICTR_SIGMA}" if ENC_OBJ == "recon_ictr" else "")
      + (f"  warmup={WARMUP}" if WARMUP else ""), flush=True)
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1500))
_S1 = 0 if (LOAD_CKPT or S1_FROM) else STEPS1
if S1_FROM and not LOAD_CKPT:
    # ⭐ 方言分辨實驗：stage 1 從別的 run 載入並凍住，之後 torch RNG 重播 SEED ⇒ 不同 SEED 只差 stage 2 的 init／資料序。
    _sk = torch.load(S1_FROM, map_location=device, weights_only=False)
    traj_enc.load_state_dict(_sk["traj_enc"]); e_pooler.load_state_dict(_sk["e_pooler"])
    if u_dec is not None:
        assert "u_dec" in _sk, f"⛔ S1_FROM ckpt 沒有 u_dec：{S1_FROM}"
        u_dec.load_state_dict(_sk["u_dec"])
    if s_embed is not None:
        assert "s_embed" in _sk, "⛔ S1_FROM ckpt 沒有 s_embed（DEC_START=hard 與 ckpt 不一致）"
        s_embed.load_state_dict(_sk["s_embed"])
    if vq is not None:
        assert "vq" in _sk, "⛔ S1_FROM ckpt 沒有 vq（LACOT_VQ 與 ckpt 不一致）"
        vq.load_state_dict(_sk["vq"])
    del _sk
    torch.manual_seed(SEED)
    print(f"  ⭐ S1_FROM={S1_FROM}：stage 1 載入並跳過訓練；torch RNG 重播 SEED={SEED}（只抽 stage 2）", flush=True)
_ed_hist, logits = [], None
for stp in range(_S1):
    traj, mask, s, g, _ = make_batch(rng, teacher_mix=TEACHER_MIX)   # stage1 不用 act ⇒ teacher 全參與
    et = etarget(traj, mask)
    if ENC_OBJ == "sg_infonce":
        q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        _main = loss
    else:
        if vq is not None:                          # ⭐ VQ：decoder 吃量化 u（直通）、加 commitment；軟錨＝吃連續 u
            et_dec, l_vq, _vqst = vq(et)
            if VQ_SOFT:
                et_dec = et
        else:
            et_dec, l_vq, _vqst = et, 0.0, None
        _pts1 = _dec(et_dec, s)                          # ⭐ 開頭綁定：hard 在這裡平移到 s
        if DEC_START and stp == 0:                       # 一次性哨兵：traj 第 0 點本來就該＝s（否則 hard 的前提錯）
            print(f"  ⭐ DEC_START={DEC_START}：‖traj[:,0]−s‖ max {float((traj[:, 0] - s).norm(dim=-1).max()):.4f}（應≈0）", flush=True)
        _main = (_pts1 - traj).pow(2).mean()             # ⭐ 重建 128 個座標點
        loss = _main + l_vq
        if DEC_START == "soft":                          # ⭐ 軟綁：第 0 點對起點的懲罰
            loss = loss + DEC_START_W * (_pts1[:, 0] - s).pow(2).mean()
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
    _warm_lr(opt1, stp)
    opt1.zero_grad(set_to_none=True); loss.backward()
    if DIAG_TRAIN and stp < 300 and stp % 10 == 0:
        # ⭐ 出生後前三百步逐幀（9/1 seed 病因診斷）：爛顆 vs 好顆的分岔點與最早死亡訊號
        _gs = [p.grad.norm() for g in opt1.param_groups for p in g["params"] if p.grad is not None]
        _gn = torch.norm(torch.stack(_gs)).item() if _gs else 0.0
        print(f"  DIAG stp {stp}  recon {_main.item():.4f}  總 {loss.item():.4f}"
              f"  grad {_gn:.3f}  effdim {eff_dim(et):.3f}"
              + (f"  vq ppl {_vqst['perplexity']:.1f} used {_vqst['used']}/{VQ_V}" if ENC_OBJ != "sg_infonce" and _vqst else ""), flush=True)
    opt1.step()
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
# ═══ 真獨立 GCBC＋chunk（LACOT_BC_OWN；8/24 floor 舊債、主人 2026-08-31 核為 paper 必做）═══
# 跟主模型【零共用】：自己的 encoder＋cond head＋action head、自己的 optimizer、獨立 backward。
# 同容量同預算（完整複製 cond 鏈的架構）、只吃真資料（real-mask）⇒ 這才是「GCBC+chunk4」對照。
# ⛔ bc_head（共 cond 的近似版）保留不動 —— 兩顆並存，歷史可比性不變。
BC_OWN = int(os.environ.get("LACOT_BC_OWN", 0))
bc_own_enc = bc_own_ch = bc_own_head = None
if BC_OWN:
    # 🚨 2026-08-31 檢討 R3 修：建構包 fork_rng ⇒ 不消耗全域 RNG 流。
    #    舊版三模組建構吃掉全域流 ⇒ 名義同 seed 的主模型初始化整個變掉（bcown 顆分段
    #    0.728/0.328 掉出常軌＝落進 seed 方差）。fork 內用自己的確定性 seed。
    with torch.random.fork_rng(devices=[device] if str(device) != "cpu" else []):
        torch.manual_seed(20260831 + SEED)
        bc_own_enc = sota_mlp(2, 512, 512).to(device)
        bc_own_ch = sota_mlp(1024, 512, COND).to(device)
        bc_own_head = CondOnlyMLP().to(device)
    def own_condvec(s, g):
        return bc_own_ch(torch.cat([bc_own_enc(s), bc_own_enc(g)], 1))
    print("  真獨立 GCBC 開啟（own encoder/head、零共用、只吃真資料；fork_rng 隔離）", flush=True)
# ⭐ BC_INDEP：bc 地板拆出去用自己的 optimizer ＋ 自己的 grad-clip。
#    主人 8/24 對 floor 的定義是「真正 BC 能到達的」—— 而共用 opt2 與全域 clip 的話，
#    它的更新會被主模型的梯度規模牽著走 ⇒ ⛔ 那不是獨立 baseline。
#    ⚠️ 開了之後 bc 的數字跟歷史結果不可直接比 ⇒ 預設仍是 0。
f_mods = ([cond_enc, cond_head, flow, refine, ahead] if BC_INDEP
          else [cond_enc, cond_head, flow, refine, ahead, bc_head])
opt2 = _apply_lr_scale(torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4))
opt_bc = _apply_lr_scale(torch.optim.Adam(bc_head.parameters(), lr=5e-4)) if BC_INDEP else None
opt_bc_own = (torch.optim.Adam([p for m in (bc_own_enc, bc_own_ch, bc_own_head)
                                for p in m.parameters()], lr=5e-4) if BC_OWN else None)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()
print("stage 2 flow+refine+action ...", flush=True)
STEPS2 = 0 if LOAD_CKPT else int(os.environ.get("LACOT_STEPS2", 2000))
# ═══ 權重 EMA（LACOT_EMA_W；「練穩」B 問題的藥之一，2026-08-31）══════════════
# 訓練不變（影子不參與 forward/backward）；ckpt 多存一份 ema 權重；
# eval 時 LACOT_LOAD_EMA=1 用影子 ⇒ 同一次訓練、兩種權重可對照（配對比較、seed 噪聲咬不到）。
# ⚠️ 對象＝stage2 會更新的供點鏈五模組（u_dec 只在 stage1 訓、refine 在 norf 慣例下閒置 ⇒ 皆不追）。
EMA_W = float(os.environ.get("LACOT_EMA_W", 0.0))    # 0=off（歷史行為不變）；慣例 0.999
_EMA_PAIRS = []
if EMA_W > 0 and not LOAD_CKPT:
    import copy as _cp
    _EMA_NAMED = [("cond_enc", cond_enc), ("cond_head", cond_head), ("flow", flow),
                  ("ahead", ahead), ("bc_head", bc_head)]
    _EMA_SHADOW = {}
    for _n, _m in _EMA_NAMED:
        _sh = _cp.deepcopy(_m)
        for _p in _sh.parameters():
            _p.requires_grad_(False)
        _EMA_SHADOW[_n] = _sh
        _EMA_PAIRS.append((_sh, _m))
    print(f"  權重 EMA 開啟：m={EMA_W:g}（影子五模組，訓練行為不變）", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng, teacher_mix=TEACHER_MIX)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    # ⭐ P1b：action loss 只算真樣本 —— teacher 樣本的 act 是零佔位，
    #    ⛔ 餵給 head/bc 等於教「這些 cond 下不要動」。_rw 全 1 時退化回原味。
    _rw = _REAL_W[0]
    _wmse = lambda p, a: ((p - a).pow(2).reshape(len(p), -1).mean(1) * _rw).sum() / _rw.sum().clamp(min=1)
    # ⭐ COND_DROP：l_anchor 這一路以 p 機率把整個 cond 歸零，逼 head 從 u 讀路徑。
    #    ⛔ 只丟 cond、不丟 u —— 丟 u 會反過來教 head 繞開 u。
    _ca = cond
    if COND_DROP > 0:
        _keep = (torch.rand(len(cond), 1, device=cond.device) >= COND_DROP).float()
        _ca = cond * _keep
    l_anchor = _wmse(ahead(_ca, _q(et)), act)         # ⭐ VQ：head 吃量化 u（與推論一致）
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
    l_bc = _wmse(bc_head(cond.detach()), act)
    if BC_OWN:
        # ⭐ 完全獨立的 backward（自己的 graph、自己的 clip）—— 對主模型零影響
        l_bco = _wmse(bc_own_head(own_condvec(s, g)), act)
        opt_bc_own.zero_grad(set_to_none=True)
        l_bco.backward()
        torch.nn.utils.clip_grad_norm_([p for m in (bc_own_enc, bc_own_ch, bc_own_head)
                                        for p in m.parameters()], 1.0)
        opt_bc_own.step()
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons + (0.0 * l_bc if BC_INDEP else l_bc)
    _warm_lr(opt2, stp)
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
    if _EMA_PAIRS:
        with torch.no_grad():
            for _me, _ml in _EMA_PAIRS:
                for _pe, _pl in zip(_me.parameters(), _ml.parameters()):
                    _pe.mul_(EMA_W).add_(_pl, alpha=1 - EMA_W)
                for _be, _bl in zip(_me.buffers(), _ml.buffers()):
                    _be.copy_(_bl)
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor {l_anchor.item():.4f} l_refine {l_refine.item():.4f}", flush=True)
TAG_SEED = SEED          # 檔名用的 seed；載入模式下改跟 ckpt 的 seed 走（2026-08-31 修互蓋病）
LOAD_EMA = 0             # 載入模式下由 LACOT_LOAD_EMA 蓋掉；訓練模式恆 0
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
    if BC_OWN:
        assert "bc_own" in _ck, "⛔ BC_OWN=1 但這顆 ckpt 沒有 bc_own 段（訓練時沒開 LACOT_BC_OWN）"
        bc_own_enc.load_state_dict(_ck["bc_own"]["enc"])
        bc_own_ch.load_state_dict(_ck["bc_own"]["ch"])
        bc_own_head.load_state_dict(_ck["bc_own"]["head"])
        print("  ⭐ 真獨立 GCBC 三模組已載入（bc 臂走 own 鏈）", flush=True)
    # ⭐ LACOT_LOAD_EMA=1：用影子權重覆蓋供點鏈五模組（同顆 ckpt 的 raw/ema 配對對照）
    LOAD_EMA = int(os.environ.get("LACOT_LOAD_EMA", 0))
    if LOAD_EMA:
        assert "ema" in _ck, "⛔ LOAD_EMA=1 但這顆 ckpt 沒有 ema 段（訓練時沒開 LACOT_EMA_W）"
        for _name, _sd in _ck["ema"].items():
            {"cond_enc": cond_enc, "cond_head": cond_head, "flow": flow,
             "ahead": ahead, "bc_head": bc_head}[_name].load_state_dict(_sd)
        print(f"  ⭐ 已切到 EMA 影子權重（m={_cfg.get('EMA_W', '?')}，五模組）", flush=True)
    if u_dec is not None:
        assert "u_dec" in _ck, (
            "⛔ ENC_OBJ=recon* 但 ckpt 裡沒有 u_dec ⇒ 那顆 ckpt 是舊版存的，"
            " 沒有 decoder 就沒有 E_geo 的眼睛")
        u_dec.load_state_dict(_ck["u_dec"])
        if "s_embed" in _ck:                              # ⭐ hard 綁定 ckpt：起點 token 的嵌入層
            assert DEC_START == "hard", "⛔ 這顆 ckpt 是 DEC_START=hard 訓的，eval 要設 LACOT_DEC_START=hard"
            s_embed.load_state_dict(_ck["s_embed"]); s_embed.eval()
        else:
            assert DEC_START != "hard", "⛔ LACOT_DEC_START=hard 但這顆 ckpt 沒有 s_embed（訓練時沒開 hard）"
        if "vq" in _ck:                                   # ⭐ VQ ckpt：eval 端自動帶起 codebook
            if vq is None:
                from lacot.vq import TokenVQ
                vq = TokenVQ(int(_ck["vq_cfg"]["V"]), D_MODEL).to(device)
            vq.load_state_dict(_ck["vq"]); vq.eval()
            VQ_SOFT = int(_ck["vq_cfg"].get("soft", 0))          # ⭐ 軟錨 ckpt ⇒ 推論也不 snap
            print(f"  ⭐ 已載入 VQ codebook（V={vq.V}）", flush=True)
        else:
            assert vq is None, "⛔ LACOT_VQ>0 但這顆 ckpt 沒有 vq 段（訓練時沒開）"
    # 🚨 2026-08-31 修：eval 檔名的 _s 段跟【ckpt 的 seed】走，⛔ 不是 env 的 LACOT_SEED。
    #    舊病：跨 seed 的 eval 支沒帶 LACOT_SEED ⇒ 全寫 _s0 ⇒ 不同顆的官方 json 互蓋
    #    （8/30 offLs0 被 offLs2 蓋、8/31 三對；⛔ 蓋掉的檔數值上完全合理、看不出來）。
    #    ⚠️ 只動檔名，⛔ 不動 SEED 變數本身 —— eval 噪聲流跟 SEED 綁著，動了就跟歷史不可比。
    if _cfg.get("SEED") is not None:
        TAG_SEED = _cfg["SEED"]
        if TAG_SEED != SEED:
            print(f"  ⚠️ 檔名 seed 跟 ckpt 走：_s{TAG_SEED}（env LACOT_SEED={SEED} 只管噪聲流）", flush=True)
    if _cfg.get("EMA_W"):
        EMA_W = _cfg["EMA_W"]        # ⭐ 只進檔名（顆身份的一部分）；影子建立條件含 not LOAD_CKPT、不受影響
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

# ═══ P2：自舉生成模式（LACOT_BOOT_GEN；主人 2026-08-31 核可「寫好＋smoke、不無人跑」）═══
# p* ∝ p_θ·exp(−βE) 的生成半邊：flow 自己生 M 份 → 幾何 E 打分 → 通過集存檔。
#   合法性＝【硬門檻】（wall ≈ 0 才算合法計畫 —— E 高分≠合法，這是 fuzz 精神的第一道縫）；
#   蒸餾權重＝【軟權重】exp(−βE)。⭐ npz 只存 E 原值，β 由訓練端算 ⇒ 換 β 檔位不用重生成。
# 課程半徑：(s,g) 從佔據圖抽、BFS 距離 ∈ [BOOT_RMIN, BOOT_RMAX] 格 —— 逐輪推遠的旋鈕。
# 跟 teacher 引擎同一張圖（GeoEnergy res=8、資訊 ⊆ D）⇒ 公平性同一條論證。
# ⚠️ 這段跑在 GEO（L745+）定義之前 ⇒ ⛔ 不可引用 GEO，自己建（同參數、便宜）。
BOOT_GEN = os.environ.get("LACOT_BOOT_GEN", "")
if BOOT_GEN:
    assert LOAD_CKPT, "⛔ 生成模式要在載入模式下跑（LACOT_LOAD_CKPT）—— 別拿沒訓過的權重生樣本"
    assert u_dec is not None, "⛔ 生成模式需要 decoder（ENC_OBJ=recon*）—— E 靠它把 u 解成座標"
    BOOT_Q = int(os.environ.get("LACOT_BOOT_Q", 512))        # 題數
    BOOT_M = int(os.environ.get("LACOT_BOOT_M", 8))          # 每題生成份數（pass@M 的 M）
    BOOT_RMIN = int(os.environ.get("LACOT_BOOT_RMIN", 8))    # 課程半徑下限（BFS 格數）
    BOOT_RMAX = int(os.environ.get("LACOT_BOOT_RMAX", 25))   # 課程半徑上限 —— 逐輪推遠的旋鈕
    BOOT_WMAX = float(os.environ.get("LACOT_BOOT_WMAX", 0.05))   # 合法性硬門檻：穿牆均值上限
    BOOT_EPT = float(os.environ.get("LACOT_BOOT_EPT", 0.25))     # 端點硬門檻：首尾偏離 s/g 上限
    from lacot.refine_grad import GeoEnergy as _BgGeo
    from lacot.subgoal import grid_bfs as _bg_bfs
    _bgeo = _BgGeo(OBS, mu, sd, res=8, device=device, w_len=W_LEN)
    _bocc = (_bgeo.dist[0, 0].cpu().numpy() == 0.0)
    _bfree = np.argwhere(_bocc)
    _blo = np.asarray(_bgeo.lo, np.float64)
    _bcell = np.asarray(_bgeo.hi - _bgeo.lo, np.float64) / (np.asarray(_bgeo.shape, np.int64) - 1)
    _brng = np.random.default_rng(20260831 + SEED)
    # ⭐ 荒漠出題（LACOT_BOOT_DESERT=1）：起點格從「均勻」改成「按訓練資料密度反比」加權抽。
    #    病理依據 2026-09-01 驗屍：稀疏區＝學習訊號弱的抽籤區、s2 死區即在此。
    #    ⛔ 半徑課程、硬門檻、軟權重全不動；預設關＝出題分布與 v1 完全一致。
    BOOT_DESERT = os.environ.get("LACOT_BOOT_DESERT", "0") == "1"
    _bp = None
    if BOOT_DESERT:
        # ⚠️ GeoEnergy 的格制建在【正規化】座標（(xy−mu)/sd，見 refine_grad.py L50）——
        #    OBS 是原始座標，先正規化再映射。⛔ 直接用原始座標＝覆蓋率 0.004 的錯位（smoke 抓過）。
        _dz = (OBS[:, :2] - np.asarray(mu, np.float64)[:2]) / np.asarray(sd, np.float64)[:2]
        _dij = np.rint((_dz - _blo) / _bcell).astype(np.int64)
        _fidx = {tuple(map(int, c)): k for k, c in enumerate(_bfree)}
        _cnt = np.zeros(len(_bfree))
        for _ij in _dij:
            _k = _fidx.get((int(_ij[0]), int(_ij[1])))
            if _k is not None:
                _cnt[_k] += 1
        _bp = 1.0 / (_cnt + 0.1 * max(float(_cnt[_cnt > 0].mean()), 1.0) + 1.0)
        _bp = _bp / _bp.sum()
        # ⚠️ 覆蓋率是座標系對齊的哨兵：資料點落進 open 細格的比例太低＝xy→格映射錯位。
        print(f"  ⭐ 荒漠出題：{len(_bfree)} 起點格、零資料格 {int((_cnt == 0).sum())} 個、"
              f"最稀/最富權重比 {float(_bp.max() / _bp.min()):.1f}x、"
              f"資料覆蓋率 {float(_cnt.sum()) / len(OBS):.3f}", flush=True)
    # 課程題庫：抽 a → BFS 距離場 → 從 d∈[RMIN,RMAX] 的格挑 b（一場 BFS 供多題、快）
    _bq = []
    while len(_bq) < BOOT_Q:
        a = tuple(_bfree[int(_brng.choice(len(_bfree), p=_bp)) if _bp is not None
                  else int(_brng.integers(len(_bfree)))])
        d = _bg_bfs(_bocc, a)                                # dict[(i,j)]=BFS 步數
        cand = [c for c, dr in d.items() if BOOT_RMIN <= dr <= BOOT_RMAX]
        if not cand:
            continue
        _pi = _brng.permutation(len(cand))[:8]               # 一個起點最多貢獻 8 題（別讓單點主導）
        for b in (cand[int(i)] for i in _pi):
            _bq.append((a, b, int(d[b])))
            if len(_bq) >= BOOT_Q:
                break
    print(f"  自舉題庫：{len(_bq)} 題（BFS 距離 {BOOT_RMIN}~{BOOT_RMAX} 格）", flush=True)
    _keep_t, _keep_e, _keep_s, _keep_g, _keep_r = [], [], [], [], []
    _npass = 0
    with torch.no_grad():
        for a, b, dr in _bq:
            sn = _blo + np.asarray(a, np.float64) * _bcell + _brng.uniform(-0.4, 0.4, 2) * _bcell
            gn = _blo + np.asarray(b, np.float64) * _bcell + _brng.uniform(-0.4, 0.4, 2) * _bcell
            st = torch.tensor(sn, dtype=torch.float32, device=device)[None]
            gt = torch.tensor(gn, dtype=torch.float32, device=device)[None]
            cond_b = condvec(st, gt)
            u_b = flow.sample(BOOT_M, cond_b.expand(BOOT_M, -1))
            pts = _dec(_q(u_b), st)                           # [M, T_CAP, 2] 正規化
            e_tot, e_terms = _bgeo(pts, st.expand(BOOT_M, -1), gt.expand(BOOT_M, -1), per_term=True)
            # 合法性硬門檻：穿牆 ≈0 ＋ 首尾錨在 (s,g) —— ⛔ 不合法的再低 E 也不進蒸餾集
            ok = ((e_terms["wall"] < BOOT_WMAX) & (e_terms["start"] < BOOT_EPT)
                  & (e_terms["goal"] < BOOT_EPT))
            if bool(ok.any()):
                _npass += 1
                idx = torch.nonzero(ok).flatten()
                _keep_t.append(pts[idx].cpu().numpy().astype(np.float32))
                _keep_e.append(e_tot[idx].cpu().numpy().astype(np.float32))
                _keep_s.append(np.repeat(sn[None], len(idx), 0).astype(np.float32))
                _keep_g.append(np.repeat(gn[None], len(idx), 0).astype(np.float32))
                _keep_r.append(np.full(len(idx), dr, np.int32))
    _pass_at_m = _npass / max(len(_bq), 1)
    if _npass == 0:
        # ⛔ 空檔案不落地 —— 載到空自舉集的訓練會安靜地退化成純 ebfs-teacher（檢查不會叫）
        raise SystemExit(f"⛔ pass@{BOOT_M} = 0/{len(_bq)} —— 一題都沒通過，自舉樣本檔不寫。"
                         f" 半徑 {BOOT_RMIN}~{BOOT_RMAX} 對這顆太遠，先降 RMAX。")
    _bt = np.concatenate(_keep_t); _be = np.concatenate(_keep_e)
    _bs = np.concatenate(_keep_s); _bgl = np.concatenate(_keep_g); _br = np.concatenate(_keep_r)
    os.makedirs(os.path.dirname(os.path.abspath(BOOT_GEN)) or ".", exist_ok=True)
    np.savez_compressed(BOOT_GEN, trajs=_bt, E=_be, s=_bs, g=_bgl, bfs_r=_br,
                        meta=json.dumps(dict(ckpt=os.path.basename(LOAD_CKPT), Q=len(_bq),
                                             M=BOOT_M, rmin=BOOT_RMIN, rmax=BOOT_RMAX,
                                             wmax=BOOT_WMAX, ept=BOOT_EPT,
                                             pass_at_m=_pass_at_m, env=ENV_NAME)))
    # 🚨 pass@M 是多樣性警報器：逐輪往下掉＝多樣性死（Goodhart 進場）⇒ 停迭代
    print(f"==== BOOT_GEN 完成 ====", flush=True)
    print(f"  pass@{BOOT_M}: {_npass}/{len(_bq)} = {_pass_at_m:.3f}   通過樣本 {len(_bt)} 條"
          f"   E 分布 p10/p50/p90 {np.percentile(_be, 10):.3f}/{np.percentile(_be, 50):.3f}/"
          f"{np.percentile(_be, 90):.3f}", flush=True)
    print(f"  半徑分布：p50 {np.median(_br):.0f} max {_br.max()} 格   寫入 {BOOT_GEN}", flush=True)
    raise SystemExit(0)

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
        if BC_OWN:                                 # ⭐ 真獨立 GCBC：全鏈自有權重（8/31）
            a = bc_own_head(own_condvec(s, g))[0].cpu().numpy()
        else:
            a = bc_head(cond)[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u == "shuf":                            # ⭐ 別人的 u，本題的 cond
        a = ahead(cond, _foreign_u(R))[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u:
        u = _oracle_u(obs, goal, 1) if U_SOURCE == "oracle" else None   # ⭐ 9/2 探針
        if u is None:
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
            if GRAD_MODE == "select" and R > 0:
                # ⭐ energy-guided selection（主人 8/29 晚核准的中間站一）：
                #    抽 N 份、E 打分、挑最低 —— 挑出來的永遠是 flow 原生樣本（殼上的點）
                #    ⇒ 零方言病，E 的幾何品味直接兌現。⛔ 不碰快取（每 chunk fresh 選）。
                cand = flow.sample(SEL_N, cond.expand(SEL_N, -1))
                with torch.no_grad():
                    _e = GEO(_dec(_q(cand), s), s.expand(SEL_N, -1), g.expand(SEL_N, -1))
                u = cand[int(_e.argmin())][None]
            else:
                _use_warm, _steps = grad_steps(R, _GRAD_CACHE["u"] is not None, GRAD_R, GRAD_R_WARM)
                if _steps > 0:
                    if _use_warm:
                        u = _GRAD_CACHE["u"]                 # 接續上一個 chunk 的計畫
                    u = grad_refine(u, cond, u_dec, flow, GEO, s, g,
                                    steps=_steps, eta=GRAD_ETA, lam=GRAD_LAM)
                    if GRAD_PROJ:
                        # ⭐ 中間站二：encoder 往返投影 —— 爬完拉回 head 熟悉的殼上再用。
                        #    這格若讓爬坡從「變差」變「不差」，方言假說確診。
                        with torch.no_grad():
                            u = encode_u(_dec(_q(u), s))
                    _GRAD_CACHE["u"] = u
                # ⛔ _steps == 0（R=0）⇒ flow 抽的 u 直接用，⛔ 不碰 _GRAD_CACHE
        else:
            u = _apply_refine(cond, u, R)
    else:
        u = torch.zeros(1, K, D_MODEL, device=device)  # (s,g)-only floor
    a = ahead(cond, _q(u))[0].cpu().numpy()  # [CHUNK,2]
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
# 幾何 energy：SUBGOAL=latent/conf（長程層要它修）與 GRAD_REFINE（短程層要它修）需要
#   GEO＋decoder；SUBGOAL=ebfs（E 圖搜索供點）只要 GEO 的【佔據圖】、⛔ 不需要 decoder
#   （它不走 latent 路，直接在資料重建的圖上搜）。
_NEED_DEC = SUBGOAL in ("latent", "conf", "conf2") or GRAD_REFINE
if _NEED_DEC:
    assert u_dec is not None, (
        f"⛔ {'SUBGOAL=' + SUBGOAL if SUBGOAL else 'GRAD_REFINE=1'} 需要 decoder，"
        f" 而 ENC_OBJ={ENC_OBJ} 沒有訓 decoder。"
        " ⇒ 用 ENC_OBJ=recon/recon_ictr（或載一顆有 u_dec 的 ckpt），或改用 SUBGOAL=bfs")
if _NEED_DEC or SUBGOAL == "ebfs":
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
if _NEED_DEC:
    # 🚨 decoder 是 E_geo 的【眼睛】—— 它若不讀 u，爬坡就是在對一條固定的平均路做最佳化，
    #    ⛔ 而且不會報錯：V 照樣會上升（它在改那條平均路），u 卻沒有任何意義。
    #    ⇒ 跟上面穿牆那格同層級：不過就停。（ebfs 不進這格：它不用 decoder。）
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

if SUBGOAL == "ebfs" or SUB_SNAP or SUB_HEADGUARD > 0:   # ⭐ 9/2：SNAP／HEADGUARD 也要 E 格 helper（只有定義與一行 print）
    # ⭐ E 圖搜索供點（主人 2026-08-30「energy 自己 reason 串接」）：subgoal 由
    #    【資料重建佔據圖】上的 BFS 生 —— 跟 oracle 格（SUBGOAL=bfs）同一套挑點邏輯，
    #    差別只在圖：bfs 用 env.maze_map（真圖＝privileged、只准當診斷），
    #    ebfs 用 GEO 的佔據圖（資訊全來自 D ⇒ 可部署、可進對標表）。
    #    可行性 gate（8/30 探針）：medium/large tier2 各 100/100 連通、端點 snap 0、
    #    E步/真格比值穩定（3.1／5.2）⇒ 4 鄰就夠。
    _EOCC = (GEO.dist[0, 0].cpu().numpy() == 0.0)      # 自由空間＝資料走過的格
    from scipy.ndimage import distance_transform_edt as _edt
    _E_CLEAR = _edt(_EOCC)                              # ⭐ 每個自由格離最近非自由格幾格（淨空；9/2 路標吸附用）
    _EFREE = np.argwhere(_EOCC)
    _E_LO = np.asarray(GEO.lo, np.float64)
    _E_SPAN = np.asarray(GEO.hi - GEO.lo, np.float64)
    _E_SHAPE = np.asarray(GEO.shape, np.int64)

    def _e_xy_to_cell(xy):
        """原始座標 → E 細格；落在牆格就 snap 到最近自由格（探針實測 snap 恆 0，保底用）。"""
        z = (np.asarray(xy[:2], np.float64) - mu) / sd
        idx = np.clip(np.round((z - _E_LO) / _E_SPAN * (_E_SHAPE - 1)).astype(int),
                      0, _E_SHAPE - 1)
        c = tuple(idx)
        if _EOCC[c]:
            return c
        return tuple(_EFREE[int(np.abs(_EFREE - idx).sum(1).argmin())])

    def _e_cell_to_xy(c):
        z = _E_LO + np.asarray(c, np.float64) / (_E_SHAPE - 1) * _E_SPAN
        return z * sd + mu

    def _e_bfs_from(_env_unused, src):
        """GEO 佔據圖上的 BFS。介面對齊 DE._bfs_from ⇒ bfs_subgoal 一行不用改。
        ⭐ 實作在 lacot.subgoal.grid_bfs（單一來源，teacher 資料引擎共用）。"""
        from lacot.subgoal import grid_bfs
        return grid_bfs(_EOCC, src)

    # DELTA_SUB（xy 單位）→ E 細格數：E 格的 xy 尺寸 = span_norm/(shape-1) × sd（兩維平均）。
    # ⚠️ E 格是各向異性的長方形（x/y 的 sd 不同），平均是近似 —— subgoal 隔多遠本來就是超參。
    _E_CELL_XY = float(np.mean(_E_SPAN / (_E_SHAPE - 1) * sd))
    _E_DELTA_CELLS = max(1, int(round(DELTA_SUB / _E_CELL_XY)))
    print(f"  E 圖 subgoal：佔據圖 {tuple(GEO.shape)} 覆蓋 {GEO.coverage:.1%}，"
          f"E 格寬≈{_E_CELL_XY:.2f}，subgoal 隔 {_E_DELTA_CELLS} 格", flush=True)


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
            "spread": [], "n_direct": 0, "n_fallback": 0, "esel_e_gap": []}


def _anchor_pts(pts_raw, obs):
    """DEC_ANCHOR：把解碼路徑整條平移到「第 0 點＝當前位置」（原始座標、[B,T,2] 各自平移）。

    ⛔ 平移是等距變換：弧長、形狀、點間距全不變，只修絕對位置 ⇒ 拆「位置錯 vs 形狀錯」。
    """
    if not DEC_ANCHOR:
        return pts_raw
    s0 = torch.as_tensor(np.asarray(obs[:2], np.float32), device=pts_raw.device)
    return pts_raw - pts_raw[:, :1] + s0


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
        box.setdefault("ep_count", {})
        box["ep_count"][task] = box["ep_count"].get(task, 0) + 1
        box["trace"] = bool(TRACE_TASK) and task == TRACE_TASK and box["ep_count"][task] == 1
        box["trace_t"] = 0
        box["hg_sticky"] = False
        if box["trace"]:
            print(f"  🔎 TRACE task {task}: start ({obs[0]:.2f},{obs[1]:.2f}) goal ({goal[0]:.2f},{goal[1]:.2f})", flush=True)

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
                pts_n = _dec(_q(u_l), s_n)               # [1, T_CAP, 2] 正規化座標
                pts_raw = pts_n * SD + MU                # ⭐ 換回原始座標 ⇒ 跟 DELTA_SUB 同單位
                pts_raw = _anchor_pts(pts_raw, obs)
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
                pts_raw = _anchor_pts(_dec(_q(u_l), s_n) * SD + MU, obs)   # [M, T_CAP, 2] 原始座標
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
            _NS = SUB_ESEL if SUB_ESEL > SUB_M else SUB_M       # ⭐ 9/2 E 選計畫：先抽 N 份
            cond_l = condvec(s_n, g_n).expand(_NS, -1)
            u_l = _oracle_u(obs, box["goal"], _NS) if U_SOURCE == "oracle" else None   # ⭐ 9/2 探針
            if u_l is None:
                u_l = flow.sample(_NS, cond_l).detach()
            u_l = grad_refine(u_l, cond_l, u_dec, flow, GEO,
                              s_n.expand(_NS, -1), g_n.expand(_NS, -1),
                              steps=GRAD_R, eta=GRAD_ETA, lam=GRAD_LAM)
            if _NS > SUB_M:                                     # 用 E 留最低的 SUB_M 份（越小越好）
                with torch.no_grad():
                    _eN = GEO(_dec(_q(u_l), s_n), s_n.expand(_NS, -1), g_n.expand(_NS, -1))
                    _keep = torch.topk(-_eN, SUB_M).indices
                SUB_DIAG["esel_e_gap"].append(float(_eN.max() - _eN.min()))
                u_l, cond_l = u_l[_keep], cond_l[_keep]
            with torch.no_grad():
                pts_raw = _anchor_pts(_dec(_q(u_l), s_n) * SD + MU, obs)   # [M, T_CAP, 2] 原始座標
            sub, _fc = farthest_confident_subgoal(
                pts_raw, box["goal"], min_arc=0.25 * DELTA_SUB, ret_stats=True,
                # 🚨 錨定把頭端分散人工歸零 ⇒ tau 校準窗挪到中段（見 subgoal.py 註解）
                calib=(0.2, 0.4) if DEC_ANCHOR else (0.0, 0.2),
                max_arc=SUB_MAX_ARC * DELTA_SUB if SUB_MAX_ARC > 0 else None)
            SUB_DIAG["spread"].append(_fc.get("spread", _fc["g_spread"]))
            if _fc["direct"]:
                SUB_DIAG["n_direct"] += 1                # 走到底（g 信心夠）
            if sub is None:                              # ③ 整條發散 ⇒ 固定弧長保底
                SUB_DIAG["n_fallback"] += 1
                sub = arc_subgoal(pts_raw, DELTA_SUB)[0].cpu().numpy()
            _d0 = float(np.linalg.norm(
                pts_raw[:, 0].mean(0).cpu().numpy() - np.asarray(obs[:2])))
            SUB_DIAG["d0"].append(_d0)
            if SUB_HEADGUARD > 0:                               # ⭐ 9/2 開頭守門（每份計畫各自的偏移取中位；黏住整集）
                _d0s = np.linalg.norm(pts_raw[:, 0].cpu().numpy() - np.asarray(obs[:2]), axis=1)
                _d0m = float(np.median(_d0s))
                _near_goal = float(np.linalg.norm(np.asarray(obs[:2]) - box["goal"])) <= DELTA_SUB
                if _d0m > SUB_HEADGUARD:
                    box["hg_sticky"] = True
                if box.get("hg_sticky") and not _near_goal:     # 計畫不從這裡出發 ⇒ 這一集改走 E 圖搜索
                    SUB_DIAG["n_headguard"] = SUB_DIAG.get("n_headguard", 0) + 1
                    _cg = bfs_subgoal(env, _e_xy_to_cell(obs), _e_xy_to_cell(box["goal"]),
                                      delta_cells=_E_DELTA_CELLS, bfs_from=_e_bfs_from)
                    sub = _e_cell_to_xy(_cg) if _cg is not None else box["goal"]
                    if box.get("trace"):
                        print(f"  🔎 headguard: d0 中位 {_d0m:.2f}（門檻 {SUB_HEADGUARD:g}，黏住）⇒ E 圖 BFS 路標 ({sub[0]:.2f},{sub[1]:.2f})", flush=True)
            if _d0 > 0.5 * DELTA_SUB:
                SUB_DIAG["n_bad_d0"] += 1
        elif SUBGOAL == "ebfs":
            # E 圖搜索供點：同一支 bfs_subgoal（含「嚴格更靠近目標」的挑點修正），
            # 只是圖換成資料重建的佔據圖、格制換成 E 細格。
            c = bfs_subgoal(env, _e_xy_to_cell(obs), _e_xy_to_cell(box["goal"]),
                            delta_cells=_E_DELTA_CELLS, bfs_from=_e_bfs_from)
            sub = _e_cell_to_xy(c) if c is not None else box["goal"]
        elif SUBGOAL == "bfs":
            c = bfs_subgoal(env, SUB_HELPERS[1](obs), SUB_HELPERS[1](box["goal"]),
                            delta_cells=max(1, int(round(DELTA_SUB / SUB_HELPERS[2]))),
                            bfs_from=DE._bfs_from)
            sub = np.asarray(env.unwrapped.ij_to_xy(c), np.float64) if c is not None else box["goal"]
        else:
            raise SystemExit(f"⛔ _plan 不認得 SUBGOAL={SUBGOAL}（新模式要顯式接，⛔ 不准靜默掉進別人的分支）")
        sub = np.asarray(sub, np.float64)
        if SUB_SNAP and SUBGOAL in ("conf", "conf2", "latent"):     # ⭐ 9/2 路標吸附
            c0 = np.asarray(_e_xy_to_cell(sub), np.int64)
            best, bestv = c0, -1.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c = c0 + np.array([di, dj])
                    if (c < 0).any() or (c >= _E_SHAPE).any() or not _EOCC[tuple(c)]:
                        continue
                    v = float(_E_CLEAR[tuple(c)]) - 0.01 * (abs(di) + abs(dj))   # 淨空優先、同淨空取近
                    if v > bestv:
                        best, bestv = c, v
            SUB_DIAG["n_snap"] = SUB_DIAG.get("n_snap", 0) + int(not np.array_equal(best, c0))
            sub = np.asarray(_e_cell_to_xy(tuple(int(x) for x in best)), np.float64)
        _ds = float(np.linalg.norm(sub - np.asarray(obs[:2])))
        SUB_DIAG["dsub"].append(_ds)
        if _ds > 2 * DELTA_SUB:                          # subgoal 遠到不像「一小段」
            SUB_DIAG["n_bad_dsub"] += 1
        if box.get("trace"):
            _cell_txt = ""
            if SUB_SNAP or SUBGOAL == "ebfs":
                _c = _e_xy_to_cell(sub); _cell_txt = f" E格{tuple(int(x) for x in _c)} 淨空{float(_E_CLEAR[tuple(_c)]):.1f}"
            _head = ""
            try:
                _pr = pts_raw[0, :6].cpu().numpy()
                _head = " 計畫頭 " + " ".join(f"({x:.1f},{y:.1f})" for x, y in _pr)
            except Exception:
                pass
            print(f"  🔎 replan@t={box.get('trace_t', 0)} 位置 ({obs[0]:.2f},{obs[1]:.2f}) → 路標 ({sub[0]:.2f},{sub[1]:.2f}) 距 {_ds:.2f}{_cell_txt}{_head}", flush=True)
        return sub

    def policy(obs, goal):
        if box["goal"] is None:                          # ⚠️ 沒接 on_start 的呼叫端也要能跑
            box["goal"] = np.asarray(goal[:2], np.float64)
        if box.get("trace"):
            box["trace_t"] = box.get("trace_t", 0) + 1
            if box["trace_t"] % 200 == 0:
                print(f"  🔎 t={box['trace_t']} 位置 ({obs[0]:.2f},{obs[1]:.2f}) 現路標 {None if planner.sub is None else tuple(round(float(v), 2) for v in planner.sub)}", flush=True)
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


# ⭐ 行為驗屍開關（LACOT_DIAG_DUMP=1）：官方 rollout 每一集落 start/goal/final/距離/步數，
#    分辨「卡死不動／走偏／差臨門」三種死法。⛔ 預設關 ⇒ 既有行為與 rollout json 完全不變。
DIAG_DUMP = os.environ.get("LACOT_DIAG_DUMP", "0") == "1"
DIAG_ROWS = []


def rollout(R, use_u, tag, policy_fn=None, on_start=None):
    """官方協定 rollout。policy_fn/on_start 給了就用它（分段臂），否則走 policy_chunk（flat）。"""
    succ, ep = 0, 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(SEEDS):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            _st0 = np.asarray(obs[:2], np.float64).tolist()
            torch.manual_seed(7 * task + sd_)  # action-sampler stream
            # 🚨 2026-08-28 修：這條官方路徑【從來沒有】重置過爬坡快取 —— 而 dev 那條有
            #    （dev_rollout 掛了 on_episode_start=_reset_grad_cache）。
            #    ⇒ 上一集爬出來的 u 會被下一集當 warm 起點，跨集、跨 task、跨 arm 互相汙染，
            #      ⛔ 而且不會報錯：後面每一集都拿到「別題的計畫」，成績照樣算得出來。
            #    ⚠️ _GRAD_CACHE 定義處的註解自己就寫著「⛔ 每集一定要重置」。
            _reset_grad_cache()
            if on_start is not None:           # ⭐ 分段 policy 有狀態 ⇒ 每集重置（同 dev 那條）
                on_start(obs, goal, task)                 # ⭐ 9/2：傳真 task（單集追蹤用；on_start 其餘不吃它）
            _reseed_shuf(1000 * task + sd_)    # #16：shuf arm 也要每集釘死 ⇒ 各 arm 配對
            while steps < MAXH and not success:
                for a in (policy_fn(obs, goal) if policy_fn is not None
                          else policy_chunk(obs, goal, R, use_u)):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
            if DIAG_DUMP:                    # 成功集也記 —— 讀屍體要有活人對照
                _fp = np.asarray(obs[:2], np.float64)
                _gl = np.asarray(goal[:2], np.float64)
                DIAG_ROWS.append(dict(
                    arm=str(tag).strip(), task=task, sd=sd_, start=_st0,
                    goal=_gl.tolist(), final=_fp.tolist(),
                    dist_final=float(np.linalg.norm(_fp - _gl)),
                    dist_start=float(np.linalg.norm(np.asarray(_st0) - _gl)),
                    steps=steps, success=bool(success)))
    print(f"  {tag}: success {succ}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep

# ⭐ embedding 軟尺（9/2 主人「先確認 embedding 好」）：五個官方任務各做一次「任務路徑→enc→(VQ)→dec」往返，
#    印 mse／解出路徑的穿牆深度／末點距終點；不跑模擬器。載入 ckpt 評測時才算（訓練跑本身不算）。
RT_GATE = None
FLOW_PROBE_OUT = None
if LOAD_CKPT and _TCH is not None and u_dec is not None and GEO is not None:
    _rt_sg = []
    for _t in range(1, N_TASKS + 1):
        _o, _i = env.reset(seed=1000 * _t, options={"task_id": _t, "render_goal": False})
        _rt_sg.append((np.asarray(_o[:2], np.float64), np.asarray(_i["goal"][:2], np.float64)))
    RT_GATE = roundtrip_gate(_rt_sg)
    if FLOW_PROBE > 0:
        FLOW_PROBE_OUT = flow_probe(_rt_sg, FLOW_PROBE)
        print(f"  ⭐ flow 探針（每題抽 {FLOW_PROBE} 份；進度中位/進度≥0.8 比例(oracle 進度)｜開頭段對路率/穿牆中位）：" + "  ".join(
            (f"t{k+1}:{r['prog_med']:.2f}/{r['prog_frac80']:.2f}({r['oracle_prog']:.2f})|{r['onroute']:.2f}/{r['wall_med']:.3f}" if r else f"t{k+1}:—")
            for k, r in enumerate(FLOW_PROBE_OUT)), flush=True)
    else:
        FLOW_PROBE_OUT = None
    print("  ⭐ embedding 往返尺（任務路徑→enc→dec；mse/穿牆/末點距）：" + "  ".join(
        (f"t{k+1}:{r['mse']:.4f}/{r['wall']:.3f}/{r['gdist']:.2f}" if r else f"t{k+1}:—") for k, r in enumerate(RT_GATE)), flush=True)
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
           sub_cap_chunks=SUB_CAP, sub_stuck_chunks=SUB_STUCK, dec_anchor=DEC_ANCHOR,
           teacher_mix=TEACHER_MIX,
           dev_tiers=DEV_TIERS, dev_eval=None, load_ckpt=os.path.basename(LOAD_CKPT) or None)
out["rt_gate"] = RT_GATE
out["flow_probe"] = FLOW_PROBE_OUT
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
    _rz, _sz = dev_rollout(0, False, "u 歸零     ")
    _R_ARM = max(1, min(RS_PRE))
    _rm, _sm = dev_rollout(_R_ARM, True, f"LaCoT (R={_R_ARM})   ")
    # ⭐ 特異度受測臂（主人 2026-08-30 裁定）：同一顆模型、同配置、只換 action-noise stream。
    #    ⛔ 受測對象必須是【會抽樣】的 arm（lacot 走 flow.sample）—— bc 不消耗 torch 亂數，
    #    拿 bc 當受測臂時換 tseed 是 no-op ⇒ 兩臂逐位元相同 ⇒ 那格什麼都沒驗到（8/28-8/29 的紅燈）。
    _rm2, _sm2 = dev_rollout(_R_ARM, True, f"LaCoT 重跑 ", tseed=71337)
    # 🚨 2026-08-28 修：舊版分段 arm 寫死 R=0，而它的對手 LaCoT 用 R=1。
    #    ⇒ 短程層一步都不爬（疊上 policy_chunk 的 R=0 凍結 bug ⇒ 整集凍在第一個 u），
    #      而 ("subgoal","lacot") 這個 pair 照樣算得出 p 值 ⇒ 差值會被讀成「階層化沒幫助」。
    #    ⇒ 兩邊用【同一個 R】，差的才只有「有沒有分段」這一件事。
    _rsg, _ssg = (dev_rollout(_R_ARM, True, f"分段 {SUBGOAL:<6}", subgoal=True)
                  if SUBGOAL else (None, None))
    # ✅ 2026-08-30 主人裁定（前身：docs/2026-08-28-fable-plan-verification.md ④ 的二選一）：
    #    特異度受測對象換成 LaCoT 主臂的兩次重跑（"bc 重跑" 臂退場，成本不變＝一個 arm）。
    #    紅燈史：8/28 發現 bc 不抽樣 ⇒ 換 tseed 是 no-op ⇒ 那格恆為位元相同；
    #    8/28~8/29 每輪誠實紅燈；8/30 主人裁定後換臂，這格才第一次真的在驗配對。
    _named = {"random": _rr, "bc": _rb, "lacot_rerun": _rm2, "null_u": _rz, "lacot": _rm}
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

    _chk = DE.sanity_check(_named, spec_pair=("lacot", "lacot_rerun"),
                           report_pairs=tuple(_pairs))
    for _n in _chk["notes"]:
        print("    " + _n, flush=True)
    print(f"  ⇒ 尺的驗收 {'✓ 通過' if _chk['passed'] else '🚨 沒過 ⇒ 這把尺還分不開已知不同的東西'}",
          flush=True)
    out["dev_eval"] = dict(n_tasks=len(DEV_TASKS), passed=bool(_chk["passed"]),
                           gates=_chk["gates"], random=_sr, bc=_sb, lacot_rerun=_sm2,
                           null_u=_sz, lacot_r1=_sm, notes=_chk["notes"],
                           **({"subgoal": _ssg, "subgoal_mode": SUBGOAL} if _ssg else {}))
    # 🚨 2026-08-28 補：per-episode 明細一定要落 json。
    #    ⛔ 只存 summary 的話【跨 seed 合併 McNemar 算不出來】—— 8/27 就是卡在這裡，
    #    只能拿單一 seed 的 bootstrap CI 去推上界，而那個假設我自己十分鐘前才說過不成立。
    #    ⭐ 配對比較的有效樣本數是 discordant pairs（實測個位數），⛔ 不是題數
    #      ⇒ 沒有明細就沒有 discordant，沒有 discordant 就沒有合法的統計。
    out["dev_rows"] = {k: [{kk: vv for kk, vv in r.items()
                            if kk in ("idx", "tier", "bfs_dist", "success", "steps")} for r in v]
                       for k, v in ([("random", _rr), ("bc", _rb), ("lacot_rerun", _rm2),
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
if SUBGOAL:
    # ⭐ 2026-08-30 補：官方協定的【分段】臂 —— 沒有這格，主打配置（分段供點）就永遠
    #    只有 dev 尺數字、上不了對標表。policy 與 dev 那條同一支（make_subgoal_policy），
    #    每集 on_start 重置狀態。
    _spol, _son = make_subgoal_policy(max(1, min(RS_PRE)), True)
    out["rates"]["subgoal"] = rollout(0, True, f"分段 {SUBGOAL}（官方協定）",
                                      policy_fn=_spol, on_start=_son)
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
               W_LEN=0.3, FINISH_R=0.0, SUB_M=4, FINISH_MODE="bc", SUB_POLICY="",
               GRAD_MODE="climb", SEL_N=8, GRAD_PROJ=0, DEC_ANCHOR=0, TEACHER_MIX=0.0,
               SUB_MAX_ARC=0.0, BOOT_TAG="", EMA_W=0.0, LOAD_EMA=0, BC_OWN=0,
               WARMUP=0, DATA_RESAMPLE=0, DATA_SEED=-1, LR_SCALE=1.0, BOOT_SEED=-1,
               SUB_ESEL=0, U_SOURCE="flow", VQ=0, STEPS1=1500, VQ_SOFT=0, SUB_SNAP=0, SUB_HEADGUARD=0.0, DEC_START="",
               S1_FROM=""):
    """檔名後綴。⭐ 只有【非預設值】才進去 ⇒ 預設跑出來的檔名跟歷史一致（⛔ 不破壞舊索引）。

    ⚠️ 預設值必須跟上面那些 os.environ.get 的第二個參數逐一對齊 ——
       對不齊的話「非預設」就判錯，而它不會報錯。（smoke_rollout_fixes.py 逐項驗這件事。）
    """
    x = ""
    if ENC_OBJ != "sg_infonce":
        x += f"_eo{ENC_OBJ}"
    if TEACHER_MIX > 0:                              # ⭐ P1b teacher 資料引擎
        x += f"_tch{TEACHER_MIX:g}"
    if BOOT_TAG:                                     # ⭐ P2 自舉輪次（⛔ 沒有它自舉 ckpt 會蓋非自舉）
        x += f"_bt{BOOT_TAG}"
    if EMA_W > 0:                                    # ⭐ 權重 EMA 訓練檔（ckpt 內容多 ema 段）
        x += f"_emw{EMA_W:g}"
    if LOAD_EMA:                                     # ⭐ eval 用影子權重（同顆 raw/ema 配對對照）
        x += "_ema"
    if BC_OWN:                                       # ⭐ 真獨立 GCBC（own 鏈、8/31）
        x += "_bcown"
    if WARMUP > 0:                                   # ⭐ 起步暖身（9/1 seed 病理藥）
        x += f"_wu{WARMUP}"
    if DATA_RESAMPLE:                                # ⭐ 荒漠重採樣（9/1 seed 病理藥）
        x += "_rs"
    if DATA_SEED >= 0:                               # ⭐ seed 拆分 2×2（9/1 病因診斷）
        x += f"_dseed{DATA_SEED}"
    if BOOT_SEED >= 0:                               # ⭐ boot 抽樣獨立流（9/2 方差溯源）
        x += f"_bseed{BOOT_SEED}"
    if SUB_ESEL > 0:                                 # ⭐ conf2 前 E 選計畫（9/2）
        x += f"_esel{SUB_ESEL}"
    if U_SOURCE != "flow":                           # ⭐ u 來源探針（9/2）
        x += f"_u{U_SOURCE[:3]}"
    if VQ > 0:                                       # ⭐ VQ 錨定（9/2）
        x += f"_vq{VQ}" + ("s" if VQ_SOFT else "")
    if STEPS1 != 1500:                               # ⭐ stage 1 步數（9/2 embedding 先做好；⛔ 不進檔名會蓋 V8 同 seed）
        x += f"_s1{STEPS1}"
    if S1_FROM:                                      # ⭐ 凍同一個 stage 1（9/2 夜；⛔ 不進檔名會蓋同 seed 原 run）
        x += "_s1from"
    if SUB_SNAP:                                     # ⭐ 路標吸附（9/2 晚）
        x += "_snap"
    if SUB_HEADGUARD > 0:                            # ⭐ 開頭守門（9/2 晚）
        x += f"_hg{SUB_HEADGUARD:g}"
    if DEC_START:                                    # ⭐ 開頭綁定（9/2 晚；hard/soft）
        x += f"_ds{DEC_START}"
    if LR_SCALE != 1.0:                              # ⭐ lr 縮放（9/1 病因診斷）
        x += f"_lrs{LR_SCALE:g}"
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
        if SUB_MAX_ARC > 0:                      # conf2 選點上限（DELTA_SUB 倍數）
            x += f"_ma{SUB_MAX_ARC:g}"
        if SUB_POLICY:                           # 歸因對照：短程走 bc
            x += f"_sp{SUB_POLICY}"
        if DEC_ANCHOR:                           # eval-time 平移錨定（供點解碼路徑）
            x += "_anch"
    if GRAD_REFINE:                              # ⭐ flat-grad 的分水嶺
        if GRAD_MODE == "select":
            x += f"_sel{SEL_N}"
        x += f"_gr{GRAD_R}"
        if GRAD_ETA != 0.1:
            x += f"e{GRAD_ETA:g}"
        if GRAD_LAM != 0.3:
            x += f"l{GRAD_LAM:g}"
        if GRAD_R_WARM != 10:
            x += f"w{GRAD_R_WARM}"
        if W_LEN != 0.3:                         # 病一快篩（w_len 只在爬坡開著時有作用）
            x += f"_wl{W_LEN:g}"
        if GRAD_PROJ:
            x += "_prj"
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
                    SUB_POLICY=SUB_POLICY, GRAD_MODE=GRAD_MODE, SEL_N=SEL_N, GRAD_PROJ=GRAD_PROJ,
                    DEC_ANCHOR=DEC_ANCHOR, TEACHER_MIX=TEACHER_MIX, SUB_MAX_ARC=SUB_MAX_ARC,
                    BOOT_TAG=BOOT_TAG, EMA_W=EMA_W, LOAD_EMA=LOAD_EMA, BC_OWN=BC_OWN,
                    WARMUP=WARMUP, DATA_RESAMPLE=int(DATA_RESAMPLE),
                    DATA_SEED=DATA_SEED, LR_SCALE=LR_SCALE, BOOT_SEED=BOOT_SEED,
                    SUB_ESEL=SUB_ESEL, U_SOURCE=U_SOURCE, VQ=VQ_V, STEPS1=STEPS1, VQ_SOFT=VQ_SOFT, SUB_SNAP=SUB_SNAP,
                    SUB_HEADGUARD=SUB_HEADGUARD, DEC_START=DEC_START, S1_FROM=S1_FROM)
tag = (f"{ENV_NAME.replace('pointmaze-', '').replace('-v0', '')}_{CONS}_K{K}_c{COND}"
       f"_ch{CHUNK}_st{STEPS2}_T{T_CAP}_ep{SEEDS}_gu{_extra}_s{TAG_SEED}")   # gu = goal uniform(official)
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
if DIAG_DUMP and DIAG_ROWS:
    _ddst = dst.replace("rollout_", "diag_")
    with open(_ddst, "w") as f:
        json.dump(DIAG_ROWS, f)
    print(f"⭐ diag dump: {len(DIAG_ROWS)} rows -> {_ddst}", flush=True)
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
            **({"vq": vq.state_dict(), "vq_cfg": {"V": VQ_V, "beta": VQ_BETA, "soft": VQ_SOFT}} if vq is not None else {}),
            **({"s_embed": s_embed.state_dict(), "dec_start": DEC_START} if s_embed is not None else {}),
            # ⭐ 權重 EMA 影子（LACOT_EMA_W>0 時）：eval 端 LACOT_LOAD_EMA=1 取用
            **({"ema": {n: m.state_dict() for n, m in _EMA_SHADOW.items()}} if _EMA_PAIRS else {}),
            # ⭐ 真獨立 GCBC 三模組（BC_OWN 時）
            **({"bc_own": {"enc": bc_own_enc.state_dict(), "ch": bc_own_ch.state_dict(),
                           "head": bc_own_head.state_dict()}} if BC_OWN else {}),
            "cfg": dict(K=K, COND=COND, CHUNK=CHUNK, D_MODEL=D_MODEL, STEPS2=STEPS2, T_CAP=T_CAP,
                        GOAL_SAMPLING="uniform-official", EVAL_EPISODES=SEEDS,
                        CONS=CONS, EMA_M=EMA_M, SEED=SEED, EMA_W=EMA_W,
                        ENC_OBJ=ENC_OBJ, LEARNED_REFINE=LEARNED_REFINE,
                        COND_DROP=COND_DROP, BC_INDEP=BC_INDEP)}, ck)
print(f"存 checkpoint {ck}", flush=True)
