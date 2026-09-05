# DESIGN — GRPO-on-thoughts：latent 計畫層的 RL 微調（設計卡 v0、2026-09-06 呈裁）

_設計使魔（Fable、主人 9/5 睡前指定；三並行之一 — 另兩線「概念語義」「前例調研」獨立跑，
前例包已收進本卡 §1.1/§1.5/§2.3/§3.5）。唯讀輸入：THEORY-0905-upgrades、FINDINGS-0905
⑭⑱'（＋⑤''⑦⑪⑫⑬⑮⑰ 佐讀）、NOTE-composition-law、REVIEW-0905-night、PLAN-0906、
DESIGN-route1（C8）、`experiments/scratch_lacot_rollout.py`＋`lacot/nf_head.py`（現況接線、唯讀）。_
_⛔ 本卡只設計、不施工不 sbatch；所有門檻數值標【猜測】＝待 rung 0/2 用實測替換。_
_引用紀律：標「隊友正文級」＝前例調研使魔查證過正文；標「訓練記憶」＝本使魔背景知識、
**未上網覆核**（web search 額度本日歸零）— 引用承重前補驗 ID。_

---

## 0. 一句話 ＋ ⓪ 四格自檢

**一句話**：把 stage-2 flow 當 policy — 每題 (s,g) 抽 G 條 latent 計畫 z、用「decode＋佔據圖
＋BFS」的零模擬器 reward 打分、group-normalized advantage 乘 **exact flow logπ** 做
policy gradient，混合目標保留 L_FM 當防塌錨 — 讓 prior 的機率質量搬到 **teacher 分佈外的
合法組合路** 上＝正面攻「攤銷天花板」（NOTE §二：蒸餾只學得會見過的路；stitch 本義）。

**⓪ 四格**（灑批前自檢慣例、先答在卡頭）：

| 格 | 答案 |
|---|---|
| 對準哪一缺？ | 缺③強化＋stitch 本義（與 A5 多路線同一個靶：「組合沒見過的路」做成可量）；不是三缺的最短路徑、是天花板那章的地基 |
| 推進還是轉圈？ | 推進 — RL 是「訓練期藥」家族（⑪ 判決：藥在訓練端）的新成員，且理論接口（超越 teacher 分佈）是 FM/蒸餾結構上給不了的 |
| 證明／證偽什麼？ | 證明「reward 通道也能內化」（第二攤銷通道、§3.3 指紋預註冊）；證偽面＝若 pass@G≈pass@1（rung 0）則 GRPO 無物可放大、整臂降級 |
| 為什麼是現在？ | **不是現在跑、是現在設計**：⑱' 說 base 未收斂＝RL 前提未滿足；本卡讓 slot 一開（f27nL 判決後）就能上，且 rung 0 免訓練、可先行。時鐘判定見 §5.3 |

---

## 1. 目標函數

### 1.1 記號與 logπ：兩條路，我們天生走 A 路

記 c ＝ flow 條件（`flow_cond(condvec(s,g,ι), ·)`，ι ∈ {route-ix, 0}），z ∈ R^{K×d} ＝
latent 計畫（現行配方 K=8 tokens）。stage-2 引擎＝`lacot/nf_head.py::Flow`（AR affine
block ×4、TARFlow 族）：

- **A 路（exact NLL、我們的原生路）**：`Flow.log_prob(z, c)` 回 per-sample
  **exact** log p_θ(z|c)（(B,) 形；change-of-variables，`base + logdet`，一次前傳）。
  NF-CoT（2606.06447、lab 已正文核驗）Eq.3.4 已示範同一個量直接當 RL 的 logπ。
  優點：無離散化偏誤、無額外噪聲超參、單 pass；我們的 flow 抽樣（`Flow.sample`）與
  密度是同一個模型的兩面，訓推一致。
- **B 路（SDE 轉移密度、FM/diffusion 系的替代）**：LaDi-RL（arXiv 2602.01705、隊友
  正文級）與 πRL（2510.25889、隊友正文級）的做法 — ODE 採樣器轉 SDE、logπ ≈ 每步
  Gaussian 轉移密度連乘。**它是「引擎沒有 exact likelihood 時」的補品**：近似
  （離散化＋注噪改變抽樣分佈）、多一組噪聲超參。⇒ 只在未來引擎換成 rectified-flow/FM
  版時啟用；現碼判準很硬：**exact `log_prob` 存在 ⇒ A 路**。
- 附註：FM 引擎理論上也有 exact likelihood（probability-flow ODE＋divergence 積分、
  Hutchinson 估計）但貴且帶估計噪音 — 實務上被 B 路壓制，列出僅為完備。

⭐ **novelty 素材（順手的）**：最近鄰全在 B 路上付近似稅；我們一句「exact logπ、
no SDE conversion needed」是結構優勢、paper 裡值一段。

### 1.2 GRPO 形式（GRPO＝DeepSeekMath 2402.03300【訓練記憶】）

每題 q=(s,g)（一個 RL batch 取 B_g 題）、從**當前** θ̄ 抽 G 條（`Flow.sample` 本身
@no_grad — 正確：抽樣無梯度、logπ 事後帶梯度重算）：

```
z_{q,1..G} ~ π_θ̄(·|c_q)                     （G 條並行、同 batch 一次抽完）
r_{q,i}    = reward(z_{q,i}; s,g)  ∈ [0,1]   （§2、零模擬器）
Â_{q,i}    = (r_{q,i} − mean_i r_q) / (std_i r_q + ε_σ)       …(GRPO 標準)
             變體〔Dr.GRPO、2503.20783【訓練記憶】〕：Â = r − mean（去 σ 除法）

L_GRPO(θ) = − (1/(B_g·G)) Σ_q Σ_i  Â_{q,i} · log π_θ(z_{q,i}|c_q) / (K·d)
```

- **/(K·d)**＝per-dim 歸一、跟迴圈裡 `l_nf = nll/DIM` 同一把尺（β 才有可比刻度）。
- **長度歸一偏誤結構性不存在**：z 固定 K token ⇒ GRPO 著名的 1/|o| 偏誤沒有著力點；
  Dr.GRPO 的兩項偏誤只剩 σ 除法那半 ⇒ 開關 `GRPO_STDNORM∈{1,0}` 進 ablation、
  預設 1（canonical）。G 小（8）時 σ̂ 噪 — 若 rung 2 見 advantage 抖動，先切 0 再加 G。
- **退化群**：某題 G 條 reward 全同 ⇒ Â≡0 ⇒ 零梯度（正確行為、不是 bug）；
  「退化群比例」進 log — 它同時是 reward headroom 錶（§4 rung 0）。

### 1.3 ratio/clip 要不要：**先不要**（μ=1 on-policy）

- 每群樣本**只用一次梯度更新**（μ=1）⇒ ratio ρ = π_θ/π_θ̄ ≡ 1、clip 是死碼 ⇒
  上式即 REINFORCE-with-group-baseline，最乾淨。
- 我們的抽樣便宜（小 flow、K=8，不是 LLM long rollout）⇒ 不缺 sample reuse 的錢；
  on-policy 純度 > 重用複雜度。
- **升級條款（預留、非現在）**：若之後重用（μ>1），才加 PPO ratio+clip
  （min(ρÂ, clip(ρ,1±ε_c)Â)、ε_c=0.2【猜測】；PPO=1707.06347【教科書級】）。
- πRL（2510.25889）在 VLA 域報 **PPO＞GRPO** ⇒ 演算法選擇⛔不鎖死：GRPO 先上是因
  critic-free＋patch 最小；若 rung 2 死因＝advantage 噪音，藥單順序＝加 G（我們抽樣
  便宜、G=16/32 可負擔，正是 GRPO 弱點的對症解）→ Dr.GRPO 去 σ → 才考慮 PPO+critic
  （要 value head、屬「要重大工程」檔）。

### 1.4 混合目標與 β 排程（LaDiR 防塌錨）

```
L_total = L_FM(data batch)＋既有各 loss 原權重原樣  +  β(t)·L_GRPO   [+ β_KL·KL̂(π_θ‖π_base)]
```

- **LaDiR 錨（2510.04573 原句「保留 flow matching loss」）**：RL 全程 L_FM 對 data
  batch 照常算、權重 1 不衰減 — RL 是**加法**不是替換。資料 batch 與 RL 題共用
  同迴圈（續訓模式 `_stage2_loop(step_off)` 現成、A2 warm-start 同一套基建）。
- **β(t) 排程**：前 W 步 β=0（ckpt 載入後讓 FM 先回穩；W=500【猜測】）→ 線性升到
  β_target 持平。β_target 定標用**梯度範數比**不用拍腦袋：起始時
  ‖β∇L_GRPO‖/‖∇L_FM‖ ≈ 0.1~0.3【猜測】（比值印進 log、rung 2 校準）；
  量級初猜 β_target ∈ [0.01, 0.1]【猜測】。
- **自動護欄**：l_nf(data) 相對 RL 起點漂 > +0.10 nat/dim【猜測】⇒ β 減半（記 log、
  先手動後自動）。
- **β_KL（預設 0、第二道鎖）**：KL̂ = mean_i[logπ_θ(z_i|c) − logπ_base(z_i|c)]、
  z_i ~ π_θ — **兩邊都是 exact**（凍結 base flow 一份前傳）⇒ 不需 k3 估計技巧、
  純 MC 噪音。FM 錨先扛；漂移偵測到才開。

### 1.5 防塌＝**必要件**，三選項並列（三個獨立來源實錘、隊友正文級）

vanilla GRPO 直上 latent 會塌：2512.11816（負結果）＋ Latent-GRPO 2604.27998
（「直接 GRPO 上 latent 一律塌」）＋ LaDi-RL 自己的 vanilla 臂 pass@k 塌 — 三來源一致
⇒ **防塌不是選配**，先導判準必含 pass@G 多樣性錶（§4 rung 2）。三選項：

| 選項 | 內容 | 我們的判 |
|---|---|---|
| (a) FM loss 錨 | LaDiR 原句；L_FM 常駐（§1.4） | **預設採用** — patch 零成本（本來就在迴圈裡）、與 A 線 L_div 哲學同族（護住通路的常駐 loss） |
| (b) decoder 邊際化 N×M＋repulsive guidance | LaDi-RL：每條 latent 配 M 次 decode 取邊際 reward＋去噪時互斥引導 | N×M **對我們自動退化為 M=1**（decoder 確定性、非 LLM 隨機 decode — 免費省掉一層）；repulsive 是去噪軌跡原生、⛔ 不直接移植到一步式 NF — 最近等價物＝reward 加 batch 內多樣性 bonus 或抽樣溫度 >1，列後備（要 patch、rung 2 塌了才上） |
| (c) KL＋熵護欄 | SofT-GRPO 系；KL 至凍結 ref＋熵 bonus | KL＝§1.4 β_KL（我們 exact、比他們乾淨）；熵 Ĥ = −mean logπ_θ(z~π_θ) 亦 **exact MC** — 先當**錶**（監控線）不當 loss，跌破 rung 2 門檻才升 loss |

⭐ 免費儀器註記：多數 latent-RL paper 連自己的熵都量不準；我們 exact logπ ⇒ 熵、KL、
per-sample NLL 全是準錶 — 防塌「偵測」端我們天生比前例強，寫進 paper 方法學。

---

## 2. Reward 設計（我們的免費優勢：全部零模擬器成本）

### 2.1 三項＋C8 閘（本體）

decode：`pts = _dec(z, s_n)` → [T,2] 座標（deterministic）；格化 `_e_xy_to_cell`；
佔據圖 `_EOCC`（**資料建的 E 圖**＝⑤'' 的「帶查(map)」合法部署語意、⛔ 不用
privileged `env.maze_map`）；D(x) ≡ grid_bfs 距離（從 goal cell 跑一次 BFS 得全圖
dist map、per-goal 快取 — `_INTENT_ROUTE_CACHE` 同款模式，攤到 G 條與後續步近乎免費）。

```
r_legal = (1/T) Σ_t 1[free(p_t)]                     （分數版、給密梯度）
r_reach = 1 − min( D(p_T) / D(s), 1 )                （D(p_T)=∞ ⇒ 0；除以 D(s)＝按題目難度歸一）
r_hit   = 1[ D(p_T) ≤ 1 格 ]
N(z)    = 1[ arclen(pts) ≥ ρ_len·L_BFS(s,g) ] · 1[ max_t‖p_{t+1}−p_t‖ ≤ δ_step ]   （C8 閘）
r       = N(z) · ( w1·r_legal + w2·r_reach + w3·r_hit ),   w=(.25,.50,.25)【猜測】
```

- **C8 非退化欄怎麼掛**：DESIGN-route1 N3 的教訓（合法率有平凡極大解 — decode 塌短/
  常數 ⇒ legality→1）直接搬進 reward：C8（位移/路徑長）從「探針報告欄」升為
  **乘法閘** — 退化計畫整包歸零、一分不給。第二個 1[·] 是 C8 的鄰步版（防 §2.2 的
  teleport 漏洞）。閘門檻 ρ_len=0.5【猜測】、δ_step＝資料計畫鄰步距 p95×1.5【猜測】
  — **rung 0 用 base 樣本＋資料分佈校準**，原則＝現任行為大多通過、閘只咬退化解。
  硬閘有 reward 懸崖 — group-centered advantage 對 0/1 稀疏訊號本來就穩（GRPO 在
  binary reward 域的原生用法）、可接受。
- **w4（可選、貴）**：真 rollout 成功（subgoal executor 跑環境）— **只進 rung 2/4 的
  驗收 eval、⛔ 永不進訓練 reward**（保住「訓練 reward 零模擬器成本」這句 claim 的
  乾淨；它同時是 proxy-gap 的錶、見 §5.1 風險 3）。
- 權重敏感度註記：group-centering 使每群只剩**相對序**承重 ⇒ w 的絕對值比看起來
  不重要；組成（哪幾項在場）才重要 — ablation 拆項不拆權重。

### 2.2 Reward hacking 逐項風險表

| 項 | 刷法 | 擋法（設計內）| 偵測欄（log 必印）|
|---|---|---|---|
| r_legal | 塌短/常數計畫貼在起點自由格 ⇒ legality=1 | C8 閘第一項（arclen floor）| arclen 分佈、常數計畫比例 |
| r_reach | 「末點瞬移」：中段亂穿牆、末點貼 goal ⇒ D(p_T) 小 | r_legal 罰中段＋C8 閘第二項（鄰步距上限＝不准 token 間跳牆）| max-step 分佈、逐點合法率的位置剖面 |
| r_hit | 「goal 貼上」：decoder 學會把 g 貼在末 token、與 s 無關 | 與 r_reach 同閘；R0 gate 兜底（executor 沿計畫走、中段爛 R0 必掉）| 末點對 g 的距離分佈 vs 末點跨題變異（跨題變異→0＝貼上指紋）|
| 群級 | 多樣性塌：G 條同質 ⇒ σ→0、advantage 失效、pass@G 塌（§1.5 三來源的塌型）| FM 錨常駐＋(c) 熵錶；塌了上 (b) 後備 | pass@G spread（z 空間＋decode 末點兩個尺度）、退化群比例、Ĥ |
| 跨項 | 縮 T 或改 decode 節奏鑽 (1/T) 歸一 | T 由架構定死（K token→decoder 固定 T）⇒ 無自由度；若未來 T 可變、歸一改 per-arclen | — |

### 2.3 前例同構與差異（reward 形的站位）

ThinkAct（2507.16815、隊友正文級）：plan 層 GRPO＋幾何密集 reward（終點距離＋
**DTW 對示範**）在 LIBERO 有效 — 我們「plan 層＋幾何密集 reward」的想法有同構前例、
可引。**關鍵差異一句**：他們的 reward 要示範軌跡（DTW 對 demo）⇒ 天花板仍是 teacher
分佈；我們的 reward 只要佔據圖＋BFS（**不引用任何 demo**）⇒ 能給「從未被示範過的
組合路」滿分 — 攤銷天花板的破口正是開在這裡（§3.4）。

### 2.4 成本帳

per-goal BFS 一次 O(格數)、快取後 per-sample reward＝T 次查表＋一次 dist 查值 —
CPU、微秒級。抽樣端：G×B_g 條 `Flow.sample`（4 block×K token 逐 token 逆推）＋
同批 log_prob 前傳 — 估步時 ×2~3【猜測、rung 1 實測】。B_g=16、G=8 起【猜測】。

---

## 3. 與現況的接線

### 3.1 base 用哪顆起：⑱' 是硬前提

- **⑱' 判決**：8000 步 base 的 l_nf 還在快掉（無 lr decay、主通路未收斂）、T 效應
  巨大（on/zero 同步 +.31/+.23）⇒ **在未收斂 base 上跑 RL＝把「FM 再練就會漲」誤記
  成「RL 的功勞」的混淆機器**。前提：base 收斂判準＝**l_nf 尾段 2000 步斜率 < ε_slope
  【猜測、以 f27nL 曲線定】**（步數是代理、斜率是本尊）；候選＝f27nL/idpxm 級長訓
  （11429+；f27nL s40 已灑、判決在 A 線本來就會出）。
- **base 選擇（三格、按用途）**：
  1. **主臂 base＝f27nL 家族**（p=0 長訓）：單通道故事最乾淨（RL 前無 dropout 糾纏）。
  2. **intent×RL 交互臂 base＝idpxm 級**（p=0.3 長訓、×8 已灑）：接 A 線劑量故事。
  3. **最純 cell（記錄備選）＝N5L 系**（無 intent 基線長訓）＋R-zero：reward 是唯一
     知識通道 — 「reward-channel-only 內化」的教科書 cell；N5L 現僅 s40 單顆、等 A 線。
- **⛔ 鐵則（⑱' 教訓制度化）：step-matched FM 對照臂必開** — RL 臂 vs「同 ckpt 續訓
  純 FM 同步數」臂，一切 RL 增益**只准對這個對照報**、不准對 RL 前的 base 報。
  加固技：GRPO 抽樣用**專用 `torch.Generator`**（`Flow.sample` 已收 generator 參數）
  ⇒ data batch 的 RNG 流兩臂逐位元相同 ⇒ 對照到 batch 級成對。

### 3.2 intent 開/關/dropout：三變體與預期指紋

RL 階段抽樣時的條件 ι：

| 變體 | ι（RL 抽樣時）| 預期 | 定位 |
|---|---|---|---|
| R-on | route-ix 常在 | reward 爬最快（樣本貼 teacher 路附近、合法率高）；但優化的是 posterior 模式 — zero 腿未必動、內化讀數最混 | 診斷臂（RL 機制 sanity）|
| **R-zero** | ι=0 常關 | **主臂**：直接優化部署模式（prior）；知識只剩 reward 通道 ⇒ 「攤銷進 prior」讀數最乾淨。險：初期樣本弱 ⇒ 退化群多（rung 0 先量）| 主 claim 臂 |
| R-drop(p) | 訓練同款 INTENT_DROP | 兩支都拿梯度、與訓練分佈一致；理論鉤子：⑭ 的鎖死＝「無梯度誘因離開」— **GRPO 給 intent 支一個非 NLL 的新梯度源**（advantage 加權），可能單靠 reward 就撬開 ⑬ 的 cond 端塌陷 | A 線藥方家族的交叉臂 |

預期序（zero 腿增益）：R-zero ≥ R-drop > R-on【猜測】。三臂共用 patch、只差 env 旗。

### 3.3 與內化錶的關係：「RL 內化」新讀數＋指紋預註冊

- **新讀數**：RL 內化 ≔ zero(post-RL) − zero(step-matched FM 對照)【主】；
  (on−zero) 配對差的變化【次】。分母紀律照 ⑱'：ref 用 f27nL 級（Int 舊分母已失效）。
- **⭐ 指紋預註冊（能跟 generic-T 分開的那格）**：⑱' 已量到 generic-T 混淆的指紋＝
  on/zero **同比例齊漲**（74%）。RL-經-prior 的預測指紋＝**zero 漲多於 on、gap 收窄**
  （R-zero 優化的就是 zero 那條腿）。收表時直接檢這兩型 — 齊漲型 ⇒ 疑 generic 效應
  沒對照乾淨；收窄型 ⇒ RL 內化成立。這是本卡對 ⑱' 教訓的正面利用。

### 3.4 理論接口（為什麼 RL 能買到 FM 買不到的）

- **攤銷天花板**（NOTE §二）：p_θ(z|s,g,∅) 蒸餾/FM 只在 data 支撐上有梯度 ⇒ 只學得會
  見過的 (s,g)→路線映射。PG 的梯度落在**自己樣本落的地方**、由 reward 定向 —
  不被 teacher 支撐綁住。
- **NF 全支撐**：z=f(gaussian)、微分同胚 ⇒ π_θ 對全空間 density>0 ⇒ 任何合法組合路
  **沒有零機率屏障**、只有低機率 — PG 可放大之；蒸餾到不了（那裡沒有 data 梯度）。
- **(S1) 引理接力**（THEORY §2、已證）：兩段各自見過的最短路拼起來＝新 (s,g) 的
  最短路（支撐分解、雙射）⇒ 組合路**天然拿滿 r_legal+r_reach+r_hit** — reward 不用
  懂 stitch，合成律替它背書。
- **⭐ 全卡最尖的一句**：GRPO 買的東西＝「把 pass@G 裡已存在但低機率的成功搬進
  pass@1」⇒ **獎品大小在訓練前就可量**：rung 0 在 stitch 題集上量 pass@G − pass@1
  （NF-CoT 的 pass@k-vs-k 診斷、前瞻用法）。gap 大 ⇒ RL 是對的工具；gap≈0 ⇒ 無物
  可放大、要先上探索輔助（溫度、intent 噪聲、字典 proposal — 後者接路線二）。
- **RL 題集可含 data 沒有的 (s,g)**（teleport 組合對）：那些題 FM 錨完全沒有對應
  data — reward 是唯一訊號 ⇒ 「超越 teacher」的實驗操作化；構造直接重用 A5
  多路線 eval-set 的 teleport spec（同一份、不重造）。
- **與路線二的分工**：RL＝把搜索**攤銷進訓練期**（推論仍 zero-shot 快）；字典 DP＝
  推論期**顯式**搜索。互補不替代 — 「幾何買多少、搜索補多少、RL 攤多少」三分帳。

### 3.5 novelty 定位（隊友 gap 判定、正文級）

「GRPO on flow-thoughts × goal-conditioned control × decode＋佔據圖免費密集 reward」
**查無先占**；最近鄰 LaDi-RL 差兩步（域：math/語言 vs 導航控制；reward 形：判別式/
模型 vs 免費幾何可驗）。加上 §1.1（exact logπ 無 SDE 稅）、§2.3（reward 不引示範）、
§3.3（內化指紋可量）— 四個差異點都落在我們已有的儀器上。

---

## 4. 最小驗證階梯（判讀樹先釘）

**分級**：〔可直接做〕＝現有工具一天內；〔要 patch〕＝單檔中量級、循 DIV_W 前例；
〔要重大工程〕＝新模組/多檔 — 本卡範圍內**沒有**必要的重大工程項。

- **rung 0 — reward headroom＋獎品量測**〔可直接做；新 experiments/ 探針腳本 ~100 行、
  既有 ckpt 唯讀、CPU＋單卡前傳 ~30 分〕：對 f27n/f27nL/idpxm 現貨 ckpt、dev＋stitch
  題集，每題抽 G=32：(i) reward 分佈與退化群比例（advantage 有沒有訊號）；
  (ii) pass@G vs pass@1 gap（§3.4 的獎品）；(iii) C8 閘門檻校準（base 通過率 ≥90%
  【猜測】）；(iv) reward fn 單元測（手造退化計畫必須拿 0）。
  **gate：headroom 與 gap 至少一個顯著 >0，否則整臂降級、不進 rung 1。**
- **rung 1 — patch＋golden**〔要 patch；估 ~150±50 行單檔（`scratch_lacot_rollout.py`
  GRPO 分支＋reward util）＋env 旗 `LACOT_GRPO_W/_G/_BG/_WARM/_STDNORM/_MODE(on|zero|drop)`；
  續訓基建（`_stage2_loop(step_off)`）現成免動〕：全段 gate 在 `if GRPO_W>0` 裡
  （DIV_W 慣例）⇒ **golden：GRPO_W=0 對現行為逐位元零差**（RNG 流不碰）；
  GRPO_W>0 抽樣走專用 generator（§3.1 對照加固）。CPU smoke：一步迴圈 reward/
  advantage/loss 形狀與手算對。
- **rung 2 — 單顆先導**〔~0.5–1 GPU-h【猜測】；base＝f27nL s40（判決後）、R-zero、
  2000~4000 RL 步〕**判準先釘**：
  1. reward 曲線升（尾窗斜率 >0）；
  2. pass@G spread 不塌（降幅 <20%【猜測】、z 與 decode 兩尺度）＋退化群比例不升；
  3. subgoal/R0（on、zero 雙模式）不退步（≥ −1 SE）；
  4. l_nf(data) 漂 < +0.10 nat/dim【猜測】。
  **判讀樹**：全過 ⇒ rung 3｜reward 升而 R0 平 ⇒ proxy-gap/hacking — 查 §2.2 偵測欄
  ＋上 w4 驗收 eval 定位｜reward 平 ⇒ 查退化群比例（headroom 病 → 回 rung 0 換題集/
  加 G）vs 梯度比失衡（β 校準病）｜多樣性塌 ⇒ β 減半＋上 §1.5(b/c) 後備、FM 錨敘事
  反而得證。
- **rung 3 — 對照判決**〔+0.5 GPU-h〕：step-matched FM 對照臂同步跑（§3.1 鐵則）；
  **決定格＝zero(RL) − zero(FM 對照) > 0 且過指紋檢（§3.3）**。
- **rung 4 — 八顆**〔RL 臂＋對照臂 8+8 ≈ 4~8 GPU-h【猜測】〕：⑤'' 全協定、成對雙
  eval、per-seed 全表；primary endpoint＝zero−FM對照zero 與 (on−zero) 配對差；
  P4 佇列位（⛔ 不搶 P0–P2）。

---

## 5. 風險、濾鏡、時鐘

### 5.1 最大三風險＋偵測

1. **T-混淆重演（⑱' 同型）**：RL 臂多吃的每一步更新都可能被記成 RL 功勞。
   偵測/擋法＝rung 3 對照臂是**結構件不是選配**、報數只准對它報；抽樣專用 generator
   讓兩臂 data 流成對。此風險排第一 — 我們四天內剛在同一個坑摔過。
2. **reward hacking／多樣性塌**（§1.5 三來源說「必然出現」）：偵測＝§2.2 偵測欄全印
   ＋pass@G/熵/退化群三錶進每次 log；擋法＝C8 閘＋FM 錨；塌了按 §4 rung 2 樹升 (b/c)。
3. **無 headroom 空轉**（base subgoal .857 — proxy 可能已近飽和；而 R0 .45–.65 的
   headroom 在 executor 層、不在 plan-proxy 層）：偵測＝rung 0 免費先量（這正是它
   排第一的理由）；藥＝題集換 stitch/teleport 難題（headroom 天然在那）— 順手把臂
   對準真靶。殘留險：proxy 與 R0 的 gap 本身 — w4 驗收 eval 當終審。

### 5.2 濾鏡自檢（設計使魔對自己）

- 「GRPO 熱」不是理由 — 本卡的存在理由是 §3.4 那條結構論證（FM 梯度只在支撐上、
  PG 不是）＋rung 0 可在訓練前量到獎品。若 rung 0 量出 gap≈0，⭐ 放大器沒有輸入，
  卡自動降級為「探索設計」問題 — 這句寫在這裡防止日後捨不得。
- 目的驅動結構檢查：reward 三項全部服務「合法組合路拿高分」這一個目的；沒有一項是
  「因為量得到所以放進來」。C8 閘是唯一例外（服務的是「別被刷」）— 它是護欄不是目的。
- 本卡未讀隊友「概念語義」線產出 — 若其對「thought/plan 語義」有裁定與本卡衝突，
  以呈裁時主人合議為準。

### 5.3 ICLR 時鐘（誠實評）

**判定：這不是 9/18 abstract 的臂。** abstract 證據包（T1–T5）由 A（藥方/Int）、
D（ant）、C（字典）填格，GRPO 一格都不佔；且硬前提（base 收斂判決）本身在 A 線
D2–D3 才落地。**它是 9/25 full 的至多一格 discussion／T4 延伸，主體是 camera-ready
或下一篇「破攤銷天花板」章的地基** — 與 A5 多路線、路線二字典同一個星座。

9/18 前**相容**的部分（不佔判決鏈、塞日間零碎/P4）：rung 0（免訓練、可直接做）＋
rung 1（patch＋golden、CPU smoke）— 做完＝slot 一開就能點火。rung 2 以後原則上
9/18 後；唯一例外通道：若 A3 藥方臂全爛（PLAN §3 A 風險枝）、「訓練期藥」家族要
擴編時，R-drop 臂可作為候補藥提呈 — **那是主人在 A3 判決日的裁量，本卡不推薦搶跑**。

---

## 附：引用清單（出處分級）

| ID | 內容 | 驗證級 |
|---|---|---|
| 2606.06447 NF-CoT | exact NLL 當 logπ（Eq.3.4）、pass@k-vs-k 診斷 | lab 正文核驗（RELATED-WORK-0905）|
| 2510.04573 LaDiR | RL 期保留 FM loss 錨（原句） | lab 正文核驗 |
| 2602.01705 LaDi-RL | 最近鄰：latent CoT＋Flow-GRPO（SDE logπ）＋N×M＋repulsive；math +7.7% | 隊友正文級 |
| 2510.25889 πRL | flow logπ 於控制域可行；PPO>GRPO | 隊友正文級 |
| 2507.16815 ThinkAct | plan 層 GRPO＋幾何密集 reward（LIBERO） | 隊友正文級 |
| 2512.11816／2604.27998 | latent 直上 GRPO 塌（負結果×2） | 隊友正文級 |
| 2402.03300 GRPO | group-normalized advantage、clip、KL 形 | 【訓練記憶、未上網覆核】 |
| 2503.20783 Dr.GRPO | σ 除法/長度歸一偏誤 | 【訓練記憶、未上網覆核】 |
| 1707.06347 PPO | clip surrogate | 教科書級 |
| 2505.11123 Cocos／2106.11230 | 鎖死動力學（⑭ 既有） | lab 既有 |

_設計卡完。呈裁點三個：①主臂選型（R-zero 起手、f27nL base、C8 乘法閘）②rung 0/1
是否准在 9/18 前的零碎時段先行 ③GRPO_STDNORM 與 β_target 的初值（全標【猜測】、
rung 0/2 校準）。_
