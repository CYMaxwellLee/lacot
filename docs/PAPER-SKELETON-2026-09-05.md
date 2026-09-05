# PAPER SKELETON — ICLR 2027（2026-09-05）

_paper 起草使魔（Fable 級、主人授權；三隻並行之一）。唯讀來源：DESIGN-0905（claim 四軸 v2＋三缺）、RELATED-WORK-0905（六類＋精讀六篇）、FINDINGS-0905 ①~⑭、THEORY-0905 ×2（合成律＋內化形式化）、PLAN-0906（T1~T5 證據包＋時鐘）。_
_時鐘：abstract **9/18**、full **9/25 AoE**〔PLAN §0，web 查證過〕。_

**分級誠實約定（全檔通用）**：每個數字／主張標三態之一 —
`[已定讞]`＝FINDINGS 收官＋複驗；`[在跑]`＝queue 中（9/5 當下：idp01 4/8＋idpxm 排隊、w=1 sanity、⑫ 分辨實驗 A）；`[待跑]`/`[待設計]`＝PLAN 已排或待主人裁。
**⛔ 佔位符規則**：`[XX@來源]` 形＝待填、來源指明由哪個實驗產出；⛔ 不准編數字（見 §6 風格鐵則）。

---

## 1. 題目候選 ×3

**T-A（主打內化度量）**
> *Train with the Map, Plan without It: Measuring Knowledge Internalization in Flow-Based Planners*

一句賣點：把「訓練期查得到的路徑知識、推論期拿掉還剩多少」從敘事變成一個有定義域、能拆三種零的度量 (Int, ε) — 三隻 sweep 獨立確認此車道查無先占〔RELATED-WORK sweep 發現 1〕。

**T-B（主打溫度族合成律）**
> *BFS at Temperature Zero: A Composition Law for Latent-Plan Generators*

一句賣點：goal-conditioned 規劃的合成律 V(s,g)=⊕_m[V(s,m)⊗V(m,g)] 在溫度族上統一 — NLL 訓練與抽樣 eval 住 T=1（log-semiring）、BFS 是 T→0 凍結極限、差距 ≤ HT·logK 有量化界〔THEORY-comp 引理 1/2〕；字典搜索＝BFS 的一般化而非替代品。

**T-C（度量＋機制合體、偏 findings 敘事）**
> *Why Your Planner Ignores Its Oracle: Conditioning Collapse, a Measurable Fix, and a Composition Law*

一句賣點：CFG 式 dropout 在條件冗餘下必然鎖死（合法全域最優＋自我維持；散度 1.1% 實錘）〔THEORY-int Prop 2.1–2.3、FINDINGS ⑬⑭〕，配三藥各打一層的理論分工＋兩個文獻空白（dropout p 臨界值、條件冗餘 vs 使用率）。

_取捨註：T-A 對應 claim 重排後主軸（⑥ 提案：④空地→內化度量軸，**待主人裁**）；T-B 理論最重、實驗端依賴 C 線字典（缺②）落地；T-C 在 A 線藥方臂全爛時仍成立（PLAN §3 風險 A 的退路敘事）。_

---

## 2. Abstract 候選 ×2（150~200 英文字；佔位符標來源）

### 2A — 主打內化度量（配 T-A）

> Inference-time search makes planners accurate but slow: recent compositional diffusion planners spend 8–530 seconds per plan [ECD Table 6; C-MCTD — FINDINGS ⑥, RELATED-WORK A]. We study the converse regime: route knowledge queryable at training time — a shortest-path oracle over the occupancy map, or free hindsight summaries — is compressed into an intent latent that conditions a rectified-flow plan generator, and the query interface is removed at inference. We formalize *internalization* as Int, a three-point-calibrated ratio whose diagnostic pair (Int, ε) provably separates "perfectly internalized" from "locked out" and "nothing to internalize" — a degeneracy naive dependence probes cannot resolve [THEORY-int Def 1.4]. On OGBench stitch tasks, anchor conditioning lifts end-to-end success from [.321@f27n-base, FINDINGS ①] to [.454@f27n, FINDINGS ①] at [ms/plan@F3 待跑] per plan, and internalization is teacher-agnostic: oracle routes and hindsight anchors match ([.918/.928@ER, PLAN §1.2 T1]). The meter further isolates a conditioning-collapse failure of classifier-free-style dropout (branch divergence [1.1%@⑬]) and prescribes remedies with provable division of labor. Post-remedy: Int = [Int@A4 待跑], ε = [ε@A4 待跑]. We position internalization as a measurable axis orthogonal to score leaderboards.

_（~190 字。⚠️ 依賴 A3/A4 藥方臂成功 — 若走 PLAN §3 風險 A 退路，末兩句改為「maze 冗餘使 Int→0 本身是 finding」敘事。）_

### 2B — 主打溫度族合成律（配 T-B）

> Planning by breadth-first search and planning by sampling from a generative model look like different algorithms. We show they are two temperatures of one composition law: V(s,g) = ⊕_m [V(s,m) ⊗ V(m,g)] over a semiring family where ⊕_T is log-sum-exp at temperature T. Training a conditional rectified flow by exact NLL and evaluating by sampling both live at T = 1 (log-semiring); BFS is the frozen T → 0 limit, with the nested-composition gap bounded by H·T·log K for horizon H and dictionary size K [THEORY-comp Lemmas 1–2, Prop 3]. Search over a small learned intent dictionary is therefore a *generalization* of BFS, and the three assumptions it needs — coverage, compositionality, decoder consistency — each carry a measurable acceptance gauge [THEORY-comp §3]; we give the failure anatomy when they break (a quantization scheme that passes every reconstruction check yet poisons downstream learnability by [−.18@①]). With internalization measured by an intent-dropout meter, our planner reaches [subgoal@T1 待填] on OGBench stitch at [ms/plan@F3 待跑] versus 8–530 s for search-based rivals [⑥]. Dictionary-space DP composes routes unseen in training: [T3@C3 待跑].

_（~185 字。⚠️ 末句押 C 線（缺②）；C 線退守時砍末句、加重 acceptance-gauge 方法學句。）_

---

## 3. 章節架構（到 subsection 級）

### §1 Introduction — claim 四軸 v2 的敘事順序

_四軸 v2＝DESIGN「ICLR 定位」四點經 ⑥ 重排提案（④空地→內化度量軸；**待主人裁**）。敘事順序：痛點→主張→度量→歸因框架→效率→附贈品。_

- **1.1 The price of searching at inference time**：競品 8~530 s/plan〔⑥、RELATED-WORK A〕；TTGS 自承「未來要用生成模型補中間 state」＝現成 motivation 引言〔RELATED-WORK A ⭐〕。
- **1.2 Internalization: train with the map, plan without it**：一般 claim 一句話〔DESIGN ⓪〕— O 壓成 intent latent、條件進 flow head、推論 O 不在場仍保留效益；O-agnostic（BFS route＝完美 oracle 特例、hindsight＝免費無圖特例）。
- **1.3 Making it measurable**（軸③內化度量軸）：Int 三點校準＋(Int,ε) 診斷對〔THEORY-int Def 1.4〕；同權重推論期開關 vs SVA 重訓式 ablation 的方法學區別〔RELATED-WORK SVA 核驗〕。
- **1.4 An attributable framework**（軸①）：latent 計畫語言＋兩腿評估（subgoal 腿／R0 端到端腿）→ 歸因乾淨（f27n 2×2 為例證〔①〕）。
- **1.5 Efficiency without search**（軸②）：毫秒 vs 秒的 3~4 個數量級〔⑥「效率軸完好」〕；數字待 F3 實測 `[ms/plan@F3]`。
- **1.6 Contributions**：四軸各一條＋合成律理論（BFS=T→0）＋方法學附贈品（軸④：recon 好≠可學 2×2、判讀樹、儀器無效判定）。

### §2 Related Work — 六類擺法與差異句（RELATED-WORK A~F 直接搬）

- **2.1 Search×generation planners**（A 類：XDiffuser/TTGS/ChronoForest/C-MCTD/ECD/CompDiffuser）— 差異句：全在原始座標、搜索付在推論期；我們 latent 語言＋內化＋毫秒。從下層進場 vs 他們從上層進場。
- **2.2 Hierarchical discrete-top continuous-bottom control**（B 類：Hydra/QPHIL/HiLAM/HDFlow）— 差異句：他們字典管 skill/motion/landmark；我們管路線拓撲、由合成律給理論位置、附內化度量。HDFlow 五軸差異表素材已齊〔RELATED-WORK 精讀③〕。
- **2.3 Discrete representation techniques**（C 類：DreamerV3/iFSQ/NSVQ/Q-FAT）— 差異句：貢獻對話＝「recon 好≠下游可學」實錘＋配對可學性驗收關＋字典層級選擇的失敗解剖。
- **2.4 Latent reasoning**（D 類：LaDiR/ETD/NF-CoT/LaST 系）— 差異句：他們語言/數學題；我們落 goal-conditioned 控制、可量內化度、環境可驗合法性。浪的證據（2026 H1 三篇 VLA）引用定調「方向熱、無人量內化」。
- **2.5 Distance geometry in GCRL**（E 類：QRL/MRN/CRL/TMD/SoRB/HIQL）— 差異句：他們學 value/距離當 policy 引擎；我們 quasimetric 當表徵幾何約束＋before 尺可量進步。**⚠️ canon 補讀 `[B0 待做]`** — 此節動筆前 RELATED-WORK E 類【缺】要清。TMD 差異（零插值量測、無凍結+整形設定）已釘〔精讀②〕。
- **2.6 Amortization & distillation**（F 類：SVA/DAPD/expert-iteration 系）— 差異句：SVA「拿掉搜尋」全在訓練期＋重訓；我們同顆權重推論期開關＝更乾淨因果讀數。DAPD「privilege illusion」由 idp 零錨 eval 正面回答。

### §3 Method

- **3.1 Setup: two-stage latent planner**：stage1 表徵（凍結）＋stage2 conditional rectified flow head；錨 a=A(τ,O)；訓練 INTENT_DROP p、推論同權重雙部署（帶查 map／免查）〔⑤'' eval 語義：⛔ 不寫 oracle、寫「帶查(map) vs 免查」〕。
- **3.2 The composition law and its temperature family**：主方程 (CL)＋對應表（Boolean/tropical/Viterbi/log/T-族）〔THEORY-comp §1.2–1.3〕；「我們的訓練目標與 eval 語意住 T=1」判決句。
- **3.3 Internalization, formally**：四候選 (a)–(d)＋簡併問題＋Def 1.4（Int 三點校準、分母 ≥ κ·SE 定義域條款）＋(Int,ε) 診斷對〔THEORY-int §1〕。
- **3.4 The idp meter: protocol**：⑤'' 協定內建（內化只在 R0 報、subgoal 內化欄 undefined＝定義的形式推論；配對差主指標；8v8 分佈對照）。
- **3.5 Remedies with provable division of labor**：三藥打三層 — 退火 p（動力學路徑）／L_div floor（結構在場、Def 3.3 margin 錨 `[.6046@⑬]`）／資料破冗餘（資訊供給、C-i~iii）〔THEORY-int §3、⑭〕。

### §4 Theory — 正文/附錄分配

_分級鐵則照搬：Prop=列明假設下可證；Conj=未證；⛔「dictionary search generalizes BFS」只准按定理級/Conjecture 級/Open 三層拆開陳述〔THEORY-comp §2〕。_

- **4.1 正文（合成律側）**：Lemma 1（log-semiring 身份）＋Lemma 2（凍結極限、HT·logK 界）＋Prop 3（定點迭代=BFS）＋D1–D3 假設×可量代理表〔THEORY-comp §1–3〕。
- **4.2 正文（內化側）**：Def 1.4＋Prop 1.3（預算恆等式=「梯度上沒錢可賺」嚴格版）＋Prop 2.1/Cor 2.2（鎖死=合法全域最優、guidance 無效一行證）＋Prop 3.7（破冗餘同拆兩支柱）〔THEORY-int §1–3〕。
- **4.3 附錄**：Prop 6/Conj 7（字典 DP 精確/近似正確性）；Prop 8/9＋R4/R9（定點存在唯一、lfp 語意-初始化警告）；Prop 1.1/1.2（ε→W₂→Δsucc 望遠鏡）；Prop 2.3（駐點三件）＋Remark 2.4（賽跑）＋Conj 2.6（p 臨界值）；Prop 3.1/Conj 3.2/Prop 3.4（藥方形式化）；Prop 4.1/4.2（幾何線相容性）；誠實邊界表〔THEORY-int §5〕。
- **4.4 雙層誠實寫法**：population 定理＋有限容量 remark（f27n +.133＝計算捷徑價值，資訊冗餘≠計算冗餘）〔THEORY-int R2.5、全域註腳〕。

### §5 Experiments — T1~T5 行列設計（PLAN §1.2 為綱）

- **5.1 Setup**：OGBench stitch 系；每格 8 顆 seed、雙 eval 成對；⑤'' 協定；per-seed 全表附錄；⛔ 不事後剔除災難 seed〔⑦〕。
- **5.2 T1 主結果表 ★**：
  - 列：pointmaze-{medium,large}-stitch ★／antmaze-stitch ★／humanoidmaze-large-stitch（stretch、遲到標 partial）。
  - 欄（每 env）：base（無 intent）／ref（p=0 全曝光）／idp-best（勝出藥方）。
  - 指標：subgoal raw／R0 帶查／R0 免查／配對差／Int／ε_rel；teacher 小欄 route vs hindsight。
  - 已有數字（搖籃 pointmaze-medium）：base R0 `.321±.095`〔①〕、ref(f27n) R0 `.454±.105`＋subgoal `.855±.029`〔①〕、idp(p=.3) on/zero `.336/.321`〔⑦〕、ER 錨 `.918`〔DESIGN ⓪〕。
  - 缺格：idp-best 整欄 `[A3/A4 待跑]`、pointmaze-large `[複製格 待跑]`、antmaze `[D2/D3 待跑·資料卡點]`、humanoid `[D4 stretch]`、Int/ε 正式值 `[A4 待跑]`。
- **5.3 T2 效率表 ★**：行=方法（ours／ECD／CompDiffuser／C-MCTD／ChronoForest／TTGS）；欄=per-plan 延遲／推論期搜索有無／同格分數誠實並列（humanoid 格 ECD `64±4`、TMD `23.0±1.5`〔⑥〕；ChronoForest `91.9s`、C-MCTD `37~530s`、ECD `8~25s`〔⑥、RELATED-WORK A〕）。我方 `[ms/plan@F3 待跑]` — **沒有實測數字前效率軸只是口號**〔PLAN F3〕。
- **5.4 T3 字典驗收＋合成表 ★（缺②本體）**：三關 D1 utilization `[C0 N1 補量 待跑]`／D2 pairability（f27n 2×2 已是實錘：FSQ 主效應 `−.18~−.19`、intent 增益被掐死 `+.03`〔①〕）／D3 round-trip+interp（現況破：C5 病灶定位〔⑨〕）；字典 DP（R4 形 (a)、lfp 初始化）vs 連續 intent 對照 `[C2/C3 待設計→待跑]`；「組合出訓練沒見過的長路」直接證據格 `[C3 待跑]`。
- **5.5 T4 劑量–反應表（文獻空白素材）**：p∈{0, 0.1, 0.3, 0.5}×8 的 (Int, ε) 曲線 — p=0 `.454`〔①〕、p=0.3 `.336/.321、ε_rel 1.1%`〔⑦⑬〕已有；p=0.1 `[idp01 在跑 4/8]`、曝光匹配 `[idpxm 在跑]`、p=0.5 `[A6 待跑]`。附：guidance 無效（w=2 `.344±.089`〔⑪〕）＋鎖死探針（B/A `1.1%`、塌在 cond 生成端〔⑬〕）＋warm-start 判決 `[A2 裁示中]`。
- **5.6 T5 幾何表（路線一）**：before 錨已立 — 插值合法率 `.705`（vs 隨機 u `.757`、t=.5 最低 `.638`）、Spearman `rho=.212`、儀器兩 gate（roundtrip `98.0%`、真−隨機差 `22.4%`）〔⑤'〕；C-battery 讀數（C1 方向盲 `cos +0.9901`、C2 `rho=.758`、C4 座標 lerp `.535` vs latent `.638`）〔⑨〕。after 欄＋三 loss ablation＋d_time↔d_bfs 交叉評 `[B1–B4 待跑]`。
- **5.7 附錄實驗**：FSQ 失敗解剖（2×2〔①〕＋全變體帳：fsqz_cont `.842/.456` 壓縮免費、snap `−.13`、u 空間災難 `.237/.004`〔③〕）；zero 探針儀器無效判定〔DESIGN 路線三〕；小抄探針（終點 R² `.999`＝「編碼糊」出局〔⑩〕）；SVA 對比（Table 9 `56.11→39.17`、重訓 vs 同權重開關〔RELATED-WORK SVA 核驗〕）；效率量測協定。

### §6 Discussion — 誠實邊界（⛔ 這節不是裝飾，是 claim 的定義域）

- **6.1 Maze redundancy**：I(τ;a|s,g)≈0 使搖籃本身壓低 intent 價值〔⑫⭐〕— 同時是 finding（Prop 1.3 實測版＋兩文獻空白：dropout p 臨界值無人分析、條件冗餘 vs 使用率無系統實驗〔⑭〕）；多路線 (s,g) 設定＝根治方向、與 stitch 本義合流 `[A5 待設計]`。
- **6.2 Population-level theorems**：全部 Prop 是 population 級；有限容量效應（f27n `+.133`）證明 population 敘事單獨不完整〔THEORY-int 全域註腳〕。
- **6.3 Stochastic transitions open**：(A1) 確定性假設；開環合成 ≠ 閉環最優（閉環 92% vs 開環 73%〔THEORY-comp §5.1 引 2605.08732〕）；升級路徑點名（options 語意或弱陳述）。
- **6.4 Learned dictionary is a goal, not yet a property**：D1 未驗（N1 無檔）、D2 有被毒死前科（f27n）、D3 實測是破的（C5）— 三個錶＝升級門票〔THEORY-comp §5.3〕。
- **6.5 Scores stay in the cradle**：「分數是搖籃裡的，能帶走的只有內化那條線」〔DESIGN 高維標準、主人 9/5 定調〕— 不 claim 分數 SOTA、賣新軸。

### §7 Conclusion（短）＋ Reproducibility statement（per-seed 全表＋協定＋json 帳）

---

## 4. Claim ↔ 證據對照表

| # | Claim | 支撐（實驗/定理/數字＋出處） | 狀態 |
|---|---|---|---|
| C1 | 錨條件化把路徑知識帶進 policy 本體（R0 腿） | f27n R0 .321→.454（+.133、≈2.7 SE）〔①〕 | **已定讞** |
| C2 | O-agnostic：teacher 換源不掉分 | ER route .918 ≈ hindsight .928〔DESIGN ⓪、PLAN T1〕；劣化 teacher 第三點 | 已定讞（兩點）；第三點 `[A6 待跑]` |
| C3 | 內化度量 (Int,ε) 拆得開三種零、儀器有效 | Def 1.4 三點校準〔THEORY-int §1.4〕；idp8 錶有效判定＋落 (0,0) 鎖死格〔⑦〕 | 定義已定讞；正式數字 `[A4 待跑]` |
| C4 | p=0.3 dropout 在條件冗餘下鎖死（機制） | Prop 2.1/Cor 2.2〔THEORY-int §2〕＋散度 B/A=1.1%＋塌在 cond 生成端〔⑬〕＋guidance 無效 w=2 .344〔⑪〕＋小抄可讀 R².999〔⑩〕 | **已定讞**（population 命題＋三方實錘） |
| C5 | 鎖死可解：三藥各打一層 | Prop 3.1/3.4/3.7〔THEORY-int §3〕；劑量二臂 `[A1 在跑]`；warm-start `[A2 裁示中]`；藥方臂 `[A3 待跑]` | 理論已定讞；實驗 **在跑/待跑** |
| C6 | 合成律：訓練/eval 住 T=1、BFS=T→0、差距 ≤HT·logK | Lemma 1/2＋Prop 3〔THEORY-comp §1–2〕 | 草稿定理（Rei 磨嚴中）；Conj 7/stochastic **open 明標** |
| C7 | 字典搜索一般化 BFS（實驗級） | 字典 DP vs 連續臂＋沒見過的長路直接證據 | `[C1–C3 待設計→待跑]`（缺②唯一通道） |
| C8 | 字典三關驗收（D1/D2/D3）是必要儀器 | D2：f27n 2×2 recon 全漏、配對錶才看到〔①〕；D3：C5 病灶〔⑨〕；D1：`[C0 N1 待跑]` | D2 已定讞；D3 已定讞(破)；D1 **待跑** |
| C9 | 效率：免搜索毫秒級 vs 競品秒級 | 競品側 8~530s 已定讞〔⑥〕；我方 `[ms/plan@F3 待跑]` | **半定讞**（缺我方實測） |
| C10 | 高維可攜（ant Int>0；humanoid 誠實對標 64±4） | 口徑五件套〔DESIGN-0904〕；hindsight-only teacher=天然測試 | `[D1–D4 待跑·資料卡點=最大單點風險]` |
| C11 | 幾何：latent 今天無測地線結構→v2 造出來 | before 尺 .705/.212＋C-battery〔⑤'⑨〕 | before 已定讞；after `[B1–B4 待跑]`；輸了轉 negative-result ablation〔PLAN §3B〕 |
| C12 | 方法學：recon 好≠下游可學 | 2×2〔①〕＋fsq 全變體帳〔③〕＋N1 教訓〔②〕 | **已定讞** |

**「待跑」彙整（=實驗排程需求端；優先序照 PLAN §4）**：
`A1` 劑量二臂收官（在跑）→ `A2` warm-start（裁示中）→ `A3` 藥方臂 → `A4` Int 正式數字（R0-200ep 硬化）→ `A5` 多路線 (s,g)（待設計）→ `A6` p=0.5＋teacher 第三點；`C0` N1 utilization（10 分 CPU、等核）→ `C1–C3` 字典 pilot→×8；`B0` E-canon 補讀（writing 前置）＋`B1–B4` 幾何階梯；`D1–D4` ant/humanoid（長桿、D1 就開）；`F3` 效率儀器化（T2 前置）。

---

## 5. Reviewer 攻擊預想 ×5

| # | 攻擊 | 回應素材在哪 |
|---|---|---|
| R1 | 「分數沒贏 ECD 64±4，humanoid 憑什麼收？」 | 不同欄競爭：ECD/CD/CDGS/GSC 全是秒級推論期搜索/校正系、無 latent 無內化、Int/ε 對它們無定義〔⑥〕；同格誠實並列＋Pareto（score×latency×search-free）＋內化車道三隻 sweep 查無先占〔RELATED-WORK sweep 發現 1；PLAN §3 E/D4 風險條敘事整段可搬〕。⛔ 不 claim 分數 SOTA。 |
| R2 | 「訓練用 privileged oracle 不公平／部署假設不現實」 | DAPD「privilege illusion」正面回答〔RELATED-WORK F〕：idp=同顆權重推論期開關（vs SVA w/o MCTS 是重訓〔SVA 核驗〕）；帶查模式=資料建佔據圖＋(s,g) 本來就知=合法部署形態、⛔ 不寫 oracle〔⑤''〕；hindsight teacher 完全免 privileged〔C2 行〕。 |
| R3 | 「HDFlow 已有 latent 階層＋flow；Hydra 已有字典＋flow — novelty？」 | HDFlow 五軸差異表（navigate-only 無 stitch 無 search、連續無字典、無內化度量、成本只報 FurnitureBench）〔RELATED-WORK 精讀③〕；B 類差異句（字典管 skill/landmark vs 路線拓撲＋合成律理論位置＋idp 錶）〔RELATED-WORK B〕；OKBE 差異五件＋限定詞引用〔THEORY-comp R5〕。 |
| R4 | 「你們自己的錶在自己的環境讀出 Int≈0 — 度量還立得住？」 | 這正是度量的賣點：(0,0) 落格=鎖死診斷、非儀器失效〔⑦ 儀器判定有效＋Def 1.4 四理由〕；機制三方定讞（⑫⑬⑭）＋藥方=可證分工〔THEORY-int §3〕；maze 冗餘升格 finding＋兩文獻空白〔⑭〕；A5 多路線設定=Int 立漂亮的正道〔⑫⭐〕。分母 undefined 條款把「不可讀」變形式推論〔Def 1.4 理由④〕。 |
| R5 | 「BFS 特例定理是 trivial／OKBE 已證；stochastic 呢？」 | 三層拆開陳述鐵則（定理級/Conj/Open）〔THEORY-comp §2〕；R5 五件差異（semiring 命名=我方觀察⛔不可寫成 OKBE 自陳；字典學出來 vs 給定；連續下層；T=1 vs exact-DP；落地量化）；stochastic 誠實 open＋開環 92 vs 73 引用〔§5.1〕；lfp 初始化警告=理論有牙齒的證據〔R9〕。 |

_後備（骨架不佔正文）：R6「maze 冗餘 ⇒ 換環境結論會變」→ 6.1＋A5＋D 線；R7「離散化毒是你們實作爛」→ ③ 全變體帳（壓縮免費、round 付稅、u 空間慘一個量級）＋C 類共識引用。_

---

## 6. 風格鐵則（骨架→成稿全程有效）

1. **⛔ 不准編數字**。骨架與成稿裡每個數字必須能指回 docs 出處（本檔一律附節號：①~⑭=FINDINGS-0905、THEORY-comp/int=兩份 THEORY-0905、RELATED-WORK/DESIGN/PLAN 同日檔）。查無出處的數字=刪。
2. **佔位符明標**：`[值@產出實驗]` 形（如 `[ms/plan@F3]`、`[Int@A4]`）— 填入時把佔位符換成「數字＋FINDINGS 節號」，⛔ 不准先填後補出處。
3. **三態分級跟著走**：已定讞/在跑/待設計 — 進稿時「在跑/待設計」的格一律標 partial 或砍，⛔ 不寫進 claim 句。
4. **用詞裁決已釘**：「帶查(map) vs 免查」⛔ 不寫 oracle〔⑤''〕；「generalizes BFS」三層拆開〔THEORY-comp §2〕；OKBE 引用帶限定詞〔R5〕；TMD 的「33x」自文與表不符、引用自己重算〔RELATED-WORK 精讀②〕。
5. **引用前升【正】**：任何【讀】【掃】級文獻進稿前升級成正文級驗設置〔RELATED-WORK 使用說明、9/4 誤報教訓〕。
6. **subgoal 格內化欄 undefined**：⑤'' 裁決=Def 1.4 分母條款的形式推論 — 表格照此渲染，⛔ 不算負殘留。
7. **per-seed 全表進附錄、災難 seed 不剔除**〔⑦〕；效率數字必須自測（F3）才上 T2。

---

_骨架完。下游：F1（claim 四軸重排定稿=⑥ 提案待主人裁）→ F2（method+theory 節 D4 開工）→ F4（9/18 abstract）。本檔三個押注點需主人/隊友回填：④軸重排裁決、A 線藥方枝走向（決定 2A 末句與 T-C 順位）、C 線是否提前（決定 2B 末句與 C7 行）。_
