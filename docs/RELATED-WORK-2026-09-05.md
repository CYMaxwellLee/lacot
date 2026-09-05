# Related Work 歸類（9/5 主人指示；paper 骨架用）

_深度標記：【正】正文級細讀｜【掃】掃描級｜【讀】使魔讀回、ルナ未開原件｜【缺】該有未讀。_
_我們的 claim 四軸（9/4 定）：latent 計畫語言＋head 可讀／兩腿評估框架／效率（毫秒 vs 秒分）／
humanoidmaze-large-stitch 空地。每類末行＝我們與該類的差異軸。_

## A. 搜索×生成的規劃競品（推論期搜索、原始 state 空間）

- 【正】XDiffuser（kNN+學距離、能量梯度 guidance；AntMaze StitchGiant 90.0）；SIHD（結構熵樹、
  D4RL 不可比）；TTGS（value 距離+Dijkstra、無生成模型；hm-giant-stitch 78.1、輕）；
  ChronoForest（per-anchor 樹；antmaze-stitch ~99；91.9 秒/query）；C-MCTD（MCTS×3+Dijkstra；
  AntMaze giant 75±18；37~530 秒/query）。【掃】同構 8 篇（MCTD 2502.07202 等）。
- ⭐ TTGS 自承 limitation「未來要用生成模型補中間 state」＝我們 motivation 引言。
- **差異軸**：五篇全在原始座標、零 latent 語言、搜索付在推論期；我們＝latent 語言＋
  內化（推論免搜、毫秒）＋兩腿歸因。競品從上層進場、我們從下層進場（9/4 定調）。

## B. 階層式「離散上層＋連續下層」控制

- 【正】Hydra 2608.28995（intent 字典 |V|=64、Gumbel 候選+cost 排序、flow 還原連續）。
- 【讀】HiLAM 2603.05815／Libra-VLA 2604.24921／QPHIL 2411.07760（landmark 字典）。
- 【掃】VQActFlow 2606.21600／AnchorRefine 2604.17787（VQ 管模式、flow 管連續）。
- **差異軸**：他們的字典管 skill／motion primitive／landmark；我們的字典（B 階段）管
  路線拓撲、由合成律給理論位置（BFS＝特例）、且附內化度量（idp 錶）。

## C. 離散表徵技術（量化怎麼做才不毒）

- 【正】DreamerV3 2301.04104（閉環 categorical+STE+unimix — 誤差不累積的對照組）；
  iFSQ 2601.17124（utilization 診斷＋sigmoid bound）。
- 【讀】NSVQ 2606.11363（凍 encoder 切斷 collapse）；Continuous-First 2605.06870；
  CODA 2503.17760／ReVQ 2507.10547（凍住＋rectifier 吸殘差）；Q-FAT 2503.14259
  （hard 量化壞 fine control）。
- **差異軸**：我們貢獻對話的是「recon 好≠下游可學」的 2×2 實錘＋「配對可學性」驗收關
  ＋字典層級選擇（per-token z vs intent 層）的失敗解剖。

## D. Latent reasoning（LLM 圈的思考在 latent）

- 【正】ETD 2510.07358（Think block 遞歸）；Parallel-TTS 2510.07745（latent TTS+LatentRM）；
  LaDiR 2510.04573（⭐ 血緣最近：β-VAE 思想 block＋latent diffusion＋「保留 flow matching
  loss 防塌」同款）；綜述 2507.06203（activation recurrence／hidden propagation／CoT 內化三分）。
- 【掃】Refinement 2506.08552（contrastive 方向場 refine）；Soft Tokens Hard Truths
  2509.19170（訓軟推硬）；SpiralThinker 2511.08983；（IterRef＝查無此篇、已記）。
- **差異軸**：他們在語言/數學題；我們把 latent CoT 落在 goal-conditioned 控制、
  提供可量的內化度（dropout 錶）與環境可驗的合法性（穿牆探針）。主人錨句：
  「這題好解 latent reasoning 不會今年才紅」。

## E. GCRL 的距離幾何線（路線一的理論鄰居）

- 【正】2605.08732（開環誤差線性疊加界＋閉環 92% vs 開環 73%）。
- 【讀】WVM 2606.24742（robust critic 候選、要加 value 時用）。
- 【缺】🚨 quasimetric RL 本家（QRL/MRN 系）、contrastive RL（CRL 系）、SoRB（圖+RL 蒸餾）、
  HIQL 原文（我們只用它的數字當地板）— 路線一動手前這格要補正文。
- **差異軸**：他們學 value/距離當 policy 的引擎；我們把 quasimetric 當【表徵幾何約束】
  （z 插值＝合法路徑的空間形狀），且有 before 尺（.705/.212）可量進步。

## F. 攤銷／蒸餾（把搜索壓進權重）— 路線三的理論鄰居

- 【缺】🚨 最薄的一類：expert iteration／AlphaZero 式攤銷、planner distillation、
  hindsight relabeling（HER 系譜）、amortized inference for planning。目前只有我們
  自己的 zero 探針教訓＋idp 錶。sweep 要補。

## 9/5 sweep 收件（三隻使魔、2025-01~2026-08；全部【讀】級、威脅高者未驗前不進判斷）

**⭐ 戰略三發現（甲乙丙交叉）**：
1. **主 claim 車道查無先占**（兩隻獨立確認）：「訓練用 search/oracle、推論免查＋量化內化殘留」
   在 GCRL/導航域查無直接命中；「字典搜索×導航×連續下層」交集也空。⇒ 內化消融＝賣點。
2. **但 stitch 賽道已擁擠**：組合式 diffusion ≥6 篇同台（ECD/CompDiffuser/CDGS/ReRoll/RCD
   ＋已知 ChronoForest/C-MCTD）⇒ 差異化不能靠「有做 stitch」，要靠內化＋latent 語言組合。
3. **latent CoT for control 正在起浪**（2026 H1 三篇真機 VLA：LaST-R1/LaRA-VLA/LaST₀）
   — 方向變熱的證據；皆操作域、無人量內化度。

**A 類新件**：ECD 2606.21646【高：claim OGBench stitch SOTA＋近啟發式速度、ICML26；
數字僅摘要級→正文隊列①】；CompDiffuser 2503.05153【中高：NeurIPS25 Spotlight、官方
Stitch 資料集 4→30 block】；HDFlow 2605.04525【中：骨架最像（latent subgoal 序列＋低層
rectified flow）但 navigate-only 無 stitch 無 search、連續非離散、88ms/step — reviewer
必問差異→隊列③】；CDGS 2601.00126／TDP 2508.21800／ReRoll 2607.19919／RCD 2605.03075【低中】。

**E 幾何 canon 補齊**：QRL 2304.01203／MRN 2208.08133／IQE 2211.15120／CRL 2206.07568／
SoRB 1906.05253／HIQL 2307.11949。
**E 新件**：TMD 2509.20478【高：Eysenbach/Levine 系、contrastive＋quasimetric 統一、claim
stitching — 路線一 novelty 必切之鄰→隊列②】；ProQ 2506.18847【中高】；Eik-QRL 2512.12046
／MAD 2506.09276【中】；LeFlow 2608.24855【高提醒：機制近「插值＝合法」但它自己
不信插值、外掛 rollout 驗證 ⇒ 我們的合法性 loss（訊號三）正是把這個不信任變成訓練目標】；
NFTR 2607.07855【中：HIQL subgoal 塌到不可達區的前車之鑑】。

**F 攤銷新件**：SVA 2607.03751【高：唯一「拿掉搜尋量殘留」分層 ablation 前例（VLA 域）；
數字二手→需核】；DAPD 2608.01735【中高：「privilege illusion」批評 — 我們 idp 零錨 eval
正面回答它、該引】；2506.07822（diffusion→單步蒸餾）／HER×AlphaZero 2511.03405／
PILOT 2601.19917（LLM 內化 ablation 設計可借）【中】。
⭐ 2605.08732 兩面確認（誤差界＋GC-IDM 攤銷）→ 細讀優先度升。

**B/D 類新件**：OKBE 2506.09499【高：唯一明講 option 序列 BFS＋等價定理；logic/生理域、
無連續下層 — 理論最近鄰、差異要釘死→隊列④】；NF-CoT 2606.06447【高：TARFlow 引擎
同構、language/code 域→隊列⑤】；LaST-R1 2604.28192／LaRA-VLA 2602.01166／LaST₀
2601.05248【中：引、浪的證據】；DCWM 2503.00653／LAFM 2606.23420／CompACT 2603.05438
【中低：discrete+control 但無字典空間搜索】；FRM 2606.29150【低】。

**精讀隊列（序）**：①ECD ②TMD ③HDFlow ④OKBE ⑤NF-CoT（＋SVA 數字核驗）。

## 使用說明

- 引用前【讀】【掃】一律升級成【正】（正文級驗設置）— 9/4 誤報教訓入規。
- ⚠️ 9/5 fan-out 教訓：已知清單要【全量】共享給每隻、不按題目裁切 — 裁切版害 TTGS
  被乙重報一次（它 9/4 已正文讀過）。checklist④ 的正確用法。
