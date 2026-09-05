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

## 使用說明

- 引用前【讀】【掃】一律升級成【正】（正文級驗設置）— 9/4 誤報教訓入規。
- 缺口 E、F＝今日 sweep 使魔的主攻面；A、B、D 掃 2025.01~2026.08 新件。
