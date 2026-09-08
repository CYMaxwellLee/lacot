# DESIGN — B2 步法字典（動作流形的離散 token 層）草案 v0

_主人 2026-09-08 凌晨裁「B2 比較稍微像是對的路線」（TG 6006、梯2 四選項之 B2）。_
_本檔＝設計空間展開＋ルナ推薦，供主人裁；⛔ 未裁前不施工。_
_素材：SURVEY-2026-09-08-action-manifold-compression（16 篇）；lo 頭 v1 判決
（NOTE-2026-09-08-ant-night-verdicts）；FSQ 重想（DESIGN-0905-internalization §路線二）。_

## 〇、一句話

把「一小段步態」壓成一個離散的**步法字**（skill token），thinking 的草稿從「只寫
xy 路線」升級成「路線骨架＋一串步法字」；執行端由凍結的 decoder 把字展開成動作。
⇒ thinking 真的想「怎麼走」，而且想的是【可讀可數的語言】。

## 〇.5、為什麼 B2（主人方向的三個支點）

1. **字典哲學一脈**：9/5 FSQ 重想的裁定＝「字典存在的理由是組合性；組合性住拓撲/skill
   層（小字典）、不在 per-token z 座標」。B2 的步法字典正是 skill 層小字典 —— 跟 fsq27
   判決同向，不是翻案。
2. **CoT 類比最緊**：離散 token 串＝語言。內化度量、開火率、逐 token verifier 全部直接適用。
3. **前人存在**：Genie（|A|=8 latent action）、LAPA（VQ-VAE latent action）、Moto
   （autoregressive motion token）—— 壓動作段成離散字有驗證過的路。

## 一、設計空間五格（每格：選項 → 推薦＋理由＋風險）

### ① 字典大小 K
- 選項：8 / 16 / 32 / 64。
- **推薦 16 起步**。理由：Genie 用 8 扛整個平台遊戲的 latent action；ant 步態直覺＝
  方向×步速的組合、8~16 級。K 過大＝退化回連續（每字只用一次、組合性消失）。
- 風險：K 太小塌到「平均步」。驗收看 code 使用率直方圖（fsq27 教訓的「利用均勻度」關）。

### ② 一個字蓋多長（token 的時間覆蓋）
- 選項：chunk 對齊（4 步）／半週期（~8 步）／全步態週期（~15 步）。
- **推薦 4 步（chunk 對齊）起步**。理由：既有管線 chunk=4（decoder、eval、記帳全部
  對齊，接口零改）；ant 每步位移 p50 0.133 ⇒ 4 步 ≈ 0.5 單位＝夠短、字串夠密。
- 風險：4 步不成「步態」語義（半步都不到）⇒ 字典學出來的是「微方向」不是「步法」。
  備選 8 步。M0 偵察（見 §三）直接比 4 vs 8 的聚類品質，資料就能裁。

### ③ 量化器
- 選項：VQ-EMA（LAPA 式）／FSQ／Gumbel-softmax。
- **推薦 VQ-EMA**。理由：LAPA/Moto 同款、collapse 有標準藥（EMA＋dead-code restart）。
  FSQ 前科註記：fsq27 的罪在「per-token z 座標量化」；本案量化的是 skill 段、層級不同
  —— 但選型仍避 FSQ（沒必要再碰）。
- 風險：codebook collapse。M0 直接量。

### ④ 跟 xy 骨架的接法
- 選項：a) 兩條並行通道（xy 骨架稀疏＋步法字稠密；Hierarchical Diffuser 形）
  b) 交錯單序列 c) 每個骨架段下掛步法字串。
- **推薦 a**。理由：現有 u 的 xy 通道**一行不動**（新開通道、旗關閉＝行為不變 ——
  沿用 lo-v1 的隔離紀律：fork_rng、旗 off 不建模組、檔名帶段）；c 是 a 的可讀重排、
  之後再說；b 動到主序列語法、最貴。
- 風險：兩通道的對齊（第 t 個字對應骨架哪一段）要在資料端定死，寫進 spec。

### ⑤ decoder（展開器）
- 選項：獨立小 MLP（token＋s → 動作 chunk）／共用主 action head 加 token 條件。
- **推薦獨立小 MLP**（lo 頭同構位、同容量對打 bc_own 的紀律）。訓練＝VQ-VAE 重建
  （動作段有直接監督 —— 跟 lo 頭 v1 的 hindsight 配對不同，這裡是壓縮重建、訊號稠密，
  純重建應可立住；V 加權留作後手藥、不進 v0）。
- lo 頭 v1 退場條款：B2 的 decoder 取代 lo 頭的執行位。lo-v1 code 保留（旗關）、
  V 加權工單凍結（TG 6007 已告主人）。

## 二、訓練流程草案（三階）

```
stage0  離線訓步法字典：資料裡切動作段（②的長度）→ VQ-VAE →
        字典 K 個字＋encoder＋decoder。零主模型改動。
stage2+ 主模型加「步法字預測 head」：thinking 生成 xy 骨架（照舊）＋
        自回歸出步法字串（新通道）。訓練目標＝hindsight 段的真字（teacher=stage0 encoder）。
eval    骨架選路標（conf2 照舊）＋ 步法字串 → stage0 decoder 展開動作。
```

## 三、最小驗證梯（先偵察再放量；M0 過不了就回頭，不燒主模型）

- **M0｜字典本身品質**（純資料、CPU/單卡小時級、可派使魔）：
  重建誤差、code 使用率直方圖、4 vs 8 步覆蓋對比、**8~16 個字的視覺化**（每個字
  render 幾段 —— 主人一眼能看出「這是往前小跑」「這是原地轉」才算字典成立）。
- **M1｜oracle 上界**：真字串（encoder 抽的）餵 decoder 接力走 —— 天花板；
  對照＝主頭接力 .448。M1 ≥ .448 才有資格進 M2。
- **M2｜thinking 出字**：主模型學出字串、端到端 eval。判準預釘後再跑。

## 三.五、u 進場後的遠景設計格（主人 2026-09-08 午後兩則洞見、TG 6112/6115）

- **u＝連貫字串的作者**：今日全部逐格選字法失敗（貪心 .063／nn 亂接／M2 逐 chunk），唯一
  證明能走的是連貫整段字串（teacher-relay .554）——「生成連貫整段」正是 u（flow）本行。
  B2 完整版＝字串生成掛進 stage2 thinking（外掛選字頭只是最小版）。humanoid 的姿態病
  同框架兩端夾：u 以當下姿態為起點鋪譜＋decoder 姿態擾動增強。
- **chunk 開大與 u 相互成就（主人 6115）**：「段長 4 優於 8」是【無 u 架構下】的結論 ——
  u 提供跨字脈絡後單字不必自扛全部信息，段長最優值會往大移 ⇒ **u 條件化後段長重掃**。
  chunk 大 ⇒ 字串短 ⇒ u 好學、test-time 搜索便宜；u 保連貫 ⇒ chunk 有本錢開大。
  前人同形狀：Hierarchical Diffuser 高層 15 步跳。條件：humanoid 開大前先配擾動增強
  （開環窗變長、4 步都摔）；ant 先彰顯。

- **GRPO for u（主人 6117）＝三級火箭的第三級**：M2 逐格版 → 字串進 u（SFT 模仿教材）→
  GRPO for u（u 生成字串候選、reward 打分強化）。離散字串上 GRPO 是標準形；reward＝
  展開實走的 per-leg 到達率（金標）／學的 V（便宜代理、三把尺統一格）；pass-G 過低的
  解＝字串版 hindsight relabel（走到哪就把哪記成目標正樣本 —— 主人 9/7 的藥在字串空間
  更自然）。順序鐵則：SFT 站住才上 GRPO、不同時開。＝主人 9/7「exploration 讓它自己
  學會走路」的離散實現。
  **reward ⛔ 不是距離型 V（主人 6119 精修）**：用我們自己的四關當骨架 —— 到得了／
  不磨蹭／不亂動／像樣（量尺包現成）；humanoid 的翻倒＝硬乘法閘（摔＝整串 0 分）；
  形狀同 pointmaze GrpoReward 三項乘法閘哲學（多項並看、閘擋單項刷分 —— 防 P2 式
  「每步更近但碎步」偏科）。組合權重＝實作時設計格、每個標量的/拍的、呈裁再燒。
  距離型 V 降級為「到得了」一關的便宜近似候補。

## 四、開放問題（留主人裁）

1. K=16 起步同意嗎？（M0 會同時量 8/16/32 的使用率，資料可改推薦）
2. 步法字要不要進 refine/GRPO 迴圈（離散 token 上的 policy gradient 是標準形）——
   v0 先不進、立住再說？
3. M0 偵察可否先派使魔跑（零主模型改動、零治療性質）？

_相關：[[SURVEY-2026-09-08-action-manifold-compression]]／[[NOTE-2026-09-08-ant-night-verdicts]]
／DESIGN-0905-internalization §路線二（字典＝組合性）／memory: lacot-action-head-is-mlp
（08-22 per-step 離散化失敗 —— 與本案 per-segment skill 字層級不同，已區辨）。_
