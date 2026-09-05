# NOTE — context 三源分類學：資訊 vs 計算（u 通道的 embedding space 語義）

_概念理論使魔（Fable 級、主人睡前指定；三並行之「概念與 embedding 語義」線）2026-09-06 凌晨。
主人的問題：「ICL 的 context 是外面來的資訊，但 u（thought）是自己想的 — 這在 embedding
space 的含義是什麼？」上游（唯讀）：NOTE-0905-composition-law §二、THEORY-0906-postA1、
FINDINGS-0905 ⑫⑬⑰⑰'。
⛔ 分級鐵則：**定理級＝證明在檔（多為一行）或逐字引已證件；類比級＝結構對應、無證明；
猜測級＝未證**。外部文獻分 [驗]＝本次抓過 abstract/正文、[記憶ID]＝ID 憑記憶未重驗。_

---

## 0. 一句話總結

> **三種 context 差在「帶不帶 world 的 bits」與「哪個時刻可得」：ICL 範例＝推論期、
> 帶 task 級 bits；我們的 oracle（BFS/hindsight）＝訓練期、帶 instance 級 bits（實測
> ~2.5 bits、走廊級 ~1 bit）；自生 u＝隨時可得、帶 **恰好 0 bits**（一行定理）—
> 所以自生 thought 的價值只能是**計算**（serial depth／把難的條件分佈拆成簡單的兩段），
> 而這正是 CoT 表達力定理定價的那種資源。我們的系統＝一條「context 來源遷移鏈」：
> oracle 建通道 → 權重存 → （未來）自生駛通道。**

---

## 1. 記號與設定

- 查詢 $(s,g)$；資料軌跡／計畫 $\tau$（或其 latent $z$）；world 變數 $W$＝任何「環境或
  該 episode 資料側」的變數（例：真實走的 route、專家的選擇），滿足 $W$ 不是 $(s,g)$ 的
  函數的部分才有戲。
- context／thought 通道統一記 $u$；oracle 實例化 $u=a=A(\tau\ \text{或}\ O)$（hindsight
  摘要／BFS route，同 postA1 記號）。
- 自生（self-generated）：$u=f_\theta(s,g,\varepsilon)$、$\varepsilon$ 為新鮮噪音，
  $\varepsilon\perp W\mid(s,g)$（$\theta$ 視為已條件、固定）。
- 模型：flow 規劃器 $p_\theta(\tau\mid s,g,u)$、CFM（$L^2$）訓練；部署零模式 $u=0$。

---

## 2. Q1 — 三種 context 來源的定理級拆解

### 2.1 自生 u 的零資訊定理（本檔核心、一行證）

**Thm CT-1（自生零資訊）**：設 $u=f_\theta(s,g,\varepsilon)$、$\varepsilon\perp W\mid(s,g)$。
則 $P(W\in\cdot\mid s,g,u)=P(W\in\cdot\mid s,g)$ a.s.，因此
$$I(W;u\mid s,g,\theta)=0.$$
_證_：$u$ 是 $\sigma(s,g,\varepsilon)$-可測；$\varepsilon\perp W\mid(s,g)$ ⇒
$u\perp W\mid(s,g)$。∎（等價 DPI 讀法：Markov 鏈 $W-(s,g)-(s,g,\varepsilon)-u$。）
**〔定理級〕** 注意量詞：對**任何** world 變數 $W$ 同時成立 — 不是「少一點資訊」，
是恆等於零；且 $f_\theta$ 可以任意深、$p(u\mid s,g)$ 可以是任意訓練出來的 prior —
prior 學到的東西住在 $\theta$（已條件掉），不改變結論。

**Cor CT-2（log-loss 一毛不賺）**：Bayes 極限下
$\inf_q\mathbb E[-\log q(\tau\mid s,g,u)]=H(\tau\mid s,g,u)=H(\tau\mid s,g)$ —
自生 conditioning 對資料 NLL 的可買降幅恰為 $I=0$（同一條資訊分解恆等式
〔轉引 2310.07972 Eq.4、沿 ⑫ 的標記〕，代 $I=0$）。**〔定理級〕**

**Cor CT-3（自生 u 建不了通道＝A1-lim 的真身）**：若**訓練時**配對用自生／shuffled u
（$u\perp\tau\mid s,g$），則 CFM 目標的條件期望不動：
$\mathbb E[v^*\mid z,t,s,g,u]=\mathbb E[v^*\mid z,t,s,g]$（因 $(z_t,v^*)\perp u\mid s,g$），
迴歸 floor 不降、adapter 驅動項 $\mathbb E[\mathrm{Cov}(u,\delta\mid s,g)]=0$ ⇒
**$\eta_{\rm eff}\equiv 0$ 精確成立**。⇒ postA1 檔裡被 ⑰ 對 oracle 資料**證偽降級**的
A1-lim（$I_{\rm data}=0$）諸命題（原 Prop 2.1/2.3），對「自生 context 訓練」這個 arm
**逐字為真** — A1 極限錨的物理真身＝自生 context 的訓練理論。
**〔定理級@population／LT 語彙；真網路按 postA1 慣例標量級指引〕**
⇒ 結構結論：**自生 u 只能「駛」一條已經建好的通道（推論期），不能「建」它（訓練期）**；
通道語義只能由相關（$I>0$）context 建立。

### 2.2 三源對照（各自「能帶什麼」的定理級差異）

**(a) 外部資料（ICL 範例）**：context $c$＝與 task/world 相關的樣本 ⇒
$I(W;c\mid s,g)\ge 0$ 且典型 $>0$；log-loss 可買降幅上限**恰等於** $I$（同上恆等式）。
語義＝**task 識別 bits**（ICL 作隱式 Bayes 推論，Xie+ 2111.02080 [記憶ID]）；
推論期進場、免改權重；且 $c$ 來自外部分佈、**可以 off-manifold**（可指定 prior 低質量區
的任務）。**〔上限恆等式＝定理級；Bayes 讀法＝文獻詮釋〕**

**(b) 外部 oracle（我們訓練時的 BFS/hindsight）**：$a=A(\tau)$ 為目標的決定函數 ⇒
$$I(\tau;a\mid s,g)=H(a\mid s,g)\quad(\text{決定性時取到資訊上限})$$
— instance 級、**標籤側** bits：⑰ 實測 $\approx2.5$ bits（引用紀律 ⑰'：其中 63% 格界
抖動、「路線多樣性」敘事只能講走廊級 $\approx1$ bit）。古典框＝LUPI／generalized
distillation（教師看得到答案側；1511.03643 [記憶ID]）。兩個定理級限制：
（i）**只在訓練期在場** ⇒ 其部署價值必須經攤銷（＝內化；context distillation
2209.15189 [驗]＝我們 §二 CFG 形的 LLM 圈同構件）；
（ii）**CFM 以變異計價、不以 bits 計價**（C-ii′ 匯率斷裂，postA1 §3）—
bits 是有沒有錢、$\eta_{\rm eff}$ 是挖不挖得動。**〔恆等式＝定理級；匯率斷裂＝
postA1 已立的量級論證〕**

**(c) 自生（scratchpad）**：Thm CT-1 ⇒ 零 world-bits；隨時可得（部署免費）。
其全部價值＝**計算**：改變「同一個條件分佈」被表示與抽樣的方式（§3、§4），
用的是 CoT 表達力定理定價的同一種貨幣（serial depth／中間態實體化）。**〔定理級〕**

### 2.3 分類表

| | (a) ICL 範例 | (b) oracle（訓練錨） | (c) 自生 u |
|---|---|---|---|
| 進場時刻 | 推論期 | 訓練期 | 隨時（部署免費） |
| 帶的 bits | task 級、$I>0$ 可 | instance 級、$I=H(a\mid s,g)$（實測 ~2.5/走廊 ~1） | **恆 0**（Thm CT-1） |
| 定理級上限 | NLL 降幅 $\le I$ | 同左＋只能經攤銷變現 | NLL 降幅 $=0$ |
| 能**建**通道（訓練訊號） | 能（相關） | 能（最強：標籤側） | **不能**（Cor CT-3） |
| 能**駛**通道（推論） | 能（要外部存取） | 部署通常不可得 | 能（唯一零成本） |
| off-manifold | 可 | 可（hindsight eval 分佈略異、⑬ 邊界） | 不可（§4.1） |
| 我們的對應 | —（未用） | route/hindsight 錨 | 未來 rung 3／route 2 引擎 |

---

## 3. Q2 — CoT 理論接口：自生 thought 的價值＝serial depth

### 3.1 離散 CoT 三件（正文級、本次全驗）

- **Feng+ 2305.15408（NeurIPS'23 oral）[驗]**：bounded-depth transformer 直接答
  arithmetic／**DP** 需 super-poly size；常數 size＋CoT 逐步生成即可解 — CoT＝把需要
  深度的計算攤成序列步。⭐ 對我們的命中：**DP 正是合成律 route 2 的引擎** —
  CoT 定理管的問題類就是我們的 stitch 類。
- **Merrill & Sabharwal 2310.07923（ICLR'24）[驗]**：中間 token 數是計算資源 —
  poly 步 CoT 把 constant-depth transformer 的可識別類實質抬升（往 P 方向）；
  無 CoT 卡在 TC⁰ 級。（⚠️ 別跟家裡已用的 2310.079**72**（資訊分解恆等式）混號。）
- **Li–Liu–Zhou–Ma 2402.12875（ICLR'24）[驗]**：$T$ 步 CoT ⇒ 可模擬 size-$T$ 序列
  電路（「inherently serial」類）；constant-depth 常數精度無 CoT 只到 AC⁰。

⇒ 三件同一句話：**thought 的價值＝把「有效串行深度」從架構常數變成生成步數**；
thought token 的**內容**可以不含資訊（filler tokens 也行、對受限類：Pfau–Merrill–Bowman
2404.15758 [驗]）— 與 Thm CT-1「零資訊」完全相容：CoT 圈早就知道 scratchpad
不是資訊通道、是計算通道。

### 3.2 連續 latent thought（我們的 u 是連續的 — 成立條件）

- **Zhu+ 2505.12514（NeurIPS'25）[驗]**：2 層 transformer、$D$ 步**連續** thought 解
  directed graph reachability（$D$＝圖直徑；離散 CoT 已知要 $O(n^2)$ 步）。機制＝
  **superposition：一個連續 thought 向量同時編多個 search frontier＝平行 BFS**；
  且此編碼在訓練中自然湧現、免顯式監督（訓練動力學續篇 2509.23365 [驗]）。
  ⭐ 對我們的命中是字面的：**我們的 oracle 就是 BFS** — 連續 thought 理論上最擅長的
  就是把 BFS frontier 疊加在向量裡。
- **Xu & Sato 2509.25239 [驗]**：形式比較 — latent thought 贏在平行計算效率；
  離散 CoT 贏在**經 stochastic decoding 的近似計數／抽樣**。⇒ 對 route 2 的設計含義：
  要「數路線／抽多樣路線」時離散字典有離散的本事，兩層混合（上離散下連續）有理論面
  支持（與 Hydra/QPHIL 共識形狀合流、④）。
- **Coconut 2412.06769（COLM'25）[驗]**：機制＝last hidden state 直接回饋當下一步輸入
  embedding；abstract 明講連續 thought 可編碼多個候選下一步＝BFS 式搜索。
  課程細節（由顯式 CoT 漸進替換訓起）[記憶、abstract 未載] — 若引用課程句需回原文補驗。

**成立條件（對【連續 latent thought＋flow】何時真的買到深度）— 各標級：**

- **C1（回饋＝深度資源）〔命題級、證梗概在此〕**：上述定理的深度全來自
  **thought 被回饋、迭代 $T$ 步**。一次性的 $u=f_\theta(s,g,\varepsilon)$ 餵進 cond
  只加 $f_\theta$ 自己的深度一次：組合後的 sampler 仍是一張固定深度映射
  $(s,g,\varepsilon')\mapsto\tau$（$\varepsilon'=(\varepsilon,\varepsilon_{\rm flow})$）—
  **one-shot 連續自生 u 數學上可被吸收成 flow 的額外 source 噪音維**，表達力增益
  只在（i）$f_\theta$ 每參數買到的串行深度高於把同參數加進 flow、（ii）$u$ 迭代
  （rounds×depth）、或（iii）$u$ 過離散瓶頸／搜索（DP＝適應性深度）時存在。
- **C2（問題要在 serial 類）〔定理級邊界、轉引〕**：增益只對 inherently serial／DP／
  reachability 類存在（2402.12875 的類）— 我們的 stitch／多步規劃在類內（合成律＝DP）。
- **C3（可學性要密集監督）〔實證級、跨三件一致〕**：filler tokens「難學、要密集監督
  才收斂」（2404.15758）；Coconut 靠顯式 CoT 課程；我們 ⑬ 的鎖死＝**同一件事的 flow 版**
  （弱相關訊號下 thought 通道根本不成形）。⇒ **oracle 錨＝thought 通道的密集監督／課程**。
- **C4（superposition 讀出）〔類比級→可探〕**：連續贏離散靠「維持一個 frontier 集合」
  ＋消費端讀得動線性疊加（attention 式讀出）。可探：probe u 是否同時線性編碼多條
  候選 route（⑰ Z2 儀器換 u 輸入即成）。

---

## 4. Q3 — embedding space 幾何三層（誠實分級）

### 4.1 第一層：自生 u 在自家可達流形上〔定義級真＋實證後果〕

$\mathrm{supp}\,p_\theta(u\mid s,g)=\overline{f_\theta(s,g,\mathrm{supp}\,\varepsilon)}$ —
自生 u **按定義**落在模型自己的可達集合；外部 context 無此保證（可 off-manifold —
這是 (a) 的本事：指定 prior 之外的東西；也是 (b) 的雷：hindsight 錨 eval 分佈略異、
⑬ 邊界註）。**後果句**（「off-support conditioning 觸發外插、行為未定義」）不是定理，
是實證形狀：⑨ C5 已量到 decoder 對 between-mode 輸入的病（合法率 .705＜隨機 .757）。
〔第一半＝定義級；後果＝實證支持〕

### 4.2 第二層：u＝把中間計算實體化成 conditioning 座標〔定理半＋經驗半，拆開標〕

- **恆真但空洞的半句〔定理級〕**：$p(\tau\mid s,g)=\int p(u\mid s,g)\,p(\tau\mid s,g,u)\,du$
  — chain rule，任何 u 都成立。
- **有內容的定理半〔定理級〕**：CFM 的 law of total variance —
  $$\underbrace{\mathbb E\|v^*-\bar v_{s,g}\|^2}_{\text{無 u 的迴歸 floor}}
  =\underbrace{\mathbb E\|v^*-\bar v_{s,g,u}\|^2}_{\text{有 u 的 floor}}
  +\underbrace{\mathbb E\|\bar v_{s,g,u}-\bar v_{s,g}\|^2}_{\text{route 間能量}}$$
  conditioning 把「route 間」項從迴歸 floor 裡拿掉 — 這一項就是 ⑬ 量的 cond 層分支
  散度能量、也是 C-ii′ 的 $\eta_{\rm eff}$ 住的地方（bits→變異橋的迷你版）。
  ⚠️ 但由 Cor CT-3：**這項 $>0$ 需要訓練配對相關** — 自生 u 在訓練期拿不到它。
- **經驗半〔猜測級、可測〕**：「拆出來的兩個因子各自**簡單**（低熵／單模／小 flow
  學得動）」— 這不是定理、是問題結構命題；它跟**合成律的中繼點語義同構**：
  $m$＝intent 時，log-semiring（T=1）合成律就是 chain rule 的 softmax 版，而合成律的
  實質主張（「分解後各段簡單＋可重組」）正是這半句要驗的。〔同構＝結構類比級；
  ⛔ 別在 paper 裡把「chain rule 恆真」冒充成「分解有用」的證明。〕

### 4.3 第三層：可測差別（自生 vs 外部 conditioning 的三支儀器）

- **P-ent（熵恆等式）〔定理級 identity＋估計誤差〕**：
  $\mathbb E_{a\sim\rm data}[H(\tau\mid s,g,a)]=H(\tau\mid s,g)-I_{\rm data}$ —
  oracle 條件的平均降熵**恰等於** $I$（~1–2.5 bits）；自生條件的降熵
  $=I_{\rm model}(\tau;u\mid s,g)$＝模型自建通道的流量，與 $I_{\rm data}$ 無函數關係
  （可大可小）。兩者都可 sample-based 量（多模計數／熵估）。
- **P-swap（配對敏感度簽名）〔定理級簽名〕**：自生 u 與 episode 真 route 條件不相關
  （Thm CT-1）⇒ d_swap 型探針（⑬ 儀器現成）在自生臂對 data-pairing 讀 0；
  oracle 臂 $>0$。**這是「資訊 vs 計算」的操作型分辨器。**
- **P-resample（三臂分解）〔操作型定義、定理背書〕**：同一顆權重三臂 eval —
  $u=$oracle／$u\sim\hat p(u\mid s,g)$（prior 重抽）／$u=0$：
  **資訊貢獻 ≔ oracle − resampled；計算貢獻 ≔ resampled − zero。**
  Thm CT-1 保證 resampled 臂零 episode 資訊 ⇒ 該格增益全是計算／模式選擇／正則化。
  半套儀器已在（idp on/zero；差一個小 $\hat p$ prior）。

---

## 5. Q4 — 「context 來源遷移鏈」敘事判定

### 5.1 判定：值得進 paper（敘事主幹級）

鏈：**訓練用 (b) 外部 oracle（ICL 型・資訊）→ 內化（context distillation／CFG 形、§二）
→ 部署 u=0（攤銷）→〔未來 rung 3〕自生／字典搜索 top-up（計算型）。**
三個理由：
1. 它把家裡三份既有結果排進**一條因果鏈**：⑰（oracle 有 bits 可攤）→ §二（怎麼攤＝
   CFG 形）→ ⑬＋C3（為什麼通道要 oracle 才建得起來）→ CT-1（部署後剩下的只能是計算）。
2. LLM 圈有**精確同構前例**（Coconut：顯式 CoT 課程→連續 latent thought；context
   distillation 2209.15189）＝審稿人熟悉的形狀；我們是 flow-planner 版第一個、
   且帶可測內化度（idp 錶）。
3. 兩端都有定理錨：(b) 端 $I=H(a\mid s,g)$ 恆等式、(c) 端 Thm CT-1 零資訊。
⚠️ 誠實邊界：今天實作到 rung 2（u=0）；rung 3（自生）是設計含義不是結果 —
paper 裡標 future／design implication，⛔ 別寫成已建。

### 5.2 它預測什麼（各標級、全部可證偽）

- **P1〔定理級@LT〕**：用 shuffled／自生 u **訓練** ⇒ $\eta_{\rm eff}\equiv0$ ⇒
  完全鎖死、A1-lim 命題逐字適用 — ⑤'' 提過的 shuffled-intent 控制臂就是檢驗
  （賭注：該臂配對差 $\equiv0$、d_zero→0）。順帶把「A1-lim 降級件」廢物利用成
  控制臂的精確理論。
- **P2〔類比級→可測〕**：內化後 zero-mode 增益上限由**計算**不由**資訊**定：
  攤銷天花板（§二：組合沒見過的長路攤不到）的殘差只能用顯式合成（route 2 DP）買。
  **T→0 連接**：R0/成功率是 max 型（T→0）泛函、NLL 是 T=1 泛函；oracle 的
  route-diversity bits 定價在 NLL、不定價在 R0（argmax route 可產即滿分）⇒
  預測 **NLL-gap 與 R0-gap 解耦、R0-gap 先閉**（idp 儀器現成：兩個 gap 分開畫）。
  溫度族合成律（Lemma 1/2）給了這兩種貨幣的正式座標：內化存在 T=1、消費在 T→0。
- **P3〔命題級設計指引〕**：one-shot 連續自生 u（從 $(s,g)$-prior 抽一次）增益 ≈0
  （C1 吸收論證）⇒ rung 3 要嘛**迭代**（rounds＝depth、Coconut 形）要嘛**離散字典
  搜索**（route 2、DP 形）— ⛔ 別蓋 one-shot 連續版，理論上它就是幾維噪音。
- **P4〔定理級簽名〕**：任何自生 top-up 的增益必 swap-insensitive（對 data-pairing）—
  oracle 臂反之。儀器現成（⑬）。

### 5.3 一句話（paper 用、接 §二 claim 之後）

> At training time the anchor is an *informational* context — it carries
> $I=H(a\mid s,g)$ bits about the target, exactly the ICL regime; after internalization
> the deployed model conditions on nothing, and any self-generated thought provably
> carries zero further information (a one-line data-processing argument) — its residual
> value is *computational*, the same currency that CoT-expressivity theory prices.
> Internalization is thus a transfer along the context-source chain:
> information builds the channel, weights store it, computation drives it.

---

## 6. 分級總表（本檔新增件）

| 條目 | 級別 | 依賴／檢驗 |
|---|---|---|
| Thm CT-1 零資訊 | **定理級**（一行證在檔） | $\varepsilon\perp W\mid(s,g)$ |
| Cor CT-2 NLL 零降幅 | 定理級 | 恆等式〔轉引 2310.07972 Eq.4〕 |
| Cor CT-3 建不了通道＝A1-lim 真身 | 定理級@population/LT；真網路量級指引 | postA1 adapter-drive 式 |
| (b) $I=H(a\mid s,g)$ | 定理級（決定性錨） | ⑰ 實測 2.5／走廊 ~1（⑰' 紀律） |
| 匯率斷裂（bits≠變異） | postA1 已立、量級論證 | C-ii′ |
| C1 one-shot 吸收 | 命題級（證梗概 §3.2） | 反例責任：$f_\theta$ 深度優勢 |
| C2 serial 類邊界 | 定理級（轉引三件） | 2305.15408/2310.07923/2402.12875 |
| C3 密集監督 | 實證級（三件跨域一致＋⑬） | shuffled 臂可再驗 |
| C4 superposition 讀出 | 類比級→可探 | Z2 儀器換 u |
| §4.1 on-manifold | 定義級；後果＝實證 | ⑨ C5 |
| §4.2 total-variance 拆分 | 定理級 | ＝⑬ 散度能量的座標 |
| §4.2 「因子簡單」 | **猜測級、可測** | per-u 熵／多模計數 |
| §4.3 P-ent／P-swap／P-resample | 定理級 identity／簽名／操作型 | 半套儀器已在 |
| 遷移鏈敘事 | 敘事級（兩端定理錨） | §5.2 P1–P4 |
| P2 gap 解耦 | 類比級→可測 | NLL-gap vs R0-gap 分畫 |

## 7. 引用清單

[驗]＝本次抓 abstract/正文：2305.15408（Feng+ NeurIPS'23）、2310.07923（Merrill &
Sabharwal ICLR'24）、2402.12875（Li+ ICLR'24）、2404.15758（Pfau+ filler tokens）、
2505.12514（Zhu+ NeurIPS'25 superposition）、2509.23365（湧現動力學）、2509.25239
（Xu & Sato 形式比較）、2412.06769（Coconut COLM'25；課程細節句[記憶]）、2209.15189
（Snell+ context distillation）。
[記憶ID・未重驗]：1511.03643（LUPI/generalized distillation）、2111.02080（Xie+ ICL
Bayesian）— 引前補驗。
〔轉引・沿上游標記〕：2310.07972 Eq.4。家內：postA1（C-ii′/A1-lim/adapter drive）、
合成律 NOTE §二、FINDINGS ⑨⑫⑬⑰⑰'。
