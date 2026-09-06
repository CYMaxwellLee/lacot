# PAPER SKELETON — ICLR 2027（v1 2026-09-05；**v2 2026-09-06 大收表判決後改寫**）

_paper 起草使魔（Fable 級、主人授權；三隻並行之一）。唯讀來源：DESIGN-0905（claim 四軸 v2＋三缺）、RELATED-WORK-0905（六類＋精讀六篇）、FINDINGS-0905 ①~⑭、THEORY-0905 ×2（合成律＋內化形式化）、PLAN-0906（T1~T5 證據包＋時鐘）。_
_**v2 增源**：FINDINGS-**0906** ①~⑮（大收表判決日）、CANON-u-semantics（u 正名）、THEORY-0906-verifier-taught-latent-thinking（統一敘事）、THEORY-0906-info-vs-function（info/function 縱軸）、DESIGN-explore-verify-bon（rung 0.5 卡）。_
_⛔ **引用鍵（v2 定案）：裸節號 ①~⑱ ＝ FINDINGS-0905；9/6 的節號一律寫 `⑥@0906` 形；`@0905` 後綴是強調、與裸號同義。連寫時後綴管整串 —「〔①⑤@0906〕」＝兩節都出自 0906。⛔「軸③」「缺②」「理由④」「（最重的三處：①②③）」是條目編號、不是引用。** 見 §6 鐵則 1。_
_時鐘：abstract **9/18**、full **9/25 AoE**〔PLAN §0，web 查證過〕。_

**分級誠實約定（全檔通用）**：每個數字／主張標三態之一 —
`[已定讞]`＝FINDINGS 收官＋複驗；`[在跑]`＝queue 中（**9/6 當下：WS 療程八顆 26 jobs 在烤、p=1 臂重灑、三腿 intent eval 施工中**〔⑮@0906〕）；`[待跑]`/`[待設計]`＝PLAN 已排或待主人裁。
**⛔ 佔位符規則**：`[XX@來源]` 形＝待填、來源指明由哪個實驗產出；⛔ 不准編數字（見 §6 風格鐵則）。
**⚠️ v2 新增第四態 `[單顆]`**：只有 s40 一顆 seed 的格 — ⛔ 一律隨數字標「single-seed s40」，⛔ 不進 claim 句、不進 abstract 主張句。理由＝②@0906：idpxm s40 單顆 .716 → 八顆 .431±.060，s40 強 seed 假象已第三次咬人。

**v2 一句話主敘事**：**latent 思考的內化不是平衡態性質、是慢動力學 —— 而且是可誘導的（療程）**〔⑨@0906〕。
證據三腳：(a) 8000 步平衡態處處負 Int＝「殘廢效應」〔⑤@0906〕；(b) 11429 步 zero 超額轉正 +.086〔①⑤@0906〕；(c) 續訓 4000 步 dropout 把崩潰的免查模式救到與帶查同分〔⑥@0906、`[單顆]`〕；(d) p=0 臂 zero 恆崩＝dropout 是免查能力的**必要條件**〔③@0906〕。

---

## 1. 題目候選 ×4（v2：新增 T-D 並升為主候選）

**T-D（★ v2 主敘事候選：慢動力學＋療程）**
> *Internalization Is Slow: Latent Planning Knowledge Arrives Late and Can Be Induced*

一句賣點：把「訓練期查得到的路線知識、推論期拿掉還剩多少」當成**時間的函數**去量 — 在同一個劑量下，8000 步時免查模式比從沒學過還糟（殘廢效應、Int 四點全負〔⑤@0906〕），11429 步時轉正（zero 超額 +.086〔①@0906〕），而 4000 步的 dropout 續訓能把已經崩潰的免查模式救回與帶查同分（配對差 .004、`[單顆]`〔⑥@0906〕）。⇒ 內化是**慢動力學**、且**可誘導**；dropout 是必要條件（p=0 臂 zero 恆為 0.000〔③@0906〕）、時間是充分側的燃料〔⑧⑨@0906〕。

_為什麼它比 T-A 強：T-A 賣的是一把錶，reviewer 會問「錶讀出 0 你怎麼辦」；T-D 賣的是**錶讀出負→轉正→可被療程拉正**的一條軌跡 — 錶的價值由它自己的動態證明。_

**T-A（主打內化度量、v2 降為第二）**
> *Train with the Map, Plan without It: Measuring Knowledge Internalization in Flow-Based Planners*

一句賣點：把「訓練期查得到的路徑知識、推論期拿掉還剩多少」從敘事變成一個有定義域、能拆三種零的度量 (Int, ε) — 三隻 sweep 獨立確認此車道查無先占〔RELATED-WORK sweep 發現 1〕。⚠️ v2 補：這把錶今天已經讀出**負值**（殘廢效應）、**零**（p=0 崩潰格）與**接近 1**（step-matched `[單顆]` Int(idpxm)=1.10、Int(idp01L)=0.88〔⑤@0906〕）三種讀數 — 「能拆三種零」從設計主張變成實測展示。

**T-B（主打溫度族合成律）**
> *BFS at Temperature Zero: A Composition Law for Latent-Plan Generators*

一句賣點：goal-conditioned 規劃的合成律 V(s,g)=⊕_m[V(s,m)⊗V(m,g)] 在溫度族上統一 — NLL 訓練與抽樣 eval 住 T=1（log-semiring）、BFS 是 T→0 凍結極限、差距 ≤ HT·logK 有量化界〔THEORY-comp 引理 1/2〕；字典搜索＝BFS 的一般化而非替代品。

**T-C（度量＋機制合體、偏 findings 敘事）**
> *Why Your Planner Ignores Its Oracle: Conditioning Collapse, a Measurable Fix, and a Composition Law*

一句賣點〔**9/6 postA1-patch**〕：CFG 式 dropout 造成**實效鎖死＝動力學陷阱** — $I>0$ 但 $\eta_{\rm eff}\ll p\lambda$ 時 ridge 把目的地壓扁（$\mathrm{Int}^*\approx\eta_{\rm eff}/(p\lambda)$）＋12× 慢時間尺度分離；散度 1.1% 實錘〔THEORY-postA1 §2.1–2.3（Prop 2.1′/2.2′/2.3′）、FINDINGS-0905 ⑬⑭⑰〕，配三藥各打一層的理論分工＋兩個文獻空白（dropout p 臨界值、條件冗餘 vs 使用率）。⛔ 舊句「在**條件冗餘**下**必然鎖死**（**合法全域最優**＋**自我維持**）」四個零件全已倒：「條件冗餘」前提被 ⑰@0905 證偽（$I\approx2.5$ bits）、「必然/合法全域最優」被 postA1 §2 改判為動力學陷阱（無 sharp 相變、是深度 crossover）、「自我維持」對應丙 B7 打掉的 Prop 2.3(iii) 無條件版。

〔**9/6 v2-patch — 標題的方向要翻面**〕：「Why Your Planner *Ignores* Its Oracle」這個框現在**只描述一半的實驗**。今天量到的是：問題不是「不理小抄」而是「**還沒學會不用小抄**」— 同一顆 f27n 續訓注入 dropout 4000 步，免查模式從 .000 直接到 .552〔③⑥@0906、`[單顆]`〕。⇒ 若走 T-C，標題應改成 *…: A Dynamical Trap and the Course That Cures It*，敘事重心從「鎖死是穩態」移到「鎖死是**還沒到站**」。⚠️ 同時：動力學陷阱的**靜態閉式**（toy 平衡態 $\mathrm{Int}^*=\eta/(\eta+p\lambda)$）在兩格 pre-registered 對決被乾淨證偽（④@0906 斜率比 0.990 vs 帶 [2.21, 3.31]；⑤@0906 劑量四點全反號）⇒ 進 paper 時**理論側只留 transient 讀法**，平衡態閉式降為附錄的「被自己實驗打掉的版本」（＝ §6.7 方法論賣點的本體）。

_取捨註（v2 重排）：**T-D 為主候選** — 它是唯一把今天四條判決（③⑤⑥⑧、皆@0906）全裝進一句話的框，且不押任何在烤的結果。T-A 是 T-D 的度量骨（可當副標）；T-B 理論最重、實驗端依賴 C 線字典（缺②）落地；T-C 需照上段翻面才不自打臉。_

---

## 2. Abstract 候選

⛔ **v2：本節的 2A/2B 已被 `docs/ABSTRACT-DRAFTS-0906.md`（變體 A/B/C，9/6 大收表後）取代 — 落筆一律用新檔。**
本節保留 2A/2B 只為留存 9/5 的框，⚠️ **裡面的數字是 0905 家族的、部分已被 9/6 判決改寫**（最重的三處：① 舊 base `.321` 現八顆讀 `.317±.036`、ref `.454±.040`；② 2A 末句押的「Post-remedy: Int = […]」現在有實測方向了 — 8000 步 Int 為**負**、11429 步才轉正；③ 2A 的「maze 冗餘」退路句已在 v1 patch 過一次、v2 再降級）。⛔ 不准從本節複製數字進成稿。

**新檔三變體的一句話定位**（詳見 ABSTRACT-DRAFTS-0906）：
A＝內化（療程＋慢動力學，主推）／B＝資訊帳本（零資訊定理→verifier 通道→內化）／C＝診斷方法論（pre-registered 對決把「不 work」解剖成「可誘導」）。

### 2A — 主打內化度量（配 T-A）〔⛔ 已凍結、數字過期〕

> Inference-time search makes planners accurate but slow: recent compositional diffusion planners spend 8–530 seconds per plan [ECD Table 6; C-MCTD — FINDINGS ⑥, RELATED-WORK A]. We study the converse regime: route knowledge queryable at training time — a shortest-path oracle over the occupancy map, or free hindsight summaries — is compressed into an intent latent that conditions a rectified-flow plan generator, and the query interface is removed at inference. We formalize *internalization* as Int, a three-point-calibrated ratio whose diagnostic pair (Int, ε) provably separates "perfectly internalized" from "locked out" and "nothing to internalize" — a degeneracy naive dependence probes cannot resolve [THEORY-int Def 1.4]. On OGBench stitch tasks, anchor conditioning lifts end-to-end success from [.321@f27n-base, FINDINGS ①] to [.454@f27n, FINDINGS ①] at [ms/plan@F3 待跑] per plan, and internalization is teacher-agnostic: oracle routes and hindsight anchors match ([.918/.928@ER, PLAN §1.2 T1]). The meter further isolates a conditioning-collapse failure of classifier-free-style dropout (branch divergence [1.1%@⑬]) and prescribes remedies with provable division of labor. Post-remedy: Int = [Int@A4 待跑], ε = [ε@A4 待跑]. We position internalization as a measurable axis orthogonal to score leaderboards.

_（~190 字。⚠️ 依賴 A3/A4 藥方臂成功 — 若走 PLAN §3 風險 A 退路，末兩句改為「**匯率斷裂**使 Int→0 本身是 finding」敘事〔9/6 postA1-patch：⛔ 原寫「maze 冗餘使 Int→0」— ⑰ 已證偽冗餘前提；正確語言＝「$I>0$（2.5 bits）但 z-度量壓縮使 $\eta_{\rm eff}\ll p\lambda$」＝C-ii′〕。）_

### 2B — 主打溫度族合成律（配 T-B）〔⛔ 已凍結、數字過期〕

> Planning by breadth-first search and planning by sampling from a generative model look like different algorithms. We show they are two temperatures of one composition law: V(s,g) = ⊕_m [V(s,m) ⊗ V(m,g)] over a semiring family where ⊕_T is log-sum-exp at temperature T. Training a conditional rectified flow by exact NLL and evaluating by sampling both live at T = 1 (log-semiring); BFS is the frozen T → 0 limit, with the nested-composition gap bounded by H·T·log K for horizon H and dictionary size K [THEORY-comp Lemmas 1–2, Prop 3]. Search over a small learned intent dictionary is therefore a *generalization* of BFS, and the three assumptions it needs — coverage, compositionality, decoder consistency — each carry a measurable acceptance gauge [THEORY-comp §3]; we give the failure anatomy when they break (a quantization scheme that passes every reconstruction check yet poisons downstream learnability by [−.18@①]). With internalization measured by an intent-dropout meter, our planner reaches [subgoal@T1 待填] on OGBench stitch at [ms/plan@F3 待跑] versus 8–530 s for search-based rivals [⑥]. Dictionary-space DP composes routes unseen in training: [T3@C3 待跑].

_（~185 字。⚠️ 末句押 C 線（缺②）；C 線退守時砍末句、加重 acceptance-gauge 方法學句。）_

---

## 3. 章節架構（到 subsection 級）

### §1 Introduction — v2 敘事順序（痛點→主張→**它是慢的**→度量→**療程**→機器圖→效率→附贈品）

_v2 改動理由：v1 的順序在「度量」之後直接跳「歸因框架」，讀者拿到的是一把錶；9/6 之後我們手上有的是**一條軌跡**（負→正→可誘導），所以「時間」必須在 intro 就進場、不能等到 §5 才出現。四軸 v2＝DESIGN「ICLR 定位」四點經 ⑥@0905 重排提案（④空地→內化度量軸；**待主人裁**）。_

⛔ **u 錨規則（CANON-u-semantics，全稿有效）**：intro 第一次出現 u／latent thought 時，第一句先寫地基「**u 是 flow 生成的 latent 序列、編碼一段想像中的軌跡（decode 出來就是一條具體路徑）**」，再進任何抽象層（context／scratchpad／工作檯）。實測錨：可讀性探針終點 R²=.999、shuf 腿 R0=0〔CANON〕。

- **1.1 The price of searching at inference time**：競品 8~530 s/plan〔⑥@0905、RELATED-WORK A〕；TTGS 自承「未來要用生成模型補中間 state」＝現成 motivation 引言〔RELATED-WORK A ⭐〕。
- **1.2 Internalization: train with the map, plan without it**：一般 claim 一句話〔DESIGN ⓪〕— O 壓成 intent latent、條件進 flow head、推論 O 不在場仍保留效益；O-agnostic（BFS route＝完美 oracle 特例、hindsight＝免費無圖特例）。
- **1.3 …but internalization does not happen when you look**（★ v2 新增、主敘事入口）：同一劑量、只差訓練時間 —— 8000 步時免查模式**比從沒學過還糟**（idp005/idp01/idp8 的 zero−base 各 −.036/−.048/−.007、Int 四點 −.26/−.35/−.05〔①⑤@0906〕＝殘廢效應：學了依賴 intent、又被抽走）；11429 步時同一個 p=.3 臂 zero 超額**轉正 +.086**〔①@0906〕。⇒ 我們主張的第一件事不是「內化會發生」，而是「**內化是慢動力學，測早了會讀到負值**」〔⑨@0906〕。
- **1.4 Making it measurable**（軸③內化度量軸）：Int 三點校準＋(Int,ε) 診斷對〔THEORY-int Def 1.4〕；同權重推論期開關 vs SVA 重訓式 ablation 的方法學區別〔RELATED-WORK SVA 核驗〕。v2 補：**分母的選擇是實驗結論不是慣例** — 原 primary endpoint「zero − f27nL_zero」已停用，因為 p=0 模型的 zero 腿是**崩潰態**（五腿 0/250、連只吃 cond 的誠實 BC 地板都 0）而非能力量測；分母改用 step-matched 無 intent 基線 N5L〔③@0906〕。
- **1.5 A course of treatment**（★ v2 新增）：dropout 有兩個身分 —— (i) **必要條件**：p=0 訓出來的模型免查模式恆為 .000（整個 conditioning OOD、全面崩潰）〔③@0906〕；(ii) **可施加的療程**：拿同一顆已崩的 f27n@8000 續訓 4000 步、只加 p=.3 的 intent dropout，免查模式 .000 → **.552**、與帶查 .556 的配對差只剩 **.004**，代價是帶查側 .656→.556〔⑥@0906、`[單顆]` s40〕。⇒ 內化不只是等來的，也是**可以誘導的**；這是本文最強的單格證據，也是 §5 T6 的本體。⚠️ 八顆確認在烤〔⑮@0906〕。
- **1.6 What teaches, what stores, what extracts**（★ v2 新增：機器圖）：訓練期 **verifier 是老師**（每次外部選擇至多注入 $\log N$ bits、GRPO 每 update 至多 $B_gG\log k$ bits、終身不超過 verifier 自己的存量）→ **權重是倉庫** → 推論期**自生的 u 不再帶任何 world 資訊**（恆 0 bits、一行 DPI），它的全部價值是**計算**：把已存進權重的東西攤開成顯式計畫〔THEORY-0906-verifier §0–§1、帳本 L2/L4、分類 CT-1〕。主人定調句：「verifier 在教 model、存在 weights 裡；inference 時從 weights 提取肌肉經驗成 u」。⇒ 這張圖同時解釋了為什麼「免查部署」在原理上是可能的、以及為什麼 dropout 是把知識從 conditioning 推進 weights 的槓桿。
- **1.7 Exploration × verification, measured once**（★ v2 新增：第二支柱）：帳本的可證偽面第一次上秤 —— 免訓練、CPU 40 秒的 BoN 探針（rung 0.5）顯示外部選擇的 bits **真的變成分數**：帶查模式 plan 層 BoN@16 實現增益 +.164/+.134/+.151（三顆 ckpt、`[單顆]` s40）；免查模式只有 idpxm 是可部署的（zero base pass@1 .794 ≈ 帶查 .788＝分佈內；f27n .226／f27nL .254＝OOD 崩），而它免查＋裁判挑到 **plan 層 .938**〔⑪@0906〕。⚠️ **這是 plan 層讀數、不是 R0 端到端讀數** — R0 兌現率未量〔BoN 卡 §3.2「R0 層無錨」〕。
- **1.8 Efficiency without search**（軸②）：毫秒 vs 秒的 3~4 個數量級〔⑥@0905「效率軸完好」〕；數字待 F3 實測 `[ms/plan@F3]`。
- **1.9 Contributions**：(1) 內化度量 (Int, ε)；(2) **內化是慢動力學**的實測（四劑量×兩時間點）；(3) **dropout 續訓療程**把崩潰的免查模式救回同分；(4) 資訊帳本＋機器圖（verifier 教／weights 存／u 提）＋BoN 首測；(5) 合成律理論（BFS=T→0）；(6) 方法學附贈品（recon 好≠可學 2×2、判讀樹、儀器無效判定、**pre-registered 對決把自家 toy 理論乾淨證偽** — 見 §6.7）。

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
- **3.5 Remedies with provable division of labor**〔**v2 改寫：主藥換人**〕：三藥打三層 — 退火 p（動力學路徑）／L_div floor（結構在場、Def 3.3 margin 錨 `[.6046@⑬@0905]`）／資料破冗餘（資訊供給、C-i~iii）〔THEORY-int §3、⑭@0905〕。**v2 排序**：
  1. **★ 主藥＝dropout 續訓療程（warm-start course）**：從 p=0 已收斂的 checkpoint 出發、續訓時才注入 intent dropout。實測 .000→.552（配對差 .004、代價 on −.10）〔⑥@0906、`[單顆]`〕。理論位置＝退火藥的一個具體排程（先 p=0 走到低 NLL、再讓 dropout 把 conditioning 推進 weights），⚠️ 但它的理論基礎**不能再引 toy 平衡態閉式**（④⑤@0906 已證偽），現階段誠實寫法＝「經驗療程＋機制假說」。
  2. **L_div（結構藥）**：dvw 先導確認**結構真的保住** — 散度全程（step 100→8000）warmup 後撐在 ~.34 沒塌，對照 f27n 原生 B/A=1.1% 塌盡；但 R0 on .508 ≈ 同 seed 劑量臂水平＝**單獨不買分數**〔⑦@0906、`[單顆]`〕。⇒ 定位＝配伍用（保可喚醒性），⛔ 不當主藥、⛔ 也不算失敗（預釘判讀樹原句）。
  3. **破冗餘（資訊供給）**：⑰@0905 後角色從「造 $I$」轉向抬匯率（C-iii 可讀性＋z-度量權重）〔THEORY-postA1 §2.3〕；`[A5 待設計]`。
  ⚠️ **退火 p 的理論基礎待重建**：④@0906 直接量到兩臂散度衰減斜率比 **0.990**（idp0.1 −2.112e-4/step vs idp0.3 −2.133e-4/step）vs pre-registered 帶 [2.21, 3.31] ⇒ 「p 驅動衰減」在此格證偽 — 衰減是 p 無關的 generic 過程。⇒ paper 裡退火 p 只能以**實驗排程**身分出現，⛔ 不得掛 toy 動力學當它的理由。

- **3.6 The information ledger: what teaches, what stores, what extracts**（★ v2 新增；理論節的接口）：四通道記帳（權重／conditioning／驗證器選擇／觀測）＋三個一行界；自生 latent thought 對 world 變數恆注入 0 bits ⇒ u 的合法工作是**計算**（分解／提取／前沿疊加／查詢規劃）〔THEORY-0906-verifier §1(1)–(3)、帳本 L0–L5、分類 CT-1〕。⛔ 引 u 時先過 CANON 錨句（§1 開頭規則）。

- **3.7 Test-time exploration under a verifier（BoN；rung 0.5）**（★ v2 新增）：同一 generator 前綴協定抽 N 個候選、外部驗證器（佔據圖＋BFS 場）打分挑一個；免訓練。這是「探索是油門、驗證器才是油箱」在部署期的最小實現，也是 GRPO 臂（rung 1）要**內化掉**的那條上界線〔BoN 卡 §1、§4；⑪@0906〕。⛔ 讀數紀律：**plan 層**（pts_J raw hit）與 **R0 層**是兩張表，rung0/rung0.5 全部錨在 plan 層。

### §4 Theory — 正文/附錄分配

_分級鐵則照搬：Prop=列明假設下可證；Conj=未證；⛔「dictionary search generalizes BFS」只准按定理級/Conjecture 級/Open 三層拆開陳述〔THEORY-comp §2〕。_

- **4.1 正文（合成律側）**：Lemma 1（log-semiring 身份、帶權 (CL-w) 版 — 9/6 深審 M8 後）＋Lemma 2（凍結極限、HT·logK 界）＋Prop 3（定點迭代=BFS）＋D1–D4 假設×可量代理表（D4=原子忠實、9/6 補）〔THEORY-comp §1–3〕。
- **4.2 正文（內化側）**〔**9/6 postA1-patch**〕：Def 1.4＋Prop 1.3（預算恆等式；⛔ 舊寫法「＝『梯度上**沒錢可賺**』嚴格版」＝⑫ 機制 1，postA1 §0 明文「**倒了**」— 現讀作「NLL 可兌現額**恰等於**注入的 $I$」，而 ⑰ 實測 $I\approx2.5$ bits $>0$）＋**Prop 2.1′/Cor 2.2**（鎖死＝**動力學陷阱**：合法但被 ridge 壓扁的小目標＋12× 慢時間尺度，⛔ 非「合法全域最優」；Cor 2.2 guidance 無效一行證只依賴 $\varepsilon\approx0$ 實測、與 A1 真偽無關 ✓ 保留）＋Prop 3.7（破冗餘同拆兩支柱；⑰ 後其角色從「造 $I$」轉向 C-iii 可讀性與 **z-度量權重**＝C-ii′ 後半）〔THEORY-postA1 §1–2；THEORY-int §1–3〕。
- **4.3 附錄**：Prop 6/Conj 7（字典 DP 精確/近似正確性）；Prop 8/9＋R4/R9（定點存在唯一、lfp 語意-初始化警告）；Prop 1.1/1.2（ε→W₂→Δsucc 望遠鏡）；Prop 2.3（駐點三件）＋Remark 2.4（賽跑）＋Conj 2.6（p 臨界值）；Prop 3.1/Conj 3.2/Prop 3.4（藥方形式化）；Prop 4.1/4.2（幾何線相容性）；誠實邊界表〔THEORY-int §5〕。
- **4.4 雙層誠實寫法**：population 定理＋有限容量 remark（f27n +.133＝計算捷徑價值，資訊冗餘≠計算冗餘）〔THEORY-int R2.5、全域註腳〕。

- **4.5 ★ v2 新增：被自己實驗打掉的那一層（進正文，不藏附錄）**。toy 的**靜態平衡態**框架在 9/6 的兩格 pre-registered 對決被乾淨證偽：
  - **④@0906 斜率格**：預測「散度衰減率由 p 驅動、$\mu_-$ 比值落在 2.759±20%＝[2.21, 3.31]」；實測比值 **0.990**（兩臂幾乎同斜率）⇒ 衰減是 p 無關的 generic 過程。⚠️ 判讀樹預釘的兩個岔（「斜率 0」「比值對上」）**都沒中** — 落在第三態（衰減在、p 不驅動），這件事本身要寫進來〔④@0906〕。
  - **⑤@0906 劑量格**：閉式預測三點全為正且落在窄區間（+.062 [.020,.167]／+.032 [.010,.091]／+.011）；實測 **−.26／−.35／−.05**（全部反號、量級也不對）〔⑤@0906〕。
  ⇒ 正文寫法：**平衡態閉式 $\mathrm{Int}^*=\eta/(\eta+p\lambda)$ 保留為極限錨與可證偽性的示範，但不作為預測工具**；活著的是 transient（慢動力學）讀法，由 ①⑤⑥⑧（皆@0906）支持。⛔ 不准把兩個版本混寫成「理論預測成功」。〔對應 §6.7 方法論賣點〕

### §5 Experiments — T1~**T7** 行列設計（PLAN §1.2 為綱；v2 加 T6 療程／T7 探索×驗證器）

- **5.1 Setup**：OGBench stitch 系；每格 8 顆 seed、雙 eval 成對；⑤'' 協定；per-seed 全表附錄；⛔ 不事後剔除災難 seed〔⑦〕。**v2 補三條**：(i) 表格一律多一欄 `steps` — 8000 與 11429 是兩個不同的物種，混在同一列讀者必誤讀〔⑧@0906〕；(ii) 分組鍵＝目錄+檔名 `_s`（⛔ json 內的 seed 欄恆 0、不可當分組鍵）〔⓪@0906〕；(iii) `[單顆]` 格一律在表內標 single-seed、不併入有 ± 的批級列。
- **5.2 T1 主結果表 ★**：
  - 列：pointmaze-{medium,large}-stitch ★／antmaze-stitch ★／humanoidmaze-large-stitch（stretch、遲到標 partial）。
  - 欄（每 env）：base（無 intent）／ref（p=0 全曝光）／idp-best（勝出藥方）。
  - 指標：subgoal raw／R0 帶查／R0 免查／配對差／Int／ε_rel；teacher 小欄 route vs hindsight。
  - **v2 已有數字（搖籃 pointmaze-medium、9/6 大收表八顆批；⛔ 取代 v1 的 0905 讀數）**〔①@0906〕：
    ```
    base ep200_base   .317±.036          ← Int 分母的 8000 步端
    ref  f27n (p=0)   .454±.040          （ep200_f27n .457±.039 錨一致）
    idp005 (p=.05)    on .363  zero .281   配對差 +.083±.016   zero−base −.036
    idp01  (p=.10)    on .335  zero .269   配對差 +.066±.015   zero−base −.048
    idp8   (p=.30,7顆) on .326  zero .310   配對差 +.015±.010   zero−base −.007
    idpxm  (p=.30@11429) on .431 zero .403 配對差 +.029±.013   zero−base +.086 ★唯一轉正
    ```
    subgoal `.855±.029`〔①@0905〕、ER 錨 `.918`≈hindsight `.928`〔DESIGN ⓪〕維持。
  - **★ v2 新的 idp-best 欄**：8000 步家族沒有贏家（全部 zero−base ≤ 0）；目前唯一正的是 **idpxm（p=.3 @11429）**。⇒ T1 的 idp-best 欄現在有東西可填，但**它靠的是訓練時間不是新藥方** — 表格要多一欄 `steps`，否則讀者會把時間效應讀成藥效〔⑧@0906〕。
  - **⚠️ 分母註**：`[單顆]` step-matched 家族（f27nL/N5L/idp01L）給出 Int(idpxm)=1.10、Int(idp01L)=0.88，但**只有 s40 一顆**〔⑤@0906〕⇒ ⛔ 正式讀數等八顆〔⑩@0906 待擴清單〕。
  - 缺格：pointmaze-large `[複製格 待跑]`、antmaze `[D2/D3 待跑·資料卡點]`、humanoid `[D4 stretch]`、Int/ε 八顆正式值 `[f27nL/N5L 擴八顆 待跑@⑩@0906]`。
- **5.3 T2 效率表 ★**：行=方法（ours／ECD／CompDiffuser／C-MCTD／ChronoForest／TTGS）；欄=per-plan 延遲／推論期搜索有無／同格分數誠實並列（humanoid 格 ECD `64±4`、TMD `23.0±1.5`〔⑥〕；ChronoForest `91.9s`、C-MCTD `37~530s`、ECD `8~25s`〔⑥、RELATED-WORK A〕）。我方 `[ms/plan@F3 待跑]` — **沒有實測數字前效率軸只是口號**〔PLAN F3〕。
- **5.4 T3 字典驗收＋合成表 ★（缺②本體）**：三關 D1 utilization `[C0 N1 補量 待跑]`／D2 pairability（f27n 2×2 已是實錘：FSQ 主效應 `−.18~−.19`、intent 增益被掐死 `+.03`〔①〕）／D3 round-trip+interp（現況破：C5 病灶定位〔⑨〕）；字典 DP（R4 形 (a)、lfp 初始化）vs 連續 intent 對照 `[C2/C3 待設計→待跑]`；「組合出訓練沒見過的長路」直接證據格 `[C3 待跑]`。
- **5.5 T4 劑量–反應表（文獻空白素材）**〔**v2 全欄收齊、且結論翻面**〕：p∈{0, .05, .1, .3}×8 顆已收（⚠️ p=.30@8000 那格是 **7 顆**、⛔ 表註要寫）〔①@0906〕，`[p=0.5 待跑@A6]`、`[p=1 臂在跑 — 卡 INTENT_DROP<1.0 assert、解鎖一行已併工單、修畢重灑@⑮@0906]`。
  - **主讀數＝殘廢效應**：8000 步四點的 Int（Def 1.4、乙檔 v2、分母 .137）＝ `−.26 / −.35 / −.05`（p=.05/.1/.3），全部**低於**無 intent 基線 — 「學了依賴 intent 又被抽走、比從沒學過更糟」〔⑤@0906〕。⇒ 這一格是本文**最反直覺的實測**，值得單獨一張圖（x=p、y=zero−base，四點全在 0 以下）。
  - **時間軸疊上去**：同 p=.3、多練到 11429 步 ⇒ zero 超額 `−.007 → +.086`〔①⑧@0906〕。⇒ T4 從「劑量曲線」升級成 **劑量×時間的 2×k 表**，這是 v2 對 v1 最大的表格結構改動。
  - **步數效應的誠實版**〔⑧@0906〕：批級可說的只有 p=.3 臂 `+.105`（idp8 .326→idpxm .431、~1.5σ）；`[單顆]` s40 同 seed 上 f27n 多練 `−.06`、無 intent `+.09`、p=.1 `+.09` — 全在 2σ 內、方向混雜。⇒ ⛔ **不准寫「generic 步數紅利」**；可寫的是「步數的作用集中在 dropout 臂＝慢收斂補課」。
  - 附：guidance 無效（w=2 `.344±.089`〔⑪@0905〕）＋鎖死探針（B/A `1.1%`、塌在 cond 生成端〔⑬@0905〕）＋**斜率判準的證偽**（見 §4.5、④@0906）。

- **5.6 ★ v2 新增 — T6 療程表（warm-start course；本文最強單格）**：
  - 設定：同一顆 f27n@8000（p=0 訓成）續訓 4000 步，續訓期注入 intent dropout；三臂＝純續訓／+p=.1／+p=.3。
  - 讀數〔⑥@0906、全部 `[單顆]` s40〕：
    ```
    純續訓（f27nL, p=0 @11429）   on .596   zero .000   ← 全崩、五腿 0/250
    ＋idp0.1                      on .584   zero .436
    ＋idp0.3                      on .556   zero .552   配對差 .004  ← 免查≈帶查
    代價：帶查側 .656 → .556（−.10、同 seed）
    ```
  - 讀法：**dropout 是免查能力的必要條件**（純續訓臂 zero 恆 0）＋**療程可以事後施加**（不必從頭重訓）。⇒ 對應 §3.5 的主藥、對應 §1.5。
  - ⚠️ **八顆確認在烤**〔⑮@0906：WS8、26 jobs〕— 成稿前這格必須換成八顆讀數，⛔ 現在不得寫成 claim 句（§6 鐵則 3 與 v2 新增鐵則 9）。
  - `[待跑@⑩@0906]`：多步數點（4000→8000）＝療程的劑量–時間曲線。

- **5.7 ★ v2 新增 — T7 探索×驗證器表（BoN rung 0.5；⛔ plan 層）**〔⑪@0906〕：
  - 設定：免訓練、CPU 40 秒；三顆 ckpt（f27n/idpxm/f27nL、`[單顆]` s40）× on/zero × N∈{1,4,8,16,32}；同 generator 前綴協定 ⇒ @1 與 @N 逐題成對。
  - 帶查模式 plan 層 realized 增益 @N=16：`+.164 / +.134 / +.151`。
  - 免查模式 headroom V0（rung0 同尺）：f27n `+.136`（邊際）、idpxm `+.189`（全開）、f27nL `+.151`（全開）。
  - ⭐ **水位混淆項（報告自標、必須寫進表註）**：zero base pass@1 劈成兩類 — idpxm `.794`（≈ on `.788`＝分佈內）vs f27n `.226`／f27nL `.254`（OOD 崩）⇒ **可部署的 zero 只有 idpxm**；它免查＋裁判挑到 **plan 層 .938**。
  - 儀器：U 系 gate 全綠、ρ_len/δ_step 與 rung0 逐位重現、golden 兩項過；支配引理 N≥2 成立。預測對決：f27n `−.003`、f27nL `+.014` 命中，idpxm `−.058`（1.8σ 偏低）。
  - ⛔ **層級紀律**：以上全是 **plan 層**；R0 端到端兌現率**沒有錨、也還沒量**〔BoN 卡 §3.2〕。完整 R0 版 ≈5 GPU-hr、需主檔加 `LACOT_BON_N` 旗 `[待派]`。⛔ 不得把 plan 層增益寫成端到端增益。
  - 關係：BoN＝GRPO 臂（rung 1）要內化掉的上界線；GRPO 術式已收件、golden zero-diff 過、`[正式開跑照時鐘 9/18 後]`〔⑫@0906〕。
- **5.8 T5 幾何表（路線一）**：before 錨已立 — 插值合法率 `.705`（vs 隨機 u `.757`、t=.5 最低 `.638`）、Spearman `rho=.212`、儀器兩 gate（roundtrip `98.0%`、真−隨機差 `22.4%`）〔⑤'〕；C-battery 讀數（C1 方向盲 `cos +0.9901`、C2 `rho=.758`、C4 座標 lerp `.535` vs latent `.638`）〔⑨〕。after 欄＋三 loss ablation＋d_time↔d_bfs 交叉評 `[B1–B4 待跑]`。
- **5.9 附錄實驗**：FSQ 失敗解剖（2×2〔①〕＋全變體帳：fsqz_cont `.842/.456` 壓縮免費、snap `−.13`、u 空間災難 `.237/.004`〔③〕）；zero 探針儀器無效判定〔DESIGN 路線三〕；小抄探針（終點 R² `.999`＝「編碼糊」出局〔⑩〕）；SVA 對比（Table 9 `56.11→39.17`、重訓 vs 同權重開關〔RELATED-WORK SVA 核驗〕）；效率量測協定。

### §6 Discussion — 誠實邊界（⛔ 這節不是裝飾，是 claim 的定義域）

- **6.1 匯率斷裂（原「Maze redundancy」）**〔**9/6 postA1-patch — 全 repo 對 A1 舊敘事最重的一句現況陳述，⛔ 進 paper 即被 ⑰ 一戳即破**〕：⛔ 舊句「I(τ;a|s,g)≈0 使搖籃本身壓低 intent 價值〔⑫⭐〕」與**直接量測正面矛盾** — ⑰ 量到 **I(τ;a|s,g)≈2.5 bits（路線級 ~0.93 bits、⑰' 紀律）$\ne0$**。**改寫**：錢是真的、倒的是**匯率** — route 資訊在 e_target 的 **z-度量**下重新計價後變異佔比只剩 **0.4%** ⇒ $\eta_{\rm eff}\approx0.003\lambda\ll p\lambda$，**bits 說有沒有錢、變異說挖錢成本**（C-ii′、兩個數量級落差）〔THEORY-postA1 §3〕— **匯率斷裂本身成為 finding**（＋兩文獻空白：dropout p 臨界值無人分析、條件冗餘 vs 使用率無系統實驗〔⑭〕）；多路線 (s,g) 設定＝根治方向、與 stitch 本義合流 `[A5 待設計]`。
- **6.2 Population-level theorems**：全部 Prop 是 population 級；有限容量效應（f27n `+.133`）證明 population 敘事單獨不完整〔THEORY-int 全域註腳〕。
- **6.3 Stochastic transitions open**：(A1) 確定性假設；開環合成 ≠ 閉環最優（閉環 92% vs 開環 73%〔THEORY-comp §5.1 引 2605.08732〕）；升級路徑點名（options 語意或弱陳述）。
- **6.4 Learned dictionary is a goal, not yet a property**：D1 未驗（N1 無檔）、D2 有被毒死前科（f27n）、D3 實測是破的（C5）— 三個錶＝升級門票〔THEORY-comp §5.3〕。
- **6.5 Scores stay in the cradle**：「分數是搖籃裡的，能帶走的只有內化那條線」〔DESIGN 高維標準、主人 9/5 定調〕— 不 claim 分數 SOTA、賣新軸。

- **6.6 ★ v2 新增 — 單圖下這把錶分不開「攤銷 BFS」與「背路線」**〔THEORY-0906-info-vs-function Remark F1.5、F2 表〕：本文的核心 claim（內化＝攤銷 teacher 的**計算**）是一個 **function** 主張；但在**單一環境**（$m{=}1$）下，「這張圖的 info」以對常數的知識形式併入先驗，**任何互資訊帳都看不見它** — info/function 之刀在數學上只在**環境重抽**存在時才存在。⇒ 誠實後果三條：
  1. 現有 Int／idp 讀數**天生**不分「學會了搜索」與「背下了這張圖的路線」— 兩者住同一顆 θ、行為可完全重合。
  2. 我們**現在站得住的台階**是 $V_1$ 級（同圖 fresh $(s,g)$ 組合＝stitch 本義）：查詢級重抽下「查表」死、「組合算子」活。
  3. $V_2$ 級（同 family fresh 環境）**一格證據都還沒有** ⇒ ⛔ 正文不得寫「學到了通用的規劃算子」，只能寫「在本圖上的組合算子」。切得開的實驗＝**換圖 transfer**（單圖訓練 vs 多圖訓練、fresh 圖 eval）`[待設計@F1.3 階梯]`。
  ⇒ 這一節不是自我批評、是**把 claim 的定義域寫清楚**；reviewer 若自己抓到（R8）我們就掉一個等級，自己先寫則變成框架貢獻。

- **6.7 ★ v2 新增 — 我們自己的 toy 理論在兩格 pre-registered 對決被證偽（方法論賣點）**：9/5 我們把 toy 的平衡態預測**寫死在跑之前**（區間、判讀樹、兩個岔都預先命名），9/6 兩格全部落空：
  - 斜率格：預測比值帶 [2.21, 3.31]、實測 **0.990**，且**落在預釘樹沒有的第三態**（衰減在、但 p 不驅動）〔④@0906〕。
  - 劑量格：預測三點皆正（+.062/+.032/+.011）、實測 **−.26/−.35/−.05** 反號〔⑤@0906〕。
  ⇒ 兩件事同時成立：**理論的那個版本錯了**、**方法有效** — 因為預測寫死在前面，我們才能在一天內乾淨地判掉它、並讀出替代讀法（transient）。paper 寫法＝把「預註冊 → 判讀樹 → 三態落點」當成**可搬的實驗紀律**寫進方法學貢獻（配 §5.9 的儀器無效判定、判讀樹、recon 好≠可學 2×2）。⛔ 不准事後把區間放寬讓它「命中」。

- **6.8 ★ v2 新增 — 療程有代價、而且我們只有一顆 seed**：WS+p=.3 把免查 .000→.552 的同時，帶查側從 .656 掉到 .556（−.10）〔⑥@0906、`[單顆]`〕。⇒ 誠實敘述＝「**把能力從 conditioning 搬進 weights，不是免費的搬運**」（帶查模式失去了它原本可以額外利用的 conditioning 增益），而非「兩全其美」。⚠️ 且此格目前只有 s40 一顆；八顆在烤〔⑮@0906〕— 成稿前沒收到八顆，這節就得從「最強證據」降級成「pilot」。

### §7 Conclusion（短）＋ Reproducibility statement（per-seed 全表＋協定＋json 帳）

---

## 4. Claim ↔ 證據對照表

| # | Claim | 支撐（實驗/定理/數字＋出處） | 狀態 |
|---|---|---|---|
| C1 | 錨條件化把路徑知識帶進 policy 本體（R0 腿） | f27n R0 .321→.454（+.133、≈2.7 SE）〔①〕 | **已定讞** |
| C2 | O-agnostic：teacher 換源不掉分 | ER route .918 ≈ hindsight .928〔DESIGN ⓪、PLAN T1〕；劣化 teacher 第三點 | 已定讞（兩點）；第三點 `[A6 待跑]` |
| C3 | 內化度量 (Int,ε) 拆得開三種零、儀器有效 | Def 1.4 三點校準〔THEORY-int §1.4〕；idp8 錶有效判定＋落 (0,0) 鎖死格〔⑦@0905〕。**v2 實測展示**：同一把錶讀出負（−.26/−.35/−.05）、零（p=0 崩潰格 zero .000）、與近 1（step-matched `[單顆]` 1.10／0.88）〔⑤@0906、③@0906〕 | 定義已定讞；**8000 步四點讀數已定讞**；step-matched 正式數字 `[f27nL/N5L 擴八顆 待跑@⑩@0906]` |
| C4 | p=0.3 dropout **實效鎖死＝動力學陷阱**（機制）〔9/6 postA1-patch；⛔ 舊 claim「在**條件冗餘下**鎖死」前提已被 ⑰ 證偽〕 | **Prop 2.1′/2.2′/2.3′**（動力學陷阱：$\mathrm{Int}^*\approx\eta_{\rm eff}/(p\lambda)$ 小目標＋12× 慢模）〔THEORY-postA1 §2〕＋散度 B/A=1.1%＋塌在 cond 生成端〔⑬〕＋guidance 無效 w=2 .344〔⑪〕＋小抄可讀 R².999〔⑩〕 | **「定讞」須撤**：三方實錘之一（⑫ 機制 1「沒錢可賺」）已倒〔postA1 §0〕、原前提已偽 ⇒ 降為**現象已定讞（散度 1.1% 直接量）＋機制改判動力學陷阱**；κT 校準疑點見 postA1〔追加勘誤 E4〕。**9/6 v2 再降一級**：鎖死的**靜態閉式**（$\mathrm{Int}^*=\eta/(\eta+p\lambda)$）在兩格 pre-registered 對決被證偽（④@0906 斜率比 0.990 vs 帶 [2.21,3.31]；⑤@0906 四點反號）⇒ 只留 **transient 讀法**、閉式降為極限錨 |
| C5 | 鎖死可解：三藥各打一層 | Prop 3.1/3.4/3.7〔THEORY-int §3〕。**v2 實驗側**：★ **WS 續訓療程**＝目前最有效（zero .000→.552、配對差 .004、代價 on −.10；`[單顆]` s40）〔⑥@0906〕；L_div 結構在場但單獨不買 R0（散度撐 ~.34、R0 on .508；`[單顆]`）〔⑦@0906〕；退火 p 的 toy 理論基礎**待重建**〔④@0906〕 | **實驗側首次為真**（單顆）；八顆在烤〔⑮@0906〕；理論側主藥的**理由**尚缺 |
| C6 | 合成律：訓練/eval 住 T=1、BFS=T→0、差距 ≤HT·logK | Lemma 1/2＋Prop 3〔THEORY-comp §1–2〕 | 草稿定理（Rei 磨嚴中）；Conj 7/stochastic **open 明標** |
| C7 | 字典搜索一般化 BFS（實驗級） | 字典 DP vs 連續臂＋沒見過的長路直接證據 | `[C1–C3 待設計→待跑]`（缺②唯一通道） |
| C8 | 字典三關驗收（D1/D2/D3）是必要儀器 | D2：f27n 2×2 recon 全漏、配對錶才看到〔①〕；D3：C5 病灶〔⑨〕；D1：`[C0 N1 待跑]` | D2 已定讞；D3 已定讞(破)；D1 **待跑** |
| C9 | 效率：免搜索毫秒級 vs 競品秒級 | 競品側 8~530s 已定讞〔⑥〕；我方 `[ms/plan@F3 待跑]` | **半定讞**（缺我方實測） |
| C10 | 高維可攜（ant Int>0；humanoid 誠實對標 64±4） | 口徑五件套〔DESIGN-0904〕；hindsight-only teacher=天然測試 | `[D1–D4 待跑·資料卡點=最大單點風險]` |
| C11 | 幾何：latent 今天無測地線結構→v2 造出來 | before 尺 .705/.212＋C-battery〔⑤'⑨〕 | before 已定讞；after `[B1–B4 待跑]`；輸了轉 negative-result ablation〔PLAN §3B〕 |
| C12 | 方法學：recon 好≠下游可學 | 2×2〔①〕＋fsq 全變體帳〔③〕＋N1 教訓〔②〕 | **已定讞** |
| **C13 ★** | **內化是慢動力學、不是平衡態性質**（v2 主 claim） | 8000 步四點 Int 全負（−.26/−.35/−.05）＝殘廢效應〔⑤@0906〕；同 p=.3 練到 11429 步 zero 超額 −.007→**+.086**、批級 R0 +.105（~1.5σ）〔①@0906、⑧@0906〕；⛔ generic 步數紅利不成立（無 intent 臂與 p=0 臂多練方向混雜、`[單顆]` 2σ 內）〔⑧@0906〕 | **八顆批級已定讞**（劑量四點＋idpxm）；轉正只有 p=.3 一個劑量點 ⇒ `[p=.1/.05 的 11429 版 待跑]` |
| **C14 ★** | **dropout 是免查能力的必要條件** | p=0 長訓（f27nL）zero 五腿全 0/250、連只吃 cond 的誠實 BC 地板也 0（on 格 bc=.244）；eval 程序健康（幾何 gate 過、往返尺正常）⇒ 真崩不是儀器壞〔③@0906〕；互證：同一顆續訓注入 p=.3 即救活〔⑥@0906〕 | **機制已定讞**（`[單顆]` s40、但為 0/250 全崩＝不需 SE） |
| **C15 ★** | **內化可事後誘導（療程）**：續訓期注入 dropout 把崩潰的免查模式救到與帶查同分 | WS+p=.3 續 4000 步：zero .000→**.552**、on .556、配對差 **.004**；WS+p=.1：zero .436；代價 on .656→.556〔⑥@0906〕 | **`[單顆]` s40 — ⛔ 尚不得寫成 claim 句**；八顆 26 jobs 在烤〔⑮@0906〕 |
| **C16** | 部署期「探索×驗證器」能把 bits 換成分數（**plan 層**） | BoN rung0.5、免訓練 CPU 40 秒：帶查 plan 層 @N=16 +.164/+.134/+.151；免查 headroom V0 f27n +.136／idpxm +.189／f27nL +.151；可部署 zero 只有 idpxm（zero pass@1 .794≈on .788；f27n .226／f27nL .254 OOD 崩），其免查＋裁判 plan 層 **.938**；支配引理 N≥2 成立、U 系 gate 全綠〔⑪@0906〕 | **plan 層已定讞**（`[單顆]` s40 三顆 ckpt）；⛔ **R0 端到端兌現率無錨、未量** `[≈5 GPU-hr＋LACOT_BON_N 旗 待派]` |
| **C17** | 誠實邊界：單圖下這把錶分不開「攤銷 BFS」與「背路線」 | Remark F1.5（$m{=}1$ ⇒ 該圖 info 併入常數、MI 帳看不見）＋value profile $V_0$–$V_3$ 階梯〔THEORY-0906-info-vs-function §F1.2–F1.3、F2 表〕 | **定義級已定讞**；$V_2$（換圖）**零證據** ⇒ 正文只准 claim「本圖上的組合算子」；`[換圖 transfer 待設計]` |

**「待跑」彙整（=實驗排程需求端；優先序照 PLAN §4；⚠️ v2 已改）**：
**在烤**：WS 療程八顆（26 jobs）／p=1 臂（解鎖 `INTENT_DROP<1.0` assert 後重灑）／三腿 intent eval（swap/noise）〔⑮@0906〕。
**v2 新增待跑（呈裁，優先序照 ⑩@0906）**：`f27nL/N5L/idp01L/ws40_idp0.3 各擴八顆`（釘 Int 正式讀數＋WS 療程確認）→ `WS 多步數點 4000→8000`（療程劑量–時間曲線）→ `殘廢效應機制探針` → `BoN 的 R0 層`（≈5 GPU-hr＋主檔 LACOT_BON_N 旗）→ `換圖 transfer`（C17 的唯一通道）。
**v1 續留**：`A5` 多路線 (s,g)（待設計）→ `A6` p=0.5＋teacher 第三點；`C0` N1 utilization → `C1–C3` 字典 pilot→×8；`B0` E-canon 補讀（writing 前置）＋`B1–B4` 幾何階梯；`D1–D4` ant/humanoid（長桿、D1 就開）；`F3` 效率儀器化（T2 前置）。
**v1 已消化**：`A1` 劑量臂＝收表完成〔①@0906〕；`A2` warm-start＝已跑、判決在 ⑥@0906；`A4` Int 讀數＝8000 步家族已出、step-matched 待八顆。

---

## 5. Reviewer 攻擊預想 ×5

| # | 攻擊 | 回應素材在哪 |
|---|---|---|
| R1 | 「分數沒贏 ECD 64±4，humanoid 憑什麼收？」 | 不同欄競爭：ECD/CD/CDGS/GSC 全是秒級推論期搜索/校正系、無 latent 無內化、Int/ε 對它們無定義〔⑥〕；同格誠實並列＋Pareto（score×latency×search-free）＋內化車道三隻 sweep 查無先占〔RELATED-WORK sweep 發現 1；PLAN §3 E/D4 風險條敘事整段可搬〕。⛔ 不 claim 分數 SOTA。 |
| R2 | 「訓練用 privileged oracle 不公平／部署假設不現實」 | DAPD「privilege illusion」正面回答〔RELATED-WORK F〕：idp=同顆權重推論期開關（vs SVA w/o MCTS 是重訓〔SVA 核驗〕）；帶查模式=資料建佔據圖＋(s,g) 本來就知=合法部署形態、⛔ 不寫 oracle〔⑤''〕；hindsight teacher 完全免 privileged〔C2 行〕。 |
| R3 | 「HDFlow 已有 latent 階層＋flow；Hydra 已有字典＋flow — novelty？」 | HDFlow 五軸差異表（navigate-only 無 stitch 無 search、連續無字典、無內化度量、成本只報 FurnitureBench）〔RELATED-WORK 精讀③〕；B 類差異句（字典管 skill/landmark vs 路線拓撲＋合成律理論位置＋idp 錶）〔RELATED-WORK B〕；OKBE 差異五件＋限定詞引用〔THEORY-comp R5〕。 |
| R4 | 「你們自己的錶在自己的環境讀出 Int≈0 — 度量還立得住？」 | **v2 換答案（更強）**：錶現在**不讀 0** — 它讀出一條軌跡：8000 步為**負**（殘廢效應 −.26/−.35/−.05）、11429 步**轉正**（+.086）、續訓療程把崩潰格拉到**與帶查同分**（.552/.556、`[單顆]`）〔⑤@0906、①@0906、⑥@0906〕。⇒ 錶的效度由它自己的**動態範圍**證明，而不是靠一個落點。舊素材保留為第二層：(0,0) 落格=鎖死診斷、非儀器失效〔⑦@0905 儀器判定有效＋Def 1.4 四理由〕；機制**兩方實錘＋一方已倒**〔9/6 postA1-patch〕；**匯率斷裂**升格 finding（⛔ 不再說「maze 冗餘」— ⑰@0905 量到 I≈2.5 bits）＋兩文獻空白；分母 undefined 條款把「不可讀」變形式推論〔Def 1.4 理由④〕。 |
| R5 | 「BFS 特例定理是 trivial／OKBE 已證；stochastic 呢？」 | 三層拆開陳述鐵則（定理級/Conj/Open）〔THEORY-comp §2〕；R5 五件差異（semiring 命名=我方觀察⛔不可寫成 OKBE 自陳；字典學出來 vs 給定；連續下層；T=1 vs exact-DP；落地量化）；stochastic 誠實 open＋開環 92 vs 73 引用〔§5.1〕；lfp 初始化警告=理論有牙齒的證據〔R9〕。 |

_後備（骨架不佔正文）：R6「**z-度量把 route 錢壓到 0.4%** ⇒ 換環境結論會變」〔9/6 postA1-patch：⛔ 原寫「maze 冗餘」— 改用 C-ii′ 語言，$I>0$ 但 $\eta_{\rm eff}\ll p\lambda$〕→ 6.1＋A5＋D 線；R7「離散化毒是你們實作爛」→ ③@0905 全變體帳（壓縮免費、round 付稅、u 空間慘一個量級）＋C 類共識引用。_

**★ v2 新增四條（9/6 判決直接製造出來的攻擊面 — 全部要正面寫進 §6，⛔ 不留給 reviewer 先講）**：

| # | 攻擊 | 回應素材在哪 |
|---|---|---|
| R8 | 「你們的理論預測被自己的實驗全部推翻（斜率比 0.990 vs 預測 [2.21,3.31]；劑量四點反號）— 這篇還剩什麼？」 | §6.7：**預註冊是這篇的方法學貢獻本體** — 因為區間與判讀樹寫死在跑之前，我們才能在一天內乾淨判掉平衡態版本、並讀出替代版本（transient/慢動力學），而 transient 版**是有正向實測支持的**（+.086 轉正、療程 .000→.552）。⛔ 不事後放寬區間。附：預釘樹的兩個岔都沒中、落第三態（衰減在、p 不驅動）— 這個「連岔都預先命名過」的細節本身就是紀律的證據〔④@0906〕。 |
| R9 | 「最強的那格只有一顆 seed（s40），而你們自己在同一篇裡說 s40 是強 seed。」 | **我們自己先講**：②@0906 就是這個教訓的紀錄（idpxm s40 .716 → 八顆 .431±.060）⇒ 全檔 `[單顆]` 標記制度、⛔ 單顆不進 claim 句〔§6 鐵則 9〕。療程八顆在烤〔⑮@0906〕；**收不到八顆就把 §5.6 降級成 pilot、把 C15 從 claim 降成 observation**（§6.8 已預寫此條款）。⚠️ 另一半答案：C14（p=0 臂 zero=0/250 全崩）**不需要 SE** — 五腿全 0 的量測不是抽樣噪音問題。 |
| R10 | 「你們怎麼知道模型學到的是『規劃算子』而不是把這張迷宮的路線背下來？」 | §6.6 正面認：**單圖下這把刀不存在**（$m{=}1$ ⇒ 該圖 info 併入常數、任何 MI 帳看不見；Remark F1.5）⇒ 我們只 claim $V_1$（同圖 fresh $(s,g)$＝stitch 本義：查表死、組合算子活），**$V_2$ 明標零證據**、換圖 transfer 列為 future work〔THEORY-0906-info-vs-function F1.2–F1.3〕。⇒ 把「刀在哪、為什麼現在切不下去」寫成框架貢獻。 |
| R11 | 「BoN 那些增益是在你們自己的 plan proxy 上量的 — 端到端呢？」 | 照實答：**rung0/rung0.5 全部錨在 plan 層、R0 層無錨也未量**〔⑪@0906、BoN 卡 §3.2〕；判讀樹早已預釘 proxy-gap 那一格（plan 層 ≥.10 而 R0 <+.02 ⇒ 判瓶頸在 decoder/executor、U4 roundtrip 82.8% 天花板，⛔ 不加 N）〔BoN 卡 §3.3-3〕。⇒ 我們沒有把 plan 層說成端到端，且**事前就寫好了它可能不兌現時該怎麼判**。 |

---

## 6. 風格鐵則（骨架→成稿全程有效）

1. **⛔ 不准編數字**。骨架與成稿裡每個數字必須能指回 docs 出處。**引用鍵（v2 定案、⛔ 全檔唯一解讀）：裸節號 ①~⑱ ＝ FINDINGS-0905（v1 沿用不動）；9/6 大收表的節號一律寫 `⑥@0906` 形**；THEORY-comp/int=兩份 THEORY-0905、RELATED-WORK/DESIGN/PLAN 同日檔。查無出處的數字=刪。
2. **佔位符明標**：`[值@產出實驗]` 形（如 `[ms/plan@F3]`、`[Int@A4]`）— 填入時把佔位符換成「數字＋FINDINGS 節號」，⛔ 不准先填後補出處。
3. **三態分級跟著走**：已定讞/在跑/待設計 — 進稿時「在跑/待設計」的格一律標 partial 或砍，⛔ 不寫進 claim 句。
4. **用詞裁決已釘**：「帶查(map) vs 免查」⛔ 不寫 oracle〔⑤''〕；「generalizes BFS」三層拆開〔THEORY-comp §2〕；OKBE 引用帶限定詞〔R5〕；TMD 的「33x」自文與表不符、引用自己重算〔RELATED-WORK 精讀②〕。
5. **引用前升【正】**：任何【讀】【掃】級文獻進稿前升級成正文級驗設置〔RELATED-WORK 使用說明、9/4 誤報教訓〕。
6. **subgoal 格內化欄 undefined**：⑤'' 裁決=Def 1.4 分母條款的形式推論 — 表格照此渲染，⛔ 不算負殘留。
7. **per-seed 全表進附錄、災難 seed 不剔除**〔⑦〕；效率數字必須自測（F3）才上 T2。
8. **「A1」三重命名衝突 — paper 化前必須改名**〔9/6 深審 S10〕。同一個符號現在指三件不同的事、且已在本檔同頁共存（§4.3 的 (A1) 確定性 vs §4 表／§5 的 A1 實驗線）：

   | 現用 | 出處 | 意思 | 建議新名 |
   |---|---|---|---|
   | A1 | 內化檔／postA1 | **條件冗餘假設**（$I_{\rm data}=0$；⑰ 已證偽、降為 A1-lim 極限錨） | **H-red** |
   | (A1) | 合成律檔 §1.1 | **確定性轉移**（$P$ 是函數） | **(D-env)** |
   | A1 | PLAN／FINDINGS／本檔 | **內化線實驗編號**（劑量二臂） | 保留（實驗編號欄） |

   ⛔ 三者不得在正文共用 "A1"；改名前的骨架階段，引用時一律帶限定詞（「A1 冗餘假設」／「(A1) 確定性」／「A1 實驗臂」）。

9. **★ v2 — `[單顆]` 標記制度**：只有 s40 一顆 seed 的格，⛔ 一律在數字旁標 single-seed，⛔ 不進 claim 句、不進 abstract 主張句、不與有 ± 的批級數字並列在同一列。理由是實測的：idpxm s40 .716 → 八顆 .431±.060（s40 強 seed 假象第三次咬人）〔②@0906〕。**例外**：全 0/250 的崩潰格（C14）可寫成機制陳述 — 五腿全 0 不是抽樣噪音問題，但仍標 single-seed。

10. **★ v2 — 還在烤的不寫**：`[在跑]` 的結果⛔ 不得出現在骨架的數字位、更不得進 abstract。今天在烤的是 WS 療程八顆、p=1 臂、三腿 intent eval〔⑮@0906〕。⇒ 每次改稿前先確認「這個數字已經在 FINDINGS 裡了嗎」。

11. **★ v2 — 層級不得混（plan 層 ≠ R0 層）**：BoN／rung0／rung0.5 的一切增益都是 **plan 層**（pts_J raw hit）；R0 端到端是另一張表、目前無錨〔⑪@0906、BoN 卡 §3.2〕。⛔ 不准把 plan 層數字寫成「成功率提升」。同理：subgoal 腿與 R0 腿不得互相代表〔⑤''〕。

12. **★ v2 — 引 u 先錨地基**：任何提到 u／latent thought 的句子，第一句先寫「u 編碼一段想像中的軌跡」再進抽象層（context／scratchpad／工作檯）〔CANON-u-semantics〕。⛔ 抽象名詞不得取代地基。

---

_骨架 v2 完（2026-09-06）。_
_v2 改了什麼（一句話）：主敘事從「一把錶」換成「**一條軌跡** — 內化是慢動力學、而且可以用療程誘導」；並把今天被自己實驗打掉的平衡態理論，從隱患改寫成方法學賣點。_
_下游：F4（9/18 abstract；草稿見 `docs/ABSTRACT-DRAFTS-0906.md`）→ F2（method+theory 節開工）。_
_**v2 押注點（需主人裁）**：(1) 主敘事採 T-D（慢動力學＋療程）還是維持 T-A；(2) WS 療程八顆若不如單顆，§5.6/C15 降級的門檻要不要現在就釘；(3) BoN 的 R0 層（≈5 GPU-hr）在 9/18 前跑不跑 — 不跑就只能用 plan 層敘事；(4) 換圖 transfer（C17／R10 的唯一通道）是列 future work 還是搶時間做 pilot。_
_v1 遺留押注點（未消）：④軸重排裁決、C 線是否提前（決定 C7 行）。_
