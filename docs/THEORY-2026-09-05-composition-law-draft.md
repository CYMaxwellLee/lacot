# THEORY — 合成律與字典搜索：定理草稿 v0.1（2026-09-05）

_理論使魔（合成律與字典線）起草，交 Rei 磨嚴後進 ICLR 理論節。_
_分級約定：**Lemma/Prop** ＝ 在明列假設下可證（多為經典結果的 instantiation，證明梗概附上）；
**Conjecture** ＝ 我們相信、但目前缺一塊才能證；**Remark** ＝ 定位與差異、不承重。
⛔ 本檔只覆蓋合成律與字典（路線二＋BFS 儀器的理論位置）；quasimetric 幾何的形式化
（路線一）與內化 gap 的正式定義（v0 第 2、3 洞）歸另一線，此處只留接口。_

---

## 1. 形式設定

### 1.1 環境、軌跡、知識源、字典

- **環境**：goal-reaching MDP $\mathcal E = (S, A, P, G)$，$G \subseteq S$。本檔預設
  **(A1) 確定性轉移**（$P$ 是函數）；stochastic 推廣見 §5。
- **軌跡**：$\tau = (s_0, \dots, s_T) \in S^*$；到達事件 $\mathrm{succ}(\tau, g) = \mathbb 1[\exists t: s_t \in B(g)]$（$B(g)$ ＝ 目標容差球）。
- **可查知識源** $O$：訓練期可查詢的路徑知識 oracle（實例：BFS route、hindsight 摘要），
  經摘要算子給出錨 $a = A(\tau, O)$。$O$ 只在訓練期在場（內化框架的前提，本檔不重述）。
- **計畫空間**：latent 計畫 $z \in Z$（token 序列），decoder $D: Z \to S^*$ 給 subgoal 座標序列。
  下層 rectified flow 學條件分佈 $p_\theta(z \mid s, g, a)$，以 exact NLL 訓練；
  推論抽樣 $z \sim p_\theta(z \mid s, g, \varnothing)$ 後 decode（＋refine）。
- **字典** $M = \{m_1, \dots, m_K\}$：intent 層的有限離散條目集。每個 $m$ 經 decoder 對應
  原空間的一個中繼區域 $R(m) \subseteq S$（由 $D$ 誘導；學出來的，非給定）。
  $K = |M|$ 意圖上是小的（intent 層小字典，非 per-token 量化 — f27n 定讞的層級選擇）。

### 1.2 Semiring 與合成律

**定義 1（semiring）**：$(\mathbb K, \oplus, \otimes, \bar 0, \bar 1)$，$\oplus$ 交換 monoid（單位 $\bar 0$）、
$\otimes$ monoid（單位 $\bar 1$）、$\otimes$ 對 $\oplus$ 分配、$\bar 0$ 吸收。〔Mohri 2002, *Semiring
frameworks and algorithms for shortest-distance problems*；Goodman 1999, *Semiring parsing*〕

**主方程（合成律）**：對值函數 $V: S \times G \to \mathbb K$，

$$V(s,g) \;=\; \bigoplus_{m \in M} \big[ V(s,m) \otimes V(m,g) \big]. \tag{CL}$$

**溫度族**：$x \oplus_T y := T \log(e^{x/T} + e^{y/T})$，$\otimes = +$，值域 $[-\infty, \infty)$。
$T = 1$ ＝ log-semiring；$T \to 0^+$ ＝ max-plus（tropical 的 max 版）。

### 1.3 對應表：哪個 semiring 是我們的目標函數（v0 待磨嚴第 1 洞）

| semiring | $\oplus$ | $\otimes$ | 值的語意 | 系統對應 |
|---|---|---|---|---|
| Boolean | $\vee$ | $\wedge$ | $\{0,1\}$ 可達 | BFS 儀器（可達性） |
| tropical (min-plus) | $\min$ | $+$ | 步數／代價 | BFS 儀器（最短路）＝ $T \to 0$ 凍結極限 |
| Viterbi (max-product) | $\max$ | $\times$ | 機率 | argmax 單計畫 — **我們不 eval 這個** |
| log-semiring ($T{=}1$) | LSE | $+$ | log-機率 | **NLL 訓練＋抽樣 eval（我們）** |
| 溫度族 $\oplus_T$ | $\mathrm{LSE}_T$ | $+$ | 插值 | refine / best-of-N（定性，見 R2） |

**統計前提（引理 1 用）**：
- **(S1) waypoint 分解**：真分佈滿足 $p^*(z \mid s,g) = \sum_{m \in M} w^*(m \mid s,g)\, p^*(z_{\mathrm{pre}} \mid s,m)\, p^*(z_{\mathrm{post}} \mid m,g)$，即給定中繼 $m$ 時前後兩段條件獨立（＝ (D2) 的分佈版，見 §3）。
- **(S2) realizable ＋ population**：$p^* \in \{p_\theta\}$，且以 population NLL 論證（有限樣本 gap 見 §5）。

**引理 1（目標函數的 semiring 身份）**：在 (S1)(S2) 下：
(i) population NLL 最小化恢復 $p^*$，且 log 域上
$\log p^*(z \mid s,g) = \mathrm{LSE}_{m \in M}\big[\log w^*(m \mid s,g) + \log p^*(z_{\mathrm{pre}} \mid s,m) + \log p^*(z_{\mathrm{post}} \mid m,g)\big]$
— 即 (CL) 在 **log-semiring**（$\oplus = \mathrm{LSE}$、$\otimes = +$）的實例，值＝log-機率。
(ii) eval 讀數 $\mathbb E_{z \sim p_\theta}[\mathrm{succ}(D(z), g)]$ 是對 $m$ **邊際化後**的期望
— sum-product 語意的泛函，與 (i) 經 $\log$ 同構等價；它不評任何 argmax 計畫。
(iii) argmax 計畫對應 Viterbi（max-product）＝ 不同的 semiring；三者由溫度族 $\oplus_T$ 連接，
$T = 1$ 是 (i)(ii)、$T \to 0$ 是 max-plus。
**⇒ 判決：我們的訓練目標與 eval 語意住 log-semiring（$T{=}1$）；「max-product 一般化」
在 paper 裡要寫成溫度族的 $T \to 0$ 端，不是我們系統的本體。**
_證明梗概_：(i) NLL＝KL＋常數，(S2) 下唯一極小＝$p^*$；(S1) 取 log 即 LSE 形。
(ii) 期望對混合分佈線性。(iii) 定義代入。∎

**引理 2（凍結極限，BFS 是 $T \to 0$）**：對 $x \in \mathbb R^K$：
$\max_i x_i \le T \log \sum_i e^{x_i / T} \le \max_i x_i + T \log K$。
故 $\oplus_T \to \max$ 一致收斂、速率 $T \log K$；且 LSE 對 sup-norm nonexpansive
（$\nabla$＝softmax、$L_1$ 範數 1），巢狀 $H$ 層合成的總偏差 $\le H \cdot T \log K$。
**⇒ 有限小字典（$K$ 小）× 有限 horizon 下，log-semiring 的 (CL) 與 tropical 的 (CL)
差距受 $H T \log K$ 控制 — 「BFS 包在合成律裡」由此從敘事升為量化陳述。**
_證明梗概_：上界＝每項 $\le e^{\max/T}$ 共 $K$ 項；下界＝丟掉其餘項。傳播＝nonexpansive 疊加。∎

**Remark R2（refine 的溫度定位；定性、不承重）**：best-of-$N$／value-directed refine 把
有效溫度從 $T = 1$ 往 $0$ 壓（$N \to \infty$ 恢復 max）。我們部署點在溫度族中段。
$N$ 與有效 $T$ 的定量關係未證，見 §5。

---

## 2. BFS 特例（草稿定理）與 OKBE 的差異

**假設**：
- **(A1)** 確定性轉移（§1.1）。
- **(A2) 字典覆蓋（精確版）**：$M$ 含格圖全部頂點（或至少每條最短路的所有中繼點），
  且一步關係 $E(s,m) = \mathbb 1[m \in \mathcal N(s)]$ 可查。
- **(A3) 代價結構**：Boolean（可達）或 min-plus 單位邊權（步數）。

**命題 3（定點迭代＝BFS）**：在 (A1)(A2)(A3) 下，取 $V_0(s,g) = \mathbb 1[s = g]$、迭代
$V_{t+1}(s,g) = V_0(s,g) \vee \bigvee_{m \in M}\big[E(s,m) \wedge V_t(m,g)\big]$，則
(i) $V_t(s,g) = \mathbb 1[d_{\mathrm{graph}}(s,g) \le t]$；(ii) 每輪「新變 1」的集合＝以 $g$ 為根的
BFS 第 $t$ 層 frontier；(iii) $\le \mathrm{diam}$ 步收斂到最小定點＝可達性。
min-plus 版同構：迭代＝Bellman-Ford，單位邊權時逐層鬆弛＝BFS。
_證明梗概_：對 $t$ 歸納；經典 algebraic path problem〔Mohri 2002〕的特例。∎

**Remark R4（兩種迭代形，B 階段實作要分清）**：(CL) 的自映射有兩形 —
(a) 單步展開 $V \mapsto V_0 \oplus (E \otimes V)$（＝value iteration，逐層；BFS＝這形）；
(b) 全合成 $V \mapsto V \otimes V$（matrix squaring，倍增步）。冪等 $\oplus$ 下兩者**定點相同**
（皆＝代數閉包／最短路），迭代軌跡不同。字典 DP／beam 用 (a) 或 (b) 皆可，
但語意保證都掛在「從 $\bot$ 起算的最小定點」上（見 Prop 8 / R9）。

**Remark R5（vs OKBE Thm 2.2〔arXiv:2506.09499, §2.10〕）**：OKBE 證的是：確定性 CTMDP
＋有限且覆蓋完整的 option 庫＋**模型已知（DP 精確算）** ⇒ option 層 tree search 最優
open-loop 且完備。與我們的差異五件：
(1) OKBE 全文無 semiring／max-product 語言 — 「其設定退化為 max-times semiring」是
**我方數學觀察，不可寫成其自陳**；(CL) 的顯式 semiring 命名是我們的框架貢獻。
(2) 他們字典（option 庫）**給定**；我們的 $M$ **學出來** ⇒ (A2) 從假設變成統計性質（§3）。
(3) 他們無連續下層；我們有 flow decoder ⇒ 中繼是 decoder 誘導的**區域／分佈**，非精確狀態。
(4) 他們 exact-DP（$T = 0$ 語意）；我們 NLL＋抽樣（$T = 1$ 語意，引理 1）。
(5) 他們 logic/生理域無量化實驗；我們落地導航＋量化內化讀數。
引用時帶限定詞：OKBE＝「option 層搜索等價定理**存在**」的旁證，非我方正確性證明。

**一般化 claim 的可辯護形式（分級明確）**：
- **定理級**（(A1)＋精確 (A2)＋(A3)）：Prop 3（BFS＝特例）＋ Prop 6（字典 DP 恢復最優）。
- **Conjecture 級**：學到的字典（$\varepsilon$-覆蓋）＋近似 decoder 下的近似保證（Conj 7）。
- **Open**：stochastic 轉移下的最優性（§5.1）。
⛔ paper 裡「dictionary search generalizes BFS」只能以上述三層拆開陳述。

---

## 3. 字典要滿足什麼：假設清單與可量測代理

**目標陳述**：「字典空間 DP ≈ 原空間規劃」。以下三條假設，各配一個 B 階段驗收關的錶。

- **(D1) $\varepsilon$-覆蓋（coverage）**：對 eval 分佈支撐內的 $(s,g)$，存在 $m \in M$ 使
  $\bigoplus_m [V(s,m) \otimes V(m,g)] \succeq V^*(s,g) \ominus \varepsilon$（分解不損超過 $\varepsilon$）。
  - **代理＝利用均勻度（utilization）**：條目使用分佈的熵／死條目率。死條目 ⇒ 有效
    $K$ 縮水 ⇒ 覆蓋半徑增大 ⇒ $\varepsilon$ 上升。⚠️ 必要非充分：均勻但擺錯位置仍可不覆蓋
    （iFSQ 病的反面）；FINDINGS-0905 ② 記載 N1 均勻度探針從未留檔 — 這關目前**未驗**。
- **(D2) $\delta$-組合性（compositionality）**：(S1) 的近似版 — 給定 $m$ 時前後段近似條件獨立：
  $\mathrm{KL}\big(p^*(z \mid s,g,m) \,\big\|\, p^*(z_{\mathrm{pre}} \mid s,m)\, p^*(z_{\mathrm{post}} \mid m,g)\big) \le \delta$。
  這是 $\otimes = +$（log 域乘積）語意的基礎＝intent 層的 Markov 性。
  - **代理＝配對可學性（pairability）**：字典 × stage1 配對後，下游 head 的可學性讀數。
    實證錨：f27n 2×2（FINDINGS-0905 ①）— FSQ 在場 subgoal $-.18{\sim}-.19$、intent 增益被掐死，
    而 recon 指標全漏 ⇒ 組合性壞在「表徵×字典交互」時，只有配對後的下游錶看得到。
- **(D3) $\gamma$-decoder 一致性（decoder consistency）**：(i) $D(R(m))$ 的支撐落在 $m$ 宣稱的
  中繼區域（語意一致）；(ii) 兩段拼接處 decode 合法（不穿牆、無跨模式假 interpolation），
  違反率 $\le \gamma$。
  - **代理＝往返一致性（round-trip）＋拼接合法率（interp-consistency）**。
    實證錨：C-battery（FINDINGS-0905 ⑨）— C5 證偽密度洞、病灶釘在「decoder 對兩模式
    之間輸入的行為」＝ (D3)(ii) 現在實測是**破的**；C4 給 before 錨（latent lerp 合法率
    .638 vs 座標 .535）。

**命題 6（精確版正確性）**：(A1)＋(D1, $\varepsilon{=}0$)＋(D3, $\gamma{=}0$)＋min-plus 下，
字典空間 DP 的定點 $\hat V$ 滿足 $\hat V(s,g) = V^*(s,g)$（原空間最短路）對覆蓋支撐內全部 $(s,g)$。
_證明梗概_：$\hat V \ge V^*$ 由每次合成對應一條可行拼接路（(D3) 保證 decode 合法）；
$\hat V \le V^*$ 由最優路的中繼點都在 $M$ 內（(D1)）、最優子結構逐段收緊。∎
（本質＝OKBE 型結果的 semiring 重述；novelty 在框架不在此證明。）

**Conjecture 7（近似合成界）**：(A1)＋(D1)–(D3)（$\varepsilon, \delta, \gamma > 0$）＋horizon $H$ 下，
$\big|\hat V(s,g) - V^*(s,g)\big| \le C \cdot H \cdot (\varepsilon + \delta + \gamma)$（線性疊加形）。
_缺的一塊_：$\delta$（KL）到值差的換算（Pinsker 給 $\sqrt{\delta/2}$ 形，可能改成
$C H (\varepsilon + \sqrt\delta + \gamma)$）；沿 horizon 線性疊加有開環誤差線性界
〔arXiv:2605.08732〕當旁證，但該界是原空間開環誤差、非字典空間 DP 誤差，接橋未搭。

---

## 4. 定點存在唯一性（v0 第 4 洞）

**命題 8（存在性，Knaster–Tarski 路線）**：值域 $\mathbb K$ 為完備格（$\{0,1\}$、$[0,1]$、
$[-\infty, 0]$、反序 $[0, \infty]$ 皆是），$\oplus$ 取逐點 sup／max、$\otimes$ 對兩參數單調。
則 Bellman 算子 $(\mathcal T V)(s,g) = \bigoplus_m [V(s,m) \otimes V(m,g)]$（含 R4 兩形）單調，
故定點集非空且成完備格〔Tarski 1955〕。若 $S, G, M$ 有限且 $\oplus, \otimes$（Scott-）連續，
Kleene 迭代 $\bot, \mathcal T\bot, \mathcal T^2\bot, \dots$ 收斂到最小定點 $\mathrm{lfp}(\mathcal T)$；
有限值格（Boolean、有限步數格）上有限步到達。
_證明梗概_：單調性逐點檢查；K-T 給定點格；Kleene 鏈由連續性收斂。∎

**命題 9（唯一性的兩條充分路）**：
(a) **Banach 路線**：折扣結構（$V(s,m) \otimes V(m,g) = c(s,m) + \gamma_{\mathrm{disc}} V(m,g)$、
$\gamma_{\mathrm{disc}} < 1$）＋ $\oplus \in \{\max, \min, \mathrm{LSE}\}$ 皆 sup-norm nonexpansive
⇒ $\mathcal T$ 是 $\gamma_{\mathrm{disc}}$-收縮 ⇒ 唯一定點＋幾何收斂〔Banach 1922〕。
log-semiring（$T = 1$）版＝soft value iteration 的收縮性，同一論證。
(b) **min-plus 無折扣**：邊代價 $\ge c_{\min} > 0$＋目標吸收 ⇒ 定點在可達集上唯一
（＝最短路距離；不可達處 $+\infty$）。_梗概_：任一定點沿自身 greedy 展開 $\ge$ 最短路；
沿最短路歸納 $\le$ 最短路；兩夾。∎

**Remark R9（lfp 語意承重 — 對 B 階段實作有直接後果）**：Boolean 可達性下**最大**定點
trivial（無孤立點的圖上 $V \equiv 1$ 也是定點）⇒「可達性」＝**最小**定點的語意，
BFS 從 $\bot$（只有 $g$ 自身）起迭代正是 lfp 構造。字典 DP／beam 的初始化必須從 $\bot$
起（或用 9(a)(b) 的收縮版），否則收斂到錯的定點而所有健康檢查都綠 — 這是
「檢查會通過因為它壞了」形狀的理論根源之一。

---

## 5. 誠實邊界：現在撐不起定理的主張

1. **Stochastic 環境最優性（open）**：全部 Prop 都吃 (A1)。stochastic 下 (i) $\otimes$ 與期望
   不交換（Jensen gap）；(ii) open-loop 合成 ≠ closed-loop 最優 — 實證上開環誤差沿長度
   線性疊加、閉環 92% vs 開環 73%〔arXiv:2605.08732〕。升級需要：值改期望代價＋合成改
   policy 層（options 語意），或接受「open-loop 值的合成律」這個較弱陳述。
2. **(S1) 分解假設本身**：確定性最短路 teacher 下，最優子結構使 (S1) 在 argmax 支撐上
   成立（可寫成小引理）；但作為**分佈等式**（含 teacher 的次優性、hindsight 源）是假設。
3. **學到的字典**：(D1)–(D3) 今天是假設不是性質 — (D3)(ii) 實測是破的（C5）、(D1) 的
   utilization 關未驗（N1 無檔）、(D2) 已有被量化毒死的前科（f27n）。⇒「字典空間 DP ≈
   原空間規劃」目前是**目標**；三個錶（utilization／pairability／round-trip）就是升級門票，
   這正是 B 階段驗收關的理論身份。
4. **有限樣本與 optimization error**：引理 1 是 (S2)（population＋realizable）下的陳述；
   有限樣本 NLL 與 flow 訓練誤差到 (CL) 偏差的換算未定量。
5. **refine 溫度**（R2）：定性。$N$–$T$ 的定量對應（order statistics of log-probs）未推。
6. **接口聲明**：quasimetric 行（$V = -d$、三角不等式取緊）在對應表佔位但本檔不展開
   — 幾何線的形式化（含對稱破缺 loss、第 2 洞）歸另一線；內化 gap 的正式定義（第 3 洞）
   同。兩線只需一致於：內化品質只影響 $p_\theta \to p^*$ 的逼近（(S2) 的鬆動），
   不改變 (CL) 的 semiring 身份。

## 參考（定義級）

Tarski 1955（lattice fixpoint）；Banach 1922；Mohri 2002（semiring shortest-distance）；
Goodman 1999（semiring parsing；log／Viterbi semiring）；min-plus／tropical：標準教材。
arXiv:2506.09499（OKBE）；arXiv:2605.08732（開環誤差線性界）；arXiv:2509.20478（TMD，
幾何線鄰居、本檔僅佔位）。內部：NOTE-0905（v0）、FINDINGS-0905 ①⑨、RELATED-WORK-0905。
