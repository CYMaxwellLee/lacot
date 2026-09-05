# THEORY — post-A1 修訂：$I>0$ 下的鎖死重述＋資訊→SNR 橋＋劑量曲線預測（revision v0）

_理論推進使魔（Fable 級、主人授權；兩並行之「理論」線）2026-09-06 凌晨。上游（唯讀）：
【內】THEORY-0905-internalization-formal（§2 為修訂對象）、【丁】THEORY-0905-upgrades（LT toy 閉式）、
【丙】THEORY-REVIEW-0905（七條需加假設＋P2.3(iii) 修正）、FINDINGS-0905 ⑦⑮⑰。
觸發：⑰ 直接量測 $I(\tau;a|s,g)\approx2.5$ bits $\ne0$ ⇒ A1 於本資料**證偽**；⑬ 鎖死實錘
（$\varepsilon$ 比 $1.1\%$）不受影響、但其解釋須從「合法全域最優」改為「動力學陷阱」。
⛔ 分級鐵則照舊：**Prop＝列明假設下完整證明在檔（或逐字引上游已證件）**、**Conj＝未證**、
Remark＝解讀。本檔不改上游、不碰 code。_

---

## 0. 修訂範圍（什麼倒了、什麼沒倒）

- **倒的**：A1（$I_{\rm data}=0$）作為本資料的實況 ⇒ 【內】Prop 2.1／2.3(i)(ii) 的前提為偽，
  兩者降為「A1 極限錨」（理論參考點、非本資料陳述）；⑫ 機制 1「沒錢可賺」倒。
- **沒倒的**：⑬ $\varepsilon_{\rm rel}=1.1\%$ 是對 $\varepsilon$ 的直接量測、不經 A1；Cor 2.2
  （guidance 無效）只依賴 $\varepsilon\approx0$、不依賴 A1，**保留、前提改標實測**；
  Def 1.4（Int 三點校準）、§3 藥方形、§4 幾何全不動。
- **升格的**：Remark 2.4 賽跑（不依賴 A1）＋【丁】LT-6 overshoot 定理 ⇒ 主敘事；
  【丁】$\mathrm{Int}^*=\eta/(\eta+p\lambda)$ 閉式 ⇒ 定量骨架（⑮ 已獨立模擬驗至機器精度）。

## 1. 假設區修訂（含丙七條落實；§4 逐條對照）

- **A1-lim（原 A1、降級）**：$I_{\rm data}(\tau;a|s,g)=0$。地位＝極限錨：$\eta\to0$ 時本檔諸
  命題退回【內】原版。⑰ 實測 $I\approx2.5$ bits ⇒ 本資料 **A1′** 成立：$I_{\rm data}>0$。
- **C-ii′（丙 B15；量化加強版）**：route 在 latent 可讀且**在 $z$-度量下有變異權重** —
  記 $\eta_{\rm eff}\equiv$ route 分量在 $z$（e_target）空間的變異、$\lambda\equiv(s,g)$-可預測
  分量的變異（per 有效模）。⛔ $I(\tau;a|s,g)>0$（bits）不蘊含 $\eta_{\rm eff}$ 大 — CFM 的
  $L^2$ 目標以**變異**計價、不以 bits 計價（§3 的匯率斷裂即此）。
- **A2u（唯一性量詞、丙 B5）**：Prop 2.1-類唯一性只對 $p\in(0,1)$；端點 $p\in\{0,1\}$ 的
  離場支零權重、行為不受約束（f27n 的 cond 差 .6046 與此相容）。
- **A3′（enc 充分性、丙 B3）**：Prop 1.3 恆等式在 $z={\rm enc}(\tau)$ 下退為
  $\mathbb E_a{\rm KL}\le I_{\rm data}(\tau;a|s,g)$（DPI 方向）；等式需 enc 對 $a$ 充分。
  本檔一律以**不等式（上限）語意**引用。
- **LT 搬運（Prop LT-0 條件）**：線性高斯＋直線 path＋仿射速度場；主線為 $t\to0$ 端切片。
  對真網路一律標「量級指引」，不標等式。

## 2. §2 重寫草稿：$I>0$ 下的條件通路鎖死

### 2.1 invariant 解的正確地位：非全域最優

**Prop 2.1′（invariant 次優；A1′＋C-ii′＋A2）**：設 $I(z;a|s,g)\ge\delta'>0$（C-ii′ 的
bits 半邊）。則 population 最優的兩支條件期望在正測度上分離、intent-invariant 解**不再是**
$L_p$ 的全域最優（$p\in(0,1)$）：由 Prop 1.3（不等式版、A3′）正折扣存在，invariant 解在
keep 支付出 $\ge$ 該折扣的超額損失。
_證_：【內】Prop 3.7 的拆除論證逐字搬入（DPI＋折扣恆等式＋場同⇒分佈同反證；丙 B15 檢過
成立、前提即 C-ii′）。∎
**Remark**：⑰ 量的是 $\tau$-層 $I\approx2.5$ bits；$z$-層 $\delta'>0$ 由 ⑰ Z2 雙峰
（長窗 $\|\varphi_-\|^2$ 佔比 21.8%、$\dim_{\rm eff}=29$）**強烈支持但未直接量** —
$I(z;a|s,g)$ 的直接量測（⑰ 儀器換 $z$ 輸入重跑）是本命題前提的補完實驗。

### 2.2 定量骨架：鎖死是連續量、不是二值

**Prop 2.2′（toy 閉式搬運；LT-D/LT-M 假設）**：線性高斯 toy 內（【丁】Prop LT-1/LT-2、
⑮ 機器精度驗訖）：
$$W_a^*(p,\eta)=\sqrt\eta\,B\,(pC+\eta I)^{-1},\qquad
\mathrm{Int}^*_i(p)=\frac{\eta}{\eta+p\lambda_i},\qquad
\kappa=(1-p)(p\lambda+\eta).$$
⇒ $I>0$ 時**最優解本身帶非零依賴**、但 $\eta\ll p\lambda$ 時被 ridge 壓到
$\mathrm{Int}^*\approx\eta/(p\lambda)\approx0$ — 鎖死的 $I>0$ 版語意＝**深度 crossover**，
無 sharp 相變（【丁】§1.3 判決）。

**Prop 2.3′（鎖死＝慢趨近小目標；LT 假設＋丙 B7 限定版）**：$I>0$、$p\in(0,1)$ 下：
(i) *目的地小*：population 終點的 $a$-讀出能量比值 $=\mathrm{Int}^*(p)=\eta/(\eta+p\lambda)$；
(ii) *走得慢*：趨近速率為慢模 $\kappa=(1-p)(p\lambda+\eta)$，與主通路收斂速率比
$\sim12\times$（$p{=}0.3,\lambda{=}1$；【丁】§1.6、⑮ 收緊①精確化 $\mu_-$ 比 2.759）；
(iii) *$\varepsilon\approx0$ 邻域內下游平坦*（Cocos 型、丙修正版）：**在 $\varnothing$ 支已
最優＋$\varepsilon_{\rm cond}=0$＋A4、且只算 adapter 以下游的共享參數**時，兩支梯度差
$\le C\varepsilon$（活化全同⇒Jacobian 全同；丙 B7 的限定詞全部收入）。adapter 參數**排除
在外**：其驅動 $\propto\mathbb E[{\rm Cov}(a,\delta|s,g)]=O(\sqrt{\eta_{\rm eff}})\ne0$ —
$I>0$ 時 invariant 流形**不是駐點集**、只是逃逸驅動 $O(\sqrt{\eta_{\rm eff}})$ 對上回拉
$O(p\lambda)$，淨平衡即 (i)。
_證_：(i)(ii)＝【丁】Prop LT-1/LT-5 逐字；(iii) 前半＝丙 B7 修正版（檢過成立）、後半＝
LT 梯度式 $\dot W_a$ 的 $\sqrt\eta B$ 項。∎
**⇒ 鎖死的修訂陳述（一句話）**：**系統不是被困在一個非法駐點，而是以慢速率 $\kappa$
合法地收斂到一個本來就被 ridge 壓扁的小目標 $\mathrm{Int}^*\approx\eta_{\rm eff}/(p\lambda)$**
— 「陷阱」的成分是 (ii)(iii) 的時間尺度分離＋(i) 的目的地渺小，不是最優性。

**Remark 2.4′（賽跑＝主敘事；升格）**：【內】R2.4 的賽跑不依賴 A1、且已被【丁】LT-6
升為 overshoot **定理**：cold-start 下 $a$-通路借 $\varnothing$ 支未收斂的 residual 先長
（單峰、峰時閉式）、$\varnothing$ 支收斂後只剩 $\eta_{\rm eff}$ 級供能 vs $p\lambda$ 級
ridge、被壓回 $\mathrm{Int}^*$。⑦ 的 8000 步×70% 曝光＝$\kappa T\gg1$、債已還完。
$I>0$ 只改終值（$0\to\mathrm{Int}^*$）、不改賽跑形狀。

**Cor 2.2（保留、前提改標）**：guidance 無效的推導只用 $\varepsilon\approx0$（實測 ⑬）：
$\tilde v_w=v_\varnothing+w(v_a-v_\varnothing)$ 的外插項範數 $\le|w|\varepsilon$ ⇒ 偏移
$\le\frac{e^L-1}{L}|w|\varepsilon$（丙 B6 加強句）。與 A1 真偽無關 — ⑪ 的 .344 vs .336
照樣被解釋。

**留 Conj 的**：
- **Conj 2.5′（$\eta_{\rm eff}$ 落差歸因）**：$\eta_{\rm eff}/\lambda\sim3\times10^{-3}$
  （§3 反推）與幾何估計 $0.3\sim2$ 的百倍落差，主嫌＝**$z$-度量壓縮**（route 資訊在
  e_target 變異度量下權重 $\approx$ Z2 反對稱能量佔比 0.37%；γ 機制）、副嫌＝非線性讀出
  成本。未證；分辨實驗＝$I(z;a|s,g)$ 直接量測＋分窗長 Z2 能量譜。
- **Conj 2.6′（真網路 transient 幅度）**：§4.3 的 $A$（overshoot 殘留幅度）無閉式；
  非凸 basin 效應全在 toy 外（【丁】§1.9 (ii)）。

### 2.3 對 ⑬⑭ 讀數的重新拼接

$(\mathrm{Int},\varepsilon)=(0,\,1.1\%)$ 落 Def 1.4 的鎖死格**不變**；變的是格子的物理：
A1 版讀「合法全域最優、無梯度誘因離開」，修訂版讀「$\mathrm{Int}^*(0.3)\approx1\%$ 級的
合法小目標＋$12\times$ 慢時間尺度」。藥方分工不動、權重照 ⑰ 移：退火 $p$＋$L_{\rm div}$ ↑
（動力學藥直接打 (i)(ii)）、資料層破冗餘 ↓（$I$ 已 $>0$；C-iii 可讀性與 **$z$-度量權重**
（C-ii′ 後半）成為新瓶頸）。

## 3. 資訊→SNR 橋（誠實版：一座半橋＋一個匯率斷裂）

### 3.1 兩個不同的 SNR（LT-3 的教訓、先立柱）

線性高斯 toy 內有兩個座標系，**橋各自成立、互不換算**：
- **目標空間**（$I$ 住這）：$y=\Phi x+B\xi+n$ ⇒
  $$I(\tau;a|s,g)\;=\;\tfrac12\textstyle\sum_i\log\!\big(1+B_i^2/\sigma^2\big)
  \quad(\text{錨無噪、}a\text{ 可逆讀出 }\xi\text{ 時}),$$
  即 $I$ 量「route 對軌跡的效應 vs 軌跡殘噪」— **有沒有錢**。單模讀法：
  $2.5\ \mathrm{bits}=1.73\ \mathrm{nats}\Rightarrow B^2/\sigma^2\approx e^{3.46}-1\approx31$
  （或多模攤分）。hindsight 錨（$a=A(\tau)$、決定性）下 $I=H(a|s,g)$ — ⑰ 量的正是
  這個：route 條件熵 $2.5$ bits $\approx2^{2.5}\approx5.7$ 條有效分支。
- **錨/讀出空間**（$\mathrm{Int}$ 住這）：$\mathrm{Int}^*=\eta/(\eta+p\lambda)$ 只看
  「錨中新資訊變異 vs 冗餘變異」— **挖錢的成本**。【丁】Remark LT-3：$I$ 對 $\eta$
  不連續、$\mathrm{Int}^*$ 對 $\eta$ 連續 ⇒ **不存在 $I\leftrightarrow\mathrm{Int}$ 的
  函數關係**；中間隔著 enc 與讀出幾何的匯率。

### 3.2 $\eta_{\rm eff}/\lambda$ 的三個獨立估計（本橋的實質內容）

| 來源 | 讀法 | $\eta/\lambda$ 估計 |
|---|---|---|
| (a) 幾何、$\tau$/錨空間 | ⑰ 決定程度中位 $R^2\!\approx\!.749$ ⇒ $(1{-}R^2)/R^2$ | $\approx0.34$ |
| (a′) 幾何、per-成分 | ⑩ 淨方向 $R^2\!\approx\!.33$（【丁】P4 用） | $\approx2$ |
| (b) $\varepsilon$ 反推、$z$/通路空間 | ⑬ $\varepsilon_{\rm rel}=1.1\%\approx\mathrm{Int}^*(0.3)$ ⇒ $\eta/\lambda=p\varepsilon/(1{-}\varepsilon)$ | $\approx0.0033$ |
| (c) Z2 能量、$z$ 空間 | ⑰ $\|\varphi_-\|^2/\|e\|^2$ 中位 $0.37\%$ | $\approx0.0037$ |

**判讀**：(b)(c) 兩個**互相獨立**的 $z$-空間估計吻合到 10%（同為 $3{\sim}4\times10^{-3}$）、
與 (a)(a′) 的 $\tau$-空間幾何估計差 **兩個數量級** ⇒ 「$p{=}0.3$ 在 $I{\approx}2.5$ bits 下
仍壓死增益」的顯式解釋：**$\tau$ 裡的 route 錢是真的（2.5 bits、$B^2/\sigma^2\sim31$），
但經 e_target 的度量重新計價後、其變異佔比只剩 $\sim0.4\%$ ⇒
$\eta_{\rm eff}\approx0.003\lambda\ll p\lambda=0.3\lambda$，深入 crossover 尾部
$\mathrm{Int}^*\approx1\%$。** bits 不給梯度定價、變異才給（Prop 1.3 是 NLL 語意、
CFM 是 $L^2$ 語意 — 匯率斷裂的形式位置）。

**與劑量實測的一致性**：$p{=}0.3$ 全鎖（⑦⑬）＝(b) 的定義域 ✓；p=0.1「部分回」方向
（idp01 4/8 初讀、待收齊）若為真，見 §4.3 — 平衡版給不出、**transient 版給得出**，
且兩版的分辨本身就是明早的判讀樹。

### 3.3 誠實標（本橋的鬆緊）

- (b) 的 $\varepsilon_{\rm rel}\approx\mathrm{Int}^*$ 用了「f27n 的 cond 讀出 $\approx$
  $p{=}0$ 平衡讀出」的 $O(1)$ 校正假設 — toy 內 f27n（$\kappa|_{p=0}=\eta$ 龜速）實停在
  min-norm 過渡平台（【丁】LT-7），分母偏小 ⇒ (b) 偏**上**界估。
- (c) 的 $\varphi_-$ 只抓反演-反對稱那一族 route 資訊（cos +0.56、未全對齊）；中位被
  短窗壓（長窗 21.8%）⇒ (c) 偏**下**界估。兩者夾出 $[10^{-3},10^{-2}]$ 的工作區間。
- 實網路非線性、$t$-切片外推、$z_t$-leak（【丁】P3）皆未入式 — **本橋是量級論證、
  不是等式**；它的可證偽形式全部外包給 §4。

## 4. 劑量曲線預測（明早 idp01/idpxm 對決用）

### 4.1 平衡版點估（假設：各臂皆達 population 平衡＋單一 $\eta_{\rm eff}$）

取 $\eta_{\rm eff}/\lambda=0.0033\ [0.001,0.01]$（§3.2 (b)(c)＋§3.3 區間）、
$\mathrm{Int}(p)=\frac{\eta/\lambda}{\eta/\lambda+p}$：

| $p$ | $\mathrm{Int}$ 中位 | 區間 | 效用換算 $\Delta R0\approx\mathrm{Int}\times.133$ |
|---|---|---|---|
| $0$ | $\equiv1$（Def 1.4 校準端） | — | $+.133$（f27n 錨） |
| $.05$ | $.062$ | $[.020,.167]$ | $+.008$ |
| $.1$ | $.032$ | $[.010,.091]$ | $+.004$ |
| $.3$ | $.011$ | $[.003,.032]$ | $+.001$（反推錨、非預測） |

⇒ 平衡版的膽子話：**除 $p{=}0$ 外四點在效用錶（SE $\approx.025$）上全平** — idp01 的
R0on 不回 .42+、增益 $\le$ 噪音級。它回了＝平衡版當場證偽（資訊量見 4.3）。

### 4.2 比值封頂（平衡版的最尖可證偽命題）

**Prop 4.2′（劑量比值界；LT 假設＋平衡）**：對 $0<p_1<p_2$：
$$\frac{\mathrm{Int}^*(p_1)}{\mathrm{Int}^*(p_2)}=\frac{\eta+p_2\lambda}{\eta+p_1\lambda}
\;\le\;\frac{p_2}{p_1},$$
等號於 $\eta\to0$。_證_：交叉相乘、$p_1\le p_2$。∎
⇒ **$\mathrm{Int}(0.1)/\mathrm{Int}(0.3)\le3$、$\mathrm{Int}(0.05)/\mathrm{Int}(0.3)\le6$**，
與 $\eta$ 取值無關。idp01 若量到 $\mathrm{Int}(0.1)>3.3\%$ 級（相對 ⑬ 的 $1.1\%$ 超過
$3\times$）⇒ **不必等曲線收齊、平衡假設已倒** ⇒ 直接進 4.3 的 transient 枝。

### 4.3 transient 修正（「p=0.1 部分回」的相容機制；Conj 級）

有限訓練 $T$ 下觀測值＝平衡項＋overshoot 殘留（【丁】LT-6 單峰、指數尾）：
$$\mathrm{Int}_{\rm obs}(p)\;\approx\;\frac{\eta}{\eta+p\lambda}\;+\;A\,e^{-\kappa(p)T},
\qquad\kappa(p)=(1-p)(p\lambda+\eta).$$
以 $\kappa(0.3)T\approx5$（⑦「債已還完」）校準：$\kappa(0.1)T\approx5/2.33\approx2.1$、
$\kappa(0.05)T\approx5/4.4\approx1.1$ ⇒ 殘留因子 $e^{-\kappa T}$ 分別
$\approx0.7\%/12\%/32\%$ — **transient 項天然給出 $10{\sim}40\times$ 的跨 $p$ 差**、
平衡項給不出（4.2 封頂 $3{\sim}6\times$）。$A=O(1)$（相對 Int 尺度、overshoot 峰值級）
無閉式 ⇒ 此式是**形狀**預測非點估。若 idp01 部分回：用其讀數反解 $A$、再前推
$p{=}0.05$ 臂 — transient 版預測 $\mathrm{Int}_{\rm obs}(0.05)$ 可達 $0.3$ 級、
且**對訓練步數敏感**（idpxm 曝光匹配臂正好當對照：同曝光異 $T$ 分離兩項）。

### 4.4 三個可證偽形狀性質（點估失效時的退守線、兩版共有／分歧標明）

1. **單調**：$\mathrm{Int}(p)$ 嚴格降於 $p$ — 平衡項 $\partial_p<0$、transient 項
   $\kappa$ 增於 $p$（$p<\tfrac12$、$\eta$ 小）⇒ 兩版皆保。任何非單調＝兩版全倒
   （查非凸/儀器）。
2. **凸性**：平衡項 $\partial_p^2\mathrm{Int}^*=2\eta\lambda^2/(\eta+p\lambda)^3>0$、
   transient 項近似 $e^{-c\,p}$ 亦凸 ⇒ 四點應斜率遞增（降得越來越慢）：
   $\frac{\mathrm{Int}(.05)-\mathrm{Int}(.1)}{.05}\ge\frac{\mathrm{Int}(.1)-\mathrm{Int}(.3)}{.2}$。
3. **半衰位置（兩版分歧點）**：平衡版 $p_{1/2}=\eta_{\rm eff}/\lambda\approx0.003$ —
   **遠左於最小非零劑量 .05** ⇒ 觀測窗內全是深尾、$\mathrm{Int}(.05)<0.5$；
   transient 版 $p_{1/2}$ 由 $\kappa T$ 決定、可落 $[.05,.1]$ 窗內。
   ⇒ **$\mathrm{Int}(.05)$ 一點就分版**：$<0.1$＝平衡版活、$\in[0.2,0.5]$＝transient 版活、
   $>0.5$＝$\eta_{\rm eff}$ 反推整組重估（(b)(c) 吻合是巧合的機率隨之上升、回頭查 (c) 的
   $\varphi_-$/route 對齊）。

## 5. 丙七條需加假設的落實對照（【內】檔部分；§1 為落點）

| 丙條目 | 內容 | 落實 |
|---|---|---|
| B3（P1.3） | enc 充分性、否則退 $\le$ | **落**＝A3′：本檔全程上限語意 |
| B5（P2.1） | 唯一性限 $p\in(0,1)$ | **落**＝A2u；Prop 2.1′/2.3′ 皆帶 $p\in(0,1)$ |
| B7（P2.3(iii)） | 加「$\varnothing$ 支已最優」＋$\varepsilon_{\rm cond}$＋A4＋排除 adapter | **落**＝Prop 2.3′(iii) 全套限定詞收入、adapter 驅動另立 |
| B10（P3.1） | 屏障門檻 $p\lesssim\tfrac14$、$p{=}0.3$ 在範圍外 | **落（引用側）**：§2.3 引退火藥時標「屏障只保 $p\lesssim.25$ 段」；原命題本體在【內】、本檔不改上游 |
| B12（Def 3.3） | margin 絕對 vs 錨 .6046 相對、標度換算 | **標不落**：屬 $L_{\rm div}$ 實作規格、非本修訂稿範圍；轉 DESIGN 收 |
| B15（P3.7） | C-ii′：$I(z;a|s,g)>0$ | **落＋加強**＝C-ii′ 量化版（bits＋變異權重雙條款）；Prop 2.1′ 的前提即它 |
| B19（P4.2） | B2 取相鄰 gap min＋$\hat q$ 換算 | **標不落**：§4 幾何線未被 A1 事件觸動、本檔不涉；留【內】v1 修訂 |
| （小註）B2 | $\rho$ 取兩分佈、$L$ uniform-in-$t$ | **落**＝Cor 2.2 引用時按丙版常數讀 |

## 6. 分級總表（本檔新增件）

| 條目 | 級別 | 依賴 |
|---|---|---|
| Prop 2.1′ | Prop（搬運＋前提換 C-ii′） | A1′＋C-ii′(bits)＋A2、A3′ |
| Prop 2.2′ | Prop（LT 內；真網路＝量級指引） | LT-D/M、⑮ 驗訖 |
| Prop 2.3′(i)(ii) | Prop（LT 內） | 同上 |
| Prop 2.3′(iii) | Prop（丙修正版限定詞全收） | $\varnothing$ 支最優＋$\varepsilon_{\rm cond}{=}0$＋A4 |
| Remark 2.4′ | Remark（敘事升格、LT-6 背書） | — |
| Conj 2.5′（$\eta_{\rm eff}$ 歸因） | **Conj** | 分辨＝$I(z;a|s,g)$ 量測 |
| §3 橋 | Remark 級量級論證（3.2 表為實測拼接） | 線性高斯、$O(1)$ 校正 |
| §4.1 點估 | 預測（平衡假設、區間列明） | $\eta_{\rm eff}/\lambda\in[10^{-3},10^{-2}]$ |
| Prop 4.2′ 比值封頂 | Prop（一行證在檔） | LT＋平衡 |
| §4.3 transient 式 | **Conj**（形狀級；$A$ 無閉式） | LT-6＋$\kappa T$ 校準 |
| §4.4 性質 1/2 | 預測（兩版共有） | — |
| §4.4 性質 3 | 判別器（兩版分歧、$\mathrm{Int}(.05)$ 單點分版） | — |

_引用註：高斯條件 MI $=\tfrac12\log(1+\mathrm{SNR})$、DPI、Pinsker＝教科書級；
Cocos/Robinson/2310.07972＝〔轉引〕沿上游標記；LT 諸式＝【丁】已證＋⑮ 數值驗訖。
本檔未新增任何未驗 ID。_
