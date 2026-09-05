# THEORY — 三洞推進：線性 toy 閉式、(S1) 引理、Conj 7 修正（upgrades v0.1、2026-09-05）

_理論推進使魔（Fable、主人授權；三並行之「往前推導」線）。上游（唯讀）：
composition-law-draft v0.1（下稱【合】）、internalization-formal v0（下稱【內】）、FINDINGS-0905。
⛔ 分級鐵則同上游：**Prop ＝ 列明假設下完整證明在檔**、**Conj ＝ 未證**、Remark ＝ 解讀/定性。
本檔不改上游、不碰 code。_

---

## 0. 完成度總表（先讀這個）

| 件 | 內容 | 完成度 |
|---|---|---|
| 1 | 線性 adapter toy（LT）：landscape 閉式＋動力學閉式＋$p^*(\eta)$＋warm-start 判決 | **閉式**（一般 $\Sigma_x$ 多維動力學的特徵結構＝部分；四個判決不受影響，見 §1.9） |
| 2 | (S1) argmax-支撐引理 | **完成**（支撐版引理＋兩類 teacher 的分佈精確版＋反例；z-空間搬運＝顯式假設 Z-align） |
| 3 | Conj 7 的 $\delta$ 換算 | **部分**：成功率語意升 Prop（$H(\varepsilon+\sqrt{\delta/2}+\gamma)$、完整證明）；min-plus 語意卡住（卡點＝值泛函無界使 Pinsker 橋失效，見 §3.4） |

**給實驗排程的三個直接輸出**：§1.6（warm-start 判「不黏」→ 判黏協定改量斜率、勿二值讀）、
§1.8（可驗預言：散度非單調／雙 $p$ 標度／per-$t$ 遞減）、§3.5（(D2) 錶的 $\delta$ 要按語意分裂）。

---

## 1. 線性 adapter toy model（LT）

### 1.1 設定

**資料生成（LT-D）**：零均值獨立成分 $x \perp \xi \perp n \perp z_0$：
- $x = (s,g) \in \mathbb R^{d_x}$，$\mathrm{Cov}(x) = \Sigma_x \succ 0$（條件恆在場）；
- route 潛變數 $\xi \in \mathbb R^{d_a}$，$\mathrm{Cov}(\xi) = I$（軌跡裡 $(s,g)$ 之外的自由度）；
- 錨 $a = Mx + \sqrt\eta\,\xi$ — $M$ 是 $M(s,g)$ 的線性化（冗餘通路），$\eta \ge 0$ 是**資訊旋鈕**：
  $\eta = 0$ ⟺【內】A1（條件冗餘）精確成立；$\eta > 0$ ⟺ 錨攜帶 route 資訊；
- 目標 $y = \Phi x + B\xi + n$，$\mathrm{Cov}(n) = \sigma^2 I$（$B$＝route 對軌跡的真實影響）。

**模型（LT-M）**：$\hat y = W_x x + \zeta\, W_a a$，$\zeta \sim \mathrm{Bern}(1-p)$ 獨立（INTENT_DROP；
$W_x$ 兩支**共享**、$a$ 段歸零，同【內】§0 的 $L_p$ 結構）。Population loss：
$$L(W_x,W_a) = \mathbb E\,\|W_x x + \zeta W_a a - y\|^2 = (1-p)L^{(1)} + p\,L^{(0)}.$$

代入獨立性分解（各成分平方和）：
$$L^{(1)} = \|(W_x + W_aM - \Phi)\Sigma_x^{1/2}\|_F^2 + \|\sqrt\eta W_a - B\|_F^2 + \mathrm{const},\qquad
L^{(0)} = \|(W_x - \Phi)\Sigma_x^{1/2}\|_F^2 + \|B\|_F^2 + \mathrm{const}.$$

**Prop LT-0（與 CFM 的精確對應，非類比）**：設 $(x,\xi,n,z_0)$ 聯合高斯、
CFM 直線 path $z_t = (1-t)z_0 + t z_1$、$z_1 = \Phi x + B\xi + n$、$u_t = z_1 - z_0$，
速度場類＝$(z_t, x, a)$ 的仿射函數（time-embedding ⇒ 權重可逐 $t$）。則 population CFM 目標
$\mathbb E\|v(z_t,t,x,\zeta a) - u_t\|^2$ 的每個 $t$-切片是一個 LT 型二次迴歸：條件期望
$\mathbb E[u_t \mid z_t, x, a]$ 在聯合高斯下恰為仿射 ⇒ 線性類 realizable（A2 成立）、
逐 $t$ 解耦。故 LT 的全部結論對「線性速度場＋高斯資料的 CFM＋INTENT_DROP」**逐字成立**
（$z_t$ 作為額外 regressor 的效應見 Remark LT-8；主線 §1.2–1.7 先分析不含 $z_t$ 的乾淨切片，
對應 $t \to 0$ 端——leak 最弱、$a$-通路價值最大的那端）。
_證_：高斯條件期望的仿射性＋二次 loss 的投影唯一性；drop 的兩支結構逐字同【內】$L_p$。∎

### 1.2 Landscape 閉式：drop ＝ 作用在冗餘子空間的 ridge

**Prop LT-1（population 最優、閉式）**：記 $C \equiv M\Sigma_x M^\top \succeq 0$（**冗餘變異算子**：
錨座標裡由 $(s,g)$ 可預測的變異）。對 $p \in [0,1)$，$L$ 是凸二次型，其最優解滿足：
$$\boxed{\;W_a^*(p,\eta) = \sqrt\eta\, B\,(p\,C + \eta I)^{-1}\;},\qquad
W_x^* = \Phi - (1-p)\,W_a^* M ,$$
且在 $a$ 的變異支撐上唯一（$C$ 的零空間方向且 $\eta=0$ 時是無害 gauge 自由度——那些方向
$W_a$ 作用在 a.s. 為 0 的輸入上）。

_證明_：一階條件 $\nabla_{W_x}L = 2[(W_x - \Phi) + (1-p)W_aM]\Sigma_x = 0$ 給第二式。代入
$\nabla_{W_a}L = 2(1-p)[(W_x + W_aM - \Phi)\Sigma_x M^\top + \eta W_a - \sqrt\eta B] = 0$：
括號第一項變 $p\,W_aM\Sigma_xM^\top$，得 $W_a(pC + \eta I) = \sqrt\eta B$。唯一性：對 $(W_x,W_a)$
的 Hessian 之 Schur 補（消 $W_x$）為 $(1-p)(pC + \eta I) \otimes I \succ 0$ 於 $p>0$ 或 $\eta>0$
（$C \succ 0$ 時）；凸性由 loss 為期望平方。∎

**機制一句話**：對 $W_x$ 消元後，$a$-通路的有效目標函數是
$$\tilde L(W_a) = p\,\|W_aM\Sigma_x^{1/2}\|_F^2 \;+\; \|\sqrt\eta W_a - B\|_F^2 \;(\text{up to } (1-p) \text{ 因子與常數}),$$
即 **INTENT_DROP 在解析上就是一個係數 $p$、只罰「冗餘讀出」$W_aM$ 的 ridge**——
「兩支必須一致」的不變性壓力（⑫ 機制 2）的線性精確形。它與資訊增益（$\eta$ 項）打對台，
勝負由下面的比值決定。

### 1.3 內化率與 $p^*(\eta)$：閉式，以及 Conj 2.6 的變數修正

沿 $C$ 的特徵分解 $C = \sum_i \lambda_i u_iu_i^\top$，keep 支經 $a$-通路實得的 route 讀出
（相對真值 $B$）逐模為：

**Cor LT-2（內化率閉式）**：
$$\mathrm{Int}_i^*(p,\eta) \equiv \frac{(\sqrt\eta\,W_a^*)u_i}{Bu_i} = \frac{\eta}{\eta + p\,\lambda_i}\;\in(0,1] ,$$
單調減於 $p$、增於 $\eta$。定義 $\rho$-鎖死為 $\mathrm{Int}_i^* \le \rho$，則
$$\boxed{\;p^*_i(\eta;\rho) = \frac{1-\rho}{\rho}\cdot\frac{\eta}{\lambda_i}\;}\qquad
(\text{半內化點 } \rho = \tfrac12:\ p^*_i = \eta/\lambda_i).$$

**判決（對 Conj 2.6）**：population 層**沒有 sharp 相變**（凸問題、解對 $(p,\eta)$ 解析）——
Conj 2.6 的「臨界值」在 population 極限的正確形是上式的 **crossover 尺度**：
$p \gg \eta/\lambda_i$ ⇒ 該模鎖死（$\mathrm{Int}\to 0$）、$p \ll \eta/\lambda_i$ ⇒ 內化，
crossover 寬約一個 decade。sharp 化只能來自 toy 之外的機制（有限容量非凸、有限樣本、
離散化）——把這句寫回 Conj 2.6 的升級註。

**Remark LT-3（概念修正：驅動變數是錨 SNR，不是互資訊）**：線性高斯下
$I_{\mathrm{data}}(y;a\mid x)$ 對 $\eta$ **不連續**——$\eta>0$ 時 $a$ 給定 $x$ 是 $\xi$ 的確定可逆
函數 ⇒ $I = \tfrac12\log\det(I+\sigma^{-2}BB^\top)$ 與 $\eta$ 無關；$\eta=0$ 時 $I=0$。但
$\mathrm{Int}^*$ 對 $\eta$ 連續。⇒ **鎖死的連續控制參數是錨的訊噪比 $\eta/\lambda_i$**
（新資訊變異 vs 冗餘變異，per 方向），不是 $I_{\mathrm{data}}$：MI 說「有沒有錢」，SNR 說
「挖錢要付多大的冗餘讀出成本」——線性讀出要放大 $1/\sqrt\eta$ 倍才能取出 $\xi$，而放大
同時放大被 ridge 罰的 $W_aM$。⇒ Conj 2.6 的參數表 $p^*(I_{\mathrm{data}},\text{容量},\text{步數})$
建議改為 $p^*(\mathrm{SNR}, \text{容量}, \text{步數})$；有限樣本下 MI 會部分回歸
（樣本噪音使「除回去」不再免費），此為 toy 外註。

**對 maze 與破冗餘設計的直接含義**：maze 現況＝$\eta \approx 0$（route 幾乎由 $(s,g)$ 決定）
⇒ $p^* \approx 0$ ⇒ $p = 0.3$ 深入鎖死區——⑦⑬ 的實測（增益沒長出來、散度 1.1%）是
此公式的預期。C-iii／teleport 構造要買到內化，需把 $\eta$ 抬到 $\gtrsim p\lambda$ 級：
**「破冗餘要破多少」第一個定量指引＝新資訊變異要與 $p\times$冗餘變異同級**。
操作化：$\lambda$ 譜可由「錨嵌入對 $(s,g)$ 的線性可預測變異」估（⑩ 的 anc-$R^2$ 探針正是
座標版：$\eta/\lambda \approx (1-R^2)/R^2$ per 成分）；但見 §1.8 預言 P4 的 caveat。

### 1.4 梯度流動力學：閉式與「賽跑」的嚴格化

梯度流（時間吸收因子 2）：
$$\dot W_x = -[(W_x{-}\Phi) + (1{-}p)W_aM]\Sigma_x,\qquad
\dot W_a = -(1{-}p)[(W_x{+}W_aM{-}\Phi)\Sigma_xM^\top + \eta W_a - \sqrt\eta B].$$

**Prop LT-4（全域收斂）**：$L$ 凸（Prop LT-1）⇒ 梯度流從任意初值收斂到最優集；
$p\in(0,1)$ 或 $\eta>0$ 時（$C\succ0$）最優唯一 ⇒ **初值不改變終點、只改變路徑與時程**。∎

這一行已含 warm-start 判決的骨架；速率見下。取標量情形把結構看穿
（$d_x{=}d_a{=}d_y{=}1$、$\Sigma_x{=}\sigma_x^2$、$M{=}m$、$C = \lambda = m^2\sigma_x^2$；
多維見 §1.9）。以 $(g, w_a) \equiv (w_x - \phi,\ w_a)$ 為座標，$\dot v = -(Av - c)$：
$$A = \begin{pmatrix} \sigma_x^2 & (1-p)\,m\sigma_x^2 \\ (1-p)\,m\sigma_x^2 & (1-p)(m^2\sigma_x^2 + \eta) \end{pmatrix},\qquad
\det A = (1-p)\,\sigma_x^2\,[\,\eta + p\,m^2\sigma_x^2\,] .$$

**Prop LT-5（速率閉式與 two-timescale）**：$A$ 對稱正定（$p\in(0,1)$ 或 $\eta>0$），特徵值
$$\mu_\pm = \tfrac12\Big[\operatorname{tr}A \pm \sqrt{\operatorname{tr}A^2 - 4\det A}\Big],\qquad
\operatorname{tr}A = \sigma_x^2 + (1-p)(m^2\sigma_x^2+\eta).$$
慢模＝$a$-通路淨調整，$\mu_- \approx \dfrac{\det A}{\operatorname{tr}A}
= \dfrac{(1-p)(\eta + p\lambda)}{1 + (1-p)(\lambda+\eta)/\sigma_x^2\cdot\sigma_x^{-0}}\Big|_{\sigma_x=1}$；
在 $W_x$ 快（$\sigma_x^2 \gg (1-p)(\lambda+\eta)$）的 slaved 極限精確為
$$\boxed{\;\kappa = (1-p)(p\lambda + \eta)\;}\qquad(\text{slaved 有效動力學 }
\dot W_a = -(1-p)[\,W_a(pC+\eta I) - \sqrt\eta B\,]).$$
_證_：直接對角化；slaved 版由 $\Delta_0 \equiv W_x - \Phi \approx -(1-p)W_aM$ 代入
$\dot W_a$（quasi-static 消去，誤差 $O((1-p)\lambda/\sigma_x^2)$）。∎

**Prop LT-6（賽跑＝overshoot 定理；Remark 2.4 的判決）**：cold init（$w_a(0)=0$、
$g(0)=-\phi$）、$\eta = 0$、$p\in(0,1)$、$m\phi \ne 0$。則
$$w_a(t) = \alpha\big(e^{-\mu_- t} - e^{-\mu_+ t}\big),\qquad
\alpha(\mu_+{-}\mu_-) = (1-p)\,m\,\sigma_x^2\,\phi \ne 0,$$
即 $a$-通路**嚴格先長後塌**（單峰、峰時 $t^* = \ln(\mu_+/\mu_-)/(\mu_+{-}\mu_-)$、終值 0）。
_證_：二維對稱線性系統、初值不在特徵向量上（off-diagonal $\ne 0$）⇒ 兩模疊加；$t{=}0$
兩係數相消、初斜率 $\dot w_a(0) = -(1-p)m\sigma_x^2\,g(0) > 0$。∎

**判決（對【內】Remark 2.4）**：賽跑的線性化敘事**嚴格成立且加強**——
(i) $a$-通路的驅動恰為 residual 的 $x$-可預測分量 $-(1-p)\,\Delta_0\Sigma_x M^\top$
（Remark 2.4 的 $\mathbb E[a\otimes\delta]$ 只剩 $(s,g)$-分量，逐字實現）；
(ii) $\varnothing$ 支收斂（$\Delta_0 \to$ 平衡）使該驅動枯竭，殘餘為 ridge 衰減——但 toy 補上
敘事沒有的一段：**早期 $a$-通路確實先長**（借 $\varnothing$ 支未收斂的 residual 的錢），
鎖死是「還債」過程，不是從未生長。⑦ 的 8000 步×70% 曝光＝$\kappa T\gg1$、債已還完。
另一致性檢查：$\{W_a = 0\}$ 在 $\varnothing$ 支未最優時**不是**不變集
（$\dot w_a|_{w_a=0} = -(1-p)m\sigma_x^2 g \ne 0$）——與【內】Prop 2.3(iii) 的前提
「$\varnothing$ 支已最優」精確吻合；前提不滿足時 $\varepsilon = 0$ 不自我維持，正是 overshoot。

### 1.5 $p=0$ 簡併谷與 f27n 的第三種解釋

**Prop LT-7（min-norm 偏置）**：$p = 0$、$\eta = 0$ 時最優集是簡併谷
$\{(W_x,W_a): W_x + W_aM = \Phi\}$（f27n 的家）。zero-init 梯度流收斂到谷上的 min-norm 點：
標量情形 $w_a(\infty) = \dfrac{m\phi}{1+m^2\sigma_x^{2}}\Big|_{\sigma_x=1} \ne 0$——
**功勞按相關度分配給 $a$-通路**。
_證_：$p{=}0,\eta{=}0$ 的 loss 只約束 $W_x + W_aM$；梯度流保持在 row-space
（沿 $(1, m)$ 方向移動）⇒ 終點＝初值在谷上的正交投影。∎

**Remark**：f27n 的 $+.133$（$p{=}0$ 時 $a$-通路長成）在線性層**不需要**「有限容量計算捷徑」
假設（【內】Remark 2.5）——zero-init GD 的隱式偏置就足夠。兩機制相容不互斥；toy 給的是
更便宜的下界解釋。$p > 0$ 打破簡併、把唯一最優釘到 $W_a = 0$（$\eta{=}0$）——
**任意小的 $p$ 在 population＋無限時間極限都足以鎖死**（$p^*(\eta{=}0) = 0^+$），
「$p$ 臨界值」的實質內容全在有限時間／有限容量修正裡（§1.3 判決的另一面）。

### 1.6 Warm-start 黏性（Conj 3.2）的 toy 判決：不黏

**Prop LT-8（衰減閉式）**：warm-start 初值＝f27n 解（Prop LT-7：$w_a(0) = m\phi/(1{+}m^2)$、
谷上點），續訓 $p \in (0,1)$、$\eta = 0$。由 Prop LT-4 終點唯一（$w_a^\infty = 0$）且系統線性：
$$\|W_a(t)\| \le \|W_a(0)\|\,\mathrm{poly}\cdot e^{-\mu_- t},\qquad
\mu_-\ \text{如 Prop LT-5（slaved: } \kappa = (1-p)\,p\,\lambda\text{）},$$
即 **$a$-通路指數衰減、必塌回、不存在第二吸子**。∎

**判決（對 Conj 3.2）**：在線性 toy 的範圍內，Conj 3.2 的「長時黏性未保證」**強化為
「必不黏」**——【內】Prop 3.1 的屏障擋不住的「同步塌修」路徑，正是這裡的梯度流本身
（凸 ⇒ 損失單調下降直達 invariant 最優，中途無屏障）。真系統要黏，只能靠 toy 沒有的
機制：非凸 basin 多重性、SGD 噪音誘導的平坦度選擇、有限容量。⇒ 明天的 warm-start
實驗**判的不是「會不會衰減」而是「衰減率是否為零」**——這改變讀錶方式：

- **量級警示**：$p{=}0.3$、$\lambda{=}\sigma_x{=}1$ ⇒ $\mu_+ {=} 1.566$、$\mu_- {=} 0.134$：
  **$a$-通路衰減比主通路收斂慢 ~12 倍**（slaved 值 $0.21$ 同量級）。⇒ loss 曲線早已平、
  散度仍在掉——「**loss plateau ≠ 散度動力學結束**」。半天測若只看端點二值（黏/塌）
  可能把慢衰減誤讀成黏住。
- **協定建議**（給明天排程）：(i) 續訓全程存散度探針（⑬ 儀器）時間序列，對
  $\log\varepsilon_{\mathrm{cond}}(t)$ 擬合斜率——斜率 $\approx 0$（置信區間含 0）才可宣「黏」，
  顯著負＝塌、只是慢；(ii) 若跑兩個 $p$ 臂，toy 預測斜率比 $\propto p(1-p)$
  （$p{=}0.3$ vs $p{=}0.1$ ⇒ $0.21/0.09 \approx 2.3\times$）——標度對上＝toy capture 真系統、
  對不上＝非凸效應在場，**兩種結果都有資訊量**；(iii) 若實測斜率確為 0：那是 toy 被證偽
  的乾淨訊號 ⇒ basin 機制實錘 ⇒ 退火可行性反而回升。

### 1.7 共享 $W_x$ 的反向代價（順手的小結果）

由 Prop LT-1：$\eta > 0$ 時 $W_x^* = \Phi - (1-p)W_a^*M \ne \Phi$ ⇒ **$\varnothing$ 支被 keep 支
拖出自身最優**，偏置 $\propto (1-p)\|W_a^*M\|$——內化越強（$W_a^*$ 大）、drop 越少（$p$ 小），
免查腿越偏。$\eta = 0$ 時 $W_a^*{=}0$ ⇒ 無代價（鎖死的解裡共享是免費的）。
⇒ 對 Def 1.4 的預言：**$U_\varnothing$ 讀數在強內化 regime 有系統性下偏**（相對「專訓 $\varnothing$」
的參考），破冗餘實驗成功時要預期 zero-模式分數略降、且降幅隨 $p$ 增而縮——
別把它讀成 regression。

### 1.8 可驗預言清單（全部免訓練或已有儀器）

- **P1（非單調散度）**：cold-start＋$p>0$ 的訓練，分支散度 $\varepsilon(t)$ 先升後降
  （Prop LT-6）。若 idp 批有中途 ckpt，直接掃即可驗。
- **P2（warm-start 斜率標度）**：$\log\varepsilon$ 衰減斜率 $\propto p(1-p)$（§1.6(ii)）。
- **P3（per-$t$ 散度遞減）**：$z_t$-leak（Remark LT-9）⇒ $a$-通路邊際價值隨 $t\to1$ 消失
  （$t{\to}1$ 時 $\mathbb E[u_t|z_t]$ 已幾乎確定 $u_t$）⇒ 散度 $d(t)$ 集中在小 $t$。
  ⑬ 探針若能出 per-$t$ 讀數可驗。
- **P4（$R^2$–鎖死張力）**：⑩ 量得淨方向成分 $R^2 \approx .33$ ⇒ $\eta/\lambda \approx 2$ ⇒
  per LT 該成分 $p^* = O(1) > 0.3$、**不該被鎖死**；但 ⑬ 量到全鎖（1.1%）。兩解：
  (a) 該成分對 residual 的 $B$-係數 $\approx 0$（方向資訊對重建計畫無增量價值——$B{=}0$ 時
  $W_a^*{=}0$ 與 $p$ 無關，「鎖死」與「無用」在散度錶上簡併）；(b) toy 失效。
  分辨法：查 route 條件下的 plan 變異是否真含方向自由度（＝量 $B$ 的代理）。
  這是 toy 給的第一個**可證偽張力點**，不是裝飾。

### 1.9 誠實邊界（LT）

(i) 一般 $\Sigma_x$、$M$ 非對齊時動力學不逐模解耦——但 Prop LT-1（閉式最優）、LT-4
（全域收斂、唯一性）、LT-8（不黏）**不依賴對角化**，只有速率的逐模顯式式要
$[\Sigma_x, M^\top M]$ 可交換類假設（或用 $\mu_{\min}(A)$ 的界代替）。
(ii) toy 是 population＋凸：有限容量、非凸、SGD 噪音全在外——所以 §1.6 的判決話術是
「線性層必不黏」，不是「真系統必不黏」；toy 的價值在把「黏」的舉證責任推到非凸機制上，
並給出可證偽的斜率標度。
(iii) $z_t$-leak 只做了定性（P3）；含 $z_t$ 的逐 $t$ 閉式是機械延伸（多一個 regressor 的
同型計算），未展開。
(iv) 步數↔梯度流時間的換算依 lr schedule，§1.6 的「12×」是相對量（robust），
絕對步數預測不可靠。

**升級對接表**：【內】Remark 2.4 → Prop（LT-6）；Conj 2.6 → crossover 閉式＋變數修正
（Cor LT-2、Remark LT-3；sharp 相變部分仍 Conj、歸有限容量）；Conj 3.2 → toy 層判決
（LT-8：不黏）＋實驗協定；Remark 2.5 → 補充機制（LT-7 min-norm 偏置）；
Prop 2.3(iii) 前提的必要性 → LT-6 一致性檢查。

---

## 2. (S1) argmax-支撐小引理（【合】§5.2 預告的那塊）

### 2.1 設定與兩個定義

(A1) 確定性環境，狀態圖 $G_{\mathcal E}$、最短路距離 $d$。$\mathrm{SP}(s,g)$＝$s{\to}g$ 最短路
（狀態序列）之集。teacher $T(\cdot\,|\,s,g)$＝支撐 $\subseteq \mathrm{SP}(s,g)$ 的計畫分佈
（最短路 teacher）。拼接 $\tau_1 \diamond \tau_2$＝首尾狀態相同時的串接。

**Def 2.1（子段封閉 / T-cons）**：路徑系統 $\{\tau(s,g)\}$（每對一條）**子段封閉**，若
$\forall\, m \in \tau(s,g)$：$\tau(s,m) = \tau(s,g)[s{..}m]$ 且 $\tau(m,g) = \tau(s,g)[m{..}g]$。

**Def 2.2（拼接相容的分段協定）**：$\sigma: \tau \mapsto m \in \tau$（從計畫讀出中繼的規則）
**拼接相容**，若 $\sigma$ 的取值由前段本身決定（例：固定深度 $\sigma_k(\tau) = \tau_k$；
或「首次進入區域 $R$ 的狀態」）。

### 2.2 支撐分解（引理本體）

**Lemma 2.3（值與支撐分解）**：(A1) 下，對任意 $m$：
(i) $d(s,m) + d(m,g) \ge d(s,g)$（三角不等式）；若 $M$ 含某條 $s{\to}g$ 最短路上至少一點
（(D1) 精確版），則 $d(s,g) = \min_{m\in M}[d(s,m)+d(m,g)]$。
(ii) 對滿足 $d(s,m)+d(m,g) = d(s,g)$ 的 $m$（最優中繼）：
$$\{\tau \in \mathrm{SP}(s,g): m \in \tau\} \;=\; \mathrm{SP}(s,m) \diamond \mathrm{SP}(m,g),$$
且此拼接是雙射（一條經 $m$ 的最短路唯一分解為前後段）。

_證明_：(i) 前半＝任兩段可拼成一條 $s{\to}g$ 可行路（**確定性在此**：狀態序列合法性只看
相鄰邊存在，拼接處狀態相同即合法——stochastic 下「軌跡」不可自由拼接，此步失效，
與【合】§5.1 的 open 邊界一致）；後半＝取最優路上的 $m$ 得等號。
(ii) $\subseteq$：最優子結構——若 $\tau \in \mathrm{SP}(s,g)$ 經 $m$ 而前段
$|\tau[s..m]| > d(s,m)$，以更短前段替換得更短 $s{\to}g$ 路，矛盾；後段同理，
又 $|\tau[s..m]| + |\tau[m..g]| = d(s,g) = d(s,m)+d(m,g)$ 配三角不等式逼出兩段各自最優。
$\supseteq$：拼接可行（同 (i)）且長度 $= d(s,m)+d(m,g) = d(s,g)$ ⇒ 最優。
雙射：$m$ 在序列裡的首次出現切點唯一（最短路無重複狀態）。∎

### 2.3 (S1) 精確成立的兩類 teacher

**Prop 2.4（點質量一致 teacher）**：teacher 為確定性路徑系統 $\{\tau(s,g)\}$ 且**子段封閉**
（Def 2.1）。則對任意拼接相容協定 $\sigma$，(S1) 以下列形式**精確成立**：
$$T(\tau\,|\,s,g) = \sum_m w^*(m|s,g)\; T(\tau[s..m]\,|\,s,m)\; T(\tau[m..g]\,|\,m,g),\qquad
w^*(\cdot|s,g) = \delta_{\sigma(\tau(s,g))},$$
兩段給定 $m$ 條件獨立（點質量的乘積）。
_證明_：子段封閉 ⇒ $T(\cdot|s,m)$、$T(\cdot|m,g)$ 恰是 $\tau(s,g)$ 兩段的點質量；
$\sigma$ 拼接相容 ⇒ $m$ 由 $\tau$ 良定義。代入驗等式。∎

**Lemma 2.5（子段封閉系統存在：lex-min 構造）**：取狀態集上任意全序 $\prec$，令
$\tau(s,g)$＝$\mathrm{SP}(s,g)$ 中頂點序列字典序最小者。則 $\{\tau(s,g)\}$ 子段封閉。
_證明_：設 $m \in \tau \equiv \tau(s,g)$ 而前段 $\tau[s..m]$ 非 $s{\to}m$ 的 lex-min 最短路，
取更小的 $\sigma'$；則 $\sigma' \diamond \tau[m..g]$ 由 Lemma 2.3(ii) 是 $s{\to}g$ 最短路且
字典序嚴格更小（在前段就分勝負），與 $\tau$ 的 lex-min 矛盾。後段：
$\tau[s..m] \diamond \sigma''$ 與 $\tau$ 前綴相同、於後段更小，同矛盾。∎

**Prop 2.6（均勻最短路 teacher）**：teacher $= \mathrm{Unif}(\mathrm{SP}(s,g))$、協定
$\sigma_k$（固定深度 $k$）。則 (S1) 精確成立，且
$$w^*(m|s,g) = \frac{|\mathrm{SP}(s,m)|\cdot|\mathrm{SP}(m,g)|}{|\mathrm{SP}(s,g)|}
\quad(\text{支撐在 } d(s,m){=}k,\ d(m,g){=}d(s,g){-}k),$$
給定 $m$ 時前後段獨立、各自均勻。
_證明_：深度 $k$ 的切點把 $\mathrm{SP}(s,g)$ 劃分（每條最短路在第 $k$ 步恰經一個 $m$；
$d(s,m)=k$ 因子段最優）；由 Lemma 2.3(ii) 每個 cell $=$ 乘積集且計數
$|\mathrm{SP}(s,m)||\mathrm{SP}(m,g)|$；均勻測度限制在乘積集上＝乘積均勻 ⇒ 條件獨立；
權重＝cell 質量比，歸一化由劃分。∎

### 2.4 邊界：反例與搬運假設

- **Per-query BFS 不一致（反例級 Remark）**：對每對 $(s,g)$ 各自跑 BFS（各自 tie-break）
  的 teacher **不保證**子段封閉：$\tau(s,g)$ 由以 $g$ 為根的樹給出時後綴一致、但
  $\tau(s,m)$ 來自以 $m$ 為根的另一棵樹，前綴可不同（tie 處選邊不同即得反例）。
  ⇒ (S1) 對這種 teacher 的前段因子**不成立**（分佈層面）。修法＝改用 Lemma 2.5 的
  全域一致 tie-break，或改 hindsight。
- **Hindsight 切段**：訓練樣本 $(s,m,g)$ 取自同一條資料軌跡 $\tau$ 時，「切段的切段＝切段」
  ⇒ 樣本層面自動子段封閉——**這正是真管線的錨生成方式**，(S1) 的訓練分佈版因此有
  結構根據；但資料軌跡**次優**時支撐溢出 $\mathrm{SP}$，Prop 2.4 的前提破——次優 teacher
  的 (S1) 仍是假設（與【合】§5.2 的定位一致；本引理只收「最短路 teacher」這格）。
- **(Z-align) z-空間搬運假設**：以上全在計畫（狀態序列）空間。搬到 latent $z$ 需
  「$z$ 的前後段分別編碼 $\tau$ 的前後段」（encoder 段落對齊）——架構性質、非自動，
  在【合】§1.1 的 tokenized 計畫下合理但**未驗**；列為 (S1) 分佈版剩餘的唯一結構假設。

**完成度**：引理本體＋兩類 teacher 分佈版＝完整證明；超出【合】§5.2 預告的部分＝
Prop 2.6（均勻 teacher 的權重閉式，順手給了 $w^*$ 的組合意義：路徑計數比）。

---

## 3. Conj 7 的 $\delta$ 換算修正

### 3.1 語意分裂：$\delta$ 進值差的兩條橋

(D2) 給的是 KL（分佈層）。值層有兩個語意（【合】§1.3 對應表），橋不同：
- **成功率語意**（log-semiring / 期望，我們 eval 的）：值 $= \mathbb E[\text{succ}] \in [0,1]$
  有界 ⇒ Pinsker 橋可用，付 $\sqrt{\delta/2}$。
- **min-plus 語意**（步數/代價）：值泛函無界 ⇒ Pinsker 橋失效（§3.4）。

### 3.2 單步分解的 $\sqrt\delta$ 界（可完整證）

**Prop 3.1（兩段拼接、成功率語意）**：設 $P = p^*(\cdot\,|\,s,g,m)$（真條件 joint）、
$Q = p^*(z_{\mathrm{pre}}|s,m)\otimes p^*(z_{\mathrm{post}}|m,g)$，(D2)：$\mathrm{KL}(P\|Q)\le\delta$。
設分段成功泛函 $F(z) = \mathbf 1[\text{前段達 } B(m)]\cdot\mathbf 1[\text{後段達 } B(g)] \in [0,1]$，
且整段成功與分段成功之差由 (D3) 控：$\mathbb E_P|\mathrm{succ} - F| \le \gamma$
（拼接處違法／語意錯位事件）。則
$$\big|\,\mathbb E_P[\mathrm{succ}] - V_{\mathrm{pre}}\cdot V_{\mathrm{post}}\big|
\;\le\; \sqrt{\delta/2} + \gamma,\qquad
V_{\mathrm{pre}} = \mathbb E[\mathrm{succ}_{s\to m}],\ V_{\mathrm{post}} = \mathbb E[\mathrm{succ}_{m\to g}].$$
_證明_：$|\mathbb E_P F - \mathbb E_Q F| \le \mathrm{TV}(P,Q) \le \sqrt{\delta/2}$（$F$ 有界、
Pinsker）；$\mathbb E_Q F = V_{\mathrm{pre}}V_{\mathrm{post}}$（$Q$ 下兩段獨立、$F$ 乘積形）；
三角不等式加 $\gamma$ 項。∎

### 3.3 沿 horizon 疊加（成功率語意：可證版）

**Prop 3.2（$H$ 層合成界）**：設字典 DP 以【合】R4 兩形之一遞迴（單步 peel 或 squaring），
且在 DP 實際訪問的三元組集合上 uniform 地有：(D1) 覆蓋損 $\le\varepsilon$（值域尺度）、
(D2) $\mathrm{KL}\le\delta$、(D3) $\le\gamma$，混合權重歸一、值 $\in[0,1]$。則
$$\big|\hat V(s,g) - V^*(s,g)\big| \;\le\; (H-1)\big(\varepsilon + \sqrt{\delta/2} + \gamma\big).$$
_證明_：記 $E_h = \sup|\hat V_h - V_h|$（$h$ 段值）。squaring 形：
$\hat V_{2h} = \bigoplus_m \hat w\,\hat V_h(s,m)\hat V_h(m,g)$。
(a) 合成誤差：$|\sum \hat w\,\hat V\hat V - \sum \hat w\,VV| \le 2E_h$（值 $\le 1$、權重歸一、
乘積對每因子 1-Lipschitz；$\oplus \in \{\text{混合}, \max\}$ 皆 sup-norm nonexpansive，
【合】引理 2）。(b) 分解誤差：$|\sum \hat w\,VV - V_{2h}| \le \sqrt{\delta/2} + \gamma + \varepsilon$
（Prop 3.1 逐項＋(D1) 收最優中繼不在 $M$ 的損）。⇒ $E_{2h} \le 2E_h + c$、
$c = \varepsilon+\sqrt{\delta/2}+\gamma$、$E_1 = 0$ ⇒ $E_H \le (H-1)c$。
單步 peel 形：$E_{h+1} \le E_h + c$，同界。∎

**⇒ Conj 7 修正後的形（成功率語意、升 Prop）**：
$$\big|\hat V - V^*\big| \;\le\; H\big(\varepsilon + \sqrt{\delta/2} + \gamma\big)
\qquad(\text{原 } CH(\varepsilon+\delta+\gamma) \text{ 之 } \delta \to \sqrt{\delta/2},\ C = 1).$$

### 3.4 Min-plus 語意：卡點（誠實標）

值 $=$ 步數（或 $-\log p$）時泛函無界，Pinsker 橋斷。兩條殘路、各缺一塊：
- **(a) 截斷**：長度 $\le H$ a.s. ⇒ $\|\cdot\|_\infty \le H$ ⇒ 每層付 $H\sqrt{\delta/2}$ ⇒ 總界
  $H^2\sqrt{\delta/2}$——**horizon 疊加從線性劣化為二次**；線性恢復不了，除非
- **(b) 強化 (D2)**：改 sup-log-ratio $\delta_\infty \equiv \sup|\log(P/Q)| \le \delta_\infty$
  ⇒ min-plus 值差每層 $\le \delta_\infty$ 直接進加法 ⇒ 總界 $H\delta_\infty$ 線性——但
  $\delta_\infty$ 是**另一個錶**（尾部敏感、比 KL 難估），pairability 代理量不到它。
- $\gamma$ 在 min-plus 的角色也變質：違法拼接＝「宣稱代價不可實現」⇒ 值差可無界，
  需另設修復假設（violation 可以 $\le r$ 步修復 ⇒ 膨脹 $\gamma r$）——**未推**。

**卡點一句話**：min-plus 版缺的不是不等式技巧，是「(D2) 該量什麼」——KL 對 min-plus
語意是錯的度量。此件標 **部分**。

### 3.5 與開環界的分工（「接橋還缺什麼」的精確化）

部署讀數到 $V^*$ 的完整鏈是三段、誤差源互異：
$$\underbrace{|\hat V - V^*|}_{\text{字典 DP 誤差：本節 Prop 3.2}}\;+\;
\underbrace{|\mathbb E_{z\sim p_\theta}[\mathrm{succ}] - \hat V|}_{\text{sampler 忠實度：(S2) 鬆動、無錶}}\;+\;
\underbrace{|\text{閉環部署} - \text{開環期望}|}_{\text{執行 gap：arXiv 2605.08732 的位置}}.$$
【合】Conj 7 的「接橋未搭」精確內容＝**中段**：flow 抽樣分佈相對 (CL) 定點的偏差
（有限樣本 NLL＋flow 訓練誤差 → 值偏差的換算，【合】§5.4 同源）目前無錶無界；
2605.08732 只旁證第三段。⇒ 升級路徑：中段需要「NLL gap → TV → 值差」的同型 Pinsker 鏈
（技術同 Prop 3.1，輸入換成訓練誤差——可做，材料未在本檔範圍）。

**對 (D2) 錶規格的輸出（B 階段直接可用）**：pairability 錶若以 KL/NLL 形估 $\delta$，
它背書的是**成功率語意**的 $H\sqrt{\delta/2}$ 界；若 paper 要對步數/代價（min-plus）陳述
近似保證，需另立 $\delta_\infty$ 錶或接受 $H^2$ 形——收表協定寫清楚，避免一錶兩用。

---

## 4. 總結（三件、一行各）

1. **LT toy＝閉式**：$W_a^* = \sqrt\eta B(pC+\eta I)^{-1}$——drop 是只罰冗餘讀出的 ridge；
   $\mathrm{Int} = \eta/(\eta+p\lambda)$、$p^* = \frac{1-\rho}{\rho}\eta/\lambda$（無 sharp 相變，
   crossover）；賽跑嚴格成立且升級為 overshoot 定理；**warm-start 判不黏**（率
   $\kappa = (1-p)(p\lambda+\eta)$、比主收斂慢 ~12×）⇒ 判黏改量斜率。
2. **(S1) 引理＝完成**：子段封閉（lex-min 構造存在、hindsight 自動滿足、per-query BFS
   反例）⇒ 點質量版與均勻版 (S1) 精確成立、$w^*$ 有路徑計數閉式；殘餘假設只剩 Z-align。
3. **Conj 7＝部分**：成功率語意升 Prop：$H(\varepsilon+\sqrt{\delta/2}+\gamma)$；min-plus 卡在
   「KL 是錯的度量」（要 $\delta_\infty$ 錶或付 $H^2$）；接橋缺口釘到 sampler 忠實度中段。

_引用註：Pinsker／Grönwall／Tarski／線性 ODE＝教科書級；arXiv ID 皆轉引自兩份上游
（2210.02747、2207.12598、2605.08732 等、上游已驗）。LT 動力學的深度線性網路近親
（Saxe et al. 2013 類）本檔未驗 ID、未引用承重。_
