# THEORY — 內化的形式化與條件通路鎖死（理論使魔 v0、交 Rei 磨嚴）

_理論使魔（Fable、主人授權；表徵幾何線）2026-09-05。對應素材：FINDINGS-0905 ⑦⑩⑪⑫⑬⑭、
NOTE-composition-law §二§四、DESIGN-route1-losses v2。⛔ 分級鐵則：**Prop = 在列明假設下可完整證**、
**Conj = 未證**、Remark = 敘事/解讀。引用標注：〔轉引〕= docs 已引未讀原件；〔驗〕= 本次上網驗過 ID。_

## 0. Setup 與假設

條件 $c=(s,g,a)$；$a$ = intent 錨（訓練時可查知識源 $O$ 的摘要 $A(\tau,O)$）、$\varnothing$ = intent 段歸零。
Flow-matching（CFM〔驗 2210.02747〕）with INTENT_DROP $p$（intent 段獨立歸零、$(s,g)$ 恆在場）：

$$L_p(\theta)=\mathbb{E}_{(\tau,a),t,z_0}\Big[(1{-}p)\,\|v_\theta(z_t,t,s,g,a)-u_t\|^2+p\,\|v_\theta(z_t,t,s,g,\varnothing)-u_t\|^2\Big]$$

$p_\theta(\cdot|s,g,\cdot)$ = 對應 ODE 的終端分佈。分支散度 $\varepsilon(\theta)\equiv\sup_{z,t,s,g,a}\|v_\theta(\cdot|s,g,a)-v_\theta(\cdot|s,g,\varnothing)\|$（期望版 $\bar\varepsilon$ 同理）。

**假設（命題各自引用）**
- **A1（條件冗餘）**：$I_{\text{data}}(\tau;a\mid s,g)=0$，即 $\tau\perp a\mid(s,g)$。maze 現況 $a\approx A(s,g)$（決定性）為其特例。
- **A2（實現性）**：函數類可表示相關條件期望（population 分析）。A2'：可附加 $O(1)$ 維輔助座標不干擾其餘。
- **A3（平滑）**：$v_\theta$ 對 $\theta$ 可微、$\nabla L$ 對 $\theta$ Lipschitz。
- **A4（通路分解）**：$a$ 只經 intent adapter + cond_head 進入 cond 向量 $c_\theta$；$v$ 對 $a$ 的依賴全部 factor through $c_\theta$（⑬ 的架構事實：$c$ 塌 $\Rightarrow\varepsilon=0$，塌在上游則下游必塌）。
- **A5（rev 相容度量）**：比較度量 $d\in\{d_{\text{time}},d_{\text{bfs}}\}$ 對時間反演不變（$|\Delta t|$、BFS 對稱皆滿足）。
- **A6（域遷移、未驗）**：同軌跡 $\Delta t$ 監督學到的序可外插到跨軌跡對（§4；由 d_bfs 臂交叉評背書）。
- **H1（效用正則）**：成功函數 $h:\mathcal Z\to[0,1]$ 可測，$\eta$-邊界層質量 $\rho(\eta)\equiv\mathbb P(z\in\partial_\eta S)$ 對小 $\eta$ 受控（env rollout + refine 管線的正則性；未驗、§5）。

## 1. 內化的形式定義

### 1.1 四個候選

- **(a) 配對成功率差**：$\Delta_{\text{succ}}(\theta)\equiv\mathbb E[h(z_a)]-\mathbb E[h(z_\varnothing)]$。idp 錶＝其成對有限樣本估計。量到：任務層依賴。與 env/eval 綁定、0/1 粗粒化、非模型內在量。
- **(b) 條件 KL**：$D(\theta)\equiv\mathbb E_{s,g,a}\,\mathrm{KL}\big(p_\theta(\cdot|s,g,a)\,\|\,p_\theta(\cdot|s,g,\varnothing)\big)$。量到：分佈層依賴。
- **(c) 資訊預算**（兩個不同的量、⛔ 不可混用）：資料版 $I_{\text{data}}(\tau;a|s,g)$＝可內化知識的上限（⑫ 機制 1 量的）；模型版 $I_\theta(z;a|s,g)$＝模型實際載送量。CFG 訓練〔驗 2207.12598〕的 $\varnothing$ 支 population 最優＝$a$-邊際 ⇒ 最優點 $D=I_\theta$（(b) 與 (c) 模型版在最優點合流）。
- **(d) 分支散度** $\varepsilon$：flow 特有、機制層、訓練期可微 — 四錶中唯一能直接當 loss 標的者（§3.2 的 $L_{\text{div}}$ 即對其 cond 層前身做 hinge）。⑬ 探針＝其 cond 層有限樣本估計（A4 使 cond 層塌 ⇒ $\varepsilon$ 塌）。

### 1.2 關係（不等式鏈）

**Prop 1.1（定性級聯）**：$\varepsilon=0\Rightarrow p_a=p_\varnothing\ \forall a\Rightarrow D=0\Rightarrow I_\theta=0$ 且 $\Delta_{\text{succ}}=0$。反向皆不成立（$\Delta_{\text{succ}}=0$ 可因 $h$ 平坦）。

**Prop 1.2（量化鏈、機制→任務）**：設 $v$ 對 $z$ 有 Lipschitz 常數 $L$。同起點 coupling + Grönwall：
$$W_2(p_a,p_\varnothing)\ \le\ W_\infty\ \le\ \tfrac{e^{L}-1}{L}\,\varepsilon .$$
任務層（兩橋、取小）：(i) $|\Delta_{\text{succ}}|\le \mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$（Pinsker）；(ii) 在 H1 下，以 $1_S$ 的 $\eta$-Lipschitz 緩和過 Kantorovich–Rubinstein：$|\Delta_{\text{succ}}|\le 2\rho(\eta)+W_1/\eta\le 2\rho(\eta)+\tfrac{e^{L}-1}{L\eta}\,\varepsilon$。
⇒ 三層望遠鏡：$\varepsilon$（機制）控 $W_2$（分佈）控 $\Delta_{\text{succ}}$（任務）。

**Prop 1.3（預算恆等式、population 最優）**〔轉引 2310.07972 Eq.4〕：最優點 $p^*_\theta(\cdot|a)=p_{\text{data}}(\cdot|s,g,a)$、$p^*_\theta(\cdot|\varnothing)=p_{\text{data}}(\cdot|s,g)$，故
$$\mathbb E_a\,\mathrm{KL}\big(p^*(\cdot|a)\|p^*(\cdot|\varnothing)\big)=I_{\text{data}}(\tau;a|s,g)=\underbrace{\mathbb E[\text{NLL}_\varnothing]-\mathbb E[\text{NLL}_a]}_{\text{條件能買到的折扣}}.$$
A1 ⇒ 折扣為 0 —「梯度上沒錢可賺」（⑫ 機制 1）的嚴格版。（註：CFM 非直接優化 NLL；上式在「CFM 最優 ⇒ 生成分佈＝條件資料分佈」的 marginal-path 唯一性下成立。）

### 1.3 與錶的對應

| 候選 | 實測儀器 | ⑦⑬ 讀數落位 |
|---|---|---|
| (a) | idp 成對 eval（R0 腿；⑤'' 協定） | $\hat\Delta_{R0}=+.015\pm.025$ |
| (d) | ⑬ 散度探針（cond 層＋$d_{\text{zero}}$） | B/A＝1.1%、塌在 cond 生成端 |
| (c) 資料版 | 兩小 flow loss-gap 積分（⑫ 分辨實驗 B、未跑） | — |

### 1.4 簡併問題與推薦定義

(a)–(d) 全是「**依賴度**」錶：都只比較 $p_a$ 與 $p_\varnothing$。而依賴 $\to 0$ 有三個 preimage：完美內化（prior 已含知識）、鎖死（通路死、知識不在）、無物可內化（$I_{\text{data}}=0$）— ⑦ 實測正踩此簡併（$\Delta\approx0$ 且 prior$\approx$基線）。拆開簡併需要外部三點校準：

**Definition 1.4（內化率、推薦的正式定義）**：固定效用 $U$（R0 成功率）、基線 $\theta_{\text{base}}$（從未見 $O$）、參考 $\theta_{\text{ref}}$（全曝光 $p{=}0$）：
$$\mathrm{Int}(\theta)\ \equiv\ \frac{U_\varnothing(\theta)-U(\theta_{\text{base}})}{U_a(\theta_{\text{ref}})-U(\theta_{\text{base}})}\ ,\qquad\text{定義域：分母}\ \ge\ \kappa\cdot SE\ \text{（知識可讀門檻）},$$
配**診斷對 $(\mathrm{Int},\varepsilon)$**：$(1,0)$＝完美內化；$(0,0)$＝鎖死；$(0,+)$＝未內化但通路活；中間＝部分內化。
- **理由**：①分母校準「有多少可內化」（$I_{\text{data}}$ 的效用影子）②三點結構是唯一拆得開三 preimage 的形 ③讀數由既有三實驗直接拼出（基線 .321／f27n .454／idp-zero .321 ⇒ $\widehat{\mathrm{Int}}\approx0$、$\varepsilon_{\text{rel}}=1.1\%$ ⇒ 落 $(0,0)$ 鎖死格 — 與 ⑭ 判決一致）④分母 $\approx0\Rightarrow$ undefined，把 ⑤''「subgoal 格內化欄 undefined」的收表裁決變成定義的形式推論而非 ad-hoc 約定。
- 理論節用 $\varepsilon$ 寫命題（可微、機制層）；實驗節用 $\mathrm{Int}$ 報內化 — 雙層各司其職。NOTE §二的「gap 小且 prior 絕對值高」即 $(\mathrm{Int}\to1,\ \Delta\to0)$ 的敘事版。

## 2. 條件通路鎖死（⑫⑭ 的理論化）

**Prop 2.1（退化全域最優；A1+A2）**：CFM 逐點為 $v$ 的二次型，兩支的函數空間唯一極小元為條件期望 $v_1^*=\mathbb E[u_t|z_t,s,g,a]$、$v_2^*=\mathbb E[u_t|z_t,s,g]$。A1 ⇒ $(u_t,z_t)\perp a\mid(s,g,t)$ ⇒ $v_1^*=v_2^*$ ⇒ **intent-invariant 解同時極小化兩支、為 $L_p$ 之全域最優，$\forall p\in[0,1]$，且（資料支撐上）population 最優解唯一＝invariant**。無 trade-off、$p$ 不動 argmin。
_Sketch_：$L^2$ 投影唯一性＋條件獨立下條件期望塔性質。結構對應 Robinson〔轉引 2106.11230 Prop 2.2〕的「冗餘條件下棄用＝合法最優」（彼為 contrastive、此為 $L^2$ 迴歸；遷移直接，因兩者最優解皆由充分統計決定）。

**Cor 2.2（guidance 無效；一行）**：$\varepsilon=0\Rightarrow \tilde v_w=v_\varnothing+w(v_a-v_\varnothing)=v_\varnothing\ \forall w$ ⇒ 生成分佈與 $w$ 無關 — CFG 外插的是零向量（⑪ 的 $.344{\pm}.089$ vs $.336{\pm}.089$；⑫ 機制 4）。

**Prop 2.3（鎖死＝穩定駐點集；A1–A4）**：
(i) *一階平坦*：設 $\theta\in M\equiv\{\theta:v_\theta$ 對 $a$ 逐點常值$\}$ 且 $\varnothing$ 支已最優。任意參數方向 $h$，其輸出擾動 $\partial_h v$ 分解為 $a$-對稱與 $a$-反變（$\mathbb E_a$ 均值零）分量；A1 使 residual 對 $a$ 條件無關 ⇒ **反變分量與 residual 的期望內積為 0** ⇒ 打開 $a$-分辨的方向導數＝0。
(ii) *無逃逸曲率*：$M\cap\{\text{最優}\}$ 由 Prop 2.1 是全域最優 ⇒ Hessian 半正定、無負曲率離開方向 ⇒ 弱穩定駐點集。
(iii) *塌後不再分化*〔結構借 Cocos 轉引 2505.11123 Thm 1〕：A3 下兩支對共享參數的梯度差 $\le C\cdot\varepsilon$ ⇒ $\varepsilon\approx0$ 自我維持（self-reinforcing；期望梯度流下 $\varepsilon(0)=0\Rightarrow\varepsilon(t)=0$）。

**Remark 2.4（賽跑機制 — 曝光稀釋的動力學形；線性化敘事）**：$a$ 通路的期望生長驅動 $\propto\|\mathbb E[a\otimes\delta]\|$，$\delta$＝residual。A1 ⇒ $\mathbb E[a\otimes\delta]=\mathbb E\big[\mathbb E[a|s,g]\otimes\mathbb E[\delta|s,g]\big]$ — 只剩 $\delta$ 的 $(s,g)$-可預測分量供能。$p>0$ 使 $\varnothing$ 支被迫自建 $(s,g)$ 通路；其收斂即令該分量 $\to0$ ⇒ **$a$ 通路的生長驅動隨 $\varnothing$ 支收斂而枯竭**。鎖死＝$a$ 通路生長 vs $\varnothing$ 支收斂的賽跑落敗（⑦ 曝光稀釋預言的機制形：8000 步 × 70% 曝光輸掉賽跑；f27n $p{=}0$ 無對手故長成 .6046）。

**Remark 2.5（資訊冗餘 ≠ 計算冗餘）**：A1 只封殺 population 最優的資訊增益；f27n 的 $+.133$ 顯示有限容量/有限步數下 $a$ 有「計算捷徑」價值（讀 $A(s,g)$ 比從 $(s,g)$ 重算便宜）〔轉引 Autoguidance 2406.02507 §3〕。$p>0$ 強迫自算通路建成後、捷徑邊際價值歸零 ⇒ 塌。⇒ 本檔所有 population 命題須配此有限容量註腳讀。

**Conj 2.6（$p$ 臨界值）**：存在 $p^*(I_{\text{data}},\text{容量},\text{步數})$：$p<p^*$ 捷徑存活、$p>p^*$ 鎖死；Remark 2.4 的 two-timescale 賽跑給出機制但未給閉式。升級：線性 adapter + 二次 loss 的 toy model 可解析（建議 Rei 出手點）；實驗上 idp01/idpxm 劑量二臂（⑪ 已灑）＋⑫ 分辨實驗 B 直接探。文獻空白（⑫⑭ 兩度確認）：影像圈 $p\in[0.1,0.2]$ 慣例僅驗樣品品質、未驗條件遵從。

## 3. 藥方的理論形

### 3.1 退火 $p$＝warm-start（吸子語言）

**Prop 3.1（屏障擋「先塌後修」；A1+A2）**：設 $\theta_A$（f27n 形：$a$ 支誤差 $\delta_a$ 小、$\varnothing$ 支誤差 $\delta_\varnothing$ 大）。任何先達 $\{\varepsilon=0,\ \varnothing\text{ 支誤差}\ge\delta_\varnothing/2\}$ 再修的連續路徑，中途損失 $\ge(1-p)(\delta_\varnothing/2)^2$ 級 — 對小 $p$、$\delta_a\ll\delta_\varnothing$ 屏障嚴格正。梯度流損失單調不增 ⇒ 從 $\theta_A$ 出發**不走「先塌後修」**；局部主梯度為修 $\varnothing$ 支（$\propto p\,\delta_\varnothing$），塌方向一階增益 $\approx0$（$a$ 支已近最優）⇒ 短期黏住。
**限制（誠實、即 ⑭「單獨不保險」的幾何形）**：「**同步塌修**」路徑（邊修 $\varnothing$ 邊縮 $a$-反變分量）可損失單調直達 invariant 全域最優 — 屏障擋不住它。He+19 lagging-encoder〔驗 1901.05534〕同構：初期弱 posterior 引發塌、warm-start 改初值不改地貌；Cyclical-Annealing〔驗 1903.10145〕的「升回目標 $p$ 仍塌」＝此路徑實測存在的旁證。
**Conj 3.2（長時黏性未保證）**：A1 下兩 basin population 損失同高（皆可達最優）⇒ 長期佔據由 basin 體積/平坦度＋SGD 噪音決定（invariant 解疑更平坦 ⇒ 偏塌）。判決實驗＝⑭ warm-start 半天測（黏住 ⇔ 塌回）。

### 3.2 cond 層散度 floor

**Def 3.3**：$L_{\text{div}}(\theta)\equiv\mathbb E\big[\max\big(0,\ m-\|c_\theta(s,g,a)-c_\theta(s,g,\varnothing)\|\big)\big]$，$m$＝margin（標度建議錨 f27n cond 層差 .6046；實作註）。

**Prop 3.4（可行集手術；A1+A2'）**：對 $L_p+\lambda L_{\text{div}}$，$\lambda>0$：任何 invariant 解付 $\lambda m$；而存在「$c$ 層附加 $a$ 的單射座標、下游忽略之」的解達 $L_p$ 最優且 $L_{\text{div}}=0$ ⇒ **invariant 解被逐出全域最優集**、新最優集 $\subseteq\{\varepsilon_{\text{cond}}\ge m\}$。
**Remark 3.5（結構藥 vs 內容藥）**：手術只保 $\varepsilon_{\text{cond}}\ge m$、**不保 $\varepsilon_v>0$**（A1 下 population 最優仍可 $v$-invariant — 下游合法地忽略被撐開的座標）。⇒ $L_{\text{div}}$ 的作用＝保「可喚醒性」（通路物理在場：guidance 有物可外插、warm-start 的 $a$ 支不被物理拆除），製造依賴要靠 §3.3 的資訊供給。⑭「雙保險」的理論分工：退火 $p$ 保動力學路徑、$L_{\text{div}}$ 保結構在場、破冗餘保資訊供給 — 三藥打三層、彼此不冗餘。margin 放 cond 層＝正中 ⑬ 定位的塌陷格（且 cond 層低維、margin 可選；速度場層 margin 會直接扭曲生成）。
**Remark 3.6（塌陷點可重啟）**：$\|x\|$ 在 $x=0$ 的 subgradient 為單位球 ⇒ hinge 在 exact 塌陷點提供非零逐出力；cos/乘積型正則在 0 梯度消失（塌陷點反成駐點）⇒ margin-hinge 是形式上正確的選擇。

### 3.3 資料層破冗餘（根治）

形式條件：**C-i（多路線）** 存在 route 變數 $R=f(\tau)$、$H(R|s,g)\ge h>0$ 於正測度 $(s,g)$；**C-ii（錨資訊性）** $I(R;a|s,g)\ge\delta$；**C-iii（效用可分）** eval 對不同 $R$ 產生可讀差異。

**Prop 3.7（鎖死前提拆除；C-i+C-ii）**：$R=f(\tau)$ ⇒（DPI）$I_{\text{data}}(\tau;a|s,g)\ge I(R;a|s,g)\ge\delta>0$ ⇒ A1 失效：Prop 1.3 給出正折扣、兩支條件期望分離（Prop 2.1 失效）、Prop 2.3(i) 的方向導數轉負（打開依賴有一階增益）⇒ **最優性與零驅動兩根支柱同時拆除**。
**Remark 3.8**：C-iii 獨立必要 — 無它則 $I>0$ 而 $\Delta_{\text{succ}}$ 恆 0（⑦ subgoal 格 intent 價值 $\approx0$ 的形式原因＝eval 只問終點不問路線；亦即 Def 1.4 分母條款的另一鏡頭）。teleport／counterfactual 構造＝直接抬 $H(R|s,g)$，與 stitch 本義及 ⑫⭐ 合流。

## 4. 路線一幾何的相容性（DESIGN v2 的命題化）

**Prop 4.1（子空間相容存在性；A5）**：時間反演 rev 為對合（$\mathrm{rev}^2=\mathrm{id}$）⇒ 函數空間 $\mathbb Z_2$-分解 $F=F_+\oplus F_-$（$\varphi_\pm=\tfrac12(\varphi\pm\varphi\circ\mathrm{rev})$）。取 $e_d\in F_-$、$e_m\in F_+$，則：(i) $e_d(\mathrm{rev}\,\tau)=-e_d(\tau)$ ⇒ 凡 $\|e_d\|>0$ 處 $\cos=-1$ 恰好達成；(ii) $e_m$ rev-不變＋A5 ⇒ rank 幾何在 rev-商空間上良定義、與 (i) 作用於互補子空間 ⇒ **兩約束無聯立矛盾**；(iii) $\varphi=\varphi_++\varphi_-$ 可重建 ⇒ 分解不丟資訊 — **recon 不犧牲的條件**＝$D_m\ge\dim_{\text{eff}}(\varphi_+)$、$D_d\ge\dim_{\text{eff}}(\varphi_-)$、decoder 讀 concat。非平凡性條件：$\tau\ne\mathrm{rev}\,\tau$ 於正測度（非回文軌跡）⇒ 非零反對稱分量存在。
**Cor 4.1'**：N1（rev 對 rank/rev 聯立矛盾）在子空間版**結構性消滅**：rev 對的 $e_m$ 距離為 0 與 $d(\tau,\mathrm{rev}\tau)=0$（A5）一致、拉遠只發生在 $e_d$ — 排池從「必要」降為「雙保險」（DESIGN 保留之、相容）。
**Remark 4.2（D_d 預算）**：C1 實測 $\cos=+0.9901$ ⇒ 現況 $e$ 幾乎全落 $F_+$、反對稱能量 $\approx0$ — $D_d{=}16$ 是預算上限、非需求證明；$\|e_d\|\to0$ 使 cos 病態 ⇒ 實作需範數下限（DESIGN 已列尺度捷徑風險）。$\dim_{\text{eff}}(\varphi_-)$ 未量 ⇒ 升級需 $\varphi_-$ 能量譜探針（§5）。

**Prop 4.2（$d_{\text{time}}$ 排序一致性；條件式）**：設行為策略使 **B1（條件單調）** $\mathbb E[\Delta t\,|\,d_{\text{bfs}}{=}k]=f(k)$ 嚴格增 — 漂移主導 $f(k)\!\approx\!k/v$、純擴散 $f(k)\!\approx\!k^2$，兩極皆單調 ⇒ B1 弱；**B2（gap 集中）** $\mathbb P\big(|\Delta t-f(k)|>\tfrac12(f(k{+}1){-}f(k))\big)\le q$。則三元組排序錯誤率 $\le 2q$ ⇒ population 級序一致。**破壞者是變異非單調性**：純擴散 first-passage $\mathrm{Var}\sim k^4$ ⇒ 長程 $q$ 爆 ⇒ 長 $\Delta t$ 段失效。
**實證錨（C2、[實測]）**：整體 $\rho=.758$＋反向三元組 $13.4\%$（$\hat q\approx.13$）＝B1+B2 於本資料尺度成立；最弱箱 $.203$＝長 $\Delta t$ 的 B2 破損實錄 ⇒ DESIGN「短 $\Delta t$ 段＋長程交 $d_{\text{bfs}}$ 臂」＝控 $q$ 協定的形式對應。跨軌跡外插屬 A6（未驗）— 交叉形式評估（$d_{\text{time}}$ 訓 → $d_{\text{bfs}}$ held-out 評；N2 解法）即其測試。

## 5. 誠實邊界（分級與升級路徑）

| 條目 | 級別 | 關鍵假設 | 升級需要 |
|---|---|---|---|
| Prop 1.1／1.2 | Prop | A3-類 Lipschitz；K-R 橋另需 H1 | H1 對 refine 管線是否成立 — **要 Rei 檢**（env rollout 使 $h$ 非 Lipschitz；$\rho(\eta)$ 可由 eval 資料估） |
| Prop 1.3 | Prop（恆等式轉引） | population 最優、marginal-path 唯一性 | 讀 2310.07972 原件確認 Eq.4 形式 |
| Def 1.4 | Definition | $\theta_{\text{ref}}$ 的選取（f27n＝目前最佳非唯一） | ref 標準化（全曝光收斂判準）寫進收表協定 |
| Prop 2.1／Cor 2.2 | **Prop（嚴格）** | A1+A2 | — （地基最穩的兩塊） |
| Prop 2.3 | Prop | A1–A4；(iii) 借 Cocos 結構 | (i) 反變分量論證的量詞順序嚴格化 — **要 Rei 檢**；讀 Cocos 原件 |
| Remark 2.4 賽跑 | Remark（線性化） | 線性 adapter 近似 | toy model 解析（線性+二次可閉式）→ 可升 Prop |
| Conj 2.6 $p^*$ | **Conj** | — | 劑量二臂（已灑）＋I 預算探針＋toy model |
| Prop 3.1 | Prop（限定範圍） | 只擋「先塌後修」 | — |
| Conj 3.2 黏性 | **Conj** | basin 幾何未知 | ⑭ warm-start 半天實驗直接判 |
| Prop 3.4／R3.5 | Prop | A2' | $\varepsilon_{\text{cond}}\ge m$ 下游是否真被用＝實驗題（idp+div 臂） |
| Prop 3.7 | Prop | DPI＋轉引恆等式 | C-iii 的 eval 改造（route 可分判準）尚無設計 |
| Prop 4.1／Cor 4.1' | **Prop（構造性、嚴格）** | A5 | $\dim_{\text{eff}}(\varphi_-)$ 能量譜探針（新 C 段、CPU 級） |
| Prop 4.2 | Prop（條件式） | B1+B2＋A6 | A6＝conjecture 級 — 交叉評臂判 |

全域註腳：本檔所有 Prop 為 **population 級**；有限容量效應（Remark 2.5 的 f27n $+.133$）證明 population 敘事單獨不完整 — paper 理論節建議「population 定理＋有限容量 remark」雙層誠實寫法。

## References

轉引（docs 已引、未讀原件）：Cocos 2505.11123（Thm 1）；Robinson 2106.11230（Prop 2.2）；資訊分解 2310.07972（Eq.4）；Autoguidance 2406.02507（§3）。
本次驗過 ID：CFM — Lipman et al., arXiv:2210.02747（ICLR'23）；CFG — Ho & Salimans, arXiv:2207.12598；lagging encoder — He et al., arXiv:1901.05534（ICLR'19）；cyclical annealing — Fu et al., arXiv:1903.10145（NAACL'19）。
教科書級：Pinsker；Grönwall；Kantorovich–Rubinstein 對偶；$\mathbb Z_2$ 表示分解；first-passage 矩（漂移 $\mathbb E\sim k$／擴散 $\mathbb E\sim k^2,\mathrm{Var}\sim k^4$）。
