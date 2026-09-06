# THEORY — 內化的形式化與條件通路鎖死（v2、A1 證偽整併版；交 Rei 磨嚴）

_理論使魔（Fable、主人授權；表徵幾何線）v0 2026-09-05；v2 2026-09-06 理論整併使魔。
對應素材：FINDINGS-0905 ⑦⑩⑪⑫⑬⑭＋⑯⑰⑰'⑱⑱'、REVIEW-2026-09-05-night-conclusions（下稱 REVIEW）、
THEORY-2026-09-06-postA1-revision（下稱 postA1）、THEORY-REVIEW-2026-09-05（下稱丙）、
NOTE-2026-09-06-context-taxonomy（下稱 CT）、NOTE-composition-law §二§四、DESIGN-route1-losses v2。
⛔ 分級鐵則：**Prop = 在列明假設下可完整證**、**Conj = 未證**、Remark = 敘事/解讀。
引用標注：〔轉引〕= docs 已引未讀原件；〔驗〕= 上網驗過 ID。_

**〔v2 修訂聲明（2026-09-06；取代 9/5 深夜插播）〕**：A1 已被直接量測**證偽於本資料**
（$I(\tau;a|s,g)\approx2.5$ bits $\ne0$、決定程度中位 .749 < 預釘 .8；⑰、對抗複核 ⑰'）⇒
本檔病因重心自「條件冗餘（$I\approx0$）＋不變性壓力 → 鎖死」移至「**dropout 動力學陷阱**」：
資訊在（bits 層），但經 e_target 度量重計價後變異佔比僅 ~0.4%（⑰ Z2）、條件通路在梯度
賽跑中輸給主通路（R2.4 升主敘事＋丁 LT-6 overshoot）。散度實錘（⑬ B/A=1.1%、塌在 cond
生成端）不受影響、全數保留。Prop 2.1／2.3 降為「A1 極限情形」（§2；$I>0$ 修訂版命題與
證明在 postA1 §2、本檔引用不重抄）。藥方分工不變、權重照 ⑰ 移：退火 $p$（動力學）＋
$L_{\rm div}$ hinge（結構在場）為主，資料層破冗餘降為輔（§3.3；其「=stitch 本義」論述
地位保留）。劑量曲線的可證偽預測住 postA1 §4（pre-registered、git 時戳先於資料 —
⛔ 本檔僅引用、不改寫）。逐條修訂摘要見 REVISION-NOTES-internalization-v2.md。

## 0. Setup 與假設

條件 $c=(s,g,a)$；$a$ = intent 錨（訓練時可查知識源 $O$ 的摘要 $A(\tau,O)$）、$\varnothing$ = intent 段歸零。
Flow-matching（CFM〔驗 2210.02747〕）with INTENT_DROP $p$（intent 段獨立歸零、$(s,g)$ 恆在場）：

$$L_p(\theta)=\mathbb{E}_{(\tau,a),t,z_0}\Big[(1{-}p)\,\|v_\theta(z_t,t,s,g,a)-u_t\|^2+p\,\|v_\theta(z_t,t,s,g,\varnothing)-u_t\|^2\Big]$$

$p_\theta(\cdot|s,g,\cdot)$ = 對應 ODE 的終端分佈。分支散度 $\varepsilon(\theta)\equiv\sup_{z,t,s,g,a}\|v_\theta(\cdot|s,g,a)-v_\theta(\cdot|s,g,\varnothing)\|$（期望版 $\bar\varepsilon$ 同理）。

**假設（命題各自引用）**
- **A1-lim（條件冗餘極限；原 A1、⑰ 後降級為極限錨）**：$I_{\text{data}}(\tau;a\mid s,g)=0$，即 $\tau\perp a\mid(s,g)$。
  ⛔ **實測不滿足**（⑰ 直接量測、⑰' 對抗複核獨立重算吻合）：本資料 $I(\tau;a|s,g)\approx2.5$ bits
  （episode 去重 3.3／2×2 粗化仍 1.4／res16 桶 2.7 — 非走廊抖動非桶粗）。引用紀律（⑰'）：
  2.5 bits 中 63% 是格界抖動 — A1 的**形式**否定（錨攜帶資訊）以 2.5 為真；「路線選擇多樣性」
  敘事只能講**走廊級 ≈1 bit（最保守 0.93）** — 引用時拆開講。決定程度中位 .749 < 預釘 .8
  （n≥20 桶 H 中位 4.64、小樣本只會低估）⇒ v0 句「maze 現況 $a\approx A(s,g)$（決定性）」
  **不成立**。地位＝**極限錨**：$\eta\to0$ 時 §2 諸命題退回原版；其**物理真身**＝
  「自生／shuffled $u$ 訓練臂」— 該臂 $u\perp\tau\mid(s,g)$ 精確成立、A1-lim 命題逐字適用
  （CT §2.1 Cor CT-3；控制臂預測＝CT §5.2 P1）。
- **A1′（實測替代；⑰）**：$I_{\text{data}}(\tau;a\mid s,g)>0$。本資料實況；與 C-ii′ 配用。
- **C-ii′（$z$-層可讀＋變異權重；丙 B15＋postA1 §1）**：route 在 latent 可讀
  （$I(z;a|s,g)\ge\delta'>0$）**且**在 $z$-度量下有變異權重 $\eta_{\rm eff}>0$。
  ⛔ bits $>0$ 不蘊含 $\eta_{\rm eff}$ 大 — CFM 以**變異**計價、不以 bits 計價（匯率斷裂；
  postA1 §3）。$z$-層 $I$ 未直接量、由 ⑰ Z2 雙峰（長窗能量 21.8%）強烈支持 —
  `TODO(驗收)`：$I(z;a|s,g)$ 直接量測（⑰ 儀器換 $z$ 輸入；postA1 Prop 2.1′ 前提補完＋
  Conj 2.5′ 分辨）。
- **A2（實現性）**：函數類可表示相關條件期望（population 分析）。A2'：可附加 $O(1)$ 維輔助座標不干擾其餘。
- **A2u（唯一性量詞；丙 B5）**：「population 最優唯一＝invariant」類陳述只對 $p\in(0,1)$；
  端點 $p\in\{0,1\}$ 的離場支零權重、行為不受約束（f27n $p{=}0$ 的 cond 層差 .6046（⑬）
  與唯一性不衝突 — 丙 D3 自洽紅利）。
- **A3（平滑）**：$v_\theta$ 對 $\theta$ 可微、$\nabla L$ 對 $\theta$ Lipschitz。
- **A4（通路分解）**：$a$ 只經 intent adapter + cond_head 進入 cond 向量 $c_\theta$；$v$ 對 $a$ 的依賴全部 factor through $c_\theta$（⑬ 的架構事實：$c$ 塌 $\Rightarrow\varepsilon=0$，塌在上游則下游必塌）。
- **A5（rev 相容度量）**：比較度量 $d\in\{d_{\text{time}},d_{\text{bfs}}\}$ 對時間反演不變（$|\Delta t|$、BFS 對稱皆滿足）。
- **A6（域遷移、未驗）**：同軌跡 $\Delta t$ 監督學到的序可外插到跨軌跡對（§4；由 d_bfs 臂交叉評背書）。
- **H1（效用正則）**：成功函數 $h:\mathcal Z\to[0,1]$ 可測，$\eta$-邊界層質量 $\rho(\eta)\equiv\mathbb P(z\in\partial_\eta S)$ 對小 $\eta$ 受控（env rollout + refine 管線的正則性；未驗、§5）。

## 1. 內化的形式定義

### 1.1 四個候選

- **(a) 配對成功率差**：$\Delta_{\text{succ}}(\theta)\equiv\mathbb E[h(z_a)]-\mathbb E[h(z_\varnothing)]$。idp 錶＝其成對有限樣本估計。量到：任務層依賴。與 env/eval 綁定、0/1 粗粒化、非模型內在量。
- **(b) 條件 KL**：$D(\theta)\equiv\mathbb E_{s,g,a}\,\mathrm{KL}\big(p_\theta(\cdot|s,g,a)\,\|\,p_\theta(\cdot|s,g,\varnothing)\big)$。量到：分佈層依賴。
- **(c) 資訊預算**（兩個不同的量、⛔ 不可混用）：資料版 $I_{\text{data}}(\tau;a|s,g)$＝可內化知識的上限 — 本資料已直接量測 $\approx2.5$ bits、路線選擇級 ≈1 bit（⑰⑰'、引用拆開講）；模型版 $I_\theta(z;a|s,g)$＝模型實際載送量。⚠️ bits 預算非效用預算 — 兌現還要過匯率（bits→變異＝C-ii′；postA1 §3）。CFG 訓練〔驗 2207.12598〕的 $\varnothing$ 支 population 最優＝$a$-邊際 ⇒ 最優點 $D=I_\theta$（(b) 與 (c) 模型版在最優點合流）。
- **(d) 分支散度** $\varepsilon$：flow 特有、機制層、訓練期可微 — 四錶中唯一能直接當 loss 標的者（§3.2 的 $L_{\text{div}}$ 即對其 cond 層前身做 hinge）。⑬ 探針＝其 cond 層有限樣本估計（A4 使 cond 層塌 ⇒ $\varepsilon$ 塌）。

### 1.2 關係（不等式鏈）

**Prop 1.1（定性級聯）**：$\varepsilon=0\Rightarrow p_a=p_\varnothing\ \forall a\Rightarrow D=0\Rightarrow I_\theta=0$ 且 $\Delta_{\text{succ}}=0$。反向皆不成立（$\Delta_{\text{succ}}=0$ 可因 $h$ 平坦）。

**Prop 1.2（量化鏈、機制→任務）**：設 $v$ 對 $z$ 有 Lipschitz 常數 $L$。同起點 coupling + Grönwall：
$$W_2(p_a,p_\varnothing)\ \le\ W_\infty\ \le\ \tfrac{e^{L}-1}{L}\,\varepsilon .$$
任務層（兩橋、取小）：(i) $|\Delta_{\text{succ}}|\le \mathrm{TV}\le\sqrt{D_{\mathrm{KL}}/2}$（Pinsker）；(ii) 在 H1 下，以 $1_S$ 的 $\eta$-Lipschitz 緩和過 Kantorovich–Rubinstein：$|\Delta_{\text{succ}}|\le 2\rho(\eta)+W_1/\eta\le 2\rho(\eta)+\tfrac{e^{L}-1}{L\eta}\,\varepsilon$。
⇒ 三層望遠鏡：$\varepsilon$（機制）控 $W_2$（分佈）控 $\Delta_{\text{succ}}$（任務）。
（丙 B2 小註：$\rho(\eta)$ 對**兩個**分佈取（或取大者）；$L$ 為對 $z$、uniform-in-$t$。
⑰ 數值對答案：Prop 1.1 方向成立、四格無嚴格反例；但 $\Delta/\varepsilon$ 斜率兩腿差 15×
⇒ 望遠鏡常數**腿依賴**、$\varepsilon$ 讀數 ⛔ 不可跨腿當定量預測子。）

**Prop 1.3（預算恆等式、population 最優；A3′ 上限語意）**〔轉引 2310.07972 Eq.4〕：最優點 $p^*_\theta(\cdot|a)=p_{\text{data}}(\cdot|s,g,a)$、$p^*_\theta(\cdot|\varnothing)=p_{\text{data}}(\cdot|s,g)$，故
$$\mathbb E_a\,\mathrm{KL}\big(p^*(\cdot|a)\|p^*(\cdot|\varnothing)\big)=I_{\text{data}}(\tau;a|s,g)=\underbrace{\mathbb E[\text{NLL}_\varnothing]-\mathbb E[\text{NLL}_a]}_{\text{條件能買到的折扣}}.$$
**A3′（enc 充分性；丙 B3＋postA1 §1）**：在 $z={\rm enc}(\tau)$ 上恆等式退為
$\mathbb E_a\,\mathrm{KL}\le I_{\text{data}}(\tau;a|s,g)$（DPI 方向）、等式需 enc 對 $a$ 充分 —
本檔一律以**上限語意**引用。A1-lim ⇒ 折扣為 0（「梯度上沒錢可賺」的嚴格版）— ⛔ 僅極限錨：
實測折扣上限 $\approx2.5$ bits $>0$（⑰）、⑫ 機制 1 已倒。⚠️ 正折扣以 NLL 計價；兌換成
CFM 梯度與效用還要過匯率（C-ii′；postA1 §3）—「有沒有錢」自 ⑰ 起不再是瓶頸、
「挖錢的成本」才是。（註：CFM 非直接優化 NLL；上式在「CFM 最優 ⇒ 生成分佈＝條件資料分佈」的 marginal-path 唯一性下成立。）

### 1.3 與錶的對應

| 候選 | 實測儀器 | 讀數落位（⑦⑬⑰） |
|---|---|---|
| (a) | idp 成對 eval（R0 腿；⑤'' 協定） | $\hat\Delta_{R0}=+.015\pm.025$ |
| (d) | ⑬ 散度探針（cond 層＋$d_{\text{zero}}$） | B/A＝1.1%、塌在 cond 生成端 |
| (c) 資料版 | 直接量測（hindsight 錨＝粗格 route 序列 ⇒ $I=H(a|s,g)$；⑰ 儀器、⑰' 獨立重算複核） | $\approx2.5$ bits（路線選擇級 ≈1 bit、⑰' 拆開講）；決定程度中位 .749 |

### 1.4 簡併問題與推薦定義

(a)–(d) 全是「**依賴度**」錶：都只比較 $p_a$ 與 $p_\varnothing$。而依賴 $\to 0$ 有三個 preimage：完美內化（prior 已含知識）、鎖死（通路死、知識不在）、無物可內化（$I_{\text{data}}=0$＝A1-lim；本資料此格已被 ⑰ 排除）— ⑦ 實測正踩此簡併（$\Delta\approx0$ 且 prior$\approx$基線；⑰＋⑬ 合看後拆定＝**鎖死格**、非「無物」格）。拆開簡併需要外部三點校準：

**Definition 1.4（內化率、推薦的正式定義）**：固定效用 $U$（R0 成功率）、基線 $\theta_{\text{base}}$（從未見 $O$）、參考 $\theta_{\text{ref}}$（全曝光 $p{=}0$）：
$$\mathrm{Int}(\theta)\ \equiv\ \frac{U_\varnothing(\theta)-U(\theta_{\text{base}})}{U_a(\theta_{\text{ref}})-U(\theta_{\text{base}})}\ ,\qquad\text{定義域：分母}\ \ge\ \kappa\cdot SE\ \text{（知識可讀門檻）},$$
配**診斷對 $(\mathrm{Int},\varepsilon)$**：$(1,0)$＝完美內化；$(0,0)$＝鎖死；$(0,+)$＝未內化但通路活；中間＝部分內化。
- **理由**：①分母校準「有多少可內化」（$I_{\text{data}}$ 的效用影子）②三點結構是唯一拆得開三 preimage 的形 ③讀數由既有三實驗直接拼出（基線 .321／f27n .454／idp-zero .321 ⇒ $\widehat{\mathrm{Int}}\approx0$、$\varepsilon_{\text{rel}}=1.1\%$ ⇒ 落 $(0,0)$ 鎖死格 — 與 ⑭ 判決一致；⑰ 後**格位不變、格子的物理改讀**＝動力學陷阱（§2.0；postA1 §2.3：A1 版讀「合法全域最優」、修訂版讀「$\mathrm{Int}^*(0.3)\approx1\%$ 級的合法小目標＋慢時間尺度」））④分母 $\approx0\Rightarrow$ undefined，把 ⑤''「subgoal 格內化欄 undefined」的收表裁決變成定義的形式推論而非 ad-hoc 約定。
- 理論節用 $\varepsilon$ 寫命題（可微、機制層）；實驗節用 $\mathrm{Int}$ 報內化 — 雙層各司其職。NOTE §二的「gap 小且 prior 絕對值高」即 $(\mathrm{Int}\to1,\ \Delta\to0)$ 的敘事版。
- **實務註（⑱'＋REVIEW §3–4；ref／分母時效）**：$\theta_{\text{ref}}$＝f27n@8000 為「目前最佳非唯一」（§5 自標）、已被長訓臂（idpxm@11429）超越 ⇒ **Int 讀數在 ref 換代前無定義**。收表協定改（⑱' 裁定）：ref／基線換 step-matched 長訓家族（f27nL／N5L）；primary endpoint＝**配對差 $U_a-U_\varnothing$ 與 $U_\varnothing-U_{\varnothing,\text{ref-long}}$** — on 絕對值受 generic 訓練長度汙染、降次要（on/zero 齊漲 74%＝該混淆的實測指紋、REVIEW §3(ii)）。`TODO(驗收)`：新 ref 具體顆與 ep200 錨（f27nL／N5L 收表後釘）。

## 2. 條件通路鎖死（v2：dropout 動力學陷阱；⑬⑭⑰ 的理論化、A1 版命題降極限錨）

### 2.0 實證地位（v2 新增：什麼倒了、什麼沒倒）

- **倒了**：A1 作為本資料實況（⑰：$I\approx2.5$ bits、決定程度 .749；⑰' 獨立重算複核）
  ⇒ ⑫ 機制 1（「沒錢可賺」）倒；Prop 2.1／2.3(i)(ii) 前提為偽、降「A1 極限情形」（下）。
  ⑭ 的機制句「忽略 intent＝合法的全域最優」隨之撤回 — ⑭ 的散度證據與藥方分工保留。
- **沒倒（量測實錘全數保留）**：⑬ 散度探針 — idp 分支散度只剩 f27n 的 **B/A＝1.1%**
  （cond 層相對差 .0109 vs .6046、差 55×）、swap 零反應、**塌在 cond 生成端**（intent
  adapter＋cond_head 輸出已 intent-invariant；d_sg 兩顆都活 ⇒ 非整體 cond 盲、專屬 intent
  段）。此為對 $\varepsilon$ 的**直接量測、不經 A1**。Cor 2.2（guidance 無效）只依賴
  $\varepsilon\approx0$、照樣成立（⑪ 的 .344 vs .336 照樣被解釋）。
- **病因重述（⑰ 裁定＋postA1 §2–3）**：資訊在 — $\tau$ 層 $I\approx2.5$ bits（路線選擇級
  ≈1 bit、⑰' 拆開講）；但經 e_target 度量重計價後 route 分量**變異佔比僅 ~0.4%**（⑰ Z2
  整體中位 0.37%）⇒ $\eta_{\rm eff}\approx0.003\lambda\ll p\lambda=0.3\lambda$（$\varepsilon$
  反推與 Z2 能量兩獨立 $z$-估計吻合 10%；postA1 §3.2）⇒ 條件通路的梯度供能輸給主通路
  （R2.4 賽跑）。**鎖死＝動力學陷阱、不是合法全域最優**。修訂陳述一句話（postA1 §2.2）：
  系統不是被困在非法駐點，而是以慢速率 $\kappa=(1-p)(p\lambda+\eta)$ **合法地收斂到一個
  本來就被 ridge 壓扁的小目標** $\mathrm{Int}^*\approx\eta_{\rm eff}/(p\lambda)\approx1\%$ —
  「陷阱」的成分是時間尺度分離（$\sim12\times$；丁 §1.6）＋目的地渺小，不是最優性。
  $I>0$ 下為**深度 crossover**、無 sharp 相變（丁 §1.3；⑮ 機器精度驗訖）。
  $I>0$ 修訂版命題（Prop 2.1′／2.2′／2.3′）之完整陳述與證明在 postA1 §2 — 本節保留
  A1-lim 原版當極限錨、不重抄。

**Prop 2.1（A1-lim 極限情形：退化全域最優；A1-lim＋A2＋A2u）**：〔⛔ 前提 A1-lim 實測不滿足（⑰）— 本命題對 maze 資料**不是**實況陳述；合法適用域見命題後。〕CFM 逐點為 $v$ 的二次型，兩支的函數空間唯一極小元為條件期望 $v_1^*=\mathbb E[u_t|z_t,s,g,a]$、$v_2^*=\mathbb E[u_t|z_t,s,g]$。A1-lim ⇒ $(u_t,z_t)\perp a\mid(s,g,t)$ ⇒ $v_1^*=v_2^*$ ⇒ **intent-invariant 解同時極小化兩支、為 $L_p$ 之全域最優，且（資料支撐上、$p\in(0,1)$）population 最優解唯一＝invariant**（量詞修正＝A2u／丙 B5：端點 $p\in\{0,1\}$ 只剩「invariant 是最優之一」）。無 trade-off、$p$ 不動 argmin。
_Sketch_：$L^2$ 投影唯一性＋條件獨立（weak union）下條件期望塔性質。結構對應 Robinson〔轉引 2106.11230 Prop 2.2〕的「冗餘條件下棄用＝合法最優」（彼為 contrastive、此為 $L^2$ 迴歸；遷移直接，因兩者最優解皆由充分統計決定）。
**合法適用域（A1-lim 的物理真身；CT Cor CT-3）**：(a) 理論參考點（$\eta\to0$ 極限錨）；
(b) **自生／shuffled $u$ 訓練臂** — 該臂 $u\perp\tau\mid(s,g)$ **精確**成立、
$\eta_{\rm eff}\equiv0$ ⇒ 本命題與 2.3(i)(ii) 對它**逐字為真**：降級件廢物利用成控制臂
理論（預測：該臂配對差 $\equiv0$、d_zero→0；CT §5.2 P1）。⛔ 除此之外別當本資料敘事引用。
**$I>0$ 版（引用、不重抄）**：postA1 **Prop 2.1′**（invariant 次優）— C-ii′ 的 bits 半邊
（$I(z;a|s,g)\ge\delta'>0$）下，兩支條件期望在正測度上分離、invariant 解在 keep 支付出
**嚴格正**超額損失（量級＝$\delta'$；由 $z$-層資訊分解恆等式 — 取 $z$ 為本體變數時
Prop 1.3 形恆等式是等式、不需 enc 充分性，A3′ 的 $\le$ 只用於 $\tau$-層量測連 $z$-層的
方向）、不再是全域最優。其 $z$-層前提未直接量：`TODO(驗收)` $I(z;a|s,g)$ 量測（§0 C-ii′ 條）。

**Cor 2.2（guidance 無效；前提改標實測、與 A1 真偽無關）**：$\varepsilon\approx0$（⛔ 不再由 A1 推出 — 直接用 ⑬ 實測 B/A=1.1%）⇒ $\tilde v_w=v_\varnothing+w(v_a-v_\varnothing)\approx v_\varnothing\ \forall w$ ⇒ 生成分佈近乎與 $w$ 無關 — CFG 外插的是接近零的向量（⑪ 的 $.344{\pm}.089$ vs $.336{\pm}.089$）。量化版（丙 B6＋Prop 1.2）：外插項範數 $\le|w|\varepsilon$ ⇒ 終端分佈偏移 $\le\tfrac{e^L-1}{L}|w|\varepsilon$。

**Prop 2.3（A1-lim 極限情形：鎖死＝穩定駐點集；(i)(ii)＝A1-lim＋A2–A4、(iii)＝條件式與 A1 無關）**：
〔⛔ (i)(ii) 同 Prop 2.1 降級：對本資料非實況、對自生／shuffled-$u$ 臂逐字為真。
**$I>0$ 版（postA1 Prop 2.3′）**：invariant 流形**不是駐點集** — adapter 驅動
$\propto\mathbb E[{\rm Cov}(a,\delta|s,g)]=O(\sqrt{\eta_{\rm eff}})\ne0$；只是逃逸驅動
$O(\sqrt{\eta_{\rm eff}})$ 對上 ridge 回拉 $O(p\lambda)$、淨平衡在
$\mathrm{Int}^*=\eta/(\eta+p\lambda)$，且趨近速率為慢模 $\kappa$（「目的地小＋走得慢」）。〕
(i) *一階平坦*：設 $\theta\in M\equiv\{\theta:v_\theta$ 對 $a$ 逐點常值$\}$ 且 $\varnothing$ 支已最優。任意參數方向 $h$，其輸出擾動 $\partial_h v$ 分解為 $a$-對稱與 $a$-反變（$\mathbb E_a$ 均值零）分量；A1-lim 使 residual 對 $a$ 條件無關 ⇒ **反變分量與 residual 的期望內積為 0** ⇒ 打開 $a$-分辨的方向導數＝0。〔實測 $I>0$ 下此內積 $=O(\sqrt{\eta_{\rm eff}})\ne0$ — 「零驅動」失效、只剩「微驅動」。〕
(ii) *無逃逸曲率*：$M\cap\{\text{最優}\}$ 由 Prop 2.1 是全域最優 ⇒ Hessian 半正定、無負曲率離開方向 ⇒ 弱穩定駐點集。〔依賴 Prop 2.1 ⇒ 同其降級。〕
(iii) *塌後不再分化*〔結構借 Cocos 轉引 2505.11123 Thm 1；丙 B7 修正版**限定詞全收**（postA1 對齊）〕：**在 $\varnothing$ 支已最優＋$\varepsilon_{\rm cond}=0$＋A4、且只算 adapter 以下游的共享參數**時，兩支梯度差 $\le C\cdot\varepsilon$（活化全同⇒Jacobian 全同）⇒ $\varepsilon\approx0$ 自我維持。⛔ 無此前提不成立（丙反例：adapter $M{=}0$、$\varnothing$ 支未收斂時 $\partial L/\partial M\ne0$、通路會復活 — 輸出差控制不了 Jacobian 差）；**adapter 參數排除在外**（$I>0$ 下其驅動非零、見命題頭）；動力學的正確敘述是 R2.4 的賽跑。

**Remark 2.4（賽跑機制＝主敘事；⑰ 後升格、不依賴 A1）**：$a$ 通路的期望生長驅動 $\propto\|\mathbb E[a\otimes\delta]\|$，$\delta$＝residual。$p>0$ 使 $\varnothing$ 支被迫自建 $(s,g)$ 通路；其收斂令 $\delta$ 的 $(s,g)$-可預測分量 $\to0$ ⇒ 供能只剩 route 專屬分量 — **A1-lim 下恰為 0**（v0 原句）、**實測下為 $O(\sqrt{\eta_{\rm eff}})$ 級**（⑰：$z$-度量計價後 $\eta_{\rm eff}\approx0.003\lambda$）— 對上 $O(p\lambda)$ 級 ridge 回拉，幾乎枯竭但非零。鎖死＝$a$ 通路生長 vs $\varnothing$ 支收斂的賽跑落敗（f27n $p{=}0$ 無對手故長成 .6046）。**量化版＝丁 LT-6 overshoot 定理**（⑮ 機器精度驗訖；postA1 R2.4′ 升格）：cold-start 下 $a$-通路借 $\varnothing$ 支未收斂的 residual 先長（單峰、峰時閉式）、$\varnothing$ 支收斂後被壓回 $\mathrm{Int}^*$；$I>0$ 只改終值（$0\to\mathrm{Int}^*$）、不改賽跑形狀。
⚠️ **$\kappa T$ 校準未定（⑱'＋REVIEW §5）**：「8000 步＝$\kappa T\gg1$、債已還完」句已被 l_nf 直接否掉（8000→11429 步 $-1.90\to-2.18$ 仍在快掉 ⇒ 主通路遠未收斂）— 各批可能仍在 overshoot **峰前**（重校準假說、假說級）；判別＝中途 ckpt 散度曲線。⛔ 引用賽跑敘事時別帶「債已還完」與任何 $\kappa T$ 定量句（⑮ 該句連帶標疑）。

**Remark 2.5（資訊價值 ≠ 計算價值；v2：兩分量並立）**：f27n 的 $+.133$ 在 v0（A1 敘事）下只能全記「計算捷徑」（讀 $A(s,g)$ 比從 $(s,g)$ 重算便宜〔轉引 Autoguidance 2406.02507 §3〕）；⑰ 後其組成＝**資訊分量（Prop 1.3 正折扣、上限 ~2.5 bits 過匯率後的兌現）＋計算分量（捷徑）**並立、比例未拆 — 分辨儀器＝P-resample 三臂（oracle−resampled＝資訊、resampled−zero＝計算；CT §4.3、半套已在）。捷徑的動力學句保留：$p>0$ 強迫自算通路建成後、捷徑邊際價值歸零 ⇒ 塌。⇒ 本檔所有 population 命題仍須配有限容量註腳讀。

**Conj 2.6（$p$ 劑量曲線；v2 改述 — toy 內已閉式、真網路仍開放）**：v0 問「臨界值 $p^*$」；toy 解析已判（丁 Prop LT-1/LT-2；⑮ 機器精度驗訖）：**線性高斯內無 sharp 相變** — $\mathrm{Int}^*(p)=\eta/(\eta+p\lambda)$ 是平滑深度 crossover、半衰位置 $p_{1/2}=\eta_{\rm eff}/\lambda\approx0.003$（遠左於任何實際劑量）。仍 Conj 的：真網路（非凸 basin）是否存在 sharp 臨界值；有限 $T$ transient 修正的幅度 $A$（postA1 §4.3、Conj 級）。⛔ 劑量曲線的可證偽預測**以 postA1 §4 的 pre-registered 版為準**（四點平坦點估／比值封頂 $\mathrm{Int}(p_1)/\mathrm{Int}(p_2)\le p_2/p_1$／單調＋凸＋$\mathrm{Int}(.05)$ 單點分版；git 時戳先於資料）— 本檔不另立、不改寫。初讀狀態（⑱⑱'）：$p$ 軸平衡版全對（idp01 增益未回、與四點平坦一致）；$T$ 軸大效應**歸因未定** —「劑量不足枝亮（曝光/總步數未拆、該臂兩者 100% 共線）」、⛔「病=曝光量」與「輾過」皆撤回（⑱'）、首要嫌疑＝generic 訓練長度（l_nf 未收斂實錘）、拆解等 f27nL／N5L／idpxm×8。文獻空白照舊：影像圈 $p\in[0.1,0.2]$ 慣例僅驗樣品品質、未驗條件遵從。

## 3. 藥方的理論形（v2 權重：退火 $p$＋$L_{\rm div}$ 主、破冗餘輔）

_權重裁定（⑰；⑭ 三藥分工不變）：病因重心移到動力學（§2.0）⇒ **退火 $p$（動力學藥）＋
$L_{\rm div}$ hinge（結構在場藥）為主**；資料層破冗餘（資訊供給藥）**降為輔** — $I$ 已實測
$>0$、「供給」不再是第一瓶頸，其角色轉向 C-iii 效用可讀性與 $z$-度量權重（C-ii′ 後半；
§3.3），「=stitch 本義」的論述地位保留。_

### 3.1 退火 $p$＝warm-start（吸子語言）

**Prop 3.1（屏障擋「先塌後修」；A2＋門檻 $p\lesssim\tfrac14$（丙 B10 顯式化）；A1 不需）**：設 $\theta_A$（f27n 形：$a$ 支誤差 $\delta_a$ 小、$\varnothing$ 支誤差 $\delta_\varnothing$ 大）。任何先達 $\{\varepsilon=0,\ \varnothing\text{ 支誤差}\ge\delta_\varnothing/2\}$ 再修的連續路徑，中途損失 $\ge(1-p)(\delta_\varnothing/2)^2$ 級；屏障嚴格正（路點損失 $>$ 出發點損失）需
$$p\;<\;\tfrac14-(1-p)\,(\delta_a/\delta_\varnothing)^2\qquad(\Rightarrow p\lesssim\tfrac14),$$
⛔ **實際運行值 $p{=}0.3$ 在證明範圍外**（$\delta_a{=}0$ 都救不回；丙 B10）— 門檻顯式化恰成退火主張的定量依據：**小 $p$ 段有屏障、$0.3$ 沒有**。梯度流損失單調不增 ⇒（門檻內）從 $\theta_A$ 出發**不走「先塌後修」**；局部主梯度為修 $\varnothing$ 支（$\propto p\,\delta_\varnothing$），塌方向一階增益 $\approx0$（$a$ 支已近最優；實測 $I>0$ 下塌另付 Prop 1.3 正折扣 — $O(\eta_{\rm eff})$ 級、方向上更黏）⇒ 短期黏住。
**限制（誠實、即 ⑭「單獨不保險」的幾何形）**：「**同步塌修**」路徑（邊修 $\varnothing$ 邊縮 $a$-反變分量）可損失（近乎）單調直達 invariant 解 — A1-lim 下它是全域最優、實測 $I>0$ 下只差 $O(\eta_{\rm eff})$ 級上坡尾段（SGD 噪音尺度下無擋）— 屏障擋不住它。He+19 lagging-encoder〔驗 1901.05534〕同構：初期弱 posterior 引發塌、warm-start 改初值不改地貌；Cyclical-Annealing〔驗 1903.10145〕的「升回目標 $p$ 仍塌」＝此路徑實測存在的旁證。
**Conj 3.2（長時黏性未保證；v2 修 basin 高差）**：A1-lim 下兩 basin population 損失恰同高；實測 $I>0$ 下 invariant basin 高出 $O(\eta_{\rm eff})$ 級折扣（$\eta_{\rm eff}/\lambda\sim3\times10^{-3}$；postA1 §3.2）— **高差微小、SGD 噪音尺度下 v0 敘事近似保留**：長期佔據仍由 basin 體積/平坦度＋噪音主導（invariant 解疑更平坦 ⇒ 偏塌）、微小高差只給極長時間尺度的定向偏壓。判決實驗＝⑭ warm-start 測（黏住 ⇔ 塌回；WS 批已灑、產物在 day_0906/ — 收表掃描要含（⑰'））；判讀尺＝slaved 域斜率比 $\mu_-=2.759\pm20\%$ 帶（⑮ 收緊①；⛔ 不用 2.3 定值）。

### 3.2 cond 層散度 floor

**Def 3.3**：$L_{\text{div}}(\theta)\equiv\mathbb E\big[\max\big(0,\ m-\|c_\theta(s,g,a)-c_\theta(s,g,\varnothing)\|\big)\big]$，$m$＝margin（標度建議錨 f27n cond 層差 .6046；實作註）。⚠️ 標度換算（丙 B12）：$m$ 是**絕對**距離、⑬ 的 .6046 是**相對**差 — 實作要寫換算句、照抄會錯尺；細則屬 DESIGN 管區。

**Prop 3.4（可行集手術；A1-lim＋A2'）**：對 $L_p+\lambda L_{\text{div}}$，$\lambda>0$：任何 invariant 解付 $\lambda m$；而存在「$c$ 層附加 $a$ 的單射座標、下游忽略之」的解達 $L_p$ 最優且 $L_{\text{div}}=0$ ⇒ **invariant 解被逐出全域最優集**、新最優集 $\subseteq\{\varepsilon_{\text{cond}}\ge m\}$。〔v2 註：兩半論證（invariant 付全額 hinge；附加座標構造）形式上皆不用 $\tau\perp a$ — 對 $I>0$ 應同型成立（從任一 $L_p$ 最優解出發附加座標即可）；丙 B13 只在 A1 語境檢過 ⇒ $I>0$ 版嚴格重檢 `TODO(驗收)` 交 Rei。實務上 $I>0$ 不減手術的必要性：動力學仍塌向 invariant（§2.0）、手術給的是結構保證。〕
**Remark 3.5（結構藥 vs 內容藥；v2 權重註）**：手術只保 $\varepsilon_{\text{cond}}\ge m$、**不保 $\varepsilon_v>0$**（A1-lim 下 population 最優仍可 $v$-invariant；實測 $I>0$ 下 $v$-invariant 雖非嚴格最優（postA1 Prop 2.1′）、超額損失僅 $O(\eta_{\rm eff})$ 級 — 下游「幾乎合法」地忽略被撐開的座標，警告不減反增）。⇒ $L_{\text{div}}$ 的作用＝保「可喚醒性」（通路物理在場：guidance 有物可外插、warm-start 的 $a$ 支不被物理拆除）；**製造依賴**在 v0 記給 §3.3 的資訊供給 — ⑰ 後資訊已在場、改記給**動力學藥（退火 $p$）＋匯率抬升（$z$-度量權重、C-ii′ 後半）**。三藥三層的分工不變（⑭ 雙保險＋⑰ 權重）：退火 $p$ 保動力學路徑【主】、$L_{\text{div}}$ 保結構在場【主】、資料層保效用可讀與匯率【輔】— 彼此不冗餘。margin 放 cond 層＝正中 ⑬ 定位的塌陷格（且 cond 層低維、margin 可選；速度場層 margin 會直接扭曲生成）。
**Remark 3.6（塌陷點可重啟）**：$\|x\|$ 在 $x=0$ 的 subgradient 為單位球 ⇒ hinge 在 exact 塌陷點提供非零逐出力；cos/乘積型正則在 0 梯度消失（塌陷點反成駐點）⇒ margin-hinge 是形式上正確的選擇。

### 3.3 資料層多路線（v2：自「根治」降為輔藥；stitch 本義保留）

形式條件：**C-i（多路線）** 存在 route 變數 $R=f(\tau)$、$H(R|s,g)\ge h>0$ 於正測度 $(s,g)$
— **⑰ 後：本資料已實測部分成立**（$H(a|s,g)\approx2.5$ bits、路線選擇級 ≈1 bit（⑰'）；
modal 路線佔比中位 .286、決定程度 .749）；**C-ii（錨資訊性）** $I(R;a|s,g)\ge\delta$ —
hindsight 錨（$a$ 為 $\tau$ 的決定函數）下自動成立；**C-ii′（$z$-層可讀＋變異權重；§0、
丙 B15）** — **新瓶頸**：encoder 若把 route 資訊的變異權重壓到 ~0.4%（⑰ Z2），資料層
破了冗餘、梯度照樣挖不動；**C-iii（效用可分）** eval 對不同 $R$ 產生可讀差異。

**Prop 3.7（鎖死前提拆除；C-i＋C-ii′）**：$R=f(\tau)$ ⇒（DPI）$I_{\text{data}}(\tau;a|s,g)\ge I(R;a|s,g)\ge\delta>0$ ⇒ A1-lim 失效：Prop 1.3 給出正折扣；配 **C-ii′（$z$-層半邊）**後兩支條件期望在正測度上分離（Prop 2.1 唯一性失效；閉合路徑＝CFM 最優＝條件資料分佈＋「場同⇒分佈同」反證、丙 B15）、Prop 2.3(i) 的方向導數轉負（打開依賴有一階增益）⇒ **最優性與零驅動兩根支柱同時拆除**。⛔ 只有 C-i＋C-ii（$\tau$-層）**不夠** — DPI 方向幫不上 $z={\rm enc}(\tau)$：enc 丟掉 route 則資料層破了冗餘、模型層兩支照樣合流（丙 B15 的縫；e_target 設計要點）。
**Remark 3.7-b（v2 地位改讀：從「根治藥」到「診斷＋輔藥」）**：⑰ 證實兩根支柱在本資料**已經拆除**（$I>0$、驅動 $O(\sqrt{\eta_{\rm eff}})\ne0$）而鎖死照樣發生（⑬）⇒ 反證病灶不在支柱、在**動力學＋匯率**（§2.0）。Prop 3.7 因此轉為：(a) **診斷** — 它列的前提哪半滿足哪半缺：C-i ✓（⑰）、C-ii ✓（hindsight）、C-ii′ 的 $\eta_{\rm eff}$ 半邊 ✗（⑰ Z2 ~0.4%）；(b) **輔藥** — teleport／counterfactual 多路線構造仍值得：抬 $H(R|s,g)$ 更高、供 C-iii 可讀評測，與 **stitch 本義（組合出沒見過的路）**及 ⑫⭐ 合流（此論述地位 ⑰ 不動）；另與合成律的非平凡性同源（多路線 ⇔ $w^*(m|s,g)$ 非退化；丙 D4 — ⑰ 實測 modal 佔比 .286 ⇒ $\oplus$ 非平凡在本資料已在場）。
**Remark 3.8**：C-iii 獨立必要 — 無它則 $I>0$ 而 $\Delta_{\text{succ}}$ 恆 0（⑦ subgoal 格 intent 價值 $\approx0$ 的形式原因＝eval 只問終點不問路線；亦即 Def 1.4 分母條款的另一鏡頭）。teleport／counterfactual 構造＝直接抬 $H(R|s,g)$。⑰ 後 **C-iii 與 C-ii′ 並列為多路線工作的主要缺口**（C-i 已部分在場）。

## 4. 路線一幾何的相容性（DESIGN v2 的命題化）

**Prop 4.1（子空間相容存在性；A5）**：時間反演 rev 為對合（$\mathrm{rev}^2=\mathrm{id}$）⇒ 函數空間 $\mathbb Z_2$-分解 $F=F_+\oplus F_-$（$\varphi_\pm=\tfrac12(\varphi\pm\varphi\circ\mathrm{rev})$）。取 $e_d\in F_-$、$e_m\in F_+$，則：(i) $e_d(\mathrm{rev}\,\tau)=-e_d(\tau)$ ⇒ 凡 $\|e_d\|>0$ 處 $\cos=-1$ 恰好達成；(ii) $e_m$ rev-不變＋A5 ⇒ rank 幾何在 rev-商空間上良定義、與 (i) 作用於互補子空間 ⇒ **兩約束無聯立矛盾**；(iii) $\varphi=\varphi_++\varphi_-$ 可重建 ⇒ 分解不丟資訊 — **recon 不犧牲的條件**＝$D_m\ge\dim_{\text{eff}}(\varphi_+)$、$D_d\ge\dim_{\text{eff}}(\varphi_-)$、decoder 讀 concat。非平凡性條件：$\tau\ne\mathrm{rev}\,\tau$ 於正測度（非回文軌跡）⇒ 非零反對稱分量存在。
**Cor 4.1'**：N1（rev 對 rank/rev 聯立矛盾）在子空間版**結構性消滅**：rev 對的 $e_m$ 距離為 0 與 $d(\tau,\mathrm{rev}\tau)=0$（A5）一致、拉遠只發生在 $e_d$ — 排池從「必要」降為「雙保險」（DESIGN 保留之、相容）。
**Remark 4.2（D_d 預算；v2 上修 — ⑰ Z2 能量譜已量）**：C1 實測 $\cos=+0.9901$（中位）⇒ 現況 $e$ 幾乎全落 $F_+$；但 ⑰ Z2 雙峰拆穿了中位數：$\|\varphi_-\|^2/\|e\|^2$ 整體中位僅 0.37%、**長窗（>61 步）中位 21.8%、cos +0.56** ⇒ 方向（反對稱）資訊住在長窗、短窗把整體讀數壓低（四分位連續、儀器 ✓ — C1 的中位數掩蓋了寬分佈）。$\dim_{\text{eff}}(\varphi_-)@90\%=29>16$ ⇒ 由 Prop 4.1(iii) 的重建條件 $D_d\ge\dim_{\text{eff}}(\varphi_-)$，**$D_d{=}16$ 偏小 — 上修至 ≥29 級、或按窗長分配**（短窗小／長窗大；⑰ 原建議「加大或分窗長」）。`TODO(驗收)`：具體選值（29 是 @90% 讀數、更高分位未量）與分窗方案 — 屬 DESIGN 管區、此處只立理論含義（現行預算違反重建條件）。$\|e_d\|\to0$ 使 cos 病態 ⇒ 實作需範數下限（DESIGN 已列尺度捷徑風險）不變。

**Prop 4.2（$d_{\text{time}}$ 排序一致性；條件式）**：設行為策略使 **B1（條件單調）** $\mathbb E[\Delta t\,|\,d_{\text{bfs}}{=}k]=f(k)$ 嚴格增 — 漂移主導 $f(k)\!\approx\!k/v$、純擴散 $f(k)\!\approx\!k^2$，兩極皆單調 ⇒ B1 弱；**B2（gap 集中；丙 B19 修正版）** $\mathbb P\big(|\Delta t-f(k)|>\tfrac12\min\big(f(k){-}f(k{-}1),\,f(k{+}1){-}f(k)\big)\big)\le q$ — 半 gap 取**相鄰兩 gap 的 min**（單邊版在凸 $f$（擴散端、gap 遞增）有縫：兩事件皆好仍可翻序；取 min 後界恢復、且恰把「長程變異爆」的破壞者寫進假設）。則三元組排序錯誤率 $\le 2q$ ⇒ population 級序一致。**破壞者是變異非單調性**：純擴散 first-passage $\mathrm{Var}\sim k^4$ ⇒ 長程 $q$ 爆 ⇒ 長 $\Delta t$ 段失效。
**實證錨（C2、[實測]）**：整體 $\rho=.758$＋反向三元組 $13.4\%$（換算修正（丙 B19）：三元組率按 $\le2q$ 應推 $\hat q\gtrsim.067$、⛔ 別把三元組率直讀成 $q$）＝B1+B2 於本資料尺度成立；最弱箱 $.203$＝長 $\Delta t$ 的 B2 破損實錄 ⇒ DESIGN「短 $\Delta t$ 段＋長程交 $d_{\text{bfs}}$ 臂」＝控 $q$ 協定的形式對應。跨軌跡外插屬 A6（未驗）— 交叉形式評估（$d_{\text{time}}$ 訓 → $d_{\text{bfs}}$ held-out 評；N2 解法）即其測試。

## 5. 誠實邊界（分級與升級路徑）

| 條目 | 級別 | 關鍵假設 | 升級需要 |
|---|---|---|---|
| Prop 1.1／1.2 | Prop | A3-類 Lipschitz；K-R 橋另需 H1 | H1 對 refine 管線是否成立 — **要 Rei 檢**（env rollout 使 $h$ 非 Lipschitz；$\rho(\eta)$ 可由 eval 資料估）。⑰ 註：望遠鏡常數腿依賴（$\Delta/\varepsilon$ 斜率 15×）— $\varepsilon$ 只當機制錶、不跨腿定量 |
| Prop 1.3 | Prop（恆等式轉引；A3′ 上限語意、丙 B3 落） | population 最優、marginal-path 唯一性；enc 充分性缺 ⇒ 退 $\le$ | 讀 2310.07972 原件確認 Eq.4 形式 |
| Def 1.4 | Definition | $\theta_{\text{ref}}$ 的選取 — ⑱' 後 f27n@8000 **已失效** | ref／基線換 step-matched 長訓家族（f27nL／N5L）＋ep200 錨 — `TODO(驗收)` 收表後釘（§1.4 實務註） |
| Prop 2.1 | **Prop（A1-lim 極限錨；⑰ 後非本資料實況）** | A1-lim＋A2＋A2u（丙 B5 落） | 適用域＝$\eta\to0$ 錨＋自生/shuffled-$u$ 控制臂（CT-3）；$I>0$ 版＝postA1 2.1′、其前提補完＝$I(z;a|s,g)$ 量測 `TODO(驗收)` |
| Cor 2.2 | **Prop（前提改實測 $\varepsilon$、與 A1 無關）** | ⑬ 實測＋A3 | —（v2 後地基最穩的一塊） |
| Prop 2.3 | (i)(ii)＝A1-lim 極限錨；(iii)＝條件式 Prop（丙 B7 限定詞全收） | (i)(ii)：A1-lim＋A2–A4；(iii)：$\varnothing$ 支最優＋$\varepsilon_{\rm cond}{=}0$＋A4、adapter 排除 | (i) 量詞丙檢畢（weak union 寫法即嚴）；讀 Cocos 原件；$I>0$ 版＝postA1 2.3′ |
| Remark 2.4 賽跑 | **Remark→主敘事**（丁 LT-6 定理背書、⑮ 驗訖） | 線性化；toy→真網路＝量級指引 | $\kappa T$ 校準（⑱' 標疑）＝中途 ckpt 散度曲線判 |
| Conj 2.6 劑量曲線 | **Conj**（v2 改述：toy 已閉式、無相變；真網路開放） | 預測以 postA1 §4 pre-registered 為準 | f27nL／N5L／idpxm×8 拆 generic-$T$；$\mathrm{Int}(.05)$ 分版點無臂（待裁） |
| Prop 3.1 | Prop（限定範圍＋門檻 $p\lesssim\tfrac14$、丙 B10 落） | 只擋「先塌後修」；$p{=}0.3$ 範圍外 | — |
| Conj 3.2 黏性 | **Conj**（v2：basin 高差 $O(\eta_{\rm eff})$ 級微小） | basin 幾何未知 | WS 批已灑（day_0906/）— 判讀尺 $\mu_-=2.759\pm20\%$（⑮） |
| Prop 3.4／R3.5 | Prop（v2 權重註：主藥之二） | A2' | $\varepsilon_{\text{cond}}\ge m$ 下游是否真被用＝實驗題（idp+div 臂） |
| Prop 3.7 | Prop（前提改 C-i＋C-ii′、丙 B15 落；v2 地位＝診斷＋輔藥） | DPI＋轉引恆等式＋C-ii′ | C-iii 的 eval 改造（route 可分判準）尚無設計；$\eta_{\rm eff}$ 半邊＝e_target 度量工作 |
| Prop 4.1／Cor 4.1' | **Prop（構造性、嚴格）** | A5 | 能量譜探針**已量**（⑰ Z2）：$\dim_{\rm eff}(\varphi_-)@90\%{=}29$ ⇒ $D_d$ 上修（R4.2）落 DESIGN |
| Prop 4.2 | Prop（條件式；B2 修正版、丙 B19 落） | B1+B2(min-gap)＋A6 | A6＝conjecture 級 — 交叉評臂判 |

全域註腳：本檔所有 Prop 為 **population 級**；有限容量效應（Remark 2.5 的 f27n $+.133$）證明 population 敘事單獨不完整 — paper 理論節建議「population 定理＋有限容量 remark」雙層誠實寫法。
v2 追加：A1-lim 命題（2.1、2.3(i)(ii)）在 paper 裡的合法用法只有兩種 — 理論參考點（$\eta\to0$ 極限）與自生／shuffled-$u$ 控制臂（該臂逐字為真；CT-3）。⛔ 別當本資料敘事引用；本資料敘事一律走 §2.0（動力學陷阱）＋postA1 §2。

## References

轉引（docs 已引、未讀原件）：Cocos 2505.11123（Thm 1）；Robinson 2106.11230（Prop 2.2）；資訊分解 2310.07972（Eq.4）；Autoguidance 2406.02507（§3）。
本次驗過 ID：CFM — Lipman et al., arXiv:2210.02747（ICLR'23）；CFG — Ho & Salimans, arXiv:2207.12598；lagging encoder — He et al., arXiv:1901.05534（ICLR'19）；cyclical annealing — Fu et al., arXiv:1903.10145（NAACL'19）。
教科書級：Pinsker；Grönwall；Kantorovich–Rubinstein 對偶；$\mathbb Z_2$ 表示分解；first-passage 矩（漂移 $\mathbb E\sim k$／擴散 $\mathbb E\sim k^2,\mathrm{Var}\sim k^4$）。
