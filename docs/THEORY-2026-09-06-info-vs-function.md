# THEORY — info vs function：權重通道的縱切刀（v0、2026-09-06）

_理論整合使魔（Fable 級、主人晨間指定）。上游（唯讀）：【帳本】THEORY-0906-information-ledger、
【分類】NOTE-0906-context-taxonomy、【敘事】THEORY-0906-verifier-taught-latent-thinking、
【一般化】DESIGN-0905-general-internalization-claim、【合成律】THEORY-0905-composition-law-draft；
佐讀：【設計卡】DESIGN-0906-grpo-thoughts、【多路線】DESIGN-0906-multiroute、【postA1】
THEORY-0906-postA1-revision、【內化】THEORY-0905-internalization-formal。
⛔ 分級鐵則：**定理級＝證明在檔（多為一行）或逐字引已證件；命題級＝列明假設下證梗概在檔；
【推測級】＝方向、未證**。外部文獻只沿 repo 已攜帶者引用、攜帶級照抄；無現成引文支撐的論斷
一律標【推測級】。節號 F\*（獨立編號、避開帳本 L\*）；本檔＝帳本續章候選（未來可併為 L6）。
本檔不改上游、不碰 experiments/ 與 results/；接口矛盾集中列檔尾「接口注意」。_

> **主人定調（原話逐字）**：「我們的u看起來有想法了，沒有外界資訊的icl其實怎麼樣也不會得到更多資訊，要靠verifier或者學到肌肉記憶，或者學會一些預設獲取資訊的能力，這可能嗎，例如學到的不是info而是function」

---

## F0. 一句話＋這把刀切在哪

> **帳本管「bits 從哪個通道進來」— 橫軸。本檔補縱軸：「進到權重裡的東西是什麼」。
> 陳述性 info＝與特定環境變數的互資訊（以 bits 計價、按環境數線性耗預算、換環境重抽即蒸發）；
> 程序性 function＝條件分佈核／算子本身（住提取赤字欄、不耗 W-bits 預算、fresh 環境上仍在，
> 其定理級後果＝在新環境把活通道的 bits 兌現成表現的能力）。權重通道 (i) 是唯一兩種都存的
> 通道，而帳本現版只按 $I(W;\theta)$ 記帳 — 只看得見 info 那份；function 那份住在每一次未來
> 部署的赤字欄裡，四通道記帳對它天生失明。這就是「沒切開的一刀」的形式位置。
> 主人句三個猜想各得座標：verifier＝info 的油箱兼 function 的老師（L4.5 防火牆＝這把刀在
> GRPO 上的既有切口，F3.3）；「肌肉記憶」＝兩者皆可、由重抽階梯分辨（F1.3）；「預設獲取
> 資訊的能力」＝acquisition function，可學、但訓練分佈要先讓「不知道」存在（F3.2）。**

**2×2 座標（content × 存放）**：

| | persistent（θ） | transient（推論期） |
|---|---|---|
| **info**（bits） | 記住的 route／地圖 — 幫浦 L4 打進去的 world-bits 份 | conditioning／verifier 選擇／觀測的當場 bits（帳本 (ii)(iii)(iv)） |
| **function**（算子） | 攤銷的算子（BFS／組合／檢查）＝核心 claim 的宣稱 | 推論期計算（search／BoN／refine；帳本 Thm L5 四格） |

帳本四通道全在 info 列記帳；function 列在帳本裡只有赤字欄一個影子。GRPO 幫浦＝同時在兩列
搬東西的搬運工（L4.5 已把它的輸出拆成 world-bits＋sharpening 兩份 — 那正是本檔這把刀）。

---

## F1. 形式區分

### F1.1 兩級層級模型（環境 ensemble）

**Def F1（設定）**：family 參數 $\rho$；環境 $e\sim\rho$ iid、world 變數 $W_e$（佔據圖 $E_e$、
BFS 場 $D_e$、episode 資料側變數 — 帳本 Def L1 的 $W$、外加環境索引）；查詢 $(s,g)\sim\mu_{W_e}$、
答案 $\tau\sim p^*(\cdot\mid s,g,W_e)$；訓練資料 $\mathcal D$＝環境 $e_1..e_m$ 的 episodes、
$\theta=\mathrm{Alg}(\mathcal D)$。部署通道束 $C$（可空）：$c=C(W_e,s,g,\dots)$。
假設：**F-A1（fresh）** $e'$ iid、獨立於 $\mathcal D$ 與 Alg 噪音；**F-A2（查詢正則）**
$(s,g)$ 取樣機制跨環境同構（maze：共用格框），使條件版獨立性論證成立；**F-A3（eval 新鮮）**
eval episode 不在 $\mathcal D$ 內。現況＝$m{=}1$ 的退化（Remark F1.5）。

**Def F2（info content）**〔定義級〕：$\theta$ 的 instance 級 info＝$I(W_{e_j};\theta)$
（對各訓練環境）；family 級 info＝$I(\tilde\rho;\theta)$（把 $\rho$ 視為隨機時 — $\rho$ 固定
則此級以「對常數的知識」存在、任何 MI 帳都看不見；這正是【分類】§2.2(a) 行「task 識別
bits」的正式身份）。以 bits 計價、受權重通道帽 $I(\mathcal D;\theta)$（帳本 (i)；框架錨
Xu–Raginsky 1705.07809、沿帳本【訓練記憶】）。

**Def F3（function content）**〔定義級〕：對通道束 $C$，
$$\Phi_C(\theta)\;:=\;\mathbb E_{e'\text{ fresh}}\Big[\mathrm{KL}\big(p^*_\rho(\tau\mid s,g,c)\,\big\|\,q_\theta(\tau\mid s,g,c)\big)\Big]$$
＝**fresh 環境上的提取赤字**（$p^*_\rho(\cdot\mid s,g,c)$＝對 $W_{e'}\mid s,g,c$ 邊際化的
Bayes 預測）。低＝function 多。不以 W-bits 計價（fresh 下 MI 恆 0、Prop F1）；計價貨幣＝
計算／描述長度。⭐ 注意：這**不是**把 function 定義成「會在新環境 re-derive info」— 那是
它的定理級後果（Cor F1.4），定義本身是「$q_\theta$ 作為固定 kernel 在環境重抽下的赤字」。
好處：$\Phi_\varnothing$ 也良定義（純邊際提取力 — 例：走廊先驗的盲規劃），
「re-derive」版在無通道時會失去定義。

### F1.2 三條基礎命題

**Prop F1（fresh 零資訊＝地板不可動）〔定理級、一行〕**：F-A1 ⇒ $W_{e'}\perp\theta$ ⇒
$I(W_{e'};\theta)=0$；且對任何 $C$：
$$\mathbb E\big[-\log q_\theta(\tau\mid s,g,c)\big]\;=\;\underbrace{H_\rho(\tau\mid s,g)-I(\tau;c\mid s,g)}_{\text{fresh 地板：只有活通道動得了}}\;+\;\Phi_C(\theta),$$
$\theta$ 只出現在 $\Phi_C$。⇒ fresh 環境的資訊地板由 $\rho$ 與部署時活著的通道決定，
**θ 裡存了什麼都動不了它** — Thm CT-1 的環境級同款、量詞同樣是「對任何 $W_{e'}$ 恆 0」。
_證_：獨立性＋帳本 Prop L0 分解逐字（$W:=W_{e'}$）。∎
⇒ 主人句前半「沒有外界資訊的 icl 怎麼樣也不會得到更多資訊」在環境級的定理形：**權重裡的
存貨對新環境不是資訊** — 部署時能當資訊用的只有當場進來的 (ii)(iii)(iv)。

**Prop F2（記憶化缺口恆等式＝info 的操作型定義）〔定理級〕**：訓練環境重查
（$E\sim\mathrm{Unif}\{e_1..e_m\}$、$W:=W_E$、fresh query，F-A3）與 fresh 環境各自分解：
$$\mathrm{NLL}_{\rm train}=H_\rho(\tau\mid s,g,c)-I(\tau;\theta\mid s,g,c)+\Phi^{\rm train},\qquad
\mathrm{NLL}_{\rm fresh}=H_\rho(\tau\mid s,g,c)-0+\Phi_C(\theta),$$
$$\Rightarrow\quad \mathrm{NLL}_{\rm fresh}-\mathrm{NLL}_{\rm train}
=\underbrace{I(\tau;\theta\mid s,g,c)}_{\text{info 項（洩進 }\theta\text{ 的 instance bits）}}
+\underbrace{\big[\Phi_C(\theta)-\Phi^{\rm train}\big]}_{\text{function 漂移項}}.$$
generalization gap **恰好**拆成兩項；「info＝訓練環境上被 $I(\tau;\theta\mid\cdot)$ 融資的
那部分表現、且恰好是換環境時蒸發的那部分」是恆等式不是比喻。
_證_：交叉熵分解兩次（train 側條件含 $\theta$：$q_\theta$ 是 $\sigma(\theta,s,g,c)$-可測；
$H(\tau\mid s,g,c,\theta)=H(\tau\mid s,g,c)-I(\tau;\theta\mid s,g,c)$）＋F-A1 使 fresh 側
$I=0$＋iid 使兩側邊際地板同值。∎
（註 1：兩個 $\Phi$ 各對自己層級的可達預測子取 KL — train 側目標是 $p(\cdot\mid s,g,c,\theta)$
— 恆等式因此精確。註 2：$\rho$ 隨機版：fresh 側殘餘 $I\ne0$ 的部分＝family 資訊 — 三層樓
自動出現、與 Def F2 對齊。）

**Prop F3（通道兌現效率 $\xi$）〔定理級 identity＋命題級量程〕**：定義
$$\xi_C(\theta)\;:=\;\frac{\mathrm{NLL}(\theta;\varnothing)-\mathrm{NLL}(\theta;C)}{I(\tau;c\mid s,g)}
\;=\;1+\frac{\Phi_\varnothing-\Phi_C}{I(\tau;c\mid s,g)}.$$
恆等式由 Prop F1 用兩次相減得〔定理級〕；$\Phi_\varnothing\approx0$（邊際側已學好）時
$\xi=1-\Phi_C/I\in(-\infty,1]$、讀作「通道許可的 bits 被程序兌現的比例」〔命題級量程〕。
**Cor F1.4（「re-derive info」的正式版）**〔定理級組合〕：fresh 環境上任何模型的價值分解＝
（當場通道供給的 bits）×（兌現效率）＋（邊際提取力）：$\mathrm{NLL}_{\rm fresh}(\varnothing)-
\mathrm{NLL}_{\rm fresh}(C)=\xi_C\cdot I$ — **function 的「在新環境 re-derive info 的能力」＝
$\xi_C\to1$ 對 family 一致成立**，是定義的後果、不是定義。
儀器註：idp on/zero 雙腿＝$\mathrm{NLL}(C)/\mathrm{NLL}(\varnothing)$ 現成、$I\approx2.5$
bits/題＝⑰（轉引 FINDINGS-0905）⇒ $\xi$ 在訓練圖上今天就可算。⚠️ C-ii′ 匯率斷裂（沿
postA1）＝「$\xi$ 不由 $I$ 預測」的既有量級版：bits 是有沒有錢、$\eta_{\rm eff}$ 是 CFM 的
變異度量下挖不挖得動 — $\xi$ 的變異面就是 $\eta_{\rm eff}/\lambda$，兩帳並存不換算。

**Remark F1.5（單環境退化＝這把刀在現資料上不存在）〔定理級（定義性）〕**：$m{=}1$ ⇒
$W_{e_1}$ 是常數 ⇒「這張圖的 info」以對常數的知識形式併入 effective-$\rho$，**任何 MI 帳都
看不見它**（MI 對常數恆 0）；info/function 之刀在數學上只在環境重抽存在時存在。後果：
（i）帳本現版沒切這刀不是疏漏 — 單圖 régime 下刀無處落；帶環境 ensemble、刀才隨之誕生。
（ii）現有 Int／idp 讀數天生不分「攤銷 BFS vs 背路線」— 兩者在單圖上住同一顆 θ、行為可
完全重合；只有 V₂ 級測試（新圖、F1.3）分得開。
（iii）⚠️ 但 V₁ 級（同圖新 $(s,g)$ 組合＝stitch 本義）已部分可分：查詢級重抽下查表死、
組合算子活 — 這是現有證據真正站的台階（F2 表）。

### F1.3 重抽階梯（function 的分級；「肌肉記憶」歧義的拆法）

**Def F4（value profile）**〔定義級〕：$V_0$＝見過的查詢／$V_1$＝同環境 fresh $(s,g)$
（stitch）／$V_2$＝同 family fresh 環境／$V_3$＝fresh family。模型的 value profile＝各級
保留值。**純 info＝值集中 $V_0$；$k$ 級 function＝值平到 $V_k$**。查表 $\,(s,g)\mapsto$route
在平凡意義下也是「函數」— 分級正是它與程序的差：查表的值在支撐外死、BFS 沿階梯活。
主人的「肌肉記憶」由此拆：背下的路線＝$V_0$–$V_1$ 部分肌肉；攤銷的搜索反射＝宣稱到
$V_2$ 的肌肉。**中間態＝非退化的 decay profile**（不是二值）— F5 的量法。

---

## F2. 我們架構各部件的歸類（對接核心 claim）

| 系統件 | 縱軸歸類 | 依據 | 證據現況（誠實） |
|---|---|---|---|
| **攤銷 BFS**（一般化內化 claim ⓪） | persistent **function** 的宣稱 | 「內化＝攤銷任意 teacher 的**計算**」（【一般化】原話）＝主張搬進 θ 的是算子不是 bits — **核心 claim 本來就是 function claim，本檔明寫這個對接** | ER route≈hindsight .918＝teacher-agnostic（另一軸、已證）；stitch＝$V_1$ 支持「本圖上的組合算子」；**$V_2$（environment-generic）一格證據都還沒有** — F-frame 第一次把這半句變可證偽（F5） |
| 特定 maze 的 route 記憶 | persistent **info** | Def F2；⑰ 的 ~2.5 bits/題＝它的 per-query 面值（轉引） | $m{=}1$ 下隱形（Remark F1.5）— 與攤銷 BFS 住同顆 θ、現讀數分不開 |
| **e_target 幾何** | function 的**載體** | 幾何是算子跑的資料結構（合成律 DP 的度量底座；路線一「z 走直線＝環境走最短路」）。⚠️ 拆兩半：「**一張圖的幾何**」＝info；「**圖→幾何的 encoder 算子**」＝function — 載體宣稱指後者 | encoder 在 fresh 圖上現算幾何、且距離仍追該圖 BFS 距離＝載體遷移（FP-3）；未測 |
| **u 的計算／結構價值**（分解、frontier 疊加） | function 的**執行痕跡** | 帳本 Thm L5 四格全是程序執行：分解＝跑 (CL) 的 DP；提取＝serial depth 兌現 θ 存量；前沿疊加＝平行 BFS 的工作記憶（Zhu+ 2505.12514、沿分類[驗]）；查詢規劃＝acquisition function 的計算（F3.2） | ⑬ P-swap／P-resample 儀器現成（分類 §4.3）＝執行痕跡 vs 資訊的操作型分辨器 |
| verifier $M_V$ | info 油箱 **兼** function 老師 | 雙身份：帳本 (iii) 供 bits；F3.1 教「檢查」算子 | L4.5 已把幫浦輸出拆兩份＝雙身份的既有實錘 |
| GRPO 幫浦 | **雙列搬運工** | F3.3 | 設計卡 rung 0–4 未跑 |

**「自生 u 零資訊」在本框架下的正確讀法**〔Remark、定理背書〕：Thm CT-1／Prop L1 說 u 恆
攜帶 0 W-bits — 帳本語氣裡這是「壞消息」（不合法期待清單 x1–x5）。縱軸上它是**本義**：
u 不是 info 通道、**是 function 的 scratchpad** — 工作記憶的定義性質就是不含外界資訊
（會漏 bits 進來的就不是 scratchpad、是通道，得按 (ii)(iii)(iv) 記帳）。CoT 圈的 filler-token
結果（2404.15758、沿分類[驗]）早就是同一句：thought 的內容可以零資訊、價值在計算。
Cor CT-3 的縱軸讀法＝**function 鑄造不了 info**：scratchpad 回填不了訓練通道 — 自生配對
建不了通道（$\eta_{\rm eff}\equiv0$）是「程序不能無中生 bits」的訓練側定理。
⇒ 對 u 的正確期待從此不在 bits 軸上：**問 u 的問題不是「它知道什麼」、是「它替哪個算子
跑了哪一段」**（L5 四格＝合法選單）。本段是既有定理的重讀、無新主張〔定理級引用＋Remark〕。

---

## F3. 「學會預設獲取資訊的能力」可能嗎（主人句的正面攻堅）

### F3.1 verifier 內化＝學到「檢查」這個 function

兩個不同的內化標的、⛔ 別混：

- **(a) 檢查算子 check(u, map-bits)**：吃活通道餵進來的環境 bits（(ii) 的佔據錨／(iv) 的觀測）、
  輸出合法性判定。學會它＝部署期 (iii) 可以自營 proposal-check 迴圈（BoN 不再需要外掛
  verifier 程式），**但 map bits 仍要當場從外面來** — 資訊帳一字不動：每選擇 $\le\log N$
  （L2(i)）、存量帽變成 $I(W;\text{活通道})$。學到的是**消費能力（function）、不是 bits**。
  fresh 圖上它能 re-derive「這條路對不對」＝ $\xi_{(iii)}\to1$ 的字面義。
  〔命題級：check 對我們域＝佔據圖查表＋逐步鄰接檢查、有限程序、可表示性平凡；
  網路實作與可學性未證〕
- **(b) 自驗證（$M_V$ θ-可測）**：L2(iii) ⇒ **0 新 bits**〔定理級、逐字引帳本〕。其真實
  價值＝把判別能力蒸回生成器＝提取赤字改善。**與 sharpening 文獻的對接**（2412.01951、
  沿帳本 L4.5【隊友正文級】）：sharpening 講的正是靠 verification–generation gap **提取**、
  不能創造 — 翻成 F-語言＝「內化 check-function 只買赤字欄、買不到地板」。主人問「可能嗎」
  的第一答：**可能；而且帳本早已證好它買得到什麼（赤字／function 側）與買不到什麼（地板／
  info 側）**；兩份的分辨照用 L4.5 指紋（大 $k$ 反超 base＝純 sharpening 簽名）。
  〔帳目＝定理級；sharpening＝check-function-內化的對接解讀＝命題級〕

### F3.2 active query／explore policy 可不可學

acquisition function $\pi_q$：belief → 該問什麼（EIG argmax＝帳本 L3 Remark 給 u 的合法
角色、L5 第 4 格）。它是 function（跨環境的**問法**、不是任何圖的 bits）。可學條件：

- **Q1（訓練分佈要含不確定性）〔命題級、退化證梗概〕**：$m{=}1$ ⇒ 訓練全程 belief 對
  $W_e$ 退化為點質量 ⇒ EIG 泛函恆 0 ⇒ 任何 reward 對「查詢品質」的梯度恆 0 —
  **沒有東西曾經未知，就學不會問**。外錨＝BARL Thm 4.1（2505.20561、沿帳本【隊友正文級】）：
  最優 policy 不依 belief 時，測試期探索零誘因 — 同一判的 RL 版。⇒ 必要條件＝**多環境＋
  部分可觀測**訓練（belief 要承重）。
- **Q2（query 要有 reward 訊號）〔命題級→實證級〕**：epistemic 行動要在訓練 horizon 內
  因果影響 return（credit assignment 到「問」的那一步）；稀疏終局 reward 下原則上可學、
  實務上餓死 — 與 C3（密集監督要件、沿分類）同族、⑬ 鎖死是它的 flow 版前科。
- **Q3（通道要在訓練迴圈裡開著）〔架構事實〕**：現行開環訓練（$(s,g)$＋錨→plan）對
  (iv)-操作的 function 施加零壓力；最便宜的上車位＝replan-on-collision（帳本 §4 表 (iv) 行、
  1 bit/步取等）。
⇒ 第二答：**「預設獲取資訊的能力」可學、而且它正是主人句裡「不是 info 而是 function」的
最純例** — 但我們目前的訓練 régime 三條件一條都沒供給；這是設計含義、不是現況能力。

### F3.3 GRPO 通道教的是 info 還是 function

- **既有切口**：L4.5 防火牆已把每次 update 拆成 world-bits（搬運）＋sharpening（0 bits、
  模式銳化）兩份 — 用 F-語言：**幫浦輸出天生就有 info 份與 function 份**、指紋可分
  （沿帳本〔啟發式〕）。本節加兩件：
- **Prop F5（centering 對比性）〔定理級、一行〕**：group advantage
  $\hat A_i=r_i-\tfrac1G\sum_j r_j$（÷σ 版同）對任何 $(s,g,e)$-可測的 reward 平移
  $r_i\mapsto r_i+\varphi(s,g,e)$ **不變** ⇒ 幫浦物理上打不進「對這題／這圖的常數事實」，
  只打得進**同題候選之間的對比結構**。_證_：per-group 常數在均值裡消去；σ 對平移不變。∎
  ⇒ GRPO 口徑天生濾掉題級 DC info、保留 plan 對比 — 這是「教程序」的**必要**形狀。
  ⚠️ 非充分：對比資訊在 $m{=}1$ 下仍可壓成 route 選擇表（instance info）。
  （與 L4(ii)「advantage＝後處理只降不升」相容：L4(ii) 說總量、本命題說**哪一種被濾掉**。）
- **充分側＝容量 schema 在 RL 題集上的實例〔命題級〕**：query 集若同時滿足
  （a）聚合上不可背 — 新組合對的總 bits 需求 ≫ 預算（Prop F7）、
  （b）可由共享算子解 — (S1) 組合性使一個算子對全題集夠用（沿合成律；設計卡 §3.4 的
  reward 接力「兩段合法拼接天然拿滿分」）,
  則訓練壓力下的最優落在算子側。**teleport／stitch 新組合對 query（設計卡已設計、data 無
  對應 FM 訊號、reward 是唯一老師）＝(a)(b) 同時最尖的一格。**
- 「怎麼想出對的路」的梯度機制（跨題共享參數收相干梯度、題專屬記憶收稀釋梯度）
  【推測級 — 無證明、無攜帶引文；分辨＝FP-4】。
⇒ 第三答：**reward 教「這條路對」是 info 面 — 且被 centering 預先濾成對比式；要教出
「怎麼想出對的路」，得靠題集設計把 info 融資變成不可負擔 — 條件與單圖/多圖之分同構
（F4 的 schema），group advantage 在 (a)(b) 成立時就是逼 function 化的壓力源。**

---

## F4. 訓練條件的推論：necessity schema（與「多路線歧義」的正式同構）

### F4.1 Schema 與雙實例

**Prop F6（necessity by conditional-entropy injection；schema）〔命題級 — 兩實例各自靠
已證件承重〕**：設任務存在「懶解」$\sigma$（不動用目標機制 $M$ 的解）。若
**(N1)** 注入的條件熵落在 $M$ 運作的層級、且 $\sigma$ 的資訊預算蓋不住它（可測性或容量論證）；
**(N2)** 蓋住它有可讀效用（loss 分離／eval 可讀）,
則 $\sigma$ 退出（population）最優集、$M$ 成為必需品。

| | **context 級（C-i、既有）** | **environment 級（本檔新增）** |
|---|---|---|
| 目標機制 $M$ | 用 intent 錨通道 | 環境通用算子（function） |
| 懶解 $\sigma$ | 忽略錨、讀 $A(s,g)$（invariant 解） | per-env 記憶（route 表壓進 θ） |
| 退化條件（懶解合法） | A1：$I(\tau;a\mid s,g)=0$ ⇒ invariant 是合法全域最優（Prop 2.1、沿內化 formal） | $m{=}1$：$W_e$ 常數 ⇒ 記憶免 bits 且對 MI 帳隱形（Remark F1.5） |
| 注入 (N1) | C-i：$H(R\mid s,g)\ge h>0$（多路線） — 錨才裝得下 $(s,g)$ 裝不下的東西 | 多環境：$\sum_e$ 所需 bits 超出 $\theta$ 預算（Prop F7） — 只有算子蓋得住全族 |
| 拆除定理 | Prop 3.7（最優性＋零驅動兩支柱同拆、沿內化 formal） | Prop F7 預算算術＋crossover |
| 可讀性條款 (N2) | C-iii（效用可分） | fresh-eval 在 loss／選模裡承重 |

**同構的正式一句**〔命題級〕：**「context 要成為解歧義的必需品」（C-i）與「function 要成為
跨環境覆蓋的必需品」（多環境）是同一個邏輯形 — 在相鄰兩個層級注入懶解吃不下的條件熵 —
的兩個實例**；C-i 注入在查詢層（$R$ 給定 $(s,g)$ 的熵）、F-條件注入在環境層（$W_e$ 給定
family 的熵）。兩者的退化端也同構：A1（錨冗餘）↔ $m{=}1$（環境退化成常數）— 都讓懶解
合法、且都讓「有沒有作弊」在該 régime 內**原則上不可觀測**。

### F4.2 預算算術

**Prop F7（info 融資按環境線性計費；function 融資不計 bits）**：
- **(i)〔定理級〕** $W_{e_1..e_m}$ 互獨（iid）⇒
  $\sum_{j}I(W_{e_j};\theta)\le I(W_{e_1..e_m};\theta)\le\min\big(H(\theta),\,I(\mathcal D;\theta)\big)$。
  _證_：鏈式 $I(W_{1:m};\theta)=\sum_j I(W_{e_j};\theta\mid W_{e_{<j}})$；
  $I(W_{e_j};\theta\mid W_{e_{<j}})-I(W_{e_j};\theta)=I(W_{e_j};W_{e_{<j}}\mid\theta)-\underbrace{I(W_{e_j};W_{e_{<j}})}_{=0\ (\text{iid})}\ge0$。∎
- **(ii)〔定理級〕** 各訓練環境上的地板增益＝$I(\tau_j;\theta\mid s,g,c)\le I(W_{e_j};\theta\mid s,g,c)$
  （Prop F2 恆等式＋DPI 沿 $\tau-(W_{e_j},s,g,c)-\theta$、F-A3 保 Markov）；F-A2 下條件版
  超加性同證 ⇒ **記憶 $m$ 個環境、每個買 $\beta$ bits 地板 ⇒ 至少耗 $m\beta$ bits 權重通道
  預算** — 線性計費。
- **(iii)〔命題級〕** 算子側：一個環境通用算子的成本＝固定描述長度、對 $m$ **常數**；
  可表示性有攜帶錨 — 連續 thought 的平行 BFS（Zhu+ 2505.12514：2 層 transformer、$D$ 步解
  reachability、沿分類[驗]）＋CoT-DP（2305.15408、沿分類[驗]）＝「BFS 在固定 size 可實現」。
- **Cor（crossover）〔命題級〕**：$m\beta\gtrsim$ 預算 ⇒ 記憶不可負擔、function 化是唯一
  蓋得住全族的（近）最優解 — 單圖訓練允許 info 偷懶、多圖訓練逼 function 化的正式形。

**⚠️ 匯率 caveat（進口 C-ii′、整條照搬）〔誠實邊界〕**：F4 全部是「最優／可行性在哪」的
陳述、**不是**「SGD 找不找得到」；bits 上不可負擔 $\ne$ CFM 的變異度量下有梯度壓力 —
postA1 的匯率斷裂在環境級同樣適用。動力學面（多圖訓練下 function 化真的被找到的速率與
條件）【推測級】。與 multiroute G3 的定量對台：環境級也該有 $\eta\gtrsim p\lambda$ 型門檻
（per-env 記憶的劣勢要在訓練壓力可感的量級浮現、光 $m$ 大不夠）— 操作化＝FP-2 的
標度曲線；門檻形式未推【推測級】。

---

## F5. 可證偽預測（info vs function 的行為分岔；判準不排程）

**FP-1（遷移二分＋關通道對照；本檔主判別器）**
設 A＝單圖訓練、B＝多圖訓練（其餘匹配）。fresh 圖 eval 三臂：
- (i) **map 通道開**（(ii)：以新圖的 route／佔據錨 conditioning）：function régime 預測
  fresh/train 表現比 $\to O(1)$（赤字是可攜的）；info régime 預測比 $\ll1$（融資項蒸發、
  Prop F2 的 NLL 版是恆等式〔定理級〕；「A 的融資混比偏 info、B 偏 function」〔命題級、
  F7〕；幅度【推測級】）。
- (ii) **verifier-BoN 開**（(iii)：新圖佔據圖＋BFS 場打分、選 1）：同向；量的是
  proposal＋選擇機器的遷移，上界照 L2（$\le\log N$／選擇）。
- (iii) **全通道關**（u=0、無錨、無驗證、無觀測）：**兩個 régime 都預測崩向 $\rho$-marginal**
  — 這臂是**對照不是判別器**（Prop F1 地板論證：沒 bits、function 也變不出資訊）。
  ⛔ 設計紀律：只跑 (iii) 的「zero-shot 測試」分不了 info/function — 兩假說同預測；判別力
  全在 (i)(ii)。反向可證偽：若 B 在 (iii) 大幅贏過盲 Bayes 基線（按 $\rho$ 走廊先驗的
  規劃器 — 此基線要真的量、不能拿 chance 冒充），則本檔的地板帳有洞（$\rho$-結構被低估）
  — **框架自身的可證偽點**。

**FP-2（容量標度簽名）**〔命題級（F7 直接後果）；定量形【推測級】〕
固定 θ 容量、掃訓練環境數 $m$：info régime ⇒ **train-set per-env** 表現隨 $m$ 衰減（預算
$C_\theta/m$ 攤薄）；function régime ⇒ 平或升（算子被多樣性磨利）。判別器＝train 曲線形
（衰減 vs 平/升）；次判＝fresh-env 表現只在 function régime 隨 $m$ 上升；crossover 點
應隨容量右移。

**FP-3（幾何載體遷移＝中間態的量法）**〔判別器＝定理級簽名沿用；預期幅度【推測級】〕
- 主探針：fresh 圖上量「latent 距離 ↔ 該圖 BFS 距離」的相關（路線一 geodesic 尺搬到
  $V_2$；probe_z_geodesic 儀器同款）。function régime：相關保留（encoder 對新圖現算幾何）；
  info régime：訓練圖高、fresh 圖掉回基線。
- **中間態＝相關量（與成功率）沿 $V_0\to V_1\to V_2$ 階梯的 decay profile** — 不是二值標籤，
  是一條曲線；「幾何探針可否區分」的答案＝可以，用 profile 的斜率位置區分。
- 配套分帳：P-resample 三臂（oracle／resampled／zero、沿分類 §4.3〔定理級操作型〕）搬到
  fresh 圖 — **oracle−resampled（info 貢獻）應追當場通道供給、resampled−zero（計算貢獻）
  在 function régime 下應保留** — 「保留的表現由哪一列融資」的操作型分帳；P-swap 簽名
  （swap-insensitive ⇔ 零配對資訊）照用。

**FP-4（GRPO 題集依賴＝F3.3 充分側的直測）**〔預測＝命題級；外溢幅度【推測級】〕
RL 題集含新組合對（teleport／stitch spec、設計卡現成）⇒ 增益**外溢**到 RL-held-out 的
其他新對（學到的是組合程序）；題集只含見過對 ⇒ 增益不外溢（學到的是題答案）。判別器＝
held-out 新對上的 RL−step-matched-FM 對照差（⑱' 鐵則照搬）。免費單元測：對 reward 加
$\varphi(s,g)$ 平移、訓練軌跡應逐位元不變 — Prop F5 的直接檢查。

**FP-5（acquisition 前提＝Q1 的直測）**〔命題級〕
單圖訓練出的 query／replan 政策：所選行動的 EIG 與隨機基線無差（且不隨 compute 改善）；
多圖＋部分可觀測訓練後 EIG-seeking 才可能浮現。判別器＝EIG(所選)−EIG(隨機)；EIG 在我們
確定性 gridworld 可直接算（L3 取等、碰撞位元 1 bit/步）。

---

## F6. 分級總表與誠實邊界

| 條目 | 級別 | 備註 |
|---|---|---|
| Def F1–F4（層級模型／info／function／階梯） | 定義級 | function 定義＝fresh 赤字、不是「會 re-derive」（後者是 Cor F1.4） |
| Prop F1 fresh 零資訊＝地板不可動 | **定理級**（一行、證在檔） | CT-1 的環境級同款＋L0 逐字 |
| Prop F2 記憶化缺口恆等式 | **定理級**（證在檔） | info＝gap 的 MI 項＝恆等式；NLL 貨幣 |
| Prop F3／Cor F1.4 兌現效率 ξ | 定理級 identity＋命題級量程 | ξ 可即算（idp 雙腿＋⑰）；C-ii′＝其變異面 |
| Remark F1.5 單環境退化 | 定理級（定義性） | 「帳本沒切這刀」的原因陳述 |
| F2 歸類表 | 定義級套用＋實證狀態陳述 | 「攤銷 BFS 證據只到 $V_1$」＝狀態、非判決 |
| u＝scratchpad 正讀 | Remark（定理背書、無新主張） | CT-1/CT-3/L5/2404.15758 重讀 |
| F3.1(a) check 算子可學 | 命題級 | 可表示性平凡、網路可學性未證 |
| F3.1(b) 自驗證＝0 bits＋sharpening 對接 | 帳目定理級（L2(iii) 逐字）；對接＝命題級 | 指紋沿 L4.5 |
| F3.2 Q1 訓練分佈要含不確定性 | 命題級（退化證梗概） | 外錨 BARL Thm 4.1（沿帳本） |
| F3.2 Q2／Q3 | 命題級→實證級／架構事實 | C3 同族 |
| Prop F5 centering 對比性 | **定理級**（一行、證在檔） | 必要形狀、非充分 |
| F3.3 容量 schema 於 RL 題集 | 命題級 | 靠 (S1)＋F7 |
| F3.3 梯度相干機制 | 【推測級】 | 分辨＝FP-4 |
| Prop F6 necessity schema＋同構句 | 命題級 | 兩實例各靠 Prop 2.1/3.7 與 F1.5/F7 |
| Prop F7(i)(ii) 預算算術 | **定理級**（證在檔） | 超加性＋DPI＋帽 |
| Prop F7(iii)＋Cor crossover | 命題級 | 可表示性錨 Zhu+／CoT-DP（沿分類[驗]） |
| 環境級 η-門檻形式 | 【推測級】 | 操作化＝FP-2 |
| FP-1 (i)(ii) 分岔 | 恆等式定理級＋régime 歸屬命題級＋幅度【推測級】 | 三層分開讀 |
| FP-1 (iii) 雙崩對照 | 定理級（Prop F1 後果） | ρ-marginal 基線定量【推測級、要實測】 |
| FP-2／FP-4／FP-5 | 命題級（定量形【推測級】） | — |
| FP-3 判別器 | 定理級簽名沿用（P-swap/P-resample）；幅度【推測級】 | 中間態＝decay profile |

**誠實邊界**：
1. **貨幣**：本檔一切以 NLL/bits 與 KL 計價；成功率／R0 隔兩道匯率（bits→變異＝C-ii′；
   $T{=}1\to T\to0$＝合成律 Lemma 1/2）— 沿帳本 §6 逐字進口。⛔ F5 凡落在成功率上的預測
   都是形狀預測、非點估。
2. **F4 是 optimum/feasibility 陳述、非動力學**：SGD 是否真走到 function 側、多快走到，
   全檔零主張【推測級】。
3. **$m{=}1$ 現況**：本檔對現有讀數的再詮釋（F1.5、F2 表）是狀態陳述；$V_2$ 實驗一格都
   還沒有 — 本檔零新實驗結果、零數字。
4. **$\rho$-marginal 基線**（FP-1(iii) 崩到哪）未定量；盲 Bayes 走廊先驗政策的成功率要真的
   量【推測級】。
5. **family 級（$V_3$）**全檔只佔位（Def F4）；跨 family 遷移零主張。
6. **多圖訓練軸與多路線軸正交**：C-i 修的是查詢層歧義（同圖內）、F-條件修的是環境層覆蓋
   — FP-1/2 需要的「多圖」資料軸是**新軸**，multiroute 檔的 (a)–(e) 選項都不自動供給它
   （teleport 仍是單圖）。此軸的資料構造未設計 — 呈裁點。
7. **外部件不承重**：承重命題全數一行／短證在檔；引文全沿 repo 攜帶、級別照抄、無新增 ID。

---

## F7. 引用清單（全部沿 repo 攜帶；級別照抄來源檔）

家內：Thm CT-1／Cor CT-2／Cor CT-3、三源表、C1–C4、P-ent/P-swap/P-resample（【分類】）；
Prop L0–L4、Thm L5、Remark L4.5、§4–6（【帳本】）；Prop 2.1/2.3、Prop 3.7、Def 1.4、
C-i~iii（【內化】＋【多路線】§1）；C-ii′、$\mathrm{Int}^*=\eta/(\eta+p\lambda)$（【postA1】）；
Lemma 1/2、(S1)、(CL)（【合成律】）；§1.2/§3.3/§3.4、⑱' 鐵則（【設計卡】）；
⑰ ~2.5 bits、⑬ 1.1%（轉引 FINDINGS-0905、沿上游標記）。
外部（攜帶級照抄）：2412.01951 Sharpening、2504.13837 RLVR、2505.20561 BARL、
2509.06861 compute-only TTS【皆隊友正文級@帳本】；2505.12514 Zhu+ superposition、
2305.15408／2310.07923／2402.12875 CoT 三件、2404.15758 filler tokens、2209.15189 context
distillation〔皆[驗]@分類〕；1705.07809 Xu–Raginsky【訓練記憶@帳本、引前補驗】；
2111.02080 Xie+ ICL-Bayes、1511.03643 LUPI【記憶ID@分類、引前補驗】。**本檔無新增 ID。**

---

## 接口注意（不改上游；列給ルナ驗收）

1. **帳本 Def L1 的 $W$ 是單環境切片** — L6 併法：外加環境期望、(i) 的帽升級為 per-family
   加總版（Prop F7(i)）。無矛盾、是延伸。
2. **帳本 §4 表「終身 ≤ $H(M_V)$（≤961 bits 級）」**在多環境下按環境數乘（每圖各自的
   $M_V$）— GRPO 卡若隨 multiroute/teleport 落地、帳的口徑要同步改。
3. **【一般化】⓪ 的「general」混了兩軸**：teacher-agnostic（ER .918、已證）與
   environment-generic（$V_2$、未測）。建議 claim 敘事拆開；F-frame 供後者判準。不動原檔。
4. **【分類】§2.2(a)「task 識別 bits」**＝F1 family 級的既有座標，兩檔可互引。
5. **teleport 破確定性**（多路線 (a3) 已註）：FP 系列若用 teleport 圖，Prop F1/F2 不受影響
   （不依確定性），但凡引 (S1)/合成律處沿其原限定。
6. **u-scratchpad 正讀與【敘事】§0 油門比喻一致**；無衝突。
7. **C-ii′ 被本檔進口到環境級**（F4 caveat）；postA1 不需改。若 ξ 錶落地、位置在【分類】
   §4.3 P-ent 家族旁。
8. **與帳本 §5「探索是油門、驗證器才是油箱」的接續句**（候選、供 L6）：**油箱裝的是 bits、
   引擎本身是 function — 訓練能把引擎造好（攤銷），永遠造不出油**。

**呈裁點**：①FP-1/FP-2 需要「多圖訓練」臂＝新資料軸（與多路線軸正交、F6 邊界 6）—
要不要開、怎麼開（多 maze 生成 vs OGBench 家族併集）待主人裁；②ξ 錶（訓練圖上今天可算、
idp 雙腿＋⑰ 現貨）要不要進儀器清單；③「攤銷 BFS」claim 是否按接口注意 3 拆軸重述。

_F 檔完（v0）。所有承重命題自帶證明；外部件只作錨與站位；ルナ明早驗收後才入庫。_



