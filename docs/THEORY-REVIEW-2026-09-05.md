# THEORY-REVIEW — 2026-09-05 兩份理論草稿逐條審查

_理論驗證官（Fable 級、主人授權；三隻並行之一、只審不改）。對象：
`THEORY-2026-09-05-composition-law-draft.md`（下稱【合】）、
`THEORY-2026-09-05-internalization-formal.md`（下稱【內】）。素材對照：FINDINGS-0905、NOTE-0905、DESIGN-0905。
判定四級：**真錯**（證明破，附反例/斷點）／**需加假設**（陳述可救，列出缺的假設）／
**建議降級**／**風格**（不承重）。每條數學我都自己重推過；「檢過、成立」＝找不到洞，不是沒看。
轉引原件（Cocos/Robinson/2310.07972/OKBE 內文/2605.08732）未上網重驗 — 草稿自己已標；
本審的判定不依賴任何轉引內容（B7 的反例獨立於 Cocos 原文對錯）。_

---

## A. 合成律檔

### A1. Lemma 1（semiring 身份）— 需加假設 ×2＋措辭修正 ×1；LSE 恆等式本體成立

**(S1) 用法逐步驗（指定重點）**：
1. **隱藏假設：z 的切分**。(S1) 寫 $p^*(z|s,g)=\sum_m w^* p^*(z_{\rm pre}|s,m)p^*(z_{\rm post}|m,g)$，
   預設 $z\mapsto(z_{\rm pre},z_{\rm post})$ 是良定義的可測切分。若切分點依 $m$ 而變（waypoint 位置不同、前段長度不同），
   「乘積密度＝拼接變數的密度」還需要拼接映射對每個 $m$ 是雙射（或把切分點納入 latent）。
   **缺的假設**：固定切分約定，或 per-$m$ 切分＋測度一致性一句話。
2. **「即 (CL) 的實例」— 形狀不合**。LSE 恆等式裡有三個物件：$\log w^*(m|s,g)$、$\log p^*(\cdot|s,m)$、$\log p^*(\cdot|m,g)$。
   (CL) 是 $V(s,g)=\bigoplus_m V(s,m)\otimes V(m,g)$ — **沒有 $w^*$ 那一項**，而 $w^*$ 同時依賴 $(s,g,m)$、
   吸不進 $V(s,m)$ 也吸不進 $V(m,g)$。另外左式多一個 $z$ 引數（型別不合）。
   **兩條修法**（擇一）：(a) 把 (CL) 改成帶權版 $V(s,g)=\bigoplus_m[W(m|s,g)\otimes V(s,m)\otimes V(m,g)]$；
   (b) 改用 partition-function 寫法：$M$ 為「每條路恰穿一次的 cut」時 $Z(s,g)=\sum_m Z(s,m)Z(m,g)$ 無權重成立
   （此時 $w^*=Z(s,m)Z(m,g)/Z(s,g)$ 是導出量不是自由量）— (b) 較漂亮但要加 cut 假設。
3. **(ii) 的「與 (i) 經 log 同構等價」— 措辭過強（風格級但必須改）**。證明只給了「期望對混合分佈線性」：
   $\mathbb E[{\rm succ}]=\sum_m w^*\,\mathbb E[{\rm succ}|m]$。succ 本身不沿 pre/post 因子化，
   所以 eval 泛函不是 (i) 恆等式的 log 像，只是「住在 sum-product semiring（其 log 像＝log-semiring）」。
   改寫成「與 (i) 同住一個 semiring、語意一致」即可；「同構等價」四個字撐不住。

(i) 的 NLL＝KL＋realizable ⇒ 唯一極小＝$p^*$（分佈層唯一）：檢過、成立。(iii) 定義代入：成立。

### A2. Lemma 2（凍結極限、nonexpansive 疊加）— 成立；需加假設（H 的語意）

夾擠 $\max\le{\rm LSE}_T\le\max+T\log K$：對。LSE 梯度＝softmax、$L_1$ 範數 1 ⇒ sup-norm nonexpansive：對。
**疊加那步（指定重點）**：誤差遞迴 $e_{t+1}\le e_t+T\log K$ 成立的前提是**每層只有一個引數帶誤差**
＝R4 的形 (a)（單步展開，$E$ 精確）。若用形 (b)（matrix squaring），兩個引數都帶誤差，
遞迴變 $e_{t+1}\le 2e_t+T\log K\Rightarrow e_H=(2^H-1)T\log K$ — 但 $H$ 次平方的**有效 horizon 也是 $2^H$**，
按有效 horizon 算仍是線性。**缺的一句**：$H$＝有效 horizon（路徑長／單步層數），不是「迭代次數」；
R4 既然開放兩形，這句不寫、實作者用 (b) 按迭代次數讀就會誤判 32 倍。
小註：形 (a) 每層 $K{+}1$ 項（含 $V_0$）⇒ 常數是 $T\log(K{+}1)$；另本引理界的是**迭代值**不是定點值（有限 horizon 讀法即可）。

### A3. Prop 3（定點迭代＝BFS）— 檢過、成立

歸納逐步重推過（含 (A2) 弱版「每條最短路的中繼點」也夠用）。兩個風格註：
「以 $g$ 為根的 BFS」在有向情形是反向圖 BFS（格圖可逆、無礙）；「$\le{\rm diam}$ 收斂」在含不可達對時
應寫「$\le$ 最大有限距離」。

### A4. R4（兩種迭代形）— **真錯（小、一行可修，但它自稱指導 B 階段實作）**

「語意保證都掛在『從 $\bot$ 起算的最小定點』」對形 (b) 是**錯的**：$\bot\otimes\bot=\bot$，
形 (b) 從 $\bot$ 起迭代**永遠停在 $\bot$**。matrix squaring 的正確寫法是從 $I\oplus E$ 起算
（或把映射寫成 $V\mapsto V_0\oplus E\oplus V\otimes V$）。且裸映射 $V\mapsto V\otimes V$ 的定點集
含大量垃圾（$\bot$、$\top$、任何傳遞閉集）—「兩者定點相同」要改成「兩者的**目標定點**（代數閉包）相同、
各自的正確初始化不同」。修一行、但不修會直接寫壞 code。

### A5. R5（OKBE 差異）— 不承重、自我限定合格；原件未重驗（草稿已自標引用帶限定詞）。

### A6. Prop 6（精確版正確性、兩夾）— **需加假設（假設清單不足；附反例）**

兩夾**方向**都對（min-plus：可行拼接⇒$\hat V\ge V^*$；最優路中繼在 $M$⇒$\hat V\le V^*$）。但：
1. **原子值未定義／缺忠實性假設（最大的洞）**。「字典空間 DP」的 base case — $V(s,m)$ 的原子數值從哪來 —
   全檔沒定義。(D3) 只保 decode **合法**、不保 decode **最優**。
   **反例**：格圖、$M$＝全頂點（(D1) $\varepsilon{=}0$ ✓）、decoder 每段都輸出合法但繞路 $+2$ 步的路徑
   （(D3) $\gamma{=}0$ ✓）⇒ $\hat V(s,g)\ge V^*+2\cdot(\text{段數})>V^*$ — 所列假設全滿足、結論破。
   **缺的假設 (D4)**：原子代價忠實 — 段值＝decode 路徑實際代價（soundness、撐 $\ge$ 方向）**且**
   decoder 可實現最優段（completeness、撐 $\le$ 方向）。
2. **(D1) 的量詞縫**：(D1) 只對「eval 分佈支撐內的 $(s,g)$」宣告；遞迴中出現的中間對 $(m,g)$、$(m,m')$
   不必在 eval 支撐內。**缺**：(D1) 對遞迴閉包成立（支撐閉包版）。
3. **(D1) 的 $V$ 指涉不明**：讀成真 $V^*$（覆蓋條件）或讀成 DP 自己的 $\hat V$（半循環）結論路徑不同 — 釘死一個。
加上 (D4)＋閉包後兩夾即閉合，Prop 級可保；照現狀陳述比證明強。

### A7. Conj 7 — 合格 conjecture（自己標了 $\sqrt\delta$ 缺塊）。建議：依 A6，誤差源應加第四項
（decoder 次優性 $\beta$，與 $\gamma$ 合法性是不同的量）— 現在的 $(\varepsilon+\delta+\gamma)$ 少一個轴。

### A8. Prop 8（Knaster–Tarski）— 檢過、成立（如陳述）；**覆蓋缺口要自報**

陳述本身對（單調⇒Tarski；有限＋連續⇒Kleene 到 lfp）。但它限定「$\oplus$ 取 sup/max」⇒
**排除了 LSE**；Prop 9(a) 只蓋「有折扣」LSE。⇒ **本系統自己住的 $T{=}1$ 無折扣情形，§4 沒有任何結果覆蓋**
（且純 (CL) 的 LSE 迭代在 log-prob 值域不封閉：$K$ 項 LSE 每步最多抬 $T\log K$，值可跑出 $[-\infty,0]$ —
這跟 A1-2 缺 $w^*$ 是同一個病：帶權/歸一化版才封閉）。§5 誠實邊界應補這條。

### A9. Prop 9（唯一性兩路）— 檢過、成立

(a) 折扣收縮：nonexpansive $\oplus$ ∘ $\gamma$-縮 內層，逐點驗過，對。
(b) 兩夾重推過；小註：唯一性論證需**有限 $S$**（或定點有下界）— 無限鏈上可有反常有限解；格圖有限、無礙，寫一句即可。

### A10. R9（lfp 語意）— 檢過、成立（$V\equiv1$ 是最大定點的例子驗過；BFS＝lfp 構造對）。
唯 R4 修正後「從 $\bot$ 起」只適用形 (a)。

---

## B. 內化檔

### B1. Prop 1.1 — 檢過、成立（ODE 良定性含在 A3-類；反向不成立的理由對：同終端分佈不逼同場）。

### B2. Prop 1.2（Grönwall 常數、指定重點）— 檢過、成立

同起點 coupling：$\delta'\le L\delta+\varepsilon$、$\delta(0)=0$、$t\in[0,1]$ ⇒ $\delta(1)\le\frac{e^L-1}{L}\varepsilon$ —
**常數對**。$W_2\le W_\infty\le$ pathwise 界：對。Pinsker 橋（含外層期望走 Jensen）：對。
K-R 橋形狀對。兩小註：$\rho(\eta)$ 要對**兩個**分佈取（或取大者）；$L$ 是對 $z$、uniform-in-$t$ — 建議寫明。

### B3. Prop 1.3 — 需加假設（$\tau\leftrightarrow z$ 的縫）

條件 MI 恆等式（KL 對 $a$ 平均＝MI＝NLL 折扣）本體對。但左邊是 $I$ of **$z$**、右邊寫 $I_{\rm data}(\boldsymbol\tau;a|s,g)$：
$z={\rm enc}(\tau)$ ⇒ $I(z;a|s,g)\le I(\tau;a|s,g)$（DPI），**等式需 enc 對 $a$ 充分／可逆**。
缺此假設時恆等式退為 $\le$ — 對 ⑫ 機制 1 的用法（上限語意）反而無傷，寫成不等式更穩。

### B4. Def 1.4 — 檢過、成立（作為定義健全；三點結構的確是唯一拆得開三 preimage 的形）

實證拼數逐位驗：$(.321-.321)/(.454-.321)=0$ ✓、$\varepsilon_{\rm rel}=1.1\%$ ✓ ⇒ $(0,0)$ 鎖死格 ✓ 與 ⑭ 一致。
分母 undefined 條款把 ⑤'' 裁決變成定義推論 — 這步是乾淨的。風格：$\kappa$ 未定（定義參數、可留）。

### B5. Prop 2.1（塔性質、指定重點）— 核心成立；**需修 $p$ 端點**

CI 那步嚴格驗過：$(u_t,z_t)\perp a\mid(s,g,t)$ ⇒（**weak union**）$u_t\perp a\mid(z_t,s,g,t)$ ⇒ 條件期望相等 —
論證合法（建議把 weak union 一詞寫進去，量詞就嚴了）。$L^2$ 投影唯一（a.e.）：對。
**斷點**：「population 最優解唯一＝invariant，$\forall p\in[0,1]$」在**端點失效** — $p{=}0$ 時 $\varnothing$ 支零權重、
其行為完全不受約束（$p{=}1$ 對稱同理），唯一性只對 $p\in(0,1)$ 成立；端點只剩「invariant 是最優之一」。
這個修正**反而幫敘事**：f27n（$p{=}0$）量到 cond 差 .6046 與唯一性不衝突（見 D3）。

### B6. Cor 2.2 — 檢過、成立（且配 Prop 1.2 可加一句：$\varepsilon$ 小時 guidance 偏移 $\le\frac{e^L-1}{L}|w|\varepsilon$ — 與 ⑪ 噪音級讀數同型）。

### B7. Prop 2.3 — (i)(ii) 成立；**(iii) 真錯（無條件版；附反例）**

**(i) 反變分量（指定重點）**：分解 $\partial_h v=\bar m+\delta$（$\delta$ 對 $p(a|s,g)$ 均值零；由 A1＋weak union，
$p(a|z_t,t,s,g)=p(a|s,g)$，均值零的測度前後一致）；$a\perp u_t\mid(z_t,s,g,t)$ ⇒ 交叉項因子化
$\mathbb E[\delta^\top r|\cdot]=\mathbb E_a[\delta]^\top\mathbb E[r]=0$；對稱項靠「$\varnothing$ 支已最優」的
$\mathbb E[r|z_t,t,s,g]=0$ 歸零 — **逐步驗過、成立**（草稿自標「量詞要 Rei 檢」— 檢完：按上述寫即嚴）。
(ii) 全域最優⇒Hessian PSD：成立。
**(iii) 斷點**：「梯度差 $\le C\varepsilon$ ⇒ $\varepsilon(0)=0\Rightarrow\varepsilon(t)=0$」不成立 —
輸出層差 $\varepsilon$ **控制不了參數 Jacobian 差**。反例（正是本系統的形）：cond 加法 adapter
$c=c_{\rm base}+Ma$、$M=0$（$\varepsilon=0$）、$\varnothing$ 支未收斂（$\bar r\ne0$）、maze 下 $a=A(s,g)$ 決定性
⇒ $\partial L/\partial M\propto(1-p)\,\mathbb E[\langle\bar r,\partial f/\partial c\rangle\otimes a]\ne0$ ⇒ $M$ 離開 0、$\varepsilon(t)>0$。
**這與同檔 R2.4 自己的賽跑機制直接矛盾**（R2.4 說 $\varnothing$ 支收斂前驅動非零 — 對的是 R2.4）。
**修法**：(iii) 加前提「$\varnothing$ 支已最優」（此時由 (i)(ii) 是駐點、塌後不再分化成立 — 但也就被 (i)(ii) 吸收）；
「訓練中自我維持」那半降級為 Remark、併入 R2.4 的賽跑（枯竭是漸進的、不是不變集）。
「梯度差 $\le C\varepsilon$」若限定 **cond 層 $\varepsilon_{\rm cond}=0$＋A4＋只算下游共享參數** 則對（活化全同⇒Jacobian 全同）—
限定詞要寫上，adapter 參數必須排除在外。

### B8. R2.4／R2.5 — 檢過、成立（R2.4 的 $\mathbb E[a\otimes\delta]$ 因子化在 A1 下對；且它反向佐證 B7）。

### B9. Conj 2.6 — 合格 conjecture（誠實）。

### B10. Prop 3.1（屏障）— 需加假設（$p$ 門檻顯式化）

sublevel-set 論證形狀合法（梯度流不穿higher-loss 路點）。但「屏障嚴格正」不夠 — 要**路點損失 > 出發點損失**：
$(\delta_\varnothing/2)^2>(1-p)\delta_a^2+p\delta_\varnothing^2\iff p<\tfrac14-(1-p)(\delta_a/\delta_\varnothing)^2$。
⇒ 需 $p\lesssim1/4$；**$p{=}0.3$（實際運行值）在證明範圍外**（$0.25<0.3$，$\delta_a{=}0$ 都救不回）。
「對小 $p$」的限定方向對、但門檻不寫出來會被誤读成蓋到 0.3 — 顯式化後恰好變成退火主張的定量依據
（小 $p$ 段有屏障、0.3 沒有）。「同步塌修擋不住」的誠實限制：對、且重要。

### B11. Conj 3.2 — 合格。B12. **Def 3.3 — 風格**：margin $m$ 是絕對距離、錨的 .6046 是**相對**差（⑬）— 標度換算一句要寫，否則實作照抄會錯尺。

### B13. Prop 3.4（附加座標 in A2'、指定重點）— 檢過、成立（合法）

判定：構造在 A2'（函數類對「附加 $O(1)$ 維＋下游投影忽略」封閉）下**合法**：$a$ 經 intent adapter 進 cond（A4 ✓）、
下游忽略 aux 座標仍實現 $v^*$（A2' ✓）、兩支同時最優 ⇒ 合成損 $=L_p^{\min}+0$ ⇒ 全域最優集 $=\{L_p$ 最優$\}\cap\{L_{\rm div}=0\}\subseteq\{\varepsilon_{\rm cond}\ge m\}$ — 逐步閉合。兩註：
(1)「單射」多餘 — $L_{\rm div}$ 只量對 $\varnothing$ 的距離，常數偏移 $\psi(a)\equiv m\,e_{\rm aux}$（$a\ne\varnothing$）就夠；弱化後更顯「結構藥不保內容」（R3.5 的誠實正確且重要）。
(2) 真架構 cond 維度**固定**、不能字面附加 — A2' 是理想化，實作對應是「有閒置方向」；建議加一句。
「任何 invariant 解付 $\lambda m$」的 invariant 需釘死＝**cond 層** invariant（v 層 invariant 而 cond 帶偏移者正是構造解、不付錢）。

### B14. R3.5 成立（關鍵誠實）；R3.6 成立 — 風格：$0$ 點次微分含 $0$，措辭改「任意小擾動後即有 $O(\lambda)$ 逐出力」更嚴。

### B15. Prop 3.7 — 需加假設（C-ii′：$z$ 層可讀）

DPI 方向對、A1 拆除對。但「兩支條件期望分離／方向導數轉負」需要的是 **$I(z;a|s,g)>0$（模型看得到的層）**，
而 C-ii 給的是 $I(\tau;a|s,g)\ge\delta$ — DPI 的方向**幫不上** $z={\rm enc}(\tau)$：encoder 若丟掉 route 資訊，
資料層破了冗餘、模型層兩支照樣合流。**缺的假設 C-ii′**：$I(z;a|s,g)\ge\delta'>0$（route 在 latent 可讀；
可由「$R=f'(z)$ 可測」保證）。有 C-ii′ 後「分離」可經 CFM 最優＝條件資料分佈＋「場同⇒分佈同」反證閉合
（比條件期望逐點分離更乾淨的路）。這個縫實務上就是「e_target 若塌掉 route、資料藥到不了模型」— 值得進設計。

### B16. R3.8 — 成立（與 ⑦ subgoal 格一致）。

### B17. Prop 4.1／Cor 4.1'（Z2 分解×商空間、指定重點）— 檢過、成立

對合 ⇒ $F_+\oplus F_-$：標準、對。(i) $\cos=-1$ 恰達成：對。**商空間那步**：$e_m$ 常值於軌道 ⇒ 降到商空間 ✓；
$d$ 依 A5 反演不變 ⇒ 也降 ✓；rank 約束在商上良定義 ✓；分離只發生在 $e_d$ ⇒ N1 的「要它最近又要它最遠」
在子空間版確實結構性消滅（Cor 4.1' 成立）。(iii) 重建＝容量條件：對（$\dim_{\rm eff}$ 非正式 — 風格）。兩註：
(1) A5 要按「**對每個引數各自**反演不變」讀（商上 $d([A],[B])$ 良定義需 $d({\rm rev}A,B)=d(A,B)$；$|\Delta t|$、對稱 BFS 都過，但要寫明）。
(2) 命題證的是「rev **不新增**矛盾」；「rank 目標自身在商空間可滿足」（有限維序嵌入存在性）是外部前提 — 加半句限定。

### B18. R4.2 — 成立（與 ⑨ C1 +0.9901 逐位一致；範數下限警告正確）。

### B19. Prop 4.2 — 需加假設（B2 的 gap 定義）＋風格一則

2q 論證在**等 gap**（漂移端）閉合；**凸 $f$（擴散端、gap 遞增）有縫**：$k{+}1$ 樣本被 $\frac12\Gamma_{k+1}$ 控住
仍可下探穿過 $\frac12\Gamma_k$ 中線 ⇒ 兩事件皆好仍可翻序。**修法**：B2 的半 gap 取**相鄰兩 gap 的 min**
（$\frac12\min(\Gamma_{k-1},\Gamma_k)$），2q 界即恢復；諷刺的是這正好把「長程變異爆」的破壞者寫進假設裡 — 與其敘事一致。
風格：$13.4\%$ 是**三元組**錯誤率，按 $\le2q$ 應推 $\hat q\gtrsim.067$、非 $\hat q\approx.13$ — 換算寫清楚。

---

## C. 對實測數字的矛盾掃描（① ⑦ ⑨ ⑩ ⑪ ⑬）— **未發現矛盾**

逐錨驗：①（$-.18{\sim}-.19$／$+.03$／$.321\to.454$）在【合】(D2)、【內】R2.5 用法一致 ✓；
⑦（$.336/.321/+.015$）進 1.3 表與 Def 1.4 拼數 ✓；⑨（C5 病灶＝decoder 模式間行為、C4 $.535$ vs $.638$、C1 $+0.9901$、C2 $.758/13.4\%$）
在【合】(D3)、【內】§4 引用一致 ✓；⑩ 與 A4/⑬ 的「塌在 cond 生成端」相容 ✓；⑪（$.344$ vs $.336$）＝Cor 2.2 的近似實例 ✓；
⑬（$.6046$／$1.1\%$／d_sg 活）進 Def 3.3 錨與 Def 1.4 診斷對 ✓。
唯二數字級註記：Def 3.3 相對/絕對標度（B12）；Prop 4.2 的 $\hat q$ 換算（B19）。

## D. 兩檔一致性 — **成立（4 驗 2 註）**

1. **接口聲明**（內化品質只動 (S2)、不動 semiring 身份）：**成立**。鎖死/內化好壞改變的是 $\varnothing$ 模式
   $p_\theta$ 對 $p^*$ 的逼近（(S2) 的 realizable/optimization 側）；$\oplus,\otimes$ 與 (CL) 形不被觸碰。
   INTENT_DROP 的 $\varnothing$ 支目標＝$a$-邊際＝Lemma 1 的 $p^*$ — 兩檔同一個目標物。✓
2. **Def 1.4 與 V 語意**：相容。$U$＝R0 成功率＝【合】Lemma 1(ii) 的 eval 泛函（邊際化期望、非 argmax）；
   推論同為 $z\sim p_\theta(\cdot|s,g,\varnothing)$。✓
3. **附加的自洽紅利**：B5 的 $p$ 端點修正讓「f27n（$p{=}0$）非 invariant（cond 差 .6046）」與 Prop 2.1
   唯一性**不再需要**只靠 R2.5 的有限容量註腳撐 — 端點本來就不保唯一。建議兩檔都收這句。
4. **(S1)×A1 合流**：【內】A1（route 由 $(s,g)$ 決定）成立時【合】(S1) 的 $w^*(m|s,g)$ 退化為單點 ⇒
   (CL) 的 $\oplus$ 塌成單項 — 不矛盾，但「合成律非平凡」與「內化度量立得住」**要的是同一件事：多路線 $(s,g)$**
   （⑫⭐、Prop 3.7 C-i、stitch 本義三方合流）。建議在兩檔接口節互引一句。
註：【合】A8 的覆蓋缺口（$T{=}1$ 無折扣定點無結果）與 A1-2 缺 $w^*$ 是同一個病的兩面 — 帶權 (CL) 一起修最省。

## E. 總表

| 條目 | 判定 |
|---|---|
| 【合】L1 | 需加假設（切分；$w^*$/帶權 CL）＋措辭（(ii)「同構等價」） |
| 【合】L2 | 成立＋需加假設（$H$＝有效 horizon／單步形；$K{+}1$） |
| 【合】P3、P9、R9 | 檢過、成立（小註見文） |
| 【合】R4 | **真錯**（形 (b) 從 $\bot$ 起得 $\bot$；一行修） |
| 【合】P6 | 需加假設（(D4) 原子忠實＋(D1) 閉包＋$V$ 指涉；附反例） |
| 【合】C7 | 合格 Conj；建議補 $\beta$（decoder 次優）第四項 |
| 【合】P8 | 成立；補自報 $T{=}1$ 無折扣缺口 |
| 【內】P1.1、Cor2.2、R2.4/2.5、R3.8、P4.1/Cor4.1'、R4.2、Def1.4 | 檢過、成立 |
| 【內】P1.2 | 成立（Grönwall 常數對；$\rho$ 兩分佈小註） |
| 【內】P1.3 | 需加假設（enc 充分性；否則退 $\le$，用途無傷） |
| 【內】P2.1 | 核心成立；需修（唯一性限 $p\in(0,1)$） |
| 【內】P2.3 | (i)(ii) 成立；**(iii) 真錯**（無條件版反例；修＝加「$\varnothing$ 支已最優」或降 Remark 併 R2.4） |
| 【內】P3.1 | 需加假設（門檻 $p\lesssim\tfrac14$；$p{=}0.3$ 在範圍外） |
| 【內】P3.4 | 成立（A2' 下合法；「單射」可弱化＋固定維度註） |
| 【內】P3.7 | 需加假設（C-ii′：$I(z;a|s,g)>0$，latent 可讀 route） |
| 【內】P4.2 | 需加假設（B2 取相鄰 gap min）＋$\hat q$ 換算風格 |
| 【內】Conj 2.6/3.2、【合】R5 | 合格（自我定位誠實） |

_建議降級彙總：P2.3(iii) → 條件式 Prop 或 Remark；【合】L1(i) 的「(CL) 實例」子句 → 修陳述（帶權 CL）而非降級；
P6 若不加 (D4) → 降 Conj 併入 C7。其餘無降級必要。_
