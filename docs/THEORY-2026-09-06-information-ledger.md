# THEORY — latent thinking 的資訊帳本：四通道記帳＋選擇界＋幫浦定理（v0、2026-09-06）

_理論使魔（Fable 級、主人睡前指定；兩並行之「formulation」線；隊友前例包本夜到貨、錨已織入）。
主人的問題：「自生的 u 對世界零資訊、只能計算 — 能 formulate 成 theory 嗎？那我們的 u
可以思考什麼？能不能靠 embedding space 探索、或跟環境互動，補 u 的資訊？」
上游（唯讀）：NOTE-0906-context-taxonomy（CT-1/2/3）、DESIGN-0906-grpo-thoughts、THEORY-
0905-composition-law（溫度族）、THEORY-0906-postA1（C-ii′）。ルナ候選統一圖＝採納並收緊。
⛔ 分級鐵則：**定理級＝證明在檔（多為一行 DPI）或逐字引已證件；Conj＝未證；Remark＝解讀；
啟發式＝量級粗估**。外部文獻三級：[驗]＝沿上游攜帶；【隊友正文級】＝前例調研使魔本夜查證
過正文（附 section）；【訓練記憶】＝未覆核（本使魔 web search 額度歸零）、承重前補驗。
本檔所有承重命題皆自帶證明、外部件只作錨與站位。
**〔9/6 午後追加〕§L6 續章**（權重通道的縱切：info vs function）＝ THEORY-0906-info-vs-function
的帳本側接口 — 導覽＋對接，本體仍在該檔；同批做了三處帳本口徑修（Def L1 的 $W$＝單環境切片、
$H(M_V)$ 帽多環境按環境數乘、L5.1 的 (CL)→(CL-w)）。深審 22 條的帳本側修檔（F1／F2／M1／M7
＋S1／S2／S4／S11）已在正文內就地標記。_

> **主人定調（2026-09-05 23:39、原話逐字，已對 closing 檔）**：「這樣我可以理解為訓練的過程中 Verifier其實是在教model東西，然後存在weights裡面，要inference的時候，則是從weights裡面提取肌肉經驗成u，然後拿來生答案？」
> — 帳本座標：Verifier 教＝(iii)；存 weights＝幫浦入 (i)（Prop L4）；提取肌肉經驗成 u＝
> Thm L5.2 提取（u 推論期加的是結構不是資訊）；生答案＝decode。整檔＝這句話的定理化。

---

## 0. 一句話＋帳本恆等式

> **世界資訊只有四個進帳通道 — (i) 權重、(ii) conditioning、(iii) 驗證器／選擇、(iv) 新觀測。
> 純自生 u 四個都不是：恆 0 bits（CT-1），全部價值在**提取赤字**欄。探索本身不進帳；探索×
> 外部驗證器每選擇 ≤ log N；互動每步 ≤ H(obs|belief)（確定性取等）；GRPO＝(iii)/(iv)→θ 幫浦。
> ⇒ 主人兩問＝同組命題兩半：同一批 bits、推論期消費（transient）或經幫浦入 θ（persistent）。**

**Prop L0（帳本恆等式）〔定理級〕**：對任何推論期 conditioning 束 $C$ 與任何實作 sampler $q$：
$$\mathbb E[-\log q(\tau\mid s,g,C)]
=\underbrace{H(\tau\mid s,g)-I(\tau;C\mid s,g)}_{\text{資訊地板（只有 bits 動得了）}}
+\underbrace{\mathbb E\,\mathrm{KL}\big(p^*(\cdot\mid s,g,C)\,\|\,q(\cdot\mid s,g,C)\big)}_{\text{提取赤字（計算住這）}}.$$
_證_：交叉熵分解＋ $H(\tau\mid s,g,C)=H(\tau\mid s,g)-I(\tau;C\mid s,g)$。∎
**「計算買提取效率、不買 bits」正式陳述**：任何不注入互資訊的操作（更深的 $f_\theta$、更多
自生樣本、自驗證搜索）只能動第二項；地板的每一分移動恰以注入的 $I$ 計價。自生 $u$：
$I(\tau;u\mid s,g)=0$（CT-1 取 $W:=\tau$）⇒ 地板不動 — Cor CT-2 的帳本座標。
⚠️ 地板以 NLL（$T{=}1$）計價；R0 是**支撐敏感、質量不敏感**的成功率泛函（對 route 邊際化
線性、對等價 route 間的質量重分佈不變＝succ 的 **route-invariance**）—〔9/6 深審 F1 修正：
原寫「R0 是 $T\to0$ 泛函」，但 R0 eval 裡沒有任何 argmax，與合成律 Lemma 1(ii)(iii)
「我們的 eval 讀數住 sum-product／log-semiring、不 eval argmax 計畫」的判決直接矛盾；
$T\to0$ 的合法住戶是 BFS（凍結極限、Lemma 2）與 refine/BoN 的部署極限方向（R2）〕。
換算分工：**Lemma 1 釘 eval 語意、C-ii′ 管 bits→變異（§6-2）；NLL→R0 的正式橋目前
沒有、標 open**。

---

## 1. 設定與四通道（Def L1）

記號沿 CT-1：查詢 $(s,g)$；world 變數 $W$＝環境／episode 資料側任何變數（佔據圖 $E$、BFS
距離場 $D$、真 route 皆是）；除註明外全程條件在固定 $\theta$ 上（部署期帳本）。
⚠️ **$W$ 是單環境切片**〔9/6 §L6 接口修、沿【F】接口注意 1〕：本檔全程固定一個環境
（$m{=}1$ 的 maze），故 $W$ 不帶環境索引。環境 ensemble 版＝外加 $e\sim\rho$ 的期望、
(i) 的帽升級為 per-family 加總版（Prop F7(i)）— **是延伸、不是矛盾**；見 §L6.1／L6.5。
⛔ 單環境下 info/function 之刀在數學上不存在（Remark F1.5）— 別拿本檔的帳去判攤銷 vs 記憶。

- **(i) 權重通道**：$\theta=\mathrm{Alg}(\mathcal D,R)$ ⇒ 鏈 $W-\mathcal D-\theta$ ⇒
  $I(W;\theta)\le I(W;\mathcal D)$〔定理級一行；框架＝Xu–Raginsky 1705.07809【訓練記憶】〕。
  進帳時刻＝訓練期；部署期唯讀。
- **(ii) conditioning 通道**：推論期輸入 $c$；注入＝$I(W;c\mid s,g)$、NLL 可兌現額恰等於它
  （Prop L0）。實例：ICL 範例（task 級）、route-ix 查圖（instance 級 ~2.5 bits、⑰）。
- **(iii) 驗證器／選擇通道**：外部資源 $M_V$（我們：資料建佔據圖 $E$＋BFS 場 $D$）經評分
  $v=V(u;s,g,M_V)$ 進場；bits 經**選擇變數**（BoN 的 $J$）或 **reward 值**（GRPO）注入。
- **(iv) 觀測通道**：閉環互動 $o_t=O(W,\text{state}_t,a_t,\xi_t)$；bits 經觀測序列注入。

三源分類（taxonomy §2.2）嵌入：(a) ICL＝(ii)@推論、(b) oracle＝(ii)@訓練→(i)、(c) 自生＝
四者皆非（零進帳）。帳本的增量＝把 (iii)(iv) 升為一級來源、GRPO 定位成 (iii)→(i) 搬運。

---

## 2. 帳本定理組（每通道一條 DPI 級命題）

### 2.1 Prop L1 — 純自生：恆 0，「想更久／更多條」聯合仍 0〔定理級〕

(a) $I(W;u\mid s,g,\theta)=0$〔＝Thm CT-1 逐字，引用〕。
(b) **聯合零**：任何自足思考程序 — $u_k=f_\theta(s,g,u_{<k},\varepsilon_k)$、
$\varepsilon_{1:N}\perp W\mid(s,g)$、無外部呼叫 — 對任意 $N$、任意深度／拓撲（迭代、
Coconut 形回饋、樹狀自我展開）：$I\big(W;u_{1:N}\mid s,g,\theta\big)=0$
〔$\theta$ 明寫一次：本檔 §1 有「除註明外全程條件在固定 $\theta$」的全域約定，但本式常被
敘事檔／設計卡單條摘引、約定不隨行 — 脫離約定後字面為偽（不條件 $\theta$ 時 $u$ 經
$\theta$ 攜帶 $W$ 資訊）。深審 S11 防漂。〕
_證_：歸納 — $u_{1:N}$ 是 $\sigma(s,g,\varepsilon_{1:N})$-可測、$\varepsilon_{1:N}\perp W\mid(s,g)$。∎
⇒ 把「多想幾條、想深一點就會多知道路」正式殺死：計算規模不是資訊變數；迭代買的是 serial
depth（taxonomy C1/C2、CoT 三件），記赤字欄。（訓練側對偶＝Cor CT-3，引用不重證。）
⭐ 外部孿生【隊友正文級】：compute-only test-time scaling 對 latent truth 零新資訊 — DPI 沿
$A\to\hat p\to T_k\to\hat R$ 鏈＋Fano 轉 accuracy 上限（2509.06861 §5.3 Thm 1＋Cor 1、證
App H）＝L1 的 TTS 版、且附帶 bits→accuracy 單向匯率模板（收進 §6-2）。

### 2.2 Prop L2 — 探索×選擇：selection channel 界〔定理級、證在檔〕

設候選 $u_i=f_\theta(s,g,\varepsilon_i)$、$i=1..N$、$\varepsilon_{1:N}\perp(W,M_V)\mid(s,g)$；
評分 $v_i=V(u_i;s,g,M_V)$；選擇 $J=\arg\max_i v_i$（平手用獨立噪音）；輸出 $u^*=u_J$。則：

- **(i) 選擇預算**：$I(W;u^*\mid s,g,\theta)\;\le\;H(J\mid s,g,\theta)\;\le\;\log N$
  （$\theta$ 固定 — 深審 S11：摘引時務必帶著，見 L1(b) 註）。
  _證_：$u^*$ 是 $(u_{1:N},J)$ 的函數 ⇒ DPI；鏈式
  $I(W;u_{1:N},J\mid s,g)=\underbrace{I(W;u_{1:N}\mid s,g)}_{=0\ (\text{L1b})}+I(W;J\mid u_{1:N},s,g)\le H(J\mid\cdot)\le\log N$。∎
- **(ii) 驗證器存量帽**：$I(W;u^*\mid s,g)\;\le\;I(W;M_V\mid s,g)$。
  _證_：$(u_{1:N},J)$ 是 $\sigma(s,g,\varepsilon_{1:N},M_V)$-可測、$\varepsilon\perp(W,M_V)\mid(s,g)$
  ⇒ $I(W;u^*\mid s,g)\le I(W;\varepsilon,M_V\mid s,g)=I(W;M_V\mid s,g)$。∎
  — 從驗證器擠不出驗證器自己沒有的東西。
- **(iii) 自驗證＝0 新 bits**：$M_V\perp W\mid(s,g,\theta)$（例：critic 是 $\theta$-可測）
  ⇒ $I(W;u^*\mid s,g,\theta)=0$。〔9/6 深審 F2 修正：原前提寫不條件 $\theta$ 的
  $M_V\perp W\mid(s,g)$ 而舉「同資料訓的 critic」為例 — 該例**不**滿足字面前提（critic 從
  $\mathcal D$ 訓、$\mathcal D$ 與 $W$ 相關 ⇒ 不條件 $\theta$ 時 $M_V\not\perp W$）；它真正
  滿足的是條件 $\theta$ 版（$\theta$-可測 ⇒ 給定 $\theta$ 為常數），與結論的條件變數一致。〕self-BoN／self-consistency 仍可能**有用** — 但收益全記
  赤字欄（改善對 $\theta$ 存量的提取），零新進帳。⛔ 兩帳欄不准混。
- **(iv) 搜索推廣**：任何「唯一 world 存取＝驗證器呼叫」的程序（beam、MCTS、迭代
  refine-with-check），選擇轉錄 $J_{1:K}$：$I(W;\text{output}\mid s,g)\le\sum_k\log|\mathcal J_k|$
  且仍 $\le I(W;M_V\mid s,g)$。_證_：(i) 鏈式對 $K$ 輪疊加＋(ii) 不變。∎
- **(v) Remark（兩種 log N 別混）**：分佈傾斜讀法是【定理】—
  $\mathrm{KL}(\pi_{\rm BoN}\|\pi_{\rm base})\le\log N-\tfrac{N-1}{N}$（2401.01879 Thm 3.1、gap
  上下界 §3.1–3.2）且同 KL 預算漸近可打滿（2404.01730 §4.2 Thm 2、Sanov）【皆隊友正文級】。
  它講**分佈動多少**、(i) 講**對 $W$ 知道多少** — 同為 $\log N$ 級是兩條不同的界。⚠️ (i) 的
  $W$-bits 側可達性仍〔啟發式〕：候選池要蓋到正確 route（$N\gtrsim1/p_{\min}$）＋排序對。
- **(vi) Remark（分離定理：驗證器是資訊來源、不是比喻）【隊友正文級】**：verifier-free TTS
  有 $\Omega(H/\sqrt n)$ 下界、verifier-based 達 $O(1)$（2502.12118 §5.2 Thm 5.4/5.7/5.8；
  假設＝heterogeneity＋anti-concentration）— (ii)(iii) 的正向錨：沒 $M_V$ 可證地虧、
  有 $M_V$ 可證地贏。⚠️ 與 L2 上界的關係是**兩側佐證、不是雙向夾**〔9/6 深審 M7 修正〕：
  資訊上界（我方、**bits**）與性能分離（外部、**accuracy 標度**）**量綱不同**，中間隔著
  §6-2 自己宣告斷裂的匯率（增益方向 bits↛accuracy）；經 Fano 只有帽方向可換算。
  〔需回原文抽驗：2502.12118 Thm 5.4/5.7/5.8 的目標量與其 heterogeneity＋anti-concentration
  假設是否適用我們的 reward 分佈。〕

### 2.3 Prop L3 — 閉環觀測：每步 ≤ H(obs|belief)、確定性世界取等〔定理級〕

互動史 $h_t=(s,g,a_{1:t},o_{1:t})$、行動 $a_t=\pi(h_{t-1},u\text{-stuff},\varepsilon_t)$（自生、
L1 適用）。則
$$I(W;h_T\mid s,g)=\sum_{t}I(W;o_t\mid h_{t-1},a_t)\;\le\;\sum_t H(o_t\mid h_{t-1},a_t),$$
且 $o_t$ 給定 $(W,h_{t-1},a_t)$ 確定（我們的 gridworld：碰撞位元、局部 patch）時**逐項取等**
— 實現的 surprisal 就是入帳的 map bits。天花板：$\sum_t\le H(W\mid s,g,\theta)$。
_證_：鏈式分解；$I(W;a_t\mid h_{t-1})=0$（L1b）；$I=H-H(\cdot\mid W,\cdot)$、確定性時後項 0。∎
- **Remark（經典錨＋邊界條件）【隊友正文級】**：閉環勝開環的量 $\le I(X;C)$、1 bit 側資訊
  至多換 1 bit 熵減（Touchette & Lloyd、chao-dyn/9905039 Thm 2）＝L3 的祖版。補充：最優
  policy 為 Markovian（不依 belief）時測試期探索零誘因（BARL 2505.20561 Thm 4.1）—
  (iv) 有價值的前提＝belief 承重（不確定性要在決策上有分量）。
- **Remark（EIG＝導向、不是來源）**：選 $a_t$ 極大化 $I(W;o_t\mid h_{t-1},a_t)$＝Bayesian
  experimental design（Lindley 1956）＝active inference 的 epistemic value（Da Costa+
  2001.07203）【皆訓練記憶】；u 的合法角色＝算這個 argmax（§3 第 4 格）、bits 由世界供給。
- 我們域的粗估〔啟發式〕：replan-on-collision＝每步 ≤1 bit；局部 $3\times3$ patch ≤8 bits/步；
  全圖存量 $H(E)\le$ 格數 bits（31×31 ≤ 961、結構先驗下遠小）。

### 2.4 Prop L4 — RL 幫浦：GRPO 每 update 打進 θ 的 bits 上界〔(i)–(iii) 定理級；(iv) Conj〕

設定沿 DESIGN-0906：第 $t$ 步查詢批 $Q_t$（$B_g$ 題）、每題 $G$ 條
$z\sim\pi_{\theta_t}(\cdot\mid c_q)$（自生）、reward $r=R(z;s,g,M_V)\in\mathcal R$、
$|\mathcal R|=k$、advantage $\hat A=h(r_{1:B_gG})$、$\theta_{t+1}=U(\theta_t,Q_t,z,\hat A)$。

- **(i) 每 update 界**：$I(W;\theta_{t+1}\mid\theta_t,Q_t)\;\le\;B_g\,G\,\log k$（binary＝$B_gG$
  bits）。_證_：給定 $(\theta_t,Q_t)$，$z\perp W$（L1b）⇒ $I(W;\theta_{t+1}\mid\cdot)\le
  I(W;z,\hat A\mid\cdot)=0+I(W;\hat A\mid z,\cdot)\le H(r_{1:B_gG}\mid\cdot)\le B_gG\log k$。∎
- **(ii) advantage＝後處理（DPI）**：$r\mapsto\hat A$ 只能降不能升；退化群（$\hat A\equiv0$）
  貢獻**恰 0** — 設計卡「退化群比例」錶＝幫浦輸入頻寬錶（字面義）。
- **(iii) 終身帽**：$\theta_T=F(\theta_0,\{Q_t\},\{\varepsilon_t\},M_V)$ ⇒
  $I(W;\theta_T\mid\theta_0,\{Q_t\})\le I(W;M_V\mid\{Q_t\})\le H(M_V)$ — 幫浦一輩子打進 θ 的
  不超過驗證器資源存量（把地圖搬進權重的上限）；同時 $\le\sum_t$(i)。_證_：同 L2(ii)。∎
  ⚠️ **多環境口徑**〔9/6 §L6 接口修、沿【F】接口注意 2〕：$M_V$ 是**每圖各自**的（佔據圖＋
  BFS 場按環境建）⇒ 訓練跨 $m$ 個環境時終身帽為 $\le\sum_{j\le m}H(M_V^{(e_j)})$、按環境數
  **乘**（各 $M_V^{(e_j)}$ 互獨時取加總，見 Prop F7(i) 的超加性）。⛔ 別把單圖的 ≤961 bits
  直接當多圖總帽用。GRPO 卡若隨 multiroute／teleport 落地，此口徑要同步改。
- **(iv) Conj L4.4（有效注入 ≪ 上界）**：可兌現成行為改善的量另有匯率 — bits 以 NLL 計價、
  CFM/PG 以變異計價（C-ii′、postA1 §3），且 (i) 粗估（$16\times8\times\log_2k\approx1400$
  bits/update〔啟發式〕）遠超實際學習量。**有效流量未定量**；rung 0 的 pass@G−pass@1 gap
  是它的操作型輸入量測（設計卡 §3.4；⚠️ 單向訊號，見上表 M1 雙錶紀律）。降 Conj、分辨
  實驗已在設計卡 rung 0/2。
  ⚠️ **根源註〔9/6 深審 S2〕**：我們的 $M_V$（$E$-圖＋BFS 場）**由訓練資料建**（⑤''）⇒
  $I(W;M_V)\le I(W;\mathcal D)$ — 油箱是資料的另一種蒸餾、不大於權重通道的上游。
  「有效注入 ≪ 上界」的一個**結構性**原因正是 $\theta_0$ 已（部分）吸收同源資訊 ⇒ 有效
  流量的正確對象是 $I(W;M_V\mid s,g,\theta_0)$（與 L4.5 防火牆同軸）。
  ⛔ 別讀成「verifier 帶來資料之外的新東西」。
- **Remark L4.5（⭐ 防火牆：sharpening ≠ 注入）【錨隊友正文級】**：GRPO 對 θ 的改動分兩份 —
  reward 攜帶的 world-bits（搬運、本命題管）與純模式銳化（$z$ 自身隨機性、0 bits＝溫度族往
  $T\to0$ 壓）。外部批評打後者：self-improvement＝靠 verification-generation gap **提取**、
  不能創造（Sharpening 2412.01951）；RLVR 只提升取樣效率、大 $k$ 時 base pass@k 反超
  （2504.13837）。**分家判準〔9/6 深審 F2 修正 — 原句以 $I(W;M_V\mid s,g)>0$ 判「批評不適用」、
  並把 2504.13837 歸為自驗證，兩者皆錯〕**：批評的適用條件＝$I(W;M_V\mid s,g,\theta)\approx0$
  （verifier 對**權重已存之外**無新資訊 — 自驗證是充分情形、真值 verifier 配吸飽的 $\theta$
  亦然）。⛔ RLVR＝RL with **Verifiable** Rewards，字面即外部可驗 reward、**不是自驗證**；
  2504.13837 的教訓正是「$I(W;M_V\mid s,g)>0$ **不足以**擋 sharpening 判定」。我方防火牆
  **主張**＝$E$-圖是近無損的地圖存儲、$\theta$ 是有損壓縮 ⇒ $I(W;M_V\mid s,g,\theta)>0$
  仍嚴格正；⚠️ 這是**假設、不是定理**（我們的 $E$-圖＋BFS 場也由訓練資料建、與 $\theta$
  同源 — ⑤''），其操作型量測就是 rung 0 headroom〔9/6 早 rung0/0.5 實測：headroom
  $.13\sim.19$ 全過 ⇒ 條件 $\theta$ 後的注入空間實測為正、與該假設同向〕。
  ⇒ **批評的適用性由 rung 0 實測判、不由 $I(W;M_V\mid s,g)>0$ 判。**
  可測指紋〔啟發式〕：sharpening 簽名＝大 $k$ 反超 base；外部注入預測＝中等 $k$ 段不反超
  （NF 全支撐、$k\to\infty$ 同飽和）— ⚠️ 指紋照跑，但其**判別力前提**改掛條件 $\theta$ 版
  ＋上面的存量論證（若批評文獻在同樣 $I(W;M_V|s,g)>0$ 的設定下量到反超，舊根據即壞）；
  加 §3.3 指紋＋P-swap、三器分辨。
- **Remark（容量對齊）〔啟發式；9/6 深審 S1 修正：兩種頻寬並陳、⛔ 不列可證偽預測清單〕**：
  兩個讀法量級差 25×，別混 — **(BoN 選 1 讀法、L2(i))**：$G=8$ ⇒ 每群選擇預算
  $\log_28=3$ bits；**(GRPO reward 向量讀法、L4(i))**：每群 $G\log k\approx8\times10=80$
  bits 級。對照 ⑰ 實測 instance 級 route 資訊 ~2.5 bits（⑰' 引用紀律：路線級 ~0.93 bits
  ⇒ 選 1 讀法是 **3 倍餘裕**、不是「剛好夠」）。$G$ 的資訊論選型論證（要更強訊號→加 $G$
  ＝加頻寬，與設計卡藥單順序一致）在兩讀法下同向、量級不同。
  ⚠️「3 ≥ 2.5 剛好夠」**無任何觀測能證偽**（$G$ 不夠時可歸因排序／可達性）⇒ 它是 Remark、
  不是預測；配套的可測件是**退化群比例錶**（L4(ii)、定理級）。

---

## 3. 「u 可以思考什麼」設計定理（Thm L5）

**陳述**：CT-1 régime 下，自生 thinking 的全部合法價值＝Prop L0 赤字欄；合法標的四格 —

1. **分解（decomposition）**：u 實體化合成律 **(CL-w)**〔9/6 深審 M8 後：主方程的統計版是
   **帶權**形，裸 (CL) 缺 $\log w^*(m\mid s,g)$ 項、且裸 LSE 迭代在 log-prob 值域不封閉 —
   引用時一律用 (CL-w)〕中繼點 $m$、思考空間跑 $\bigoplus_m$ 的
   DP — 難分佈拆兩段簡單因子（Lemma 1 log-semiring；(S1)／(S1′)）。⚠️「因子各自簡單」仍猜測級
   （taxonomy §4.2）— 分解合法、增益待驗。
2. **提取（extraction）**：θ 存量（地圖／路線知識）展開成顯式計畫 — serial depth 貨幣
   （CoT 三件、沿上游[驗]），要件 C1–C3（迭代或離散瓶頸、serial 類、密集監督）。
   ＝主人定調句「提取肌肉經驗成 u」的那格；外部語義同構＝Sharpening（2412.01951）。
3. **搜索前沿疊加（superposition）**：連續 u 同時線性編碼多條候選 route＝平行 BFS
   （Zhu+ 2505.12514、沿上游[驗]）—「embedding space 探索」的合法形式＝**在自家假設空間
   搜 θ 已有的東西**，不是對世界採樣。
4. **查詢規劃（query planning）⭐**：計算「該問什麼」— EIG 的 argmax（L3）、BoN 的
   proposal 生成（L2 的 $f_\theta$ 端）、GRPO 的行為分佈。u 不能供給 bits，但能決定 bits
   從哪、以多大效率進來 — 自生思考＝(iii)/(iv) 通道的**導向系統**。對主人這問最深的一答。

**⛔ 不合法期待清單**（每條有定理擋）：(x1) 憑空生 instance／route bits — L1(a)；
(x2) 想更久、抽更多條→更多資訊 — L1(b)；(x3) 自驗證 BoN 注入新 bits — L2(iii)＝0
（可改善提取、帳欄別混）；(x4) 自生 $(u,\tau)$ 配對建通道 — Cor CT-3；(x5) 自生達成
off-manifold 探索 — 定義上 on-manifold（taxonomy §4.1）。

---

## 4. 與我們系統的對應表

| 系統件 | 通道 | 現況 | 缺格 | bits 粗估〔啟發式〕 | 設計含義 |
|---|---|---|---|---|---|
| 佔據圖＋BFS 檢查（免費 reward） | (iii) 的 $M_V$ | 儀器現成（`_EOCC`＋dist 場；rung 0 可直接做） | **部署期 BoN 臂未建**：eval 抽 $G$ 條、免費驗證器選 1 — (iii) 最便宜消費者、零訓練 | 每候選 reward ≤ $\log k\approx$10–13 bits；BoN 選擇 ≤ $\log G$（$G{=}8$⇒3 bits） | pass@G−pass@1 gap（rung 0）＝這條臂的**價值錶**；對幫浦輸入只是**單向訊號**（gap>0 ⇒ 有輸入；gap=0 在 dense reward 下**不判死**、看退化群錶）—「一魚兩吃」只在 reward 二值時兩錶重合〔9/6 深審 M1〕 |
| route 查圖 eval（idp on-mode） | (ii) 部署版 | 已建（⑤'' 帶查(map) 語意） | 部署時 route-ix 可得性（訓練圖之外） | ~2.5 bits/題（⑰；走廊 ~1、⑰'） | (ii) 是唯一「免搜索直給」的 instance 通道；增益上限＝地板下移 ≤ I |
| GRPO 卡 | (iii)→(i) 幫浦 | 設計 v0；rung 0–4 未跑；前提＝base 收斂（⑱'） | 有效流量（Conj L4.4）；退化群頻寬實測 | ≤ $B_gG\log k$/update；終身 ≤ $H(M_V)$（≤961 bits 級 — ⚠️ **單環境口徑**；多環境按環境數乘 $\sum_j H(M_V^{(e_j)})$〔§L6 接口修、【F】接口注意 2〕） | 退化群錶＝頻寬錶；$G$ 選型＝容量對齊；**雙錶紀律**〔9/6 深審 M1〕：**退化群比例＝1 ⇒ 幫浦零輸入**（L4(ii) 字面義、定理級）；pass@G−pass@1 gap＝0 **不是**零輸入的定理（dense reward 下群內 reward 仍可有變異 ⇒ $\hat A\ne0$）⇒ ⛔ 別按單錶砍臂 |
| 閉環 replan | (iv) | **未建** | 全部 | 碰撞位元 1 bit/步（確定性⇒取等）；patch ≤8 bits/步 | 最便宜 (iv)＝replan-on-collision；閉環 92% vs 開環 73%（2605.08732、沿上游）＝實證錨、理論帽＝T&L Thm 2 |
| oracle 錨→內化（主線） | (ii)@train→(i) | 已建（遷移鏈本體） | 內化 gap 定量（另線） | 2.5 bits/episode × 資料量、帽＝$I(W;\mathcal D)$ | 帳本收遷移鏈為特例：資訊建通道、權重存、計算駛 — **驗證器與環境續帳、RL 入帳** |

Novelty 站位【隊友正文級 gap 判定】：「flow-latent 探索＋佔據圖免費驗證器＋GRPO 內化」三件
同堂於 GC 導航查無前例 — 最近鄰 MCTD 2502.07202（純 test-time、無內化）、Flow-GRPO
2505.05470（文生圖、無探索迴圈）、Searchformer 2402.14083（token 空間、監督式）。

---

## 5. 主人兩問的直接答案

- **「在 embedding space 探索，能補 u 的資訊嗎？」** 探索**本身不能** — L1(b)：任意多條、
  任意深的自生探索聯合仍恰 0 bits；合法買的是搜索計算（L5.3）。**探索×外部驗證器能** —
  L2：每次選擇 $\le\min(\log N,\;I(W;M_V\mid s,g))$；自驗證器＝0 新 bits（L2(iii)）；
  且驗證器有無＝可證分離（L2(vi)）。一句話：**探索是油門，驗證器才是油箱。**
- **「跟環境互動補？」** **能** — L3：每步 $\le H(o_t\mid\text{belief})$、我們的確定性世界
  逐步取等（碰撞位元＝1 bit/步）；u 的角色＝挑最大 EIG 的那步（L5.4）。
- **「跟 GRPO 的結合？」** ＝幫浦定理 L4：同一批 (iii)/(iv) bits 兩條出路 — 推論期當場消費
  （transient、免訓練：BoN／replan）或經 GRPO 打進 θ（persistent、amortized）。兩問＝
  同組命題兩半（ルナ候選圖成立；本檔升為 L1–L4）。守恆句：**GRPO 不創造 bits、只把
  驗證器的地圖搬進權重** — 零輸入的**定理級**判準是「退化群比例＝1（群內 reward 全同
  ⇒ $\hat A\equiv0$）」；rung 0 的 pass@G−pass@1 gap 是它的**單向訊號**（gap>0 ⇒ 有輸入；
  gap=0 在 dense reward 下不判死）〔9/6 深審 M1 雙錶修正；設計卡 §4 rung 0 gate 原本就
  寫「兩錶至少一個顯著>0」— 是帳本這裡把雙錶壓成單錶又升格成定理〕；
  且「搬運 vs 銳化」有防火牆與指紋（L4.5）。

---

## 6. 分級總表與誠實邊界

| 條目 | 級別 | 備註 |
|---|---|---|
| Prop L0 帳本恆等式 | 定理級（恆等式） | 交叉熵分解；NLL 貨幣 |
| L1(a)(b) 純自生聯合零 | 定理級（引 CT-1＋一行歸納） | 外部孿生 2509.06861【隊友正文級】 |
| L2(i)–(iv) 選擇界 | 定理級（證在檔） | $W$-bits 可達性仍啟發式；傾斜側定理＋可達（2401/2404） |
| L2(vi) 驗證器分離 | 引用級【隊友正文級】 | 2502.12118；與 L2 上界＝**兩側佐證**（bits vs accuracy 標度、量綱不同）⛔ 非雙向夾〔深審 M7〕 |
| L3 閉環界＋取等 | 定理級（證在檔） | 取等＝觀測確定性；祖版 chao-dyn/9905039 |
| L4(i)(ii)(iii) 幫浦界 | 定理級（證在檔） | 計 world-bits、非總行為改變 |
| Conj L4.4 有效注入 | **Conj** | 匯率斷裂 C-ii′ 承接；分辨＝rung 0/2＋指紋 |
| Remark L4.5 防火牆 | Remark＋指紋〔啟發式〕 | 2412.01951/2504.13837 對 L2(iii)/(ii) 分家 |
| Thm L5 合法四格 | **定理級（價值歸屬：全部合法價值＝L0 赤字欄，由 L0＋CT-1）＋taxonomy（「恰為四格」＝分類敘事、不可證偽）**〔深審 S4：⛔「框架」二字會被引為已證窮舉〕；各格承上游分級 | L5.1 增益半句猜測級；L5.4 依 EIG【訓練記憶】 |
| 容量對齊 3 vs 2.5、表內 bits 粗估 | 啟發式 | 全部上界方向、未扣結構冗餘 |

**誠實邊界**：
1. **嚴的**：L0（恆等式）、L1（恰 0）、L3 在確定性域（等式）。**只上界的**：L2 的 $\log N$
   （$W$-側可達性要覆蓋＋排序對）、L4 計數界（gross bits；有效流量＝Conj L4.4）。
2. **貨幣**：帳本以 NLL/bits 計價；R0 是**支撐敏感、質量不敏感**的成功率泛函（succ 對 route
   標籤不變 — sharpening 塌到單一合法 route ⇒ NLL 變差、R0 不動＝解耦例）、⛔ 非 $T\to0$
   泛函〔9/6 深審 F1〕。bits 進帳後要過兩道匯率（bits→變異＝C-ii′；NLL→R0＝Lemma 1 釘
   eval 語意、**正式橋 open**、P2 解耦是它的可測面）。⛔ 別把「注入 ≤ x bits」讀成 R0 界。
   單向例外：資訊不足→accuracy 上限的 Fano 方向有外部模板（2509.06861）— 帽方向可換算、
   增益方向仍斷。
3. **(iv) 的界鬆**：$H(o_t\mid\cdot)$ 是通道容量、非 task-relevant 量；task 相關那份（route 級
   ~2.5 bits）遠小於容量 — 「幾次好查詢就夠」是設計含義、「多互動＝多學」不是。
4. **外部件不承重**：承重命題全數一行證在檔。錨三級 — 【隊友正文級】十一件（§7、附
   section、隊友查證）；【訓練記憶】三件（Lindley 1956、Da Costa+ 2001.07203、Xu–Raginsky
   1705.07809）引前補驗。承重內部件：CT-1/2/3、⑰（2.5 bits）、C-ii′、Lemma 1/2、⑤''⑬⑱'。
5. **接口**：不動合成律、不動內化 gap 定義；只需一致於「(ii)@train 建通道、帳本為其進帳
   計價」。GRPO 卡呈裁點不變；本檔為 rung 0 添讀法（headroom＝幫浦輸入頻寬）、不加新工。

## §L6. 權重通道的縱切：info vs function（續章、9/6 午後追加）

_本體＝`THEORY-2026-09-06-info-vs-function.md`（下稱【F】；節號 F0–F7、承重命題自帶證明）。
⛔ 本章**不複製 F 檔正文** — 它是帳本側的正式接口：核心定義／承重命題的**摘要＋級別**、
與 L0–L5 的**對接表**、以及帳本自身需要的口徑修正。每一條都指回 F 節號，**引用一律引 F 檔**。
分級鐵則照帳本原制（定理級＝證在檔或逐字引已證件；命題級＝列明假設下證梗概；【推測級】＝未證）。_

### L6.0 這把刀切在哪〔導覽〕

帳本 §1 的四通道是**橫軸**：「bits 從哪個通道進來」。L6 補**縱軸**：「進到權重裡的東西是什麼」。

- **陳述性 info**＝與特定環境變數的互資訊。以 bits 計價、按環境數**線性**耗權重通道預算、
  換環境重抽即蒸發。
- **程序性 function**＝條件分佈核／算子本身。住 **Prop L0 的提取赤字欄**、**不耗 W-bits 預算**、
  fresh 環境上仍在；其定理級後果＝在新環境把**活通道**的 bits 兌現成表現的能力。

⇒ 通道 (i) 是唯一**兩種都存**的通道；而帳本 §1–§2 只按 $I(W;\theta)$ 記帳 ⇒ **只看得見 info
那一份**。function 那份住在每一次未來部署的赤字欄裡，四通道記帳對它**天生失明**。這就是
「帳本沒切的那一刀」的形式位置。〔【F】F0〕

**2×2 座標（content × 存放）**：帳本四通道全在 info 列記帳；function 列在帳本裡只有赤字欄
一個影子。GRPO 幫浦＝同時在兩列搬東西的搬運工 — **Remark L4.5 把每次 update 拆成
world-bits（搬運）＋sharpening（0 bits、模式銳化）兩份，那正是這把刀在帳本裡的既有切口**。
〔【F】F0 表、F3.3〕

### L6.1 核心定義（摘要級；完整式子與良定義性討論在【F】F1.1／F1.3）

- **Def F2（info content）**〔定義級〕：$\theta$ 的 instance 級 info＝$I(W_{e_j};\theta)$；
  family 級 info＝$I(\tilde\rho;\theta)$（$\rho$ 固定則此級以「對常數的知識」存在、任何 MI 帳
  都看不見）。以 bits 計價、受權重通道帽（帳本 (i)）。⇒ **family 級正是【分類】§2.2(a) 行
  「task 識別 bits」的正式身份**（兩檔可互引）〔【F】接口注意 4〕。
- **Def F3（function content）**〔定義級〕：$\Phi_C(\theta)$＝**fresh 環境上的提取赤字**
  （對 $W_{e'}\mid s,g,c$ 邊際化的 Bayes 預測與 $q_\theta$ 的期望 KL）。低＝function 多。
  不以 W-bits 計價、計價貨幣＝計算／描述長度。
  ⭐ **它不是**把 function 定義成「會在新環境 re-derive info」— 那是它的**定理級後果**
  （Cor F1.4）；定義本身是「$q_\theta$ 作為固定 kernel 在環境重抽下的赤字」，好處是
  $\Phi_\varnothing$ 也良定義（純邊際提取力）。
- **Def F4（value profile／重抽階梯）**〔定義級〕：$V_0$ 見過的查詢／$V_1$ 同環境 fresh
  $(s,g)$（＝stitch）／$V_2$ 同 family fresh 環境／$V_3$ fresh family。
  **純 info＝值集中 $V_0$；$k$ 級 function＝值平到 $V_k$**；中間態＝非退化的 **decay profile**
  （不是二值標籤）。主人句裡的「肌肉記憶」由此拆：背下的路線＝$V_0$–$V_1$、攤銷的搜索反射
  ＝宣稱到 $V_2$。

### L6.2 ⛔ 帳本現版對這刀失明 — 而那不是疏漏（Remark F1.5）

**Remark F1.5（單環境退化）〔定理級（定義性）〕**：$m{=}1$ ⇒ $W_{e_1}$ 是常數 ⇒
「這張圖的 info」以對常數的知識形式併入 effective-$\rho$，**任何 MI 帳都看不見它**
（MI 對常數恆 0）⇒ **info/function 之刀在數學上只在環境重抽存在時存在**。三個後果：

1. 帳本現版沒切這刀不是疏漏 — **單圖 régime 下刀無處落**；帶環境 ensemble、刀才隨之誕生。
2. ⛔ 現有 Int／idp 讀數**天生不分**「攤銷 BFS vs 背路線」— 兩者在單圖上住同一顆 $\theta$、
   行為可完全重合。**只有 $V_2$ 級測試（新圖）分得開，而 $V_2$ 一格證據都還沒有。**
3. ⚠️ 但 $V_1$（同圖新 $(s,g)$ 組合＝stitch 本義）**已部分可分**：查詢級重抽下查表死、
   組合算子活 — 這是現有證據真正站的台階。

⇒ **本章對帳本讀者的第一條紀律**：⛔ 別拿 L0–L5 的單環境帳去判「我們內化的是算子還是路線表」。
帳本能判的是 bits 從哪來、能兌現多少；判不了進 θ 的是哪一列。

### L6.3 承重命題摘要（級別照抄【F】F6 分級總表；證明全在 F 檔、⛔ 不重抄）

| 命題 | 一句話 | 級別 | F 節號 |
|---|---|---|---|
| **Prop F1** fresh 零資訊＝地板不可動 | F-A1 ⇒ $I(W_{e'};\theta)=0$；且 $\mathbb E[-\log q_\theta]=\{H_\rho(\tau\mid s,g)-I(\tau;c\mid s,g)\}+\Phi_C(\theta)$，$\theta$ 只出現在 $\Phi_C$ ⇒ **fresh 環境的資訊地板由 $\rho$ 與部署時活著的通道決定，θ 裡存了什麼都動不了它** | **定理級**（一行、證在檔） | F1.2 |
| **Prop F2** 記憶化缺口恆等式 | $\mathrm{NLL}_{\rm fresh}-\mathrm{NLL}_{\rm train}=\underbrace{I(\tau;\theta\mid s,g,c)}_{\text{info 項}}+\underbrace{[\Phi_C-\Phi^{\rm train}]}_{\text{function 漂移}}$ — generalization gap **恰好**拆成兩項；「info＝訓練環境上被融資、且恰好是換環境時蒸發的那部分」是**恆等式不是比喻** | **定理級**（證在檔） | F1.2 |
| **Prop F3** 通道兌現效率 $\xi$ ＋ **Cor F1.4** | $\xi_C:=\frac{\mathrm{NLL}(\varnothing)-\mathrm{NLL}(C)}{I(\tau;c\mid s,g)}=1+\frac{\Phi_\varnothing-\Phi_C}{I}$；fresh 價值分解＝（當場通道供給 bits）×（兌現效率）⇒ **「在新環境 re-derive info」＝$\xi_C\to1$ 對 family 一致成立**，是定義的後果 | 定理級 identity＋命題級量程 | F1.2 |
| **Prop F5** centering 對比性 | group advantage 對任何 $(s,g,e)$-可測 reward 平移**不變** ⇒ GRPO 幫浦物理上打不進「對這題／這圖的常數事實」、只打得進**同題候選之間的對比結構** ⇒ 這是「教程序」的**必要**形狀（⚠️ 非充分：$m{=}1$ 下對比資訊仍可壓成 route 選擇表） | **定理級**（一行、證在檔） | F3.3 |
| **Prop F7(i)(ii)** 預算算術 | iid 環境 ⇒ $\sum_j I(W_{e_j};\theta)\le I(W_{1:m};\theta)\le\min(H(\theta),I(\mathcal D;\theta))$；記憶 $m$ 個環境、每個買 $\beta$ bits 地板 ⇒ 至少耗 $m\beta$ bits 權重預算 — **info 融資按環境線性計費** | **定理級**（證在檔） | F4.2 |
| **Prop F7(iii)＋Cor** crossover | 環境通用算子成本＝固定描述長度、對 $m$ **常數**（可表示性錨＝Zhu+ 2505.12514 平行 BFS、CoT-DP 2305.15408，沿【分類】[驗]）⇒ $m\beta\gtrsim$ 預算時 function 化是唯一蓋得住全族的（近）最優解 | 命題級 | F4.2 |
| **Remark F1.5** 單環境失明 | 見 §L6.2 | 定理級（定義性） | F1.1 |

⚠️ **匯率 caveat（環境級版，整條沿 C-ii′ 進口）**：F4 全部是「**最優／可行性在哪**」的陳述、
**不是**「SGD 找不找得到」。bits 上不可負擔 $\ne$ CFM 的變異度量下有梯度壓力 —
postA1 的匯率斷裂在環境級同樣適用；動力學面【推測級】。〔【F】F4.2 caveat〕

### L6.4 與 L0–L5 的對接表

| 帳本條目 | 縱軸對接 | F 節號 | 性質 |
|---|---|---|---|
| **Prop L0** 帳本恆等式（地板＋赤字） | Prop F1 是它的**環境級同款**：把 $W$ 換成 fresh $W_{e'}$，$\theta$ 就只出現在赤字欄 ⇒ 「function 住赤字欄」不是比喻、是 L0 在環境重抽下的字面讀法 | F1.2 | 定理級延伸 |
| **Prop L1**（自生 u 恆 0、聯合零）＋**Cor CT-3** | **u＝function 的 scratchpad**：工作記憶的定義性質就是不含外界資訊（會漏 bits 進來的就不是 scratchpad、是通道）。⇒ 帳本語氣裡的「壞消息」（不合法期待 x1–x5）在縱軸上是**本義**。Cor CT-3 的縱軸讀法＝**function 鑄造不了 info**（自生配對建不了通道＝「程序不能無中生 bits」的訓練側定理） | F2 末段 | Remark（定理背書、無新主張） |
| **Prop L2(i)(ii)** 選擇界 | 內化「check 算子」後部署期可自營 proposal-check 迴圈，**但 map bits 仍要當場從外面來** — 每選擇 $\le\log N$、存量帽變成 $I(W;\text{活通道})$。學到的是**消費能力（function）、不是 bits** | F3.1(a) | 帳目定理級；可學性命題級 |
| **Prop L2(iii)** 自驗證＝0 新 bits | 縱軸讀法＝內化 check-function **只買赤字欄、買不到地板**；與 sharpening（2412.01951）的對接就在這一格 | F3.1(b) | 帳目定理級（逐字）；對接命題級 |
| **Prop L2(vi)** 驗證器分離 | 縱軸不動它 — 仍是**兩側佐證、非雙向夾**〔深審 M7〕；F 檔未新增主張 | — | 不變 |
| **Prop L3** 閉環觀測＋EIG Remark | 「學會預設獲取資訊的能力」＝acquisition function $\pi_q$（belief → 該問什麼）。**可學條件三條**：Q1 訓練分佈要含不確定性（$m{=}1$ ⇒ belief 退化成點質量 ⇒ EIG 泛函恆 0 ⇒ 對「查詢品質」的梯度恆 0；外錨＝BARL Thm 4.1，沿帳本【隊友正文級】）／Q2 query 要有 reward 訊號／Q3 通道要在訓練迴圈裡開著（最便宜上車位＝replan-on-collision，＝§4 表 (iv) 行） | F3.2 | Q1 命題級（退化證梗概）；Q2/Q3 命題級→實證級／架構事實 |
| **Prop L4(i)(ii)** 幫浦界 | **Prop F5** 加一層：advantage centering 使幫浦**天生濾掉題級常數事實**、只保對比結構。與 L4(ii)「advantage＝後處理只降不升」相容 — L4(ii) 說**總量**、F5 說**哪一種被濾掉** | F3.3 | 定理級 |
| **Remark L4.5** 防火牆 | 縱軸上這正是「幫浦輸出天生有 info 份與 function 份」的既有切口；指紋（大 $k$ 反超＝純 sharpening 簽名）照用、判別力前提仍掛條件 $\theta$ 版〔深審 F2〕 | F0、F3.3 | 不變、只換讀法 |
| **Thm L5** 四格 | 四格**全部是程序執行**：分解＝跑 (CL-w) 的 DP；提取＝serial depth 兌現 θ 存量；前沿疊加＝平行 BFS 的工作記憶；查詢規劃＝acquisition function 的計算（F3.2）。⇒ 對 u 的正確期待從此不在 bits 軸上：**問 u 的問題不是「它知道什麼」、是「它替哪個算子跑了哪一段」**（L5 四格＝合法選單） | F2 表 u 行 | Remark（L5 的縱軸重讀） |
| **§4 對應表** | F 檔 F2 給了系統各部件的縱軸歸類＋**誠實的證據現況**：攤銷 BFS＝persistent function 的**宣稱**（ER route .918 teacher-agnostic 已證＝另一軸；stitch＝$V_1$；**$V_2$ 零證據**）／特定 maze route 記憶＝persistent info（$m{=}1$ 下隱形）／e_target 幾何＝function 的**載體**（⚠️ 拆兩半：「一張圖的幾何」＝info、「圖→幾何的 encoder 算子」＝function）／verifier＝info 油箱**兼** function 老師／GRPO＝雙列搬運工 | F2 | 定義級套用＋狀態陳述 |
| **§6 誠實邊界 2**（兩道匯率） | F 檔逐字進口：⛔ 凡落在成功率上的預測都是**形狀**預測、非點估 | F6 邊界 1 | 不變 |

**核心 claim 的對接（一句話）**：一般化內化 claim ⓪「內化＝攤銷任意 teacher 的**計算**」
＝主張搬進 θ 的是**算子不是 bits** ⇒ **核心 claim 本來就是 function claim**。L6 明寫這個對接，
而 F-frame 是第一次把那半句變**可證偽**的（判別器＝【F】F5 的 FP-1/FP-2/FP-3）。
⚠️ 但【一般化】⓪ 的「general」混了兩軸 — teacher-agnostic（已證）與 environment-generic
（$V_2$、未測）；**建議 claim 敘事拆軸重述**〔【F】接口注意 3；⛔ 不在帳本管區、呈裁〕。

### L6.5 帳本側的接口修正（本章就地吸收【F】接口注意 1／2／8）

1. **〔已修〕Def L1 的 $W$ 是單環境切片** — §1 已補註：環境 ensemble 版＝外加 $e\sim\rho$ 期望、
   (i) 的帽升級為 per-family 加總版（Prop F7(i)）。**是延伸、不是矛盾。**
2. **〔已修〕終身帽 $\le H(M_V)$ 的多環境口徑** — §2.4 L4(iii)＋§4 表 GRPO 行已補：$M_V$ 是
   每圖各自的 ⇒ 多環境時 $\le\sum_{j\le m}H(M_V^{(e_j)})$、按環境數乘。⛔ 別把單圖的
   ≤961 bits 當多圖總帽。
8. **〔本章落點〕§5「探索是油門、驗證器才是油箱」的接續句**：
   > **油箱裝的是 bits、引擎本身是 function — 訓練能把引擎造好（攤銷），永遠造不出油。**

   〔前半＝Prop L2/L3 的既有帳；後半＝Prop F1 的字面義（fresh 環境上 θ 動不了地板）。
   ⇒ 主人句「沒有外界資訊的 icl 其實怎麼樣也不會得到更多資訊」在環境級的定理形。〕

**不屬帳本、列而不動**（留給各自管區）：接口 3（【一般化】⓪ 拆軸）／接口 4（【分類】§2.2(a)
與 Def F2 family 級互引，本章已單向引妥）／接口 5（teleport 破確定性 — Prop F1/F2 不受影響、
但凡引 (S1)／合成律處沿其原限定）／接口 6（u-scratchpad 與【敘事】§0 油門比喻一致、無衝突）／
接口 7（C-ii′ 進口環境級，postA1 不需改；$\xi$ 錶若落地位置在【分類】§4.3 P-ent 家族旁）。

### L6.6 L6 自身的誠實邊界

1. **本章零新主張、零新數字** — 全部是【F】的摘要與帳本側口徑對齊；承重命題的證明一律在 F 檔。
2. **$m{=}1$ 現況**：F1.5／F2 表對現有讀數的再詮釋是**狀態陳述**；$V_2$ 實驗一格都還沒有。
3. **FP-1/FP-2 需要「多圖訓練」臂＝新資料軸**，與多路線軸**正交** — multiroute 檔的 (a)–(e)
   選項都不自動供給它（teleport 仍是單圖）。此軸的資料構造未設計 ⇒ 呈裁。
4. **family 級（$V_3$）全檔只佔位**；跨 family 遷移零主張。
5. **外部件不承重**：F 檔承重命題全數一行／短證在檔、引文全沿 repo 攜帶、無新增 ID；
   本章亦無新增 ID。

### L6.7 呈裁點（沿【F】、帳本側複述）

① FP-1/FP-2 要的「多圖訓練」臂要不要開、怎麼開（多 maze 生成 vs OGBench 家族併集）；
② $\xi$ 錶（訓練圖上今天就可算：idp on/zero 雙腿＝$\mathrm{NLL}(C)/\mathrm{NLL}(\varnothing)$
現成、$I\approx2.5$ bits/題＝⑰ 轉引）要不要進儀器清單；
③「攤銷 BFS」claim 是否按 L6.4 末段拆軸重述。

---

## 7. 引用清單

【隊友正文級・附 section】：2401.01879（BoN KL Thm 3.1、§3.1–3.2）、2404.01730（可達 §4.2
Thm 2、Sanov）、2502.12118（verifier 分離 §5.2 Thm 5.4/5.7/5.8）、2509.06861（compute-only
§5.3 Thm 1＋Cor 1、App H）、chao-dyn/9905039（Touchette & Lloyd Thm 2）、2505.20561（BARL
Thm 4.1）、2412.01951（Sharpening）、2504.13837（RLVR 批評）；gap：2502.07202、2505.05470、2402.14083。
沿上游[驗]：2305.15408、2310.07923、2402.12875（CoT 三件）；2505.12514（Zhu+）；
2605.08732（開環誤差）；2209.15189（context distillation）。
【訓練記憶・引前補驗】：1705.07809、2001.07203、Lindley 1956。教科書級：DPI、鏈式、$H$ 上界。
家內：NOTE-0906-context-taxonomy（CT-1/2/3、三源表、C1–C4、P-swap）、THEORY-0906-postA1
（C-ii′）、THEORY-0905-composition-law（Lemma 1/2、(CL-w)、(S1)/(S1′)）、DESIGN-0906-grpo-thoughts、
FINDINGS ⑤''⑬⑰⑰'⑱'；luna-2026-09-05-closing.md（主人 23:39 定調原話、已逐字對）。
**§L6 續章專屬**：THEORY-0906-info-vs-function（【F】F0–F7；Def F1–F4、Prop F1/F2/F3/F5/F6/F7、
Cor F1.4、Remark F1.5、FP-1~5）— **L6 的一切承重內容以該檔為準**、本章只作導覽與對接。
L6 引用的外部件全部沿 F 檔攜帶（2505.12514 Zhu+、2305.15408 CoT-DP、2404.15758 filler tokens
〔皆[驗]@分類〕；2412.01951 Sharpening、2505.20561 BARL Thm 4.1〔【隊友正文級】@本檔〕）—
**L6 無新增 ID**。

_帳本完（§0–§7＋§L6 續章）。呈裁點：①部署期 BoN 臂要不要排（rung 0 儀器順手可量）
②Conj L4.4 分辨實驗掛 GRPO rung 0/2 還是另立 ③【訓練記憶】餘三件補驗排程；
④⑤⑥＝§L6.7 三條（多圖訓練臂／$\xi$ 錶／攤銷 BFS claim 拆軸）。_
