# THEORY — latent thinking 的資訊帳本：四通道記帳＋選擇界＋幫浦定理（v0、2026-09-06）

_理論使魔（Fable 級、主人睡前指定；兩並行之「formulation」線；隊友前例包本夜到貨、錨已織入）。
主人的問題：「自生的 u 對世界零資訊、只能計算 — 能 formulate 成 theory 嗎？那我們的 u
可以思考什麼？能不能靠 embedding space 探索、或跟環境互動，補 u 的資訊？」
上游（唯讀）：NOTE-0906-context-taxonomy（CT-1/2/3）、DESIGN-0906-grpo-thoughts、THEORY-
0905-composition-law（溫度族）、THEORY-0906-postA1（C-ii′）。ルナ候選統一圖＝採納並收緊。
⛔ 分級鐵則：**定理級＝證明在檔（多為一行 DPI）或逐字引已證件；Conj＝未證；Remark＝解讀；
啟發式＝量級粗估**。外部文獻三級：[驗]＝沿上游攜帶；【隊友正文級】＝前例調研使魔本夜查證
過正文（附 section）；【訓練記憶】＝未覆核（本使魔 web search 額度歸零）、承重前補驗。
本檔所有承重命題皆自帶證明、外部件只作錨與站位。_

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
⚠️ 地板以 NLL（$T{=}1$）計價；R0 是 $T\to0$ 泛函 — 換算屬 Lemma 1/2 與匯率斷裂管區（§6-2）。

---

## 1. 設定與四通道（Def L1）

記號沿 CT-1：查詢 $(s,g)$；world 變數 $W$＝環境／episode 資料側任何變數（佔據圖 $E$、BFS
距離場 $D$、真 route 皆是）；除註明外全程條件在固定 $\theta$ 上（部署期帳本）。

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
Coconut 形回饋、樹狀自我展開）：$I\big(W;u_{1:N}\mid s,g\big)=0$。
_證_：歸納 — $u_{1:N}$ 是 $\sigma(s,g,\varepsilon_{1:N})$-可測、$\varepsilon_{1:N}\perp W\mid(s,g)$。∎
⇒ 把「多想幾條、想深一點就會多知道路」正式殺死：計算規模不是資訊變數；迭代買的是 serial
depth（taxonomy C1/C2、CoT 三件），記赤字欄。（訓練側對偶＝Cor CT-3，引用不重證。）
⭐ 外部孿生【隊友正文級】：compute-only test-time scaling 對 latent truth 零新資訊 — DPI 沿
$A\to\hat p\to T_k\to\hat R$ 鏈＋Fano 轉 accuracy 上限（2509.06861 §5.3 Thm 1＋Cor 1、證
App H）＝L1 的 TTS 版、且附帶 bits→accuracy 單向匯率模板（收進 §6-2）。

### 2.2 Prop L2 — 探索×選擇：selection channel 界〔定理級、證在檔〕

設候選 $u_i=f_\theta(s,g,\varepsilon_i)$、$i=1..N$、$\varepsilon_{1:N}\perp(W,M_V)\mid(s,g)$；
評分 $v_i=V(u_i;s,g,M_V)$；選擇 $J=\arg\max_i v_i$（平手用獨立噪音）；輸出 $u^*=u_J$。則：

- **(i) 選擇預算**：$I(W;u^*\mid s,g)\;\le\;H(J\mid s,g)\;\le\;\log N$。
  _證_：$u^*$ 是 $(u_{1:N},J)$ 的函數 ⇒ DPI；鏈式
  $I(W;u_{1:N},J\mid s,g)=\underbrace{I(W;u_{1:N}\mid s,g)}_{=0\ (\text{L1b})}+I(W;J\mid u_{1:N},s,g)\le H(J\mid\cdot)\le\log N$。∎
- **(ii) 驗證器存量帽**：$I(W;u^*\mid s,g)\;\le\;I(W;M_V\mid s,g)$。
  _證_：$(u_{1:N},J)$ 是 $\sigma(s,g,\varepsilon_{1:N},M_V)$-可測、$\varepsilon\perp(W,M_V)\mid(s,g)$
  ⇒ $I(W;u^*\mid s,g)\le I(W;\varepsilon,M_V\mid s,g)=I(W;M_V\mid s,g)$。∎
  — 從驗證器擠不出驗證器自己沒有的東西。
- **(iii) 自驗證＝0 新 bits**：$M_V\perp W\mid(s,g)$（例：critic 是 $\theta$-可測、同資料訓的）
  ⇒ $I(W;u^*\mid s,g,\theta)=0$。self-BoN／self-consistency 仍可能**有用** — 但收益全記
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
  有 $M_V$ 可證地贏；與 L2 上界組合成雙向夾。

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
- **(iv) Conj L4.4（有效注入 ≪ 上界）**：可兌現成行為改善的量另有匯率 — bits 以 NLL 計價、
  CFM/PG 以變異計價（C-ii′、postA1 §3），且 (i) 粗估（$16\times8\times\log_2k\approx1400$
  bits/update〔啟發式〕）遠超實際學習量。**有效流量未定量**；rung 0 的 pass@G−pass@1 gap
  是它的操作型輸入量測（設計卡 §3.4）。降 Conj、分辨實驗已在設計卡 rung 0/2。
- **Remark L4.5（⭐ 防火牆：sharpening ≠ 注入）【錨隊友正文級】**：GRPO 對 θ 的改動分兩份 —
  reward 攜帶的 world-bits（搬運、本命題管）與純模式銳化（$z$ 自身隨機性、0 bits＝溫度族往
  $T\to0$ 壓）。外部批評打後者：self-improvement＝靠 verification-generation gap **提取**、
  不能創造（Sharpening 2412.01951）；RLVR 只提升取樣效率、大 $k$ 時 base pass@k 反超
  （2504.13837）。**防火牆＝L2(iii) vs (ii) 分家**：自驗證 RLVR 的 $M_V\perp W\mid(s,g)$ ⇒
  注入恰 0（純 sharpening、批評成立）；我們的 $M_V=E$-圖＋BFS 場、$I(W;M_V\mid s,g)>0$ ⇒
  真 (iii) 注入、批評不適用。可測指紋〔啟發式〕：sharpening 簽名＝大 $k$ 反超 base；外部注入
  預測＝中等 $k$ 段不反超（NF 全支撐、$k\to\infty$ 同飽和）；加 §3.3 指紋＋P-swap、三器分辨。
- **Remark（容量對齊）〔啟發式〕**：$G=8$ ⇒ 每群選擇預算 $\log_28=3$ bits ≥ ⑰ 實測
  instance 級 route 資訊 ~2.5 bits — 一群頻寬剛好夠指定一條 route；$G$ 的資訊論選型論證
  （要更強訊號→加 $G$＝加頻寬，與設計卡藥單順序一致）。

---

## 3. 「u 可以思考什麼」設計定理（Thm L5）

**陳述**：CT-1 régime 下，自生 thinking 的全部合法價值＝Prop L0 赤字欄；合法標的四格 —

1. **分解（decomposition）**：u 實體化合成律 (CL) 中繼點 $m$、思考空間跑 $\bigoplus_m$ 的
   DP — 難分佈拆兩段簡單因子（Lemma 1 log-semiring；(S1)）。⚠️「因子各自簡單」仍猜測級
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
| 佔據圖＋BFS 檢查（免費 reward） | (iii) 的 $M_V$ | 儀器現成（`_EOCC`＋dist 場；rung 0 可直接做） | **部署期 BoN 臂未建**：eval 抽 $G$ 條、免費驗證器選 1 — (iii) 最便宜消費者、零訓練 | 每候選 reward ≤ $\log k\approx$10–13 bits；BoN 選擇 ≤ $\log G$（$G{=}8$⇒3 bits） | pass@G−pass@1 gap（rung 0）＝這條臂的價值量測**兼**幫浦輸入量測 — 一魚兩吃 |
| route 查圖 eval（idp on-mode） | (ii) 部署版 | 已建（⑤'' 帶查(map) 語意） | 部署時 route-ix 可得性（訓練圖之外） | ~2.5 bits/題（⑰；走廊 ~1、⑰'） | (ii) 是唯一「免搜索直給」的 instance 通道；增益上限＝地板下移 ≤ I |
| GRPO 卡 | (iii)→(i) 幫浦 | 設計 v0；rung 0–4 未跑；前提＝base 收斂（⑱'） | 有效流量（Conj L4.4）；退化群頻寬實測 | ≤ $B_gG\log k$/update；終身 ≤ $H(M_V)$（≤961 bits 級） | 退化群錶＝頻寬錶；$G$ 選型＝容量對齊；「rung 0 無 headroom ⇒ 幫浦無輸入」是定理不是比喻 |
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
  驗證器的地圖搬進權重** — rung 0 無 headroom ⇒ 幫浦無輸入（設計卡 §5.2 的帳本身份）；
  且「搬運 vs 銳化」有防火牆與指紋（L4.5）。

---

## 6. 分級總表與誠實邊界

| 條目 | 級別 | 備註 |
|---|---|---|
| Prop L0 帳本恆等式 | 定理級（恆等式） | 交叉熵分解；NLL 貨幣 |
| L1(a)(b) 純自生聯合零 | 定理級（引 CT-1＋一行歸納） | 外部孿生 2509.06861【隊友正文級】 |
| L2(i)–(iv) 選擇界 | 定理級（證在檔） | $W$-bits 可達性仍啟發式；傾斜側定理＋可達（2401/2404） |
| L2(vi) 驗證器分離 | 引用級【隊友正文級】 | 2502.12118；與上界組成雙向夾 |
| L3 閉環界＋取等 | 定理級（證在檔） | 取等＝觀測確定性；祖版 chao-dyn/9905039 |
| L4(i)(ii)(iii) 幫浦界 | 定理級（證在檔） | 計 world-bits、非總行為改變 |
| Conj L4.4 有效注入 | **Conj** | 匯率斷裂 C-ii′ 承接；分辨＝rung 0/2＋指紋 |
| Remark L4.5 防火牆 | Remark＋指紋〔啟發式〕 | 2412.01951/2504.13837 對 L2(iii)/(ii) 分家 |
| Thm L5 合法四格 | 定理級框架＋各格承上游分級 | L5.1 增益半句猜測級；L5.4 依 EIG【訓練記憶】 |
| 容量對齊 3 vs 2.5、表內 bits 粗估 | 啟發式 | 全部上界方向、未扣結構冗餘 |

**誠實邊界**：
1. **嚴的**：L0（恆等式）、L1（恰 0）、L3 在確定性域（等式）。**只上界的**：L2 的 $\log N$
   （$W$-側可達性要覆蓋＋排序對）、L4 計數界（gross bits；有效流量＝Conj L4.4）。
2. **貨幣**：帳本以 NLL/bits 計價；R0 是 $T\to0$ 泛函 — bits 進帳後要過兩道匯率（bits→變異
   ＝C-ii′；$T{=}1\to T\to0$＝Lemma 1/2、P2 解耦）。⛔ 別把「注入 ≤ x bits」讀成 R0 界。
   單向例外：資訊不足→accuracy 上限的 Fano 方向有外部模板（2509.06861）— 帽方向可換算、
   增益方向仍斷。
3. **(iv) 的界鬆**：$H(o_t\mid\cdot)$ 是通道容量、非 task-relevant 量；task 相關那份（route 級
   ~2.5 bits）遠小於容量 — 「幾次好查詢就夠」是設計含義、「多互動＝多學」不是。
4. **外部件不承重**：承重命題全數一行證在檔。錨三級 — 【隊友正文級】十一件（§7、附
   section、隊友查證）；【訓練記憶】三件（Lindley 1956、Da Costa+ 2001.07203、Xu–Raginsky
   1705.07809）引前補驗。承重內部件：CT-1/2/3、⑰（2.5 bits）、C-ii′、Lemma 1/2、⑤''⑬⑱'。
5. **接口**：不動合成律、不動內化 gap 定義；只需一致於「(ii)@train 建通道、帳本為其進帳
   計價」。GRPO 卡呈裁點不變；本檔為 rung 0 添讀法（headroom＝幫浦輸入頻寬）、不加新工。

## 7. 引用清單

【隊友正文級・附 section】：2401.01879（BoN KL Thm 3.1、§3.1–3.2）、2404.01730（可達 §4.2
Thm 2、Sanov）、2502.12118（verifier 分離 §5.2 Thm 5.4/5.7/5.8）、2509.06861（compute-only
§5.3 Thm 1＋Cor 1、App H）、chao-dyn/9905039（Touchette & Lloyd Thm 2）、2505.20561（BARL
Thm 4.1）、2412.01951（Sharpening）、2504.13837（RLVR 批評）；gap：2502.07202、2505.05470、2402.14083。
沿上游[驗]：2305.15408、2310.07923、2402.12875（CoT 三件）；2505.12514（Zhu+）；
2605.08732（開環誤差）；2209.15189（context distillation）。
【訓練記憶・引前補驗】：1705.07809、2001.07203、Lindley 1956。教科書級：DPI、鏈式、$H$ 上界。
家內：NOTE-0906-context-taxonomy（CT-1/2/3、三源表、C1–C4、P-swap）、THEORY-0906-postA1
（C-ii′）、THEORY-0905-composition-law（Lemma 1/2、(S1)）、DESIGN-0906-grpo-thoughts、
FINDINGS ⑤''⑬⑰⑰'⑱'；luna-2026-09-05-closing.md（主人 23:39 定調原話、已逐字對）。

_帳本完。呈裁點：①部署期 BoN 臂要不要排（rung 0 儀器順手可量）②Conj L4.4 分辨實驗掛
GRPO rung 0/2 還是另立 ③【訓練記憶】餘三件補驗排程。_
