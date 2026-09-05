# THEORY — latent thinking 的資訊帳本：四通道記帳＋選擇界＋幫浦定理（v0、2026-09-06）

_理論使魔（Fable 級、主人睡前指定；兩並行之「formulation」線 — 隊友做文獻調研、獨立跑）。
主人的問題：「自生的 u 對世界零資訊、只能計算 — 能 formulate 成 theory 嗎？那我們的 u
可以思考什麼？能不能靠 embedding space 探索、或跟環境互動，補 u 的資訊？」
上游（唯讀）：NOTE-0906-context-taxonomy（Thm CT-1／Cor CT-2/CT-3、三源分類、遷移鏈）、
DESIGN-0906-grpo-thoughts（免費幾何 reward）、THEORY-0905-composition-law（溫度族）、
THEORY-0906-postA1（C-ii′ 匯率斷裂）。ルナ的候選統一圖（四通道帳本）＝本檔採納並收緊。
⛔ 分級鐵則：**定理級＝證明在檔（多為一行 DPI）或逐字引已證件；Conj＝未證；Remark＝解讀；
啟發式＝量級粗估**。外部文獻：[驗]＝上游已抓正文、沿上游攜帶；【訓練記憶】＝本使魔背景
知識、未上網覆核（web search 額度本 session 歸零）— 引用承重前補驗 ID。本檔所有承重
命題皆自帶證明、不依賴任何【訓練記憶】件。_

---

## 0. 一句話＋帳本恆等式

> **latent thinking 的資訊帳本：世界資訊只有四個進帳通道 — (i) 權重（過去資料）、
> (ii) conditioning 輸入、(iii) 驗證器／選擇、(iv) 新觀測（閉環互動）。純自生 u 四個都不是
> — 它恆注入 0 bits（CT-1），其全部價值在**提取赤字**那一欄（計算）。探索本身不進帳；
> 探索×外部驗證器每次選擇進帳 ≤ log N；環境互動每步進帳 ≤ H(obs|belief)（確定性世界取等）；
> GRPO＝把 (iii)/(iv) 的進帳打進 θ 的幫浦（每 update ≤ B·G·log k、終身 ≤ 驗證器自己的存量）。
> ⇒ 主人的兩問（探索補資訊、跟 GRPO 結合）是同一組命題的兩半：同一批 bits、
> 推論期消費（transient）或經幫浦入 θ（persistent）。**

**Prop L0（帳本恆等式）〔定理級〕**：對任何推論期 conditioning 束 $C$ 與任何實作 sampler $q$：
$$\mathbb E[-\log q(\tau\mid s,g,C)]
=\underbrace{H(\tau\mid s,g)-I(\tau;C\mid s,g)}_{\text{資訊地板（只有 bits 動得了）}}
+\underbrace{\mathbb E\,\mathrm{KL}\big(p^*(\cdot\mid s,g,C)\,\|\,q(\cdot\mid s,g,C)\big)}_{\text{提取赤字（計算住這）}}.$$
_證_：交叉熵分解＋ $H(\tau\mid s,g,C)=H(\tau\mid s,g)-I(\tau;C\mid s,g)$（互資訊定義）。∎
**「計算買提取效率、不買 bits」的正式陳述**：任何不注入互資訊的操作（更深的 $f_\theta$、
更多自生樣本、自驗證搜索）只能動第二項；第一項的每一分移動都恰好以注入的 $I$ 計價。
自生 $u$：$I(\tau;u\mid s,g)=0$（CT-1 取 $W:=\tau$）⇒ 地板不動 — Cor CT-2 的帳本座標。
⚠️ 地板以 NLL 計價（$T{=}1$ 泛函）；R0／成功率是 $T\to0$ 泛函 — 兩種貨幣的換算屬
合成律 Lemma 1/2 與 postA1 匯率斷裂的管區，本檔不重證（§6 誠實邊界 3）。

---

## 1. 設定與四通道（Def L1）

記號沿 CT-1：查詢 $(s,g)$；world 變數 $W$＝環境／該 episode 資料側的任何變數（佔據圖 $E$、
BFS 距離場 $D$、真 route 皆是）；除註明外全程條件在固定 $\theta$ 上（部署期帳本）。

- **(i) 權重通道**：$\theta=\mathrm{Alg}(\mathcal D,R)$、$R$ 為訓練噪音 ⇒ Markov 鏈
  $W-\mathcal D-\theta$ ⇒ $I(W;\theta)\le I(W;\mathcal D)$〔定理級一行；「information in
  weights」框架＝Xu–Raginsky 1705.07809【訓練記憶】〕。進帳時刻＝訓練期；部署期唯讀。
- **(ii) conditioning 通道**：推論期輸入 $c$；注入量＝$I(W;c\mid s,g)$、NLL 可兌現額
  恰等於它（Prop L0）。實例：ICL 範例（task 級）、route-ix 查圖（instance 級 ~2.5 bits、⑰）。
- **(iii) 驗證器／選擇通道**：外部資源 $M_V$（我們：資料建佔據圖 $E$＋BFS 場 $D$）經
  評分 $v=V(u;s,g,M_V)$ 進場；bits 經**選擇變數**（BoN 的 $J$）或 **reward 值**（GRPO）注入。
- **(iv) 觀測通道**：閉環互動 $o_t=O(W,\text{state}_t,a_t,\xi_t)$；bits 經觀測序列注入。

三源分類（taxonomy §2.2）嵌入：(a) ICL＝(ii)@推論、(b) oracle＝(ii)@訓練→(i)、(c) 自生＝
四者皆非（零進帳）。帳本的增量＝把 (iii)(iv) 升為一級來源、並把 GRPO 定位成 (iii)→(i) 搬運。

---

## 2. 帳本定理組（每通道一條 DPI 級命題）

### 2.1 Prop L1 — 純自生：恆 0，且「想更久／更多條」聯合仍 0〔定理級〕

(a) $I(W;u\mid s,g,\theta)=0$〔＝Thm CT-1 逐字，引用〕。
(b) **聯合零**：任何自足的思考程序 — $u_k=f_\theta(s,g,u_{<k},\varepsilon_k)$、
$\varepsilon_{1:N}\perp W\mid(s,g)$、無外部呼叫 — 對任意 $N$、任意深度／拓撲（迭代、
Coconut 形回饋、樹狀自我展開）：$I\big(W;u_{1:N}\mid s,g\big)=0$。
_證_：歸納 — $u_{1:N}$ 是 $\sigma(s,g,\varepsilon_{1:N})$-可測、$\varepsilon_{1:N}\perp W\mid(s,g)$。∎
⇒ 把「多想幾條、想深一點就會多知道路」正式殺死：計算規模不是資訊變數。迭代買的是
serial depth（taxonomy C1/C2、CoT 三件），帳本上記在赤字欄、不記在地板欄。
（訓練側對偶：自生配對建不了通道＝Cor CT-3，引用不重證。）

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
- **(iii) 自驗證＝0 新 bits**：$M_V\perp W\mid(s,g)$（例：critic 是 $\theta$-可測的、同資料
  訓的）⇒ $I(W;u^*\mid s,g,\theta)=0$。self-BoN／self-consistency 仍可能**有用** — 但其
  收益全記赤字欄（改善對 $\theta$ 存量的提取），零新進帳。⛔ 兩個帳欄不准混。
- **(iv) 搜索推廣**：任何「唯一 world 存取＝驗證器呼叫」的程序（beam、MCTS、迭代
  refine-with-check），選擇轉錄 $J_{1:K}$：$I(W;\text{output}\mid s,g)\le\sum_k\log|\mathcal J_k|$
  且仍 $\le I(W;M_V\mid s,g)$。_證_：(i) 的鏈式對 $K$ 輪疊加＋(ii) 不變。∎
- **(v) Remark（兩種 log N 別混）**：分佈傾斜讀法 $\mathrm{KL}(\pi_{\rm BoN}\|\pi_{\rm prior})
  \le\log N$（folklore 精確式 $\log N-\tfrac{N-1}{N}$；exactness 分析 Beirami+ 2401.01879、
  overoptimization 用法 Gao+ 2210.10760【皆訓練記憶】）講的是**分佈動多少**、(i) 講的是
  **對 $W$ 知道多少** — 同為 $\log N$ 是兩條不同的界。⚠️ (i) 是上界不是可達性：真拿到
  ~2.5 bits 需要候選池蓋到正確 route（coupon-collector、$N\gtrsim1/p_{\min}$）＋驗證器
  排序對 —〔啟發式〕。

### 2.3 Prop L3 — 閉環觀測：每步 ≤ H(obs|belief)、確定性世界取等〔定理級〕

互動史 $h_t=(s,g,a_{1:t},o_{1:t})$、行動 $a_t=\pi(h_{t-1},u\text{-stuff},\varepsilon_t)$（自生、
L1 適用）。則
$$I(W;h_T\mid s,g)=\sum_{t}I(W;o_t\mid h_{t-1},a_t)\;\le\;\sum_t H(o_t\mid h_{t-1},a_t),$$
且 $o_t$ 給定 $(W,h_{t-1},a_t)$ 確定（我們的 gridworld：碰撞位元、局部 patch）時**逐項取等**
— 實現的 surprisal 就是入帳的 map bits。
_證_：鏈式分解；$I(W;a_t\mid h_{t-1})=0$（L1b）；$I=H-H(\cdot\mid W,\cdot)$、確定性時第二項 0。∎
- **天花板**：$\sum_t\le H(W\mid s,g,\theta)$ — 學不到比「還不知道」更多。
- **Remark（EIG＝導向、不是來源）**：選 $a_t$ 極大化 $I(W;o_t\mid h_{t-1},a_t)$＝Bayesian
  experimental design（Lindley 1956）＝active inference 的 epistemic value／EIG 項
  （Da Costa+ 2001.07203）【皆訓練記憶】。u 的合法角色＝算這個 argmax（§3 第 4 格）；
  bits 由世界供給。
- 我們域的粗估〔啟發式〕：replan-on-collision＝每步 ≤1 bit（碰撞位元）；局部 $3\times3$
  patch ≤8 bits/步；全圖存量 $H(E)\le$ 格數 bits（31×31 ≤ 961、結構先驗下遠小於此）。

### 2.4 Prop L4 — RL 幫浦：GRPO 每 update 打進 θ 的 bits 上界〔(i)–(iii) 定理級；(iv) Conj〕

設定沿 DESIGN-0906：第 $t$ 步取查詢批 $Q_t$（$B_g$ 題）、每題 $G$ 條
$z\sim\pi_{\theta_t}(\cdot\mid c_q)$（自生）、reward $r=R(z;s,g,M_V)\in\mathcal R$、
$|\mathcal R|=k$（可分辨等級數）、advantage $\hat A=h(r_{1:B_gG})$（群內標準化）、
$\theta_{t+1}=U(\theta_t,Q_t,z_{1:B_gG},\hat A)$。

- **(i) 每 update 界**：$I(W;\theta_{t+1}\mid\theta_t,Q_t)\;\le\;B_g\,G\,\log k$（binary reward
  ＝$B_gG$ bits）。_證_：給定 $(\theta_t,Q_t)$，$z\perp W$（L1b）⇒
  $I(W;\theta_{t+1}\mid\cdot)\le I(W;z,\hat A\mid\cdot)=0+I(W;\hat A\mid z,\cdot)
  \le H(\hat A\mid\cdot)\le H(r_{1:B_gG}\mid\cdot)\le B_gG\log k$（$\hat A$ 是 $r$ 的函數）。∎
- **(ii) advantage＝後處理（DPI）**：$r\mapsto\hat A$ 只能降不能升；退化群（reward 全同
  ⇒ $\hat A\equiv0$）貢獻**恰 0** — 設計卡的「退化群比例」錶＝幫浦輸入頻寬錶（字面義）。
- **(iii) 終身帽**：$\theta_T=F(\theta_0,\{Q_t\},\{\varepsilon_t\},M_V)$ ⇒
  $I(W;\theta_T\mid\theta_0,\{Q_t\})\;\le\;I(W;M_V\mid\{Q_t\})\;\le\;H(M_V)$ —
  幫浦一輩子打進 θ 的不超過驗證器資源自己的存量（＝把地圖搬進權重的上限）；
  同時 $\le\sum_t$(i)。_證_：同 L2(ii) 的可測性論證。∎
- **(iv) Conj L4.4（有效注入 ≪ 上界）**：可兌現成行為改善的量另有匯率 —
  bits 以 NLL 計價、CFM/PG 以變異計價（C-ii′ 匯率斷裂、postA1 §3），且 (i) 的粗估
  （$16\times8\times\log_2 k\approx1400$ bits/update、$k\approx10^3$〔啟發式〕）顯然遠超
  實際學習量。**幫浦的有效流量未定量**；rung 0 的 pass@G−pass@1 gap 是它的操作型
  輸入量測（設計卡 §3.4）。降 Conj、分辨實驗已在設計卡 rung 0/2。
- **Remark（幫浦不創造，只搬運＋降溫）**：GRPO 對 θ 的改動分兩份 — reward 攜帶的
  world-bits（本命題管）與純模式銳化（$z$ 自身隨機性驅動、0 bits、＝溫度族往 $T\to0$ 壓）。
  兩者的分辨器＝設計卡 §3.3 指紋（zero/on 收窄 vs 齊漲）＋P-swap 型探針（taxonomy §4.3）。
- **Remark（容量對齊）〔啟發式〕**：$G=8$ ⇒ 每群選擇預算 $\log_2 8=3$ bits ≥ ⑰ 實測
  instance 級 route 資訊 ~2.5 bits — 一群的頻寬「剛好夠指定一條 route」；這是 $G$ 的
  資訊論選型論證（要更多樣化 reward 訊號→加 $G$＝加頻寬，與設計卡藥單順序一致）。

---

## 3. 「u 可以思考什麼」設計定理（Thm L5）

**陳述**：CT-1 régime 下，自生 thinking 的全部合法價值＝Prop L0 赤字欄；具體合法標的四格，
每格有既證背書 —

1. **分解（decomposition）**：u 實體化合成律 (CL) 的中繼點 $m$、在思考空間跑
   $\bigoplus_m$ 的 DP — 把難的條件分佈拆成兩段簡單因子（Lemma 1 log-semiring；(S1)）。
   ⚠️「因子各自簡單」仍是猜測級（taxonomy §4.2）— 分解合法、增益待驗。
2. **提取（extraction）**：把 θ 存量（地圖／路線知識）展開成顯式計畫 — serial depth
   貨幣（CoT 三件 2305.15408/2310.07923/2402.12875、沿上游[驗]），成立要件 C1–C3
   （迭代或離散瓶頸、serial 問題類、密集監督）。
3. **搜索前沿疊加（superposition）**：連續 u 同時線性編碼多條候選 route＝平行 BFS
   （Zhu+ 2505.12514、沿上游[驗]）—「在 embedding space 探索」的合法形式＝**在自家
   假設空間搜索 θ 已有的東西**，不是對世界採樣。
4. **查詢規劃（query planning）⭐**：計算「該問什麼」— EIG 的 argmax（L3 Remark）、
   BoN 的 proposal 生成（L2 的 $f_\theta$ 端）、GRPO 的行為分佈。u 不能供給 bits，
   但能決定 bits 從哪裡、以多大效率進來 — 自生思考是 (iii)/(iv) 通道的**導向系統**。
   這格是對主人「那 u 可以思考什麼」最深的一答。

**⛔ 不合法期待清單**（每條有定理擋）：
(x1) 憑空生 instance／route bits — 違 L1(a)；(x2) 想更久、抽更多條→更多資訊 — 違 L1(b)；
(x3) 自驗證 BoN 注入新 bits — L2(iii)＝0（可改善提取，帳欄別混）；(x4) 拿自生 $(u,\tau)$
配對當訓練訊號建通道 — Cor CT-3；(x5) 靠自生達成 off-manifold 探索 — 自生按定義
on-manifold（taxonomy §4.1）。

---

## 4. 與我們系統的對應表

| 系統件 | 通道 | 現況 | 缺格 | bits 粗估〔啟發式〕 | 設計含義 |
|---|---|---|---|---|---|
| 佔據圖＋BFS 檢查（免費 reward） | (iii) 的 $M_V$ | 儀器現成（`_EOCC`＋dist 場；rung 0 可直接做） | **部署期 BoN 臂未建**：eval 抽 $G$ 條、免費驗證器選 1 — (iii) 的最便宜消費者、零訓練 | 每候選 reward ≤ $\log k\approx$10–13 bits；BoN 選擇注入 ≤ $\log G$（$G{=}8$⇒3 bits） | pass@G−pass@1 gap（rung 0）＝這條臂的價值量測**兼**幫浦輸入量測 — 一魚兩吃 |
| route 查圖 eval（idp on-mode） | (ii) 部署版 | 已建（⑤'' 帶查(map) 語意） | 部署時 route-ix 的可得性假設（訓練圖之外） | ~2.5 bits/題（⑰；走廊級 ~1、⑰' 紀律） | (ii) 是唯一「免搜索直給」的 instance 通道；它的增益上限＝地板下移 ≤ I |
| GRPO 卡 | (iii)→(i) 幫浦 | 設計 v0；rung 0–4 未跑；前提＝base 收斂（⑱'） | 有效流量（Conj L4.4）；退化群頻寬實測 | ≤ $B_gG\log k$/update；終身 ≤ $H(M_V)$（≤961 bits 級） | 退化群錶＝頻寬錶；$G$ 選型＝容量對齊；「rung 0 無 headroom ⇒ 幫浦無輸入」在帳本裡是定理不是比喻 |
| 閉環 replan | (iv) | **未建** | 全部 | 碰撞位元 1 bit/步（確定性⇒取等）；patch ≤8 bits/步 | 最便宜 (iv)＝replan-on-collision；開環 73% vs 閉環 92%（2605.08732、沿上游）＝(iv) 的實證價值錨 |
| oracle 錨→內化（主線） | (ii)@train→(i) | 已建（遷移鏈本體） | 內化 gap 定量（另線） | 2.5 bits/episode × 資料量、帽＝$I(W;\mathcal D)$ | 帳本把遷移鏈收為特例：資訊建通道、權重存、計算駛 —**驗證器與環境續帳、RL 入帳** |

---

## 5. 主人兩問的直接答案

- **「在 embedding space 探索，能補 u 的資訊嗎？」** 探索**本身不能** — L1(b)：任意多條、
  任意深的自生探索聯合仍恰 0 bits；它合法買的是搜索計算（L5.3 疊加前沿）。
  **探索×外部驗證器能** — L2：每次選擇注入 $\le\min(\log N,\;I(W;M_V\mid s,g))$；
  自驗證器＝0 新 bits（L2(iii)）。一句話：**探索是油門，驗證器才是油箱。**
- **「跟環境互動補？」** **能** — L3：每步 $\le H(o_t\mid\text{belief})$、我們的確定性世界
  逐步取等（碰撞位元＝1 bit/步）；u 的角色＝挑最大 EIG 的那一步（L5.4 查詢規劃）。
- **「跟 GRPO 的結合？」** ＝幫浦定理 L4：同一批 (iii)/(iv) bits 有兩條出路 —
  推論期當場消費（per-episode、transient、免訓練：BoN／replan）或經 GRPO 打進 θ
  （persistent、amortized；每 update ≤ $B_gG\log k$、終身 ≤ $H(M_V)$）。
  兩問是同一組命題的兩半（ルナ候選圖成立；本檔把它從圖升為 L1–L4）。
  守恆句：**GRPO 不創造 bits、只把驗證器的地圖搬進權重** — rung 0 量無 headroom
  ⇒ 幫浦無輸入（設計卡 §5.2 濾鏡句的帳本身份）。

---

## 6. 分級總表與誠實邊界

| 條目 | 級別 | 備註 |
|---|---|---|
| Prop L0 帳本恆等式 | 定理級（恆等式） | 交叉熵分解；NLL 貨幣 |
| L1(a)(b) 純自生聯合零 | 定理級（引 CT-1＋一行歸納） | 量詞：對任何 $W$、任何 $N$、任何拓撲 |
| L2(i)(ii)(iii)(iv) 選擇界 | 定理級（證在檔） | (i)(iv) 上界非可達性；(v) 傾斜讀法【訓練記憶】 |
| L3 閉環界＋取等 | 定理級（證在檔） | 取等條件＝觀測確定性（我們的域滿足） |
| L4(i)(ii)(iii) 幫浦界 | 定理級（證在檔） | 計的是 world-bits、非總行為改變 |
| Conj L4.4 有效注入 | **Conj** | 匯率斷裂 C-ii′ 承接；分辨＝rung 0/2＋指紋 |
| Thm L5 合法四格 | 定理級框架＋各格分級承上游 | L5.1 增益半句猜測級；L5.4 依 EIG【訓練記憶】 |
| 容量對齊 3 vs 2.5 | 啟發式 | 上界 vs 實測熵的併排、非可達性證明 |
| 表內 bits 粗估 | 啟發式 | 全部上界方向、未扣結構冗餘 |

**誠實邊界**：
1. **嚴的**：L0（恆等式）、L1（恰 0）、L3 在確定性域（等式）。**只上界的**：L2 的 $\log N$
   （可達性要覆蓋＋排序對）、L4 的計數界（gross bits；有效流量未知＝Conj L4.4）。
2. **貨幣**：全帳本以 NLL/bits 計價；R0 是 $T\to0$ 泛函 — bits 進了帳、R0 動不動要過
   兩道匯率（bits→變異：C-ii′；$T{=}1\to T\to0$：Lemma 1/2、P2 解耦預測）。⛔ 別把
   「注入 ≤ x bits」讀成「R0 至多／至少動多少」。
3. **(iv) 的界很鬆**：$H(o_t\mid\cdot)$ 是通道容量、不是 task-relevant 量；task 相關的
   那部分（route 級 ~2.5 bits）遠小於容量 — 所以「幾次好查詢就夠」是設計含義、
   「多互動＝多學」不是。
4. **外部件全數不承重**：Beirami 2401.01879、Gao+ 2210.10760、Lindley 1956、
   Da Costa+ 2001.07203、Xu–Raginsky 1705.07809 皆【訓練記憶、未上網覆核】—
   只作定位；引進 paper 前補驗。承重內部件：CT-1／CT-2／CT-3、⑰（2.5 bits）、
   C-ii′、Lemma 1/2、⑤''⑬⑱'（沿上游各自分級）。
5. **接口**：本檔不動合成律、不動內化 gap 定義；只需一致於「(ii)@train 建通道、
   帳本為其進帳計價」。GRPO 卡的呈裁點不變；本檔為其 rung 0 添一個讀法
   （headroom＝幫浦輸入頻寬）、不加新工。

## 7. 引用清單

沿上游[驗]：2305.15408、2310.07923、2402.12875（CoT 三件）；2505.12514（Zhu+
superposition）；2605.08732（開環誤差）；2209.15189（context distillation）。
【訓練記憶、未上網覆核；承重前補驗】：2401.01879（Beirami+ BoN exactness）、
2210.10760（Gao+ overoptimization、$\mathrm{KL}_{\rm BoN}$ 公式用法）、1705.07809
（Xu–Raginsky）、2001.07203（Da Costa+ active inference）、Lindley 1956（EIG）。
教科書級：DPI、鏈式法則、$H$ 上界、$I=H-H(\cdot\mid W)$。
家內：NOTE-0906-context-taxonomy（CT-1/2/3、三源表、C1–C4、P-swap/P-resample）、
THEORY-0906-postA1（C-ii′）、THEORY-0905-composition-law（Lemma 1/2、(S1)）、
DESIGN-0906-grpo-thoughts（reward 形、rung 0、指紋）、FINDINGS ⑤''⑬⑰⑰'⑱'。

_帳本完。呈裁點：①部署期 BoN 臂（§4 列缺格 — rung 0 儀器順手就能量、要不要排）
②Conj L4.4 的分辨實驗掛在 GRPO rung 0/2 還是另立 ③【訓練記憶】五件的補驗排程。_
