# THEORY — Verifier 教的、weights 存的、u 提的：latent thinking 一條故事線（v0、2026-09-06）

_統一敘事使魔（Fable 級、主人授權的最後一隻）。本檔【只織不添】：把今晚各線織成一份
連貫敘事 — 每步引來源檔節號、不重證明、不新增理論主張；分級標記照抄來源。
來源縮寫：【帳本】=THEORY-0906-information-ledger、【分類】=NOTE-0906-context-taxonomy、
【postA1】=THEORY-0906-postA1-revision、【合成律】=THEORY-0905-composition-law-draft、
【內化】=THEORY-0905-internalization-formal、【設計卡】=DESIGN-0906-grpo-thoughts、
【F⑫…】=FINDINGS-0905 各節。衝突時一律以來源檔為準。_

---

## 0. 主人的定調（2026-09-05 23:39 原話）

> 「訓練的過程中 Verifier其實是在教model東西，然後存在weights裡面，要inference的時候，
> 則是從weights裡面提取肌肉經驗成u，然後拿來生答案」

**白話翻譯（比喻歸位）**：這句話把今晚三個比喻一次擺對位置 —
- **油箱＝verifier**（外部驗證資源 $M_V$：佔據圖＋BFS 場）。world 的 bits 只能從外面來，
  verifier 是四個進帳通道之一、也是我們最便宜的一個【帳本 §1 Def L1 (iii)】。
- **幫浦＝GRPO／訓練**。「Verifier 在教 model、存進 weights」＝幫浦把油箱的 bits 打進
  $\theta$：每 update 有帳、終身有帽【帳本 §2.4 Prop L4】。教的不是姿勢，是可計量的資訊。
- **油門＝推論期的自生 u**。「從 weights 提取肌肉經驗成 u、拿來生答案」＝u 自己不產油
  （恆 0 bits【分類 §2.1 Thm CT-1】）、踩多深都只是消耗 $\theta$ 裡已存的油 — 它的合法
  工作是提取與導向【帳本 §3 Thm L5】。「肌肉經驗」＝已攤銷進權重的存量，正是 L5 第 2 格
  「提取」的字面義。

一句收攏：主人句＝遷移鏈的 verifier 版 — **資訊建通道（verifier 教）→ 權重存（weights 記）
→ 計算駛通道（u 提、生答案）**【分類 §5.1；帳本 §4 表「oracle 錨→內化」行：驗證器與
環境續帳、RL 入帳】。

---

## 1. 一條故事線（八步、每步引節號、不重證）

**(1) 起點：latent thinking 自己是零資訊的。** 自生 $u=f_\theta(s,g,\varepsilon)$ 對任何
world 變數恆注入 0 bits — 不是「少」、是恰等於零〔定理級、一行 DPI〕【分類 §2.1 Thm CT-1】；
且「想更久、多想幾條、任意拓撲」聯合仍 0〔定理級〕【帳本 §2.1 Prop L1(b)】。計算規模不是
資訊變數；訓練側對偶：自生配對建不了通道（$\eta_{\rm eff}\equiv0$ 精確成立）〔定理級@LT〕
【分類 §2.1 Cor CT-3】。

**(2) 那 u 的合法工作是什麼。** 帳本恆等式把 NLL 拆成「資訊地板＋提取赤字」、只有 bits
動得了地板〔定理級〕【帳本 §0 Prop L0】⇒ u 的全部價值住赤字欄（計算），具體四格
【帳本 §3 Thm L5】：**分解**（實體化合成律的 DP 中繼；增益半句仍猜測級）、**提取**
（serial depth、CoT 三件的貨幣，成立要件 C1–C3【分類 §3】）、**前沿疊加**（連續 u 平行
BFS — 在自家假設空間搜索 θ 已有的東西）、**查詢規劃**（算 EIG argmax、BoN proposal、
GRPO 行為分佈 — u 不能供給 bits，但決定 bits 從哪裡、以多大效率進來）。

**(3) 資訊從哪來：四通道與各自的界。**【帳本 §1–2】(i) **權重**：訓練期入帳、
$I(W;\theta)\le I(W;\mathcal D)$；(ii) **conditioning**：NLL 可兌現額恰等於注入的 $I$
〔定理級〕；route-ix 實測 ~2.5 bits/題【F⑰】；(iii) **驗證器／選擇**：每次選擇
$\le\log N$、且終身擠不出驗證器自己沒有的（$\le I(W;M_V)$）；自驗證＝0 新 bits
〔定理級〕【帳本 §2.2 Prop L2】；(iv) **觀測**：每步 $\le H(o_t\mid\text{belief})$、我們的
確定性世界逐步取等（碰撞位元＝1 bit/步）〔定理級〕【帳本 §2.3 Prop L3】。
一句話【帳本 §5 原句】：**探索是油門，驗證器才是油箱。**

**(4) 我們的病，在帳本語言裡＝通道活著、匯率被壓。** 實測兩頭：⑬ 鎖死實錘 — idp 的
intent 分支散度只剩 f27n 的 1.1%、塌在 cond 生成端（d_sg 兩顆都活 ⇒ 不是整體 cond 盲）
【F⑬】；⑰ 又量到 $I(\tau;a|s,g)\approx2.5$ bits $\ne0$（A1 證偽；引用紀律：路線多樣性
敘事只能講走廊級 ~1 bit【F⑰'】）。⇒ 錢是真的、通道物理也在，倒的是**匯率**：bits 以
NLL 計價、CFM 以變異計價（C-ii′ 匯率斷裂）【postA1 §1、§3.1】；route 錢經 e_target 度量
重計價後變異佔比只剩 ~0.4% ⇒ $\eta_{\rm eff}\approx0.003\lambda\ll p\lambda$ ⇒
$\mathrm{Int}^*(0.3)\approx1\%$ — 與 ⑬ 的 1.1% 對上〔量級論證、兩獨立 z-估計吻合 10%〕
【postA1 §3.2】。動力學面：系統不是被困在非法駐點，而是**以慢速率 $\kappa$ 合法收斂到
被 ridge 壓扁的小目標**（$12\times$ 時間尺度分離＋overshoot 賽跑）〔Prop（LT 內）＋
Remark 升格〕【postA1 §2.2 Prop 2.3′、Remark 2.4′】—「動力學陷阱」。⚠️ ⑱' 補：κT
校準句（「債已還完」）已標疑，活路＝重校準假說（8000 步或在 overshoot 峰前、l_nf 未收斂
佐證），判別＝中途 ckpt 散度曲線【F⑱'】。

**(5) 三藥在帳本的位置。**（藥方形【內化 §3】；權重按 ⑰ 移：退火＋L_div ↑、破冗餘 ↓
【F⑰；postA1 §2.3】）— **退火 $p$＝動力學藥**：改走的路徑與時間尺度（屏障擋「先塌後修」
〔Prop、限定範圍〕；⚠️ 屏障門檻 $p\lesssim1/4$【postA1 §5 B10】；長時黏性未保證
〔Conj 3.2〕→ WS 實驗直接判）【內化 §3.1】。**$L_{\rm div}$＝通道保活**：把 invariant 解
逐出最優集、保 $\varepsilon_{\rm cond}\ge m$＝「可喚醒性」，不保證被用（結構藥 vs 內容藥）
〔Prop 3.4／R3.5〕【內化 §3.2】。**破冗餘＝抬匯率**：⑰ 後它的角色從「造 $I$」轉向
C-iii 可讀性與 z-度量權重（C-ii′ 後半）— 打的是 $\eta_{\rm eff}$ 那頭、不是 bits 那頭
【內化 §3.3；F⑰】。三藥打三層、彼此不冗餘【內化 R3.5；F⑭ 雙保險】。

**(6) verifier＋GRPO＝把地圖 bits 打進權重的幫浦。** reward $=V(z;s,g,M_V)$ 把通道 (iii)
的 bits 經 advantage 入 $\theta$：每 update $\le B_gG\log k$、終身 $\le H(M_V)$（把地圖
搬進權重的上限）〔(i)–(iii) 定理級〕；退化群 $\hat A\equiv0$ 貢獻恰 0 ⇒ **退化群比例錶＝
幫浦輸入頻寬錶（字面義）**【帳本 §2.4 Prop L4】。幫浦不創造、只搬運＋降溫（world-bits
vs 模式銳化兩份、指紋可分）【帳本 L4 Remark；設計卡 §3.3】。GRPO 買的東西一句話：
「把 pass@G 裡已存在但低機率的成功搬進 pass@1」— 獎品大小訓練前就可量【設計卡 §3.4】；
「rung 0 無 headroom ⇒ 幫浦無輸入」在帳本裡是定理不是比喻【帳本 §4 表、§5】。
有效流量另有匯率（gross bits ≫ 實際學習量）〔Conj L4.4〕【帳本 §2.4】。前提：base 收斂
（⑱' 硬前提）＋step-matched FM 對照鐵則【設計卡 §3.1】。這是「reward 通道也能內化」
的第二攤銷通道【設計卡 ⓪】— 主人句前半的正式身份。

**(7) 溫度族給「存在哪、消費在哪」的正式座標。** 訓練目標與 NLL 住 log-semiring
（$T{=}1$）、argmax／成功率是 $T\to0$ 泛函、BFS＝凍結極限（偏差 $\le H\,T\log K$、
$H$＝horizon〔⑮ 釘死〕）〔Lemma 1/2〕【合成律 §1.2–1.3】；refine/BoN＝把有效溫度往 0 壓
〔R2 定性〕。⇒ **內化存在 $T{=}1$、消費在 $T\to0$**；bits 進了帳、R0 動不動要過兩道匯率
（bits→變異 C-ii′；$T{=}1\to T\to0$）【postA1 §6 誠實邊界 2；帳本 §6】⇒ NLL-gap 與
R0-gap 可解耦、R0-gap 先閉〔類比級→可測〕【分類 §5.2 P2】。

**(8) 搬了多少＝內化率 Int。** 三點校準（從未曝光 base／全曝光 ref／被測 $\theta$）拆開
「完美內化／鎖死／無物可內化」三重簡併，配診斷對 $(\mathrm{Int},\varepsilon)$
〔Definition〕【內化 §1.4 Def 1.4】；toy 閉式 $\mathrm{Int}^*=\eta/(\eta+p\lambda)$
（⑮ 機器精度驗訖）給出它的理論值與劑量曲線【postA1 §2.2 Prop 2.2′、§4】。⑱' 紀律：
ref／分母改用長訓家族（f27nL／N5L）、一切增益只准對 step-matched 對照報；on/zero
齊漲 74%＝generic-T 混淆的實測指紋，收表先過指紋檢【F⑱'；設計卡 §3.3】。

---

## 2. Claim 敘事段（paper intro 素材；整合遷移鏈＋內化度量軸）

> During training the verifier is a teacher, and what it teaches is countable: each act of
> external selection injects at most $\log N$ bits of world information, and each GRPO
> update pumps at most $B_g\,G\log k$ bits into the weights — never more, over a lifetime,
> than the verifier's own information stock (one-line data-processing arguments). At
> inference the deployed model conditions on nothing, and any self-generated latent
> thought provably carries zero further information about the world — its entire residual
> value is *computational*: it extracts the knowledge already stored in the weights (the
> "muscle memory") into an explicit plan, the same serial-depth currency that
> CoT-expressivity theory prices. Latent thinking is thus a transfer along the
> context-source chain — *information builds the channel, weights store it, computation
> drives it* — and the transfer is measurable: an internalization rate Int, calibrated
> between a never-exposed baseline and a fully-exposed reference, reports how much of the
> teacher's information actually moved into the weights, while a temperature-family
> composition law fixes the two currencies involved — storage is priced at $T{=}1$ in
> log-likelihood, consumption happens at $T\to0$ in success rate — predicting that the
> two corresponding gaps need not close together.

> 訓練時 verifier 是老師，而它教的東西可以記帳：每一次外部選擇至多注入 $\log N$ bits 的
> world 資訊、每個 GRPO update 至多把 $B_gG\log k$ bits 打進權重 — 終身總量不超過
> verifier 自己的資訊存量（各為一行 DPI 論證）。推論時部署模型不 condition 任何外部輸入，
> 任何自生的 latent thought 可證明地不再攜帶任何 world 資訊 — 它的全部剩餘價值是
> **計算**：把已存在權重裡的知識（「肌肉經驗」）提取成顯式計畫，用的正是 CoT 表達力理論
> 定價的那種 serial-depth 貨幣。latent thinking 因此是沿 context 來源鏈的一次遷移 —
> **資訊建通道、權重存、計算駛** — 且遷移可量：內化率 Int 以「從未曝光的基線」與「全曝光
> 的參考」兩端校準，報告老師的資訊實際搬進權重多少；溫度族合成律則釘死兩種貨幣的座標 —
> 儲存以 $T{=}1$ 的 log-likelihood 計價、消費發生在 $T\to0$ 的成功率 — 並預測兩個對應的
> gap 不必同時閉合。

（英中兩段每一句都有來源：選擇界【帳本 L2】、幫浦界【帳本 L4】、零資訊【分類 CT-1】、
serial depth【分類 §3】、遷移鏈【分類 §5.3】、Int【內化 Def 1.4】、溫度座標【合成律
Lemma 1/2；分類 P2】。⛔ 無新增主張。）

---

## 3. 可證偽預測清單（各檔收攏、去重、標出處與分級）

1. **劑量四點平坦**：除 $p{=}0$ 外，$p\in\{.05,.1,.3\}$ 的效用增益全在噪音級
   （$\mathrm{Int}(.1)\approx3\%$、$\Delta R0\approx.004$）；idp01 R0on 回 .42+ ＝平衡版
   當場證偽〔預測、平衡假設＋區間列明〕【postA1 §4.1】。初讀：⑱ idp01 增益沒回（.335）、
   方向符合；正式判等 ep200 錨＋新分母【F⑱】。
2. **劑量比值封頂**：$\mathrm{Int}(.1)/\mathrm{Int}(.3)\le3$、$\mathrm{Int}(.05)/\mathrm{Int}(.3)\le6$，
   與 $\eta$ 取值無關；量到 $\mathrm{Int}(.1)>3.3\%$ 級即平衡假設倒、直接進 transient 枝
   〔Prop 4.2′、一行證〕【postA1 §4.2】。
3. **劑量形狀三性質＋單點分版**：$\mathrm{Int}(p)$ 單調降、凸（斜率遞增）兩版共有；
   分版點＝$\mathrm{Int}(.05)$：$<0.1$ 平衡版活、$[0.2,0.5]$ transient 版活、$>0.5$
   $\eta_{\rm eff}$ 整組重估〔預測／判別器；transient 式 Conj 級〕【postA1 §4.3–4.4】。
   ⑱' 補判別：中途 ckpt 散度曲線（重校準假說 vs 平衡）【F⑱'】。
4. **NLL-gap 與 R0-gap 解耦、R0-gap 先閉**（idp 儀器現成、兩 gap 分開畫）
   〔類比級→可測〕【分類 §5.2 P2；座標＝合成律 Lemma 1/2】。
5. **G=8 ↔ 2.5 bits 容量對齊**：每群選擇預算 $\log_2 8=3$ bits ≥ ⑰ 實測 route 資訊
   ~2.5 bits —「一群頻寬剛好夠指定一條 route」；reward 訊號不夠多樣 → 加 $G$＝加頻寬
   〔啟發式；配套頻寬錶＝退化群比例、L4(ii) 定理級〕【帳本 §2.4 Remark 容量對齊】。
6. **rung 0 headroom 判準**：stitch 題集上 pass@G − pass@1 gap 與 reward headroom 至少
   一個顯著 $>0$，否則 GRPO 無物可放大、整臂降級（幫浦無輸入＝帳本內定理；門檻【猜測】）
   【設計卡 §3.4、§4 rung 0；帳本 §4–5】。
7. **WS（warm-start）雙判**：黏住 ⇔ 塌回（判退火藥是否單獨可行、Conj 3.2）【內化 §3.1；
   F⑭】；判讀尺＝slaved 域斜率比 $\mu_-=2.759\pm20\%$ 帶、⛔ 不用 2.3 定值
   〔⑮ 收緊①、toy 精度驗訖〕【F⑮】。
8. **shuffled／自生 u 訓練臂完全鎖死**：$\eta_{\rm eff}\equiv0$ ⇒ 配對差 $\equiv0$、
   d_zero→0（A1-lim 命題對此臂逐字為真＝廢物利用成控制臂）〔定理級@LT〕
   【分類 §5.2 P1＝Cor CT-3】。
9. **one-shot 連續自生 u 增益 ≈0**（可被吸收成 flow 的噪音維）⇒ rung 3 要嘛迭代、
   要嘛離散字典搜索，⛔ 別蓋 one-shot 連續版〔命題級設計指引〕【分類 §5.2 P3、§3.2 C1】。
10. **自生 top-up 必 swap-insensitive**（對 data-pairing 讀 0；oracle 臂 $>0$）— 「資訊
    vs 計算」的操作型分辨器、⑬ 儀器現成〔定理級簽名〕【分類 §5.2 P4、§4.3 P-swap】。
11. **GRPO 內化指紋（預註冊）**：RL-經-prior ⇒ zero 漲多於 on、gap 收窄；on/zero 齊漲
    ⇒ generic-T 混淆未對照乾淨（⑱' 實測齊漲 74% 為其錨）【設計卡 §3.3；F⑱'】。

---

## 4. 明早行動接口：這份敘事等什麼數字餵

| 收表組 | 判什麼 | 餵本檔哪節 |
|---|---|---|
| idpxm ×8（p=.3 曝光匹配 11429、s40–47） | 劑量/曝光主判、Int 正式讀數 | §1(4) 動力學陷阱、§3 預測 1–3、§1(8) |
| f27n@11429 s40（step-matched 對照） | 拆「曝光 vs 總步數」（⑱' caution①） | §1(4)；§3 預測 11 的 generic-T 基線 |
| f27nL（p=0 長訓） | base 收斂判準（l_nf 尾段斜率）＋ Int ref | §1(6) 幫浦前提；§1(8) Def 1.4 分母 |
| N5L s40（無 intent 基線@11429） | Int 新分母 $\theta_{\rm base}$ | §1(8)；§3 預測 1 的正式化前提 |
| idp01L s40（p=0.1@11429） | 劑量點（T 匹配版） | §3 預測 1–2（$\mathrm{Int}(.1)$ 封頂檢） |
| WS 產物（day_0906/、收表掃描要含【F⑰'】） | 退火藥黏性＋斜率帶 | §1(5) 退火藥；§3 預測 7 |

另兩支免費探針（不佔 GPU、待主人點頭）：rung 0 headroom（§3 預測 6 的輸入；設計卡
rung 0 免訓練）；$I(z;a|s,g)$ 直接量測（⑰ 儀器換 $z$ 輸入 — postA1 Prop 2.1′ 前提補完
＋Conj 2.5′ 分辨）。缺口誠實列：$\mathrm{Int}(.05)$ 分版點（§3 預測 3）目前無臂、屬待裁。

_織完。本檔無任何新理論主張；所有分級（定理級／Prop／Conj／啟發式／類比級→可測／
【猜測】）照抄來源檔，衝突時以來源檔為準。_
