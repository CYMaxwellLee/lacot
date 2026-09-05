# DESIGN — 多路線 (s,g) 資料／評測設計（A5 槽位、multiroute v0）

_設計使魔（Fable、主人授權；9/5 三並行之「多路線資料/評測」線）。上游（唯讀）：
THEORY-internalization-formal §1.4（Int）/§3.3（C-i~iii）、THEORY-upgrades §1.3（η≳pλ）、
FINDINGS-0905 ⑫⑭（病因）、DESIGN-general-internalization-claim、RELATED-WORK-0905、
PLAN-forward A5。標記：〔驗 web/ssh/code 9/5〕＝本次一手驗過；【推論】＝ルナ由已驗事實
推出、未在資料上量；【猜測】＝不確定。⛔ 本檔為設計、無任何已執行變更；全文待主人裁。_

---

## 0. TL;DR

**推薦：主戰場＝(a) OGBench 現成 `pointmaze-teleport-stitch-v0`**（官方環境＋官方地板數字
＋C-i 天然＋C-iii 部分內建於官方成功率），**第一步＝選項無關的 H/η 探針**（CPU、唯讀資料、
明天可跑；現行 large-stitch 當陰性對照、同時當儀器 gate）。確定性備援：(d) 官方生成器改窗
重生（理論線相容、CPU 級、不卡 expert）；(c) 合成拼接只在 (a)(d) 都出問題才上（保證有 H、
但縫合風險最大）；(b) 現資料 counterfactual 配對【推論】機械上被餓死（§2.3 論證）、
但其探針就是所有選項共用的儀器，照建。

一句話的理論帳：三個構造全是在抬 $H(R|s,g)$（C-i）與錨 SNR $\eta$；toy 給的門檻是
$\eta \gtrsim p\lambda$（p=0.3 時新資訊變異要到冗餘變異的三成級），探針把 $\hat H,\hat\eta,\hat\lambda$
先量出來、Int 錶才有站起來的前提（Prop 3.7：兩根支柱同拆）。

---

## 1. 要買什麼（形式目標、一段收攏）

病根（⑫⑭＋Prop 2.1）：現行 pointmaze-large-stitch 上 $a\approx A(s,g)$（A1 冗餘）⇒ 忽略
intent 是合法全域最優 ⇒ 鎖死（⑬ 實測 ε_rel=1.1%、⑦ Int≈0）。要拆它需三條件（§3.3）：

- **C-i（多路線）**：$H(R|s,g)\ge h>0$ 於正測度 (s,g)，$R=f(\tau)$＝路線變數；
- **C-ii（錨資訊性）**：$I(R;a|s,g)\ge\delta$ — 我們的 hindsight 錨＝軌跡 cell 序列的 32 點
  弧長重採樣〔驗 code·intent.py/intent_anchor.py〕，只要管線不毀掉跳點資訊，C-ii 結構上自帶；
- **C-iii（效用可分）**：eval 對不同 R 產生可讀差異 — 這格現在完全沒有（eval 只問終點）。

定量指引（THEORY-upgrades §1.3）：$\mathrm{Int}^*=\eta/(\eta+p\lambda)$ ⇒ 要買到內化需
$\eta\gtrsim p\lambda$；並帶 P4 caveat：$\eta$ 要是「route 承載」的變異（$B\ne0$），
抖動噪音不算數 — 所以 §3 的 $\hat\eta$ 操作化用「桶內按 R 分組的組間變異」而非 raw 殘差。

**病根的機械量化（本檔新增、【推論】待探針證實）**：官方 stitch 生成器把目標距離寫死在
`adj_steps = 4`（scripted oracle＋noise 0.2；pointmaze 的 actor 是 `ob[-2:]` 純腳本、
無 learned expert）〔驗 web·generate_locomaze.py〕。而 large 佈局（9×12）〔驗 web·maze.py〕
的迴圈周長：最小 8（右上、繞 (2,9) 壁塊的 {(1,8)…(3,8)} 環）、次小 10（左上）。
兩條 alternative 路各 ≈ 半周長 ⇒ 除了右上那個 8-環（兩條等長 4 步路恰好貼窗）之外，
**路線選擇塞不進 4-cell 的 hindsight 窗** — A1 不是 maze 的宿命，是「窗長 < 迴圈半周長」
的幾何後果。這同時給了 (b) 的精確預言（§2.3）與 (d) 的旋鈕（§2.5）。

---

## 2. 構造選項比較（≥3＋2）

### 2.1 概覽表

| 選項 | C-i | C-ii | C-iii | 工程檔位 | 外部可比性 |
|---|---|---|---|---|---|
| (a) teleport 現成環境 | ◎ 天然（傳送門分支） | ○ 自帶（跳點要處理） | ◎ 部分內建官方成功率＋側錶 | 資料下載＝可直接做；錨跳點＋route 圖語義＝小 patch | **◎ 最高**（官方 env/data/eval、地板見 2.2） |
| (b) 現資料 counterfactual 配對 | ✗【推論】餓死（§2.3）；探針裁決 | ○ 自帶 | △ 全靠側錶 | 探針＝可直接做；重加權＝小 patch | ◎ 高（資料不動或僅重加權、要聲明） |
| (c) 合成拼接（同起訖不同中繼） | ◎ 可控（自己造） | ○ 自帶 | △ 全靠側錶 | 拼接資料腳本＝小 patch 級；縫合品質工程另計 | ○ 中高（eval 官方、訓練資料增強要聲明；SCoTS 2506.00895 先例） |
| (d) 官方生成器改窗重生 | ○ 自然湧現（窗 ≥ 迴圈半周長；oracle tie-break 是變數） | ○ 自帶 | △ 全靠側錶 | CPU 級可直接做（無 expert 死結）；【猜測】oracle 行為要 500 ep 先驗 | △ 中（env/eval 官方、資料非官方旋鈕 ⇒ 另立設定名、⛔不冒充官方 large-stitch 格） |
| (e) 自製雙走廊 maze | ◎ | ○ | △ | **要新環境** | ✗ 零地板 | 

(e) 一行帶過：C-i 可控性不輸 (c)(d) 但外部零可比、成本最高 — ⛔ 不推薦、僅列完整性。

### 2.2 (a) OGBench teleport 現成環境【主推】

**機制**〔驗 web·2410.20092 HTML＋maze.py 原始碼〕：teleport maze＝9×12（與 large 同尺寸級
⇒ 管線超參大概率直搬）；黑洞 2 格 `(4,6),(5,1)`、白洞 3 格 `(1,7),(6,1),(6,10)`、
`teleport_radius=1`；踩進黑洞→ `np.random.randint` **均勻隨機**傳到三白洞之一、
**其一是死巷**。官方原文：「must learn to avoid the black holes, without being optimistically
biased by 'lucky' outcomes」。資料集 `pointmaze-teleport-stitch-v0` 現成（軌跡同樣 ≤4 cell）。

**地板（官方 Table 2＋TMD Table 1、% 成功率）**〔驗 web〕：

```
pointmaze-teleport-stitch-v0:  GCIVL 44±2 | HIQL 34±4 | GCBC 31±9 | TMD 29.3±2.2
                               GCIQL 25±3 | CMD 15.7±2.9 | QRL 9±5 | CRL 4±3
對照 pointmaze-large-stitch-v0: QRL 84±15 | GCIQL 31±2 | HIQL 13±6 | GCIVL 12±6 | GCBC 7±5
```

兩個判讀：①天花板低（44）＝這格對所有人都難、我們有真實上升空間；②**QRL 84→9 的崩塌**
＝quasimetric 法被隨機性擊殺的實證 — 對內化線是機會（我們不靠 quasimetric），
對路線一是張力警報（見 §6）。

**C-i**：多路線天然存在 — 同 (s,g) 的「走廊路」vs「傳送門路」；且黑洞在走廊**內**
（radius 1、cell 邊長 4）⇒ 路過就觸發，資料裡 teleport 事件不需要 oracle 刻意去踩
（noise 0.2 也會推進去）。事件率＝探針量【待探針：oracle 的 BFS 是否把傳送邊納入規劃、
資料裡 teleport 事件佔比】。**C-ii**：hindsight 錨含跳點（非相鄰 cell 對）⇒ 錨對 R 高資訊
by construction；但見 §6 錨跳點病理。**C-iii ⭐ 獨有賣點**：風險差內建 — 傳送路成功率
結構性受 1/3 死巷拖累 ⇒ **選對路線直接反映在官方成功率裡**、不用側錶就部分成立
（可操縱性那一面仍要側錶，§4）。

**成本**：資料下載（jasmine `/archive/cymaxwelllee/data/ogbench` 已在、無 teleport 檔
〔驗 ssh 9/5〕⇒ `ogbench.make_env_and_datasets` 下載即可、pointmaze 級很小）＝可直接做。
訓練管線接上＝小 patch 三件：①dataset 名 plumb ②錨跳點處理（分段重採樣，§6-a1）
③帶查 route 圖的傳送邊政策（v0 建議：**排除傳送邊**、map 錨只下安全走廊路 — 與「學會避開
黑洞」的最優行為一致；hindsight-訓練 vs map-推論的錨分佈差變成被量的量、不是被埋的雷）。

### 2.3 (b) 現資料 counterfactual hindsight 配對【探針必建、構造本身預測餓死】

**機制**：不是新資料、是發現＋重加權 — (s,g) 粗格桶內撈不同實走路線的軌跡；若桶內真有
$H(R|s,g)>0$，hindsight 錨各自已含自己的 R ⇒ 什麼都不用改、A1 自動不成立；介入只剩
①oversample 多路線桶（dataloader 小 patch）②從這些桶建 route-conditioned eval 對（§4）。

**餓死論證【推論】**：§1 的機械量化 ⇒ 4-cell 窗裝不下任何 ≥10 周長迴圈的路線選擇；
唯一例外＝右上 8-環（(1,8)↔(3,10) 型對、兩條 4 步等長路）— 且 oracle 是 scripted 最短路
追蹤器、tie-break 若決定性則連這格也塌成單路線（噪音 0.2＋連續起點是唯一變異來源）。
⇒ **精確可證偽預言：large-stitch 的路線質量 ρ_route < 1%、且殘餘全集中在右上 8-環桶。**
探針半天判生死；若意外撈到肉、(b) 是零成本首選 — 所以照建、不因預測負就跳過（一格不
下結論、先量）。陰性對照的價值另立：探針若在 large-stitch 的 junction 粒度也報高 H
＝儀器在把抖動當路線 ⇒ 儀器 bug、先修再信任何讀數。

### 2.4 (c) 合成拼接（同起訖不同中繼、SCoTS 式）

**機制**：large-stitch 現資料上，cell 對齊撈段 — τ₁ 給 s→m₁→g 用的 s→m₁ 段、τ₂ 給
m₁→g 段，縫成合成軌跡（狀態黏合：位置貼 cell 中心、速度縫隙平滑或直接以低速段接）；
同 (s,g) 造不同中繼 m₁/m₂ ⇒ H 可控、$\eta$ 是自己的旋鈕（造幾條、路線差多開）。
理論同盟：(S1) 拼接引理（upgrades §2）明說 hindsight 切段自動子段封閉、確定性環境下
拼接合法 — (c) 是它的資料面實作；SCoTS（GCIQL 21→79）當外部先例背書。

**代價**：縫點不是真 dynamics ⇒ 凍結 stage1 encoder 吃到 off-support 輸入、flow 學到
「穿縫」；資料增強要在 paper 聲明。定位＝**保證有 H 的最後備援**（(a) 有隨機性風險、
(d) 有 oracle 行為風險，(c) 的 H 不依賴任何環境行為）。

### 2.5 (d) 官方生成器改窗重生（bonus、確定性伴星）

**機制**：官方 `generate_locomaze.py` 只改一個旋鈕 `adj_steps: 4→8~10` 重生 large 資料
（pointmaze 的 actor 是 scripted `ob[-2:]`、**無 learned expert 依賴** — 不踩 A0 的
experts.tar.gz 死結〔驗 web·原始碼〕；CPU 級）。窗 ≥ 迴圈半周長後，8-環與 10-環的
兩條路都進窗 ⇒ H 自然湧現（真 rollout、零縫合、確定性環境 ⇒ 理論線全相容）。

**風險與檔位**：oracle tie-break 若決定性、H 仍可能塌（同 (b) 的病）⇒ 先花 1 小時 CPU
生 500 ep 跑探針再談 GPU【猜測：oracle 用 env 內建 BFS 方向、tie-break 行為未讀原碼】。
可比性：eval 任務照官方、但訓練資料的 stitch 難度變了 ⇒ **另立設定名**（如
large-multiroute@w8）、⛔ 不填官方 large-stitch 的 T1 格。

---

## 3. 推薦方案＋最小驗證階梯

**主推 (a) teleport**，理由三條：①唯一「官方 env＋官方資料＋官方 eval＋現成地板」四全的
選項 — T1 可直接多一欄、外部敘事零折扣；②C-iii 部分內建於官方成功率（其他選項全靠側錶）；
③理論文件點名的正是它（§3.3 Remark 3.8、⑫⭐、⑭ 根治條）。(d) 並行當確定性伴星
（CPU 便宜、餵理論線的確定性假設）；(c) 押後備援；(b) 的探針先行、所有選項共用。

### 3.1 階梯

```
R0（明天、CPU、唯讀資料、可直接做）
    H/η 探針跑現行 large-stitch ⇒ 儀器 gate＋陰性對照＋(b) 生死判
R1（明天~後天、可直接做＋下載）
    teleport-stitch 下載 → 同探針 ⇒ gate G1/G2/G3（下方）全過才進 R2
    並行（CPU）：(d) 500 ep @ adj_steps=8 樣本 → 同探針（oracle tie-break 判）
R2（單顆先導、~0.5 GPU-h：24 分訓＋eval）
    teleport 上 base/ref/idp 三臂裡先跑 ref(p=0)＋idp(p=0.3) 各 1 顆
R3（放量 ×8、~6 GPU-h、PLAN A5 預算內）
    勝出構造三臂 ×8＋route-conditioned eval（§4）＋Int 正式讀數（§5）
```

### 3.2 探針操作型定義（可跑級、全部用現有零件）

**R＝粗格路線簽名（雙粒度、都要報）**：
- 抽取：`traj_to_cells()`（現成、相鄰去重〔驗 code〕）→ 抖動壓平（A→B→A 三連縮成 A；
  `jitter_rate` 現成當診斷）→
  - **R_fine**＝壓平後 cell 序列的 tuple-hash；
  - **R_coarse**＝只留「決策 cell」（自由格圖上度數 ≥3 的格）的訪問序列；teleport 資料
    另加一維事件標記：R_tp ∈ {無傳送, 經 BH₁, 經 BH₂}（跳點偵測＝連續 cell 非相鄰）。
- ⛔ 單一粒度的 H 讀數不下結論 — 粒度是自由參數，兩檔一起報、差一個量級就先查儀器。

**H(R|s,g) 估**：桶＝(cell(s), cell(g))；桶內樣本 n≥10 才進表。報三個數：
①質量加權 $\hat H$（plug-in＋Miller–Madow 修正 $(\hat K-1)/2n$）；
②**ρ_route**＝「桶內 ≥2 個相異簽名、且次多簽名佔比 ≥10%」的桶所含 hindsight-pair 質量比
（主判準 — 對小樣本熵偏差穩健）；③per-桶分佈圖＋熱點定位（large 預言：右上 8-環）。

**η̂/λ̂ 估（錨 SNR、對齊 Cor LT-2）**：對 32×2 錨（與 ix 嵌入各做一次、對照 ⑩ 的 anc-R²
儀器）做三層變異分解：λ̂＝桶均值解釋的變異（(s,g)-可預測＝冗餘）、
η̂_route＝桶內**按 R 分組的組間變異**（route 承載 — P4 caveat 的正解，抖動不計入）、
σ̂_jit＝組內殘差。逐主成分報譜、主讀數＝top 成分的 η̂_route/λ̂。

**Gates（R1 過關判準；門檻【猜測】、主人調）**：
- **G1（C-i）**：teleport 資料 ρ_route ≥ 5%（且 large-stitch 對照 < 1% — 反之儀器疑）；
- **G2（C-ii）**：桶內 linear probe 從錨判 R 的 acc ≥ 90%（錨本來就是路線 polyline、
  應近 100%；掉下來＝重採樣把跳點抹掉了 → 先修 §6-a1 再前進）；
- **G3（η 門檻）**：top 成分 η̂_route/λ̂ ≥ 0.3（=目標 p；不足 ⇒ 選項內加碼 — teleport
  可加 (b) 式多路線桶 oversample — 或降 p 併入 T4 劑量軸）。

### 3.3 單顆先導判準與放量

單顆（R2）看三格、全是現成儀器：①⑬ 散度探針 ε_rel **脫離 1% 級**（≥10×、即 >11%
【猜測門檻】）— 這是 Prop 3.7「支柱拆除」的機制層直讀；②route-consistency 側錶（§4）
方向正確（跟指定 R 的匹配率 > 換路 chance）；③官方成功率 sanity（不塌出家族分佈）。
三格齊 ⇒ R3 放量；①亡 ⇒ 先回 G3 查 η̂ 是否真達標、再談加藥（L_div／退火 — A3 的臂庫）；
①活②亡 ⇒ 通路活但沒學會跟路 — 訓練時數/容量線、按 (0,+) 格處置（§5）。

---

## 4. eval 側 C-iii 設計（官方口徑不動、側錶另立）

**鐵則：官方成功率的定義、任務集、episode 協定一字不動** — T1/T2 的外部可比性是本設計
的最大資產、⛔ 不拿去換靈敏度。C-iii 的可讀差異加在**側錶**：

- **側錶 1（per-route 成功率）**：多路線 (s,g) eval 對上、分別以 R₁ 錨/R₂ 錨條件、各報
  官方口徑成功率。route 錨構造＝waypoint 約束最短路：`route_cells(s,m)+route_cells(m,g)`
  （現成 `grid_shortest_path` 拼兩段 — **正是合成律 (s,m,g) 機器**、C 線字典 DP 未來直接
  複用同一介面）＝小 patch。
- **側錶 2（route-consistency, RC）**：rollout → `traj_to_cells` → R_coarse 簽名、與指定
  R 匹配（決策 cell 序列相等；teleport 另比 R_tp 標記）。主報 P(跟對路 | 成功)＋
  **R₁/R₂ 混淆矩陣**。
- **側錶 3（counterfactual steerability）**：同 (s,g) 同 eval seed、錨 R₁→R₂ 對調、量執行
  簽名翻轉率 — ⑬ 的 d_swap 抬到行為層、C-iii 最直接的一格。
- **eval-set 構造**：從探針熱點桶抽多路線 (s,g) 對（teleport：白洞鄰域桶＋黑洞走廊桶
  必含）；每對兩錨皆下側錶 1-3。route-holdout（訓見 R₁、eval 指定 R₂）＝組合泛化加分題、
  ⛔ 不混入主錶。
- **teleport 的兩面注意**：效用差那面官方口徑自己會讀（風險差）；但「都到得了 g」的對上
  官方成功率對 R 鈍 ⇒ 可操縱性那面只有側錶讀得到。兩面都綁回 C-iii 的形式敘述、
  防 reviewer 說側錶 ad-hoc。

---

## 5. 與 Int 錶的整合（判讀樹先釘）

**分母（$U_a(\theta_{\text{ref}})-U(\theta_{\text{base}})$）預期怎麼變**【猜測、方向論證】：
ref（p=0 全曝光）拿到真資訊錨 ⇒ 增益應 ≥ 搖籃的 +.133（teleport 上 map 錨還多送「避開
黑洞」的行為資訊 — 對照地板 GCBC 31 vs GCIVL 44 的差距＝幸運偏差的價碼）；base 不變或
略掉（多路線讓免錨的 (s,g)→plan 更難）⇒ **分母變大、Int 從 undefined 域進可讀域** —
這就是整個構造的目的（Def 1.4 分母條款的正向面）。同時 teleport 的隨機性會吹大 R0 的
SE ⇒ κ·SE 門檻同步變嚴 — A4 的 ep≥200 硬化在這裡是前置不是加分、且 teleport-事件
分層報（見 §6-a2）。

**雙 Int（新增、先釘再收數）**：
- $\mathrm{Int}^{\text{succ}}$＝Def 1.4 原式（U＝官方成功率）— 外部可比、進 T1；
- $\mathrm{Int}^{\text{route}}$＝同式、U 換側錶 2 的 RC — C-iii 對齊的靈敏錶；
  多路線下可能 $\mathrm{Int}^{\text{route}}>0$ 而 $\mathrm{Int}^{\text{succ}}\approx0$
  （兩條路都到得了）⇒ 兩錶並報、⛔ 不拿靈敏錶冒充官方錶。

**判讀樹（收數前釘死）**：
1. 分母 < κ·SE？→ C-iii 沒立住：先看 $\mathrm{Int}^{\text{route}}$ 的分母 — 也死 ⇒ 構造
   失敗、回 G1/G3 查 H/η；只有 succ 版死 ⇒ 效用面鈍、側錶版照讀＋記錄口徑差。
2. 分母活、落 (Int≈0, ε≈0) ⇒ 仍鎖死 ⇒ 對照 η̂ vs pλ̂：η̂<pλ̂＝toy 預期內（構造加碼或
   降 p）；η̂≥pλ̂ 仍鎖 ⇒ **toy 反例、升級 Conj 2.6 的有限容量項** — 兩枝都有資訊量。
3. (Int≈0, ε>0) ⇒ 通路活未內化 ⇒ 時數/容量線（曝光匹配臂的老路）。
4. Int 中間值 ⇒ 部分內化：對 $\mathrm{Int}\approx\eta/(\eta+p\lambda)$ 做定量比對 —
   **LT toy 的第一次真環境檢驗**、直接進 T4。
5. ⚠️ LT §1.7 預警照搬：強內化 regime 的 $U_\varnothing$ 有系統性下偏（共享通路被 keep 支
   拖）⇒ zero-模式略降 ⛔ 不讀成 regression、降幅應隨 p 增而縮。

---

## 6. 風險（每方案一條「會怎麼爛」＋偵測法）

- **(a1) 錨跳點病理**：傳送跳點被 32 點弧長重採樣攤成一條穿牆直線 ⇒ intent MLP 吃到
  off-support 幾何、C-ii 名存實亡。偵測：錨相鄰點距 max-gap 探針（> maze_unit×√2 即異常）
  ＋G2 linear probe。修法（小 patch）：分段重採樣（跳點兩側各自弧長分配、點數按段長比例）。
- **(a2) 隨機傳送毀開環一致性**：plan 無法承諾傳送後綴 ⇒ RC 側錶在傳送路上讀成噪音、
  Int 的 SE 被吹大。偵測：所有 eval 讀數按「是否發生 teleport 事件」分層報；傳送路的 RC
  只比到黑洞入口為止（承諾得了的前綴）。
- **(a3) 理論邊界**：隨機轉移破 (S1)/Lemma 2.3 的確定性前提（合成律線）＋QRL 84→9 警示
  路線一的 quasimetric 幾何在 teleport 上病態。處置：**teleport 只掛內化線**；合成律與
  路線一的陳述限定在無傳送子圖／large-stitch（確定性）— 這正是 (d) 伴星存在的理由。
  寫進 E 線交接包給 Rei。
- **(b) 餓死或假多路線**：預言 ρ_route<1%（餓死）；或 fine 粒度把抖動報成路線（假陽性）。
  偵測：探針本身＋雙粒度交叉＋shuffle 對照（打亂 R 標籤後組間變異應歸零 — pipeline
  虛報偵測、⑩ 的 shuffle 欄同款）。
- **(c) 縫點教穿牆**：合成縫隙是非法 dynamics ⇒ decoder 學會穿縫、合法性探針（⑤' 儀器）
  退步。偵測：縫點局部合法性檢查＋凍結 encoder 在合成軌跡上的 recon 誤差百分位（對真軌跡
  分佈）— 超 P95 的縫棄用。
- **(d) oracle tie-break 塌縮**：決定性 tie-break ⇒ 窗加長了 H 還是 0。偵測：500 ep CPU
  樣本先過探針、⛔ 不先燒 GPU。
- **共通①（簽名粒度不穩）**：H 讀數隨粒度晃一個量級 ⇒ 永遠雙粒度並報、gate 用 ρ_route
  不用裸熵。**共通②（可比性稅）**：凡動了訓練資料（(b) 重加權/(c)/(d)）的格、T1 内
  ⛔ 不與官方資料格混排、腳註聲明 — 只有 (a) 免稅。

---

## 7. 分級總表與排程對接

```
可直接做（唯讀/CPU）   H/η 探針（large-stitch＋teleport＋(d) 樣本）；teleport 資料下載；
                       (d) 500 ep 重生樣本
要小 patch             (a) dataset plumb＋錨分段重採樣＋route 圖傳送邊政策；
                       側錶 1-3（route eval）；(b) 桶 oversample；(c) 拼接資料腳本
要新環境               只有 (e)、不推薦
```

- **PLAN-forward 對接**：本檔＝A5 的「eval-set 設計」交付（D2 槽）；R1 探針可提前到 D1 晚
  CPU 跑；R2/R3 進 A5 的 pilot+×8 預算（≈6 GPU-h、P0 鏈之後）。T1 提案新增
  pointmaze-teleport-stitch 欄（官方地板 44/34/31 現成）— 併入 D1 晚的呈裁清單。
- **要主人裁的**：①主推 (a)＋(d) 伴星的雙軌是否核准 ②G1/G3/ε 門檻數值（全標【猜測】）
  ③T1 加欄 ④(d) 的設定命名與揭露格式。

_自檢：C-i/C-ii/C-iii 三條件每個選項都對過表；η≳pλ 的量法落到可跑（G3）；官方口徑
一字未動；每個風險都有偵測器不是祈禱。探針先行 — 一格不下結論、先定位再開藥。_
