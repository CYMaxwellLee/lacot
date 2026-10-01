# 突破第 1 步雙探針發射單 v4（F5 預檢靶）— Move 3 途經點三世界 ＋ Move 1 steerability gate

_lead=ルナ 2026-10-01。v1 NO-GO 31→v2 修→複審 8 must-fix→v3 修→三審 6 洞→v4 修
（exact-zero＋self-rollout-u）→四審 NO-GO 5 殘（全為契約凍結級）→本 v5 逐項修
〔修④-N 標四審五項〕。判決重心＝Move 1（u 層）。⛔ v5 過①五審 → 呈主人 → 點頭才發。_

## 共同底座

- **題集**〔gate0b 親算、①複審重算相合〕：
  - 主判集＝20 題多路雙掛（task 4 十四題 D=14＋task 5 六題 D=15；各恰 2 條等長路；
    幾何僅 2 組 (s,g)、一切分層報）。
  - 對照集＝task 2 十題單路雙掛（D=19；只跑 A/C 臂；A-w 規則見構造器節〔修②-①〕）。
  - 管線 gate 集〔修②-③〕＝**8 題全收**：task 1 episode **10,15,20,23,25,32,34**（七題）
    ＋task 3 episode 16（一題）（規則＝a_bits=64 且 b_bits=64 全部題；〔修③-③〕v3 手抄
    錯列 38/漏 32、本列已由 gate0b rows 機器重出——⛔ 列舉型事實一律機器輸出）。
- **編碼接口**：τ→E(τ)＝凍結源 encode_u／e_target（scratch:826 帶）→ head(cond,u)；
  s33 載入契約（EMA、normalization、T_CAP=128【encoder 軌跡點數、非 episode 預算
  〔修②-⑤〕】、mask 全 False）impl 自測證可呼叫。
- **凍結基底**：PIN 276c68fb… 唯一基底＋新 probe harness（import 凍結模組、sha pin 驗後；
  不走三處 sed derived；flat-only guard 不觸）；harness 過 S2、sha 進 receipt。
  〔v6.1、impl 首棒 BLOCKED 解〕：主樹現為 b7d3…（n64 後被改）、git 歷史無 276c
  （n64 跑的是未 commit 工作樹版）；**276c 實體＝experiments/_workorders/ucontrast1/
  smoke/mutants/M9/scratch_lacot_rollout.py**（create_frozen_worktree.sh 的
  source_copy、lead 10/1 親驗 sha 命中）——harness 以此路徑載入、啟動驗 sha。
- **episode 預算**〔修②-⑤：v2 把 T_CAP 誤當預算、撤〕：**H=1000**（large 環境 episode
  上限、同 n64 真協定）；三臂同 H、兩段共用不重置；model-call 計數隨 receipt 報。
- **streams（逐臂 seed 表）**〔修②-②：「雙流全新」宣稱撤、改逐臂明釘〕：
  ```
  公式（collector 凍結版）: base=7·task+ep；stream_seed=base+1000003·draw；
  draw 索引一律 64..79（Move 3）／64（Move 1、N 臂同）——與舊 0..63 的
  stream_seed 交集為零（①複審枚舉驗證）。
  Move 3 A/B/C 三臂統一協定〔修②-⑥〕：flow＝A 型 fresh（flow_seed=stream_seed、
    per draw 新流）；動作噪音＝全臂同 stream 同 σ=.05（noise generator 用同
    stream_seed）。⇒ 三臂唯一差＝w 介入（無/demo-w/alt-w）。
    ⚠️ 如實聲明：C 臂（fresh flow＋噪音）與 n64 B 臂（replay draw-0 flow）協定
    不同＝本輪新對照、⛔ 不與 n64 歷史數字直接比。
  Move 1 P/Q/R＝固定注入（無 flow 抽樣）；N＝A 型 fresh flow draw=64（⛔ 不退
    legacy torch.manual_seed、不掛 B 型 collector——impl 驗 N 的實際 seed 分支）。
  跨題 base 碰撞（如 (4,30)/(5,23) 同 base=58）：exact 判決一律【分幾何層】做、
    不跨幾何合併（碰撞只在跨 task 合併時成混淆）〔修②-②〕。
  實跑 seeds 逐 rollout 記錄進產物。
  ```
- **寫入面**：jasmine /archive/cymaxwelllee/breakthrough1/{move3,move1}/；
  PYTHONDONTWRITEBYTECODE=1；MuJoCo 暫存本機；XY trace **全解析度**〔修②-⑤、修 28；
  v6.1 修：提早結束（成功/terminated/truncated）時湊不足 H+1 點——長度契約改
  **len(trace)=實際 steps＋1 且 steps≤H**、終止原因欄必填、⛔ 不 padding 不拼接；
  validator 驗長度與 steps 欄一致〕。〔v6.3 裁、S2 B4：卡估 <40MB 是 binary 假設、
  JSON 實測 ~166MB（RNG 快照佔八成）——**分段落盤**（每題一檔或 shard）＋ RNG
  快照壓縮或獨立 sidecar；sbatch 時限 Move 3 **120 分**（CPU 實測 3.5-4s/rollout
  ×1280≈75 分留裕）、進度心跳每 50 rollouts 一行；ada-lite REQUEUE 搶佔=分段落盤
  後可續跑〕。
- **F5 鏈**：v3 ①三審 GO → 呈主人 → impl（S2）→ smoke（見 smoke 節）→ ETA 定版 →
  vFinal ①終審（補指令全文/配額/設定快照/失敗重提）→ ⑤發射官（smoke 級）→
  正式（ルナ親手）→ ③哨兵 → ④收割（獨立重算）→ 分支判讀。

## 路徑構造器（新立、確定性；兩探針共用）

- 輸入＝gate0b 的 (s,g) 格＋large 圖；輸出（凍結、sha 進 receipt）：
  - 主判 2 幾何：兩條等長路全格序、分岔點／合流點、互斥段、**route-A/route-B 固定
    命名**（字典序最小格序＝route-A；⛔ 命名不預設誰是 demo）。
  - **demo 標籤（獨立欄、可 unverified）**〔修②-①：不能只套 OGBench 粗圖平手序〕：
    對照訓練資料生成 oracle（ogbench stitch 生成器的細佔據圖＋tie-break 枚舉序）標
    route-A/B 哪條是 demo；核不出＝該幾何標 `demo_label=unverified`——Move 1 的
    可轉向判定**不依賴**此標籤（對稱設計、見下）、只有吸附判讀與 N 基線解讀降級。
  - **w（整點規則固定）**〔修②-①：task 5 中點 2.5〕：w 深度 d*＝floor(互斥段中點深度)、
    **兩臂同步取整同深**（task 4：d*=4、候選 (1,6)/(5,10)；task 5：d*=2、候選
    (3,1)/(1,3)——①複審枚舉之候選、構造器重導出核對）；w 必在互斥段內。
  - task 2（單路）A-w＝唯一路深度 floor(D/2)=9 的格；無 B 臂。
  - gate 幾何模板：task 1/3 各建 DAG（兩條等長路、D=15/12、路數 2——gate0b rows
    既有）、同規則出 route-A/B 與注入軌跡模板。
  - R donor 固定表〔修②-③；v6.1 修：task 2 不在 Move 1 題集、donor 項刪（impl
    首棒抓到的歧義）〕：task 4 題←task 5 ep 最小題 route-A 軌跡；task 5 題←task 4
    ep 最小題 route-A；gate task 1 題←task 3 幾何 route-A、gate task 3 題←task 1
    幾何 route-A。執行者零裁量。

## 探針一：Move 3 途經點三世界

- **在證明什麼、判準是什麼**：20 題多路雙掛題上，等深途經點介入（demo-w／alt-w）
  相對無 w 噪音對照（C）的救回格局 ⇒ 判失敗型態屬哪個世界、決定下一步分支。
- **三臂**：A=demo-w（demo 標籤 unverified 時＝route-A-w、結論寫 route 語言）、
  B=alt-w（同上另條）、C=無 w（協定見 streams 表、三臂唯一差=介入）。
- **兩段狀態機**：追 w→逐 step reach 鎖存（rho=1.875 同式同參）→命中才切 g；
  reached/cap/stuck 三因分離；w 段 cap=d*·步/格帳×2〔拍、smoke 校準〕；
  **bypass-g 事件**〔修②-⑤；修④-3：舊「語義=無 w 救回」句刪、以 d1 定義為唯一〕：
  未命中 w 先到 g＝記 bypass、該 draw 不計入 w 介入效果（獨立欄、定義見 d1）。
- **draw**：K=16（draw 64..79）；單題 0/16 只寫「此預算未觀察到」（95% 上限 17.1%）。
- **draw 分類（唯一、互斥、預註冊）**〔修③-⑤：歸因契約凍結〕：
  ```
  C 臂 draw：w 欄=N/A；分類＝{成功, 失敗}（g 於 H 內、同全臂成功口徑）。
  A/B 臂 draw（按序判、第一個命中即分類）：
    d1 未命中 w 先到 g           → bypass-success（觀測語言=「未完成指定 w 的成功」；
                                    ⛔ 不等同「無 w 救回」——此前仍受追 w 政策影響）
    d2 w 未達（cap/stuck）且未到 g → w-miss-fail
    d3 命中 w、g 失敗             → segment2-fail
    d4 命中 w、g 成功、穿越核實過 → full-rescue（唯一計入「介入救回」的類）
    d5 命中 w、g 成功、穿越未核實 → rescue-unverified（獨立欄、不入 d4、細讀）
  穿越核實【對稱口徑、機械判準】〔修③-⑤；v6.1 定準（impl 首棒要求）〕：
  核實過＝全程 trace 落格【入過本臂專屬互斥段 ≥1 格】且【入過另一臂專屬段
  =0 格】（走了本臂分岔、沒走對面；w 本在本臂段內故命中 w 蘊含前半、後半防
  折返走對面）。A 臂尺=route-A 專屬段、B 臂尺=route-B 專屬段——同一把尺。
  ```
- **題×臂級旗標與歸因順序**：w-miss 率>50% ⇒ `w-unreliable`。衝突裁定〔修③-⑤〕：
  題級落格先看 C（C 成功 ⇒ 落「C 對照救回」格、⛔ 不再讀 A/B）；C 掛才讀 A/B，
  此時 `w-unreliable` 的臂視為「無有效介入觀測」（≠失敗）、該題落
  「介入觀測不足」格入細讀——兩規則作用在不同層、無覆寫衝突。
- **model-call 帳**〔修③-⑤；修④-3：換段計帳凍結〕：計數單位=flow sample call
  （head forward 不限）；fresh flow 每 chunk 1 call＋命中 w 換段即刻重算 1 call ⇒
  上限=**ceil(H/chunk_steps)＋2**（換段裕度；H=1000、chunk=4 ⇒ 252）；超限＝
  `budget-exceeded`、該 draw 無效。**無效 draw 流轉**〔修④-3〕：先判有效性再分類
  ——無效 draw 不入 d1-d5、不入任何分母（含 w-miss 率）；題×臂有效 draws <8（K 半數）
  ⇒ 該題該臂「觀測不足」、按 w-unreliable 同路處置（⛔ 無有效觀測不得落成失敗）。
- **落格**（預註冊；觀測語言〔修③-③：C 因果命名越界、撤「噪音即救」〕）：
  ```
  C 成功 ≥1 draw        → 「C 對照救回」格（本輪 C=fresh flow＋噪音的合成效果、
                           ⛔ 不歸因到噪音單因；該題退出 w 介入判讀）
  C 全掛、A d4 B d4     → 時程形＋岔路可執行
  C 全掛、A d4 B 無 d4  → 時程形、alt 介入未見效（B 的 d2/d3/d5 分布隨格附上）
  C 全掛、A 無 d4 B d4  → demo 段局部洞指紋（單題細讀）
  C 全掛、兩臂 d1 合計≥K/2 → 「C 對照救回」格（bypass 版註記）〔v6.3 裁：S2 主錘
                          的 d1 劫持反例——15 個 d1 不得被單顆 d3 定格〕
  C 全掛、A B 皆無 d4   → 按兩臂 draw 分布主導分流〔修④-5〕：主導=兩臂【合併】
                          d2/d3/d5 計數的唯一最大者（⛔ d1 不入此分流〔v6.3〕）；
                          d3 主導=深執行洞候選、d2 主導=介入觀測不足、
                          d5 主導=介入疑似有效穿越未核實（細讀、⛔ 不入深執行洞）；
                          **平手/無唯一最大**〔修⑤-②〕=「介入觀測歧義」格
  介入觀測不足/歧義     → 細讀清單（不計世界主導數）
  ```
  〔v6.3 明文〕w-miss 率＝d2 數／該臂有效 draws（d1 不算 w-miss、它到了 g）。
- **兩層判決（方向修正）**〔修②-⑥：v2 停損句方向錯置、作廢〕：
  - 操作性停損（行動用）：**B 專屬救回**（B 救∧C 掛；兩向 discordant 並報）＜3 題
    ⇒ 「alt-w 介入在此預算下未見效」⇒ 資料注入線（軸一）降優先、先走 consumer
    診斷。⛔ 此句【不得】寫成「岔路執行洞」——介入無效≠執行洞。
  - 機制宣稱（CANON 用語門檻）：
    「岔路可執行」：B 專屬救回 exact 單向 p<.05（分幾何層、≥5:0 級）。
    「岔路執行洞」：需獨立證據＝B 臂 w 可達（w 段命中率高）而 g 段系統性掛＋
    per-step 證據；B/C 的 p ⛔ 不得借來定性此洞。
    零成功臂只寫「預算內未觀察到」＋預定實務成功率區間、⛔ 不判「走不了」。
- **規模**：(20×3＋10×2)×16＝1280 rollouts；ETA smoke 定版。

## 探針二：Move 1 steerability gate ⭐ 判決重心

- **在證明什麼、判準是什麼**：固定注入兩條路線的 E(τ)，看 rollout 的分岔段選擇
  是否**跟著注入內容走**（對稱可轉向檢定、不依賴 demo 標籤）⇒ 判 consumer 讀 u
  內容（跟隨者）或只認身分（吸附者）。
- **五臂（每題每臂 1 rollout、確定性、統計以題為單位）**：
  - P=route-A-u、Q=route-B-u（**對稱雙注入**〔修②-④〕）
  - R=別幾何-u（donor 固定表；身分負對照）
  - N=無注入 fresh flow（draw 64；自然傾向基線）
  -（gate 集只跑 N→P/R：**gate 的 P 源唯一=self-rollout-u**（該題 N 臂成功軌跡）、
    ⛔ 不用 route-A；主判 P/Q 才用構造器 route-A/B〔修⑤-①：雙定義殘句刪〕）
- **注入協定**：全程固定一條 E(τ)；逐 chunk 記送入 head 的 u hash（驗未被覆蓋）；
  「過期軌跡窗」列已知限制。
- **表示層 manipulation check（rollout 前置閘）**〔修②-④：round-trip 要保留身分〕：
  E(τ_A) vs E(τ_B) per 幾何可區分（距離>同路重抽噪聲帶、gate 幾何校準）＋
  **round-trip 身分保留**：decode(E(τ_A)) 的分岔段歸屬必須判回 route-A（B 同理）；
  不過＝「表示層不可判」、該幾何 P/Q 不跑。
- **compliance＝互斥段首入事件**（入 A 段／入 B 段／未進＋撞牆卡死獨立記）。
- **判定（預註冊、題級、分幾何層；出口優先序見合讀節）**：
  - **兩道 gate 分離**〔修④-2：送達 ≠ 行為資格；反例「注入全正確、consumer 忽略
    u、P/R 都復現 8/8」不得導向修管線〕：
    - **送達閘（機械）**：接口自測＋逐 chunk 送入 head 的 u hash 覆蓋檢查＝注入
      張量真的到了 head。不過＝修管線（E0）。
    - **行為資格閘（工程閘〔拍、未校準〕）**：gate 題先跑 N 臂（draw 64）1 rollout；
      成功者以該題成功軌跡為 P 注入源（self-rollout-u；統一定義：gate 的 P 源＝
      self-rollout、主判的 P/Q 源＝構造器 route-A/B〔修④-2 統一句〕）、N 失敗題
      除名。**合格題 ≥5 才開閘**、不足＝「gate 樣本不足」出口（探針改期）。
      判準：復現事件＝P rollout 的互斥段/終點歸屬與源軌跡一致【且】P 成功
      〔v6.3 明文：and、採實作語義〕；P 復現 ≥⌈合格題×3/4⌉ 且 **R 復現 ≤2 題**
      〔修④-2：R 門檻凍結〕⇒ 行為閘過。
      **P 過而 R 不低**＝「consumer 內容不敏感候選」（獨立出口名：⛔ 不是管線壞、
      不停探針）——主判 P/Q 照跑、結果掛 `gate-content-insensitive` 旗一併判讀。
      P 不過＝「行為資格不足」、主判照跑但 exact-zero 第二層降級為描述性。
  - **第一層：內容敏感性（exact-zero、硬檢定；確定性＝硬閘）**〔修③-①；修④-1：
    硬前提與失敗出口補〕：P/Q 兩臂除注入張量外協定全同 ⇒ 零假設「u 內容不被讀」
    下 P/Q 軌跡 byte-identical ⇒ n_diff>0。**硬前提（全部成立才准用 exact-zero
    語言）**：torch deterministic 模式（use_deterministic_algorithms＋固定後端旗標、
    官方 randomness 文件清單）、環境/policy 快取逐題重置、全 RNG 檔案化、dtype/
    batch/後端設定凍結、head 前後處理無非確定源；**符合性檢查＝實際 P/Q 路徑上
    同 u 重跑**（每幾何抽 1 題、P/Q 各重跑一次、byte-identical 才放行；R 重跑
    降為輔證）〔修④-1〕。**失敗出口**：任一項不符＝撤 exact-zero 宣稱、第一層
    降級為「行為差異觀測（不可歸因）」、第二層照報描述性。n_diff>0 的結論語言
    限縮＝「行為對注入張量有因果敏感性」；路線跟隨只由第二層判〔修④-1〕。
  - **第二層：方向（描述性反事實響應、⛔ 不配校準 p）**〔修③-①：v3 的「P/Q 同
    分布⇒切換等概率」零假設不成立（邊際同分布容許切換不對稱）、exact 宣稱撤〕：
    報凍結題集上的完整配對響應表——n_switch（P 入 A∧Q 入 B）、n_anti（P 入 B∧
    Q 入 A）、含 O（未進段）之九格計數、分幾何層。**行動門檻（工程判準、非檢定）**：
    n_switch ≥5 且 n_anti=0 ⇒ 按「跟隨者」行動；1≤n_switch<5 或 n_anti>0 ⇒
    方向性訊號、補證再判。校準推論留 confirmatory 階段（屆時預註冊置換法）。
  - 吸附（需絕對正證據＋demo 標籤 verified）〔修③-②：零 demo 偏好反例修〕：
    **P、Q 的 demo 段入題數各自 ≥⌈幾何層題數/2⌉**（絕對門檻：注入雙方內容都被
    拉回 demo 的多數證據）且 n_switch ≤1。demo 標籤 unverified ⇒ 只報「內容不
    敏感」弱判（n_diff 低且分布穩）、⛔ 不用「吸附」字樣。
  - N 天花板出口（分層）〔修③-⑥〕：task 4 層 N 同段入 ≥11/14、task 5 層 ≥5/6
    ⇒ 該層吸附判讀停用、只報兩層判定。
  - 其餘＝部分跟隨/不可判、題級分布細讀。
- **產物**：per 題×臂 XY 全 trace、E(τ) hash＋逐 chunk u hash 序列、首入事件、
  manipulation check 數值、實跑 seeds、receipt 全口徑。
- **規模**〔v6.2 裁：二棒抓到 v6.1 的「≤14」手滑（8 題全過=16）——採它的算式〕：
  主判 20×4＋gate N 前置 8＋gate P/R 合格題×2（上限 16）＋確定性符合性重跑 4
  ＝ **上限 108 rollouts**（N 失敗 gate 題與閘不過的層按條件遞減）＋encode 前處理；
  ETA smoke 定版。

## 合讀（出口優先序＋分支表〔修②-⑦〕）

**出口規則（E0 全局；E1-E4【按幾何層各自流轉】——除名一層、另一層繼續）〔修③-⑥：
「第一命中即止」與局部除名衝突修〕：**
```
E0 送達閘不過          → 全停：修編碼-注入管線（⛔ 不碰讀出端結論）
   〔修④-2：行為資格閘不過≠E0——「內容不敏感候選」「行為資格不足」各走獨立旗、
    主判照跑，見 Move 1 gate 節〕
per 幾何層（task 4 層、task 5 層獨立走）：
E1 該層表示層不可判    → 該層除名（細讀）；兩層全除＝先修表示層、探針改期
E2 該層 P/Q 失效主導   → 唯一公式〔修④-4〕：分子=該層「P 或 Q 任一臂失效
                          （撞牆∨卡死∨未進段∨無效）」的題數、分母=該層配對
                          題數（14 或 6）；>50% ⇒ 該層 consumer 判定停用、
                          注入工程細讀（⛔ 不用 rollout 合併口徑）。
                          〔v6.3 裁、S2 B3〕Move 1 各臂掛 stuck_check（參數同
                          Move 3 的 stuck 判準、smoke 校準）＝撞牆/卡死有紀錄；
                          校準前跑的資料 E2 分子標「下界」。
E3 該層 N 天花板       → 該層吸附判讀停用、兩層判定照報
E4 該層正常合讀（下表）；兩層結果併排呈報
```
**E4 分支表**（Move 3 落格主導 × Move 1 判定；「主導」＝count(top)−count(second)≥2
〔修②-⑦〕；落格表互斥 ⇒ 每題恰一格、主導數無多標籤〔修③-⑥〕。
**行選擇唯一序**〔修④-5：兩列同時命中時按表列順序、第一命中即止——明文〕：
主導格決定「行」後若另有輔訊號（如複合格主導差=2 且 B/C 8:0），輔訊號寫進該行
行動註記、⛔ 不開第二行。**d5（rescue-unverified）主導**〔修④-5 漏接補〕：A/B
無 d4 但 d5 主導 ⇒ ⛔ 不落「深執行洞候選」、落「介入疑似有效、穿越未核實」格
（細讀/不可判；行動=修核實口徑後重判；d5 計入自己格的主導數、不計 d4）：
```
Move 3 ＼ Move 1   跟隨者（行動門檻過）    內容不敏感/吸附            方向性/不可判
岔路可執行         軸一劑量反應三組對照    修讀出端（Astra 分段介面）  加題補證後重判
時程形主導         階層線另立案            階層線＋讀出端並行診斷      階層線另立案
深執行洞候選/
alt 介入未見效     consumer 診斷/字典線    同左（u 層關閉候選）        同左
C 對照救回主導     → support 假說重審、回 BoN 框內重讀
B d4 主導但未站穩  → 「方向性訊號」格：補 episode 擴證工單呈裁〔修③-⑥：
                      如 task 5 的 4:0——⛔ 不升格也不落混合格〕
混合格/幾何間分歧  → 暫不分支：分層明細呈主人裁（⛔ 不自動選線）
無合格題（除名殆盡）→ 「探針失效」出口：工程細讀報告呈裁
```
停損（B 專屬救回<3）只影響「軸一降優先」行動註記、不改表內判讀。

## smoke 計畫〔修②-⑧〕

正式 harness＋正式寫檔/收割路徑，覆蓋：
- 幾何×臂：task 4、task 5 各 1 題全臂＋**task 2 A/C**＋**gate 模板 task 1、task 3
  各 1 題 P/R（含 N 前置跑）**〔修③-⑧〕。
- 負例清單（收割器必須逐項正確拒收/標記）：缺列、重列、錯 seed、錯 reset 指紋、
  注入被 flow 覆蓋（故意不接注入層的 mutation）、**合法 w＋短 cap**（真跑進
  runtime 的 w 未達路徑〔修②-⑧：不可達 w 會被構造器先攔、測不到 runtime〕）、
  cap/stuck 三因、mid-chunk 換段異常、表示 check 過但 rollout 失效樣本、
  人工污染列。
- Move 1 確定性雙跑＝標「確定性驗證」（同 rollout 重現、⛔ 不算兩個 draw）。
- 測速 → ETA／時限定版進 vFinal。

## self_checklist

- [x] 四審 5 項逐項〔修④-N〕：1=exact-zero 硬前提清單＋同 u 重跑符合性＋失敗出口
  ＋結論語言限縮　2=送達閘/行為資格閘分離＋內容不敏感獨立出口＋R≤2 凍結＋合格題
  ≥5＋P 源定義統一　3=bypass 舊句刪＋無效 draw 流轉（先有效性後分類、<8 觀測不足）
  ＋換段計帳 ceil(H/chunk)+2＋計數單位分離　4=E2 唯一公式（配對題分母）
  5=E4 行選擇第一命中明文＋d5 主導分流（不入深執行洞）
- [x] 三審 6 洞、複審 8 must-fix：前輪 CLOSED 維持
- [x] 在證明什麼、判準是什麼：兩支頭行就位；exact-zero 有硬閘與降級出口
- [x] 拍的數字明標；教訓隨卡：列舉型事實機器輸出、未校準判準不冒充檢定
- [ ] ①五審（待）→ 主人點頭（待）→ impl/S2/smoke/vFinal（待）
