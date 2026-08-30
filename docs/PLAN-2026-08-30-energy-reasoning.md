# 作戰計畫 2026-08-30：energy-guided planning ＋ 蒸餾自舉

_主人 8/30「好，規劃吧」。整合今天的討論（u reason 多段 energy、expert iteration、三路調研、SCoTS 批註）。_
_每個 phase 開工前跟主人確認；判讀規則【事先】釘死，⛔ 不准開獎後補。_

## 故事主軸（一句話）

reasoning 在規劃層：非參數 energy（資料建圖）先當 teacher 教 u 推理、再當 verifier 陪 u 自舉，
最後把圖 amortize 掉 — 部署時只剩一個會長程規劃的 latent planner。
執行層交反應式（bc）＝三軸解剖量出來的發現，不是妥協。

賣點四張牌：amortization（GAS/TTGS 的 limitation 是我們的動機）、teacher 最優性（vs SCoTS 覆蓋導向）、
condition-consistency 病理學（93% 量化=文獻空位）、E-verified 閉環（verifier 不可 tamper/不 drift/保守可審計）。

---

## P0：收官 ＋ ebfs 對照（今天）

- ✅ 已收：large 矩陣（bcbfs 1.000／conf2 0.240／S0 0.440／bc 0.190、L2 ckpt 同源配對）、
  C2-S0a（chunk1 對齊 0.580、重想 103→15、特異度臂二連綠）。
- ⏳ 在跑：C2-S0（chunk1 未對齊、同 ckpt 配對格）、L2-latbc（large 矩陣最後一格）、
  原 L-train/C1-train 長尾（⛔ 不砍，它們最後存自己的 ckpt）。
- 🔜 待下（主人點頭即發）：**ebfs 四支**（eval-only、n=100）
  ```
  M-ebfs-bc  medium A2s0  SUBGOAL=ebfs SUB_POLICY=bc GRAD_REFINE=1 FINISH_R=2   對照 oracle 0.910
  M-ebfs-u   medium A2s0  SUBGOAL=ebfs GRAD_REFINE=1 FINISH_R=2                對照 S0+fin 0.750
  L-ebfs-bc  large  L2s1  同上 bc 版                                            對照 oracle 1.000
  L-ebfs-u   large  L2s1  同上 u 版                                             對照 S0 0.440
  ```
  判讀規則（先釘）：
  1. `ebfs+bc vs oracle(bcbfs)` 差 ＝「資料圖 vs 真圖」的代價。差 ≤5 分 ⇒ 資料圖夠好、teacher 上界確立。
  2. `ebfs+u vs ebfs+bc` ＝ u 執行劣勢在合法供點下的複測。
  3. ebfs+bc 若在 large 大幅掉（>15 分）⇒ 先修 E 圖品質（res/連通後處理），P1 暫停。

## P1：治根因 — decoder 錨定 ＋ ebfs teacher 資料引擎（1~2 天）

**P1a. decoder 錨定 s（結構修 93% 病）**
- 改 `traj_decoder`：輸出改相對位移、路徑第 0 點結構上恆＝s（Diffuser 的 inpainting 精神、我們的架構版）。
- 驗法：重訓 medium 一顆 → d0 診斷（‖路頭−s‖）應塌到 ~0；large eval 看 flat/conf2 變多少。
- ⚠️ 架構改動 ⇒ 舊 ckpt 不相容 — 跟 P1b 重訓【合併成同一次】，別重訓兩趟。

**P1b. ebfs teacher 資料引擎（治 cond OOD）**
- 生成器：從 E 圖抽 (s,g) — 距離分布【重尾】（p50 落在原分布內、尾巴蓋到考題級 12+ 格；調研配方：
  Huginn/Looped-WM 的 Poisson 家族）→ 圖 BFS 路徑 → 內插成軌跡（速度按資料統計）。
- 混合：teacher 配對【混入】原資料（起手 1:1，ablation 掃）。⛔ 永不取代原資料（防塌鐵律）。
- 訓練：架構零改（u 照壓軌跡）；medium＋large 各一顆（含 P1a 的錨定 decoder）。
- **主判決格**：large 的 conf2 供點（現 0.240）與 flat（0.080）在新 ckpt 上重量。
  判讀（先釘）：conf2 ≥0.70 ⇒ 供點器被治好、供點戰收官在望；0.4~0.7 ⇒ 有效但要疊 P2；
  <0.4 ⇒ cond 分布內化不足，查 (s,g) 覆蓋與 d0。
- 配套：隨機 E／打亂 E 對照組實作（Spurious Rewards 教訓 —「E 在承重」需要對照證明）。

## P2：自舉閉環 — E-verified expert iteration（3~5 天）

- Loop：抽 (s,g)（課程半徑逐輪推遠）→ flow 生 M 份 → E 逐段打分（E-to-go 差分＝進度）→
  通過集【按 exp(−E) 加權】蒸餾（⛔ 不用過線 argmax）＋保留 FM/NLL loss 錨（LaDiR 原句）→ 下一輪。
- 監控（警報器先裝）：pass@M（掉=多樣性死、停迭代）、E 分數 vs 真成功率相關（Gao 曲線、衰減=Goodhart 進場）、
  per-iteration 保留 ckpt（回滾點）。
- E 封縫三件套（P2 開跑前做）：
  1. fuzz：對抗性生成「E 高分」計畫丟模擬器、量假陽性率；
  2. 同構不變性抽查（平移/鏡射/重標格）；
  3. held-out 真 rollout 抽查 — ⛔ 永不進訓練訊號。
- 深度隨機化：refine/重想圈數訓練時用抽的（Poisson）＋E 收斂 early-exit。

## P3：距離場 ＋ 收割對標（下週）

- **d 蒸餾**：同一個圖 teacher 的第二個 student — quasimetric 參數化（IQE/MRN）、監督＝圖 BFS 精確距離
  （無 bootstrap 複利）。用途：①部署時替代圖當「更靠近」判準（去圖化完成）②E 第四項（verifier 懂進度）
  ③課程難度尺。
- **humanoidmaze-stitch**：SCoTS 沒做的地 — xy 投影圖 teacher、同管線。antmaze-stitch 同批。
- **對標表**（官方協定）：GCBC/GCIVL/GCIQL/QRL/HIQL（official impls）＋SCoTS/GAS/TTGS（引數字或跑其 code）
  ＋兩格自設最硬對照：`GCBC+engine`（同 teacher 資料餵 GCBC — 資料贏還是方法贏）、
  `d+greedy`（距離場直接出 policy — u 的計畫層價值）。
- **ablation 套件**：同 teacher 蒸 policy vs 蒸 u（泛化假說判決格）、E 各項消融、隨機 E、
  M/R scaling 曲線、chunk、K（計畫句長）、teacher 混合比。
- **病理學標準化**：condition-consistency 違反率（d0 分布）當正式 diagnostic，
  跨方法量（我們 vs baseline）＝paper 的病理節。

## v2（本篇之後）

VQ 離散錨（連續 proposer＋離散錨＝調研藥方）、codebook 圖搜索（高維版 ebfs）、
E 距離場全面化、cube/scene（節點=狀態聚類、邊權=d）。

## 風險與依賴

- P0-3 判讀規則影響下游：ebfs+bc 掉太多 ⇒ P1 暫停修圖。
- P1 重訓×2 顆（各 ~2h、zeldajr AMD 驗證過）；P2 每輪重訓 ~2h×3-5 輪；GPU 現有配置足夠。
- 對標表的 SCoTS/GAS/TTGS：優先引官方數字（protocol 對齊要驗）；跑其 code 是 fallback（時間貴）。
- ⚠️ 特異度臂：large 崩區出現過 4 分漂（不顯著）— 續累積各支數據再裁門檻，⛔ 先不動。
