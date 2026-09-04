# 兩層架構第一戰：intent 三接法同日對比 — 2026-09-04

_主人裁示鏈：「先 A 再 B」（intent 來源＝現成 E 格路線，字典版之後）→「探討 categorial
加 continuous」→ (iii) residual 是主人提的 →「開始吧，我也想知道…能節省多少token」。
⭐ 本實驗同時是 fleet-command skill 的第一個正式案子（三 sonnet 寫模組、opus 施工＋
opus 驗收、ルナ設計＋抽驗）。_

## 設計

上層（A）：E 佔據圖路線 — 訓練＝hindsight 實走 cell 序列、推論＝BFS 最短路；
`lacot/intent.py` 沿弧長重採樣成 [T_A=32, 2] 正規化錨點（smoke 5/5）。

三接法（統一 IntentAdapter 介面、都騎在 N3 配方＝8d 連續 z 上）：

| 臂 | 機制 | 對照拆什麼 |
|---|---|---|
| (i) embed `_ite` | 錨→MLP 全域向量→cond_head（軟） | 路線「資訊」的價值 |
| (ii) anchor `_ita` | 錨切 K 段 per-token 進 flow prefix（硬） | 「結構化擺放」的加值（vs i） |
| (iii) residual `_itr` | 訓練目標＝traj−錨軌跡；推論+回（同 (i) cond） | 「輸出空間分工」的加值（vs i） |

底層改動：nf_head Flow 支援 3D cond（per-token prefix、Permutation flip 同步在
Flow 內處理；golden 鎖 2D 行為 bit 級不變）。(iii) ⛔ 不能配 S1_FROM（殘差語言重訓）。

## 判讀表（先釘死）

```
(i) > N3(.842)     路線資訊有用；差值＝資訊價值
(ii) > (i)         結構化擺放再加分 ⇒ Hydra 式耦合對
(iii) > (i)        層該切在輸出空間 ⇒ 下一步上層真 categorical＋可搜索
全 ≈ N3            分開驗：a 執行層瓶頸（看開火率/命中診斷）
                   b 訓推錨分佈差（診斷開關：eval 餵實走式錨）
(iii) 大掉         殘差 scale 沒控（l_nf 絕對值、9/3 臂 B 教訓）
R0 掉              u 的路線負擔被 intent 分掉 — 預期內，⛔ 別報退步（N4 教訓）
```

## 預註冊風險

1. **訓推錨不匹配**：hindsight（彎、實走）vs BFS（直、最短）。預期傷硬約束 (ii)(iii)
   多於 (i)。診斷開關（privileged、eval only）等主人點頭再加。
2. TA=32 對 large-stitch（中位 48 步）可能欠採樣 — TA 是後續 ablation 軸。（v1 誤寫
   「只跑 medium」— 基準 N3/.842 全是 large-stitch，對照純度要求先導同 env。）
   先導 4 支（9/4 午）：ITE/ITA 騎 N3 配方；ITR 無 FSQ＋重訓 stage1；ITE0＝ITR 的
   配對對照（同底、只差 target 變換）。jobs 23522/23524/23526/23528＋各自 eval。
3. 先導 2 顆只判「值不值得 8 顆」：看 nll 形狀＋不炸，⛔ 不看單顆分數（一顆不是量測）。
4. 對照組＝N3 已有八顆（.842 sd .108）— 三臂各 8 顆對它，雜訊 ±.03 規則沿用。

## 主人兩問（9/4 午、rebuttal 素材）

1. **「推論靠 BFS 會被批評不公平」** — 成立一半。防守：資訊合法（圖從 D 建、同
   model-based 學 dynamics 的地位）；但 claim 要擺成「研究怎麼把 planner 知識接進
   生成式管線」而非「打贏 baseline」，表上標注。ebfs .99 早證明此 env 選路已被 BFS
   解完 — 有價值的是「怎麼餵」的判決，不是分數。
2. **「高維沒有 BFS 怎麼辦」** — 即「先 A 再 B」的 B：離散 intent 改學（Hydra 式
   字典 k=64＋字典空間搜索）；「查」從格圖 BFS 換成學到的圖（k-NN／quasimetric／
   value 造的距離）上找路。**今天的判決跨表示可攜**（切段釘 flow／壓摘要給 head／
   residual 吃輪廓品質）— A 買接口答案，B 換引擎不重猜。

## 先導判決（9/4 午、單顆 s40 只判方向）

ITA .976（subgoal 王）／ITE R0 淨 .524（R0 王、破字典 .505）／ITE0 .960（無 FSQ 也強）
／residual 兩錨源判死（hindsight＝殘差無資訊 u 失業、gate 正確攔截；route＝.412）。
⇒ 放量 21 支（ITA/ITE/ITE0 × s41~47）already 灑。
⭐ residual 死因入教訓：**殘差接法的資訊分配吃錨的品質 — 錨越準連續層越失業**；
蹺蹺板兩端（太準/太粗）都不在甜蜜點。

## 基準格（9/4 早收齊）

subgoal 腿：s20 .792 sd .040／s27 .857 sd .032（新王座）／ebfs 天花板 ~.99／N3 .842。
R0 腿：.373（256d）→ .456（8d 壓縮）→ .505（字典、plateau ~.51）。

## 高維偵察＋競品定位（9/4 晚；細讀正文級、五篇）

**OGBench 高維迷宮地形**：ant 29 維／humanoid 69 維、xy=qpos[:2] 直接 slice ⇒ E 圖照建。
stitch＝5000 ep（收集政策本身就是 BFS 子目標）。地板（官方 Table 2）：antmaze-large-stitch
HIQL 67；humanoidmaze-large-stitch HIQL 28；giant 全塌（2~4）。eval＝5 task×20 ep、xy≤0.5。
⚠️ giant 瓶頸可能在低層 locomotion、無 runtime oracle 可借。

**競品族（搜索＋生成、細讀後修正版）**：
```
             圖/搜索              waypoint 進法       生成空間      可比分數(細讀逐字)         成本/query
XDiffuser    kNN+學距離           能量梯度 guidance   原始 state    AntMaze StitchGiant 90.0   未報
SIHD         kNN+結構熵樹(無搜路) 純量 CFG            原始 state    D4RL——不可比               未報
TTGS         value距離+Dijkstra   無生成模型(檢索直餵) N/A          hm-giant-stitch 78.1       輕
ChronoForest per-anchor樹+multi-tree guidance 注入    原始 state    antmaze-stitch m/l/g ~99   91.9 秒
C-MCTD       MCTS×3+圖上Dijkstra  classifier-guided   原始 state    AntMaze giant 75±18        37~530 秒
```
🚨 掃描級誤報教訓（主人擺正）：「C-MCTD giant 100%」＝PointMaze 那格；「XDiffuser 98.5」
＝Explore 非 Stitch。⛔ 摘要級聲稱一律細讀驗設置再信（insight 已存）。

**我們的 claim 空間（五篇檢驗後全立）**：① latent 計畫語言＋head 可讀（R0 腿）— 五篇
全在原始座標、零 latent；② 兩腿評估框架 — 唯一；③ 效率：毫秒 BFS＋8×8 token flow vs
秒~分鐘級搜索迴圈；④ humanoidmaze-large-stitch＝五篇全空、baseline 28 的空地。
⭐ TTGS limitation 自承「未來要用生成模型補中間 state」＝我們的 motivation 引言。
同構鄰居另 8 篇（MCTD 2502.07202 等）在掃描報告；Hydra 引用鏈太新無收穫。
