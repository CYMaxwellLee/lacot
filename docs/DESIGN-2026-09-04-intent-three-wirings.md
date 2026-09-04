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
