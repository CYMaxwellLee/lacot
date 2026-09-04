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
2. TA=32 對 large-stitch（中位 48 步）可能欠採樣 — 本輪只跑 medium。
3. 先導 2 顆只判「值不值得 8 顆」：看 nll 形狀＋不炸，⛔ 不看單顆分數（一顆不是量測）。
4. 對照組＝N3 已有八顆（.842 sd .108）— 三臂各 8 顆對它，雜訊 ±.03 規則沿用。

## 基準格（9/4 早收齊）

subgoal 腿：s20 .792 sd .040／s27 .857 sd .032（新王座）／ebfs 天花板 ~.99／N3 .842。
R0 腿：.373（256d）→ .456（8d 壓縮）→ .505（字典、plateau ~.51）。
