# 實驗 1 判決（教師資格；2026-09-28、lead 親讀 s0 代表＋blocks 三 seed 同型）

正式資料（197k 窗/domain、20k steps、hidden 256、split hash 三 seed 一致：
pointmaze 9aec9ac9 / ant ca2dd9cf）。產物 jasmine:/archive/cymaxwelllee/refine_v3/exp1/。

## Wφ（動作條件後果模型）：兩 domain 資格【過】

判準（預註冊、L1/L2 修訂版）：world NMSE ≤ 0.9×obs-only 且配對改善 CI 下界>0，分 h 分子空間。

```
                h=4                h=16               h=32
pointmaze xy    9.1e-7/7.9e-4 ✓    3.4e-6/7.0e-3 ✓    8.8e-6/2.4e-2 ✓
ant xy          3.1e-4/5.3e-4 ✓    2.0e-3/5.7e-3 ✓    5.3e-3/2.2e-2 ✓
ant yaw         .041/.071 ✓        .081/.258 ✓        .099/.408 ✓
ant gait        .104/.296 ✓        .137/.594 ✓        .145/.724 ✓
（world/obs-only；全格比率 <0.9、CI 下界全 >0、shuffled 對照全顯著劣化=判別力有牙）
```

⭐ 判決級意義：ant 的 xy/yaw 專訓就過（含 h=32 長程）——C1 刀的「ŝ 頭 xy/yaw 是
頭病可修非碼病」正式蓋章，修法=獨立訓一顆 Wφ。v3 的「替自己領域造便宜 oracle」
第一根柱子立起。PI 覆蓋 ~.90、validity ~.90-.92（健康欄、不擋資格——L1/L2 修訂
的正確性側證：這兩格數字對好壞教師都差不多）。

## Aω（同計畫動作老師）：兩 domain 資格【未過】（差距大）

controller_action_nmse：pointmaze .764 / .456=ant（門檻 ≤.1；三 seed 同型）。
＝從「完整起始 state＋插值路徑窗」推不回原始動作——DESIGN 的資格風險預言命中
（pointmaze obs 僅 2 維、路徑為插值點、action 可辨識性不足）。

按 DESIGN (d) 預裁停損：**pointmaze Aω 不過 ⇒ refine 支線（實驗 4）暫停**；
selector 線（實驗 3）只依賴 Wφ、不受影響。ant Aω 本為 diagnostic。
⛔ 不硬修不繞道（「不拿插值 XY 差冒充 action」紅線在案）；若要救 Aω 需另案
（時間 adapter/更多 state 資訊）呈裁。

## generated 兩 gate：missing=預期（需真 flow 候選；實驗 3 掛真候選時填）

## 對後續的縫合

- 實驗 3（selector-only）：Wφ 供應器資格已備。ant 側（privileged relay 橋接、
  三法候選）照走；pointmaze 側的候選供給疑慮與今晚 u 對照 200 題判決同看
  （smoke 級「8 抽無解鎖」若確認 ⇒ pointmaze 供給病、先修供給再談挑選）。
- 實驗 4（refine）：前置未過、暫停——spec 停損門正常運作、省下其預算。
- 實驗 2（接線）：獨立、照排（mod-actor/objective 下一批）。
