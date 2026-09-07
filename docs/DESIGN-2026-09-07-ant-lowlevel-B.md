# DESIGN — B 案：ant 階層低層（短程路標到達頭）｜2026-09-07、待主人裁後施工

_主人裁示鏈（9/7 TG）：5909「螞蟻的動作學不起來才是麻煩的…看這幾天推導的理論來規劃」→
5910 ルナ提 A/B 兩路 → 5911「好」。本檔＝B 案設計；⛔ 施工等 A 曲線＋GCBC 錨落地、主人裁。_

## 理論定位（一段）

三物件鏈：intent（訓練鷹架）→ weights（知識）→ u（路線）→ 動作。計畫棧在 ant 上已驗活
（pilot 25436：stage1 收斂、decoder 強讀 u、幾何全綠）；死的是動作端＝「肌肉」那格。
帳本語言：走路資訊全在資料（1M 步示範）、走模仿通道 — 不需要 verifier/GRPO（那是之後救
路線完成率的）。低層化＝把「遠程模仿」（8 維步態×遠 goal、訊號稀）拆成「短程模仿」
（每段實走都是稠密教材）。文獻錨：同地圖 HIQL 94±1 vs flat GCBC 45±11（arXiv:2410.20092 T2）。

## 一手量測（9/7、val set）

- ant 每步位移 |Δxy|：p50 0.133、p90 0.210 單位。
- 現行 conf2 subgoal 間距（pilot log：E 格寬 0.71×隔 11 格 ≈ 7.8 單位）≈ **59 步**的路程。
- ⇒ 訓練 k 範圍與 eval leg 長度必須重疊（下面 3.）。

## 設計

1. **新頭 π_lo(a | s_full, w_xy)**：輸入＝完整 obs（29）⊕ 路標 xy（2、獨立小 encoder 直吃、
   ⛔ 不走 goal_to_obs 補 0 那套 — 低層是新頭、乾淨直給）；輸出＝CHUNK×ADIM 動作塊（同現行）。
   規格對齊 bc head（sota_mlp 512）。
2. **訓練配對＝短程 hindsight**：(s_t, xy_{t+k})、k ~ U[10, 60]（依上面量測 ≈ 1.3~8 單位路程）。
   loss＝BC（與官方同款）、加進 stage2 迴圈：`LACOT_LO_W`（預設 0＝off ⇒ golden zero-diff）。
3. **eval 接線**：`LACOT_SUB_POLICY=lo` 新模式 — conf2 分段選出 subgoal 後、該 leg 交 π_lo
   （吃當下 state＋subgoal xy）。leg 長度要能調（現行間距 59 步在 k 上限邊緣；
   若 G1 顯示 k=60 到達率差、優先縮 eval 間距對齊訓練分佈、⛔ 不是拉長 k 硬撐）。
   終局接管維持 takeover=off(ant)。
4. **Gates**（順序、掛了回設計）：
   - G1 π_lo 單獨到達率：held-out 短程配對、到達判定用 env 官方 success 半徑（⛔ 不自訂尺）；
     要求＝顯著非零且隨步數上升（走路溫度計 #2）。
   - G2 conf2 全程成功率 vs A 臂（同預算對照、同 seed）。
   - G3 Int 三臂（base/ref/idp-best）騎上選定執行器 — 執行器一旦選定、三臂共用、⛔ 不混。
5. **施工紀律**：旗預設 off；pointmaze golden A/B 重驗（同 9/7 三跑法：A1/A2/B 同卡）；
   ⛔ 不動既有臂與 eval 語義；施工使魔 CPU-only、GPU 由ルナ排。

## 成本粗估

施工＝使魔一單（頭＋配對取樣＋eval 接線＋smoke）；訓練＝π_lo 短程模仿收斂快（【猜測】
數萬步級、單卡 1~2h）；G1/G2 eval 照現行批。

_相關：DESIGN-2026-09-07-ant-migration-v1（v1 落地）；NOTE-2026-09-07-ant-judge-options（裁判 A 案）；
PLAN-0906-forward 主線 D；官方表（GCBC 45±11／HIQL 94±1）＝地基錨。_
