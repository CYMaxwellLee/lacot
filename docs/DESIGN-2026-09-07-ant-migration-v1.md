# DESIGN — ant 移植 v1（antmaze-medium-stitch；2026-09-07、主人裁「先小地圖、確認方法＋能否走路」）

_主人 9/7 早裁示：先把螞蟻做在小地圖、確認我們方法以及能否走路。線一＝官方 GCBC 地基（另單、
jasmine）；本檔＝線二（我們管線）的 v1 設計決定。判準照 9/5 定調：Int>0＋配對差站起來、不追分數。_

## 核心設計決定：兩個空間、一條投影

**計畫棧（stage1 e_target／intent 錨／flow 的 u／裁判／teacher／subgoal E 圖）＝全部住 xy 投影空間。**
**決策端（action head、BC 地板頭、cond 的當下狀態）＝吃完整 obs（ant 29 維）。**

理由：
1. u 的正名地基＝軌跡規劃（CANON-u-semantics）。路線就是 xy — 把計畫棧釘在 xy 是語義對齊，
   不是偷懶；關節姿態不進計畫、由決策端從資料模仿吸收（走路住 action head 權重）。
2. 迷宮幾何本來就是 2D：佔據圖／BFS 場／穿牆檢查／intent 錨（[T_A,2] cell 路線）全部原樣可用。
3. pointmaze 上 obs≡xy ⇒ 投影＝恆等 ⇒ **golden zero-diff 天然成立**（開關其實可以是無條件的
   `to_xy()`，但仍照規矩掛旗）。
4. e_target 超參（K/COND/sigma/T_A）在 xy 空間下有機會直接沿用 — v1 先沿用、不掃。

一手驗證過的事實（9/7 早、zeldajr）：ant obs[:2]==qpos[:2]==全域 xy；goal 為 29 維 state、
其 [:2]＝目標 xy；資料 xy 範圍 x∈[-1.6,37.6] y∈[-1.6,25.7]；episode 結構 201 步/集。

## 具體改動面（主檔 scratch_lacot_rollout.py、3420 行）

1. `XY = OBS[:, :2]` 投影 helper；所有幾何消費者（GeoEnergy 佔據圖×3 份獨立建構、teacher 題庫、
   intent 錨 hindsight/route、subgoal E 圖、GRPO reward 眼睛、eval 端 E 圖）改吃 XY。
   模型 I/O（e_target 輸入軌跡窗、flow、頭）：**計畫側軌跡窗也投影到 xy**（設計決定 1），
   決策側 state/goal 保持全 obs。
2. 維度全部從資料推導（obs_dim=29、act_dim=8 不寫死；pointmaze 值退化回 2/2）。
3. **終局接管必須按 env 家族關閉**（pointmaze 是速度控制、可直接朝目標走；ant 不能）：
   ant 下 disable、計數照印、log 印明「takeover=off(ant)」。已呈主人（9/7 TG 5904）。
4. teacher：hindsight（實走 xy cell 序列）與 route（xy 佔據圖 BFS）兩模式在投影下皆可運作 —
   v1 預設 hindsight（現行預設），route 留旗不動。
5. eval：MAXH 走 env.spec（官方、已如此）；success 用 env 官方判定；conf2 分段協定的 subgoal
   語義在 xy 下不變。
6. goal conditioning：訓練 hindsight goal＝未來 state（29 維全餵決策端；計畫端投影）；
   eval goal 用 env 給的 goal obs。

## 紀律（不可妥協）

- 開關式：`LACOT_ENV` 指到 antmaze 才走新 codepath；pointmaze 預設路徑 bit 級不變。
- **golden gate**：gold-post2 設定（pointmaze）在新 code 下 ckpt SHA 必須仍= a315d385；
  獨佔跑（slurm 取卡、單卡無鄰居 — 9/6 事故教訓：共卡下同 code step 375 就分岔）。
- CPU smoke（ant）：資料載入→stage1 幾步→stage2 幾步→rollout 1 題不炸、shape 全對。
- ⛔ 共用機 GPU 一律走 slurm；⛔ 施工使魔不准自行提 GPU job（工單止於 golden＋smoke，
  GPU pilot 由ルナ驗收 diff 後親自排）。

## Gate 順序（v1 pilot、migration 過 golden 後）

①stage1 ×1（antmaze-medium、zeldajr slurm）→ ②往返尺（decode→xy 忠實度：mse／穿牆／末點距
三錶照搬）→ ③stage2 pilot＋誠實 BC 地板（＝「會不會走」溫度計、對表線一 GCBC）→
④R0／配對差初讀。任一 gate 掛＝回設計，⛔ 不硬調。

_相關：NOTE-2026-09-07-ant-judge-options（裁判選項、A 案已由本檔採用）；PLAN-0906-forward 主線 D；
FINDINGS-0905（antmaze regen 出處）；主人裁示鏈 TG 5893/5897/5900/5902。_
