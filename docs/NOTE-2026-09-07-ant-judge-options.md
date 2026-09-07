# NOTE — ant 裁判（verifier）重設計選項（2026-09-07 早、跟主人討論用）

_背景：主人 9/7 早裁「早點看怎麼讓它更好＋進展高維」；資料格已解（HF 鏡像
ryanhoangt/ogbench_data、antmaze m/l stitch 四檔 9/7 09:15 下載入 zeldajr ~/.ogbench/data、
三層驗證過：schema 同構／ogbench API 端到端／env reset）。本檔只整理選項、不動工。_

## 一手驗證（9/7 早、zeldajr、lacot-rocm venv）

- `ob[:2] == qpos[:2] == ant 全域 xy`（obs 29 維＝qpos15＋qvel14、含全域座標）[實測]
- goal 是完整 29 維 state、其 `[:2]` ＝目標 xy（36.8, 24.8）[實測]
- 資料 xy 範圍 x∈[-1.6,37.6]、y∈[-1.6,25.7] ⇒ 佔據圖可照 pointmaze 流程從資料重建 [實測]
- 上游 rail server 仍死（Berkeley EECS 全站 outage、issue #49 open、rail 302→iris WordPress）[實測]
- 鏡像出處紀律：第三方鏡像、結構驗證過；官方復活後回頭對位元；paper 若用需揭露。

## 核心事實：迷宮幾何還是 2D

ant 的「身體」高維（29），但「迷宮」不是 — 佔據圖／BFS 場／穿牆檢查全部活在 xy 平面。
⇒ 裁判的重設計不是「換一個裁判」，是「裁判讀計畫的哪個影子」。

## 選項

**A. xy 投影裁判（v1 提案、最小改動）**
decode(latent plan) → 29 維軌跡 → 取 xy 影子 → 進既有佔據圖＋BFS 場＋穿牆檢查（管線同
pointmaze、從資料重建佔據圖）。
- 前置要量的一格：**往返尺在 ant 空間重跑**（decode→xy 忠實度；pointmaze 版 mse/穿牆/末點距
  三錶直接搬）。這格不過、裁判讀到的影子是假的 — 先量再上。
- 不做的：身體姿態合法性（翻倒等）v1 不裁 — 判準是 Int>0 重現、不是分數 SOTA（主人 9/5 定調）。
- 風險：plan 空間（e_target/u）在 29 維下超參（K、COND、sigma）要重整；stage1 重跑是本來就要的。

**B. 表示空間裁判（缺課地圖族、第二章）**
e_target／quasimetric 空間的 kNN 距離＋NLL 兩把尺（DESIGN-DRAFT-0902-highdim-gap-map 起點頁）。
- 適用：非迷宮 env（手臂等）、或 humanoid 要更細的失敗定位時。
- 校準紀律：先在 maze 上驗「分數高＝死點聚集」再帶走（gap-map §3）。
- v1 不用它裁 ant — antmaze 有 2D 真相可用，不需要先過校準關。

**C. 模擬器 rollout 裁判（真值、最貴）**
直接展開 env 驗計畫。留作 BoN/GRPO 的 oracle 上界對照格，不當常規裁判（成本）。

## 建議路徑（呈主人）

1. A 選項 v1：管線移植（obs/action/cond 維度過渡＋xy 投影點）→ 往返尺 → stage1 ×2 pilot →
   stage2 pilot（PLAN D2、~6 GPU-h）。teacher 只剩 hindsight ⇒ O-agnostic claim 自然被測。
2. 裁判用 A；B 留 humanoid／失敗定位；C 當 oracle 對照。
3. 機台：zeldajr 四卡（已呈主人、待裁）。

_相關：PLAN-2026-09-06-forward §主線 D；DESIGN-2026-09-04（口徑五件套）；
DESIGN-DRAFT-2026-09-02-highdim-gap-map；THEORY-0906-verifier-taught（幫浦理論、env 無關）。_
