# NOTE — ant 移植 v1 實作筆記（2026-09-07、branch `ant-v1`）

_設計依據＝`DESIGN-2026-09-07-ant-migration-v1.md`，衝突時以設計檔為準。_
_本檔只記「設計的每一條落在哪一行」＋「驗了什麼、怎麼驗的」＋「沒做的事」。_
_行號皆為 `experiments/scratch_lacot_rollout.py`（本 branch 的狀態）。_

---

## 一、設計條目 → 落實位置

### 核心：兩個空間、一條投影

| 名稱 | 行 | 說明 |
|---|---|---|
| `XY_DIM / OBS_DIM / ADIM` | 48-50 | 維度從資料推導（ant 29/8、pointmaze 2/2） |
| `OBS_XY / MU_XY / SD_XY / MU_XY_T / SD_XY_T` | 52-54 | 計畫棧用的 xy 切片（numpy ＋ torch 兩套） |
| `ENV_FAMILY / IS_ANT` | 58-59 | env 家族判定（給終局接管的閘） |
| `to_xy()` | 62 | **唯一**投影入口；pointmaze 上是恆等 |
| `xy_to_obs_slot()` | 69 | 正規化 xy → 正規化 obs（非 xy 維補 0＝資料平均）；給只有位置的合成樣本 |
| `goal_to_obs()` | 79 | 原始 xy 路標 → 決策端的完整 obs（非 xy 維填 mu） |

### 設計檔 1：幾何消費者改吃 XY

| 消費者 | 行 | 做法 |
|---|---|---|
| GeoEnergy 佔據圖 ×6 份（teacher／intent／GRPO／BoN／BOOT_GEN／主 GEO） | 445, 479, 1363, 2449, 1980, 2325 | 建構參數換 `OBS_XY, MU_XY, SD_XY` |
| **e_target 的軌跡窗**（設計決定 1：計畫側軌跡窗也投影） | 645 | `make_batch` 的 traj 用 `OBS_XY/MU_XY/SD_XY` 內插 |
| `etarget()` 的守門 | 778-784 | `assert Dc == XY_DIM` —— 餵完整 obs 進來當場炸 |
| `traj_enc` 輸入維 | 718 | `sota_mlp(XY_DIM, ...)` |
| decoder 起點 token `s_embed` | 899 | `nn.Linear(XY_DIM, ...)` |
| `_dec()` 的起點投影 | 963 | 呼叫端可能給完整 obs 或 xy ⇒ 一律 `to_xy` |
| DEC_START 哨兵／soft 懲罰 | 1072, 1077 | 跟 `to_xy(s)` 比 |
| E_geo 端點（select 臂／conf2 的 E 選計畫） | 2266, 2722 | `to_xy(s)/to_xy(g)` |
| `grad_refine` 端點（flat／latent／conf／conf2） | 2277, 2671, 2692, 2717 | 同上 |
| GRPO reward 的眼睛 | 1717 | `_grpo_reward(pts, to_xy(s_zn), to_xy(g_zn))` |
| GRPO 的 δ_step 校準軌跡 | 1407 | 用 `OBS_XY` 建 |
| BoN 打分端點 | 2586 | `to_xy(s[0]), to_xy(g[0])` |
| BoN 的 C8 校準窗 | 2501 | 用 `OBS_XY` 建（roundtrip-decode 空間＝計畫棧） |
| E 圖 cell↔xy 轉換 | 2390, 2398 | `MU_XY/SD_XY`（原本用完整 mu/sd，ant 上會廣播錯位） |
| E 格寬 | 2430 | `SD_XY` |
| 往返尺／flow 探針的座標還原 | 835, 843, 847, 2676, 2696, 2727 | `* SD_XY_T + MU_XY_T` |
| subgoal 長程層的幾何 | 2660-2661 | `s_xy/g_xy/goal_xy` 三個投影本地變數 |
| 選路標 / headguard / fallback | 2728, 2745, 2754, 2764, 2769 | 目標一律用 `goal_xy` |
| 荒漠重採樣的格 | 581 | `to_xy(OBS)`（4×4 格的常數已查證 ant 也對齊，見該行註解） |

**決策端維持完整 obs**：`cond_enc`（1122）、`sg_c`（718）、`bc_own_enc`（1180）輸入改 `OBS_DIM`；
`make_batch` 的 `s/g`（646）不投影；`normstate()`（2072）加 assert —— **只吃完整 obs**，
餵 xy 路標進來當場炸（分段規劃的路標要先過 `goal_to_obs`，見 2822）。

### 設計檔 2：維度從資料推導

`ADIM` 從 `ACT.shape[1]` 推（50），原本寫死在 424 行那一排常數裡（`B, D_MODEL, TEMP, ADIM = ..., 2`）。
`ActionMLP` / `CondOnlyMLP` 的輸出（`CHUNK*ADIM`）、teacher 佔位動作、dev 尺的亂走臂因此自動跟著走。

### 設計檔 3：終局接管按 env 家族關閉

`FINISH_ON = FINISH_R > 0 and not IS_ANT`（376）；兩個觸發點（flat 2215、分段 2811）改判 `FINISH_ON`。
⛔ **不動 `FINISH_R` 本身** —— 檔名的 `_fin` 段是這條臂的身份，要留著。
log：檔頭印 `takeover=off(ant)`（377-379 與 426-430），收工那行補「0 次是【預期】的」（3435）。

### 設計檔 4：teacher hindsight/route

兩模式都走 xy cell，投影之後語義不變 ⇒ **一行沒改**。預設仍是 hindsight。
teacher 樣本的 `s/g` 因為只有位置，改用 `xy_to_obs_slot` 補到 obs 維（655-656）——
合法性：那些樣本的 action loss 本來就被 real-mask 擋掉，只餵幾何側。

### 設計檔 5：MAXH / success

照 env 官方（`env.spec.max_episode_steps`、`info["success"]`）—— 現行已如此，**沒改**。

### 設計檔 6：goal conditioning

訓練 hindsight goal＝未來 state（完整 obs 進決策端、計畫端由 traj 投影承擔）；
eval goal＝env 給的 goal obs。分段規劃的路標只有 xy ⇒ 過 `goal_to_obs` 補成完整 obs（2822）。
`_rt_sg`（2884）改存完整 obs，往返尺與 flow 探針內部各自 `to_xy`。

### 產物可追溯

非退化 env 才寫入：rollout json 加 `obs_dim/act_dim/xy_dim/env_family/finish_takeover`（2925），
ckpt cfg 加 `ENV/OBS_DIM/ACT_DIM/XY_DIM/FINISH_TAKEOVER`（3528）。
⛔ pointmaze 上兩者都是空 dict ⇒ **json 與 ckpt 逐位元不變**（golden gate 的立足點）。

---

## 二、驗證（CPU、zeldajr 本機、`~/venvs/lacot-rocm`）

### 關卡 1：pointmaze 零差

兩份配置、改動前後同設定各跑一次，**stdout 逐字 diff 相同 ＋ ckpt/json 的 SHA256 相同**：

```
A 輕量（sg_infonce 預設路徑）      st20 T128 ep1 MAXH20   19 秒
B 重裝（9/6 正式配方的整條計畫棧）  st20 T64  ep1 MAXH20   2 分 35 秒
   recon_ictr decoder + teacher0.5 + intent embed + GEO + conf2 分段
   + BoN N=4 + GRPO + EMA + DEC_START=soft + COND_DROP + L_div + FINISH_R
```

`ckpt_..._s120_s0.pt  1848f7c2…  OK`（A，前後同）
`ckpt_..._s1150_..._s0.pt  27d26965…  OK`（B，前後同）

⚠️ **零差測試前先修好一個會讓它報假警的東西**：ogbench 的 maze `add_noise` 走
**全域 `np.random`**（`locomaze/maze.py:565`），而官方 `rollout()` 只釘
`env.reset(seed=)` 與 `torch.manual_seed` ⇒ 同一份 code 跑兩次，每集的起點/終點抖動就不一樣。
實測：B 配置跑兩次，BoN 的統計欄逐次不同。
⇒ 驗證 harness 在 exec 主檔前先 `np.random.seed(20260907)`（**只在 harness，⛔ repo code 一行沒改**），
之後 A/B 兩份都「同 code 跑兩次逐字相同」，零差比對才成立。
（`lacot/dev_eval.py:131` 早就為了配對而釘了這條流；官方 `rollout()` 那條沒有 —— 見下面殘留風險。）

### 關卡 2：ant CPU smoke（`antmaze-medium-stitch-v0`）

四支全部 rc=0：

1. **輕量**（20 秒）：資料載入 → stage1 20 步 → stage2 20 步 → 5 個 task 各 1 集 rollout。
   `obs_dim=29 act_dim=8、計畫棧投影 xy_dim=2（env 家族 antmaze；終局接管 takeover=off(ant)）`
2. **重裝**（2 分 39 秒）：teacher 題庫 4096 條建得起來（細格步數 p50 27 / max 63）、
   intent 錨 fallback 0/1280、GEO 佔據圖 `(39,43)` 覆蓋 40.9%、
   `格心 round-trip 2.10e-06`、`盒內隨機點穿牆中位 0.1207`、**健康檢查通過**、
   `decoder 讀得到 u（打亂後 0.3975 → 1.7478，差 +1.3504）`、
   GRPO β 定標開火（`β_target=1.836e-01`）、BoN 換過計畫 1.9%、
   `終局接管觸發 0 次　⭐ 本次 takeover=off(ant)：0 次是【預期】的`
3. **dev 尺 ＋ D0/D4 前置量測**：`build_dev_tasks` 在 ant 上生得出 3 層題（BFS 3~7）、
   亂走臂（8 維動作）跑得動、D0/D4 兩格都算得出來。
   ⚠️ 「尺的驗收沒過」是 smoke 尺度的必然（20 步訓練 × MAXH 30 ⇒ 全臂 0/6），**不是** code 問題。
4. **LOAD_CKPT 只評估**：往返尺 `t1~t5 mse 0.10~0.40 / 穿牆 0.03~0.25 / 末點距 0.7~2.0`、
   flow 探針 oracle 進度 0.59~0.76 —— 兩支都是這次改過 `SD_XY_T/MU_XY_T` 的路徑。

**佔據圖 xy 範圍**（獨立腳本、兩個 env 對照）：

```
pointmaze-medium-stitch  obs  2  格(39,43) 覆蓋0.390 格寬0.70
    資料   xy x[ -1.29, 21.25] y[ -1.16, 21.25]
    自由格 xy x[ -1.20, 21.17] y[ -1.07, 21.16]  mapping_err 1.59e-06  health_ok True
antmaze-medium-stitch    obs 29  格(39,43) 覆蓋0.409 格寬0.71
    資料   xy x[ -1.74, 21.43] y[ -1.43, 21.24]
    自由格 xy x[ -1.63, 21.33] y[ -1.36, 21.17]  mapping_err 2.10e-06  health_ok True
```

兩個 env 是同一張 8×8 迷宮（maze_unit 4.0、offset 4，一手查過）⇒ 佔據圖形狀、格寬、覆蓋率
全部對得上 ⇒ **xy 投影沒有把座標系搬歪**。

**ckpt 內的形狀**（ant 重裝那顆）：`traj_enc` 入 `[512, 2]`（計畫棧）、
`cond_enc` 入 `[512, 29]`（決策端）、`ahead` 出 32 ＝ CHUNK 4 × ADIM 8。

### 關卡 3：語法／邊界

`python -m py_compile experiments/scratch_lacot_rollout.py` 過。
「寫死 2 維」逐點清單見下一節。

---

## 三、「寫死 2 維」清單與處置

**改掉的（原本在 ant 上會錯）**

- `ADIM = 2`（舊 362 行常數排）→ 從 `ACT.shape[1]` 推
- `traj_enc / sg_c / cond_enc / bc_own_enc = sota_mlp(2, ...)` → 計畫棧 `XY_DIM`、決策端 `OBS_DIM`
- `s_embed = nn.Linear(2, D_MODEL)` → `XY_DIM`
- `etarget` 的 `traj.reshape(Bc*Tc, 2)` → `XY_DIM` ＋ 加 assert
- `_dec` 的 `reshape(-1, 2)` → `to_xy(...).reshape(-1, XY_DIM)`
- `GeoEnergy(OBS, mu, sd, ...)` ×6 → `OBS_XY, MU_XY, SD_XY`
- `_e_xy_to_cell` 的 `(xy[:2] - mu) / sd` → `MU_XY / SD_XY`（**ant 上會直接廣播炸**）
- `_e_cell_to_xy` 的 `z * sd + mu` → `SD_XY / MU_XY`
- `_E_CELL_XY` 的 `* sd` → `SD_XY`
- `* SD + MU` ×6（往返尺／flow 探針／三個 subgoal 臂）→ `* SD_XY_T + MU_XY_T`
- GRPO 與 BoN 的 C8 校準軌跡（`OBS[...]-mu)/sd`）→ `OBS_XY/MU_XY/SD_XY`
- `box["goal"] = np.asarray(goal[:2])` → 存完整 obs，幾何用時 `to_xy`
- `policy_chunk(obs, planner.sub, ...)` → `goal_to_obs(planner.sub)`
- BOOT_GEN 的 `condvec(st, gt)`（xy 進決策端）→ `xy_to_obs_slot` 補到 obs 維
- `_rt_sg` 存 `_o[:2]` → 存完整 obs

**查過、確認留 2 是對的（都在計畫棧或純幾何路徑）**

- `_i_zn_to_cell` 的 `z[:2]`、`_route_traj._cell` 的 `mu[:2]/sd[:2]`、BOOT_DESERT 的 `OBS[:, :2]`
- teacher cell jitter `size=(1,2)`、intent 錨 `np.zeros((INTENT_TA, 2))`
- `_anchor_pts / planner.observe / arc_subgoal / SUB_DIAG` 的 `obs[:2]`（全是幾何距離）
- `DIAG_DUMP` 的 start/goal/final `[:2]`、`_xy_to_ij` 的 `xy[:2]`
- FINISH 判距的 `obs[:2] / goal[:2]`
- `lacot/traj_decoder.py` 的 `nn.Linear(d_model, 2)`（decoder 解的就是 xy 路徑）
- `lacot/intent_{embed,anchor,residual}.py` 裡所有 `2`（錨序列 `[T_A,2]` 本來就是 xy）
- `lacot/refine_grad.py` `GeoEnergy` 內部的 2 維格制（它的契約就是吃 `[N,2]`）
- `PerceiverPooler(512, D_MODEL, K, 2, 4, ...)` 的 `2`＝num_layers、
  `torch.stack([sg_c(s), sg_c(g)], 1)` 的 2＝兩個 token、`cond_head` 的 1024＝512×2 —— 都不是空間維

---

## 四、殘留風險 / 沒做的事（誠實列）

1. **⛔ GPU golden gate 沒跑** —— 工單範圍止於 pointmaze CPU 零差。
   本檔給的是「同一台 CPU、同設定、前後 ckpt SHA256 相同」，`gold-post2 = a315d385` 那格由ルナ親自跑。
   ⚠️ 我覆蓋的兩份配置沒有涵蓋 `FSQ / VQ / DEC_START=hard / INTENT=anchor|residual / BC_OWN /
   CONT_TRAIN / SUBGOAL=latent|conf|bfs|ebfs / U_SOURCE=oracle / BOOT_GEN / AMP / COMPILE`。
   這些路徑我逐行改過也 py_compile 過，但**沒有跑過** ⇒ 只算「讀過」，不算「驗過」。
2. **官方 `rollout()` 沒有釘全域 np.random** —— 這是本次查出來的**既有**問題，⛔ 沒有修
   （修它會改變所有歷史 pointmaze 結果）。後果：官方協定下 `bc / null_u / shuf / R{k} / intent 三腿`
   各臂拿到的每集起點抖動**不一樣** ⇒ 那些臂之間**不是配對比較**（dev 尺那條有釘、是配對的）。
   要修的話是獨立一單、要主人裁（它會讓所有官方數字跟歷史不可比）。
3. **teacher／自舉樣本的姿態維填 0（＝資料平均）** —— 那是一個選擇，不是唯一解。
   它們只餵幾何側（action loss 被 real-mask 擋掉）所以不會污染 head；
   但 `cond_enc` 會看到「xy 合理、姿態是平均」的 (s,g)，ant 上這是訓練分佈裡沒有的組合。
   ⚠️ `TEACHER_MIX=0.5` 時有一半 batch 長這樣 —— **v1 pilot 要盯 cond 分佈**，必要時改成
   「從資料裡撈一個 xy 最近的真 state 當姿態」。
4. **分段規劃的路標同上** —— `goal_to_obs` 把路標的姿態維填 mu。pointmaze 上不存在這件事，
   ant 上「走到那個 xy、姿態隨便」正好是我們要的語義，但它同樣是 OOD 的 cond。
5. **`DATA_RESAMPLE` 的 4×4 格是常數** —— 查過 pointmaze／antmaze 的 medium/large 都是
   `maze_unit=4.0 / offset=4` 所以對齊；換到別的迷宮尺寸要重算（已在該行留註解）。
6. **`_INTENT_ROUTE_CACHE` / `_GRPO_RCACHE` 的 key 是 cell 對** —— ant 上不同姿態、同 xy 格的
   兩個 state 會共用同一份錨。這是**對的**（錨本來就只看 xy），但值得記著：
   快取命中率在 ant 上會比 pointmaze 高，⛔ 別把它讀成「規劃變少了」。
7. **smoke 的成功率全是 0** —— 20 步訓練 × MAXH 30~40，那是**預期**且**不構成任何訊號**。
   ⛔ 別把這份筆記裡的 0 讀成「ant 上跑不動」。
