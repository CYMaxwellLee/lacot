# NOTE — B 案 ant 階層低層 π_lo 實作筆記（2026-09-07、branch `lo-v1`）

_設計依據＝`DESIGN-2026-09-07-ant-lowlevel-B.md`，衝突時以設計檔為準。_
_本檔只記「設計的每一條落在哪一行」＋「驗了什麼、怎麼驗的」＋「沒做的事」。_
_行號皆為 `experiments/scratch_lacot_rollout.py`（本 branch commit `4ff80c6` 的狀態）。_
_分支基底＝`main@712216e`（goldab harness 入庫那顆）。⛔ 全程沒有動 main。_

---

## 〇、一句話

`LACOT_LO_W>0` 之後多出一條**跟主模型零共用**的短程頭 π_lo(a | s_full, w_xy)；
`LACOT_SUB_POLICY=lo` 讓 conf2 分段的**開車那顆頭**換成它（⛔ 分段機制一行沒動）。
⛔ 兩支旗都不開 ⇒ 模組不建、graph 不進、RNG 不碰、檔名不變。

---

## 一、規格條目 → 落實位置

| 規格 | 行 | 做法 |
|---|---|---|
| `LACOT_SUB_POLICY` 新值 `lo` | 336 | assert 擴成 `("", "bc", "lo")` |
| 旗：`LO_W / LO_KMIN / LO_KMAX / LO_ADV / LO_ADV_BETA` | 344-355 | 全部 env var；`LO_ADV_WMAX=5.0` 寫死（356，⛔ 護欄不是旋鈕） |
| `LO_ON`（模組要不要在場） | 358 | `LO_W>0`（訓它）**或** `SUB_POLICY=="lo"`（載它來評） |
| **π_lo 頭**（規格 1） | 1211-1231 | `lo_s_enc(OBS_DIM→512)` ⊕ `lo_w_enc(XY_DIM→512)` → `lo_ch(1024→COND)` → `lo_head=CondOnlyMLP()`。⭐ 結構＝`bc_own` 那條鏈的同構版，只把 goal 端 encoder 換成路標 xy encoder ⇒ 同容量、可直接跟 bc_own 對打 |
| 建構包 `fork_rng`＋固定 seed | 1219-1220 | `torch.manual_seed(20260907 + SEED)`（同 bc_own 8/31 R3 前例） |
| 路標**不**走 `goal_to_obs` | 1226-1228、3037-3040 | 低層直吃 2 維 xy（自己的 encoder 的語言） |
| **短程 hindsight 配對**（規格 2） | 1238-1256 | 同軌跡內 `(s_t, xy(s_{t+k}))`、`k~U[KMIN,KMAX]` 整數；`t+k` **clamp 到 `traj_end`**（⛔ 不 `continue` 重抽 —— 同 `make_batch` F6 的理由：重抽會系統性丟掉靠近軌跡結尾的起點） |
| 動作目標 `a_t..a_{t+CHUNK-1}` | 1281 | `ACT[r:r+CHUNK]`；抽樣時已要求 `traj_end[r]-r >= CHUNK` |
| ⛔ 不碰主資料流 | 1234 | 專用 `_LO_RNG`（同 GRPO `_GRPO_QRNG` 紀律）；⇒ **主模型與 `LO_W=0` 那顆逐位元相同**（已實測，見二-③） |
| loss＝與現行 BC 同款 | 1758 | `mse`（π_lo 的 batch 全是真資料、沒有 teacher 佔位 ⇒ 等價於 `_wmse` 權重全 1） |
| 訓練掛進 stage2 迴圈 | 1753-1778 | 獨立 batch／graph／`opt_lo`／clip（完全照 `BC_OWN` 那格的形狀，含 AMP 三件套） |
| `opt_lo` | 1302-1303 | Adam lr 5e-4（同 bc_own；⛔ 同樣不吃 `LR_SCALE`，跟 bc_own 對齊） |
| **梯級一 advantage**（規格 3） | 1258-1275、1759-1772 | `v = ‖xy(s_{t+k}) − xy(s_t)‖ / (t+k−t)`（路標就是 `s_{t+k}` 本人 ⇒ 朝它的位移＝這段淨位移）；`v̄`＝5 萬筆同抽法配對的中位；`w = min(exp(β(v−v̄)/v̄), 5)` |
| v̄ 定標流不吃 SEED | 1268 | `default_rng(20260907)` ⇒ ⭐ 各 seed 用同一把尺（否則跨 seed 的 w 不可比而且不會報錯） |
| w 的 p10/p50/p90 印一次 | 1762-1768 | 首批印，附 max ＋「打到上限幾筆」 |
| `l_lo` 進 log | 1943-1945 | 接在既有 `step` 行尾（⛔ `l_lo is None` 時整段不接 ⇒ 舊 log 逐行不變） |
| **eval 接線**（規格 4） | 2446-2456、3037-3040 | `lo_policy(obs, w_xy)`：`normstate(obs)` ⊕ `(w−MU_XY)/SD_XY` → `lo_act` → clip ±1。分段的選點／換段／stuck 判定**一行沒動** |
| **per-leg 統計**（規格 4） | 2789-2830（錶）、2850／2857／3035（記帳）、3210／3404（印） | 見下面〈三〉 |
| 檔名 `_loW{值}` | 3588-3593 | `LO_W>0` 才加；`+a{β}`（LO_ADV）、`+k{min}-{max}`（非預設 k）——⚠️ 它們改的是**權重本身**，照 GRPO 那段的防互蓋慣例一起帶 |
| 檔名 `_splo` | 既有 3549 那格 | `SUB_POLICY` 本來就進檔名 ⇒ `lo` 自動變成 `_splo`，⛔ 沒有新 code |
| ckpt 存／載 | 3767-3770（存）、1991-2001（載） | `lo` 段四模組＋cfg 血緣（`LO_W/KMIN/KMAX/ADV/BETA/VBAR`，3797-3800）；⛔ 兩者都 gate 在 `LO_ON` ⇒ 預設 ckpt 逐 key 不變 |
| `.eval()` | 2098-2100 | π_lo 不在 `f_mods` ⇒ 同 `bc_head` 的理由要自己切 |
| json 落地 | 3646-3651 | `leg_stats`（每個分段 arm 一列）＋ `lo`（設定＋v̄）；⛔ gate 住 ⇒ 舊 json 逐 key 不變 |

### 三道護欄（都**實測炸過**，⛔ 不是只有讀過）

| 護欄 | 行 | 實測 |
|---|---|---|
| `SUB_POLICY=lo` 但沒 `LO_W>0` 也沒 `LOAD_CKPT` | 1213-1215 | `AssertionError: ⛔ …π_lo 會是隨機初始化的權重，而它【不會報錯】` |
| `LO_ON` 但 ckpt 沒有 `lo` 段 | 1992-1994 | 把 lo 段拿掉的複本餵進去 ⇒ `AssertionError: …這顆 ckpt 沒有 lo 段` |
| `LACOT_SUB_POLICY` 亂填 | 336 | `AssertionError: ⛔ LACOT_SUB_POLICY 只能是空/bc/lo，收到 xx` |

---

## 二、驗證（CPU、zeldajr 本機、`~/venvs/lacot-rocm`；`CUDA_VISIBLE_DEVICES=""` `HIP_VISIBLE_DEVICES=""`）

### 關卡 1：pointmaze 零差（旗全關、改動前後同設定）

harness＝`slurm/goldab/detrun.py`（exec 主檔前先 `np.random.seed(20260907)`；ogbench maze
`add_noise` 走全域 `np.random`，不釘它零差比對會報假警 —— 見 ant-v1 note）。
「改動前」＝`git show main:experiments/…` 落到一個借用 repo root 的目錄（`lacot/` symlink），
⛔ repo 一行沒改、⛔ 沒有動 main。**兩份配置、stdout 逐字 diff 相同 ＋ ckpt/json SHA256 相同**：

```
A 輕量（sg_infonce 預設路徑）      st20 T128 ep1 MAXH20         20 秒
B 重裝（9/6 正式配方整條計畫棧）    st20 T64  ep1 MAXH20   2 分 39 秒
   recon_ictr decoder + teacher0.5 + intent embed + GEO + conf2 分段（sp bc）
   + BoN N=4 + GRPO + EMA + DEC_START=soft + COND_DROP + L_div + FINISH_R
```

```
A  ckpt_..._st20_T128_ep1_gu_s120_s0.pt        1848f7c2dd9757a5…  前後同
   rollout_...json                             f86cdc8097e5bb5a…  前後同
B  ckpt_..._s1150_..._bon4_s0.pt               27d26965ba126782…  前後同
   rollout_...json                             be148dbb7a302d99…  前後同
   divlog / grpolog jsonl                      65a80aa6… / 5e501cd8…  前後同
```

⭐ A 的 `1848f7c2…` 與 B 的 `27d26965…` **跟 ant-v1 那一單記下來的同一顆對得上** ⇒ 兩單串起來
都沒有動到 pointmaze 預設路徑。
⚠️ 這一關**跑了兩次**：第一次在 commit 前，之後又改了 `l_lo = LO_W * l_lo` 那一行，
所以照最終 code 重跑一遍 —— 上面的數字是**最終狀態**的。

### 關卡 2：ant CPU smoke（`antmaze-medium-stitch-v0`、`OGBENCH_DATA_DIR=~/.ogbench/data`）

**① 訓練有印、有降勢、不炸**（輕裝 ＋ `LACOT_LO_W=1`、stage2 120 步、rc=0、1 分 3 秒）

```
⭐ π_lo 低層頭建立（obs 29 ⊕ 路標 xy 2 → 4×8；LO_W=1  k∈[10,60]  adv=0）
  step 10  l_nf/dim -0.005 l_anchor 0.5367 l_refine 0.5612 l_lo 0.4961
  step 40  l_nf/dim -1.335 l_anchor 0.3797 l_refine 0.3862 l_lo 0.3629
  step 80  l_nf/dim -2.077 l_anchor 0.3369 l_refine 0.3326 l_lo 0.3176
  step 120 l_nf/dim -2.015 l_anchor 0.2943 l_refine 0.2825 l_lo 0.2856
```

**② advantage 那格**（同配置 ＋ `LACOT_LO_ADV=1`、rc=0）

```
⭐ π_lo advantage 定標：v̄（位移/步 中位）=0.0697  p10/p90 0.0102/0.1499  (n=50000、β=1、w 上限 5)
⭐ π_lo advantage 權重 w：p10 0.451 / p50 0.916 / p90 3.054  (首批 n=64、max 5.000、打到上限 1 筆)
```

⭐ v̄=0.0697 對得上設計檔的一手量測：每步**路徑長** p50 0.133，而 v 是 10~60 步窗的**淨位移/步**
⇒ 必然比路徑長小（螞蟻不走直線），約一半是合理的。w 的分散度不退化（p10/p90 差 6.8 倍）。

**③ ⭐「LO_W 不污染主流」實測**（⛔ 不是只有讀過）：同設定跑 `LO_W=0` 與 `LO_W=1`，
主模型的 12 行 log（`l_nf/l_anchor/l_refine`）**逐字相同**、四個 arm 的成功率也相同
⇒ 專用 rng ＋ fork_rng ＋ 獨立 optimizer 這三道確實兌現。

**④ eval 接線 ＋ per-leg**（重裝 ant 訓一顆帶 π_lo 的 ckpt ⇒ 再 `LOAD_CKPT` 只評估、
`LACOT_SUB_POLICY=lo`、MAXH 200、rc=0、38 秒）

```
⭐ π_lo 低層四模組已載入（訓練時 LO_W=1.0 k∈[10,60] adv=0）
  分段 conf2（官方協定）: success 0/5 = 0.000
  ⭐ per-leg [official conf2]: legs attempted 35 / legs reached 0 / reach rate 0.000 / median steps per leg 24   (集末未結算 5 段；到達半徑 rho 1.875、路標目標間距 DELTA_SUB 7.5)
```

檔名：`rollout_antmaze-medium-stitch_…_sgconf2_ma2_splo_gr0_fin2_bon4_loW1_s0.json`
（`_splo` ＋ `_loW1` 都在）。訓練那支的 ckpt 檔名是 `…_loW1_s0.pt`。

⚠️ **reach rate 0.000 不構成任何訊號** —— 那顆 π_lo 只訓了 40 步。
⛔ 別把這份筆記裡的 0 讀成「B 案不行」。

**⑤ dev 尺那條 call site**（`_leg_report` 的第二個呼叫點）也跑過了，但**得靠 harness 繞過
main 上的兩個既有 bug**（見〈四〉1、2）：

```
  分段 conf2 : 0.000 ± 0.000  (n=6)   步數中位 100   最近距離中位 11.01
  ⭐ per-leg [dev 分段 conf2]: legs attempted 23 / legs reached 0 / reach rate 0.000 / median steps per leg 20   (集末未結算 6 段；到達半徑 rho 1.875、路標目標間距 DELTA_SUB 7.5)
```

### 關卡 3：語法／維度盤點

`python -m py_compile experiments/scratch_lacot_rollout.py` 過。

**新 code 的「寫死維度」逐點**

| 寫死的數 | 在哪 | 判定 |
|---|---|---|
| `sota_mlp(OBS_DIM, 512, 512)` / `CHUNK*ADIM` | 1221、`CondOnlyMLP` | ✅ 從資料推導，⛔ 沒有寫死 |
| `sota_mlp(XY_DIM, 512, 512)` | 1222 | ✅ 留 2 是**對的** —— 路標的契約就是 xy（同 ant-v1 的計畫棧判準） |
| `sota_mlp(1024, 512, COND)` 的 `1024` | 1223 | ✅ ＝512×2（兩個 encoder 的輸出），⛔ 不是空間維（同 `bc_own_ch`） |
| `OBS_XY[...]`、`axis=1` | 1260-1262 | ✅ 幾何量，只該吃 xy |
| `np.asarray(w_xy)[..., :XY_DIM]` | 2453 | ✅ ＝`to_xy` 語義（呼叫端給 xy 或完整 obs 都對） |
| `_leg_close` 的 `obs[:2]` | 3035 | ✅ 跟上一行 `planner.observe(obs[:2])` 同一把尺，⛔ 不可以改成完整 obs |
| `0.25 * DELTA_SUB` | 2817 | ⚠️ 只是 `pl` 不在手上時的 fallback 顯示值；實際判定一律讀 `pl.rho` |
| `512` / `lr 5e-4` / `clip 1.0` | 1221-1223、1303、1776 | ✅ 刻意對齊 `bc_own`（同容量同預算才比得動） |
| seed `20260907` | 1220、1234、1268 | ✅ 常數，⛔ 不是維度 |

---

## 三、per-leg 統計怎麼算的（⛔ 判讀前先看這節）

一段 **leg ＝「這個路標從被 `planner.set()` 設下、到被換掉」之間走的路**。

- **attempted**：被**結算**的段數 ＝ `planner.observe()` 回 True（到了／走滿 cap／卡住）那一刻。
- **reached**：結算時 `‖現在 − 路標‖ < planner.rho`（rho ＝ `0.25·DELTA_SUB`）。
  ⭐ **跟換段的判定用同一把尺** —— ⛔ 不自訂第二把（兩把尺會給出互相矛盾的「到了」）。
- **median steps per leg**：`planner.since * CHUNK`（`since` 是 **chunk 數**，⛔ 不是 env step）。
- **集末未結算 N 段**：episode 在段中途結束（success／MAXH）⇒ **只計數**，
  ⛔ 不進 attempted、不進到達率、不進步數中位。
  理由：把它算成「沒到達」會把**成功收尾**的那一段也記成失敗 ⇒ 憑空的悲觀偏差。
- 錶跟著 policy 走：`make_subgoal_policy()` 一建就歸零 ⇒ 每個 arm 各自從零數。

⚠️ 這條線量的是**低層單獨**的短程到達率，⛔ 不是全程成功率（那格混著長程供點的品質）。
⭐ 這正是 G1 要的那一格。想拿 bc 頭當對照量同一格：`LACOT_LEG_STATS=1`（預設只有
`SUB_POLICY=lo` 自動開；⛔ 開了會多印一行 ⇒ 跟歷史 log 逐字比對時要記得）。

---

## 四、殘留風險／沒做的事（誠實列）

1. 🚨 **既有 bug（⛔ 不是我這單造成的、⛔ 也沒有修）：`DEV_EVAL=1` ＋ 分段 arm 在 main 上就會炸。**
   `lacot/dev_eval.py:142` 把 task **dict** 傳給 `on_episode_start`，而主檔 `on_start` 拿它當
   dict 的 key（`box["ep_count"][task]`，9/2 的 TRACE 那格帶進來的）⇒ `TypeError: unhashable type: 'dict'`。
   **實測**：同一份配置在 `main` 上炸在同一行（`mainref/…:2649`），在本分支炸在 `:2860`。
   ⇒ 修法二選一（都要主人／ルナ裁）：`dev_eval.py` 改傳 index，或 `on_start` 改用可 hash 的 id。
   **⚠️ 這會擋住 G1/G2 走 dev 尺的分段 arm。**
2. 🚨 **第二個既有 bug（同一條路徑）：`_q` 被覆蓋。**
   主檔 946 行 `def _q(u)` 是 VQ/FSQ 量化器（`policy_chunk` / `_dec` 都在用），而
   `DEV_EVAL` 的 subgoal 診斷區 3023（main 行號）寫 `_q = lambda v: …` **把它蓋掉**
   ⇒ 之後跑的官方協定 arm 會拿到那個 lambda ⇒ `RuntimeError: Boolean value of Tensor…`。
   ⚠️ 觸發條件＝`DEV_EVAL=1` ＋ `SUBGOAL` 非空 ＋ 診斷有樣本 ＋ **官方 arm 跑在 dev 尺之後**
   —— 而那正是 G2 想要的跑法（同一顆、同一輪、dev 尺＋官方協定）。⛔ 沒修，理由同上。
3. **`LO_W` 在梯度被 clip 的期間是「看不出來」的。**
   π_lo 的 clip 是 1.0，而起步梯度範數 > 1 ⇒ `LO_W=0.5` 與 `LO_W=1` 的**更新完全一樣**
   （實測：前兩個 log 點 `l_lo` 剛好是 2 倍關係，第三點才開始岔開）。
   ⇒ ⛔ 別把 `_loW0.5` 讀成「訓練訊號減半」；它只在梯度掉到 clip 以下之後才真的有作用。
4. **`v̄` 是 5 萬筆配對的中位，⛔ 不是「全資料」的精確中位。**
   規格寫「全資料該量的中位數」—— 全枚舉是 N×51 ≈ 5×10⁷ 對，所以用同一把抽法抽 5 萬筆估。
   定標流固定 seed、不吃 `LACOT_SEED` ⇒ 可重現且跨 seed 一致。要改：`LACOT_LO_ADV_CALIB_N`。
5. **advantage 的 `v` 只認「淨位移／步」，⛔ 不認繞路的合理性。**
   ⇒ 必須繞牆的段會被系統性地打低權重（它的淨位移天生小）。這是一個**選擇**不是唯一解；
   要更貼的話得換成「沿 BFS 最短路的進度／步數」，但那要 E 圖、成本高一級。
   ⚠️ 梯級一判負時，先確認負的是「advantage 這個想法」還是「v 這個定義」。
6. **`k` 的 clamp 讓靠近軌跡結尾的段實際 k 變小。**
   `t+k` 夾到 `traj_end` ⇒ 那些樣本的實際 k < KMIN。這是照 `make_batch` F6 的選擇
   （clamp 不重抽 ⇒ 起點分布不歪）。⚠️ 代價是 k 的邊際分布在低端多一個小尖峰。
   ⭐ advantage 的 v 用的是**實際步數**（`tgt−r`），⛔ 不是名目 k ⇒ 這一格沒有被算錯。
7. **pointmaze ＋ `FINISH_R>0` ＋ `SUB_POLICY=lo` 時，終局那幾步走的是 bc 頭不是 π_lo**
   （分段 policy 的終局分支照舊）。ant 上 `takeover=off` ⇒ 不存在這件事，
   ⛔ 但如果之後拿 pointmaze 跑 lo 對照，這一格要先想清楚。
8. **⛔ GPU golden gate 沒跑。** 本單範圍止於 pointmaze CPU 零差；A/A/B 三跑那格由ルナ親自排
   （harness 已入庫：`slurm/goldab/gold_ab_wrap.sh`，`BRANCH_ROOT=<lo-v1 的 worktree>`）。
9. **沒有覆蓋到的配置**：`FSQ / VQ / DEC_START=hard / INTENT=anchor|residual / BC_OWN /
   CONT_TRAIN / SUBGOAL=latent|conf|bfs|ebfs / U_SOURCE=oracle / BOOT_GEN / AMP / COMPILE`。
   π_lo 那段的 AMP 三件套是照 `BC_OWN` 抄的、逐行對過，但**沒有跑過** ⇒ 只算「讀過」，不算「驗過」。
10. **`CONT_TRAIN` 沒有存 `opt_lo` 的狀態**（只存 `opt2` / `opt_bc`）⇒ 續訓時 π_lo 的 Adam m/v
    從零起算。⚠️ 現在還沒有人續訓 π_lo，但真的要做的時候這一格要補，⛔ 別靜默接受。
11. **per-leg 的「集末未結算」不分「成功收尾」與「MAXH 用完」。** 兩者都只計數。
    要分得開的話得把 episode 的結局傳進 `on_start`，那要動 `rollout()`／`dev_eval` 的介面 ⇒ 沒做。
