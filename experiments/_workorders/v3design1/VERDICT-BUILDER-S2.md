# VERDICT-BUILDER-S2：Wφ quality 服務 builder 檢察（S2、Claude opus；實作方 GPT astra）

在證明什麼：本件接上 Wφ 之後，R>0 在現狀（Aω .764 未過、generated gates 缺）下仍然跑不起來；flag-off 逐位元不變；座標往返、RNG、metadata 正確。
判準：任何 env 組合能讓 R>0 走到第一個 opt2 step 即 FAIL；其餘逐點要有兩態證據或親讀。

VERDICT: PASS-with-notes

受檢 bytes（檢察期間未變）：refine_service_builder `c1d63cdd…`、test_refine_service_builder `0c24d3d2…`、scratch `b7d3a0d9…`。
參照未動：refine_quality `c59e78cd…`、refine_training `fedcd61e…`、refine_objective `4f4c196e…`、model.py `1f6f8b9d…`。後三者與 `evidence-exp2/source-hashes.log` 一致，是獨立錨；mtime 全早於本件開工 23:39。

本檢察的 CPU 重跑（皆 `CUDA_VISIBLE_DEVICES=''`；logs 在 `/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/37e41886-10cf-41d0-9aae-34351f439be3/scratchpad/`）：

- builder 13/13 PASS，數字逐位重現（u grad .494005859、.002749476）。
- wiring 16/16、oracle 24/24、R0 model 1/1 PASS，合計 54，與報告一致。
- 獨立探針 `s2_probe.py`：45 格，44 PASS。唯一 FAIL 是 1h，屬我的 harness 預期寫錯：我預期 ValueError，實際是 TypeError（`build_training_services` 沒有 `exposure` 參數），正是要的 fail-closed。

## 1. 授權鏈（靈魂格）：PASS，沒有 env 繞得過

親讀路徑如下：

- hook 只做 `build_from_env` → `build_training_services`。後者寫死 `exposure=None`，函式簽名也不收 exposure（探針 1h：TypeError）。
- `QualityTrainingServices` 沒覆寫 `authorize`、`restore`、`for_batch`。
- 起跑檢查有兩道：
  - `_stage2_loop` 開頭：`n_steps and LEARNED_REFINE` 時呼叫 `require_services`。
  - 每個 R>0 step：`compose` 再呼叫一次 `require_services`。
- `authorize` 先跑原封未動的 `authorize_training`（三項＋generated＋source＋digest），再檢查 `exposure is None`，兩層串聯。
- 續訓（CONT_TRAIN）路徑的 `restore()` 也先 `authorize`。
- scratch 內沒有任何地方設定 `.exposure`。

會影響這條路徑的 env 只有 CKPT、LEARNED_REFINE、STEPS2、CONT_TRAIN、LOAD_CKPT、CONS、INTENT。逐一推過，沒有一個能讓 exposure 非 None：

- LEARNED_REFINE=0：rounds 恆為 0，只有 R0。
- 其餘組合：全部停在 require_services。

兩態實證：我偽造了一顆 production 版 toy。把 source 改標成 offline_dataset、重算 joint fingerprint 與 readings digest，其餘不動。

| 探針 | 狀態 | 結果 |
|---|---|---|
| 1a | production 路徑正態 | 能載入，所以拒絕有鑑別力，不是恆拒 |
| 1b | EXP1 形狀的 readings | 被**原** gate 擋下，訊息 `oracle qualification failed: generated validity…; generated nominal_plus_alternative_coverage…; A action NMSE >.1` |
| 1c/1d | readings 偽造成全過 | quality 層放行；R>0 **只**剩 `exposure unavailable; R>0 refused` 在擋 |
| 1f/1k/1l/1m | 真 hook 用 env 建服務，真 `_stage2_loop(1)` | 開頭就拒絕；0 次 backward，權重逐位元不變 |
| 1n | restore（續訓） | 拒絕 |

diagnostic_cpu 逃生門：確認只開給 CPU fixture，而且永不授權。

- 1g：`build_from_env` 拒收 diagnostic_cpu。
- 1j：非 CPU device 在任何 IO 之前就拒絕。
- 1i：diagnostic＋全過的 offline 檔，`authorize_training` 恆為 `diagnostic/unqualified`。
- 它唯一放寬的是「toy 可以載入」，不放寬授權。

## 2. flag-off 慣性：PASS

- 位元組層（最強證據）：
  - 現檔剝掉 BEGIN/END 區塊之後，sha 等於 `cffa0491…`。
  - 也和本件開工快照 `/tmp/lacot-pilot-builder/scratch-before.py`（23:39:55）逐位元組 `cmp` 相等。
- hook 條件 `LEARNED_REFINE and (STEPS2 or CONT_STEPS) and key in os.environ` 無副作用。不設 env 時整支程式語義等同快照。
- 快照 vs HEAD 的差異，關鍵字全部歸屬 u 線（ORACLE_*、oracle_draw）或 exp2（refine_training／CONS／OBJECTIVE_VERSION）。builder 字樣只出現在 hook 內。
- R0 差分重跑 PASS（內含於 13 考場）。但見 §3：那 16 格對 hook 本身是恆真的，真正的證據是上面的位元組比對。

## 3. 考場兩態＋恆真斷言：PASS-with-notes

我另外做了重放，全部是兩態（拒絕 → 還原後通過）：

- 壞檔 9 種：
  - obs_only 權重、controller 權重、threshold、state_norm：原 fingerprint 擋下。
  - hidden、t_cap 欄位：strict load 擋下。
  - readings 未同步 digest、episode_split：各自的 guard 擋下。
  - NaN：JSON allow_nan 擋下。
- toy 走 production：`load_quality_teacher` 與 `build_from_env` 都拒絕，訊息為 `toy/unverified … production`。
- 反態：同一顆 toy 在 diagnostic_cpu 下能載入。

恆真掃描：

- `test_flag_off_bitwise_full_scratch_loop_16_cells` 的 16 格對 hook 是**結構性恆真**：
  - `load_scratch` 只 exec 五個函式定義，hook 是模組層的 `if`。
  - learned=False 時，第一個 conjunct 就短路。
  - 所以它只守住「那五個函式沒被改」。結尾「learned=True＋缺 env → 拒絕」那段是有牙的。
- `test_revised_pi_validity…` 的尾段 `authorize → diagnostic/unqualified`，對任何 diagnostic 服務都恆成立。有意義的只有載入那段。
- NaN 與「readings」兩個 case 都擋在 digest／JSON 層。`_check_readings` 裡對 A NMSE 的 NaN guard 實際上走不到，也沒有測到。結論（NaN 必拒）不受影響。
- 有牙的：RNG（見 §5 的突變實證）、非單位 MU/SD 下的 cost 等值、frozen 翻轉、輸入擾動、全 invalid。

## 4. 座標／正規化往返：PASS

親讀 `bind` 與 `decode`：

- 入口轉 raw：`raw = s*SD+MU`、`raw_goal = g[:, :2]*SD[:2]+MU[:2]`。
- 每次 decode：`(current−MU)/SD` → `_dec(_q(u), ·)` → `_intent_inv` → `*SD[:2]+MU[:2]`。
- 與 scratch 的 `traj` 正規化（`MU_XY=mu[:2]`、`SD_XY=sd[:2]`，都含 +1e-6）同源。
- W 與 A 都吃 raw，各用自己的 normalizer。兩套統計不必相同，設計正確。

原考場 `test_batch_coordinates…` 釘住的範圍：非單位 stats 下的 path／goal 反正規化（cost 等值），以及**第一次** decode 的 state 正規化。它沒有釘到的是：真 `_dec` 搭配非單位 stats，以及後續 chunk。

我用探針 4 補上：真 scratch `_dec`（hard start）、MU=[2,−3,1,4]、SD=[2,3,4,5]、h16（3 members × 4 chunks = 12 次呼叫）。

| 格 | 驗什麼 | 結果 |
|---|---|---|
| 4b | 每次 `_dec` 的輸入 ==(current_raw−MU)/SD | max diff 0 |
| 4c | 每次 raw path[0]== current raw xy（往返閉合） | max diff 1.2e-7 |
| 4d | 後續 chunk 用的是 W 新預測的 state | 成立 |
| 4e | 第一次 current == raw state | 成立 |

## 5. RNG 隔離：PASS

- 5a：載入不消耗 CPU RNG。
- 5b：突變實證。把 `fork_rng` 換成 nullcontext 後，RNG **會**被消耗，所以原測試有鑑別力。
- 5c：quality forward 也不消耗 CPU RNG。
- 模型在 CPU 上建構，全檔沒有 `set_default_device`，CUDA RNG 不會被碰到。GPU 端屬 UNVERIFIED（本機沒有 CUDA），只是由構造推得。

## 6. metadata：PASS

偽造的 production 版（Aω 用 toy 的 .8197）結果：

- `training_blocks` 恰好是 [generated validity, generated coverage, A NMSE >.1] 三條，與 EXP1-VERDICT 的「Aω 未過＋generated 兩 gate 缺」一致，不多不少。
- `exposure_available=False` 另列，屬 DESIGN 接線缺口，不是 EXP1 的資格項。
- `world_qualification` 只驗 h=4（服務用的 horizon）。EXP1 三個 horizon 全過，沒有矛盾。
- 其他欄位齊全：fingerprint、split、dataset、readings_digest、絕對路徑、source、diagnostic。

UNVERIFIED：真 exp1 檔不在本機。split 前綴 `9aec9ac9`，以及真 readings 所得的 blockers，只能由 F5 對。

## pilot 上機前必改清單

F5 的「載入＋拒啟」smoke 可以照草稿跑。以下是可訓練 pilot 之前必須處理的：

1. **〔授權完整性，exposure 接線落地前必改〕readings 沒有出處保護。**
   - `readings_digest` 從檔案讀出後只能驗自洽。joint fingerprint 不涵蓋 readings（探針 1e：釘住真 fingerprint，照樣放行只改 readings 的偽造）。
   - hook 也完全沒傳 `expected_fingerprint`。
   - 現在只有 exposure=None 在擋（1c/1d）。exposure 一旦接上，唯一的閘門就變成可偽造的數字。
   - 修法：env 路徑必須帶釘值，由 builder 驗證後才載入。兩種做法擇一：
     - 整檔 sha256。
     - expected fingerprint＋expected readings_digest，值取自 jasmine 真檔。
   - 在修好之前，F5 要人工比對 log 裡的 fingerprint、readings_digest、split hash 與真檔。
2. **〔續訓正確性，任何 CONT_TRAIN R>0 之前必改〕hook 在 LOAD_CKPT 區塊之前建服務。**
   - ckpt 帶 VQ 而 env 沒開 VQ 時，`vq` 會在 hook 之後被重新綁定成新的 TokenVQ（:2091-2096；可訓練、`VQ_SOFT` 也跟著改）。
   - decoder lambda 是晚綁定，會用到新的 vq；但 `frozen_modules` 是 hook 當下的快照，不含它。每次呼叫的 frozen 檢查因此漏掉 codebook。
   - 修法：在載入之後重建服務，或在 `_stage2_loop` 入口從現行 globals 重取 frozen 集。
   - 草稿 CONT_TRAIN=0，這次不受影響。
3. 〔建議〕把探針 4b/4c 的不變式（真 `_dec`、非單位 stats、每個 chunk）收進考場，當往返的回歸守門。

跨線提醒（不是本件 bug）：hook 又改了一次整檔 hash。u 線的 gate 釘的是 `276c68fb…`，自 exp2 起就已經 stale。重投會大聲拒絕，commit 順序照 VERDICT-EXP2-S2 的 M4 協調。

## 自我懷疑

- production 正態只用「source 改標的 toy」證過：4 維 state，有 velocity 子空間。真 pointmaze 是 2 維、只有 xy 子空間，這條分支是讀碼推的，沒有實跑。
- 真檔的 blockers、split hash、GPU／AMP 路徑都是 UNVERIFIED。
- 開工快照是 builder 自己產的。post-rework 的 scratch 沒有獨立 hash 錨（VERDICT-EXP2-S2 已記）。所以「hook 是唯一改動」這個結論，是「相對快照逐位元組」加「HEAD→快照差異的關鍵字歸屬」，不是對 rework 定稿的逐位元組比對。
- scratch 沒有端到端跑（本機沒有 OGBench 資料）。hook 是用 exec 在 fixture 命名空間裡跑真原始碼，`_stage2_loop` 是真的 AST。
- §必改 2 只靠讀碼推得，沒有造 VQ ckpt 實測。snap 對 codebook 是否真的有梯度，我也沒驗。

STATUS: DONE
