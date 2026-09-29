VERDICT: PASS-with-notes

判的是：v3 實驗 2 接線批（refine compose 接進三個真入口、解對抗官 B1/B2）的接線與邊界聲稱。不判科學有效性，也不審 u 線本身。實驗 2 的三 seed 學習格本來就上不了機：Aω 沒過（EXP1-VERDICT），`REFINE_TRAINING_SERVICES=None`，R>0 會拒啟。下面「上機前必改」指的是：有合格 provider 之後、真正開格之前，這幾件要先補。

檢察員：S2（Claude Opus，跟實作方 GPT astra 不同家族）。唯讀。只做了 CPU 重跑，沒跑 GPU，沒送 sbatch。
受檢 bytes（與 evidence-exp2/source-hashes.log 一致，檢察期間沒變）：scratch `f3b569f7…`、model.py `1f6f8b9d…`、refine_objective `4f4c196e…`、refine_training `fedcd61e…`、test_refine_wiring_v3 `e5a85e58…`。
重跑產物放在 `/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/37e41886-10cf-41d0-9aae-34351f439be3/scratchpad/s2exp2/`（下稱 `$SP`）。

---

## 1. 兩個阻斷走的是真路徑　PASS（附一個設定相依的缺口，見 M2）

**B1（輸入來源）**
- 親讀 `refine_objective.refinement_terms`：簽名裡沒有 actions。`clean` 只拿來做 shape/dtype 驗證和 `new_zeros`。迭代的起點是 `u = sampled.detach()`，之後每輪 `u = un`，走的是 student 自己的遞迴。teacher 包在 `no_grad` 裡，吃 `(cond.detach(), u.detach())`。
- 親讀 `refine_training.compose`：
  - `sampled = adapter.sample().detach()`。
  - R0 在 sample 之前就 `return nf + anchor`。
  - quality closure 由 `services.quality_factory(sampled, cond)` 綁在 sample 上，不經過 actions。
- 取樣來源：兩個 actor 的 adapter 是 `self.flow.sample(b, cond)`；scratch 是 `sample_plan(B, cond, anc, s, g)`，在 `INTENT_GUID_W=0` 時就等於 `flow.sample(n, flow_cond(cond, anc))`。kernel 確實不讀資料 actions，也沒有 clean±noise 這條路。
- 證據：
  - Trace 逐位元比對 `states[0]` 等於當次 flow sample。每步只抽一次，而且連續兩次的 sample 不同。
  - actions 置換後 quality 梯度逐位元不變，anchor 有變（正對照成立）。
  - 我重跑 12/12 PASS（`$SP/wiring-rerun.log`）。
- 缺口（設定相依，見 M2）：scratch 開 intent 時，sample 的條件 `anc = intent_anchors_of(traj)` 是從資料的未來軌跡算出來的。
  - hindsight 模式：錨點就是實際走過的格子序列。
  - route 模式：錨點是從 `traj[0]` BFS 到 `traj[-1]`（scratch:538-558）。終點用的是資料未來的終點，不是 goal。
  - `cond` 裡的 ix 也被 INTENT_DROP 丟過。
  - 部署時錨點來自 `_intent_anchor_eval(s,g)`（route 到 goal）。這不符合 DESIGN (c) 寫的「部署相同 temperature／intent／quantization」。錨點越密，u0 越貼近資料計畫，B1 那個「輸入落在資料附近 ⇒ identity 近最優」的病就會從側門回來。

**B2（主線繞過）**
- 親讀 scratch :1690-1768。舊的 inline 三輪 `ahead(cond, us[r+1]) vs act` 已經拿掉。`_stage2_loop` 在 :1766 呼叫 `refine_training.compose`，傳入 `refine, cond, _u_head, rounds=_gstp%4, lam_cons=.1`。
- `_et_nf`、density 的原位求值（:1741，在 COND_DROP 的 `torch.rand` 之前）、`_ca/COND_DROP`、`_REAL_W/_wmse`、`flow_cond(cond,anc)` 都保留了。
- Trace 記到的 caller 是 `_stage2_loop`。我自己的 mutation replay 也獨立確認，compose 是從 `_stage2_loop` 呼叫進來的（見第 3 點）。

## 2. R0 逐位元迴歸：證據成立，沒有恆真格　PASS

- 照報告的命令，用 `git show 5b256d4:…` 當基線重跑，16/16 BITWISE PASS（`$SP/r0-diff-head-rerun.log`）。
- 基線是否等價：
  - 把 own-scratch.diff 反向套回現檔，還原出來的檔 sha 是 `276c68fb…`，跟 /tmp 開工快照、差分 log 裡記的 pre-edit SHA 完全一致。
  - 五個受測函式（`_stage2_loop/_flow_nll/sample_plan/_amp_step/_amp_update`）在 HEAD 和 pre-edit 之間 AST 原文相同。所以用 HEAD 當基線是有效的。
- 恆真檢查。我另外寫了儀器化版 `$SP/r0_diff_instrumented.py`：
  - 每格實際比到 1 個 backward loss、35 或 37 組參數與 grad，外加 RNG。沒有空比對。
  - 負對照一：在舊源的 total 加上 `1e-7*l_nf`。
  - 負對照二：在舊源多插一個 `torch.rand(1)`。
  - 兩個負對照在 16 格裡全部被抓到（32/32）。
- 小弱點：原腳本的 `zip` 沒有檢查長度。要是某一側的 backwards 是空的，會空轉通過。我已實測現況下沒有空轉，但建議補上 `assert len(a)==len(b)>0`。
- 覆蓋範圍：只驗 `LEARNED_REFINE=0`。這是對的，因為 R>0 的語意本來就是要改的。AMP scaler、GRPO/LO/BC_OWN、intent residual/anchor、INTENT_GUID_W 都不在 16 格內。讀碼判斷這些路徑沒改或數值等價（`_amp_step` 改走 hook，數值相同）。

## 3. 六個 mutation 的兩態　PASS

- 機制：考場用 `patch.object(module, symbol, mutant)` 換掉模組屬性。production 在呼叫當下才去解析這個屬性：`_stage2_loop` 走 `refine_training.compose`，compose 走 `objective.refinement_terms`。所以是真的換掉 production callable，不是旁路。mutant 是拿 production 源碼做字串替換、在模組 globals 的副本裡編譯出來的。
- 重跑：6 個都是 FAIL→RESTORED PASS，witness 訊息跟 wiring.log 對得上。
- 我自己重放兩個（`$SP/mutation_replay.py`）。witness 是自己寫的，不用考場的 Trace；jitter=0，也就是 production 的主臂：
  - truncate_bptt：mutant 從 compose 被呼叫 1 次，9 個梯度輸入全部對不上獨立算的 full-BPTT 期望值（|dLq/dcond| 3.6e-2 vs 8.3e-2）。還原後 0 個不符。
  - detach_cond：mutant 被呼叫 1 次，只有 cond 那一格對不上（0 vs 8.29e-2）。還原後一致。

## 4. total 代數　PASS

- 兩個 actor 直接回傳 compose 的 total（考場用 `assertIs` 驗）。
- scratch :1815 是 `total = base_total + (0.0*l_bc if BC_INDEP else l_bc)`，之後只有 DIV/GRPO 的外層加項。backward 在 BC_INDEP 時吃 `total + l_bc`，否則吃 `total`。我讀了 :1815-1955，沒有地方再把 l_nf/l_anchor 加一次。GRPO 的 β 定標用的 `l_nf` 就是同一個 `_nf_value` tensor。
- `grad(total)-.25*grad(exposure)==grad(anchor)` 這條斷言有方向性（`$SP/head_identity_check.py`）：
  - 非空：head 上的 |grad exposure| = 3.7e4。
  - mutant 把 λexp 改成 .5：斷言 FAIL。
  - mutant 讓 exposure 吃沒 detach 的 u：這條恆等式本身抓不到，但同一個測試的「exposure→refine 梯度=0」斷言會抓到（實測 42.5≠0）。
- 考場缺口（低優先）：「R>0 時 anchor 改算在 sample 上」這種 mutant，head 恆等式和 scratch mask 測試都抓不到（`head_values[0]` 會跟著變）。production 我讀過，是 `adapter.features(clean.detach())`，寫對了，只是沒有 witness 守著。

## 5. 邊界聲稱　PASS（附一個跨線影響）

- **u 線 bytes 未動：驗過，而且比 ownership log 更強。** 現檔 = HEAD + u 線 hunks + 本批 hunks：
  - HEAD→pre-edit 的 diff 只碰 `ORACLE_*`、`oracle_draw`、`policy_chunk`、`rollout`。
  - pre-edit→現檔 就是 own-scratch.diff（sha 對得上）。
  - 兩組 hunk 不重疊。
- **跨線影響（不是本批 code 的 bug，但 lead 要知道）：**
  - u 線 `run.py` 的 `code_hashes()/flagoff_contract/gate` 釘的是整支 scratch 的 sha256，smoke/mutant preflight 記的是 `276c68fb…`。本批把整檔 hash 改成了 `f3b569f7…`。
  - 叢集 job 是直接從這個 NFS 工作樹跑的（exp1.sbatch 與 u 線都 `cd /home/cymaxwelllee/Projects/lacot`）。
  - 正在跑的 u 線 main 33906_0/1 約 14:05 開跑，早於本批動檔（14:41–15:02），執行的是 pre-edit 版，前後一致。這是從開跑時間推的，它們的 preflight 在 jasmine /archive，我沒讀到。
  - 但 u 線之後任何重投都會被它自己的 gate 以「stale／flag-off proof mismatch」拒絕（大聲失敗）。
  - 本批要 commit 或開分支之前，必須先跟 u 線排好順序。
- **mod-oracle 未改：** `test_refine_quality_v3` 24/24 重跑 PASS。refine_quality、refine_models、測試檔的 mtime 分別是 12:33、12:44、12:45，都早於開工快照 14:41。rework 之後沒有留 hash 錨點（VERDICT-ORACLE-S2 記的 hash 比 rework 早），所以「bytes 未改」用 hash 驗不到，標 UNVERIFIED；mtime 看起來一致。
- **原 R0 測試檔：** `tests_repair/test_model_zero_rounds.py` 跟 HEAD（ad45d67）比沒有差異，重跑 1/1 PASS。

## 6. 新引入的洞

- **resume／ckpt（:2088、:3845）：沒有靜默載入舊檔，PASS。**
  - `restore()` 先 `authorize`；metadata 缺了或對不上就 raise「migration required」。考場涵蓋 None 和 legacy dict 兩種。
  - CONT_TRAIN＋LEARNED_REFINE 遇到舊 ckpt 會大聲拒絕。
  - teacher 就是 `refine_ema`，不在 `f_mods` 裡，所以續訓時 blanket `.train()` 不會把它解凍。`_stage2_loop` 開頭還會再做一次 `assert_frozen`。
  - 限制（N3）：metadata 沒寫進 λcons=.1、λexp=.25、深度輪換規則；data rng、torch 全域 RNG、GradScaler scale 都沒存。CONT_TRAIN 不是逐位元續訓，報告已自承。
- **EMA 只在 opt2 真的 step 時更新：兩態都成立，PASS。**
  - 考場：假 scaler 打在 scratch 真正的呼叫點，外加真的 CPU GradScaler。
  - 我補驗（`$SP/fused_skip_check.py`）：hook 判別法對 non-fused Adam 正確，opt2 是 `torch.optim.Adam` 預設值，也就是 non-fused。
  - 但 fused Adam（`_step_supports_amp_scaling=True`）在 inf 被 skip 時，`optimizer_step` 仍回 True（權重其實沒動），EMA 會在假 step 上更新。這是潛在陷阱（N4）。
- **本批新洞：CONS 預設翻轉＋沒有 objective 版本 ⇒ 結果檔 provenance 會混（M1）。**
  - `CONS` 預設（:725）在 `LEARNED_REFINE=1`（也就是預設值，:154）時從 "self" 變成 "ema"。
  - CONS 會進三個地方：檔名 tag `{env}_{CONS}_K…`（:3720）、`ckpt_{tag}.pt`、out JSON 的 `cons`（:3180）。
  - 後果 (a)：預設環境下對舊 LEARNED_REFINE=1／CONS=self 的 ckpt 做 LOAD_CKPT 評估，現在會寫出 `_ema_` 檔名、`cons:"ema"`。這是標錯，還可能跟真正 ema ckpt 的舊結果撞名。也違反了 scratch 自己 :3528-3531 的規則：「預設跑出來的檔名不變」。
  - 後果 (b)：scratch 的 tag 和 out JSON 都沒有 objective 版本。v3 R>0 的跑檔如果其他旋鈕跟舊 LEARNED_REFINE=1／CONS=ema 的跑檔相同，會產生同名的 `rollout_*.json`／`ckpt_*.pt`。非續訓的存檔沒有 exists 防線，會靜默覆蓋，或讓 v2、v3 兩種目標的結果混在同一個名字底下。
  - R0 差分沒有涵蓋到這件事，因為 fixture 直接注入 `CONS='ema'`，不走模組層的預設值。
  - u 線顯式設了 `LACOT_CONS=self`、`LEARNED_REFINE=0`，不受影響。

---

## 上機前必改清單（實驗 2 三 seed 學習格）

- **M1（必改）provenance。**
  - v3 objective 要進檔名 `_extra` 和 out JSON，例如 `objective_version=generated-plan-v3` 與 `_objv3`。只在 v3 R>0 時加，這樣預設檔名才不會變。
  - CONS 預設：LOAD_CKPT 時從 ckpt 的 `cfg["CONS"]` 讀，或只在 v3 訓練時預設 ema。
  - 實驗 2 用專屬的 `LACOT_OUT_DIR`。
  - 驗法：純函式 `_tag_extra` 考場＋一個「v2 與 v3 同旋鈕不同名」的測試。
- **M2（必裁）intent 錨點。** 二選一並寫進 launch card：
  - 實驗 2 關 intent；或
  - 訓練取樣改用部署來源的錨點（route s→goal，不用 traj[-1]，也不吃 INTENT_DROP 過的 ix）。
  - 另外量一次 u0 到 e_target 的距離，intent on/off 各一次，當作「u0 沒有被資料拉近」的證據。
  - batch binder 收到的 `anchors` 只准拿去 decode（`_intent_inv`），不准當 quality 目標。這條要寫進未來合格 service 的契約。
- **M3（必做）用實驗 2 的確切 env 跑一次 1-step GPU smoke。**
  - 如果開 AMP=1：驗 sampled 與 clean 同 dtype 不會 raise。CUDA fp16 下這件事 UNVERIFIED；讀碼判斷 e_target 以 LayerNorm 收尾是 fp32、FSQ-z 兩端都過 `fsq.up`，應該一致。
  - 目標節點 venv 跑一次 `import lacot.model`。jasmine 已驗 Python 3.11.15／torch 2.6.0，其他節點沒驗。
- **M4（順序）** 本批 commit 或開分支前，先跟 u 線對齊。u 線重投會因整檔 hash 釘而被拒（見第 5 點）。

## notes（不擋上機，建議修）

- N1：原 `r0_differential.py` 補上 zip 長度斷言。考場補「R>0 時 anchor 必須用 clean」的 witness，例如 R3 時對 anchor 的 head 輸入做 hook 比對 `_u_head`。
- N2：LEARNED_REFINE 預設是 1，舊的預設設定訓練會在 stage 1 跑完之後才在 stage 2 拒啟，白燒 stage 1 的 GPU。建議把資格檢查移到 stage 1 之前（fail fast）。
- N3：若實驗 2 或 relay 會用到 CONT_TRAIN，要補存 data rng、torch RNG、GradScaler 狀態，並把 λ／輪換寫進 metadata。
- N4：在 `optimizer_step` 或 `_stage2_loop` 前檢加一條防線：`getattr(opt, "_step_supports_amp_scaling", False)` 為真就拒絕。或改用 found_inf 判別。
- N5：兩個 actor 的 `lam_cons` 預設仍是 .5，設計值是 .1。實驗 2 走 scratch、吃 .1，不受影響；但走 actor 的呼叫端靠預設值時，會安靜地拿到 .5。

## 自我懷疑

- 沒跑 GPU。AMP 的 dtype、實際的 ROCm/CUDA 行為都是讀碼推的（M3）。
- 我的 mutation 重放和 R0 儀器化用的是考場同一個 `scratch_fixture`（CPU globals），fixture 層的盲點我也會一起繼承。
- M2 的嚴重程度沒有量過。u0 被 hindsight／route 錨拉近資料的程度、會不會讓 refiner 重回 identity，要靠 M2 那次量測回答。我只判它違反了「部署相同 intent」這條契約。
- u 線 main 執行的是 pre-edit 版，是從開跑時間推的，沒讀到 jasmine /archive 上的 preflight。
- oracle 源碼未改是看 mtime，沒有 hash 錨點。refine_quality 內部（`authorize_training`／QualityService 語意）不在本次範圍，我當成已知。
- M1 的撞名機率（舊 ema 系列跟實驗 2 旋鈕完全重合）沒有去逐檔清點。標錯是確定的，覆蓋與否要看旋鈕。

STATUS: DONE
