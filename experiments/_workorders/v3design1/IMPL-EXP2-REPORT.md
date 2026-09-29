在做什麼：DESIGN-v3.md (c) 檔案表的 mod-actor/objective 批——新增 lacot/refine_objective.py＋lacot/refine_training.py（唯一 compose）＋改造 lacot/model.py 兩處 losses_given 與 experiments/scratch_lacot_rollout.py 主線 inline／total。

STATUS: DONE

此 STATUS 僅指本次「實作＋CPU 接線考場」。未跑正式訓練、GPU、sbatch、環境考試；未作科學有效性判決、未 commit、未安裝套件、未刪檔。EXP1-VERDICT 的 Aω 未過照案保留：正式 R>0 沒有預設老師，明確拒啟；實驗 2 的三 seed 學習格及實驗 4 未執行。本次沒有開啟 selector／depth／timing gate，沒有拿接線 PASS 代替救援資格。

**交付與兩個 production 阻斷**

- B1 輸入來源病：三入口的 R>0 均由當前 flow freshly sample，compose 停 u0 梯度後才建立 sample-bound quality closure；kernel 不讀資料 actions，不以 clean±noise 當輸入。每輪用自己的上一輪 state，主臂無 jitter；可選輔臂為 sample＋獨立 generator noise，逐輪 quality 權重遵循 v2 的 .25／1.25。
- B2 主線繞過：scratch 完整 `_stage2_loop` 現在呼叫同一個 `refine_training.compose`；兩個 `losses_given` 同步接線。資料 action 僅留在 clean anchor／原 BC 等外層，舊 sample→本批 act 項移除，`l_act_refine=0` 保留相容 log key。
- 唯一總式：`nf + anchor + quality + lam_cons*consistency + .25*exposure`。scratch 使用 `.1` consistency；兩 actor 保留原 positional args／`.5` 預設值，可顯式傳 `.1`。這是舊 API 的保留，不是兩份 objective。
- 復用 v2 Adapter 的 density／features／anchor／sample／exposure 分工與 refinement_terms 語意；TrainingServices 收攏 teacher、quality_factory、exposure 及資格／生命週期。v3 quality 接現有 `QualityReport`，個別 invalid 只進 valid mean，coverage 另報；全 invalid、fingerprint 不符、未 frozen/eval 或無有效 action labels 都拒絕。
- scratch 保留 `_et_nf` 與原 density normalization、`_u_head` clean、`_ca/COND_DROP`、`_REAL_W/_wmse`、`flow_cond(cond,anc)`。density closure 在原位置求值，再交 compose，保住 flow dropout 和 COND_DROP 的 RNG 次序。sample 使用既有 `sample_plan`，沒有另造 decoder／sampling 方言。
- same-plan exposure 是資格未備即拒啟的接點。`REFINE_TRAINING_SERVICES=None`；沒有 fake production teacher。未來合格服務須用 batch binder 接本批 state／goal／anchors，沿原 `_dec/_intent_inv` 與座標轉換，並提供 `exposure_loss` 同型的 `(loss, validity_info)`。目前不提供繞過 EXP1 的 builder。
- refiner EMA 使用 frozen/eval teacher；僅 opt2 真正執行 step 才更新參數，buffers 同步複製。以公開 optimizer post-step hook 分辨各 optimizer，而非用全域 scale 猜哪個 optimizer 跳步。保存 objective version、joint quality/action artifact fingerprint、Rtrain、EMA config／state、嘗試／成功步數與輔臂 RNG；缺 metadata／不符版本明確要求 migration。原 inference 參數結構不變。

回鍋 M3 記錄：GPU 1-step smoke 由 F5 鏈執行段做。
回鍋 M4 記錄（VERDICT-EXP2-S2 原文）：「**M4（順序）** 本批 commit 或開分支前，先跟 u 線對齊。u 線重投會因整檔 hash 釘而被拒（見第 5 點）。」

**三個真入口 trace**

考場使用真正 LaCoTActor／LaCoTActorState 的 `losses_given`、真正 TARFlow（縮小寬度）與真正 RefineOperator。scratch 從現檔抽取並編譯完整 `_stage2_loop`／`sample_plan`／NLL／AMP helper 函式，未抽換 loss branch；dataset、環境 globals、head 和 oracle 使用明標 CPU fixtures。沒有 import／執行整支 rollout 的環境初始化。三入口各做 4 個微型 CPU optimizer steps，深度 0/1/2/3；這些步數只是接線測試。

| 真入口 | compose trace | flow sample calls | states 長度 | Jacobian witness |
|---|---|---|---|---|
| image | `losses_given` ×4 | R0=0；R1/2/3 各1 | 0/2/3/4 | own recurrence、full BPTT、cond 非零 |
| state | `losses_given` ×4 | R0=0；R1/2/3 各1 | 0/2/3/4 | 同上 |
| scratch | `_stage2_loop` ×4 | R0=0；R1/2/3 各1 | 0/2/3/4 | 同上，包含真正外層 backward／step |

每次 positive-round state[0] 與當次 real flow sample 逐位元相同，連續 samples 不同；逐輪輸出與獨立 recurrence 數值／Jacobian 對照。image/state 返回的 total 就是 compose total；scratch backward input 另與 base_total＋原 BC/div 權重逐位元對照。

三入口 actions_data batch permutation：refiner quality 梯度逐位元不变，anchor 明確改變。head 的 `grad(total)-.25*grad(exposure)==grad(anchor)` 通過；exposure 對 refiner 梯度為零，clean／sample／noise／teacher 不收梯度，student／cond 留完整圖。

**六 mutation 的兩態證據**

以 production 函式 source 編譯 mutation，暫時替換實際 module callable，再跑真正 `_stage2_loop→compose→objective`；不是旁路 fake compose。每個 mutant 必由 AssertionError 語意 witness 抓到，import／shape error 不算；還原 callable 後同一 witness 必過。完整原始輸出在 [wiring.log](evidence-exp2/wiring.log)。

| mutation | 注入後 | 還原後 |
|---|---|---|
| total 缺 quality | FAIL：base_total 各項一次的代數等式 | PASS |
| sample 錯接 clean target | FAIL：state[0] ≠ fresh flow sample | PASS |
| 忽略 rounds，固定一輪 | FAIL：R3 的 states 長度不符 | PASS |
| 零 noise | FAIL：真輔臂 quality 數值不符 | PASS |
| 截斷 BPTT | FAIL：refiner parameter Jacobian input 0 | PASS |
| detach cond | FAIL：cond Jacobian input 8 | PASS |

零 noise 特別說明：fixture 啟用專用 generator 的 `.35` 測試擾動；compose 真正呼叫 `services.noise_for(sampled)`。mutant 在這個位置把抽出的 noise 置零；期望值用相同 generator state 重建，經真正 RefineOperator 的輔臂計算。`.35` 只是有辨識力的接線 fixture，production 預設 jitter=0，沒有把零 jitter 主臂誤判為 mutant。

**CPU 驗證結果與重跑**

環境：現有 `.venv`，PyTorch 2.6.0+cu124，所有 tensor／GradScaler 測試在 CPU。沒有使用 CUDA backend。

| 考場 | 結果 | 原始證據 |
|---|---|---|
| 新 wiring tests | 12 tests PASS；含六 mutant 的 FAIL→RESTORED PASS | [wiring.log](evidence-exp2/wiring.log) |
| 原 R0 考場 | 1 test PASS；原檔 bytes 未改 | [r0-original.log](evidence-exp2/r0-original.log) |
| scratch 開工前快照 vs 新 loop | 16/16 loss／grad／更新後權重／torch RNG 逐位元 PASS | [r0-scratch-differential.log](evidence-exp2/r0-scratch-differential.log) |
| mod-oracle 原考場 | 24 tests PASS；未修改 oracle 來源 | [oracle-regression.log](evidence-exp2/oracle-regression.log) |
| ownership | oracle hooks／兩個 rollout 函式與 R0 原測試 bytes 相同 | [ownership-check.log](evidence-exp2/ownership-check.log) |

16 格為 FSQ z on/off × intent on/off × BC_INDEP on/off × div on/off；fixture 使用非均勻 real weights。FSQ z 的 density 維度 2、head u 維度 4，另驗 flow condition 帶 anchors、head 用 COND_DROP 後條件與量化 clean。R0 不呼叫 sample/refine/require_services；兩 actor total 與獨立 nf＋anchor 完全相等。

額外通過：float64 recurrence、cond 與 latent 不同 dtype、CPU bf16 autocast、structural ValueError、nonfinite 留給 scaler、all-invalid/fingerprint/freeze/exposure 拒絕、stale EMA consistency gradient、兩種相反的 opt2／opt_bc skip，以及真正 CPU GradScaler 的兩 optimizer 區分。EMA parameters/buffers 與 noise RNG checkpoint roundtrip 通過；scratch 在原 opt2 call site 的 skip→EMA 行為也有考場。

```bash
.venv/bin/python -m unittest discover -s tests -p test_refine_wiring_v3.py -v
.venv/bin/python -m unittest discover -s tests_repair -p test_model_zero_rounds.py -v
.venv/bin/python -m unittest discover -s tests -p test_refine_quality_v3.py -v
# 開工前 snapshot 還在本機 /tmp；可直接重跑：
.venv/bin/python experiments/_workorders/v3design1/evidence-exp2/r0_differential.py /tmp/refine_v3_scratch_before.py
# 若 /tmp 已清，該基線 commit 的五個受测函式與開工快照相同：
git show 5b256d4bafdd9f1fdee0673fb147cae49d24e6c3:experiments/scratch_lacot_rollout.py > /tmp/refine-v3-r0-base.py
.venv/bin/python experiments/_workorders/v3design1/evidence-exp2/r0_differential.py /tmp/refine-v3-r0-base.py
```

**改動清單與 git diff 分段歸屬**

本批新增：`lacot/refine_objective.py`、`lacot/refine_training.py`、`tests/test_refine_wiring_v3.py`、本報告，以及 `evidence-exp2/` 的原始 logs／R0 比對腳本／本批 scratch diff。

本批修改 `lacot/model.py`：import／舊說明修正、image `losses_given:118`、state `losses_given:309`；`training_losses:145` 只增加 keyword services 轉送。沒有改 constructor、RefineOperator、inference 或 state_dict shapes。

本批 scratch hunks（行號以本次交付檔為準）：

| 區段 | 本批內容 |
|---|---|
| import `:16` | refine_training import |
| `CONS :725` | 回鍋撤回預設 EMA，所有預設回到 self；v3 EMA 要顯式設 `LACOT_CONS=ema`；eval-only 由 ckpt cfg 校正 |
| `:1154`、`:1195` | s_embed freeze、refine_ema eval、未合格 services 空位 |
| `_amp_step :1644` | 回傳該 optimizer 的實際 step 狀態 |
| `_stage2_loop :1690` 至 compose `:1766` | 前置拒啟、density／anchor／sample adapters、depth cycling、每批服務綁定 |
| total `:1815` | base_total 只加一次；BC/div/GRPO 外層保留 |
| `:1950`、`:1978` | opt2 成功才更新 refiner EMA；新增 quality/exposure/coverage logs |
| resume `:2088`、`:2132` | metadata／EMA 載回及續接深度步號 |
| checkpoint `:3845`、`:3866` | refine service state 與 R>0 optimizer state 保存 |

[own-scratch.diff](evidence-exp2/own-scratch.diff) 是「開工時已含 ucontrast 改動的快照 → 本次檔案」的 diff，只含本批 hunk。普通 `git diff` 中 `ORACLE_ARM` 初始化段、`policy_chunk` hook、`rollout(...oracle_draw...)`、最下方 ORACLE_COLLECTOR 執行段都是**開工前既有 u 對照線改動**，不列本批功勞；已驗其原文 bytes 未動。`refine_quality.py`／`refine_models.py`／proposal-v2 與 DESIGN／EXP1 文件均未修改。

**最容易寫歪的三處：自查**

1. 三 caller 的 total 各只加一次 base_total：兩 actor 直接返回 compose tensor；scratch 只有 `base_total + 原BC項`，div/GRPO 繼續原外層。考場驗 total 代數，scratch 另驗 BC_INDEP／div 的真 backward input；缺 quality mutant 先 FAIL。
2. sg 位置：sample/clean/noise 停梯度；EMA target 在 no_grad 內且 cond／previous state detached；quality teacher 參數 frozen，但 quality forward 不包 no_grad。student 所有自身輪次與 cond full BPTT；exposure 僅拿 detached plan 與 detached 同計畫 action labels。BPTT／cond mutants、anchor-gradient 及 frozen teacher assertions 分別取證。
3. 零 noise mutation 走真路徑：由 scratch 的 injected service 啟用輔臂，mutant 改真正 compose 的 noise，執行完整 production loop；不是 kernel 旁路測試，也沒有改成 data±noise。

**自我懷疑／尚未證明**

- WiringOracle 是明確 test-only 的二次函數與同計畫 label fixture。它能驗接線／梯度，不證明 Wφ/Aω 於 generated plans 的品質、coverage、exposure gap，亦不證明真 RefineOperator 學得會；Aω 未過的正式停損仍有效。
- scratch 考場執行完整 production 函式，但以 CPU globals 隔離環境／資料載入；沒有宣稱整支 OGBench trainer、checkpoint artifact 載入或 GPU autocast 已端到端跑通。GRPO/LO/BC_OWN branch 保持原碼，這次未開其環境相關考場。
- FSQ／intent 用帶可辨識數值的 adapter fixture 驗接線，未證 hard-snap Jacobian／訓推 discrepancy 合格。正式 codec/provider builder 仍須通過 mod-oracle 的版本與資格門，不能以本次 fixture 放行 image/ant R>0。
- checkpoint 考場覆蓋 refiner EMA、buffer、步數及輔臂 RNG；不冒稱整支 scratch 的 data RNG／CUDA RNG／GradScaler runtime 已可逐位元跨程序續訓。一般 actor loss API 不擁有 optimizer，外部 trainer 仍须在成功 step 後調用 services.after_step。
- 相同 condition 的 fresh flow sample 與 full BPTT 消除了已指出的輸入／錯配接線原因；是否仍因模型能力、品質代理或支持域而接近 identity，必須由後續獨立合格教師與正式學習考場回答。
