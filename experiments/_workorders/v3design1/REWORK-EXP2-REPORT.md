# EXP2 第二輪小回鍋

範圍：VERDICT-EXP2-S2 §2 小弱點、§6 M1/M2、上機前必改清單 M3/M4。未改 u 對照線 hook，未 commit、未跑 GPU/sbatch、未安裝套件。

## 逐項改動

- **M1（§6、上機前必改清單）**：撤回 `LEARNED_REFINE=1` 將 CONS 預設翻成 `ema` 的改動；未設 `LACOT_CONS` 時一律是歷史預設 `self`。v3 訓練仍由服務綁定段要求 `CONS=ema`，因此須顯式設 `LACOT_CONS=ema`。`objective_version` 寫入 rollout JSON，`OBJECTIVE_VERSION` 寫入 ckpt `cfg`；tag 不加版本段。新 learned-refine 訓練記 `generated-plan-v3`，R0 記 `r0-baseline`。eval-only（`LOAD_CKPT` 非空、`CONT_TRAIN=0`）先從 ckpt `cfg["CONS"]` 取 CONS；若 env 顯式值不符或 cfg 缺有效值，立即 assert。eval-only 的版本採用來源 ckpt `cfg`，其次採用 refine metadata；舊檔兩者都沒有時明記 `legacy-unspecified`，不把舊結果誤標成 v3。
- **M2（§1、上機前必改清單）**：`_stage2_loop` 的 refine 服務綁定前檢查 `LEARNED_REFINE=1` 且 `INTENT` 非空，直接 raise；訊息指出 hindsight anchors 未過 same-intent-as-deployment 資格並指向 VERDICT-EXP2-S2 M2。intent 本體未改；R0 intent 差分仍跑。
- **M3（上機前必改清單）**：IMPL 報告補「GPU 1-step smoke 由 F5 鏈執行段做」。本回鍋沒有 GPU 執行證據。
- **M4（§5、上機前必改清單）**：IMPL 報告照引 M4 原文，記錄 u 線整檔 code hash 錨與 commit／開分支的順序協調事項。
- **§2 小弱點／N1**：`r0_differential.py` 兩個 zip 前加兩側等長且非空斷言，防空列表或截短比較靜默通過。

## CPU 綠證據

| 驗證 | 結果 | 證據 |
|---|---|---|
| v3 wiring，含六 mutation 兩態、M2 intent 拒絕、CONS 兩態、版本欄 | 16/16 PASS | [rework-wiring.log](evidence-exp2/rework-wiring.log) |
| 開工前 scratch SHA256 `276c68fb…` vs 現版 R0；FSQ × intent × BC_INDEP × div | 16/16 BITWISE PASS | [rework-r0-differential.log](evidence-exp2/rework-r0-differential.log) |
| 原 R0 model 考場 | 1/1 PASS | 本回鍋 `.venv/bin/python -m unittest discover -s tests_repair -p test_model_zero_rounds.py -v` |
| `git diff --check`（scratch） | PASS | 本回鍋執行，exit 0 |

檔名逐字元證據：預設 `LEARNED_REFINE=1`、無 `LACOT_CONS` 時，新 tag 與開工前 `CONS=self` 規則同為 `medium-stitch_self_K4_c256_ch4_st2000_T128_ep50_gu_s0`；顯式 `LACOT_CONS=ema` 時為 `medium-stitch_ema_K4_c256_ch4_st2000_T128_ep50_gu_s0`。測試直接執行 production 的 CONS assignment、`_tag_extra` 與 tag expression，對兩個完整字串逐字元相等斷言。eval-only 測試對 ckpt `CONS=self`／`ema` 兩態均驗證無 env 與同值 env 會採用 cfg，異值 env 必 assert，缺值亦 assert。

## 自我懷疑

- 沒有在真 OGBench eval-only 程序載入完整 ckpt；測試執行純 config／tag AST 並檢查載入段確實呼叫 cfg guard。真 ckpt I/O 與 GPU dtype 留給 F5 鏈的 smoke。
- objective version 保持檔名不變是 lead 本輪裁定；同一輸出目錄裡若舊版與新版其他旋鈕完全相同，仍可能撞名。實驗 2 執行時必須使用專屬 `LACOT_OUT_DIR`，讀結果時以 JSON／ckpt 版本欄辨識。
- 舊 ckpt 若沒有 `cfg["CONS"]` 會明確拒絕 eval-only；這是防錯名取捨，該舊檔須另案確認來源設定後遷移。

STATUS: DONE
