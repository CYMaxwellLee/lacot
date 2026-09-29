# F2 mod-oracle 第二輪回鍋報告

STATUS: DONE

本輪只改 VERDICT-ORACLE-S2 裁定的 M1–M3、G1/G2、L1–L3 與兩處文字／witness 小修。未跑 GPU、sbatch，未 commit；沒有把 CPU toy 讀數當實驗 1 資格。

| 項目（判決節號） | 改動與證據 |
|---|---|
| M1（§8-C、上機前必改） | `--split-seed` 預設固定 `1729`；`--seed` 只管模型初始化／訓練及評估 shuffle。split hash 是 dataset hash、split seed、三個 split 的排序 episode IDs 的 canonical JSON SHA256，進 record metadata、readings、checkpoint metadata／calibration 和 teacher fingerprint。三個 CPU toy seeds 0/1/2 的 dataset 與 split hash 完全一致，teacher fingerprint 各異，見 [rework-split-seeds.txt](oracle-smoke/rework-split-seeds.txt)。 |
| M2（§8-B1/B2/B3、上機前必改） | 新增 [DATA-MANIFEST.json](../../refine_v3/DATA-MANIFEST.json)：僅兩份 large-stitch 本機副本，完整 SHA256、檔名、domain、obs/action 維度；明標未對 upstream 驗證。loader 重算 SHA256 並核全部欄位；CLI 移除 `--dataset-hash` 和自建 permit，domain 只用於核對。兩份本機 NPZ 通過，見 [rework-manifest-check.txt](oracle-smoke/rework-manifest-check.txt)。 |
| M3（§6、上機前必改） | `available` 向下取 4 的倍數；mask 為真的每一個 chunk 都有四個真實 action。`partial_chunks_dropped` 按 split 計數進 metadata／report；toy 三 seed 均為 calibration 210、test 210、train 476 個所抽候選窗。相關逐步配對與整 chunk 測試通過。檢察官所報官方候選窗 10.66% 是舊行為測量，這裡未重跑全量比例。 |
| G1（§8-B4、gate 使用前必改） | `build_records` 的 `offline_dataset` 只能從 `load_offline_npz` 實際登記的 `VerifiedArrays` 產生；陣列回傳後唯讀，record 帶 loader 發的來源證明，訓練／校準入口再次核對。原始陣列自標、偽造容器與 toy record 改標均拒收。 |
| G2（§8-B4、gate 使用前必改） | held-out `evaluate` 封存 readings digest；`authorize_training` 核 teacher fingerprint、split hash 與 readings digest。`measure_generated_candidates` 由實際模型 report 計 validity／coverage，記 parent hash 與 candidate IDs，不收手填 bool 矩陣。手填「全過」readings 的真 offline fixture 在移除 seal 的對照中可授權，現行程式拒收。 |
| L1/L2（§2、需 lead 裁） | lead 預註冊修訂已列於 [README.md](../../refine_v3/README.md) 新舊並排及 `qualification_reasons` 註解。PI 85–95% 與 data validity ≥90% 保留健康報告讀數、移出資格阻擋；資格仍核 W held-out NMSE ≤0.9×obs-only、配對改善 CI 下界 >0、Aω NMSE ≤0.1；原有實際候選門檻維持。測試逐一使保留門檻失敗並驗證健康讀數不擋。 |
| L3（§需 lead 裁） | readings 新增 `known_limitation`：Wφ chunk 內非因果，主讀數以 chunk 末端／段級為準，causal masking 待實驗 1 過線另裁；未改模型行為。 |
| 小修（§1、§5、§7、optional O1/O2） | [原報告](IMPL-ORACLE-REPORT.md) 改正原版 metadata **11** 欄與 injection 對照 **4 FAIL＋3 ERROR**。`test_action_blind_witness_is_detected` 改用實際 W 預測、shuffle 動作與真 future；動作清零 mutant 使測試 FAIL。 |

## 驗證

- [全測試](oracle-smoke/rework-green.txt)：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v`，**24 tests，OK**。
- [注入兩態](oracle-smoke/rework-injections.txt)／[重跑腳本](oracle-smoke/rework-injections.py)：G1 自標來源、G2 移除 readings seal、action-blind W 各 **FAIL**；原樣三條各 **PASS**。G2 red 用測試專屬 manifest 驗證的 `offline_dataset` fixture；手填全過 readings 在舊核對路徑確實獲授權，不是只改了錯誤訊息。
- [三 seed CPU toy](oracle-smoke/rework-split-seeds.txt)：固定 split hash `e88f477a9b9825448f0a417b400b87c19fdd5c8e80d157757287b8265b58624e`，不同 teacher fingerprints；各 seed CLI／checkpoint roundtrip exit 0。個別輸出見 [seed 0](oracle-smoke/rework-toy-fixed-s0.txt)、[seed 1](oracle-smoke/rework-toy-fixed-s1.txt)、[seed 2](oracle-smoke/rework-toy-fixed-s2.txt)。

## 自我懷疑與局限

- manifest hash 只證明本機副本內容；未與 upstream 官方 hash 比對。此輪未做正式資料長訓練，也不能宣稱 Wφ／Aω 已取得科學資格。
- `source_proof` 與 readings seal 防止 library 正常 API 的自標／手填；同一 Python 行程內有意改動私有物件、或改寫 checkpoint 與其 digest 的攻擊者不在這輪威脅模型。候選 `parent_model_hash`／IDs 是 API 記錄的出處宣稱，真生成器的綁定仍需 F5 接線驗證。
- partial chunk 的新計數針對實際抽中的窗；檢察官 10.66% 是全候選舊口徑，兩者不可直接相比。Wφ 非因果敏感度只沿用判決書的 toy 2.1–5.4%，本輪未在真資料重測。
