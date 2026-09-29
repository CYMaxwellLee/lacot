在證明什麼：檢察官 S2 的 FAIL(8,2) 與 lead 裁定的回鍋項已完成 CPU 可驗的實作；真 GPU gate 仍待 lead 執行，本報告不冒充 GPU 結果。

## 逐項回鍋

1. **VERDICT-S2 §8 / FAIL-8**：oracle `rollout()` 每次 reset 前以同題 reset seed 執行 `np.random.seed` 和 `env.action_space.seed`，照 `dev_eval.py:126-135`。flag-off 仍走原分支。ToyEnv reset 抖動改用全域 NumPy。直接復用檢察官 `realenv_pairing.py` 的現行碼段，A σ0、B′ σ0、B′ σ.1 均 `summarize OK`，A/B 起點相同；B′ σ0 八 draws 完整 trajectory hash 逐位元一致。證據：`smoke/realenv_pairing_rework.py`、`.log`。
2. **VERDICT-S2 §2a / FAIL-2a**：smoke 和 σ0 mutant 使用 ep 40–44、`SWEEP_SALT=2000011`；main 使用 ep 0–39、salt 0。reset、flow、noise seed 均納入 salt，`(task,ep,seed)` 無交集。README 預註冊，CPU test 核對。
3. **VERDICT-S2 §2b / lead B′**：刪除整題一顆 u cache。B′ 每 chunk replan，每 draw 的 flow RNG 都從該題 draw-0 seed 重播；noise 仍按 draw 獨立配對。σ0 B′ 與 A draw0 trajectory/u sequence 逐位元相同，且 B′ 自身八 draws 相同；測試已驗。README 臂定義同步更新。
4. **VERDICT-S2 §3 / Item 3**：刪除 .6680/s13 與原訓練起跑時間 gate。s33 校準為 A draw0 的 200 題 R1 對 j_signal noclimb .350、容忍 ±.08〔拍〕；GPU flag-off 三題 hash 證據為前置 gate，缺證據即 BLOCKED；A smoke sanity 帶 .15–.60。checkpoint 逐字路徑加完整 SHA256：s33 `88180676…`、s35 `ba5009ec…`（本機重驗與檢察官一致）；provenance 訓練起跑時間為 null、註明凍結輸入資產不適用。GPU 證據格式與 gate 順序見 README；目前未產生 GPU proof 或通過真 GPU gate。
5. **VERDICT-S2 §4/§5 建議 R1–R3 / 測試缺口**：R1 每 draw 比 rollout 回傳 R1 與 collector per-draw 成功率；R2 CPU 比 B′ 正 σ 和 σ0 的 env.step 軌跡 hash，select 亦比同題 σ0 mutant；R3 `run.py` 用不呼叫 `summarize` 的獨立公式重算所有 oracle@k、per-draw quality、n_success，並核對 raw draws 與 success_bits。

## 驗證證據

- `smoke/cpu-wiring-rework.log`：17 tests，全部 OK；含 flag-off 對 git HEAD 逐位元、真 AST oracle loop、B′/A draw0、reset 配對、sweep 不交集、gate、R1/R2/R3。
- `smoke/realenv_pairing_rework.log`：A σ0、B′ σ0、B′ σ.1 無 `Unpaired resets`；B′ σ0 八 draws trajectory hash 逐位元同；B′ σ.1 同題三條 trajectory 不同。驗證器只用玩具 flow/head，環境是真 ogbench CPU。
- `smoke/mutants/`：檢察官 driver 的副本加上 unittest exit-code 回傳，對工作樹唯讀、在副本注入三個 mutant。M6 `end(False, steps)`：`M6.mutant.log` exit 1（R1 交叉核錯），`M6.repaired.log` exit 0 / 17 pass。M7 噪音未送 `env.step`：exit 1（正 σ 軌跡未改），修復 exit 0 / 17 pass。M9 oracle 只看 draw0：exit 1（獨立 oracle 兩測試抓出），修復 exit 0 / 17 pass。各兩態 log 均保留。
- Python AST/compile、`git diff --check` 與 sbatch/batch bash 語法通過；未跑 GPU、sbatch，未 commit。

## 自我懷疑

GPU flag-off proof 尚未生成；gate 的 JSON 驗證只能驗來源 SHA、ckpt SHA、三題 hash 對齊，仍須 lead 確認產生 hash 的 GPU runner 確實記錄完整 trajectory。真模型 decoder/模擬器確定性尚未驗。B′ 在正 σ 下每 chunk 的 cond 會隨噪音改變，所以「唯一外加差異」是動作噪音，後續 u tensor 仍可能因 cond 不同而變；這是閉環設計的預期。既有 term/trunc 內圈 break 問題僅記錄，未跨範圍修復。

STATUS: DONE
