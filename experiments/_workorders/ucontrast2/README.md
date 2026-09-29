# ucontrast2 預註冊補正（2026-09-29，GPU 結果產生前）

本輪依 lead 的 D1 拍板：s35 是獨立粗掃臂。它先以 s35 凍結 checkpoint 跑 B σ0 mutant，再以同一 checkpoint 跑 A σ0、B σ=0.05/0.10/0.20，各 5 official tasks × ep 40–44 × 8 draws。從三個 B 的 pooled oracle@8 選最大者，平手取最小 σ；selection 只選對照參數，不作 A/B 判決。選出的 σ 用於 s35 主實驗的 B 臂。s35 主實驗仍是 ep 0–39、200 題 × 8 draws × A/B；A 為 σ0。s33 9/28 calibration、flag-off 證據作為共通程式碼 gate，s35 的 mutant/selection 則須另立證據。B 對照臂取對它最有利的 σ，讓對照採最強形式；若沿用 s33 的 σ，判準會偏鬆。原工單「口徑與 s33 完全同」指量測口徑與 collector/analyze，不指校準參數共用。

固定 source、checkpoint SHA、provenance、flag-off、s33 calibration 與 s33 selection gate 先由 `validate_inputs()` 驗證。s35 mutant 的完整結果與 preflight 均須有不可變 SHA；負控制須 `negative_mutant_passed=true`。四格粗掃用原 `ucontrast1.run.select()` 的配對、題號、oracle、正 σ 軌跡變異與 A draw0 sanity 檢查；CPU select 輸出 `sigma-selection.json`，正式 s35 啟動前再核其四個結果 SHA、s35 checkpoint、preflight 與勝出 σ。所有結果以新目錄 exclusive write；失敗即 BLOCKED。所有產物是 offline 評估證據，禁止作訓練資料。

執行順序（在 lead 建妥凍結 worktree 並跑 `validate_inputs()` 後）：

1. `sbatch experiments/_workorders/ucontrast2/s35-mutant.sbatch`；確認 `s35-calibration/mutant/B/mutant-gate.json` 為 PASS。
2. `sbatch experiments/_workorders/ucontrast2/s35-sweep.sbatch`；四個 array index 全部 verified 後才往下。
3. `/archive/cymaxwelllee/LaCoT/.venv/bin/python -B experiments/_workorders/ucontrast2/run.py s35 --s35-stage select --outdir /archive/cymaxwelllee/ucontrast1/s35-calibration`；確認 `sigma-selection.json` 為 SELECTED。
4. `sbatch experiments/_workorders/ucontrast2/s35-smoke.sbatch`，再提交 `s35.sbatch`。n64 的提交順序照原方案。正式兩臂齊後才執行 `analyze.py s35`。

資源依 9/28 主跑實測 6.9 秒/rollout、MaxRSS 約 1.22 GiB：n64 每臂 3200 rollouts 約 6.2 小時，s35 每臂 1600 約 3.1 小時；s35 mutant 200 rollouts、四格粗掃各 200 rollouts，單格約 23 分。若只有一顆 GPU，mutant 加四格粗掃約 115 分；若四格同時分配 GPU，mutant 後約再 23 分。所有新舊 sbatch 申請 gpu:1、CPU 8 核、記憶體 16G、12 小時上限；ada-lite MaxTime 為 12 小時。n64 smoke 每臂 50 rollouts 約 6 分，s35 smoke 每臂 200 約 23 分。以上不含排隊、載入與 I/O 波動。

CPU 自驗只能覆蓋選檔、gate、sbatch 語法與檔案契約。jasmine 的 CUDA/MuJoCo 真 rollout、Slurm 限制與 archive 寫入仍由上機 smoke 驗證。
