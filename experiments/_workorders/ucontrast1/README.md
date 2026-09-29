在證明什麼：pointmaze 的 u 抽樣執行變體是否優於同預算的動作噪音變體。此包只交付實作與 gate；GPU 評估及科學判決尚未執行。所有結果為 offline 評估產物，禁止作訓練資料。

## 預註冊判準

固定 s33 後補 s35；每 seed 5 official tasks × 40 episodes = 200 題，每題 8 條完整 rollout。Primary = pooled oracle@8：每題任一 draw 的 R1 `info.success` 為真，然後按題平均。A − 所選 B σ ≥5pp 且兩 seed 各自同號，才支持 u 的執行變體抽樣器角色；否則降級。兩 seed 合成分母為 400 題。per-draw quality、oracle@k 與 leg 統計只作 secondary；flat policy 的 leg_stats 為空。

## 臂、題集與隨機流

A 與 B′ 都在每個 policy chunk 根據當下 obs/goal 重新抽 u 並解碼。A 的每條 draw 用獨立 flow stream；B′ 每條 draw 從該題 draw-0 的 flow seed 重新起播，所以相同條件下逐 chunk 的 u 序列相同，跨 draw 的唯一外加差異是動作噪音。B′ 不快取一顆 u。σ0 時 B′ 的八條完整軌跡逐位元相同，且 B′ draw0 與 A draw0 逐位元相同。

σ = 0.05、0.10、0.20 × 動作半幅；當前 action range [-1,1]，半幅=1。Gaussian 加在 policy clip 後、env.step 前，再 clip 回 [-1,1]。noise 用獨立 CPU torch.Generator，不推進 flow RNG。A 也消耗配對 noise stream，但係數為零。

主實驗題號 ep 0–39、salt=0；σ 粗掃及 σ0 mutant 題號 ep 40–44、`SWEEP_SALT=2000011`。每題 reset seed=`1000*task+ep+salt`，noise/draw stream seed=`7*task+ep+salt+1000003*draw`；B′ flow seed 固定為同題 draw0 的 `7*task+ep+salt`，A flow seed為各 draw stream seed。故粗掃任何 rollout 均不與主實驗共用 `(task,ep,seed)` 三元組；env reset、flow 及 noise 三條 stream 均隔離。每次 oracle reset 前還以 reset seed 釘住 legacy 全域 `np.random` 和 `env.action_space`，因 ogbench 的 init/goal 抖動先於 `reset(seed=...)` 取全域 NumPy RNG。這只在 `LACOT_ORACLE_ARM` 啟用時生效；flag-off 官方 rollout 保持原行為。

主檔只加 opt-in hook 與外層 draw 收集；env.step、MAXH、R1 reached 仍走官方 `rollout()`。`LACOT_ORACLE_ARM=A/B`、`LACOT_ORACLE_DRAWS`、`LACOT_ORACLE_SIGMA`、`LACOT_ORACLE_OUT` 控制收集；`LACOT_ORACLE_EPISODE_OFFSET` 和 `LACOT_ORACLE_STREAM_SALT` 由 launcher 固定。要求 flat、EMA、R1、no refine、flow、BON_N=0；拒絕 DEV_EVAL、SUBGOAL、INTENT、FINISH、probe。`run.py` 清除父行程 LACOT_* 後重建配置。

## Gate 順序

1. 建立 `provenance.json`（格式見 `provenance.template.json`）。s33、s35 是凍結輸入資產：checkpoint gate 逐字比對路徑和完整 SHA256，分別為 `88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787`、`ba5009ec197d1d11b9902a2b92917d1ca9198e8b32882c442638380786ab0d7d`；`training_started_unix=null`，註明不適用。沒有訓練起跑時間或 mtime gate。
2. lead 側提供 GPU flag-off 三題逐位元等價證據 `smoke/flagoff-proof.json`（欄位見 `flagoff-proof.template.json`）：含 `device="cuda"`、baseline/current source SHA256、目前 `code_hashes`、s33 `ckpt_sha256`、三個相異 `(task,episode)` 的 `baseline_trajectory_sha256` / `current_trajectory_sha256`，兩側逐題相同。`run.py` 先驗這份證據，沒有則 BLOCKED。CPU 版 flag-off 等價已在 wiring test；GPU 版尚待執行。
3. `calibration.sbatch`：s33、A、draw0、200 題 R1 與 j_signal noclimb s33=.350 差值 ≤.08〔拍、預註冊〕。200 題的 naive binomial SE 約 .034；五個 official task 相關，.08 是保守放大。只比整體率，因釘 reset 抖動後逐題 RNG 消耗可不同。未過即停。
4. `mutant.sbatch`：B′ σ0、ep 40–44 × 8 draws。逐題檢查 reset/goal、flow 首 u、完整 trajectory、steps、成功位元相同，oracle@8=@1。真 GPU 結果未取得。
5. `smoke.sbatch`：A 和 B′ 三 σ 檔，各 ep 40–44 × 8 draws。A draw0 R1 必須在 .15–.60 sanity 帶。select 用獨立公式從每條 draw success 重算 oracle@k 與 per-draw quality；B 各正 σ 至少一題 trajectory hash 與同題 σ0 mutant 不同，且檢查 A/B reset、noise seed 和首 u 配對。選 B pooled oracle@8 最大者，平手取最小 σ；不產 A/B 判決。
6. `main.sbatch` 只供審查，不由本包提交。s33 的 A/B 各 200 題×8；s35 須另跑其 σ0 mutant、獨立粗掃與選檔，再用其 selection。

所有 gate 綁定 code hash、provenance hash 和結果 SHA256；結果 exclusive write。部署沿 J-signal 的 ada-lite/jasmine、指定 venv/ogbench 資料及 gpu:1。CPU 無 allocation 不能 launch GPU stage。

## 結果帳

`tasks[]` 每題含 task、episode、success_bits、八條 draws；每條記 env_seed、stream_seed、flow_seed、success、steps、reset/goal/first-u/u-sequence/trajectory SHA256、u_calls/u_samples。頂層存 oracle_at_k、pooled_oracle、per_draw_quality、n_oracle_success；末三欄依序為 `n_tasks,n_draws,n_success`。`n_tasks` 是 task/episode 題對，不是獨立 official task 數。R1 是任一步 `info.success`，不讀 `per_task.completed`。主檔每 draw 將 rollout 回傳率與 collector 位元平均交叉核；不符立即失敗。

## 自我懷疑

GPU 上 decoder/模擬器逐位元確定性與 flag-off 三題仍待 lead 實跑。σ 粗掃只有 25 題，最佳 σ 有取樣誤差；.08 錨容忍帶是依五個 task 相關性放大後的預註冊判準，並非已量得的 CI。既有 rollout 在 term/trunc 時只跳出 action 內圈，未在本回鍋範圍修正。真 env 驗證器用玩具 flow/head，能驗 reset 配對與 B σ0 確定性，不能替代真模型 GPU smoke。
