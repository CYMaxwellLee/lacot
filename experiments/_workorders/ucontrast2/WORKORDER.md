# ucontrast2 工單：u 線兩補充臂（N=64 折衷＋s35 重現）

主人 2026-09-29 裁「都跑」（TG 6934；N=64 折衷案＋s35 走 worktree 凍舊碼方案）。

## 在證明什麼（兩臂各自、預註冊）

**臂甲（N=64 折衷）**：u 的 oracle 優勢是否隨 draw 數放大到過門檻。
判準（預註冊、不得事後改）：50 題×64 draws 配對兩臂（A=u 抽樣、B=噪音 σ=.05），
pooled oracle@64 的 A−B ≥ +5pp 且配對 McNemar 單向 p<.05 ⇒ u 站穩；
否則維持 9/28「未站穩」判決。50 題子集＝從原 200 題池以 seed=20260929
均勻抽（抽法寫進 code、抽出的題單入結果 JSON——別台重抽必須逐題一致）。

**臂乙（s35 重現）**：9/28 的 u 判決（A−B=+3.0pp、配對 6:0 單向、解鎖 9v2）
是否在第二顆凍結 ckpt（s35）上重現。口徑與 s33 判決完全同（200 題×8 draws×
兩臂）。判準：方向與解鎖結構照實記，不設通過門檻——這臂是重現性測量，
合併判準（兩顆 pooled）等兩臂都齊後另行預註冊。

## 靶與既有鏈（全在 `experiments/_workorders/ucontrast1/`）

- `run.py`＋`collector.py`＋`main.sbatch`＝現行兩臂主跑鏈（jasmine、gpu:1）。
- `README.md`＝gate 規矩全文（code hash gate、ckpt SHA gate、provenance、
  exclusive write、schema 末三欄 `n_tasks,n_draws,n_success`）。
- s35 ckpt SHA（README 已釘）：`ba5009ec197d1d11b9902a2b92917d1ca9198e8b32882c442638380786ab0d7d`。
- ⚠️ hash 錨現況：exp2/builder 兩批改了 `experiments/scratch_lacot_rollout.py`，
  `run.py` 的 code hash gate 釘的是 s33 判決當時的舊 SHA ⇒ 在現工作樹直接跑
  **會被 gate 拒（gate 正常運作、不是 bug）**。

## 要做的活

1. **worktree 方案（兩臂共用）**：從 gate 的 baseline source SHA（讀
   `ucontrast1` 的 provenance/flagoff-proof 所記 `code_hashes`／baseline sha，
   找到對應 git commit）建 `git worktree`（建議 `/home/cymaxwelllee/Projects/
   lacot-u-frozen`，jasmine 側跑時同法）。⛔ 不准為過 gate 改 gate 邏輯或
   重算 hash 釘值；worktree 內容 hash 必須等於 gate 既釘值（這就是驗收）。
2. **臂乙**：s35 版 sbatch（複製 main.sbatch 改 ckpt 指向 s35＋新 OUT 目錄
   `/archive/cymaxwelllee/ucontrast1/s35/`）；provenance 照 README 規矩
   （凍結資產、training_started_unix=null）。
3. **臂甲**：N=64 版（50 題子集 seed=20260929、64 draws、兩臂 A/B）＋新 OUT
   `/archive/cymaxwelllee/ucontrast1/n64/`。50 題抽樣 code 寫在 run 入口、
   題單（task,episode 對）落結果 JSON。獨立 sbatch。
4. **smoke 檔**：沿 `smoke.sbatch` 慣例（ep 40-44 縮版）給兩臂各一份，
   smoke 輸出寫 scratch、不碰正式 OUT。
5. **成本帳**：臂甲 50×64×2 臂=6400 rollouts、臂乙 200×8×2=3200——對照
   9/28 主跑（3200 rollouts 約半天）給 ETA 估計、寫進交付。

## 紀律（⛔ 級）

- 不碰凍結資產（s33/s35 ckpt）內容；只讀。
- 不改 gate 邏輯；gate 拒＝BLOCKED 回報，不繞。
- 結果 exclusive write；OUT 目錄必須是新的（既有檔=拒started）。
- 不跑 GPU／不 sbatch（上機是 F5 鏈的事）；你交付 code＋sbatch 檔＋smoke 驗證
  （CPU 可驗的部分：50 題抽樣重現性、sbatch 語法、worktree hash 對齊）。
- 測試環境差異聲明（工單模板 9/28 補強）：你在 zeldajr CPU 驗的與 jasmine
  GPU 正跑的差異面明列。

## 交付（寫進 out 檔）

- 改動清單（檔案、diff 摘要）＋兩態證據（正：worktree hash=釘值、50 題抽樣
  兩次重跑逐題一致；反：改一字元的 worktree hash 必不等、壞 OUT 目錄必拒）。
- ETA 估計＋sbatch 資源申請（gpu:1、CPU 核數對 code 實際用量）。
- 自我懷疑欄（沒把握的格明標，⛔ 不准留空）。
- 尾行 `STATUS: DONE` 或 `STATUS: BLOCKED(原因)`。
