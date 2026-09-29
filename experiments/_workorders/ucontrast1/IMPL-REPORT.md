在證明什麼：u 抽樣產生的執行變體是否優於等預算的動作噪音變體（u 的「執行變體抽樣器」角色存亡）。

本次交付為**待檢實作包；驗收 BLOCKED**。沒有主實驗、沒有科學判決、沒有 commit。
真 GPU 校準、負向 mutant、25 題 A/B smoke 均未取得執行結果；不能標 DONE。

## 已完成的實作

- `experiments/scratch_lacot_rollout.py`：新增 opt-in `LACOT_ORACLE_ARM/DRAWS/SIGMA/OUT`；沿用原 `rollout()` 的 env/reset/step/MAXH/R1 reached，新增 hook 和多 draw 收集段，不另起 eval loop。
- `collector.py`：A fresh-per-chunk flow；B 首顆 u 全題／全 draws 固定；配對 Gaussian action stream；逐位元成功矩陣、u／trajectory hash、oracle@k、per-draw quality、末端完成計數；exclusive-write 防誤蓋。
- `run.py`：固定 ckpt、原訓練 mtime＋最大 `_st` gate、歷史錨／SE gate、σ0 gate、25 題三檔選 σ、結果重算與配對核對。資料／code hash 改變會讓 gate 失效。
- `batch.sh`、`calibration.sbatch`、`mutant.sbatch`、`smoke.sbatch`：沿用 J-signal 的 venv／資料／ada-lite／jasmine，所有 GPU 作業 `gpu:1`。
- `main.sbatch`：**僅交檢，未提交**。array 0–1 各跑 A／勝檔 B，200 題×8，預設 s33；s35 後補沿用腳本和其獨立粗掃選檔檔案。
- `README.md`：預註冊判準、σ=.05/.10/.20 動作半幅、schema／計數、gate 次序、offline 禁訓紅線及非空自我懷疑。
- `provenance.template.json`：不猜歷史帳、SE、原訓練起跑時間，未知值保留 null；**尚未建立可通過的 provenance.json**。
- `implementation.patch`：主檔 diff 加上新增程式／sbatch／README 的可檢查補丁；工作樹為實際交付。

## 實際驗證證據

| 驗證 | 實際狀態 | 證據 |
|---|---|---|
| CPU wiring integration | 9 tests PASS | `smoke/cpu-wiring-v2.log` |
| Python AST、sbatch/bash 語法 | PASS | `smoke/static-checks.log` |
| `git diff --check` | PASS | 同上 |
| 無 Slurm allocation 時拒絕 eval | exit 2，符合預期 | `smoke/no-allocation-guard.log` |
| s13/s33/s35 檔案及 cfg/EMA | 本機存在，CPU 讀取、SHA256 已記錄 | `smoke/checkpoint-inspection.log` |
| ckpt mtime＋最大編號 | **未通過驗收**，缺原訓練起跑來源 | 同上、provenance template |
| calibration sbatch 提交 | **FAIL，exit 1，無 job ID** | `smoke/calibration-submit.log` |
| s13 T1 N16 `.6680 ± SE` | **NOT RUN／未復現** | 缺歷史來源與 Slurm 連線 |
| 真 decoder B σ0 ×8 mutant | **NOT RUN** | 前置校準 gate 未過 |
| A／B 三檔各25題×8 smoke | **NOT RUN** | 同上 |
| 正式200題主實驗 | **未提交** | 本工單禁止 |

CPU 測試從主檔 AST 取出真正的 `policy_chunk()` 和 `rollout()`，只替換 toy env/head，沒有 import 主檔觸發訓練或 GPU。
`CUDA_VISIBLE_DEVICES=''`，使用本機既有 `.venv` 的 torch 2.6.0+cu124；沒有裝套件。
測試涵蓋：flag-off 與 git HEAD 原函式逐位元相同；A draw0 與原版一致；B u 抽樣次數／跨 draw hash；
A/B reset／stream seed 對齊；noise 不消耗 flow RNG；完整軌跡 σ0 相同；故意重抽 u／改 success／改 reset／
改 trajectory hash 都會被拒絕；missing/duplicate draws 被拒絕；oracle 單調；完成計數與拒絕覆寫。
第二輪 toy σ0 有成功與失敗兩種題（2/6 reached），不是只在全0位元上驗。**這不等於工單要求的真模型 mutant 親跑。**

提交校準時，Slurm 回覆：

```text
sbatch: error: Error creating slurm stream socket: Operation not permitted
sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)
sbatch_exit=1
```

本機 `/archive/cymaxwelllee/LaCoT/.venv/bin/python` 與 `/archive/cymaxwelllee/data/ogbench` 不存在；
它們是指定 compute node 的部署路徑，此次無法查證 jasmine 的可用性。沒有改用別處資料或裸跑 GPU。

## 三處易寫歪的自查

1. **B 真固定 u**：cache key 是 `(task,episode)`，跨全部 draws 保留實體 clone；每題僅第一 draw 首 chunk 呼叫 sampler，後續 `u_samples=0`。CPU 已驗；真 GPU 尚未驗。
2. **成功尺**：沿原 R1 的 `info.get('success')` reached；完全沒有讀 `per_task.completed`；any-success 在完整執行後才聚合。
3. **配對**：題序皆 task 外層／episode 內層；env seed 不含 draw、arm、σ、ckpt seed；draw seed 同公式。選 σ 前另逐題核對 reset/goal hash、stream seed、A/B 首顆 u。

## 盤點結論、未修既有問題及偏離點

- 已逐條讀 `materials/KO-INJECT-0928.md` 主題②，並讀 J-signal README／sbatch／判決材料。
- `EVAL_RS` 不是多樣本維度；`DIAG_DUMP` 沒有 oracle draw 維度；`BON_N` 是事前 GrpoReward 挑選；`probe_bon_rung05.py` 是 plan raw-hit。本件新增收集 hook，但沒有宣稱重建了 9/16 歷史管線。
- 搜尋面限 repo 的 docs／experiments／slurm 與指定材料；未無界掃描 home。`.6680` 原始帳、pooling、SE 未查到。
- 既有 rollout 的 term/trunc 只跳出內層 action loop，外層不含終止旗標；記錄為可能問題，**未順手修正**，尚未證明本配置會提前終止而觸發。
- 既有 `7*task+episode` 在 task 之間有重複 seed；為維持歷史 draw0 配對未修改。
- B 全題凍住首顆 u 的字面解讀、mtime 採原訓練起跑的解讀已在 README 與澄清問題明列；未收到回覆，不能當作已獲確認。
- 首輪 CPU fixture 使用 `TemporaryDirectory`，自動清除了該輪自行建立的 `/tmp` 測試檔；這偏離了「不刪檔」的最嚴格字面要求。未刪任何既有 repo／使用者資料，第二輪已改為保留 `/tmp` fixture，不再清除。
- 沒有 GPU smoke 結果；沒有冒用 CPU toy 或歷史結果補上；沒有主實驗提交、沒有 commit、沒有套件安裝、沒有改工單外的既有檔案。

## 自我懷疑與解除阻塞所需

最主要的不確定是歷史 `.6680` 的 pooling 是否根本不是 flat official episode。數字近似也不能證明語義相同；
須取得原 sbatch／逐位元帳和 SE 定義，先確認題集與 pooling，必要時依同一 rollout 擴充收集，然後重跑正向錨。
另外，B 凍住 u 會改變題內 replan，GPU decoder 確定性尚未驗，σ range 對實際 action RMS 的強度未量；均未隱去。

交檢後仍需：提供歷史錨與訓練起跑 provenance、確認固定 u／mtime 解讀、在能連線 Slurm 的執行環境提交校準；
正向校準 PASS 後才能親跑 σ0 mutant，再跑 A/B 各25題級 smoke。任一異常須停止，不能直接上200題。

STATUS: BLOCKED(校準 sbatch 連線被拒且無 job ID；s13 .6680 歷史 pooling/SE 與 ckpt 原訓練起跑來源缺失；正向校準、真模型 σ0 mutant 與 A/B GPU smoke 均未完成)
