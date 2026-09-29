# Flag-off GPU proof driver

STATUS: DONE

## 設計

`gen_flagoff_proof.py` 從 `git show HEAD:experiments/scratch_lacot_rollout.py` 取得 baseline 原始碼，暫存在 `experiments/` 同層，與工作樹版分別動態載入。載入器只略過腳本末尾與本件無關的多臂整批評估；兩側的模型初始化、s33 checkpoint 載入、EMA 覆蓋、`policy_chunk()` 和官方 `rollout()` 函式原樣執行。配置對齊 `run.py` 的 flat、R1、flow、s33 EMA，且先清除父行程所有 `LACOT_*`，不設 `LACOT_ORACLE_*`。

driver 在兩側使用同一個 `TracedEnv`。官方 reset 要求的 task/episode 先經核對，wrapper 隨後在 `env.reset` 前以 `1000*task+episode+SWEEP_SALT`（`SWEEP_SALT=2000011`）設定全域 `np.random.seed` 及 `env.action_space.seed`，並將同一 seed 傳入真 env。這是外部配對修正；不更動任一側的 rollout。wrapper 對 reset seed、起點 obs、goal、每步 action/obs/reward/term/trunc/info.success 按順序做 SHA256。`range` 外部選題器只作用於 `rollout()` 的兩層題號迴圈，故每次呼叫原版 `rollout(1, True, ...)` 只執行指定的一題，且保留原版 `7*task+episode` torch action stream。

執行前驗 s33 checkpoint 完整 SHA256；結束時驗 current source 未變。每題比較 SHA256 與步數，三題全同才以 exclusive create 寫出 proof，任一差異印出摘要並 exit 1。JSON 包含模板的 device、baseline/current source SHA256、`run.py` 的 `code_hashes`、s33 `ckpt_sha256` 與三題 hash。輸出已有檔案時拒絕覆蓋。

交給 lead 的入口：`sbatch experiments/_workorders/ucontrast1/flagoff.sbatch`。腳本限定 jasmine、ada-lite、gpu:1、30 分鐘，使用既有 archive venv 與 OGBench data，輸出 `smoke/flagoff-proof.json`。本次未提交 sbatch、未使用 GPU。

## CPU 自驗（真 checkpoint、真 OGBench env）

本機 `.venv/bin/python`、`/home/cymaxwelllee/.ogbench/data/pointmaze-large-stitch-v0.npz`、s33 checkpoint 執行：

```bash
OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
  .venv/bin/python -u experiments/_workorders/ucontrast1/gen_flagoff_proof.py \
  --tasks '1:3,2:7,4:11' --device cpu \
  --out experiments/_workorders/ucontrast1/smoke/flagoff-proof.cpu.json
```

結果檔：`smoke/flagoff-proof.cpu.json`。

| task:episode | baseline/current SHA256 | steps（兩側同） |
| --- | --- | ---: |
| 1:3 | `469ba720a9d8d59ce6fb35c1df909976525f7874f0f9d44037aadf3823e0ad71` | 490 |
| 2:7 | `1e49485dba26b6a799de7ba4b6c3e183113df426f13c52c3873b0aedab09208e` | 1000 |
| 4:11 | `82520b3c1ac006434f81240a30a8b20c9486ca130055953081ec90ce7531de9c` | 1000 |

動態 import `run.py` 後直接呼叫 `flagoff_contract(cpu_proof, 33)`：以預期的 `ValueError: GPU flag-off three-question bitwise proof missing/mismatched` 拒絕 CPU device。僅在 `/tmp` 暫存副本將 `device` 欄改成 `cuda`，其餘 proof 原值不變，再呼叫同函式獲接受；這只檢查「除 device 外」的 gate 欄位，不是 GPU 等價證據。

最終 driver 再跑一次得到逐欄相同的 CPU proof；Python AST compile 與 `bash -n flagoff.sbatch` 通過。未改 `run.py` 或既有 rollout，未安裝套件、未 commit。

## 自我懷疑

CPU 對等不能推論 CUDA 上的 decoder 與模擬器也逐位元相同；lead 必須實跑 `flagoff.sbatch`。`run.py` 的 contract 只看三題 hash 相等，不驗 hash 的序列化規格，因此 driver 的外部 wrapper 與原版 rollout 路徑仍須一起審查。若工作樹其他 `code_hashes` 檔在 lead 跑完後變動，gate 會拒絕既有 proof，需重新生成。
