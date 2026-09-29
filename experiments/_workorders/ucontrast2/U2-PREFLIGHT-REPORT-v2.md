# ucontrast2 發射單 v2：①預檢對抗報告（第二輪）

日期：2026-09-29

受檢：`U2-LAUNCH-CARD.md` v2、六支 ucontrast2 sbatch、
`ucontrast2/run.py`、`ucontrast1/run.py`／README。範圍嚴格限於前一輪五個
必改，另掃 v2 新增的 ETA／死點與 smoke 盲區文字；不重開前輪已 PASS 的格。
本機只能做靜態檢查，沒有再次 SSH、sbatch 或 GPU 實跑；題示的
`validate_inputs()=READY`、jasmine／archive 事實按 lead 已驗結果採信。

## 總判定

**VERDICT: GO（五項必改全 CLOSED；限本輪發射契約預檢）**

五項沒有 `NOT-CLOSED`。第 3 項的 GO 意義是「卡片已誠實降級、人工流程已明寫」；
不是宣稱 Slurm 或 launcher 會機械 enforce 全鏈順序。

## 五必改逐項判定

| # | 判定 | 核驗結果 | 自我懷疑／邊界 |
|---|---|---|---|
| 1. select 命令 | **CLOSED** | 卡片 `:44-47` 是單行 `ssh jasmine`，使用實路徑 `/archive/cymaxwelllee/LaCoT/.venv/bin/python`，`-B`，且 `--outdir /archive/cymaxwelllee/ucontrast1/s35-calibration`。該行通過 `bash -n` 語法檢查。`run.py:262-276` 將 `args.outdir` 解析為 calibration root，驗 mutant gate、四個 sweep verified，並在同一 root 寫 `sigma-selection.json`；不帶錯誤的 `--calibration-root`。 | 未在本地實際 SSH jasmine；venv、archive 可見性採 lead 已驗事實。卡片 `:38` 的「從 zeldajr 提交」只描述 sbatch 提交來源，不是 select 的執行端；select 明確只在 jasmine。 |
| 2. 計數表 | **CLOSED** | 卡片 `:69-79` 與程式逐格一致：n64 formal `50×64=3200`（`run.py:187-190,234-238`）；s35 formal `200×8=1600`；n64 smoke `25×2=50`；s35 smoke `25×8=200`；s35 mutant／每個 sweep bin `25×8=200`（`run.py:325-330`）。`n_tasks` 是 `(task, episode)` 題對數，`n_draws=n_tasks×draws_per_task`，且末三欄硬驗相同。 | 未跑真 rollout 產生結果 JSON；判定依 launcher 硬驗與題單生成式，沒有把靜態表冒充 runtime 結果。 |
| 3. 鏈序聲明 | **CLOSED（誠實化成立）** | 卡片 `:55-61` 明寫 code 只對前置產物 fail-closed：sweep 驗 mutant gate（`run.py:293`）、select 驗四個 verified（`:265-276`）、s35 formal 與 smoke-B 讀 selection（`:181-185`）。同時明寫無 `--dependency`、smoke-A 不驗 selection、formal 不驗 smoke receipt，並把「每步收割後才發下一步」交給 lead／⑤官人肉執行。這與實際控制流相符；s35 smoke-A 的跳過條件在 `run.py:184`。 | 若人工順序被違反，formal 仍可能早於 smoke，或 smoke-A 先留下 stranded OUT；這是卡片已揭露的殘餘風險，不再誤稱為 code enforce，故不構成此必改未閉合。 |
| 4. 六支 bytecode export | **CLOSED** | 六支 `n64-smoke.sbatch`、`n64.sbatch`、`s35-mutant.sbatch`、`s35-smoke.sbatch`、`s35-sweep.sbatch`、`s35.sbatch` 均有 `export PYTHONDONTWRITEBYTECODE=1`；每支均位於 OMP export 之後、`exec` 之前。逐支行序檢查與 `bash -n experiments/_workorders/ucontrast2/*.sbatch` 全部通過。launcher 建 child env 時保留既有環境（`run.py:210-212`、calibration `:311-313`），所以 export 會傳給 child subprocess；NFS home 的 Python `__pycache__` 寫入缺口已補上。 | 未做 jasmine 的檔案監看；此結論針對前輪指出的 Python bytecode 寫入面。外部 runtime／系統 cache 若另有寫入，不由此靜態檢查證明不存在。 |
| 5. ckpt／seed 映射 | **CLOSED** | 卡片 `:31-32` 已成為明文契約：n64 = s33 ckpt／seed 33，s35 = s35 ckpt／seed 35。`run.py:178-179` 固定 mode→seed，`:205` 以 `old.CKPTS[seed]` 放入 rollout config，`:237` 再硬驗結果 ckpt；`ucontrast1/run.py:15-16,54-64` 以路徑與 SHA gate 釘住兩顆 frozen ckpt。s35 calibration 亦在 `run.py:307` 固定使用 35。 | 未在本機重新讀 jasmine frozen ckpt bytes；實際 SHA／worktree READY 依題示 lead 驗證，程式 gate 本身仍會在上機重驗。 |

## v2 新增內容掃描

### ETA、GPU 波次與死點

- ETA 修訂正確：每格 200 rollouts、以 6.9 秒／rollout 約 23 分；s35 sweep 是
  array `0-3` 的四格，在三張 GPU 上至少兩波，故約 46 分，不再誤寫成四格同時
  23 分。formal 兩臂並行時 s35 約 3.07 h／臂、n64 約 6.13 h／臂；這些是
  題示實測基準的推算，不含 queue／I/O，卡片也沒有把它寫成保證。
- 另有一個已被「不含 queue」涵蓋的排程邊界：兩條鏈若同時提交兩個 formal
  array，總請求是 4 個 GPU job，而 jasmine 只有 3 卡，不能保證四 job 同時起跑。
  這只會令其中一臂排隊；按 3.1 h／6.2 h 基準仍在 12 h 牆鐘內，故是 ETA
  caveat，不是新的契約洞或本輪 NO-GO。卡片的「各 array 兩臂並行」應讀作單一
  array／理想可用 GPU 的估算。
- 死點聲明正確且足夠醒目（卡片 `:96-98`）：結果在 rollout loop 收尾才寫，
  timeout／OOM／搶占會留下 partial OUT；`reserve_outdir` 拒絕同 arm 重跑，沒有
  checkpoint 續跑，須換新 OUT 並人工調查。這是已揭露的殘餘風險，不是新洞。

### smoke 盲區與收割判讀

- n64 smoke 的 `2 draws`、25 題對／50 raw rows 與 `run.py:187-190,234-238`
  相符；因 `questions=None`，它不走 `rollout_n64.py` 的 64-draw derived path。
- 卡片 `:105-108` 明確要求正式完成後由 `run.py:234-259` 重驗 shape／題單／schema，
  並明說 harvest 時不得把 n64-smoke PASS 升格為全 derived 路徑 PASS。這把前輪
  smoke 數字錯誤與過度解讀兩個洞都補齊；derived 首航風險仍存在，但已是誠實揭露，
  不在本輪重開為必改。
- 卡片 `:109-110` 對 s35 B 的 selection 讀取路徑限定在 select 後的人工順序；
  與第 3 項已揭露的 smoke-A blind spot 一致，未見新增矛盾。

## 結論

五項：`CLOSED × 5`、`NOT-CLOSED × 0`。可按卡片的人工序列進入 F5 發射；保留
上述 GPU derived 首航、queue／OOM、partial OUT 與 jasmine 實機可見性等已明示的
runtime 殘餘風險。

STATUS: DONE
