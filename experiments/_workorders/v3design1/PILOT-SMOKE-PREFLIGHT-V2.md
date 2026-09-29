# PILOT 拒啟 smoke 預檢對抗報告（第二輪／v2）

日期：2026-09-29。受檢：`PILOT-SMOKE-LAUNCH-CARD.md` v2；對照上一輪
`PILOT-SMOKE-PREFLIGHT.md`、`experiments/scratch_lacot_rollout.py`、
`lacot/refine_service_builder.py`、`lacot/refine_quality.py`。

## VERDICT: NO-GO

v2 已實質修掉 CPU 完成錨與「任意非零 rc 即 PASS」兩個問題；RUN_ID、
`batch_status`、三 seed 同批檔也已進卡。但仍有可直接造成誤判或破壞批次契約的缺口：
實際 smoke 的 clean-env 不是 probe 的同一環境、wrapper 沒有真正 fail-fast、
stale 檢查仍可把 symlink／不可讀目錄當成新目錄，且完整 hash 錨沒有進入執行時或
完成檢查。故不能把三個 `rc=1` 判為「可判讀的 gate 拒啟」。

## 六 blocker 對照

| 上輪 blocker | v2 判定 | 對抗證據 |
|---|---|---|
| 1. import dry-run／完整 import 鏈 | **部分 PASS，未獨立閉合** | 卡上補述了 jasmine 的 `env -i` probe，且列出 numpy、torch、ogbench 與動態 builder 鏈；本工作樹的 `scratch`／builder SHA 前綴也分別與卡上 `b7d3a0d9…`／`c1d63cdd…` 相符，shell 語法檢查 PASS。但卡只給摘要與省略號，沒有可稽核的完整 probe 指令／stdout／exit；更關鍵是實際 wrapper line 53 用 `PATH="$PATH"`，不是 probe 宣稱的固定 `PATH=/usr/bin:/bin`，所以不能把 probe 直接套到發射環境。 |
| 2. CPU／完成錨 | **PASS（僅就此 blocker）** | v2 line 84 要求先有 `device: cpu`，並明定出現 `device: cuda` 即 FAIL；clean probe 也宣稱 `cuda.is_available()==False`。這補上了上一輪缺失的硬 CPU marker。 |
| 3. clean env／環境邊界 | **NO-GO** | `env -i` 確實清掉大部分 host env，但 line 53 把父 shell 的 host `$PATH` 帶入 child；它與 line 18 的 `/usr/bin:/bin` 實測不是同一契約。若要以 probe 證明 smoke，應把 wrapper 也寫死為 `PATH=/usr/bin:/bin`，或提供同一完整命令的證據。`set -u` 也不是環境或 command failure 的 fail-fast。 |
| 4. stale／覆蓋面／批次新鮮度 | **NO-GO** | RUN_ID 生成、START／DONE、seed `run_id` 同值與 STALE 排除，對「普通新鮮目錄」是進步；但 `[ -d "$out" ]` 會跟隨 symlink，沒有 `test ! -L`／`test ! -h`；`ls -A ... 2>/dev/null` 失敗時輸出被吞，unreadable 目錄可能被當空目錄。`mkdir -p`, `printf run_id`, `printf exit_code`, START／DONE 寫入均未逐一檢查，失敗仍可能落到 DONE／summary。這仍不能稱為 fail-closed 的批次契約。 |
| 5. 路徑／依賴／來源錨 | **NO-GO** | `test -r "$NPZ"`、`test -r "$CKPT"` 與 BASE 可寫檢查已補上，但只是存在／權限檢查。wrapper 沒有驗 NPZ 的 `sha256` 對 `dataset_hash`，也沒有驗 CKPT 的完整 SHA；卡只留 `ckpt sha256=de6338b6d513…` 前綴。程式實際在 scratch line 45 直接 `np.load`，未走 `lacot.refine_models.load_offline_npz` 的 manifest/hash whitelist（manifest 內雖有完整 `9add335e…`）。builder hook line 2006–2010 也沒有傳 `expected_fingerprint`，loader 的預設值是 `None`。因此卡上完整 split/readings/dataset 值目前只是人工參考，不能防止同路徑換檔或錯資料。 |
| 6. exact gate／crash 分類 | **部分 PASS，仍不足以 GO** | `exit_code == 1`、`0` 大 FAIL、`24` stale、`137` signal/OOM，以及「沒有 marker 即 FAIL」都已寫清楚；這一部分修對了。可是 primary marker 仍是 prefix 加三個 substring（`oracle qualification failed:`、generated validity／coverage、`A action NMSE`），不是完整 expected exception line；alternate marker 只靠「人工核對 blockers 全空」，沒有獨立可稽核證據。另 batch summary 只有每 seed rc，沒有 gate class；任何 Python 非 1 rc 仍被捕捉後繼續跑，最後照樣寫 DONE。 |

## 直接可重現的關鍵缺口

1. `set -u` 不等於 fail-fast。v2 line 69 對 Python 失敗只做 `rc=$?`，即使 `rc=2/137/24`
   也繼續下一 seed；line 73 無條件追加 `DONE`。這讓 wrapper 的成功狀態與批次內是否已 crash 脫鉤。應只容許預期 gate `rc=1` 後繼續，其他 rc 立即寫明確 ABORT／wrapper failure 並停止；所有 mkdir／marker 寫入也要檢查。
2. stale 判斷 line 46–50 沒有拒絕 symlink，也把 `ls` 錯誤吞掉。這正是上一輪要求明修的兩個條件，RUN_ID 本身沒有補上它們。
3. 卡 line 22–26 鎖了完整 `split_hash`、`readings_digest`、`dataset_hash`，但完成錨 line 84–88 沒要求 `dataset_hash`，fingerprint 只要求非空；line 101 的「對真檔」沒有提供可比對的完整 fingerprint 錨。完整值若不進 harvest gate，就不能證明載入的是指定真檔。
4. `scratch_lacot_rollout.py` 的 gate 確實在 `_stage2_loop`（line 1708–1713）早於後段 eval／JSON／ckpt；因此 output-only-if-gate 的推理合理，但正因如此，真正的通過條件必須由完整 marker、來源 hash 與批次寫入結果共同鎖住，不能只靠 `rc=1`。

## 已通過／非 blocker 註記

- v2 的 `device: cpu` 順序錨、metadata-before-traceback、marker 後不得有產物，方向正確。
- `bash -n` 對卡上 wrapper 通過；這只證明 shell 語法，不證明 jasmine 執行時 I/O。
- 按卡上 lead 事實，clean-env import probe 與真檔三個 hash 已在 jasmine 測過；本報告沒有把該宣稱升格成我方獨立實測。

## 殘餘風險／放行前必改

- 固定 wrapper 與 probe 的完整環境（至少 `PATH=/usr/bin:/bin`），保存完整 probe command、stdout、rc。
- 用明確的 symlink／readability 檢查與逐步寫入錯誤處理；非預期 Python rc fail-fast，summary 寫 gate class 與 wrapper status。
- 將完整 CKPT SHA、teacher fingerprint、NPZ SHA／`dataset_hash` 放入可執行或收割硬門，並把 `dataset_hash` 納入每 seed 完成錨。
- 將目前真檔預期 traceback 鎖成完整最後一行；alternate marker 只有在同批、可稽核的 qualification-blocker 清單全空時才可用。

## 自我懷疑

- 本機沒有 jasmine `/archive`、目標 venv 或 production `oracle.pt`，所以無法獨立重跑 clean-env probe、讀檔、完整 hash、真 readings blocker 或三 seed smoke；上述 NO-GO 主要由卡與本地原碼的可比對矛盾推出。
- 我接受 lead 提供的「jasmine import probe 過」作為事實性補充，但它沒有消除 wrapper `PATH="$PATH"` 與摘要證據的差異。
- symlink／I/O／同秒併發情境是靜態 shell 對抗，不是本機實跑；若部署平台另有強制隔離，仍應把該保證寫進契約，不能默認。
- 沒有執行 smoke、Python、sbatch、測試或 archive I/O；本輪只新增本報告，保留既有 dirty worktree。

STATUS: DONE
