# PILOT 拒啟 smoke 預檢對抗報告（第四輪／v4 限定 #2、#4）

日期：2026-09-29。受檢：`PILOT-SMOKE-LAUNCH-CARD.md` v4。本輪只核驗三輪
尚未閉合的 #2、#4；不重開已 CLOSED 的 #1、#3，也不重做一、二輪覆蓋面。

## VERDICT: GO

| 修訂點 | 判定 | 對抗核對 |
|---|---|---|
| #2 fail-fast／寫入與 ABORT 分類 | **CLOSED** | 四個 in-batch `batch_status` append 都是 `|| exit 45`；`smoke.log` 有獨立的預開寫入門；pre-batch 與 in-batch 的 batch_status 要求已按可觀測性分界。 |
| #4 `dataset_hash` 硬錨／程式契約 | **CLOSED** | loader 成功返回前必須有合法 64-hex `dataset_hash`；builder provenance 無條件放入；metadata print 只在 builder 返回後執行，因此到達該 print 行時欄位必然存在。 |

本輪總判定 **GO**，意義是 #2、#4 的預檢修訂閉合；不等同於三 seed smoke 已經實跑成功。

## #2：CLOSED

### 寫入與 redirection

- `batch_status` 的 START 建立（launch card L50）、SYMLINK／UNREADABLE／STALE／unexpected-rc 四個 in-batch append（L54、L56、L58、L86）均在寫入失敗時轉 `exit 45`。
- 相關產物寫入也保持 fail-closed：seed 目錄、`run_id`、`smoke.log` 預開、`exit_code`、DONE append 分別在 L62–64、L83、L91 有失敗出口。
- L64 的 `: >"$out/smoke.log"` 在 Python 啟動前獨立測試 create/truncate 可寫；失敗時先嘗試寫 `ABORT LOGWRITE`，再以 `exit 45` 結束。故先前「bash output redirection 失敗回 rc=1、被當成 gate 拒啟」的直接路徑已被封住；L82 的 Python command 只有在預開門成功後才會執行。
- 我另以 `bash -n` 對卡片中的完整 bash code block 做語法檢查，結果 PASS。

### pre-batch／in-batch 分界

L95–100 與錨 1（L107–112）現在明確承認：40–47 的 pre-batch failure 可能發生在
`batch_status` 建立前，或建立本身已不可信；這時硬要求 `batch_status` 原因行是
自相矛盾，stderr／sbatch log 才是可靠原因面。這個修訂可接受地消除了三輪所指的
「早期 ABORT 沒有 batch_status 行」契約矛盾。

對於正常的 in-batch ABORT（24／48／49／50），原因 append 成功時必先留下
`batch_status` 行；append 本身失敗則立即轉 45，不能繼續跑、不能寫 DONE、也不會
冒充 gate rc=1。這是合理的 fail-closed 例外：記錄器本身失效時，不能再硬要求一個
無法可靠寫出的原因行。

因此我接受 `45` 為 wrapper-I/O failure sentinel，而非 gate 拒啟碼。它在 START
建立失敗時可屬 pre-batch，也可能在 START 已成功後因後續寫入失敗而出現；收割時
不得只按數字範圍把所有 45 當成正常 pre-batch。這是操作解讀注意事項，不改變
「寫入失敗即整批 FAIL」的 fail-closed 性質。

## #4：CLOSED

程式契約逐段核對如下：

1. `Calibration` dataclass 將 `dataset_hash` 列為必要欄位（`lacot/refine_quality.py` L88–105）。
2. `load_quality_teacher` 以 `Calibration(**saved["calibration"])` 建構 calibration（`lacot/refine_service_builder.py` L109–115）；缺欄無法成功返回。其後 L120–123 對 `dataset_hash`、`split_hash`、`teacher_fingerprint` 逐欄要求字串、長度 64、且每字元為小寫十六進位；不合格即 raise。
3. `build_training_services` 只有在 loader 成功返回後才到 provenance 建構處（L176–187），而 provenance dict 在 L187–195 無條件寫入 `dataset_hash=cal.dataset_hash`。
4. `QualityTrainingServices.metadata()`（L152–158）把整份 provenance 放入 `quality_teacher`；所以該欄不會因 print 層的條件分支而消失。
5. smoke caller 在 L2005–2010 呼叫 `build_from_env`，L2011 才執行 `print("refine quality metadata:", REFINE_TRAINING_SERVICES.metadata(), ...)`。因此「走到 metadata print 那行」的必要條件就是 builder 已成功返回，屆時 `dataset_hash` 必然存在且已通過 64-hex 驗證。

這使 launch card L113–121 的 `dataset_hash` 已是無條件硬錨；缺欄不再能以
「metadata 沒印所以略過」繞過。即使 loader 因缺欄或非法值在 print 前失敗，完成
錨仍會因 metadata／traceback marker 不成立而 FAIL，不會被當作成功 gate 拒啟。

## 殘餘風險

- `: > smoke.log` 與 L82 的再次 `>` 之間仍有一般檔案系統 TOCTOU 窗口；若檔案
  在兩個命令間被替換、unlink 或權限突變，第二次 redirection 理論上仍可能回 rc=1。
  本輪接受預開門對原先的 deterministic write-failure 偽裝缺口已足夠，但未宣稱
  它是 descriptor-level 原子保證。
- `exit 45` 是跨階段的 wrapper-I/O sentinel，與「40–47=pre-batch」的純數字讀法
  有重疊；收割官必須把 45 視為 wrapper failure、整批 FAIL，不得把它判成 gate
  拒啟或要求不存在的 batch_status 原因行。
- 本機沒有 jasmine `/archive` 目標 venv／真實 `oracle.pt`，本報告沒有實跑三 seed、
  沒有獨立重驗 production metadata 值；本結論是卡片 shell 控制流與本地 builder
  原碼的靜態契約核對。
- 既有 hash check 與 Python 實際讀檔之間的檔案替換風險沿用前輪記錄，本輪不把它
  擴大成新的 #4 blocker。

## 自我懷疑

- 我把預開 `: >` 判為足以閉合本輪所指的 redirection 偽裝路徑；若團隊把
  「獨立門」定義成必須覆蓋並發替換／第二次 shell redirection 的所有競態，#2
  應改用已開啟的 file descriptor 或等價不可變交接，現判定就會過寬。
- 我接受卡片對 45 的文字例外，因其明寫 wrapper 寫入失敗時 fail-closed；若 lead
  要求 ABORT 類別必須是互斥的數字集合，則卡片還需把 45 明列為第三種 phase-
  independent I/O failure，否則不能只靠「40–47」做機械分類。
- #4 的程式契約證明的是「到達 print 時欄位必然存在且格式合法」，不是實際
  checkpoint 一定含本單指定的 `9add335e...` 值；該值仍須由 wrapper hash gate
  與收割硬錨共同驗證。本輪只核驗契約存在性，未冒充真檔實跑證據。

STATUS: DONE
