# PILOT 拒啟 smoke 預檢對抗報告（第三輪／v3 限定四點）

日期：2026-09-29。受檢：`PILOT-SMOKE-LAUNCH-CARD.md` v3；對照二輪報告
`PILOT-SMOKE-PREFLIGHT-V2.md`。本報告只核對使用者指定的四個修訂點，不重開一、二輪已覆蓋的全文面向。

## VERDICT: NO-GO

四點中 #1、#3 CLOSED；#2、#4 尚未完全閉合。故 v3 目前不能以本四點限定審查放行。

## 四點判定

| 修訂點 | 判定 | 對抗核對 |
|---|---|---|
| 1. PATH 契約 | **CLOSED** | v3 line 62 的 wrapper 明寫 `env -i PATH=/usr/bin:/bin`；probe 說明 line 19 也是同一字串。二輪的 `PATH="$PATH"` 差異已消失，且發射命令在 loop 前沒有另一個 PATH 覆寫。這裡接受卡上 lead 對 probe 已在 jasmine 實測的宣稱；本報告未把它升格成本機獨立重跑。 |
| 2. fail-fast／寫入與 ABORT 分類 | **NOT-CLOSED** | v3 line 81–84 已正確做到 Python `rc != 1` 時寫一行 ABORT、`exit 50`、不進下一 seed；line 101–102 也把 40–50 類 ABORT 納入錨 1 的整批 FAIL 語義。但「各寫入步 `|| exit 45`」未閉合：line 51、53、55、82 對 `batch_status` 的 append 都沒有 `|| exit 45`；失敗時可能仍以 48/49/24/50 結束，或沒有記因。另 line 78 的 `smoke.log` redirection 沒有獨立的寫入失敗門。並且 line 33、40–45、47 的早期 ABORT 路徑沒有 `batch_status` 原因行，與錨 1 所寫「40–50、batch_status 記原因行」的整體契約不一致。 |
| 3. symlink／不可讀 | **CLOSED** | line 51 先以 `[ ! -L "$out" ]` 拒絕 symlink（含 broken symlink），再進目錄判斷；line 53 以 `ls -A "$out" 2>&1` 捕捉錯誤，將 `listing` 外顯並以 exit 49 中止；symlink 走 exit 48。二輪「跟隨 symlink」與「吞掉 ls 錯誤」兩個缺口均已由控制流修補。相關 `batch_status` append 自身未加 `|| exit 45`，已列入 #2，不重複把它算成 #3 的控制流缺口。 |
| 4. hash 執行門／錨 2、3 | **NOT-CLOSED** | 子項 A 已 CLOSED：line 43–44 在 Python wrapper 執行前比對 NPZ、CKPT，兩個常數均為完整 64 hex，失配分別 exit 46/47。子項 B 未 CLOSED：line 106 對 `dataset_hash` 寫成「**若 metadata 印出此欄**」，不是每一 seed 的硬完成條件；二輪要求的「把 `dataset_hash` 納入每 seed 完成錨」因此仍可被缺欄輸出繞過。子項 C 可接受為分階段閉環：line 111–115 要求三 seed traceback 尾行逐字一致，並明寫首航沒有先驗 exact 行、由收割官記錄全文及人工判讀，記錄值成為後續 exact 錨。這是誠實的首航政策，不冒充已有先驗；但實際收割仍須留下人工判讀明文，否則只有卡上程序要求、尚無證據。 |

## 殘餘風險

- `batch_status` 的 abort/stale/unreadable append 沒有 fail-closed 寫入檢查；I/O 失敗時狀態碼與原因行可能脫鉤。
- `dataset_hash` 已進入 hash 常數與文字錨，但完成錨的「若 metadata 印出」使每 seed 不必然驗證它；這是本輪仍阻擋 GO 的實質缺口。
- hash 是啟動前的 time-of-check；在 hash 比對與 Python 讀檔間若檔案可被替換，卡上沒有鎖檔或同一 file descriptor 的不可變交接保證。本輪只記錄，不以此另開全文 blocker。
- 首航尾行的人工語義判讀可作 staged gate，但它依賴收割官實際產出可稽核明文，且下一輪應把首航記錄升格為預先固定的 exact anchor。

## 自我懷疑

- 本次是對卡片與二輪報告的靜態逐行對抗；本機沒有 jasmine `/archive`、目標 venv 或 production `oracle.pt`，未獨立重跑 clean-env probe、hash 或三 seed smoke。
- 我將「若 metadata 印出此欄」解讀為不滿足二輪要求的硬錨，因二輪原文要求 dataset_hash 進入每 seed 完成錨；若 lead 能另以程式契約證明該欄必然輸出，應把 v3 卡上的條件句改成無條件要求後再複審。
- 我把 `smoke.log` 的 shell redirection 視為寫入步；即使排除這一項，line 51/53/55/82 的 `batch_status` 未接 `|| exit 45` 仍足以使 #2 NOT-CLOSED。
- 首航「無先驗 exact 行、記錄後成日後錨」我判為可接受的誠實分階段安排，但只接受其政策表述，不把未產出的 harvest 明文當成已存在證據。

STATUS: DONE
