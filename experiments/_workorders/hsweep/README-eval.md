這張單只交「能對任何一顆新 checkpoint 做同樣評估的程式」。證明：(a) 用新載入器載 s33，離線尺的三道閘與摘要數字跟既有那支（sha e5b8a49b…）**逐位元一樣**；(b) 新的 SG 臂每個 chunk 都是「flow 看最終終點抽 u → 解碼取終點當近子目標 → 動作頭的條件換成近子目標」，留痕查得到。判準＝§6 的自測全過（含殺手輸入真的 FAIL 過）。

hsweep-eval-v1-r2 已完成，E1–E7 全過。C 臂相對修改前已存檔的 patch 版，40 步全部既有欄位（含 RNG、chunks、actions、provenance 及浮點位元）diff=[]，1000 步 trace SHA 相同。SG 的動作頭吃遠條件與近遠條件相同兩個殺手均 FAIL；E7 舊 import 殺手也 FAIL。歷輪及本輪輸出原文都保留在 `selftest-eval-output.txt`。

本輪修改 `eval_common.py`、`harness_sg.py` 與本文件／測試原文。`selfsub_probe_ckpt.py` 完全不動，SHA256=`20c446c25ef7a7a8a7c613b6ef6f17a026b63aabc7638adc653a05a9cf56cc87`，快取鍵仍為（checkpoint SHA256, 本離線尺檔 SHA256）。PB、原 e5b8a49b 尺及五個 trainer 工單檔案前後 SHA256 diff=[]。

- `load_ckpt` 保留上一輪的 PB loader 逐行複製與參數化、source/dataset PIN、EMA、確定性設定及 import boundary；本輪函式未改。`eval_common.smooth` 逐行複製原尺，`smooth_anchor` 用相同 float64 平滑及錨定運算。逐行 diff=[]，兩組測試的錨定結果 bytes 完全相同；匯入共用檔不寫環境變數。
- rollout 不匯入離線尺；保留呼叫者提供的 CUDA 與 MuJoCo 設定。既有 CUBLAS/MUJOCO 預設只在變數未設時套用。每列與 summary 的 `execution` 記 `module_device`、`torch_cuda_is_available`、載入前當下的 `CUDA_VISIBLE_DEVICES`／`MUJOCO_GL`（未設為 null）。
- C 臂直接呼叫 PB `harness_detour.run_draw_detour(..., 'Q-C')`。`DetourView` 提供 Q-C 不使用的空 oracle table，不往模型加欄位。PB 入口既有的 flow 計數 hook 由原函式自行復原；本評測不改任何 PB 模組屬性。
- Q-C 使用 float64 保存環境位置，舊 PB `run_draw` 的輸出契約使用 float32。`c_compat_row` 只轉換結果表示法，沿用既有 `xy`／`trace_sha256`、flow/event 欄位；另保留 PB 原始 `detour_trace_sha256`。不改環境步進或 RNG。E4 比較全部舊欄位，沒有略過 RNG 或 hash 差異。
- SG 保留上一輪 chunk 迴圈，reset 改為 PB detour 的 inline Python／torch seed、cache／shuffle reset，collector 使用原 `make_collector_detour`。兩臂 plan 都走 `seeds_detour`；`runtime.seeds is common.seeds` 和 `harness_move3.seeds is common.seeds` 在入口、loader 前後、rollout 前後及 profile 事件斷言。E4 另用逐行 trace 驗證，23,253,995 次事件均保持身分。
- SG 每個 chunk 的 flow 看最終終點，解碼後做 9 點 valid 平滑、錨定首點至當下 xy、取末點 w，動作頭吃 `cond_near`。`audit.validate(require_distinct=True)` 要求每個 chunk 的近遠條件不同；實際 forward pre-hook／profile 驗證 head 輸入與呼叫次數。audit 用明確參數傳入，沒有模型上隱藏 audit 欄位。
- 題單完整 SHA256 釘：harvest-m3.json=`46de59ed8d46f7990e514c4c7fd9e52a4e1c0ecc50c9c31cf07504678fe421ce`；builder.json=`66a6957d7958729ba2775418cbcf2b86e5cec9571098db550d028d4760799846`。載入題目及模型前先驗，錯誤拒跑；原始題單不改。
- 每支 rollout 例外寫成失敗列，含 plan、`error=true`、`success=false`、`traceback` 原文、provenance、execution，stderr 亦輸出原文。其餘支繼續，summary 記總 `errors` 與每臂 × task 的 `errors`，全跑完只要有錯就 exit 1。載入／輸入／覆蓋保護失敗依既有入口拒絕；不把出錯支當成成功。

指令與參數（CPU 自測輸出在暫存目錄）：

```bash
cd /home/cymaxwelllee/Projects/lacot
export CUDA_VISIBLE_DEVICES=''
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
CKPT=/home/cymaxwelllee/Projects/lacot/results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt
CKPT_SHA=88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787
EVAL_TMP=$(mktemp -d /tmp/hsweep-eval-XXXXXX)

$PY -B experiments/_workorders/hsweep/selfsub_probe_ckpt.py \
  --ckpt "$CKPT" --ckpt-sha256 "$CKPT_SHA" --out "$EVAL_TMP/selfsub.json"
$PY -B experiments/_workorders/hsweep/selfsub_probe_ckpt.py \
  --ckpt "$CKPT" --ckpt-sha256 "$CKPT_SHA" --out "$EVAL_TMP/geometry.json" --selftest

# CPU 接線自測：同一題、draw 80、最多 40 步。
$PY -B experiments/_workorders/hsweep/harness_sg.py \
  --ckpt "$CKPT" --ckpt-sha256 "$CKPT_SHA" --arm SG --questions trap \
  --task 4 --episode 4 --draw 80 --max-steps 40 --out "$EVAL_TMP/SG.jsonl"
$PY -B experiments/_workorders/hsweep/harness_sg.py \
  --ckpt "$CKPT" --ckpt-sha256 "$CKPT_SHA" --arm C --questions trap \
  --task 4 --episode 4 --draw 80 --max-steps 40 --out "$EVAL_TMP/C.jsonl"
```

離線尺必填 `--ckpt`、`--ckpt-sha256`、`--out`；`--selftest` 跑原幾何殺手、不載入模型。可選 `--gate-evidence <先前 out.artifacts/gates-evidence.json>` 指定新 checkpoint 的證據；不指定時查原 s33 證據，其他 checkpoint 會走重算。指定證據檔不存在也走重算。真測量固定 CPU 及 `/home/cymaxwelllee/data/ogbench`。

rollout 必填 `--ckpt`、`--ckpt-sha256`、`--arm {SG,C}`、`--questions {trap,gate}`、`--out`。`--max-steps` 預設 1000、範圍 1..1000；陷阱 draw=80..83、易題 draw=80..81。`--task`／`--episode`／`--draw` 只過濾題庫，`--dataset-dir` 可指向經 PIN 驗證的副本。未寫死 hostname 或 /archive。CPU wrapper 在指定步數回報 truncated；程式保留正式 horizon，這輪未跑正式評估。

離線輸出保留 50 格 × 16 份、三道閘、殺手、分类與摘要，endpoint 描述不進判準；`<out>.artifacts/` 內圖表、證據及完整快取鍵照 r1。閘或殺手未達判準，依原尺停止。rollout JSONL 每支一列：保留成功、步數、最後格、軌跡 SHA、seed／RNG、actions、實際 provenance，每 chunk 記 step、當下 xy/格、w_xy/w_cell、sample_plan／flow／head 次數及條件 SHA。C 的 w 為最終 goal、conditions_differ=false；SG 條件相同即失敗。summary 的 `by_arm_task["SG:4"]={n,successes,errors}`，`n` 含出錯列。兩入口任何輸出或 companion 路徑已存在（含 symlink）於模型載入前拒絕，不覆寫。

E1 正 s33/EMA 載入與錯 SHA 拒絕；E2 三閘、50 格 × 16 份分類與摘要全部原欄位／float64 bits diff=[]；E3 task4/episode4/draw80 40 步共 10 chunks，每 chunk flow 恰一次、近遠不同、w 完整、head 吃 near，兩個 SG 殺手均 FAIL；E4 與預先存檔的 patch 版 C 對照，40 步全欄位及 RNG 逐位元一致、1000 步 trace SHA 一致；E5 兩入口與 companion 已存在均拒絕且 bytes 不變；E6 換查詢 checkpoint SHA 後實際重算 50 格、62 個 encode/decode 參照、三閘與三殺手，不借用舊 checkpoint 的閘；E7 以 CUDA_VISIBLE_DEVICES=0、MUJOCO_GL=osmesa 啟動，import＋完整 CLI 驗證走到 loader 函式本體前仍原樣，加回舊 import 後變成空字串／egl 並 exit 2，殺手 FAIL。E7 未載模型或初始化 GPU。

題单殺手以暫存副本添一個空白，各自 SHA 驗證 FAIL；注入 draw80 rollout 例外後，原 traceback 字串同時存在該列與 stderr，draw81..83 繼續完成，共 4 列、summary errors=1、exit 1。

原文位置可用 `rg -n 'hsweep-eval-v1-r2|E[1-7].*PASS|KILLER|ROLLOUT ERROR|SECOND PASS|n_tests=7|STATUS: DONE' experiments/_workorders/hsweep/selftest-eval-output.txt` 查。E4 修改前結果在 `/tmp/hsweep-r2-C-patch-40.json`、`/tmp/hsweep-r2-C-patch-1000.json`；新结果在 `/tmp/hsweep-r2-rollout-goh6c1sm/C-40.json`、`C-1000.json`。drivers：`/tmp/hsweep-r2-baseline.py`、`hsweep-e1.py`、`hsweep-r1-e2.py`、`hsweep-r2-rollout-test.py`、`hsweep-r2-error-retry.py`、`hsweep-r2-e5.py`、`hsweep-r1-e6.py`、`hsweep-r2-e7.py`、`hsweep-r2-second-pass.py`。

測試 driver 首次有兩個錯誤：錯誤列 fixture 的 sys.argv 夾帶 Python 的 -B／腳本路徑、E5 driver 在加入 PB 路徑前匯入 runtime。失敗原文已保留；只修 driver 後對相應測項重跑通過。實作與判準未為測試修改，E1/E2/E3/E4/E6/E7 通過後未重跑。

SECOND PASS: 核對 73 個凍結檔 SHA256 diff=[]，含 PB、原尺、離線尺與五個 trainer 檔；loader 授權參數還原後逐行 diff=[]，smooth 逐行 diff=[]。AST 無離線尺 import／PB 模組屬性賦值，運行中 seed factory 身分不變；E2、E6 保留原快取鍵與原數值，E4 舊版對照在改碼前落盤，沒有略過欄位。全部 10 種殺手確實 FAIL，例外列原文與非零退出已驗。

測試環境差異聲明：全部模型載入與 env.step 測試皆 CPU。E4 依 r2 指定額外跑同一題 1000 步作比較；其他接線最多 40 步，錯誤續跑 fixture 每支 4 步。未跑正式 rollout、未上 GPU、未送 Slurm、未驗 jasmine 路徑；CPU 結果不涵蓋 GPU 浮點、CUDA kernel 或 GPU RNG，沒有跨 CPU/GPU 位元相同保證。新 H checkpoints 尚未提供。E6 只改查詢 SHA、真模型仍 s33，其 fixture 不能當另一顆 checkpoint 的證據。所有測試 artifact 在 /tmp；未 commit、未裝套件、未刪檔。

| 成功路徑必須保留的輸出 | 允許變更的行為 |
| --- | --- |
| 離線尺全部原欄位、50 格 × 16 份、三閘、分母、摘要、實際 checkpoint/source/dataset SHA、正規化及相符鍵的閘／殺手證據 | 沿用 r1 的 checkpoint／輸出參數化及 endpoint/cache 描述；本輪離線尺與快取鍵一個 byte 不改 |
| rollout 的成功、步數、最後格、float32 軌跡 SHA、seed／RNG、動作、provenance、每 chunk 的實際接線證據 | C 改走 PB Q-C 並保留原 float64 hash 描述；SG 相同條件拒絕；新增裝置／載入環境、題單 SHA 保護、失敗列 traceback、summary errors 與非零退出 |

n_tests=7 n_pass=7 n_blocked=0 n_killers=10（SHA、SG head-far、SG equal-cond、離線 G1/G2/G3/上界、E7 old-import、trap SHA、builder SHA）
STATUS: DONE
