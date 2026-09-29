# PILOT 拒啟 smoke 預檢對抗報告（F5 ①）

日期：2026-09-29。靶為 `PILOT-SMOKE-LAUNCH-CARD.md`，旁讀
`PILOT-BUILDER-REPORT.md`、`VERDICT-BUILDER-S2.md`、
`experiments/scratch_lacot_rollout.py` 與授權鏈原碼。

本查唯讀：沒有跑 smoke、Python、sbatch、測試或資料／checkpoint；沒有改靶、程式或
commit。只新增本報告。工作樹原先已是 dirty；本次沒有碰既有變更。現檔
`scratch_lacot_rollout.py`／`refine_service_builder.py` 的 SHA256 分別為
`b7d3a0d9…`／`c1d63cdd…`，與卡上前綴相符。

**VERDICT: NO-GO**

核心原因不是「拒啟本身不該跑」，而是本卡目前不能把「授權 gate 正常拒啟」和
「import／資料／checkpoint／環境炸掉」可靠分開；stale 也沒有批次級新鮮度錨。找不
到 gate exact marker 前，不能把三個非零 `exit_code` 當完成錨 2。

## 八格證據

| 格 | 判定 | 證據與攻擊結論 |
|---|---|---|
| 1. env import dry-run | **NO-GO／證據不夠** | lead 只在卡 `:17-19` 宣稱驗過 `torch`、`mujoco`、`ogbench`。實際 smoke 先做 `scratch_lacot_rollout.py:8-17` 的 `numpy`、`lacot.e_target`、`lacot.nf_head`、`lacot.model`、`lacot.refine_training`、`lacot.dev_eval`、`ogbench` 匯入；再在 `:2005-2010` 動態匯入 `lacot.refine_service_builder`，其後還牽 `refine_models`／`refine_quality`。這些不是「只 import torch/mujoco/ogbench」能證明的。`mujoco` 甚至不是 scratch 的直接 import。必須用卡上**同一個絕對 PY＋同一個 `env -i`**做 exact import／entrypoint probe，或以 log 中 metadata 前的成功標記證明；否則一個 `ModuleNotFoundError` 也會留下非零 rc。 |
| 2. 機器格 | **機制 PASS；完成錨 NO-GO** | `CUDA_VISIBLE_DEVICES=''` 在標準 PyTorch 語義下會使 `torch.cuda.is_available()` 為 false；原碼 `:37-38` 因而選 `device='cpu'`，後續 `.to(device)` 同走 CPU，且 `torch.cuda` RNG 分支 `:814-823` 也因 false 不取用。可是卡的四錨沒有要求 log 必須有 `device: cpu`、不得有 `device: cuda`；GPU 可見性若因目標 venv／站點差異不如預期，現行錨仍可能被人工誤放。收割必須把 `device: cpu` 設為硬 marker。 |
| 3. 卡上有誰／clean env | **CONDITIONAL，未閉合** | 目前明確保留 `PATH`、`CUDA_VISIBLE_DEVICES`、`OMP_NUM_THREADS` 與所有 `LACOT_*`；PY 是絕對路徑，所以 `VIRTUAL_ENV` 不應是必要條件；`scratch:11` 自己插入 repo root，所以 `PYTHONPATH` 不應是必要條件；資料也明確傳入 `OGBENCH_DATA_DIR`。`HOME` 對這條預期拒啟路徑從原碼看不是直接必要：`np.load` 用絕對資料路徑，`ogbench.make_env_and_datasets` 也明確傳 `dataset_dir`（`:2533-2537`），避免預設下載到 home。但 `HOME`、`LD_LIBRARY_PATH`、`TMPDIR`、locale 等清掉後的目標 venv／shared-library 行為沒有在 exact clean env 驗過；不能用普通 shell import 代替。不要為了遮錯盲目補回整個 host env，應先把 exact clean-env probe 做成前置證據。 |
| 4. 碟餘量 | **PASS（以 lead 事實為前提）** | 預期 gate 發生在 `scratch:1708-1723` 的 `_stage2_loop` 入口，早於 eval、`rollout_*.json` 與 `torch.save`；因此每 seed 的實際寫入主要是 shell 先開的 `smoke.log` 與 1–2 bytes 級 `exit_code`，三份合計是 KB～低 MB 量級，不是訓練 checkpoint 量級。`/archive` 5.2T 足夠。但若 gate 沒有被辨識、流程意外走過 `:2537` 及尾端 `:3825`／`:3876`，就會進入完全不同的 eval／JSON／ckpt 寫入面；卡目前只有「不得出現」文字，沒有執行後硬檢查。 |
| 5. 覆蓋面／stale | **部分 PASS；整體 NO-GO** | `exit 24` 在 for 內退出整支 wrapper，對三 seed smoke 是合理的 fail-closed 設計：不應在一顆 stale 時繼續湊一批。真正的雷是：沒有批次級 `run_id`／`batch_status`／wrapper rc 檔，也沒有要求收割時 wrapper 的 stale 結果一律丟棄。若舊的 `s0/s1/s2` 都已有看似合格的 log／exit_code，命令在 `s0` 以 24 退出而不改任何檔，四錨仍可能被舊檔湊過；若只在 `s1` stale，還可能留下新 s0＋舊 s1/s2 的混批。`ls -A` 出錯時 stderr 被吞掉，也把 unreadable dir 當成空目錄看待。 |
| 6. 路徑／依賴存在 | **NO-GO／清單漏項** | 卡 `:28-29` 只 `test -x "$PY"`；沒有在 wrapper 前置核 `test -r` checkpoint、data、也沒有核 output parent 可寫。程式在 gate 前已於 `scratch:45` 讀精確檔：`/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz`；卡只寫「ogbench 資料在地」，漏掉 basename、讀權限與 hash/size。另漏掉直接 `numpy`、repo `lacot.*` 模組、動態 `refine_service_builder` 鏈，以及 wrapper 的 `env`／`ls`／`mkdir`。若 gate 意外放行，後段還依賴 `ogbench.make_env_and_datasets` 與其環境資產；普通 import OK 也不等於該呼叫可用。 |
| 7. 參數拼寫／草稿 diff | **env PASS；「只改四處」說法不精確，wrapper 仍有缺口** | 對 builder 草稿 `:72-90` 做逐行 diff：所有 `env -i` 鍵和值均未改錯；完整逐字 env block 見下方。`LACOT_OUT_DIR` 只是同一字串改由 `$out` 展開。絕對 jasmine PY 與 `-B` 正確，後者吻合不寫 NFS `__pycache__` 的目的。可是 literal diff 不只四處：還新增 `PY`/`test -x`、`out`/`mkdir -p`、stale 分支、重導向、rc/exit_code/echo；這些雖大多合理，應完整列帳。且 wrapper 沒有 `set -u`／fail-fast 前置處理；`test -x`、`mkdir`、`cd` 失敗後仍可能繼續，最後只能靠人工看 log。 |
| 8. ETA × 死點／拒啟分類 | **NO-GO** | `<15 分`對「載入 full NPZ＋CPU 建模＋3.4MB teacher load＋立刻 gate」的量級或許合理，但卡沒有 jasmine exact wrapper walltime。卡說最貴死點是 ckpt schema，實際 gate 前更早的 `np.load`（完整資料）、大量 CPU module construction 與 import 也可能是主要成本；若 gate 未觸發，後段 eval/ckpt 是另一個成本級。最重要的是分類：`TrainingServices.authorize()` 先走 `QualityService.authorize_training()`（`refine_training:59-70`、`refine_quality:361-376`），目前真檔預期是 `ValueError: oracle qualification failed: ...generated...; A action NMSE >.1`；若 qualification 全過，才是 `qualified same-plan action teacher/exposure unavailable; R>0 refused`。兩者與 `ModuleNotFoundError`、`FileNotFoundError`、`invalid quality checkpoint`、schema／fingerprint／split mismatch、OOM 都是不同事件，但未捕捉的 Python `ValueError`／其他錯誤多半同樣寫 rc=1。更糟的是每輪 `echo "seed ... rc=$rc"` 讓 for loop 最後狀態回 0；wrapper 整體 rc 不能代表三 seed。`exit_code≠0` 絕不能單獨算錨 2。 |

逐字核對的 env block（builder 草稿 `:73-87` 與發射單 `:36-50`；唯一路徑重構是
`LACOT_OUT_DIR`）：

```text
PATH="$PATH" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench
LACOT_ENV=pointmaze-large-stitch-v0
LACOT_REFINE_QUALITY_CKPT=/archive/cymaxwelllee/refine_v3/exp1/pointmaze-s0/oracle.pt
LACOT_LEARNED_REFINE=1 LACOT_CONS=ema LACOT_EMA_M=0.996
LACOT_INTENT='' LACOT_INTENT_GUID_W=0
LACOT_SEED="$pilot_seed" LACOT_DATA_SEED=-1 LACOT_BOOT_SEED=-1
LACOT_ENC_OBJ=recon_ictr LACOT_W_ICTR=0.2 LACOT_ICTR_SIGMA=0.05
LACOT_DEC_START=hard LACOT_K=4 LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128
LACOT_STEPS1=0 LACOT_STEPS2=1 LACOT_WARMUP=0 LACOT_LR_SCALE=1
LACOT_LOAD_CKPT='' LACOT_CONT_TRAIN=0 LACOT_S1_FROM=''
LACOT_COND_DROP=0.1 LACOT_TEACHER_MIX=0 LACOT_EMA_W=0
LACOT_AMP=0 LACOT_COMPILE=0 LACOT_DIV_W=0 LACOT_GRPO_W=0 LACOT_LO_W=0
LACOT_BC_OWN=0 LACOT_BC_INDEP=0 LACOT_BON_N=0 LACOT_ORACLE_ARM=''
LACOT_EVAL_EPISODES=2 LACOT_DEV_EVAL=0 LACOT_PREREQ=0
```

## 錨 2 必須改成可判讀的契約

目前卡 `:72-73` 的「非零＋訊息含 A/generated」方向是對的，但不夠精確。每一 seed 至少要同時滿足：

1. `exit_code` 是預期 Python gate rc（目前應為 `1`，不是任意非零）。
2. log 先出現 `device: cpu`，再出現 `refine quality metadata:`；metadata 要逐 key 看
   `domain: pointmaze`、`source_kind: offline_dataset`、完整 `teacher_fingerprint`、
   `split_hash`、`readings_digest`、`exposure_available: False`。
3. metadata 後的 traceback 最後一行必須是目前真檔預期的 exact gate：
   `ValueError: oracle qualification failed: ...generated validity...; ...generated nominal_plus_alternative_coverage...; A action NMSE >.1`；若人工核對確認 qualification blockers 已全空，才可接受另一個 exact marker
   `ValueError: qualified same-plan action teacher/exposure unavailable; R>0 refused`。
4. 沒有上述 exact gate marker，即使 `exit_code=1` 也算 FAIL；`exit_code=24` 是 stale／wrapper
   FAIL；`137` 等 signal/OOM 也是 FAIL。三 seed 要逐個同樣分類，不能用三個舊檔補齊。
5. gate marker 之後不應有 backward、opt2 update、rollout JSON 或 ckpt；預期 output 只能是
   當次新開的 `smoke.log`＋`exit_code`。`refine quality metadata:` 不能只以「有印字」算，還要確認
   它出現在 gate 之前。

## NO-GO 的改哪裡

1. **改發射單 `:63-78` 的完成錨**：加入上面的 exact gate／crash 分類、`device: cpu`、
   metadata key 名稱與順序；把「非零」降為必要條件，不再是判決條件。把目前真檔的 expected
   blocker 或完整 fingerprint／readings digest／split hash 以可比對的完整值鎖定，不只留 SHA 前綴。
2. **改 wrapper `:26-56` 的批次契約**：在三 seed 前核精確 NPZ、teacher ckpt、output parent
   的存在／可讀／可寫；拒絕 unreadable 或 symlink 目錄；為整批寫一個明確的 start／status
   marker，或由收割器強制要求當次 wrapper 結果為 fresh 且 `exit 24` 一律丟棄。`exit 24` 整批停留著，
   但必須明文定義為整批 NO-GO。
3. **改 shell 控制流**：前置檢查失敗立即停；Python 命令以條件式捕捉預期 rc，不讓最後的
   `echo` 把整批錯誤變成 wrapper rc=0；收工輸出 batch summary，列出三 seed 的 rc 與 gate class。
4. **改依賴段 `:86-90`**：列出 exact NPZ、teacher 檔、`numpy`、所有直接／動態 repo module、
   output parent、`env/ls/mkdir` 與目標 venv 的 clean-env import 證據；不要把普通 shell 的
   `torch/mujoco/ogbench` import 當成 smoke import chain 已驗。
5. **改 CPU 完成條件**：保留 `CUDA_VISIBLE_DEVICES=''`（原碼判斷是對的），但把 log 的
   `device: cpu` 設為 hard gate；若要證明 clean env，直接在同一 `env -i` wrapper 下做，不要另外
   依賴 host env。完成後才可重新評估 `<15 分`。

## 自我懷疑

- 本工作站沒有 `/archive/cymaxwelllee/LaCoT/.venv`、jasmine 的 `/archive` 或真 production
  `oracle.pt`，所以沒有獨立驗證檔案存在、真 readings blockers、完整 hash、磁碟 5.2T、NFS／ext4
  與目標 venv 的 clean-env 行為；這些只按卡上 lead 事實標示為 conditional。
- 沒有執行 `torch.cuda.is_available()`、import、NPZ load、checkpoint load 或 smoke；CPU 結論是
  PyTorch 的標準 `CUDA_VISIBLE_DEVICES` 語義加原碼讀查，不冒稱實測。
- 我把 `HOME`／`VIRTUAL_ENV` 判為「從這條原碼路徑看不出必需」，不是保證第三方 `ogbench`／
  shared library 永不讀它；這正是 exact clean-env probe 尚缺的地方。
- `VERDICT-BUILDER-S2` 已證明 hook 會在 stage2 入口 fail-closed，且 `exposure=None`；但它也明記
  真 exp1 檔的 blockers、split hash、GPU 路徑尚未由檢察端驗，不能把 toy／普通 import 證據升格成
  jasmine production smoke 已可判讀。

STATUS: DONE
