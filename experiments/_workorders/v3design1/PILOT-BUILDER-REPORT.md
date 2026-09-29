在做什麼：新增 production Wφ checkpoint／quality builder 與 scratch 唯一 opt-in hook；本件只完成 quality 半邊，Aω／exposure 留空拒啟。

STATUS: DONE

此狀態只指實作＋CPU 驗證交付。**正式 R>0 仍拒啟，沒有 pilot 學習結果。真 ckpt 載入由 F5 smoke 驗。** 本機沒有讀 jasmine `/archive`，沒有跑 GPU／sbatch、安裝套件、commit 或改動 u 線及 exp2 既有行為。

## 交付與服務介面

- `lacot/refine_service_builder.py`：`load_quality_teacher`、`build_training_services`、`build_from_env` 與 `QualityTrainingServices` metadata 擴充。
- `experiments/scratch_lacot_rollout.py`：僅在第一個 `_stage2_loop(STEPS2)` 前新增一段 opt-in hook。須 learned refine、有訓練步數且 env key 存在才載入；不設 env 保留原先空位與 R>0 拒啟。空字串／壞路徑明確拒絕。R0 與 eval-only 不載入，即使 env 指向壞檔也不碰 I/O。
- `tests/test_refine_service_builder.py`：13 個 CPU 考場；自行用原 `train_offline --toy` 產 pointmaze／ant checkpoint，不依赖 `/tmp` 既有檔。
- 原始證據：本目錄 `evidence-pilot-builder/`，包含 tests、原考場回歸、開工快照差分、ownership、toy 指標及本件 scratch diff。

讀檔固定 `torch.load(..., map_location="cpu", weights_only=True)`，依原 exp1 格式還原 Wφ、obs-only、Aω、normalizers、support banks、calibration/readings。既有 `QualityService` 重算 joint artifact fingerprint（包含 Wφ、Aω、obs-only、normalizers、support/calibration）；另檢查 readings digest、split/source/domain、版本、horizon、缺失子空間與非有限數值。模型建構在 CPU `fork_rng(devices=[])` 中，載入不消耗 trainer 的 CPU RNG。驗證後才搬到 caller 指定 device，所有教師 frozen/eval。

`build_from_env` 只認明確的 `LACOT_REFINE_QUALITY_CKPT`；沒有預設路徑或 toy env 逃生門。`diagnostic_cpu=True` 僅為 Python CPU fixture 入口，保留 `toy_offline` 身份，仍拒絕 `authorize_training`。

服務經 `for_batch(state=s, goal=g, anchors=anc)` 綁本批，再以 `quality_factory(sampled, cond)(u)` 回 `QualityReport`。沿 scratch 原 `_q → _dec → _intent_inv`，沒有新解碼器。trainer normalized state/goal 先用 MU/SD 轉 raw；每個 W ensemble member／chunk 的新 raw state 再轉回 normalized 給 `_dec`；decoded path 轉 raw XY 給原 A→W 路徑。anchors detach/clone，沒有捕獲資料 action 作 label。Ant 使用同一建構介面，batch latent refinement 明確 `ant R>0 unsupported`。

quality forward 不包 `no_grad`，明確關閉巢狀 autocast 並用 fp32 teacher/reduction，保留 u→decoder→A→W→cost 的圖。回傳有限 fp32[B] cost、原 validity；全 invalid 拒絕，partial invalid 由原 objective 的 valid mean 處理。exposure 固定 `None`，沒有零 loss、假 action labels、改公式或覆寫原授權。

metadata 沿既有 `checkpoint_state()`／scratch checkpoint 寫入 `refine_training.metadata.quality_teacher`：joint teacher fingerprint、split hash、dataset hash、readings digest、原始 checkpoint 絕對路徑、domain／representation／source、frozen/eval、diagnostic 身份、W 數值資格與 training blockers。上層原 fingerprint／Rtrain／EMA 欄保留。metadata roundtrip 與 batch metadata 相同已驗；完整 scratch 保存／恢復仍待 F5。

## 修訂版三項的邊界

現存 `qualification_reasons` 修訂版為：分 horizon／子空間的 W NMSE ≤ .9×obs-only、paired CI 下界 >0、A action NMSE ≤ .1；另保留 generated validity／nominal+alternative coverage ≥.8。PI coverage 與 data validity 是健康欄，不擋資格。

EXP1-VERDICT 的 Wφ 判決只過前兩項；真 pointmaze Aω=.764 未過，generated gates 尚缺。故本件分兩層：**W 載入**驗前兩項及完整性，可建立待用 quality；**完整訓練授權**仍由未改的 `QualityService.authorize_training`＋`TrainingServices.authorize` 檢查三項、generated gates、source 與 exposure。metadata 原樣列出全套 blockers。即使未來 quality 全部合格，本版也仍因 exposure=None 拒啟。

「toy 自產合格 ckpt」在原 API 下不能解讀成 production 合格。實際 `train_offline --toy` 的 source 永遠是 `toy_offline`，且本次 Aω=.8197417；沒有改 source、偽造三項全過或補假 generated readings。CPU 正態證據是 **W 數值條件通過的 diagnostic checkpoint**，反態包括同檔 production 載入拒絕。此限制與真 ckpt 待 F5 明確保留。

## CPU 結果與兩態證據

現有 `.venv`／CPU；測試程序設 `CUDA_VISIBLE_DEVICES=''`。新測試 13＋原 wiring 16＋oracle 24＋原 R0 1＝54 tests PASS；另有開工 snapshot 對照 16/16。

| 考場 | 結果 | 證據 |
|---|---|---|
| builder 新考場 | 13/13 PASS | [builder-tests.log](evidence-pilot-builder/builder-tests.log) |
| artifact 兩態 | weights、bank、readings、split、版本、schema、domain 注入拒絕；原檔還原通過；fingerprint pin 錯誤拒絕 | 同上 |
| 資格兩態 | NMSE、CI、空子空間、缺 horizon、null CI、split 注入拒絕；還原通過；NaN 拒絕；PI/data validity 低不誤擋 W | 同上 |
| quality 梯度 | cost 全有限 fp32，valid 7/8，u grad norm=.494005859；教師無梯度；外層 CPU bf16 autocast 同型 | 同上 |
| 原 decoder 路徑 | 抽取真 `_dec/_intent_inv`，真小型 TrajDecoder＋hard start；u grad norm=.002749476，decoder／s_embed 無梯度 | 同上 |
| batch binder | 非單位 MU/SD、anchors 快照、兩批不同 goal、h16 各 member 每 chunk 新 state；與獨立 raw-coordinate report 比對 | 同上 |
| flag-off | 新測試 16 格 loss／grad／更新後權重／RNG 逐位元；env 缺失的 learned 路徑拒啟 | 同上 |
| 開工快照差分 | FSQ × intent × BC_INDEP × div，16/16 BITWISE PASS | [r0-snapshot.log](evidence-pilot-builder/r0-snapshot.log) |
| exp2 原 wiring | 16/16 PASS，含原六 mutation 與 CONS／intent guards | [wiring-regression.log](evidence-pilot-builder/wiring-regression.log) |
| oracle 原考場／R0 原考場 | 24/24、1/1 PASS | [oracle-regression.log](evidence-pilot-builder/oracle-regression.log)、[r0-model.log](evidence-pilot-builder/r0-model.log) |
| ownership | 移除新增 hook 後 scratch bytes 等於開工快照；model／四份 refine 來源及兩份原 tests hash 未變 | [ownership.log](evidence-pilot-builder/ownership.log)、[own-scratch.diff](evidence-pilot-builder/own-scratch.diff) |

本次 standalone toy：200 steps/model、hidden=64、seed=0、split seed=1729、T_CAP=128。W 的 h4 xy world/obs-only 比=.004623、CI lower=.477424；velocity 比=.015139、CI lower=.311866；h16/h32 同過。完整 hash／指標在 [toy-summary.json](evidence-pilot-builder/toy-summary.json)。暫存 checkpoint `/tmp/lacot-pilot-builder/toy/oracle.pt`，可清除，不作永久交付依賴。

```bash
# 重跑全部本件考場（會自產兩 domain toy artifact）
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover \
  -s tests -p test_refine_service_builder.py -v
# 單獨產 checkpoint：OUT 必須是新的目錄，原 trainer 拒絕覆蓋 report
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.refine_v3.train_offline \
  --domain pointmaze --toy --out /tmp/pilot-builder-toy-new \
  --steps 200 --hidden 64 --seed 0 --split-seed 1729 --t-cap 128 --threads 1
```

## pilot 發射單草稿（未執行）

**目前可驗的是 F5 載入／拒啟 smoke，不是可放行的三 seed refine 訓練。** 真檔的設定值為 `LACOT_REFINE_QUALITY_CKPT=/archive/cymaxwelllee/refine_v3/exp1/pointmaze-s0/oracle.pt`（jasmine 上）；本機 zeldajr 不探該 archive。使用固定 s0 教師，refine seeds 0/1/2，避免同時改 teacher seed。lady 選機理由、資料／checkpoint 可見性、partition／資源由 **⓪官填**，本件不代選、不投 sbatch。

步數建議〔拍〕：先三 seed 各 **STEPS1=0、STEPS2=1** 驗 checkpoint metadata 與拒啟；這一步不證 decoder 品質。Aω、generated gates 與 exposure 另案通過且實際接入後，再考慮 stage1=1500、stage2=2000（Rtrain=0/1/2/3 各500步）的短 pilot；不是本版可直接啟動的命令，也不是學習充分性判決。

以下是 **CPU 拒啟 smoke** 的 env 全文草稿（獨立乾淨 env，未執行）。F5 GPU 1-step smoke 的 GPU 配額與 CUDA_VISIBLE_DEVICES 由 F5／⓪官另填，不由本件啟動。

```bash
cd /home/cymaxwelllee/Projects/lacot
for pilot_seed in 0 1 2; do
  env -i PATH="$PATH" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
    OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench \
    LACOT_ENV=pointmaze-large-stitch-v0 \
    LACOT_REFINE_QUALITY_CKPT=/archive/cymaxwelllee/refine_v3/exp1/pointmaze-s0/oracle.pt \
    LACOT_LEARNED_REFINE=1 LACOT_CONS=ema LACOT_EMA_M=0.996 \
    LACOT_INTENT='' LACOT_INTENT_GUID_W=0 \
    LACOT_SEED="$pilot_seed" LACOT_DATA_SEED=-1 LACOT_BOOT_SEED=-1 \
    LACOT_ENC_OBJ=recon_ictr LACOT_W_ICTR=0.2 LACOT_ICTR_SIGMA=0.05 \
    LACOT_DEC_START=hard LACOT_K=4 LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128 \
    LACOT_STEPS1=0 LACOT_STEPS2=1 LACOT_WARMUP=0 LACOT_LR_SCALE=1 \
    LACOT_LOAD_CKPT='' LACOT_CONT_TRAIN=0 LACOT_S1_FROM='' \
    LACOT_COND_DROP=0.1 LACOT_TEACHER_MIX=0 LACOT_EMA_W=0 \
    LACOT_AMP=0 LACOT_COMPILE=0 LACOT_DIV_W=0 LACOT_GRPO_W=0 LACOT_LO_W=0 \
    LACOT_BC_OWN=0 LACOT_BC_INDEP=0 LACOT_BON_N=0 LACOT_ORACLE_ARM='' \
    LACOT_EVAL_EPISODES=2 LACOT_DEV_EVAL=0 LACOT_PREREQ=0 \
    LACOT_OUT_DIR="/archive/cymaxwelllee/refine_v3/pilot-builder-smoke/s${pilot_seed}" \
    .venv/bin/python experiments/scratch_lacot_rollout.py
done
```

預期輸出／完成錨：

1. F5 先核對 exp1 真檔與本批來源 hashes；log 出現 `refine quality metadata:`，domain=pointmaze、source=offline_dataset、fingerprint 與真 artifact 一致、split hash 前綴應對照 EXP1 的 `9aec9ac9`，`exposure_available=False`。
2. 隨即由原授權 gate 非零退出，訊息含 A action NMSE／generated blockers（或未來 qualification 全過後的 exposure unavailable），**不得出現第一個 opt2 成功更新、不得產新訓練 checkpoint 或 rollout 結果**。拒啟是本版預期，不是 pilot 學習通過。
3. F5 另用真 checkpoint＋真 frozen decoder／本批 state/goal 建 binder，驗有限 quality、coverage 與非零 u 梯度；全 invalid 必如實拒絕，不能放寬 cutoff。記錄 CPU/GPU dtype、fingerprint、split hash、cost 範圍、valid fraction、u grad norm、老師 grad=None。真 ckpt 載入由 F5 smoke 驗。
4. 後續可訓練 pilot 的完成錨才是三 seed 完整步數、保存 `generated-plan-v3`＋quality provenance、quality/exposure/coverage 曲線、有限梯度／成功 step 計數。現版 exposure 空位不可能達此錨，需另件補齊，不以刪 guard 達成。

## 自我懷疑／尚未證明

- 真 exp1 checkpoint 不在本機；production `offline_dataset` 正態與完整 OGBench process 尚無端到端證據。toy 正態不是 production 資格，沒有把這一缺口藏在 DONE 裡。
- Wφ 的 qualified consequence model 仍要經未合格 Aω 才能把 latent plan 變成 actions。這個 closure 現階段只能接線／preflight；exposure=None 與原授權共同防止把這種 surrogate 用於正式訓練。
- 新 loader 複用底層 classes／fingerprint 與 schema，但因不從 production import 實驗 CLI，重建步驟與 `train_offline.load_checkpoint` 有重複；未來 exp1 artifact schema 改動須同步更新並重跑本考場。
- 原 `_q` 的硬量化可能截斷 u 梯度。主草稿關閉 VQ/FSQ；CPU 真 decoder 考場只證連續 u。量化部署須另案驗 Jacobian，不冒稱已合格。
- 本件未驗 GPU AMP、scheduler、完整 resume 或真 generated coverage。metadata 包含 checkpoint 絕對路徑，搬路徑會觸發原 strict resume metadata 不符；需顯式遷移，不靜默忽略。
- scratch hook 在 stage2 前，沿用既有位置契約；若誤把拒啟 smoke 的 STEPS1 設大，可能先做無用 stage1。草稿故明設0；正式 gate 前置到整支程式開頭不屬本件唯一 hook 範圍。

STATUS: DONE
