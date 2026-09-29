# F2 mod-oracle v3 第一批實作報告

STATUS: DONE

DONE 指本批「新增實作＋CPU smoke＋wiring」交付完成；**不表示實驗 1 科學資格通過**。未跑正式資料訓練、GPU、sbatch、環境執行、套件安裝或 commit。Aω、generated candidates 與正式 horizon 資格仍由後續 F5／實驗 1 處理。

## 交付檔案

| 新檔 | 內容 |
|---|---|
| `lacot/refine_models.py` | metadata／record 嚴格 schema、evaluation 欄位遞迴拒收、NPZ path＋SHA256 白名單、episode 先切分、兩種資料視圖、train-only normalization、Wφ×3／obs-only×3／Aω×3、CPU 訓練、freeze 與 fingerprint。 |
| `lacot/refine_quality.py` | calibration-only p95、90% PI、分 horizon／子空間讀數、episode-cluster paired CI、QualityReport、實際 executor 模型閉環、D→Aω→Wφ、same-plan exposure／gauge、資格拒啟、decoder shuffle preflight。 |
| `experiments/refine_v3/train_offline.py` | `--domain pointmaze/ant`、`--toy`、CPU-only CLI、報告／metadata／checkpoint 儲存與載入、forward/backward smoke。 |
| `tests/test_refine_quality_v3.py` | 19 項標準庫 unittest；不依賴環境中缺少的 pytest。 |
| 本檔與 `oracle-smoke/` | red／green 證據、toy logs／JSON／checkpoint、來源 hash 與共享工作區差異快照。 |

所有實作均為新增檔案。沒有改寫既有 rollout、model、actor、compose 或 proposal-v2 介面；沿用 `quality(u)->Tensor[B]` 的 closure、same-plan detached exposure、fp32 reduction／保留品質梯度的約定。新 `QualityReport` 提供 v3 的 validity／coverage 接口給後續 objective／actor 接線。

## 按 DESIGN-v3 (c) 落地

1. **隔離入口**：真資料只可透過明示 `OfflinePermit(path, dataset_hash, domain)` 讀取，resolve 後路徑及檔案 SHA256 均核對。NPZ 僅接收 observations/actions/terminals 及白名單原始輔助欄位。訓練入口再次驗證全部 record／metadata；頂層或 metadata 的 reward/rewards、success、oracle_index 等均拒收，未知欄位也拒收。
2. **metadata**：完整保存 `source_kind,dataset_hash,episode_split,parent_model_hash,candidate_id,noise_seed,iteration,domain,horizon,representation_version,eval_only`。raw records 的 parent/candidate 可為 null；model-generated 必須有 parent SHA256 和 candidate_id。原始 Wφ/Aω 的物理配對訓練明確拒收 model-generated labels；generated records 不能冒充真後果。toy 單獨標為 `toy_offline`，無法取得 production training 資格。
3. **Wφ**：完整起始 state＋原始動作，每次預測 4 個物理 transition，遞推到 4/16/32；各 horizon 直接對實際多步 targets 驗誤差，不能以 h4 通過代替 h32。缺未來的窗保留 mask／計數；不存在完整 train/calibration horizon 時不發該 horizon 門檻。以 train state 尺度正規化 displacement，按子空間等權重建。ensemble=3，獨立初始化、共用 minibatch，沒有宣稱 bootstrap ensemble。
4. **子空間量尺**：依 shat_probe FIX3 的定義，ant xy=[0,1]、yaw=qpos quaternion[3:7]、gait=[2,7:29]。這是 quaternion 座標重建誤差，不是 Euler yaw 誤差。pointmaze 額外維度另報 velocity。常數狀態、obs-only、shuffled-action 三種對照均保留。
5. **Aω**：三個 controller，輸入完整起始 state＋固定 T_CAP 點 XY path；target 是同一 row 的 `ACT[row:row+4]`。path 插值／goal clamp 沿用 make_batch；endpoint index 僅用於 audit，未輸入模型。沒有逐步 inverse dynamics，也沒有拿相鄰插值點差分作 action。
6. **calibration**：normalized state/action nearest-neighbor bank 只來自 train（有界 512 個 complete windows／h），p95 support 與 p95 W ensemble disagreement 只來自 calibration。另支援明示 frozen reference NLL module 的 per-dimension NLL p95；A ensemble disagreement 也從 calibration 鎖定。component scales、PI radius 同樣只用 calibration；fit 後再次 fit 會拒收。checkpoint／fingerprint 綁 model weights、normalizer、dataset metadata、reference、bank、thresholds、scales、PI 及 A uncertainty threshold。
7. **品質與接線**：normalized arrival＋hold＋nonnegative behavior-support penalty 等權重；support 是密度／鄰近度限制，不是成功率。實際 executor callback 每個 chunk 重新讀取每個 ensemble member 的預測完整 state。pointmaze 同計畫 D→Aω→Wφ 保留對 plan/state 的圖；exposure label detach，梯度只到 head／其捕捉的 cond。老師 freeze＋eval，outer `.train()` 不會切回訓練狀態。
8. **拒啟與未知值**：版本／fingerprint 不符、未校準 horizon、toy／未合格 teacher 都拒啟。ant R>0 拒收；ant A 訓練僅供 smoke 診斷。個別 invalid 的 report cost 不歸零；strict v2 closure 以 inf 表示 invalid，後續 objective 應使用 report mask 並記 coverage，全 invalid 明確拒絕。exposure 無 valid labels 時傳回零 loss **和** `valid=false/reason`，不是靜默有效。gauge 缺值用 null。

## 測試兩態證據

- [red.log](oracle-smoke/red.log)：在新增入口仍為 permissive scaffold 時先跑；reward／success／oracle_index 注入造成 3 個預期 FAIL。當時 action-blind 的另 1 個 FAIL 是 scaffold 恆假斷言，**不能算 witness 證據**；第二輪已改為真 Wφ witness 並留下 action-blind mutant 兩態。scaffold 與當時測試保存於 `red-models.py.txt`、`red-tests.py.txt`，不是把 import error 當 red。
- [red-trained.log](oracle-smoke/red-trained.log)：已訓練 obs-only 取代 W，跑真正的 held-out shuffled-action 測試；h4/16/32 × xy/velocity 共 6 個預期 FAIL，paired degradation CI 下界為 0。
- [red-controls.log](oracle-smoke/red-controls.log)：最終測試的同一真實 `train_models` 路徑，在記憶體中繞過 schema guard 後，7 個 injection 子檢查為 **4 FAIL＋3 ERROR**；同一 shuffle 測試換 trained obs-only 後另 6 FAIL。重跑腳本在 `witness-controls.py.txt`，清楚標為 negative controls，不改 production 檔。
- [green-initial.log](oracle-smoke/green-initial.log)：中途測試抓到 freeze 後殘留舊 parameter.grad；已修正 freeze 清除舊 grad，保留這筆失敗。
- [green.log](oracle-smoke/green.log)：最終 **19 tests，全部 OK**，約 1.9 秒。

覆蓋：eval 欄位的實際 loader／training API 拒收、**原版 11 個 metadata 欄位**缺一拒收、來源／parent 規則、NPZ hash／path／field 白名單、跨 split episode／實際 row 無交集、原始動作與插值配對、train-only statistics、calibration-only p95 與 lock、masked short windows、held-out shuffle 顯著劣化、QualityReport schema、invalid 不作零品質、CPU bf16 fp32 reduction、plan/state backward、老師無 grad、closed-loop executor 每 chunk 重讀 state、same-plan exposure 梯度歸屬、gauge RNG／train mode／null、NLL p95、fingerprint／provenance 拒收、checkpoint roundtrip、candidate coverage、constant-decoder witness。

## CPU smoke

正式驗收產物：

- [pointmaze-final.log](oracle-smoke/pointmaze-final.log)／[report.json](oracle-smoke/pointmaze-final/report.json)／[metadata.json](oracle-smoke/pointmaze-final/metadata.json)
- [ant-final.log](oracle-smoke/ant-final.log)／[report.json](oracle-smoke/ant-final/report.json)／[metadata.json](oracle-smoke/ant-final/metadata.json)

每域 50 個代數 toy episodes，train/calibration/test=30/10/10 episodes；train 最多 1024 windows，calibration/test 各 450。每個 model 200 updates、batch 64、hidden 64、ensemble 3、T_CAP=128、CHUNK=4、CPU 1 thread。這不是環境 simulator，也不是真 ant 動力學；不能解讀成 domain 結果。

| domain | W train probe loss（前→後） | A train probe loss（前→後） | elapsed |
|---|---:|---:|---:|
| pointmaze | 7.834375 → 0.007112 | 1.262417 → 0.635704 | 2.49 s |
| ant | 5.423854 → 0.018360 | 1.318211 → 0.879515 | 3.73 s |

obs-only 對照 loss 也下降。兩域 action→quality backward、teacher freeze、checkpoint roundtrip 通過；pointmaze plan→D→A→W→quality backward 通過；ant latent refine 拒收通過。JSON 包含完整 `cost,components,valid,uncertainty,support,teacher_fingerprint`。

管線能產出實驗 1 要求的讀數：各 h／子空間 NMSE、obs-only／constant／shuffle 誤差、配對改善及 shuffle degradation 的 episode-cluster CI、90% PI coverage／raw 與 normalized width、data validity／拒收計數、A action NMSE、A uncertainty coverage。toy A held-out NMSE 分別為 **0.798916／1.056742**，不冒充 ≤.1；pointmaze toy 的 data validity 也沒有全數達到 .9。程式如實列 qualification_blocks，未因讀數不好調整門檻。generated validity／nominal＋替代候選 coverage 在尚無真候選時為 null；已有 `candidate_coverage` 可接後續候選輸出。

重跑（output 必须是新目錄，避免覆寫證據）：

```bash
.venv/bin/python -m unittest discover -s tests -p test_refine_quality_v3.py -v
.venv/bin/python -m experiments.refine_v3.train_offline --domain pointmaze --toy --out /tmp/f2-pointmaze-new
.venv/bin/python -m experiments.refine_v3.train_offline --domain ant --toy --out /tmp/f2-ant-new
.venv/bin/python < experiments/_workorders/v3design1/oracle-smoke/witness-controls.py.txt
```

真資料入口是 `--domain ... --data /明示路徑.npz --dataset-hash SHA256 --out 新目錄`；沒有隱式資料搜尋、evaluation input、GPU device 選項。這個入口本批沒有用正式資料啟動。

## 最容易寫歪的三處：lead 指定自查

1. **episode 不跨 split 重疊窗**：先把 episode 分 train/calibration/test，再抽 row。future 到 terminal 前停止，path interpolation 的 hi index 夾在同窗終點。測試枚舉各 split 的實際使用 row 集合，確認交集為空，並刻意注入 episode 重疊確認拒收。
2. **normalization 僅 train**：state mean/std 由 train record 起始 states 計算；action mean/std 由 train 的真實 masked actions 計算（重疊窗可能重複加權，但無 held-out 資料）。測試直接核算兩組統計，再大幅改動 test state/action，確認統計與 calibration artifact 不變。quality component 尺度用獨立 calibration，和 train normalizer 明確分開。
3. **A target 是原始 action sequence**：資料測試逐窗比對 `ACT[row:row+CHUNK]`，同時核對插值 path；A forward 只接 state/path，不接 path_end、未來速度或時長。h=4/16/32 是 W 的物理步數，T_CAP 是 path 取樣點數，兩者沒有互換。

## 自我懷疑與實際邊界（非空）

- 真 pointmaze 的 observation 可能只有 XY；本 toy 有 4 維 state，不能證明真資料 action 可辨識。A held-out NMSE 本來就未達門檻，必須留待實驗 1，不能靠 path 差分補答案。
- ensemble 的獨立初始化和 empirical interval calibration 不能保證反事實準確；同 episode 重疊窗也不具逐窗獨立性。CI 因此按 episode resample；PI 是 marginal coverage 讀數，仍須真 held-out episodes 驗證，不宣稱長程聯合 coverage 保證。
- ant toy 的 29 維是線性代數 fixture；quaternion 不代表真姿態。gait displacement／quaternion norm 只是 diagnostic，真 fall/freeze／physical feasibility 沒有標籤時保持 unknown，不能以這些欄位取得安全或 goal-selector 資格。
- generated flow／own iterates、實際 codec／executor、live-head exposure gap 尚無本批資料。服務提供 callback、gauge、coverage 與資格拒啟；沒有假造候選、成功率或 deployment evidence。真 decoder 要沿用 caller 的 `_dec/_intent_inv`，由 caller freeze＋eval 並跑 pairing preflight；本批沒有改主線建構時序。
- 沒有 flow 的 CLI 使用 train-neighbor support。optional NLL module 必須由後續 caller 提供同表示的 frozen reference；現有主線 latent flow 不能在未寫 adapter／未驗表示時直接當 action density。CLI checkpoint 是 neighbor 版本；帶 reference flow 的 checkpoint 載入會要求其 explicit adapter，缺失則拒收。
- strict `quality(u)` closure 對 invalid 用 inf，避免其變成便宜好樣本；後續 objective 要透過 report valid mask／coverage 正確處理，不能把目前 smoke 當三個 production caller 已接好。

## 共享工作區稽核

本輪開始時 `experiments/scratch_lacot_rollout.py` 已有未提交修改。本次任何 patch／寫入命令都只針對本批新檔；沒有編輯該主檔。

中途 tracked diff 與起始快照一致；最後稽核發現主檔另有並行變更，因此不宣稱整個共享 repo 的 diff 完全不變，也不回復他人的內容。前後快照：`oracle-smoke/preexisting.diff`（SHA256 `cdc6241773575960517d561ad0490f3e79a8689e403d944037ece72f306dead8`）與 `oracle-smoke/concurrent-final.diff`（SHA256 `541d644d799c977571e447b3b5bfe688de9f4c79c9adc568417e28b5ab023ba1`）。原始稽核 assertion failure 留存脈絡；後續稽核記錄此為共享工作區差異，不是本批需要修改主檔的 BLOCKED 情形。

最終程式可載入兩域 smoke checkpoint；`gym/gymnasium/ogbench/mujoco` 均未出現在本次訓練模組的 imported modules。來源 SHA256／驗證詳見 `oracle-smoke/final-audit.log` 與 `oracle-smoke/final-audit-complete.log`。
