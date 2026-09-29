在證明什麼：16pp 包絡、J-signal 與 v3 是可追溯但不混域的帳；並把「teacher/selector 不靠 evaluation labels」落成可跑的反洩漏測試設計。

# 實驗 0：封口徑與反洩漏

**狀態範圍**：本輪只讀既有結果並以 CPU 重算逐題布林交集；沒有新模擬、GPU、訓練或程式修改。反洩漏兩測是後續實作規格，**尚未有可通過的 production loader／selector，未宣稱測試 PASS**。判準依 [DESIGN-v3.md](DESIGN-v3.md) (a)「不可混帳」及 (d)「實驗 0」；5.5% 原轉引見 [CONVERGENCE.md](materials/CONVERGENCE.md) 14–17 行。

## 1. 同題 leg-1 包絡重算

**結果：已由逐題 evidence 重建 11/200＝5.5%。** K32 schedule 失敗 43/200＝21.5%，K32 nn 失敗 54/200，G16K16 schedule 失敗 48/200；三法同題皆失敗 11 題。逐題事後至少一法可到達首路標為 189/200＝94.5%；相對 K32 schedule 的失敗率差為 **16.0 個百分點**。這是固定三個完整候選策略的事後包絡，尚無事前 selector，亦不是 16 個候選或 16pp 的已實現改善。

**題 key 與重算方法**：讀 `teacher_relay_byleg_summary.json` 的 `results.schedule/nn.per_task` 與 `teacher_relay_g16k16_summary.json` 的 `results.schedule.per_task`；以 `(ti, episode)` 作 key，不用檔內列序偷偷 join。三臂各 200 個 key、無重複，key 集與 `ti→episode` 對應逐項相同；`seed=20260908`、`antmaze-medium-stitch-v0` 的 val episode、`DELTA_SUB=7.5`、`rho=1.875`、共同 ruler、跳過的 episode 386、路標總數 595 也一致。leg-1 在 200 題均被嘗試；`legs_reached == 0` 即首段未進 `rho`，與 K32 byleg `leg_rows[m=1].reached == false` 的 43／54 筆逐題一致。原始 K32 `teacher_relay_summary.json` 的 schedule／nn `per_task` 與 byleg 版逐項相同。這裡不用 `completed`（整條多 leg 任務）、不限時全 leg 平均，亦不用 1.25×`N_data/N_table` 的限時尺。

可覆核的純讀重算式（Python 3）：

```python
import json
from pathlib import Path
w = Path("experiments/walk_verify")
k = json.loads((w / "teacher_relay/results/teacher_relay_byleg_summary.json").read_text())
g = json.loads((w / "teacher_relay_g16k16/results/teacher_relay_g16k16_summary.json").read_text())
def keyed(rows):
    out = {(r["ti"], r["episode"]): r for r in rows}
    assert len(out) == len(rows) == 200
    return out
s = keyed(k["results"]["schedule"]["per_task"])
n = keyed(k["results"]["nn"]["per_task"])
f = keyed(g["results"]["schedule"]["per_task"])
assert s.keys() == n.keys() == f.keys()
fail = {key for key in s if all(d[key]["legs_reached"] == 0 for d in (s, n, f))}
assert len(fail) == 11
```

共同失敗 episode（升序）：`62, 99, 169, 225, 271, 381, 414, 424, 444, 484, 495`。題集指紋是 **按 `ti` 排序的** `[(ti,episode), ...]`，以 `json.dumps(..., separators=(',',':'))` UTF-8 編碼後 SHA256：`ebb00bb2ba407fcbedaf0bbf6b74a800e2f0424193e2263c87d77efc88990be1`。這是輸出題 key 指紋，**不是** OGBench val NPZ 的內容 hash；原資料檔未在本輪求內容 hash。

| 三法 variant | 候選來源／每步所見 | 實際 codec checkpoint SHA256 | 逐題 evidence SHA256 |
|---|---|---|---|
| `K32 schedule` | 同一題教材 episode 的第 `k` 個 4-step action chunk 編真字；decoder 讀 live obs，碼序照時間表 | `experiments/walk_verify/results/p0_dict_v1_L4K32_50k.pt`：`b6df04119899942abb7d9d073bdf6a65915eaf38d1dfeb6459309300b25bae53` | `teacher_relay_byleg_summary.json`：`51f24f68fe8bafcd4c6c6aa91e214e19be62be67f412b22f30bd2abbf9b29dd3` |
| `K32 nn` | 同一題教材軌跡全部 4-step 格點座標中，按 live xy 找最近格點，再取該格點真字；decoder 讀 live obs | 同上；K=32 單碼本，與 schedule 共用 checkpoint | 同上，`results.nn.per_task` |
| `G16K16 schedule` | 同題第 `k` 個 action chunk 真字；16 組各 K=16 的完整 tuple，decoder 讀正規化 live obs | `experiments/gk_scan/ckpt/G16_K16.pt`：`b201a05088410820b6a3c083a3f41bc0897c9c36c413d23beed0c4ec6d5af4ab` | `teacher_relay_g16k16_summary.json`：`45e8559d0b8f31335074b864d48b755f203595ddfaf78b4b5180aa40c8fecb45` |

兩份 evidence 均在 `experiments/walk_verify/{teacher_relay,teacher_relay_g16k16}/results/`；K32 原版 `teacher_relay_summary.json` SHA256 `694a4ad7951c3ba6d87763be38463b9902f4ae8d15c29a4d74ebb5564e7d4ef6`。`lacot-gensft` repo 的 K32 byleg／原版 JSON 與 K32 checkpoint SHA256 與本 repo 相同；G16K16 逐題檔在本 repo 找到。兩個 `_raw.npz` 只存合併速度／高度／動作差等陣列，**沒有題 key 或 success 位元**；逐題重算依上述 JSON 的 `per_task`／K32 `leg_rows`，不可拿 NPZ 當交集原始表。三法本輪所用 evidence 完整，故此 11/200 **不標「轉引，尚未重建」**；但若要求從模擬逐步姿態獨立重新評分，G16K16 此 NPZ 仍缺逐步題 key／位置軌跡，這一層尚未重建。

原尺：首段由同一條 val episode 弧長的第一個 7.5 路標構造，`rho=1.875`；從原 qpos/qvel 起步，4-step chunk，該題 `T//4` chunk 上限（此 200 題 leg-1 `horizon=200` 物理步）。200 步內 **任一時刻** 依序到達首路標算成功；不是 `ruler_pack.json` 的 `N_table` 限時成功。原 runner 仍載入 `experiments/walk_verify/results/ruler_pack.json`，SHA256 `88afcda331701e46906f625b14b6bb9f84ad3fd9888b17312a5584aae0c341fa`，主要供 `N_table` 診斷。程式錨點：`run_teacher_relay_byleg.py:60-91,103-140,225-275`；`run_teacher_relay_g16k16.py:73-100,137-187,417-423`。

## 2. 帳面封口：禁止跨格相減／移植

| 帳 | 域、題集、原 ruler／horizon | 候選來源與可說的結論 | 禁止混格 |
|---|---|---|---|
| **ant 16pp 包絡** | `antmaze-medium-stitch-v0` val 的 200 個零漂移起點，**leg-1**；首路標 `rho=1.875`，每題 200 物理步，不限時 first-hit | K32 schedule／K32 nn／G16K16 schedule 三個**完整策略**；都從本題教材未來 action 或座標取字，屬 `privileged_relay`。21.5%→事後共同失敗 5.5%，差 16pp | 不是部署可得候選庫、不是事前挑中率；不能把 G16K16 的 16 組寫成 16 候選，不能拿全 leg／限時尺代換 |
| **J-signal 歷史判決** | `pointmaze-large-stitch-v0`；每 checkpoint 5 官方 task×40 episodes＝200，s33/s35 兩顆；官方 `R1` reached／`MAXH=1000` | frozen flow 計畫；noclimb vs post-F6 GeoEnergy gradient climb vs `SEL_N=8` 以 GeoEnergy 最低選樣（`BON_N=0`）。s33 `.350/.050/.325`，s35 `.395/.030/.405`（依序三臂） | GeoEnergy@8 不是 GrpoReward BoN@8，更非 ant 教材 relay；pointmaze 的 ±pp 不能記作追回 ant 16pp |
| **今日 u 抽樣對照（待跑）** | `pointmaze-large-stitch-v0`，s33 後 s35；每 seed 預定 5×40 題，每題 8 次**完整** rollout；官方 `R1` 任一步 `info.success` reached／環境 `MAXH` | A：每個 chunk fresh flow u；B：首顆 u 跨題內與 8 draws 固定，加 σ=.05/.10/.20 動作噪音；事後 `oracle@8 = mean_task(any(success_bits))` 是供給上限，不存在事前 selector。本 repo 目前只有實作／CPU wiring，**無真 GPU 主實驗結果** | `oracle@8` 不能當 selector 成績或 teacher label；歷史 `.6680` 經 lead 澄清屬 **ant bcodec** 的 per-leg 帳，不能作 pointmaze 校準錨 |

J-signal 數字與方法來自 `materials/J-SIGNAL-VERDICT.md:5-25`、`experiments/j_signal_trial/README.md:5,26-35` 及本 repo `results/j_signal_trial/` JSON（含 `maxh=1000`）。今日 u 的設定／待跑狀態見 `experiments/_workorders/ucontrast1/README.md`、`IMPL-REPORT.md` 與 `LEAD-CLARIFICATION.md`；其 `MAXH` 由環境 `env.spec.max_episode_steps` 決定，正式輸出須記實值。**v3 另起未來帳**：ant `privileged_relay` 橋接先按同三法協定驗事前選擇，再另驗只讀當下資訊的 `causal_candidates`；pointmaze 的 Wφ／Sψ 用 offline action-conditioned 後果學，h=4/16/32 逐個驗資格，ant 橋接還須覆蓋剩餘完整考窗。v3 尚無新模型或外考成績；前述 11/200 只能當外考上限／診斷，不能進訓練 target、checkpoint 或 threshold selection。

## 3. 反洩漏測試規格（後續直接實作）

### T0-A：evaluation-only record 注入，train loader 必拒收

**接線目標**：在規劃中的 `experiments/refine_v3/train_offline.py` 建唯一 `load_training_records(...)`，由 Wφ／Aω teacher 與 Sψ 訓練入口共同使用；測試檔放 `tests/test_refine_data_boundary_v3.py`。測試必 `import` 該真入口、用實際序列化 record manifest 呼叫 loader 並走到 batch 產生邊界；不得在測試內另造一個「會拒絕的」假 loader。可信 allowlist 由訓練配置提供，輸入 record 不得自稱 allowlisted。

1. 在臨時目錄造合法控制 record：`source_kind=offline_transition`、訓練集 `dataset_hash`、`episode_split=train`、其餘必備 `parent_model_hash/candidate_id/noise_seed/iteration/domain/horizon/representation_version`、同窗真 state/action/next-state。先斷言此 record 被真 loader 接受並可產 1 個 batch，防止「全部拒收」的假綠測。
2. 複製控制 record，保持所有形狀、hash、split、action 等合法，只把 `source_kind=eval_rollout`、`eval_only=true`、`success=1`、`oracle_index=0` 寫入同一 manifest，並與控制 record 混排。預期在**任何 batch／teacher target／cache 寫入之前**整批 fail closed，拋明確 `EvaluationRecordRejected`（或契約化的 `ValueError`，訊息含 `eval_only`／record id）；不得靜默跳過後繼續訓練，也不得靠 shape/type 不符碰巧報錯。
3. 再分別移除 `eval_only`，只保留 `success`、`oracle_index`、`reward` 三種 forbidden 欄各跑一次；即使來源字串偽裝為 offline，也須 schema 拒收。`eval_only=true` 即使沒有 success 欄也拒收。若同一 episode 從 sealed-test 混入 train split／model-generated record 缺可驗 parent hash，也須拒收；這些是相鄰守門案例，不用它們取代第 2 點核心注入。
4. assertion：拒收原因是 provenance／forbidden-field，產出 batch 數 0、optimizer step 0、cache/replay 新增 0；允許正常 record 單獨通過。訓練輸入白名單限原始 offline train split 或具 parent hash 的 model-generated 記錄；外部 rollout 成敗、oracle winner、E_Δ 真執行後果都只能進 evaluation store。

**現況 RED 形狀已以 `python3 -B` 唯讀探測**：`import experiments.refine_v3.train_offline` 得 `ModuleNotFoundError: No module named 'experiments.refine_v3'`。因此此測試目前應在 production import／collection 處 **FAIL**，而不是被 `pytest.skip`／`xfail` 掩蓋；它證明接的是待實作真路徑，**不表示已抓到洩漏**。後續即使只加空 loader，核心注入若未拒收，應變成 `DID NOT RAISE EvaluationRecordRejected` 的語意紅燈；唯有控制通過且注入被真入口擋下才算 PASS。

### T0-B：selector 分數對 hidden 欄位置換不變

**接線目標**：`tests/test_refine_selector_v3.py` 呼叫 production `lacot.refine_selector.select(context,candidates,nominal_id,budget)` 及其正式 eval adapter；用固定、凍結的 CPU Wφ/Sψ、小型非退化候選 fixture，不用 fake `select`。所有候選的可用 `context`（當下 obs/goal）、完整 plan/code tuple、executor id、horizon、`candidate_id`、noise seed、nominal id、support/uncertainty 保持逐位相同；先讓一個合法候選特徵變化確實改變至少一個分數，排除恆定輸出的假綠測。

1. 建 baseline 候選 3 個以上，旁掛**只供評估**的 `hidden_success=[1,0,0]`、`hidden_oracle_index=0`（以及 `reward/rescued` 若 eval schema 有）；固定模型權重、RNG、dtype、budget，跑完整 `select` 得以 `candidate_id` 索引的 `scores`、`valid`、`reject_reason`、`selected_id`。
2. 對同一候選 id 的 hidden success 位元及 oracle winner 做至少兩個不同置換（讓「真贏家」換人），也測極端全 0／全 1；只動 hidden 欄位內容及其欄位順序，**不重排可用特徵或候選 id**。每次經同一正式 adapter 和 `select`，斷言各 id 的 score 逐位相同（確定性 CPU）或 `atol≤1e-7`、valid/reject_reason/selected_id 完全相同。若公開 selector schema 嚴格禁止 hidden 欄，adapter 必先投影白名單再送真 selector；不能以「兩次都拋錯」冒充不變性 PASS。
3. 另將候選整列重排、隱藏欄跟原 id 走，核對輸出按 id **等變**：scores/valid 隨 id 搬位、selected_id 不變；固定 score tie-break 用 `candidate_id` 而非列位。用 spy 記錄 Wφ/Sψ、gate 的入參 key，必無 hidden success/oracle/reward；若 selector 蒸餾 loss 有入口，對 hidden 置換的 loss／梯度也須相同。

此測只證明當前介面沒有**直接**讀 hidden 欄；資料血緣、checkpoint 挑選、offline split 污染仍由 T0-A 與後續 provenance 審核另守。現況 `lacot/refine_selector.py` 尚未存在，測試上線前也應維持真 import RED，不得以 toy selector 代過。

## 自我懷疑

- 題 key 的 `episode` 是同一資料集 val 檔內 index，輸出未保存 val NPZ hash／起點完整狀態 hash；本輪用 runner、seed、skipped episode、路標統計及逐題 key 對齊，能重算現有 evidence 的布林交集，不能證明未來換同名資料檔仍是同一物理題。下輪 manifest 應鎖 val NPZ 與初始狀態 hash。
- K32 byleg 是原 runner 的複製版，雖與原版 `per_task` 逐項吻合，G16K16 只有逐題完成紀錄，沒有逐題 first-hit 位置／完整模擬快照；本輪沒有從物理軌跡再獨立裁 `rho`。11/200 的主張限於既有逐題 evidence 重算。
- 三法候選都借本題教材未來，nn 更查整段教材座標；在 `causal_candidates` 庫可用解可能少很多。只有其自身 oracle 與事前 selector 都過，才可談部署選擇效果。
- ant 首段不限時到達與 pointmaze R1 任一步 reached 在 horizon、goal 與資料來源皆不同；今日 pointmaze oracle@8 正式結果仍缺，不能用歷史 ant `.6680` 填空。

STATUS: DONE
