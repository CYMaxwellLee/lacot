# wallpen-eval-v1-r1（CPU）

交付的是預註冊 v1.1＋v1.2＋v1.3（含補充）判讀程式。只讀指定 H3／H4／H8／Hinf 替身 JSON 與 H3／H4／H8 checkpoint 做自測；沒有評估任何新臂、訓練、rollout、GPU、slurm、commit、安裝套件或修改訓練工單目錄。

三支程式：`model_metrics.py` 每個 process 只呼叫一次指定 frozen `eval_common.load_ckpt`；`analyze_wallpen.py` 讀 Stage O＋model JSON；`selftest_wallpen_eval.py` 執行七組自測及十二個會被 invariant 擋下的 mutant。

## 執行

從 `/home/cymaxwelllee/Projects/lacot` 執行。輸出請只放本目錄或系統暫存。三支程式都在 import torch 前設定 `CUDA_VISIBLE_DEVICES=''`，停用 bytecode；matplotlib cache 放 `/tmp`。

```bash
.venv/bin/python -B experiments/_workorders/wallpen_eval/model_metrics.py \
  --ckpt /home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/ckpt-local/H3.pt \
  --ckpt-sha256 c42e4389a6f29188a9f8d5f1f3bbad732ccc8597bd26faae00d7fbef9d924d4f \
  --out experiments/_workorders/wallpen_eval/H3-model.json

.venv/bin/python -B experiments/_workorders/wallpen_eval/selftest_wallpen_eval.py
```

完整自測預設重新用三個 subprocess 計算 H3／H4／H8，並以第四個 fresh process 再算 H3。`--reuse-models` 只重用本目錄已算出的三份 JSON，仍核對指定 checkpoint 與 model provenance 的完整 sha。這次交付以三個獨立 model process 真算之後，使用 `--reuse-models` 跑 E1–E7；此旗標仍必定啟動 fresh process 重算 H3，輸出 `H3-repeat-model.json`／`H3-repeat-model-output.txt`，E7 比較兩次獨立計算的 MSE 與完整直路結果。重用的三份 model JSON 必須含新的「至少三個不同格」定義。各 process 原文存於 `H3-model-output.txt` 等檔。

替身 analyzer CLI：

```bash
.venv/bin/python -B experiments/_workorders/wallpen_eval/analyze_wallpen.py \
  --arm B0=/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/stageO/H3.json,experiments/_workorders/wallpen_eval/H3-model.json \
  --arm C1=/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/stageO/H4.json,experiments/_workorders/wallpen_eval/H4-model.json \
  --arm C2-hi=/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/stageO/H8.json,experiments/_workorders/wallpen_eval/H8-model.json \
  --arm C2-lo=/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/stageO/Hinf.json,experiments/_workorders/wallpen_eval/Hinf-model-synthetic.json \
  --outdir experiments/_workorders/wallpen_eval
```

`B0` 與 `--outdir` 必給，其他臂可省略。選用雜訊臂加 `--arm B0s=STAGEO_JSON,MODEL_JSON`。缺 C2-hi 時不把 lo 當預設主判；缺比較臂時不產生該差值／歸因。`G` 若提供，checkpoint sha 必須逐位元等於 B0。schema、尺、evidence key、summary、geometry／dataset provenance 或配對錯誤會拒絕分析並 exit 2；G1–G3 非 PASS 則保留描述數字，該臂標「不出結論」。不為缺失的關鍵數值補零。

`model_metrics.py` 可指定 `--dataset`、`--cells`、`--routes`；預設使用工單指定資料集、H3 的 50 格清單與 `detour-u/routes-density.json`。dataset 必須符合固定 DATASET_PIN。analyzer 不載 checkpoint，從同一 pinned dataset 按 M9 的 float32 `mean/std+1e-6` 建尺，並對各 Stage O 的 normalization 做逐值核對。

## 方法與一手來源

規格：[PREREG-wallpen-v1.md](/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/wallpen/PREREG-wallpen-v1.md) §三、§四、§六（v1.2）、§七（v1.3＋補）。工單 §5 對應該檔 §六。

lead 原型均唯讀，沒有 import 其頂層執行段。方法取自：

| 原型 | sha256 |
| --- | --- |
| `clean_baselines.py` | `f0c39a8d0b1dbfa946a4deb0c5ed027c30d5622defc82b6abccaf9da581e56b1` |
| `calib_tol.py` | `d133a37a66e618c0babf097bff39e734ac3a3d0d6487858f77fe62322e2808e2` |
| `breakdown_h3.py` | `f07bc2f466eb4eda618b603ae1739a1e66f5461d2c0304750f413d4f61819cd6` |
| `path_vs_landing.py` | `bcdd97b5a300934d5a20cb7ea38a66cc31adb854139210ce1252a927eef91bb0` |
| `same_ruler_teacher_vs_flow.py` | `09e2b6dff7c69d57e98fd507d63a42d5127df0c1479f89b57ea1afe59eb800bb` |

以上檔案都在 `/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/hsweep/stageO/`。

關鍵一手片段，與本實作對應：

```python
# clean_baselines.py / calib_tol.py：保留第一個跨過 12 的離散點，不插值截斷。
c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=1))]
return len(sm) - 1 if c[-1] < 12 else int(np.searchsorted(c, 12))
ok = dep(sm[:land_k(sm) + 1]).max() <= TAU

# calib_tol.py：每格每条經過它的最短路，前四格。
k = path.index(c); seg = path[k:k + 4]
if len(seg) < 2: continue
sm = prep(d.decode(module, m1.encode_trajectory(module, pts), pts[0]), pts[0])

# breakdown_h3.py：牆內落點才吸到最近自由格；不同類距差 <1 判分不清。
if other is not None and dist[other] - dist[order[0]] < 1.0:
    k = '分不清（兩邊差不到 1 單位）'

# frozen M9 stage 1：全 128 點、正規化 XY 的重建式。
et = etarget(traj, mask)
_pts1 = _dec(et_dec, s)
_main = (_pts1 - traj).pow(2).mean()
```

正對照先走 frozen `harness_move1.encode_trajectory`（128 點弧長內插、全 False mask），再用同一 decoder、9 點 valid 平滑與起點錨定。佔據深度固定 `GeoEnergy(..., res=8).wall_depth`，τ=.0664；真牆深度照原型 20 格／世界單位光柵、`distance_transform_edt`，只量截取段中的離散點。

四分類以 BFS 的 detour／progress 判方向；自由格不吸附、牆內落點比較最近自由格中心。JSON 使用縮短後的四類名稱，判法與 lead 完全相同。失敗比例的分母是所有非 VALID，並非全部抽樣；沒有失敗、最大类並列或「太近」最大但增幅未達 .15 的未規定分支都明標「待討論」。

直路是 `make_batch(rng, teacher_mix=0.0)` 傳回的 normalized 128 點還原世界座標後，呼叫環境 `xy_to_ij` 的格序列；去連續重複、至少三個**不同**格、全同列或同欄（A→B→A 排除）。固定 RNG 20261004，保留最先 2048 條；不足三個不同格的窗口不算直路。轉彎是格序列同時改變列與欄，報同一段抽樣過程中全部轉彎窗口（此次 4726 條）的 MSE；其餘短窗口不算轉彎。每條 128×2 點 `.pow(2).mean` 後以 float64 平均，等權相當於全點平均，不套平滑、不套錨定、不加 soft start loss。座標 hash 是有順序的世界座標 `<f8` C order bytes，shape 固定 `[2048,128,2]`。

bootstrap 是各 task 的 `[陷阱格,16]` 矩陣：重抽格、每个抽中的格各自重抽 16 draws；同一對臂共用兩組索引。10,000 次、seed 20261004、95% percentile。單臂柱圖的 CI 使用同一兩層算法；JSON 同時保留五組差值對所有機制指標的區間。格、away 標記、seed 或 draw 順序不同就拒絕配對，不排序修補。

Stage O 閘和守門分開記錄。C1 守門對 B0，C2-hi／lo 守門各對 C1；另保留 C2 對 B0 的獨立整套守門欄。直路 MSE、非陷阱格 VALID、陷阱格 NEAR 都按預註冊檢查；後兩項包含 task 2。只有 task 4／5 做原因判定；task 2 的守門仍可使整臂不能出結論。hi→lo 只取決於 Stage O G1／G2／G3 與對 C1 守門，理由列出所有未過項；hi 主指標完整保留。比較臂 B0／C1 的 Stage O 閘必須 PASS 才歸因；C1 對 B0 守門 FAIL 時 C2 相對 C1 的結論仍可報，但必附指定警語並獨立報整套對 B0 的結果。門檻等號納入，比例運算只容許 1e-12 浮點舍入誤差。

## v1.3 變更

- `arms["C2-hi"/"C2-lo"].guards` 改為對 C1，`guard_reference` 明標參照；`guards_vs_b0` 是獨立整套欄。`interpretation.c2_guard_conclusions` 分成 `penalty_side_effects`（懲罰有沒有副作用）與 `overall_degradation`（整套跟原本比有沒有變差）。C1 守門任一 FAIL，C2 所有結論均附「⚠️ C1 對 B0 已有副作用，C2 對 C1 通過不代表整套沒問題」。task 2／4／5 全守。
- hi 任一 Stage O 閘非 PASS 或對 C1 守門 FAIL 即檢查 lo；兩顆不合格或缺必要參照時不出結論。`select_primary` 不讀 VALID；KILLER 3 直接猴補這個函式，測試相同 invariant 必須 FAIL。
- B0s 先走既有 Stage O 校驗、同格／seed／draw 配對與直路集合核對。`interpretation.noise.by_task` 只報 task 4／5 各五個絕對差：`valid`、`clean`、`straight_mse_ratio`、`nonaway_valid`、`near`。直路 MSE 比差明定為 `abs(MSE_B0s / MSE_B0 - 1)`，即兩臂各以 B0 正規化後的比值之差。無 B0s 印「未提供」；B0s 閘未過則該欄不出結論、列出原因，不拿無效尺的結果判噪音。任一判定 task VALID 絕對差 ≥.10，選臂理由、C1／C2 結論、四分類結論、兩層守門結論與 rollout 提示全部加指定雜訊註記。B0s 不進主判。
- rollout 提示只看已選 C2 的 task 4、5 VALID 是否都 ≥.70；B0／C1 閘等阻擋歸因的原因另列 `conclusion_blockers`，不壓掉達標提示。
- `--outdir` 必填；直路抽滿 2048 條，至少三個不同格且全同列／欄。H3／H4／H8 重新 CPU 真算；H3 再由 fresh process 獨立驗證。既有尺、正對照、四分類、bootstrap 方法及固定 E1–E4 預期不變。
- 新殺手：拿掉格內 draw 重抽、漏看 G3、四分類錯對 B0，以及六個嚴格不等號突變（VALID .15、clean .30、雜訊 .10、MSE 1.10×、nonaway .10、NEAR .15）。十二個 mutant 全部各自實際得到 AssertionError；測試套件恢復原函式後繼續。另驗 A→B→A 排除、B0s 五值／閘／配對、雙層警語、rollout 隱藏條件與必填 outdir。


## 自測與替身結果

本輪最終驗證：`n_tests=7 n_killers=12 n_checks=129`；E1–E7 全 PASS，十二個突變均實際 FAIL。

[selftest-output.txt](selftest-output.txt) 保留完整七組原文與十二個 mutant 的 AssertionError；[example-analysis.txt](example-analysis.txt) 是替身等寬表格；[wallpen-analysis.json](wallpen-analysis.json) 保留全部計數、區間、守門、每 draw 深度／四分類、正對照明細、直路每條 MSE；[wallpen-analysis.png](wallpen-analysis.png) 為兩欄圖。

E1–E4 重現 lead 的固定數字。H3 正對照實測 `n=62 p50=.00728186 p95=.06638879 max=.14309558`；點名 (5,6)=.08524871、(5,1)=.12759110（真牆深度 .4）、(3,9)=.00668797。

H3／H4／H8 均抽 470 批、直路 2048 條，新的共同 sha＝`9a295f038b84666e1fe9d6ac20e1eb98b506158fa8a42afce212989048635a6f`。直路 MSE 分別 `.012006182343`／`.014335873203`／`.009988202150`；H4／H3 約 1.19404，超過 1.10。H3 第二次獨立 process 的直路結果完整相同，MSE 比恰為 1.0。替身 Hinf 的 MSE 明確構造等於新 H3，沒有 Hinf checkpoint 正對照或轉彎 MSE。

E6 新參照的手算（原始 counts 來自各 Stage O JSON）：

| 守門層 | task 2 nonaway VALID 掉 | task 4 掉 | task 5 掉 | 結果 |
| --- | --- | --- | --- | --- |
| C1=H4 對 B0=H3 | (113−74)/144=.270833 | (129−146)/208=−.081731 | (117−126)/192=−.046875 | task 2 FAIL；直路 MSE 亦 FAIL |
| hi=H8 對 C1=H4 | (74−61)/144=.090278 | (146−123)/208=.110577 | (126−137)/192=−.057292 | task 4 FAIL |
| lo=Hinf 對 C1=H4 | (74−62)/144=.083333 | (146−137)/208=.043269 | (126−140)/192=−.072917 | PASS |
| hi 對 B0 | (113−61)/144=.361111 | (129−123)/208=.028846 | (117−137)/192=−.104167 | task 2 FAIL |
| lo 對 B0 | (113−62)/144=.354167 | (129−137)/208=−.038462 | (117−140)/192=−.119792 | task 2 FAIL |

所有 NEAR 增幅均不超過 .15；hi／lo 對 C1 的 MSE 均不超過 1.10 倍。故主判由 hi 改 lo，原因明寫 task 4 nonaway VALID 守門。lo 相對 C1 的 VALID 差 task 4 `(6−14)/80=−.10`、task 5 `(15−28)/64=−.203125`，路乾淨差也未達 .30，兩 task 都判「懲罰沒教到牆（權重太小或被躲掉）」；附 C1 副作用警語。lo 的懲罰副作用層 PASS、整套層 FAIL，C1 自身歸因不出結論；rollout 未達 .70。此為不同 horizon 替身的程式自測，沒有 wallpen 真臂效果結論。

殺手原文摘錄（完整十二項見 `selftest-output.txt`）：

```text
KILLER 3：實際 select_primary 被猴補成看 hi r=.9 > lo r=.5，NEAR 守門 invariant FAIL
KILLER 4：兩層 sparse-draw CI 非零；一層 CI [.0625,.0625]，invariant FAIL
KILLER 5：Stage O 副本只有 G3 FAIL，gate_pass 漏看 G3，invariant FAIL
KILLER 6：四分類應對 C1 的太近 .20，突變對 B0 的 .05，分岔 invariant FAIL
KILLER 7–12：等號邊界改成嚴格不等號，各自 invariant FAIL
```


## 測試環境差異聲明

替身是 hsweep 的不同 teacher horizon，並非 wallpen 的乾淨 teacher／懲罰臂。只有 H3／H4／H8 指定 ckpt 被 CPU 真算；Hinf 只讀現有 Stage O，model JSON 明標 synthetic，正對照與轉彎 MSE 為 null（未量）。真臂的 checkpoint 仍須先經指定 loader 與 sha 核對；目前沒有測新臂相容性、λ／ρ 定標 log、pen／重建比或 G 的新訓練結果。這些輸入沒有提供，不推定其值。

固定 DATASET_PIN 與 source bytes、相同 50 格、相同抽樣 seed／draw 才能配對；新臂若換 batch、資料或 normalization，應拒絕配對／直路守門。CPU torch 環境與訓練 GPU 可能有數值差異，程序仍只在 CPU 判讀；不以此更改預註冊數值。9 點平滑本身會切角，深度絕對值會偏深；真牆光柵只量離散點，沒有聲稱連續路徑碰撞證明或 rollout 成功率。

## 輸出契約（兩欄）

| 必須保留的輸出 | 允許變更的行為 |
| --- | --- |
| model JSON：schema、cpu、完整 checkpoint/dataset/source sha 與 normalization | 輸出檔路徑可換到本目錄或系統暫存；指定 dataset 路徑可換，但 pin 不變 |
| 62 個 cell-route 正對照條目、三格點名、τ、通過率、p50/p95/max、真牆深度 | 額外附描述統計可增加；不得換尺、刪共享路尾或改容差 |
| 直路 n=2048、抽樣 seed、批數、座標 sha、MSE；轉彎定義／n／MSE | 增加 metadata 可；直路定義只依本輪 B2 改為至少三個不同格；其餘抽樣順序與重建 loss 保留 |
| analysis：每 task 原始 counts、主指標與機制率、五組可用差值及 CI | 可省略未提供的臂和不可算的比較，必須明標缺失 |
| 每臂 G1/G2/G3、參照與兩層守門 PASS/FAIL／原因、四分類 counts/proportions | 表格寬度、數字顯示小數位可改；JSON 原始精度保留 |
| 主判與選臂理由、hi 全部主指標、task 4/5 分別結論、task 2 描述 | 追加解說可；不得看主指標改選 lo、合併 task 或把描述當 rollout |
| selftest 原文、n_tests/n_killers、十二個實際 FAIL 的 mutant | 只可增加有意義的檢查；不得修改 E1–E4 預期值來遷就輸出 |
| 兩欄各 task 長條＋95% CI、明標替身／synthetic／未量 | 圖的顏色、尺寸、字型與排列可改 |

SECOND PASS: 最沒把握的兩處是尚未取得真臂／B0s checkpoint，故 frozen loader 對未來真臂 metadata 的相容性與真實雜訊幅度未實測；以及 lead 的真牆光柵／9 點平滑只支持同尺比較，絕對深度與連續碰撞的偏差未另校準。兩處都已在輸出／聲明標註，不用預設數字填補。

n_tests=7 n_killers=12 n_checks=129
STATUS: DONE
