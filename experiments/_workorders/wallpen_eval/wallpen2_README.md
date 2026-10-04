# wallpen2-eval-v1-r2（CPU）

結論：下限比例與雜訊帶邊界補上 `1e-12` 浮點容差；V8 用整數計數與獨立有理數預期，驗證 n=80、n=112 的等號不擠、下一個計數會擠及 `1/N` 退回。V1–V8 單次完整執行全部 PASS，六個 subprocess 突變殺手都確實 FAIL，包含移除下限比例容差。這是替身資料的程式驗收，不能當成 v2 實驗結果。

SECOND PASS: PASS；`descriptor_checks` 以外判讀程式 AST 一致，V1–V7 斷言與預期值一致（V4 僅更新突變來源比對字串）；25 個保護檔 SHA256 diff=[]。六個同斷言 subprocess 殺手全部 FAIL。

`n_tests=8 n_killers=6 n_checks=387`

## 檔案與依據

- `wallpen2_analyze.py`：CLI、v2 描述量、seed R、守門、hi→lo 選擇、判定、JSON／等寬表／四面板圖。
- `wallpen2_selftest.py`：V1–V8；被測程式只在新 subprocess 執行，突變為獨立來源副本；沒有 preload 或猴補。
- `wallpen2-selftest-output.txt`：回鍋單單次完整執行 V1–V8 的 stdout＋stderr 原文（覆寫）。
- `wallpen2_selftest_first_failure.txt`：第一輪失敗原文。
- `wallpen2_example/`：正常有 draw 的替身組，含 JSON、文字、PNG。
- `wallpen2_example_nodraw/`：真實 C2hi G2 FAIL／無 draw 的替身組，含 JSON、文字、PNG。

預註冊完整 SHA256：`2131461b93bd339d79eec94ec043687a757dc31e86c88f4c08795e0b10309563`。

本次 `wallpen2_analyze.py` SHA256：`68038b856fc6a08dad16506f6444799d93dd7df110df56823876cea359aff8da`。

沿用而未修改：

- `analyze_wallpen.py` SHA256 `536e6a9eb907a521fbd6a6aa05fa3c21da2e38cff9938639f6636972dc146c00`。
- `model_metrics.py` SHA256 `291ead053320d669a814157f2dbe125b50d549c5251ced4473b704b99f2f929c`。

只使用 CPU；未訓練、未載 checkpoint、未 commit、未刪檔、未裝套件。訓練工單檔案未碰。

## CLI

臂名稱固定，`--outdir` 必填。以下為替身例，不能當成 v2 實驗結果：

```bash
BG=/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u
EV=/home/cymaxwelllee/Projects/lacot/experiments/_workorders/wallpen_eval
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -B "$EV/wallpen2_analyze.py" \
  --base "C1p=$BG/wallpen/stageO/C1.json,$BG/wallpen/model/C1-model.json" \
  --seed "C1p-s34=$BG/hsweep/stageO/H3.json,$EV/H3-model.json" \
  --seed "C1p-s35=$BG/wallpen/stageO/B0s.json,$BG/wallpen/model/B0s-model.json" \
  --arm "C3-b=$BG/hsweep/stageO/H3.json,$EV/H3-model.json" \
  --arm "C3-a=$BG/wallpen/stageO/C2lo.json,$BG/wallpen/model/C2lo-model.json" \
  --outdir "$EV/wallpen2_example"
```

輸出固定名稱 `wallpen2-analysis.json`、`wallpen2-analysis.txt`、`wallpen2-analysis.png`。JSON 記錄每份輸入的路徑與 SHA256。

無 draw 例的三個 seed 與 C3-a 都用 C1；C3-b 用真實 C2hi Stage O／model。該例沒有補造 draw，保留 G2 FAIL；`interpretation.primary` 為 `C3-a`，原因列出 G2 與無 draw，C3-b 主指標為 null。

## 實作口徑

有 draw 的 Stage O 直接呼叫 v1 `validate_stage`、`assert_paired`、`arm_metrics`；主指標、區間、路乾淨尺與四分類沿用 v1。空 flow 專用驗證只檢查 envelope／尺／evidence key／inventory／seed／閘／provenance，不補造樣本。對照及兩個 seed 都必須有 draw 且閘全 PASS；候選臂閘失敗仍保留已量到的指標。

落點距離為 `sqrt((p_x-start_x)^2+(p_y-start_y)^2)`，start 是 `smooth_anchored_xy[0]`。陷阱與非陷阱分報中位數及閉區間 `[5.65,6.5]` 比例。中位數的圖上區間採兩層 percentile bootstrap；比例直接沿用 v1 bootstrap。皆 10,000 次，seed 20261004。

R 按 task／量取三 seed 極差；零極差用相同樣本數的 `1/N`，每個 R 在文字與 JSON 列出。非陷阱 VALID 與陷阱 NEAR 用往壞方向 `>2R` 守門。直路 MSE 相等檢查僅警告，不進守門。四分類使用 v1.2 函數，基準 C1p。主判只看閘與守門；task 4／5 分判，task 2 描述。

本輪逐條比較自查（所有容差皆為 `1e-12`；報出的門檻值維持原數值）：

| 比較 | 自查與處理 |
| --- | --- |
| 下限比例嚴格 `> max+2R` | 補為 `floor > limit + 1e-12`，等號不擠。 |
| 雜訊帶下緣（含等號） | 補為 `band[0]-1e-12 <= med`。 |
| 雜訊帶上緣（含等號） | 補為 `med <= band[1]+1e-12`。 |
| 「仍在縮」嚴格小於下緣 | 補為 `med < band[0]-1e-12`，與含等號的帶內比較一致。 |
| VALID Δ ≥ .15 | 原 `V1.reaches(dv, .15)` 已有容差，維持。 |
| 路乾淨 Δ ≥ .30 | 原 `V1.reaches(dc, .30)` 已有容差，維持。 |
| VALID R ≥ .10 | 原 `V1.reaches(R, .10)` 已有容差，維持。 |
| 非陷阱 VALID／陷阱 NEAR 守門嚴格 > 2R | 原 `value > limit+1e-12` 已有容差，維持。 |

落點距離 `[5.65,6.5]` 是原始描述量定義，維持；零極差退回判斷與直路 MSE 相等警告也維持。

替身組的三個「seed」其實是 C1、H3、B0s，不能代表 v2 共享 stage 1 的訓練雜訊。輸出中的 `synthetic` 反映 model JSON 原有標記；這兩組交付例的臂別都是替身名稱，並非真的 v2 訓練臂。

## 自測狀態

執行命令：

```bash
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -B \
  experiments/_workorders/wallpen_eval/wallpen2_selftest.py \
  > experiments/_workorders/wallpen_eval/wallpen2-selftest-output.txt 2>&1
```

V1：逐值核對 v1 指標、paired 區間、四分類。V2：獨立重算 H3／C2lo 距離與比例。V3：C1／H3／B0s 手算 R 與三份相同 JSON 的 `1/N`。V4：原始 JSON 落點全改成 5.8；邊界殺手被殺。V5：VALID 大增仍需兩個描述量；所有判定分支、四分類、縮短及雜訊註記測試通過，成功條件殺手被殺。

V6：真實無 draw CLI、G1／G2／G3、task 2／4／5 兩種守門、等於 2R、兩臂都 FAIL 的檢查全部通過；exit-on-no-draw 殺手確實 exit 2。原 :329 的過度寬泛斷言改成兩條：有直路 MSE 不等的警告；每臂的守門失敗清單只含 VALID／NEAR 守門，且直路 MSE 不進守門 checks 或 pass_all。保留替身組原有的 VALID／NEAR FAIL。新增殺手把直路 MSE 相等納入守門 checks，這條斷言確實 FAIL。

V7：固定同一組閘／守門狀態，把 C3-b VALID 分別設為 .9、.01（C3-a 為 .5），兩次主判都選 C3-b，並保留 C3-b 主指標。改成比較 VALID 的殺手選出 C3-b、C3-a，確實 FAIL。

V8：下限比例計數組 `(0,0,3)/80`、`(7,7,7)/80`、`(0,0,3)/112`，候選臂各測 8/N、9/N、10/N。以 `Fraction` 獨立手算門檻；R=0 時為 `7/80 + 2*(1/80) = 9/80`，等號不擠。兩臂、task 4／5 的擠判定與結論全部核對。另驗距離雜訊帶 `[639/80,642/80]` 的上下緣及外側；等號留在帶內且不標「仍在縮」。拿掉下限比例容差的突變在 `(0,0,3)/80`、臂 9/80 的 V8 斷言 FAIL。

本次判讀程式只補上述容差；未改 V1–V7 預期值。殺手總數由五調為六，計入新增的容差殺手。最終原文：

```text
SECOND PASS: source review and all same-invariant subprocess killers passed
n_tests=8 n_killers=6 n_checks=387
STATUS: DONE
```

完整 stdout／stderr 見 `wallpen2-selftest-output.txt`。subprocess fixture、突變副本與各 CLI 原文保留於 `/tmp/wallpen2_selftest_7vejpmcg`。本輪修改前來源、輸出與 SHA256 清單保留於 `/tmp/wallpen2_r2_before/`。

STATUS: DONE
