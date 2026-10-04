# wallpen2-eval-v1-r3（CPU）

結論：擠下限比例改用「所有最短路前 3 步格心位移皆 ≥ 5.65」的陷阱格子集；臂與 C1p 三個 seed 共用子集，比例的 R、零極差 1/N 與 max+2R 隨之使用子集樣本。落點中位距離、雜訊帶與其他計算維持原樣。V1–V10 首次單次完整執行全部 PASS，7 個 subprocess 殺手都實際 FAIL。

SECOND PASS: PASS；V10 用 SHA256 為 `68038b85…` 的原始程式執行真 CLI，在原替身組與 U 回折組逐值比較所有非擠下限 JSON 欄位。原替身組除新增子集清單外，TXT 全文相同，PNG 位元相同。V1–V8 自測語句與預期值 AST 全同；v1 評估目錄 20 個保護檔 SHA256 diff=[]。

`n_tests=10 n_killers=7 n_checks=506`

## 依據與變更

開工前已讀原工單 `WORKORDER-wallpen2-eval-v1.md`、預註冊 §七 v2.1、r2 交付與自測原文；上一版 commit 為 `eccece8`。

- §七 v2.1 完整 SHA256：`e8cef8e86ed1447021bcb491be5aea662b6bc502507e192c90ea569ebb34ecd3`。
- 上一版 analyzer SHA256：`68038b856fc6a08dad16506f6444799d93dd7df110df56823876cea359aff8da`。
- 本版 analyzer SHA256：`0ca418f614fc34a63ba05001c0e49e553c079849e529fca8424181c04f9c18d8`。

程式新增部分標記 `wallpen2 r3:`。從 C1p Stage O 的 `cells` 取陷阱格；在 `routes[task]['paths']` 中找所有包含該格的路，取該格索引 +3（超出路長取最後一格）。格心位移使用格索引差乘原始格距 `ruler.cell_size=4`，共同座標原點抵消；每條路皆 ≥ 5.65 才納入。子集依原 cells 次序保留。

子集只計算一次；既有配對／routes 一致性驗證確保五臂共用。JSON 新增 `floor_subset_cells`，TXT 每 task 印同名清單，包含無 draw 情境。`descriptors.away.floor_fraction` 使用子集的 16 draws／格；非陷阱比例不變。`noise_report` 與 `descriptor_checks` 原碼不變，直接接收新的比例與 N。中位距離仍用所有陷阱格。

原 JSON 的 `prereg_sha256` 欄位依「其他欄位逐值不變」保留 v2.0 基線值；本次 v2.1 規則依據的 SHA 記於上方。

原替身組子集未排除任何陷阱格：task 2＝7 格／112 draws，task 4＝5 格／80 draws，task 5＝4 格／64 draws。此為程式驗收，不能當成 v2 訓練臂的實驗結果。

## 自測與殺手

`wallpen2-selftest-output.txt` 是以下單次完整執行的 stdout＋stderr 原文，exit 0；本輪沒有失敗後重跑。

```bash
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -B \
  experiments/_workorders/wallpen_eval/wallpen2_selftest.py \
  > experiments/_workorders/wallpen_eval/wallpen2-selftest-output.txt 2>&1
```

被測 analyzer 只在新 subprocess 執行，殺手為來源副本；沒有 preload、猴補或載入 checkpoint。既有 V4–V8 的判讀分支 fixture 仍為明示 synthetic aggregates；新增 V9、V10 執行真正 analyzer CLI。

- V1–V8：全部 PASS，沿用上一版所有斷言與預期值。V4 的全 5.8 落點仍判擠；V8 的 n=80／112 計數邊界與 1/N 退回均不變，無任何預期值修訂。
- V9：不呼叫被測函數，以 `math.hypot` 獨立逐格重算 H3 前 3 步位移，核對 JSON／TXT 清單，原文列出每格位移。
- V9 U 回折：明示合成 routes fixture；task 4 的 `[3,8]` 一條路位移 √80，另一條路三步 U 回折位移 4，必須排除。此為 JSON 路徑條件 fixture，不是實測 maze 最短路。
- V9 子集比例：五臂 task 4 都只用 4 格／64 draws；被排除格的 16 個 5.8 落點不計入。三 seed 都 0，R＝1/64；候選臂 2/64 等於 max+2R 不擠，3/64 必須擠。
- V9 短路徑：一步位移 4／終止格位移 0 排除，兩步直角位移 √32 納入，確認不足三步取最後格。
- V9 新殺手：把子集条件改成 `if True`（全部陷阱格），同一套 CLI 斷言實際 FAIL：`V9 exact independent subset list`。
- V10：從 `git show eccece8:…/wallpen2_analyze.py` 取原始來源並核對完整 SHA，再對完全相同輸入路徑執行 CLI。僅剔除新增子集、陷阱 floor_fraction、其 seed noise 與三個判讀欄位 `floor_fraction/floor_limit/squeezed`，其餘 JSON 全結構逐值相等，包含結論、守門、來源與所有中位距離。一般替身組的原 TXT 與 PNG 亦相同。

七個實際被殺的突變：V4 基準改 C1p；V5 拿掉成功的描述量条件；V6 無 draw exit；V6 直路 MSE 納入守門；V7 按 VALID 選主判；V8 移除下限容差；V9 子集改全部陷阱格。

最終原文：

```text
SECOND PASS: independent subsets, pinned r2 non-floor equality and all same-invariant subprocess killers passed
n_tests=10 n_killers=7 n_checks=506
STATUS: DONE
```

完整 fixture、突變來源與每次 CLI 原文保留於 `/tmp/wallpen2_selftest_sdv15916/`；修改前檔案及保護檔 SHA 清單保留於 `/tmp/wallpen2_r3_before/`。只寫入 `wallpen2_*` 交付檔與其輸出；未 commit、刪檔、裝套件、訓練或使用 GPU。訓練目錄同時有外部變更，本次未寫入其中任何檔案。

## CLI 與輸出

CLI 不變：`--base C1p=STAGEO,MODEL`，兩個 `--seed C1p-s34=…`／`C1p-s35=…`，兩個 `--arm C3-b=…`／`C3-a=…`，`--outdir DIR` 必填。

輸出固定為 `wallpen2-analysis.json`、`wallpen2-analysis.txt`、`wallpen2-analysis.png`。原替身例在 `wallpen2_example/`；真實 C2hi G2 FAIL／無 draw 的替身例在 `wallpen2_example_nodraw/`，保留退回 C3-a 的判定與原因。

STATUS: DONE
