# patchrun1 harness 收割驗收

HARVEST-OK

範圍只到 harness 驗收；沒有判讀科學結論。產物以 jasmine 的唯讀掛載
`/archive/jasmine/cymaxwelllee/diag-g16/` 讀取；job log 內的 `saved:` 路徑是
`/archive/cymaxwelllee/diag-g16/`。

## 1. 終態

先前在 jasmine 成功執行相同格式的 `sacct --format=JobID,State,Elapsed,ExitCode -X`，
分別查到 33852 array 與 33853 單 task；終態摘錄如下：

```text
JobID             State    Elapsed ExitCode
------------ ---------- ---------- --------
33852_8       COMPLETED   00:36:12      0:0
33852_9       COMPLETED   00:36:08      0:0
33852_10      COMPLETED   00:36:39      0:0
33852_11      COMPLETED   00:36:40      0:0
33853         COMPLETED   00:03:28      0:0
```

逐格：33852 的 8/9/10/11 與 33853 均 `COMPLETED`, `ExitCode=0:0`。

## 2. 完成錨

四份 JSON 的 `metrics.steps` 都是 `50000`；每個對應 log 都有兩行 `saved:` 與一行
`=== done ... (wall Xs) ===`：

### 33852_8 → s80001 / G32K32_lam0p0

JSON check: `metrics.steps=50000`。

```text
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80001/results/G32K32_lam0p0.json
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80001/ckpt/G32K32_lam0p0.pt
=== done G32K32_lam0p0 (wall 2168.7s) ===
```

### 33852_9 → s80002 / G32K32_lam0p0

JSON check: `metrics.steps=50000`。

```text
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80002/results/G32K32_lam0p0.json
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80002/ckpt/G32K32_lam0p0.pt
=== done G32K32_lam0p0 (wall 2165.2s) ===
```

### 33852_10 → s80001 / G32K32_lam1p0

JSON check: `metrics.steps=50000`。

```text
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80001/results/G32K32_lam1p0.json
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80001/ckpt/G32K32_lam1p0.pt
=== done G32K32_lam1p0 (wall 2195.8s) ===
```

### 33852_11 → s80002 / G32K32_lam1p0

JSON check: `metrics.steps=50000`。

```text
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80002/results/G32K32_lam1p0.json
saved: /archive/cymaxwelllee/diag-g16/seedrep/s80002/ckpt/G32K32_lam1p0.pt
=== done G32K32_lam1p0 (wall 2197.1s) ===
```

四顆 G32K32 ckpt 也各存在（s80001 兩顆、s80002 兩顆）。

單 B：`/archive/cymaxwelllee/diag-g16/seedrep/logs/shat3-33853.out` 有
`=== 結果 ===`、四行數值與判定行（原文見第 8 節），且
`/archive/cymaxwelllee/diag-g16/shat3-results/` 恰有 1 份 JSON：

```text
/archive/cymaxwelllee/diag-g16/shat3-results/bcodec_shat_gk16_obsonly_srep80001_summary.json
```

## 3. schema 逐 key

驗收腳本以完成的
`/archive/cymaxwelllee/diag-g16/seedrep/s80001/results/G16K16_lam0p0.json`
為參照，逐一要求 top-level、`metrics`、`config` key set 相同；並要求
`metrics` 含 `G/K/recon_a_val/recon_s_val/any_group_collapsed/mean_active_codes/train_seconds`。
參照的 top-level key 是
`config, data, loss_curve, metrics, per_group_usage`；G32K32 四份均無缺鍵。

逐格強制檢查也通過：`metrics.G=32`、`metrics.K=32`、`metrics.lam_s` 分別與檔名
`lam0p0/lam1p0` 相符、`metrics.steps=50000`；同樣的 G/K/λ/steps 也在 `config`。

```text
PASS: /archive/jasmine/cymaxwelllee/diag-g16/seedrep/s80001/results/G32K32_lam0p0.json
PASS: /archive/jasmine/cymaxwelllee/diag-g16/seedrep/s80001/results/G32K32_lam1p0.json
PASS: /archive/jasmine/cymaxwelllee/diag-g16/seedrep/s80002/results/G32K32_lam0p0.json
PASS: /archive/jasmine/cymaxwelllee/diag-g16/seedrep/s80002/results/G32K32_lam1p0.json
```

## 4. 負對照

把已知 G16K16 的內容暫放在 `/tmp/G32K32_lam0p0.json` 冒充 G32K32；沒有寫入任何
`results/`。驗收腳本輸出：

```text
FAIL: /tmp/G32K32_lam0p0.json: G/K mismatch: filename expects 32/32, JSON has 16/16
exit_code=1
```

壞檔已從 `/tmp` 丟棄。

## 5. 正對照

對已知完成好的 G16K16 λ0 檔、按其自身檔名預期 G/K 執行同一腳本：

```text
PASS: /archive/jasmine/cymaxwelllee/diag-g16/seedrep/s80001/results/G16K16_lam0p0.json
```

## 6. G32K32 數字抽驗

以下直接讀 JSON；數字保留原始位數，未四捨五入。

| seed | λ 檔名 | recon_a_val | recon_s_val | any_group_collapsed | mean_active_codes | train_seconds |
|---|---|---:|---:|---|---:|---:|
| s80001 | lam0p0 | 0.0006816941895522177 | 1.2355849742889404 | false | 32.0 | 2160.934250831604 |
| s80001 | lam1p0 | 0.009447637014091015 | 0.02419516071677208 | false | 32.0 | 2187.5861542224884 |
| s80002 | lam0p0 | 0.0006253454484976828 | 1.2658734321594238 | false | 32.0 | 2157.0547165870667 |
| s80002 | lam1p0 | 0.009439931251108646 | 0.023865167051553726 | false | 32.0 | 2189.152905225754 |

## 7. 健康格與 smoke 外推

- `any_group_collapsed=True`：0/4；四格都是 `false`。
- `mean_active_codes=32`：4/4；四格都是 `32.0`。
- smoke 基準：1500 步 66.5s；50k 外推為 `2216.6666666666665s`、
  `36.94444444444444 min`。

| seed | λ | log wall (s) | wall / smoke 外推 | 偏差 >1.5× |
|---|---|---:|---:|---|
| s80001 | lam0p0 | 2168.7 | 0.9783609022556391 | 否 |
| s80002 | lam0p0 | 2165.2 | 0.9767819548872181 | 否 |
| s80001 | lam1p0 | 2195.8 | 0.9905864661654137 | 否 |
| s80002 | lam1p0 | 2197.1 | 0.991172932330827 | 否 |

## 8. shat3 stdout 原文（不解讀）

```text
=== 結果 ===
(real) val ŝ MSE (外掛頭，正確配對)     = 0.119072
(a)    λ1 自帶頭 ŝ MSE（參照上界，讀檔） = 0.052035  (ratio real/lambda1 = 2.288, 越接近1越好)
(b)    爛錨 control ŝ MSE（碼打亂）       = 0.533421  (ratio bad/real = 4.480, [拍]門檻=2.0；只檢查碼×狀態配對依賴)
(c)    obs-only ŝ MSE（只給狀態）         = 0.234513  (ratio obs-only/full = 1.970；碼有增量資訊=True，判準 err_real < err_obs_only × 0.9)
判定：碼有增量資訊，但離λ1自帶頭有距離，見數字，人工判讀
```

## 自我懷疑

1. 本次 sandbox 直接執行 `ssh jasmine '<cmd>'` 時在 socket 層回 `Operation not permitted`；
   直接重跑 `sacct` 也因連 `zeldajr:6819` 被拒。四格產物則由同一 jasmine archive 的
   唯讀掛載讀到。第 1 節終態表是先前成功的遠端 monitor 所留的 `sacct` 原始摘錄，
   不是從 JSON 反推；本次無法在當前 sandbox 對 Slurm DB 做第二次 live query。
2. schema gate 是檔名推導預期 G/K/λ，再和 G16K16 參照逐 key 比對；它能抓到第 4 節的
   冒充檔，但不替代產物內容的科學審查。shat3 判定只照錄原文，沒有解讀。
3. 未讀取或納入 smoke0928 產物作為正式格；時間比較只使用工單給的 1500 步／66.5s
   基準與正式格 log 的 wall 錨。

STATUS: DONE
