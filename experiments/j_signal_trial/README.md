# J 訊號生死判：三臂 eval 包

`run_三臂.sbatch` 是 6 個 array job：每顆固定 ckpt 各跑 noclimb、climb、select。只組包；本副本沒有兩顆 ckpt，亦未提交或執行 rollout。部署目錄沿用 `slurm/run_rollout.sh` 的 `~/Projects/lacot`、本機 `/archive/cymaxwelllee/LaCoT/.venv/bin/python` 與 OGBench 資料路徑；Slurm 的 admin/it/great-mage、單 GPU、24 小時沿用 `slurm/night_0903_dialect.sh`。

**C 臂判準差異。** `docs/ENERGY-FRAMEWORK.md:143` 的 9/2 方言役是抽 8 條、選 `GeoEnergy` 最低者，即 `LACOT_GRAD_REFINE=1 LACOT_GRAD_MODE=select LACOT_SEL_N=8`（rollout:427-431, 2407-2418）。工單指定的 `LACOT_BON_N=8` 在目前程式卻選 **GrpoReward 最高**，並有 C8 乘法閘（rollout:1560-1568, 2683-2710, 2741-2750）；它不是 9/2 的 E selection，而且 `slurm/night_0906_bon_r0.sh` 記錄了長程題退化風險。為維持 9/2 的選樣分數，此包把 C 臂的 `BON_N` 固定為 0。不可把本 C 臂結果標成 BoN@8 結果。

**F6 前置條件。** 這份副本的 `lacot/refine_grad.py:153` 目前仍是 `wall = self.wall_depth(pts).mean(1)`，只查頂點；與工單「F6 已修」不符。sbatch 遇到此已知舊式碼會在 rollout 前停止。部署時須核對 `GeoEnergy.__call__` 的穿牆項確實有線段內插，才可把結果稱為 F6 後實驗。腳本亦會先檢查 ckpt 與 large-stitch 資料檔。

## 固定配置

表中 `rollout` 均指 `experiments/scratch_lacot_rollout.py` 的行號；其餘未列的環境變數須維持程式預設值。

| 設定 | A noclimb | B climb | C select | 依據 |
|---|---:|---:|---:|---|
| `LACOT_GRAD_REFINE` | 0 | 1 | 1 | rollout:421；分支 rollout:2395-2426 |
| `LACOT_GRAD_MODE` | climb | climb | select | rollout:427-431 |
| `LACOT_SEL_N` | 8（未用） | 8（未用） | 8 | rollout:431, 2411-2417；Energy Framework:143 |
| `LACOT_BON_N` | 0 | 0 | 0 | rollout:1560-1568；與 9/2 E selection 隔離 |
| `LACOT_GRAD_R` | 50（未用） | 50 | 50（選樣分支未用） | rollout:377, 2407-2426 |
| `LACOT_GRAD_ETA` / `LACOT_GRAD_LAM` | 0.1 / 0.3（未用） | 0.1 / 0.3 | 0.1 / 0.3（未用） | rollout:378-379, 2426 |
| `LACOT_GRAD_R_WARM` / `LACOT_W_LEN` | 10 / 0.3（未用） | 10 / 0.3 | 10 / 0.3（選樣使用 E 長度權重） | rollout:424, 382, 2486 |
| `LACOT_EVAL_RS` | 1 | 1 | 1 | rollout:2407, 3154, 3379；R=0 不會啟動爬坡或 selection |

| 共用設定 | 值 | 依據 |
|---|---|---|
| `LACOT_LOAD_CKPT` | 下表固定 s33 或 s35 路徑 | rollout:406, 1964-2043；load 時 `STEPS2=0`（1431） |
| `LACOT_SEED` | 對應 ckpt seed 33 或 35 | rollout:703, 2031-2037；官方題目 seed 在 rollout:3060 起按 task/episode 固定 |
| `LACOT_LOAD_EMA`, `LACOT_CONT_TRAIN` | 1, 0 | rollout:1993-2002, 3746-3749；兩臂共用 EMA 權重且只 eval |
| `LACOT_ENV` | `pointmaze-large-stitch-v0` | rollout:28-29；與 ckpt 的 large-stitch 對齊 |
| `LACOT_EVAL_EPISODES`, `LACOT_DEV_EVAL` | 40, 0 | rollout:2462；5 個官方 task × 每 task 40 集 = 200 題，輸出 `episodes` 須為 200（3117-3119） |
| `LACOT_CONS`, `LACOT_K`, `LACOT_COND`, `LACOT_CHUNK`, `LACOT_TCAP` | `self`, 8, 256, 4, 128 | rollout:709, 95-110；ckpt 名各段對齊 |
| `LACOT_ENC_OBJ`, `LACOT_LEARNED_REFINE` | `recon_ictr`, 0 | rollout:128, 138；ckpt 名與載入檢查（1970-1974） |
| `LACOT_COND_DROP`, `LACOT_BC_INDEP`, `LACOT_DEC_START` | 0.1, 1, `soft` | rollout:141, 145, 209；ckpt 名 `_cd0.1_bci_dssoft` |
| `LACOT_TEACHER_MIX`, `LACOT_WARMUP` | 0.5, 500 | rollout:461, 999；ckpt 名 `_tch0.5_wu500` |
| `LACOT_SUBGOAL`, `LACOT_FINISH_R`, `LACOT_INTENT` | 空、0、空 | rollout:156, 385, 217；只比較 flat 三臂 |
| `LACOT_U_SOURCE`, `LACOT_SUB_ESEL`, `LACOT_INTENT_GUID_W`, `LACOT_DEC_ANCHOR` | `flow`, 0, 0, 0 | rollout:173, 169, 308, 364 |
| `LACOT_GRAD_PROJ`, `LACOT_BON_MODE` | 0, `plan` | rollout:432, 1567 |
| `LACOT_BOOT_DATA`, `LACOT_S1_FROM` | 空、空 | rollout:550, 413；不引入額外資料或權重 |
| `LACOT_PREREQ`, `LACOT_FLOW_PROBE`, `LACOT_DIAG_DUMP` | 0, 0, 0 | rollout:116, 321, 3051；不加探針批次 |
| `LACOT_BOOT_TAG`, `LACOT_OUT_DIR` | `jsignal_${ARM}`, `results/j_signal_trial` | rollout:551, 3474-3475, 3638-3642；arm 進 JSON 檔名 |

ckpt 路徑（不得替換）：

| seed | 檔案 |
|---:|---|
| 33 | `results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt` |
| 35 | `results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s35.pt` |

## 輸出檔與讀法

六份 JSON 都在 `results/j_signal_trial/`；依 rollout 的 `_tag_extra`（rollout:3474-3578）與 `tag`（3634-3642）推得下列檔名。實際存檔時以 stdout 的 `寫入` 路徑核對，並檢查每份 JSON 的 `episodes == 200`、`rates.R1`、`load_ckpt` 與 `grad_refine`。

| array id | 臂 | ckpt seed | JSON 檔名 |
|---:|---|---:|---|
| 0 | noclimb | 33 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_noclimb_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_s33.json` |
| 1 | climb | 33 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_climb_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_gr50_s33.json` |
| 2 | select | 33 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_select_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_sel8_gr50_s33.json` |
| 3 | noclimb | 35 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_noclimb_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_s35.json` |
| 4 | climb | 35 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_climb_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_gr50_s35.json` |
| 5 | select | 35 | `rollout_large-stitch_self_K8_c256_ch4_st0_T128_ep40_gu_eorecon_ictr_tch0.5_btjsignal_select_emw0.999_ema_wu500_dssoft_norf_cd0.1_bci_sel8_gr50_s35.json` |

每顆 seed 各以 `rates.R1` 比較：若 `climb < noclimb` 且 `select > noclimb`，判為 **J 可挑不可爬**；其他序列不套用此判讀。先看兩顆各自的配對方向，再看合併趨勢；200 集跨 5 個官方 task，不能把 200 集當成 200 個獨立 task。
