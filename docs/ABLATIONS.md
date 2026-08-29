# LaCoT ablation 清單

_主人 2026-08-29 裁：主線不因這些變形分岔 —— 它們是 ablations，等主線定了再一輪掃。_
_新想到的變形先進這張表，⛔ 別直接動主線。_

## 待掃（主線定案後）

| # | 變形 | 開關 | 狀態與備註 |
|---|------|------|-----------|
| 1 | 每 chunk 完全重想 vs 接續修（主線） | `LACOT_GRAD_R_WARM=0` vs `10` | 主人 8/29 提出並裁歸 ablation。量「重新想會不會逃出爛計畫／接續修會不會被困住」 |
| 2 | learned refine（RefineOperator）vs 梯度爬坡（主線） | `LACOT_LEARNED_REFINE` | 8/28 已降級成 ablation；⚠️ 未訓練時 ≡ LayerNorm(u)，不是隨機垃圾 |
| 3 | chunk=1 vs 4 雙報 | `LACOT_CHUNK` | 對標公平性必要：chunk=4 的時間平滑本身值 0.15→0.85（POMDP 效應）。最終表兩個都報 |
| 4 | 爬坡步數 R = 0/1/3/5/8（test-time scaling 曲線） | `LACOT_GRAD_R` ×輪數 | 主線本來就列的 scaling 軸；修好 steps=R×GRAD_R 之後才可信 |
| 5 | (η, λ) 敏感度（rollout 端） | `LACOT_GRAD_ETA/LAM` | 機制層已有探針 3×3 格點（refineprobe_*_tasks-t2）：flow 起點最佳 η=0.1 λ=0.1~0.3、萬用檔 η=0.5 λ=1.0。rollout 端待驗 |
| 6 | K（query 數）rollout 端 | `LACOT_K` | decoder 側新考題已掃：K=4 夠、K=16 回頭（EXPERIMENT-INDEX Q1）。rollout 端待 |
| 7 | cond-dropout p | `LACOT_COND_DROP` | 現 0.1（head 讀 u 三件套之一）。值的敏感度沒掃 |
| 8 | subgoal 間距 | `LACOT_DELTA_SUB` | 現 7.5＝訓練分布路長中位（讓短程層坐在資料最肥處）。S1/S0 有訊號後再掃 |
| 9 | consistency 目標 self vs ema | `LACOT_CONS/EMA_M` | 8/23 anti-collapse 掃過：ema 系列全過、byol 全塌 —— 但每格 1 seed。若 refine 回主線才需重驗 |

## 已定案（不重掃，證據在索引）

- ENC_OBJ：`recon_ictr`（A2）勝 `recon`（A1）—— 替身指標反著看（A1 門票漂亮但 D0 1.2%、flow 塌成點質量）。見 EXPERIMENT-INDEX Q3。單 seed，headline 前補 multi-seed。
- T_CAP=128（主人 8/24 裁、8/28 落進預設）。
- 主線 refine＝梯度爬坡（主人的更新式），learned refine 退居 #2。
