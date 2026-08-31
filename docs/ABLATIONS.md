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
| 6 | K（計畫摘要長度）rollout 端；離散版與 codebook 大小聯掃 | `LACOT_K` | decoder 側新考題已掃：K=4 夠、K=16 回頭（Q1）。rollout 端待。⭐ 主人 8/29：離散版 K=句長、codebook=字彙量，聯掃小格網；最佳 K 可能隨環境難度變（large 路深一倍）＝好圖表；未來可信心自適應 K（同 M 的尺） |
| 7 | cond-dropout p | `LACOT_COND_DROP` | 現 0.1（head 讀 u 三件套之一）。值的敏感度沒掃 |
| 8 | subgoal 間距 | `LACOT_DELTA_SUB` | 現 7.5＝訓練分布路長中位（讓短程層坐在資料最肥處）。S1/S0 有訊號後再掃 |
| 9 | consistency 目標 self vs ema | `LACOT_CONS/EMA_M` | 8/23 anti-collapse 掃過：ema 系列全過、byol 全塌 —— 但每格 1 seed。若 refine 回主線才需重驗 |
| 10 | E_geo 四項權重（10/3/3/0.3）敏感度 | `refine_grad.py` GeoEnergy.w | 撞牆×10 是主人 8/26 裁示（大懲罰），其餘是初設沒掃過。reviewer 必問。η λ 已有探針格點、權重還沒有 |
| 11 | 中繼點選法：固定弧長 7.5 vs 信心選點 | `LACOT_SUBGOAL=conf/conf2` | 主人 8/29 提案。「信心」候選定義：(a) 同題多抽幾份 u、選多條計畫的【共識點】（跟 D4 多樣性量測共用機制、成本低）；(b) 候選短程 cond 的 flow 密度；(c) GeoEnergy 局部安全度。7.5 的現行理由＝訓練分布路長中位（短程坐在資料最肥處），是信心選點的固定近似 |

| 12 | 自適應抽樣數 M（主人 8/29：「不見得每次都要抽這麼多」） | 待實作 | sequential sampling：先抽 M_min=2 判信心，過門檻就用、不過再加抽到 M_max=8。信心高的題 2 份就走 ⇒ per-episode 算力分布更陡的自適應，compute-matched 曲線的加分項 |

## 已定案（不重掃，證據在索引）

- ENC_OBJ：`recon_ictr`（A2）勝 `recon`（A1）—— 替身指標反著看（A1 門票漂亮但 D0 1.2%、flow 塌成點質量）。見 EXPERIMENT-INDEX Q3。單 seed，headline 前補 multi-seed。
- T_CAP=128（主人 8/24 裁、8/28 落進預設）。
- 主線 refine＝梯度爬坡（主人的更新式），learned refine 退居 #2。

- **flow engine 對照（NF vs FM/RF）**〔主人 2026-08-31 裁「維持原案，頂多加 ablation」〕：
  同配方換 flow-matching 引擎訓一顆＋官方 — 把「為什麼用 exact-NF」變 empirical 決定。
  預期敘事：exact likelihood 的 tilt 蒸餾精確性 vs FM 的生成速度；⛔ 主線引擎不換。
