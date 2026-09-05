# 路線一 loss 設計 — 三個訓練訊號（主人 9/5 提出、ルナ落形）

_目標：給 e_target 的 z 空間長出測地線結構。before 尺（9/5 探針、[實測]）：插值合法率
.705（低於亂猜 .757）、latent-BFS 距離 Spearman rho=.212。三訊號全是 stage-1（e_target）
loss 追加；現有 contrastive 端子＝recon_ictr 的 w_ictr=0.2。_

## 訊號一：方向反演（A→B vs B→A 高度負相關）

- **免費自監督對**：任何軌跡 τ 時間倒放＝rev(τ)（同一套正規化後 flip token 順序）。
- **loss 形**：L_rev = 1 + cos( e(τ), e(rev(τ)) ) → 把餘弦壓向 −1（＝主人的「高度負相關」）。
  rev(τ) 同時餵 recon（它就是合法軌跡、白拿的資料增強）。
- 買到：空間長出「方向」座標＝quasimetric 不對稱性的入口。
- 風險：反演對稱約束吃容量 → 張力錶（recon、之後的 R0）一起盯。

## 訊號二：跨路徑 contrastive／metric 對齊（rho 那格的直接藥）

- **rank 形（首選、比 InfoNCE 更直接對 rho）**：三元組 (A,B,C)、若 d(A,B) < d(A,C)
  則要求 ‖e_A−e_B‖ < ‖e_A−e_C‖ − margin（hinge）。直接把 latent 距離排成環境距離的序。
- **⭐ 設計岔（要主人的眼睛）**：d 用哪把尺 —
  - `d_time`＝同軌跡內兩段的時間間隔（**可攜**：高維、無圖都有；teacher-agnostic，
    跟大方向濾鏡對齊）→ **建議當主臂**。
  - `d_bfs`＝佔據圖 BFS 距離（**精準**但 maze 限定＝儀器）→ 當上界對照臂。
- 負樣本：跨軌跡 windows（batch 內互為 negative）；InfoNCE 版保留為備選。

## 訊號三：不合法推開（70.5 那格的藥）

- **negative 製造**：真軌跡 waypoints 往牆內擾動（佔據圖現成）→ encode → 兩用：
  ① 進訊號二的 negative 池（推離合法流形）② aux 合法性 head（BCE、e → P(legal)）。
- aux head 白拿一個推論期濾器（decode 前先看 P(legal)）。
- 探針抓到的穿牆中點之後也能回收進 negative 池（先用擾動版、管線簡單）。

## 最小驗證階梯（照 9/5 濾鏡＋階梯）

```
0. patch：三訊號 env 開關（w=0 零行為差 golden ⛔ 老契約）＋檔名 tag（不進檔名會蓋）
1. 單顆 stage1 變體（s27 同款＋三訊號小權重 0.1/0.1/0.1）
   → 用 before 尺量 after：插值合法率要 > .757（贏亂猜）、rho 明顯抬（≥.35 算清楚贏）
   → gate：recon 不明顯退步（張力錶第一格）
2. 過了→疊 stage2（f27n 配方）單顆：subgoal/R0 不退步 gate（張力錶第二格：R0 特別盯）
3. 都過→八顆。任一步 fail＝回設計，⛔ 不硬調權重刷過
```

環境變數（實作時）：LACOT_S1_REV_W / LACOT_S1_RANK_W（LACOT_S1_RANK_D=time|bfs）/
LACOT_S1_LEG_W；tag `_s1g<rev><rank><leg>` 進檔名。施工＝使魔＋opus 驗收（9/4 流程重放）、
判讀與驗收ルナ；e_target 核心方程式的變動屬「跟 Rei 一起扛」範圍 → converge 後交接包裡
這節要附完整 ablation。
