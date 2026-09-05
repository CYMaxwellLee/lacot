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

## 9/5 晚·對抗審查 — ルナ逐條複判版（⛔ 初版寫「凍結」是被攻擊手嚇到，主人糾正：
## 逐條檢查→判斷→定參考價值。設計【不凍結】、修訂如下）

**ルナ複判四級**：
- N1（rev×rank 聯立矛盾）＝**採納、但一行可修**：rev 增強樣本逐出 triplet 池即解。
  更好的結構解（v2 候選）：e 拆 [e_metric‖e_dir] 子空間 — L_rev 只作用 e_dir、rank 只
  作用 e_metric ⇒ N1 與「時間箭頭捷徑」（④）同時結構性消滅。
- N2（d_bfs 判準循環）＝**修訂採納**：非「必然無效」而是證據力弱 — 修法＝交叉形式評估
  （d_time 訓→對 d_bfs 評；或 held-out 區域對）。協定級修正、不動設計。
- N3（合法率平凡解）＝**降級**：既有階梯第 2 步（stage-2 subgoal/R0 不退步 gate）本來就
  蓋住塌掉的 decoder — 攻擊手沒對上提供的階梯。採 C8（位移/路徑長欄）當便宜加固。
- ③（density 病 vs ordering 藥）＝**高參考價值、當假設測**：強版（「metric 越好插值越糟
  ＝預設結果」）過度宣稱 — 非凸性繼承假設低維等距嵌入，K×D 高維空間弦有空間貼流形。
  測法＝C5（弦中點 k-NN vs Gaussian）。若證實 ⇒ v2 加 density 成分：**interp-consistency**
  （用訊號三的 legality head 當可微場、把弦中點往高合法區推 — 訊號三升級成密度藥）。
- C4＝**降級使用**：座標 lerp 本來就會穿牆 — 它是 baseline 樓地板（latent 目前有沒有
  贏過天真平均），不是 falsifier。
- d_time 偏差（②）＝**採納**：C2 量出 Δt 死點、只用短 Δt 段＋長程交給 d_bfs 臂。
- C9（ictr 順序盲）＝**採納**（一次 grep 的事）。

**修訂後前置檢查（跑這四支、CPU）**：C5（密度 vs 排序）、C1（rev 對現況）、C2（d_time
效度）、C9（取樣器）；C4 當 baseline 欄一起出。過了再進 patch。
〔9/5 晚收官：C1b 成立（rev 排池）、C2 兩判準不成立（d_time 堪用 rho .758）、C5 強版
證偽（病在 decoder 對模式間輸入）、C4 樓地板 53.5%<63.8%、C9 無雷（ictr 本來就不用反轉
增強）。⇒ **解凍、進 v2**。詳 FINDINGS-0905 ⑨。〕

## v2 規格（C-battery 後定形；待主人過目）

**結構**：e ∈ R^{K×D} 攤平後拆兩個子空間 — e_m（metric、前 D_m 維）與 e_d（direction、
後 D_d 維，建議 D_d=16）。⛔ 拆分只在 loss 計算層、不動模組結構（zero-diff 契約可守）。

1. **L_rev（訊號一、只作用 e_d）**：cos(e_d(τ), e_d(rev τ)) → −1。C1 證明現在方向整個
   不存在（+0.99）— 只要求一個小子空間長出方向、不強迫整體翻轉。rev(τ) 照舊餵 recon。
2. **L_rank（訊號二、只作用 e_m）**：三元組 hinge、d=d_time 主臂（C2 驗過 rho .758、
   反向 13.4%）；⛔ rev 增強樣本排除於三元組池（C1b）。d_bfs 上界臂另跑、評估交叉形式
   （d_time 訓→對 d_bfs held-out 對評，N2 解法）。margin 用距離分位數自適應（防 loss3
   撐大範數讓 margin 失效）。
3. **L_leg（訊號三、升級版）**：牆內擾動 negative＋legality head（BCE）；⭐ 加
   **interp-consistency 項**：batch 內真 e 對的弦中點過 legality head、往 legal 推
   （C5 證明病在模式間輸入 — 這項直接雕那裡）。⛔ C8 非退化欄（decode 位移/路徑長）
   進探針報告、跑後必看。
4. env：LACOT_S1_REV_W／RANK_W／LEG_W／INTERP_W（全 default 0＝零行為差）＋
   LACOT_S1_DIR_DIM（default 0＝不拆）；tag `_s1g...` 進檔名。
5. 階梯照舊：patch＋golden smoke → 單顆 stage1（before 尺量 after：插值合法率>.757、
   rho 用交叉形式評、recon gate）→ 疊 stage2 單顆（subgoal/R0 gate）→ 八顆。

**必然級三條（跑之前就判死的）**：
- **N1（條件式必然）**：rev 對若進 triplet pool — d_time/d_bfs 對反轉皆近似不變 ⇒ rank 要
  它最近、L_rev 要它最遠，聯立不可滿足。規避：rev 對逐出 pool 或改方向感知 d。
- **N2（判準循環）**：d 取 d_bfs 時「rho 抬升」＝訓練目標與評估統計同一個量 — 不構成證據。
  評估必須用訓練未觸及的 state 對或不同函數形式的 d。
- **N3（平凡解）**：插值合法率有平凡極大解（decode 塌短/常數 ⇒ legality→1、loss3 正往
  保守推）— 合法率必須與非退化量（位移/路徑長/多樣性）成對釘死。

**最深一刀（③）**：「metric 修好 ⇒ decode(lerp) 合法」因果鏈在 decoder 斷 — off-manifold
行為無 loss 約束；且 metric 若真變成迷宮圖度量的等距嵌入，e 繼承迷宮【非凸性】⇒ 直線弦
穿牆、metric 越好插值可能越糟。已量的病徵（t=.5 最低、Gaussian 贏弦）是 **density/support
洞**、不是 ordering 錯 — 三個 loss 開的是 ordering 的藥。⇒ 設計要補 density 那半
（interp-consistency／prior shaping／或把「插值」改成測地線插值而非線性弦）。

**其他高風險**：反演 cos 的尺度不變捷徑（大範數時間箭頭座標、淹掉 rank、弦中點變 OOD）；
d_time＝√Δt 壓縮＋混合時間後歸零＋監督域（同軌跡）與評估域（跨軌跡）不相交；
d_bfs(start,start) 方向盲長度盲、同起點反向對與 recon 對撞；ictr positive 若順序盲
與 L_rev 反號（C9 查）；loss3 在評估從不造訪的區域雕花（C7 查）。

**C-battery（九支 CPU 前置檢查、各 ≤30 分；先跑 C4/C5/C1/C2）**：
C1 rev 對現況 cos 與距離百分位｜C2 d_time 效度（Spearman(Δt,BFS)、反向三元組比例 >15% 即
有錯拉底線）｜C3 d_bfs 退化度｜**C4 非凸性判死：座標空間 lerp 合法率 ≈ latent lerp ⇒
病在迷宮非凸、本提案前提失效（最強 falsifier）**｜C5 弦中點 k-NN 距離（密度洞 vs 排序）｜
C6 rank 一致性×中點合法相關（因果鏈直測）｜C7 loss3 雕哪裡｜C8 非退化門檻預釘｜
C9 ictr 取樣器順序盲檢查。
