# NOTE — 完美教材開環播放為什麼還是只有 .554：量化損失 vs 執行漂移拆帳（2026-09-09）

_對應 [[NOTE-2026-09-08-teacher-relay]]（schedule 版 per-leg 不限時到達率 .554）與
[[NOTE-2026-09-08-relay-byleg]]（失敗 77% 是「走歪」不是「走慢」）之後的追問：這個
.554 的失敗，是【字典量化本身有損】還是【開環執行漸漸脫離教材軌跡】，各佔多少？
只量 schedule 版（「完美譜開環播放」＝這一版；nn 版是閉環修正，不在本次範圍）。_

_全部程式：`experiments/walk_verify/drift_analysis/`（新目錄，只 import 上一層
`wv_common.py`／`p1_replay.py`與同目錄 teacher_relay 的
`run_teacher_relay_byleg.py`，未修改任何既有檔案）。一手 log：
`experiments/walk_verify/drift_analysis/logs/drift-analysis-26830.out`（最終正確版；
26826／26828 是畫圖 bug 的兩次迭代，log 留著，見§四）。_

**在證明什麼、判準是什麼**（工單原文，抄一份在這裡對齊）：定量拆解 .554 失敗的機制，
判準＝三張可重現量測——①漂移曲線（成功/失敗題分開）②拆帳表（重置對照臂拆量化 vs
漂移）③漂移-失敗關係（有無可用門檻，沒有也如實報）。

---

## 〇、一句話

**執行漂移是大頭，但量化／單段執行誤差不是零貢獻，漂移對失敗的預測力只有中等。**
拆帳表：單一 chunk 的純量化基線 E_indep≈0.049m，但開環連續執行的漂移 D_cont 在
j=10（約一個 leg 的長度）就漲到 1.33m（淨貢獻是基線的 26 倍），j=50（整集預算用完）
漲到 7.57m（淨貢獻是基線的 153 倍）——而且比「每 chunk 誤差不相關地線性加總」的天真
預測（50×0.049≈2.45m）還多出 3 倍，代表漂移不只是累積、還會自我放大（decode 吃到
越來越偏的 obs）。但失敗前兆分佈顯示漂移不是唯一原因：leg 1（起點定義上漂移恆為 0）
仍有 21.5% 失敗率，遠低於全體 44.6%，但不是 0——代表就算完全沒有漂移，單段本身的
量化／執行誤差也能造成失敗；漂移量對失敗的預測力是中等（AUC=0.770，最佳單一門檻
accuracy 74.2% vs 多數類 baseline 55.4%），不是乾淨可用的門檻。

---

## 一、量測構造

沿用 `run_teacher_relay_byleg.py` 的題目構造（`build_tasks`，**直接 import 不複製**，
保證跟原 run 同源）：seed=20260908、val 檔前 200 條可用 episode（跳過 episode 386）、
路標每 7.5 弧長（`DELTA_SUB`）、chunk 每 4 步、rho=1.875。**確認 EP_LEN 固定 201 步
⇒ 全部 200 題 n_chunks 都是 50**（原 docstring 寫「通常 50」，這次量出來是全部都是 50，
不是「通常」，見程式碼裡的 assert）。

兩條新臂（因為要加新記錄，`run_one` 無法只 import，複本＋擴充自
`run_teacher_relay_byleg.run_one` 的 schedule 分支，邏輯逐行對齊，見程式檔頭註解）：

- **臂 1（cont，開環連續執行）**：跟原 schedule 版完全同一條軌跡（新增的記錄只讀
  `u.data.qpos`／呼叫 `model.norm_obs`，不寫任何模擬狀態、不影響 `env.step` 的呼叫
  序列），額外在每個 chunk 邊界 j（= 第 j 個 chunk 即將開始那一刻）記錄：xy 漂移
  （模擬 xy 對教材軌跡第 j 個 chunk 起點真 xy 的距離）、state 漂移（`model.norm_obs`
  normalize 後的 29 維 obs 距離，用 decoder 自己內部就在用的同一個 normalize，不是
  另外發明的尺）、當下 active leg。另外用「leg m 開始那一刻」的精確物理步數
  （`leg_active_step[m]`，對應教材時刻 s0+leg_active_step[m]）算 leg-start 漂移。
- **臂 2（indep，重置對照）**：跟 `p1_replay.py` 的 `dict_indep` 同定義，但套用在
  200 題×50 chunk 這個網格上：每個 chunk 開始前 `set_state` 回教材真起點，執行 1
  chunk，量這 1 個 chunk 的位移誤差／state 誤差。不追蹤到達率（重置後軌跡鎖回真
  路徑，到達率無意義，工單原文已點名）。**固定跑滿全部 50 個 chunk**（不像臂 1
  提早結束）——這樣每個 chunk index 都有全部 200 題的基線樣本。

**受控變因自檢（重要，是拆帳因果解讀成立的前提）**：`encode_idx` 只吃動作段，不吃
obs，所以臂 1／臂 2 在同一個 (task, chunk index) 上理論上必須選到同一個字——本次
9485 次比較（臂 1 早停前實際跑過的 chunk 數）**0 次不一致**，證明兩臂唯一的受控
差異就是「obs 有沒有漂移／狀態有沒有重置」，拆帳的因果解讀站得住。

---

## 二、復現斷言（做到，附對帳）

3 條題目（ti=0/2/3，涵蓋「部分成功後失敗」「M=4 從頭失敗」「完整成功」三種路徑）
逐欄位（m／reached／active_step／end_step／first_hit／d_start／d_end）跟
`teacher_relay_byleg_summary.json` 的 `leg_rows` 對帳，**全部 PASS**：

```
ti=0 episode=319 M=3 legs=2 completed(gt/mine)=False/False  對帳=PASS
ti=2 episode=56  M=4 legs=1 completed(gt/mine)=False/False  對帳=PASS
ti=3 episode=423 M=2 legs=2 completed(gt/mine)=True/True    對帳=PASS
```

額外（比工單要求的 3 條更嚴、免費多做）：200 題全量 completed／legs_attempted／
legs_reached／n_steps_run 跟 `teacher_relay_summary.json` 的 `per_task` 逐項對帳，
800 項（4 欄位 x 200 題）**全部一致**；本檔 completion_ratio=0.125，跟原 note 逐位元
相同。兩次獨立重跑（job 26826 vs 26828，中間只改了畫圖程式碼，量測邏輯沒動）Step
1-4 的數字輸出**逐位元相同**（`diff` 兩份 log 無差異），確認整條 pipeline 全確定性。

---

## 三、三張主產出

### ① 漂移曲線（`results/drift_curve_xy.png`）

x 軸＝chunk index j（第 j 個 chunk 即將開始那一刻），y 軸＝模擬 xy 對教材軌跡同一
時刻真 xy 的距離，兩組（完整走完 vs 中途卡死的題）各畫 p50 實線 + p25-p75 色帶。

讀法：兩組漂移都隨 j 上升，卡死題（紅）明顯比走完題（綠）漲得快、漲得高——完整走完
的題本身漂移也在 2-3m 量級打轉，不是貼著 0。**走完題的資料量在 j>35 之後迅速變薄**
（n 從 25 一路掉到 j=44 之後剩 0——完整走完的 25 題全部在 j<44 就結束了），曲線末段
只是描述性參考、不是統計有力的比較，圖例已標每組的點數，不掩蓋。

### ② 拆帳表（`results/drift_summary.json` 的 `accounting_table`，數字表非圖）

E_indep（臂2 全部 chunk 合併中位數，「純量化」基線）：**xy=0.0491m　state=2.2842**

```
   j  n_active  D_cont_xy_p50  E_indep_xy_p50(k=j-1)  net_xy=Dcont-E_overall
   1       200         0.0474                 0.0474                -0.0017
   2       200         0.1561                 0.0433                 0.1070
   3       200         0.3049                 0.0465                 0.2559
   4       200         0.4682                 0.0460                 0.4191
   5       200         0.5979                 0.0455                 0.5488
   6       200         0.7485                 0.0442                 0.6994
   8       200         1.0900                 0.0530                 1.0410
  10       200         1.3289                 0.0432                 1.2798
  15       198         2.3332                 0.0488                 2.2842
  20       198         4.0739                 0.0443                 4.0248
  30       187         6.7759                 0.0517                 6.7269
  40       178         7.4011                 0.0533                 7.3520
  50       175         7.5684                 0.0499                 7.5193
```

`E_indep_xy_p50(k=j-1)` 那欄幾乎在 0.043~0.053m 之間打轉——證實「單 chunk 量化基線
不太隨 chunk index 變」這個假設站得住，用單一個 E_indep_overall=0.0491m 當扣除基線
是合理的。**自檢 #2（首 chunk 誤差同量級）**：臂1 j=1 的 xy drift median=0.0474m
(n=200) vs 臂2 k=0 median=0.0474m (n=200)，**比值=1.000**；state drift 兩邊都是
2.1416，比值同樣=1.000。⚠️ 誠實講這條自檢比工單原文要求的「同量級」更強到近乎
逐位元相等，**原因是構造上保證、不是巧合或深刻的經驗發現**：j=1（臂1）跟 k=0（臂2）
都是「從同一個真起點 s0 reset、選同一個字（已由字選擇自檢確認）、執行同一段動作」，
是同一個計算，逐位元相等是代數上必然的結果。它的價值是**驗證程式碼沒寫錯**（兩個
分開寫的函式互相對得上），不是驗證某個經驗假設——真正有經驗內容的是 j>1 之後兩條
數字開始分岔，那才是漂移在發生作用的地方（見拆帳表其餘各列）。

state drift（normalize 後）的逐 j 中位數（`accounting_table` 裡的
`d_cont_state_p50` 欄）：j=1 時 2.14、j=10 時 4.73、j=50 時 4.73——漲得比 xy 溫和
（且到 j=15 左右就大致飽和），只留數字沒有另外畫成第二張分位數帶圖（見§四第 1 項，
是刻意的範圍取捨，不是漏做）。

### ③ 失敗前兆分佈（`results/drift_failure_precursor.png`）

x 軸＝該 leg 開始那一刻的 xy 漂移（截到 0-5m 方便看兩條線怎麼分岔，完整分位數含
尾巴在 json 裡沒有被砍），y 軸＝密度。失敗 leg n=175、成功 leg n=217（217/392=.554，
跟原 note 的 per-leg 到達率逐位元對上）。

```
                    p25     p50     p75     p95     p99
失敗 leg 開始漂移   0.249   1.183   2.496   8.007   9.174
成功 leg 開始漂移   0.000   0.000   0.358   1.819   2.306
```

AUC 分離度（P(隨機失敗樣本漂移 > 隨機成功樣本漂移)，0.5=分不開、1.0=完全分開）
**= 0.770**。最佳單一門檻 thr=0.235m，accuracy=74.2%，比多數類 baseline
（55.4%，永遠猜「會到達」）高 18.8 個百分點——**有一個中等強度、可用但不乾淨的
門檻**，不是「完全分不開」也不是「一刀切準確」。

leg-1 專門子群（leg_start_drift 依構造恆為 0，n=200）：失敗率 **21.5%**，遠低於
全體失敗率 44.6%，但不是 0——這是「即使起點完全沒漂移，單段內的量化/執行誤差本身
也會造成失敗」的直接證據，跟②拆帳表看到的「執行漂移量級遠超量化基線」放在一起看，
形成本篇的判讀主軸（見§〇、§四判讀邏輯）。

---

## 四、拆帳判讀邏輯（依據哪張表）

判讀依據**②拆帳表為主、③失敗前兆分佈為輔**：

1. ②顯示 D_cont（開環連續執行的漂移）在 j=10 就是 E_indep（單 chunk 量化基線）的
   ~27 倍（1.3289/0.0491）、j=50 是 ~154 倍（7.5684/0.0491）——量化基線本身很小
   且穩定，累積下來的漂移量級完全由「執行漂移」主導，不是量化損失的量級能解釋的。
2. 用最天真的「每 chunk 誤差互不相關、線性加總」模型算一個參考值：j×E_indep_overall，
   j=50 時 ≈ 50×0.0491=2.455m，遠低於實測的 7.5684m（約 3.1 倍）——代表漂移不只是
   噪音累加，還有正回饋（decode 吃到的 obs 越漂越偏，選出的動作也跟著更偏）。這句是
   從①②兩張表的形狀推論出來的描述，不是另外做了因果實驗量到的，標記為推論。
3. ③顯示漂移量對「這個 leg 會不會失敗」有中等預測力（AUC=.770），不是接近 0.5（分
   不開）也不是接近 1.0（乾淨門檻）——所以漂移是失敗的重要因子，但不是唯一因子。
4. ③的 leg-1 子群（起點漂移恆 0）失敗率 21.5%，直接量到「漂移=0 時仍有的失敗地板」，
   這個地板只能來自量化／單段執行誤差本身（因為那一刻沒有累積漂移可以背這個鍋）。
5. 綜合 1-4：**執行漂移是量級上的大頭（②），但量化/單段執行誤差撐起一個不可忽略
   的失敗地板（③的 leg-1 子群），且漂移對失敗只有中等預測力（③的 AUC）——「失敗
   全部由漂移解釋」站不住，「漂移完全不重要、全是量化的鍋」也站不住。**

---

## 五、沒做到 / 不確定清單

1. **state drift 沒有另外畫成分位數帶圖**——只有 xy drift 有專門的曲線 PNG，state
   drift 的逐 j 中位數只留在拆帳表的數字欄裡。這是刻意的範圍取捨（工單要求「三張
   主產出」，額度用在 xy 曲線 + 失敗前兆分佈），不是漏做；數字都在
   `drift_summary.json` 的 `accounting_table`／`drift_curve.state_completed`／
   `drift_curve.state_failed` 裡，要重畫隨時可以從這份 json 生圖。
2. **「執行漂移有正回饋、不只是線性累加」是§四第 2 點的描述性推論**，不是另外做了
   對照實驗（例如「拆開純累加 vs 有回饋」兩種合成模型去對比）量出來的——只是拿①②
   兩張表的形狀跟一個天真的線性參考值比大小，讀者可以照 json 裡的數字自己重算，
   但這句判讀比其他幾句弱，標記出來。
3. **失敗前兆分佈是所有 leg index m 合併（pooled）算的**，只額外拆出 leg-1 這個
   特殊子群（漂移恆 0），沒有對 m=2/3/4 個別做 AUC／門檻——`drift_raw.npz` 裡留了
   `leg_ti/leg_m/leg_M/leg_reached/leg_start_drift` 逐列資料，要拆可以直接拆，
   本次工單範圍沒有要求拆到這一層。
4. **只有 1 個字典 seed、1 次模擬**（沿用同一份 ckpt p0_dict_v1_L4K32_50k.pt），沒
   有量 run-to-run variance——跟 teacher-relay／relay-byleg 兩篇既有 NOTE 同一個
   開放項，不是本次新增的缺口（本次流程本身全確定性，兩次獨立重跑數字逐位元相同，
   已在§二點名）。
5. **畫圖程式（`drift_plots.py`）第一版有 bug，點名不掩蓋**：第一次跑（job 26826）
   完整資料量到／算對了，但 `draw_percentile_band` 沒有濾掉「completed 組在 j 很
   大時沒有樣本」產生的 NaN，畫出一條座標爆掉的雜線；且 ylabel 跟 title 疊字。已修
   （NaN 先過濾成只畫有樣本的子集、ylabel 併進 title 一行）並重新提交驗證圖形正確
   （job 26828 數字跟 26826 逐位元相同，只有圖修好；job 26830 再把失敗前兆分佈圖的
   x 軸縮到 0-5m 方便看兩條線分岔，數字同樣沒變）。三次 log 都留著沒刪
   （26826/26828/26830），可以自己對照。**這裡沒有刪除或覆蓋掉「不同條件」的結果**
   ——三次是同一套 200 題同一套算法，只有畫圖呈現方式在修，數字本身逐位元未變，
   跟 teacher-relay NOTE §五#6 點名過的「不能用 rm 重來」情況不同（那次是刪掉舊
   結果重新產生；這次是同 tag 疊代同一份分析直到畫圖正確，Steps 1-4 的計算結果從
   未被刪除或視為需要重新驗證的對象）。
6. **maze 邊界/物理限制對漂移飽和的可能影響沒有深入查**——①漂移曲線在 j>30 之後
   明顯趨緩（可能是因為卡死的螞蟻在迷宮邊界或角落附近打轉，漂移量被空間本身的有限
   範圍限制住），這是看圖描述出來的觀察，沒有另外去量「卡死的螞蟻最終停在哪裡」
   來證實，工單範圍沒有要求做這一層。
7. 沒做到的「查不到／跑不出」項目：無。復現斷言、200 題全量、臂1/臂2 全部跑完，
   兩張圖與 json/npz 都有存檔，log 是一手（`logs/drift-analysis-26830.out`）。

---

## 一手 log / 產物路徑

```
程式：experiments/walk_verify/drift_analysis/run_drift_analysis.py（主程式）
      experiments/walk_verify/drift_analysis/drift_plots.py（新畫圖工具，百分位帶）
      experiments/walk_verify/drift_analysis/run_drift_analysis.sbatch（CPU-only 提交）
log： experiments/walk_verify/drift_analysis/logs/drift-analysis-26830.out（最終版一手 log）
      experiments/walk_verify/drift_analysis/logs/drift-analysis-26826.out（迭代#1，畫圖 bug，數字正確）
      experiments/walk_verify/drift_analysis/logs/drift-analysis-26828.out（迭代#2，畫圖修 NaN，數字同26826）
json： experiments/walk_verify/drift_analysis/results/drift_summary.json
npz：  experiments/walk_verify/drift_analysis/results/drift_raw.npz
圖：   experiments/walk_verify/drift_analysis/results/drift_curve_xy.png
      experiments/walk_verify/drift_analysis/results/drift_failure_precursor.png
對帳來源（既有檔，唯讀）：
      experiments/walk_verify/teacher_relay/results/teacher_relay_summary.json
      experiments/walk_verify/teacher_relay/results/teacher_relay_byleg_summary.json
```
