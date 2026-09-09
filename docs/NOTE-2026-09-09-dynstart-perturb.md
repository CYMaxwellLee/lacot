# NOTE — 主人裁示①②：動態隨機起點 + 漂移型擾動（rewrite-v1 步2 升級版，2026-09-09）

_對應 [[NOTE-2026-09-09-midstart-aug]] 定讞的『固定 2 刀』教材（R=8 三 seed 均值
.6639，std≈.026）。主人裁示升級兩件事：①切點改成訓練時每步動態隨機挑（不預先剪
死，起點連續覆蓋整條軌跡）；②起點姿態加漂移型擾動（教 u「題目的起點本來就可能
歪」，對應重寫時真實遇到的漂移態，擾動幅度校準到 walk_verify/drift_analysis 量到
的真實漂移分佈，不是拍的常數）。⚠️ 這次擾動是『腦的題目端』（u 的條件起點 obs）,
跟已判無效的『decoder obs 噪音』（身體執行端、i.i.d. 形狀）是兩件不同的事，設計
上刻意區分開。_

**在證明什麼、判準是什麼**（工單原文逐字照抄）：
1. 動態隨機起點 vs 固定 2 刀——R=8 重寫的 3-seed 均值超過 2 刀版 .6639（差 ≥.02）
   或 3 顆散佈（std）明顯小於 2 刀版的 ≈.026，兩個子判準各自獨立檢查。
2. 起點加漂移型擾動再多賺多少——擾動組相對無擾動組的 R8 均值差 ≥.02 算有效。
   兩問各自「無差」都是合法答案，如實報。

全部程式（新檔，未修改任何既有檔案；本 worktree `lacot-dyn`、分支 `rewrite-dyn`）：
`dyn_corpus_common.py`（軌跡級語料資料結構＋動態抽樣核心函式）、
`build_corpus_dyn.py`（建軌跡級語料＋驗收①自檢＋驗收④擾動幅度稽核，CPU、不經
slurm，同 build_corpus.py 慣例）、`train_gen_sft_dyn.py`（GPU 訓練，動態每步取樣）、
`train_gen_sft_dyn.sbatch`（六顆訓練共用）、`eval_rewrite_dyn.sbatch`（直接呼叫既有
`eval_rewrite_aug.py`，機制 100% 沿用，不改一行）、`analyze_dyn.py`+
`collect_dyn.sbatch`（收表算判準）。

---

## 一、機制：軌跡級語料 + 每步動態取樣

### 1.1 跟固定 2 刀版（aug）的關鍵差異

固定 2 刀版在 corpus-build 時就把每條 episode 的 2 個切點抽死、造成靜態子樣本存進
`corpus_aug_v1.pt`，訓練時只是對這個固定清單做 index。本工單設計點①要求「不預先
剪死」——所以語料本身改存**軌跡級原料**（`corpus_dyn_v1.pt`，經
`dyn_corpus_common.build_trajectory_record()` 建出），每條 episode 存：

- `obs_chunks`：每個 chunk 邊界 k=0..n_chunks-1 的『真實』obs（`raw["observations"]`
  直接索引，不模擬）——邊界安全性：只存到 k=n_chunks-1（不含 k=n_chunks 那個『最後
  一個 chunk 之後』的邊界），因為 `4*(n_chunks-1) <= T-4 < T`，保證每一列都落在該
  集自己的資料範圍內，不會讀到下一集或越界（`build_corpus_dyn.py` 有邊界自檢，見
  §三）。
- `code_seq` / `wp_xy`：完整字串／完整路標序列（母 episode 全長）。
- `wp_local_idx`：每個路標第一次被真實軌跡到達的 local timestep（重新呼叫一次
  `rt.build_tasks()` 取得，跟 aug 版取得方式完全相同，本檔自檢①逐位元驗證是同一組
  task，見 §三）。
- `usable_cuts` / `usable_cuts_idx`：預先過濾好『哪些中途 chunk 邊界 k 可用』（避開
  最後 4 個 chunk、且切下去後至少留 1 個尚未到達的路標）＋對應的『尚未到達路標』
  起始 index——這就是工單設計點①說的『路標弧長表——或等價可即時切的形式』的具體
  實作：`wp_local_idx` 本身就是既有 aug_corpus_common 判定『尚未到達』的機制，這裡
  只是把每個 episode 的可用切點事先篩好存起來，訓練時 O(1) 查表，不必每步重算
  searchsorted，也不需要 retry 邏輯（候選已經是『保證可用』的子集，均勻抽樣落在
  這個子集上，數學上等價於『對全部候選做拒絕抽樣直到可用』，但不用真的重試）。

**真正的『動態』發生在訓練迴圈**（`train_gen_sft_dyn.py` 的 `iterate_batches_dyn`）：
每次取樣，對抽到的 episode 丟一枚硬幣（`cut_rng`）——50% 用原起點（k=0，從頭）、
50% 從該 episode 的 `usable_cuts` 均勻隨機挑一個 k，即時建出 (條件, 目標) pair。
教材因此『連續覆蓋整條軌跡』：任何一個 usable 的中途點都可能在任何一次訓練被抽到，
不像 aug 版死死只有 2 個固定點。

### 1.2 隨機性分流（工單要求「跟權重初始化 seed 分開記錄」）

三條獨立 rng 流，彼此不互相消耗：

| 用途 | 變數 | 固定值 | 是否隨 3 個訓練 seed 變動 |
|---|---|---|---|
| 權重初始化 + batch-order（哪些 episode 進哪個 batch、順序） | `--seed` | 20260909/20260910/20260911 | **是**——這是 3-seed 比較唯一該變動的自由度 |
| 切點選擇（原起點 vs 中途切點、切哪個 k） | `--dyn-cut-seed` | 20260909（`dc.DYN_CUT_SEED`） | 否，六顆 dyn 訓練全部共用同一個值 |
| 擾動（方向/幅度） | `--perturb-seed` | 20260912（`dc.PERTURB_SEED`） | 否，僅擾動組 3 顆共用同一個值 |

**理由**（本檔的方法論選擇，非工單逐字規定，這裡明講）：如果『哪個點被切到』或
『擾動多大』也隨訓練 seed 變動，3-seed 的 R8 散佈（std）就會同時混進『權重初始化
+batch-order 的抖動』跟『切點/擾動抽樣的抖動』兩種來源，沒辦法乾淨對應到判準1
要量的『同語料同架構只換訓練 seed』的那種抖動（跟 2 刀版 std≈.026 的定義口徑一致）。
固定切點/擾動流之後，dyn 三顆之間的差異來源跟 2 刀版三顆之間的差異來源是同一種
（只有權重初始化+batch-order 不同），兩邊的 std 才是可比的。

每個 episode 每『輪』（reshuffle 週期）被抽到恰一次，週期長度＝train episode 數
（4457，跟 `corpus_v1.pt` 的 train 集完全同一批，見 §1.3），跟固定 2 刀版的
`iterate_batches` 對『樣本清單』重洗牌是同一個角色，只是這裡洗牌的對象是『episode』
不是『預先切好的樣本』。

### 1.3 Held-out 評測樣本

沿用 `corpus_aug_v1.pt` 的 `val` split（固定 2 刀版六顆訓練共用的同一批 held-out
樣本）當 `evaluate()` 的輸入——這批 val episode 跟本次 dyn 訓練用的 train episode
是同一個 `split_seed=42` 切分（`corpus_v1.pt` 一路沿用到底，不會有洩漏），好處是
三個 recipe（orig/aug/dyn）的 held-out teacher-forcing 數字可以直接互比，不用另外
從頭寫一份 held-out 建構邏輯（複用已驗證過的資料，降低新增程式碼的錯誤面）。
⚠️ 這個 held-out 只是訓練時的健康度描述性指標，不是本工單的判準——判準看的是
`eval_rewrite_aug.py` 跑出來的 R8/R∞ per-leg（見 §四）。

---

## 二、漂移型擾動的校準

工單指定量級對齊「成功題 p75≈0.36m」（`walk_verify/drift_analysis/results/
drift_summary.json`，`failure_precursor.succ_quantiles.75`，唯讀引用主 repo
`lacot`，一手數字 = **0.357983m**，不是四捨五入的 .36）。

**xy 主擾動**：方向 uniform[0,2π)，幅度 = 半常態分佈（`scale=xy_sigma`）。校準：
半常態分佈的 p75 分位數 = `xy_sigma × halfnorm.ppf(0.75)`（`scipy.stats.halfnorm`，
標準半常態 scale=1 時 p75 = **1.150349**，等於 `Φ⁻¹(0.875)`，因為
`P(X≤x)=2Φ(x/σ)-1`）。解出 `xy_sigma = 0.357983 / 1.150349 = 0.311195`。

**其餘 27 維**：獨立 Gaussian，std = `OTHER_DIM_COEF × 該維 train 資料 std`（std 用
train episodes 的全部 timestep raw obs 算，n=895,857 步，27 維 std 範圍
[0.0864, 3.2753]，中位數 0.7556）。`OTHER_DIM_COEF` 是本檔自訂係數（工單原文
「係數你定並記錄」），**校準過程本身值得記一筆**：第一次嘗試 `0.05`（各維看起來
很小），但 27 個獨立 Gaussian 疊成 L2 norm 後量級隨 √27≈5.2 放大，稽核算出來
p75=0.4238，反而**大於** xy 主擾動 p75（0.3659，比例 1.158）——不符合工單「量級
明顯小於 xy 主擾動」的要求。改成 `0.005`（縮小 10 倍）後 p75≈0.0424，比例 0.116，
明顯小於，見下方稽核表。**這是 `build_corpus_dyn.py` 稽核輸出當場抓到的，不是
訓練完才發現**——corpus 重建成本低（1.5 秒），抓到就直接修正重跑，附兩次嘗試的
數字在此存證。

### 驗收④：擾動幅度分佈稽核（1000 樣本，`PERTURB_SEED=20260912`）

```
                    p50      p75      p95      mean     目標(p75)   備註
xy 主擾動         0.2193   0.3659   0.5856   0.2531    0.3580     量出來的 p75 跟目標差 +0.0080（n=1000 有限樣本噪音，方向合理）
其餘27維 L2 norm  0.0362   0.0424     —        —          —       vs xy p75 比例=0.116，明顯小於（OTHER_DIM_COEF=0.005，第二次嘗試後修正）
```

一手 log：`experiments/gen_sft/logs/build_corpus_dyn.out`（本檔跟 `build_corpus_aug.py`
同慣例直接在本機跑、不經 slurm）；數字同時存在 `corpus_dyn_v1.pt`
`meta.perturb_calibration` / `meta.perturb_audit_xy`。

---

## 三、驗收標準①：動態切正確性自檢（20/20）

強制 cut 分支（`use_full_prob=0`，`perturb=None`——先驗證不擾動時的基礎構造邏輯是
對的），從 train episodes 均勻抽 episode、對每個抽到的 episode 用它自己的
`usable_cuts` 抽一個 k，逐一核對：(a) `code_seq` 是否為母 episode 完整字串的對應
後綴位元一致、(b) 條件 obs 是否跟 `raw["observations"][s0_idx+4k]` 位元一致（沿用
`build_corpus_aug.py` 驗收①的比對法，這裡額外對 obs 也做，不只做字串）。

**結果：20/20 全部 PASS**（`code_seq` 後綴 bit-exact、`s0_obs` L2=0.00000000 逐條
皆是）。抽樣涵蓋 cut_k 從 1 到 40（範圍夠廣，不是全部擠在同一個區段）、M_sub 從
1 到 3。一手 log：`experiments/gen_sft/logs/build_corpus_dyn.out`；結構化紀錄：
`experiments/gen_sft/results/dyn_selfcheck20.json`（seed=20260909777，跟訓練實際用
的 `dyn-cut-seed`/`perturb-seed` 分開，避免自檢消耗到訓練會用的那條 rng 流状態——
雖然這兩者是不同行程，即使同值也不會真的互相干擾，但用不同 seed 讓這件事不必
依賴這個推理，更乾淨）。

軌跡級語料規模（`build_corpus_dyn.py` ledger）：4952 episodes 全部收錄
（n_episodes_zero_usable_cuts=0，沒有任何 episode 完全沒有可用切點）、候選切點
總數 222,840、通過『至少留 1 個尚未到達路標』過濾後的可用切點 196,244（約
88%）。Train/val 切分沿用 `corpus_v1.pt` 原本的 episode 級切分（split_seed=42，
n_train_episodes=4457 / n_val_episodes=495，跟 orig/aug 六顆訓練完全同一批）。

---

## 四、六顆訓練 + 主表

訓練/eval 全程走 Slurm sbatch + dependency 鏈（6 顆訓練 GPU job 同批提交、
`--gres=gpu:1`，zeldajr 4 卡跑 4 顆同時+2 顆排隊；6 顆 eval CPU job 各自
`--dependency=afterok:<對應訓練job>`；最後一顆 `collect_dyn`
`--dependency=afterok:<全部6顆eval>`、`sbatch --wait` 拿到單一阻塞點），全程非裸跑、
前景無 squeue 輪詢迴圈。

**一次提交事故（誠實記錄，仿 [[NOTE-2026-09-09-midstart-aug]] 記錄 --export bug 的
慣例）**：第一次提交（job 27211-27216）六顆全部 FAILED（9 秒內），原因是提交腳本
把 `CORPUS`/`GENSFT_CKPT` 寫成相對路徑（相對於「submit script 執行時的 cwd」），但
`train_gen_sft_dyn.sbatch`/`eval_rewrite_dyn.sbatch` 內部會先 `cd "$WT"`（worktree
根目錄，不是 `experiments/gen_sft/`），相對路徑在 job 的 cwd 下解析不到檔案
（`FileNotFoundError: results/corpus_dyn_v1.pt`）。已 `scancel` 掉六顆 FAILED 訓練
連帶卡住的 6 顆 eval + collector（27217-27223，皆處於 Dependency 狀態，afterok 永遠
不會滿足），改用絕對路徑重新提交（27227 起），全部成功。舊 log 保留在
`logs/gensft_dyn_*-2721{1..6}.out` 供查，不刪除。

### 4.1 六顆訓練摘要（實際成功跑的 job：27227-27232）

```
group    seed     | total_steps best_step | smoke loss[0]→loss[500](drop)   | held_out top1_codes / top1_eos | wall(s)
--------------------------------------------------------------------------------------------------------------------
nopert 20260909   |    8500     4500      | 3.664→2.758 (0.906)             | 26.20% / 60.00%                | 328.6
nopert 20260910   |    8000     4000      | 3.671→2.780 (0.892)             | 26.12% / 65.86%                | 310.5
nopert 20260911   |    8500     4500      | 3.663→2.788 (0.875)             | 26.30% / 62.78%                | 328.5
pert   20260909   |    8000     4000      | 3.663→2.754 (0.909)             | 26.48% / 55.82%                | 325.7
pert   20260910   |    8500     4500      | 3.670→2.788 (0.882)             | 25.98% / 64.03%                | 343.5
pert   20260911   |    8500     4500      | 3.663→2.786 (0.878)             | 26.19% / 63.59%                | 344.7
```

全部 6 顆 smoke 全過（前 500 步 loss 從 ~3.66 明顯降到 ~2.76-2.79，遠超亂猜基準
ln(32)=3.4657）——**smoke test 必須 pass 才交，六顆皆過**。held-out top1_codes（~26%）
/top1_eos（56-66%）數字量級跟固定 2 刀版（aug，26-29% / 60-64%，見
[[NOTE-2026-09-09-midstart-aug]] §二）相近，訓練健康度正常，不是壞掉的訓練。
`cut_stats` 逐顆核對 n_cut/n_full 皆貼近 1:1（例：nopert s20260909 為
1,088,871:1,087,129，比例 50.04%:49.96%，符合設計的 50/50）；`pert` 三顆的
`n_perturbed` 精確等於 `n_cut`（擾動只加在切點分支，未誤加到原起點分支——設計
落實，非僅宣稱）。一手 log：`experiments/gen_sft/logs/gensft_dyn_{group}_s{seed}
-{jobid}.out`；summary json：`experiments/gen_sft/results/gensft_dyn_{group}_s{seed}
_train_summary.json`。

### 4.2 主表：{2刀(引用), 動態無擾, 動態輕擾} × 3 seed × {R8, R∞}

```
group        seed |     R8   |    R∞
-------------------------------------------
2cut(引用) 20260909 | 0.6899 | 0.5900
2cut(引用) 20260910 | 0.6651 | 0.5337
2cut(引用) 20260911 | 0.6368 | 0.5931
           mean/std | 0.6639 / 0.0266 | 0.5722 / 0.0334

dyn無擾    20260909 | 0.6344 | 0.5946
dyn無擾    20260910 | 0.6635 | 0.5417
dyn無擾    20260911 | 0.6527 | 0.5819
           mean/std | 0.6502 / 0.0147 | 0.5727 / 0.0276

dyn輕擾    20260909 | 0.6513 | 0.5823
dyn輕擾    20260910 | 0.6325 | 0.5934
dyn輕擾    20260911 | 0.6811 | 0.6099
           mean/std | 0.6549 / 0.0245 | 0.5952 / 0.0139
```

（2 刀版數字直接引用既有檔案 `experiments/gen_sft/results/rewrite_eval_aug_s{seed}
_summary.json`，非重新輸入的手抄數字——`analyze_dyn.py` 程式本身直接讀這三個檔案。
std 一律 sample std，ddof=1，除以 n-1——見下方 ⚠️ footnote，跟工單引用的『2 刀版
std≈.026』口徑對齊。200 題、貪心解碼、`build_tasks(seed=20260908)` 同一批任務，
跟 aug 版完全同分佈可比。）

⚠️ **std 算法校準記錄**：`analyze_dyn.py` 第一版用 population std（ddof=0），算出
baseline R8 std=.0217，但工單原文引用的是「≈.026」——反推發現工單那個數字是
**sample std（ddof=1）**：baseline [.6899,.6651,.6368] 的 ddof=1 std=.02657≈.026，
跟 ddof=0 的 .0217 對不上。3 點小樣本，ddof=1（無偏估計）本來就是統計上比較標準
的選擇，已改成這個口徑重跑——**兩種算法下 Q1「更穩」子判準的結論沒有變**（皆為
「未明顯小」），只是數字精度現在跟工單引用的錨點對得上，過程如實記在這裡而不是
悄悄改掉。

---

## 五、判準判讀

### Q1：動態隨機起點 vs 固定 2 刀 —— **無差**

```
mean_diff = 0.6502(dyn無擾) - 0.6639(2刀) = -0.0137        門檻 >= +.02 → 未達（且方向是負的，不是打平）
dyn std = 0.0147  vs  threshold 0.0133(=0.5×baseline std .0266)   → 未明顯小（自訂門檻，見下方）
```

兩個子判準都不成立，**判讀：無差，如實報**（不是「更好」也不是「更穩」，字面判準
不允許因為某個數字方向對就升級成 PASS）。誠實補充兩點語境，讓「無差」不是一個
空洞的結論：

1. **均值方向其實是略負**（-0.0137），不是貼著 0 的雙向噪音——3 顆 dyn 無擾（.6344/
   .6635/.6527）比 3 顆 2 刀版（.6899/.6651/.6368）逐顆看有 2 顆更低、1 顆持平略高，
   不是「打平」而是「方向上略遜、但沒有大到可以說『更差』」（-.0137 遠小於 .02
   門檻，也遠小於兩組各自的 std）。3 個點的比較，這個差距在統計上站不住「有意義
   地更差」，但也同樣站不住「持平」——最誠實的說法是「量出來的點估計偏負，但
   信心不足以下任何方向性結論」。
2. **std 有縮小但沒有『明顯』**：0.0147 對 0.0266，比例 55%——確實比一半（我方
   自訂的「明顯」門檻）大一點，如果把門檻放寬到「六成」就會變成 PASS。**這正是
   為什麼要先把門檻寫死再看數字**（工單開頭「在證明什麼、判準是什麼」的精神）：
   如果看到 55% 才回頭選一個門檻，那就是拍腦袋逆推，這裡沒有那樣做——門檻在
   `analyze_dyn.py` 寫定為「一半」在先，55% 就是「未過」，不因為差一點就破例。

「std 明顯小於」門檻＝dyn std ≤ 0.5×baseline std，本檔自訂（工單沒給精確數字），
已在 `analyze_dyn.py` 明標，非工單既有判準（同 [[NOTE-2026-09-09-midstart-aug]] §五
姿態置換探針 .10 門檻自訂的慣例）。

### Q2：漂移型擾動 vs 無擾動 —— **無效/無差**

```
mean_diff = 0.6549(輕擾) - 0.6502(無擾) = +0.0047        門檻 >= +.02 → 未達
```

方向是正的（跟「擾動應該有幫助」的物理直覺一致：教 u 見過偏移的起點，重寫時遇到
真實漂移態應該更從容），但量級只有門檻的四分之一不到，**在 3-seed 的量測精度下，
這個差距不足以宣稱『有效』**——**如實報：無效/無差**，不因為方向對就放寬。

側面觀察（非本工單判準，質性點名，不下結論）：擾動組的 **R∞**（開環、完全不重寫）
mean=0.5952 比 2 刀版（0.5722）與 dyn 無擾（0.5727）都高，std（0.0139）也是三組
裡最小的。R∞ 理論上只吃「原起點」條件（擾動只加在中途切點分支），照理不該被
擾動訓練影響——如果這個側面觀察是真訊號，可能的故事是『訓練一個要應付偏移條件
的輔助任務，讓共享的模型表徵整體更穩』，但**只有 3 個 seed，這個解讀本身很可能
只是抽樣噪音**，本檔不據此下任何結論，附數字存查，留給之後更多 seed 或專門設計
的實驗去驗證。

### 5.1 跟主人裁示原文的對照

工單原句「兩問各自『無差』也是合法答案，如實報」——**兩問這次都是無差**，不是
湊巧只有一問。這件事本身有一個誠實的合理故事：固定 2 刀版本來就已經覆蓋了『從
頭』與『兩個中途點』三種條件，增強過的教材已經教過『半路接寫』這個病灶的
主要部分（見 [[NOTE-2026-09-09-midstart-aug]] R4 代價從 -.046 反轉為 +.085 的
乾淨結果）；把切點從『固定 2 個』換成『連續覆蓋整條軌跡』，如果病灶已經被固定
2 刀治得差不多了，動態版本能再擠出的邊際收益本來就可能很小、甚至被『訓練
本身的 seed-to-seed 抖動』（std≈.02-.03 這個量級）蓋過去，量不出來是合理的，不是
「本工單失敗了」。擾動同理：條件本身的『歪』如果本來就已經被中途切點的『真實
姿態』覆蓋了相當一部分的分佈差，額外再疊加人工漂移擾動，能再擠出的邊際收益
也可能落在同一個抖動量級以下。**兩問都無差，比只有一問無差更一致地指向同一個
故事**——但這是本檔的解讀，不是量出來的事實，跟 §五的量測結果分開標記。

---

## 六、沒做到 / 不確定清單

1. 判準1的『std 明顯小於』沒有工單給定的精確數字，本檔自訂『dyn std ≤ 2 刀版 std
   的一半（=.0133，2 刀版 R8 sample std=.0266 的一半）』當『明顯』門檻，已在
   `analyze_dyn.py` 明標，非工單既有判準（同 [[NOTE-2026-09-09-midstart-aug]] §五
   姿態置換探針 .10 門檻自訂的慣例）。量出來的 dyn 無擾 std=.0147，比例 55%，
   沒有過線（見 §五）。
2. `OTHER_DIM_COEF=0.005` 是本檔自訂係數，只保證『27 維合成 L2 norm 明顯小於 xy
   主擾動』這個工單要求的定性關係，沒有進一步校準『多小才是最合適的小』——如果
   之後想細調這個係數，§二的稽核法可以直接重跑（corpus 重建成本 1.5 秒，很便宜）。
3. 動態訓練的『每步取樣』用 Python-level 逐項迴圈構造（非向量化批次構造），正確性
   優先於速度（§一設計理由），沒有做效能優化；如果之後要多 seed/多變體掃描、
   這裡的 wall time 值得回頭看一眼是否成為瓶頸。
4. 擾動只加在中途切點的條件 obs（`use_full_prob` 抽到『原起點』分支時不擾動）——
   這是本檔對工單「動態切點的條件 obs 加擾動」這句話的字面解讀（原起點對應 env
   reset 剛完、語意上不該有『漂移』），但工單沒有明講『原起點分支要不要也擾動』，
   這裡是本檔的判斷，如果跟主人原意不同、可以在下一輪快速調整重跑（只改
   `draw_dynamic_sample` 一個函式的呼叫方式）。
5. held-out 描述性指標（teacher-forcing top1/loss）沿用 `corpus_aug_v1.pt` 的 val
   split，不是本工單自建的『dyn 專屬』held-out 抽樣——理由見 §1.3，好處是可以跟
   orig/aug 直接比、壞處是它本身仍是『固定 2 刀』形狀的 held-out（不是動態的），
   如果之後想量『held-out 本身也用動態抽樣會怎樣』，這裡目前沒做。

---

## 七、一手 log / 檔案索引

```
experiments/gen_sft/dyn_corpus_common.py                動態抽樣核心（新檔）
experiments/gen_sft/build_corpus_dyn.py                  建軌跡級語料+驗收①④（新檔，CPU 直跑不經 slurm）
experiments/gen_sft/results/corpus_dyn_v1.pt             軌跡級語料
experiments/gen_sft/results/dyn_selfcheck20.json         驗收①20條逐條紀錄
experiments/gen_sft/logs/build_corpus_dyn.out            corpus 建置一手 log（含驗收①④輸出）
experiments/gen_sft/train_gen_sft_dyn.py                 動態訓練腳本（新檔）
experiments/gen_sft/train_gen_sft_dyn.sbatch             六顆訓練共用 sbatch（新檔，WT=lacot-dyn）
experiments/gen_sft/eval_rewrite_dyn.sbatch              六顆 eval（新檔，直接呼叫既有 eval_rewrite_aug.py 不改一行）
experiments/gen_sft/analyze_dyn.py + collect_dyn.sbatch  收表算判準（新檔）
experiments/gen_sft/results/dyn_analysis.json            判準計算結果（機器可讀，ddof=1 校正後的最終版本，
                                                           見§四footnote；由直接重跑 analyze_dyn.py 產生，
                                                           非 collect_dyn-27239.out 那個 sbatch job 的版本——
                                                           該 log 是校正前的舊版本，保留供查不刪除)

訓練 job id：nopert s20260909=27227 s20260910=27228 s20260911=27229
             pert   s20260909=27230 s20260910=27231 s20260911=27232
             （第一次提交 27211-27216 因相對路徑 bug 全 FAILED，已 scancel，見 §四）
eval   job id：nopert s20260909=27233 s20260910=27234 s20260911=27235
             pert   s20260909=27236 s20260910=27237 s20260911=27238
collector job id：27239（sbatch --wait 阻塞點；log 是 ddof=0 舊版本，dyn_analysis.json
             已用直接重跑的 ddof=1 版本覆寫，見上）

checkpoint：experiments/gen_sft/results/gensft_dyn_{nopert,pert}_s{20260909,20260910,20260911}_best.pt
eval summary：experiments/gen_sft/results/rewrite_eval_dyn_{nopert,pert}_s{seed}_summary.json
```

隨機性固定：切點抽樣（`--dyn-cut-seed`）=20260909；擾動（`--perturb-seed`）=20260912；
擾動幅度稽核 seed=20260912（跟訓練實際用的流相同，直接驗證『真的會用到的那條流』，
不是替代 seed）；自檢 20 樣本 seed=20260909777；6 顆訓練 seed=
{20260909,20260910,20260911}（只管權重初始化+batch-order）；held-out/eval 200 題
沿用既有 `build_tasks(seed=20260908)`。生成一律 `temperature=0` 貪心、決定性。
「我不知道」與「答案是否」分開表達，見 §五。
