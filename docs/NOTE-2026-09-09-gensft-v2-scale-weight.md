# NOTE — gen-sft-v2：容量掃 + 四關離線打分挑教材（2026-09-09）

_對應 [[NOTE-2026-09-09-gensft-v1]]。工單兩問：①寫譜模型 u 現在的成績是不是被容量卡住
——掃容量看 held-out 準確率與 per-leg 是否跟著漲；②「四關離線打分挑教材」（好教材
加權）能不能拉高譜的品質。_

_在證明什麼、判準是什麼：同 200 題上，貪心單條 per-leg ≥.66（v1 錨 .588＋.07 抖動尺
之外）或 BoN(16) per-leg ≥.90（v1 錨 .828＋.07 之外）任一格達標算清楚贏；帶內（在錨
±.07 內）＝無效；「held-out top-1 大漲但 per-leg 不動」是重要訊號，不論哪一格達標都要
點名。_

_全部程式：`experiments/gen_sft/`（既有 v1 檔案一律不改，新檔一律 `*_v2*` 或
`score_corpus*` 命名）。一手 log 全部在 `experiments/gen_sft/logs/`。_

---

## 〇、一句話

**兩個都沒清楚贏，而且理由不同、都查得到。** 容量掃 3 格（2x~7.7x 參數）held-out
top1（28.0%~28.8%）與 200 題 per-leg（貪心 .588~.593、BoN16 .828~.868）全部
落在 v1 錨 ±.07 抖動尺以內——不是容量不夠，是這個任務在現有配方下已經吃滿
（容量越大過擬合越早越重，widedeep train/held-out loss 落差最大）；四關加權
（軟／硬各一顆，都在 C0=C* 架構上）per-leg 也一樣落在帶內，**但方向一致地
偏低**（held-out top1 明顯降到 26.1%~26.3%、greedy/BoN16 都比錨低），不是雜訊
各打各的，是同一個方向——最可能的原因是這份語料裡「磨蹭」跟「壞教材」不是
同一件事（reach_frac 全 4457 條 train 樣本都是 1.0，代表全部教材本來就都
「到得了」，磨蹭只是走比較久不是走錯），砍掉磨蹭那一半只是讓模型看到的
（起點+路標）→code 配對變少了，沒有真的濾掉會教壞的東西。打分器本身驗收
①乾淨過（adversarial 兩種構造都落在真實分佈第 0 百分位，遠低於 p10），問題
不在「打分器不準」，在「這個 axis 跟這個任務的相關性」。

---

## 一、配方

### 1.1 v1 錨（不重跑，直接引用既有結果）

C0＝v1 現版：d_model=128, n_layers=3, n_heads=4, dropout=0.1, 620,577 參數。
held-out top1_codes_only=28.71%（best_step=2500，提早停在 step=6500）。200 題（同
build_tasks seed=20260908）：貪心 per-leg=.5875、BoN(16) per-leg=.8282。
來源：`experiments/gen_sft/results/gensft_v1_train_summary.json`、
`gensft_eval_N16_summary.json`（job 26889）。

### 1.2 容量掃三格（design 第一步，教材維持均勻抽樣，不加權）

沿用既有、完全未修改的 `model.py`（`GenSFT(d_model,n_layers,n_heads,dropout,max_len)`
本來就參數化）與 `eval_gen_sft.py`（讀 ckpt 內存的 config 字典建模型，架構無關）。
只新增 `train_gen_sft_v2.py`（import `train_gen_sft.py` 的 `collate_batch()`/
`evaluate()` 重用，訓練迴圈重寫以支援 `--weight-mode`）＋ `train_gen_sft_v2.sbatch`
（`$EXTRA` passthrough，比照 `eval_gen_sft.sbatch` 的既有慣例）。

| tag | d_model | n_layers | n_heads | 參數量 | 對 C0 倍數 |
|---|---|---|---|---|---|
| gensft_v2_wide | 256 (×2) | 3 | 8（跟著 d_model 等比例，維持 head_dim=32 跟 C0 一致）| 3,282,977 | 5.3x |
| gensft_v2_deep | 128 | 6 (×2) | 4 | 1,236,257 | 2.0x |
| gensft_v2_widedeep | 256 (×2) | 6 (×2) | 8 | 4,790,049 | 7.7x |

其他超參（lr=3e-4 cosine、batch=256、warmup=500、weight-decay=0.01、
patience-evals=8、eval-every=500、min-delta=1e-3、seed=1234）維持 C0 原樣不動
（design 指示：lr 等超參本次不掃）。`--steps` 上限設 30000（design 指示的
patience 早停上限；v1 本身在 6500 步就已經 plateau，30000 只是安全上限）——
三格全部提早停在 5000~5500 步，遠不到上限。

Job：27020(wide)／27021(deep)／27022(widedeep)，`sbatch --export=...
train_gen_sft_v2.sbatch`，`-p admin -A it -q great-mage --nodelist=zeldajr
--gres=gpu:1`，三顆同批獨立提交、平行跑，wall time 391.8s／389.3s／759.8s。

**訓練結果（held-out，來自 `results/gensft_v2_{wide,deep,widedeep}_train_summary.json`）**：

```
config       best_step  stopped_at  held-out top1_codes_only  held_out_loss@best  過擬型態
C0 (v1 錨)   2500       6500        28.71%                    2.2853              輕微：見底後小幅回升
wide         1000       5000        28.09%                    2.3099              明顯：train ema1.10 vs held_out3.58（step5000）
deep         1500       5500        28.82%                    2.2866              明顯：train ema1.66 vs held_out2.80（step5500）
widedeep     1000       5000        28.02%                    2.3117              嚴重：train ema0.38 vs held_out5.63（step5000）
```

一句話判讀（held-out 這關）：**三格容量掃全部落在 28.0%~28.8% 這個窄帶內，
跟 C0 的 28.71% 統計上難以分辨——held-out top-1 完全沒有跟著容量漲**，而且
容量越大、過擬合越早出現越嚴重（widedeep 的 train/held-out loss 落差三格中
最大）。這是「held-out 準確率沒有大漲」的直接證據，比工單提醒的「大漲但
per-leg 不動」更乾脆一步——這裡連 held-out 這關本身都沒有漲，如實點名；
per-leg 數字見 §三主表。

### 1.3 加權教材兩檔（design 第二步，在容量掃勝出的 C* 架構上跑）

**C* 的選擇**：容量掃三格（§1.2）沒有一格在 held-out 或 per-leg 上清楚贏過
C0，甚至 greedy 最佳（deep .5931）跟 BoN16 最佳（widedeep .8678）指向不同的
config——兩個指標挑不出同一個贏家，本身就是「差異是雜訊」的證據。在沒有證據
支持任何一個變體優於 C0 的情況下，選更大的架構進第二步只會讓加權實驗背上
不必要的過擬合風險（且已經實測到容量越大過擬合越重），沒有帶來對應的好處。
**C\* = C0**（d_model=128, n_layers=3, n_heads=4, dropout=0.1，跟 v1 完全同
架構，只是這次改吃加權/過濾過的語料），這是一個實測支持的判斷，不是省事。

- 軟加權：`--weight-mode soft --scores-path results/corpus_scores_v2.pt`，
  `weights=exp(beta*score_quality)`，beta=3.5156（調到 ESS≈N/2，實測
  ESS_frac=0.501）。Job 27061（訓練，wall=209.0s，提早停在 step 5500，
  best_step=1500）→ job 27063（eval，wall≈103s）。tag=`gensft_v2_softw`。
- 硬過濾：`--weight-mode hard --scores-path results/corpus_scores_v2.pt`，
  只留 `score_quality>=median`（median=0.2692，前 50%＝2229/4457）。Job 27062
  （訓練，wall=209.4s，提早停在 step 5500，best_step=1500）→ job 27064
  （eval，wall≈103s）。tag=`gensft_v2_hardf`。

---

## 二、打分器設計與會亮證據（驗收①）

### 2.1 四關的離線算法（`score_corpus_common.py`，純陣列運算，不碰模擬器）

- **到達**：hindsight 恆真可略——但沒有整個丟掉這個訊號：`leg_reach_steps()` 對每
  條腿重新（不查表既有 metadata）從給定的 xy 陣列偵測『第幾步進到 wp_xy 的 rho
  容忍圈內』，某腿完全找不到就把它、以及之後所有腿，一起判定「用完整個 episode
  的步數預算」（固定罰 `horizon`）——把『到達失敗』折進磨蹭分，不必另開一個獨立
  門檻。**實測**：4457 條 train 樣本 `reach_frac` 全部是 1.0（`frac_all_legs_
  reached=1.0000`）——在真實資料上這關對分數貢獻剛好是 0，是「hindsight 恆真」
  這個假設本身在這份資料上被實測證實，不是假設沒被檢查；它的存在意義只在
  adversarial／未來可能出現的退化教材上才會被觸發。
- **不磨蹭**（`dawdle_steps`）：上面重新偵測到的逐腿實際步數，episode 內取平均。
  ⚠️ **踩過一次雷**：原始設計是『實際步數 / ruler 查表典型步數』的比值，第一次
  跑驗收①直接 FAIL——診斷發現 ~21% 的腿，其『起點→路標』或『路標→路標』的
  **直線距離**本身就小於 rho（antmaze-stitch 資料弧長 DELTA_SUB=7.5 不保證換到
  >1.875 的直線淨位移，路徑會晃），這時 `ruler_steps_for_distance` 的
  below-grid 線性外插把分母推到接近 0，比值炸成幾十萬～百萬量級離群值，把整個
  分位排名淹沒。改法：不做除法——每條腿弧長本身幾乎是常數（≈DELTA_SUB），『同
  距離分佈』這件事在對全 corpus 做分位排名時已經自動成立，直接把逐腿實際步數
  本身當原始量即可。ruler 查表版函式（`leg_typical_steps`/`leg_distances`）留著
  沒刪，只是不再進最終分數。
- **不亂動**（`jerk_mean`）：該集動作序列相鄰差 L2 範數的均值（跟
  `wv_common.build_ruler` 算 `action_smoothness` 同一條公式，這裡是單集自己的
  均值，不是全資料集攤平的池）。
- **像樣**（`fall_frac`）：z<fall_line（ruler p1=0.2573）的步數比例。

### 2.2 合成 scalar（`build_score_table`）

三個原始量各自對訓練集全體算經驗分位（0~1，越大越差），取三者**最大值**當
badness，`score_quality=1-badness`。
⚠️ **也踩過一次雷**：第一版取三者**平均**，adversarial 構造（見下）只打壞『一關』
打到 100 百分位，另外兩關正常甚至更好，平均下來被稀釋回中段（badness≈0.3~0.4），
沒有靠近真實 p10。改成取 **max**（哪一關最差就以它為準）之後兩種構造應聲落底。
這個改動同時也更貼近『好教材』這個目的本身：一集動作平滑步速正常但常翻倒，不該
被前兩關拉回「普通」，翻倒本身就該否決它。

### 2.3 opus 抽驗：發現並修好三個真的會影響數字的問題

三份新程式寫完、第一次跑過驗收①之後，派 opus 做一次獨立、只給產出物不給思路的
對抗性審查（fleet-command §4 慣例）。抓到 4 個非風格類發現，其中 3 個會實質改變
數字（已全部修好、重驗過，下面是修完之後的版本），1 個是已知限制（點名不修）：

1. **內部（train 自評）分位 vs 外部（val／adversarial）分位，原本是兩套不同公式**
   （內部用「中點分位」`(嚴格小於+0.5*相等)/n`，外部用「嚴格小於/n`）。對
   `dawdle_steps`／`jerk_mean` 這種連續量差 O(1/n) 可忽略，但 `fall_frac` 有
   ~96% 樣本恰好是 0（零膨脹）——中點分位把這 96% 全部打到 pct_fall≈0.48，
   結果 **19.4%（864/4457）的 train 樣本被人工天花板卡在同一個
   score_quality=0.519、彼此分不出高下**（會實質傷害加權方案的分辨力），且
   外部樣本算出來的分數跟內部樣本不在同一把尺上（同一個真實 train episode，
   當「外部」樣本算得 0.577，比任何內部 train 樣本能拿到的上限 0.519 還高，
   自相矛盾）。**修法**：兩邊統一只用「嚴格小於/n」（這個計數本身跟 self
   在不在母體裡無關，因為 self 不會嚴格小於自己，只差分母 n 對 n-1，
   O(1/n) 可忽略）——重驗：最大並列組從 864 條（19.4%）降到 76 條（1.7%），
   score_quality 從 2363 個不同值（原本更少，未精確記錄）內部分辨力明顯提升。
2. **「顯著低於 p10」的判斷式標錯了名字**：原本印的 `pct_rank_in_real` 其實
   就是 `score_quality` 本身換個名字（`1-badness`），不是「這個分數在真實
   score_quality 分佈裡排第幾百分位」——兩者只有在 badness 剛好卡在 0 或 1
   邊界時數字才會重合（這次兩個 adversarial 案例剛好都卡在 badness=1 邊界，
   所以 PASS 的結論沒有錯，但判斷式本身邏輯是錯的，只是巧合沒讓結論翻盤）。
   **修法**：真的用 `percentile_of()` 去查 `score_quality` 在 `sorted(真實
   train score_quality)` 裡的排名，不是重算 `1-badness`。修完之後重印：底本
   （未打壞的真實樣本）的真實百分位是 84.1%（原本誤標成「57.7%」），兩個
   adversarial 案例的真實百分位都是 0.00%（跟原本顯示的數字剛好一樣，因為
   它們踩在邊界上）。
3. **`leg_reach_steps()` 逐腿分開搜尋，語意其實跟 `run_teacher_relay.run_one()`
   的 `while` 迴圈不完全一樣**：`run_one()` 允許同一步同時吃下好幾條腿（如果
   那一步剛好同時在兩個相鄰路標的 rho 圈內），原本的逐腿版本把下一腿的搜尋
   起點強制設成「上一腿命中步+1」，永遠不會讓同一步吃兩腿。診斷過 wp_xy 兩點
   間直線距離的分佈（mean=4.4，但 ~21% 小於 rho=1.875），有一部分腿的間距
   確實小到 <2·rho=3.75，同一步吃兩腿在這份資料上不是空論。**修法**：改寫成
   逐『步』掃、內層用 `while`（不是 `if`）一次判完當步所有已啟用的腿，逐行
   對齊 `run_one()` 的邏輯，不是「同構」用嘴巴講講，是真的搬過去。重驗：
   3 個人眼核對樣本的 dawdle_steps 數字幾乎沒變（這 3 條的路標間距夠開，沒
   踩到這個情形），全體 train 的 dawdle_steps 統計量也幾乎沒變（mean 32.79→
   32.73），確認這個修正主要影響邊界情形（含 adversarial 構造本身），沒有
   把正常資料的分數弄壞。
4. **已知限制、不修**：`train_gen_sft.py`（v1、不可改）的 `iterate_batches`
   在過濾後樣本數 < batch_size 時會卡住不報錯，不會 hang 死當前的硬過濾設定
   （2229 遠大於 256），只是留一個未來若把過濾門檻設更嚴會踩到的地雷，
   點名記錄。

修完之後重跑 `score_corpus.py`（§2.4 是修完之後的最終版本），也順手加了兩個
低成本防呆（`train_gen_sft_v2.py::load_weighting`）：分數檔內部陣列長度一致性
assert、分數檔記錄的 corpus 路徑跟這次訓練實際讀的 corpus 路徑不一致時印警告
（opus 也點出原本的 assert 沒檢查這兩件事）。

### 2.4 驗收①證據：adversarial 構造會亮（修完後的最終版本）

底本：`train[1378]` episode=3247（M=3），真實分數 dawdle_steps=31.0
jerk_mean=1.832 fall_frac=0.0 reach_frac=1.0 → score_quality=0.573（在真實
train score_quality 分佈裡排第 **84.1 百分位**，中上等）。

```
mode      dawdle_steps(pct)   jerk_mean(pct)     fall_frac(pct)   reach_frac  score_quality  真實百分位  below_p10(.035)
reverse   180.0  (100.0%)     1.832  (26.7%)      0.0000 (0.0%)   0.333       0.0000         0.00%       True
shuffle     5.0  (  0.0%)     2.577 (100.0%)      0.0000 (0.0%)   1.000       0.0000         0.00%       True
```

兩種構造的 score_quality 都是 **0.0000**，在真實訓練集 score_quality 分佈裡排
**第 0 百分位**（4457 條真實樣本裡沒有一條比它們更差），遠低於真實 p10=0.035
門檻，滿足驗收①「顯著低於真實分佈 p10」（這裡用「真實百分位 ≤5%」當「顯著」
的可判定義，兩者都是 0%）。

**推導自我核對**（不只是事後看數字，是動手前就能預測、動手後對上的）：reverse
對 jerk_mean 的變化 = **+0.0000**（理論：時間反轉下相鄰差的 L2 範數是不變量，
反轉後相鄰對只是原本相鄰對掉個方向，範數對稱不變——量出來的數字跟推導完全
吻合，不是巧合）；shuffle 對 jerk_mean 的變化 = **+0.7455**（理論：打散相鄰
配對會破壞原本的時間平滑相關性，範數應該上升——量到的方向與量級都合理）。這也
解釋了為什麼兩種構造分別由不同關頂上去（reverse 靠磨蹭關 100%，shuffle 靠亂動
關 100%），是刻意設計成兩種不同性質的反例，不是隨手挑的擾動。

⚠️ **誠實的限制**（opus 審查點出、值得記錄）：shuffle 這個構造反而讓
`dawdle_steps` **變好**（31.0→5.0，pct 從中段掉到 0%）——隨機打亂位置時，
偶然「傳送」到路標附近會被判定「秒到」，磨蹭這一關不是每種破壞都會偵測到，
這次能抓到 shuffle 完全是靠亂動這一關頂上去（max-aggregation 的價值正在這裡：
只要有一關頂住就夠）。如果有一種破壞方式是「只打亂 xy／z 但動作序列本身維持
連續平滑」，本打分器目前的三關可能都抓不到——但這種輸入本身在真實資料上不會
出現（動作跟位置是同一次物理模擬因果配對出來的，硬把它們拆開已經不是「一條
可能的真實軌跡被弄壞」，而是憑空拼接），不在這次驗收要防守的範圍內，這裡點名
是誠實記錄打分器的已知盲區，不是隱藏它。

一手 log／json：`experiments/gen_sft/logs/score_corpus.log`、
`experiments/gen_sft/results/corpus_scores_v2_report.json`（人類可讀摘要）、
`experiments/gen_sft/results/corpus_scores_v2.pt`（含全部原始特徵/分位/權重的
完整陣列，訓練腳本讀這份）。

### 2.5 人眼合理性核對：3 條真實 train episode（seed=4242）

```
episode  M  arclen  dawdle_steps  jerk_mean  fall_frac  reach_frac  score_quality  說明
4775     3  27.32   37.3          1.847      0.0000     1.000       0.168          到齊全部腿；比中位數磨蹭；動作平滑；不翻倒
1903     3  27.59   32.0          1.920      0.0000     1.000       0.077          到齊全部腿；比中位數磨蹭；動作較跳動；不翻倒
321      3  27.28   28.0          1.859      0.0000     1.000       0.503          到齊全部腿；比中位數快(不磨蹭)；動作平滑；不翻倒
```

三條的『數字特徵』跟『說明』欄位互相一致（episode 321 兩項都偏好，分數也明顯
最高；1903 兩項都偏差，分數最低）——這是不畫 gif、不跑模擬器版本的『分數與軌跡
特徵一致』核對，用離線量到的磨蹭/亂動/翻倒三個數字本身當軌跡特徵。⚠️ 沒有做
gif 版的人眼核對（design 允許用「軌跡特徵」替代，見驗收標準原文）。

### 2.6 加權方案（修完後的最終版本）

軟加權：beta=3.5156，effective sample size=2232.0/4457（frac=0.501，目標
0.5），weights_soft 範圍 0.237~6.953（正規化到平均 1）。
硬過濾：median_score=0.2692，保留 2229/4457（50.0%）。

---

## 三、主表

全部 6 格同一套協定：200 題（同 build_tasks seed=20260908）、貪心 1 條、BoN
temperature=1.0 N=16（固定 seed=20260909）、隨機字串對照組先過 <0.30 閘門才
繼續（全部 6 次都 PASS，control per-leg=0.2000 全部一致）。C0 直接引用 v1
既有結果，不重跑。

```
config              d/L/H     held-out top1   貪心 per-leg  (Δ vs 錨)   BoN16 per-leg (Δ vs 錨)   判讀
C0 (v1 錨)          128/3/4    28.71%          .5875         —           .8282          —           錨
C_wide              256/3/8    28.09%          .5911        +.0036       .8543         +.0261       帶內／無效
C_deep              128/6/4    28.82%          .5931        +.0056       .8600         +.0318       帶內／無效
C_widedeep          256/6/8    28.02%          .5877        +.0002       .8678         +.0396       帶內／無效
C*_soft (=C0，軟加權) 128/3/4    26.32%          .5282        −.0593       .8089         −.0193       帶內／無效（方向偏低）
C*_hard (=C0，硬過濾) 128/3/4    26.08%          .5663        −.0212       .8138         −.0144       帶內／無效（方向偏低）
```

判準：貪心 ≥.66 或 BoN16 ≥.90 任一達標＝贏；|Δ|<.07＝帶內／無效。**六格全部
落在帶內，沒有一格達標——沒有清楚贏家。**

---

## 四、判讀

**①容量卡不卡：不卡。** 三個判準一致指向同一個結論：(a) held-out top1 三格
全部落在 28.0%~28.8%，跟 C0 的 28.71% 幾乎無法分辨；(b) per-leg 兩個指標
（貪心、BoN16）六個數字全部落在 ±.07 帶內，沒有一格達標；(c) 貪心最佳
（deep +.0056）跟 BoN16 最佳（widedeep +.0396）指向不同架構——如果容量真的
是瓶頸，應該看到兩個指標一致地往同一個方向、同一個 config 集中，而不是
「兩把尺各挑各的贏家」，這是雜訊而非訊號的直接證據。且容量越大過擬合越早
越重（widedeep 的 train/held-out loss 落差三格最大），代表現有配方對這個
任務的複雜度而言容量已經**過剩**，不是不足——加更多容量只會加速過擬合，
不會多學到東西。本次沒有出現工單提醒的「held-out top-1 大漲但 per-leg
不動」——這裡連 held-out top-1 都沒有漲，是更乾脆的「兩者都沒漲」。

**②加權有沒有用：沒有，而且方向一致偏低（沒到「反效」但也不是雜訊各打各
的）。** 軟加權與硬過濾兩顆在 C0（=C*）架構上訓練，per-leg 六個數字裡的四個
（softw 的貪心/BoN16、hardf 的貪心/BoN16）全部**低於**錨且全部落在帶內
（|Δ| 最大 .0593，未達 .07 的反效門檻）；held-out top1 明顯降（28.71%→
26.1%~26.3%，工單沒有替這個指標定門檻，這裡不宣稱「顯著」，只如實描述方向
與量級）。三個指標（held-out top1、貪心、BoN16）對兩顆加權版本**全部同方向
偏低**，不是三個指標各自隨機分散——這個一致性本身是值得留意的訊號，即使
單獨每一格都沒有跨過「顯著」門檻。

**可能原因（假說，非定論）**：打分器驗收①乾淨過關（§2.4），問題不在「打分器
不準」。真正的落差可能在於**這個 axis 跟這個任務的相關性**：`reach_frac` 全
4457 條 train 樣本都是 1.0——這份語料裡沒有「教材本身走錯路」這種東西，
「磨蹭」量的只是走比較久、比較繞，不是走去了不該去的地方。而 u 學的是
（起點, 路標序列）→**離散 code 序列**，不是連續軌跡本身；K=32 的量化字典
在把連續動作段壓進 32 個字的過程中，可能已經吸收掉了一部分「同一個大致
動作、走快走慢」的差異——磨蹭與否對量化後的 code 選擇的區分度，不見得跟
它對連續軌跡的區分度一樣大。加權/過濾實際做的事情主要是**減少訓練時看到
的（起點,路標）→code 配對多樣性**（硬過濾直接砍半、軟加權把 ESS 壓到
N/2），如果「磨蹭」這個 axis 跟「這個 (起點,路標) 該配哪個 code」的相關性
本來就弱，那砍掉/降權磨蹭教材主要效果就是減少多樣性，沒有對應的品質提升
補償回來——這與實測到的「held-out/per-leg 全部同向下降」吻合。這是本檔
能給的最佳解釋，沒有進一步做消融去證明（例如只加權磨蹭、固定另外兩關）。

**回答工單兩問**：①寫譜模型現在的成績**不是**被容量卡住（held-out 與
per-leg 一致無感，且過擬合隨容量惡化）；②四關離線打分挑教材**沒有**拉高
譜的品質（六格全部帶內，且加權兩顆方向一致偏低）。打分器本身的正確性
（驗收①）跟這兩個負面結果是分開的兩件事——打分器會亮，只是它量到的
「磨蹭」跟這個任務的學習訊號關聯不強，這是本次最值得留給下一步的判讀，
不是「打分器沒做好」。

---

## 五、沒做到／不確定清單

1. 打分器的『到達』軸在真實資料上是恆真、貢獻 0 訊號（reach_frac 全 1.0）——
   design 允許略過，這裡是實測確認，不是假設。
2. `dawdle_steps` 這關對「只打亂位置、保留動作平滑」這種假想破壞方式沒有防禦
   （見 §2.4 誠實限制段），但這種輸入本身在真實資料上不可能出現（動作跟位置
   因果配對），沒有進一步構造這種反例去驗證。
3. `iterate_batches`（v1、不可改）在過濾後樣本數 < batch_size 時會 hang，本次
   硬過濾配置遠離這個邊界（2229 vs 256），沒有實測踩線行為，只是點名記錄。
4. **只有 1 個訓練 seed、1 個 BoN 採樣 seed**（延續 v1 §五#2 同樣的已知限制）
   ——六格之間的小差異（+.0056~+.0396、−.0144~−.0593）有多少是 run-to-run
   variance、多少是真實效應，本檔沒有量（design 給的抖動尺 ±.07 是 v1 NOTE
   對單次 eval 的經驗估計，不是本次針對這六個新 config 各自重新量的）。
5. **加權/過濾方向偏低的假說（§四「可能原因」）沒有做消融驗證**——沒有跑
   「只用 fall_frac 或只用 jerk_mean 單一 axis 加權」之類的對照去確認是不是
   真的是「磨蹭跟 code 選擇相關性弱」，這是留給下一步的方向，本檔只提出
   假說、沒有證明。
6. **只掃了容量的『寬/深/寬深』三格**（design 指定的範圍），沒有掃 lr／
   batch／dropout 等其他超參，也沒有掃介於 C0 與這三格之間的中間容量點
   （例如只加深 2 層、只加寬 1.5x）——如果真實的容量-表現關係是非單調的，
   本次三個離散點可能沒有覆蓋到。
7. **加權方案只試了 β 對到 ESS≈N/2 這一個操作點、硬過濾只試了前 50% 這一個
   切點**——design 兩種各一顆，沒有掃 β 或切點的敏感度，不知道更溫和（例如
   ESS≈N/0.8）或更激進（前 25%）的設定會不會有不同結果。

---

## 六、一手 log／artifact 路徑索引

- 打分器：`experiments/gen_sft/score_corpus_common.py`、`score_corpus.py`，
  log `logs/score_corpus.log`，輸出 `results/corpus_scores_v2.pt` /
  `results/corpus_scores_v2_report.json`。
- 訓練：`experiments/gen_sft/train_gen_sft_v2.py`、`train_gen_sft_v2.sbatch`，
  log `logs/train_gen_sft_v2.sbatch-{27020,27021,27022(容量掃),27061,27062
  (加權)}.out`；checkpoint/summary/loss 曲線 `results/gensft_v2_{wide,deep,
  widedeep,softw,hardf}_{best.pt,train_summary.json,loss_curve.png}`。
- CPU 相容性 smoke（train_gen_sft_v2.py 三種 weight-mode + eval_gen_sft.py 讀
  v2 checkpoint schema）：`results/SMOKE_v2_*`。
- eval：沿用既有 `eval_gen_sft.py`/`eval_gen_sft.sbatch`，不修改，只換
  `--gensft-ckpt`；log `logs/eval_gen_sft.sbatch-{27058,27059,27060(容量掃),
  27063,27064(加權)}.out`；summary json/overlay png/gif
  `results/gensft_v2_{wide,deep,widedeep,softw,hardf}_eval_*`。
- 獨立 opus 抽驗：對抗性審查三份新檔，抓到 §2.3 列的 4 個發現（3 修 1 記錄），
  過程與完整發現清單在本 session 記錄，未另存檔案。
