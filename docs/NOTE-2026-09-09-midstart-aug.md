# NOTE — rewrite-v1 步2：鐘形定讞 + 半路起點增強教材（2026-09-09）

_對應 [[NOTE-2026-09-09-rewrite-receding]] 留下的兩個未定案：(1) R=8 的 +.0506 增益是
真效應還是訓練抖動——沒有跨 seed 量過；(2) 假說 H（u 訓練教材全部「從乾淨開頭寫到
底」，沒學過「從半路接寫」）只有間接證據支持，沒有直接測試。本檔跑兩件事：3 個訓練
seed 的原配方模型定讞鐘形是否為真，以及建「中途起點增強」教材重訓後直接測 H。_

**在證明什麼、判準是什麼**（工單原文）：
1. 鐘形定讞——3 個原配方訓練 seed，各量 (R=8 − R=∞) 差值：全部 >0 且平均 ≥.03 算
   「重寫增益為真」。
2. 增強判讀——增強版 R=8 的 per-leg 平均 ≥.66（錨 .5875+.07 之外）算清楚贏；並檢查
   增強版 R=4 是否不再明顯低於 R=∞（病灶治對的直接證據）。
3.（主人 2026-09-09 追加）姿態置換探針——u 有沒有真的在用姿態 token；掉幅大＝有用，
   幾乎不掉＝忽略，重寫收益要重新解讀。

全部程式（新檔，未修改任何既有檔案）：`aug_corpus_common.py`（增強語料構造邏輯）、
`build_corpus_aug.py`（建語料+自檢，CPU、不經 slurm，同 build_corpus.py 慣例）、
`train_gen_sft_aug.sbatch`（GPU 訓練，複用 train_gen_sft.py 不變）、
`eval_rewrite_aug.py`+`eval_rewrite_aug.sbatch`（CPU eval，R grid 可配置）、
`pose_permute_probe.py`+`.sbatch`（姿態置換探針）、`analyze_bell_aug.py`（收表算判準）。

---

## 一、增強語料構造

### 1.1 設計（design 第1點逐字照做）

每條 episode（build_corpus 收錄的全部可用集，4952 集）除原樣本（從頭起筆）外，從
中途 chunk 邊界（k∈{1,...,n_chunks-5}，避開最後 4 個 chunk）均勻隨機不重複抽 2 個
切點（cut_seed=20260909，按 episode id 排序後固定走訪順序抽），每個切點造一個子
樣本：條件＝該中途點的真實 obs（`raw["observations"][s0+4k]`，資料裡的，非模擬）＋
其後尚未經過的路標序列（用路標第一次被真實軌跡到達的 local timestep
`wp_local_idx`，`searchsorted(wp_local_idx, 4k, side='right')` 之後的路標）、目標＝
其後字串（`code_seq[k:]`）。

實作關鍵：`gen_sft_common.split_corpus()` 假設「每個 episode 只有一個樣本」（用
樣本數當 episode 數算 9:1），增強後同一 episode 有 3 個樣本會讓它的 rng 消耗方式
算錯、甚至 `assert train_eps.isdisjoint(val_eps)` 會炸——`aug_corpus_common.py` 另
寫 `split_corpus_by_episode()`，對「唯一 episode id」集合做切分，同一個 episode
的原樣本＋全部子樣本永遠同側（防洩漏），且在無重複時跟原函式逐位元同構（見自檢②）。

### 1.2 樣本數帳

```
原始樣本（build_corpus 收錄）      4952
切點抽樣候選                        4952 集 × 2 切點 = 9904 次抽樣
子樣本成功造出                      8706
子樣本跳過（切點之後已無剩餘路標）   1198   （M_sub<1，主要發生在 M 本身很小
                                            且路標很早就被真實軌跡到達的集，
                                            晚切點後沒有路標可留）
增強後總樣本                        13658  (4952 原 + 8706 子)
```

Split（episode 級 9:1，split_seed=42）：train_episodes=4457 / val_episodes=495（跟
`corpus_v1.pt` 完全相同——自檢②逐集比對 PASS），對應樣本數 train=12293 / val=1365。

### 1.3 自檢（驗收標準第1條，逐條附證據）

**自檢①**（wp_xy 逐位元比對）：重新呼叫一次 `rt.build_tasks()`（同 seed=0、同
raw/ruler/rho）只為了拿 `wp_local_idx`（`corpus_v1.pt` 既有樣本格式沒存這個欄
位），跟 `corpus_v1.pt` 全部 4952 集的 `wp_xy` 逐位元比對：**0 mismatches，PASS**
——證明重建的 task 真的是同一組，不是「seed 一樣就假設一樣」。

**自檢②**（split 一致性）：增強後的 episode 級切分（`train_eps`/`val_eps`）跟
`corpus_v1.pt` 原本的切分逐集比對：**完全相同，PASS**。

**驗收①（工單指定）**：抽 5 個子樣本，逐一核對「字串＝母 episode 完整字串的對應
後綴」與「條件 obs＝資料原值位元一致」：

```
episode=  530 cut_chunk=19/50  code_seq len=31 (parent len=50): PASS bit-exact suffix   s0_obs: PASS bit-exact (L2=0.00000000)   M=2 (parent M=3)
episode=  360 cut_chunk=23/50  code_seq len=27 (parent len=50): PASS bit-exact suffix   s0_obs: PASS bit-exact (L2=0.00000000)   M=2 (parent M=3)
episode= 1960 cut_chunk=36/50  code_seq len=14 (parent len=50): PASS bit-exact suffix   s0_obs: PASS bit-exact (L2=0.00000000)   M=1 (parent M=3)
episode= 3383 cut_chunk= 6/50  code_seq len=44 (parent len=50): PASS bit-exact suffix   s0_obs: PASS bit-exact (L2=0.00000000)   M=3 (parent M=3)
episode= 2128 cut_chunk=18/50  code_seq len=32 (parent len=50): PASS bit-exact suffix   s0_obs: PASS bit-exact (L2=0.00000000)   M=2 (parent M=3)
```

**5/5 PASS，構造正確性確認。** 語料檔：`experiments/gen_sft/results/corpus_aug_v1.pt`
（10.53 MB）。一手 log：`experiments/gen_sft/logs/build_corpus_aug.out`（本檔跟
`build_corpus.py` 同慣例直接在本機跑、不經 slurm，執行輸出另存進 repo 的 logs/
目錄以求跟其餘步驟一致）。

副作用觀察（訓練 log 可見、非本節驗收項目但值得記）：增強語料的 `top1_eos` 訓練
中段只有 57~75%（原語料恆為 100%，因為原語料 EOS 永遠在第 50 格，是平凡解）——
增強後模型必須學「不同起點該在哪裡喊停」，這是任務本身變難、不是訓練壞掉。

---

## 二、六顆訓練

3 訓練 seed（20260909/20260910/20260911）× {原語料 corpus_v1.pt, 增強語料
corpus_aug_v1.pt}，d128/L3/H4，held-out 走平早停（patience=8×eval_every=500=4000
steps 無改善，上限 30000）。GPU：zeldajr（AMD R9700 ×4，sbatch `--gres=gpu:1`，
6 顆同批獨立 job，Slurm 排卡）。

| recipe | seed | total_steps_run | best_step | held_out top1_codes | held_out top1_eos | smoke drop(loss[0]→loss[500]) | wall(s) |
|---|---|---|---|---|---|---|---|
| orig | 20260909 | 6500 | 2500 | 28.49% | 100.00% | 3.647→2.615 (Δ1.032) | 245.6 |
| orig | 20260910 | 6500 | 2500 | 28.45% | 100.00% | 3.678→2.623 (Δ1.055) | 247.1 |
| orig | 20260911 | 6500 | 2500 | 28.93% | 100.00% | 3.649→2.609 (Δ1.040) | 244.7 |
| aug | 20260909 | 8500 | 4500 | 25.95% | 59.78% | 3.662→2.848 (Δ0.815) | 324.9 |
| aug | 20260910 | 8000 | 4000 | 26.34% | 63.52% | 3.684→2.811 (Δ0.872) | 305.6 |
| aug | 20260911 | 8500 | 4500 | 26.07% | 63.66% | 3.666→2.853 (Δ0.813) | 326.5 |

全部 6 顆 smoke 全過（前 500 步 loss 明顯下降，遠超亂猜基準 ln(32)=3.466）。六顆
訓練 log 用同一個 `train_gen_sft.py`（未修改），只有 `--corpus`/`--seed`/`--tag`
不同，走新 sbatch `train_gen_sft_aug.sbatch`（WT 指到本 worktree，其餘含
GPU-visibility-guard 重試邏輯照抄既有 `train_gen_sft.sbatch`）。

觀察：aug 三顆的 `held_out top1_codes`（~26%）比 orig 三顆（~28.6%）低約 2.5pp、
`top1_eos` 從 100% 掉到 60~64%——這不是訓練變差，是任務本身變難：aug 語料裡
1/3 強是「中途起點、不定長度」的子樣本，模型要多學「這個中途點該在哪裡停」，
teacher-forcing top1 本來就該比「EOS 永遠在第 50 格」這種平凡解低。`corpus_meta`
欄位裡的 `n_train=4457/n_val=495` 是繼承自原始 `corpus_v1.pt` 的 meta（欄位沒有
覆寫，屬本檔紀錄疏漏，見六、沒做到清單）——**訓練實際用的樣本數以 log 為準**：
`corpus: n_train=12293 n_val=1365`（三顆 log 逐字相同），這才是真正吃進去的量。

---

## 三、主表（6 顆 × R 格 per-leg）

```
recipe       seed |       4 |       8 |      16 |     inf
-----------------------------------------------------------
  orig   20260909 |    ---  |  0.6132 |    ---  |  0.6187
  orig   20260910 |    ---  |  0.6545 |    ---  |  0.5890
  orig   20260911 |    ---  |  0.6087 |    ---  |  0.5455
   aug   20260909 | 0.6643  |  0.6899 | 0.6604  |  0.5900
   aug   20260910 | 0.6575  |  0.6651 | 0.6598  |  0.5337
   aug   20260911 | 0.6490  |  0.6368 | 0.6342  |  0.5931
```

（原配方只跑 {8,inf} 兩格，設計上不需要 4/16——鐘形定讞只需要 R8 vs R∞ 的差值；
增強組按 design 補掃全部 4 格。200 題、貪心解碼、`build_tasks(seed=20260908)`
同一批任務，跟前一份 NOTE 完全同分佈可比。）

### 差值表（R=8 − R=∞，原配方 3 seed）

```
seed        R8       Rinf     diff
20260909  0.6132   0.6187   -0.0055
20260910  0.6545   0.5890   +0.0656
20260911  0.6087   0.5455   +0.0632
                    mean    +0.0411
```

---

## 四、判準判讀

### 判準1：鐘形定讞——**FAIL（按字面判準）**，但方向偏正

字面判準＝「全部 >0 **且** 平均 ≥.03」。**不成立**：seed 20260909 的差值是
**-0.0055**（R8 反而略低於 R∞），不滿足「全部 >0」，所以按 design 定死的合取判準，
**這是 FAIL，如實報**——不能因為平均漂亮就跳過「全部」那個字。

但拆開看不是均勻的雜訊：3 個差值裡 2 個清楚為正（+.0656、+.0632），1 個是貼著 0
的小負值（-.0055，量級遠小於下面量到的訓練抖動 std=.030）。平均 +.0411 本身遠超
.03 門檻。**誠實的中間結論**：R=8 的重寫增益**方向上偏正、但沒有乾淨地對全部 3
個訓練 seed 都成立**——第 3 顆 seed 的極小負值，數值上完全落在下面量到的訓練抖動
範圍內，不能排除它只是抖到了門檻另一側，但字面判準不允許把「合理懷疑是雜訊」
自動升級成「PASS」，所以判定仍是 FAIL。

### 訓練抖動幅度（原配方 3 seed 的 R=∞ per-leg，順便量出來，這個數字本身有價值）

```
seed=20260909  Rinf=0.6187  對錨點(.5875)差=+0.0312
seed=20260910  Rinf=0.5890  對錨點差=+0.0015
seed=20260911  Rinf=0.5455  對錨點差=-0.0420
mean=0.5844  std=0.0301  range=[0.5455, 0.6187]（跨幅 0.0732）
```

**這個數字本身值得記住**：只換訓練 seed（同語料、同架構、同超參），R=∞（開環播放，
決定性 eval）的 per-leg 就能在 3 顆之間跨 **7.3 個百分點**、標準差 **3.0 個百分點**。
均值（.5844）貼著原始單 seed 錨點（.5875，差 -.0031），錨點本身沒有問題；但
**前一份 NOTE 報告的「R=8 對錨點 +.0506 增益」，量級跟這個訓練抖動的 std 是同一個
數量級**——這件事本身就是本工單存在的理由（單 seed 量不出這個，鐘形有可能部分是
幸運的 seed 抽樣）。判準1 FAIL 的結果跟這個抖動幅度是同一個故事的兩面。

### 判準2：增強判讀——**PASS**

```
seed=20260909  R8=0.6899
seed=20260910  R8=0.6651
seed=20260911  R8=0.6368
平均=0.6639   門檻>=0.66
```

平均 **0.6639 ≥ 0.66**，**PASS——增強版清楚贏**。但誠實點名：門檻是壓線過的
（超出僅 +.0039），且 3 顆裡有 1 顆（20260911, .6368）單獨看沒有過 .66——是平均
過線，不是「每顆都乾淨過線」。跟增強前的原配方 R=8（3 顆平均 (.6132+.6545+.6087)/3
=.6255）比，增強版 R=8 平均高出 **+.0384**，方向一致支持增強有效，但不是壓倒性
差距。

### R=4 代價檢查——**代價消失，而且反轉成清楚收益**

```
seed=20260909  R4=0.6643  Rinf=0.5900  (R4-Rinf)=+0.0743
seed=20260910  R4=0.6575  Rinf=0.5337  (R4-Rinf)=+0.1238
seed=20260911  R4=0.6490  Rinf=0.5931  (R4-Rinf)=+0.0559
平均 (R4-Rinf) = +0.0847
```

對照：原始（未增強、單 seed）量測（[[NOTE-2026-09-09-rewrite-receding]] §2.2）
R=4 .5415 vs R=∞ .5875，**(R4-Rinf)=-0.0460**（密集重寫是淨損失）。增強後 **3
個 seed 全部反轉為正**、平均 +.0847——不只是「代價消失」，是從「密集重寫倒扣
4.6 分」變成「密集重寫倒賺 8.5 分」，方向完全一致、沒有一個反例。**這是本工單
目前最乾淨的結果**：直接支持假說 H（u 沒學過從半路接寫，是密集重寫代價的主因）
——教模型見過中途起點之後，密集重寫（R=4）不再是負擔，反而變成用得起的優勢
（更頻繁用真實姿態修正、又不再付「沒見過這種條件」的稅）。

附：增強版 R=16（甜蜜點是否漂移）：0.6604 / 0.6598 / 0.6342，平均 .6515——跟 R=8
（平均.6639）、R=4（平均 .6636）三者非常接近，鐘形在增強版裡明顯**變鈍/變平**
（R=2 沒測，但 R=4/8/16 三點幾乎打平），不像原始版那樣尖銳地在 R=8 出現單一峰值
——這本身也是支持假說 H 的間接證據：如果密集重寫的代價主要來自「條件分佈差」，
補上這個分佈之後，重寫頻率的最適點應該變得不那麼敏感（各頻率都堪用），量到的
形狀正是如此。

---

## 五、姿態置換探針（主人 2026-09-09 追加驗收）

**在證明什麼、判準是什麼**：u（GenSFT 生成頭）的鐘形/重寫收益，會不會其實是純
waypoint 條件撐起來的、u 根本沒在用姿態（obs）token？若是，半路重寫「餵當下真實
姿態」這件事本身不帶資訊增益，前面§四的判讀都要重新想。判準（主人原話，質性）：
掉幅大＝有在用；幾乎不掉＝忽略——本檔另外自訂 .10 當「大」的參考門檻（工單沒給
精確數字，已在腳本與此明標，非工單既有判準）。

**設計**：對一顆原配方 ckpt（best_orig=seed 20260910）與增強組最佳 ckpt
（best_aug=seed 20260909，兩者皆按 R=8 per-leg 選出，跟 gif/§四引用同一個選法），
各取 40 題子集（來自標準 200 題固定前綴，`build_tasks(seed=20260908)` 同一母體），
固定 seed（20260909314）配對出 derangement，只把 u 的姿態條件輸入（初始
`generate()` 的 s0、R=8 重寫時 `regenerate` 的姿態）換成配對到的另一題的真實資料
obs；decode_from_idx 的動作解碼 obs 條件不動（仍是真實模擬姿態）——精確隔離「u
決定寫哪個字」跟「字怎麼被解碼成動作」兩件事。

```
ckpt                        arm                    R=inf per-leg   R=8 per-leg
orig(s20260910, n=40)       baseline(真姿態)              0.5584        0.6163
orig(s20260910, n=40)       permuted(換別題姿態)           0.2778        0.1667
                             掉幅                          +0.2807       +0.4496

aug(s20260909, n=40)        baseline(真姿態)              0.6203        0.6512
aug(s20260909, n=40)        permuted(換別題姿態)           0.1875        0.1111
                             掉幅                          +0.4328       +0.5401
```

**判讀：兩顆 ckpt 掉幅都遠超自訂門檻 .10（原配方 +.28~+.45，增強版 +.43~+.54）——
u 有在用姿態 token，不是被 waypoint 條件單獨撐起來的。** 換成別題的姿態後，
per-leg 直接腰斬甚至更低（原配方 R=8 從 .6163 掉到 .1667；增強版 R=8 從 .6512 掉
到 .1111），這是「餵錯姿態 ⇒ 寫錯字 ⇒ 走錯路」的直接證據。附帶觀察（非本探針
主判準，質性點名）：增強版兩格掉幅都比原配方對應格更大（R=inf: .43 vs .28；
R=8: .54 vs .45）——方向上支持「增強訓練讓 u 對姿態的依賴變得更強/更精確」（合理：
原配方只見過『乾淨開頭』單一種姿態分佈，增強版被迫學會在不同姿態下都要正確回應），
但只有 2 顆 ckpt、40 題子集，這個附帶觀察是**觀察不是定論**，沒有跨 seed 驗證過。
**§四的重寫收益判讀因此站得住腳**：既然 u 確實在用姿態，半路重寫「餵當下真實姿態」
帶來的資訊增益是有意義的機制，不是安慰劑。

一手 log：`experiments/gen_sft/logs/pp_orig_s20260910-27167.out`、
`experiments/gen_sft/logs/pp_aug_s20260909-27168.out`；JSON：
`experiments/gen_sft/results/pose_permute_orig_s20260910_summary.json`、
`experiments/gen_sft/results/pose_permute_aug_s20260909_summary.json`。

---

## 六、沒做到 / 不確定清單

1. **判準1（鐘形定讞）按字面是 FAIL**，不是「弱 PASS」——如實報，見上方判讀，
   不因為平均數字好看就模糊字面判準。
2. **`corpus_aug_v1.pt` 的 meta 頂層 `n_train`/`n_val` 欄位是原封不動繼承自原始
   `corpus_v1.pt` 的舊值（4457/495，增強前的樣本數），沒有覆寫成增強後的真實
   計數**——真實計數存在 `meta["aug_sample_ledger"]` 裡（train=12293/val=1365）
   且訓練 log 本身印的是正確數字（`corpus: n_train=12293 n_val=1365`），**訓練
   沒有用錯資料**，這只是 meta 欄位的記錄疏漏，寫本節點名，不是事後才發現的地雷。
3. R=4/16 只在增強組補掃，原配方 3 seed 沒有量 R=4/16——design 只要求增強組補掃，
   原配方沒有這兩格的原配方鐘形（無法判斷原配方本身在增強前 R=4 是否也有這麼大
   的 seed-to-seed 差異，只能引用前一份 NOTE 的單 seed 數字當對照）。
4. 訓練抖動只在原配方 3 seed 上量，增強組的 3 seed 之間是否有類似量級的抖動沒有
   額外拆解（增強組 R=∞ 三顆：0.5900/0.5337/0.5931，肉眼看跨幅也有 ~6 個百分點，
   跟原配方同量級，但本檔沒有把這個當正式判準另外算 std——附原始數字在此，
   讀者可自行核）。
5. gif 只做了檔案格式/決定性重跑一致性檢查，沒有逐幀人眼看過——人眼關要主人自己
   看過才算數（同前兩份 NOTE 慣例）。
6. 姿態置換探針的「掉幅>=.10 算有在用」門檻是本檔自訂（工單沒有給精確數字），
   已在探針腳本與本節明標，不冒充工單既有判準。

---

## 七、一手 log / 檔案索引

```
experiments/gen_sft/aug_corpus_common.py                             增強語料構造邏輯（新檔）
experiments/gen_sft/build_corpus_aug.py                               建語料+自檢（新檔，CPU 直跑不經 slurm）
experiments/gen_sft/results/corpus_aug_v1.pt                          增強語料（10.53MB）
experiments/gen_sft/train_gen_sft_aug.sbatch                          六顆訓練共用 sbatch（新檔）
experiments/gen_sft/eval_rewrite_aug.py + .sbatch                     六顆 eval + 附加 gif follow-up（新檔）
experiments/gen_sft/pose_permute_probe.py + .sbatch                   姿態置換探針（新檔）
experiments/gen_sft/analyze_bell_aug.py                                收表算判準（新檔）
experiments/gen_sft/results/bell_aug_analysis.json                    判準計算結果（機器可讀）
experiments/gen_sft/results/gifs_rewrite_aug/R8_success_0_ep260.gif    增強最佳設定(aug s20260909,R8)成功例，32幀，legs 3/3
experiments/gen_sft/results/gifs_rewrite_aug/R8_fail_0_ep319.gif       增強最佳設定(aug s20260909,R8)失敗例，50幀，legs 0/3

訓練 log（sbatch job id）：
  orig s20260909=27135  orig s20260910=27136  orig s20260911=27137
  aug  s20260909=27138  aug  s20260910=27139  aug  s20260911=27140
eval log（sbatch job id）——⚠️ 首次提交用 `sbatch --export=ALL,VAR=a,b` 傳
R_GRID="8,inf" 這種含逗號的值，被 Slurm `--export` 的逗號分隔語法吃掉只剩第一段
（`R_GRID=8`，"inf" 消失）：orig 三顆（27143/27144/27145）跑完才發現只有 R=8 一格；
aug s20260909 的 27146 更巧——它依賴的訓練 job 剛好在我 `scancel` 指令送出前就已
完工，dependency 觸發搶跑掉了（22:13:09~22:13:28 完整跑完，只是同一個 bug 只留下
R=4 一格），27147/27148（aug s20260910/20260911）當時訓練還沒完工，`scancel`
成功攔下（狀態 CANCELLED，未執行）。修法＝`export R_GRID=... ; sbatch --export=ALL`
（不把含逗號的值放進 `--export` 的逗號分隔清單，改讓 `ALL` 從當前 shell 環境繼承），
驗證 log 行 `r_grid=8,inf` 正確後，全部 6 顆重新提交跑完整 grid（27149~27151 為
orig 三顆重跑，27152/27164/27165 為 aug 三顆重跑/補跑，其中 27152 覆寫掉 27146
的殘留檔案，最終落地的 `rewrite_eval_aug_s20260909_summary.json` 是 27152 的完整
4 格結果，非 27146 的殘留——已用 `r_grid` 欄位核對過)：
  orig s20260909=27149  orig s20260910=27150  orig s20260911=27151
  aug  s20260909=27152（27146 為同 bug 下的殘留跑，已被覆寫，log 仍保留供查）
  aug  s20260910=27164  aug  s20260911=27165
gif follow-up=27166   pose-permute(orig s20260910)=27167   pose-permute(aug s20260909)=27168

checkpoint：
  experiments/gen_sft/results/gensft_{orig,aug}_s{20260909,20260910,20260911}_best.pt
```

隨機性固定：切點抽樣 `cut_seed=20260909`；6 顆訓練 seed 各自 `{20260909,20260910,
20260911}`（同時決定權重初始化與 batch 走訪順序）；eval 200 題固定
`build_tasks(seed=20260908)`（跟前兩份 NOTE 同一批任務）；姿態置換探針固定配對
`pair_seed=20260909314`。生成一律 `temperature=0` 貪心、決定性。「我不知道」與
「答案是否」分開表達：判準1 是查過數字後的**否（FAIL，字面判準不成立）**；判準2
是**是（PASS，但壓線）**；R4 代價檢查是**是（反轉為正，最乾淨的結果）**；抖動
幅度是**量出來的數字**（std=.030，非拍的）。
