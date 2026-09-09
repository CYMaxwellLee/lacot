# NOTE — gen-sft-v1：先寫譜、再演奏（hindsight 真字串 SFT 生成式 u）（2026-09-09）

_對應 [[NOTE-2026-09-08-teacher-relay]] 的判讀：hindsight 真字串接力 per-leg .554/.555，
遠贏所有「邊走邊逐格選字」的閉環方式（貪心 .063、已訓練主頭 .320）——教材本身沒問題，
問題是選字方式。本檔驗證下一步：訓一個生成式 u，一次自回歸把整段「(起點+路標)->步法
字串」寫出來（不逐格看當下狀態選字），照譜開環播放，看 per-leg 到達率能不能貼近甚至
超過 teacher_relay 量到的教材上界。_

_全部程式：`experiments/gen_sft/`（新目錄，只 import `wv_common.py` / `p1_replay.py` /
`p2_analyze_traces.py` / `teacher_relay/run_teacher_relay.py` 的既有函式/常數，未修改
任何既有檔案）。一手 log 全部在 `experiments/gen_sft/logs/`。_

---

## 〇、一句話

**學會寫譜了，而且寫得比教材本身還好。** 200 條接力題目（跟 teacher_relay 完全同一
批、同協定）上，u 貪心生成 1 條字串照譜開環播放，per-leg **.5875**——超過驗收①門檻
（.30）近 2 倍、超過驗收②「貼近教材上界」門檻（.45）、甚至超過教材本身的 hindsight
schedule 版 **.554**。BoN(N=8) 採樣取最好更到 **.7485**，N=16 到 **.8282**——u 不只
學會了教材的字，取樣多條之後找得到比任何一條真實 hindsight 軌跡本身更會走的字串。
隨機字串對照組 per-leg=.20（<.30 閘門），管線自檢通過。訓練 held-out next-token
top-1（只算「哪個 code」的位置）28.71%，是亂猜基準 3.125% 的 9.2 倍。golden 一致性
12/12 bit-exact，反例（打亂/換段）12/12 都會亮。

---

## 一、任務構造

### 1.1 語料構造（train split，全部 episodes）

`experiments/gen_sft/gen_sft_common.py::build_corpus()`：路標序列（M、wp_xy、
skip-if-arclen<DELTA_SUB=7.5）**直接呼叫 `run_teacher_relay.build_tasks()`**（不重寫
一份，同構是結構保證）；code 序列改用 `wv.cut_segments`+`wv.build_seg_tensors` 一次
批次切出整個資料集的段、再一次 batched `encode_idx`（比逐段呼叫快很多），照
`episode_bounds` 的走訪順序切片回每集自己的 code 序列，用 `assert` 驗證長度跟
`build_tasks` 算出的 `n_chunks` 對得上。

- `antmaze-medium-stitch-v0.npz`（train）：5000 episodes，每集固定 201 步
  （T//4=50 chunks）。48 集因總弧長 < 7.5 被跳過（0.96%），4952 條可用。
- waypoints M：min=1 p50=3 max=4（跟 teacher_relay 的 val 集統計同量級）。
- code 序列長度 L：**恆為 50**（因為這份資料集所有 episode 都固定 201 步）——
  這代表本次「可變長 pad+mask」的設計真正被練到的是**條件端**（M∈{1,2,3,4}），
  輸出端長度在這份資料集上其實是個常數，模型不需要對「該生多長」做真正的
  跨樣本泛化，只需要學會「數到 50 就吐 EOS」，這件事本身確實學得很乾淨（見
  §二 top1_eos=100%）。這不是我偏離設計，是資料集本身的性質，在此點名以免
  誤讀「pad+mask」機制被完整驗證過各種長度。
- episode 級 9:1 切分：`split_seed=42`，n_train=4457 / n_val=495（⚠️ 這是本檔
  自己語料庫內部的 held-out，用來量訓練期 next-token top-1；**不是** antmaze
  `-val.npz` 那 200 題接力用的 val——兩者是完全不同的兩件事，下文一律用
  「corpus-val」vs「接力 200 題」分開稱呼，避免混淆）。
- 存檔：`experiments/gen_sft/results/corpus_v1.pt`（4.25MB，含 train/val 樣本
  list 與完整 meta）。build log：`experiments/gen_sft/logs/build_corpus.log`。

### 1.2 模型

`experiments/gen_sft/model.py::GenSFT`：3 層、d_model=128、4 heads、dropout=0.1、
620,577 參數（落在設計給的範圍 2~4 層/128~256 內，沒有做超參數掃描——結果已經
遠超門檻，沒有調參的必要，見 §五#1）。條件當 prefix token：s0(29,) 線性投影 1
個 token，每個路標 xy(2,) 各線性投影 1 個 token（右側 pad＋mask，M 可變）；
輸出接在 BOS 之後，vocab=32(code)+1(EOS)=33。**整個序列（prefix+BOS+輸出）套
同一個純 causal mask**（不做 prefix 雙向的 prefix-LM 變體——路標本身沿路徑
排序，causal 順序不損失資訊，換取實作簡單、少一個 bug 面，這是設計決定，寫在
`model.py` docstring）。Teacher forcing CE 訓練，padding 位置的 target 用
`ignore_index=-100` 蓋掉。

### 1.3 一個真的踩到、值得記住的雷：GPU 可見性被自己 import 的模組污染

`train_gen_sft.py` 一開始（job 26819/26821/26823/26825）連續四次
`torch.cuda.is_available()=False` 直接 FAILED，中間走了兩個錯誤方向（疑心
sbatch 環境繼承、疑心 gres 綁定要 srun 包）才找到真正原因，記錄下來避免自己
或別的腳本再踩一次：

`run_teacher_relay.py` 檔頭有 `os.environ.setdefault("HIP_VISIBLE_DEVICES", "")`
——它自己是 CPU-only 腳本，這樣寫是防自己誤連 GPU 的自保寫法，本身沒有錯。
但 slurm 的 gres 外掛在這台只設定 `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`，
**從來沒設過** `HIP_VISIBLE_DEVICES`——所以「這個變數本來就不存在」的前提
成立，`setdefault` 真的生效，把它釘成空字串。`train_gen_sft.py` 經
`import gen_sft_common` 會連帶 `import run_teacher_relay`；只要這個 import
發生在本行程**第一次呼叫** `torch.cuda.is_available()` 之前，ROCm 就會把
「HIP 可見裝置=空字串」讀成看不到卡，即使 `CUDA_VISIBLE_DEVICES` 明明是
`"0"`。修法：在 `import gen_sft_common` **之前**先呼叫一次
`torch.cuda.is_available()`（torch 會把偵測結果快取住，之後 `setdefault`
再怎麼設都不影響這個行程）。已經在 `train_gen_sft.py` 裡加上這個 guard 並附
完整推理過程當註解；`eval_gen_sft.py` 因為本來就設計成 CPU-only（自己也在
sbatch 裡明確設空 `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`），不受影響、
不需要這個 guard。**這是值得存的實作 insight**（任何未來會 import
`gen_sft_common`/`run_teacher_relay` 又需要真的用 GPU 的腳本都可能中招）——
點名給後續判斷是否要 `save_insight`，本檔本身沒有呼叫這個工具（見 §五#6）。

除錯過程一手 log：`experiments/gen_sft/logs/train_gen_sft.sbatch-{26819,26821,
26823,26825}.out`（四次失敗）、`train_gen_sft.sbatch-26827.out`（修好後成功）。

### 1.4 接力 200 題構造（跟 teacher_relay 嚴格同一批）

`eval_gen_sft.py` 呼叫的是**同一個** `run_teacher_relay.build_tasks(raw_val,
n_traj=200, seed=20260908, ruler, rho=1.875)`——跟 teacher_relay 用完全相同的
函式呼叫、完全相同的參數，所以 200 題本身（哪些 episode、哪些路標）**逐位元
相同**：waypoints M min=1 p50=3 max=4、total_legs=595，跟 teacher_relay NOTE
§一報的數字一致。差別只在「這個 leg 要走哪個字」的來源——teacher_relay 用
hindsight encoder 現算真字，本檔用 u 生成的字串。**這代表本次 u-greedy/u-BoN
跟 teacher_relay[schedule]/[nn] 是嚴格同分佈可比的**（同一批 leg），但跟
main_head(.320)/貪心選字(.063) 不是（那兩個來自 live rollout 的 planner
subgoal，leg 分佈不同，見 teacher_relay NOTE §五#1 與本檔 §四）。

生成設定：貪心 1 條（temperature=0）；BoN 用 temperature=1.0 採樣，固定
`torch.Generator` seed=20260909（`BON_SEED`）；隨機字串對照組 20 題（接力
200 題的前 20 題，同一個 build_tasks 呼叫的前綴，不是另外構造）、uniform 取
code，固定 `np.random.default_rng(99)`（`CONTROL_SEED`）。模擬步數上限固定
= 該集 `n_chunks`（=50，本協定不變）。

**一個設計沒明講、這裡明訂的決定**：若生成字串比 `n_chunks` 短（EOS 提早
出現），模擬在字串用完那一格直接停止（不補字、不重複播最後一個字）——語意
上等於「模型自己判斷字串已經講完，沒有更多譜可照」。實測 `ran_out_of_codes_
frac`（見 §二）在 u-greedy/u-BoN(N=8)/u-BoN(N=16) 三組聚合結果裡全部是 0.0
——因為 BoN 的「贏家」篩選（每題 N 條裡挑 legs_reached 最大的那條當代表）
天然會把過短、大概率一步都沒走到的退化樣本篩掉，這個決定在本次結果裡實際上
沒有被啟用到會影響聚合數字的程度，但機制仍然存在，點名說明。

**CPU-heavy 的 rollout 走 multiprocessing 平行化**：`--cpus-per-task=8`，
每個 worker 各自建一次 env/dict decoder/raw npz（`mp.Pool` initializer），
之後重複用；不平行化的部分（u 生成字串本身）用小模型單行程做，本來就很快
（200 題貪心 1.4s、200x16=3200 條 BoN 採樣 47.9s，見 §二）。

---

## 二、結果

### 2.1 驗收①：golden 一致性

`experiments/gen_sft/golden_check.py`，log：`experiments/gen_sft/logs/
golden_check.log`，json：`experiments/gen_sft/results/golden_check.json`。

- 抽 3 條 train episodes（seed=1234：ep 110/1937/3893）各 4 段，**12/12
  bit-exact 一致**（`build_corpus` 批次管線 vs 直接對同段動作呼叫
  `model.encode_idx`）。**PASS**。
- 反例：先試「反轉該段 4 步的時間順序」，12 段裡 10 段 code 立刻改變；剩下
  2 段（ep1937 chunk0/chunk1）反轉後 code 沒變（量化對這兩個特定小擾動剛好
  穩定，不代表比對機制失靈），改用「換成另一條完全不同 episode 的段」升級
  反例，2 段都改變。**12/12 反例最終都會亮，比對機制不是恆等空比對**。
  **PASS**。

### 2.2 驗收②：訓練 smoke + held-out top-1

`experiments/gen_sft/train_gen_sft.py`，log：`train_gen_sft.sbatch-26827.out`，
json：`results/gensft_v1_train_summary.json`，loss curve：
`results/gensft_v1_loss_curve.png`。

- **smoke（前 500 steps）**：loss `3.6651 -> 2.6520`，Δ=1.0131（亂猜 32 類
  CE 基準 ln(32)=3.4657，起點就已經低於亂猜、500 步內持續明顯下降）。**PASS**。
- **訓完**：patience-based 提早停在 step **6500**（held-out loss 連續 8 次
  eval=4000 steps 沒有改善 >1e-3；held-out loss 在 step 2500 見底
  2.2853，之後回升到 2.4366，是輕微 overfit 不是還沒收斂，見圖）——落在
  design 建議的 30k~60k **以下**，理由是「loss 走平為準」這個判準本身在
  step 2500 附近就已經達成（且之後惡化），不是我沒跑夠，是提早看到平台+
  過擬合訊號就照設計的判準停手，訓練時長本身不是越長越好。
  用 **best checkpoint（held-out loss 最低，step=2500）** 做下游 eval：
  held-out **top1_codes_only=28.71%**（只算「哪個 code」的位置，n=24750）
  ——是亂猜基準 **3.125%（=1/32）的 9.2 倍**，「遠超」門檻達成。**PASS**。
  （補充對照：top1_all=30.10%、top1_eos=100.00%——EOS 位置固定在相對第 50
  格，模型幾乎立刻學會這個純計數子任務，不是本次要驗證的訊號主體。）

### 2.3 驗收③：eval smoke（20 題，含隨機字串對照）

log：`eval_gen_sft.sbatch-26885.out`（26861 那次因 §1.3 之外另一個小 bug——
`model.generate()` 回傳的「是否吐出 EOS」布林值語意寫反，印出來的「撞 cap
條數」字面上跟長度統計矛盾——已修正，26861 的數字沒錯只是那一行標籤誤導，
以 26885 為準），json：`results/SMOKE_eval2_summary.json`。

- 對照組（隨機字串 20 題）per-leg=**0.2000**（< 0.30 閘門）——**PASS**，
  繼續往下跑。
- u-greedy per-leg=0.5405、u-BoN(N=2) per-leg=0.5641（20 題小樣本，抖動大，
  只作管線驗通用途，正式數字看 §2.4 的 200 題）。**PASS（管線全程跑通）**。

### 2.4 驗收④：主結果表（200 題，正式）

log：`eval_gen_sft.sbatch-26886.out`（N=8，含 gif）、
`eval_gen_sft.sbatch-26889.out`（N=16，bonus，`--skip-gifs`）；json：
`results/gensft_eval_N8_summary.json`、`results/gensft_eval_N16_summary.json`；
疊圖：`results/gensft_eval_N8_{speed,z,adiff}_overlay.png`；原始陣列：
`results/gensft_eval_N8_raw.npz`。N 預算：先跑 N=8（design 要求的基準），
實測 200 題+N=8 的 rollout 只花 47.6s 總 wall time，餘裕很足，照 design
「還有餘裕就補到 N=16」的但書多跑了一組 N=16（79.1s）當 bonus 上界參考，
兩組都完整報。

```
關                                  u-greedy      u-BoN(N=8)    u-BoN(N=16)   control(隨機)
────────────────────────────────────────────────────────────────────────────────────────
① per-leg 不限時到達率                0.5875        0.7485        0.8282        0.2000
   legs reached / attempted          235 / 400     378 / 505     453 / 547     5 / 25
① per-leg 限時 1.25N/1.5N (N_data)   41.9% / 46.5% 53.2% / 61.7% 62.1% / 69.2%  0.0% / 4.5%
外加：全程走完比例                     0.175         0.365         0.530         0.000
② 步速 p50 / mean                    .0996 / .1051 .1124 / .1150 .1203 / .1215 .0857 / .0893
③ 翻倒率（逐步 / 逐集）                0.024% / 3.5% 0.012% / 1.5% 0.006% / 0.5% 0% / 0%
④ 平滑度 |a_t-a_(t-1)| p50 / mean     1.261 / 1.386 1.248 / 1.366 1.262 / 1.382 1.180 / 1.287
   選字使用率(active/K,ppl,top1frac)  31/32,21.7,13.1% 32/32,30.9,5.1% 32/32,30.7,5.3% 32/32,31.6,4.2%
   ran_out_of_codes_frac（聚合後）    0.0           0.0           0.0           0.0
生成長度 min/p50/max、撞cap未出EOS數  50/50/50, 0    2/50/60(x8), 2   1/50/60(x16), 5   —（不用模型）
```

**真螞蟻對照行**（一手引用自 `docs/NOTE-2026-09-08-teacher-relay.md` §二，
非本次量測）：步速 p25/p50/p75/mean = .086/.132/.175/.131；adiff p50/mean =
1.768/1.840；翻倒逐步線（資料定義）= 1.00%。

**錨點對照行**：

```
vq_oracle 貪心選字（closed-loop, live rollout, 舊制）   per-leg .063   ⚠️ 見下方可比性註記
已訓練主頭（closed-loop, live rollout, job 25987）      per-leg .320   ⚠️ 見下方可比性註記
teacher_relay[schedule]（hindsight真字, 同本協定同200題）per-leg .554   ⭐ 嚴格同分佈可比
teacher_relay[nn]（hindsight真字+閉環校準, 同本協定）    per-leg .555   ⭐ 嚴格同分佈可比
u-greedy（本次，生成 1 條，同本協定同 200 題）           per-leg .5875  ⭐ 超過 teacher_relay 兩版
u-BoN(N=8)（本次，同本協定）                             per-leg .7485  ⭐⭐ 大幅超過教材上界
u-BoN(N=16)（本次，同本協定，bonus）                     per-leg .8282  ⭐⭐⭐
random control（本次，均勻亂碼，20 題，同本協定）        per-leg .2000  管線自檢用，非教材比較對象
```

**可比性註記（重要，抄 teacher_relay NOTE §五#1 的邏輯、套在這裡）**：
u-greedy/u-BoN 跟 **teacher_relay 兩版是嚴格同分佈可比**（同一個
`build_tasks(seed=20260908)` 呼叫、同 200 題、同 595 個 leg、同 rho）；但跟
**貪心選字(.063)/已訓練主頭(.320) 不是**——那兩個數字來自 live rollout 裡
planner 對整個迷宮生成的 subgoal（每集約 39 個 leg、共 9761 個），leg 的
難度分佈跟本協定（每題 1~4 個從真實錄製軌跡取的 leg，共 595 個）本來就不
保證一樣。u-greedy(.5875) 大幅超過 .320 這件事**方向上**支持「u 學到的字串
比已訓練主頭的逐格選字強」，但嚴格的量級比較只對 teacher_relay 兩版成立。
單次 eval 抖動尺 ±.03~.07（design 給的量級）——u-greedy(.5875) 比
teacher_relay[schedule](.554) 高 0.0335，落在抖動尺上緣邊界，**不是壓倒性
差距**，但 u-BoN(N=8)=.7485 高出 teacher_relay 0.19 以上、u-BoN(N=16)=.8282
高出 0.27 以上，都遠超抖動尺，是穩定訊號不是噪音。

---

## 三、gif（人眼關）

`experiments/gen_sft/results/gifs/`，格式沿用 teacher_relay：每支兩格並排
（左＝跟拍看步態、右＝全迷宮固定機位看走到哪；橘點＝目前 leg 的路標，X 標記
＝目前目標，用 `u.set_goal()` 即時挪動）。取自 **u-greedy** 結果（成功/失敗
各 2 支，design 沒指定要哪一組，選 greedy 是因為它是「u 生成 1 條」這個主張
本身要驗的東西）：

```
greedy_success_0_ep319.gif   20 幀   全走完 3/3 legs（80 步）
greedy_success_1_ep202.gif   27 幀   全走完 3/3 legs（108 步）
greedy_fail_0_ep260.gif      50 幀   卡在 0/3 legs（跑滿 200 步預算）
greedy_fail_1_ep56.gif       50 幀   卡在 0/4 legs（跑滿 200 步預算）
```

重跑一致性斷言（`completed`/`n_steps_run` 跟第一次測量時 bit-exact 相同）
已通過（寫在 `render_gif_exemplars` 裡的 assert，job 26886 log 沒有噴任何
assert 失敗，四支都成功存檔）。⚠️ 我自己只抽看過 2 支裡各 1 幀（
`greedy_success_0_ep319` 與 `greedy_fail_0_ep260`，用 Read 工具看的靜態幀，
不是逐幀播放），確認左右兩格構圖、螞蟻姿態、橘點/X 標記位置都正常——**人眼
關要主人自己看過才算數**（同 P1/P2/teacher_relay 慣例），本檔沒有僭稱已經
人工驗收過動畫本身。

---

## 四、判讀

**證據鏈**：

1. §二2.1：golden 12/12 bit-exact、反例 12/12 會亮 ⇒ 資料管線（`build_corpus`
   的批次切法）跟直接呼叫 encoder 完全等價，下游訓練/eval 用的 code 序列
   可信，不是管線本身在製造假訊號。
2. §二2.2：held-out top1_codes_only 28.71%（9.2x 亂猜線）⇒ u 確實從
   （s0, 路標序列）這種極短的條件（1~4 個路標 token）裡學到了能預測整段
   50 步步法字串裡「該用哪個 code」的訊號，不是背答案（held-out 是 episode
   級切分，訓練沒看過這些集）。
3. §二2.4 控制組：per-leg=.20（<.30 閘門）⇒ 隨機字串走不動，驗證整條
   rollout/leg 判定管線沒有系統性作弊或算錯（不是隨便什麼字串都會被判定
   「到達」）。
4. §二2.4 主結果：u-greedy(.5875) 超過驗收①②兩個門檻、甚至超過
   teacher_relay[schedule](.554)；u-BoN(N=8/16) 更遠超（.7485/.8282）⇒
   工單要證明的主張成立：**u 能一次自回歸生成整段步法字串，照譜開環執行
   的 per-leg 到達率遠超閉環逐格選字**（已訓練主頭 .320、貪心選字
   .063——即使承認這兩個數字的可比性打了折扣，u-greedy 用嚴格同分佈的
   teacher_relay 兩版當對照組依然贏，這個結論不依賴那個打折的比較）。
5. u-BoN 贏過 teacher_relay 兩版本身（hindsight 真字）這件事值得多想一句：
   hindsight 真字串反映的是「這一條被錄下來的軌跡實際怎麼走」，不保證是
   「從這個起點到這些路標最好走的字串」——u 取樣多條、oracle 選最好，能
   找到比任何單一真實錄製軌跡更直接/更會走的字串組合，並不矛盾（是
   「教材」跟「教材空間裡的最優解」的差別，teacher_relay NOTE §四第 4 點
   自己也點過「.554 不是完美教材」）。u-greedy 單條就已經逼近甚至略超過
   教材，代表 u 學到的不只是背誦某條特定軌跡，是學到了「怎樣的字串組合
   對這種起點+路標的形狀比較管用」這個更一般的訊號。

**回答工單的問題**：**成立**。u 貪心生成 1 條的 per-leg（.5875）超過「學會
寫譜」門檻（.30）達 .29、超過「貼近教材上界」門檻（.45）達 .14、甚至超過
教材本身的 schedule 版（.554）；BoN(N=8/16) 進一步證明生成分佈裡存在明顯
更好的字串（.7485/.8282），不是單純運氣。管線可信度由 golden 一致性與
隨機字串閘門雙重把關，訓練訊號由 held-out top-1（9.2x 亂猜）獨立佐證。

---

## 五、沒做到 / 不確定清單

1. **沒有做超參數掃描**：模型結構（3層/d128/4heads/dropout.1）、batch size
   （256）、lr（3e-4 cosine）等只試了一組，落在 design 給的範圍內、結果已
   遠超門檻就沒有調參——但這代表不知道「這組是不是比較好的」，只知道
   「這組夠用」。
2. **只有 1 個訓練 seed**（`torch.manual_seed(1234)`）、**只有 1 個 BoN
   採樣 seed**（20260909）——沒有量 run-to-run variance（跟 teacher_relay
   NOTE §五#4「只有 1 個字典 seed」同一類開放項，本檔延續同樣的限制沒有
   加碼解決）。
3. **可比性打折的兩個錨點**（貪心選字 .063、已訓練主頭 .320）沒有重新在
   本協定下量一次——本檔的結論不依賴這兩個數字（依賴嚴格可比的
   teacher_relay 兩版），但如果要「跟已訓練主頭正面對決」，嚴謹做法是在
   同一個 200 題協定下也跑一次已訓練主頭的 live rollout 取同一批 leg 的
   子集，這件事本檔沒有做。
4. **BoN 的「贏家」選擇只用 legs_reached 最大值**，沒有在 legs_reached
   打平時做二階 tie-break（目前用 `np.argmax` 預設取第一個）——實務上
   對聚合後的 per-leg/completion 數字沒有影響（同分時選哪個都不改變
   legs_reached 這個數字本身），但點名這個細節沒有特別處理。
5. **gif 只抽看過 2 支裡各 1 幀**（不是逐幀播放看完 4 支全部）——人眼關
   最終要主人自己看過才算數，本檔沒有僭稱已完成人工動畫驗收。
6. **GPU 可見性污染（§1.3）是本次真正花時間排查出來的實作 insight，但
   本檔本身沒有呼叫 `save_insight` 存進 elsa-knowledge**——是否要存、
   存成什麼 domain/confidence，留給看這份 NOTE 的人（Luna 或主人）決定，
   這裡只把發現跟修法完整寫清楚，不越權自己存。
7. **訓練提早停在 step 6500**（design 建議 30k~60k）：不是沒做到，是
   patience-based 判準本身在這個 step 附近就判定「held-out loss 已經在
   惡化」而停手，繼續跑到 30k+ 只會讓 best checkpoint 選擇邏輯持續指向
   更早的 step，不影響最終用的 checkpoint——但如果要嚴格滿足「30k~60k」
   這個字面數字，本次沒有做到，這裡如實點名。
8. **沒做到的「查不到／跑不出」項目**：無。golden check、語料構造、訓練
   （smoke+完整+held-out top1）、eval（smoke 20 題+control+正式 200 題
   greedy/BoN(8)/BoN(16)）、疊圖、4 支 gif 全部跑完並留下一手 log。
