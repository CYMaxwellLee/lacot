在證明什麼：ucontrast1 實作包上機後，能否產出「A/B 逐題配對、全執行事後 oracle@8」而且能自證的帳。判準＝PLAYBOOK §5 八條逐條掃；驗不到的格標 UNVERIFIED，不當 PASS。

# VERDICT-S2（S2 檢察官：Claude Opus 5.5，與實作方 GPT/astra 不同家族；判官不動手）

```
VERDICT: FAIL(8, 2)
  FAIL-8  env reset 在真 env 上不配對：所有 GPU 階段都會在收尾 crash，README 的「同 env reset」主張不成立
  FAIL-2  2a 粗掃 25 題與正式 200 題題集沒有隔離（題號完全包含）
          2b B 臂被構造成「開環」：主尺 A-B 混入開環稅（必裁，不是實作錯；lead 已裁語義，本件只呈風險）
  其餘    1 PASS / 4 PASS / 6 PASS / 5 PASS-with-notes / 3 PASS-with-notes（gate 碼與 LEAD-CLARIFICATION 不一致，上機前必改）/ 7 notes
```

核心邏輯親驗正確：collector 的 u 凍結、noise 專用 stream、oracle 算術、flag-off 惰性、exclusive write、patch＝工作樹。FAIL 的面很窄、可修，但 FAIL-8 不修，任何一關都拿不到結果。

受檢檔未動：collector.py／run.py／test_wiring.py／scratch_lacot_rollout.py 的 SHA256 驗前驗後一致（`sha256sum -c` 全 OK），`__pycache__` 未寫入。所有重跑都在 scratchpad。

---

## 逐點證據

### 8. RNG 配對　FAIL

torch 與 noise 兩條 stream 對齊是對的，env reset 那條不對。

- 手推 (task=3, episode=17, draw=5)：stream_seed = 7·3+17+1000003·5 = 5000053；collector 實印 5000053。公式不含 arm／σ，所以 A/B/各 σ 同 seed；draw 0 = 38 = 原版 `torch.manual_seed(7*task+sd_)`（scratch_lacot_rollout.py:3082）。noise 是專用 CPU Generator（collector.py:39），不推進 global RNG（M4 mutant 被抓）。flow 在 CUDA 上抽（`device` L34；nf_head.py:229 `torch.randn(..., device=device)`），和 CPU MT19937 的 noise 是不同演算法。
- env reset：rollout 只呼叫 `env.reset(seed=1000*task+sd_)`（L3079），全主檔 `np.random.seed` 命中 0 次。ogbench 的 init/goal 抖動在 `locomaze/maze.py:401-405` 呼叫 `add_noise`，而 `add_noise` 用的是 global legacy `np.random.uniform`（maze.py:564-566），而且在 `super().reset(seed=...)` 之前。所以 `reset(seed=)` 釘不住起點與終點。
- 這是已知病：`lacot/dev_eval.py:126-135`（2026-08-26 稽核）註解寫明，並用 `np.random.seed(seed0+idx)`＋`env.action_space.seed(...)` 修了 dev 路徑；commit cf78926 訊息「ogbench reset抖動吃legacy全域np.random流⇒arm間非配對比較（既有病、排修）」。官方 `rollout()` 那條從來沒修。
- 真 env 重現（交付的 policy_chunk/rollout AST＋交付的 Collector＋真 ogbench pointmaze-large-stitch，只有 flow/head 是玩具，CPU）：

```
DELIVERED
  B σ0 : (task1,ep0) initial_sha per draw = 323f2e0469 / 6397fc935b / 5eef58ca2e   env_seed 欄 = 1000,1000,1000
         summarize RAISES: Unpaired resets: (1, 0) initial_sha256
  A σ0 : summarize RAISES: Unpaired resets: (1, 0) initial_sha256
  B σ.1: summarize RAISES: Unpaired resets: (1, 0) initial_sha256
  A vs B(0.1) 同 (task,ep,draw) 起點相同: False
SCRATCH-PINNED（只在 oracle 路徑 reset 前加 np.random.seed + action_space.seed，dev_eval 同款）
  B σ0 : 三個 draw 起點相同，summarize OK（σ0 軌跡逐位元相同）
  A σ0 / B σ.1 : summarize OK；A vs B 起點相同: True；B σ.1 同題三條軌跡互異: 3
```

- 後果：`summarize` 在 `write_result` 裡才跑（collector.py:104-106），所以每個 GPU job 會把 8 個 draws 全跑完，然後才 crash；rows 沒落檔、只剩 traceback；run.py 收到 CalledProcessError 後 BLOCKED。calibration／mutant／smoke／main 全部中。select 的 A/B 首 u 配對（run.py:221-224）也會跟著失敗。好消息是它**會大聲失敗**（initial/goal hash 那道檢查有牙），不會靜默產出假帳。
- CPU 套件為什麼抓不到：ToyEnv.reset（test_wiring.py:28-32）是 seed 奇偶的純函數，沒有模擬 global np 抖動。M8（reset seed 混入 draw）有被抓，所以檢查本身沒問題，是玩具 env 的模型不夠真。
- 附帶：`env_seed` 欄是 collector 自己從 (task,ep) 算的（collector.py:84），不是讀自真的 reset 呼叫，所以上面那個 env_seed 欄一直是 1000，但真起點每次都不同。見第 7 點。

### 2. 判準格自由度　FAIL（2a）＋必裁（2b）

2a 題集沒有隔離。smoke／mutant 用 `episodes=5`（run.py:127），main 用 40（run.py:149）；rollout 的題號是 `for sd_ in range(SEEDS)` 從 0 起（L3078）。所以粗掃題 (task 1-5, ep 0-4) 完全包含在正式題 (task 1-5, ep 0-39) 裡，佔 25/200 = 12.5%；同 (task,ep,draw) 的 stream seed 也完全相同（collector.py:34-35）。現在（未釘抖動）是「同題號、不同抖動」；一旦照第 8 點把抖動按 (task,ep) 釘住，勝檔 σ 的 smoke rollouts 就會是正式 B 臂前 25 題的逐位元複本（GPU 確定性允許的話），等於在測試集上選格。方向偏袒 B，也就是偏向「u 降級」那一支判決。量級〔拍〕：3 個相關檔在 25 題上取最大的贏家詛咒約 +2~5pp，稀釋 25/200 後約 +0.3~0.6pp，比 5pp 門檻小，但它有系統方向，而且免費就能拿掉。

2b B 臂被構造成開環。B 在 draw 0 第一個 chunk 抽一顆 u，之後全 chunk、全 draws 都讀同一顆（collector.py:49-66；主檔 L2401-2405）；A 每 chunk 重抽。實驗室自己的文件把這個形狀定義成 open-loop：`docs/FINDINGS-2026-08-23.md:211-218`「open-loop：u 只在起點算一次，全程沿用；cond 仍每 chunk 更新」，而且「兩者的差距本身就是 reasoning quality 的直接量測」。主檔 L2420-2424 也把「整集用第 0 個 chunk 那個 u」記成過去的 bug。所以主尺 A-B oracle@8 ＝（閉環 vs 開環的單條品質差）＋（擾動源差），主尺分不開。CONVERGENCE.md:164-166 對殺手對照的措辭是「換擾動源」，現在的 B 換了擾動源，也換了迴路。實驗室在別條線量過開環稅 17.5 點（CONVERGENCE.md:70，bcodec 教師重放，不是本 ckpt），遠大於 5pp 門檻，所以「A ≥ B+5pp」有可能純靠構造達成，不需要任何多樣性優勢，這就是一格「B 構造上拿不到分」。lead 已裁「B 凍 u 語義：實作正確」（LEAD-CLARIFICATION §2），本件不推翻裁示，只呈：照此語義，主尺無法區分「u 是執行變體抽樣器」與「開環比較差」。可選做法見必改清單第 3 項。

### 3. 預註冊合成規則　PASS-with-notes

- 已預註冊：primary＝pooled oracle@8（README:6-10，標〔拍、預註冊〕）、A-B≥5pp、兩 seed 同向、400 題合成；σ 選擇規則＝B pooled oracle@8 最大、平手取小 σ（README:60；run.py:225），先於資料；s35 另做 sweep（README:61）。
- 缺口一：LEAD-CLARIFICATION §1 把 .6680 換成 (a) GPU flag-off 3 題＋(b) A draw0 的 200 題 R1 落在 j_signal noclimb s33=.350 的「抽樣誤差內」＋smoke sanity 帶 .15~.6。「抽樣誤差內」的數字寬度哪裡都沒寫，要先寫死再跑。提醒：獨立單位是 5 個官方 task（主檔 L3208-3209），binomial SE（p=.35、n=200 約 .034，兩次獨立跑的差約 .048）會低估。
- 缺口二：run.py 還是舊合約。`calibration_contract` 寫死 expected 0.6680（run.py:88），calibration 固定 seed 13（L123），PASS 規則是 |pooled-.6680| ≤ se（L183）。這份碼跑不了 lead 的新錨，所以 calibration→mutant→smoke→main 這條 gate 鏈照現在的碼走不通。
- 缺口三：`checkpoint()` 在 training_started_unix 為 null 時直接 raise（run.py:56-57），跟 LEAD-CLARIFICATION §3「填 null、註明不適用」衝突。SHA256 今天重驗：s33 `88180676…`、s35 `ba5009ec…`，與 checkpoint-inspection.log 一致；`_st*` 家族兩個 seed 各只有 st8000 一顆（glob），所以「編號最大」本來就自動成立。
- 小註：「兩 seed 同向」建議寫明＝各 seed 的 A-B 同號（我的讀法）。

### 4. 尺的出處　PASS

collector 記的成功位元就是 rollout 自己的 `success` 變數：任一步 `info.get("success")`（L3103-3104），在 L3109 `ORACLE_COLLECTOR.end(success, steps)` 傳入。J-signal 的 `rates.R1` 是 `rollout(1, True, …)` 的回傳，也就是同一個變數聚合（L3107、L3120；j_signal README 以 rates.R1 比較）。collector／run／oracle 區塊對 per_task／completed 零引用（grep）。LEARNED_REFINE=0 時 `_apply_refine` 是恆等（L2286-2287），所以 R1＝flow-only，與 J-signal noclimb 同路。
注意：M6（成功位元寫死 False）通過全部 9 tests，主量測的接線沒有任何測試或執行期檢查釘住。見建議 R1。

### 5. mutant 走真路徑　PASS-with-notes

- σ0 mutant 那關：run.py:130-131（arm B、σ0、8 draws、5 episodes）→ LACOT_ORACLE_* → 主檔 L3189-3205 → 跟 smoke／main 同一個 rollout()／Collector。是真路徑。
- 交付的兩份 log（cpu-wiring.log、cpu-wiring-v2.log）只有最後 OK，包裡沒有「先 FAIL 過」的證據。兩向證據由我補（下表，真路徑 mutant 都改在 scratchpad 的副本上）：

```
mutant                              改哪裡                              結果
M0 baseline（driver 自檢）           無                                  9/9 pass
M1 測試期望 (6,48)->(6,47)          test_wiring                         FAIL  test_sigma_zero
M2 flag-off seed+1                  主檔 rollout L3082                  FAIL  test_flag_off, test_a_draw_zero
M2b flag-off 多耗一次 torch RNG      主檔 policy_chunk else 分支         FAIL  test_flag_off, test_a_draw_zero
M3 B 只在 draw 內凍 u（每 draw 重抽）  collector.get_u                     ERROR ValueError: B u differs across draws（4 tests）
M4 noise 改吃 global RNG            collector.action                    FAIL  test_sigma_stream, test_a_draw_zero
M5 拿掉 σ0 軌跡 hash 檢查            collector.summarize                 FAIL  test_leak...(trajectory_sha256)
M6 成功位元寫死 False                主檔 L3109                          存活  9/9 pass
M7 noise 算了但沒送進 env.step        主檔 L3098                          存活  9/9 pass
M8 reset seed 混入 draw              主檔 L3079                          ERROR Unpaired resets（5 tests）
M9 oracle 只看 draw 0                collector.summarize                 存活  9/9 pass
```

- 照現況上 GPU，σ0 mutant 會因為錯的原因失敗（第 8 點的起點不配對），u／軌跡檢查還來不及發揮意義。修掉第 8 點後，上面 SCRATCH-PINNED 的真 env 跑顯示 B σ0 的 draws 完全相同，mutant 才有鑑別力。
- 缺正向對照：M7 存活，代表 B σ>0 可以靜默退化成 σ0；B 就輸在構造上、A 白贏。見建議 R2。

### 6. flag-off 逐位元　PASS

- 交付套件原樣重跑：9 tests OK（0.537s）。
- 兩向：M1、M2、M2b 都 FAIL（上表），測試有牙。
- 讀碼：頂端 L18-29 在 flag-off 時只讀 env、不 import、不動 RNG；oracle 區塊由 `if ORACLE_COLLECTOR is not None`（L3189）守；hooks 由 `oracle_draw is not None` 守。
- LEAD-CLARIFICATION §1(a) 的 GPU 3 題版 flag-off 比對，包裡沒有對應腳本（UNVERIFIED，屬 lead 側）。

### 7. 斷言方向　notes

- `env_seed` 欄是自算的（collector.py:84），所以 summarize 的 env_seed 相等檢查（L104）和 select 的 paired()（run.py:205）碰到真資料永遠不會失敗。真正有牙的是 initial/goal hash（M8 與真 env 重現都證明它會抓）。
- `oracle@k decreased`（collector.py:121-122）永遠不會觸發，因為 prefix-any 構造上就單調；測試 `vals == sorted(vals)`（test_wiring.py:116-117）同理。
- `negative_mutant_passed`（collector.py:128）只是設定回聲（B ∧ σ0 ∧ 8 draws），實際意義是「summarize 沒 raise」；test_sigma_zero 的斷言（L75-78）大多已被這件事蘊含。只要 summarize 保持「失敗就 raise」就沒問題。
- M9 存活；select 的重算（run.py:216-218）呼叫的是同一個 summarize，不是獨立重算。現行公式由我獨立重算確認正確（第 1 點）。
- `zip()`（test_wiring.py:84、87）長度不同時會靜默截斷（小）。

### 1. 留痕檔自證　PASS

用交付的 `write_result` 產出三份 JSON（玩具 A σ0、B σ.1、B σ0；6 題×8 draws），再用不呼叫 summarize 的獨立程式，從 `draws[].success` 重算：oracle_at_k、pooled、per_draw_quality、n_success、n_oracle_success、n_tasks、n_draws、最後三欄順序。三份全部一致。手算 A：draw-0 欄 = [0,1,1,0,1,0] → .500；8 條任一成功 = 5/6 = .833 = 存檔值。
限制：成功位元本身就是原始帳，存檔資料推不回它（trajectory 只存 hash，沒存軌跡），只有 B σ0 有內部不變量可驗；A 與 B σ>0 的位元只能信其記錄（所以有建議 R1）。只竄改 `success_bits`、不動 `draws[].success`，select 式重算照樣通過（O4）。

### 額外

- implementation.patch 與工作樹一致（主檔 hunks 相同；collector／run／test／batch／README 逐字相同）。
- `experiments._workorders.ucontrast1.collector` 靠 L11 的 sys.path insert 以 namespace package 解析，本機驗過可 import（jasmine venv UNVERIFIED）。

---

## 上機前必改清單（依序）

1. 〔8〕在 oracle 路徑釘住 reset 抖動：L3079 的 `env.reset` 之前，`oracle_draw is not None` 時做 `np.random.seed(q)`＋`env.action_space.seed(q)`，q 為每題 seed（照 dev_eval.py:131-135）。只放在旗標下，flag-off 才能維持逐位元。ToyEnv 改成從 global np.random 抽 reset 抖動，讓 CPU 套件能看到這一類病。修後 A draw0 不再重播歷史的抖動實現，跟 .350 只能比整體率（本來就是 lead 的計畫）。
2. 〔2a〕粗掃題與正式題的 seed 要不相交（例如 q 與 stream seed 都加 salt，sweep 的 salt ≠ main 的 salt）。要跟第 1 項一起設計，不然修完第 1 項就直接變成測試集選格。
3. 〔2b，必裁〕smoke 之前，lead 要重裁或先預註冊讀法，三擇一：
   (i) B′＝閉環、flow RNG 凍在 draw 0 的 stream（每個 B draw 用 draw 0 的 stream seed 抽 u、仍逐 chunk 依當下 cond 抽；跨 draw 的差異只來自動作噪音）。這是字面上的「換擾動源」，而且 σ0 時 B′ 逐位元等於 A draw0，σ0 mutant 照樣成立、還多一個 A/B 錨。
   (ii) 保留 B，另加一臂「開環、每 draw 新 u、無噪音」，把開環稅和擾動源拆開。
   (iii) 保留 B，預註冊：只有當 B 在選定 σ 的 per_draw_quality 均值不低於 A 超過 δ 時，A-B 才能讀成「多樣性」；否則報「開環稅＋多樣性、未拆分」。
4. 〔3〕run.py 的 gates 對齊 LEAD-CLARIFICATION：calibration 改成 (a)＋(b)，拿掉 .6680／s13 合約（run.py:85-96、123、183）；ckpt gate 改成 SHA256 釘選（上面兩個值）＋路徑逐字，拿掉 training start 必填（run.py:56-57）。改完重跑 CPU 套件（code_hashes 會重綁；目前還沒有任何 gate 檔，不會作廢既有證據）。
5. 〔3〕(b) 錨的數字容許帶、smoke sanity 規則，送出前寫進 README。

## 建議（不阻擋，但便宜，防靜默失效）

- R1（M6）：oracle 迴圈裡 `rate = rollout(...)`，assert rate == collector 該 draw 的成功率。主量測的接線就被釘住了。
- R2（M7）：B σ>0 的正向對照：要求至少一定比例的題目跨 draws 有 >1 個不同的 trajectory hash，否則 FAIL。
- R3（M9）：測試與 select 裡用獨立公式重算 oracle，不要回呼同一個 summarize。
- R4：summarize 之前先把 raw rows 落檔，驗證失敗時才有屍體可驗，不會只剩一段 traceback。
- R5：勝檔落在網格邊緣（σ=.20）時先擴格再上 main（偏袒 B 原則）。

## optional

- O1：`env_seed` 改記真的傳給 reset 的值，或不要再拿它當配對證據。
- O2：主檔在沒有 CUDA 時會靜默退回 CPU（L34）。退回 CPU 時 flow 的 z 和動作 eps 是同一串數字（本機實測前 4 個數相同：True）。建議 noise seed 加 salt，並在子行程斷言 `torch.cuda.is_available()`。
- O3：`negative_mutant_passed` 改成真的反映檢查結果；拿掉永不觸發的單調檢查。
- O4：select 核對 `success_bits` 與 `draws[].success` 一致，以及 k<8 的 oracle_at_k。

## UNVERIFIED

- GPU 上 flow／decoder／head 的逐位元確定性（要等 σ0 mutant 真跑）。
- jasmine venv 的 ogbench 是否同樣走 global np 抖動（本機 venv 已實證；dev_eval 註解引用同一行 maze.py:565，推定相同）。
- jasmine venv 的 import 路徑（本機已驗）。
- 本 ckpt 上 B 開環的單條品質損失實際有多大（2b 的量級）；smoke 的 A 與 B σ0 per-draw 就能量到。
- jasmine 各卡是否同型（四個 smoke job 若分到不同型號的卡，首 u hash 的跨 job 配對可能被硬體差異打破）。

## 自我懷疑

- 真 env 重現用的是玩具 flow/head；但結論只依賴 env.reset，rollout／collector 用的是交付的 AST，所以我對第 8 點有把握。我驗的是本機 venv 的 ogbench，版本差異 UNVERIFIED。
- 2b 的量級是推的：依據是實驗室自己的開環定義，加上別條線（bcodec 教師重放）量到的 17.5 點開環稅，不是本 pointmaze ckpt 的量測。這裡可能其實很小。lead 已裁語義，我是呈風險，不是推翻。
- 2a 的偏差量級是〔拍〕。
- M6／M7／M9 存活代表的是測試覆蓋的洞，不是現行碼的錯。現行碼讀碼加 CPU 跑都正確，而且真 env 跑也顯示 B σ.1 確實有擾動。

## 重跑產物（scratchpad，非受檢目錄）

`/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/37e41886-10cf-41d0-9aae-34351f439be3/scratchpad/s2/`：`realenv_pairing.py`＋`.log`（第 8 點）、`selfproof.py`＋`.log`＋`selfproof_out/`（第 1 點）、`mut/`（driver 與 M0-M9）、`pre-hashes.txt`（受檢檔驗前 hash）。

STATUS: DONE
