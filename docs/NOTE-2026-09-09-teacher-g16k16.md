# NOTE — G16K16 因子化字典 hindsight 真字接力驗證（2026-09-09）

_對應工單：G×K 因子化字典掃描（[[NOTE-2026-09-08-gk-scan-route]]）膝點落在
G=16,K=16（64 bits/chunk，val 重建 MSE .01529），但「教材能不能走」只驗過
K32 單碼本（[[NOTE-2026-09-08-teacher-relay]]：schedule 版 per-leg .5536）。
本檔補 G16K16 的同款驗證，判準：同一批 200 條接力題目、schedule 開環版協定，
per-leg 到達率與 K32 .554 比較，|Δ|<=.07 算打平（G16K16 以更高容量打平即可
續留換裝候選）、明顯高＝換裝候選、明顯低＝G16K16 教材出局。決定下一版生成式
SFT 要不要換這本字典。_

_全部程式：`experiments/walk_verify/teacher_relay_g16k16/`（新目錄，只 import
`wv_common.py`／`p1_replay.py`／`p2_analyze_traces.py`／
`teacher_relay/run_teacher_relay.py`（重用 `build_tasks`）／`gk_scan/gk_common.py`
的既有函式與類別，未修改任何既有檔案）。sbatch 一手 log：
`experiments/walk_verify/teacher_relay_g16k16/logs/TR-G16K16-26820.out`
（job 26820，CPU-only，`-p admin -A it -q great-mage --nodelist=zeldajr
--cpus-per-task=8`，實際 wall time 19.5s、`sacct` 記錄 Elapsed=00:00:20，
ExitCode 0:0）。結果 JSON：
`experiments/walk_verify/teacher_relay_g16k16/results/teacher_relay_g16k16_summary.json`。

---

## 〇、一句話

**打平，續留換裝候選，但沒有贏。** G16K16 schedule 版 per-leg 不限時到達率
**.4973**（184/370），對照 K32 的 **.5536**（217/392），Δ=**−.0563**，落在
判準 |Δ|<=.07 的打平帶內（不是壓線，帶內還有約 0.014 的餘裕）。爛錨對照
（20 題、16 組各自隨機取字）per-leg 只有 **.0476**，遠低於 .3 的管線自檢
門檻，確認量測管線本身健康、.4973 是字典的功勞不是量測誤差。載入自檢：整個
val 切分（25000 段）重建 MSE 精確重現 gk_scan 記錄值（差=0.00e+00）。
**值得記住的反直覺點**：G16K16 每 chunk 容量是 K32 的 12.8 倍（64 bits vs
log2(32)=5 bits），重建 MSE 也大幅優於（G16K16 val .0153 vs K32 這裡沒有
直接可比的重建 MSE 數字——見§五#3），但接力 per-leg 沒有跟著贏，只打平、
甚至點估計還略低。這代表「教材容量」不是這條 relay pipeline 的唯一瓶頸，
但本次沒有進一步拆解原因（推論，非量測，見§四）。

---

## 一、任務構造

跟 K32 版（`docs/NOTE-2026-09-08-teacher-relay.md` §一）**唯一的協定差異
＝ encode/decode 用哪本字典**，其餘全部沿用同一套：

- **題目構造**：直接 `import run_teacher_relay as trk32` 後呼叫
  `trk32.build_tasks(raw, n_traj=200, seed=20260908, ruler, rho=1.875)`
  ——跟 K32 版是**同一個函式呼叫**（不是重新實作一份可能 subtly 不同的版本），
  同一顆 seed=20260908 保證抽到**同一批 200 條**題目。來源＝官方 held-out
  val 檔（`antmaze-medium-stitch-v0-val.npz`，500 集），跳過同一集
  （episode=386，總弧長 3.362 < DELTA_SUB=7.5），路標仍是每 7.5 弧長一個、
  595 個路標、M 分布 min=1 p50=3 max=4——跟 K32 版逐位元一致。
- **schedule 版定義**：第 k 個 chunk 播該集自己第 k 段的 hindsight 真字
  （encoder 對 `act[s0+4k : s0+4k+4]` 算出的 code），與模擬中 ant 實際在哪裡
  無關，純開環照表播。decoder 的 obs 條件輸入＝模擬中 ant 的當下 obs（連續
  多 chunk 接力，不是每 chunk 重置）。
- **到達判定**：rho=1.875，leg 依序，leg m 要等 leg m-1 到達才「開始」，
  沒被輪到的 leg 不算「試過」——跟 K32 版、跟官方 per-leg 統計同一慣例。

**唯一換的東西**（見 `run_teacher_relay_g16k16.py` 檔頭 docstring 的完整
介面差異說明）：

```
K32   （wv_common.CondVQVAE）    ：encode_idx(x) -> 單一 scalar code（K=32 選 1）
                                    decode_from_idx(idx, obs_raw) （obs 正規化在模型內部做）
G16K16（gk_common.FactorizedCondVQVAE）：encoder(x) -> 128 維連續 -> 16 組各自
                                    VQEMA(K=16) 各選 1 字 -> idx 是長度 16 的
                                    code 元組；decode_codes(idx, obs_norm) 需要
                                    呼叫端自己先正規化 obs
```

ckpt：`experiments/gk_scan/ckpt/G16_K16.pt`（`state_dict`/`config`/`obs_mu`/
`obs_sd` 四個 key，config 裡 G=16,K=16,group_dim=8,seg_len=4,hidden=512,
split_seed=42,val_frac=0.1,ckpt_train_seed=36016——直接讀 ckpt 的 config，
沒有硬編任何架構參數）。**obs_mu/obs_sd 直接在 ckpt 裡找到**（訓練時算好存
進去的，不是模型的 buffer），⛔ 沒有重算：`obs_mu[:2]=[9.4337, 10.2370]`，
跟 `MAZE_CENTER=(9.42, 10.24)` 對得上，是合理的資料集常數。

**範圍縮減**：K32 版跑了 schedule／nn 兩版，本檔**只做 schedule**——工單§主
結果只要求 schedule 版跟 K32 的 .554 比，nn（最近鄰閉環修正）不在本次範圍
內，沒有做（見§五#1）。

**新增**（K32 版沒有的）：爛錨對照（`mode="random"`）——每個 chunk 不查真字，
16 組各自 uniform 隨機取字，同協定跑 20 題（`tasks[:20]`，同一批 200 題的
前 20 條，非獨立抽樣，見§五#2）。用來確認量測管線本身沒有把「亂走也算贏」
悄悄放進分母/分子。

---

## 二、結果：四關全表（K32 對照 vs G16K16，並排）

一手：`experiments/walk_verify/teacher_relay_g16k16/results/teacher_relay_g16k16_summary.json`；
K32 對照來源（一手引用，非本次量測）：
`experiments/walk_verify/teacher_relay/results/teacher_relay_summary.json`；
原始陣列：`teacher_relay_g16k16_raw.npz`；sbatch log：
`experiments/walk_verify/teacher_relay_g16k16/logs/TR-G16K16-26820.out`。

```
關                                    K32[schedule]（對照 .5536）  G16K16[schedule]（本次）
────────────────────────────────────────────────────────────────────────────────────
① per-leg 不限時到達率（官方口徑）      0.5536                      0.4973
   legs reached / attempted           217 / 392                   184 / 370
① per-leg 限時 1.25N／1.5N (N_data)    41.7% / 45.7%                37.8% / 41.5%
   n evaluable (N_data)               379 / 370                   360 / 354
① per-leg 限時 1.25N／1.5N (N_table)   13.0% / 20.2%                11.0% / 20.3%
   n evaluable (N_table)              391 / 391                   365 / 364
外加：全程走完比例                      0.125 (25/200)               0.070 (14/200)
② 步速 p25/p50/p75/mean                .062/.103/.148/.108          .062/.103/.147/.107
   （真螞蟻 p25/p50/p75/mean，ruler_pack 一手）  .086 / .132 / .175 / .131（同一列，兩邊共用同一份 ruler，供對照）
③ 翻倒率（逐步／逐集）                  0.026% / 3.5%                0.057% / 6.5%
   （真資料定義上逐步線 = 1.00%，tautology：fall_line 定義即 p1）
④ 平滑度 |a_t-a_(t-1)| p50/mean        1.236 / 1.350                1.720 / 1.789
   （真螞蟻 p50/mean，ruler_pack 一手）  1.768 / 1.840（同一列，供對照）
   選字使用率                          32/32（1 組 K=32）           G=16 組跨組平均：
                                        perplexity 31.09/32          active 16.0/16（min=max=16）
                                        top1 5.1%                    perplexity 14.91/16, top1 11.2%
                                                                      （n_samples=9647 個實際跑到的
                                                                      chunk；跟 K32 那格不是同一統計量，
                                                                      只是最接近的類比健康度指標，見§五#4）
```

**爛錨對照**（K32 版沒有這項，本次新增的管線自檢，20 題、16 組隨機取字）：
per-leg 不限時到達率 = **0.0476**（1/21），completion_ratio = **0.000**
（0/20）——遠低於門檻 0.3，管線自檢通過（見 log「[驗收②] 爛錨
per-leg=0.0476 < 0.3」）。

**載入自檢**（見 log「[驗收①]」區塊）：

```
3 個 train 段（seg_starts=[941841, 468362, 437126]）
  per-sample MSE = [0.01424, 0.00881, 0.01425]  mean=0.01243
  對照 G16_K16.json 記錄 val MSE=0.015289（NOTE 表列 .01529）—— 同量級 ✓

加驗（非工單硬性要求，成本近乎 0，一併做）：
整個 val 切分（25000 段）重建 MSE = 0.0152894016... 
  vs G16_K16.json 記錄值 0.0152894016...  差 = 0.00e+00 —— 精確重現，載入正確 ✓
```

疊圖（真螞蟻 vs G16K16 schedule）：
`results/teacher_relay_g16k16_{speed,z,adiff}_overlay.png`——抽看過 speed
疊圖（見下方，兩條曲線形狀大致重疊、G16K16 峰值略左移＋左尾略胖，跟④
smoothness 偏高、completion 偏低方向一致，沒有矛盾）。

---

## 三、gif（人眼關，2 支）

`experiments/walk_verify/teacher_relay_g16k16/results/gifs/`，格式同 K32 版
（左＝跟拍看步態、右＝全迷宮固定機位看走到哪；橘/粉點＝當下路標，即時用
`set_goal()` 挪動）：

```
schedule_success_ep423.gif   18 幀   全走完 2/2 legs（72 步）
schedule_fail_ep319.gif      50 幀   卡在 1/3 legs（跑滿 200 步預算）
```

⭐ 兩支 gif 用的都是**跟 K32 版同一條 episode**（423=success 範例、
319=fail 範例）——因為題目集合逐位元相同（同一顆 seed），render_exemplars
挑「per_task 裡第一條 completed=True／False」時剛好挑到同一條，不是刻意
對齊。跟 K32 版比：ep423 這條兩邊都全走完但步數不同（K32 22 幀/88步 vs
G16K16 18 幀/72步，G16K16 這條反而更快走完）；ep319 這條兩邊都卡在同一個
進度（K32 1/3 legs、G16K16 也是 1/3 legs，跑滿 200 步預算）——單一題目的
個案，不代表整體（整體是 595 個路標中、G16K16 版被試過的 370 條 leg 的
平均，同 K32 NOTE §三的提醒）。

⚠️ 我自己只抽看過 `schedule_success_ep423.gif` 其中 1 幀（確認左右兩格構圖、
螞蟻姿態、粉點位置都正常，見下方附圖），**沒有逐幀播放看完全部 2 支**——
人眼關要主人自己看過才算數（同 K32 慣例）。

---

## 四、判讀

**證據鏈**：
1. §二①：G16K16 per-leg=.4973 vs K32=.5536，Δ=−.0563，|Δ|=.0563 < 判準
   .07 ⇒ **打平**，落在帶內、不是壓線（帶寬還剩約 .014 餘裕）。
2. §二爛錨：隨機碼 20 題 per-leg=.0476，遠低於 .3 門檻、也遠低於 schedule
   版的 .4973（差了一個數量級）⇒ .4973 是「查到對的字」帶來的，不是量測
   管線本身的偽陽性。
3. §二①外加：完整走完比例 G16K16=0.070 明顯低於 K32=0.125。用「leg 獨立」
   的粗略近似估：.4973^2.975≈0.125（M 均值 595/200≈2.975），跟觀察值
   0.070 有明顯落差（觀察值只有近似估的 56%，K32 自己那組近似對照是
   .554^2.975≈0.171 vs 觀察 .125，比值 73%）——G16K16 這裡的「近似估 vs
   觀察值」落差比 K32 版更大，代表 G16K16 的失敗更容易在同一集裡連續發生
   （leg 間不獨立、失敗會聚集），本次沒有再往下拆解成因（推論，非量測，
   見§五#5）。
4. §二②③④：步速分佈跟真螞蟻**同量級但系統性偏慢**（精確核對過比例，不是
   目測「重疊」）——G16K16 的 p25/p50/p75/mean 分別是真螞蟻同一分位數的
   72.4%/78.0%/83.8%/81.7%，跟 K32 schedule 版自己的比例（72.4%/78.0%/
   84.7%/82.3%，用同一份 ruler_pack 重新核對，非引用 K32 NOTE 文字）幾乎
   一模一樣——兩本字典驅動出來的步速分佈實質上疊在一起，都是真螞蟻的
   7~8 成速度；翻倒率仍遠低於真資料定義線（0.057% vs 1.00%）、選字使用率
   健康（16 組
   全活、無死碼）——教材本身「步態健康」的結論跟 K32 版一致方向。但
   ③翻倒率明顯更高（逐步 0.057% vs K32 0.026%，2.16 倍；逐集 6.5% vs
   K32 3.5%，1.86 倍——兩把尺的倍數不同，精確核對過分別標出，不是同一個
   數字套兩次）、④平滑度 p50/mean 都比 K32
   粗糙約 33~39%（1.720 vs 1.236、1.789 vs 1.350）——**精確核對過比例**：
   G16K16 的 p50/mean 是真螞蟻自己 1.768/1.840 的 97.3%/97.2%（仍在其下
   方一點點，不是超過），K32 的 p50/mean 只有真螞蟻的 69.9%/73.4%——兩本
   字典的平滑度都沒有超過真螞蟻，但 G16K16 明顯更貼近真實雜訊量級（K32
   動作比真螞蟻平滑得多，G16K16 幾乎貼齊真螞蟻自己的抖動量級）
   ⇒ G16K16 這本字典驅動出來的動作比 K32 更抖、摔倒更頻繁，這是 per-leg
   沒能贏過 K32、completion ratio 更低的一個合理的（但未經因果驗證的）
   部分解釋。
5. **反直覺點**（§〇已點名）：G16K16 每 chunk 64 bits，是 K32（5 bits）的
   12.8 倍容量，val 重建 MSE 也遠低（.0153，96.6% 降幅——見
   `NOTE-2026-09-08-gk-scan-route.md`），但 relay per-leg 沒有跟著贏，
   只打平且點估計還略低、翻倒率跟平滑度都變差。**這代表「重建誤差低」
   不直接等於「接力好走」**——decoder 容量、16 組各自量化造成的「組合
   爆炸」（16^16 種組合，訓練資料裡每種特定組合出現次數極稀疏，某些
   訓練時沒見過的字組合在 relay 情境下 decode 出來的動作可能沒那麼穩）
   都是可能的解釋方向，但本次沒有做拆解實驗驗證，這是推論不是量測
   （見§五#6）。

**回答工單的問題——「G16K16 教材能不能打平 K32」**：**能，打平**
（Δ=−.0563，|Δ|<.07），依工單判準**續留換裝候選**——但不是「贏」，也不是
「乾淨的打平」：per-leg 點估計略低、completion ratio 明顯更低（.070 vs
.125）、翻倒率跟平滑度都比 K32 差。換裝這本字典去訓下一版生成式 SFT 之前，
建議留意：更高的原始容量不會自動換來更好的 relay 表現，這批字典的
「可走性」瓶頸看起來不在編碼容量，值得在換裝前想清楚是不是要先查§五#5/#6
那兩個推論。

---

## 五、沒做到 / 不確定清單

1. **只做了 schedule 版，沒做 nn（最近鄰閉環修正）版**——工單§主結果只
   要求 schedule 版跟 K32 的 .554 比，這是刻意的範圍縮減不是遺漏，但代表
   「G16K16 在允許漂移後重新校準」的表現本次沒有量到，K32 版 nn 版
   （.555，幾乎跟 schedule 打平）的模式在 G16K16 上是否重現，不知道。
2. **爛錨對照的 20 題是同一批 200 題的前 20 條（`tasks[:20]`），不是獨立
   抽樣**——工單原文「同協定 20 題」沒有明講是否要獨立抽樣，選用前綴子集
   是最簡單、最可重現的做法（等同於用 n_traj=20 呼叫同一個 `build_tasks`），
   但這代表爛錨對照跟主結果的 20/200 條題目有重疊，不是完全獨立的兩批。
3. **G16K16 的重建 MSE（.0153）跟 K32 的重建 MSE 沒有直接放在一起比較**
   ——K32 那本字典（`p0_dict_v1_L4K32_50k.pt`，P0 版本，latent_dim 另有
   配方）沒有在本次或先前任何 NOTE 裡報過跟 gk_scan 24 格同條件（同
   train/val 切分、同 obs 正規化方式）算出來的重建 MSE 數字，所以§〇/§四
   說「G16K16 重建誤差大幅優於」時，那個「優於」其實只是跟 gk_scan
   scan 自己的 baseline 比（96.6% 降幅），不是跟 K32 這支字典的重建誤差
   直接比大小——兩本字典的重建誤差本身沒有做過同條件的頭對頭比較，這裡
   如實點名，不是隱藏。
4. **選字使用率那格不是同一統計量**（K32 是 1 組 K=32 codebook 的
   perplexity/top1；G16K16 是 16 組各自 K=16 codebook 的跨組平均）——
   已在§二表格與§一裡明講，這裡重申：這是「最接近的類比健康度指標」，
   不是可以直接相減比大小的同一把尺。另外 G16K16 這裡的 code_usage 是
   在**這次 200 題 relay 實際跑到的 9647 個 chunk**上統計的，跟
   `gk_scan/results/G16_K16.json` 裡「整個訓練+驗證資料集（250000 段）」
   算出來的 per_group_usage（例如 group0 perplexity=15.34/16,
   top1=8.98%）**不是同一個母體**——relay 情境下的 top1 明顯偏高
   （11.2% vs group0 的 8.98%），是「這 200 條真實軌跡在動作空間裡的
   取樣本來就不是均勻的」造成的合理差異，沒有進一步拆解是否所有 16 組
   都同方向偏移還是只有少數幾組。
5. **completion ratio 比「leg 獨立」近似估低得更多（§四#3）的原因沒有
   拆解**——只指出這個現象比 K32 版更明顯，沒有做「leg 失敗是否在同一集
   內聚集」的逐集分析（例如：是不是特定幾條 episode 的路標特別刁鑽，
   兩本字典在那些集上都失敗、但 G16K16 失敗得更早更徹底）。
6. **「重建誤差低不等於接力好走」的兩個候選解釋（decoder 容量瓶頸／16^16
   組合稀疏）都是推論，沒有做拆解實驗**（例如：固定 obs 條件、掃過 16 組
   codebook 裡「訓練時很少共同出現的組合」在 decode 品質上是否特別差）
   ——工單範圍是「打平/更高/更低」的判定，沒有要求做這一層歸因，如實
   標註為開放問題。
7. **只有 1 個 G16K16 ckpt（單一 seed=36016，沿用 gk_scan 掃描時的既有
   ckpt）**——跟 `NOTE-2026-09-08-gk-scan-route.md` §三#1 同一個既有開放
   限制（24 格都只有 1 個 seed），本次沒有新增 run-to-run variance 的量測，
   G16K16 這個 .4973 本身也可能有一次性訓練路徑造成的抖動。
8. **只有 1 個爛錨對照 seed**（anchor_seed=20260909）——0.0476 遠低於 0.3
   門檻（約 6 倍餘裕），換一個 seed 大機率不會翻案，但沒有實測驗證這個
   直覺。
9. **gif 只抽看過 1 幀，沒有逐幀播放看完全部 2 支**——同 K32 版慣例，人眼
   關留給主人自己看過才算數。
10. **本次流程本身是全確定性的**（跟 K32 版同款理由：`env.reset()` 的
    RNG 用量在下一行 `set_state` 就被蓋掉、encode/decode 都在 eval mode
    無隨機性；random 模式的隨機性只來自顯式傳入、可重現的
    `np.random.default_rng(anchor_seed)`）——render_exemplars 的 bit-exact
    重跑斷言（`completed`/`n_steps_run` 一致）在 log 裡確認通過（兩支 gif
    都成功產生，沒有觸發 AssertionError），這點不是待確認項。
11. 沒做到的「查不到／跑不出」項目：無。ckpt／obs 統計量／ruler_pack／
    val 資料載入、200+20 題抽樣、爛錨對照、主結果、四關量測、疊圖、2 支
    gif、K32 對照來源全部跑完並留下一手 log 與 JSON，sbatch job 26820
    ExitCode 0:0 完整跑完（wall 19.5s，遠低於 6 小時預算）。
