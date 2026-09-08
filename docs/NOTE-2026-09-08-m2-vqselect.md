# NOTE — M2 寫譜頭 vq_select（2026-09-08）

_對應 [[DESIGN-2026-09-08-b2-gait-dictionary]] §二（stage2+ 那格）。
前置：[[NOTE-2026-09-08-walk-verify]]（P0 字典 v1、P1 重放、P2 貪心選字判負 .063）
＋ [[NOTE-2026-09-08-teacher-relay]]（hindsight 真字接力 .554 ⇒ 教材能走、值得訓 M2）。_

_全部程式：worktree `~/Projects/lacot-m2`、分支 `vq-select-v1`（已 push）。
主 repo main 一行未動、⛔ 未 merge。結果與一手 log：`experiments/m2_vqselect/`。_

---

## 〇、一句話

**它學得會寫，寫出來的譜也像譜 —— 但一放進閉環，螞蟻會蹲下來卡死。**
寫譜頭在**沒進過訓練的 held-out 資料**上 top-1 到 46.07%（亂猜線 5.13%、8.98 倍），
寫出的字串統計跟教材真字串幾乎一樣（連續同字率 2.4% vs 教材 2.8%、perplexity
30.27 vs 31.10）；但實戰接力 **per-leg 只有 .060**，⛔ 沒過 lo 頭守門線 .124，
跟貪心 .063 打平。病因**不是**選字視野、**不是** argmax、**不是**訓練分佈太窄
（三個假說各被一個量測推翻）—— 是 **29.4% 的 chunk 螞蟻蹲在 z≈0.267 完全不動**，
而那個姿勢資料裡沒有、頭出不去（凍結時連續同字率 94.6%，非凍結只有 4.5%）。

---

## 一、要證明的事與判準

P2 已把責任釘清楚：**壞的不是字典，是選字的方式**（貪心單 chunk 前瞻 per-leg .063，
照 hindsight 真字串接力 .554）。M2 問下一題：

> **一顆小 head 能不能【臨場】把那條字串寫出來？**

預釘判準：守門 per-leg > **.124**（lo 頭 v1）／目標 **.320**（主頭 job 25987）／
教材上界 **.554**（teacher-relay；⚠️ 非嚴格同分佈，見 §六#2）。

---

## 二、施工

### 2.1 頭的形狀（逐項照 π_lo v1 的紀律）

```
輸入  正規化【完整 obs】29 維（決策端 mu/sd）
    ⊕ 【相對】路標 xy 2 維 =(w_xy − obs_xy)/ARC     ⭐ 相對，⛔ 不是絕對座標
鏈    sota_mlp(29,512,512) ‖ sota_mlp(2,512,512) → sota_mlp(1024,512,COND=256)
      → sota_mlp(256,512,K=32,n=3)                 ⇒ 32-way logits
loss  cross-entropy(logits, hindsight 真字)
執行  argmax 選字 → P0 條件化 decoder(字 ‖ 當下原始 obs) → 4 步動作 → clip[-1,1]
```

⭐ 路標吃**相對量**：頭要決定的是「往哪個方向踏這 4 步」；餵絕對座標等於請它背迷宮位置。
⭐ 執行端**沒有前瞻**：P2 的 vq_oracle 每 chunk 在模擬器裡走 128 步枚舉 ⇒ 作弊上界、
不可部署；M2 一個 chunk 只做一次前傳 ⇒ **可部署**。

### 2.2 隔離三道

1. `fork_rng` 建構 —— ⚠️ **字典的 Linear 初始化與 codebook `randn` 也吃 RNG**，一起包進去。
2. 專用抽樣流 `_VQSEL_RNG = default_rng(20260908 + SEED)`。
3. 獨立 optimizer／backward／clip；字典 `requires_grad_(False)`。

旗預設 off ⇒ 模組不建、字典不載、graph 不進、RNG 不碰。

### 2.3 訓練資料＝【接力尺度】

```
起點 t：任一「後面還有 ≥4 步真動作」的 index
路標  ：沿該軌跡的【累積弧長】往前 a 單位的點，a ~ U[1.875, 7.5]
        ⛔ 不是 k 步後的點（lo 頭 v1 的分佈教訓）
        1.875 = rho（planner 到達半徑）、7.5 = DELTA_SUB
        超出 episode 尾巴 ⇒ clamp 到 traj_end（⛔ 不重抽 —— _lo_pairs 的 F6 教訓）
標籤  ：P0 encoder 對 act[t:t+4] 算的 code（全資料 985,000 段一次算完）
```

🚨 **亂猜線是 5.1%，⛔ 不是 1/32 = 3.1%**。教材類別分佈：985,000 段、32 字全活、
perplexity 31.10/32、**最大類 5.1%** ⇒ 最強的常數預測器就是 5.1%。訓練起手就印這行。

### 2.4 防互蓋

ckpt 多存 `vqsel` 段並帶 **`dict_ckpt` 字典身份**；載入端 assert 同一本
（換字典＝32 個輸出換意思，**而它不會報錯**）。檔名：訓練 `_vqsW{W}`、eval `_spvq_select`。

---

## 三、驗收（施工合格才跑實驗）

一手：`experiments/m2_vqselect/logs/M2-verify-ab-26311.out`（zeldajr GPU、job 26311）

```
（1）goldab（main vs vq-select-v1 worktree、pointmaze、GPU）
     A1/A2/B ckpt sha256 全同 1c203ff189359672bd29e53d67c6f411bb83e7a9e8616ce96fb5d0545a47590d
     A1/A2/B json sha256 全同 fd028208c806a35b915d8848b6037fd92db9ce107a726c5fbfa1af71ac29418f
     VERDICT: PASS — 受測分支對 pointmaze 於 GPU 亦零差（ckpt+json 全同）
     ⚠️「A1!=A2 stdout: DIFFER」那行是輸出【路徑字串】不同（A1/ vs A2/），ckpt+json sha 全同。

（2）ant 側旗關閉零差
     A/B rollout json sha256 全同 e77f3b6dea1629dd81e00015faf9747dddd1aaba534157d67f3f8e96cac38b9a
     檔名相同、stdout diff【空】（逐字相同）
     FLAGOFF VERDICT: PASS — ant 路徑旗關閉零差（rollout json sha256 全同）
```

**（3）旗開著也不污染主流**（CPU 對照）：同組輕裝設定跑 `VQSEL_W=0` 與 `=1`，
主模型 40 行 stdout **逐字相同**，rollout json 除多一個 `vqsel_cfg` key 外**逐位元相同**，
唯一差是檔名多 `_vqsW1`（防互蓋、刻意的）。

**（4）100k 步真訓練也對得上 —— 而且是直接比權重**：
`l_nf/l_anchor/l_refine` **100 行對 lo 頭 v1（job 25775）零差**；再把兩顆 ckpt 拆開逐張量比：

```
cond_enc / cond_head / flow / refine / ahead / bc_head / traj_enc / e_pooler / intent_ad / u_dec
  以及 EMA 影子六模組（eval 實際用的那份）        max|Δ| 全部 = 0.000e+00
⇒ M2 這顆的【計畫棧與 lo_100k 逐位元相同】—— per-leg 是同題同錶的比較。
```

⚠️ 但訓練期附帶的 ep2 小 eval（n=10）兩顆數字不同（BC 地板 .500 vs .600）——
**權重相同而數字不同**，那正是既知的「官方 `rollout()` 沒釘 global np.random」在說話
（[[NOTE-2026-09-08-ant-night-verdicts]]）。⇒ ⛔ 別把 eval 的小差讀成權重差。

---

## 四、訓練

slurm job 26312（`-p admin -A it -q great-mage --nodelist=zeldajr --gres=gpu:1`），
**02:00 內跑完（Elapsed 01:58:09、COMPLETED）**；⛔ 全程沒有裸跑上卡。
咒語逐項照 lo 頭 v1（S1_FROM＝`results/ant_pilot/idp03` pilot stage1、STEPS2=100000、
SEED=40、EMA_W=0.999、WARMUP=500、INTENT=embed／drop 0.3、DEC_START=soft、BC_INDEP），
唯一差別＝`LACOT_LO_W=1` 換成 `LACOT_VQSEL_W=1`。

一手：`experiments/m2_vqselect/logs/M2-vqsel-tr-26312.out`

```
step   1000   l_vqsel(ce) 2.1481  top1 0.3252
step  10000   l_vqsel(ce) 1.8328  top1 0.3985        （EMA，m=0.99）
step  50000   l_vqsel(ce) 1.6419  top1 0.4532
step 100000   l_vqsel(ce) 1.5254  top1 0.4848
均勻亂猜 ce = ln32 = 3.4657 ；⭐ 亂猜線（教材最大類）= 5.1%
```

**held-out 探針**（`probe_headout.py`、CPU、各 200,000 配對；
一手 `results/m2_headout.json`）——⭐ 訓練 log 的 top1 量的是訓練分佈上的，
⛔ 不回答「背下來還是學到規律」：

```
                       cross-entropy   top-1     top-5    亂猜線(最大類)   倍率
訓練切分（同分佈）          1.5438      47.75%   89.21%      5.11%       9.34×
官方 held-out val          1.6144      46.07%   87.71%      5.13%       8.98×
  （val = antmaze-medium-stitch-v0-val.npz，⛔ 沒進過主模型或字典的訓練）
train − val 差 +1.68 個百分點 ⇒ ⛔ 沒有過擬合的跡象
頭的輸出分佈：32/32 全活、perplexity 30.07/32、top1 6.2%（教材側 30.98 / 5.1%）
```

⇒ **「學不學得會」這一格：學得會。**

---

## 五、eval 四關全表

slurm job 26313（同卡、同 quota；ep50 / s40 / DEV_EVAL=0 / MAXH 1000 / `SUB_POLICY=vq_select`；
設定逐項照 job 25987 與 P2 job 26202，⛔ 沒有 `LACOT_LO_W`——M2 那顆 ckpt 沒有 lo 段）。
一手：`logs/M2-vqsel-ev-26313.out`、`results/m2_vqsel_gates.json`。

**stdout 原話三行**：

```
  分段 conf2（官方協定）: success 56/250 = 0.224
  ⭐ per-leg [official conf2]: legs attempted 10281 / legs reached 615 / reach rate 0.060
     / median steps per leg 16   (集末未結算 250 段；到達半徑 rho 1.875、路標目標間距 DELTA_SUB 7.5)
  🔮 vq_select 選字統計：chunks 56137 / 有被選到的字 32/32 / perplexity 24.49 / top1 10.0%
```

**四關全表**（分析器＝`walk_verify/p2_analyze_traces.py`，跟 vq_oracle 同一支 code、同一把尺）：

```
關                                 M2 vq_select      對照
──────────────────────────────────────────────────────────────────────────────────
① per-leg 不限時到達率（官方口徑）    0.060           貪心 .063 ／ lo 頭 .124 ／ 主頭 .320 ／ 教材 .554
① per-leg 不限時（trace 重建口徑）    0.095           貪心 .111（同口徑；⚠️ 兩把尺定義不同，見 §六#5）
① per-leg 限時 1.25N ／ 1.5N        8.0% ／ 8.2%    貪心 9.0% ／ 9.0%
① 全集 goal 限時 1.25N ／ 1.5N      1.6% ／ 4.4%    貪心 0.0% ／ 0.0%   ⚠️ N_table 對全集過嚴
   median steps per leg             16              貪心 24 ／ 主頭 16 ／ lo 頭 20
② 步速 p50                          0.0515          真螞蟻 0.1325 ⇒ 39% 🚨（貪心 0.0490＝37%）
   步速 p25 ／ p75                   0.0014 / 0.0990 真螞蟻 0.0859 / 0.1752
③ 翻倒率（逐步／逐集）               0.06% ／ 36.0%  真資料逐步線 1.00% ⇒ 逐步過 ✅，逐集偏高 ⚠️
③ 平滑度 |a_t−a_(t-1)| p50          1.012           真螞蟻 1.768 ⇒ 更平滑、非抽搐 ✅
   選字使用率                        32/32、perplexity 24.49/32、top1 10.0%（教材 31.10 / 4.9%）
④ 人眼關                            gif 6 支（3 成功 3 失敗），主人親判
   疊圖：results/m2_vqsel_{speed,z,adiff}_overlay.png
```

**全鏈成功率參考錶（本次同一支 job 的七個 arm，250 集）**：

```
誠實 BC 地板 .512 ／ u 歸零 .116 ／ 別人的 u .084 ／ LaCoT R=0 .236
intent_swap .236 ／ intent_noise .240 ／ 分段 conf2 (vq_select) .224
對照（job 25987 同一顆計畫棧）：BC .504 ／ 分段 conf2 主頭 .448 ／ (job 25776) lo 頭 .288
對照（job 26202 貪心）：分段 conf2 .100
```

**gif**（`results/m2_gifs/`，每支兩格並排：左＝跟拍看步態、右＝全迷宮固定機位看走到哪；
每 4 步 1 幀、上限 200 幀、0.08 秒/幀；最大 4.3 MB、六支合計 21 MB）：

```
success_trace_t1_s0.gif (109幀)  success_trace_t2_s1.gif (152幀)  success_trace_t3_s26.gif (200幀)
fail_trace_t1_s1.gif    (200幀)  fail_trace_t2_s0.gif    (200幀)  fail_trace_t3_s0.gif    (200幀)
```

---

## 六、判讀：學得會寫，但螞蟻會蹲下來卡死

判準：**⛔ 沒過**。per-leg .060 < 守門線 .124，跟它要打敗的貪心 .063 打平。

但 ⛔ **不要把它讀成「寫譜頭學不會寫」** —— 那跟 §四的 held-out 46.07% 直接矛盾。
下面三個假說**各被一個量測推翻**，第四個才是留下來的（⭐ 先定位再開藥）：

### 6.1 假說 A：eval 的路標超出訓練分佈 ⇒ **推翻**

先量到訓練分佈確實沒蓋滿：eval 實際看到的 |路標−位置|（56,137 chunk）
p50 = 12.0、**68.7% 大於訓練上限 7.5**（conf2 的 direct 分支直指終點、不受
`SUB_MAX_ARC` 限制）。看起來像鐵證 —— 但按 leg 起始距離分箱之後：

```
d0 區間      legs   到達率        d0 區間      legs   到達率
[0, 3)       516    0.579        [7.5, 10)    598    0.005
[3, 5)      1034    0.013 ⭐內    [10, 15)     311    0.019
[5, 7.5)    1131    0.007 ⭐內    [20, ∞)      221    0.032
訓練分佈內（[1.875,7.5]）到達率 0.011  vs  分佈外（>7.5）0.014
```

⇒ **分佈內跟分佈外一樣爛**（唯一會到的 [0,3) 是「本來就快到了」，中位只花 1 個 chunk）。
⛔ 拓寬訓練弧長窗治不了這個病 —— 這一格省下了一輪 3 小時的重訓。

### 6.2 假說 B：argmax 把譜壓平了 ⇒ **推翻**

eval 實走的字串「連續 chunk 同字率 **30.9%**」，而教材真字串只有 **2.8%**
（官方 val 500 集、25,000 chunk 實測）—— 真步態幾乎每個 chunk 都換字。
看起來就是 argmax 的 mode-seeking。但把頭放回**資料狀態**上重量
（`probe_string.py`、同 500 集）：

```
                        連續同字率   perplexity   top1
教材真字串（上界）          2.8%      31.10/32    4.9%
寫譜頭 argmax（eval 用的）  2.4%      30.27/32    5.8%
寫譜頭 依 p 抽樣            2.8%      31.20/32    4.9%
argmax 對教材的 top-1 一致率 45.2%
```

⇒ 在資料狀態上 **argmax 寫出來的譜跟真譜統計上分不出來**。⛔ 不是解碼法的錯。

### 6.3 假說 C：P2 §3.5 的「相位連續性被打斷」 ⇒ **反過來了**

P2 推論貪心之所以崩，是因為「每 4 步換一種步法、互相打斷」。
本次直接量到：**真步態字串本來就 97.2% 的時候在換字**。
⇒ ⛔ 「一直換字」不是病；P2 §3.5 那段推論（它自己標了「是推論不是量測」）**方向錯了**。

### 6.4 留下來的：閉環的蹲姿吸子（有量測）

```
⭐ 凍結 chunk（4 步位移 < 0.02 m）：29.4% 的 chunk（每集平均 26.4%）
   凍結時 軀幹 z p50 = 0.267   非凍結 0.549   真螞蟻 0.552（翻倒線 0.2573）
   z 落在 [0.26, 0.28] 窄帶的 chunk 佔 28.6%，其中 93.7% 是凍結的
   凍結時連續同字率 94.6%   非凍結 4.5%（⇒ 跟教材的 2.8% 同一量級）
   拿掉凍結之後的步速 p50 = 0.0588/步（真螞蟻 0.1325 ⇒ 仍只有 44%）
朝路標的淨進展中位 +0.0000／chunk、位移方向與路標方向 cos>0 只有 52.9%（亂走＝50%）
```

⇒ **機制**：螞蟻常常掉進一個 z≈0.267 的**蹲姿**，那個姿勢在資料裡不存在
（真螞蟻 z p50 0.552）⇒ 頭進了沒見過的狀態 ⇒ 反覆吐同一個字（94.6%）⇒ 出不來。
整個「30.9% 同字率」完全由這 29.4% 的凍結 chunk 產生 —— 會動的時候，它寫的譜是正常的。
這是**閉環 covariate shift 的絕對吸子**，⛔ 不是選字能力的問題。
⚠️ 而且拿掉凍結之後步速仍只有真螞蟻的 44%，⇒ 蹲姿吸子**不是唯一**的缺口。

**證據鏈總結**：§四 held-out 46.07%（8.98×亂猜）＋ §6.2 字串統計跟真譜同級
⇒ 頭會寫；§五 per-leg .060、步速 39%、cos>0 52.9% ⇒ 走不到；
§6.1／6.2／6.3 三個推翻 ＋ §6.4 的凍結量測 ⇒ 卡在閉環，不是卡在選字。

---

## 七、沒做到 / 不確定清單

1. **⭐ 最該做而沒做的**：治蹲姿吸子的那帖藥**一帖都沒試**。工單範圍是「訓 M2 ＋ 量」，
   ⛔ 不含開藥；而任何一帖（DAgger／在頭自己走出來的狀態上重標、把「站起來」放進
   教材、字串層加時間結構、或直接在 z 落到窄帶時換策略）都要再一輪訓練＋eval，
   ⚠️ 沒有主人點頭我不動。⇒ **這是下一步最高價值的一格。**
2. **教材上界 .554 跟本次 .060 不是嚴格同分佈**（沿用 teacher-relay §五#1 的註記）：
   .554 的 leg 來自真實錄製軌跡自己的路徑（595 個、保證可走），本次的 leg 來自
   live rollout 裡 planner 對整個迷宮生成的 subgoal（10,281 個）。方向上的結論站得住，
   ⛔ 但別把 .554 當成「這顆頭差多少」的精確缺口。⭐ 相對地，
   **.063／.124／.320 這三個對照是同協定同 250 集**，跟 .060 直接可比。
3. **單一 seed、單次 eval**。既知抖動尺 ±.03~.07（官方 `rollout()` 沒釘 global np.random；
   本次自己又量到一次：權重逐位元相同的兩顆，n=10 的 BC 地板跑出 .500 vs .600）。
   ⇒ ⛔ .060 vs .063 之間的差**不能**當成有意義的差；⭐ 但 .060 vs 守門線 .124
   是 2 倍，超出抖動尺，這個判負站得住。
4. **只有 1 個字典 seed**（10432，沿用 P0/P1/P2 同一份 ckpt）—— 跟 walk-verify §五#3
   同一個開放項，沒量 run-to-run variance。
5. **兩把 per-leg 尺並存**（.060 內建 `LEG_STATS` ＝官方口徑「這一段是不是【因為到了】才結束」；
   .095 是我從 trace 重建的「這一段期間有沒有進過 rho」）。⛔ 不保證逐段對齊，
   兩個都印，**內建那行是官方口徑**。同 P2 §五#6。
6. **gif 是人眼關的素材、不是判決**。我實際抽幀看過兩支（⛔ **沒有逐幀播完 6 支**）：
   - `success_trace_t1_s0`（第 5/50/100 幀）：螞蟻在跟拍格裡看得到，四腳著地、軀幹低伏著在動；
     右格顯示它從左下一路走到右上的橘點 ⇒ 跟「成功」對得上。
   - `fail_trace_t2_s0`（第 5/60/120/197 幀）：第 5 幀螞蟻攤在地上；
     ⚠️ **第 60/120/197 幀跟拍格裡看不到螞蟻本體**（右格的十字仍在迷宮裡漂、沒收斂到橘點）。
     ⛔ 我沒有查出跟拍鏡頭為什麼會漏掉它（相機在牆內／螞蟻在視錐外都有可能）——
     這是**渲染端的疑點，不是量測端的**（四關的數字全部由 trace 重跑算出，不經過 render）。
   ⇒ 主人自己看過 6 支才算第四關過。
7. **「全集 goal 限時」那格照舊要打折**：N_table 用直線距離查表、迷宮要繞路 ⇒ 系統性過嚴
   （walk-verify §五#12 同一個成因）。本次 1.6%/4.4% ⛔ 不能單獨拿來當結論。
8. **§6.4 的「頭進了沒見過的狀態所以出不來」是機制推論**；量到的是「凍結 29.4%、
   凍結時 z=0.267、凍結時同字率 94.6%」這三個**事實**。⛔ 我沒有直接證明
   「z≈0.267 這個姿勢在訓練資料裡不存在」（只比了 z 的中位數 0.267 vs 0.552），
   也沒量「從蹲姿出發時頭的 logit 熵有沒有塌」。這兩格補起來才算閉合。
9. **沒有跑「亂選字」在 live conf2 協定下的對照組**。P1 §2.3 的教訓是「沒有打假球對照
   那張表不能用」；⭐ 這次沒補，理由是**這個協定上的尺已經證明有鑑別力**
   （同協定同 250 集：貪心 .063 → 主頭 .320，5 倍展開）。⚠️ 但這是我的判斷，不是量測。
10. **抽樣解碼（temperature / 依 p 抽）只在資料狀態上量過字串統計（§6.2），
    ⛔ 沒有真的跑一次 rollout。** 依 §6.2 它跟 argmax 幾乎同分佈，預期改變不大 ——
    ⚠️ 但那是預期，不是量測。
11. **⛔ 沒動的（工單範圍外，照約定停手）**：G16K16 字典、beam / 多 chunk 前瞻、GRPO。
12. 沒做到的「查不到／跑不出」項目：**無**。goldab、ant 旗關零差、污染對照、
    權重逐位元對照、100k 訓練、held-out 探針、250 集 eval、四關表、字串探針、
    凍結診斷、6 支 gif 全部跑完並留下一手輸出。

---

## 八、復現

```bash
git -C ~/Projects/lacot worktree add ~/Projects/lacot-m2 vq-select-v1
cd ~/Projects/lacot-m2
sbatch experiments/m2_vqselect/verify_ab.sbatch        # goldab ＋ ant 旗關零差
sbatch experiments/m2_vqselect/train_vqselect.sbatch   # 100k 步，約 2 小時
sbatch experiments/m2_vqselect/eval_vqselect.sbatch    # ep50，約 1 小時
bash   experiments/m2_vqselect/analyze.sh              # 四關表＋6 gif（CPU）
python experiments/m2_vqselect/probe_headout.py --ckpt <ckpt>   # held-out top-1
python experiments/m2_vqselect/probe_string.py  --ckpt <ckpt>   # 字串統計
```
