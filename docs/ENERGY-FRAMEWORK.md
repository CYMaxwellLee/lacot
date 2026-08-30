# Energy 框架（主人 2026-08-29 裁定的定位）

_一句話：LaCoT 的 test-time 修正＝「在合成 energy 地形上引導生成」；NF 只是引擎選擇。_
_主人 8/29 裁：方向核准 ——「如果前置實驗順利的話，可以準備往這方向，慢慢來規劃」。_
_⇒ ⛔ 不搶跑：先把目前的前置（G3 對照、分段、信心選點）跑順，再按下節觸發條件規劃。_

## 框架句

```
E_total(u) = E_geo(u) − λ·log p(u | s,g)
u ← u − η·∇E_total        （實作上兩項各自 normalize 再相加 —— 主人的 trust region）
```

- `E_geo`：幾何 energy（撞牆×10、終點×3、起點×3、路長×0.3，越小越好）。
  【算的不是學的】⇒ 沒有可被優化器鑽的破綻。實作在 `lacot/refine_grad.py` GeoEnergy。
- `log p`：生成引擎的密度（結界）。其梯度 ∇log p 在文獻裡就叫 **score**。
- 家族定位：energy-guided sampling / classifier guidance / Diffuser 系 guided planning。
  差異點：guidance 加在【軌跡 latent】上，且 E_geo 幾何可驗。

## 為什麼叫 energy 不叫 value（主人 8/29 的三個理由）

1. 不跟 RL 的 value（expected return、從資料學）混淆 —— reviewer 不會追問「value 怎麼訓的」。
2. 聽起來（也真的）更接近它的文獻家族。
3. energy 框架自然接 score-based generative 一族 ⇒ 引擎未必永遠是 NF。

## score-based 引擎的替換路（沒立刻要做，先立此存照）

| 零件 | 現在（NF） | 換成 score-based 後 |
|------|-----------|---------------------|
| sample | 一步過可逆橋 | 反向擴散（多步） |
| 結界 ∇log p | NF 精確 log p 的梯度 | score 網路 s(u,t) 直接給（近似） |
| 爬坡 | 生成後另跑 η 步 | 併進每個去噪步（guided diffusion 標準形）|
| refine 與生成的關係 | 兩個階段 | **合流**：生成本身就是一連串 refine |

⭐ 8/24 的疑慮「refine 改去噪會跟 flow 重疊」在 score 框架下是 feature：整個生成即 refine。

## 數學身分：更新式＝乘積分布的 score（主人 8/29 追問後補）

真正想取樣的分布是「引擎分布 × 幾何 Boltzmann 因子」的乘積：

```
p_guided(u) ∝ p(u|s,g) · exp(−E_geo(u)/λ′)
∇log p_guided = ∇log p − ∇E_geo/λ′          ← 分布相乘 ⇒ score 相加
```

對照更新式 `u ← u + η(−clip(∇E) + λ·clip(∇log p))`：方向就是乘積分布的 score。
⇒ 爬坡不是 heuristic —— 它是在 p_guided 上找眾數（Langevin 拿掉噪聲項的 MAP 版）。
⇒ 這個等式跟引擎無關（NF / score model 都成立）＝引擎可換的數學保證。
⚠️ 誠實註記：per-term clip（trust region）讓實作偏離嚴格的 score 方向 —— 那是刻意的
（兩項量級差兩個數量級，8/26 主人裁），寫 paper 時照實講。

## 統一框架：整套方法＝對 energy-tilted 分布的迭代 amortization（主人 8/30 裁定）

_主人 8/30：「把 latent reasoning 與 verifier 的最新 insights 拿來用，但要避免拼裝車 —
在數學上精巧融合」。⇒ 下面一個式子統攝全部零件；上一節（乘積分布 score）是它的梯度視角特例。_

**目標分布（全法的中心對象）**：

```
p*(u|s,g) ∝ p_θ(u|s,g) · exp(−β·E(u; s,g))
```

flow 先驗 × energy Boltzmann 傾斜。每個零件都是對 p* 的一個運算元：

| 零件 | 數學身分 |
|---|---|
| select／BoN（抽 M 挑 E 低） | 對 p* 的 self-normalized importance sampling（M→∞ 收斂到 p*） |
| exp(−E) 加權蒸餾 | min KL(p*‖p_θ) 的蒙特卡羅梯度（M-step，⛔ 不是外掛 trick） |
| expert iteration 自舉 | 反覆 amortize：p_θ ← p*；分布沿 energy 地景逐輪下降（ReST-EM 的 EM 視角） |
| ebfs teacher 蒸餾（P1b） | β→∞ 極限：圖最短路 ≈ argmin E ⇒ **冷啟動與自舉＝同一式的兩個 β 檔位** |
| 課程 | (cond 難度, β) 的退火 schedule（annealed importance sampling 一族） |
| E 逐段打分＋beam search | E 因子化成段位能 E(τ)=Σφ(seg_k) ⇒ p* 成因子圖 ⇒ 逐段剪枝＝因子圖近似推斷 |
| R／深度隨機化 | 對計算深度的先驗（邊際化 compute；Poisson 家族） |
| 爬坡（已判死的那條） | 對 p* 找眾數（MAP）— 判死的是「用梯度找眾數」這個**採樣器**，⛔ 不是 p* 本身 |

**血統對位（2026-08 參照系；主人 8/30 裁「RLHF 舊了、GRPO 也舊了」後全面換新）**：
1. **MaxRL（2602.02710, ICML 2026）＋ power sampling（2510.14901, ICLR 2026）**＝
   「RL≈tilted likelihood」的正典：前者證明 expected-reward RL 只是 tilted likelihood 的
   一階近似、後者不訓練直接 MCMC 抽 p^α 就追平 RL post-training。我們＝把同一個 tilted
   分布在訓練時**迭代 amortize** 掉的 exact 版；且 tilt 來自外部 verifier E 而非自身
   likelihood — 恰好躲開「純 sharpening 不穩」（2604.16259）。
2. **On-policy distillation 原語（MOPD 2606.30406、SDPO 2601.20802；DeepSeek-V4／
   Kimi K3／GLM-5 的工業 default）**：2026 年「RL 出 teacher → 蒸餾回 student」已是一級
   原語。我們的 M-step＝它的 verifier 版 — teacher 不是另一顆網路，是「自己被 exp(−βE)
   tilt 過的分布」（BOND→OPD 血脈的下一步）。
3. **Tilt Matching 家族＋SOC 統一觀點（2512.21829、2512.03234、2605.00229）**：同一個
   目標 p·exp(−βE)，他們在無 exact likelihood 的 diffusion/interpolant 上只能走速度場／
   SOC 近似；我們的 exact-likelihood NF 讓 tilted 權重逐樣本解析精確 — test-time select
   與訓練蒸餾用**同一組權重**。
4. **正確性血統**：Boltzmann generators（NF exact-IS reweight 二十年標準作業）— 我們的
   權重正確性宣稱直接接這條正統；差異＝E 是 verifier 定義＋迭代自舉＋planner conditioning。

**撞題檢查（2026-08 調研）**：「exact-likelihood NF＋exact tilt 蒸餾＋verifier 閉環」五個
方向查遍仍空著。四鄰居點名：Boltzmann（無 verifier/自舉）、DISA（AR LLM、離線凍 Z）、
Tilt Matching（velocity 近似、無 exact weight）、NF-CoT 2606.06447（TARFlow 進 LLM 但走
policy gradient）。機器人 BoN 線（CoVer/EVE）全停在 test-time selection、無人蒸餾閉環。
⛔ 措辭：別喊「免 clip」（會與丟 ratio 的 GPG 族混淆）— 喊「IS 權重解析精確；clip-free
是 exact likelihood 的**推論**，不是手工選擇」（以 CTPO 2605.07331 承認 exact sequence IS
在 LLM 不可行當反襯）。

**NF 的結構性優勢（「為什麼不用 diffusion」的第二個理由）**：p_θ 是 exact likelihood ⇒
tilted 分布的重要性權重 w = exp(−βE) **精確**、零密度估計誤差；diffusion 只有 ELBO ⇒
tilted sampling 必然近似（LaDiR 族做不到 exact tilt）。第一個理由（exact NLL 當訓練目標）
8/19 已錄。

**自舉的外部錨（2026 self-play 文獻的教訓）**：R-Zero/Absolute Zero 一族的已知病＝
pseudo-label 噪音＋多樣性幻覺（R-Diverse 2602.13103 實錘），公認需要外部錨 —
我們的非參數 E 正是那個錨；此為 E-verified 自舉相對純 self-play 的結構優勢。

**誠實邊界**：E 是真成功率的 proxy（8/30 verifier 調研的定論）⇒ p* 的「好」上限在
proxy gap；封縫三件套（fuzz／同構不變性／held-out rollout）與 pass@M 警報是這個框架的
配套量測，⛔ 不是可選項。β 的 schedule 是新超參（退火太快＝Goodhart 提早進場）。

## NF ＋ score 一起用的三種真組合（主人 8/29 問「數學上有什麼好處」）

1. **score 疊加性**（上節）—— energy 可加 ↔ 分布可乘，guidance 是精確操作不是近似。
2. **互補組合**：diffusion 的 log p 難算（要沿 probability-flow ODE 積分）、NF 的 log p
   一次前向且精確 ⇒ score model 當生成器（表達力、抗塌）、NF 當守門員（精確結界）。
3. **統一形式＝flow matching**：把連續 NF 與 diffusion 統一 —— 訓練是穩定回歸
   （不碰 NLL 的 Jacobian）、取樣是 ODE、密度仍可估。NF 的兩個實測痛點
   （NLL 不穩、塌成點質量）它都對症；「refine 改去噪會跟 flow 重疊」的疑慮在此變成
   feature：生成本身就是一串 refine。主人 8/18 的 flow-matching 直覺在 energy 框架下歸位。

## 什麼時候值得換（觸發條件，都是實測痛點）

- NF 的 cond 泛化偏：tier2 探針量到 flow 抽的計畫終點偏 9 倍且 log p 無自覺
  （`results/refineprobe_*_tasks-t2_*.json`）⇒ 爬坡目前救得回；救不完就是換引擎的訊號。
- NF 的多樣性塌：D4 線索（recon 目標下塌成點質量）。score/diffusion 的 mode coverage 是強項。
- 成本形狀不變：我們本來就在迭代修 u ⇒ 多步生成不是新增成本。

## 方言病與離散化階梯（主人 2026-08-29 晚定向）

病：爬坡沿 decoder 靈敏方向把 u 推出 encoder 流形薄殼（log p 沒掉不代表在殼上），
head 對殼外輸入行為未定義 ⇒「decoder 視角的改善」不轉化成成功率。
證據：noclimb+fin 0.580 > climb+fin 0.520；R 線 0.460→0.360；D0 弱；shuf 崩 0.056。

三層階梯（由軟到硬，逐層驗證後再上）：

1. **投影回殼（軟）**：`[實測 8/29 夜]` ❌ 失敗 —— climb+prj+fin 0.430 < climb+fin 0.520。
   ⚠️ 打掉的是「enc∘dec＝乾淨投影」這個實作（往返有損＋encoder 對合成路有域差 0.64），
   ⛔ 不是方言假說本身：selection 0.600 > noclimb 0.580 > climb 0.520 仍完整支持
   「殼上的點就是好、手續越多越傷」。⇒ v2 立項證據降一級，硬錨定論證仍活。
   姊妹招 selection ✅ 成功：抽 N=8 挑 E 最低＝flat 家族新王 0.600（`LACOT_GRAD_MODE=select`）。
2. **VQ 錨定（中）**：encoder 出口加 codebook，連續爬坡＋週期 quantize（Diffusion-LM/CDCD/
   bit-diffusion 的「操作連續、表示離散」哲學）。flow 與 E 全保留、引擎不換，
   只重訓 encoder(VQ)+head+decoder。⭐ v2 首選候補 —— prj 若有效，此路只會更穩。
3. **全 categorical（硬）**：DreamerV2 式 one-hot 組合，先驗換 discrete 系
   （AR / discrete diffusion），refine 變離散搜索（座標搜索/退火、引導生成、遮罩重填 ——
   第三種與分段 thinking 敘事同構）。⚠️ 8/22 淘汰的是 action 離散化，⛔ 與 latent 離散化無關。

## Stitch 配方：組合知識放在哪（主人 8/29 晚之問）

TT/AR 把 stitch 全押在「序列先驗泛化出沒見過的長序列」（結構性難）。我們拆三份，
離散化後每份都在：
1. **字彙從短段學**：code/u 學「一小段路的幾何模式」，每條短軌跡都是合法教材。
2. **組合合法性由 E 裁決**：佔據圖＝兩萬條短軌跡的【聯集】，全域組合知識在這張地圖裡、
   不在任何序列模型裡。離散版＝「先驗提議、幾何裁決」。
3. **組合負擔小**：只組合 K=4 個 code 的計畫摘要（搜索可及），不是千 token 長序列；
   分段機制讓短程多數時刻就在分布內。
保險絲：BFS 資料引擎（訓練時造跨軌跡長配對＋合成參考路）—— 密度病治本，離散版更好造。
一句話：有限的字、一張全域地圖、只寫四個字的短句。
