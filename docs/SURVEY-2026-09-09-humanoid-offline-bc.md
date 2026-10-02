# 文獻調研：Offline BC/Imitation 練 Humanoid 行走 —— 前人配方與可搬技巧

_2026-09-09。範圍：純 offline（訓練期不與環境互動）設定下，前人如何用 BC / imitation
類方法把 humanoid（雙足）練到能走。為 LaCoT 的 OGBench humanoidmaze 管線服務
（84 維動作段、21 關節；ant 四足「步法字典＋整段開環播放」可行 .554 per-leg 到達率，
humanoid 同款崩掉：到達率只有 ant 的 5~22%、摔倒率 75~100%；現轉向「u 一次生成整段
動作字串（開環）＋ Best-of-N 挑譜」，訓練全 offline BC/SFT）。_

**標記說明**：`[paper 明講]` = 論文原文直接陳述（附出處）；`[推論]` = 調研者自己的判斷／
外推，不是論文寫的；`[查不到]` = 認真找過但沒找到，不用沉默帶過。

---

## 軸① OGBench 直接對標

_我們自己的對標線，最重要，本節由 Luna 親自查證（未外包）。_

### 1. OGBench: Benchmarking Offline Goal-Conditioned RL

- **arXiv**: [2410.20092](https://arxiv.org/abs/2410.20092)（已用 WebFetch 打開確認標題／作者／摘要）
- **作者**：Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine；ICLR 2025。
- **一句話**：離線目標條件 RL 的標準化 benchmark，8 類環境、85 個資料集、6 個參考演算法實作
  （GCBC / GCIVL / GCIQL / QRL / CRL / HIQL）。`[paper 明講]`
- **訓練訊號**：對這 6 個被評測的演算法而言是純 offline —— 固定資料集、訓練期不碰環境。
  `[paper 明講]`（Section 8.1：「GCBC is the simplest goal-conditioned behavioral cloning
  method」，標準 offline BC）。⚠️ 但資料集本身的生成用了「low-level directional policy
  trained via SAC」`[paper 明講, §7.1]` —— 也就是產生軌跡的底層 locomotion 技能是用 online
  RL（SAC）練出來的，只是這件事發生在資料集建置階段、下游被評測的演算法沒有碰環境。
  `[推論]`：這代表 OGBench 的資料集本身已經內含一個「會走路」的底層 controller 產生的
  軌跡，不是從零散的隨機資料硬拼出走路動作 —— 我們自己的資料是否有同等品質的底層
  locomotion 訊號，會直接影響 GCBC 類方法的地板。
- **HumanoidMaze 環境**：21-DoF humanoid，三種尺寸（medium/large/giant），三種資料變體
  （navigate=noisy expert、stitch=需要拼接的短軌跡、explore=隨機探索）。`[paper 明講, §7.1]`
- **GCBC 網路架構**（來源：官方 repo `impls/agents/gcbc.py`，已用 WebFetch 打開確認）：
  3 層 MLP（512, 512, 512）、goal 用 concat encoder 直接串接、**逐步單步輸出動作
  （無 chunking）**、連續動作用 Gaussian NLL loss。`[paper 明講]`
  —— 這點很關鍵：OGBench 的 GCBC baseline 是純逐步閉環，跟我們原本「步法字典逐步閉環選字」
  同構，也跟我們正在轉向的「整段開環生成」形成對照組。
- **GCBC 在 humanoidmaze 上的成績**（Table 2，標準資料集規模）：
  `[paper 明講]`

  ```
  humanoidmaze-medium-navigate-v0   8 ± 2%
  humanoidmaze-large-navigate-v0    1 ± 0%
  humanoidmaze-giant-navigate-v0    0 ± 0%
  humanoidmaze-medium-stitch-v0    29 ± 5%
  humanoidmaze-large-stitch-v0      6 ± 3%
  humanoidmaze-giant-stitch-v0      0 ± 0%
  ```

  對照 GCBC 在 antmaze 上：medium-navigate 29±4%、large-navigate 24±2%
  `[paper 明講, Table 2]` —— 同一個演算法，四足 antmaze 中等地圖能到 29%，雙足
  humanoidmaze 中等地圖只有 8%，giant 直接掉到 0。
  對照當前最強 baseline HIQL 在 humanoidmaze：medium-navigate 89±2%、large-navigate
  49±4%、giant-navigate 12±4% `[paper 明講, Table 2]` —— 代表同樣的標準資料規模下，
  換成階層式方法可以把 medium 從 8% 拉到 89%，headroom 巨大，不是資料不夠而是
  flat/單步方法的結構性問題。
- **論文有沒有解釋為什麼 humanoidmaze 比 antmaze 難**：`[查不到]` —— OGBench 論文正文
  沒有針對「為什麼 GCBC 在 humanoidmaze 特別差」給出機制層級的解釋，只泛泛提到
  high-dimensional control。這代表我們自己「雙足連續執行姿態漂移放大」的診斷是
  **我們自己的推論**，OGBench 論文本身既沒有證實也沒有反駁它。

### 2. Horizon Reduction Makes RL Scalable

- **arXiv**: [2506.04168](https://arxiv.org/abs/2506.04168)（已用 WebFetch 打開確認）
- **作者**：Seohong Park, Kevin Frans, Deepinder Mann, Benjamin Eysenbach, Aviral Kumar,
  Sergey Levine；NeurIPS 2025。跟 OGBench 同一個第一作者脈絡的後續工作。
- **一句話**：研究 offline RL 演算法的資料規模擴展性，發現「horizon 長度」是擴展性的
  主要瓶頸，提出 SHARSA（階層式 subgoal + n-step）來降低有效 horizon。`[paper 明講]`
- **訓練訊號**：純 offline。`[paper 明講]`：「Offline RL aims to train a
  reward-maximizing policy from a static dataset without online interactions」；
  評測的是 offline goal-conditioned 設定，「purely from binary sparse rewards and an
  unlabeled (reward-free) dataset」。
- **⚠️ 資料規模不是同一件事**：這篇用的資料集比 OGBench 預設釋出的規模大到 1000 倍
  （最多到 1B transitions）`[paper 明講]` —— **這裡的數字不能直接跟上面 OGBench
  Table 2 的標準規模數字並排比較**，是完全不同的資料量級。`[推論]`
- **humanoidmaze-giant 成績**（Appendix D Table 1，1B-transition 規模）：
  SAC+BC baseline 44±35% vs SHARSA 87±3%。`[paper 明講]`
  `[推論]`：baseline 的 ±35% 標準差非常大，很像是「有時候學會走、有時候整組垮掉」的
  雙峰分佈（bimodal），跟我們自己觀察到的「humanoid 摔倒率 75~100%」這種近乎全有全無
  的失敗模式在性質上相似 —— 但這是我們的類比推論，論文沒有明講這個變異數的成因。
- **為什麼 flat/單步策略在 humanoidmaze 失敗**：`[paper 明講]`「the mapping between the
  optimal actions and distant goals can be highly complex as it depends on the entire
  topology of the state space」；對 value-based 方法而言「biases (errors) in the target
  Q values accumulate over the horizon」。
- **SHARSA 機制**：階層式 —— high-level policy 輸出 subgoal，low-level policy 在給定
  subgoal 下輸出動作，用 n-step returns 訓練（時間上的分段/chunking 發生在 value
  learning 層級，不是逐 token 的動作序列生成）。`[paper 明講]`
- **對我們的可搬性**：`[推論]` 這篇的核心論點——flat/單步策略在長 horizon 的
  humanoidmaze 上結構性地失敗（因為 state→遠距 goal 的映射太複雜／誤差沿 horizon
  累積），而時間分段/階層化能解決——是對我們「從逐步閉環選字改成整段開環生成」這個
  轉向方向的強力側面佐證（同屬「降低有效決策 horizon」這個大類手段）。但 SHARSA
  本身是 value-based（SARSA/TD）階層式 RL，不是 BC/SFT 字典方法，具體演算法不能直接
  搬，能搬的是「壓縮決策 horizon」這個設計原則本身，不是 SHARSA 的實作細節。

### 3. HIQL: Offline Goal-Conditioned RL with Latent States as Actions

- **arXiv**: [2307.11949](https://arxiv.org/abs/2307.11949)（已用 WebFetch 打開確認；NeurIPS 2023）
- **作者**：Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine。
- **一句話**：階層式 offline GCRL —— high-level policy 把「未來狀態」當成動作輸出
  （預測一個 latent subgoal 表示），low-level policy 學習抵達這個 subgoal。這就是上面
  OGBench Table 2 裡把 humanoidmaze-medium-navigate 從 GCBC 的 8% 拉到 89% 的方法。
  `[paper 明講]`
- **訓練訊號**：純 offline，「learn directly from diverse offline data」`[paper 明講]`。
  low-level policy 用 **advantage-weighted regression**（不是純 BC，是用 IQL 式 value
  function 算出的 advantage 對 BC loss 加權）：

  ```
  J_πℓ(θℓ) = E[exp(β·Ã(st, at, st+k)) · log πθℓ(at | st, st+k)]
  ```

  `[paper 明講，原文附公式]`。Subgoal 用學到的 latent 表示、非固定步數的原始狀態
  `[paper 明講]`；具體 k 值論文本文沒有明確給單一定值 `[查不到]`。
- **⚠️ 出處澄清（容易搞混，特此分清楚）**：89±2% / 49±4% / 12±4% 這組 humanoidmaze
  數字來自**OGBench 論文**（拿 HIQL 演算法去跑 OGBench 新環境當 baseline），**不是**
  HIQL 原始論文自己的實驗結果 —— HIQL 原論文（2023）評測環境只有 AntMaze / Kitchen /
  CALVIN / Procgen Maze / Visual AntMaze / Roboverse `[paper 明講，原文環境清單]`，
  **不含任何 humanoid/雙足環境**（2023 年時 humanoidmaze 這個環境還沒被提出）。
- **對我們的可搬性**：`[推論]` 可以拆成兩件獨立的事——
  (a) low-level policy「條件在近距離 subgoal、不是條件在遠距最終目標」這個結構性想法，
  理論上可以直接搬進純 BC/SFT：不需要 value function、不需要 advantage 加權，只要用
  hindsight relabeling 把訓練軌跡未來第 k 步的狀態當成訓練時的「目標」來條件化。
  (b) advantage-weighted regression 這個具體加權機制需要額外訓練一個 value function，
  不是純 BC —— 這部分如果要完整複製會偏離我們現在「純 BC/SFT」的路線，是可選的加分項，
  不是必要品。

---

## 軸② MoCap / tracking 線

_由 sonnet 使魔調研（它自己又平行分派 5 條子查證，5 篇皆用 WebFetch 開過 abs＋html
全文）。這軸最重要的一個訊息：**這 5 篇全是 online RL**（PPO 或 online distillation），
沒有一篇是純 offline——調研的重點因此是「拆解出裡面哪個子機制不需要 online rollout」，
不是找到一篇可以整篇搬的論文。Luna 已檢查標記，1 條列為待抽驗。_

### 1. DeepMimic

- **arXiv**: [1804.02717](https://arxiv.org/abs/1804.02717)（已用 WebFetch 打開 abs＋ar5iv 全文確認；SIGGRAPH/TOG 2018）
- **作者**：Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne。
- **一句話**：PPO 在物理模擬器中模仿 mocap 片段，RSI（reference state initialization）
  ＋ early termination 是關鍵樣本效率技巧，可疊加 task reward。
- **訓練訊號**：`[paper 明講]` 純 online——rollout 由「目前策略」在模擬器中逐步生成，
  PPO on-policy 更新（§6）。
- **humanoid 成績**：`[paper 明講]` Normalized Return：Walk 0.985／Cartwheel 0.804／
  Backflip 0.729／Frontflip 0.485（Table 2/3/4）；Spinkick-strike 成功率 99%（有 task
  reward）vs 19%（純 imitation，100 trials）。無整體摔倒率表。
- **可搬性拆解**：
  - RSI 的**效果**（訓練涵蓋多樣相位起點）`[推論]` 可用離線切 phase-shifted 訓練視窗
    重現；但 RSI 的**原始動作**（在模擬器裡即時重置到某個參考幀）`[paper 明講]` 本質仍是
    online 操作。
  - Early termination 的**效果**（避免在已倒地的壞狀態上浪費訓練訊號）`[推論]` 可用
    離線 kinematic 摔倒分類器（骨盆高度/傾角閾值）事後篩選重現；但作為即時判斷本身
    `[推論]` 脫離持續 rollout 沒有意義，不可搬。
  - root-relative + phase 狀態表徵`[paper 明講, §5.1]`是純特徵工程，離線可算，但
    `[推論]` 它沒有 future-frame 目標，不解決我們「長 horizon 條件化」的問題。
  - Exploration noise 依賴「噪聲—reward 修正」的線上迴圈`[paper 明講]`，離線 SFT 沒有
    這個修正訊號，`[推論]` 不可搬。
- **給我們的建議**：`[推論]` ①訓練集切 phase-shifted 視窗做資料增強；②做離線摔倒分類器
  當 Best-of-N 的拒絕規則；③root-relative+phase 表徵可當 tracking-error 評分特徵。

### 2. AMP (Adversarial Motion Priors)

- **arXiv**: [2104.02180](https://arxiv.org/abs/2104.02180)（已用 WebFetch 打開 abs＋ar5iv 全文確認；SIGGRAPH/TOG 2021）
- **作者**：Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa。
- **一句話**：用判別器（discriminator）取代手工 pose-matching reward，跨多段風格化動作
  泛化，疊加 task reward。
- **訓練訊號**：`[paper 明講]` 非純 offline——Algorithm 1 + §6.3：判別器每個 iteration
  同時用固定 reference set 與「目前策略」on-policy rollout 的 replay buffer 共同訓練，
  buffer 必須持續追上策略分佈，否則判別器過擬合舊分佈。
- **humanoid 成績**：`[paper 明講]` Target Heading 0.90±0.01／Target Location
  0.63±0.01／Obstacles 0.27±0.10／Strike 0.73±0.02／Dribble 0.78±0.05（Table 1）；
  Cartwheel pose error 0.067±0.014m（Table 3）。`[查不到]` 量化摔倒率／存活時間。
- **可搬性拆解**：
  - 判別器輸入特徵 Φ（root 速度、local 關節旋轉/速度、end-effector 位置，`[paper 明講,
    §5.3]`）是純運動學特徵，離線可算；least-squares 判別器架構本身`[paper 明講, §5.2]`
    跟訓練框架無關。
  - `[推論]` 不可搬的是判別器與策略的 replay-buffer **共同訓練迴圈**——這是 AMP 能work
    的核心機制，離線固定資料下沒有等價物。
- **給我們的建議**：`[推論]` 把 Φ 特徵 + 判別器架構改造成**離線訓練的靜態 Best-of-N
  realism scorer**（真 mocap vs 我方生成候選池一次性訓練即可，因為只是排序已生成的
  候選、不必追 on-policy 分佈；scorer 可能對未來生成分佈逐漸過時，用定期離線重訓緩解）。
  這條在 5 篇裡機制改造的可行性最高。

### 3. MoCapAct

- **arXiv**: [2208.07363](https://arxiv.org/abs/2208.07363)（已用 WebFetch 打開 abs＋全文 html 確認；NeurIPS 2022 D&B track）
- **作者**：Nolan Wagener, Andrey Kolobov, Felipe Vieira Frujeri, Ricky Loynd, Ching-An
  Cheng, Matthew Hausknecht。
- **一句話**：PPO 訓練大量單片段 expert 後一次性 rollout 出巨大離線資料集，下游
  multi-clip/GPT 動作補全策略只在這個固定資料集上訓練。
- **訓練訊號**：分兩段。資料集建立 `[paper 明講]` 是 online（§4.1 expert PPO 訓練
  「約 50 GPU-年」；§4.2 帶噪聲 expert 多次 rollout 建資料集）。下游訓練 `[paper 明講]`
  是**真 offline**（§5.1.1 multi-clip policy 對已記錄的 reward/advantage 做加權概似，
  不再呼叫模擬器；§5.2 GPT 動作補全模型對固定軌跡做 MSE 回歸監督學習）。
- **humanoid 成績**：`[paper 明講]` Expert normalized reward 0.816（Table 1）；離線
  multi-clip 最佳 RWR 0.688±0.002（約 expert 的 84%，Table 2）；go-to-target 成功率：
  有 low-level policy 96.3±2.8 vs 無 7.5±1.1（Table 3）；⚠️velocity-control 任務反而是
  無 policy baseline 較高，遷移並非全面正向。
- **⚠️ 灰色地帶（要老實講）**：`[推論]` 下游 BC 確實不碰模擬器，但這個能力是作者端
  約 50 GPU-年 online RL 換來的資料——**不能拿它當「純 offline 能從零生出人形控制力」
  的證據**，只能借用它的機制或別人已公開的資料集，這個區別要講清楚不能含糊。
- **給我們的建議**：`[推論]` ①GPT 式固定片段 MSE 回歸配方——**5 篇裡跟我們「開環整段
  生成」結構最像的機制**，最值得直接參考；②RSI 效果同 DeepMimic，可用隨機視窗重現；
  ③0.688/0.816≈84% 可當「我方離線 BC 相對於一個很強 online expert」的粗略期望錨點
  （但錨點本身建立在 expert 已經很強的前提上，不是我們自己資料的保證）。

### 4. PHC (Perpetual Humanoid Control)

- **arXiv**: [2305.06456](https://arxiv.org/abs/2305.06456)（已用 WebFetch 打開 abs＋全文 html 確認；ICCV 2023）
- **作者**：Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, Weipeng Xu。
- **一句話**：漸進神經網路（PMCP）+ hard-negative mining，把 tracking 能力擴展到全
  AMASS 資料集，另加專門的失敗恢復 primitive。
- **訓練訊號**：`[paper 明講]` 純 online——§3.1 PPO 訓練，Algorithm 1 有 live simulation
  迴圈；1536 個 humanoid 平行、約 7 天收集百億筆樣本。
- **humanoid 成績**：`[paper 明講]` AMASS-Train 成功率 98.9%（MPJPE-g 37.5mm）、
  AMASS-Test 96.4%（47.4mm）（Table 1）；帶噪聲影片輸入 88.7%（Table 2）；失敗恢復
  （5 秒內）：跌倒後 95.0%、遠離後 83.7%（Table 4）。成功率定義：關節平均偏離參考
  >0.5m 即判失敗（§4）。
- **可搬性拆解**：
  - 失敗判準本身（>0.5m 偏離或跌倒接觸地面，`[paper 明講, §3.2]`）`[推論]` 是純
    kinematic 距離/幾何準則，不依賴物理模擬，可離線套用在生成候選上評分。
  - RSI 效果`[推論]` 同前可離線視窗重現；AMASS 多片段資料組織`[paper 明講]` 天生離線。
  - `[推論]` 不可搬：PMCP 難度訊號需要「目前策略實際 rollout 成功/失敗」判定
    （`[paper 明講, §3.2]`），架構想法可借、難度算法本身不能照搬；hard-negative
    mining（`[paper 明講, §3.1]` 沿用 UHC）靠「用目前控制器評估整個資料集」找難例，
    是 rollout-based——`[推論]` 或許能用離線 proxy（BC 重建 loss、動作複雜度特徵）
    替代，但論文沒有驗證過這個替代是否有效；失敗恢復 primitive 是獨立 PPO 訓練＋
    動態換目標，`[paper 明講]` 沒有離線等價訓法。
- **給我們的建議**：`[推論]` ①失敗判準改成 Best-of-N 離線評分規則（懲罰路徑上出現類
  跌倒姿態的候選）——⚠️但若評分方式是把候選丟回真實物理模擬器 replay 驗證而非純幾何
  距離，就等於評分階段偷偷引入了模擬器，這條邊界要明確跟主人確認怎麼算「純 offline」；
  ②RSI 同前；③hard-negative 想法可借，但難度分數必須換成離線 proxy 且要先小規模驗證。

### 5. PULSE（補充第 5 篇 —— 反面教材，防我們誤搬）

- **arXiv**: [2310.04582](https://arxiv.org/abs/2310.04582)（已用 WebFetch 打開 abs＋全文
  html 確認，另有 ICLR 官方 proceedings 頁佐證；ICLR 2024 spotlight，PHC 直接後續、同作者群）
- **作者**：Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris
  Kitani, Weipeng Xu。
- **一句話**：把 PHC 專家蒸餾進 VAE 式低維連續動作 latent space，下游任務在此空間裡
  重新用 RL 訓練。
- **訓練訊號**：`[paper 明講]` **非純 offline，且是本篇最重要的發現**——§4.2「Online
  Distillation」：訓練資料由「學生策略自己 rollout 產生的狀態」即時去查詢凍結的 PHC+
  專家，學生分佈隨訓練改變，**不是**對固定 buffer 做行為克隆。
  ⭐ `[推論]`：這代表「蒸餾（distillation）」這種聽起來像監督學習的字眼，**不代表
  自動離線友善**——要具體查訓練資料的狀態分佈是誰產生的，PULSE 是一個清楚的反例。
- **humanoid 成績**：`[paper 明講]` Table 1（AMASS train/test 成功率／誤差）：
  PHC 97.1%／47.5；PHC+ 99.2%／36.1；PULSE 97.1%／54.1。`[查不到]`
  誤差單位本次未逐字重新核對（應為 mm，沿用同表慣例），精確引用前建議再確認。
- **可搬性拆解**：VAE bottleneck 架構（encoder/decoder/learned prior + KL+平滑正則，
  `[paper 明講]`）本身純架構設計，不內在依賴 rollout；動作時間平滑正則化`[paper 明講]`
  可直接加進任何離線 SFT 損失。`[推論]` 不可搬的是蒸餾機制本身（學生 rollout+即時查詢
  專家）——這是核心賣點，離線固定資料下無法預先算好等價物；所有下游任務策略都是在
  凍結 decoder 上疊加 online PPO（`[paper 明講]`），繼承自 PHC+ 的問題是起點本身就要
  一個很貴的 online RL 專家先存在。
- **給我們的建議**：`[推論]` ①VAE bottleneck 或可單獨改造成純 autoencoder，直接在
  我方已生成好的固定動作字串資料集上做 encode/decode（KL+平滑損失，不碰模擬器），換
  一個 Best-of-N 可取樣/插值的緊湊空間——但這是未經驗證的外插，不是論文驗證過的用法；
  ②不要援用它的 student-rollout+query 蒸餾配方本身。

### 軸②小結（跨 5 篇的一致 pattern）

`[推論]` 這 5 篇全是 online RL／online distillation，沒有一篇整篇可搬——但拆開機制看，
有一條線索的信心特別高：**RSI 的「效果」在 4 篇 online-RL 論文（DeepMimic/AMP 隱含/
MoCapAct/PHC）裡都指向同一件可離線重現的事：訓練時用多樣化的相位/時間起點做資料增強**。
判別器/scorer 類機制（AMP）改造成離線 Best-of-N 排序器，是這一軸裡機制改造可行性最高
的一條。PULSE 則是最有價值的警訊：訓練訊號是否純 offline，要看「狀態分佈是誰產生的」，
不能只看方法名字聽起來像不像監督學習。

## 軸③ 動作 chunking / 序列生成式 BC

_由 sonnet 使魔調研，Luna 已檢查來源與標記，2 條最重的結論標為待親自抽驗（見文末抽驗紀錄）。_

### 1. ACT (Action Chunking Transformer)

- **arXiv**: [2304.13705](https://arxiv.org/abs/2304.13705)（使魔已用 WebFetch 打開 abs＋html 全文確認）
- **作者**：Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn；RSS 2023
  （*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*）。
- **一句話**：低成本雙臂遙操作系統 + CVAE-transformer，一次預測長度 k 的未來動作序列。
- **訓練訊號**：純 offline（遙操作收集資料後監督式訓練，全文無 online rollout）。`[paper 明講]`
- **關鍵參數**：chunk size k=100`[paper 明講]`。⚠️ 但**實際執行不是真開環**——ACT 每個
  timestep 都重新推論一次、產生互相重疊的 chunk，再用指數加權做 temporal ensembling
  平滑`[paper 明講]`：也就是雖然一次吐 100 步，作者自己也不敢盲執行 100 步不看資料。
- **locomotion/humanoid 成績**：`[查不到]` 只有操作任務（6 真實雙臂 + 2 模擬），無 locomotion。
- **可搬性**：`[推論]` k=100 可當「單次前向能吃多長 chunk」的量級參照；但 temporal
  ensembling 這個折衷本身是對「整段開環、不 replan」設計的一個警訊——連 ACT 都不敢賭到底。

### 2. Diffusion Policy

- **arXiv**: [2303.04137](https://arxiv.org/abs/2303.04137)（已用 WebFetch 打開 abs＋html v5 確認）
- **作者**：Cheng Chi, Zhenjia Xu, Siyuan Feng 等；RSS 2023。
- **一句話**：條件去噪擴散生成動作序列，receding-horizon 執行。
- **訓練訊號**：純 offline，論文明講「under the behavior cloning formulation」。`[paper 明講]`
- **關鍵參數**：定義觀察窗 To／預測 horizon Tp／實際執行步數 Ta（執行完才重新規劃）；
  ablation 明確指出 **Ta=8 對多數任務最優**，並定性描述這是「temporal consistency vs.
  responsiveness」的 trade-off。`[paper 明講]` 精確 To/Tp 數值：`[查不到]`（兩次 WebFetch
  都沒能在可讀內容裡撈到）。
- **locomotion/humanoid 成績**：`[查不到]` 純操作 benchmark，無 locomotion。
- **可搬性**：`[推論]` 「執行長度存在明確 sweet spot、不是越長越好」是可直接借用的方法論——
  我們該對自己的開環執行段長度做同款 ablation，不是預設「整段」就是最優；DP 連在動態
  不敏感的操作任務都保留 replanning（Ta<Tp），是對「完全一次生成到底」的間接警訊。

### 3. Behavior Transformer (BeT)

- **arXiv**: [2206.11251](https://arxiv.org/abs/2206.11251)（已用 WebFetch 打開確認）
- **作者**：Nur Muhammad Mahi Shafiullah, Zichen Jeff Cui, Ariuntuya Altanzaya, Lerrel Pinto；2022。
- **一句話**：transformer 讀歷史觀察窗，用 k-means 離散化動作 bin + 殘差修正頭逐步預測單一動作。
- **訓練訊號**：純 offline。`[paper 明講]`
- **關鍵參數**：⚠️ BeT **不是**多步開環 chunk 生成器——h 是輸入觀察窗長度（依環境
  2~10），動作離散化用 k-means（bin 數依環境 2~64），執行方式是**逐步 closed-loop**，
  每步重新取樣一個動作。`[paper 明講, Table 4]`
- **locomotion/humanoid 成績**：`[查不到]` 只測 point-mass／CARLA／block-push／kitchen，無 locomotion。
- **可搬性**：`[推論]` 離散化跟「chunk 開環生成」是兩條獨立軸線的證據之一——BeT 只做
  離散化沒做開環 chunk，是 4 篇裡跟我們結構最不吻合的一篇，參考價值主要在殘差修正頭這個小技巧。

### 4. VQ-BeT ⭐（deep dive，對我們結構最相關）

- **arXiv**: [2403.03181](https://arxiv.org/abs/2403.03181)（已用 WebFetch 打開 abs＋html 確認；ICML 2024 Spotlight）
- **作者**：Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi
  Shafiullah, Lerrel Pinto（*Behavior Generation with Latent Actions*）。
- **一句話**：BeT 加裝階層式（residual）VQ-VAE 動作 tokenizer + transformer prior 預測
  code 序列，目標是同時拿到 BeT 的推論速度與 Diffusion Policy 的多模態表達力。
- **訓練訊號**：純 offline，標準 BC 形式化。`[paper 明講]`
- **關鍵參數**（Table 13, Appendix C.1）：`[paper 明講]`
  - Residual VQ 層數：預設 Nq=2；主 codebook 大小依環境 8~16。
  - **Chunk 長度**：Kitchen=1、Ant=1、BlockPush=1、UR3=10、PushT=5、NuScenes=6、
    real-world=1 —— ⚠️ **多數環境 chunk 長度其實是 1**，只有 UR3/PushT/NuScenes 有真正多步 chunk。
  - **執行方式：全 closed-loop、每步重新規劃**。Sec 4.7「Practical concerns」原文機制：
    作者原本想用 Diffusion Policy 式 receding-horizon（開環執行一段再 replan），但
    **「fails completely in our environment」**——他們的低成本 mobile manipulator 控制
    噪聲導致開環僅 3 步就跑出分布外，因此改成全 closed-loop（代價是推論次數暴增：
    VQ-BeT 15.1ms vs Diffusion Policy 98.6ms/次，Table 3；real robot 差距在 closed-loop
    下反而縮小到 18ms vs 573ms，Table 7）。
    **`[paper 明講，Luna 已親自 WebFetch §4.7 核對，原句：「This controller noise
    causes models to go out of distribution during even a short period (three
    timesteps) of open-loop rollout.」，逐字相符]`**。
  - **locomotion 成績**：8 個評測環境含「Multimodal Ant」（四足 locomotion），但該環境
    chunk 長度=1（非開環多步）；`[查不到]` 具體到達率/reward 數字（使魔本次未深挖）。
    **無 humanoid 實驗。**
- **`[推論]` 與我們框架逐點對照**：
  - **同構**：離散字典 ≈ residual-VQ codebook，都是有限可重用的原子動作庫；兩者都用
    序列模型吐出一串 codebook index 代表一段軌跡；都要回答「一次 commit 多長才 replan」。
  - **不同**：VQ-BeT 的 token 絕大多數是**單步低階動作**（chunk 長度多為 1），我們的
    token 是**整個步態片段**（本身已是多步 primitive）——粒度差一個量級；VQ-BeT 對
    「commit 多長」的實測答案是**極短**（1 步、每步重規劃），而且是**被硬體雜訊逼出來
    的結論**，我們的計畫在光譜的另一端（整段一次生成＋Best-of-N，完全不 replan）；
    VQ-BeT 沒有 Best-of-N／候選評分機制，是單次取樣直接執行。
  - **最值得先試**：residual/hierarchical VQ 建碼法（先訓主 codebook 再用殘差層修正
    量化誤差）是一份可執行的「flat 字典→階層式字典」升級 recipe；但 8~16 的 codebook
    大小對應的是單步動作量級，跟我們的整段步態 token 不是同一個量級，只能當數量級起點。
  - **最重的警訊**：VQ-BeT 用真實硬體證明「開環 commit 幾步就 OOD」不是雙足特有現象，
    連相對不那麼動態敏感的 mobile manipulator 都中招——跟我們 ant→humanoid 診斷出的
    「連續執行姿態漂移放大」是同一病灶的獨立案例。

### 5. Humanoid Locomotion as Next Token Prediction

- **arXiv**: [2402.19469](https://arxiv.org/abs/2402.19469)（已用 WebFetch 打開 abs＋html 確認；NeurIPS 2024）
- **作者**：Ilija Radosavovic, Bike Zhang, Baifeng Shi 等（UC Berkeley）。
- **一句話**：把真實雙足 humanoid 控制當成 GPT 式 causal transformer 的 next-token
  prediction，混合 sim RL rollout／model-based controller／mocap／YouTube 人類影片訓練，
  zero-shot 部署到真實全尺寸 humanoid。
- **訓練訊號**：純 offline——對固定資料集做自回歸密度建模，模型自身訓練無 DAgger/RL/
  fine-tune（資料裡部分軌跡的「上游來源」是別的 RL policy 產生的，但那是資料生成階段的事，
  不是這個模型訓練時的線上互動）。`[paper 明講, Sec 3.1]`
- **關鍵參數**：⚠️ 跟標題給的直覺相反——**沒有 chunk、也沒有離散化**：Sec 3.6 明講逐步
  單步自回歸執行（非多步開環 chunk，Luna 親自核對原句：「we predict the next actions
  ... and execute the action」）；Sec 3.1 明講試過離散化成 bin／VQ，但最終選連續回歸，
  原文理由是「we found the regression approach to work reasonably well in practice
  and opt for it for simplicity」（Luna 親自核對逐字相符）；context 長度預設 16 步
  （Sec 5.1 原句「we only keep the past 16 steps in input」），Sec 5.9 額外 ablate 過
  16/32/48，結論是「Larger context windows produce better policies」——**context 越長
  越好，不是越短越好**，這點跟「短 commit horizon」的警訊是兩件不同的事，不要混為一談：
  這篇警訊在於**沒有 chunk/離散化**，不是在於「context 要短」。`[paper 明講，Luna 已親自
  WebFetch §3.1/§3.6/§5.1/§5.9 核對]`
- **locomotion/humanoid 成績**：真實全尺寸 humanoid 在舊金山戶外多種地面 zero-shot
  行走，僅用 27 小時走路資料即可 transfer，質化上優於 RL baseline（追蹤誤差更小）。
  `[paper 明講]` `[查不到]` 量化的成功率／摔倒次數／穩定度百分比——使魔查了 arXiv 全文
  ／GitHub／WebSearch 三管道都沒找到表格化數字，論文這部分主要用質化描述與圖表呈現。
- **可搬性**：`[推論]` 目前找到最接近「humanoid + next-token」的旗艦論文，卻在真實雙足
  部署時**放棄了 chunking 也放棄了離散化**，用連續動作逐步自回歸執行——這是對我們框架
  兩個核心設計選擇的直接反例，該當成需要主動 ablate 的風險項，不是預設操作任務的經驗值
  可以照搬到雙足。⚠️ **修正一個容易混淆的地方**：「16 步 context」跟「commit horizon
  多長」是兩個不同的軸——前者是模型看多少步歷史當輸入（且原文 ablation 顯示 context
  越長越好，不是越短越好），後者是模型一次要不要盲執行多步（這篇答案是 1 步、每步都
  重新自回歸）。這篇真正的警訊是「**沒有 chunk/commit 承諾**」，不是「context 要短」——
  跟 VQ-BeT 的警訊（開環幾步就 OOD）是獨立的第二個資料點，兩篇分別從操作和雙足角度，
  都指向「雙足/精細動態偏好短 commit horizon（每步或近乎每步重新決策）」，但**不代表
  輸入歷史窗也要短**。

### 搜尋涵蓋度補充（使魔的排除紀錄）

- *HumanPlus*（[2406.10454](https://arxiv.org/abs/2406.10454), CoRL 2024）：33-DoF 全身
  humanoid，但 locomotion（shadowing）是 **sim RL 訓練**、非 offline BC；ACT 風格
  chunking BC 只用在手臂類技能（穿鞋、摺衣服），不覆蓋 locomotion 本體——不符合本軸篩選
  門檻，未列入正式清單。（僅 WebSearch，未 WebFetch 逐條驗證，故不算入來源總表）
- *Data-Efficient Approach to Humanoid Control via Fine-Tuning a Pre-Trained GPT on
  Action Data*（[2405.18695](https://arxiv.org/abs/2405.18695)）：純模擬、無離散化，
  評測用 FID/ADE/FDE，穩定度弱於 Radosavovic 這篇（~5.75 秒後跌倒 vs MoCapAct-Small
  ~5 秒），判斷不如上面第 5 篇，未獨立成篇。
- 整體：「離散 VQ chunking ＋ 開環執行 ＋ humanoid/locomotion」三者同時具備的單一論文，
  認真搜尋後**沒有找到**——最接近的組合是 VQ-BeT 的 Ant（locomotion + VQ，但 chunk=1、
  非開環）與 Radosavovic 這篇（真雙足 + next-token，但無 chunk、無離散化），兩篇分別
  覆蓋其中兩個維度，沒有第三篇同時覆蓋三者。`[查不到]`

## 軸④ 抗 covariate shift 的 offline BC 技巧

_由 sonnet 使魔調研，5 個技巧全用 WebFetch 打開驗證，且使魔自己抓到並丟棄了兩個
WebSearch 摘要工具捏造的假線索（見文末排除紀錄）。**Luna 抽驗時發現 1 個實質錯誤
並已訂正**（RvS-G/RvS-R 兩欄數字對調，見下方第 5 篇，訂正後結論從「警訊」翻轉成
「本軸最強的正面證據」）。_

### 1. DART — 噪音注入示範者，不是注入學習中的 policy

- **arXiv**: [1703.09327](https://arxiv.org/abs/1703.09327)（已用 WebFetch 打開 abs 確認；CoRL 2017）
- **作者**：Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, Ken Goldberg。
- **附註**：arXiv 掛的正式標題是「DART: Noise Injection for Robust Imitation
  Learning」，不是「Disturbances for Augmenting Robot Trajectories」（後者是常見的
  縮寫展開但不是標題本身）——使魔已核實作者/年份/演算法一致，同一篇，不是查錯番號。
- **一句話**：把校準過的高斯噪音注入 **supervisor（示範者）的動作**，不是注入正在
  學習的 policy，讓示範資料本身就包含「偏離一點之後怎麼修正回來」的例子。
- **訓練/資料生成階段是否需要 online rollout**：`[paper 明講]` 分兩層：①不需要執行
  我們的 BC learner——原文「We propose an off-policy approach that injects noise into
  the supervisor's policy while demonstrating」，全程只跑 supervisor，這是它跟 DAgger
  的根本差異。②但完整演算法仍是疊代式的——要根據「supervisor 與目前 learner 的誤差」
  校準噪音共變異數，代表它假設有一個**可重複呼叫的 supervisor**。`[推論]` 嚴格說，
  DART 需要「資料生成階段可與 supervisor 互動」，只是這個互動不牽涉我們自己在訓練的
  learner。
- **humanoid 成績**：`[paper 明講，Luna 已親自 WebFetch abs 頁核對逐字相符]` MuJoCo
  Humanoid（前向行走不摔倒任務，演算法型 supervisor 用 TRPO 訓練，非人類示範）：
  「For high dimensional tasks like Humanoid, DART can be up to 3x faster in
  computation time」；「DART...only decreases the supervisor's cumulative reward by
  5% during training」對比「DAgger...executes policies that have 80% less cumulative
  reward than the supervisor」。
- **可搬性**：`[推論]` ①若我們手上還留著當初產生 humanoidmaze 資料集的 controller
  （不是我們的 BC learner），可以離線用它 + DART 式噪音重新生成一批「擾動後示範修正」
  的資料，全程不碰我們的模型，完全符合「訓練不碰環境」的限制——可行性取決於那個
  data-generating controller 現在還能不能被重新呼叫。②若只剩死資料、沒有可重跑的
  supervisor，DART「依 learner 誤差校準噪音」這個核心步驟就沒辦法照搬，只能借概念、
  換一種不需要重新示範的方式實作（例如資料集內部做鄰近點 stitching），不是真的 DART。

### 2. DAgger — 對照組：為什麼不能搬

- **arXiv**: [1011.0686](https://arxiv.org/abs/1011.0686)（已用 WebFetch 打開確認；AISTATS 2011）
- **作者**：Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell。
- **一句話**：每輪執行「目前正在學的 policy」跟環境互動、走到哪就問一次專家「正確動作
  是什麼」，把新標註疊加進訓練集重訓。
- **訓練/資料生成階段是否需要 online rollout**：`[paper 明講]` 需要，且兩件事缺一不可：
  「At iteration i, execute the current learned policy π_i to collect a dataset of
  states, and query an expert for the optimal action at each of those states.」——
  (a) 要執行**正在訓練、還不成熟**的 policy 本身跟環境互動（跟 DART 只跑 supervisor
  本質不同），(b) 要有一個**當場可即時問答**的專家 oracle。我們的 pipeline 兩者都
  沒有，`[推論]` 不能直接搬。
- **humanoid 成績**：`[查不到]` 原始論文本身的實驗細節本次未能完整核實；可交叉引用
  上面 DART 論文裡拿 DAgger 對比出的 MuJoCo Humanoid 數字，但那個數字出自 DART 論文
  不是 DAgger 原論文，僅供參考不算 DAgger 自己的一手數字。
- **可搬性**：`[推論]` 不可搬。要用 DAgger 必須把 pipeline 改成線上——每輪跑 learner、
  每輪問活專家——直接違反「訓練全程不與模擬環境互動」的前提，是設計上排除的選項。

### 3. HER (Hindsight Experience Replay) — relabeling 的起源，但仍是線上 RL

- **arXiv**: [1707.01495](https://arxiv.org/abs/1707.01495)（已用 WebFetch 打開確認；NeurIPS 2017）
- **作者**：Marcin Andrychowicz 等 10 位作者。
- **一句話**：把軌跡「事後實際到達的狀態」拿來當那條軌跡的目標重新標籤，讓失敗的嘗試
  也變成可學的成功案例。
- **訓練/資料生成階段是否需要 online rollout**：`[paper 明講]` 論文驗證的版本需要——
  摘要「combined with an arbitrary off-policy RL algorithm」，實作接 DDPG，off-policy
  RL 本身代表每輪仍持續跟環境互動收新資料。`[推論]` 但「重新標籤」這個資料轉換動作
  本身，理論上可以直接套用在一批固定不動的離線資料上——這正是下面 GCSL/RvS-G 在做的
  簡化版本，只是 HER 這篇論文自己沒有用純離線方式測過。
- **locomotion 成績**：`[paper 明講]` 只在非 locomotion 任務驗證——摘要列出 pushing、
  sliding、pick-and-place 三個 Fetch 機械手臂操作任務，沒有腿式移動或 humanoid 實驗。
- **可搬性**：`[推論]` 不要照搬 HER 整套 off-policy RL 骨架，只借「重新標籤」概念本身，
  改用下面 GCSL/RvS-G 驗證過的單輪離線版本。

### 4. GCSL — relabeling 的「純監督式」版本，但本身仍要 rollout

- **arXiv**: [1912.06088](https://arxiv.org/abs/1912.06088)（已用 WebFetch 打開確認；2019）
- **作者**：Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Devin,
  Benjamin Eysenbach, Sergey Levine。
- **一句話**：疊代地 (a) 用目前 policy 跑一條新軌跡、(b) 把軌跡實際到達的終點狀態
  重新標成目標、(c) 對 (state, 事後目標)→action 做監督式訓練，再重複。
- **訓練/資料生成階段是否需要 online rollout**：`[paper 明講]` ⚠️ 值得特別提醒——GCSL
  **本身不是零 rollout 演算法**。摘要「Each iteration, the agent collects new
  trajectories using the latest policy」；Algorithm 1「collect data with
  π_k(⋅|⋅,g)」。它不需要外部專家或 reward（比 DAgger 好），但每輪都要把**自己的
  policy** 丟進環境收新軌跡，`[推論]` 仍是一種 online 互動。
- **locomotion 成績**：`[paper 明講]` 沒有。實驗環境為 2D Room Navigation、Robotic
  Pushing、Lunar Lander、Door Opening、Claw Manipulation，全不是腿式/humanoid 任務。
- **可搬性**：`[推論]` 不要照搬 GCSL 疊代演算法本身（假設每輪能重新 rollout，違反零
  互動限制）；該搬的是背後那句話的**單輪、純離線版本**——直接在固定資料集上把每條
  軌跡未來到達的狀態當目標重新標籤一次，訓一輪監督式 BC，完全不需要 rollout——這正是
  下一項 RvS-G 實際驗證過的做法。

### 5. RvS-G ⭐ —— 純離線 hindsight-relabeled 目標條件 BC，本軸最強證據

_⚠️ 含 Luna 抽驗訂正：使魔原回報的數字對錯了欄，結論方向被訂正為相反。_

- **arXiv**: [2112.10751](https://arxiv.org/abs/2112.10751)（已用 WebFetch 打開確認；ICLR 2022，*RvS: What is Essential for Offline RL via Supervised Learning?*）
- **作者**：Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, Sergey Levine。
- **一句話**：在固定離線資料集上，對同一條軌跡「之後任一時刻實際到達的狀態」均勻取樣
  當目標標籤（hindsight relabeling），單輪訓練一個兩層 MLP 做 max-likelihood 監督式
  擬合，訓練全程不跟環境互動。論文同時報告兩個變體：**RvS-R**（條件在 return-to-go，
  跟 Decision Transformer 同類）與 **RvS-G**（條件在未來狀態，才是跟 GCSL/我們的
  goal-conditioned BC 同構的版本）。
- **訓練/資料生成階段是否需要 online rollout**：`[paper 明講，Luna 已親自 WebFetch 核對]`
  不需要。RvS-G 的目標定義：`f(ω∣τ_{t:H}) = Unif(s_{t+1}, s_{t+2}, ..., s_H)`——對每條
  既有軌跡取時間點 t，目標從同一條軌跡「之後」visited 過的狀態均勻取樣（Algorithm 1）。
  環境只在**評估**時被用來量成功率，訓練階段沒有互動。
- **⚠️ 抽驗訂正記錄**：使魔原始回報把 RvS-G 的 antmaze-medium/large 成績寫成
  7.7/4.5/3.7/3.5（讀成「腰斬式崩潰」），Luna 親自用 WebFetch 打開同一個 arXiv HTML
  兩次交叉核對（含 ar5iv 版的完整表格逐欄轉錄）後發現：**那組數字其實是 RvS-R 欄位
  的**，RvS-G 欄位的真實數字完全不同、而且方向相反。以下是訂正後的正確數字。
- **humanoid/locomotion 成績（AntMaze，四足非雙足，但結構上跟 humanoidmaze 最接近）**：
  `[paper 明講，Luna 親自用 WebFetch 兩次交叉核對 Table 1，含 ar5iv 版逐欄轉錄]`

  ```
                          RvS-G   RvS-R   CQL-p   TD3+BC   DT
  antmaze-umaze            65.4    64.4    74.0     78.6*  65.6
  antmaze-umaze-diverse    60.9    70.1    84.0     71.4*  51.2
  antmaze-medium-play      58.1     4.5    61.2     10.6*   1.0
  antmaze-medium-diverse   67.3     7.7    53.7      3.0*   0.6
  antmaze-large-play       32.4     3.5    15.8      0.2*   0.0
  antmaze-large-diverse    36.9     3.7    14.9      0.0*   0.2
  平均                     53.5    25.6    50.6*     27.3*  19.8
  ```

  （*CQL-p/TD3+BC 用 antmaze-v0 資料集，其餘用 v2，論文原表如此並排，非 Luna 自己混用）。
  **RvS-G 在這張表裡平均分數最高**（53.5，贏過 CQL-p 的 50.6、TD3+BC 的 27.3、DT 的
  19.8、甚至贏過同篇的 RvS-R 25.6），而且在最難的 large-diverse/large-play 上也沒有
  崩掉（36.9／32.4，還贏 CQL-p 的 14.9／15.8）。
- **可搬性（訂正後）**：`[推論]`
  1. 這是 5 篇裡唯一「純離線＋有量化 locomotion-迷宮數字」的技巧，而且訂正後看到的是
     **正面證據，不是警訊**：目標條件（不是 return 條件）的 hindsight-relabeled 純
     BC，在一組涵蓋 CQL/TD3+BC/DT 的廣泛比較裡是最強的離線方法，包含在較難的
     medium/large 迷宮上。這支持「純 BC 配上對的 relabeling 方式，不是結構性地打不過
     value-based 方法」，跟軸①的 GCBC 在 humanoidmaze 上慘輸 HIQL 的落差，很可能不是
     「BC 天生不行」，而是關在別的地方（relabeling 分佈？架構？雙足特有的漂移？）。
  2. ⚠️ 開放問題（尚待查證，`[查不到]`）：我們現在的 goal-conditioned 訓練標的，
     跟 OGBench 官方 GCBC 一樣，究竟是只對「任務指定的最終目標」條件化，還是也有做
     某種 hindsight relabeling？這點本次調研沒有查證 OGBench GCBC 的確切 relabeling
     分佈（軸①只確認了它是逐步單步、無 chunking，沒有確認 goal 標籤怎麼抽樣）。
     若我們目前只對最終目標條件化，RvS-G 這種「均勻抽樣未來任一狀態當目標」的作法
     本身就是一個便宜、值得先試的改動。
  3. 因為 RvS-G 的證據是 Ant（四足）不是 Humanoid（雙足），雙足在姿態穩定性上更脆弱、
     更容易因誤差累積摔倒，這條證據只能算「同類技巧在鄰近 embodiment 上可行」的支持
     訊號，不能直接保證雙足也會有相近數字——但至少排除了「純 BC 在迷宮類任務上必然
     打不過 value-based 方法」這個悲觀假設。

### 額外項：訓練時對輸入 state 加噪音——`[查不到]`（誠實記錄一次搜尋落空）

使魔認真搜尋「訓練時對 BC 輸入 state 注入噪音、且在 humanoid/locomotion 上有驗證、且
有量化數字」這個組合，兩條原本看起來有希望的線索，WebFetch 核實後都不成立：

- **Stable-BC**（[2408.06246](https://arxiv.org/abs/2408.06246)，已用 WebFetch 打開
  確認存在）：這篇是控制理論式穩定性正則化（不是噪音注入），2024 新技巧，明講「訓練
  純用離線資料，不需要環境互動」`[paper 明講]`，但驗證環境是路口雙車互動駕駛／四旋翼
  避障／影像式點質量導航／機械手臂打空氣曲棍球，`[paper 明講]` **沒有任何腿式/四足/
  humanoid 實驗**（先前 WebSearch 摘要曾聲稱它有「quadruped locomotion」實驗，使魔
  用 WebFetch 直接核對論文摘要與作者專案頁後確認不存在這個實驗，判斷是摘要工具誤植，
  已排除未採用——這是使魔自己抓到的錯誤，不是 Luna 抽驗抓到的）。
- **RL with Evolutionary Trajectory Generator**（[2109.06409](https://arxiv.org/abs/2109.06409)）：
  確實是四足 locomotion，但 WebFetch 核實後是純線上 RL（policy 與 trajectory
  generator 交替優化、靠 live simulator rollout），不是 BC 加噪音，先前搜尋摘要描述
  有誤，已排除未採用。

`[查不到]` 同時滿足「訓練時對 state 加噪音」＋「humanoid/locomotion 驗證」＋「有量化
數字」三個條件的論文——目前看起來這是文獻裡的一個空白，如果我們想做這個消融，得自己
在 humanoidmaze 上跑出來，不是在借用一個已驗證的數字。最接近的替代證據還是第 1 篇
DART（資料生成階段對 supervisor 加噪音，不是訓練時對 state 加噪音，但「小擾動＋觀察
怎麼修正回來」的邏輯相通，且它確實有 humanoid 數字）。

---

## 可搬清單 Top 5

排序原則：預期收益（對「抗摔/抗漂移」這個主人最在意的問題有多直接）× 搬運成本（要
改多少東西、可行性有多確定）。每條一行理由，rationale 先於結論。

1. **開環 commit 長度要先做 sweet-spot 消融，不要預設「整段」就是最優。**
   理由：Diffusion Policy 的 ablation 明講執行長度 Ta=8 是多數任務最優、不是越長越好
   `[paper 明講]`；VQ-BeT 在真實硬體上「開環僅 3 步就跑出分布外」被迫全部改回
   closed-loop `[paper 明講]`；Radosavovic 的真雙足旗艦論文乾脆放棄 chunking、逐步
   單步執行 `[paper 明講]`——三個獨立來源收斂在同一個方向：commit 太長有結構性風險。
   成本最低（用既有基礎設施跑一組消融即可），直接回答我們新框架最大的未定設計問題。

2. **Hindsight-relabel 對「未來任一狀態」條件化，不是只對最終目標條件化。**
   理由：HIQL 的 low-level policy 這樣做，把 OGBench 官方 humanoidmaze-medium-navigate
   從 GCBC 的 8% 推到 89%`[paper 明講, Table 2]`；RvS-G（同款純 offline
   hindsight-relabeled 目標條件 BC，**Luna 抽驗訂正後**確認的真實數字）在一組涵蓋
   CQL/TD3+BC/DT 的廣泛 AntMaze 比較裡是**平均分數最高的方法**（53.5，含較難的
   large 迷宮也沒崩）`[paper 明講]`。這是全份調研裡最強的正面實證：純 BC 配對的
   relabeling 方式，不是結構性地打不過 value-based 方法。成本低（純資料標籤方式，
   不需要 value function）；⚠️ 但要先查清楚我們現在的 goal-conditioning 訓練標的是
   只對最終目標、還是已經有某種 relabeling——這是開放問題，`[查不到]`，值得先花小
   成本確認再決定要不要動。

3. **Best-of-N 評分器加一個離線可算的「像不像會摔」判準。**
   理由：目前的 Best-of-N 若只看「有沒有到達目標」，沒有專門對「穩不穩」評分；PHC
   的關節偏移幾何判準（>0.5m 判失敗，`[paper 明講]`）與 AMP 的判別器特徵（root 速度、
   local 關節旋轉/速度、end-effector 位置，`[paper 明講]`）都是可以離線改造成 scorer
   的現成起點——`[推論]` 兩者都不需要在評分階段呼叫模擬器，純幾何/運動學特徵。成本
   中等（要建一個 scorer/classifier，但範圍明確、一次性訓練）。

4. **訓練資料做 phase / 時間位移的隨機視窗增強（RSI 效果的離線版）。**
   理由：DeepMimic／MoCapAct／PHC 三條獨立 online-RL 線都靠 RSI 做樣本效率
   `[paper 明講×3]`，`[推論]` 其「效果」——訓練涵蓋多樣的相位/時間起點——可以完全用
   離線資料切窗重現，不用碰模擬器。成本最低（純資料前處理）。排在第 4 是因為證據
   強度是「多篇 online RL 論文的機制成分＋我方推論它能離線重現」，不是直接在 offline
   BC 上量化驗證過的效果，比前三條更依賴外推。

5. **若還能重新呼叫產生 humanoidmaze 資料集的底層 controller，用 DART 式噪音重新生成
   一批「擾動—修正」示範。**
   理由：DART 是本次調研唯一有**真實 MuJoCo Humanoid** 量化數字的抗漂移技巧——比
   DAgger 快 3 倍、supervisor 累積 reward 只掉 5%（DAgger 掉 80%）`[paper 明講]`，且
   噪音注入示範者、不用把我們的 learner 丟進環境。排最後是因為可行性不確定：需要
   一個「可重複呼叫的資料生成 controller」，如果我們手上只剩死資料集，這條就無法
   照搬，只能借「小擾動＋觀察修正」的概念另外實作。

## 來源總表

| # | 軸 | 論文 | arXiv | 驗證狀態 |
|---|---|---|---|---|
| 1 | ① | OGBench: Benchmarking Offline Goal-Conditioned RL | [2410.20092](https://arxiv.org/abs/2410.20092) | 已打開確認（Luna 親自，abs+html） |
| 2 | ① | Horizon Reduction Makes RL Scalable | [2506.04168](https://arxiv.org/abs/2506.04168) | 已打開確認（Luna 親自，abs+html） |
| 3 | ① | HIQL: Offline GCRL with Latent States as Actions | [2307.11949](https://arxiv.org/abs/2307.11949) | 已打開確認（Luna 親自，abs+html） |
| 4 | ② | DeepMimic | [1804.02717](https://arxiv.org/abs/1804.02717) | 已打開確認（使魔，abs+ar5iv） |
| 5 | ② | AMP: Adversarial Motion Priors | [2104.02180](https://arxiv.org/abs/2104.02180) | 已打開確認（使魔，abs+ar5iv） |
| 6 | ② | MoCapAct | [2208.07363](https://arxiv.org/abs/2208.07363) | 已打開確認（使魔，abs+html） |
| 7 | ② | PHC: Perpetual Humanoid Control | [2305.06456](https://arxiv.org/abs/2305.06456) | 已打開確認（使魔，abs+html） |
| 8 | ② | PULSE | [2310.04582](https://arxiv.org/abs/2310.04582) | 已打開確認（使魔，abs+html+ICLR proceedings） |
| 9 | ③ | ACT (Action Chunking Transformer) | [2304.13705](https://arxiv.org/abs/2304.13705) | 已打開確認（使魔，abs+html） |
| 10 | ③ | Diffusion Policy | [2303.04137](https://arxiv.org/abs/2303.04137) | 已打開確認（使魔，abs+html v5） |
| 11 | ③ | Behavior Transformer (BeT) | [2206.11251](https://arxiv.org/abs/2206.11251) | 已打開確認（使魔，abs+html） |
| 12 | ③ | VQ-BeT | [2403.03181](https://arxiv.org/abs/2403.03181) | 已打開確認（使魔＋**Luna 親自抽驗 §4.7**） |
| 13 | ③ | Humanoid Locomotion as Next Token Prediction | [2402.19469](https://arxiv.org/abs/2402.19469) | 已打開確認（使魔＋**Luna 親自抽驗 §3.1/3.6/5.1/5.9**） |
| 14 | ④ | DART | [1703.09327](https://arxiv.org/abs/1703.09327) | 已打開確認（使魔＋**Luna 親自抽驗 abs**） |
| 15 | ④ | DAgger | [1011.0686](https://arxiv.org/abs/1011.0686) | 已打開確認（使魔，abs） |
| 16 | ④ | HER (Hindsight Experience Replay) | [1707.01495](https://arxiv.org/abs/1707.01495) | 已打開確認（使魔，abs） |
| 17 | ④ | GCSL | [1912.06088](https://arxiv.org/abs/1912.06088) | 已打開確認（使魔，abs+ar5iv） |
| 18 | ④ | RvS: What is Essential for Offline RL via Supervised Learning? | [2112.10751](https://arxiv.org/abs/2112.10751) | 已打開確認（使魔＋**Luna 親自抽驗並訂正 Table 1，共 2 次交叉核對**） |

**考慮過但排除／未逐條驗證的（誠實列出，不算入上表 18 篇正式來源）：**

| 論文 | arXiv | 狀態 |
|---|---|---|
| HumanPlus | [2406.10454](https://arxiv.org/abs/2406.10454) | 僅 WebSearch，未 WebFetch 逐條驗證；且其 locomotion 是 sim RL 非 offline BC，不符軸③篩選門檻 |
| Data-Efficient GPT Humanoid Control | [2405.18695](https://arxiv.org/abs/2405.18695) | 已 WebFetch 但穩定度弱於 Radosavovic 那篇，未獨立成篇 |
| Stable-BC | [2408.06246](https://arxiv.org/abs/2408.06246) | 已 WebFetch 確認存在，但驗證環境無 humanoid/locomotion，不符軸④篩選門檻 |
| RL with Evolutionary Trajectory Generator | [2109.06409](https://arxiv.org/abs/2109.06409) | 已 WebFetch 確認存在，但是純 online RL 非 BC+噪音，不符軸④篩選門檻 |

## 抽驗紀錄（Luna 親自核對，未照單全收使魔回報）

三隻使魔的原始回報整體品質很高（自己就抓到、丟棄了 2 個 WebSearch 摘要工具捏造的假
線索），但仍照 fleet-command 的紀律「使魔的結論＝待驗貨物」，對會進入上面 Top 5 的
關鍵主張做了親自抽驗：

- **VQ-BeT §4.7「開環 3 步就 OOD」**——WebFetch 打開原文逐字核對，**相符**。
- **Radosavovic §3.1/3.6「無 chunk、無離散化」＋ §5.1/5.9 context 長度**——WebFetch
  打開原文逐字核對，**相符**；額外發現使魔的敘述有個容易誤導的地方（把「context
  長度 16」跟「commit horizon 短」混在一起講），**已在報告內文訂正**：這篇的 context
  ablation 其實顯示「越長越好」，跟「commit horizon 該多短」是兩個獨立的軸，已分開講清楚。
- **DART 摘要「3x 快、supervisor reward 只掉 5% vs DAgger 掉 80%」**——WebFetch 打開
  abs 頁逐字核對，**相符**。
- **RvS-G 在 AntMaze 上的成績**——WebFetch 兩次打開同一篇（含 ar5iv 完整表格版）交叉
  核對，**發現使魔把 RvS-G 欄位的數字回報成了 RvS-R 欄位的數字**（7.7/4.5/3.7/3.5
  其實是 RvS-R，不是 RvS-G；RvS-G 真實數字是 67.3/58.1/36.9/32.4，本軸表現最好的方法）。
  **已在軸④第 5 篇與 Top 5 第 2 條訂正**，這是本次調研裡目前抓到唯一一個實質數字錯誤，
  幸好訂正方向對我們是好消息不是壞消息。
