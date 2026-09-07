# Survey：動作/姿態軌跡的壓縮潛在表示（2026-09-08）

_為 LaCoT 下一步設計題調研：目前 u 只編碼 2D xy 位置軌跡（計畫棧 xy 投影），
下一步要把「動作/姿態」也壓進 thinking。本檔調研前人怎麼壓「動作流形」、
怎麼訓、下游怎麼消費，四條線各 ≥2 篇。_

**分級標記**（依任務書定義）：
`[摘要級]` = 讀過 arXiv abstract 頁；`[正文級]` = 讀過方法/實驗章節原句或作者專案頁原文；
`[記憶級、番號未驗]` = 番號不確定（**本檔沒有任何一篇用這個標記** — 16 篇番號全部經
WebSearch 命中 arXiv listing 頁 + WebFetch `arxiv.org/abs/<id>` 交叉確認）。
每條引用盡量附一手原句（英文原文，未翻譯以免失真）。

---

## 索引表（快速掃描；細節見下方各節）

| # | 篇名 | 番號 | 分級 | 線 | 壓縮對象 | latent 形狀（濃縮） |
|---|------|------|------|----|----|----|
| 1 | AMP | 2104.02180 | 摘要級 | 1 | state transition（無顯式 latent） | 無 latent，判別器純量 reward |
| 2 | ASE | 2205.01906 | 正文級 | 1 | state transition + skill z | z ∈ 單位超球面（維度未驗） |
| 3 | PULSE | 2310.04582 | 正文級 | 1 | 整條 motion（含姿態） | 32 維連續 |
| 4 | PHC | 2305.06456 | 摘要級 | 1 | 無壓縮（萬用 imitator，PULSE 的原料） | 無 latent |
| 5 | CALM | 2305.02195 | 正文級 | 1 | state transition + skill z | 64D 超球面 |
| 6 | OPAL | 2010.13611 | 正文級 | 2 | c 步動作序列 | 8 維連續，c=10 步/z |
| 7 | SPiRL | 2010.11944 | 正文級 | 2 | 動作序列 + state-conditioned prior | 10 維連續，horizon H 可調 |
| 8 | TACO | 2306.13229 | 摘要級 | 2 | state 表徵 + action 表徵（聯合對比） | 維度未載明 |
| 9 | SkiMo | 2207.07560 | 摘要級 | 2 | 動作序列（skill）+ skill dynamics | 維度/horizon 未查得 |
| 10 | LAPA | 2410.11758 | 摘要級 | 3 | 影片幀對之間的隱式動作 | 離散 VQ-VAE，codebook 大小未載明 |
| 11 | Genie | 2402.15391 | 正文級 | 3 | 影片幀對之間的隱式動作 | 離散，`\|A\|=8` |
| 12 | GR00T N1 | 2503.14734 | 正文級 | 3 | 連續動作 chunk（非量化） | H=16 步 chunk，維度依 embodiment |
| 13 | Moto | 2412.04445 | 摘要級 | 3（bonus） | 影片→motion token 序列 | 離散 token 序列，維度未載明 |
| 14 | Diffuser | 2205.09991 | 正文級 | 4 | state ⊕ action 聯合（不壓縮） | 原始維度串接，無額外 latent |
| 15 | Decision Diffuser | 2211.15657 | 正文級 | 4 | 只壓 state；action 另算 | state-only 軌跡 + 獨立 inverse dynamics |
| 16 | Hierarchical Diffuser | 2401.02644 | 正文級 | 4 | 兩層：稀疏 subgoal + 稠密 state⊕action | 高層每 K=4/15 步跳一次，低層無額外 latent |

---

## 線 1：Physics character / locomotion 圈的 motion prior

### 1. AMP — arXiv:2104.02180 `[摘要級]`（Peng, Ma, Abbeel, Levine, Kanazawa；SIGGRAPH/TOG 2021）

- **壓縮對象**：unstructured motion clip 裡的 state transition `(s_t, s_{t+1})`，不是個別動作序列，壓的是「風格」。
- **latent 形狀**：**無**顯式可採樣 latent。判別器 `D(s_t,s_{t+1})` 只輸出一個純量 style-reward，是隱式先驗而非 latent 空間。
- **訓練目標**：對抗模仿學習（GAN 式）——`reward = -log(1-D(s_t,s_{t+1}))`，不需要逐幀模仿目標或 clip selection。原文：
  > "These motion clips are used to train an adversarial motion prior, which specifies style-rewards for training the character through reinforcement learning (RL)."
- **下游消費**：純粹當 RL 的 style-reward 訓練訊號；沒有可控 interface，策略本身無可讀的「thinking」層。是 ASE/CALM 的地基。
- **對我們**：本身沒有 latent code 可借，但證明「discriminator-as-prior」能學出好的動作分佈——若我們想幫動作流形加一個「像不像真實動作」的正則項（而非只有重建 loss），這是最早的模板。

### 2. ASE — arXiv:2205.01906 `[正文級]`（Peng, Guo, Halper, Levine, Fidler；SIGGRAPH/TOG 2022）

- **壓縮對象**：state transition，但現在有 latent skill code `z` 中介，是 AMP 的「可控制」升級版。
- **latent 形狀**：`z` 落在單位超球面 `𝒵={z: ‖z‖=1}`（正規化連續空間）。**具體維度數字在可讀全文段落中未驗到**（不編號猜測；狀態空間 120D、動作空間 31D 有驗到，但那是角色本體維度不是 z 的維度）。
- **訓練目標**：對抗 + 互資訊雙目標。原文：
  > "The reward for the policy at each time step is then specified by `r_t = -log(1-D(s_t,s_{t+1})) + β·log q(z_t|s_t,s_{t+1})`."
  判別器項 + skill-discovery 編碼器項，逼 z 與 transition 互資訊最大化，形狀像 InfoGAN 而非 VAE。
- **下游消費**：兩層架構——低層 `π(a|s,z)` 訓練後凍結常駐；高層 `ω(z|s,g)` 針對新任務重新訓練，只輸出 z。原文：
  > "a task-specific high-level policy `ω(z|s,g)`, which receives as input the state of the character `s` and a task-specific goal `g`, then outputs a latent `z`."
- **對我們**：「z 當高層 policy 的輸出空間、凍結 decoder 當低層」的最直接先例——若把 u 換成動作 latent，高層 planner 可以直接套 `ω(z|s,g)` 的角色。

### 3. PULSE — arXiv:2310.04582 `[正文級，32 維數字來自作者 GitHub README 原文]`（Luo, Cao, Merel, Winkler, Huang, Kitani, Xu；ICLR 2024 spotlight）

- **壓縮對象**：把「萬用動作模仿控制器」（PHC，見下）distill 成通用 humanoid motion 表徵——壓的是整條 motion trajectory（含姿態），覆蓋範圍刻意做大（不只 locomotion）。
- **latent 形狀**：低維、**32 維**。GitHub README 原文："low dimensional (32)"；覆蓋 99.8% AMASS motion。
- **訓練目標**：encoder-decoder + variational information bottleneck（VAE 式蒸餾），額外聯合學一個以 proprioception（自身姿態/速度）為條件的 prior，提升採樣效率與穩定性。摘要原文：
  > "This is achieved by using an encoder-decoder structure with a variational information bottleneck. Additionally, we jointly learn a prior conditioned on proprioception... to improve model expressiveness and sampling efficiency."
- **下游消費**：當 hierarchical RL 的動作空間——高層 policy 輸出 32 維 z，丟給凍結的 decoder 展開成 humanoid 控制訊號；也可直接從 prior 採樣生成長時間穩定動作（當生成模型用）。
- **對我們**：四條線裡「壓縮維度」證據最紮實的一篇——**32 維就能覆蓋近乎全部人類動作**，直接回答「動作流形 latent 要多大」這個量級問題；也證明「先蒸餾萬用 imitator、再壓成小 latent」的兩階段做法可行。

### 4. PHC — arXiv:2305.06456 `[摘要級]`（Luo, Cao, Winkler, Kitani, Xu；Meta Reality Labs + CMU；ICCV 2023）

- **壓縮對象**：PHC 本身不壓縮——它先修出一個能無限期模仿上萬條 motion clip、不摔倒、不需重置的萬用控制器，是 PULSE 拿去蒸餾的原料。
- **latent 形狀**：無顯式 latent。核心是漸進式乘法控制策略（PMCP），動態擴充網路容量學更難的動作序列，避免 catastrophic forgetting。
- **訓練目標**：模仿學習（imitation），最小化與 reference motion 的追蹤誤差，對雜訊輸入（video pose estimate / language-generated motion）容錯。
- **下游消費**：直接部署成即時多人虛擬化身控制器；也是 PULSE 的教師模型。
- **對我們**：跟我們的關係是「地基」而非「介面」——如果我們也要「先有一個吃得下任意動作序列的萬用低層 controller，再蒸餾出 thinking-friendly latent」，PHC→PULSE 這條兩階段管線可以照抄骨架。

### 5. CALM — arXiv:2305.02195 `[正文級]`（Tessler, Guo, Mannor, Chechik, Peng；NVIDIA + Technion + SFU；SIGGRAPH 2023）

- **壓縮對象**：跟 ASE 一樣壓 state transition，但加了「語意可讀」目標——encoder 要重建動作的關鍵特徵，而不是任由判別器黑箱決定 z 的意義。
- **latent 形狀**：**64 維超球面**。原文："The latent space `𝒵` is defined as a 64D hypersphere."
- **訓練目標**：conditional 對抗模仿（`reward = -log(1-D(s,s'|z))`）+ encoder 雙正則損失——**alignment loss**（同一動作片段時間重疊的片段要編到相近 z）+ **uniformity loss**（隨機取樣動作要盡量撐滿超球面）。**沒有顯式 reconstruction loss**，encoder 純靠 policy gradient 端到端訓練（`stop grad(E(M))` 阻斷 encoder 從判別器收梯度）。
- **下游消費**：階層式——高層 policy 在「precision training」階段學會輸出 `z_t` 給低層 `π(a|s,z)` 執行；推論時也可用 FSM 直接餵 motion encoding 給低層做單一動作，不經高層。原文：
  > "the high-level policy produces latent variables z_t. These are then provided to the low-level policy which controls the character."
- **對我們**：alignment/uniformity 雙損失是「不用重建 loss、也能讓 latent 語意可讀、且可插值」的具體範例——若想讓動作 u 的空間「插值出來的中間點也是合法動作」，這兩個 loss 項可以直接借。

---

## 線 2：Offline RL / skill 圈的 trajectory/skill embedding

### 6. OPAL — arXiv:2010.13611 `[正文級]`（Ajay, Kumar, Agrawal, Levine, Nachum；ICLR 2021）

- **壓縮對象**：offline dataset 裡以 state 為條件的**動作序列**（action sequences），學一個 continuous primitive 空間。
- **latent 形狀**：`dim(Z)=8`，每個 z 對應 `c=10` 個環境時間步的動作。原文："Unless otherwise stated, we use `c=10` and `dim(𝒵)=8`."
- **訓練目標**：序列 VAE——encoder `q(z|s_{0:c},a_{0:c})` 編碼一段軌跡，decoder `π(a_t|s_t,z)` 逐步重建動作序列，加 KL 正則。
- **下游消費**：離線 RL 直接在 8 維 z 空間做策略優化（取代高頻 raw action space），減少 effective horizon、避開分佈外動作；也用於 few-shot imitation 與線上 RL 的 exploration prior。
- **對我們**：「c 步動作序列 → 8 維連續 code」是最直接可比的先例——若 u 要吃「一段時間內怎麼走」，OPAL 給了具體、維度很小的參考答案（8 維/10 步）。

### 7. SPiRL — arXiv:2010.11944 `[正文級]`（Pertsch, Lee, Lim；CoRL 2020）

- **壓縮對象**：與 OPAL 同源，但額外多學一個「以當前 state 為條件的 skill prior」。
- **latent 形狀**：**10 維** skill embedding 空間。原文："the skill encoder...outputs the parameters (μ_z,σ_z) of the Gaussian posterior distribution in the **10-dimensional** skill embedding space `𝒵`." Horizon `H` 可調（消融顯示太短沒有時間抽象、太長難探索）。
- **訓練目標**：三項 loss——重建 `log p(a_i|z)`；KL 到標準常態先驗 `β(log q(z|a_i)-log p(z))`；state-conditioned prior 逼近 posterior：`D_KL(q(z|a_i), p_a(z|s_t))`。
- **下游消費**：下游 RL 用學到的 skill prior 取代標準熵正則——`-α·D_KL(π(z_t|s_t), p_a(z_t|s_t))`，讓探索偏向資料裡真的出現過的技能而非均勻亂試。
- **對我們**：「state-conditioned prior 取代熵正則」機制很乾淨——若想讓 planner 在動作流形裡探索時「偏好像 demonstration 的走法」，可直接借這個 prior-regularized RL 的公式形狀。

### 8. TACO — arXiv:2306.13229 `[摘要級]`（Zheng, Wang, Sun, Ma, Zhao, Xu, Daumé III, Huang；NeurIPS 2023）

- **壓縮對象**：與前兩篇不同——不是動作序列 VAE，而是**同時**學 state 表徵和 action 表徵，兩者用對比學習聯合對齊。
- **latent 形狀**：摘要未給出具體維度數字（未查得，不編號）。核心是「(當前 state + action 序列) 聯合表徵」與「未來 state 表徵」配對。
- **訓練目標**：時序對比學習——最大化「當前 state+action 序列表徵」與「對應未來 state 表徵」的互資訊（InfoNCE 風格），同時學 state 和 action 兩個 encoder。摘要原文：
  > "TACO simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states."
- **下游消費**：學出來的表徵當 RL policy 的輸入特徵，不是生成式 decoder；可當 plug-and-play 模組插進既有 offline visual RL pipeline 提升樣本效率。
- **對我們**：提醒「壓縮動作」不一定要走生成式 VAE/decoder 路線——用對比學習把 action 表徵和 state 表徵綁在同一空間，也是讓 u 感知「動作流形」的方式，可當訓練目標候選之一（而非只能是重建）。

### 9. SkiMo — arXiv:2207.07560 `[摘要級]`（Shi, Lim, Lee；CoRL 2022）

- **壓縮對象**：延續 SPiRL 的 skill 空間，但額外學一個「skill dynamics model」——直接預測執行完一整個 skill 之後的結果 state，不用逐步展開低層動作。
- **latent 形狀**：摘要未載明具體維度/horizon 數字（未查得，不編號）。
- **訓練目標**：三件事聯合訓練——skill repertoire（skill autoencoder）、skill dynamics model（skill 層級的下一狀態預測）、skill prior，三者一起從 offline 資料學。
- **下游消費**：下游用 skill dynamics model 在 skill 空間做長 horizon 的 model-based 規劃/想像，而非在原始動作空間逐步規劃，直接處理長 horizon、稀疏獎勵任務。
- **對我們**：「在 skill 空間裡做 model-based 展開」跟我們的 u（在 latent 軌跡空間裡想）概念同構——若要讓 u 也能「模擬展開」（而不只是編碼），SkiMo 的 skill-dynamics-model 角色可以借。

---

## 線 3：VLA / robot learning 圈的 latent action

### 10. LAPA — arXiv:2410.11758 `[摘要級]`（Ye et al.；ICLR 2025）

- **壓縮對象**：網路影片幀對之間的隱式動作（無機器人動作標籤）——壓的是「這兩幀之間發生了什麼動作」。
- **latent 形狀**：離散 latent action，VQ-VAE codebook。**具體 codebook size 摘要未載明（未查得，不編號——注意這是跟 Genie 不同團隊的獨立設計，不能借用 Genie 的 `|A|=8`）**。
- **訓練目標**：三階段。摘要原文：
  > "We first train an action quantization model leveraging VQ-VAE-based objective to learn discrete latent actions between image frames, then pretrain a latent VLA model to predict these latent actions from observations and task descriptions, and finally finetune the VLA on small-scale robot manipulation data to map from latent to robot actions."
- **下游消費**：finetune 階段把 latent action 當「中介語言」，少量真機資料就能解碼成真實 continuous 動作；預訓練完全不需機器人動作標籤，可吃 web-scale 人類操作影片。
- **對我們**：跟我們問題最貼的模板之一——證明「先無監督學一個離散動作語言、再用少量標註資料接到真實動作」可行。若想讓 u 帶動作資訊、又想維持 offline dataset 不夠大的彈性，這個兩階段 pretrain→finetune 接口值得抄。

### 11. Genie — arXiv:2402.15391 `[正文級]`（Bruce et al.；Google DeepMind；ICML 2024）

- **壓縮對象**：跟 LAPA 概念一致（影片幀對→隱式動作），但目的是當「世界模型的控制輸入」而非機器人動作。
- **latent 形狀**：離散，codebook 大小**明確限制為 `|A|=8`**。原文：
  > "We limit the vocabulary size `|A|` of the VQ codebook, i.e. the maximum number of possible latent actions, to a small value to permit human playability and further enforce controllability (we use `|A|=8` in our experiments)."
- **訓練目標**：VQ-VAE 式 Latent Action Model（LAM）——encoder 吃連續影片幀輸出量化動作，decoder 吃「歷史幀 + 目前選的 latent action」預測下一幀；跟自回歸動態模型、時空 video tokenizer 三件構成 11B 參數世界模型。
- **下游消費**：使用者/agent 逐幀從 8 個離散動作選一個「玩」生成的世界（當 action space）；學到的 latent action 空間也可拿去模仿未見過影片裡的行為，當 agent 訓練的動作標籤來源。
- **對我們**：證明「動作語言」可壓到極小（僅 8 個離散選項）還保有可控性與可解讀性——若動作流形也想要「不多、但每個都對應一種可辨識走法」，`|A|=8` 是具體錨點（對比 PULSE 的 32 維連續空間，呈現「離散小字典 vs 連續小維度」兩種收斂路線）。

### 12. GR00T N1 — arXiv:2503.14734 `[正文級]`（NVIDIA；2025）

- **壓縮對象**：跟 LAPA/Genie 不同——不走離散 latent action 中介，直接學連續動作 chunk 的生成式表示，兩系統緊密耦合端到端訓練。
- **latent 形狀**：System 1（diffusion transformer）一次輸出 **H=16** 步動作 chunk。原文："we set `H=16` in our implementation"，"the model uses `A_t=[a_t,a_{t+1},…,a_{t+H-1}]`"。動作維度依 embodiment 而異，用「每個 embodiment 各自的 MLP 投影到共用 embedding 維度」處理跨機型動作維度不一致。原文：
  > "To process states and actions of varying dimensions across different robot embodiments, we use an MLP per embodiment to project them to a shared embedding dimension as input to the DiT."
- **訓練目標**：System 2（VLM）解讀影像+語言，System 1（diffusion transformer）生成動作，兩者聯合端到端訓練，混合真機軌跡、人類影片、合成資料。
- **下游消費**：System 2 語意輸出直接條件化 System 1 的動作生成；推論時即時跑 diffusion transformer 產生 16 步動作 chunk 執行，不需離散中介。
- **對我們**：「高層語意 condition 低層 diffusion/flow 動作頭、動作用固定 horizon chunk 表示」結構上就是「thinking 條件化動作 decoder」的樣板——若想讓 u 直接條件化連續動作 chunk 生成器（而非量化成離散 token），GR00T 的雙系統介面是最貼近的參考。

### 13. Moto（bonus，line 3 延伸）— arXiv:2412.04445 `[摘要級]`（Chen et al.，Tencent ARC；ICCV 2025 oral）

- **壓縮對象**：跟 LAPA 同一類問題（影片→無動作標籤的隱式動作），但走「Motion Token 自回歸語言模型」路線，而非單純預測單步 latent action。
- **latent 形狀**：Latent Motion Tokenizer 把影片轉成一串 latent Motion Token 序列（具體 codebook 大小/維度摘要未載明，未查得，不編號）。
- **訓練目標**：Moto-GPT 用自回歸方式預測 motion token 序列（類似語言模型 next-token prediction，但 token 是動作/motion）。
- **下游消費**：co-fine-tuning 策略把「預測下一個 motion token」跟「真實機器人控制」橋接，讓預訓練動作知識轉移到真機操作。
- **對我們**：把動作處理成像文字一樣可自回歸生成、還能用 likelihood 評估軌跡合理性的 token 序列——跟我們「u 是 latent thinking 序列」的既有比喻幾乎同構。若要把動作也做成「一步步吐出來」的形式，Moto 的自回歸 motion token 是最直接的參照。

---

## 線 4：Diffusion / generative planner 圈把動作包進生成的做法

### 14. Diffuser — arXiv:2205.09991 `[正文級]`（Janner, Du, Tenenbaum, Levine；ICML 2022）

- **壓縮對象**：state 和 action**聯合**放進同一條軌跡擴散——不特別分開。原文：
  > "states and actions in a trajectory are predicted jointly; for the purposes of prediction the actions are simply additional dimensions of the state."
- **latent 形狀**：軌跡陣列 `τ=[[s_0,...,s_T],[a_0,...,a_T]]`，**沒有**額外壓縮 latent——直接在 (state ⊕ action) 串接後的原始維度上做 diffusion，是「不壓縮、直接生成」的路線。
- **訓練目標**：標準 diffusion 去噪目標，搭配 classifier-guided sampling（reward/約束梯度引導採樣）與 inpainting 技巧（固定起點/終點條件生成中段）。
- **下游消費**：receding-horizon control——採樣整條軌跡，只執行第一個動作，執行完重新採樣規劃。原文：
  > "The first action of a sampled trajectory `τ∼p(τ|O_{1:T}=1)` may be executed in the environment, after which the planning procedure begins again in a standard receding-horizon control loop."
- **對我們**：「不分離 state 和 action、直接聯合生成」的極端對照組——若讓 u 直接包含動作維度（串接而非分開的 latent），Diffuser 的 receding-horizon 推論介面（只吃第一步、重新規劃）現成可搬。

### 15. Decision Diffuser — arXiv:2211.15657 `[正文級]`（Ajay, Du, Gupta, Tenenbaum, Jaakkola, Agrawal；ICLR 2023）

- **壓縮對象**：跟 Diffuser 相反——**刻意只擴散 state 軌跡**，動作完全不進 diffusion model。原文："we choose to diffuse only over states"，理由是動作常是離散或高頻（如關節扭矩），比 state 更難建模。
- **latent 形狀**：state-only 軌跡（無額外壓縮 latent）；動作透過**另外單獨訓練**的 inverse dynamics 模型事後算回來：`a_t := f_φ(s_t, s_{t+1})`（論文 Eq.7）。
- **訓練目標**：state 軌跡上的 classifier-free guidance diffusion，條件變數可以是 return `R(τ)`、約束或 skill。引導公式（Section 3.2）：
  > `ε̂ := ε_θ(x_k(τ),∅,k) + ω·(ε_θ(x_k(τ),R(τ),k) − ε_θ(x_k(τ),∅,k))`
  混合有條件/無條件去噪預測，偏向高 return 軌跡，不需顯式 Q function。
- **下游消費**：先生成整條 state 軌跡，再用單獨的小型 inverse dynamics 網路把相鄰兩個 state 轉成動作，兩階段解耦。
- **對我們**：**全部調研裡跟我們問題最貼的架構先例**。明確證明「thinking/plan 只管 state（我們現在的 xy）、動作用另一個小網路事後算」是可行、已驗證的路線——等於我們現有設計（u 只編 xy）的更完整版本：把「動作」做成輕量、獨立訓練的解碼模組，而非硬塞進主 latent。

### 16. Hierarchical Diffuser — arXiv:2401.02644 `[正文級]`（Chen, Deng, Kawaguchi, Gulcehre, Ahn；ICLR 2024，GitHub 上又稱 "Simple Hierarchical Planning with Diffusion"）

- **壓縮對象**：兩層——高層對「每隔 K 步的 subgoal state」做 jumpy diffusion（壓縮的是「時間」，不是維度）；低層對相鄰兩個 subgoal 之間的完整 state-action 軌跡做 diffusion。原文：
  > "it consists of two Diffusers: one for high-level subgoal generation... and the other for low-level subgoal achievement..."
- **latent 形狀**：無額外 latent code。高層跳步 `K=4`（Gym-MuJoCo 類任務）或 `K=15`（長 horizon 任務），原文分別給出兩個數字；低層直接對相鄰 subgoal 間的完整軌跡片段建模，**同時**輸出 state 和 action（joint，不用 inverse dynamics）。
- **訓練目標**：兩個獨立 diffusion 模型分別訓練——高層對子取樣（每 K 步取一次）的 state 軌跡做標準 diffusion loss；低層對兩個 subgoal 之間的完整片段做 diffusion loss。
- **下游消費**：高層先生成一串稀疏 subgoal，低層對每一對相鄰 subgoal 生成中間稠密的 state+action 軌跡並直接執行，不需額外 inverse dynamics。
- **對我們**：「兩層 latent/表示」的直接先例——高層（對應我們現在的 xy thinking）用稀疏/跳步方式規劃，低層（對應我們要新增的動作流形）處理稠密動作細節、且可直接輸出動作而不必反推。若要讓 u 保留現有稀疏 xy 骨架、另開一個「填細節」的低層生成器，這篇的兩層切分方式（K 步跳）具體可比。

---

## 設計啟示（≤5 條，每條指向「u 包動作流形」的具體接口選項）

1. **分開兩條 latent（Decision Diffuser 範式）**：u 維持只管 state/xy，動作另開一個小 decoder 事後算（`a_t=f_φ(s_t,s_{t+1})` 式獨立 inverse-dynamics 模組），不動主 latent 的訓練——是四條線裡跟我們現狀最貼、改動最集中在新模組的選項，風險最小。
2. **u 本身就是動作流形（ASE / CALM / PULSE 範式）**：z 直接當高層輸出/thinking 座標，凍結的低層 `π(a|s,z)` 展開成動作。有具體維度參考可挑：PULSE 32 維連續、CALM 64 維超球面——兩者都是「小維度、大覆蓋率」的連續空間。
3. **逐段離散 skill/動作 token（Genie `\|A\|=8` / LAPA VQ-VAE / Moto 自回歸 motion token 範式）**：把動作流形量化成小字典，thinking 從連續向量換成一串可解讀的離散 token，語言模型化、可用 likelihood 評估軌跡合理性。
4. **串接 state+action 同一條軌跡一起生成（Diffuser / Hierarchical Diffuser 範式）**：不分離兩種 latent，直接把 pose/action 當成 xy 之外的額外維度串進同一個 u 序列；若要保留現有稀疏規劃優勢，可比照 Hierarchical Diffuser 的兩層切法（高層稀疏 subgoal、K 步跳；低層稠密填動作，且低層可以是聯合輸出不需 inverse dynamics）。
5. **訓練目標可以跟 latent 形狀脫鉤選（OPAL/SPiRL 的 VAE-KL vs TACO 的對比學習 vs ASE 的對抗+互資訊）**：不管選哪種形狀，訓練訊號至少有三種互不相同、都驗證過可行的家族可挑——形狀（選項 1~4）與訓練目標是兩個可以分開決定的維度，不必綁死。

---

_調研方法：WebSearch 找 arXiv 番號 → WebFetch `arxiv.org/abs/<id>` 取摘要 → 對關鍵設計事實（latent 維度、訓練目標公式、下游架構）用 `ar5iv.labs.arxiv.org` 或作者 GitHub README 再次 WebFetch 核對原句。16 篇全部拿到 arXiv 番號一手確認；10 篇拿到正文/專案頁層級的具體數字或公式原句，6 篇止於摘要層級（TACO/SkiMo/LAPA/Moto/PHC/AMP，均已在條目內標明哪些數字「未查得」，未用記憶值填補）。_
