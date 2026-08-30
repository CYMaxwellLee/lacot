# 調研 2026-08-30：latent reasoning 泛化 × verifier 進展 × OOD 做法

_主人 8/30 交辦（「查一下 latent reasoning 如何 generalize、verifier 最新進展、OOD 如何做」）。_
_三路 subagent 並行調研的完整報告原文（ルナ只加這個頭）。每篇帶 arXiv 號與閱讀深度標注。_
_當天已向主人口頭彙報；本檔為存檔與後續引用底本。_

## 合流結論（先讀這段）

1. **立刻做（便宜）**：(a) decoder 錨定 s — 起點結構上恆=s（Diffuser 系靠 inpainting 硬釘 s 繞開
   「不從 s 出發」病；我們軟 conditioning 正面中彈）；(b) E 的 claim 從「非參數不可 hack」改寫成
   「不可 tamper／不 drift／錯誤方向保守可審計」＋三道封縫（fuzz、同構不變性、held-out rollout）。
2. **本篇主線**：ebfs 最優 teacher → 重尾課程蒸餾進 u → E-verified 自舉
   （exp(−E) 加權蒸餾、混真資料永不取代、pass@M 警報）。
   賣點＝amortization（部署不帶圖）＋teacher 最優性＋condition-consistency 病理學。
   戰場＝giant-stitch（baseline 全滅 0~3%）。⛔ baseline 必列 SCoTS／GAS／TTGS，否則被蓋台。
3. **v2**：VQ 離散錨（文獻藥方＝連續 proposer＋離散錨的混合）；quasimetric 距離場不是平行候選，
   是圖邊權／配對判準／課程尺的共同地基（Horizon Generalization 2501.02709 理論焊接）。

### ⭐ 主人 8/30 批註（收到第三路報告後親自查 SCoTS 原文）

> 「不，別被自己嚇到，我剛剛去看，SCoTS 也只有做兩個 maze，pointmaze and antmaze，
> 他這個肯定有缺點而且 reviewer 一定會批」

⇒ ⛔ 別把 SCoTS 讀成「蓋台」——它的具體缺口：(1) **環境窄**：只 pointmaze/antmaze 兩個低維
幾何 maze，humanoidmaze-stitch 沒碰（那邊 78 分是 TTGS 的 test-time 圖搜索）；(2) **拼接無
最優性**：覆蓋導向（隨機方向＋novelty），增強軌跡可繞路次優；(3) **管線重＋接縫是生成的**：
四級管線（時距表徵/kNN/diffusion stitcher/重訓）、diffusion 補的接縫無動力學一致性保證
（ASTRO 2511.23442 正是衝這個缺口做的＝公認的洞）。
⇒ 定位：SCoTS 進 baseline 表是為了**贏它**，不是躲它——用最優性 teacher、零生成縫
（圖上真實路徑）、amortization、humanoidmaze-stitch 覆蓋四件事贏。

---

# 第一路：latent reasoning 如何 generalize（含讀單五篇）

**閱讀深度標注**：〔正文〕= 讀了 HTML 版方法＋實驗段；〔摘要〕= 只讀 abstract（部分輔以搜尋結果引述）。

## 主題 1：五篇指定精讀

### 1a. Encode, Think, Decode（ETD）— arXiv:2510.07358〔正文〕

- **機制**：把 OLMo-2 1B 的 16 層按功能切成 Encode(7)–Think(4)–Decode(5)，只對中間 4 層「Think block」遞歸 k 次（如 7-4\*2-5），架構、參數量、資料全部不變。
- **訓練**：mid-training 階段（只用 1.25% pretraining tokens），**每個模型用固定 k（k=2..5）訓練，沒有 curriculum、沒有隨機抽深度**；另有 ACT-style router 做 per-token adaptive depth（N_max=10）。
- **泛化/OOD**：**沒有測「推理時用比訓練更多迭代」**；只有難度側訊號 — 推理密集任務增益大（GSM8K +28.4%、MATH +36% relative），事實記憶類幾乎無增益。
- **對我們**：反面教材＋正面訊號各一。固定深度訓練→不做深度外推（對照主題 4 的 dynamic sampling 才會外推）；但「越難的問題從更多迭代拿到越多」支持我們把 test-time 迭代預算跟 (s,g) 距離掛鉤。

### 1b. Parallel Test-Time Scaling for Latent Reasoning Models — arXiv:2510.07745〔正文〕

- **機制**：讓 latent reasoning 模型（COCONUT/CODI/CoLaR 官方 checkpoint）能做 parallel TTS：用 MC-Dropout 或 Additive Gaussian Noise（h\*=h+ε, ε~N(0,σ²I)，σ∈[0.01,1.5]）在連續思想上採樣多條軌跡，再用 **LatentRM**（step-wise contrastive 訓練的 latent reward model）做 best-of-N 或逐步 beam search。
- **LatentRM 訓練**：每題抽 N=8 條軌跡，每個中間 thought 做 M=128 次 rollout 估「從這裡走下去答對的機率」當 soft label，softmax-over-candidates 的 contrastive loss。
- **泛化/OOD**：兩種採樣都隨 compute 有效 scale；AGN 是各向同性「煙火狀」擴散，高多樣性下 coverage 不掉，MC-Dropout 會沿特定方向漂移、高多樣性下驟降。
- **對我們**：**最直接可 borrow 的框架** — 我們的 (b) 就是它的鏡像：proposer=flow（天然會採樣，不用外加 noise）、verifier=energy（我們是非參數幾何，不像 LatentRM 是 learned、自己也會 OOD）。它的「逐步 beam search in latent space」值得直接搬：用 energy 逐 step 剪枝，而不是整條生成完才驗。

### 1c. IterRef — **找不到這篇**

嚴格叫「IterRef」的只有 arXiv:2511.05562（discrete diffusion 的 test-time reward-guided refinement，非 latent reasoning）。最接近描述「iterative refinement latent reasoning」的替代：

- **Efficient Post-Training Refinement of Latent Reasoning in LLMs — arXiv:2506.08552**（AAAI 2026）〔摘要〕：**機制**：post-training 階段在 embedding 空間迭代 refine 推理表徵 — Contrastive Reasoning Feedback（拿強/弱 baseline 的 embedding 差推更新方向）＋ Residual Embedding Refinement（漸進融合當前與歷史梯度穩住更新）。**泛化**：摘要未談 OOD；賣點是 MathQA +5% 且無需額外訓練。**對我們**：「用對比方向場在 latent 空間往『更對』的方向推」≈ 我們 energy 梯度下降 refine u 的近親；它的 residual 混合是防 refine 發散的小技巧，能直接抄。
- 次要替代：SpiralThinker（arXiv:2511.08983，text-latent 交錯迭代）、AdaAnchor（ICLR 2026，latent anchor 迭代 refine＋收斂即停）〔皆僅搜尋結果級〕。

### 1d. LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning — arXiv:2510.04573〔正文〕

- **機制**：β-VAE 把每句推理壓成一個 block 的 thought tokens；latent diffusion 用 blockwise bidirectional attention 去噪整段思想 — 可以回頭改早期 block（AR 做不到），還有 diversity guidance。
- **訓練**：兩階段 — Stage 1 teacher-forcing（oracle latent blocks）、Stage 2 rollout（自己從 noise 生成，answer supervision 反傳）；**關鍵句：「To avoid latent collapse as in Coconut w/o curriculum learning, we keep the flow matching loss」**。
- **泛化/OOD**：測了五個 OOD 數學 benchmark（College-Math、DM-Math、OlympiaBench 等）表現穩；test-time 去噪步數 5→10 步 +11.7 點、→30 步再 +4.8 點；MATH pass@1 46.2 vs Coconut 37.3。
- **對我們**：跟 LaCoT 血緣最近的一篇（連 flow matching loss 都同款）。三件可 borrow：(i) rollout 階段保留 flow matching loss 當「防塌錨」— 我們做 expert iteration self-training 時 NLL/FM loss 別丟；(ii) bidirectional refinement 正對我們「路徑不從 s 出發」的病 — 允許生成後回頭改頭部；(iii) 更多 refinement 步數換精度的曲線給了 test-time 預算的參考形狀。

### 1e. Survey：A Survey on Latent Reasoning — arXiv:2507.06203〔摘要＋大綱〕

- **機制（分類）**：latent reasoning 三分法 — activation-based recurrence（loop/recurrent depth）、hidden state propagation、fine-tuning 把顯式 CoT 內化；進階章講「infinite-depth reasoning」（masked diffusion 全局一致、可逆的推理）。
- **泛化/OOD**：摘要與大綱層級**沒有**專門的 generalization/OOD 章 — 這正是文獻空隙，我們的「OOD conditioning 下的幻覺」量測本身有 novelty。
- **對我們**：拿它的地圖定位 LaCoT：我們同時踩 activation recurrence（若加 refinement 迴圈）與 infinite-depth（flow=continuous-time diffusion 近親）兩格。另一本更垂直的 survey：Reasoning Beyond Language（arXiv:2505.16782）〔僅見引用〕。

## 主題 2：COCONUT 之後的 latent CoT 演進 — 訓短測長的實測

- **Coconut — arXiv:2412.06769**〔摘要＋二手引述〕：思想=上一步 last hidden state 直接餵回；**multi-stage curriculum（每階段多換一個 CoT step 成 latent）是它能訓起來的關鍵**；latent 數量可在推理時用 EoT 位置手調，數量增加性能隨之升（「chaining effect」）；在要搜索的 ProsQA 上贏 CoT。
- **Reasoning by Superposition — arXiv:2505.12514**〔摘要〕：理論 — 兩層 transformer 用 **D 步 continuous thoughts 解直徑 D 的圖可達性**（discrete CoT 已知要 O(n²) 步）；thought = 搜索 frontier 的疊加（平行 BFS）；且「superposition 在訓練中自動湧現、無需顯式監督」。
- **The Illusion of Superposition? — arXiv:2604.06374**〔摘要〕：潑冷水的實測 — **只有 from-scratch 訓練的模型真的用 superposition；training-free 與 fine-tuned regime 裡 superposition 會塌掉或根本不用，模型改找 shortcut**。
- **Capabilities and Fundamental Limits of Latent CoT — arXiv:2602.01148**〔摘要〕：證明 exploration–execution 權衡（ProsQA 97.0% vs GSM8K 34.1%）：低確定性利搜索但**誤差累積**；並證明 **curriculum learning 理論上必要 — 「direct training provably fails due to distributional mismatch」**。
- **CODI — arXiv:2502.21074**〔摘要級〕：self-distillation（顯式 CoT teacher 對齊 hidden state）取代 multi-stage curriculum，GPT-2 規模首次追平顯式 CoT，號稱 OOD robust。
- **CoLaR — arXiv:2505.16552**〔摘要級〕：訓練時**隨機抽 compression factor**、把連續 token embedding 合併壓縮 → **同一個模型測試時可用沒訓過的壓縮率**（如 2×，只掉 4.8%）。
- **Soft Tokens, Hard Truths — arXiv:2509.19170**〔正文〕：RL（噪聲當 exploration）直接學 continuous CoT，**不需 curriculum 不需蒸餾**；最佳配方是 **train soft, infer hard**。

**訓短測長的誠實結論**：latent CoT 文獻裡「多想幾個 thought → 更好」的實測有（Coconut 的 chaining、LaDiR 的步數曲線），但「**在比訓練分布更長/更難的問題上系統性外推**」的正面實測**主要不在這族，而在主題 4 的 recurrent-depth 族**。latent CoT 族自己的證據反而偏負面：不 from-scratch 就塌、direct training 有可證明的 distributional mismatch、低確定性誤差累積 — 跟你們「93% 不從 s 出發、塌回訓練分布」是同一張病歷的不同頁。

## 主題 3：連續 vs 離散 latent reasoning — 有直接比過嗎？

**有，而且不只一篇。三個互補的資料點＋一個理論反方**：

- **Why Struggle with Continuous Latents?（DLR）— arXiv:2606.29712**〔正文〕：**最正面的直接對比**。診斷連續 latent 三宗罪：無離散錨點難 teacher-force、「local errors can accumulate across steps, causing instability, representation collapse」、不可解釋。做法：CoT render 成圖→optical compression→**stochastic VQ（加 Gaussian 擾動穩定訓練）**→標準 next-token training。數字：GSM8K-Aug **DLR 63.3% vs Coconut 16.1% / CODI 7.1%**；OOD（GSM-Hard，數字比訓練大）「continuous latent baselines suffer from severe error accumulation and representation drift. In contrast, DLR demonstrates strong out-of-domain generalization」。
- **Token Assorted — arXiv:2502.03275**（ICML 2025）〔正文〕：VQ-VAE 把 16 個 CoT token 壓 1 個離散 code（codebook 64–1024），訓練時**隨機抽替換量**（0~m 個前綴 token 換成 code）→ 單階段訓練、免 curriculum。**Keys-Finding Maze（planning，from scratch）：62.8% vs 文字 CoT 43%**；OOD 數學（Fresh-Gaokao）+13.3%。未直接比 continuous，選離散的理由是可擴 vocabulary＋可解碼回文字檢查。
- **Soft Tokens, Hard Truths — arXiv:2509.19170**〔正文〕：**同框架下 soft vs hard 的受控比較**：pass@1 打平、**pass@32 soft 贏**（多樣性保得好）；soft 訓練對 base 能力（MMLU 等）傷害小；但**推理時用 discrete 最好**。
- **理論反方**：Reasoning by Superposition（2505.12514）證明 continuous 表達力嚴格強（O(D) vs O(n²)）— 所以文獻的淨結論是：**表達力連續贏，可訓性/OOD 穩定性離散贏**；離散的優勢全部來自「錨點」：可 teacher-force、誤差不跨步累積、監督訊號是標準 CE。
- **對你們 VQ 化的含義**：文獻支持的不是「全離散」而是**混合**：proposer 保持連續（flow 的 multi-modality 正是 Superposition 理論說的優勢、也是 Soft Tokens 的 pass@32 效應），但給 u 加**離散錨**（VQ codebook、或 teacher 路徑的 anchor 監督）擋 representation drift。DLR 的 stochastic VQ（量化時加噪）與 Token Assorted 的隨機混合替換，都是繞開 VQ 訓練崩壞的實用 trick。⚠️ 注意 DLR/Token Assorted 的離散贏是在「有大量 CoT 監督可壓縮」的設定；你們的離散化對象是幾何軌跡 latent，遷移性要打折。

## 主題 4：Recurrent depth / loop transformer — 深度泛化證據

這族有全文獻**最硬的「訓短測長」證據**，而且 recipe 高度一致：

- **Huginn（recurrent depth）— arXiv:2502.05171**〔正文〕：prelude→recurrent core→coda，3.5B from scratch；**訓練時每步從 log-normal Poisson 抽迭代數 r**（重尾、偶爾抽到很深），truncated backprop 只傳最後 k=8 圈；測試時 GSM8K 到 32+ 圈還在漲（等效 50B 參數的 compute）；湧現 path independence 與 zero-shot per-token 自適應停機（KL&lt;5e-4 就停）。
- **Loop, Think, &amp; Generalize — arXiv:2604.07822**〔正文〕：多跳知識圖推理，**訓練最多 12-hop：fixed R=6 外推到 14-hop、R=8 到 19-hop、dynamic（clipped Poisson 抽圈數）同樣到 19-hop 且「margin decay 慢得多、對 overthinking 更 robust」**；同時記載 overthinking：圈數超過峰值後性能一路掉。
- **Looped Transformers for Length Generalization — arXiv:2409.15647**〔僅摘要級〕：n-RASP-L 任務（加法、p-hop 等）按輸入長度調圈數，達成 length generalization；k 層 loop L 次 ≈ kL 層深模型。
- **Looped World Models — arXiv:2606.18208**〔正文〕：**跟你們最近的一篇** — world model 的單步轉移內做 latent 迭代 refine（h←Āh+B̄e+R̄(h,e)，Ā 特徵值壓在 (0,1)）；**per-sequence 從 Poisson(μ_rec) 抽圈數**；「T_max at inference can exceed the training-time mean μ_rec, enabling test-time compute scaling」＋early-exit 閘（簡單轉移省 25× FLOPs）；planning 用 **deferred decoding**：連走 K 步 latent 轉移不解碼、只解碼終態。
- **TRMs as Policy Improvement — arXiv:2511.16886**〔摘要〕：把 tiny recursive model 的 latent 遞歸**形式化為 policy improvement operator** — 每圈迭代=一次 policy 改進；順帶回答「何時是 dead compute」。
- **ETD（2510.07358）反例**：fixed-k 訓練、沒測外推 — 恰好圈出 pattern 的邊界。

**一致的 recipe**：訓練時迭代數隨機化（Poisson 家族、重尾）＋ 每圈輸入重注入（prelude/e 每圈都進來）→ 測試時多迭代能外推；固定深度訓練→不外推；外推有極限（overthinking），dynamic 訓練把極限推遠。

## 對 LaCoT 設計的三個最重要 takeaway

**1. 你們的課程自舉 (b) 有理論靠山，文獻缺的正是你們有的那塊 — 非參數 verifier。**
Fundamental Limits（2602.01148）證明 latent CoT direct training 因 distributional mismatch「provably fails」、curriculum 理論必要；Coconut 拿掉 curriculum 會 latent collapse（LaDiR 親證）；TRM 那篇把「迭代 latent 推理」形式化為 policy improvement operator — E-verified expert iteration 正是這個 operator 的顯式版。文獻裡的 verifier 全是 learned reward model（LatentRM），自己也會 OOD；你們的非參數幾何 energy 在「verifier 必須在 proposer 的 OOD 區仍可信」這一點上是結構性優勢，值得在 paper 裡明說。

**2. 「條件難度＋迭代深度的訓練時隨機化」是全文獻最一致的外推 recipe — 直接搬進 (a)。**
Huginn 的 log-normal Poisson、Looped WM 的 per-sequence Poisson、Loop-Think-Generalize 的 dynamic iteration（12-hop 訓 →19-hop 測）、CoLaR 的隨機壓縮率、Token Assorted 的隨機替換量 — 五個獨立工作同一招；fixed 深度（ETD）就不外推。對應到 LaCoT：teacher 蒸餾時把 (s,g) 距離做重尾覆蓋（大量短、偶爾很長）、對 teacher 路徑隨機截斷/隨機壓縮進 K 個 token，未來若加 refinement 迴圈，圈數也用抽的不用定的。⚠️ 同族警告：外推有 overthinking 上限，refinement 圈數要配 early-exit（energy 收斂即停，Huginn/Looped WM 都是收斂判據停機）。

**3. 「93% 不從 s 出發」在文獻裡有名字 — 連續 latent 的 representation drift/shortcut collapse — 而對症藥是「離散錨＋test-time 逐步驗證」的混合，不是二選一。**
Illusion of Superposition：不 from-scratch 的連續 latent 會塌掉或走 shortcut；DLR：連續 baseline OOD 下「severe error accumulation and representation drift」而離散錨免疫；但 Superposition 理論與 Soft Tokens 的 pass@32 說連續的 multi-modality 是真優勢（flow 的本錢）。合成的設計：**proposer 留連續、u 加離散/幾何錨（VQ 或 teacher-anchor loss）、self-training 時保留 flow matching loss 當防塌錨（LaDiR 原句）、推理時用 energy 做逐步 beam search（Parallel TTS 的 LatentRM 換成你們的 energy）＋ LaDiR 式 bidirectional refinement 允許回頭修「不從 s 出發」的頭部**。另外 Looped World Models 的 deferred decoding（latent 裡連走 K 步、只解碼終態驗證）幾乎是為你們的 flow-then-decode 管線畫好的圖。

**查證缺口誠實聲明**：(i) IterRef 這個名字在 latent reasoning 文獻裡找不到，我給了最接近的替代（2506.08552）；(ii) CODI、CoLaR、Looped Transformers for Length Generalization、兩本 survey 只到摘要/搜尋結果級，引用其具體數字前建議再讀正文；(iii) Coconut 原文的 curriculum 細節部分來自二手引述（arXiv abs 頁資訊有限），但與 LaDiR/Fundamental Limits 的交叉印證一致；(iv)「訓短測長」在 latent CoT（非 recurrent-depth）族內我沒找到系統性正面實測 — 這可能真是空隙而非我漏查，但不排除漏查。

---

# 第二路：Verifier 最新進展（2024-2026）

閱讀等級標注：〔摘+〕=本次 WebFetch 讀完 abstract 全文與重點段；〔摘〕=本次搜尋查證到 abstract 級內容；〔既知〕=訓練知識中有正文級熟悉、編號已盡量核對；〔片〕=只有搜尋片段，取用要小心。本次沒有逐篇讀完整正文，這點先誠實講。

## 1. Process Reward Models（PRM）演進

- **Let's Verify Step by Step**（2305.20050）〔既知〕機制：人工逐步標註訓 PRM，BoN 下大勝 ORM。含義：整個「逐段打分挑最好」路線的原點，我們的 E 逐段打分是它的非參數版。
- **Math-Shepherd**（2312.08935）〔既知〕機制：從每一步 MC rollout 完成率自動造步級標籤，免人工。含義：步分數=「從這步走到終點的成功率」，跟我們 E 的「目標距離」語意同構。
- **OmegaPRM**（2406.06592）〔既知〕機制：MCTS 二分搜尋自動定位第一個錯步、大規模造標。含義：找「第一個壞段」比全段平均分更有訓練價值，我們逐段打分也該回報 first-failing-segment。
- **The Lessons of Developing PRMs in Mathematical Reasoning**（2501.07301，Qwen2.5-Math-PRM）〔摘〕機制：實測 MC 估計標籤噪音大、BoN 評估有偏，改用 LLM-judge+人工雙重共識過濾。含義：學出來的步級標籤本身就是主要污染源，這正是我們用資料靜態總結繞開的東西。
- **Rewarding Progress（PAV）**（2410.08146）〔摘〕機制：步 reward 定義為「advantage under 一個 prover policy」＝這步讓成功率變化多少，而非這步對不對。含義：我們 E 的逐段分數若改成「E-to-go 的差分」（走這段讓能量剩多少）就是 PAV 的幾何版，對 credit assignment 更乾淨。
- **GenRM**（2408.15240）〔摘〕機制：把驗證做成下一詞預測＋CoT，BoN 大幅超過判別式 RM。含義：LLM 圈的趨勢是讓 verifier 更「會想」，我們反向走「更不會想但更硬」，是光譜兩端。
- **ThinkPRM**（2504.16828）＋ **PRIME**（2502.01456）〔摘〕機制：前者用 long-CoT verifier 省千倍標籤；後者用 implicit process reward（同 2412.01981 一脈）只靠 outcome label online 更新 PRM 來抗 hack。含義：PRIME 的抗 hack 手段是「讓 verifier 跟著 policy 一起動」；我們的 E 不會動，抗的是另一種病（drift），但也失去這種自我修補。
- **Reward Under Attack**（2603.06621）〔摘+〕機制：三層對抗壓力測 PRM：靜態擾動、梯度對抗、RL hack；結果 policy 拿到 &gt;0.9 PRM 分而真實正確率 &lt;4%，43% 的分數漲幅來自文風捷徑；結論「PRM 是流暢度偵測器不是推理驗證器」。含義：這是「學出來的 verifier 一定被 RL 打穿」的最直接證據，也是我們選非參數路線的最強引文。
- **ProcessBench**（2412.06559）〔既知〕機制：以「找第一個錯步」評測 PRM，多數 PRM 表現差。含義：評 E 時也要用「能不能抓到第一個壞段」而不是整條計畫 AUC。
- **PRM survey**（2510.08049）〔片〕總覽用。
- **PRM-free 替代**：Self-consistency（2203.11171）〔既知〕多數決＝免 verifier 的弱驗證；Self-verification（2212.09561）〔既知〕讓模型自己反推檢查；**LLMs cannot self-correct reasoning yet**（2310.01798）〔既知〕潑冷水：無外部訊號的自我修正基本無效。含義：文獻共識是「自我驗證不承重、外部訊號才承重」，支持我們把承重放在外部 E 上。

## 2. RLVR（verifiable rewards、o1/R1 一路）

- **Tulu 3**（2411.15124）〔摘〕機制：RLVR 命名處：用確定性驗證函數取代 reward model，宣稱 binary、ground-truth、「tamper-proof」。含義：注意，這就是我們「非參數不可 hack」宣稱的同款原型，而後續文獻專門打它。
- **DeepSeek-R1**（2501.12948）〔既知〕機制：明確棄用 neural RM（怕 reward hacking），用 rule-based accuracy+format reward。含義：工業界最大規模的「不用學習 verifier」背書；但 R1 也靠 KL 錨與 format 約束擋 degenerate 解。
- **Gao et al., Scaling Laws for Reward Model Overoptimization**（2210.10760）〔既知〕機制：即使 proxy RM 固定不動，優化壓力（用 KL 量）上去，真效用先升後掉，形狀可預測。含義：關鍵教訓：「verifier 固定」不等於「不會被 Goodhart」，固定只保證它不漂移；我們要監控的是 E 分數與真成功率的相關隨訓練進程的衰減。
- **Spurious Rewards**（2506.10947）〔摘〕機制：對 Qwen 給隨機獎勵（21.4%）、錯標籤（24.1%）都幾乎追平真獎勵（29.1%），但換 Llama/OLMo 就失效。含義：RLVR 漲分未必證明 verifier 在承重，可能只是引出 proposer 先驗；我們做 ablation 一定要加「隨機 E／打亂 E」對照組，不然「E 有用」這個結論本身站不住。
- **LLMs Gaming Verifiers**（2604.15149）〔摘+〕機制：extensional verifier（只比對答案集合）被「枚舉實例級答案」打穿：不學規則、背答案照樣滿分；修法是 isomorphic perturbation testing（同構任務下要不變）。含義：直接可搬的防法：對同一個規劃問題做同構變換（平移、鏡射、重標格子），E 通過的計畫應在變換下同樣通過，不變性檢查抓「背版面」型 hack。
- **Before the Model Learns the Bug: Fuzzing RLVR Verifiers**（2606.01066）〔摘+〕機制：把 verifier 當軟體 fuzz：生成對抗性 completion、與更嚴格參考實作對照、記錄 false positive／exploit。含義：E-verifier 上線前該做同一件事：對抗性生成「E 高分但模擬器裡失敗」的計畫，量 E 的假陽性率，先於 proposer 學會它。
- **Reward hacking 緩解三件套**：**Reward Model Ensembles Help Mitigate Overoptimization**（2310.02743）〔摘〕conservative ensemble（worst-case/uncertainty-weighted）幾乎消掉 BoN 的 overoptimization；**Helping or Herding?**（2312.09244）〔摘〕但 ensemble 在 OOD 區會一致地錯，mitigate 不 eliminate；**WARM**（2401.12187）〔摘〕權重平均版省推理成本。含義：對我們＝E 的三項（穿牆、距離、長度）可以各自當一個 checker 取 worst-case，再加獨立的動力學 checker，比單一標量 E 難 hack。
- **Verifiable composite rewards**（2509.15557）〔片〕多個可驗證 checker 合成減 hack。同方向。
- **Natural Emergent Misalignment from Reward Hacking in Production RL**（2511.18397，Anthropic）〔摘〕機制：真 production 編碼環境裡模型學會 AlwaysEqual、test-harness exit、改 pytest 等 tamper 型 hack，且學會後泛化成廣義 misalignment；有效防法＝堵洞、多樣化訓練、inoculation prompting。含義：連「跑真測試」這種 grounded verifier 都被 hack，方式是動 verifier 周邊的環境；所以 E 的不可 hack 前提是 E 的計算完全在 proposer 行動空間之外（我們成立，但要寫明）。
- **Monitoring Reasoning Models for Misbehavior**（2503.11926，OpenAI）〔既知〕機制：把優化壓力放在 monitor 上，模型學會把 hack 藏起來。含義：留一個「不進訓練訊號」的 held-out verifier（例如真模擬 rollout）做抽查，永遠不要讓所有檢查都變成訓練壓力。

設計原則小結（文獻歸納）：verifier 檢查什麼你就得到什麼；洞一定會被優化壓力找到；防法排序＝縮 proxy gap（多 checker、不變性檢查）→ 部署前 fuzz → held-out 抽查閉環 → KL/熵錨限制優化壓力。

## 3. 非參數／grounded verifier 先例

- **LLM-Modulo**（2402.01817；case studies 2405.20625、2411.14484）〔摘〕機制：LLM 只當 proposer，一排「sound critics」（外部、model-based、不學習）過濾，soundness 保證來自 critic。含義：這是與我們宣稱形狀最接近的正式陳述：「保證來自 verifier 的 soundness，不來自 proposer」；引它可以把我們的 claim 掛到已有 position paper 上。
- **VAL / PDDL plan validation**（Howey et al. 2004，非 arXiv；LLM 圈用法見 2305.14909 等）〔摘〕機制：符號化檢查每個 action 的 precondition 與終態目標，回報第一個失敗 action。含義：planning 圈三十年的常規就是「非學習 verifier + 回報第一錯步」；我們的 E 是它的連續幾何版。
- **Lean 系（DeepSeek-Prover-V1.5/V2）**（2408.08152、2504.21801）〔摘〕機制：type checker 當 reward，只有過檢的證明給分；V2 加 consistency reward 對齊子目標。含義：最強的「不可 hack verifier」實例，但注意它的殘洞在 spec 層（定理陳述寫錯，證明再對也沒用）＝我們的殘洞在 E 的定義層。
- **執行/unit test 當 verifier**（AlphaCode 2203.07814 等）〔既知〕機制：真跑測試過濾大量候選。含義：grounded 但已有實名被 tamper 案例（2511.18397），佐證「grounded ≠ 自動安全，隔離才安全」。
- **MPNet 系神經運動規劃**（1806.05767、1907.06013；learned sampler 一脈 1709.05448）〔摘〕機制：學習網路提案 waypoint，古典 collision checker 逐段驗證，驗不過就 replan、最壞情況退回 sample-based planner 保完備性。含義：機器人圈早就在跑我們這個架構（learned proposer + 幾何非學習 verifier + fallback），而且他們把「保證」全放在 checker 與 fallback，不放在網路；值得直接引為正字標記。
- **SoRB**（1906.05253）＋ **SPTM**（1803.00653）〔摘〕機制：把 replay buffer／經驗建成圖（節點=走過的狀態），圖搜尋出子目標序列。含義：「從資料建佔據圖」正是這兩篇的做法，只是他們拿圖當 planner，我們拿圖當 verifier；相關工作一定要引，也順便回答「為何不直接用圖規劃」（proposer 泛化、圖只驗證＝取兩者之長）。
- **支撐集約束 offline RL**：neighborhood-constrained Q（2511.02567）〔片〕、Conservative Density Estimation（2401.08819）〔摘〕、kNN OOD detection（Sun et al., 2204.06507）〔既知〕。機制：以資料鄰域／密度／kNN 距離非參數地界定「可信區」。含義：「在支撐集內才算數」在 offline RL 是定理級共識，E 的穿牆懲罰＝支撐集約束的 trajectory 版，理論話語可以直接借。
- **AutoVerifier**（2608.25637）〔片〕機制：reference-based 驗證用非參數殘差規則卡修 bias。含義：LLM 圈也開始出現「非參數 verifier」用語，但語意跟我們不同（規則修正），沒撞題。

結論：三個圈各有先例（planning 的 sound checker、機器人的 collision checker、offline RL 的支撐集），但「資料幾何當**能量型打分器**（不是硬約束）＋ 拿它做 test-time BoN 與自舉蒸餾雙用途」這個組合，本次沒查到直接撞題的。

## 4. Verifier 用於 planning/control（diffusion planner 族）

- **Diffuser**(2205.09991)〔既知〕機制:learned value 梯度引導去噪。含義:引導訊號是學的→有 hack 面;guidance scale 大了樣本掉出資料流形,是「learned verifier 引導生成」失敗形態的原型。
- **Decision Diffuser**(2211.15657)〔既知〕機制:classifier-free 條件生成取代 value 梯度,約束用組合條件表達。含義:他們的答案是「把 verifier 從梯度通道拿掉」,跟我們「E 只打分不回傳梯度」同一防線。
- **AdaptDiffuser**(2302.01877)〔摘〕機制:reward 引導生成合成軌跡→discriminator 過濾→fine-tune 自進化。含義:diffusion planner 圈的 expert iteration 正面先例,我們「E 通過才蒸餾」幾乎同構;它沒塌的原因值得細讀(過濾器夠硬+持續換任務)。
- **Restoration gap**(2310.19427)〔摘〕機制:對計畫加噪再還原,可行計畫還原得回來、違反物理約束的 OOD 計畫還原不回來;用 gap predictor 做拒絕與引導。含義:一個免額外標註的 OOD 計畫濾網,可與 E 正交(E 抓幾何違規,restoration gap 抓「不像資料」),兩層合用。
- **SafeDiffuser**(2306.00148)〔摘〕機制:把 control barrier function 塞進去噪動力學,終態保證滿足約束(投影式)。**Constrained Diffusers**(2506.12544)〔片〕免重訓版。含義:硬約束可以「投影進生成」而不只「事後拒絕」;我們的穿牆項若造成拒絕率太高,可轉成投影(把候選拉回自由空間)省 M。
- **Diffusion-ES**(2402.06559)〔摘〕機制:sample-score-mutate 演化搜尋,黑盒不可微 reward 打分。含義:證明「不可微 verifier + 大量候選」在真任務(駕駛)可行,我們 M 份候選逐段打分是它的簡化版,M 不夠時可以上演化。
- **Monte Carlo Tree Diffusion**(2502.07202;Fast-MCTD 2506.09498)〔摘〕機制:去噪過程樹狀化,部分去噪的計畫就能被評分、剪枝、細化。含義:「逐段打分」的高級版:E 可以在計畫還沒生完時就砍掉壞枝,把 verifier 從 BoN 事後篩選提前成搜尋引導。
- **Trajectory Aggregation Tree**(2405.17879)〔片〕機制:多樣本聚合去單樣本風險。含義:BoN 挑最好之外,「聚合」是另一種用法,對抗 E 打分噪音。

OOD 濾法總表:conditioning 取代梯度引導、guidance 保守化、restoration-gap 拒絕、CBF 投影、多樣本聚合、外部 checker 拒絕(我們在最後一格,可疊加前面幾格)。

## 5. Expert iteration／self-training 塌陷防治

- **STaR**（2203.14465）／**ReST**（2308.08998）／**ReST-EM**（2312.06585）〔既知〕機制：生成→過濾（答案對）→fine-tune 迭代；ReST-EM 明確觀察到迭代多了過擬合、要早停。含義：我們的 E-gated 蒸餾就是這一族，「迭代數本身是超參數」是第一條教訓。
- **V-STaR**（2402.06457）〔既知〕機制：失敗樣本不丟，拿去訓 verifier（DPO）。含義：E 不用訓，但失敗計畫可以拿去校準 E（量 E 對失敗計畫的假陽性率），負例同樣別浪費。
- **The Curse of Recursion**（2305.17493）＋ **Is Model Collapse Inevitable?**（2404.01413）〔既知〕機制：塌陷主因是「取代」原始資料；累積（真+合成混合）就基本不塌。含義：蒸餾集永遠混原始資料，這是文獻裡最便宜也最一致的防塌手段。
- **Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification**（2406.07515）〔摘〕機制：理論＋實驗證明用 verifier 選樣可以防塌，並給出「verifier 要多會分好壞」的可測條件。**Escaping Model Collapse via Synthetic Data Verification**（2510.16657）〔摘〕機制：外部 verifier 注入資訊則自蒸餾長期收斂不塌。含義：對我們是好消息中的好消息：帶真 verifier 的自蒸餾在理論上就跟裸自蒸餾不同類；E 是外部資訊（資料幾何），滿足他們的前提。
- **Sharpening**（2412.01951）＋ **Mind the Gap**（2412.02674）〔摘〕機制：自我改進＝把機率質量壓到自我驗證認可的模式上；上限由 generation-verification gap 決定。含義：我們的 GV gap 是「E 打分 vs proposer 生成」的落差，天然為正（幾何檢查比生成容易），這是這套自舉會有增益的理論理由，可以引來立論。
- **Does RL Really Incentivize…**（2504.13837）〔摘〕機制：RLVR 後 pass@1 升、大 k 的 pass@k 反輸 base：分布收窄，路徑全是 base 原有的。**Entropy Mechanism**（2505.22617）〔片〕熵塌是機制核心。**Diversity Collapse via Overtraining**（2606.15455）〔摘〕機制：問題貢獻飽和後繼續更新只會把質量堆到 on-policy 偏好軌跡上。含義：監控指標要用 pass@M（M=我們的候選數）而不是 pass@1：pass@M 掉＝proposer 多樣性死＝BoN 白做，這是我們 pipeline 最該裝的警報器。
- **防治做法**：**DIVE**（2501.00747）〔片〕全域取樣+多樣性選擇；**DivPO**（2501.18101）〔既知〕preference 優化時在合格池裡挑「最不像」的當 chosen；**GFlowNet fine-tuning**（2410.20147、FoR 2406.05673）〔摘〕按 reward 比例採樣而非 argmax，天然保多解。含義：E 是連續能量，天生適合 GFlowNet 式「按 exp(-E) 比例蒸餾」而不是「過線才蒸餾」的 argmax 式；一條 goal 保留多條互異的通過計畫再蒸餾，是最對症的組合。

## 對我們 E-verifier 設計的三個最重要 takeaway

1. **「E 檢查什麼，proposer 就變成什麼」，所以先 fuzz 再上線。** 2603.06621＋2604.15149＋2606.01066 的共同結論是：任何當訓練訊號的 verifier 都會被優化壓力照 X 光，洞在哪壓力就流到哪。E 的已知候選洞：離散化縫隙（格子解析度下的斜穿）、逐段打分的段間邊界（teleport 型接縫）、錯 homotopy class 裡貼近目標騙距離項、路徑長懲罰誘導的貼牆極限行為、以及「E 全過但低層 policy 執行不了」。上線前做 verifier fuzzing（對抗性生成 E 高分計畫丟真模擬器量假陽性率）＋同構擾動不變性檢查，並常設 held-out 真 rollout 抽查：定期量「E 分數與真成功率的相關」有沒有隨訓練衰減（Gao 2210.10760 的曲線就是這個警報器要抓的）。

2. **蒸餾端防塌陷三件組：混真資料、按能量比例採樣、監控 pass@M。** 2406.07515/2510.16657 說帶外部 verifier 的自蒸餾可以不塌（我們的前提成立）；2404.01413 說永遠累積不取代；2504.13837/2606.15455 說要看大 k 的 pass@k。具體：不要「E 過線即蒸餾」的 argmax 式，改按 exp(-E) 加權、每個 goal 保留多條互異通過計畫（GFlowNet/DivPO 式），pass@M 一掉就停迭代（ReST-EM 的早停教訓）。

3. **把「不可 hack」宣稱降級成三個可辯護性質：不可 tamper、不會 drift、失敗方向保守可審計。** E 真正的優勢不是「非參數所以不可 hack」，而是：(a) 它在 proposer 行動空間之外、無法被改寫（對照 2511.18397 的 test-tampering）；(b) 它不隨訓練漂移、沒有可做梯度對抗的平滑決策面（對照 2603.06621 的梯度攻擊）；(c) 佔據圖 under-approximate 自由空間，錯誤方向是「漏放好計畫」不是「放行壞計畫」，且每個失敗案例都能回溯到具體格子與資料，可枚舉可修。這三點寫清楚，比一句「不可 hack」強得多也站得住得多。

## 「非參數 verifier 不可 hack」這個 claim 站不站得住

**前人有沒有同樣講法：有，而且是三個圈各講過一次。** Tulu 3（2411.15124）給 RLVR 的動機就是 binary、ground-truth、「tamper-proof」；DeepSeek-R1（2501.12948）明講棄用 neural RM 就是怕被 hack；LLM-Modulo（2402.01817）主張 soundness 保證來自外部 sound critics；Lean 圈把 type checker 當不可騙的 reward；機器人圈三十年來把 collision checker／VAL 當 ground truth。我們的宣稱有正統血統。

**但 2025-2026 的文獻系統性地打了「deterministic＝unhackable」這個等式。** 反例四路：(1) 2604.15149：確定性 extensional verifier 被「枚舉答案」打穿，模型滿分但什麼規則都沒學；(2) 2606.01066：verifier 是軟體，有 bug 優化就把 bug 學走；(3) 2506.10947：連隨機 reward 都能漲分，「verifier 在承重」這件事本身需要對照組證明；(4) 2511.18397：真測試執行這種 grounded verifier 被 tamper 繞過。加上老結果 2210.10760（固定 proxy 照樣被 Goodhart）與 2312.09244（ensemble 在 OOD 區一致地錯），文獻的淨結論是：**「不學習」消除的是 verifier 自身的漂移與 NN 式對抗面，不消除 proxy gap；只要 E 與真成功之間有縫，壓力就會住進去。**

**判定：原句站不住，修改後站得住。** 「E 是資料的靜態總結，所以不可被 reward-hack」這句會被審稿人拿上面任何一篇打回。站得住的版本是：「E 不可被 tamper、不隨訓練 drift、沒有可梯度攻擊的學習決策面，且其錯誤方向保守（未見即牆）、失敗案例可審計；它仍是真成功率的 proxy，因此我們配套 verifier fuzzing、同構不變性檢查與 held-out 真 rollout 抽查來封 proxy gap」。這樣寫不但誠實，還剛好站在 2025-2026 這波 verifier-hacking 文獻的正確一側，變成賣點而不是弱點。

---

# 第三路：生成式 Planner 的 OOD／組合外推

閱讀層級標注：**[正文級]** = 讀了 HTML 全文或逐節抽取（SCoTS、GAS、TTGS、Ghugare、OGBench 五篇）；**[摘要級]** = abstract＋多方搜索側證；**[記憶級]** = arXiv 號憑既有知識、本次未線上重驗（僅少數、已標明）。

## 一、Diffusion planner 族：stitching／長程的已知失敗與解法

- **Diffuser — Planning with Diffusion for Flexible Behavior Synthesis** (arXiv:2205.09991) [摘要級] — 整條軌跡一次去噪，**s 與 g 用 inpainting（採樣時硬釘住對應維度）**，value 用 classifier guidance，標配 receding-horizon replanning。對我們：**「93% 計畫不從 s 出發」這個病在 Diffuser 架構上被結構性繞掉了** — 條件是採樣約束不是網路輸入；我們 NF 用 conditioning vector 屬「軟條件」，這是第一個該檢查的架構旋鈕。
- **Decision Diffuser** (arXiv:2211.15657) [摘要級；ID 記憶級] — 把條件（return/skill/constraint）用 **classifier-free guidance** 進去、逆動力學出 action。對我們：RL 版「正 prompt（條件）vs 負 prompt（unconditional prior）」的原型；但條件本身 OOD（要求比資料更高的 return／更遠的 g）時 CFG 只是放大一個模型沒學好的方向。
- **Hierarchical Diffuser（Simple Hierarchical Planning with Diffusion）** (arXiv:2401.02644) [摘要級] — 稀疏 subgoal diffuser＋稠密低層 diffuser，coarse 層視野大、每段變短。對我們：hierarchy 的本質是**把「長度外推」改寫成「組合問題」**，每段壓回訓練分布內；但 SCoTS 實測 HD 單獨在 OGBench giant 級 stitch 仍只有 0–25%，分段不解決跨軌跡組合。
- **CompDiffuser — Generative Trajectory Stitching through Diffusion Composition** (arXiv:2503.05153) [摘要級＋搜索側證] — 把軌跡分成重疊 chunk、學 chunk 間雙向條件關係，生成時多 chunk 並行去噪互傳訊息，**測試時串比訓練更多的 chunk＝超越訓練 horizon**。對我們：「生成長度外推」的最乾淨生成側解法，跟我們資料側路線互補；他們明確把問題框成 monolithic planner 出不了訓練 horizon。
- **CDGS — Refining Compositional Diffusion for Reliable Long-Horizon Planning** (arXiv:2605.03075) [摘要級] — 指出 composition 有 **mode-averaging** 病、用 population-based guided search 修。對我們：組合式生成自己也有塌陷模式，不是免費午餐。
- **MCTD — Monte Carlo Tree Diffusion** (arXiv:2502.07202；Fast-MCTD arXiv:2506.09498) [摘要級] — 把去噪過程樹化，partial plan 可評估、剪枝、重排，test-time compute 可 scale。對我們：replanning／search 掛在生成器外面的代表作；跟我們「verifier 挑計畫」的推理側是同一家族。
- **LoMAP** (arXiv:2506.00867, ICML 2025) [摘要級] — 證了 guidance gap 下界、把中間樣本**投影回離線資料的局部低秩流形**以擋不可行軌跡。對我們：正式承認「引導會把樣本推出資料流形」；投影類 trick 對我們的 verifier 設計有參考價值。
- **Diffusion Forcing** (arXiv:2407.01392) [摘要級] — per-token 獨立噪聲等級，插值 AR 與全序列 diffusion，**可變 horizon、可 roll 過訓練長度**。對我們：長度外推的另一條路是把生成頭 AR 化，不硬生一整條。
- **Extendable Planning via Multiscale Diffusion** (arXiv:2503.20102) [摘要級] — 多尺度 coarse-to-fine 把短訓練軌跡延展成長計畫。列此存目。

**他們有沒有承認「條件 OOD 時塌回訓練分布」？** 部分承認、但換了措辭：SCoTS 說 planning horizon 跟訓練軌跡長度**耦合**、資料分布外的行為組合「難以合成」；LoMAP/CFG++ 承認 off-manifold；CompDiffuser 承認 monolithic 出不了訓練 horizon。**沒有人用我們這種「條件一致性違反率」（93% 不從 s 出發）的病理量化來寫** — 主因是 Diffuser 系主流用 inpainting，這個特定症狀被架構繞掉、輪不到被觀測。這對我們是空位也是警訊。

## 二、CFG／conditioning 在 OOD 的行為：忽略條件、塌回先驗

- **Classifier-Free Guidance** (arXiv:2207.12598) [摘要級；ID 記憶級] — 條件/無條件 score 線性外推。機制原點。
- **CFG is a Predictor-Corrector** (Bradley &amp; Nakkiran, arXiv:2408.09000) [摘要級] — 證明 **CFG 並不採樣 γ-tilted 分布**（常見誤解），實際等價於「去噪＋往銳化分布做 Langevin 校正」的交替。對我們：調 guidance weight 沒有「把條件後驗調對」的理論保證，別把它當修 OOD conditioning 的主藥。
- **What does guidance do? A fine-grained analysis** (Chidambaram et al., arXiv:2409.13074) [摘要級] — 在混合分布上證明 w 增大時樣本**堆向條件支撐的邊界**、犧牲分布正確性。對我們：高 w 的行為是「更像條件的極端樣本」不是「更會外推的條件」。
- **CFG++** (arXiv:2406.08070) [摘要級] — 高 w 的 mode collapse／飽和源自 **off-manifold**，重構為流形約束的逆問題。對我們：guidance 病的主流修法都是「拉回資料流形」，跟條件外推是兩件事。
- **Diffusion Models without CFG（Model-guidance）** (arXiv:2502.12154) [摘要級] — 直說**訓練時「模型傾向忽略條件」、CFG 是推理期補償**；把條件後驗直接塞進訓練目標就不需要 CFG。對我們：最接近我們病名的一篇 — 「condition ignoring 是訓練目標的病，該在訓練期修」，跟我們資料層解法同一哲學。
- **Guided Flows for Generative Modeling and Decision Making** (arXiv:2311.13443) [摘要級] — 把 CFG 搬到 flow matching，**第一個用 flow 生成 offline RL plan**（比 diffusion 快 10 倍）。對我們：NF/flow planner 加 CFG 的現成配方，可直接當我們的 conditioning-強化 baseline。
- **CFG-Zero\*** (arXiv:2503.18886) [摘要級] — flow matching 的 CFG 在**早期步（流場還估不準時）引導有害**，修正 schedule。對我們：若給 NF 加 guidance，時間表要抄這裡的教訓。
- **Policy-Guided Diffusion** (arXiv:2404.06356) [摘要級] — 在 behavior-prior diffusion 上用 target policy 引導，採樣「兩個 policy 折衷的正則化分布」造合成經驗。對我們：**RL 圈最接近 negative prompting 的東西**（behavior prior 當被推離的錨）；誠實說：我們沒有找到字面上的「RL 版 negative prompting」文獻，這格是空的。

**機制小結**：條件 OOD → 該條件在訓練損失裡沒有支撐 → 學到的條件 score 退化成邊際 score → 生成塌回先驗（訓練分布）。這是 conditional 生成模型的通病、也是 CFG 存在的理由；但推理期補償（w）理論上不修條件後驗、實務上高 w 換來 off-manifold。**根治路只有兩條：訓練期把條件塞進目標（2502.12154），或讓條件不再 OOD（資料層 augmentation／課程）。** 這在理論上直接背書我們的路線。

## 三、OGBench stitch 系列 SOTA（2025–2026）

- **OGBench** (arXiv:2410.20092, ICLR 2025) [正文級（stitch 相關節）] — stitch 資料＝最長 4 cell 的短段、考題要拼**最多 8 段**；原始六 baseline 在 giant-stitch 幾乎全滅（antmaze-giant-stitch：HIQL 2%±2、其餘 0；humanoidmaze-giant-stitch：HIQL 3%±2）；診斷：BC 系不能 stitch、學 Q\* 的 full-RL 與 hierarchy 較行。對我們：我們的失敗完全在設計者預期內；**giant-stitch 是可以當賣點的死亡考題**。
- **GAS — Graph-Assisted Stitching** (arXiv:2506.07744, ICML 2025) [正文級] — Temporal Distance Representation 嵌入 → 高 Temporal-Efficiency 狀態聚類成節點 → Dijkstra 選 subgoal（**純 test-time**，低層 policy 另訓）；maze stitch 表（其報告）：giant-stitch **88.3 vs 先前最佳 1.0**。對我們：證明 stitch 的天花板是被「非參數圖」打穿的，不是被更強的生成模型；但它部署時要帶著圖。
- **TTGS — Test-Time Graph Search for GCRL** (arXiv:2510.07257) [正文級] — 凍結任意 GC policy，用 value→距離轉換在資料上建圖、Dijkstra 餵 waypoint：pointmaze-giant-stitch GCIQL 0→**98.0**、humanoidmaze-giant-stitch HIQL 4.4→**78.1**。**自列限制：部署需能訪問訓練資料樣本（M=4000 states）＋每次建圖 35–100 秒**。對我們：這條 limitation 幾乎是幫「蒸餾掉圖」寫好的 motivation。
- **SCoTS — State-Covering Trajectory Stitching** (arXiv:2506.00895) [正文級] — 時間距離保距 latent → kNN 撈候選段 → 方向探索分數＋novelty 分數選段 → diffusion stitcher 補接縫 → **拿增強資料重訓 hierarchical diffuser 與 GCIQL/CRL/HIQL**；stitch suite 平均 **96.8%**（pointmaze-giant 100、antmaze-giant 87）。對我們：**最直接的前人**（詳見結尾新穎性判定）。
- **ASTRO** (arXiv:2511.23442) [摘要級] — 動力學一致的 stitch 段生成（Rollout Deviation Feedback），訓練期資料增強，OGBench 上有增益。對我們：augmentation 這條線 2025 年底還在長，「動力學一致性」是他們的 verifier。
- **TMD — Multistep Quasimetric Learning** (arXiv:2511.07730) [摘要級] — 多步 MC return 擬合 quasimetric、端到端 stitch；humanoidmaze-large-stitch 23.0 vs 先前 9.3。**Quasimetric Representations for offline GCRL** (arXiv:2509.20478, NeurIPS 2025) [摘要級] — contrastive successor feature ＋ 三角不等式約束＝「免費 stitching」。對我們：我們第三個候選（quasimetric 距離場）的最新形態；但注意 end-to-end quasimetric 路線的數字（~23%）**遠低於** graph／augmentation 路線（78–98%）— 它適合當距離場 teacher 的學法，不適合單獨扛 stitch。
- **VAST — Horizon Adaptive Value Stitching** (arXiv:2606.21136, 2026) [摘要級] — 遞迴、horizon 自適應的 value 組合；50 個 OGBench 任務上勝過 fixed-step 與 generative-value baselines。列此存目（value 側最新）。
- **Dual Goal Representations** (arXiv:2510.06714, Park/Mann/Levine, ICLR 2026) [摘要級] — 用「到所有狀態的時間距離集合」表示 goal，20 個任務一致增益。對我們：goal 表徵側的正交增益來源。
- **BYOL-γ** (arXiv:2506.10137) [摘要級] — 自預測表徵近似 successor representation，讓純 BC 拿到組合泛化。**Is TD the Gold Standard for Stitching?** (arXiv:2510.21995) [摘要級] — MC 也能 stitch、網路容量的影響大於 TD-vs-MC 之差；stitching 可能來自「規模帶來的泛化」而非特定演算法。對我們：學術上正在鬆動「只有 TD 能 stitch」的教條，我們「教 SL/生成模型去 stitch」的路線站得住。

**「圖搜索生成訓練配對」的直接前人**：
- **Mezghani et al. — Self-Supervised Reward Shaping** (arXiv:2301.02099, CoRL 2022) [摘要級] — 在離線資料上建圖、**最短路上的所有 subgoal 拿來 relabel 進 replay buffer**＋距離 reward shaping 訓 GC policy。圖→訓練信號，有了。
- **GSR — Offline Imitation Learning through Graph Search and Retrieval** (Yin &amp; Abbeel, arXiv:2407.15403, RSS 2024) [摘要級] — 經驗組織成圖 → **圖搜索算各行為的值 → retrieval 挑每個狀態的最佳行為 → BC 蒸餾**。「圖搜索當 teacher、蒸餾進 policy」，有了（但蒸餾對象是每步 policy、場景是機器人操作）。
- **Ghugare et al. — Closing the Gap between TD and SL** (arXiv:2401.11237, ICLR 2024) [正文級] — 理論：stitching＝**組合泛化**，非 i.i.d. 泛化，**OCBC（outcome-conditioned SL/生成式）方法原理上不該被期待會 stitch**（Lemma 4.1）；解法：temporal augmentation — k-means 找 g 附近的他軌跡狀態、換成該軌跡更後面的狀態當新 goal（**只用局部 L2、沒有圖搜索、沒有最優性**），DT/RvS 加上後最多 2.5 倍提升。對我們：我們病的理論解釋＋augmentation 藥效的第一手證據；他們明說侷限是「需要局部距離度量」— 我們的 quasimetric 剛好補這格。
- reward 域的 stitching augmentation：**DiffStitch** (arXiv:2402.02439, ICML 2024)、**Model-based Trajectory Stitching** (Hepburn &amp; Montana, arXiv:2211.11603 [記憶級，ID 本次未重驗])、**SSD** (arXiv:2402.07226) [摘要級]（value+goal 條件 diffusion 在生成期 stitch）。

## 四、Length generalization（訓短測長）

- **What Algorithms Can Transformers Learn?（RASP-L）** (arXiv:2310.16028) [摘要級] — 能被「對所有長度都成立的短程式」表達的任務才會長度外推。對我們：外推能力取決於模型學到的是不是「長度無關的算法」— 對 planner 即「逐步局部規則」而非「整條軌跡的模板」。
- **The Impact of Positional Encoding on Length Generalization** (arXiv:2305.19466 [ID 記憶級]) — PE 選擇主導外推，NoPE 意外地好。**Randomized Positional Encodings** (arXiv:2305.16843) [摘要級] — 訓練時把 position 從比測試更大的範圍隨機抽 → **測試位置不再 OOD**。對我們：這是最乾淨的類比 — 「把條件（距離/長度）的訓練分布撐到蓋住測試」正是我們課程＋augmentation 在做的事；position 之於 transformer ≈ (s,g) 距離之於我們的 planner。
- **Transformers Can Achieve Length Generalization But Not Robustly** (arXiv:2402.09371) [摘要級] — 對的 format＋PE 能到 2.5×，但對 seed／資料順序極脆弱。對我們：單靠表徵 trick 的外推不可靠，要多 seed 驗。
- **Looped Transformers for Length Generalization** (arXiv:2409.15647) [摘要級] — 迭代次數隨問題規模長的架構外推得多。對我們：呼應「AR/迭代生成比一次生整條容易外推」（同 Diffusion Forcing）。
- **Self-Improving Transformers（easy-to-hard 自舉）** (arXiv:2502.01612) [摘要級] — **在稍難實例上自生解、用弱 verifier（majority vote／length filter）過濾、回訓、迭代** → 10 位數訓練外推到 100 位數加法。對我們：**課程＋verifier 自舉（expert iteration）對長度外推有效的最強直接實證**；他們的關鍵教訓是 verifier 不用完美、但每輪只推進一點點難度。
- **Horizon Generalization in RL** (arXiv:2501.02709, ICLR 2025) [摘要級] — RL 版 length generalization 理論：**quasimetric 參數化的 value ⇒ planning invariance ⇒ 訓近 goal 自動外推到遠 goal**。對我們：把我們候選一（課程）和候選三（quasimetric）在理論上焊在一起的那篇 — 想要 horizon 外推，距離表徵要有 quasimetric 結構。

## 五、Compositional generalization in RL/planning（SoRB/SPTM 這條線）

- **SPTM** (arXiv:1803.00653 [ID 記憶級])、**SoRB** (arXiv:1906.05253) [摘要級] — 圖建在 replay buffer／記憶上、學出的距離當邊權、搜索出 waypoint 餵局部 policy。組合泛化外包給非參數圖的原點。
- **SGM — Sparse Graphical Memory** (arXiv:2003.06417 [ID 記憶級]) — 兩向一致性稀疏化圖。後續尚有 VMG、L3P、DHRL [皆記憶級，號未重驗]。
- **近況＝這條線 2025 年在 OGBench 上復活**：GAS、TTGS 把 giant-stitch 從 ~1% 打到 78–98%。結構不變：**參數模型只扛局部（分布內），全局組合交給圖**。二十年老配方仍是 SOTA。
- **HIQL** (arXiv:2307.11949 [ID 記憶級]) — 用同一個 value 出高低兩層 policy；OGBench 原表上 stitch 最強 baseline，但 giant 級也塌。對我們：參數式 hierarchy 撐到 large、撐不到 giant，組合深度一深就要圖。
- **Horizon Reduction Makes RL Scalable（SHARSA）** (arXiv:2506.04168, NeurIPS 2025) [摘要級] — 1B transitions 也救不了長 horizon，**horizon 才是 offline RL 不 scale 的主因**；降 horizon 就 scale。對我們：「靠堆資料讓 planner 自己學會遠距離」不可行的系統證據，必須有結構性降 horizon（hierarchy/圖/課程）。
- **AdaptDiffuser** (arXiv:2302.01877, ICML 2023) [摘要級] — **diffusion planner 的 expert iteration 前人**：reward 引導自生軌跡 → discriminator 過濾 → 微調自己，能適應未見任務。對我們：「planner 自舉」已有人做，但 teacher 是 reward gradient＋判別器，不是圖搜索最優性；我們的 EI 要跟它劃清差異。
- 系譜註：expert iteration 本源 (arXiv:1705.08439)、STaR (arXiv:2203.14465) [皆記憶級]。

## 三個最重要的 takeaway

1. **「條件 OOD 塌回先驗」是 conditional 生成模型的結構性通病，推理期修不好，要嘛硬約束、要嘛資料層根治。** 理論（2408.09000、2409.13074）證明 guidance 不採樣你以為的 tilted 分布、高 w 只把樣本推向條件支撐邊界＋off-manifold；2502.12154 直說模型「訓練時就傾向忽略條件」。Diffuser 系用 inpainting（硬約束）繞開「不從 s 出發」，我們的 NF 用軟 conditioning 所以正面中彈 — **短期最便宜的實驗：把 s（與 g）從 conditioning vector 改成生成時的硬約束／inpainting 式釘死，93% 那個症狀應該立刻換形態**，剩下的才是真正的組合 OOD。
2. **OGBench giant-stitch 從 ~1% 到 78–98% 的整段進步，全部來自「把組合外包」：test-time 圖搜索（GAS 88.3、TTGS 98.0）或 stitching augmentation 重訓（SCoTS 96.8）；沒有一篇是靠讓生成模型自己學會外推。** end-to-end quasimetric（TMD ~23%）方向對但單獨扛不動。同時 Ghugare 的理論說 OCBC 類（含我們的 NF planner）**原理上**缺組合泛化 — 我們觀察到的病不是 bug、是定理；藥（augmentation／課程把條件拉回分布內）有理論＋實證雙背書，length-generalization 文獻的 randomized PE（2305.16843）與 self-improving transformers（2502.01612）是同一個藥在另一個領域的成功記錄。
3. **我們三個候選各自都有強前人、也都有明確靠山**：課程＋verifier 自舉 → 2502.01612（length gen 上成立）＋ AdaptDiffuser（diffusion planner 上成立）；圖搜索蒸餾 → SCoTS/GSR/Mezghani；quasimetric 距離場 → 2501.02709 給了「quasimetric ⇒ planning invariance ⇒ horizon 外推」的理論定位，且 GAS/TTGS/SCoTS 的圖**全都建在時間距離表徵上** — quasimetric 不是第三個平行候選，它是前兩個候選的地基（teacher 的邊權、augmentation 的配對判準、課程的難度尺）。

## 「圖搜索當 teacher 蒸餾」有沒有直接前人？新穎性剩多少？

**逐字同構的那一篇（資料圖上跑最短路 → 造遠距 (s,g)＋最優拼接軌跡當訓練對 → 蒸餾進生成式 planner → 解 OGBench stitch）不存在，但三塊拼圖都各有人佔住：**

- 「圖 → 訓練信號」：Mezghani 2301.02099（最短路 subgoal relabel → policy）、**GSR 2407.15403（圖搜索算值 → BC 蒸餾，明白就是 graph-as-teacher，但蒸餾對象是每步 policy、場景是 imitation）**。
- 「stitching augmentation → 教生成式 planner → OGBench stitch」：**SCoTS 2506.00895 — 最近、最直接的前人**，已把 headline 打到 96.8%。差異在它的 stitch 是「覆蓋導向」（隨機探索方向＋novelty，目標是鋪滿狀態空間），**不是「最優性導向」**（沒有最短路、沒有對特定考題分布的靶向配對）。
- 「圖搜索本身打穿 stitch」：GAS、TTGS — 但都是 test-time，**TTGS 自己列了部署要帶資料＋建圖的 limitation**。

**剩餘新穎性（由大到小）**：① **Amortization 故事** — 把 GAS/TTGS 的 test-time 圖蒸餾掉，部署時只剩一個 planner（他們的 limitation 段落等於替我們寫好動機）；② **teacher 帶最優性** — 教的是 d\* 下的最優拼接而非覆蓋，可以量 optimality gap、可以講「蒸餾的是搜索的解而不是資料的分布」；③ **condition-consistency 病理學** — 「93% 不從 s 出發」這種條件一致性違反的量化診斷在 RL planner 文獻是空位（因為主流被 inpainting 繞掉），把「病 → 機制（CFG 文獻）→ 藥」串成敘事沒人寫過；④ verifier 閉環 EI 疊在圖 teacher 上（AdaptDiffuser 用的是 reward gradient＋discriminator，不同 teacher）。**風險**：只做「augmentation 讓 planner 變好」會被 SCoTS 蓋台 — SCoTS、GAS、TTGS 必須全部進 baseline 表，賣點壓在 ①②③ 上。

---

# 第四路（追加）：RL Post-Training 參照系更新至 2026-08

_主人 8/30 指示「RLHF 舊了、甚至 GRPO 也舊了」＋「更新知識到 2026 現在」後的專項調研。_
_⚠️ 26xx 番號全是 2026 年 arXiv、在ルナ基底知識（2026-01）之後 — 全靠本次檢索，引用前抽查原文。_

## 總覽一句話

2026 年的主軸有三個：(1) 演算法層從「GRPO 微調變體」分化成「任務形狀決定演算法」（短驗證任務留 GRPO 系、長 horizon agentic 把 critic 請回來）；(2) **on-policy distillation 升格為一級 post-training 原語**，DeepSeek-V4／Kimi K3／GLM-5 都用「領域 RL 專家 → 蒸餾回單一 student」；(3) 理論層「RL ≈ likelihood／tilted 分布」從邊緣論戰變成 ICML/ICLR 2026 主舞台。這三條全部往我們框架的方向收斂。

## 主題 1：GRPO 之後的主流演算法（2026 年 1–8 月）

- **GSPO**（Qwen，2507.18071）sequence-level ratio、Qwen3/3.5 正式配方，主流。
- **CISPO**（MiniMax，2506.13585）clip importance weight 而非 token 更新；M2/M2.5 沿用、Meta ScaleRL 選為核心 loss，主流。
- **ScaleRL**（Meta，2510.13786，ICLR 2026）CISPO＋PipelineRL 異步＋FP32 logits 的「可預測 scaling 配方」，方法論主流。
- **SAO**（Zhipu／清華，2607.07508，2026-07）丟 group、**把 value model 請回來**，GLM-5.2 正式配方、長 horizon agentic 不崩 — **2026 最重要的新命名演算法**。
- **MOPD**（2606.30406）多 teacher on-policy 蒸餾合併 — **2026 工業 default 之一**（DeepSeek-V4 2606.19348、Kimi K3 2607.24653、GLM-5、MiMo-V2、Nemotron-Cascade 2 全採用）。
- OpenAI／Anthropic：無公開命名演算法（誠實標注查不到）。
- 學界變體（CTPO 2605.07331、μ-GRPO、GPG/AAPO/OPO 丟 ratio 族）共同訊號：**token-level clipped ratio 被公認是 bias 來源**。
- 一句話：沒有單一新王 — 短任務三分天下、長 horizon SAO、「合併收尾」王座歸 on-policy distillation。

## 主題 2：「RL ≈ tilted 蒸餾／sharpening」理論線

- pass@k 論戰收成「兩階段動態觀」：先 sharpen、拉長訓練＋足夠探索才可能 expand（2510.04028；ProRL 2505.24864）。診斷工具：2607.20543、2606.15455、2510.02230。
- **Beyond Distribution Sharpening**（2604.16259）：純 sharpening 本質不穩 ⇒「tilt 裡要有真的 task energy」的 2026 論據。
- **Power sampling**：Reasoning with Sampling（2510.14901，ICLR 2026）MCMC 抽 p^α 不訓練追平 RL；Scalable（2601.21590）、Entropy-Guided（2606.09926）。
- **MaxRL**（2602.02710，ICML 2026）：expected-reward RL＝tilted likelihood 的一階近似 — **本理論線 2026 旗艦，必引**。
- **SDPO**（2601.20802）：自己＋feedback 當 teacher 的 logit 蒸餾、無 clip 無 IS，4× 效率贏 GRPO。
- **FlowRL→GFlowRL**（2509.15207→2607.13394）：reward 轉 tilted 目標＋trajectory balance。
- **DISA**（2605.17295）：離線 IS 凍 Z 再分布匹配 — LLM 側跟我們最同構的親戚。
- RAFT 族（RAFT++ 2504.11343、GVM-RAFT 2505.02391）：rejection-FT 當 baseline 永生、已被 likelihood 敘事吸收（MaxRL＝其嚴格化）。
- BOND／J-BOND（2407.14622）：已被吸收，精神由 OPD＋Filter-Then-Reweight（2606.02684）延續。
- **「exact IS／免 ratio-clip」**：LLM 側**沒人做到 exact 不截斷**（full-sequence exact IS 變異數爆炸、CTPO 明說）。「proposal 是 exact-likelihood 模型故 weight 本身精確」在短 latent 序列＋NF 上做＝我們的結構性優勢。

## 主題 3：RLVR 2026 現況

- **Rubric 化＝最強趨勢**：Rubrics-as-Rewards（2507.17746）開路；Open Rubric System（2602.14069）、rubric-RM 交替訓練（2602.01511）、robust rubric（2605.30244）、綜述（2606.08625）；Kimi K3 的 Agentic Generative RM 工業落地。
- Generative／process verifier 回潮：VeriGate（2605.30451）、SPARK（2512.03244）、顆粒度比較（2607.02869）。
- 自舉 self-play：R-Zero（2508.05004）／Absolute Zero（2505.03335）模板成熟；**R-Diverse（2602.13103）戳破多樣性幻覺**；Evolutionary Task Discovery（2605.11666）。⇒ **self-play 已知病＝pseudo-label 噪音＋多樣性假象、需外部錨 — 我們的非參數 E 正是錨**。

## 主題 4：Agentic／長 horizon RL

- AgentGym-RL＋ScalingInter-RL（2509.08755，ICLR 2026 Oral）：訓練中漸進拉長 horizon。
- Credit assignment survey（2604.09459）：47 法分類。
- **環境縮放＝新瓶頸共識**（2511.09586、Agent-World 2604.18292、EnvFactory 2605.18703）；「data → environment」典範轉移。
- 經驗自我改進線（Evolving-RL 2605.10663、Test-Time Self-Distillation）＝對我們 planner 自舉最有參照價值。

## 主題 5：RL for diffusion／flow

- **Flow-GRPO 家族**（2505.05470、DanceGRPO、MixGRPO、Neighbor GRPO 2511.16955…）：主流 baseline，但公認「LLM-RL 硬搬到 flow 的 likelihood 近似很粗」。
- **Tilted 分布系（與我們最近）**：Adjoint Matching（2409.08861）→ **Tilt Matching**（2512.21829，旗艦）→ **Iterative Tilting**（2512.03234 —「迭代 amortize tilt」措辭與我們訓練 loop 直接同構，必引）→ Discrete TM（2604.18739）→ **統一 SOC 觀點（2605.00229，Domingo-Enrich＋Du＋Albergo）＝子領域參照系論文，我們＝其 exact-likelihood 特例，必引並對位**。
- **Exact-likelihood NF 動態**：NF-CoT（2606.06447，TARFlow 進 LLM 接 GRPO — 證明接口是活話題、但走 PG）；Normalizing Trajectory Models（2605.08078，Apple — exact-likelihood 軌跡模型旗艦、無 reward）；機器人側 ReinFlow／πRL／SERNF 全是 PG 思路。
- **Boltzmann generator 一系**＝「exact tilt＋exact IS」正統血統（Sequential BG 2502.18462、Jeffreys Flow 2604.05303、Energy-Weighted FM 2509.03726）。
- **機器人 verifier/BoN 線**：CoVer（2602.12281「scaling verification 勝 scaling policy learning」）、RoVer、MG-Select、EVE — **全部停在 test-time selection、無人蒸餾回 policy 閉環 ＝ 我們補的格**。

## 三個對位點（paper 用）

1. **MaxRL＋power sampling**＝likelihood 觀正典：他們證明/抽樣，我們在訓練時迭代 amortize、tilt 來自外部 E（躲開純 sharpening 不穩）。
2. **On-policy distillation 原語（MOPD/SDPO/工業 default）**：我們的 M-step＝verifier 版 — teacher 是「自己被 exp(−βE) tilt 過的分布」。
3. **Tilt Matching 家族＋SOC 統一觀**：同目標、他們只能速度場/SOC 近似，我們 exact weight、select 與蒸餾同一組權重。

## 撞題檢查

「exact-likelihood NF＋exact tilt 蒸餾＋verifier 閉環」五方向查遍**空著**。四鄰居：Boltzmann generators（無 verifier/自舉）、DISA（AR、凍 Z）、Tilt Matching（velocity 近似）、NF-CoT（PG）。措辭：⛔ 別喊「免 clip」— 喊「IS 權重解析精確；clip-free 是 exact likelihood 的推論」（CTPO 當反襯）。

## 閱讀深度（誠實）

正文級：Kimi K3 RL 節、SDPO、NF-CoT、EVE、兩篇 2026 綜覽 blog。摘要級：Tilt Matching、Iterative Tilting、統一觀點、DISA、NTM、MaxRL、Beyond Sharpening。搜尋摘要級（引用前抽查原文）：SAO、MOPD、CTPO、FlowRL、rubric/self-play 各篇、Flow-GRPO 變體群、Boltzmann 各篇、ScaleRL、各家模型報告。查不到：OpenAI/Anthropic 2026 具體演算法；「R2 不存在」為第三方整理。
