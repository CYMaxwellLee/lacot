# 契約 v2-b（提案，待 lead／主人裁）

u 編碼想像軌跡。v2-b 指本輪主案「generated-plan quality + paired head exposure + EMA」；不是 v1-b 去噪主案。canonical API 見 api.py；小型 executable specification 見 fakes.py。以下規則涵蓋 production，fake 僅證明接口與梯度，不宣稱 maze 教師已合格。

## C1 輸入、dtype、數值與 R=0

`refinement_terms(refine, teacher, quality, cond, clean, sampled, *, rounds, noise) -> Terms(quality, consistency, states)`。
cond [B,C]；clean、sampled、noise [B,K,D]，B/C/K/D>0，浮點、同 device。只有三個 latent 輸入要求相同 dtype；**cond 可以不同 dtype**。autocast 下 refiner／decoder 中間輸出可為 bf16/fp16，不能因不同於 clean dtype 就 raise；loss reduction 宜 fp32。測試以 float64 recurrence 驗梯度、CPU bf16 autocast 驗相容。

rounds 為非 bool 的非負整數；shape/device/型別違反為 ValueError。hot path 不掃 tensor finiteness、不 `.item()` 判斷 finite、不因 nan/inf raise；交由 GradScaler skip，monitor 在 no-grad log 非有限事件。structural checks 不自動 cast cond 或 detach cond。不得 mutate 輸入。noise 由 caller 專用 generator 抽 N(0,.08²)，與 sampled 同 dtype/device；.08 僅玩具設定，production 按 held-out latent 尺度另定，但不能代替真 sample 臂。

R=0 合併語意：total=nf+anchor，cons/舊 act_refine/quality/exposure 都為同 device scalar 0；不除以0，不必存在 teacher/quality provider，不抽 sample、不呼叫 refine；inference 直接 decode 原 sample/injected plan。kernel R0 states=[sg(sampled)]；composition R0 states=[]（未抽樣）。保持 R2 已合併的 log keys 與有限 total；`tests_repair/test_model_zero_rounds.py` 原測試是合併必過，不能改弱。

## C2 訓練輸入與可微展開

在**當前** flow、同一 state/goal/intent/temperature/quantization 設定下取 `u0=sg(adapter.sample())`，每 optimizer step 更新；不以 clean+小噪聲取代。主臂 `u[r+1]=refine(cond,u[r])`；輔臂 `v0=u0+sg(noise)`、`v[r+1]=refine(cond,v[r])`，只補 flow 周圍局部魯棒性。兩臂每一輪都受 quality supervision，student recurrence 全 BPTT；clean/sample/noise/teacher 目標停梯度，cond/refiner 保留梯度。R=0/1/2/3 分布用 batch depth cycling 起步，正式可隨機 depth；跨深度的訓練算力需報出。

refine callable 可封裝 frozen decoder feedback。初版保留既有 RefineOperator(cond,u) 架構，decoder 透過 loss Jacobian 回饋；不把玩具 sign 特徵當已實作的 production 架構。若要把 explicit feedback 拼進網路，屬 checkpoint shape 變更，另列變更單，不偷偷改原寬度。

## C3 對本樣本有效的目標

`quality(u) -> Tensor[B]`，closure 綁本批的 state/goal/map、原樣本 mode metadata，**沒有 actions_data 或 u_target 作任意配對真值**。主案：P=D_frozen(u,s)，J=goal arrival + hold + collision + kinematic feasibility + behavior support。每項用觀測座標單位正規化、非負權重、versioned config；每輪 `Lq=mean_r (J(u[r+1])+.25 J(v[r+1]))/1.25`。production 初始建議 goal=1、hold=.25、wall=1、feasibility=.1、support=.1；它們是待校準設定，不能當實證最佳權重。障礙物與 hold 需用真 horizon mask，不能以到終點距離冒充可執行性。

主線 `_dec`、`_intent_inv`、座標 normalizer 是唯一解碼路徑；decoder、s_embed、geometry、inverse dynamics 全 freeze+eval，**不在 quality forward 套 no_grad**（仍需 ∂J/∂u）。GeoEnergy 現在在 stage2 之後建構，M2 要把供應器提前至訓練前，由 M4 提供 safe builder；不能第一步讀 None。geometry 的健康檢查需涵蓋未走過的障礙格與 mapping roundtrip。硬 VQ/FSQ 離散下若 Jacobian=0，須明列 soft/ST approximation 並重新驗部署 hard-snap discrepancy；未驗設定禁止宣稱支援。

可用 decoder feedback 直接學 J，也可同 sample 起步跑受信 optimizer 得 paired endpoint（mode/feasibility 驗證後）作輔助蒸餾；禁止隨機 batch data action/latent 作 target。玩具 oracle 以 sample 的 corridor sign 與本題 goal 定義可解地形；這是合法 same-plan optimizer endpoint，不是假定能在實際 maze 免費取得 mode。

## C4 F5 待裁 fork

**建議 EMA**：`Lc=mean_r ||u[r+1]-sg(F_ema(cond,u[r]))||²`，直接沿用 scratch:1724–1727 的定義，非 v1 反向保守步長。teacher 同步時梯度可以為0；stale 非固定點探針 R1 必有 student 梯度。quality 必啟用；consistency NEVER used alone。現有 EMA 初始化及參數更新可複用，但 student optimizer 成功才更新；AMP skip 不動 EMA，buffer 和 checkpoint/resume 也要帶版本。

備選 self：保留原 `||u[r]-sg(u[r+1])||²`，訓練有 refine 時限定 R>=2；R0 例外。R1 inference 仍可測，但不再聲稱 R1 consistency 能教 student；需申報與原 U{0..3} 訓練設計的衝突。採這案要重作 b-consistency 考場，不改頂層行為門檻。

## C5 head exposure 橋與 gauge

anchor 維持原有效資料 `(head(features(clean)), actions_data)` 及 mask/normalization；新增 `Lexp=mean_r head.nll(head(features(sg(u[r+1]))), sg(A_teacher(s,D(sg(u[r+1]))))).mean()`，權重 .25。A_teacher 是**same sampled decoded plan**的合格 frozen inverse-dynamics/controller；座標不是 actions，不拿 live head 自蒸餾冒充真值。教師 validity mask、覆蓋率、模式覆蓋和誤差都要報，無有效 labels 時 `Lexp=0` 並 `exposure/valid=false, reason=...`，合併科學 gate FAIL，不能靜默放行。

動作教師由 M4 用已有離線 transition 的 (state,next-state,action) 配對建立並在 held-out 真 transition、rollout-generated plans 驗證；ant 等非完整狀態路徑不可用 xy 差分硬當 action。最先支援 pointmaze；未合格領域保留介面並明確拒啟，不捏造已能普遍修復。玩具 action(u)=decoded x 是明定的一步 dynamics，只驗機制。

每次 gauge 用固定 held-out conditions、專用 RNG，在 no_grad 下算 `exposure/anchor_mse`、`exposure/r{0,1,3}_mse` 與 `..._gap=sample_mse-anchor_mse`，另 log n_valid/n_total、教師誤差與 quality。不可把缺值填0；使用 null + valid=false + reason。continuous 報 MSE，discrete 報 NLL/decoded action error 分開命名。gauge 不消耗訓練 RNG、不改 .train 狀態、不回傳 tensor graph。head 梯度檢查改成 `grad(total)-.25 grad(exposure)==grad(anchor)`，名稱 **anchor-gradient**；v1 的「全 head 梯度只來自 anchor」撤回。

## C6 共用組合函式及 adapter

`compose(adapter, refine, teacher, quality_factory, cond, clean, *, rounds, noise, lam_cons=.1)` 返回 `(total, tensor_logs, states)`。抽樣後調用 `quality_factory(sampled,cond)` 綁定該條 sample 的 context/mode metadata，再傳給 kernel；不能在 sample 尚未抽出時就把另一條 sample 的 target 關進 closure。
`Adapter(density,anchor,sample,exposure,features)`：density()->已正規化 scalar；features(clean)->原 head 輸入；anchor(features)->scalar 原 action loss；sample()->部署空間 sample；exposure(detached_u)->同計畫 head loss。features 可是 tuple 供 ActionMLP(cond,u)，故不可假定都扁平 concat。clean 為 head 的語意 latent；density closure 可用另一個 `_et_nf`，不能把 FSQ z 與 decoded u 混用。closure 封裝只做表示轉換，不另寫一份 loss 展開。

`total=nf+anchor+Lq+lam_cons*Lc+.25*Lexp`。三處只調用這個 total；主線 BC/div/GRPO 等非 refine loss 保留外層加總及其權重。舊 `l_act_refine=0` 表示移除錯配項；新 `l_plan_quality/l_head_exposure/l_cons` 不偷換名；在 log 邊界轉 float，gauge 是獨立 nullable 結構。lam_cons 必有限非負。附帶 checkpoint 記 objective_version、Rtrain、quality/action teacher fingerprint、cons mode/EMA state；旧 inference state_dict 可載，續訓缺 metadata 要顯式 migration，不承諾 bitwise 等價。

## C7 三個接入面與舊 API

scratch 實際主線、image actor、state actor 共用 compose；兩 actor 的 losses_given 舊位置參數／預設值保留，新增可選 keyword-only training_services 或先 attach services；R0 不要求 services。R>0 未配置 qualified oracle 清楚 ValueError，**不得 fallback 到舊錯配 loss**。state features=[cond,flat(u)]，image=flat(u)，scratch=(anchor_cond,head_clean) 並保留 _REAL_W、COND_DROP、FSQ/intent。對純 actor 的部署端不增加 teacher 依賴。

這是 v1「所有合法 caller 完全不變」的契約變更：R>0 的訓練 caller 必須提供 quality services；不支持缺 oracle 的無聲訓練。建構子與 inference 簽名不改。主線 teacher_mix 未支持的 assert 先保留，只有 M2 實接有效 mask 並考過才可移除。

## C8 科學驗收與策略獨立性

頂層 gate：固定條件、多 mode、actions_data 與 sample 独立，label permutation 不改 refine 梯度；真 flow samples 起始且 off-manifold，held-out R3 cost <.15 R0 且 <.12、mode error<.05；R1<R0、R3<=1.05 R1；identity 同門檻必 FAIL；head exposure valid 且玩具 MSE<.08。production threshold 由獨立 held-out 軌跡/實際 success 單列，不能套用玩具數值。

上述不規定 consistency 公式、不規定 action teacher 型式。EMA C4、total algebra/full BPTT/cond/noise 等 v2-b 接線考場獨立存在。six mutations 每個實際替換程式後跑梯度/數值測試，不是印六個假 PASS。R>3（含6）只報曲線與最差值，沒有無條件單調保證。
