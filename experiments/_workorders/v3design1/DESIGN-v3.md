在設計什麼：在 offline 紅線內，用學出來的機制取代已判死的手刻 GeoEnergy 爬坡／挑選，主攻 oracle 包絡的 16 點選擇器空間（5.5% oracle vs 21.5% 現況）。

## (a) v2 逐模組判決對照與 v3 邊界

本件為 F2 架構官 v3 實施提案，供 lead 驗收後呈主人裁；完成的是 spec，未宣告方法有效或准予部署。沿用 v2 四模組與契約骨架，主要變更是撤掉 GeoEnergy 品質教師、加入 learned selector、把品質資格與觸發時機拆開。以下「建議／擬定門檻／預算」均為本件設計，不冒充既有裁決。

引文約定：`M/`＝本件 `materials/`；`P/`＝`materials/proposal-v2/`；其他路徑自 repo 根目錄起算。行號為本次唯讀核對版本；文獻只轉述提供的證據包，未另查網頁或重現論文。對抗官的「identity 最優阻斷」「主線 inline 繞過阻斷」與實驗的「開環稅三臂刀」「四步效果命令刀」一律用全名。

| 承重判決 | v3 採納及適用範圍 | 出處 |
|---|---|---|
| 開環稅三臂刀 | 教師動作開環失敗 17.5%；閉環字典 21.5%；字典開環 46%→閉環 21.5%，既有回授已值 24.5pp。主攻選擇／重規劃；不能把 17.5pp 解讀成任何閉環都不可突破的下界，也不能把相减的約 4pp 當獨立因果效應。 | `M/CONVERGENCE.md:69-83`；後兩項為本件對比較條件的限制 |
| oracle 包絡 | 同題 schedule／nn／G16K16 共同失敗 11/200＝5.5%，對 21.5% 差 16pp；是事後可選空間，不是已存在的事前 selector。**16 點不是 16 個候選**。 | `M/CONVERGENCE.md:14-17,80-81` |
| 四步效果命令刀 | 效果選字救 6 壞 135、累積誤差回授救 1 壞 140；判死四步 SE(2) greedy 換卡語言，未判死換卡。 | `M/CONVERGENCE.md:85-93` |
| 碼後果差分刀 | gait 的 E_Δ=.27/.32/.30，碼有效且 ŝ 可讀；xy/yaw 不合格，不能當已可用的方向教師。「頭病可修」是修理方向，非已修好。 | `M/CONVERGENCE.md:112-131` |
| 擾動雙胞胎刀 | 分布外主格 N-benefit=-.135；交叉項 -.176 方向正確，d²=.311 蓋過修正；44 真修正／60 修被吃／56 外推。增益校準、阻尼、退回名義仍未裁。 | `M/CONVERGENCE.md:133-145` |
| 今日 J-signal | climb .350→.050、.395→.030；select 相對原樣 -2.5/+1.0pp。停止 GeoEnergy 梯度爬坡與 GeoEnergy@8 挑選；不能推成所有 learned score 或所有選擇法都無效。 | `M/J-SIGNAL-VERDICT.md:5-17` |
| 判死用尺可靠 | lead 三證確認 post-F6：0916c97 14:30:37 早於提交 15:03:10、舊碼 guard 六臂全過、現碼含線段內插；知識官早先的時序空手已由此補齊。 | `M/J-SIGNAL-VERDICT.md:19-25`；對照 `M/KO-INJECT-0928.md:43-55` |

**不可混帳**：16pp 出自 ant 的零漂移 leg-1；J-signal 出自 pointmaze-large-stitch、5 task×40 episodes×兩 checkpoint。原 relay 會從本題教材軌跡取真字，nn 亦查同條教材座標，故包絡還帶候選来源特權；部署候選未必包含同一批解。先驗相同教材協定下的事前選擇，再獨立驗部署可得候選；pointmaze 成功不能記作追回 ant 16pp。〔`M/CONVERGENCE.md:14-17,52-57`；`experiments/j_signal_trial/README.md:26-35`；`experiments/walk_verify/teacher_relay/run_teacher_relay_byleg.py:103-140`〕

| v2 模組／契約 | 今日 J-signal：支持／推翻／未觸及 | 四把刀、對抗官與 v3 承接 |
|---|---|---|
| mod-actor：三入口 compose、clean/sample 分離、head exposure | **未觸及**接線正確性；不可把 today 陰性當 actor 已修 | 保留；對抗官主線 inline 繞過阻斷仍須 production 銷項。新增 selection/depth 推論接點，不能只改 model.py。〔`P/mod-actor/TASK.md:3-13`；`M/ADVERSARY-SUMMARY.md:8-11`〕 |
| mod-objective：flow＋own iterates、逐輪 same-plan quality、full BPTT | **推翻**以既有 GeoEnergy 可直接擔任有效 J 的落地前提；**未觸及**learned update 構造 | 保留真 sample、自身迭代和有效梯度；把 quality 改為離線學出的動作後果模型。identity 最優阻斷不靠換 loss 權重消失。〔`P/ARCH.md:11,21-27`；`P/contracts/CONTRACT.md:14-26`；`M/ADVERSARY-SUMMARY.md:4-7`〕 |
| mod-verification：多 seed、identity、scaling、預算對照 | **支持其必要性**：proxy 成績不能替代實際執行；**未證明**任何 learned arm 有效 | 保留六 mutation、真三入口、off-manifold、R0/1/3/R6；新增 candidate oracle gap、四關與兩段觸發驗收；GeoEnergy 歷史敗臂只入帳不重跑。〔`P/mod-verification/TASK.md:5-13`；`M/J-SIGNAL-VERDICT.md:14-25`〕 |
| mod-oracle：合格 frozen quality/action teacher、validity/gauge | **推翻**「修好牆圖即可相信 quality」；**未觸及**same-plan teacher 介面 | 擴為 learned consequence/quality 供應器；gait 可作候選資訊，xy/yaw 須修理重驗；coords 不等於 action，缺教師禁 R>0。〔`P/mod-oracle/TASK.md:5-9`；`M/CONVERGENCE.md:122-131`〕 |
| contracts：dtype、R0、EMA、梯度歸屬、metadata | **未觸及** | 保留 API 與數值規則；C3 品質定義、C5 教師资格、C8 production gate 需升版，EMA 待裁不默改。〔`P/contracts/CONTRACT.md:5-59`〕 |
| integration：wiring／current inline／reference 三種考場 | **未觸及**既有考場結果 | fake wiring PASS≠production；toy 只在小 MLP＋sign feedback 全 seed 過，真 RefineOperator 700步×3seed 的 seed0 FAIL 未解。v3 必接實際三 caller 重考。〔`P/README.md:19-21`；`P/EVIDENCE.md:210-216`〕 |

知識注入主題①逐條落位：五坑＝identity→(c)輸入；inline→(c)檔案與 trace；幅度候選未裁→(b)；toy 限制→上表與(d)；F6 時序→上表已補證。三先備＝四項待裁→(b)；FIX3 可復用但頭本體未修→(c)/(d)；同名異物→全名口徑。三輪子＝v2 接口考場沿用；shat_probe 子空間／位移／存權重沿用；GeoEnergy 僅保留診斷工具，不再產學習或選擇標籤。〔`M/KO-INJECT-0928.md:11-87`〕

## (b) 待裁四項：逐項建議、代價、退路

四項原文出處為 `P/CHANGE-ORDER.md:24`。以下均為呈裁建議；「現在就救 learned refine」的大方向已裁，不等於下列細節已裁。人日與算力為規劃估算，無本機吞吐實測支撐。

| 待裁項 | 建議與具體代價 | 否決／失敗後的處置 |
|---|---|---|
| 主案 LePlanner-style vs ReflexFlow／DMD | 採 **v2 LePlanner-style generated-plan、逐輪監督主架構＋learned quality**；先驗 selector-only 的低成本增益，再增 refiner。代價：新增離線後果模型、selector 及校準，約 4–7 工程人日；每次訓練要展開並反傳模型後果，部署增加 N 候選評分。 | 品質資格或真 operator 不過，停該臂；不拿 sign toy 補 PASS。ReflexFlow 留到有同樣本合格 endpoint；DMD 留到有條件 target/fake score 的另案，不能視為免費替換 loss。 |
| EMA vs self／R≥2 | 建議 EMA；沿用 existing CONS=ema 語義，quality 必為主訊號。新增成功 optimizer step 才更新 EMA、buffer／resume 保存，約 0.5–1 人日；多一份 refiner 權重與 teacher forward，不多一份 optimizer。 | 若裁 self，R>0 訓練只准 R≥2，R0 合法；R1 推論另驗。改 depth 訓練分布且重考 consistency；任何方案都不得單用 consistency。 |
| 先救 pointmaze，其他 actor 缺 oracle 禁 R>0 | 同意 **pointmaze-first learned refiner**，三 caller 同步接線；另以 **ant selector-only、R=0、凍結現有執行器**正面攻 16pp。它不產新 latent、不訓 ant refiner，不繞過禁令；仍須有合格 action-conditioned 後果 predictor。 | pointmaze 教師未過就不開 refine；ant selector 可獨立通過或失敗。ant／image 的 R>0 一律等各自 quality＋action teacher 資格，不能用 xy inverse dynamics 冒充。兩域各自 200 題評估，成本不能合併省略。 |
| same-plan action teacher 的資料／校準成本 | 批准作為 pointmaze refine 前置：既有 offline 窗配對訓 qualified controller ensemble；有可靠物理時間 adapter 才可用逐步 inverse dynamics（見(c)）。按 trajectory 分 train/calibration/test，保存 validity、誤差與 coverage。建議 2–3 人日、3 個 teacher seeds，算力併入(d)教師預算；不採購新 simulator 標籤。 | coverage不足則拒啟 refine／head exposure，selector-only 保留。若 teacher 的條件資訊不足，補完整 offline state 或縮小支援域，不以 live head 自蒸餾或座標差分偽造動作真值。 |

主案理由只到「值得試」：LePlanner 提供 PushT 配對 50 集、逐 block 重規劃 94% vs CEM 34%；執行全部五 block 卻是 88% vs 90%。iter5 改善、iter6 惡化支持深度 gate，不保證 LaCoT 會贏或越深越好。TARFlow 精確 inverse roundtrip 幾乎無 drift，ReflexFlow endpoint coupling 不能直接照搬；DMD 多兩個 score 系統且有 mode dropping 風險。〔`M/refine-evidence-last.txt:41-51`；`P/ARCH.md:9-17`〕

| 提案臂（含備案） | 一行合規論證 |
|---|---|
| learned selector-only | predictor 只學 offline 真 transition／重建；selector 只學 frozen predictor 對同條候選的預測，不接 oracle 贏家或環境成敗。 |
| nominal／uniform／sampler／既有 BoN／Wφ 直接評分對照 | 封存既有模型與離線定下的設定，不以本輪環境結果再訓練或調參；事後 candidate/time oracle 僅算上限，永不當教師。 |
| learned refiner＋same-plan exposure | flow 真 sample／own iterates 作輸入；梯度只來自 offline 學出的 frozen quality 與同計畫 action teacher；環境 rollout 僅外部考試。 |
| EMA 主案／self 深度備案 | teacher 都是模型自身副本／迭代，配合合格 offline quality；不產生環境 supervision。 |
| ReflexFlow 備案 | 只接受同 noise／同 sample、由合格 offline 模型內 optimizer 產生的 endpoint；GeoEnergy endpoint 與 simulator 搜尋贏家一律不合格。 |
| DMD 備案 | target score 只學 offline 資料，fake score 只學模型樣本；不注入 rollout return；分布匹配本身不算任務改善證據。 |
| 深度 gate／時機 gate | gain、support、uncertainty 的訓練／閾值校準只用 offline held-out 與模型自身預測；oracle 枚舉時機僅考試，不作分類標籤。 |
| 增益校準／阻尼／退回名義候選 | 增益只由 offline 重建殘差或模型內擾動校準；阻尼係數预註冊；退回保留原策略。擾動真環境結果只判通否、不回填訓練。 |

**幅度三候選額外建議，未裁**：先比較「不確定則保留原策略」與小增益／阻尼的模型內診斷，主輪暫不改 decoder obs 通道。實際名義雙胞胎的未來 obs 不可部署；若將來做退回名義，名義只能是部署可得的原 policy／plan 或 frozen model 預測。成本是另加 gain/uncertainty 校準與擾動考試；「更不理」未必「修更對」，不因縮幅就宣布成功。〔`M/CONVERGENCE.md:139-153`〕

## (c) 實施計畫：契約增量、資料路徑、檔案與主線 inline

本節列裁後改動；本次只寫此文件。四模組職責不變：objective 算目標、actor 管三 caller／生命週期、oracle 管可信監督、verification 管外部考試。新增 selector 歸 actor，品質與 gate 標籤資格歸 oracle，不另起第五套獨立訓練框架。〔承接 `P/ARCH.md:38-64`〕

**資料隔離契約（所有臂必守）**：訓練 loader 僅白名單原始 offline dataset 與帶 parent hash 的 model-generated records。原始 offline transition 的已記錄動作／狀態是許可資料；本計畫新跑或歷史實驗跑出的模擬軌跡、成功率、oracle winner、救回／弄壞、E_Δ 真執行後果，一律 evaluation-only。實際 obs 可作推論當下輸入，不能存入訓練 replay。按 episode 切分，禁止重疊窗跨 split；所有閾值和 checkpoint 在看環境考卷前由 offline validation 固定。〔offline 紅線：本工單【合規約束】；`M/KO-INJECT-0928.md:142-144`〕

模型內資料至少帶 `source_kind, dataset_hash, episode_split, parent_model_hash, candidate_id, noise_seed, iteration, domain, horizon, representation_version`；評估另帶 `eval_only=true`。訓練 API 不接受 reward/success/oracle_index 欄位。加入故意混入 evaluation record 必拒收的測試；評估結果只准決定「過／停／下一版假說」，不得回歸權重、挑 checkpoint、重調 gate 閾值或暗中挖 hard cases 回訓。

**oracle 增量：先學後果，再學選擇；不把 GeoEnergy 換名字蒸餾。**

1. 以原始 offline `(s_t,a_{t:t+h},s_{t+1:t+h})` 訓 frozen action-conditioned 後果模型 `Wφ`，ensemble=3 為建議首輪配置。目標是分子空間的位移／姿態／速度重建，按 train 統計正規化；multi-step h=4/16/32 起步，只有驗過的 h 可啟用。資料不足的窗加 mask 並報數，不補虛構未來。ant ŝ-gait 作輔助讀數／初始化；不假定 gait 已能預測 xy/yaw，也不把四步 ŝ 直接遞推當長程世界模型。〔動機：`M/CONVERGENCE.md:122-131`；可复用 `experiments/bcodec_shat_probe/shat_probe.py:66-88,157-162`〕
2. selector 候選以 frozen **實際執行器**在模型狀態上 decode 動作，再由 Wφ 展開；每個 chunk 重讀預測狀態，對應既有 live-obs 閉環，而非教師 obs 替換。輸出到達／限時、gait／翻倒／凍結、動作平滑、support、uncertainty 的向量。優先學整段／多 chunk 後果，不做四步 SE(2)→最近字查表。整個 horizon 的預測資格不足就拒選，不以終點幾何距離補缺。
3. 可微品質 `Jφ` 由上述預測後果計算 normalized arrival＋hold＋behavior support，外加 gait/feasibility validity；分量尺度取 offline calibration、權重先等權並鎖版本。到達與 hold 對本題 goal，非指派另一條資料動作；資料 NLL 是支持域限制，**不是成功率**。GeoEnergy、其排名／梯度均不作 label 或主目標；F6 牆圖僅作獨立穿牆診斷。這保留 v2 generated-plan quality 的構造，替換其未合格的後果來源。〔v2 原構造：`P/contracts/CONTRACT.md:20-26`；反證：`M/J-SIGNAL-VERDICT.md:14-23`〕
4. pointmaze 的 `Aω(s,D(u))` 只以 offline 同窗 state/path/action 訓 controller；若改逐步 inverse dynamics，另需同 transition 的 state/next-state/action 及時間資格。校準可讀狀態是否含速度等足夠資訊。R>0 用 `D→Aω→Wφ` 給 same-plan quality；另驗 live head 與 Aω 的 exposure gap。若 D 只給 xy 而缺可辨識 action 的必要資訊，資格失敗，不把 xy 差當 action。ant selector 用現有 frozen code decoder，不要求新 action labels；ant R>0 仍不支援。〔`P/contracts/CONTRACT.md:34-40`；`experiments/scratch_lacot_rollout.py:977-995`〕
5. `QualityReport={cost[B], components, valid[B], uncertainty[B], support[B], teacher_fingerprint}`。teacher parameters freeze+eval；quality 對 u 的 graph 保留，不能全包 no_grad。校準失敗、全 invalid、版本不符都顯式拒啟訓練；個別 invalid mask 不當零品質，coverage 入 gate。模型內 OOD 沒有反事實真標籤：uncertainty／模型互證只篩選，不稱已證正確；環境外考仍不可省。

**時間與配對資格不可省**：主線 make_batch 把不同長度的軌跡插值成固定 T_CAP 點，actions 卻取原始前 CHUNK 步；所以 D 的相鄰點不能直接當相鄰物理步。Aω 首案先學 `Aω(完整起始state,固定點數路徑窗)→原始CHUNK動作` 的離線配對 controller，沿 v2 的 qualified controller 選項；只有另有部署可得時間標記的 adapter 才用逐步 inverse dynamics。不得把資料終點時刻／未來速度當部署特徵；h=4/16/32 一律是執行步數。若同一路徑形狀對多種速度導致 action teacher 無法辨識，就停 refine，不拿插值 XY 差冒充 action。〔`experiments/scratch_lacot_rollout.py` 的 `make_batch:634-682`；`P/contracts/CONTRACT.md:36-38`〕

pointmaze provider 因此先校準「原始物理 transition 的 Wφ」與「同窗 controller Aω」兩種資料視圖；本節及後文 Aω 的資格／成本一體適用，逐步 inverse dynamics 不是已選定可行的實作。Wφ 用 chunk transition 遞推，並以實際 h 步資料重建考 multi-step 誤差；h=32 不合格不能拿八次 h=4 PASS 代替。品質首次只判已合格的 horizon，超出 horizon 的到達／hold 記 unknown，不杜撰遠期成功率。

validity 的擬定可實作規則：有 flow 的臂以 frozen reference flow 的 per-dimension NLL≤offline calibration第95百分位；沒有 flow 的碼庫以 train-only normalized state/action-neighbor 距離≤calibration第95百分位作 support 篩選，兩者都另要求 ensemble disagreement≤calibration第95百分位及合法完整 tuple。門檻只在 offline 鎖定；這是支持域篩選，不是反事實保證。refiner 的 support penalty 使用 frozen reference，避免同時把尺學寬；所有 valid ratios／reject reasons 另報，禁止只在容易留下的子集算提升。

**objective 沿用／增量**：`u0=sg(current_flow.sample(cond))`；`u[r+1]=Fθ(cond,u[r])`，各 optimizer step fresh sample，部署相同 temperature／intent／quantization。每輪 `Lq=mean_valid Jφ(u[r+1])`；主臂零 jitter，輔臂只可 `u0+noise`，不能退回 data±小噪聲。clean 只進 nf／anchor，資料 actions 不進 sampled-quality closure。student full BPTT、cond 可微、teacher 停梯度。首輪 depth cycling R=0/1/2/3，R6 僅壓力診斷。〔`P/contracts/CONTRACT.md:14-18`；`M/ADVERSARY-SUMMARY.md:4-7`〕

總式維持 `L=nf+anchor+Lq+λcons Lcons+λexp Lexp`；selector 獨立 `Lsel` 更新自己的 optimizer，freeze backbone／執行器，便於歸因。`Lexp=head(sg(u_r))` 對同條 plan 的 detached Aω labels，梯度只訓 head／可訓 cond，不回推 refine。沿用 v2 起始 λcons=.1、λexp=.25 作呈裁初值，非最佳實證；anchor-gradient 檢查須扣 λexp 梯度後比較。R0 composition 嚴格 nf+anchor、不 sample、不建 provider、不呼叫 refine；selector-only 是獨立模式，不偷塞進 R0 loss。〔`P/contracts/CONTRACT.md:12,30-47`〕

**learned selector 具體接口與訓練**：`select(context,candidates,nominal_id,budget)->{selected_id,scores,valid,reject_reason}`。每個候選包含完整 plan／完整 code tuple、執行器 id、horizon；G16K16 與 K32 不混成同一 latent。Sψ 以當下 obs／goal、候選表示與首 chunk predicted-consequence features 預測較長 horizon 的 quality vector／score；由 Wφ 蒸餾長程評估以節省部署展開。對照「直接 Wφ 全 horizon 評分」以辨認是否值得另學 Sψ；若無省時／品質收益，可刪 Sψ 留 learned-model selector，不能因多一個網路就稱改善。

Sψ 直接回歸 frozen Wφ 的分量，另以同 context 候選對做 pairwise ranking：只有預測改善量超過 offline calibration 誤差帶且雙方 valid 才給順序，否則 abstain；不指定 clean candidate 永遠最佳。採 cross-fit 的 predictor folds／held-out trajectory 防學生與教師同批自證。每個訓練 epoch 刷新當前真 flow 候選；refiner 加入後同時收其各輪自身 iterates，不只訓資料碼或舊版本樣本。candidate 庫包含 nominal；預測改善下界≤0、support 不合格或無有效候選時保留 nominal。先 N=8，matched-budget 的 N 由測速鎖定，不把 N=8 當神奇常數。

ant 分兩種 manifest：`privileged_relay` 精確沿用三法教材協定，只作 16pp 包絡橋接考試；`causal_candidates` 僅當下觀測／goal、train-only 字串庫或現有 flow samples 產候選，禁本題未來教材／episode id 查表。兩者學習仍只用 train split；在第一種協定勝出也只能聲稱教材上界條件下會選。碼後果差分刀使用的 behavior codec 與三法包絡的字典不是自動同一物，每個 codec hash 都須獨立校準 Wφ；有 gait 證據不等於所有候選都有合格 ŝ。〔`M/CONVERGENCE.md:14-17,112-129`；relay `run_teacher_relay_byleg.py:126-140`；`experiments/firstcuts/c1_edelta/run_c1.py:32-62`〕

包絡橋接的選擇單位是「leg-1 起點選一個完整候選策略，固定續行」，nn 在候選內仍照原規則讀自己的實際 obs；不是每 chunk 任意拼三法的字。逐 chunk 重選／新 suffix 是另一候選庫，須重算其 oracle，不能冒用5.5%。三法確切 variant／題目交集由實驗0鎖定；教師 horizon 要覆蓋該策略剩餘考試窗，h=32 的資格不自動涵蓋整個 leg。超過32的 H 用 offline 足長窗另考、費用計入同一上限；沒有足長資料／資格便把「追回16pp」橋接格標未具前置，不妨礙已合格短窗的獨立新包絡實驗。

沿用 v2 數值與離散契約：clean/sample/noise 同 dtype，cond 可異；reduction fp32，hot path 不以 `.item()` 掃 finite 或 raise，nonfinite 交 scaler，結構錯誤才報錯。硬 FSQ/VQ 的 latent refine 若需 soft/ST Jacobian，必先驗 hard-snap discrepancy；未驗只准完整候選離散 selection。cond/full BPTT 與 noise 非零的梯度考場仍在，不能以 validity mask 把圖全剪斷過關。〔`P/contracts/CONTRACT.md:8-10,16,24`〕

**兩種 gate 分工**：

- 深度 gate 管同一決策內「多算一輪值不值」：保留 u0 和所有有效 iterate；在 R≤3 的訓練深度內，只有 Jφ 改善下界>0 且 support 有效才接納下一輪，否則回到最後有效候選；R6 不部署。不得以 LayerNorm 有界宣稱品質收斂。gate 是模型內品質規則，不是已證明的環境救援器。〔`M/refine-evidence-last.txt:41-49`；`P/ARCH.md:70-77`〕
- 救援資格 gate 問「能不能救」：固定介入時刻，測 frozen selector/refiner 的候選救援前緣與實際事前選擇；失敗則停止時機研究。通過後才驗 timing gate：只讀當下／過去 innovation、模型 gain/uncertainty、剩餘算力；不得讀未來 drift、成功標籤。兩問各有外考，見(d)。〔主人三裁②；`M/CONVERGENCE.md:11-13,147-153`〕
- timing 的訓練只是 offline 模型中「介入與不介入」預測 gain 的回歸／閾值，不訓 simulator rescue classifier；leg index 作分層與混淆對照，不讓 pooled AUC .77 作通行證。

| 裁後檔案／錨點 | owner 與實作內容 | 必交證據 |
|---|---|---|
| 新 `lacot/refine_objective.py` | mod-objective；沿 v2 refinement_terms，接 learned J、mask、full BPTT／rounds；不直接讀 dataset 或環境 | label permutation 不改 sampled refine 梯度；逐輪梯度、dtype、invalid 檢查 |
| 新 `lacot/refine_quality.py`、`lacot/refine_models.py` | mod-oracle；Wφ、Aω、校準／fingerprint、QualityReport／exposure gauge；domain／codec adapters | train-only provenance、子空間校準、OOD/coverage 報告；質量未合格會拒啟 |
| 新 `lacot/refine_training.py` | mod-actor；唯一 compose，延用 Adapter.features/density/anchor/sample/exposure | 三入口相同 total，nf／anchor 不重加；FSQ z 與 head-u 空間分明 |
| 新 `lacot/refine_selector.py` | mod-actor；Sψ、nominal 保留、candidate/depth/timing gate；provider 注入 | 不讀 env、輸出 candidate id 可追溯、同候選置換等變、nominal fallback |
| `lacot/model.py:119-162,325-362` 的兩個 losses_given | mod-actor；image/state 同 compose；保留舊 positional args，新增 keyword services；R>0 缺資格報錯 | 真 actor trace、舊 inference 權重載入、R0 原測試 |
| **`experiments/scratch_lacot_rollout.py:1695-1734,1779`** | **mod-actor 唯一 owner；必換主線 inline**。保留 `_et_nf` density closure、`_u_head` clean、`_ca/COND_DROP`、`_REAL_W/_wmse`、`flow_cond(cond,anc)`；移除 sample→本批 act 的錯配項 | **真 `_stage2_loop`→compose spy**，非均勻 mask／FSQ z≠u／intent 案例；total 只用一次 base_total＋既有 BC/div/GRPO 外層 |
| 同檔 `1130-1137,1172-1173,1948,2085-2092` | mod-actor；凍結 decoder＋s_embed；首次 stage2 前建立並核資格 services，resume 後恢復 teacher metadata 再續訓 | 第一個 batch 不讀 None；不得在 blanket `.train()` 後解凍 teacher |
| 同檔 `1903-1920,1943-1945,1976-1979,3758-3783` | mod-actor；opt2 真成功才更新 refine EMA；新增 logs／teacher、selector、optimizer、RNG、provider version 保存與 migration | opt_bc skip 與 opt2 skip 分別驗；stale EMA R1 梯度，resume 重現 |
| 同檔 `_apply_refine:2267-2277`、選樣呼叫 `2395-2426`、`_bon_plan:2718`、輸出 `3117` | mod-actor；獨立 learned selector/depth 配置与真推論 trace；不借 GRAD_REFINE=1 開新臂；eval 記選樣／迭代／reject 數 | 明列 `method, selector, N, R_requested, R_executed`；停用時保持原 nominal；Geo select 與現有 GrpoReward BoN 分開 |
| 新 `experiments/refine_v3/{train_offline,eval_selector,eval_rescue,eval_timing}.py` | mod-oracle 管 train、mod-verification 管 eval；復用 relay／完整快照／子空間量尺，原腳本保持可對照 | train/eval 不互讀；privileged／causal 兩種 manifest；一口氣存快照／逐步姿態／actions／候選與決策 |
| 新 `tests/test_refine_{objective,training,quality,selector}_v3.py`、`tests/refine_acceptance_v3/` | 四模組各自考場；搬用 v2 fixture／integration，再對真 merged candidate 取證 | 六 mutation、constant/swap/identity、三 caller、教師血緣、gate leakage、AMP／resume |
| 既有 `tests_repair/test_model_zero_rounds.py`、`experiments/bcodec_shat_probe/shat_probe.py` | 復用，不改弱原考場；FIX3 probe 統計與存權重工具可沿用，頭重訓新 provider 做 | 原 R0 回歸、子空間量尺；與既有修理艦 head 重訓避免重複派單 |

以上 production 錨點已唯讀核對；v2 原行號有些已漂移，以 symbol 定位。原 scratch 1721–1730 仍是 sample 後三輪搭本筆 act；model.py 345–352 同型；只改 model.py 不算完成。R0 不抽 sample 是 v3 必達契約，不是現行兩 actor 已達事實。〔`M/ADVERSARY-SUMMARY.md:8-11`；上述 repo 行號；`P/integration/WIRING-SKETCH.md:6-36`〕

整合顺序：鎖 versioned contracts／fake → oracle 資格與 objective/compose 可獨立實作 → 接三個真 caller → selector-only → pointmaze R>0 → 外部科學 gate。fake 只解除實作依賴，不能解除教師資格；AST branch 抽不到、import error 或 reference PASS 都不得冒充 production PASS。〔`P/ARCH.md:64,79-83`〕

## (d) 第一輪實驗序列：證明、判準、成本與停損

以下是裁後工作單，不是本次已跑結果。所有新數字門檻均為**擬預註冊**，開跑前 lead 鎖定；失敗保留 raw log，不就地降門檻。算力是首輪預算上限／計數公式，非吞吐測量：GPU 型號與可用卡未知，CPU 時為 core-hours；pilot 記實際 `t_batch/t_step/峰值記憶體` 再把 operation counts 換算，超額停止呈成本差，不盲目加碼。

共通方法：trajectory 分 train／calibration／sealed-test，split hashes 固定；模型 seeds=3，主推論候選／noise 配對。所有環境評估只在封存 checkpoint 後做、禁止 online update；公開 per-seed／per-task、救回／弄壞、bootstrap 95% CI（ant 以 episode 配對，pointmaze 另列五官方 task，不能假裝 200 個獨立 task）。重複測試同一 200 題只算診斷；升格結論須新封存題組。〔限制出處：`M/J-SIGNAL-VERDICT.md:25`；`P/ARCH.md:68-77`〕

依賴順序：selector 主線是 **0→1的Wφ資格→3→5→6**；2的接線考場先行但其 refiner 學習失敗不阻塞 selector。pointmaze refine 支線是 **1的Wφ/Aω資格＋2的真operator→4→5→6**。時間 gate 對每個已合格救援器分別解鎖，不拿另一域或另一方法的成功代過。

**實驗 0：封口徑與反洩漏（先做，零新模擬）**

- 在證明什麼：16pp 包絡、J-signal 與 v3 是可追溯但不混域的帳；teacher/selector 不靠 evaluation labels。
- 判準：由既有 raw evidence 按相同題 key 重算三法交集；若缺 raw 或無法對齊，標「11/200 轉引，尚未重建」，不得宣稱本輪重現。列 candidate 來源特權、codec/hash、horizon、原 ruler；evaluation-only record 注入必被 train loader 拒收。selector 分數對更換 hidden success/oracle 欄位不變。
- 成本：CPU ≤0.5h，GPU 0；現有結果只讀。合規：oracle outcome 只進上限／診斷帳，不產 selector target。出處：`M/CONVERGENCE.md:14-17`、relay 程式 `:126-140`。

**實驗 1：教師與候選品質資格（主線前置）**

- 在證明什麼：Wφ 看得出實際動作後果、Aω 能教同計畫動作，並且 selector 有足夠有效候選；不只重建 obs 複本。
- 設計：ant／pointmaze 各 train-only Wφ ensemble=3，凍結原執行器；pointmaze Aω=3，同窗path→原始CHUNK動作的時間配對必驗。用真 offline held-out actions 檢查 h=4/16/32（包絡橋接再加實際剩餘H）後果；per-codec、xy／yaw／gait 分讀，做 shuffled-action/code、obs-only／constant 對照。再在真 flow samples／完整 donor tuple 上產模型內預測，量 support／uncertainty／action-decode discrepancy，沒有真反事實 label 的格明標未知。
- 擬判準：使用中的子空間每個 h 的 held-out normalized MSE ≤obs-only 的 0.9 倍，且配對改善 CI 下界>0；90% prediction interval 覆蓋率 85–95% 並報寬度。Aω action NMSE≤.1（以 train action variance 正規化）；data validity≥90%、generated candidates validity≥80%，每狀態至少 nominal＋1 個有效替代候選覆蓋≥80%。這些只准開外考，不能證明 OOD 預測已真。
- ant 若只有 gait 過，只准標 gait 安全診斷，不能用它聲稱 goal selector 合格；xy/yaw 不過則停 goal selector。pointmaze Aω 不過則停 refine，selector 可續。費用上限 GPU 12h（兩域 W＋pointmaze A 合計）、CPU 2h；先記步數與資料量，超額另裁。
- 合規：所有有真值的 supervision 都來自原 offline records；flow／異碼候選只用模型预测，不用新環境後果補 labels。FIX3 工具及尺度來源：`M/KO-INJECT-0928.md:65-87`、`M/CONVERGENCE.md:122-131`。

**實驗 2：真三入口接線＋既有 operator 能不能學（不以 toy 過關）**

- 在證明什麼：對抗官兩個阻斷在 production 資料路徑解除；原 RefineOperator 有可用梯度而非回 identity。
- 判準：image/state/scratch 三真入口各一次 compose trace，sample provenance=fresh flow、自身 states 覆蓋 R1/2/3；資料 action permutation 不改 refiner quality 梯度；六 mutation（total 缺項、錯接 target、忽略 rounds、零 noise、截斷 BPTT、detach cond）各有 witness。R0 原考場、不同 cond dtype、AMP skip/EMA resume 全過；無 teacher 時 R>0 明確拒啟。shape/import 失敗不算抓到語意 mutant。
- 真 operator 用 offline 合格 provider 做 3 seeds 小訓練；預註冊每 seed R1 品質改善>0、R3 不比 R1 差超過5%、coverage不跌破實驗1門檻；identity 的「非零改善」負對照必 FAIL，同報 absolute cost、diversity／code usage，不能只報 correction ratio。此為模型內 gate，外部成功另考。
- 成本：CPU ≤1h、GPU ≤2h pilot；fail 不擴訓，不把 toy sign feedback 暗塞 input。若需改寬度或顯式 feedback，另呈 shape/checkpoint 變更單。合規：只用 offline／model quality，不執行環境。既有真 operator seed0 FAIL：`P/EVIDENCE.md:210-216`；接線考場：`P/ARCH.md:79-83`。

**實驗 3：selector-only 是不是真的會事前挑（優先判 16pp 主攻）**

- 在證明什麼：候選供給有救援前緣，而且 learned score 能取用它；改善不是多抽幾條或特權輸入造成。
- 設計：固定候選與原執行器，R=0；ant 先 privileged 三法同協定考試，再 causal N=8；pointmaze N=8 真 flow。對照 nominal、uniform candidate（同庫同成本）、Sψ selector、Wφ直接評分、只作事後統計的 candidate oracle。sampler temperature／增 N／現有 GrpoReward BoN 的收益另列；GeoEnergy 歷史 select 不重跑、不與 GrpoReward BoN 混名。先同N比排序，再於共同p95 latency上限比可容納的N；不能用同N冒稱同成本。
- 擬判準：在原 200 題 ant 配對協定至少救回12、弄壞≤2，且 Δsuccess CI 下界>0；對 causal 庫及 uniform 同時滿足淨改善 CI 下界>0，才稱可部署來源下「會選」。pointmaze 每模型 seed 淨方向正、五 task 分列、配對 CI 下界>0。報 recovered-gap=`(fail_nominal-fail_select)/(fail_nominal-fail_oracle)`；分母≤0 記 N/A，不能固定代入16pp。
- 若 oracle 有空間而 selector 無益→排序病；oracle 也無空間→供給／協定病；privileged 過、causal 不過→特權候選依賴；任何一項不過不得移到 timing 來掩蓋。ant 另列到達／限時、步速、翻倒／凍結、平滑與人眼 review 待辦；統計提案要求安全／平滑惡化 CI 上界≤預註冊5%相對容忍，翻倒／凍結用≤1pp絕對容忍，不能只刷到達。
- 成本：Sψ 3seeds×最多20k updates，GPU ≤6h；環境評估按 `200×seeds×需執行候選數×每題步數上限` 算，第一批先固定庫避免每個 scorer 重跑同候選。CPU 預算併實驗5的48h；未量步速前不承諾完工時間。合規：oracle labels只作報表，Sψ只收實驗1的 frozen model targets。
- 門檻依據：救回／弄壞沿 `M/CONVERGENCE.md:54-57`；四關口徑沿 `docs/NOTE-2026-09-08-teacher-relay.md:63-85,104-110`；BoN 方言沿 `experiments/j_signal_trial/README.md:5`。

**實驗 4：pointmaze learned refine 是否增加候選品質與可執行性**

- 在證明什麼：在 selector 固定後，更新 plan 比單純重抽／排序有額外價值，且 head 能讀修改後的 plan。
- 設計：合格 Aω／Wφ frozen，先原 RefineOperator；3seeds×最多10k updates。同起始 u 比 R0/1/2/3（R2用於曲線、不任選最好 R），R6 壓力診斷；因子比較 selector off/on × refine off/on。保留原 sampler、同 walltime 增 N，與不加 refiner 的 learned selector，記 flow/decoder/predictor/refine calls、backwards、peak memory、p50/p95 latency、離線訓練成本。
- 擬判準：模型內實驗2規則仍過，exposure generated valid≥80%、normalized error≤實驗1 Aω 容忍；以封存的未refine head作對照，gap=`sample_error-anchor_error` 不增加超過0.01 NMSE（避免近零 gap 的相對比失真）。外部 success 對 R0 及等 walltime selector-only 的配對 CI 下界>0，三 seed 淨方向全正。若只同 N 贏而等成本不贏，記「品質可能有效、算力不划算」，不升格部署。模型 cost 降但外部無顯著改善記「未證實有效」；若外部顯著變差則記 proxy exploitation 警訊，均不放行。
- R6 若惡化照實留存；depth gate 只能在訓練內 R≤3，對照固定 R1/R3 量省算力與損失。pointmaze 淨改善不回填 ant 16pp 帳。成本：GPU ≤9h；外考可復用實驗5固定介入，不多跑同一設定。合規：每輪 supervision 只來自模型後果／same-plan teacher，所有環境數字只判決。
- 出處：`P/ARCH.md:68-77`、`P/contracts/CONTRACT.md:34-40,55-59`、`M/refine-evidence-last.txt:41-49,77-85`。v2 的 GD baseline 在此改為歷史死臂 ledger；首輪不再執行 GeoEnergy GD。

**實驗 5：第一問「能不能救」——固定時刻，先於觸發器**

- 在證明什麼：把介入時機固定後，已封存的 selector／合格 refiner 在原失敗題上有實際救援能力；這才是 timing 的前提。
- 設計：先零漂移 leg-1，固定預註冊 chunk 時刻與一次介入預算；同完整 snapshot 比 no-op、nominal、learned intervention、事後 candidate oracle。另在已有擾動雙胞胎協定診斷 Γ=2rᵀWd+‖d‖²_W、救回／弄壞；不更換 decoder obs 通道。oracle 時刻／candidate 枚舉只作上限。
- 擬判準：ant 200 題救回≥12、弄壞≤2、淨改善 CI 下界>0；同時過實驗3四關限制，並勝過同算力 random/no-op churn。若候選 oracle 能救、實際 selector 不能，仍判「可用救援器未成立」，不開 timing。擾動臂要 N-benefit CI 下界>0 才稱修正有效；只交叉項負或縮幅不算救援。
- 成本：評估首輪總 CPU 上限48h（含實驗3/4；與原開環三臂 CPU<1天僅量級參考），計數表分 `states×candidates×times×suffix_steps`；已有候選 outcomes 可重算不同 frozen scorer，但不能拿來挑分數函數。合規：snapshot 與實際後果只在 sealed evaluator 使用；不可送 teacher retraining。來源：`M/CONVERGENCE.md:54-68,133-153`。

**實驗 6：第二問「會不會挑時機」——僅實驗5過才解鎖**

- 在證明什麼：同一救援器、同介入次數與算力，因果 timing gate 優於固定／隨機時機，而不是挑較容易的 leg。
- 設計：frozen timing gate（offline model gain label，至多5k updates或固定已校準閾值），對照 fixed interval、budget-matched random、leg-index-only，以及事後 oracle timing 上限；固定 intervention budget=每 leg最多1次，記實際算力。按 leg／進度分層、另報排除首 leg；新增測試未來資料置換不得改 gate 決策。
- 擬判準：對 fixed/random/leg-only 的配對 Δsuccess CI 下界均>0，且救回／弄壞、四關限制仍過；報 precision、recall、觸發覆蓋／錯觸／漏救與預算，AUC 只作附表。只有 pooled AUC 高、分層失效就判 timing 不成立，保留已通過的固定時刻方法。
- 成本：GPU ≤1h；CPU 另≤24h，先算 `200×3seeds×4可部署時機臂×suffix_steps`，oracle 枚舉另列時刻數，超額不啟動。合規：oracle timing、真 rescue label不參與學習／閾值 tuning。來源：主人三裁②、`M/CONVERGENCE.md:11-13,147-153`。

首輪總預算上限：GPU 30h（12+2+6+9+1）、CPU 76h（0.5+2+1+48+24=75.5，向上留整），不是本件3小時文書努力預算；人日估算有共用工作不得相加當排程。主要停損點在實驗1品質資格、實驗2真 operator、實驗3選擇、實驗5救援；前置不過就省下後續成本。每次只報此次能證明的層級：接線／模型內／固定時刻環境／時機選擇／跨域，不能跨級。

## (e) 死路對照表：不靠改名復活

| 死路／失效推進 | 判死或撤回出處 | v3 禁止事項／仍可保留的邊界 |
|---|---|---|
| 手刻 GeoEnergy 梯度爬坡 | `M/J-SIGNAL-VERDICT.md:9-23` | 不重跑調步長找翻案，不以其 optimizer endpoint 訓 refiner；post-F6 已確認，不能訴諸壞牆尺。 |
| GeoEnergy@8 事前挑選 | 同檔 `:16-17,25` | 不蒸餾其排名成「learned selector」；歷史敗臂只記錄。未判死的是用獨立 learned consequences 的候選評分，须重新過資格與外考。 |
| 四步 SE(2) 效果命令換卡語言 | `M/CONVERGENCE.md:85-93` | 不用相同短窗 nearest-effect lookup 換名重上；完整 code tuple＋多 chunk 後果的 selector 是新假說，並非既有陽性。 |
| Jacobian／mjd_transitionFD 線 | `M/CONVERGENCE.md:154-155` | ant RK4 不支援該介面，且它是有限差分；不列為實施依賴。模型內 autograd 僅對 neural predictor，與此不是同一條線。 |
| token 級 R_min 套用 | 同檔 `:154-155` | 不以資訊下界判死 token；code 信息只用候選差分／實際行為驗證。 |
| data±小噪聲主訓練輸入 | `M/ADVERSARY-SUMMARY.md:4-7` | 不以加正則／增噪幅假裝解分布問題；必用 flow 真 sample 與自身 iterates，jitter僅輔助。 |
| 只改 model.py 修主線 | 同檔 `:8-11` | 必含 scratch inline 與 total，三真 caller trace 缺一不可。 |
| AUC .77 即可上觸發器 | `M/CONVERGENCE.md:11-13` | 先救援、後時機，分層排除進度混淆；不可用 pooled AUC 換部署資格。 |
| toy feedback PASS＝production 成功；LayerNorm＝越深越好 | `P/EVIDENCE.md:210-216`；`M/refine-evidence-last.txt:41-49` | seed0 FAIL 留存；真 operator 重驗、R6僅診斷；改 input/checkpoint shape 另呈變更。 |
| 教師 obs 替換 live obs 作救場捷徑 | `M/CONVERGENCE.md:77-79` | 救16壞65僅歸因儀器；不把名義雙胞胎的未來觀測當部署通道。 |

GeoEnergy F6 幾何查詢工具可保留於獨立診斷，保存它不等於恢復它的品質排序權。任何未來重新開死臂，都須帶改變承重前提的新證據與獨立變更單；本 spec 沒有授予這項復活。

## (f) 自我懷疑、未知事實與 lead 驗收清單

| 我可能錯在哪裡 | 能推翻本案的讀數／後果 |
|---|---|
| 把「學出的」當成「可信的」：Wφ、Aω、Sψ 會共同鑽模型洞 | 離線 calibration 過、環境 success 不升或掉，或候選 validity 低→停止相應臂。cross-fit／ensemble 不能創造缺少的反事實真值。〔風險已見 `P/CHANGE-ORDER.md:20`；文獻警示 `M/refine-evidence-last.txt:29-37`〕 |
| 16pp 只是特權 candidate／彩券 churn 的事後包絡 | causal 庫 oracle 沒剩空間、或 selector 不勝 random→縮小目標，不能拿 privileged 成功宣稱部署突破。17.5pp 開環稅也不是本案可回收額保證。〔`M/CONVERGENCE.md:14-17,54-81`；relay `:126-140`〕 |
| offline 根本無法辨識重要反事實／長程後果 | h=16/32 或 xy/yaw 資格不過→selector 研究停在短程診斷；不得借 simulator 產訓練資料補洞。資料單路線／flow 不自帶新路線資訊的既有判決仍有效。〔`M/CONVERGENCE.md:23-37,122-131`〕 |
| 真 refiner 表示能力不足；選擇已吃掉全部收益 | 真 operator 仍 seed0 FAIL 或等成本不勝 selector→保留 selector，另呈 feedback／寬度案；不能把 refine 當必需品。〔`P/ARCH.md:75-77`〕 |
| same-plan teacher 看的是理想路徑，live head 實際走另一條 | exposure gap／action support 不過→即使 Jφ 更低仍拒啟 refine；坐標 plan 不能自動推出可執行 action。〔`P/contracts/CONTRACT.md:34-40`〕 |
| 拒選／深度 gate 太保守，實際只剩 identity | valid低／介入率近0、success與nominal相同→報無效；不能以「沒有弄壞」當救援成立。C2 60題修被吃只是潛力，不是可領回的收益。〔`M/CONVERGENCE.md:133-144`〕 |
| 同步更新 flow／head 會讓資格過期 | 初輪 freeze selector 的執行器，refine 訓練仍須定期 fresh sample；每換 backbone／codec／teacher hash 重新校準，不沿用舊 valid 印章。〔v2 覆蓋要求 `P/contracts/CONTRACT.md:16,47`〕 |

尚未查得／未完成，不能寫成事實：兩顆 J-signal checkpoint 是否本地可載、遠端即時卡量與吞吐；ant 三法逐題原始交集在本輪未重算；可用 offline 完整狀態／horizon coverage 與 pointmaze action teacher 可辨識性未量；新 Wφ／Sψ／Aω 尚未實作或合格；「refine-v2 批」與今日 v3／patchrun shat3 的派單對應未確認。建議將本件 oracle 資格列為接口責任、接受其他修理艦交付的合格 head，避免重複重訓；不冒認已完成頭修理。〔`M/KO-INJECT-0928.md:65-68,97-99`〕

lead 可逐項驗收：

- (a) 四模組＋contracts/integration 都有今日支持／推翻／未觸及對照；四把刀、兩個對抗阻斷與 F6 補證無混名；toy／production 明分。
- (b) 待裁四項逐項有建議、代價、退路；幅度三候選仍標未裁；各提案臂有 offline 合規行。
- (c) 真 flow／own iterates、learned quality、same-plan teacher、selector／深度／時機契約可落檔；scratch inline、total、生命週期與兩 actor 都列改動及驗收。
- (d) 實驗0–6逐項寫明證明什麼、門檻、成本與合規；先「能不能救」後「會不會挑時機」；16pp ant 帳與 pointmaze 帳分開。
- (e) 六條指定死路全部封存，其他已否決捷徑亦列；沒有以 F6／新命名復活 GeoEnergy。
- (f) 自我懷疑非空、缺證明示；本次只讀 repo／materials，只交此檔，沒有實驗、production 修改或 commit。

未見判決互斥到無法出 spec：分布內回授有效與分布外幅度失控可並存；GeoEnergy 失效與 learned selector 值得試可並存。真正可能卡住的是 offline 模型對反事實候選的辨識資格，本案把它放在實驗1即停的門，而不是寫成已解。STATUS 僅指設計交付完成；四項裁示及所有實驗結果仍待後續。

STATUS: DONE
