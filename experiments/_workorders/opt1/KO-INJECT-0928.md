# 知識官注入（S4；2026-09-28 晨）——供今日三張工單定稿

_檢索：全 grep/讀檔（無 MCP）。紅線：無界 find/du/grep -r 一律不跑，grep 限定任務書列的路徑；
只讀不寫（除本檔）。出處逐條攜帶；ACTIVE-INSIGHTS 條目標 _source: user_correction 的一律標
【主人裁示】；查不到的進空手欄，不編造出處。_

---

## 【知識官注入】主題：①（v3 設計，LePlanner 式 learned refine）

### 坑（按殺傷排序）

1. **對抗官 BLOCKER B1**：目前訓練分布（資料 ± 小噪聲）下，identity refine 是閉式最優解——
   方盒外 f\*(x)=x；KD=1024 時 flow 樣本落盒機率 σ=.02 時 2.77e-6、σ≥.05 時≈0；correction
   ratio 實測 .59-.98（σ≥.05）＝幾乎不修（lead 已親自重跑腳本確認）。
   出處：`data/fleet-runs/refine1/ADVERSARY-SUMMARY.md:3-6`。
   → 對本件：v3 若不換訓練輸入來源（改用 flow 真實樣本＋refine 自己的迭代，而非資料+小噪聲），
   loss 怎麼調都會收斂回 identity refine；proposal-v2 已提修法方向（C3）但只在 toy 上驗證過（見坑4）。

2. **對抗官 BLOCKER B2**：主線 trainer（`scratch_lacot_rollout.py:1720-1730`）直接 inline 了
   F1 病（`model.py:345`，抽樣 u 與本筆 actions 條件獨立），不走 `model.py` 的 `losses_given`；
   只改 model.py 完全碰不到主線。
   出處：`data/fleet-runs/refine1/ADVERSARY-SUMMARY.md:8-11`；同病見
   `data/fleet-runs/review1/VERDICT.md` 🅱-1（F1）。
   → 對本件：v3 的修改範圍必須包含主線 inline 分支（或抽共用 compose 函式、三個呼叫端同步），
   ⛔ 不能只改 model.py 就宣稱修好。

3. **C2 幅度失控的操作方向三個都還沒裁**：CONVERGENCE.md 記錄「分布內反射有效（+24.5點）、
   分布外方向對但幅度失控（correction_overwhelmed 是主模式）」後，列出候選操作方向——增益
   校準／阻尼／不確定時退回名義——但明文標「未裁、僅記錄」；60 題「修被吃」是潛在翻正空間。
   出處：`data/fleet-runs/breakthrough1/CONVERGENCE.md:142-144`。
   → 對本件：v3 若要處理「回授方向對、幅度失控」，這三個是現成候選但主人還沒拍板選哪個，
   ⛔ 別假裝已經選定其中一個方向。

4. **proposal-v2 的驗證只到 toy 級**：只有「小 MLP＋toy decoder＋sign feedback」才全 seed
   通過；無 feedback 的兩次失敗、真 RefineOperator 額外診斷（700 步×3 seed）仍留有 seed0
   FAIL，原樣留存未解。
   出處：`data/fleet-runs/refine1/proposal-v2/README.md:21`；
   `data/fleet-runs/refine1/proposal-v2/CHANGE-ORDER.md` 末段。
   → 對本件：「LePlanner 式回饋可行」目前只被 toy 支持，⛔ 不能把 toy 成功當成 production
   RefineOperator 已經能學會的證據。

5. **F6 修復時序與 J-signal 判死結論的關係未經確認**：`j_signal_trial/README.md`（寫於 9/27
   15:02）明文警告「此份副本的 `lacot/refine_grad.py:153` 目前仍是只查頂點的舊碼，與工單
   『F6 已修』不符，部署前須核對」。S4 額外查證（單檔 grep＋git log，非任務書列的檢索面，
   因案情直接動搖①的背景前提才多查一步）：F6 修復 commit `0916c97`（"F6 穿牆線段內插"）
   實際落在同日 **14:30:37**，比 README 寫作時間早 32 分——本地 repo 當時已經修好；但
   J-signal 判死爬坡/挑選那次真正執行（jasmine job 33812）用的是不是這個 post-fix 副本，
   S4 在唯讀範圍內查不到（那是遠端節點的部署狀態，非本機讀檔能驗）。
   出處：`lacot/experiments/j_signal_trial/README.md:3-7`；`lacot` repo `git log` commit
   `0916c97`（`2026-09-27T14:30:37+0800`，`lacot/refine_grad.py` 現況已含
   `wall_interp_k` 線段內插，行 44/149-159 核對過）。
   → 對本件：v3 若把「climbing/selection 已判死」當乾淨背景事實，先跟主人/實作方核對 33812
   那次跑的是不是 post-F6 code；目前證據偏向「應該是」（修復比 README 還早），但沒有第一手
   確認，別講成已驗證。

### 先備

1. **proposal-v2 待裁四項尚未拍板**：①主案 LePlanner-style vs ReflexFlow/DMD 備案　②EMA vs
   self/R≥2　③先救有合格 decoder/geometry 的 pointmaze、其他 actor 缺 oracle 禁 R>0 訓練
   ④新 same-plan action teacher 的資料/校準成本。主人已裁「現在就救 learned refine」（大方向），
   但這四項細節未見裁決記錄。出處：`refine1/proposal-v2/CHANGE-ORDER.md` 尾段「待裁項」＋首段
   「主人已選『現在就救 learned refine』」。

2. **FIX3（修理批3）已於 9/27 落地可用**：shat_probe 判準子空間化＋ŝ 目標位移正規化＋probe
   權重保存，25 支 tests_repair 迴歸綠才 commit。C1 提到的 ŝ 頭本體重訓仍待「refine-v2 批」
   ——本次 v3 是不是就是這個「refine-v2 批」，命名對應待跟主人核對。
   出處：`data/fleet-runs/firstcuts1/WORK-FIX3.md`；`CONVERGENCE.md:156-157`。

3. **命名口徑（跨文件同名異物）**：`CONVERGENCE.md` 的「B1」＝開環稅 17.5 點（題b首刀）；
   `ADVERSARY-SUMMARY.md` 的「B1」＝identity-refine blocker——兩份文件同代號指完全不同的東西。
   出處：對照 `CONVERGENCE.md`（題b段）vs `refine1/ADVERSARY-SUMMARY.md`。
   → 寫 v3 工單時兩邊代號都要展開講清楚是哪一個，⛔ 不要單獨寫「B1」。

### 輪子

1. `data/fleet-runs/refine1/proposal-v2/`（`ARCH.md`、`contracts/`、
   `mod-{actor,objective,verification,oracle}/`、`integration/`、`EVIDENCE.md`）——四模組設計
   ＋CPU 玩具證據＋wiring PASS/現行主線 top-level FAIL 兩向測試，已解掉對抗官 B1/B2/F5語意/
   exposure-bias/M3空殼/C1 dtype/R2/六種mutation 等 major 項的「設計層回應」（尚未落地
   production）。v3 可直接接著 v2＋其 CHANGE-ORDER/REVIEW-RESPONSE 往下走，不必重新設計。

2. `lacot/experiments/bcodec_shat_probe/shat_probe.py`（含 `gen_fix3_before_after.py`）——
   FIX3 已修好的子空間化判準＋位移正規化，C1/C2 分析已在使用，可直接復用。

3. `lacot/lacot/refine_grad.py` 的 `GeoEnergy`（`wall_interp_k=7` 可調線段內插密度，
   commit `0916c97` 已修好穿牆判定）——若 v3 仍要用 energy 型判準，這支不必重造。

### 空手

- `EXPERIMENT-INDEX.md`、`CANON-u-semantics.md`、`DESIGN-2026-09-11-behavior-codec.md`
  本文未直接讀（僅透過其他文件間接引用），未見≠不存在。
- J-signal 判死結論所依據的 jasmine job 33812 實際執行程式碼版本是否為 post-F6，S4 查不到
  （超出唯讀讀檔範圍，需查遠端節點部署狀態）——見坑5。
- `refine1/proposal/`（v1，非 v2）只看了目錄結構，沒有細讀內容；判斷 v2 本身的
  CHANGE-ORDER.md 已完整交代 v1→v2 差異，讀 v2＋變更單應已足夠，v1 全文未展開查。
- G16/G17 行為字典容量掃描（`review1/VERDICT.md` 🅱-9/10）與今天 `patchrun1`（seedrep2/
  shat3）的關聯只看了 `patchrun1/LAUNCH-CARD.md` 摘要，沒有查這條線是否已經被排進 v3 設計
  範圍——時間預算內未查完，不在任務書列的檢索面內。

---

## 【知識官注入】主題：②（u 殺手對照：u-BoN vs 純動作噪音 BoN）

### 坑（按殺傷排序）

1. **BoN 尺陷阱**：pooled 全譜 vs leg 級是不同的世界、N=64 未飽和；u-BoN 對照隊與（原定的）
   蒸餾隊口徑要對齊。出處：`data/fleet-runs/distill1/KO-INJECT-distill-maiden.md` 坑#5（挪用）。
   → 對本件：這次對照如果也跨 temperature/窗口設定，先把「哪個尺」釘死再比，兩臂不能一邊
   看 pooled 一邊看 leg 級。

2. **09-16 的實戰教訓（同型陷阱曾撤回兩個結論）**：驗收官當天翻案「溫度序只在 pooled 尺、
   繞圈『閉環湧現說』撤回＝視窗混淆」；BoN 上限 .9451（T0.6 N64 pooled）就是那次量出來的。
   出處：`MEMORY.md` 2026-09-16 收工條目；`CONVERGENCE.md:159-166`（u 改名＋殺手對照段，
   引用同一個 .9451 數字）。
   → 對本件：這次殺手對照的判決基準（.9451 那把尺）跟 09-16 是同一批材料，先確認兩臂用
   同一個尺讀數，別讓溫度序結論只活在某一個尺度上。

3. **BON_N/BON_MODE 現在的實作細節**：`LACOT_BON_MODE` 目前只實作 `"plan"`
   （抽 max(n,BON_N) 條→decode→三項乘法閘打分→取分數最高的 n 條）；`BON_N=0`（預設）＝整段
   不建、逐位元不變；`BON_N=1` 仍會在檔名加 `_bon1` 但行為＝不開 BoN，容易跟 `BON_N=0` 的
   產物混讀。出處：`lacot/experiments/scratch_lacot_rollout.py:1560-1586`
   （`_bon_score`/`_bon_plan` 實作在 2683-2750/2718-2750）。
   → 對本件：換成 u-BoN vs action-noise-BoN 時要確認兩臂是否真的共用同一套
   `_bon_plan`/`_bon_score`，檔名 `_bon1` vs `bon0` 的混讀陷阱要避開。

4. **action-noise stream 配對設計是既有規則、不是自由選擇**：現有「特異度受測臂」設計是
   「同一顆模型、同配置、只換 action-noise stream」，且各 arm 必須共用同一條 noise stream
   做配對比較，否則差值會混進取樣噪聲。出處：`lacot/experiments/scratch_lacot_rollout.py:3198`
   （配對註解）、`:3224`（主人 2026-08-30 裁定）。
   → 對本件：u-BoN vs action-noise-BoN 比較要沿用這個配對設計（同 model/config 只換噪音
   來源），不要讓兩臂各自獨立抽樣。

5. **ckpt 挑選需雙條件**：mtime 晚於本次起跑＋編號最大，缺一會挑到舊模型且沒有欄位會顯示
   異常。出處：`MEMORY.md` 索引「選產物要用『本次起跑時間』＋『編號最大』」
   （`pick-artifacts-by-run-not-by-name.md`，08-07；S4 僅讀過索引一行，未展開讀該檔全文，
   故不套用【主人裁示】分級，僅轉引索引本身的描述）。
   → 對本件：挑 jasmine 上跑 u-BoN/action-noise-BoN 用的 ckpt 時套用這個雙條件檢查。

### 先備

1. **【主人裁示】offline 紅線**：執行結果不得進訓練資料。
   出處：`data/fleet-runs/distill1/KO-INJECT-distill-maiden.md` 先備#2（挪用，原標【主人裁示】）。
   → 這次殺手對照產出的 rollout/成功率統計，⛔ 不能被後續拿去當訓練資料用。

2. **u 的語義地基＝軌跡規劃**：flow 生成的 latent 軌跡、decode 成具體路徑；抽象說法
   （context／工作檯／scratchpad）是側面不是替代；canonical 在 `lacot/docs/CANON-u-semantics.md`
   （本檔僅間接引用，未直接讀）。【主人裁示】
   出處：`ACTIVE-INSIGHTS.md:71-72`（`_source: user_correction, 2026-09-06`）。
   → 討論「u 存亡」時先錨地基再抽象，別讓措辭漂走。

3. **殺手對照的判決邊界**（work order 背景已給，這裡是 CONVERGENCE 的精確措辭）：同預算上限
   若相同→u 降級為可替代噪音源；若 u 顯著高→「執行變體抽樣器」角色站穩。
   出處：`data/fleet-runs/breakthrough1/CONVERGENCE.md:159-166`。

### 輪子

1. `lacot/experiments/scratch_lacot_rollout.py:1560-1568,2597-2750`
   （`BON_N`/`BON_MODE`/`_bon_plan`/`_bon_score`）——現成 plan 模式 BoN 管線，已解掉
   「抽 N 條→decode→打分→取 top」的整套機制，u-BoN 臂可直接用。

2. `lacot/experiments/scratch_lacot_rollout.py:3198-3224`（action-noise stream 配對機制，
   主人 8/30 裁定）——已解掉「同 model/config 換噪音源」的配對設計，action-noise 臂可直接
   當實作基礎。

3. `lacot/experiments/j_signal_trial/README.md`＋`run_三臂.sbatch`——同一份 rollout、同
   ckpt 路徑、同輸出檔命名規則的部署骨架；其「C 臂判準差異」段（BON_N 現在選 GrpoReward
   +C8 乘法閘，不是 9/2 的 E-selection）是現成的「別把 BON_N 跟 GeoEnergy selection 混為
   一談」前車之鑒，寫工單時可直接引用。

### 空手

- 沒有在給定檢索面內找到明確標「2026-09-16」日期的獨立 BoN 管線檔案；work order 提到的
  「複用 9/16 BoN 管線」，S4 只確認了 `scratch_lacot_rollout.py` 裡 `BON_N` 機制存在且與
  `j_signal_trial`/`CONVERGENCE.md` 的引用吻合（.9451 那次測試用的應該就是這套機制），但
  沒找到一份單獨標記 9/16 日期的檔案或報告——未見≠不存在，不排除在 jasmine 本機或本次
  檢索面外的路徑。
- `lacot/experiments/probe_bon_rung05.py`（檔案 mtime 9/6）是否是後來「9/16 五臂」的前身
  或同一支程式，只看了檔案存在與大小，內容沒有展開比對——時間預算內未查完。
- pointmaze-large-stitch 這次要跑的 jasmine 上實際 GPU/資源分配（是否有空卡）未查——這是
  即時集群狀態，不在本次「讀檔/grep」的檢索面內。

---

## 【知識官注入】主題：③（S0 最佳化首航：lady 8×2080Ti golden baseline＋profiling）

### 坑（按殺傷排序）

1. **【主人裁示，挪用】lady 雙隊衝突**：蒸餾 fine-tune 與最佳化 profiling 若同日都排 lady，
   發射單須明寫時段/卡分配；疊卡判斷要 VRAM＋SM 雙量一起看。
   出處：`data/fleet-runs/distill1/KO-INJECT-distill-maiden.md` 坑#3（挪用，原標【主人裁示】，
   原情境：蒸餾今天不點火）；雙量判斷本體見 `ACTIVE-INSIGHTS.md:49-50`
   （`_source: user_correction, 2026-08-14`）。
   → 對本件：即使蒸餾今天沒點火，這條規則仍是通用的——S0 上 lady 前先確認今天 lady 上
   沒有其他工作會搶卡/搶 SM（例如若蒸餾臨時改點火，或別的排隊工作）。

2. **fp16≠bf16（Turing/2080Ti）**：`torch.cuda.is_bf16_supported()` 在 Turing 回 True，
   但實測 bf16 只有 7.38 TFLOP/s，比 fp16 的 56.57 慢 7.66 倍、甚至比 fp32 的 12.22 還慢；
   照習慣寫 bf16 會讓訓練慢七倍且不會有任何錯誤訊息。
   出處：`ACTIVE-INSIGHTS.md:37-38`（`_source: daily_observation, 2026-08-09`）；同病見
   `data/fleet-runs/distill1/KO-INJECT-distill-maiden.md` 坑#1（fp16 需配 GradScaler，
   否則靜默不收斂，8/17 gradient underflow 實錄）。
   → 對本件：S0 的 golden baseline 在 lady（2080Ti＝Turing）上，先實測 matmul 選 dtype，
   別憑 API 或習慣選 bf16；若最終選 fp16 務必掛 GradScaler。

3. **主人裁示的 profile 方法論三條**：(a) 先查框架有沒有內建 profiler，別自己手寫取樣程式；
   (b) 一直改設定重跑看指標是 sweep 不是分析，要下去 profile 看時間花在哪一行；(c) 量測
   可信度是地基，Profile（找瓶頸）與 Optimize（怎麼修）要分開，中間還有「診斷為什麼慢」
   這一段。出處：`ACTIVE-INSIGHTS.md:11-16`（三條皆 `_source: user_correction, 2026-07-24/07-27`）。
   【主人裁示】
   → 對本件：S0 是「首航」，golden baseline＋profiling 階段只做 profile／診斷，⛔ 不要一邊
   調參一邊把結果當分析結論（sweep 半天建立在錯前提上的教訓就是這個病）。

4. **optimize skill 的 Step 0 紀律**：量測前先讀 `hardware/<vendor>.md` 與
   `architectures/<family>.md`，別重新量測已經記過的事實——2026-08-04 曾因沒做這步，花
   15 分鐘重新量到 `amd-rocm.md` 07-29 已經記載的兩個事實，還當新發現報告給主人。
   出處：`.claude/skills/optimize/SKILL.md:50-57`。
   → 對本件：S0 開跑前先讀 `hardware/nvidia.md`（目前只有 AMD/NV 類比表，**沒有 Turing 實測
   列**，見空手欄）＋ACTIVE-INSIGHTS 的 Turing 數字，別把已知的 2080Ti dtype 特性當新發現
   重新量一次。

5. **消費級多卡無 NVLink 的 NUMA 拓樸陷阱**：實測 3×RTX 2080 Ti／3×RTX 4060 Ti 顯示加卡
   不一定加吞吐——兩卡 1061 樣本/秒、三卡因跨 NUMA 反降到 971；同 NUMA node 的兩張（topo
   顯示 PHB 而非 SYS）比跨 NUMA 快 16-17%。
   出處：`ACTIVE-INSIGHTS.md:41-42`（`_source: daily_observation, 2026-08-09`）。
   → 對本件：S0 在 lady 8×2080Ti 上做 baseline 時，先用 `nvidia-smi topo -m` 看拓樸，別
   假設 8 卡 DDP 線性擴展；golden baseline 的卡數/配置選擇要先看 topology 再定。

### 先備

1. **主人 2026-07-27 定的魔法書體系**：量測可信度是地基；Profile 與 Optimize 必須分開
   （中間插「診斷」段）；Optimize 按瓶頸落在哪個資源分冊——CPU、GPU（再分 NV/AMD、
   單卡/多卡）、通訊傳輸、I/O 儲存；調參 HPO 自成一本但依賴量測魔法。【主人裁示】
   出處：`ACTIVE-INSIGHTS.md:13-14`（`_source: user_correction, 2026-07-27`）。
   → S0 的產出應該歸位到這個分冊體系裡（GPU/NV/多卡 那一冊），不要混著寫成一份雜記。

2. **optimize skill 的核心原則：Measure, don't assume**——第一個案例（LeWM）裡，blanket
   bf16 看起來是最大槓桿，結果是淨 **-18%**；vmap 玩具給 3.4×，真模型卻 ~1×。所有假設都
   要被 ablation 驗證，不能只信直覺或猜測。
   出處：`.claude/skills/optimize/SKILL.md:18-23`。

3. **S0 是這個 skill 第一次真正落在「lady/2080Ti＋LaCoT 主線訓練迴圈」這個組合上**——
   `case-studies/` 現有三案（LeWM narrow model、Horizon Imagination、config-sweep）都不是
   這個模型/硬體組合，沒有現成 case study 可直接抄；本次是 F4 最佳化艦隊的第一戰，跑完後
   應該回填一份新的 case-study。
   出處：`.claude/skills/optimize/SKILL.md` 結構段（`case-studies/` 清單）。

### 輪子

1. **optimize skill 全套方法論**——`workflow.md`（8 步法：profile→診斷→optimize）、
   `diagnose.md`（一症狀多病因、對症解藥相反）、`levers.md`（槓桿目錄，每項附驗證方式與
   會破壞什�麼）、`harness/`（`ablation_template.py`、`bit_exact_check.py`）、
   `hardware/characterize-new-gpu.md`（如何 profile 一張沒摸過的卡）——S0 可以直接照 8
   步法＋harness 模板走，不必重新發明量測腳手架。

2. **ACTIVE-INSIGHTS 裡已經累積的 Turing/2080Ti 實測數字**（dtype TFLOP/s、NUMA 拓樸吞吐，
   見坑2/5）——可以直接當 S0 baseline 的「已知先驗」起點，省掉重新量測那一段，只需要在
   S0 自己的環境（lady 具體機況）上覆核一次數量級是否吻合。

3. `cluster-inventory/inventory.db`（SQLite：machines/gpus/benchmarks/opt_notes）——查
   lady 的 GPU/VRAM/topology 等硬體規格的既有本機實測快照，不用重新用印象推測。【主人裁示】
   出處：`ACTIVE-INSIGHTS.md:25-26`（`_source: user_correction, 2026-08-02`）。

### 空手

- `hardware/nvidia.md` 目前**沒有 Turing（2080Ti）的實測列**（只有 AMD R9700／RTX 3090／
  Blackwell 6000 三卡的量測表，Turing 只在 partitions 表尾提到「turing(lady 2080Ti)」這個
  名字）；S0 要用的 Turing 專屬數字目前只存在 ACTIVE-INSIGHTS，還沒回填進 optimize skill
  本身——這是 skill 文件落後於 insight store 的落差，不是「沒有這份資料」。
- lady 今天（9/28）實際的 squeue/佔用狀態未查——即時集群狀態不在「讀檔/grep」的檢索面內，
  且會需要 ssh/slurm 指令，非本次唯讀讀檔範圍。
- `optimize/workflow.md`、`diagnose.md`、`levers.md` 本文只看過 `SKILL.md` 裡的摘要說明，
  沒有展開全文讀——任務書只要求「結構與 AMD/NVIDIA 分工看一眼」，此為刻意範圍，非查不到。
- `case-studies/` 三案（LeWM narrow model 1.84×、Horizon Imagination、config-sweep 十四臂）
  只看了檔名，沒有細讀內容比對是否有可遷移到 LaCoT 訓練迴圈的具體手法——時間預算內未查完。

---

STATUS: DONE
