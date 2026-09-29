在證明什麼：ucontrast2 交件（worktree 腳本、n64 與 s35 兩臂 launcher、分析器）能否在凍結樹上產出「50 題×64 draws 配對兩臂」與「s35 200 題×8 draws 重現」的可自證帳，且沒有繞 gate、沒有偷改釘值或預註冊判準。判準＝lead 指定的四個咬點逐條獨立重算，加全面查；驗不到的格標「未驗」，不當 PASS。

# VERDICT-S2（S2 檢察官：Claude Sonnet 5.5 max，與實作方 GPT-6 Sol 不同家族；判官不動手）

```
VERDICT: PASS-with-notes
  重點1 50 題抽樣        PASS   seed 釘死、獨立重算逐題一致、題單落結果 JSON
  重點2 worktree 腳本    PASS   驗證邏輯是「worktree 對齊釘值」，釘值全來自留存證據，沒有遷就現樹
  重點3 N=64 參數鏈      PASS   64 draws 走到 collector；oracle@64、schema 末三欄、判準皆未被改
  重點4 兩態證據         PASS   自報屬實；未執行的腳本我在沙盒替身跑了正態與 6 個反態
  must-fix（會改結果、使 gate 失效、或使預註冊判準走樣的缺陷）: 0 項
  lead 拍板: 1 項（D1：s35 臂沿用 s33 的 mutant/選檔證據，偏離 ucontrast1 README:24，需 lead 明確認可）
  上機前建議動作: 5 項（A1-A5；A3 是二選一，其餘皆低成本）
```

一句話：交件的核心邏輯（釘值 gate、抽樣、衍生 source、判準）經獨立重算與真模型 CPU 實跑後，沒有找到缺陷。要 lead 處理的是一個政策拍板（D1）和幾個 sbatch 表頭與煙霧測試的風險敞口。

受檢檔未動：ucontrast1＋ucontrast2 底下 93 個檔（排除 `__pycache__`）SHA256 驗前驗後逐檔一致（`diff` 為空）。唯一副作用：我第一次 import 時在 `ucontrast2/` 留下 `__pycache__`，之後改用 `python -B`，並已把自己產生的那個 `__pycache__` 刪掉，所以該目錄 mtime 變過、內容沒變。s33/s35 ckpt 只讀（mtime 仍是 9/3，SHA 不變）。所有重跑都在 scratchpad 的沙盒替身裡，沒有碰真的 `.git`、沒有建真的 `lacot-u-frozen`。

---

## 重點 1. 50 題抽樣　PASS

驗法與證據：

- 釘死：`QUESTION_SEED = 20260929` 是 run.py:19 的常數，沒有 CLI 旋鈕或環境變數能改；同值出現在 `n64-questions.json` 的 `seed`、rollout_n64.py 衍生 source 的字面值（被 DERIVED_PIN 綁住）、run.py:155/177 與 analyze.py:52-58 的事後比對。抽法是 hashlib 的 SHA256 鍵排序，不靠 numpy 或 random，跨版本可攜。
- 獨立重算：我只看 manifest 的 `method` 字串（`SHA256 ASCII seed:task:episode; 50 smallest; task/episode sorted`）自己重寫抽法（用 hex 字典序比較，與實作的 bytes 比較不同寫法），結果與 `sample_questions()`、`registered_questions()`、manifest 三者逐題相同。Python 3.12.3 與 3.11.15（jasmine venv 同版）各跑一次，`PYTHONHASHSEED=random` 下輸出 md5 相同兩次。manifest SHA256＝`78ee1ae717ad16728008d05bfa7b6414d9acea08e062212205241928232cf684`，與交付報告宣稱一致。
- 無偏檢查：把同一鍵法換 20000 個不同 seed，200 題的入選頻率 0.2422 至 0.2588（理想 0.25，3σ 帶 0.2408 至 0.2592），各 task 平均入選 9.95 至 10.03 題；seed 確實有作用（與 seed±1 的交集 15 與 12，隨機期望 12.5）。
- 竄改偵測：manifest 換一題、改 seed、反轉順序，三種都讓 `registered_questions()` 丟 `ValueError`（沙盒實測）。
- 題單落結果 JSON：真模型 CPU 實跑（3 題、50 題、64 draws 三種規模）的輸出 JSON 都有頂層 `sampled_questions`（`[{task,episode}...]`）與 `question_seed=20260929`，位置在末三欄之前；launcher 與 analyze.py 都會拿它對 `registered_questions()`，缺、錯或順序不同一律 BLOCKED（沙盒實測 `sampled_questions` 缺、`question_seed` 錯、題單改一題、A/B 題序反轉，全被擋）。
- 每 task 題數 10/10/10/14/6：這是簡單隨機抽（非分層）的自然結果，工單寫「均勻抽」，不算偏離；lead 若想要分層要另下令。

## 重點 2. worktree 腳本　PASS

判決級問題是「對齊釘值」還是「改釘值遷就現樹」。結論：對齊釘值，釘值來自留存證據，現樹被拒。

- 釘值來源獨立核對：我直接從 `flagoff-proof.json`、`calibration-gate.json`、`mutant-gate.json`、`sigma-selection.json` 抽 `code_hashes`，再從 jasmine 的 9/28 正式主跑 preflight（`/archive/jasmine/cymaxwelllee/ucontrast1/main/*A*.preflight.json` 與 B）抽一次，六處全部一致：`scratch_lacot_rollout.py=276c68fb…ecfc`、`collector.py=7d8e2bf3…26a5`、`run.py=e78e2a70…7b7a06`；baseline `21f4cf30…6705` 等於 flagoff-proof 的 `baseline_source_sha256`，也等於 `git show 5b256d4:experiments/scratch_lacot_rollout.py` 的 SHA。腳本 6-8、42-43 行與 run.py 的 `PINNED_HASHES` 寫死的常數，都等於這組值，不等於現樹（現樹 source＝`b7d3a0d9…4a234`）。
- 沒動 ucontrast1：現樹的 `run.py`、`collector.py` 仍等於釘值；ucontrast1 下沒有任何檔的 mtime 晚於 9/28 14:30（`find -newermt`，`__pycache__` 除外）。
- 凍結 source 的出處另外獨立重建：baseline commit 的 source 套 `ucontrast1/implementation.patch` 的 scratch_lacot_rollout.py 部分（`git apply --include`），得到的 SHA 正好是 `276c68fb…`。所以腳本從 `smoke/mutants/M9/` 複製的雖是變異測試工作區裡的檔，但它的位元組與「baseline＋當時的 patch」完全相同（M9 的變異在它的 collector.py，scratch 那份沒被變異）。
- 沙盒替身實跑（`git clone --shared` 一份、複製 ucontrast1/2 並逐檔核 SHA 相同、ckpt 用 symlink 指向真檔）：腳本 exit 0，印出三個 hash 與 READY。我另用 `sha256sum` 獨立重算三檔，等於釘值；`git status` 只有 `M scratch_lacot_rollout.py`（相對 baseline 64 增 3 刪，就是 ucontrast1 的 opt-in 儀器化）與未追蹤的 `_workorders/`；`lacot/model.py` 等於 HEAD。在新樹裡跑 `validate_inputs()`：Python 3.12 與 3.11 都 PASS（含重驗 calibration、mutant、4 個 sweep 輸入的 SHA，以及 flagoff 合約的 `git show HEAD:` 比對）；`rollout_n64.source_hash()` 等於 DERIVED_PIN。
- 反態（沙盒，每案只改一處）：見重點 4 表。現工作樹（真的那棵）跑 `validate_inputs()` 得 `ValueError: BLOCKED: frozen code hash differs from pinned gate`，`derived_source()` 得 `Frozen rollout source differs from pinned code hash`。
- gate 沒被弱化到 n64 臂：n64 用 s33，與 calibration、mutant、selection 證據同 ckpt；老 gate 的各項檢查在新 launcher 都保留或更嚴（另加 provenance＋flagoff SHA 一致、證據路徑重定位後逐檔重驗）。s35 臂的差別見 D1。

## 重點 3. N=64 參數鏈　PASS

- 64 draws 走到 collector：`run.py` 的 `draws=64` → 環境變數 `LACOT_ORACLE_DRAWS=64`（我攔截 `launch()` 的 `subprocess.run` 取得實際要交給 GPU job 的 env，n64 兩臂皆為 64、`EVAL_EPISODES=40`、`LACOT_UCONTRAST2_QUESTIONS_JSON` 為 50 對、指令為 `rollout_n64.py`）→ 凍結 source 第 29 行 `Collector(..., int(env DRAWS))` → 第 3205 行 `for _draw in range(ORACLE_COLLECTOR.draws)` → `summarize(draws)`。真模型 CPU 實跑 3 題×64 draws（A、B 兩臂）：`draws_per_task=64`、`n_draws=192`、`oracle_at_k` 有 "1" 至 "64" 共 64 鍵、`pooled_oracle==oracle_at_k["64"]`、`independent_oracle` 重算與檔內一致；高 draw 索引的 seed（最大 `stream_seed=63000226`）正常。
- 與 9/28 的參數差異僅有：n64 是 `DRAWS 8→64` 與 tag／輸出路徑；s35 是 ckpt、SEED、tag／輸出路徑。把捕捉到的 env 與 9/28 主跑 preflight 的 `config` 逐鍵 diff，n64 差 4 鍵、s35 差 5 鍵（config 共 43 鍵），其餘 38 至 39 鍵完全相同（σ、CHUNK、K、EVAL_RS、BON_N、U_SOURCE 全同）。
- 衍生 source 三處替換在真模型上驗過（CPU、jasmine 同版 venv：torch 2.6.0、numpy 2.4.6、ogbench 1.2.1、mujoco 3.11.0）：只執行入選題（3 題那次只跑 3 題）；同一題的 rollouts 在「只跑它」與「跟另兩題一起跑」時逐位元相同（無跨題 RNG 洩漏）；同機重跑逐位元相同；`(task,ep,draw)` 的 `env_seed`、`stream_seed`、`flow_seed`、`initial_sha256`、`goal_sha256` 與 9/28 GPU 存檔逐列相同（3 題 6 列、50 題 A 臂 100/100、B 臂 100/100、64 draws 的前 8 draws 24/24）；B 臂 `flow_seed`、首 u 跨 draw 恆定、`u_samples==u_calls`，A 臂每 draw 獨立（50/50 題首 u 皆相異）；A/B 配對欄位全等。
- 50 題×2 draws×2 臂的完整衍生路徑（真模型、CPU）：執行的題單＝manifest 順序內容，`n_tasks/n_draws/n_success = 50/100/31（A）與 50/100/34（B）`，末三欄在最後，`launch()` 式的事後重算全部吻合。
- 末三欄合 README：9/28 存檔與所有新輸出的最後三個 key 皆為 `n_tasks, n_draws, n_success`；`launch()` 也把這件事寫成檢查（我打亂末尾三個 key 的順序並多加一個 key，被擋）。
- 預註冊判準沒被改：`analyze.py` 的 `u_stands = delta_pp >= 5 - 1e-9 and p_one_sided < .05`，p 為配對 discordant 的精確二項單向（A_only 對 B_only，與工單「配對 McNemar 單向」相符）；否則字串為「維持 9/28 未站穩」。我用 `launch()`（真 gate＋真 collector＋真事後驗證）配合假的 rollout 子行程，灌入設計好的 50×64 結果，餵給 `analyze.py`，9 個情境的 discordant、both、unlock、p、`delta_pp`、`u_stands` 全與我獨立算的期望值一致，含邊界：5:0（p=.03125，過）、7:1（p=.0352，過）、6:1（p=.0625，不過）、4:1（+6pp 但 p=.1875，不過）、3:0（+6pp、p=.125，不過）、0:4（負向，不過）。s35 模式的重放（6:0、+3.0pp）得 `A>B`、解鎖 `6v0`。
- 9/28 參考數字獨立重算：從 jasmine 存檔的 9/28 兩個結果 JSON 逐 draw 重算，A pooled oracle@8=.375、B=.345、差 +3.0pp、配對 6:0（p=.015625）、解鎖 9v2，全等於 `analyze.py` 寫死的 `reference_2026_09_28`，且「解鎖＝draw0 失敗、之後有 draw 成功」的定義能重現 9v2。

## 重點 4. 兩態證據　PASS（自報屬實）

實作官報告的每一條，我都獨立重跑過；未執行的部分我在沙盒替身補上。

```
自報項目                                        我的獨立結果
正 題單兩次重跑一致、50 對唯一、SHA=78ee1ae7…   一致（另加獨立重寫、兩個 Python 版本、隨機 hash seed）
正 衍生 source 錨替換成功、compile、SHA=4e02a2c3… 在凍結樹重現，SHA 等於 DERIVED_PIN，py3.11／3.12 都 compile
正 ckpt SHA 88180676…／ba5009ec…                  重算相同
正 三檔 hash＝釘值（腳本建後重核，自報未實測）      沙盒實跑：exit 0，三個 hash 相同，validate_inputs PASS
反 source 加一個換行→derived_source 拒絕          重現（也測了改一個位元組）
反 現工作樹→validate_inputs BLOCKED               在真的現工作樹重現（用 -B，不寫 bytecode）
反 同 arm 第二次 reserve_outdir→FileExistsError    重現；base 放雜檔→ValueError；全新 base 已有檔→ValueError
反 bash -n 五支                                    重現，五支皆過
```

我加測的反態（沙盒，每案只改一處，全數被擋且訊息明確）：

```
凍結 source 加換行 / 翻一個位元組            BLOCKED: frozen code hash differs from pinned gate
collector.py 語意 1 字元（draws < 1 → < 2）  同上
ucontrast1/run.py 語意 1 字元                同上
calibration 結果檔翻 1 位元組                Missing, failed, or stale calibration gate
mutant 結果檔翻 1 位元組                     Missing, failed, or stale mutant gate
sweep 輸入檔翻 1 位元組                      Changed sigma sweep evidence
flagoff-proof 改一對軌跡 hash                GPU flag-off three-question bitwise proof missing/mismatched
gate JSON 的 code_hash 被改成別值            stale（因為與 PINNED 不等）
selection σ 改 0.1／provenance 容忍帶改 0.2   Missing/stale selected sigma；J-signal R1 anchor contract mismatch
s33 路徑下換成 s35 的 bytes                   Frozen checkpoint bytes differ from pinned SHA256
衍生 source 錨命中 0 次、2 次                 N64 transform anchor count 0／2（連 PIN 一起改也擋得住錨數）
DERIVED_PIN 不符                              N64 derived source differs from preregistered SHA
launch 事後：pooled_oracle 或 oracle@64 被灌水、draws_per_task=32、n_draws 少 1、n_tasks=49、
  題單缺／seed 錯／改一題、某 draw 的 success 位元被翻、ckpt 欄錯、末三欄被打亂     全部 ValueError
analyze：結果檔驗後被改、preflight 驗後被改、verified 缺、B 題序反轉且刷新 verified   全部拒絕；summary 第二次寫入 exit 2（exclusive）
worktree 腳本：目標已存在／M9 副本改一字節／ckpt 內容被換／ckpt 缺／baseline 釘值錯   全部 BLOCKED（exit 2），且不建 worktree
```

---

## 其餘全面查

### 5. lead 拍板項 D1：s35 臂沿用 s33 證據，偏離 README:24（非實作錯誤）

`ucontrast1/README.md:24` 寫「s35 須另跑其 σ0 mutant、獨立粗掃與選檔，再用其 selection」。老 `launch()` 對 s35 會因 `mg["checkpoint"]` 與 `selection["seed"/"ckpt_sha256"]` 都不是 s35 而拒絕。新 launcher 的 `validate_inputs()` 把 calibration、mutant、selection 全部綁 s33，s35 直接沿用並固定 σ=.05。實作官在自我懷疑欄有揭露，且工單寫「口徑與 s33 判決完全同」，所以這是合理讀法；但工單另有「不改 gate 邏輯；gate 拒＝BLOCKED 回報，不繞」，兩句在 s35 上互相拉扯，實作官選了前者。我不替 lead 裁：

- 若 lead 認可「s35 是同協定重現、σ 沿用 s33 的選檔」：請在預註冊裡寫下這一句，讓它不是事後補的。技術面我已驗過 s35 臂能跑（見 N5）。
- 若 lead 認為 s35 須有自己的 mutant 與選檔：目前 launcher 不支援，需補 s35 的 mutant＋sweep＋selection 三道（以 9/28 實測約 mutant 21 分、sweep 23 分×4，三卡並行約 1 至 1.5 小時牆鐘）。

flag-off 證據與 calibration 只綁 s33 沒有問題，前者驗的是程式碼等價，與 ckpt 無關。

### 6. 時間與資源（上機前建議動作 A1、A2 的依據）

實測數字取代報告裡的估計：

```
sacct 9/28 主跑 33906_0／_1：Elapsed 03:04:15／03:04:16（兩臂並行）、Timelimit 12:00:00、AllocCPUS 8、ReqMem 16000M
  => 1600 rollouts／臂／3.07 小時 => 6.9 秒／rollout（含啟動）；50 題子集平均步數為全集的 1.003 倍
  => n64 每臂 3200 rollouts ≈ 6.2 小時；s35 每臂 1600 ≈ 3.1 小時；smoke：s35 約 23 分，n64 約 7 至 8 分
  報告的估計（n64 24 GPU 小時、s35 12、9/28「約半天」）高估約 2 倍（GPU 小時）到 4 倍（牆鐘）；方向保守，但與實測不符
```

- `--time=24:00:00` 不會生效：`ada-lite` 的 `MaxTime=12:00:00`（`scontrol show partition`），叢集用 lua job_submit 且 `EnforcePartLimits=NO`；9/28 的 main.sbatch 同樣寫 24:00:00，`sacct` 顯示 Timelimit 12:00:00，等於被靜默夾成 12 小時。n64 每臂預估 6.2 小時，有 1.9 倍餘裕，不致出事，但表頭說謊、且沒有續存機制（結果只在收尾一次寫出，逾時或崩潰會失去全部）。
- `--cpus-per-task=1` 是相對 9/28（8 CPU）的未實測變更，報告未列為偏離。連帶預設記憶體降為 `DefMemPerCPU=2000`（MiB，cgroup 強制）。9/28 的 MaxRSS 約 1.22 GiB（1,263,884K 至 1,279,884K，即約 1,250 MiB），佔 2000 MiB 的約 62%，不會直接爆，但餘裕比 9/28 少；速度影響不明（GPU 端 kernel 啟動綁 Python 單執行緒，理論上不敏感，但沒有量）。
- QOS：9/28 的 job 跑在 `great-mage`（最高階，不被別人的 QOS 搶占），所以 REQUEUE 搶占風險低；不列項目。

### 7. n64 煙霧測試沒有走衍生路徑（A3 的依據）

`launch()` 在 `smoke` 時 `questions=None`，所以 n64-smoke 跑的是原凍結 source（ep 40–44、2 draws），從未經過 `rollout_n64.py` 的三處替換與 `exec`；衍生路徑的第一次 GPU 執行會是正式主跑。報告的自我懷疑寫「待 jasmine smoke／主跑」，容易讓人以為 smoke 會涵蓋它，其實不會。我已在真模型 CPU 上把衍生路徑實跑過（重點 3），GPU 專屬風險趨近於零（替換只動純 Python 控制流，不含 device 相關程式），所以列為建議而非 must-fix。另外我把 n64-smoke 的設定（ep 40–44、2 draws，A、B 各一次）在真模型 CPU 上跑過：形狀 25 題／50 draws、A/B 配對、salt（`env_seed=2001051` 即 `1000*1+40+2000011`）、`independent_oracle` 皆通過，A draw0 R1＝.40（在 README:23 的 .15–.60 帶內）。s35 主跑設定的 A 臂另跑過 draw-0（見 N5）。

### 8. `/scratch/cymaxwelllee` 是實作官新造的路徑（A4 的依據）

兩個 smoke sbatch 把輸出寫到 `/scratch/cymaxwelllee/ucontrast2/…`。repo 內別處沒有這個慣例（我 grep 過 `_workorders` 與 `experiments` 全部 sbatch），zeldajr 上沒有 `/scratch`；jasmine 上有沒有、可不可寫，我看不到（未驗）。失敗模式是良性的：`reserve_outdir` 的 `mkdir` 會在啟動 GPU 之前丟 PermissionError 被 BLOCKED。另一個不便：`/scratch` 若是 jasmine 本機碟，smoke 的 `verified.json` 從 zeldajr 看不到。相對地，`/archive/cymaxwelllee/ucontrast1/` 確定存在、可寫、且從 zeldajr 以 `/archive/jasmine/cymaxwelllee/ucontrast1/` 讀得到。

### 9. worktree 腳本的次要毛病（notes）

- 建後斷言（腳本 41-43 行）失敗時 exit 1 且沒有任何訊息，並留下一個已登記在 `git worktree list` 的半成品，之後重跑會被 `target already exists` 擋住。什麼情況會走到：`ucontrast1/run.py` 或 `collector.py` 在來源樹被動過（前置檢查只驗 baseline、M9 副本、兩顆 ckpt，沒驗這兩檔）。我用沙盒竄改 run.py 實測：rc=1、沒印 READY、半成品殘留。清理指令：`git -C ~/Projects/lacot worktree remove --force ~/Projects/lacot-u-frozen`。建議把這兩檔的 SHA 檢查搬到前置階段並印 BLOCKED。
- `target` 給相對路徑會得到假 READY（沙盒實測，rc=0、印出三個 hash 與 READY）：`git -C "$source_root" worktree add` 把 git worktree 建在 `$source_root/<相對路徑>`，而其後的 `mkdir`、`cp`、`ln`、`cd` 與建後斷言都以 shell 的 cwd 為基準，作用在另一個普通資料夾。預設（不帶參數，絕對路徑）不受影響，所以請照報告不帶參數執行，或只給絕對路徑；建議腳本開頭加 `case "$target" in /*) ;; *) exit 2;; esac` 之類的檢查。
- `cp -a` 會連 `__pycache__` 一起複製，mtime 保留所以 pyc 仍有效；良性。跑驗證時用 `python3 -B`。
- `/home` 是 zeldajr 對全叢集的 NFS export（`/etc/exports`：`/home 172.16.165.0/24(rw,…)`），jasmine 看到同一份 `~/Projects`，所以在 zeldajr 建 worktree、jasmine 上 `cd $HOME/Projects/lacot-u-frozen` 是通的，ckpt symlink 也會解析（`sha256`、`torch.load` 都經 NFS）。

### 10. gate 的碼雜湊只涵蓋三個檔（notes，非本件新增）

`lacot/*.py` 不在雜湊內。現主樹相對 HEAD 有 `lacot/model.py` 修改與 5 個未追蹤 `refine_*.py`（mtime 9/28 12:33 之後，與 9/28 gate 期間重疊），凍結樹用 HEAD 版。`model.py` 的 diff 只在 `LaCoTActor*.losses_given`／`training_losses`（訓練損失），rollout 唯一用到的 `RefineOperator` 未改；實證上，HEAD 版 lacot 在 CPU 重現 9/28：reset／seed 100% 相同、成功位元一致率 95/100（A）與 96/100（B）（CPU／GPU 浮點差異造成軌跡分歧的正常量級），s35 的 A draw-0 R1 為 .390（J-signal noclimb s35 為 79/200＝.395）。

### 11. n64 用 s33 是推論

工單對臂甲沒寫 ckpt；實作用 s33，與「維持 9/28 未站穩判決」和 50 題來自 s33 的 200 題池吻合，也是唯一有配套 gate 證據的 ckpt。請 lead 確認這是意圖。

---

## 一、必改（must-fix，上機前）

無。我沒有找到會改變結果、使 gate 失效、或使預註冊判準走樣的缺陷。

## 二、需 lead 拍板

- D1：s35 臂沿用 s33 的 mutant／選檔證據並固定 σ=.05（第 5 節）。拍板前 s35 臂不建議提交；n64 臂不受影響。

## 三、上機前建議動作（都低成本；不做不會出錯，但敞口放大）

- A1：四支 sbatch 的 `--cpus-per-task=1` 改回 8（等於 9/28 已量測的包絡：3h04m、MaxRSS 1.22 GiB、16000M），或保留 1 並加 `--mem=16G`。理由：這是報告未列的偏離，速度與記憶體餘裕都沒量過，而失敗代價是 6 小時 GPU。
- A2：n64.sbatch／s35.sbatch 的 `--time=24:00:00` 改成 `12:00:00`，並把報告的 ETA 改成實測值（n64 每臂約 6.2 小時、s35 約 3.1 小時）。
- A3：n64 的衍生路徑沒有 GPU smoke（第 7 節）。二選一：接受我的真模型 CPU 證據；或請實作官加一個小規模衍生路徑 smoke（需要讓 `rollout_n64.main()` 接受非 50 題的清單，會動到受檢碼，要重審）。
- A4：提交 smoke 前在 jasmine 上先查 `ls -ld /scratch /scratch/cymaxwelllee`；不可寫就把兩個 smoke 的 `--outdir` 改到 `/archive/cymaxwelllee/ucontrast1/smoke2-…`（已知可寫）。
- A5：建 worktree 時請不帶參數（或只給絕對路徑，見第 9 節相對路徑的假 READY），建好後先做兩步再提交：
  ```
  cd $HOME/Projects/lacot-u-frozen
  python3 -B -c "import sys; sys.path.insert(0,'.'); from experiments._workorders.ucontrast2 import run as L, rollout_n64 as R; L.validate_inputs(); print('gate PASS', R.source_hash()[:16])"
  ```
  預期印 `gate PASS 4e02a2c38994a9e3`（沙盒已重現）。並比對本檔末的 SHA 表，確認複製進 worktree 的 ucontrast2 檔就是我審的那版。

## 四、notes（資訊，非缺陷）

- N1（煙霧驗收門檻，建議寫進提交清單）：s35-smoke（200 rollouts／臂，與 9/28 smoke 33885 同形）Elapsed 應約 22 至 24 分（實測基準）；超過約 35 分〔拍，約 1.5 倍〕視為 1 CPU 拖慢。用 `sacct -j <id> -o JobID,Elapsed,MaxRSS,State` 看 MaxRSS 是否約 1.2 GiB（9/28 基準 1.20 至 1.22 GiB）。A draw-0 R1（25 題）落在 [.15,.60]（README:23）。
- N2（迴歸錨，正式主跑收工時用）：50 題子集在 9/28 GPU 主跑 draws 0–7 的成功題數，A 為 [17,15,14,15,16,16,15,15]、oracle@8＝17/50；B 為 [17,17,17,17,16,17,17,15]、oracle@8＝17/50（我從存檔重算）。n64 的 seed 與 reset 已驗與 9/28 逐位元相同，若 GPU 逐位元確定，n64 兩臂 `per_draw_quality[0:8]×50` 與 `oracle_at_k["8"]×50` 應完全等於上列；不等要先查確定性與接線，再解讀 @64。s35 錨：A draw-0 R1 應在 J-signal noclimb s35 的 .395（79/200）±.08 內，我的 CPU 實跑為 .390。
- N3（子集組成，影響「怎麼解讀」，不是實作問題）：9/28 在 8 draws 內「動過」的 9 題（A 解鎖 9、B 解鎖 2，含 6 題 A_only）全在 task 3，episode 為 2、4、6、7、9、18、23、26、34；登記的 50 題含 task 3 的 10 題（1、3、8、16、19、21、24、25、31、36），與這 9 題完全不相交。隨機抽 50 題全避開 9 題的機率是 7.1%，不算離譜；抽法不依賴結果、我也驗過無偏，沒有跡象顯示被挑過。但這意味著子集在 8 draws 時 A、B 都是 17/50、discordant 0:0，n64 若過門檻，靠的全是 draws 8–63 的新解鎖。
- N4（預註冊的有效門檻）：N=50 時每題 2pp，「+5pp」那條腿不綁：任何能過 McNemar 單向 p<.05 的結果都已 ≥ +10pp（最小過關形是 5:0＝+10pp；7:1、6:0 也過）。所以實際門檻等於 McNemar 那條。這是預註冊本身的性質，不是實作偏差；提醒 lead 別把「差 +6pp 但 p 不顯著」讀成「接近過關」。
- N5（s35 臂技術面已驗）：真模型 CPU、s35 ckpt、與 launch() 相同的 env，A 臂 200 題 draw-0 R1＝78/200＝.390（對 .395）；輸出 schema、題單（5×40 格）、`independent_oracle` 重算皆通過。
- N6：沒有交付單元測試檔；我的探針腳本留在 scratchpad（第八節）。`analyze.py` 沒有自己的測試，我用合成結果驗了 9 個情境。
- N7：結果只在最後一次寫出（`write_result`），崩潰或逾時即全失；這是 9/28 設計就有的性質，n64 拉長到 6 小時後代價變大。可選：每 draw 落一份 JSONL 供續跑；不建議為此重審。
- N8：FYI，非審查項：這 200 次 rollout（50 題×2 draws×2 臂）在 zeldajr CPU（4 執行緒，同時有別的 job）用了 8 分 21 秒，約 2.5 秒／rollout，比 jasmine GPU 的 6.9 秒快；但 CPU 與 GPU 的軌跡會分歧（成功位元一致率 95–96%），不能與 9/28 的 GPU draws 混算。只列數字，不建議改路徑。
- N9：分析指令可在 zeldajr 用 `/archive/jasmine/cymaxwelllee/ucontrast1/n64` 的路徑讀 jasmine 存檔（`analyze.py` 只用標準函式庫，主樹或 worktree 都能跑），summary 會寫進該 `--outdir`；報告寫的 `/archive/cymaxwelllee/...` 路徑只在 jasmine 上成立。

## 五、未驗（未驗不等於通過）

- jasmine 側一切：`/scratch` 是否存在可寫、sbatch 表頭是否被 Slurm 接受（我沒有跑 `sbatch --test-only`）、cpus=1 下的實際速度與 RSS、GPU 逐位元確定性與 9/28 前 8 draws 是否重現、衍生路徑在 GPU 上的第一次執行。
- 真模型的 64 draws×50 題完整規模：真模型只跑到 3 題×64 draws 與 50 題×2 draws；50×64 是用假 rollout 灌真 collector 與真 launch()／analyze 驗的（機制驗證，不含真結果分布）。
- 對真的 `.git` 執行 `git worktree add`：我在 `git clone --shared` 的替身 repo 上跑（tmpfs、路徑不同）；真 repo 的其他 worktree、hooks 的影響未驗。
- 沒有重審 ucontrast1 自身的碼（前次 S2 已審）；只驗了它被 ucontrast2 引用與釘住的部分。
- Slurm 搶占：只看到 QOS 設定與 9/28 job 的 QOS，沒有壓力測試。

## 六、測試環境差異聲明（我的驗證 vs jasmine 正跑）

```
                   我的驗證（zeldajr）                        jasmine 正跑
裝置                CPU（torch 2.6.0+cu124 於 CPU，無 CUDA）    GPU（同版 torch），CUDA 浮點路徑
軟體堆疊            .venv：py3.11.15、numpy 2.4.6、ogbench 1.2.1、mujoco 3.11.0、gymnasium 1.3.0、scipy 1.17.1，與 jasmine venv 逐項相同
執行緒／CPU         OMP=4（真模型跑）；gate／分析測試無關      OMP=1，cpus 1（待 A1 決定）
檔案系統            沙盒在 tmpfs；資料在 ~/.ogbench/data        NFS home＋/archive 本機碟
排程                無 Slurm，SLURM_JOB_ID 由測試腳本偽造       真 sbatch、cgroup 記憶體限制
軌跡                CPU 與 GPU 逐步浮點分歧，成功位元一致率 95–96%（reset、seed 100% 相同）
```

## 七、受檢檔 SHA256（前 16 碼；完整值可由檔案重算，驗前驗後一致）

```
867a3402e6de6952  ucontrast2/analyze.py
be6302065cf301d7  ucontrast2/rollout_n64.py
97d32d8a6fdc707a  ucontrast2/run.py
d0874ccc370b2968  ucontrast2/create_frozen_worktree.sh
01431f12be8b80da  ucontrast2/n64.sbatch
924910e300261efd  ucontrast2/n64-smoke.sbatch
706930911927937d  ucontrast2/s35.sbatch
eeef046bc66d512d  ucontrast2/s35-smoke.sbatch
78ee1ae717ad1672  ucontrast2/n64-questions.json   (完整 78ee1ae717ad16728008d05bfa7b6414d9acea08e062212205241928232cf684)
adfacf07005b9c7c  out-impl-ucontrast2-last.txt
釘值：scratch_lacot_rollout.py=276c68fba6c41a05…  collector.py=7d8e2bf3b8760388…  run.py=e78e2a704317bf47…  DERIVED_PIN=4e02a2c38994a9e3…
```

這份 VERDICT 只對上表這一版檔案成立；任何一檔改動，請重審。

## 八、證據位置與重跑

全在 scratchpad：`/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/f8052efa-ca20-4e69-a6e3-98158ead28b0/scratchpad/`

- `probe_sampler.py`：抽樣獨立重算（第 1 點）。
- `sbx/`：沙盒替身（`repo/` 是 shared clone＋byte-identical 複製，`frozen/` 是腳本建出的 worktree，`neg/`、`neg2/` 是反態替身）；`neg_harness.py`、`neg_harness2.py` 跑 gate 反態。
- `capture_launch.py`＋`captured_env.json`：攔截 `launch()` 得到的實際 env（第 3 點）。
- `probe_928.py`：9/28 參考數字與子集錨重算。
- `pipeline_dry.py`、`pipeline_neg.py`：假 rollout 灌真 collector 的 launch()／analyze 全鏈與反態。
- `tiny_driver.py`、`tiny_run.sh`、`plain_run.sh`、`analyze_tiny.py`＋`tiny_out/`：真模型 CPU 實跑與其分析（3 題、50 題、64 draws、s35、smoke 設定）。
