# VERDICT S2 (Claude 側檢察) : 突破雙探針 probe harness 三棒交件

日期 2026-10-01 | 檢察官 S2 (Claude Sonnet 5.5, 對抗審查) | 契約基準 PROBE-LAUNCH-CARD.md v6.2 (sha256 34634832ccaf...)
受檢目錄 /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/ (28 檔, 不含本 VERDICT; 我審前後逐檔 sha 比對相同, 未改、未 commit)

在證明什麼、判準: 證明「實作 = 契約」, 判準 = 每一條契約句都對得到碼上的行為, 且該行為由我自寫探針或真跑獨立重算過; selftest 與 IMPL-NOTES 的自我宣稱一律不採信。

## 0. 判定

**PASS-with-notes**

- 契約違反 (FAIL 級): 0 條。凍結源 / sha gate / streams / d1-d5 / d4 機械判準 / trace 契約 / call 上限 / 注入送達 / gate P 與 R 來源 / 規模 108 / fail-closed, 全部有獨立證據 (第 2 節)。
- 上機前 (GPU smoke) must-fix: 1 條 (A1, 程序陷阱, 非科學錯), 另有 1 條強烈建議 (A2)。
- 正式前 (vFinal) must-fix: 7 條 (B1-B7)。主題: 合讀出口只有純函式沒接線、selftest 對 runtime 協調層與 Move 1 判決算式零牙齒、落盤與時限。
- 沒驗到: GPU 上的一切 (第 8 節)。未驗不等於通過。

一句話: 實作碼本身我找不到 bug (CPU 真迷宮、真模型端到端全規模 smoke 庫存跑通, 128 + 14 rollouts 與獨立 oracle 全相符); 但「selftest 30 PASS」不能當作這些性質的證據, 因為對 83 個實質受測碼變異體, 只殺掉 48 個 (見第 4 節), 沒被殺的集中在最關鍵的 runtime 協調層。

## 1. 基準與前置核對

| 項 | 我怎麼驗 | 結果 |
|---|---|---|
| 凍結源 sha | sha256sum M9/scratch_lacot_rollout.py | 276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc, 命中 PIN |
| ckpt sha | sha256sum results/ckpt_...s33.pt vs common 的 CKPT_PIN vs ucontrast1/provenance.json | 88180676e1f8..., 三方一致 |
| collector | 實際被 import 的檔: sys.modules[...collector].__file__ 的 realpath + sha | ucontrast1/collector.py, 7d8e2bf3b876... (不是鄰居 M9/collector.py, 那份是 oracle 洩漏的壞 mutant, sha 0a63b93d...) |
| 正規化資料集 | zeldajr 與 jasmine 兩份 pointmaze-large-stitch-v0.npz 的 sha | 同為 9add335e598e...; harness 只記錄、未 pin (見 B7) |
| 審前審後 | 28 檔逐檔 sha vs 我的鏡像; find -newer IMPL-NOTES.md 掃整個 repo | 無差異、無新檔、無 __pycache__; git tracked 檔零修改 (ucontrast1 與 HEAD 相同) |
| 基線 selftest | 在鏡像目錄 python3 -B selftest.py | 30 tests OK (4.73 s), 與 selftest-output.txt 逐行相同 (僅耗時秒數不同) |
| 輸入稽核 | third-baton-input-audit.json 內 11 個輸入 + 10 個 .py 的 sha 逐一重算 | 全相符 (沒有過期稽核) |

鏡像法: 把整個目錄複製到 scratchpad, 以 symlink 掛 ucontrast1 / lacot / results / .venv; 所有變異、端到端、fuzz 都在鏡像做, 受檢目錄零寫入 (本 VERDICT 是唯一新檔, 副檔名 .md, 不在 receipt 的 *.py glob 內)。

## 2. 契約 vs 碼, 逐條證據

| # | 契約句 | 怎麼驗 | 結果 |
|---|---|---|---|
| 1 | 題集: 主判 4:14 + 5:6, 對照 task2:10, gate task1 七題 + task3 ep16 | 自寫腳本從 gate0b rows 用卡面規則重篩 (a_bits/b_bits/n_shortest), 與 builder.json 逐題比 | 全同 (gate = 1:10,15,20,23,25,32,34 + 3:16) |
| 2 | 幾何 / 互斥段 / w 整點 / route-A 字典序 / 單路 task2 | 自寫 BFS + 路徑 DP 計數 + DFS 全枚舉 (不 import builder), 比 routes / branch / merge / exclusive / w / DAG 全節點與邊 | 5 個 task 全相符; task4 D=14 d*=4 w=(1,6)/(5,10); task5 D=15 d*=2 w=(1,3)/(3,1); task2 D=19 w=(3,5); DAG 節點 20/20/20/22/20 全比對 |
| 3 | R donor 表 (無 task 2) | 自算各 task 最小 ep | 4<-5 ep5, 5<-4 ep4, 1<-3 ep16, 3<-1 ep10, 無 key '2' |
| 4 | streams: base=7t+ep, stream=base+1000003*draw, draw 64..79, 與舊 0..63 交集為零 | 對全 50 題枚舉; seeds() 對公式; draw 護欄 | 舊 2432 個 stream、新 608 個, 交集 0; 公式逐題相符; 63/80 等被拒 |
| 5 | N 的實際 seed 分支 (A 型 fresh, 非 legacy, 非 B 型) | 真模型: collector.first_u 對「torch.manual_seed(stream_seed) 後 flow.sample」; 對照 legacy 7t+ep | 3 組 (task,ep,draw) 全等於 stream 版、全不同於 legacy; Collector.arm=='A' |
| 6 | 動作噪音 = 全臂同 stream 同 sigma=.05 | 獨立 Generator(stream_seed)+randn 重算 150 步 (3 組) 對 noisy_action | bit 相同; 且 A/C 兩臂 rng_before.noise 相同、不同 draw 不同 |
| 7 | 協定等價 (harness N 臂 = 凍結源自己的 oracle A 臂路徑) | 在真迷宮 CPU 上, 用凍結源的 policy_chunk + ORACLE_COLLECTOR 重現 rollout() 的 oracle 分支 (補一行 _bon_plan 存根, 該函式在 import 邊界 2480 之後, BON_N<=1 時逐字等於 sample_plan), 跑 160 步與 run_draw(N) 比 | 3 組 (4,4,64) (1,10,64) (5,5,71): 動作序列與 XY trace 位元相同 |
| 8 | 凍結源載入 / sha gate / 不 sed | 讀碼 + 測: load_frozen 先 import_boundary()=verify_source() 再 import torch; 邊界行 2480 (AST 唯一匹配, 'line' 事件在該行執行前拋例外); 夾 SOURCE 到壞檔 -> Blocked 在 import 前 | 成立; 邊界後的名稱 (_bon_plan 等) 確實未定義, 所以 harness 用自己的 sample_plan+ahead 取代 policy_chunk (已由第 7 列等價測試背書) |
| 9 | d1-d5 語義 | (a) rules.classify vs harvest.recompute_move3 隨機差分 4000 組; (b) 獨立 oracle 對真 artifact 128 列重算; (c) 真環境 cell 映射下 d4 判準 | (a) 0 不一致; (b) 128/128 category、w_step、ledger 全相符; (c) 本臂折線=True、對臂折線=False、先本臂後跳對臂=False (task4/5 x A/B 全中) |
| 10 | d4 機械判準 = 入本臂專屬段>=1 且對臂=0, 全程 trace | 讀 harvest.exclusive_crossing:120-131 + 上列 (c) + 變異 H4/H5 被殺 | 成立, A/B 對稱 |
| 11 | trace 契約 len=steps+1<=H+1, 不 padding, 終止原因必填 | 真 artifact 全列檢; validate_rows 負例; fuzz 1500 次 | 成立 (steps 範圍 0<steps<=1000; 缺 'steps<=H' 的負測見第 4 節 H2) |
| 12 | model-call 上限 ceil(H/chunk)+2 = 252, 超限 = 無效且不入任何分母 | 差分 + 自寫判例 B1-B7 | 成立; 實際最大 call = 251 (w 在 4m+1 步命中), 252 不可達 |
| 13 | 換段: 命中 w 當步即刻重抽、丟棄舊 chunk 剩餘動作 | 真 Move3 artifact: 所有 w%4!=0 的命中皆在 step=w 有 target=g 的 call, 無 w-call 在命中後、無 g-call 在命中前; fuzz 1500 次 vs 獨立 oracle | 128 列 0 異常; fuzz 0 failure (涵蓋 d1/d2/d3/d4, cap/stuck/success 與命中同步、chunk 邊界) |
| 14 | 落格 / 平手歧義 / 主導數排除 / d5 分流 | 自寫判例 A1-A8 + 變異 R6-R9 | 成立 |
| 15 | gate: 合格題>=5、P>=ceil(3/4 q)、R<=2、獨立出口名 | 自寫 behavior_gate 9 格 (q=0,4,5,6,7,8) | 全如卡; 但測試套件沒釘 ceil 與 q=5..7 (見第 4 節 R10) |
| 16 | gate P 來源 = 該題 N 成功軌跡 (self-rollout-u); R 用 donor route-A | 真模型重編碼: 對 Move1 artifact 每列 encoded_u_sha256 重算候選 (10 條路線 + 各題 N trace) | 8 列 (P/Q/R x 主判 + gate) 注入張量全如契約 (見 p6); 重採樣前處理與我獨立的弧長 128 點實作逐位元相同 |
| 17 | 注入送達 (逐 chunk head u hash) 與「被 flow 覆蓋」偵測 | 真 artifact 上的 24 種竄改 (下表) | 21 種拒收、1 種正確標記 (conformance=False)、2 種是設計上抓不到 (見 2.1) |
| 18 | 確定性 + 同 u 重跑 (CPU) | e2e Move1 smoke: 在真迷宮重跑 (4,4,P/Q) | 兩條 conformance True; 用竄改 1e-3 的重跑列, 收割正確回 False |
| 19 | 規模 108 | 計畫列計數: 主 80 + gate N 8 + gate P/R 16 + 重跑 4 | 108 (smoke 版 18 = 12 跑 + 2 重跑 + 4 因 task5 表示層不過而跳過, 與 validate_artifact 獨立重導的期望集合一致) |
| 20 | E0-E4 出口 / 分支表 | 讀 rules.exits/e2/ceiling/e4 + 判例 D/E 組 | 純函式對卡正確 (E2 嚴格 >50%, E3 11/14 與 5/6, 主導差>=2, 行選擇第一命中, d5 主導獨立格); 但管線未呼叫, 見 B1 |

### 2.1 真 artifact 竄改矩陣 (Move 1 e2e 產物, 24 例)

拒收 (21): 缺 gate P 列 / 缺主判 R 列 / 重列 / N 錯 seed / 注入列帶 flow seed / 錯 reset 指紋 / xy 改一點但沒修 hash / injection_source 標錯 / R donor 竄改 / P 帶 flow_calls (被 flow 覆蓋) / head hash 中途不符 / head hash 少一格 / P 的 encoded_u 與 representation 證據不符 / skipped 少一筆 / skipped 偽造一筆 / receipt 的 code hash 竄改 / receipt.smoke 翻面 / builder 竄改 / representation 證據 u_A,u_B 對調 / noise_band 設極大使表示層不過但列仍在 / gate P 的 source_trace 雜湊錯。
正確標記 (1): 重跑列 xy 被改 -> 收割回 conformance False, 不 raise。
設計上抓不到 (2): (i) xy 竄改並同步重算 trace_sha256 (無獨立真值來源, 屬 receipt 信任邊界); (ii) goal_success 末位翻 True 但 termination_reason 仍 truncated (validate_rows:93 只驗「success 蘊含末位為 True」, 缺逆向, 見 B5)。

## 3. 寫入面 / sbatch / fail-closed

- 寫入面: 在 Move 1 smoke 端到端外掛 sys.addaudithook 記所有開檔寫入與 mkdir: 只有 outdir/result.json (+ .sha256) 與 TMPDIR 下的 MuJoCo / torch 暫存檔。write_json 用 'xb' 獨占建檔, outdir 已存在則 Blocked, 輸出必須在 /archive/cymaxwelllee/breakthrough1 下 (runtime.py:229)。sbatch 有 PYTHONDONTWRITEBYTECODE=1, trap 清 mktemp。
- jasmine (唯讀探查, 未提 job): /home 是 zeldajr NFS (repo / gate / card / ogbench 原始碼 / ckpt / M9 全可見); /archive 本機 ext4; venv = Python 3.11.15, torch 2.6.0+cu124, CUDA True; /archive/cymaxwelllee/breakthrough1/ 已存在 (含 gate0b/), 子目錄 move1/move3/smoke 尚不存在 (正合 fresh-dir 要求); n64 兩份歷史檔 sha 與 gate0b receipt 一致; 在 jasmine 以 CPU 跑 harvest.historical_fingerprints -> 50 題指紋全取得。
- sbatch: 3 支 bash -n 通過; sbatch --test-only 三支全接受 (partition ada-lite, nodelist jasmine, gres gpu:1, 皆有效; 未建立任何 job, squeue 為空)。ada-lite: MaxTime 12h, PreemptMode=REQUEUE (叢集 PreemptType=preempt/qos)。資源 vs 碼: cpus=1 與 torch 單執行緒一致; mem 16G 夠 (Move3 全規模估 JSON 約 166MB, RSS 數 GB 內)。
- 時限: 三支都是 00:30:00 佔位 (卡內已註明非 ETA)。我的 CPU 實測 Move3 smoke 庫存 (128 rollouts) 耗 443 s, 單 rollout 約 3.5-4 s (實測 env.step 僅 0.04 ms/步, 1000 步 0.04 s, 時間幾乎全是模型: 由 3.5-4 s / 250 chunk 推算約 14-16 ms/chunk); 全規模 1280 在 CPU 約 75 分鐘; GPU 未測 (小批次 GPU 可能受 kernel 啟動延遲限制, 不保證比 CPU 快)。smoke 估計夠 30 分, 全規模肯定不夠 (見 B4)。
- fail-closed: 實際執行 5 種入口, 全部 exit 2 並印 BLOCKED: harness_move3.py / harness_move1.py 無 --smoke (含附校準檔) -> "PRODUCTION_READY=False"; --smoke 缺校準 -> "explicit smoke calibration required"; --smoke 在本機 (無 n64) -> "historical reset reference unavailable"; smoke.py 缺參數 -> argparse exit 2。未建立任何輸出目錄。注意: 旗標只擋 CLI 入口 production(); runtime.execute(smoke=False) 本身無旗標檢查 (內部 API, 僅提醒)。

## 4. 假 PASS 掃描: selftest 30 項有沒有牙齒

方法: 對受檢碼逐一施加單點文字變異 (每個變異獨立複本、跑完整 selftest), 85 個條目中: 83 個為實質變異, 1 個無改動對照 (S3, 應存活, 實際存活, 證明工具無誤殺), 1 個樣式錯誤作廢 (X6)。

結果: 殺掉 48 (含 B1 因 builder 內建斷言在 setUp 崩潰), 存活 35。

被殺的代表 (測試確實有牙): 凍結源 sha gate 失效 (C1)、reset 三行順序與 seed (C7 C8)、injected 列帶 flow seed (C5)、noise seed (C6)、reach <= rho (S1)、cap 差一 (S2)、stuck 標成 cap (S4)、collector 換 B 臂 (S6)、兩個 PRODUCTION_READY 閘移除 (S9 M9)、warn_only=True (M1)、表示層 check 忽略 round trip (M8)、trace hash 檢查 (H1)、seed 檢查 (H10)、polluted 檢查 (H11)、d4 only-own / only-other (H4 H5)、同步 w=g 當 bypass (H7)、ledger 異常 (H8 H9)、gate 非條件執行 (M5)、重跑只做 P (M10)、gate P 源改 route-A (M11)、tie 不歧義 (R7)、d5 併入 DEEP (R8)、A/B 對調 (R9)、w_unreliable >= (R5)、C 成功不再優先 (R6)、E2 >= (R13)、follower 閾值 (R16)、p 閾值 (R19)、E0 失效 (R20)、donor 加 task2 (B3)、路線命名反轉 (B2)、gate P 源檢查移除 (H17)。

存活, 依實質度分類:

高 (決定性路徑上沒有任何測試守):
- runtime 協調層, 全部存活: X1 命中 w 後不即刻重抽 (循環 break 拿掉)、X2 cap 係數 x2 改 x1、X3 R 注入自己的 route A 而非 donor、X4/S5 Move3 動作噪音關掉、X5 stuck 永不觸發、X7 條件向量永遠用最終目標 (w 追逐失效)、X9 N 失敗的 gate 列照跑、X10 主判不被 >=5 閘擋、X11 gate P 改用幾何路線而非 self-rollout、X12 接口自測整段略過、X14 路線反向編碼。原因: selftest 完全沒有跑 runtime.execute() 的測試; test_23 只跑 5 步合成環境加 cap=1 的真環境單步。這些性質我已用端到端 / fuzz / 重編碼各自驗過 (第 2 節 #6 #13 #16), 目前碼是對的, 但回歸時 selftest 抓不到。
- Move 1 判決核心算式, 存活: H18 n_diff 強制為 0、H19 conformance 恆為 True、H13 復現判準不看成功、H15 主判啟動閘 5 改 4、H20 n64 A/B 指紋一致性檢查、H3 head chunk 計數檢查。原因: test_28 的合成 artifact 讓 P/Q 的 xy 全同, n_diff 永遠 0, 重跑永遠相同, 沒有一格正向例。

中 (邊界沒釘): R10 ceil 改 floor (只測 q=8)、R12 gate 最低題數 5 改 4 (只斷言 run_main)、R14 E3 task5 閾值 5 改 4、R15 主導差 >=2 改 >=1、R17 吸附 ceil(n/2) 改 floor、R18 停損 <3 改 <2、M7 零長度點保留 (np.interp 非嚴格遞增)、H2 steps<=H、H12 成功後仍有觀測。
低 / 等價: C2 draw 護欄收 63 (計畫本身不產生)、R1 R2 H6 (252 邊界不可達, 實際最大 251)、M2 TF32 旗標 (mock 沒斷言)、M3 M4 (FixedInjection 前後兩道檢查互相冗餘, 擇一移除不影響)、X8 (GRAD_REFINE=0 時快取惰性)。

mock 邊界誠實度: smoke.py inventory 與 IMPL-NOTES 大體誠實 (GPU 項都標 pending, 不編 ETA)。兩處用語偏強: (i) 'representation-pass-rollout-fail: CPU fixture passed' 實際只測了 rules.e2 純函式, 管線沒有 E2 路徑; (ii) test_23 的 REAL_CPU_PASS 印出 'four P/Q same-u reruns', 實際是 5 步合成環境; 我另在真迷宮 CPU 補驗過 (兩條 conformance True), 所以結論成立, 只是測試比標籤弱。

## 5. 自寫判例 (全部對鏡像、碼未改)

- 41 格判例 41/41 通過: 平手 / 歧義 A1-A8 (含 d2=d3=16 在各臂 d2=8 時不得誤成 INSUFFICIENT; d1 全滿 -> AMBIGUOUS; C 觀測不足先於平手); budget / 無效流轉 B1-B7 (9 無效 + 7 有效 d4 -> INSUFFICIENT; 超限的 C 成功不得救回; 250 call 的 C 成功有效、253 call 無效); bypass / 同步 C1-C3; behavior_gate 九格; E2/E3 邊界; exact p; E4 行選擇與停損旗標 E1-E7。
- 差分: rules.classify vs harvest.recompute_move3 隨機 4000 組, 0 不一致。
- fuzz: runtime.run_draw (真 TwoStage + 真迴圈 + 真 recompute) 對獨立真值 oracle, 腳本化軌跡 1500 次, 隨機臂 / cap / stuck 窗 / 命中時刻 / 成功時刻, 0 failure; 覆蓋 d1 80, d2 678, d3 4, d4 184, C 成功 552、失敗 2 (d5 因 verifier 固定為 True 沒生成, 由 3 組真映射判例補)。
- 端到端 (CPU, 真迷宮, 真 s33, 真 n64 指紋): Move1 smoke 17.2 s, Move3 smoke 442.8 s, 收割 CLI 輸出與跑內 readout 位元相同。

## 6. 必改與建議

### A. 上機前 (GPU smoke)

A1 [must, 中高, 程序陷阱]: receipt 與收割綁「活的、會變的」檔。harvest.py:112 把 receipt.card_sha256 對「現在磁碟上的」PROBE-LAUNCH-CARD.md; harvest.py:113 把 code_sha256 對「現在 HERE/*.py 的 glob」(common.py:87,90 同源)。卡在 vFinal 前必被改 (ETA 要寫進去; 卡的 mtime 19:07 就在實作期間又動過), 目錄裡多一支 .py 也一樣。後果: smoke 產物在卡一改之後收割一律 Rejected('receipt launch card hash')。修法二選一, 不可皆缺: (a) 把 card 與 gate 複本凍進 breakthrough-probe/frozen/ 並讓 receipt 與收割對複本; code hash 改固定檔名清單; (b) 寫進發射官 SOP: smoke 收割完成前不改卡、不加 .py。

A2 [強烈建議, 中]: runtime.py:331-334 先 validate_artifact 後寫檔, 任何 Rejected 都讓整支 job (smoke 約 8-10 分) 零 artifact 可除錯。修法: 先把 raw (rows / receipt / representation) 獨占寫成 raw.json, 再驗; 驗失敗時另寫 validation_error。smoke 是第一次上 GPU, 失敗機率不低, 這條的投資報酬最高。

前置條件 (非缺陷, 發射官須備): 校準 JSON 六鍵 (dataset_dir, steps_per_cell, stuck_window, stuck_distance, noise_band.4, noise_band.5) 沒有範本、沒有估計工具; 缺鍵要到模型載入後才 KeyError。建議附 calibration.example.json (標 provisional) 並在載模型前先檢查鍵。我的佔位值 (window 20 / dist 0.2) 讓 task2 A 臂 16/16 在第 29-44 步被判 stuck (起步慢), 可見 stuck 窗必須用 smoke 實測校準, 不可沿用任何猜值。

### B. 正式前 (vFinal / 發射官放行前)

B1 [must, 高]: 合讀出口沒有接線。grep 證實: rules.e2 / ceiling / e4 / adsorption / exact_one_sided / content_sensitivity 在 harvest / runtime / harness 全無呼叫點 (exits 只被 --fixtures 的合成報表呼叫)。收割輸出沒有 B-vs-C discordant 的 forward / reverse (幾何層)、E2 分子、N 天花板、停損旗標; conformance 為 False 時也不會自動把 n_diff 語言降級。這些要靠人手把數字餵進純函式, 手算口徑無碼可對。修法: 加一個確定性 combine 步驟 (從 artifact 導出各輸入並套 rules.exits), 並把 conformance False -> content_sensitivity 降級語言寫進輸出。在此之前, 任何人不得把 harvest 輸出稱為「判讀」。

B2 [must, 高]: 補測試牙齒 (第 4 節高 / 中存活)。至少: (a) Move1 判決: P/Q xy 不同時 n_diff>0; 重跑 xy 不同時 conformance False; 復現須含成功; q=5..7 的 ceil 閾值; 主判啟動閘 5 的邊界; head chunk 計數負例; (b) runtime: 把我的 stub-module + 腳本化環境 fuzz (p15) 收進 selftest (純 CPU, 不需模型, 5 秒內), 加一個 Move3 端到端微型測試釘住噪音 / cap 係數 / stuck / 換段 / R 與 gate P 的注入內容; (c) 規則邊界: 主導差 2 vs 1、E3 task5 4/6 vs 5/6、停損 2 vs 3、吸附 ceil。

B3 [must, 中]: Move 1 沒有撞牆 / 卡死紀錄。卡寫「未進 + 撞牆卡死獨立記」, 且 E2 分子含「撞牆 或 卡死 或 未進段 或 無效」; 目前 Move 1 呼叫 run_draw 不傳 stuck_check, E2 只能算「未進段」下界, 會系統性低估失效率而少觸發 E2。需 lead 先凍結 Move 1 的 stuck 判準 (卡尚未凍結), 凍結前 E2 結果須標「下界」。

B4 [must, 中]: 落盤、時限、心跳。全規模 Move3 約 166MB JSON (我測 130KB/列: RNG 快照占約 80%, 卡估 <40MB), 單次寫入、無分段落盤, ada-lite 有 REQUEUE 搶佔; 30 分佔位不夠 1280 rollouts; 整個 run 沒有任何進度輸出 (哨兵無從得知進度)。修法: 逐題 flush rows.jsonl; 每 N 個 rollout 印一行; --time 依 smoke 實測 x2 定; 卡內「<40MB」更正或把 RNG 快照改存雜湊加 checkpoint 檔。

B5 [must, 中]: 收割不變量缺口 (都不貴): (i) termination_reason == 'success' 與 goal_success[-1] 雙向 (validate_rows:93 只有單向, 見 2.1 (ii)); (ii) chunk_steps == 4 (現在只驗 >0, 而 ledger 期望與 call 上限都用 row 自報的 chunk); (iii) receipt.interface (E0 接口自測: mask_all_false、[1,8,256]/[1,4,2]) 存在且正確 (現在 X12 把整段略過也能過); (iv) R 與 gate P 的注入內容目前只驗標籤 / 自報雜湊: 建議每列多記 injection_traj_sha256 (餵進 encode 前的折線雜湊), 收割用 builder 幾何與 N 的 trace 在 CPU 重算; (v) rng_before.torch_cpu / noise 可由 seed 重算比對 (我已驗 140 列全相符), 把「標籤 seed」升級為「實際 seed 證據」; (vi) 任務 2 對照集成功列現在標 d5 (rescue-unverified), 應改標專屬類別避免與主判 d5 混讀。

B6 [must, 中]: PRODUCTION_READY 放行程序。selftest.py:387-388 與 runtime_selftest.py:84 把兩旗釘死 False, 翻旗即 selftest FAIL; 且翻旗要改 harness_move1.py / harness_move3.py, 使「S2 審過的 sha」不等於「實跑的 sha」。修法: 旗標移到獨立非 .py 檔 (或發射官環境變數), selftest 對旗標中性 (測「旗標 False 時入口擋、True 時放行」兩個分支), S2 對僅旗標差異簽字。

B7 [must, 低中]: 校準與 pin。noise_band 沒有估計器 (卡: 同路重抽噪聲帶, gate 幾何校準), 只能手填; 手填可為任意小值使「距離 > 帶」恆真 (我的 CPU 距離 52 / 31 遠大於任何合理帶, 所以此次無影響); cap / stuck 無校準腳本; 資料集 sha 只記錄不 pin (兩機同, 但無 fail-closed)。修法: 補校準腳本並把輸出入 receipt; load_frozen 加 DATASET_PIN。

### C. 建議 / 給 lead 的契約模糊 (不算實作缺陷)

C1 d1 主導格卡沒定義: 我的判例 "A=15 個 d1 + 1 個 d3, B=16 個 d1, C 全掛" 回「深執行洞候選」(單一 d3 draw 決定)。同源問題: w-miss 率現為 d2 / 有效 draw (d1 在分母、不在分子), 卡文可兩讀。建議卡明文。
C2 復現事件定義: 實作取「首入互斥段同路 且 P 成功」; 卡文「互斥段 / 終點歸屬與源軌跡一致」可讀成 or。影響 gate 通過率, 建議卡明文。
C3 task5 層在目前 s33 上表示層 round-trip 即不過 (CPU 真跑: decode(E(route-B)) 在 B 專屬段 0/128 點、A 專屬段 15/128 點, 路徑還穿過牆格 (2,2)(2,3); task4 層 A 30 點 / B 41 點歸屬正確)。預期 E1 該層除名, Move 1 實質只剩 task4 層 14 題。這是模型性質, 不是實作 bug, 但會影響「判決重心」的樣本數, 建議 smoke 前先讓主人知道。
C4 demo 標籤: builder 固定 unverified, 沒有任何核實嘗試 (卡允許)。後果: 吸附判讀永遠停用, 只能報「內容不敏感」弱判。
C5 base 碰撞: Move3 題集內三組 (base 33: (2,19)/(4,5); 37: (2,23)/(4,9); 58: (4,30)/(5,23)), 其中兩組跨對照 / 主判; 卡只舉 58。exact 分層不受影響, 但主導數把題當獨立。
C6 凍結源住在 ucontrast1/smoke/mutants/M9/, 隔壁 collector.py 是故意壞的 mutant; 該目錄被重生或清理即 PIN 失效。建議複本入 frozen/ 並在 receipt 記雙路徑。
C7 ada-lite 會被 REQUEUE 搶佔 (preempt/qos); 結果確定性所以重跑結果相同, 但時間全丟 (見 B4)。
C8 harness 不用凍結源的 policy_chunk (其 _bon_plan 在邊界後未定義), 而是重寫 sample_plan+ahead。等價性我已用位元級測試背書 (第 2 節 #7), 建議把該測試收進 selftest (需一行 _bon_plan 存根)。
C9 smoke 建議時限提高到 60 分 (逾時 = 全丟, 且 smoke 的產出之一就是時間)。

## 7. 我跑過的指令與輸出摘要

(探針腳本在 /tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/0e65b72e-3d4a-41dd-99f0-671eeb533b4a/scratchpad/probes/, p1-p19 + mutate.py + driver_e2e.py; 結果檔 mutation-results.json)

- p1 幾何 / 題集獨立重算: main 4:14 5:6, ctrl 2:10, gate [(1,10),(1,15),(1,20),(1,23),(1,25),(1,32),(1,34),(3,16)]; 5 個 task 的 routes / w / exclusive 全相符; p10 DAG 全邊比對 OK。
- p2 streams: 舊 2432 / 新 608 / 交集 0; 公式逐題相符; 9 組 base 碰撞 (全 50 題), 3 組在 Move3 題集內。
- selftest 基線 + 85 條變異 (mutate.py): killed 47, survived 36 (含對照 S3), crash 1 (B1, 視同殺), 作廢 1 (X6)。
- p4 / p4b 竄改矩陣 24 例 + 重跑 conformance 傳播; p4b grep: rules.content_sensitivity / e4 / e2 / ceiling / adsorption / exact_one_sided 在管線無引用。
- driver_e2e.py 1 與 3: 真迷宮 + 真模型 + 真 n64 指紋 (從 jasmine 唯讀取出), CPU。Move1 smoke: rows 12, repeats 2, skipped 4, conformance 兩條 True, gate qualified 2 (smoke 不能證 >=5), task5 表示層不過。Move3 smoke: rows 128, 0 無效, 0 ledger 異常; task4 A/B 皆 16/16 命中 w (A 步 73-93, B 步 80-94) 後 g 失敗 (d3), task5 A 16/16 (步 39-45)、B 15/16 (步 41-47) 命中 w 後 g 失敗, task5 B 另一次 cap@48, task2 A 16 次 stuck (我的佔位校準), C 臂全失敗。
- p6 注入內容重編碼比對 8 列全 OK; p8 128 列獨立重算 0 不符; p13 真映射 d4 判準 OK; p14 Move3 重跑與跨臂 noise 配對 OK; p15 fuzz 1500 次 0 failure; p16 協定等價 3 組位元相同; p5 / p11 / p12 噪音 / seed 分支 / collector 路徑 OK; p18 RNG 快照與 seed 重算 140 列全相符; p17 擴縮: 全規模 JSON 外推 166MB, 驗證時間可忽略。
- ssh jasmine 唯讀: hostname, ls, df, mount, sha256sum, scontrol show partition / config, squeue, sbatch --test-only (三支), 以 jasmine 的 LaCoT venv (CPU, 不碰 GPU) 跑 harvest.historical_fingerprints 取回 50 題指紋。未提交 job、未用 GPU、未寫任何檔。
- 5 種 fail-closed 入口試跑: 全 exit 2。

## 8. 未驗 (不等於通過)

1. 任何 GPU 行為: use_deterministic_algorithms(True, warn_only=False) 在 CUDA 上對 flow.sample / encoder / decoder 是否有 op 會 raise (我靜態掃了 nf_head / e_target / model / traj_decoder, 無 cumsum / scatter / index_add 等, 風險低但沒跑); GPU 上同 u 重跑是否位元相同; cuBLAS workspace 設定實效; 單 rollout 速度。
2. 全規模 (Move3 1280、Move1 108) 實跑; 全規模下的 JSON 寫入與收割 (只外推)。
3. 校準值 (cap 步 / 格、stuck 窗與距離、noise band) 的合理性; 我只證明「給任何值, 管線忠實執行」。
4. demo 標籤 oracle (OGBench stitch 生成器的 tie-break) 未核。
5. gate0b 的 a_bits / b_bits 與 n64 結果本身 (我只用規則重篩題集, 沒重算 gate0b 的底層位元)。
6. Slurm 實際排程 / 搶佔行為 (只驗 --test-only 與 partition 設定)。
7. Move 3 w 段條件向量與凍結源 policy_chunk(goal=waypoint) 的位元級等價 (只對 Move 1 的 N 臂做了位元級; w 段靠讀碼 + e2e 命中 w 的事實)。
8. 多人共用下的資源爭用。

## 9. 自我懷疑欄

- 變異集是我手挑的 85 個, 偏向我讀碼時覺得關鍵的點; 存活率 (35/83) 是下界估計, 真正的測試盲區可能更多; 殺掉率也不能反映未被我想到的失誤型態。
- 我的獨立 oracle (fuzz 的 truth()、p8) 是依卡文與碼的共同語義寫的, stuck 窗語義、同步事件的優先序 (命中先於成功先於 stuck 先於 cap) 取自碼的行為而非卡的明文; 卡對這些優先序沒有逐字規定, 若 lead 意圖不同, fuzz 的 0 failure 不能證明對。
- 端到端用的是 CPU, 且校準是我的佔位值; 所有「d3 / stuck / cap」的實際分布都只證管線, 不是科學結果, 不可拿去讀。
- Move 1 的 task5 表示層不過是 CPU 上一次確定性前向的結果 (表示層 check 對幾何而非對題, 全規模同一幾何結果相同); 解碼路徑離邊界很遠 (B 專屬段 0/128 點), 我判斷 GPU 數值差異不會翻盤, 但未在 GPU 重算。
- 我說「碼沒 bug」的範圍僅限於我讀過 / 跑過的路徑; 對 harvest.py 的 Move 1 分支我讀過全部並做了竄改矩陣, 但 GPU 專屬路徑 (如 torch_cuda RNG 快照的實際型態) 只靠推理。
- ssh jasmine 與 e2e 佔用了 zeldajr / jasmine 的 CPU 時間 (最多 6 個並行 selftest + 2 個 e2e), 時間短、無寫入; 若主人認為這超出「驗證性指令」範圍, 責任在我。
- 關於 A1 是否算 must: 它是程序陷阱不是科學錯, 我選擇升到 must 是因為出現機率高 (卡幾乎必改) 而代價是整批 smoke 產物收割失敗; 若 lead 以 SOP 吸收, 可降為 note。
