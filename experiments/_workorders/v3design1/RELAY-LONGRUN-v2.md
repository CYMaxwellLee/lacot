# 接力棒長跑 F6 v2：呈主人終裁的契約文本

## §0 修訂地位與證據錨〔回應 M1–M6；FAIL 1–8〕

在證明什麼：無人看管的多小時自主推進，不退化成密度農場，也不因平台限制失去交接。保留 v1 的五項首航判準，以 §7 的尺驗收。
本件只修設計；**不是主人批准、實作完成或首航 PASS**。下列規則及數值均為呈裁提案；主人批准完整包後才生效，缺批准即禁止啟航。
基稿：[RELAY-LONGRUN.md](RELAY-LONGRUN.md)，SHA256 `c8db09e69a31b18f4503e4eaeed215f9938ad1c940dd864bdfbf7507ad95f8ae`；逐條回應 [S2 全文](VERDICT-RELAY-S2.md)，不改原稿。
判決索引：FAIL 1→§3–5；2→§2、§4；3→§5；4→§3、§8；5→§4.5；6→§5.5；7→§7；8→§6。M1→§3；M2→§2；M3→§4；M4→§5；M5→§4.5、§5.5；M6→§7。
制度先例：[ucontrast 發射權澄清](../ucontrast1/LEAD-CLARIFICATION.md) §4、[凍結資產與逐關 gate](../ucontrast1/README.md)「Gate 順序」、[patchrun 發射卡](../patchrun1/LAUNCH-CARD.md)「F5 鏈位置」。它們證明制度慣例，不把舊 GO 當本夜授權。
完成錨先例：[patchrun 收割](../patchrun1/HARVEST-REPORT.md) §1–4；CPU／scratch 先例：[EXP1 卡](EXP1-LAUNCH-CARD.md) 及 [預檢](EXP1-PREFLIGHT.md) C1–C3；新 run／保留失敗與 smoke interlock：[opt1 卡](../opt1/S0-LAUNCH-CARD.md)、[opt1 預檢](../opt1/S0-PREFLIGHT.md) §9。

## §1 為什麼是接力不是馬拉松〔回應 M2；FAIL 2〕

v1 以 Anthropic 九迴圈的單 agent 長跑作對照；這裡沿用其論證，不新增外部實證：session 會凍死、午夜會重啟，長跑應靠不失真的交接，而非假定跑者不死。
v1 自述背景約 58 分的限制尚待首航前確認。**棒是 90 分鐘排程時段，不是 agent 存活 1–2h**；agent 工作段 ≤50 分，另留 5 分交接，最遲第 55 分結束。若實測平台更短，須縮短並重裁，不能硬撐。
sbatch 脫離 session 只保護 job 存活；授權、帳與交接另由本契約保護。快審最多 5 分、到期深審最多 15 分，各為獨立工作段；未完成閘不因下一時段到來而放行。

## §2 架構、所有權與停棒協議〔回應 M2；FAIL 2、8〕

架構：主人睡前裁預裁樹 → 獨立 timer 喚棒 → agent 交提案 → 唯一執行閘核准／記帳／執行 → 原子交接 → S5 → 下一棒。F5 ⓪①②③④仍管棒內真跑。
timer／監督器不依賴上一棒排程；每 15 秒檢查租約、工作期限、審計期限及持久停態。timer 只請求啟動，無發射權；Restart 不等於恢復批准。
單一 coordinator 是 state／帳本提交者；agent、S5 只交具身分提案。戰役租約綁 owner、epoch、boot_id、expiry，60 秒有效、每 15 秒續租；用單調時鐘判期限，跨開機租約一律失效。
取得租約、budget 預留、intent 提交及外部提交在同一序列化閘內；每個外部動作即時驗 epoch、phase、工作／夜間／審計期限，不能利用監督器 15 秒輪詢間隙越限。過期 owner 不能寫 state 或發射；若平台不能隔離舊 owner 的憑證，首航 NO-GO。
啟動閘驗批准錨、schema／hash、revision 鏈、停態、磁碟容量／inode、帳本完整、未決 intent、既有 jobs 與應到審計；任一不明拒啟，不能建空帳、借舊 OK 或直接載舊快照續跑。
交接用不可覆寫 generation：先持久寫事件、jobs／budget／audit／progress 及各 hash，再寫 manifest、fsync，最後以 expected revision 的 CAS 原子切換 state 指標並 fsync 目錄；state 指向完整 manifest 才算 commit，僅有 manifest 不算已提交。
每一中途副作用亦有獨立 durable transaction，不能等棒末才落帳；取消／寫產物同樣先記 intent 再作用，控制帳自身的 append/CAS 則以其原子交易記錄為證，不遞迴另造 intent。`progress.md` 與 state 引同一事件 ID／generation；未提交片段留存為 orphan，回放不得當已完成棒。
發射順序固定：驗 gate → 同交易預留額度與 attempt → fsync `INTENT` → 以唯一 `intent_id` 提交 → 記 receipt／cluster/job/task → 對帳。idempotency key 綁 campaign/action/attempt/task，換棒不換 key。
提交中斷先依 gateway 提交帳及 scheduler 的唯一 token 查詢；0 筆僅在提交程序已被 fence、controller 確認全時間窗完整且證明未提交時才算 NOT_SUBMITTED；1 筆收養入帳；多筆或未知即停。首航不自動重送任何未決 intent。
STOP 是持久安全閘：停止新科研副作用、fence agent；只准已裁的留痕、唯讀對帳、審計與精確取消本戰役可控制 jobs。不能刪產物、搬移凍結檔或取消他人的／僅觀察的正式 job。
每個新 job 必須先帶可驗身分、取消授權及 scheduler 硬時限；STOP 時取消其 queued/running tasks，保留預留至終帳。取消失敗記 UNCONFIRMED，靠已預留的硬上限封頂；不能宣稱已停止耗用。
獨立控制儲存保存 STOP latch／准入許可；啟動預設 deny，停態或許可缺失、控制儲存不可用均拒啟。資料盤寫不成時由監督器關許可；全機失聯則許可過期，重啟仍拒啟，不靠失敗的「寫停態」當唯一保護。
STOP 後唯有主人可發具 hash 的 resume 批准；lead 須附可信快照＋完整事件回放、未決 intent 結案、job／資源對帳、故障排除、適用 F5／S5 證據。新 epoch 接續原計數／原窗口，禁止藉改 campaign 洗帳；FINISHED 不恢復。bootstrap 僅准信任根登記的新戰役用一次，既有 campaign 缺 state 必走恢復，不得再初始化。

| 戰役 phase／觸發 | 唯一下一狀態與准許動作 |
|---|---|
| 未批准／初啟不過閘 | STOPPED；只留缺件單；批准完整 bootstrap 包且全閘過才 READY |
| READY；時段到且取得租約、全閘有效 | RUNNING；僅執行命中的已裁 action |
| RUNNING；正常完成交接 | AUDIT_PENDING；本棒 COMMITTED，關新科研副作用直到 §5 判決 |
| WAITING；同一依賴新事件或定時觀測 | 僅唯讀核身分／期限；仍有效則 WAITING，依賴完成且全閘過→READY；失效／到期→STOPPED |
| RUNNING；第 50 分未結束工作 | 停新 action，交接；第 55 分仍未 commit→ABORTED／STOPPED |
| 任意非終態；夜間 action 截止時間到 | STOPPED(reason=NIGHT_END)；取消可控制 jobs、對帳及尾段深審，無自動續夜 |
| 任意；故障／硬違約 | STOPPED；按下表原因落帳，壞 state 時以控制 latch 等價拒啟 |
| STOPPED；無主人 resume | 持續 STOPPED；timer、遲到審計及 job 完成都不能清停態 |

十種意外採下列互斥條件；同時發生多故障一律 STOPPED、保留全部原因。判定先看故障／持久 STOP，再看終點、應到審計、等待，最後才判 READY 的啟動許可；終點亦先關閘完成應到審計，新故障可把 FINISHED 降為 STOPPED。正常交棒原子釋放 agent 租約，審計／等待／兩棒間無 agent 租約不算 OWNER_LOST；coordinator 的准入許可仍須有效。

| 意外 | 唯一出口（首航不做故障自癒） |
|---|---|
| 1 job FAIL／TIMEOUT／非預期 CANCELLED | STOPPED(JOB_FAILURE)，留 scheduler state/exit；infra 失敗不是科學否定；取消的連鎖回報不重複觸發取消 |
| 2 機器掉／agent 被殺／租約失效 | STOPPED(OWNER_LOST)，該棒 ABORTED；監督器或重啟啟動閘落停，先對帳而非接管續發 |
| 3 形狀怪／少檔／NaN／完成錨不符 | STOPPED(INVALID_ARTIFACT)；不得把 parse error 當 predicate=false |
| 4 預算不足、燒超、計量失聯 | STOPPED(BUDGET_GATE)；取消可控制 jobs，真超額另標契約違約，不回填分類 |
| 5 S5 crash／格式壞／timeout／缺回覆 | STOPPED(AUDIT_ERROR 或 AUDIT_TIMEOUT)，不重抽、不沿用前棒判決 |
| 6 state 壞／失蹤／hash 不符 | STOPPED(STATE_INVALID) 的 latch 拒啟；可信舊版只能用於唯讀重建，不能直接發射 |
| 7 兩棒重疊 | 第二棒取租約失敗→該棒 ABORTED(OVERLAP)，無寫入／副作用；既有 owner 續按原閘；若雙方都持有效權的證據出現→戰役 STOPPED(OWNERSHIP_BREACH) |
| 8 Slurm 排不上 | 已裁 job 身分／依賴／觀測有效且未到 queue deadline→提 WAITING，RUNNING 棒先 commit 經 S5 才進 WAITING；已在 WAITING 則維持；其餘→STOPPED(WAIT_INVALID_OR_EXPIRED)，不改 partition、不重送 |
| 9 盤滿／inode 用盡／寫入失敗 | STOPPED(STORAGE_ERROR) 或 latch 等價拒啟；不清盤、不刪 ckpt，未能留痕標證據缺失 |
| 10 上棒未完整 commit／提交後 ID 未落帳 | STOPPED(INCOMPLETE_COMMIT)，該棒 ABORTED；查 intent 收養或列未決，不把 orphan 當完成或再提交 |

## §3 夜間決策權與凍結授權模板〔回應 M1；FAIL 1、4、5、6、8〕

預裁樹仍取 3–5 個成果分支、每支額度；lead 睡前提出、主人裁整包。批准包含本契約、樹／里程碑、已知知識基線、模板實值、故障／取消動作白名單、資源與驗收表。
批准包 bytes 的 SHA256 由主人批准紀錄固定於 agent 不可寫的信任根；紀錄含 approver、approved_at、source_ref、package_hash。包不含自己的 hash 以免循環；state、action、milestone 以 approval_ref 回指該 hash。
下表為規範性模板，`<…>` 必須睡前填實值且通過 schema；不是可執行示例或預設授權。所有 ID 永久穩定、唯一；改 ID／別名不改等價身分或累計。

| 模板 | 必填欄位／凍結含義 |
|---|---|
| branch | `branch_id, order, purpose, condition{validator_ref,hash,input_refs,expression}, action_ids, milestone_ids, budget_ref, terminal_kind, approval_ref`；order 只供呈現，不解決多命中 |
| action | `action_id, branch_id, experiment_id, dead_end_id, purpose, kind, classification, classification_reason, approval_ref, pre_predicate, post_predicate`；kind 限 READ/HARVEST/SMOKE/CANCEL/CONTROL；classification 限 ADVANCE/NECESSARY_VALIDATION/DENSITY，CONTROL 另帳 |
| action 執行面 | `argv_allowlist, script/code/env_hashes, cwd, input_manifest{paths,hashes}, output_manifest{new_paths,schema_hash,count,completion_anchor}, write_allowlist, overwrite=false, dependency_ids`；路徑解析到固定資產，不准 latest 別名或未列動態依賴 |
| 輸入綁定〔N1；移植 ucontrast LEAD-CLARIFICATION §3 先例〕 | 兩類輸入分開凍結：**既有資產**（含 t0 前既有 job 的產物）＝睡前 SHA256＋路徑逐字進批准包，缺 hash 不得為輸入；**夜間新產物**＝上游 action 的 output_manifest（schema_hash/count/完成錨）由凍結收割器驗過才可當下游輸入，收割記錄其實測 hash 後即凍結——下游 input_manifest 以 `producer_action_id+manifest_slot` 引用、執行閘核實測 hash 相符。⛔ 無 manifest 驗證的產物（含 latest 別名）不得入任何 input |
| action 限額／F5 | `seed_set,dose_set,step_cap,array_tasks,max_invocations,cluster,partition,nodes,gpu_type_count,cpu_count,memory,wall_limit,kill_grace,queue_deadline,completion_deadline,bytes/inodes_cap,F5_refs{⓪,①,②,③,④},sentinel_spec,cancel_scope,result_validator_hash`；數量均有界 |
| milestone | `milestone_id,version,branch_id,outcome,dependency_ids,predicate{validator_ref,hash,expression},evidence_contract{source,version,schema_hash},baseline_ref,semantic_rubric_ref,approval_ref`；只能凍結成果，不得拆成讀 log／抄 log 小格 |
| night／控制面 | `campaign_id,t0,action_end,close_deadline,slot=90m,limits,known_jobs,controller_actions,writer_acl,storage_limits,verifier_versions,acceptance_ref,approval_ref`；任何所需讀寫／取消／監督行為也須在白名單 |
| 棒語義〔N2；凍結防挑語義湊 PASS〕 | `actions_per_baton`（每棒至多執行的已裁 action 數）、`slot_mode`（fixed_timer 或 event_triggered）睡前凍結；「等 job」的唯一形狀＝該棒先 commit 再入 WAITING（棒內⛔不輪詢等待）；§7 的「≥3 工作棒」只計有已裁 action 執行或有效收割判讀的棒——純 WAITING 觀測棒與 STAGNATION 棒不計；批准包附覆蓋可行性算式（由 known_jobs 終態時刻＋slot 推出預期棒數，達不到最小覆蓋則首航改期，⛔ 不夜間挑語義補棒） |

所有產物 exclusive write；僅 CONTROL 白名單中明列的 state 指標／許可／租約可依 §2 CAS 替換，不能覆寫歷史或科研證據。無資源／F5 適用性的控制動作亦填有理由的 NOT_APPLICABLE，須睡前批准，夜間不能自填豁免。
條件採 TRUE/FALSE/UNKNOWN/INVALID；依賴尚在合法等待先走 §4.5，不拿未到產物判分支；決策點先驗輸入 provenance／schema，再算 predicate。UNKNOWN（缺值）／INVALID（壞證據）、零分支命中或多分支命中均 STOPPED，不能以預設 false、優先序或自由判斷補洞；正常樹終點亦須顯式終端分支。
`terminal_kind` 限 NONE/TREE_EXHAUSTED/BLOCKED_BY_AUTHORITY；後兩者 action_ids 為空、只附待晨請求，不把禁止的正式命令塞進可執行清單。非終端分支的 action 序列／依賴凍結；多個已就緒 action 依已裁序號取首個，不能夜間改排程意圖。
成果 predicate 由凍結 validator 重算；語義格引用 S5 的 rubric 與證據判決，不能宣稱全機械可判。runner 只交產物，不能改 desc、門檻、分類、已知基線或手填 done。
smoke 必須是主人逐 action 明批的 SMOKE，命令／script／資料／參數／步數／seeds／array／資源／寫入面逐一匹配，且當前 F5 gate 有效；短時、名稱含 smoke 或舊 GO 均不構成許可。
**scratch ckpt 預設禁止，首航維持禁止；所有新 ckpt 寫入均不代發。** 若主人要例外，另裁精確 action/hash、scratch 路徑、產物數／上限及無覆寫證據並重驗本契約；未裁例外不能引用 EXP1 的 scratch 先例自開。
F5 接點：lead 睡前提供適用 ⓪① 與逐 gate 證據；執行閘比對目前命令和凍結 hash，② 點火驗證與 ③ 生理哨兵由已裁 hook 啟動，④ 由凍結收割器驗 schema／完成錨。③ 原則隨真跑掛，豁免須逐 action 主人明裁。
下一 action 的必要 F5 結果缺失即 STOPPED；smoke exit=0 不代替 schema、科學資格或正式 GO。任何新 script／資料／參數使原 gate 不再適用，夜間不得自行修腳本過關。

## §4 防退化契約：可信計帳與死路〔回應 M3；FAIL 1、2、8〕

保留四條：成果里程碑階梯、加密度 ≤20%、新知識心跳、同死路有界重試；修正「三條半機械」說法：分類／等價／語義先裁，機械只驗凍結規則及可信輸入。
20% 約束本戰役可控制實驗 jobs 自 `t0` 起的**每一時間前綴及終帳**，不只棒末；`action_end` 停新動作並取消殘留，終帳到最後 task 終止，超過 `close_deadline` 仍未對清則未驗，不切掉尾耗用。
計量 lane 分開：每種 GPU 型號各用「配置張數×配置秒」，CPU 實驗用「配置 core×配置秒」（GPU job 的 CPU 亦記 CPU lane）；不用利用率、不同 GPU 不任意換算。每個非空 lane 均須 D/T≤0.20。
T=該 lane 全部已裁實驗耗用，D=其中凍結 DENSITY 耗用；ADVANCE／NECESSARY_VALIDATION 計 N=T−D。重複／加 seed／加劑量點預設 DENSITY，必要驗證例外須主人逐項裁理由；未知／爭議不發，S5 不可改成較寬分類。
failed／取消／重試／array 每個 task 與 requeue 每次 allocation 的耗用均入原分類；pending 不算實耗但全額預留。首航禁自動 requeue。T=0 且證明無耗用時比值定 0，仍不能免 §7 有效覆蓋；來源缺失不能填 0。
跨夜既有 jobs 只讀觀察，不收養為可發／可取消 jobs，其本夜耗用另列 inherited_observed、未知也標明；不納本契約 D/T、不拿歷史 GPUh 稀釋。新發 F6 jobs 不許跨夜甩帳，尾帳沿原戰役。
coordinator／收割控制工作與 S5 模型成本另列 overhead（CPU 秒、API 次數／token／費用上限），有獨立硬額度且不進 T；不能把實驗改名為 overhead。記憶體、儲存 bytes／inode 亦分別有硬額度，不發明算力換算率。
每個 task 提交前原子預留最大耗用 U=配置資源×(wall_limit＋站點可證明 kill_grace)，加總 total／branch／lane；`available=cap−settled_used−outstanding_reservation`，排隊／在跑／未決 intent 都占預留。
在跑 task 的 reservation 保持全 U，直到可信終帳以實耗上界替換，禁止重複相加或提前釋放。若 scheduler／wrapper 無法保證含終止延遲的硬 U，該 action 不准發。
密度信用只用已終態且對帳的非密度耗用下界 `N_verified`：每次預留後須 `D_upper + R_density ≤ N_verified/4`（逐 lane；D_upper 為已結密度上界，R 含全部未結密度 U）。以來源誤差向不利側取整；不借未來 N；D=20,T=100 時沒有餘額再發 10。
上述不等式保守保證任一前綴 `4D≤N`；N=0 時不得先發密度。總額／分支額度和密度閘同交易檢查，兩台機器不能各花同一餘額。
執行中由 scheduler 強制 allocation／wall／requeue 禁令，監督器每 15 秒查可控制 jobs、分類、reservation 和限制。原始 accounting 來源、查詢時間、單位與誤差上界須留存；終態帳最長容許延遲 120 秒，超時或矛盾→STOPPED，預留不釋放。
允許延遲期間仍按 U 計上界，不授信用；資源重疊、輸入 hash 漂移或未列依賴使受影響結果 INVALID/CONTAMINATED，停其後繼，保留全部耗用與證據，禁止補跑抹帳。
死路等價睡前凍結：`experiment_id` 綁假說／資料版本／對照／方法；`dead_end_id` 綁該 experiment 的失敗命題與等價參數集合（含 seed／path 別名）；`attempt_id` 唯一標一次授權嘗試。新 seed／改 path 不自動變新死路。
「同一死路重試 ≤2」明定為首次 attempt 後至多兩次重試（最多三次），跨棒／resume 累計；重試須有逐 action 授權，這個數字本身不授權。首航 scientific/infra 自動重試上限均為 0，job 失敗即 §2 停待晨。
嘗試在 durable intent 時占次數；只有可信 NOT_SUBMITTED 證明才能留事件解除占位。requeue 若意外發生仍逐 allocation 入帳並視硬違約；未知 failure class／等價歧義停待主裁，不由 runner 改類清零。科學否定只由有效成果 predicate 判，不由 exit code 推導。

## §4.5 推進引擎〔回應 M5；FAIL 1、5〕

棒的預設任務仍是推成果一格：收割→驗成果→按樹判讀→執行允許下一步。只有凍結成果 predicate 首次成立，或對照睡前基線證明一條未排除分支應排除，才可提 PROGRESS；行政 diff、改 evidence 字串、pending↔blocked、拆 M1a／M1b 都不算。
里程碑 execution status 限 PENDING/VERIFIED/INVALID；waiting／blocked 另記活動分類，不能來回改成果刷進度。VERIFIED 若證據後來失效，留 invalidation 事件並 STOPPED，不倒回 PENDING 重做。
合法等待必填 `wait_id,milestone_id,dependency_id,cluster/job/array_task,started_at,queue_deadline,completion_deadline,last_observed_at,raw_state_ref,next_check_at`；依賴在批准 DAG、job 可唯一找到且確為 PENDING/RUNNING，觀測不超過 60 秒。
同 dependency 的 deadline 自首次 intent／起等固定，換棒／換顯示 ID 不展延；排隊 deadline 取已裁 action 值，完成期限取已裁絕對值與 action_end 較早者。查不到、已終態仍稱等待、未列依賴、過期觀測均 STOPPED。
WAITING 每 60 秒或 job 事件作唯讀觀測，不為等候湊空棒；COMPLETED 先過產物契約才可用結果，不能直接記成果。無真依賴也無成果就是 STAGNATION；需正式發射則 BLOCKED_BY_AUTHORITY。
樹走完→FINISHED(TREE_EXHAUSTED)；只剩需主人／lead 正式親手動作→STOPPED(BLOCKED_BY_AUTHORITY)，皆為有證據的合法終點，先經 §5 尾段審計；已 STOP 不為審計暫開閘。終點候選即停止新 action，殘留可控制 jobs 比照 STOP 取消並對帳；未消帳仍列未決、不能冒充首航 PASS，監督器持續收尾至 close_deadline。
深審見收益遞減或樹將耗盡，須產明早新樹草稿並列證據；草稿與批准樹分存，不算成果、不改夜間授權。

## §5 S5 分層及審計官自身閘〔回應 M4；FAIL 1、3、5、8〕

快審仍用小模型 sonnet，每個取得過租約的棒結束（COMMITTED／ABORTED）即到期，5 分硬 deadline；初棒前須有針對 bootstrap 的有效預審，無前棒不等於免審。due_at 由事件／棒期限機械產生，deadline=due_at+5/15分，不因晚發審計請求而順延。
深審仍用 opus 或 astra：每第三個取得租約的棒結束或逾第 55 分時即到期，15 分硬 deadline；尾段未滿三棒亦強制深審。深審與快審可並行讀同一凍結輸入，但應到的全完成前不放行。
審計輸入固定為包 hash、input revision、該棒 intent／事件與原始成果、baseline/claims、jobs／budget／tried、待審清單；深審必讀自上次深審後全部棒的原證據並核快審誤判／漏審／validator 失效，不只讀摘要。深審 activity 專指本次末棒，整段方向另記，避免與快審範圍混淆。
三問各存 `claim,baseline_ref,evidence_refs+hashes,reason,verdict`：Q1 真新知識或舊結論加密；Q2 分類／預算對得上；Q3 死路等價／次數合規。Q2/Q3 的數字由獨立機械重算，S5 判語義，不能覆蓋硬閘。
審計 `status` 唯一枚舉 PENDING/VALID/ERROR/TIMEOUT；VALID 的 `activity` 唯一枚舉 PROGRESS/WAITING/STAGNATION/BLOCKED_BY_AUTHORITY，另有 `hard_checks=PASS|FAIL`、`tree_exhausted`、`direction=IN_SCOPE|DISPUTED`。不再混用 OK／PROGRESS-OK。
`stagnation_count` 只依有效快審更新；深審不重複加次。PROGRESS 歸零，WAITING 暫停但不清零（STAGNATION→WAITING→STAGNATION 仍為二次）；審計錯誤、blocked 不冒充無新知識。

| 到期／結果（由上往下首個命中；硬違約永遠優先） | 唯一轉移／計數 |
|---|---|
| 機械硬閘 FAIL 或審計 hard_checks=FAIL | 立即 STOPPED(HARD_VIOLATION)，不等二棒 |
| crash、格式壞、輸入 hash／revision 錯、三問缺件、語義無法判定或方向爭議 | ERROR→STOPPED(AUDIT_ERROR)，留申覆；不抽到 OK 為止 |
| now≥deadline 尚無全份 VALID（含深審）；無回覆 | TIMEOUT→STOPPED(AUDIT_TIMEOUT)；遲到回覆只留 ledger，不能恢復 |
| 到期未發請求或仍 PENDING，且 now<各自 deadline | AUDIT_PENDING；監督器只補發尚未發出的請求，禁止新科研副作用 |
| 所有應到審計 VALID，tree_exhausted=true | FINISHED；保留 final audit 和未消帳，終帳缺失仍不得驗收 PASS |
| VALID／BLOCKED_BY_AUTHORITY | STOPPED(BLOCKED_BY_AUTHORITY)，計數不變，留需主裁／親手發射單 |
| VALID／PROGRESS 且無上述終態 | count=0；有已驗真依賴待等→WAITING，否則 READY |
| VALID／WAITING 且 §4.5 等待有效 | WAITING，count 不變；不免除等待期限 |
| VALID／WAITING 但等待無效 | STOPPED(WAIT_INVALID)，不算一棒停滯 |
| VALID／STAGNATION | count+1；=1→READY（僅剩餘已裁 action），≥2→STOPPED(STAGNATION) |

上表只對尚未持久 STOP 的戰役決定放行；STOP 中的尾審／遲到審計只更新 ledger 與結果標記，不轉 READY/FINISHED。快／深審活動判斷不一致即 ERROR；tree_exhausted 與 BLOCKED_BY_AUTHORITY 同時為真屬格式矛盾，不能擇寬解。
深審若推翻快審、懷疑過寬或過嚴，關閘留單；夜間沒有二審放行權。主人晨間可命獨立 verifier 復核原證據，append overturn 紀錄，按 §2 resume；runner 不得自我翻案。
`state.audit.ledger_ref` 必須指向批准包指定的 **VERIFIER-LEDGER** 邏輯帳。實址已在：
`luna/.claude/skills/fleet-command/VERIFIER-LEDGER.md`（複審旁證、commit e20c09a 起）——
它是每審一列的總帳；本節要求的逐次審計欄位（三問逐項/input revision/原始回覆）由
戰役目錄的審計帳檔承載、總帳留一列摘要＋ref。
ledger 每筆含 audit/verifier ID、model/prompt/rules/validator 版本與 hash、scope/deadline/input revision/hash、三問逐項輸出、原始回覆、錯誤、深審核查、申覆／二審／推翻及主裁 ref；append-only，缺 ledger 寫入即不放行。

## §5.5 多機資源分配〔回應 M5；FAIL 6〕

各分支仍帶機型／卡數／CPU／時長、資源獨佔與 budget 標籤；睡前主線先排資源時段，執行閘再驗實際輸入就緒及 GPU／CPU／IO／寫入面不衝突。機器空閒不構成發射理由。
**首航停用候補佇列，`candidate_queue_enabled=false` 且候補 action 清單為空。** 夜間三軸自評、latest.pt 或「不等主線」聲稱都不能加入新活；不承諾七八成利用率。
後續若恢復候補須另修訂呈裁：lead 睡前交固定資產 hash、DAG／無上游證明、判決用途、讀寫面與資源／預算，主人裁入 branch/action；此處不是預先准入。目標仍是填滿判決力，無合法有價值工作就空卡。

## §6 state.json 規範性 schema〔回應 M1–M5；FAIL 8〕

位置沿 v1：`data/fleet-runs/<campaign>/state.json`。以下是待實作的型別／必填／一致性契約，可拆檔；本次不建立任何 runtime 檔。所有 ref 具有 `uri,sha256,event_id,generation`，由 state 可達；schema 拒未知欄位／枚舉、漏必填或懸空 ref。
ID/hash 為非空字串（SHA256 固定 64 hex）；revision/epoch/seq/count 為非負整數；時間用 UTC timestamp＋boot/monotonic deadline；額度用整數 resource-seconds/bytes/token，禁止 NaN／負數。只有明列 optional／尚未 receipt 的欄可 null，且須 reason。
條件必填：stop 在 STOPPED 必填、其餘 null；棒 end 在終態必填；audit activity 僅 VALID 必填、其餘 null；receipt/job ID 在 SUBMITTED/RECONCILED 必填；尚未產生的 evidence、未持租約的 owner/expiry 可 null 並附狀態理由。其他不適用欄以已批 NOT_APPLICABLE 物件表明，不能省略。

| 判決八類／必填物件 | 最小欄位、寫入權與不變量 |
|---|---|
| 1 版本／批准 `anchors` | `schema_version,campaign_id,contract/tree/milestones/baseline{version,ref},approval{package_hash,source_ref,approver,approved_at},actions_ref`；主人信任根不可由 runner 寫；所有 action／predicate／write_scope 綁 §3 |
| 2 生命週期 `lifecycle,batons[],milestone_state_ref` | `phase,activity,t0,action_end,close_deadline,stop{reason,event,affected_jobs,resume_authority,evidence},last_heartbeat`；棒含 `id,slot,owner,start,work_deadline,commit_deadline,end,status=RUNNING|COMMITTED|ABORTED,reason`；成果含 `milestone_id,status,evidence,predicate_result,verified_event,invalidation_event`；合法轉移只按 §2、§4.5、§5 |
| 3 寫入權／交接 `ownership,commit` | `owner,epoch,boot_id,lease_expiry,writer_acl,revision,prev_revision,prev_hash,manifest_ref,event_head,progress_ref,commit_status`；只有 coordinator CAS 提交；owner 過期及舊 revision 拒寫；state/progress 同事件錨 |
| 4 jobs／副作用 `intents_ref,jobs_ref` | `intent_id,idempotency_key,action/branch/attempt_id,epoch,status=INTENT|SUBMITTED|RECONCILED|NOT_SUBMITTED|UNKNOWN,receipt,cluster/job/array_task,argv/script/env_hash,F5_refs,input/output_manifest,scheduler_state/exit,completion_anchor,cancel_authority,observed_at`；未決不得當不存在 |
| 5 資源 `resources_ref` | 各 total/branch/lane 的 `cap,settled_used,outstanding_reservation,available,D_upper,N_verified,R_density`；逐 task `classification,approval_ref,U,allocations,actual,source_ref,observed_at,error_bound,accounting_status`；另有 inherited/overhead/storage 帳，不信 runner 自填總數 |
| 6 等待／重試 `dependencies_ref,tried_ref` | §4.5 wait 全欄、`candidate_queue_enabled=false`；experiment/dead_end 等價規則 hash、failure_class、attempts/occupied/retries/limits 及解除占位證據；同依賴不展延、改名不清帳 |
| 7 審計／知識 `audit,claims_ref` | `due[],deadline,scope,input_revision,status,activity,hard_checks,direction,tree_exhausted,verifier_ref,ledger_ref,stagnation_count,exemption_reason,claims{baseline,evidence}`；三問／深審／申覆皆由 ref 可達，VALID 必須完整證據 |
| 8 事件史／例外 `history_ref,pending_for_master[]` | append-only `seq,event_id,prev_hash,time,actor,epoch,kind,before/after_revision,intent/evidence_refs,decision_reason`；保存全部 generation、ABORTED/orphan、對帳與未決事件；待裁單有來源事件，不能覆寫失敗史 |

hash 鏈只供完整性檢查，不取代 ACL／可信批准錨；事件先後序以 durable seq，不靠會跳動的 wall clock。控制 latch／提交帳有獨立持久來源，資料缺失一律拒啟，不把無紀錄當新戰役。
progress 自由體記「為何選這條」，state 記可判的「是什麼」；驗收允許從 state 導航完整證據包，不要求 state 單檔塞下所有理由或重叫 LLM 重現同句話。

## §7 首航驗收與後續義務〔回應 M6；FAIL 7，兼驗 M1–M5〕

首航仍是過夜實驗管理夜：收割既有 job→判讀→只發已明批、無 ckpt 的 smoke；正式下一步留晨，真跑走 sbatch。日期、t0/action_end/close_deadline、具名量測者與批准包均須睡前凍結。

| 五項判準 | 量測者／尺 | 證據／缺失處理 |
|---|---|---|
| 零樹外行動 | 非跑棒者的驗收官逐事件核准入；全窗口實際未授權副作用數=0；主人終裁 | 批准包、predicate 證據、提交／取消／寫入原始帳與 scheduler 雙向對帳；被擋嘗試另列；缺事件或未知 intent→未驗，不能算 0 |
| 加密度 ≤20% | 獨立計帳器重算 §4 每 lane 全前綴及終帳，S5/lead 查分類；各 lane D/T≤.20 且 total/branch 不超額 | 原始 allocation／終帳、分類批准、預留史與硬 U；失敗／尾帳全含；缺可信耗用→未驗，已證超標→FAIL |
| 每棒交接可回放 | 獨立回放器＋驗收官，全部取得租約棒含 killed/aborted，從前版＋事件重建後版、jobs、budget、停因；副作用無重複 | generation／事件鏈／intent／receipt／progress 共同錨；斷鏈→未驗，重複或越權→FAIL；不要求 LLM 文句重現 |
| 審計三問全有證據 | 深審及晨間 lead 核每次應到快審三問，可重算／核原證；列應審、實審、缺審與深審到期數 | VERIFIER-LEDGER、原輸入／輸出與機械核算；空字串或無來源不是證據；timeout 正確關閘仍是缺審，不報此項 PASS |
| lead 十分鐘重建全夜 | 非跑棒 lead 首次打開 state 包起計 ≤600秒；另一驗收官計時並對原事件答案，全題正確完整 | 只用 state 可導航的 progress／凍結證據包，不靠跑棒者口述；回答分支、jobs 終態、新知、耗用/餘額、停因、待裁六題；漏錯／超時→FAIL，材料缺失→未驗 |

最小有效覆蓋提案：實際跨度 ≥3 小時、≥3 個取得租約工作棒、≥2 次成功交接；至少一次有效收割→成果判讀→已允許下一步、一次第三棒到期深審、所有尾段終審。數字待主人隨包裁，不能事後降低湊 PASS。
另須首航前受控演練覆蓋午夜重啟：故障後拒啟、持久 STOP、對帳、主人明批 resume、新 epoch 接續而無重發；正式夜不要求自動復活。演練證據與實夜分列，不能以演練冒充實夜三小時推進。
總結果固定：`PASS`=五項全過＋有效覆蓋＋首航前義務證據齊；`FAIL`=已證契約違反或有效量測不達尺；`SAFE_STOP_INSUFFICIENT_EVIDENCE`=已證安全關閘但欠推進／可靠性／審計覆蓋；`UNVERIFIED`=連安全性也因缺帳無法證明。FAIL 優先，缺證不得包裝安全通過。
安全停不強迫續跑：樹外、需正式親手或樹早完導致覆蓋不足，按上述分類如實報，不能靠拆格、補密度或候補刷滿。主人終裁可接受安全結果，仍不改成整套首航 PASS。
**後續義務（全未在本次實作／驗收）**：產生機器 schema/validator、信任根與 ACL、fencing gateway、原子帳／回放器、獨立 timer/控制 latch、scheduler 硬上限、F5 hooks、S5/VERIFIER-LEDGER adapter，填實全部 action／預算／期限並由主人批准。
首航前故障注入至少逐一覆蓋 §2 十種意外；特別驗：審計 timeout 不放行、壞 state 拒啟、雙棒不重發、sbatch 成功後失聯可對帳、磁碟滿仍持久拒啟、過期 owner 不寫入、預算／分類不能繞過、有效 WAITING 不算停滯。
配對正反驗收另含：缺值／零／多分支、改 script/seed/hash 冒充 smoke、scratch ckpt 禁令、pending↔blocked 假進度、假／過期 wait、D=20/T=100 再發密度拒絕、CPU-only／array／失敗尾帳、第一／二次停滯、深審與尾審缺席、全停空過及 600 秒重建；結果留獨立證據，不在本稿聲稱已過。

## §8 邊界與依賴〔回應 M1、M2、M5；FAIL 4、6〕

- watch daemon 仍是後續基建：systemd 級喚棒與監聽可作實作選擇，但 Restart=always 必須服從持久 STOP／租約／審計閘；本件不安裝、不重啟、不實作 daemon。
- **不碰 Tier A、對外發訊、正式長跑／新 ckpt 發射。** 正式由 lead ルナ親手，夜間留單；TG/Discord/email 等皆禁自動發送，適用 runner、喚棒器、審計器、job hook；晨間由ルナ本人處理。
- Tier A 完整制度版本未取得，首航前須綁適用定義；當前採封閉白名單，daemon 安裝、服務更動、權限更動、刪檔清盤及未列動作都無夜間授權，不臨時猜等級。
- 每 action 綁適用工單的讀寫／儲存邊界、固定輸入 hash、新輸出 exclusive write；不把某卡的 /archive/NFS 規則推成全 repo 通則。失敗留存、新 run ID 仍受次數與預算約束，禁止覆寫、偷搬或換路徑洗帳。
- F6 管棒與棒之間，F5 仍管棒內；S5 不取代 F5、小模型不繼承 lead 正式發射身分。候補停用；閒置可以，樹外動作不可以。

## §9 自我懷疑（非空）〔回應 M1–M6；FAIL 1–8 的證據界線〕

- 這份契約以保守停態換取唯一出口，可能使首航很早停；若沒有無 ckpt 且有判決力的已批下一步，應報覆蓋不足，不能為證明長跑削弱發射邊界。
- 每型 GPU／CPU 各守 20% 及只用已對帳 N 信用較 v1 嚴，可能堵住合理先驗證後推進；此為呈裁選擇，不冒稱主人已接受，改法須重裁並保留不能借未來耗用的保證。
- hash／租約文字不等於部署隔離；外部提交和本地 commit 不能假裝單一原子交易。唯一 token 對帳、控制儲存失效與 scheduler 終止上界必須有故障證據，否則首航 NO-GO。
- 58 分限制、站點取消延遲、Tier A 全文與既有 VERIFIER-LEDGER 實址未獨立驗證；本稿只固定缺件拒啟的行為，未聲稱遠端現況、制度接線或任何注入已通過。

STATUS: DONE
