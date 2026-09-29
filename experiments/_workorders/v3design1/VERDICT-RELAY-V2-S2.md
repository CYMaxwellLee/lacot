# S2 複審判決：接力棒長跑 F6 v2

VERDICT: PASS-with-notes（呈裁前必改 2 項 N1、N2，均為文字級；補完由 lead 按 §4 所列要素自核，不需第三輪 S2）

檢察官：Claude opus（與 v2 修訂者 GPT astra 不同家族）。日期：2026-09-28。本輪為複審：只驗 M1–M6 是否落成可裁決契約、第一輪六個具名反例是否擋得住、修訂是否引入新洞；不重打架構。判官不動手：不改受檢稿、不實作、不發射、不對外發訊。

**結論：** v2 把第一輪六項必改落成了可裁決的契約（M1 差一塊），六個具名反例都擋得住，4D≤N 前綴閘在它自己寫明的前提下數學成立。§2 phase 表與 §5 審計表的交界沒有「一條放行、一條停」的矛盾格；剩下的缺格與缺碼都往 STOPPED 收斂，不會放行未授權副作用。兩項必改：**N1** 夜間產物與既有 job 產物「作為輸入」的綁定規則沒寫，而這正是首航第一步（L168 收割既有 job）；嚴讀則首航做不到，寬讀則輸入完整性沒有保證（第一輪 §6 的 latest.pt 類危害在樹內重開）。**N2**「≥3 棒」能否達到取決於沒凍結的語義，同一晚同一棵樹可以是 2 棒也可以是 3 棒。

## 受檢版本與證據界線

- **V2**：RELAY-LONGRUN-v2.md，201 行，SHA256 `964f5414627445746835b3007d1a47670013efad3199e5c81b146c53bad0fc08`；本報告 Lx 均指此版。
- **V1**：RELAY-LONGRUN.md，SHA256 `c8db09e69a31b18f4503e4eaeed215f9938ad1c940dd864bdfbf7507ad95f8ae`，與 V2:L7 所引一致。
- **R1**：VERDICT-RELAY-S2.md（第一輪判決），SHA256 `cdb62768380dcb7f8737c3ffb0d8a5cae149d206fd6aec58d978fd78d4e24aef`。
- **旁證（唯讀、有界）**：`../ucontrast1/LEAD-CLARIFICATION.md` §3–4；`~/Projects/elsa-agent-workspaces/luna/.claude/skills/fleet-command/VERIFIER-LEDGER.md`（commit e20c09a 13:03 開帳、c206d6e 14:25 補第 7 筆）；`~/Projects/elsa-agent-workspaces/luna/RULES-DESTRUCTIVE-OPS.md`。V2:L9–10 引的八份先例檔都存在，引到的節名（LEAD-CLARIFICATION §4、README「Gate 順序」、LAUNCH-CARD「F5 鏈位置」、HARVEST-REPORT §1–4、EXP1 預檢 C1–C3、opt1 預檢 §9）也都在；只核存在與標題，未重審內容。
- **沒做的**：未跑任何 code、未連 Slurm；平台 58 分上限、kill grace、sacct 延遲、憑證隔離均 UNVERIFIED。下文可行夜與反例都是設計推演，不是故障注入結果。

## 1. M1–M6 逐項

| 必改 | 判 | v2 證據 | 缺什麼／附註 |
|---|---|---|---|
| M1 凍結授權 | **PARTIAL** | 批准包與信任根 L62–64；六類模板 L66–73（kind 枚舉不含正式發射，L69）；exclusive write、NOT_APPLICABLE 須睡前批 L75；TRUE/FALSE/UNKNOWN/INVALID，零／多命中皆停 L76；終端分支不帶可執行命令 L77；validator 重算、runner 不能改 L78；smoke 逐 action 明批 L79；scratch ckpt 首航禁 L80；F5 接點 L81–82；候補停用 L143 | 缺「夜間產物／既有 job 產物作為輸入」的綁定規則（N1）。其餘 M1 要求均已落字 |
| M2 停棒／恢復／交接 | RESOLVED | 90 分時段取代 1–2h L15；獨立 timer L21；租約、序列化閘、憑證不能隔離即 NO-GO L22–23；啟動閘 L24；generation＋CAS 原子交接 L25–26；intent／idempotency／對帳 L27–28；持久 STOP、控制 latch、主人 resume L29–32；phase 表 L34–43；十種意外互斥唯一出口 L45–58 | 附註：初始 STOPPED 的再 bootstrap（J6）、coordinator 自身互斥（O8） |
| M3 可信計帳／有界重試 | RESOLVED | 前綴＋終帳 L87；分 lane L88；T/D/N 與分類凍結 L89；失敗／requeue／T=0 L90；既有 jobs 不入帳 L91；overhead 另帳 L92；預留 U L93–94；密度閘 L95–96；執行中硬限、120 秒延遲、污染處置 L97–98；死路等價 L99；「≤2」語義與首航自動重試 0 L100–101 | 數學見 §3c；兩處精度（O4） |
| M4 S5 自身閘 | RESOLVED（附註） | 到期／硬 deadline／初棒預審／尾段深審 L115–116；固定輸入 L117；三問逐項 L118；唯一枚舉 L119；停滯計數 L120；首個命中轉移表 L122–133；STOP 中審計只記帳、夜間不翻案 L135–136；ledger L137–138 | bootstrap 預審無對應列（J1）；方向爭議與審計器故障同碼（J5）；VERIFIER-LEDGER 已存在，L137「實址未取得」過時（O6） |
| M5 真推進／候補 | RESOLVED | PROGRESS 只認成果首次成立或分支排除 L105；成果三值、不能倒回 L106；合法等待必填與不展延 L107–109；兩種合法終點 L110；新樹草稿分存 L111；首航停用候補 L143–144 | PROGRESS 宜綁機械事件（O2） |
| M6 驗收與最小覆蓋 | RESOLVED（附註） | 量測者睡前凍結 L168；五項尺與缺失處理 L170–176；最小有效覆蓋 L178；午夜重啟演練分列 L179；四種總結果 L180–181；R1 指定的六項首航前驗收全在 L183（另加 L184 配對正反案例） | 「≥3 棒」的可達性取決於未凍結語義（N2） |

第一輪 FAIL 1–8 經 V2:L8 索引對到 M1–M6，上表即涵蓋。

## 2. 第一輪具名反例重放

| 反例 | 擋住它的 v2 條文 | 判 |
|---|---|---|
| 死路改名洗次數 | ID 永久穩定、別名不改等價與累計 L64；action 睡前綁 dead_end_id L69；等價集合含 seed／path 別名 L99；未知等價停待主裁、不能改類清零 L101；改名不清帳 L159 | 擋住。更根本的是 runner 不能新增 action（L78），改了名也發不出去。L184 沒把「改名／換 seed 不清帳」列為配對驗收案例，宜補（optional） |
| milestones 來回切刷進度 | 成果只凍結、不准拆小格 L72；runner 不能手填 done L78；pending↔blocked、改 evidence 字串、拆 M1a/M1b 不算 L105；狀態僅 PENDING/VERIFIED/INVALID 且不能倒回 L106 | 擋住，且已入 L184 驗收。殘留：PROGRESS 由 S5 下判，未明文要求引用機械事件（O2） |
| 「等 job X」萬能藉口 | 等待必填身分、期限、最近觀測 L107；期限固定不展延、查不到或已終態即停 L108；60 秒觀測、COMPLETED 先過產物契約 L109；等待無效即停 L132 | 擋住，且已入 L184。殘留：L108 的排隊 deadline 只寫「取已裁 action 值」，既有 job 沒有 action（併入 N1） |
| D=20、T=100 再發 10 | 每次預留後 D_upper＋R_density ≤ N_verified/4，L95 | 擋住：20＋10 ≤ 80/4＝20 不成立，此時 R 上限為 0；已入 L184。見 §3c |
| 兩棒重疊雙花 | 租約／epoch／boot_id L22；序列化閘、過期 owner 不能發射 L23；expected-revision CAS L25；換棒不換 idempotency key L27；第二棒取不到租約即 ABORTED L55；總額與密度同交易 L96 | 契約層擋住，已入 L183。前提是 L23 的憑證隔離，做不到即 NO-GO，v2 自己寫明了 |
| 58 分 vs 1–2h | 棒＝90 分時段、工作 ≤50 分、最遲 55 分 commit L15；第 50／55 分轉移 L40 | 解消。58 分本身仍未驗（L15、L199 已自承） |

## 3. 修訂新引入的洞

### 3a 內部一致性與複雜度

逐格對過 phase 表（L34–43）、十種意外（L47–58）、審計表（L122–133）與 §4.5 終點（L110）。**沒有矛盾格**：同一情境下兩條規則要嘛都導向 STOPPED，要嘛一條導向 STOPPED、另一條沒寫。沒寫的地方如下；最壞代價是多耗一個 90 分時段或標籤錯，不會放行越權：

- **J1 bootstrap 預審無轉移列。** L115 要求初棒前有效預審，但 L119、L150 規定 VALID 必填 activity，bootstrap 沒有棒可判；§5 表各列以 activity 為鍵。L36「全閘過才 READY」接得住，只是 schema 會卡在必填欄。
- **J2 停因碼與枚舉不封閉。** L76（零／多命中、UNKNOWN、INVALID）、L82（F5 缺件）、L98（CONTAMINATED）、L106（VERIFIED 失效）只寫 STOPPED、無 reason 碼；L56 叫 WAIT_INVALID_OR_EXPIRED、L132 叫 WAIT_INVALID，同一事兩名；phase 與 stop reason 沒有一處列全，而 L148 要求 schema 拒未知枚舉。L45「保留全部原因」與 L155 單一 `reason` 欄也對不上。
- **J3 零分支命中的路由不明。** L76 說直接 STOPPED；若實作照 L38 正常交接進 §5，S5 可能判 STAGNATION，要兩棒後才停（L133）。
- **J4 終點是機械事實，卻放在審計欄。** 終點由命中分支的 `terminal_kind` 決定（L77），`tree_exhausted` 與 BLOCKED_BY_AUTHORITY 卻是 S5 輸出（L119）；兩者分歧時如何處置沒寫，最壞是在無可執行 action 下空轉兩棒。
- **J5 方向爭議被歸為審計器故障。** 「語義無法判定或方向爭議」與 crash、格式壞同歸 AUDIT_ERROR（L125）。停是對的；但 L175 的驗收會把「審計官盡責提出爭議」記成審計缺失，晨間 lead 看到 AUDIT_ERROR 也會先懷疑審計器壞了。
- **J6 初始 STOPPED 能否重來。** L36 允許未過閘的新戰役補件後由批准轉 READY；L32 卻規定 bootstrap 只准一次、L43 規定 STOPPED 只有主人 resume 才出。睡前閘沒過（零副作用）時，是主人重新批准同 ID 再 bootstrap，還是必須走整套 resume（快照、回放、對帳），兩讀皆通。

**複雜度本身。** V2 幾乎逐句對齊 R1 用語；VERIFIER-LEDGER 第 7 筆記 R1 檢察官為 gpt-6-astra，與 v2 修訂者同家族，文字對齊不等於實質到位，所以本審改用 §3b 的構造與 §3c 的證明驗實質。實質站得住。代價有兩個，屬主人裁決時該看到的成本，不是契約錯誤：一是 L182 列的八項基建（外加填實與批准）全是首航前置件；二是每夜批准包要主人逐 action 看 hash、預算與寫入面。建議呈裁時附一頁摘要（夜裡准做什麼、什麼會停、首航證明什麼、要先蓋哪些基建），否則 201 行契約難以被真正審過，只會被蓋章。

### 3b 會不會保守到湊不出三棒

**結論：合法三棒夜存在；但最可能的真實夜型是 2 棒還是 3 棒，取決於 v2 沒凍結的語義。**

骨架：一棒的結束方式只有 commit→審計→{WAITING、READY、STOPPED、FINISHED}。首航無候補、無重試、正式留晨，棒數主要由核准鏈上的依序等待決定，另受三個未定語義影響：每棒准執行幾個 action（L77 只說「取首個」）、收割是獨立 HARVEST action 還是併在分支條件 validator 裡（L76、L81）、棒內可否輪詢等待（L56 傾向先 commit 再 WAITING，但未禁止棒內等）。時段模式（固定 90 分格或 READY 即起）則決定跨度；L15、L37、L73 傾向固定格，但未明寫。

**可行夜 N\*（構造）。** 睡前：lead 已親手發正式 job F，t0 時 F 仍在跑。批准包：兩個互補非終端分支（條件＝凍結 validator 判 F 的產物），各帶一個明批 smoke s（精確 argv／hash、1 seed、wall 15 分、只寫 metrics.json 到新目錄、無 ckpt、F5 ⓪①② 證據與 ③ hook）及 s 的收割；終端分支 BLOCKED_BY_AUTHORITY（正式下一步）；零 DENSITY action；候補關。
夜裡（固定格）：

1. t0 棒1：F 在跑，提 WAITING（F 在 DAG、唯一可找、RUNNING、觀測 ≤60 秒），commit；快審 VALID/WAITING，進 WAITING。
2. 約 t0+100：F 完成，轉 READY。
3. t0+180 棒2：收割 F，分支恰一個命中，M1 VERIFIED（PROGRESS），發 s，commit；快審後因有已驗依賴進 WAITING。
4. 約 t0+200：s 完成，轉 READY。
5. t0+270 棒3：收割 s，命中終端，commit；快審加第三棒深審（兼尾審），進 STOPPED(BLOCKED_BY_AUTHORITY)。

合計：3 個持租約棒、2 次成功交接、一次「收割→判讀→准許的下一步」、第三棒深審與尾審齊全，跨度約 4 小時 50 分，每一步都只用 v2 已准的動作。前提有三：N1 以「產出後綁定」讀法成立；棒要等 job 時先 commit；F 在棒1 commit 之後才完成。若時段是 READY 即起，F 在 t0 後須再跑約 160 分以上，跨度才 ≥3 小時。N\* 對「每棒幾個 action」不敏感，因為每棒的工作都被一次等待切開。

**翻轉反例 C（最可能的真實夜型）。** F 在 t0 前已完成，樹上只有一個 smoke，接著是正式下一步：

- 一棒可執行多個 action：棒1 收割 F、判讀、發 s，然後 WAITING；棒2 收割 s，命中終端。只有 2 棒、1 次交接，依構造達不到覆蓋，整夜最好也只能拿到 SAFE_STOP_INSUFFICIENT_EVIDENCE。
- 一棒只執行一個 action，收割為獨立 HARVEST，時段為固定格：收割 F、發 s、收割 s 各佔一棒，得 3 棒，跨度約 3 小時 20 分，可達標。
- 同上但時段為 READY 即起：3 棒擠在約 50 分內，跨度不到 3 小時。

同一晚、同一棵樹，覆蓋判決由實作選擇決定，等於開了一條「挑語義湊 PASS」的路。這是 L178「不能事後降低湊 PASS」要防的事換了一個形狀出現，所以列為 N2。

**機率面**（不是契約錯）：自動重試為 0，每棒又掛著約六個硬計時器（5／15 分審計、60 秒租約、120 秒 sacct、55 分 commit、60 秒觀測），任一抖動即安全停；v2 已接受此取捨（L196）。棒越多暴露越大，「一棒一 action」雖幫覆蓋，也放大誤停率。各計時器實際餘裕未量，建議在 L179 演練順手量。

### 3c 4D≤N 前綴不等式（L95–96）

**判：數學正確，是充分條件（保守、非必要）。**

- 等價：T>0 時 D/T ≤ 0.2 ⇔ D ≤ 0.2(D+N) ⇔ 4D ≤ N；T=0 時 D=N=0，兩邊皆成立（L90）。
- 證明概要：設 t_r 是時刻 t 之前最後一次預留。t_r 之後沒有未預留的密度耗用（L93），所以 D(t) ≤ [已結密度實耗]＋Σ[未結密度 task 整生命耗用] ≤ D_upper(t_r)＋R_density(t_r) ≤ N_verified(t_r)/4 ≤ N(t_r)/4 ≤ N(t)/4。最後兩步用到：N_verified 是「已發生」非密度耗用的下界，且 N 隨時間不減。首次密度預留前 D=0。逐 lane 各自成立。
- 前提（v2 都有寫，只是散在各處）：A1 每個密度 task 在每條 lane 的整生命耗用 ≤ 其 U，靠硬 wall、kill grace 與禁 requeue（L93–94、L97、L101）；A2 N_verified 為下界（L95）；A3 D_upper 為上界（L95）；A4 無預留即無耗用（L93）；A5 分類凍結不回溯（L89）。A1 破掉（如意外 requeue）時保證不成立，v2 正確地記為契約違約而非 PASS（L52、L101）。
- 兩處精度（O4，不影響上述真值保證）：(i) 結帳時的計量上界若因誤差超過 U，帳面上的不變式會在兩次預留之間自己破掉，連帶讓下一次非密度預留被 BUDGET_GATE 擋下；應以 min(U, 計量上界) 入帳，或把超過 U 視為矛盾停下。(ii) N_verified 單調不減要明寫（下修即 STOPPED），L97 的「矛盾」只算間接涵蓋。另外，§7 獨立計帳器（L173）須用與閘同一套取整慣例，否則會在誤差範圍內報出假違約。

## 4. 呈裁前必改與 optional

**分寸句：** 只報影響可行性或安全的為必改。照 v2 原文執行不會放行未授權副作用；N1 保護首航做得成與輸入完整性，N2 保護驗收有意義，兩項都是文字級，補完由 lead 按下列要素自核即可，不需第三輪 S2。本判決不要求先造 daemon、不下放正式發射權、不以行數或文風阻擋。v2 樣樣往 STOPPED 收的保守，是主人可以接受的取捨；主人只需看清它的代價：首航要挑對夜，而且可能依構造只拿到安全停。

**N1｜夜間產物與既有 job 產物作為輸入的綁定規則（M1 的 PARTIAL）。**
問題：L64 要求 `<…>` 睡前填實值並過 schema；L70 的 input_manifest 要 {paths, hashes}；L148 規定 schema 拒懸空 ref、所有 ref 帶 sha256。可是首航第一步就是收割既有 job（L168）；既有 job 只讀觀察（L91），模板沒有它產物契約的槽位（L73 的 known_jobs 只有欄名）。smoke 若吃收割出的產物，L79「資料逐一匹配」也無從在睡前凍結 hash。
兩種讀法：嚴讀，批准包過不了 schema，首航做不到；寬讀，既有 job 的產物沒有任何產物契約可驗（它沒有 output_manifest），本戰役 smoke 的產物也缺「新鮮度」與「綁定不可變」兩條，第一輪 §6 的 latest.pt 類危害（讀到被換掉或舊的檔）在樹內重開。
先例可直接移植：ucontrast1 LEAD-CLARIFICATION §3 已把「凍結輸入資產＝SHA256＋逐字路徑」與「新產物＝起跑後產生＋編號最大」分成兩條規則，v2 只寫了第一條。
自核要素：

1. 輸入分兩類：FROZEN（睡前 hash 已知，沿 L70）與 BOUND（夜間產生）。BOUND 必須列在 dependency_ids；綁定前在 schema 中是「待綁定、附理由」（接 L150），不算懸空 ref。
2. 睡前凍結 producer 與產物契約：本戰役 action 用其 output_manifest；既有 job 在 known_jobs 補模板：job／array 身分、預期路徑、schema_hash、count、completion_anchor、result_validator_hash、queue／completion deadline、cancel_scope=NONE。
3. 綁定時點與執行者：producer 終態後，由 coordinator 或凍結收割器驗完成錨、schema 及「產生於 producer 提交之後」，把實得 sha256 寫成事件；**棒 agent 不能寫綁定**；綁定後不可變。
4. 下游一律對綁定 hash：L76 的 provenance、L79 的資料匹配、L172 的零樹外核對，遇 BOUND 輸入皆以綁定事件的 hash 為準；綁不成（缺檔、多檔、hash 漂移）走 #3 INVALID_ARTIFACT。
5. 等既有 job 時，L108 的排隊與完成 deadline 取 known_jobs 的睡前值。

**N2｜凍結決定棒數的語義（M6 附註升為必改）。**
問題：見 §3b 反例 C。「≥3 棒」能否達到取決於三個未凍結語義，時段模式另決定跨度；同一晚可以是 2 棒也可以是 3 棒。
自核要素：

1. night 模板（L73）補 `actions_per_baton` 與 `slot_mode`（固定格或 READY 即起）兩欄。
2. 明寫：需要已發 job 的結果時，棒先 commit 並經 S5 進 WAITING，不在棒內輪詢。
3. 定義 L178 的「工作棒」：持租約且 commit 的棒；activity=STAGNATION 的不計入門檻。WAITING 棒是否計入由主人裁；本審傾向計入，因為它實際走過租約、commit、審計與交接。
4. 批准包附覆蓋可行性算式：依核准鏈與上述語義列出每棒執行哪些 action（含收割是否為獨立 HARVEST）、預期棒數與跨度；算不到門檻就不選那晚首航。

**Optional（實作前宜補，不擋呈裁）**

- **O1 封閉枚舉與停因碼（J2、J3）。** phase 與 stop reason 在一處列全；L76、L82、L98、L106 補 reason 碼；兩個 WAIT 名稱合一；stop 改 reasons[]，對齊 L45。零命中明定當場 STOPPED、不經 STAGNATION。L184 補「改名／換 seed 不清帳」案例。
- **O2 審計表交界（J1、J4、J5）。** bootstrap 預審 activity=NOT_APPLICABLE，補「VALID→READY」一列；`tree_exhausted` 與 BLOCKED_BY_AUTHORITY 由命中分支的 terminal_kind 機械導出；activity=PROGRESS 須引用本棒的 milestone VERIFIED 或分支排除事件，S5 只確認，不一致即 ERROR。DIRECTION_DISPUTED、SEMANTIC_UNDECIDABLE 與 AUDIT_ERROR 分碼，停的行為不變，並在 L175 註明「有效爭議不算缺審」。
- **O3 J6。** 未曾進 READY、零副作用的戰役，閘失敗後可否由主人重新批准、沿用同 ID 再 bootstrap，二擇一寫明。
- **O4 §3c 兩處精度。** 以 min(U, 計量上界) 入帳；N_verified 單調不減；計帳器與閘同一套取整。
- **O5 首航 profile。** 批准包凍結零 DENSITY action（D 恆為 0，密度閘退化為靜態核對）、單一 GPU 型號 lane。不削弱任何保證，只縮小首航前要蓋的基建面。
- **O6 更正「未取得」。** VERIFIER-LEDGER 已存在（見證據界線），L137「實址未取得」過時；但它是每次審查一列的錯判率總帳，沒有 L138 要的逐次審計欄位。建議逐次審計寫入戰役內 append-only 審計帳（L138 欄位），每夜結一列到 VERIFIER-LEDGER。Tier A：本機 RULES-DESTRUCTIVE-OPS.md 是 Rei 版（列 LanceDB、transcripts、backups 等），沒有 Slurm、ckpt、/archive；L190 的封閉白名單比它嚴，引它為上位定義即可，不必等全文。
- **O7 〔拍〕逐數字標記。** v1 的 20% 標了〔拍〕，v2 改成 L6 一句總括。90／50／55 分、5／15 分、15／60／120 秒、≥3h／3／2、600 秒逐一標明「拍」或「由 58 分推得」，主人才分得出哪些是量出來的。
- **O8 隔離與互斥前置。** ucontrast1 LEAD-CLARIFICATION §4 已記 codex 沙盒結構性擋 sbatch：棒放在網路受限的沙盒內、只由沙盒外 coordinator 提交，就是 L23 的現成隔離路徑。棒若是以同一 Unix 使用者直跑 Bash 的 Claude agent（v1 §7 的設定），須另證隔離；信任根「agent 不可寫」同理，同一使用者下需主人側簽章或另開帳號。L183 故障注入宜加「coordinator 雙開或凍住後復甦」一條。

## 自我懷疑（非空）

- N1 列必改可能偏嚴。把 L150（尚未產生的 evidence 可 null 附理由）、L76（決策點先驗 provenance）、L81（④ 凍結收割器驗 schema／完成錨）拼起來，細心的實作者重建得出綁定規則。我仍列必改，因為它在首航第一步，而寬讀那一側牽動輸入完整性；若主人認為拼讀已足，降為「實作前必補」也說得通。
- 反例 C 的「最可能的真實夜型」是依 L168 的首航定義推的，沒看實際的樹。若實際的樹本來就有兩段依序等待，N2 那一晚不會咬人，但語義未凍結的問題仍在。
- N\* 的存在性只證「契約允許」，不證「這台機器做得到」。58 分上限、sacct 延遲、kill grace、租約續約、沙盒憑證隔離皆 UNVERIFIED。
- §3c 證明是非形式的，依賴 A1–A5。若排程器實際做不到硬 U（例如 CG 狀態卡住的 GPU 繼續累積配置秒），保證不成立；v2 對此的處理是不准發（L94），方向正確。
- 我與 v1 作者同屬 Claude 家族，對 v1 框架可能偏寬；對 GPT 修訂者的密集文風可能偏嚴。本審以構造與證明而非用語比對抵銷，但偏差不能自證為零。
- 40 分預算內未逐格核對 §6 schema 表（L152–161）與 §3 模板（L66–73）的全部欄名交叉引用，只抽查了與 M1–M6 及首航路徑相關的格。其餘若有欄名不一致，屬實作 schema 時的工作，本判決不宣稱沒有。
- 未讀 C25-DESTRUCTIVE-OPS-PROTOCOL.md（elsa-system repo 不在可碰範圍），Tier A 上位定義內容 UNVERIFIED。

唯讀核驗：V2 審前 SHA256 為 `964f5414627445746835b3007d1a47670013efad3199e5c81b146c53bad0fc08`，交付前重核；本次唯一新增檔為本判決。

STATUS: DONE
