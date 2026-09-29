# M4：監督有效性、exposure 橋

u 編碼想像軌跡。L4，只寫新 lacot/refine_quality.py 與測試；**scratch 所有接線交 M2**。本地 oracle fake 是解析雙路世界，不是已完成的真教師。

供應 frozen differentiable decoder/geometry quality service，使用主線原 `_dec/_intent_inv` 語意，不另造另一份解碼。先校準真資料 held-out 再校準 flow samples/own iterates；拒絕常數 decoder（shuffle gap）、假的空牆圖、norm/xy mapping 錯誤。report 資料版本、teacher fingerprint、所有 cost 權重與實測 validity，不得 no_grad 包 quality forward。

新增 same-plan action teacher：用離線有效 transition 配對訓練/驗證 inverse dynamics 或合格 controller；以当前 sampled plan 的 decode 取得 target，不能拿 data batch act。pointmaze 先行，xy 對 ant 的 action 不足，需拒绝而非用差分冒充。主線 _REAL_W 與 teacher 的 validity 分開。head exposure 只學 detached iterates，保存 Q3 的非資料 latent 可讀性橋。

gauge @no_grad：anchor/r0/r1/r3 errors+gap，valid/n_valid/n_total/reason，缺值 null 不是0；獨立 RNG/held-out，不改訓練狀態。候選工單不限於把 fake 的 Quality 複製出來；必交 decoder OOD 校準與 action teacher 資格報告，沒有證據就誠實 FAIL。

起跑用本地 synthetic plans/callbacks，不等 M1/M2；codebase 唯讀調查，production patch 在可寫 checkout 由 lead 指派。tests.py --self-check 只驗 API；tests.py --candidate solution.py 加上上述 production fixtures 才可交件。CPU toy 可以跑，真訓練/GPU 非本輪授權。
