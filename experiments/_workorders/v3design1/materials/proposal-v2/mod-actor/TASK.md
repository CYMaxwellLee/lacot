# M2：三入口及主線唯一 owner

u 編碼想像軌跡。L4，所有 production hunk 如 ARCH ownership，尤其 scratch:1720–1734/1779 必改；兩 actor 同步。同族 loss 僅一 compose，不能留第三份 inline 漂移。

從 solution.py 實作 compose，再整合 production training module 與三 caller。Adapter.features(clean) 不等於 density target；scratch 保留 _et_nf normalization、_u_head/_ca、_REAL_W/_wmse、_q/FSQ_SPACE=z、flow_cond(cond,anc)。不可誤把 z nll target 拿去 refine，不可把資料 act 洩漏進 quality。total 在主線只加一次，BC/div/GRPO 外層依舊。

M4 builder 尚未完成也可用本地 services fake 寫完三入口、AMP skip、EMA state/checkpoint/migration 與 R0。M2 唯一能改 scratch；M4 不進該檔。既有 GeoEnergy 太晚初始化，提前以共用 builder 建訓練服務；s_embed/decoder 都 freeze；保留 eval coordinate semantics。

新增 optional training services（兩 actor 舊 positional signature 保持），R0 不需服務；R>0 缺 oracle 明確拒絕，不回退舊 F1。R2 原 tests_repair/test_model_zero_rounds.py 必過。用 spy trace 證明 inline/state/image 三個 production 入口真的呼叫同一 compose；新增三種 feature/clean adapter 的 differential tests，fake compose PASS 不算此項。

EMA 可沿用 CONS=ema forward formula，但 opt2 scaler skip 時不得 update；包含 teacher parameters/buffers/resume。這項要用可控 skip fixture 驗，不必 GPU。新增 metadata，不把缺 teacher 的舊 ckpt 假裝無縫續訓。

本地兩命令同 M1：tests.py --self-check，tests.py --candidate solution.py。回交三入口 trace、API/state_dict/migration diff、所有 hunk/測試、R2 output。合體時和 M3 更新 AST namespace 支援真新 adapter；不得把 reference 假 PASS 當 production。
