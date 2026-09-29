# M3：真正的科學驗收（不只填 metrics）

u 編碼想像軌跡。L4，拥有新 acceptance tests、本包 integration/與證據；不改 production。起跑有完整本地 contracts/integration，不等 M1/M2/M4。

**solution.assess_run 只是一個小入口；它通過不算完成。** 工作主體是把 scientific harness 對接真 merged 三入口，完成 off-manifold 多 seed 訓練、B1 population optimum、R0/1/3與R6、exposure/teacher校準、matched-budget baselines、六 mutation 在真实整合中的 injection。fake 現成數值不能當答案。raw-latent toy 失敗及 sign-feedback 成功均須保存，不准把最終 PASS 推成 production 已救。

本地 integration/integration_test.py current mode 對唯讀真 inline AST/actors 跑負對照；--reference 明示提案。不 import 整支 trainer（避免真訓練/GPU副作用），不可 monkeypatch 真 branch 來製造成功。合體時配合 M2 的新正式 adapter 綁測試 namespace，且 trace 三真 caller→同 compose；抽不到 branch、import error、例外不得當語意 FAIL/PASS。

identity 在同 off-manifold gate 必 FAIL；F1 factorial/permutation 必獨立於方法。top-level 不規定 EMA 公式；b-consistency/mutation 接線測試另檔。六種：total缺項、接錯u_target、忽略rounds、零noise、truncated BPTT、detach cond，皆有獨立數值/梯度 witness，不能只計 source text。

CPU 真 Flow.sample 訓練每次 fresh，holdout不同RNG，保留兩 mode absolute cost、route swap、constant decode/head；head exposure 不能只有 anchor好看。R6 超訓練深度只報，不硬拗收斂。baseline至少原 sampler/temperature、加樣本、BoN、GD；walltime/calls/train成本分開報。將 noise消耗、round-depth分布與seed列入manifest。

本地 `tests.py --self-check` 與 `--candidate solution.py` 是接口考場，scientific 完工還必須跑本地 integration 全套（含真正 candidate audit）與 R2 原測試。交回 current FAIL／merged PASS stdout+exit、source hashes、真训练 steps、mutation matrix、三入口 trace、baseline ledger。不能以 reference PASS 或 metrics self-check充數。
