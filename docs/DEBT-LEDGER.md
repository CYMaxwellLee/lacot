# 技術債登記簿（裁決過、掛條件觸發的批次債）

_規則：每條債＝裁決出處＋觸發條件＋到期動作。翻新 handoff 續帶時逐條核對；
觸發條件滿足時【必須】提出來裁，⛔ 不准靜默滑過。_

## F2-b：e_target 出口 norm 換不降維版（根治 LayerNorm 奇異目標）

- **裁決**：主人 2026-09-27「照這個組合吧但要記得」＝ a（NLL target 加噪聲）併入
  refine-v2 實作批先上；b（換 norm、如 RMSNorm）掛本條。
- **病**：LayerNorm 每 token 減均值＝目標流形天生少一維；滿維 NLL 對奇異目標有
  無底方向（opus 複驗：無噪聲 NLL 不收斂、rank 6/8）。a 是標準緩解、b 才是根治。
- **觸發條件**：下一次 e_target／encoder 前置【本來就要重訓】的批次
  （任何改 e_target 架構、換 backbone、或 latent 幾何大改版的工程）。
- **到期動作**：該批次的變更單必須含「出口 norm 換不降維版」評估；採用則同批攤掉
  重訓成本。⛔ 不為本條單獨發起全量重訓。
- **關聯**：refine-v2 的 quality 幾何調在「a 加噪後」的 latent 上；若 b 落地、
  幾何再變一次，refine 的尺要重校（記進該批 blast radius）。
