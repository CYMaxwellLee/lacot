# v3 offline oracle：第二輪預註冊修訂

資料入口只認 [DATA-MANIFEST.json](DATA-MANIFEST.json) 的 SHA256、檔名、domain 與 observation/action 維度。清單中的兩份檔是**本機副本，未對 upstream 驗證**。`--domain` 用來核對 manifest；`--seed` 只控制模型，`--split-seed` 預設固定為 `1729`，控制 episode split 與抽窗。同一資料的三個模型 seed 必須共用 split seed，並比對輸出中的 `split_hash`。

CPU 接線命令：

```bash
.venv/bin/python -m experiments.refine_v3.train_offline --domain pointmaze --toy --out /tmp/oracle-toy --seed 0 --split-seed 1729
.venv/bin/python -m experiments.refine_v3.train_offline --domain pointmaze --data /path/to/pointmaze-large-stitch-v0.npz --out /tmp/oracle-real --seed 0 --split-seed 1729
```

## lead 裁定的判準修訂（VERDICT-ORACLE-S2 §2、L1/L2）

| 讀數 | 原判準 | 第二輪判準 |
|---|---|---|
| held-out Wφ NMSE | ≤ obs-only 的 0.9 倍 | 維持資格門檻 |
| episode 配對改善 95% CI | 下界 > 0 | 維持資格門檻 |
| 90% PI 覆蓋 | 85–95% 資格門檻 | 健康報告欄，不擋資格 |
| offline data validity | ≥90% 資格門檻 | 健康報告欄，不擋資格 |
| Aω held-out action NMSE | ≤0.1 | 維持資格門檻 |

訓練授權另需實際候選的 validity 與 nominal＋alternative coverage 各 ≥80%。候選讀數只由 `QualityService.measure_generated_candidates` 以模型 report 計算，並記 parent hash／candidate IDs；外部手填的 readings 或 bool 矩陣不能授權。

**已知局限（VERDICT-ORACLE-S2 L3）**：Wφ 在 chunk 內同時讀四個動作，故 `s[t+1]` 可受後三個動作影響；toy 敏感度約為當步的 2.1–5.4%。主讀數以 chunk 末端／段級為準；causal masking 留待實驗 1 過線後另裁。
