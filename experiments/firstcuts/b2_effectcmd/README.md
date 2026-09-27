**在證明什麼、判準是什麼**：證明「17.5 點開環稅可用『效果命令＋累積誤差回授』型的
字典介面退掉一部分」——判準（預註冊）：B 臂或 C 臂相對 A 臂（原 schedule）救回
≥12 題且弄壞 ≤2 題〔churn 校準沿用 B1〕；B/C 差異用整題配對 bootstrap 95% CI〔拍〕。
形狀擾動下 B/C 能換字維持效果誤差減半〔拍〕＝「意圖與相位分離」的機制證據。

# B2 效果介面三臂

`run_b2.py` 使用 B1 的快照、restore、`findmnt` 守門、教師軌跡及 relay 對齊方式。後果表的每行是從同一個 leg-1 快照解碼一個完整 K32 字並實際執行四步得到的體座標 `(Δxy, Δyaw)` 與去群形狀變化。每題各存 `snapshot.npz`、`effect_table.npz`、`teacher_R.npz`、`A/B/C.npz` 及擾動流。

B 的要求效果只來自預先固定的 `g*_k⁻¹ g*_{k+1}`；C 使用 `g_live⁻¹ g*_{k+1}`。兩者以同一表、同一固定的 SE(2) 距離及各自的即時 decoder obs 執行。yaw 權重由前 40 題教師四步增量的 xy/yaw RMS 比值凍結。smoke 題數少時權重只供接線檢查，不作推論。

擾動附測前 40 題沿用 C2 的起始兩步 action pulse、四步切線位移 50% 校準和 1% 容差。A/B/C 使用相同要求 pulse，逐步存請求與實現。`command_effect_error` 衡量各臂自己所下命令的實現偏差；三臂共通的 `effect_error` 衡量每個 chunk 終點與教師下一姿態的 SE(2) 差。B/C 的共通效果誤差相對 A 擾動臂計比值，≤0.5 且換字才形成機制證據。校準 rollout 步數也計入 `n_steps_total`。

`preregistered.json` 在實驗開始前寫門檻；`summary.json` 含 `n_tasks/n_arms/n_table_entries/n_steps_total`、B/C 對 A 的救回及弄壞數、整題配對 bootstrap CI、選字一致率及擾動比值。只有 200 題才評估 43 題 A 基線、churn 及 40 題擾動門檻；smoke 明示沒有科學判決。

```bash
.venv/bin/python -B experiments/firstcuts/b2_effectcmd/test_b2_wiring.py \
  --synthetic --n-tasks 1 --out-dir experiments/firstcuts/b2_effectcmd/synthetic_wiring_new

# 官方 val 的四題接線檢查（須在 zeldajr 本機資料路徑）
.venv/bin/python -B experiments/firstcuts/b2_effectcmd/test_b2_wiring.py \
  --n-tasks 4 --out-dir experiments/firstcuts/b2_effectcmd/official_wiring

# 200 題入口存在，但本次不得執行
# .venv/bin/python -B experiments/firstcuts/b2_effectcmd/run_b2.py --n-tasks 200 --main
```

`test_b2_wiring.py` 先檢查 SE(2) wrap 與後果表行列 mutant，再跑真 MuJoCo 全通路；它逐字檢查 A 與原 relay action/xy 對齊、表中 0/17/31 字的物理重播、B/C 目標左元素、逐步流與擾動校準。synthetic 不是官方 val 的 43/200 複現。

本次四題 synthetic wiring 輸出（原樣）：

```text
B2 SYNTHETIC PASS tests=3 n_tasks=4 n_arms=3 n_table_entries=128 n_steps_total=6376
B2 MUTANT CAUGHT both_arms=true
B2 A_RELAY_MATCH action_xy_bit_exact=true
B2 MAIN NOT_RUN expected_A_failures=43/200
B2 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true
```

synthetic fixture 的 B 選中原 schedule 字共 7/200；fixture 教師動作由隨機帶生成，這個比例只當接線觀測。教師字效果本身可選時，selector 的一致性另由 `test_02_mutant_caught_before_full_wiring` 驗證。
