# 工單 B1：T/N/L 三臂快照庫＋劑量＋真降精度（b 題首刀）

**在證明什麼、判準是什麼**：把 21.5% 零漂移失敗地板拆到三個病因（資料重放極限／
decoder 回授／字典量化）——判準（預註冊）：N 臂相對 L 臂救回 ≥12 題且弄壞 ≤2 題
〔churn 校準：null 介入能矇中 23/43，此門檻按其上緣定〕⇒ 存在「decoder 條件可移除」
的失敗；α 劑量下十倍小殘差仍同等失敗〔配對 bootstrap 95% CI〔拍〕〕⇒ 才支持飽和
放大器說；真降精度對照一步殘差 ≥ 資料一步殘差的 50%〔拍〕⇒ 精度病才立案。

〔拍〕門檻跑前寫死，回填新舊並排（主人裁③）。口徑固定：ant、原 200 題、200 步、
ρ=1.875、leg-1（reached，非 per_task.completed —— freebie-b 複驗確認過的口徑）。

## 設計（來源：CONVERGENCE 題 b 首刀；opus-b 候選 1 × astra2-b 放大合成）

同一完整模擬快照（每題 leg-1 起點）分三臂，物理狀態各自連續演化：
- **T 臂**：教師原動作開環重放（動作帶固定）。存本機參考軌跡 R（逐步 qpos/qvel/
  obs/動作 全存 —— drift_raw.npz 沒存逐步姿態的教訓，⛔ 別再只存誤差標量）。
- **N 臂**：原字串，但 decoder 每 chunk 吃 R 上同時刻的教師 obs（歸因臂、特權資訊）。
  ⭐ 語義釘死：字串固定時 decoder 仍在讀 obs 算動作 —— L 不是純動作開環，N 把
  「obs 通道」單獨切換到教師世界。
- **L 臂**：現行 schedule（複現基線 43 失敗 —— 對不上先停，⛔ 別在歪基線上疊臂）。

劑量臂：T 與 N 的已裁切動作帶凸組合 α∈{0, 0.1, 1}〔拍〕（α=0 純 N、1 純 T；
⛔ 不另造白噪音）。連續量全程記：xy 位置、軀幹 yaw、去群座標形狀誤差 ——
防二元到達門檻掩蓋劑量效應。

ulp 集成（跑在 L 臂）：每題 16-32 個〔拍〕float32-ulp 級初態擾動（qpos/qvel，
四元數擾動後重新正規化保持合法）→ 每題成功機率分布＋穩定失敗率（＝「不可重放段
存在率」的正確量法，取代單次二元判定）。

真降精度對照（獨立小工單，同批交付）：取 R 中途原生 float64 快照 →
(i) 完整還原重放（自檢，應零誤差）
(ii) 只把 qpos/qvel cast float32 再升回 float64，重放
(iii) 另行改 warmstart 狀態
三者的一步/整段殘差 vs 資料一步殘差（e_data）並列。
背景：p1_diag_replay_fidelity.py:67 的 f64roundtrip **從未做過 (ii)**——同起點
重放零誤差只證明決定性（放大官翻案）。⚠️ venv mujoco 3.12.0 vs lock 3.11.0：
本工單先不動版本；(ii)(iii) 判完若精度/warmstart 無罪、e_data 仍大 ⇒ 版本差成為
主嫌，再另立版本對照工單（裁過的工程附帶）。

## 判讀表（寫進輸出 README）

- N 達救回門檻 ⇒ 存在 decoder 條件可移除的失敗（obs 通道病）。
- N 比 L 差 ⇒ decoder 回授正在救場（回饋有正貢獻的直接證據）。
- 固定參考下 α 把殘差縮十倍仍同等失敗 ⇒ 支持飽和放大器（混沌極限）。
- ulp 穩定失敗率高 ⇒ 地板是分布性質不是單次壞運。
- ⛔ 三個失敗率不可直接相加拆份額（各臂參考不同 —— opus 原案這步被放大官擋下）。

## 資料與資源

- 快照/reset 路徑：復用 walk_verify 的 relay 實作（run_teacher_relay*.py 的
  save_state/restore_state 慣例；lacot_vqo.py:86 有含 warmstart 的版本可抄）。
- 教師動作/obs：ogbench 官方 val（四台 /archive 都有 —— ⛔ 用哪台就取那台的
  /archive，不經 NFS〔主人 9/27 再糾正〕）。
- 規模上限：200 題 × (3 臂 + 3 劑量 + ~24 ulp) × ~188 步 × 5 物理步 ≈ 1.1M 物理步
  —— CPU array job，2080ti 台或 4060ti 台 sbatch，牆鐘 <1 天。
- 完整快照定義：qpos/qvel/act/warmstart/time/RNG/leg bookkeeping —— 只存 qpos/qvel
  不算完整複製（astra2-c 同款警告）。

## 接線自檢（兩向）

- 正向：T 臂對 R 自身重放零誤差（決定性自檢）；L 臂 leg-1 失敗數 = 43。
- 負向：故意在 N 臂餵錯時刻教師 obs（offset+3），行為必須可見地變。
- 完成錨：輸出寫 n_tasks/n_arms/n_ulp/n_steps_total；缺=未完成。

## 交付形狀

- 程式：~/Projects/lacot/experiments/firstcuts/b1_triarm/（run_b1.py、
  run_b1_precision.py、README、test_b1_wiring.py）。
- 輸出：快照庫（.npz 每題）＋ per-題 per-臂逐步流 ＋ summary（2×2 救回/弄壞表、
  劑量曲線、穩定失敗率分布、精度對照殘差表）。
- 流程同 C1：實作 → 混家族雙檢（重點：N 臂 obs 對時、α 凸組合作用在動作帶不是
  參數、ulp 擾動後四元數合法、快照欄位齊全）→ 過了才上機。
