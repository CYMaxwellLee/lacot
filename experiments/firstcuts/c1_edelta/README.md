**在證明什麼、判準是什麼**：證明「bcodec 的碼帶有可執行的效果資訊、且 ŝ 頭讀得出來」——
判準（預註冊）：碼間差分預測誤差比 E_Δ < .5〔拍〕，且正確 ŝ-碼對應優於置換對照
（整題配對 bootstrap 95% CI 支持收益〔拍〕）；前置判別力關卡：碼間真實後果差異 ≥
教材同子空間四步變化 RMS 的 10%〔拍〕，不過關記「此視窗不可辨識」而非陰性。

# C1 E_Δ 差分審判

## 第 0 步 checkpoint 盤點

`bcodec_gk_train.py` 的命名為 `G{G}K{K}_lam{lam_s:.1f}`（小數點換成 `p`），存於 `out_dir/ckpt/`。本 repo 實際找到：

| 字典 | 本機 checkpoint |
| --- | --- |
| G16K16 λ1 | `experiments/bcodec_v0/ckpt/G16K16_lam1p0.pt` |
| G32K32 λ1 | `experiments/bcodec_v0/ckpt/G32K32_lam1p0.pt` |
| K32 λ1 | `experiments/bcodec_v0/ckpt/K32_lam1p0.pt` |

`experiments/gk_scan/ckpt/` 有 G×K 掃描模型，但不是本次 λ1 bcodec 受測物。三顆指定 checkpoint 均已在本機找到；若在別台缺檔，需从 jasmine:/archive 取，並用 `--ckpt` 指定實際路徑，程式不推測遠端路徑。

## 執行

從 repo 根目錄：

```bash
.venv/bin/python experiments/firstcuts/c1_edelta/test_c1_wiring.py
.venv/bin/python experiments/firstcuts/c1_edelta/run_c1.py \
  --ckpt experiments/bcodec_v0/ckpt/G16K16_lam1p0.pt \
  --n-starts 4 --out experiments/firstcuts/c1_edelta/results/c1_G16K16_smoke.json
```

完整規模於檢察過關後才跑：每顆字典 `--n-starts 200`，輸出各自的 `results/c1_{dict}.json`。官方 val 資料預設在 `$OGBENCH_DATA_DIR` 或 `/home/cymaxwelllee/.ogbench/data`，可用 `--data-dir` 指定。執行只用 CPU。

## 協定與欄位

- 一半 clean：官方 val episode 起點；一半 drift：從該 episode 起點用原碼執行一段後，在下一段的模擬狀態起跑。`coverage` 分別記 attempted、accepted、donor_shortage。drift 預備段額外花四步，末端 `n_steps` 只計四碼反事實測試的步數。
- `env.reset()`、`set_state(qpos,qvel)`、用 `sim_obs` 做 decoder 條件、`obs_mu/obs_sd` 正規化、â clipping 與四次 `env.step` 沿用 `bcodec_gk_drift_eval.run_reset_chunks`。每個起點保存 MuJoCo integration state（含 warmstart）、wrapper 步數、環境與 action-space RNG，以及 goal 欄位；每碼還原同一快照。C1 不設路標 leg，沒有 active-leg bookkeeping。
- donor 從官方 val 的**另一 episode**完整段取一整個碼 tuple。依 checkpoint 的正規化 obs 姿態/速度維度 `[2:29]` 距離排序，排除相同碼，並要求與原段四步動作的 L2 距離至少 0.5。缺三個時記 coverage，不偷補逐 group 拼接碼。
- `starts[].cells[]` 每格存 `start_id` 所屬起點、`code_id`、完整碼、預測正規化/原尺度 ŝ 四步流、實際 obs 四步流、clipped â 四步流、各子空間絕對平方誤差。`starts[].stats` 存六個碼對的真實後果差分平方、預測差分誤差平方、置換誤差平方。置換只換 ŝ 與碼的對應，預測集合保持不變。
- `xy` 使用 obs 前兩維，單位公尺；`yaw` 從 MuJoCo wxyz quaternion 求角並 wrap，單位弧度；`gait` 去掉平移與 heading，保留身高、roll/pitch、關節姿態、身體方向的平移速度及其餘 qvel；姿態/速度按 checkpoint 的 obs_sd 定尺度。ŝ 先從模型正規化空間還原再投影。
- **子空間可比性警告**：C1 的 yaw/gait 採 SE(2)-quotiented 定義：yaw 是從 quaternion 導出的 1 維 wrap 角；gait 是 25 維去群座標表示，包含 roll/pitch 歐拉分解、旋轉到機體座標系的 xy 速度及其餘 qvel。`shat_probe` 則直接按原始索引切分（xy=`[0,1]`、yaw=4 維 quaternion、gait=23 維原始索引）。兩者量測不同，per-subspace 數字不能直接放在同一張表比較。
- `E_delta` 分母是同一起點**碼對之間**的真實後果差平方總和。`true_pair_rms` 與教材四步 `teacher_four_step_rms` 決定辨識力。`paired_benefit_ci95` 在整個起點層級配對 bootstrap，收益定義 `permutation_E_delta - E_delta`。summary 內保留預註冊門檻、每子空間判定，JSON 末端有 `n_starts/n_codes/n_steps`。

## 三病分讀

| 結果 | 解讀 |
| --- | --- |
| 差分準、絕對不準 | 碼有可用效果資訊，原漂移讀數受共同偏差污染。 |
| 真實碼效果明顯、差分不準 | ŝ 頭不會預測自身動作的後果；這是頭的問題。 |
| 真實碼效果很小 | 四步尺度或候選庫不可辨識；不得判碼沒有導航資訊。 |

smoke 只驗通路，其統計判定不作科學結論；主表必須等雙檢後再上機。
