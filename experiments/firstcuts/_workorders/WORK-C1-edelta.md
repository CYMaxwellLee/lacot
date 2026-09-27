# 工單 C1：E_Δ 差分審判（c 題首刀）

**在證明什麼、判準是什麼**：證明「bcodec 的碼帶有可執行的效果資訊、且 ŝ 頭讀得出來」——
判準（預註冊）：碼間差分預測誤差比 E_Δ < .5〔拍〕，且正確 ŝ-碼對應優於置換對照
（整題配對 bootstrap 95% CI 支持收益〔拍〕）；前置判別力關卡：碼間真實後果差異 ≥
教材同子空間四步變化 RMS 的 10%〔拍〕，不過關記「此視窗不可辨識」而非陰性。

〔拍〕= 拍的門檻，跑前寫死；回填時新舊門檻並排報（主人 2026-09-27 裁③）。

## 設計（來源：breakthrough1/CONVERGENCE.md 題 c 首刀；astra2-c 候選二放大）

每顆字典 200 個起點 × 每起點 4 個完整碼：
- 原計畫碼 1 個 ＋ donor 碼 3 個：從姿態/速度相近（bcodec 正規化 obs 空間近鄰）、
  但動作內容有差異的**別的段**借。G×K 字典必須借完整 tuple（一段的全部 group 碼
  一起搬），⛔ 不准逐 group 隨機拼裝。donor 來自別段 ⇒ hindsight 自動排除。
每個（起點, 碼）：
1. 執行前凍結 ŝ 四步預測（主表用最後一步，完整四步流另存）。
2. 從相同模擬器快照 reset，執行經 clipping 的 â 四步，記真實後果 Y_c。
3. ⛔ 禁止用新產生的狀態重新編碼。
4. 按 xy／yaw／gait（去群座標形狀）三個子空間分別算：
   E_Δ = Σ_{c<c'} ‖(Ŷ_c−Ŷ_{c'})−(Y_c−Y_{c'})‖² / Σ_{c<c'} ‖Y_c−Y_{c'}‖²
   性質：任何只依賴起點、忽略碼的預測器 E_Δ=1 ⇒ 不需要另訓 obs-only 對照頭。
5. 置換對照：同一起點內打亂 ŝ-碼對應（保留預測集合、破壞行為對應），重算 E_Δ。
6. 同時報絕對執行預測誤差（跟差分誤差並排，三病分讀見下）。

三病分讀表（寫進輸出 README）：
- 差分準、絕對不準 ⇒ 碼有可用效果資訊，原漂移讀數被共同偏差污染。
- 真實碼效果明顯、差分不準 ⇒ ŝ 頭不會預測自己動作的後果（頭的病，不是碼的病）。
- 真實碼效果很小 ⇒ 四步尺度/候選庫無辨識力，⛔ 不得判「碼沒有導航資訊」。

## 受測物與資料

- 主測字典：G16K16 λ1、G32K32 λ1；輔測：K32（單本）。ckpt 位置候選：
  jasmine:/archive/cymaxwelllee/…（bcodec_gk 掃描產物）、zeldajr ~/Projects/lacot/
  experiments/bcodec_v0|gk_scan 下 —— 實作第 0 步先盤點，找不到就停下回報，⛔ 不猜。
- 起點：官方 val 段（教材有完整 qpos/qvel 可 reset —— 復用 walk_verify 已解過的
  reset 路徑）。乾淨起點與漂移起點分開記（覆蓋率要報，防「只留容易 donor」）。
- 核心復用：experiments/bcodec_v0/bcodec_gk_drift_eval.py 的 run_reset_chunks
  （約 :67），只需加「保存預測流+實際流」。⛔ 沿用其快照/正規化語義，別重造。
- 規模：每顆 200×4×4 = 3,200 env steps —— CPU 級、單機小時內。

## 接線自檢（跑主表前必過、兩向）

- 正向：v0 的 ŝ 只記錄不進動作通道 ⇒ 置換 ŝ 不得改變動作輸出（bit 級同）。
- 負向：故意把 donor 碼餵錯位置（如全零碼），E_Δ 與動作必須可見地變 —— 證明
  探針走的是真通路、真 payload。
- 完成錨：輸出 JSON 末端寫 n_starts/n_codes/n_steps 三個計數；缺任一=未完成。

## 交付形狀

- 程式：~/Projects/lacot/experiments/firstcuts/c1_edelta/（run_c1.py + README +
  test_c1_wiring.py 含上面兩向自檢）。
- 輸出：results/c1_{dict}.json（每格：起點id、碼id、ŝ四步流、真實四步流、子空間
  誤差）＋ summary 表（三子空間 × {E_Δ, 置換E_Δ, 絕對誤差, 判別力} × 字典）。
- ⛔ 存流不是只存 summary（shat_probe 只存 summary 的教訓，CONVERGENCE 立即修項）。
- 部署：2080ti 四台之一的本機 /archive（⛔ 不經 NFS；rsync 程式+ckpt 過去、
  結果 rsync 回來）。sbatch 單卡或純 CPU 皆可。

## 流程（主人 2026-09-27 裁①）

實作使魔寫 → 混家族檢察雙檢（走真路徑跑 wiring test、對 CONVERGENCE 核設計）→
過了才上機。檢察重點：E_Δ 分母語義（碼對之間、不是對起點）、G×K tuple 完整性、
reset 快照含 warmstart 與 RNG 與 leg bookkeeping、置換對照沒有洩漏正確配對。
