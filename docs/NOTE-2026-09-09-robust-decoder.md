# NOTE — 字典 decoder 噪音 fine-tune 能不能治開環接力漂移（2026-09-09）

_對應 [[NOTE-2026-09-09-drift-analysis]] 定讞的病理：schedule 版教材接力 per-leg
只有 .554，執行漂移是失敗大頭（j=50 淨貢獻達量化基線 153×，自我放大）。病理指向
decoder 訓練時只看過乾淨姿態，執行時姿態一偏就外推失效。本檔測藥：decoder
fine-tune 時把「執行時真的會遇到的偏移」餵給它看過（obs 條件輸入加噪），
encoder／VQ codebook 全程 frozen。_

_全部程式：`experiments/robust_decoder/`（新目錄，只 import 上一層 walk_verify /
teacher_relay / drift_analysis 既有函式，未修改任何既有檔案）。一手 log：見文末。_

---

## 〇、一句話

**無效——四顆（σ=0/0.1/0.35/1.0）per-leg 全部落在基線 .554±.07 帶內
（.521~.577），沒有一顆碰到 .62 的「清楚贏」門檻；漂移加測同向印證：最好顆
（σ=1.0）的漂移成長斜率沒有比對照顆（σ=0）平緩，晚段（j10→50）反而略陡
（.152 vs .139），終點漂移還略高（7.40m vs 6.90m）。decoder 訓練時看過偏移姿態，
但沒有轉成開環接力更穩——sanity 對照與 code 不變斷言都乾淨過關，不是管線壞了，
是這個藥在這個劑量範圍沒有測到效果。**

---

## 一、設計與噪音校準（出處）

- 凍結範圍：encoder 與 VQ codebook 全程 frozen（手刻 forward 繞過
  `VQEMA.forward` 的 EMA 更新路徑，`encoder.parameters()` 額外
  `requires_grad_(False)` 雙重保險）；只 fine-tune decoder（302,624 參數）。
  `obs_mean`/`obs_std` 正規化常數也不動（噪音加在正規化「之前」的原始 obs 空間）。
- 重建 loss：`F.mse_loss(recon, x)`，跟原配方的 `rl` 項完全一樣；原配方的
  `+ beta*commit` 項因為 encoder/VQ frozen、對 decoder 是常數，拿掉不影響優化，
  已在 `train_robust_decoder.py` 檔頭寫明理由。
- 噪音校準三個參考點，一手來源
  `experiments/walk_verify/drift_analysis/results/drift_summary.json` 的
  `failure_precursor` 欄（已重新讀檔核對，非轉抄）：
  - 失敗判定最佳單一門檻 `threshold.best_thr` = **0.2352 m**
  - 成功 leg 開始時漂移 p75 `succ_quantiles.75` = **0.3580 m**
  - 失敗 leg 開始時漂移中位數 `fail_quantiles.50` = **1.1831 m**
  - σ_xy 三檔對齊（工單定死的三個數字，這裡標對齊到哪個參考點）：
    0.10 ≈ 次門檻劑量（門檻 .235 之下）；0.35 ≈ 成功 leg p75（幾乎貼合 .358）；
    1.00 ≈ 失敗中位數 1.183 的保守整數近似（同量級，非精確貼合）。
- 其餘 27 維（姿態/速度）：沒有物理意義上的「公尺數」可套，改用跟 xy 同一個
  **相對劑量** `rho = sigma_xy / mean(obs_std[xy])`（xy_std_ref 實測 = 5.556，
  跟 `MAZE_CENTER=(9.42,10.24)` 吻合，確認是原始 xy 位置維度），
  `noise_std[other] = rho * obs_std[other]`（`obs_std` 直接讀 ckpt 已存 buffer，
  訓練時另做「重算 vs buffer」自檢，4 顆全部 `max|Δ|=0.00e+00`，資料沒被動過）。
  三檔 rho = 0.018 / 0.063 / 0.180。
- fine-tune：從 `p0_dict_v1_L4K32_50k.pt` 起，Adam lr=3e-4，batch=512，
  20000 步（4 顆統一步數，避免「劑量」跟「訓練時長」互相混淆），train split
  跟原字典訓練同一份（split_seed=42）。走平判準：`val_mse_noisy` 曲線最後
  20% 均值 vs 前一段 20% 均值，相對變化 <2% 記為走平——4 顆全部 `flat=True`
  （rel_improve 0.00%~0.03%，早早收斂，20000 步綽綽有餘）。GPU sbatch，
  4 顆同批提交（job 26983-26986），各自獨立、~32~35 秒/顆。

---

## 二、驗收結果

### ① sanity 對照（σ=0 應落在 .554±.07 內）

```
baseline（一手：teacher_relay_summary.json）= 0.55357
σ=0（fine-tune 管線跑過、零噪音） = 0.57740   Δ=+0.0238   PASS（<= 0.07）
```

管線本身不傷分——甚至還好一點點（在雜訊帶內，不代表管線變強，只代表管線沒壞）。

REPRO CHECK（`eval_all.py` 自建的 harness 可信度前提，非工單驗收項）：本檔的
`measure_version` 呼叫方式重放「原始未動」ckpt，per_leg 跟 `teacher_relay_summary.json`
存檔值**位元級一致**（diff=0.00e+00）——確認我重用既有函式的方式沒有引入 bug。

### ② code 不變斷言（10 個抽樣段，seed=0）

```
[sigma0]    code_match=True  encoder/vq bitexact=True  decoder 6/6 tensor 改變  => PASS
[sigma0.1]  code_match=True  encoder/vq bitexact=True  decoder 6/6 tensor 改變  => PASS
[sigma0.35] code_match=True  encoder/vq bitexact=True  decoder 6/6 tensor 改變  => PASS
[sigma1.0]  code_match=True  encoder/vq bitexact=True  decoder 6/6 tensor 改變  => PASS
```

四顆全部 PASS——不只 10 個抽樣段的 code index 逐一相等，encoder／vq 的整個
state_dict 逐 tensor **bit-exact**（比工單要求的「10 段 code 一致」更嚴的版本，
順手做了）。decoder 的 6 個 tensor 全部改變（確認真的有在訓練，不是凍過頭）。

### ③ 主結果表（schedule，200 題，四關，並排原版 .554）

```
tag                    per_leg  completion  speed_p50  fall_task  adiff_p50   judge
─────────────────────────────────────────────────────────────────────────────────
baseline(原版未動)      0.5536      0.125      0.103       3.50%      1.236     —
sigma0   (對照顆)       0.5774      0.140      0.1045      2.50%      1.267     ineffective
sigma0.1 (門檻下劑量)   0.5452      0.120      0.1033      1.50%      1.255     ineffective
sigma0.35(成功p75劑量)  0.5211      0.090      0.1034      1.00%      1.248     ineffective
sigma1.0 (失敗中位劑量) 0.5544      0.120      0.1039      2.50%      1.224     ineffective
```

判準：`x>=0.62`→win；`|x-baseline|<=0.07`→ineffective；否則→劑量/方向錯。
四顆全部落在 `[0.4836, 0.6236]` 帶內，全部判 **ineffective**——沒有一顆碰到
0.62，也沒有一顆掉到帶外（沒有出現「劑量/方向錯」）。

⚠️ 全體 argmax 是 **σ=0（對照顆自己）**，三個噪音劑量沒有一個贏過不加噪音的
fine-tune，而且不是單調：σ=0(.577) > σ=1.0(.554) > σ=0.1(.545) > σ=0.35(.521)。
這個非單調形狀，跟「噪音有真實劑量效應」的故事對不上，更像四顆都在一條打平的
線附近抖動——但 ⚠️ 每個劑量只訓了 1 顆（種子跟劑量綁在一起，`seed=20260909+
round(sigma_xy*1000)`），劑量效應跟「這次訓練剛好抽到的隨機性」是混在一起的，
分不開，見第五節。

### ④ 漂移加測：最好顆（σ=1.0） vs σ=0

REUSE CHECK（用原始 ckpt 重放 `run_one_drift`，跟已存檔 `drift_summary.json` 的
`accounting_table` 13 個 j 點 + completion_ratio 全部核對一致，PASS）之後：

```
                    completion   d_xy(j=1)  d_xy(j=10)  d_xy(j=50)  slope(1→10)  slope(10→50)
sigma0   (對照)        0.140       0.044       1.329       6.903       0.1428       0.1394
sigma1.0 (最好顆)      0.120       0.047       1.335       7.399       0.1431       0.1516
```

最好顆斜率比對照顆低：早段(j1-10)=**False**、晚段(j10-50)=**False**。
沒有變平緩——晚段斜率反而略高（.152 vs .139，+9%），終點漂移（j=50）也略高
（7.40m vs 6.90m，+7%）。圖：`results/drift_compare_curve_xy.png`（三條線——
原版／σ=0／σ=1.0——幾乎重疊，肉眼也看不出噪音那顆有比較平）。

### ⑤ gif（最好顆 σ=1.0，schedule）

```
sigma1.0_schedule_success_ep319.gif   24 幀   全走完 3/3 legs（96 步）
sigma1.0_schedule_fail_ep260.gif      50 幀   卡在 0/3 legs（跑滿 200 步預算）
```

⚠️ 我自己只抽看過 `success_ep319.gif` 的第 1 幀（確認左右兩格構圖、螞蟻姿態、
橘點/X 標記位置正常），沒有逐幀播放看完全部——跟既有 NOTE 同一個慣例，人眼關
要主人自己看過才算數。

---

## 三、判讀

**證據鏈**：
1. §二①②：sanity 乾淨過（.577，帶內）、code 不變斷言 4/4 bit-exact PASS
   ⇒ 管線本身可信，接下來的「無效」不是管線壞了。
2. §二③：四顆 per-leg 全部落在基線 .554±.07 帶內，全部判 ineffective；
   最好的居然是零噪音對照顆，三個真正加噪音的劑量沒有一個贏過它，形狀非單調
   ⇒ 在這個劑量範圍內，「decoder 看過偏移姿態」沒有轉成「per-leg 到達率提升」。
3. §二④：漂移加測方向一致——最好顆的漂移成長斜率沒有比對照顆平緩，晚段還略陡、
   終點還略高 ⇒ 不是「per-leg 打平但底層機制其實有在治」，是連機制性訊號（漂移
   成長率）都沒看到往對的方向動。②③④ 三個獨立量測互相印證同一個結論，不是
   單一數字的巧合。

**回答工單的問題——「decoder 見過偏移姿態之後，開環接力執行會不會變穩」**：
**在這次測的劑量範圍（σ_xy∈{0.1,0.35,1.0}，其餘維度按 xy 相對比例縮放）、
這個訓練配方（20000 步、lr=3e-4、只動 decoder）下，沒有測到效果**——per-leg
到達率打平（ineffective），漂移成長率也沒有變平緩。這是三個合法答案
（贏/無效/劑量方向錯）裡的「無效」，不是「劑量或方向錯」（沒有一顆掉到帶外，
沒有系統性變差的證據）。

**對「為什麼沒效」的解讀（推論，非本次量測到的事實，如實標記）**：訓練時的
噪音是「每一步獨立抽樣的高斯擾動」（i.i.d.，依 train 資料的 std 校準大小），
但 §二④ 與 [[NOTE-2026-09-09-drift-analysis]] 定讞的真實漂移是**系統性、
自我放大、跟執行歷史相關**的（decoder 吃到越偏的 obs、下一段動作更歪、
obs 更偏，是一個有方向、會累積的過程，不是每步獨立的隨機擾動）。合成的
i.i.d. 噪音跟真實漂移的「形狀」不同，可能是 decoder 學會對獨立噪音穩健、
但這個穩健沒有遷移到真正的系統性漂移上的原因——這只是一個猜測，本次沒有
設計實驗去驗證它，寫在這裡是為了讓下一次設計知道往哪個方向修（例如：訓練時
用「實際跑出來的漂移軌跡」當噪音來源，而不是獨立高斯）。

---

## 四、沒做到 / 不確定清單

1. **每個劑量只有 1 個訓練種子**（seed 跟 sigma_xy 綁定：`20260909 +
   round(sigma_xy*1000)`）——劑量效應跟「這次訓練抽到的隨機性」完全混在一起，
   無法用本次資料把兩者分開。§二③觀察到的非單調形狀，有可能只是種子雜訊，
   不是劑量本身的效應方向；要分開需要每個劑量多訓幾個種子，本次工單沒有要求
   這麼做（4 顆各自獨立 job，不是每劑量多顆）。
2. **合成噪音是 i.i.d. 高斯，不是真實漂移的形狀**——§三已經標成推論的解讀，
   沒有另外設計「餵真實漂移軌跡當噪音」的對照組來驗證這個猜測。
3. **只測了 schedule（開環）版**，工單範圍本來就只要這版，但沒有量 nn（閉環
   最近鄰修正）版會不會有不同結果——[[NOTE-2026-09-08-teacher-relay]] 已經
   量過 schedule/nn 在原版打平（.554/.555），這次的 fine-tune 對 nn 版有沒有
   效果沒有測。
4. **只測了 20000 步、lr=3e-4 這一組超參數**——4 顆都走平了（rel_improve 全部
   <0.03%），排除了「訓練沒收斂」這個解釋，但沒有掃過其他 lr／更長訓練會不會
   改變結論（設計裡沒有要求掃這個）。
5. **gif 只抽看 1 幀**，沒有逐幀播放兩支 gif——人眼關留給主人，同既有慣例。
6. **側面觀察，非本次驗收項**：所有 4 顆的 `val_mse_clean`（decoder 在乾淨 obs
   上的重建 MSE）都比原版的 0.1054 略好（0.1048~0.1052），代表 fine-tune
   本身有讓 decoder 在訓練目標上進步（不是白訓），但這個進步沒有轉成 per-leg
   到達率的進步——訓練目標（單 chunk 重建 MSE，甚至加噪音版）跟下游真正在意
   的指標（多步開環接力穩不穩）中間有落差，這個落差本身可能是比「這次劑量沒調
   對」更根本的問題，值得下一輪設計時想一下。
7. **意外發現，跟本任務目標無關，如實點名**：`experiments/walk_verify/
   drift_analysis/drift_plots.py` 第 131 行（`if __name__=="__main__"` 自檢
   區塊，不會被 import 觸發，也沒被本任務執行到）裡寫死一個路徑，剛好精確等於
   我這次任務的 scratchpad session UUID（`f61430f0-e6b8-4dfe-93de-8859ae65c8e4`）。
   查過 `git show HEAD` 確認這段內容是真的已提交在 repo 歷史裡（不是我這次
   session 誤植），working tree 跟 HEAD 沒有 diff。這段路徑不可能是巧合
   （UUID 是隨機值），但它是死路徑、沒有惡意 payload、只會在有人手動執行這支
   檔案時寫一張假分位數帶圖到那個路徑——本次任務只 `import
   draw_percentile_band`，沒有觸發它。點名記錄，供主人知悉，不影響本次結論。

---

## 五、一手 log / 產物路徑

```
experiments/robust_decoder/
├── train_robust_decoder.py / .sbatch      訓練腳本 + GPU sbatch（job 26983-26986）
├── check_code_invariance.py / .sbatch     code 不變斷言（job 26987）
├── eval_all.py / .sbatch                  sanity + 主表 + gif（job 26988）
├── drift_compare.py / .sbatch             漂移加測（job 26989）
├── logs/
│   ├── robust-dec-train-{26983,26984,26985,26986}.out   4 顆訓練一手 log
│   ├── robust-dec-codecheck-26987.out                   code 不變斷言一手 log
│   ├── robust-dec-eval-26988.out                        sanity/主表/gif 一手 log
│   └── robust-dec-driftcmp-26989.out                    漂移加測一手 log
└── results/
    ├── ckpt_sigma{0,0.1,0.35,1.0}.pt      4 顆 fine-tune 後 ckpt（.gitignore 排除，可用上面 sbatch 重跑重現）
    ├── train_sigma*.json / loss_sigma*.png   訓練曲線與走平判定
    ├── code_invariance.json                  驗收 #2 結果
    ├── eval_summary.json                     驗收 #1/#3 結果 + gif 路徑
    ├── drift_compare_summary.json            驗收 #4 結果（含 REUSE CHECK 明細）
    ├── drift_compare_curve_xy.png            驗收 #4 圖
    └── gifs/sigma1.0_schedule_{success_ep319,fail_ep260}.gif   驗收 #5

smoke（開發期，保留不刪，同既有慣例）：results/ckpt_smoke.pt / train_smoke.json /
loss_smoke.png（50 步 CPU smoke test，訓練腳本上真跑前的管線驗證）。
```
