# DESIGN — GRPO for u：整合更新目標 v0（2026-09-08）

_主人 6123：「怎麼同時把這些整合在一起變成好的 grpo 更新目標，規劃一下」。_
_整合的零件：四關 reward（6119）／字串進 u（6112-6115）／hindsight（9/7）／
covariate shift 教訓（M2 蹲姿吸子・humanoid 摔）／教材上界 .554／乘法閘哲學。_
_自檢標記：〔量〕＝量出來的、〔例〕＝沿既有先例、〔拍〕＝拍的（初版值、要掃或呈裁）。_

## 〇、一句話

u（thinking）當字串的作者：SFT 學教材、GRPO 用四關 reward 磨「自己走出來的狀態」，
hindsight 把失敗變教材 —— 更新目標＝ **L = L_GRPO(四關R, KL錨) + λ·L_SFT(教材＋hindsight buffer)**。

## 一、八格設計

### ① policy 與候選單位
一個候選＝一條字串（當下狀態→路標，5~10 字）。組＝同 (s, 路標) 抽 G 條。
骨架（xy 路線）照舊 conf2 —— 一次動一層〔例：9/8 裁示〕。

### ② 字串怎麼進 u（實作形式、三選一）
- (i)〔推薦〕**自回歸字串 head，conditioned on (u, s, 路標)** —— u 提供跨字脈絡、head 當作者。
  最小改動；＝「M2 的序列版＋u 脈絡」。GRPO 作用在它的 logits 上（離散、標準形）。
- (ii) u 擴成混合通道（flow 出離散要 ST/argmax，複雜）——後手。
- (iii) 獨立 transformer 不掛 u —— 簡單但 thinking 整合故事弱。

### ③ reward＝四關組合〔主人 6119 定調〕
```
R = gate × (w1·到達 + w2·限時 + w3·步速貼真 + w4·平滑/像樣)
gate = 1[沒翻倒] × 1[沒凍結]        ← 凍結＝4步位移<0.02（M2 吸子的直接判死）〔量：M2 診斷〕
各分項從量尺包算〔量：9/8 已建〕；w 初版等權〔拍：掃敏感度或呈裁〕
```
乘法閘擋單項刷分（防 P2 碎步偏科）〔例：pointmaze GrpoReward〕。

### ④ rollout 起點＝covariate shift 的正面攻擊（本設計的靈魂）
起始狀態混合：教材乾淨態 50% ＋ **policy 自己走出的狀態 50%**〔拍：比例〕——
讓模型在「蹲姿邊緣」也練選字。DAgger 精神進 GRPO：探索分佈＝自己的閉環分佈。
（M2 的死因＝只見過教材態；這格直接治它〔量：蹲姿吸子 29.4% 凍結〕。）

### ⑤ hindsight relabel〔主人 9/7 藥〕
字串沒到 g 但走到 g′ ⇒ (s, g′, 字串) 進 SFT buffer 混訓（成功經驗回收、自蒸餾）。
解 pass-G 過低；字串空間的 relabel 比連續版乾淨。

### ⑥ 完整目標
```
L = L_GRPO(R 組內標準化 advantage、KL 錨到 SFT policy〔例：LLM 標準形〕)
  + λ·L_SFT(教材 ＋ hindsight buffer)          λ〔拍〕
```
KL 防跑飛、λ 防忘教材。全在離散 logits 上做。

### ⑦ 模擬預算〔量：估算〕
G=8〔拍〕× ~40 步/條 × batch 64 ≈ 2 萬模擬步/batch；MuJoCo CPU ~1 萬步/s ⇒ ~2s/batch，
rollout CPU 池並行、GPU 只跑 policy —— 可行。

### ⑧ 驗收判準（預釘）
主錶＝閉環 per-leg 到達率（對照：M2 .060／守門 .124／主頭 .320／教材 .554）；
副錶＝**凍結率**（vs M2 29.4% —— 吸子有沒有被治的直接讀數）＋翻倒率＋四關全表。
訓練中看 reward 曲線＋凍結率曲線。

## 二、階段化（順序鐵則：SFT 站住才 GRPO）

- **階段 0｜字串進 u 的 SFT**（前置、還沒做）：②-(i) 的 head、教材＝hindsight 真字串
  （接力尺度、M2 的標註管線直接重用）。**⚠️ 先量它自己的閉環** —— 序列連貫性可能
  已經比 M2 好，說不定不用 GRPO 就過守門；先量再 GRPO，歸因才乾淨。
- **階段 1｜GRPO**（本設計）：④ 的混合起點＋③ 的四關 R＋⑤ hindsight buffer。
- gate between：階段 0 的閉環數字出來、跟主人一起裁階段 1 的開法。

## 三、自檢彙總

〔量〕四關量尺包／教材 .554／M2 凍結 29.4%／模擬成本估算／亂猜線 5.1%
〔例〕KL 錨與組內標準化（LLM 標準形）／乘法閘（pointmaze GrpoReward）／一次動一層
〔拍〕G=8／起點混比 50:50／w 等權／λ／KL 係數 —— 全部初版值，階段 1 開跑前呈裁或掃

_相關：DESIGN-0908-b2 §三.五（u 遠景）／NOTE-0908-m2-vqselect（吸子診斷）／
NOTE-0908-teacher-relay（.554 上界）／DESIGN-0906-grpo-thoughts（路線層 GRPO 前例）。_
