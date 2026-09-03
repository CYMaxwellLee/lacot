# 離散化的正確層：文獻五路＋兩層架構草案 —— 2026-09-03 晚

_主人 19:02 指示「記錄好這些路，然後我們開始規劃實驗」。出處：9/3 全日實驗（FINDINGS-2026-09-03 ③③'③''）＋晚間調研。_

## 一、今天的完整數據圖（兩條管線、五種字彙劑量）

```
subgoal 管線（計畫→路標→BC 執行；吃座標精度）      端到端管線（head 直讀 u；吃 u 可讀性）
基線(凍s20+soft) .792 sd .040                      R0 .373  u淨貢獻 +.08
z甲(8維瓶頸+round) .712 −.08                       R0 .455  +.26
u snap .651 −.14 ／ z乙 .604 −.19 ／ u dequant .237  R0(z乙) .505  +.31（升7平1降0）
⇒ 字彙化：傷 subgoal 腿（單調劑量線）、救端到端腿（head 終於讀 u）
```

## 二、文獻五路（2025-2026；全部引用前需正文核，摘要頁已核）

1. **DreamerV3**（arXiv 2301.04104；Nature 2025 版）：32×32 categorical latent 贏連續控制。
   ⭐ 關鍵對照：它是【每步閉環】（每 timestep 重新觀測重出 latent）⇒ 量化誤差不累積成位置誤差；
   我們是【開環計畫吐路標座標】⇒ 誤差直接是座標誤差。配方：straight-through＋unimix 1%（0.99softmax+0.01uniform 防塌）。
2. **Hydra**（arXiv 2608.28995，2026）：⭐ 與「兩條腿」同構的既有解——離散 latent 只壓 intent
   （pose/action 字典各 64！visual 2048）、規劃搜索全在離散空間（Gumbel-Max 候選＋Kinematic-Perceptual
   cost 挑）、選中的 intent 由 flow-matching 連續 decoder 還原精確軌跡。**離散管「選哪種未來」、連續管「精確在哪」**。
3. **iFSQ**（arXiv 2601.17124，2026）：vanilla FSQ 的 activation collapse——激活高斯 vs 格子等距 ⇒
   中央擠爆（利用率 83%）。一行修：bound 由 tanh 改 **2·sigmoid(1.6z)−1**（分佈匹配 ⇒ 格點均勻用）。
   ⚠️ 我們 9/3 只驗「每維 8/8 有人」未驗均勻度。
4. **Soft Tokens, Hard Truths**（arXiv 2509.19170，2025，LLM）：訓軟推硬的系統實測；訓推不匹配代價
   視任務。我們 z甲 的 −.08 是同形狀量測。
5. **VQActFlow**（2606.21600）／**AnchorRefine**（2604.17787）：VQ 管動作模式＋flow 管連續／離散錨＋連續殘差。
   「離散粗、連續細」＝機器人圈共識形狀。

## 三、兩層架構草案（Hydra 式、我們的版本；⛔ 設計題、跟主人一起敲定才動）

- **上層（離散、小字典）**：intent／路線 token——管拓撲（走哪條走廊、經過哪些區塊）。
  可搜（離散空間 BFS/beam）、可讀（head/verifier 讀它）、跨 seed 錨定（FSQ 格固定）。
  候選來源：現有 E 格制的區塊序列？或 u 的 coarse 字典（k 小、如 64）。
- **下層（連續）**：現行 decoder／flow——給定上層 intent，還原精確路標座標。
  精度住這層（今天 subgoal 腿的教訓）；可用 conditional flow（cond=intent+s,g）。
- 對應今天資產：上層≈守門用的 E 圖搜索的「學習版」；下層≈現行 u_dec；R0 腿吃上層 token。

## 四、9/3 夜實驗計畫（灑到明天中午；主人核准的牌面見 Telegram）

- **N1 格點直方圖探針**（本機、分鐘級）：現 fsq 每維 8 格佔用分佈 ⇒ 有中央擠就觸發 N2。
- **N2 iFSQ bound 修**（fit 幾分鐘＋z甲z乙 各 8 顆重跑）：字典本身修好再判。
- **N3「只壓縮不離散」臂**（8 顆）：z甲拿掉 round ⇒ 把 −.08 拆成「壓縮費」vs「離散費」。
- **N4 端到端腿×讀 u 加壓**（z乙 配方 × COND_DROP∈{0.3,0.5} × 8 顆 ×2＝16 支）：
  主人 9/2「要加強讓模型讀 u」——字典化已讓 R0 +.13，加壓 cond 遮蔽看能推多高。
- **N5 凍 stage 1 穩健性**（s26、s27 各凍一次 ×8 seed＝16 支）：「語言只教一次」防
  「只是 s20 運氣好」的 reviewer 攻擊；順帶第二三顆官方數字。
- 機隊：Jasmine／Moana／Lady／Pocahontas 訓練＋eval、ZeldaJr eval 主力（主人 19:02 指定五台）。
