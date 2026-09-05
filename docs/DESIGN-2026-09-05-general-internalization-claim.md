# 一般化內化 claim 的規劃 — 2026-09-05

_主人開題：「BFS 也只是我們的一種內化，我們要 claim 更 general 的部分，先從這邊開始規劃。」_
_⚠️ 三條路（幾何/組合/攤銷）的名字與順序（三→一→二）出自 9/4 handoff；內文當天只在對話裡，
本檔是ルナ照名字＋「B 階段理論化＝max-product 一般化」「路線一與 Rei 理論線交接」兩根錨
重推的版本 — 與昨天對話有出入的地方以主人記憶為準。_

## ⓪ 一般 claim（草稿，一句話）

**內化＝把「訓練時查得到的路徑知識源 O」壓成 intent latent、條件進生成式 head，
推論時 O 不在場仍保留其效益。** BFS 只是 O 的一個特例（完美 oracle）；hindsight 是另一個
（免費、無圖、可攜去高維）。Claim 的一般性＝架構對 O agnostic ＋ 內化程度可量。

已有的證據雛形：ER（route 錨 ≈ hindsight 錨 .918）＝ O 換源不掉分；f27n R0 .321→.454
＝內化真的把知識帶進 policy 本體；zero 探針教訓＝內化度要用 intent-dropout 臂才量得到。

## 路線三【攤銷】— teacher-agnostic 內化（先走，empirical、maze 內可完成）

- 主張：內化＝攤銷任意 teacher 的計算。驗法＝teacher 光譜實驗：
  完美 BFS route／hindsight（無圖）／劣化 teacher（子採樣 BFS、greedy 啟發式、
  帶折扣 value-iteration 路徑）→ 各 8 顆，量 subgoal/R0/內化度三格。
- 預期 claim 形：內化效益對 teacher 品質退化是平滑的、不依賴 oracle 完美性。
- 儀器：**intent-dropout 臂**（訓練時 p=0.3 把 intent 段獨立歸零、s,g 保留）＝內化度的錶，
  同顆模型雙部署模式。3 行 patch、先給主人過目。
- 高維扣件：A0（ant）上 teacher 只剩 hindsight ⇒ 路線三的 claim 在高維被自然測試。

## 路線一【幾何】— quasimetric 線（與 Rei 理論線交接）

- 主張：內化的 latent 實作了一個學來的 quasimetric／geodesic 場；BFS＝格圖度量的
  geodesic oracle 特例。
- 資產：d 蒸餾（quasimetric student、8/30 續帶未動）、往返尺、intent.py 弧長重採樣。
- 便宜探針（可先跑）：intent embedding 距離 vs 真實路徑距離的相關 — 既有資料可量。
- 交接 Rei：形式化「intent＝geodesic 結構的充分統計量」＋ hindsight 摘要在什麼條件下
  恢復 quasimetric。
- ⛔ 先探針＋交接、不先大灑（單設定只講單設定）。

## 路線二【組合】— max-product 一般化（B 階段理論化、最深、最後）

- 主張：BFS＝min-plus／max-product semiring DP 的定點；我們的兩層系統＝攤銷 semiring DP。
  字典（B 階段）＝DP 可組合性所需要的離散支撐。
- ⭐ 與今日 FSQ 重想的接點：字典存在的理由是【組合性】不是分數 — 而組合性住在
  intent／拓撲層（小字典），不在 per-token z 座標。fsq27 的毒（2×2 定讞）與
  「放錯層」假說一致；使魔調研回來後在此節收斂。
- 等 Rei 理論線＋主人；實驗（intent 層小字典、k~64）排在路線三、一之後。

## 順序與分工

```
路線三（攤銷）  現在可動：intent-dropout patch → teacher 光譜批（等主人核）
路線一（幾何）  便宜探針＋寫交接包給 Rei（等主人點頭）
路線二（組合）  理論化，等 Rei／主人開；字典實驗押後
A0（ant）      rail 一通就走（口徑五件套已入 DESIGN-0904）
```

高維標準（9/5 主人定調）：**分數是搖籃裡的，能帶走的只有內化那條線** —
所有零件（teacher／subgoal 腿／字典）一律用「在高維還存在嗎」判去留。
