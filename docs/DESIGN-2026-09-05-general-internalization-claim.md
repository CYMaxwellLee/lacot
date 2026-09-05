# 一般化內化 claim 的規劃 — 2026-09-05

_主人開題：「BFS 也只是我們的一種內化，我們要 claim 更 general 的部分，先從這邊開始規劃。」_
_✅ 9/5 早主人搬回 9/4 晚原話（第四次恢復記憶），三條路內文已由重推版換成【昨晚定案的真版】；
原話全文另存 memory/luna-2026-09-04-closing.md。_

## ⓪ 一般 claim（草稿，一句話）

**內化＝把「訓練時查得到的路徑知識源 O」壓成 intent latent、條件進生成式 head，
推論時 O 不在場仍保留其效益。** BFS 只是 O 的一個特例（完美 oracle）；hindsight 是另一個
（免費、無圖、可攜去高維）。Claim 的一般性＝架構對 O agnostic ＋ 內化程度可量。

已有的證據雛形：ER（route 錨 ≈ hindsight 錨 .918）＝ O 換源不掉分；f27n R0 .321→.454
＝內化真的把知識帶進 policy 本體；zero 探針教訓＝內化度要用 intent-dropout 臂才量得到。

## 路線三【攤銷蒸餾】— BFS 當 teacher、內化進權重（先走、免費前哨）

- 主張（9/4 原話）：flow 在 route 錨條件下訓練＝一直在學「BFS 的輸出分佈」；
  推論拿掉 BFS 看內化多少。內化＝攤銷任意 teacher 的計算（O-agnostic 的 empirical 脊柱）。
- **⚠️ 天花板（9/4 定、卡在我們痛點上）**：蒸餾只能內化「見過的路線分佈」，
  而 stitch 的核心是組合出【沒見過】的長路 ⇒ 路線三單獨走不到終點 —
  定位＝**前哨**，量「內化的極限在哪」。
- 儀器：zero 探針已判儀器無效（COND_DROP 整組歸零 ⇒ 部分歸零是真 OOD）；
  正確版＝**intent-dropout 臂**（訓練時 p=0.3 intent 段獨立歸零、s,g 保留）＝內化度的錶
  ＋同顆模型帶查/不帶查雙部署。3 行 patch、過目→單顆→八顆。
- 延伸（teacher 光譜、降後）：route≈hindsight（ER 打平）已是兩個 O；OGBench 軌跡本非
  最短路 ⇒「不完美 teacher 可內化」已有免費證據；受控劣化 teacher 版寫 paper 時補。
- 高維扣件：A0（ant）teacher 只剩 hindsight ⇒ 內化 claim 在高維被自然測試。

## 路線一【距離幾何】— quasimetric 進 e_target（B 階段前置、Rei 理論線接口）

- 主張（9/4 原話）：讓 latent 距離 ≈ 環境最短路距離（quasimetric 約束加進 **e_target
  訓練**、即改 stage1 目標）。做到則「z 空間走直線」＝「環境走最短路」— BFS 變成
  latent 的測地線、不用搜。
- 理論錨：goal-conditioned value 的 Bellman 不動點＝最短路距離的變換；
  TD learning＝隨機化的 Bellman-Ford。
- **可驗證預測（便宜、既有資產可量）**：兩點 z 插值 decode 出的路徑應合法 — 直接量。
  〔9/5 施工中：probe_z_geodesic（使魔）。判讀先釘：這是「加 quasimetric 之前」的基線尺 —
  真軌跡 decode 合法率高＋隨機 u 低＝儀器有效兩 gate；插值合法率與距離相關的絕對值
  不設 pass/fail，是路線一之後要打敗的 before 錨。〕
- 交接 Rei：形式化「intent＝geodesic 結構的充分統計量」＋ hindsight 在什麼條件下恢復
  quasimetric。資產：d 蒸餾（8/30 續帶）、往返尺、intent.py 弧長重採樣。
- **⚠️ 要盯的張力（9/4 定）**：距離幾何壓進 z 會跟「z 表示路線形狀細節」搶容量 —
  R0 卸貨故事反過來演。到時候量、不猜。

## 路線二【組合推理】— max-product 一般化（B 階段主體、paper 的骨）

- 主張（9/4 原話）：p(計畫|s,g) 分解成子計畫乘積、對中繼點取 max ＝ max-product
  message passing；BFS＝它在格圖上的特例。⇒ B 階段字典空間 DP／beam **不是 BFS 的
  替代品、是 BFS 的一般化** — 包法的正式版。
- 與 FSQ 重想（9/5）的接點：字典存在的理由是【組合性】不是分數；組合性住 intent／
  拓撲層（小字典），不在 per-token z 座標 — fsq27 2×2 定讞與外部共識（Hydra 64 格等）
  同向。字典驗收＝配對可學性＋利用均勻度兩關。
- 等 Rei 理論線＋主人開工；實驗排在三、一之後。

## 三條的關係（9/4 定案）

- 改的組件不同所以相容：一改 stage1 目標（z 幾何）、二改推論結構（生成怎麼分解）、
  三改訓練資料流（teacher）。
- **一×二＝連續體**：幾何做得越好、推論期搜索越少 — 「幾何買多少、搜索補多少」的分帳。
- **三×二＝同一推理的兩種付費時點**：攤銷（訓練期壓進權重）vs 顯式（推論期真搜）。

## 順序與分工

```
路線三（攤銷）  現在可動：intent-dropout patch → teacher 光譜批（等主人核）
路線一（幾何）  便宜探針＋寫交接包給 Rei（等主人點頭）
路線二（組合）  理論化，等 Rei／主人開；字典實驗押後
A0（ant）      rail 一通就走（口徑五件套已入 DESIGN-0904）
```

高維標準（9/5 主人定調）：**分數是搖籃裡的，能帶走的只有內化那條線** —
所有零件（teacher／subgoal 腿／字典）一律用「在高維還存在嗎」判去留。

## ICLR 定位（9/5 主人核「記錄這幾個大方向，來推」）

有相的四點：①可歸因的內化框架（BFS 儀器→兩腿歸因乾淨）②效率軸（毫秒級 vs 競品
diffusion 規劃 37~530 秒/query）③空地插旗（humanoidmaze-large-stitch 五篇競品全沒碰、
HIQL .28）④方法學附贈品（recon 好≠空間好學 2×2、判讀樹、儀器無效判定）。

缺的三塊＝路線圖的三個目的地（⛔ 缺一塊就從 ICLR 掉回 maze study）：

```
缺① 高維實證          → A0 ant（rail 通了就走）→ humanoidmaze-large-stitch 插旗
缺② 主 claim 系統本體  → 路線二（intent 層小字典＋合成律 DP、B 階段）
缺③ 內化數字立得住    → idp 錶（內化度讀數）＋R0 線；門檻按環境看
                        （humanoid-large-stitch HIQL .28 ⇒ 內化線 .3~.5 即插旗）
```

式子美化軌：合成律 note（NOTE-2026-09-05-composition-law.md）＝一條主方程三取法＋
CFG 式內化寫法；待 Rei 有空磨嚴（她忙完前ルナ先推、主人 9/5 裁）。
