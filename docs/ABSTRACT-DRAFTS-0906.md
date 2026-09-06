# ABSTRACT 草稿 v0 — 2026-09-06（大收表判決後）

_目標：ICLR 2027 abstract **9/18**（D-12）。三個變體、不同主打角度，各 150~200 英文字。_
_骨架對應：`PAPER-SKELETON-2026-09-05.md` v2（T-D 主候選）。⛔ 骨架 §2 的 2A/2B 已凍結、數字過期，不要回頭引。_

**⛔ 數字紀律（寫這份時逐條套過、改稿也要照套）**
1. 只用 **FINDINGS 已入庫**的數字（本檔一律標 `⑤@0906` 形出處）。⛔ 不編、不推、不四捨五入到更好看。
2. **`[單顆]` 必標**：只有 s40 一顆 seed 的格，英文寫 `single seed`／`one seed`，⛔ 不得與八顆批的數字寫成同一種語氣。
3. **⛔ 不寫還在烤的**：WS 療程八顆（26 jobs）、p=1 臂、三腿 intent eval — 明早才有〔⑮@0906〕。今天的稿子裡它們**一個字都不能出現**。
4. **層級不得混**：BoN 是 **plan 層**（pts_J raw hit），⛔ 不是 R0 端到端；R0 層目前**無錨也未量**〔⑪@0906、BoN 卡 §3.2〕。
5. **不 claim 分數 SOTA**〔主人 9/5 定調〕；效率數字我方 `[ms/plan@F3 待跑]` ⇒ ⛔ 三個變體現在都**不准寫我們自己的延遲數字**。

---

## 變體 A — 主打「內化」（療程＋慢動力學）★ 推薦

> Route knowledge queryable during training — a shortest-path route over the occupancy
> map, or a free hindsight summary — can be compressed into an intent latent that
> conditions a rectified-flow plan generator, and that interface removed at inference. On a
> maze stitching benchmark, how much survives removal depends on training time far more
> than on the dropout dose. At 8,000 steps no dose we measured does better without the
> interface than a model that never had one, and most do worse (excess success −.036, −.048
> and −.007 at p = .05, .10 and .30; eight-seed batches, seven at p = .30), with a negative
> internalization rate throughout: the model learns to lean on the interface and is crippled
> when it is withdrawn. Trained to 11,429 steps the same p = .30 arm reverses sign (+.086,
> eight seeds). Internalization can also be induced after the fact: continuing a fully
> conditioned checkpoint for 4,000 steps with dropout injected lifts interface-free success
> from .000 to .552, matching the interface-using mode at .556 (paired gap .004; single
> seed). Dropout is necessary — without it the interface-free mode never works — but time is
> what makes internalization appear.

_（~200 字。出處：8000 步四點與 11429 轉正＝①⑤@0906；Int 負值＝⑤@0906；療程＝⑥@0906〔單顆〕；p=0 全崩＝③@0906。⚠️ 兩處精度：idp8（p=.30@8000）是**七顆**不是八顆〔①@0906〕；p=.30 那格的 −.007 相對 base `.317±.036` **在噪音內** ⇒ 英文用 "no dose does better … most do worse"，⛔ 不寫成三點全部顯著更糟。）_

**這個角度的風險**：整篇的頭是那個 `[單顆]` 的療程格 —— 八顆若不重現，abstract 的最後一句當場作廢，而且「時間比劑量重要」很容易被 reviewer 讀成「你們只是多訓練了而已」（我們批級能說的步數效應只有 p=.3 一臂 +.105、~1.5σ〔⑧@0906〕，其餘方向混雜）。

---

## 變體 B — 主打「資訊帳本」（理論框架掛帥）

> Latent thinking cannot create information. A self-generated latent u — a sequence that
> encodes an imagined trajectory — provably carries zero bits about the world, however long
> or however branched the computation; its entire value is computational, extracting what
> the weights already hold. World information must therefore enter through countable
> channels: conditioning, whose likelihood credit equals exactly the mutual information
> injected; external verification, worth at most log N bits per selection and never more,
> over a lifetime, than the verifier's own stock; and observation. The ledger recasts
> internalization as context distillation with a meter — a verifier teaches, the weights
> store, and at inference a self-generated plan extracts. On a maze stitching benchmark the
> meter reads a trajectory rather than a constant: negative at 8,000 steps, where removing
> the interface leaves the model no better than never having had one; positive at 11,429
> (+.086 over a no-interface baseline, eight seeds); and restorable by a 4,000-step course
> (interface-free success .000 → .552, matching the conditioned mode; single seed). A
> training-free best-of-N probe confirms the verification channel pays, at the plan level
> (+.164, +.134 and +.151 at N = 16; single seed).

_（~190 字。出處：零資訊定理＝分類 Thm CT-1；通道界＝帳本 L2/L4；conditioning 兌現額＝帳本 §1(ii)；實驗＝①⑤⑥@0906；BoN＝⑪@0906。u 的地基句已照 CANON 錨在第一句。）_

**這個角度的風險**：把 reviewer 換成理論人 —— 而「自生 thought 零資訊」是一行 DPI，容易被判成 known（CoT filler-token 那條線早就同義），同時我們的實驗**沒有一格在測帳本本身的界**（測的是內化）。框架與證據不同軸，是這個變體最軟的地方。

---

## 變體 C — 主打「診斷方法論」（pre-registered＋探針體系）

> We report a negative result that turned into a positive one, and the protocol that made
> the difference. Conditioning a rectified-flow planner on route knowledge and removing that
> interface at inference kept producing a meter reading of about zero — the usual
> diagnosis being that the model ignores its conditioning. Before the deciding runs we
> pre-registered two competing accounts: an equilibrium one with closed-form predictions
> (dose–response of +.062, +.032 and +.011; a divergence-decay slope ratio of 2.759 ± 20%),
> and a transient one. Both named branches of the readout tree missed. Measured dose–response
> was −.26, −.35 and −.05 — sign reversed — and the slope ratio was 0.990, landing in a third
> state we had named but not predicted: decay is real but independent of the dropout rate.
> The equilibrium account is dead. The transient account survives and predicted what we then
> measured: internalization arrives late (+.086 at 11,429 steps, eight seeds) and can be
> induced by a 4,000-step dropout course (.000 → .552; single seed). We argue that
> pre-registered intervals, named readout trees and instrument-invalidity gates belong in
> any claim about internalization.

_（~195 字。出處：預測值與帶＝postA1 §4／FINDINGS-0905 ⑮ 收緊①；實測 0.990 與 −.26/−.35/−.05＝④⑤@0906；轉正與療程＝①⑥@0906。）_

**這個角度的風險**：開頭就把「我們的理論錯了」放在最大字級 —— 審稿人可以只讀第一句就得到「這篇沒有正面貢獻」的印象，而方法論貢獻在 ICLR 一向給分保守。且它把 §5 的正面結果（轉正＋療程）擠到最後兩句，等於用最強的證據去支撐一個比較弱的賣點。

---

## 一句話比較與推薦

```
變體  主打            最強的一句                     最脆的一點
A     內化（療程）    .000 → .552、配對差 .004       那一格是單顆；八顆在烤
B     資訊帳本        自生 thought 零資訊（定理）    框架與證據不同軸
C     診斷方法論      預測全滅但方法有效             第一印象＝「沒有正面貢獻」
```

**推薦 A**，理由三條：
1. **它是唯一一個「主張＝我們手上最強的量測」的變體。** B 的主張住理論、C 的主張住流程，只有 A 的主張本身就是實驗（負→正→可救）。
2. **它同時吃掉 A/B/C 三邊的料**：殘廢效應（反直覺、記得住）＋轉正（八顆、站得住）＋療程（可操作、審稿人會想試）。B 的帳本可以縮成 intro 的一段機器圖，C 的預註冊可以縮成一句 “pre-registered” 修飾語掛在數字上 —— **反過來不成立**（A 的實驗塞不進 B/C 的骨）。
3. **它的風險是可管理的、而且明早就知道結果**：八顆若重現，A 直接變全篇最強；若不重現，退路是把最後一句換成「dropout 是必要條件」（③@0906，0/250 全崩、不吃 seed 噪音），A 仍然成立，只是少一隻腳。

⚠️ **收 A 之前必須先解一個誠實缺口**：三個變體目前都靠**單一環境**（pointmaze-large-stitch — ⚠️ ルナ驗收修正：使魔原寫 medium、但主線 f27n/idpxm/WS 全家都在 large；medium 是八月舊實驗）。⛔ 現在不能寫 “on OGBench stitch tasks”（複數、暗示跨環境）；今天只能寫 “on a maze stitching benchmark” 或直接點名該環境。若 9/18 前 antmaze（A0 線）沒進來，這個措辭就得一路留到成稿。相關的更深一層在骨架 §6.6：單圖下這把錶**分不開**「攤銷 BFS」與「背下這張圖的路線」〔THEORY-0906-info-vs-function Remark F1.5〕。

---

## 待主人裁（三件）

1. **走 A 嗎**（還是要把 B 的理論框拉到 abstract 前半、做 A/B 混血）。
2. **療程八顆若明早只有部分回來**：A 的最後一句是照寫並標 partial、還是先退成「dropout 必要條件」版。
3. **BoN 要不要進 abstract**：現在只有 plan 層（變體 B 用了一句）。若要進，得先決定 R0 層那 ≈5 GPU-hr 在 9/18 前跑不跑；不跑就只能維持 plan 層措辭。
