# NOTE — 中繼點合成律與內化的統一寫法（式子美化、v0 草稿）

_主人 9/5 問「式子能不能做得更好看」；主人裁：Rei 忙完前ルナ先推、之後請 Rei 一起磨。_
_目標：一條主方程＋三個特例（=三條路）＋內化的 CFG 式寫法。⛔ v0 是敘事級，符號待 Rei 磨嚴。_

## 一、主方程：中繼點合成律（waypoint composition law）

到目標的「好」滿足以中繼點分解的合成律：

    V(s, g) = max_{m ∈ M} [ V(s, m) ⊗ V(m, g) ]        （合成律）

- (max, ⊗) 是一組 semiring 選擇：⊗ 在機率語言是乘積（max-product message passing）、
  在成本語言是加法（min-plus／tropical）。V 的定點＝最短路結構。
- **三個取法＝三條路**：
  1. M＝格圖鄰居、V＝0/1 可達 ⇒ 定點迭代＝**BFS**（我們的儀器；「包住 BFS」是字面的）。
  2. V＝−d(s,g)、d 是 quasimetric ⇒ 合成律＝三角不等式取緊 ⇒ **路線一（距離幾何）**：
     z 空間走直線＝環境走最短路；理論錨＝goal-conditioned value 的 Bellman 不動點是
     最短路距離的變換、TD＝隨機化 Bellman-Ford。可測預測：兩點 z 插值 decode 出合法路徑。
  3. M＝小字典（intent 層條目）⇒ 合成律上的 DP／beam＝**路線二（字典搜索）**：
     BFS 的一般化不是替代品。字典的存在理由＝合成律需要離散支撐（組合性），
     ⇒ 字典住 intent 層、不住 per-token z（9/5 FSQ 2×2＋外部共識 Hydra |V|=64 同向）。

## 二、內化：把合成律的定點攤銷進生成式 prior（CFG 形）

訓練時可查的知識源 O（BFS route、hindsight 摘要）給出錨 a＝A(τ 或 O)：

    訓練：z ~ p_θ(z | s, g, a)     （posterior 形；flow 看得到錨）
    推論：z ~ p_θ(z | s, g, ∅)     （prior 形；O 不在場）
    intent-dropout（p）＝兩者混訓 —— classifier-free guidance 的訓練式，
    我們的 COND_DROP 已是同構（整組版）；INTENT_DROP＝intent 段獨立版（9/5 實裝）。

- **內化度** ≔ gap(posterior 用法, prior)：實測讀數＝帶錨 eval 與零錨 eval 的成功率差
  （idp 臂的錶）。內化好 ⇒ gap 小且 prior 絕對值高。
- **路線三（攤銷蒸餾）的天花板在式子裡看得到**：p_θ(z|s,g,∅) 只能攤銷「訓練分佈裡
  見過的 (s,g)→路線」映射；stitch 要的是用合成律**組合出沒見過的長路** ⇒
  攤銷單獨走不到終點，要合成律的顯式那半（路線二）補 —— 「幾何買多少、搜索補多少」。

## 三、claim 一句話（paper 用）

> Route knowledge that is queryable at training time — a BFS oracle, or hindsight summaries
> of suboptimal trajectories — can be amortized into the conditional prior of a flow-based
> planner; the waypoint composition law makes BFS, quasimetric geometry, and dictionary
> search three instantiations of one operator, and the internalization gap is directly
> measurable by intent-dropout.

## 四、待 Rei 磨嚴的點（ルナ標好的洞）

1. (max,⊗) 選哪組 semiring 對應我們 flow 的 NLL 目標 — max-product 對 argmax 計畫、
   sum-product 對抽樣計畫；我們 eval 是抽樣＋refine，嚴格版可能是 log-semiring。
2. 「quasimetric 約束加進 e_target」的具體 loss 形（對稱破缺怎麼保證）— 接 d 蒸餾資產。
3. 內化度 gap 的正式定義（KL？成功率差？）與 idp 錶讀數的關係。
4. 合成律定點存在唯一性的條件（有限 M、⊗ 單調即可？）。
