# 重新設計「上界」— 現在這個 ORACLE 不是天花板

_2026-08-23。待主人裁示要走哪一個（或哪幾個）。_

## 問題

現在的 ORACLE 這樣做：

```python
def oracle_etarget(obs, goal):
    poss = expert_positions(obs, goal)   # env 內建 BFS ＋ 簡化點動力學 xy += 0.2*a
    return e_pooler(traj_enc(poss))      # 編成 e_target 餵給 head
```

`[實測]` 它只拿到 **0.120**，而 LaCoT R=3 open 是 **0.700**。⇒ 它不是上界。

`[推論]` 原因是 **OOD**：head 訓練時只看過兩種 `u` —— 資料集軌跡編出的 `e_target`、
以及 `refine(flow.sample())`。而 oracle 餵的是**第三種**：另一個規劃器用**簡化動力學**
走出來的路徑。head 沒學過怎麼讀它。

⇒ 「上界」的定義應該是「**餵 head 一個它讀得懂、而且盡可能好的 u**」，
而不是「餵一個由更強的規劃器產生、但 head 沒見過的東西」。

---

## 三個候選

### A. Dataset-hindsight oracle（最貼近訓練分布，但只能離線）

在**資料集**裡取 (s, g)，用**真實的未來軌跡**（跟訓練時同一種 relabel）算 `e_target`，
量 action 誤差。

- ✅ 完全 in-distribution，是「如果 `u` 完美，head 能多準」的乾淨上界
- ❌ **不能在 live env 做** —— rollout 時還沒走過未來
- ⇒ 它給的是 **action 準度**的上界，不是 success rate 的上界

### B. Best-of-N oracle（live env 可做；ルナ推這個）

rollout 時從 flow 採 **N 個 `u`**，用一個 oracle 準則挑最好的那個（例如「哪個 `u` 解出來
的動作最接近 BFS 的方向」，或直接用訓好的 value）。

- ✅ **完全 in-distribution** —— 每個候選都是 flow 自己生的
- ✅ 直接量的是 **test-time scaling 的上界**：「如果我們每次都挑對，能到多好」
- ✅ **直接接上主人的 value-directed refine** —— 那個點子要的就是「往哪個 u 爬」，
  而 best-of-N 先告訴我們「爬到最好能有多少」
- ⇒ ルナ覺得這個對 LaCoT 最有意義：它量的是**方法自己的頭**，不是別人的規劃器

### C. 修現有 oracle 讓它 in-distribution（最小改動）

把 `expert_positions()` 從「簡化點動力學」改成**用真實 env 模擬**
（複製 env state、跑專家 policy、收集真的 observations）。

- ✅ 改動小，保留「有人給你正確的路」這個語意
- ⚠️ 仍不完全 in-distribution（訓練時的軌跡來自資料集的行為 policy，不是 BFS 專家）
- ⚠️ 而且 `[實測]` 顯示 head 對 `u` 的分布很敏感，改了動力學未必就夠

---

## ルナ的建議順序

1. **B（best-of-N）** —— 它才是 LaCoT 該有的上界，而且直接鋪好 value-directed refine 的路
2. **A** 當附帶的離線 sanity（便宜、幾行就能加）
3. **C** 低優先 —— 修好也只是讓一個「別人的規劃器」的分數好看一點

⛔ 無論選哪個，**現在那個 0.120 不能再被當成天花板寫在任何地方**。
「u 還欠 0.51」那種算式已經作廢。
