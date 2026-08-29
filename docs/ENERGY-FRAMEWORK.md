# Energy 框架（主人 2026-08-29 裁定的定位）

_一句話：LaCoT 的 test-time 修正＝「在合成 energy 地形上引導生成」；NF 只是引擎選擇。_

## 框架句

```
E_total(u) = E_geo(u) − λ·log p(u | s,g)
u ← u − η·∇E_total        （實作上兩項各自 normalize 再相加 —— 主人的 trust region）
```

- `E_geo`：幾何 energy（撞牆×10、終點×3、起點×3、路長×0.3，越小越好）。
  【算的不是學的】⇒ 沒有可被優化器鑽的破綻。實作在 `lacot/refine_grad.py` GeoEnergy。
- `log p`：生成引擎的密度（結界）。其梯度 ∇log p 在文獻裡就叫 **score**。
- 家族定位：energy-guided sampling / classifier guidance / Diffuser 系 guided planning。
  差異點：guidance 加在【軌跡 latent】上，且 E_geo 幾何可驗。

## 為什麼叫 energy 不叫 value（主人 8/29 的三個理由）

1. 不跟 RL 的 value（expected return、從資料學）混淆 —— reviewer 不會追問「value 怎麼訓的」。
2. 聽起來（也真的）更接近它的文獻家族。
3. energy 框架自然接 score-based generative 一族 ⇒ 引擎未必永遠是 NF。

## score-based 引擎的替換路（沒立刻要做，先立此存照）

| 零件 | 現在（NF） | 換成 score-based 後 |
|------|-----------|---------------------|
| sample | 一步過可逆橋 | 反向擴散（多步） |
| 結界 ∇log p | NF 精確 log p 的梯度 | score 網路 s(u,t) 直接給（近似） |
| 爬坡 | 生成後另跑 η 步 | 併進每個去噪步（guided diffusion 標準形）|
| refine 與生成的關係 | 兩個階段 | **合流**：生成本身就是一連串 refine |

⭐ 8/24 的疑慮「refine 改去噪會跟 flow 重疊」在 score 框架下是 feature：整個生成即 refine。

## 數學身分：更新式＝乘積分布的 score（主人 8/29 追問後補）

真正想取樣的分布是「引擎分布 × 幾何 Boltzmann 因子」的乘積：

```
p_guided(u) ∝ p(u|s,g) · exp(−E_geo(u)/λ′)
∇log p_guided = ∇log p − ∇E_geo/λ′          ← 分布相乘 ⇒ score 相加
```

對照更新式 `u ← u + η(−clip(∇E) + λ·clip(∇log p))`：方向就是乘積分布的 score。
⇒ 爬坡不是 heuristic —— 它是在 p_guided 上找眾數（Langevin 拿掉噪聲項的 MAP 版）。
⇒ 這個等式跟引擎無關（NF / score model 都成立）＝引擎可換的數學保證。
⚠️ 誠實註記：per-term clip（trust region）讓實作偏離嚴格的 score 方向 —— 那是刻意的
（兩項量級差兩個數量級，8/26 主人裁），寫 paper 時照實講。

## NF ＋ score 一起用的三種真組合（主人 8/29 問「數學上有什麼好處」）

1. **score 疊加性**（上節）—— energy 可加 ↔ 分布可乘，guidance 是精確操作不是近似。
2. **互補組合**：diffusion 的 log p 難算（要沿 probability-flow ODE 積分）、NF 的 log p
   一次前向且精確 ⇒ score model 當生成器（表達力、抗塌）、NF 當守門員（精確結界）。
3. **統一形式＝flow matching**：把連續 NF 與 diffusion 統一 —— 訓練是穩定回歸
   （不碰 NLL 的 Jacobian）、取樣是 ODE、密度仍可估。NF 的兩個實測痛點
   （NLL 不穩、塌成點質量）它都對症；「refine 改去噪會跟 flow 重疊」的疑慮在此變成
   feature：生成本身就是一串 refine。主人 8/18 的 flow-matching 直覺在 energy 框架下歸位。

## 什麼時候值得換（觸發條件，都是實測痛點）

- NF 的 cond 泛化偏：tier2 探針量到 flow 抽的計畫終點偏 9 倍且 log p 無自覺
  （`results/refineprobe_*_tasks-t2_*.json`）⇒ 爬坡目前救得回；救不完就是換引擎的訊號。
- NF 的多樣性塌：D4 線索（recon 目標下塌成點質量）。score/diffusion 的 mode coverage 是強項。
- 成本形狀不變：我們本來就在迭代修 u ⇒ 多步生成不是新增成本。
