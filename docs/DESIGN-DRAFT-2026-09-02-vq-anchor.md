# VQ 錨定（離散化階梯第二層）設計草稿 —— 2026-09-02（跟主人一起敲用、⛔ 未動手）

_出處：ENERGY-FRAMEWORK §方言病與離散化階梯（主人 8/29 定向）、SURVEY 8/30（Token Assorted / DLR 的混合與 stochastic VQ）、
9/2 全日證據（FINDINGS-2026-09-02 ⑥⑦⑧）。_

## 為什麼是它（9/2 的證據鏈）
- 執行通道無罪：路標對就 1.000／1.000／0.996／1.000（ebfs 分辨器）。
- 失敗全在計畫內容：低分顆的 u 解出的路標在岔路選錯邊／放太遠；四份計畫「一致地錯」時共識選路標救不了、E 選計畫只救回幾分（s23 +0.056）。
- 計畫器好壞由訓練抽籤決定：同 init 同資料、任一隨機源都能把結果從 0.12 拉到 0.56（C 批）；定版配方下仍 0.49~0.87 分級。
- ⇒ 假說：計畫表示 u 沒有錨，每顆 seed 長出自己的方言。VQ 把計畫變成固定字彙的句子，從結構上擋方言病。

## 插在哪（scratch_lacot_rollout.py 的一手位置）
- u 的產生：`e_pooler(traj_enc(traj))` → et [B, K, D]（L482；K=句長、D=每字向量）。
- stage 1 loss：重建 `u_dec(et)` 對 traj（L531）＋ ictr 對比（L542）＋ var/cov（L546）。
- stage 2：flow 對連續 et 算 NLL（L654），另有 anchor／refine／cons／bc（L695）。
- 消費端：`u_dec(·)` 解成路徑（rollout L854／L1006 等）；E 在解出的路徑上算。

**插入點＝pooler 出口**：et → VQ → et_q（straight-through）。
- 決定 u_dec 吃 et_q（訓練與推論一致）；ictr 對比用 et_q 或 et（待敲）。
- flow 的目標：⛔ NF 不能對點質量建模（離散支撐 ⇒ log-density 爆）。走階梯寫的「操作連續、表示離散」：
  flow 仍對【連續】et 建模（encoder 被 commitment loss 拉到 code 附近、但不塌成點）；推論時 flow.sample → snap 到最近 code → u_dec。
  可選：對 et 加小量 dequantization 噪聲讓 NLL 穩。⇒ flow 與 E 完全不改，只重訓 encoder(+VQ)、head、decoder。
- refine／E 選計畫在連續 u 上操作，最後 snap 一次。

## 要跟主人一起決定的兩格
1. **字彙量與粒度**：每個 token 各自量化（V^K 句子，V 待掃：64／256／1024）還是整個 u 一個 code（不建議、容量太小）。
   主人 8/29 的直覺：K＝句長、codebook＝字彙量，聯掃小格網。ルナ建議先 K=8（現配方）× V∈{64,256}。
2. **防 codebook 崩＋訓練排程**：EMA codebook 更新＋dead-code 重置（標準）、commitment β 0.25 起、
   stochastic VQ（量化時加噪、DLR）當保險；從頭重訓 stage 1（不做 fine-tune，避免舊方言殘留）。
   ⚠️ 8/22 淘汰的是 action 離散化（同容量仍輸「什麼都不學」）——這次離散的是計畫 latent，不同物件，但容量損失的風險同型：
   驗收要看 recon 有沒有明顯變差（重建 mse 對比無 VQ）。

## 驗收怎麼算（事前註冊）
- 場：定版配方（warmup500＋EMA）× held-out s20~s27（V8 同八顆）＋ V 兩檔 ⇒ 16 訓 ＋ 16 官方 eval。
- 主指標：八顆 sd 從 0.149 縮多少；次指標：平均（對 0.665）、爛顆數、BC 通道。
- 陽性：sd 明顯縮（< 0.10）且平均不掉 ⇒ 進配方；陰性：sd 不動 ⇒ 方言假說降級，方差另有來源（回 D 批結果）。
- ⛔ 雜訊線 ±0.03（同 ckpt 重評）；別拿單顆講故事。

## 不做的事
- 不換 flow 引擎、不改 E；不上第三層（全 categorical）——那是先驗換 discrete 系，另一題。
- 不在同一批混其他旋鈕（warmup 長度、EMA、β）。
