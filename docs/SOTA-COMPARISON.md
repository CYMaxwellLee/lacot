# 對標 SOTA 的注意事項（2026-08-29 與主人議定）

_對象：LAVL（SOTA）、HIQL（階層近親）、TTGS（test-time 同軸）、官方 GCBC/GCIQL/QRL。_

## 六條紅線（按殺傷力排）

1. **chunk 公平性（最大的坑）**：SOTA 全是每步推理；chunk=4 的時間平滑自帶 0.15→0.85 級
   的紅利（POMDP 效應，實測）。最終表 chunk=1/4 雙報，**主打數字用 chunk=1**。
2. **上表只用官方協定**：5 task × 50 seeds、官方成功判定。dev 尺（BFS 分層 300 題）是
   開發儀器，只做設計決策與 ablation，⛔ 不上對標表。
3. **S0（BFS 中繼點）永遠只當 oracle 上界與拆功勞的尺**（它偷看迷宮）。提案數字必須是
   latent 自己供點的版本（S1 修復版／conf2）。
4. **資訊使用申報**：energy 佔據圖從資料蓋、⛔ 不讀 env.maze_map（已做）；test-time
   compute 較多（每段抽 M 份＋修正）要披露，與 TTGS 同軸比。
5. **SOTA 數字用原 paper 或官方 code 重跑**，⛔ 不拿自己復刻的弱版本墊背（主人 8/22 裁）。
6. **覆蓋**：上表前補 3+ seeds；主戰場是 **large**（最難層 bc 0.08 級、TTGS 自承流形外
   收益趨零 ⇒ 遠程規劃＋信心分段該贏最多的地方）。medium 是練兵場。

## Compute-matched comparison（主人 8/29 提，reviewer 必問的先發制人）

「增益是方法還是算力」→ 給 SOTA 同樣的 test-time 算力：

| 方法族 | 加算力的方式 | 成本 |
|---|---|---|
| bc / GCBC | ensemble N 顆平均動作（我們的 bc 先做） | 低 |
| HIQL / LAVL（有 critic） | best-of-N：抽 N 動作挑 Q 最高 | 低 |
| TTGS | 加大 sample 數（算力軸同構＝最公平對手） | 低 |
| 任何方法 | 加大模型 | 重（只對關鍵 baseline 或引其 scaling 數據） |

呈現：**compute-performance 曲線**（x=每題推論算力、y=成功率、每法一條）——
要嘛整條 dominate、要嘛高算力端拉開。⛔ 不用單點表講 test-time 故事。

⭐ 我們的獨有賣點：**信心機制＝算力自適應** —— 簡單題直達（算少）、難題多抽多修（算多），
per-episode 算力分布可直接量、畫進圖（adaptive test-time compute 敘事）。
⭐ 第二層自適應（主人 8/29）：**抽樣數 M 也自適應** —— 先抽 2 份判信心、夠就用、
不夠再加抽（sequential sampling，至 M_max）⇒ 簡單題的算力再砍一半以上。ablation #12。

## 已知的自家強 baseline 事實

- bc 0.710（medium-stitch dev tier2）之所以高：chunk=4 平滑紅利＋同資料同預算的獨立
  head＋小迷宮岔路有限。⏳ 官方 GCBC 在 stitch 的原表數字待查核後填入此處。
