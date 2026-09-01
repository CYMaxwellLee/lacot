# 對標 SOTA 的注意事項（2026-08-29 與主人議定）

_對象：LAVL（SOTA）、HIQL（階層近親）、TTGS（test-time 同軸）、官方 GCBC/GCIQL/QRL。_

## 六條紅線（按殺傷力排）

1. **chunk 公平性（最大的坑）**〔🚨 2026-08-31 依官方尺數據改打法、主人核〕：官方尺實測
   chunk1 vs chunk4＝medium 0.544/0.920、large 0.284/0.796（−37.6/−51.2），且 chunk1 之下
   bc≈分段＝導航加值歸零（執行端每步抖動吃掉供點價值）。⛔ 舊條文「主打用 chunk1」作廢 —
   會把表打回 QRL 之下且錯誤歸因。**新打法＝同 chunk 尺度比**，理由四層：
   ①先例合法（ACT／Diffusion Policy 皆以 chunk 出段動作為核心設計；OGBench 協定管 task/seed/
   成功判定、不管 policy 內部）②變因隔離（表列 GCBC+chunk4 對照 — bc 地板＝近似版、真獨立版
   必補 — 同尺度下規劃增益 +20~49 分仍在 ⇒ chunk 從變因中消去）③chunk1 消融誠實列＋機制講明
   （執行抖動、本身是發現）④貢獻定位一致（claim 在規劃層、執行器是載體，同 GAS 另訓低層）。
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

## 官方對標表現況（2026-08-31 白天收官；全部 5 task × 50 seed 官方協定）

```
pointmaze-＊-stitch     medium          large
我方主打（蒸餾+ma2）     0.852 (K8,2sd)  0.535 (K8,3sd) / 0.796 (best sd)
我方 ebfs 上界（帶圖）   0.800           0.972
官方行情  QRL            0.80            0.84
          HIQL           0.74            0.13
          GCIVL          0.70            —
          GCIQL          —               0.31
          GCBC           0.23            0.07
```

- ⚠️ 紅線遵守註記：medium K8 只有 2 seeds（0.784/0.920）—— 上正式表前補 s2（紅線 6）。
  我方全部 chunk=4 —— **chunk=1 主打數字未跑**（紅線 1、8/29 舊債），上表前必還。
- large 報法：平均 0.535 誠實報＋best-seed 0.796 另欄標明（seed 方差 0.388~0.796 是
  已量化的已知問題，藥＝EMA／dev 挑顆，8/31 三問已判：步數無效、K12 無效）。
- ⭐ 敘事鏈（表的故事）：ebfs 上界 97.2 證天花板 ≫ QRL ⇒ 蒸餾（medium 已到 0.920 單 seed、
  超上界＝amortization 不設限）⇒ large gap＝訓練方差工程債，非方法債。

### Test-time 圖搜索三 baseline（SURVEY 8/30 正文級抽讀；主人批註定位）

| 法 | 已知數字（其報告尺） | 我們的差異化 |
|---|---|---|
| SCoTS (2505.20983) | 只做 pointmaze/antmaze 兩 maze | 贏法四件：最優性 teacher、零生成縫、amortization（部署無圖）、humanoidmaze 覆蓋 |
| GAS (2506.07744) | giant-stitch 88.3（vs 先前最佳 1.0） | 它部署帶圖＋另訓低層；我們蒸餾掉圖 |
| TTGS (2510.07257) | pm-giant GCIQL 0→98.0；hm-giant 4.4→78.1 | 它自列限制：部署訪問訓練資料＋建圖 35-100s/次 ＝我們「蒸餾去圖」的現成 motivation |

⏳ 三法在 medium/large-stitch 的逐格數字待抽原文表（現只有 giant 級）；⛔ 引用前抽查原文（26xx 番號全在基底知識後）。

## 已知的自家強 baseline 事實

- bc 0.710（medium-stitch dev tier2）之所以高：chunk=4 平滑紅利＋同資料同預算的獨立
  head＋小迷宮岔路有限。⭐ 8/31 官方段的對應事實：我方 bc 頭 medium 0.60~0.78（顆各異）
  vs 官方 GCBC 0.23 —— 差距三因子（chunk4＋cond 表徵＋teacher 外溢）已列 8/30 續帶、待拆。

## Related work 結構（主人 2026-08-29 裁：就這三類）

定位一句話：**offline 學一個「路網先驗」，測試時在上面做 energy 引導的 latent thinking，
想多深、走多遠由信心自適應。**（廣義 offline model-based —— model 的是軌跡先驗，非轉移動態。）

| 族 | 代表 | 我們拿了什麼 | 我們補了什麼 |
|---|---|---|---|
| offline model-based RL | MOPO、COMBO、MOReL | 不用環境互動 | 不學轉移 f(s,a)→s'（少一個誤差源）；model 的是「世界允許哪些路」 |
| 生成式軌跡規劃 | Diffuser、Decision Diffuser | 整條路一次想、分布上引導 | latent 壓縮的計畫（u）＋信心自適應（選點與抽樣數兩層） |
| test-time thinking / scaling | TTGS、LLM CoT | 測試時算力換品質、自己決定想多深 | 幾何可驗的 energy —— 想錯了推得回來，不是純自我一致性 |

⭐ 三族交界不是身分尷尬、是賣點：每族的優點各拿一塊，各族缺的那塊都有補。

## Anticipated QA：「離散了幹嘛不直接 Transformer 出軌跡？」（DT/TT 之問，主人 8/29 晚）

四張牌（硬度排序）：
1. **AR 序列模型結構性不會 stitch**：它是密度模型，抗拒「資料裡沒出現過的片段組合」，
   而那正是 stitch 的定義（DT can't stitch 有專文；OGBench stitch 上 GCBC 23/7 分同病）。
   我們把組合搬到 latent 規劃層＋分段執行：短程踩分布內、長程只需幾何合法。
2. **token 粒度差三個數量級**：TT 的 token 是軌跡逐步轉錄（千級、beam 貴、改=重生）；
   我們的 u 是計畫摘要（K=4、成本與 horizon 解耦、修正=改四個字）。
   categorical 化之後仍是「計畫的字」，不會變成「軌跡的字」。
3. **幾何可驗的 E**：TT 的剪枝靠 likelihood / learned reward（有破綻可鑽）；E 算的沒破綻。
4. **test-time compute 形狀**：TT 只有加寬 beam 一招；我們有結構化階梯（抽樣數、修正輪、
   信心深度）可畫劑量曲線。
誠實邊界：短程、資料有完整示範的 regime，TT 簡單有效 —— claim 圈定在
「長程＋無完整示範」（large-stitch 除 QRL 全滅的那塊）。

## 高維戰場 baseline 表（2026-09-01 抓 OGBench 原文 arXiv:2410.20092 HTML）

大家比的 stitch/play 系、官方六 baseline 的每題最強（success %）：

```
antmaze-large-stitch     HIQL  67±5    （GCBC 3, QRL 18, CRL 11）
antmaze-giant-stitch     HIQL   2±2    ⭐ 全滅級無人區
humanoid-med-stitch      HIQL  88±2
humanoid-large-stitch    HIQL  28±3    ⭐ 半滅、最高維 locomotion（21-DoF）
cube-single-play         GCIQL 68±6
cube-double-play         GCIQL 40±5
scene-play               GCIQL 51±4
puzzle-3x3-play          GCIQL 95±1
```

- QRL（pointmaze 的對手）在高維 stitch 全弱（18/0/18/3）⇒ 高維對手換人：
  locomotion＝HIQL、manipulation＝GCIQL、＋TTGS 78 於 humanoidmaze-stitch
  （size 未確認、SURVEY 記錄、⛔ 引用前抽原文）。
- 甜蜜點：humanoid-large-stitch(28)＋antmaze-giant-stitch(2)＝長程拼接×高維交叉、方法主場。
- 主人 9/1 路線 antmaze→cube→機械手臂 正踩得分谷；高維缺課地圖（9/2 第一題）＝入場券。
