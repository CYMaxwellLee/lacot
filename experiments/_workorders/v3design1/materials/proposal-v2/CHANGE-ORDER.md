# v1 → v2 變更單（呈 lead／主人）

u 編碼想像軌跡。本單申請的是設計契約變更，不代主人裁案。觸發：對抗 B1/B2 由 lead 親驗；主人已選「現在就救 learned refine」。範圍是主線修法提案與 CPU 研究證據，未改唯讀 codebase。

| 契約/範圍 | v1 | v2 提議 | blast radius／重考 |
|---|---|---|---|
| C3 訓練分布 | data±.05 | real flow + own iterates；jitter 只補 flow 周圍 | M1 全重作；B1、off-manifold、R sweep 新增，v1 paired denoise PASS 作廢 |
| C3 目標 | denoise 到 clean | frozen decoder 的 same-plan quality，每輪監督；同 sample optimized endpoint 可輔助 | 新 M4、teacher validity；不可再獨立資料動作配對 |
| C4 F5 | 反向「保守更新」 | **呈裁 EMA（建議）vs self/R>=2**，撤回默改 | F5 stale-teacher R1/2/3 重考；原非固定點梯度 probe 構想沿用 |
| C5/Q3 | head 只訓 clean，sample log 填0 | qualified same-plan action teacher 訓 head exposure，no-grad gap 真量 | anchor-gradient 改為扣除 exposure 後等式；缺 gauge 用 null+invalid，不當0 |
| C1 | cond 同 dtype、finite 即 raise | latent dtype 同；cond 可異；autocast 中間可降精；nonfinite 交 scaler | M1/M2 CPU bf16、nonfinite propagation、skip/EMA 重考 |
| C7 主入口 | 只修兩 actor | 共用 compose + 主線 inline 必改，M2 唯一 scratch owner | 三入口 wiring 全重考；BC mask/COND_DROP/FSQ/intent 舊語意需保 |
| C7 R0 | 留給修理艦，不入新 contract | nf+anchor，全部 refine loss 0，不除0，不需服務 | R2 原測試納必過；加 no-sample/refine 負呼叫測試 |
| C7 相容性 | training API/weights 不變 | 舊位置参数保持，新 training services；R>0 缺 oracle 清楚報錯；EMA/provider metadata checkpoint | 舊 inference 權重相容，續訓明確 migration；image/ant 需各自合格 oracle，不能假装通用 |
| C9/M3 | metrics fake 幾乎等於工作 | 真 CPU train、B1、off-manifold、scaling、exposure、six mutants、baselines | M3 全考場重做；v1 factorial 路線 oracle、常數/swap 控制沿用 |
| 分工 | 3模組 | 4模組，增加 oracle | 4份快照／四個搬移考場重跑；不超過4 |

沿用件：五件套格式、versioned contracts + neighbors + stub + local exam 骨架、v1 的固定條件×資料路線×sample 路線因子設計、route metrics 程式（保存在 contracts/metrics.py）、anchor-gradient 原理、非固定點 F5 真 RefineOperator probe、唯讀 hash/原始 stdout/exit 舉證方式。沿用的是工具與判準，不沿用 v1 的科學 PASS 結論。

**值得的理由**：只改 pairing 不能避開 B1 的 identity 最优；只修 model.py 根本碰不到 B2 的主線。增加 oracle 與 head exposure 成本，是為了讓「會改善」和「改善後能執行」都有對該樣本有效的梯度及外部量尺。不是用更多 loss 掩飾目標無效。代價是 decode 每輪開銷、teacher 校準、EMA checkpoint，以及 quality proxy 被鑽漏洞的風險；因此 BoN/GD、外部成功率和 teacher validity 是必交，而不是選做。

toy 補充變更：第一次 300-step/raw features、第二次600-step/raw features 都有 seed1 失敗。最終 toy 加 sign feedback 可通過，**未申請直接把 sign 或任意新 features 塞進 production**；這只支持 LePlanner 式回饋可行。若現有 RefineOperator 真 CPU／實際資料仍學不會，下一圈應呈另一張網路輸入/寬度改動单。真RefineOperator額外700步×3 seed診斷也有seed0 FAIL（詳EVIDENCE）；不能把toy feedback成功推成既有operator成功。所有嘗試檔附 evidence/try-*，沒有提高門檻容忍度來洗 PASS。

待裁項：① 主案 LePlanner-style vs ReflexFlow/ DMD 備案；② EMA vs self/R>=2；③ 先救有合格 decoder/geometry 的 pointmaze，其他 actor 接同函式但 oracle 缺席禁止 R>0 訓練；④ 新 same-plan action teacher 的資料/校準成本。已授權工作全部做到可審閱；本包不啟動 worker、不部署、不發送他人訊息。
