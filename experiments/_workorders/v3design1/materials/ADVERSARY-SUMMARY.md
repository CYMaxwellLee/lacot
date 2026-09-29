# 對抗官判決摘要（opus 5.5、9/27；lead 親驗 B1/B2）

## BLOCKER
B1: C3(u_data±0.05 去噪)+C4(保守更新) 在部署分布上的最優解=identity refine。
  閉式解: 方盒外 f*(x)=x；KD=1024 時 flow 樣本落盒機率 2.77e-6(σ=.02)~0(σ≥.05)。
  correction ratio 實測 .59-.98（σ≥.05）＝幾乎不修。lead 重跑腳本確認。
  ⇒ 修法方向：訓練輸入必須來自【flow 真實樣本與 refine 自己的迭代】，不是資料+小噪聲。
B2: 主線 trainer(scratch_lacot_rollout.py:1720-1730) inline 了 F1 病、不走 model.py 的
  losses_given（呼叫端只有 legacy 與 8/23 老實驗）。lead 親驗：inline else 分支註解早已
  診斷此病；LEARNED_REFINE 預設 1、42/64 支 slurm 顯式關 0。
  ⇒ v2 範圍必須改主線 inline 分支（或抽共用組合函式、model.py 同步），⛔ 不是只改 model.py。

## major（v2 必須處理）
- F5 語意：C4 反轉了主人設計的 consistency 用途（「NEVER used alone」）；替代案=EMA-teacher
  consistency（codebase 已有：CONS=ema、scratch:1723-1726）或 rounds≥2 規定。改語意要呈裁。
- 因果鏈第 2 環（head 能解讀非資料 latent）失去訓練訊號＝exposure-bias 橋被拆（設計文件 Q3）
  ⇒ v2 要有 exposure-bias gauge（no-grad 診斷、新 log key、⛔ 不用 0.0 哨兵）。
- M3 空殼：fake 即完整實作＝沒活；v2 的 M3 要負責 off-manifold 臂（identity refine 必須 FAIL）、
  B1 型閉式/訓練探針、R=0/1/3 scaling。
- C1 契約：dtype 規則只管 clean/sampled/noise（cond 可異 dtype）；hot path 不做 finiteness raise
  （autocast/GradScaler 相容、實測撞過）。
- R2 已合併的 tests_repair/test_model_zero_rounds.py 必須進合併 gate；合併後 rounds=0 語意寫進契約。
- mutation 漏洞六種（total 缺項/接錯 u_target/忽略 rounds/零噪聲/truncated BPTT/detach cond）
  ⇒ v2 的 integration 要能抓這六種。
- anchor-positive 標籤改名（它守的是 anchor-gradient）。
- loss 組合抽共用函式（兩 actor+主線 inline 三份重複）。
