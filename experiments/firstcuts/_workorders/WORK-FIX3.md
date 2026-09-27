# 工單 FIX3：修理批 3（量尺修正，三小修＋一措辭）

**在證明什麼、判準是什麼**：把三把舊尺修直，讓後續量測不再被結構性盲區誤導——
判準：每修一項附兩向驗證（修前能重現病、修後病消失且既有正確行為不變），
25 支既有 tests_repair 迴歸全綠。

## 修理項

1. **shat_probe 判準子空間化**（experiments/bcodec_shat_probe/shat_probe.py）：
   pooled 判準對 xy 結構性盲（xy 只佔 recon_s loss 0.02-0.3%）→ 輸出按 xy/yaw/
   gait 子空間分報；pooled 保留為附欄不作判準。既有 margin 0.9 與 obs-only 臂
   語義不動。
2. **ŝ 目標位移正規化**（同檔）：目標改「相對當前態位移」正規化（opus-c 候選二
   的一行改），修前後各跑一次小樣本並排存檔（這就是回填素材）。
3. **probe 權重保存**（同檔 :230 附近）：現在只存 summary → 加存 probe 權重與
   per-sample 預測（npz），讓後續免重訓重測。
4. **EXPERIMENT-INDEX 措辭修訂**（lacot/docs/EXPERIMENT-INDEX.md ⑦ 區）：五次
   decode-probe 的受測物標明「Enc(τ)（凍結 encoder 於真軌跡），非 flow 生成 u」；
   18.5% 那格補「sg_infonce 目標、低回收為預期」；回收率標尺註記（RMSE/MSE/log
   三尺擺動）。⛔ 只改措辭與標註，不改任何數字。

## 流程

luna-tier 一隻包（雜活級、主人裁的四級表）→ Claude 家族檢察一隻複核 diff（重點：
子空間切分的維度索引對、正規化只動目標不動輸入、措辭修訂沒有偷改結論方向）→
tests_repair 全綠 → commit（訊息前綴 fix(firstcuts)）。
