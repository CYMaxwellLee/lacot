# ucontrast2 回鍋單（S2 檢察 PASS-with-notes 後；lead 拍板落地）

檢察報告：experiments/_workorders/ucontrast2/VERDICT-S2.md。四咬點全 PASS、0 must-fix。
本回鍋落 lead 對 D1 的拍板＋A1/A2/A4 三個建議動作。改動全在 ucontrast2/ 檔群。

1. 【D1 拍板：判給 README】s35 臂改為【獨立粗掃】——照 ucontrast1/README.md:23-24
   規矩：s35 須另跑其 σ0 mutant、獨立 σ 粗掃（smoke 掃 3 σ 檔、ep 40-44×8 draws）、
   選 B pooled oracle@8 最大者（平手取最小 σ）。理由（寫進預註冊）：B 對照臂必須
   取對它最有利的 σ＝對照最強形式，綁 s33 的 σ 會讓判準偏鬆。工單原句「口徑與
   s33 完全同」指量測口徑（200 題×8 draws×兩臂、同 collector/analyze），不含
   校準參數共用——此句由 lead 修訂澄清。
2. 【A1】四支 sbatch：--cpus-per-task 改 8、加 --mem=16G（9/28 實測 MaxRSS
   1.22GiB 的餘量版）。
3. 【A2】--time 表頭改 12:00:00（ada-lite MaxTime、lua 會靜默夾）；ETA 註解改
   實測推算（6.9 秒/rollout：n64 每臂 ~6.2h、s35 ~3.1h、s35 粗掃另 +分鐘級）。
4. 【A4】smoke 輸出路徑改 /archive/cymaxwelllee/ucontrast1/smoke2-<arm>/（不用
   /scratch——repo 無此慣例、jasmine 上未驗）。
5. A5（worktree 絕對路徑＋建後 validate_inputs）由 lead 親手執行、不在本單。

交付：改動 diff 摘要＋改動面自驗（sbatch 語法、粗掃鏈 CPU 可驗部分）＋
自我懷疑欄；尾行 STATUS: DONE。⛔ 不碰 n64 臂已 PASS 的部分（除 sbatch 資源行）。
