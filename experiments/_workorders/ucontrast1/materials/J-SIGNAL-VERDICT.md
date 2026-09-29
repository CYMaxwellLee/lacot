# J-signal 判決（2026-09-28 晨收割；lead=Luna 親驗）

判準（預註冊於 j_signal_trial/README.md）：per ckpt「climb<noclimb 且 select>noclimb」⇒ J 可挑不可爬。

實測（rates.R1、每臂 200 題=5 官方 task×40 eps、兩顆獨立 ckpt；六 JSON 在 results/j_signal_trial/）：

```
seed   noclimb   climb    select
 33     .350     .050     .325
 35     .395     .030     .405
pooled  .3725    .040     .365
```

判定：**判準不成立**。
- 爬坡半邊：兩 seed 一致崩 30-36.5pp ⇒「不可爬」確鑿——GeoEnergy 梯度方向對 rollout 是毒藥。
- 挑選半邊：select vs noclimb = −2.5pp / +1.0pp（pooled −0.75pp），雜訊內 ⇒ GeoEnergy@8 挑不出好計畫。
- ⇒ J（GeoEnergy）既不能爬也不能挑。蒸餾線的「便宜事前挑選器」候選判死；按主人 9/27 預裁分支走 v3。

前提驗證（lead 三證）：
1. F6 修復 commit 0916c97（穿牆線段內插）= 9/27 14:30:37；job 33812 提交 = 15:03:10。
2. run_三臂.sbatch 內建 guard：grep 到舊式 wall_depth 碼即 exit 5——六臂全 COMPLETED = guard 全過。
3. refine_grad.py 現況無舊 pattern（grep=0）、含 wall_interp_k 線段內插。
⇒ 判決跑的是 F6 修復後的 GeoEnergy，非壞尺假陰性。

局限：200 集跨 5 個官方 task，不可當 200 獨立 task 讀；select 的 E-selection 口徑=9/2 方言役（GeoEnergy 最低、SEL_N=8、BON_N=0 隔離 GrpoReward 閘），詳 README「C 臂判準差異」段。
