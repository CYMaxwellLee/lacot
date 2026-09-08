# NOTE — humanoid「照譜跳」驗證：段獨立重放＋教材接力（2026-09-08）

_對應主人裁示：把 ant 上已完成的步法字典驗證鏈（P1 段獨立重放、teacher-relay 真字串接力）
搬到 humanoid。開工前讀過的資產：`docs/NOTE-2026-09-08-humanoid-dict.md`（humanoid 字典
偵察，推薦格 G16K16、量尺包已建）、`experiments/humanoid_dict/`（26 顆 ckpt）、
`experiments/walk_verify/teacher_relay/run_teacher_relay.py`（ant 版協定範本）、
`docs/NOTE-2026-09-08-teacher-relay.md` 與 `docs/NOTE-2026-09-08-walk-verify.md` §二（P1
協定細節）。全部新程式與結果都在 `experiments/humanoid_dict/teacher_relay/`（新目錄）；
⛔ 沒有修改任何既有檔案，⛔ 沒有訓練任何模型，⛔ 沒有 git commit。_

---

## 〇、一句話結論

**humanoid 教材在「單段、給真狀態」時能被準確複述（block 1 過），但一旦連續執行、狀態
自己往前滾動（block 2「照譜跳」），兩顆字典都幾乎每一集都會跌倒，per-leg 到達率
0.029～0.122，只有 ant 同協定（0.554／0.555）的 5～22%。** 判讀：humanoid 字典的教材
**還不到能直接拿去訓 M2 選字頭的程度**——問題不在字典會不會編碼（block 1 顯示字典跟亂字
分得很開），而在**接力執行對狀態漂移的容忍度太低**，這跟 ant（同一套協定跑出 .554）形成
清楚對比。詳見 §七。

---

## 一、資產與環境（跟工單一致，逐項核過）

- venv：`/home/cymaxwelllee/venvs/lacot-rocm/bin/python`（torch 2.9.1+rocm6.3、
  mujoco 3.12.0、numpy 2.5.2）
- `OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data`，`humanoidmaze-medium-stitch-v0`
  train（5000 ep×401 步）／val（500 ep×401 步，一手驗證 `qpos/actions/observations` 維度
  28/21/69）皆已在位。`MUJOCO_GL=osmesa`，CPU-only，全程未碰 GPU、未用 slurm。
- 兩顆字典（`experiments/humanoid_dict/ckpt/`，使魔已訓好、本次只載入不訓練）：
  `G16_K16.pt`（推薦格，capacity=64 bits）、`G1_K32.pt`（單碼本對照，capacity=5 bits，
  跟 ant 的單碼本 K=32 架構同形狀）。兩者皆 seg_len=4。
- 量尺包：`experiments/humanoid_dict/results/ruler_pack.json`（沿用既有，未重建）——
  fall_line_p1=0.1582、goal_tol=0.5、真實步速 mean/p50=0.0478、真實軀幹 z p50=1.117。

---

## 二、DELTA_SUB／rho 要不要重新校：查了，結論是不用（沿用 ant 數值 7.5／1.875）

工單要求「尺度明顯不同就重新校，並標理由與數值」。查證鏈（一手，程式碼 grep 過）：

1. `ogbench/locomaze/maze.py` 的 `MazeEnv.__init__` 對 `maze_unit` 的預設值是 `4.0`。
2. `ant.py`／`humanoid.py` 兩份檔案都 grep 不到 `maze_unit` 字樣 ⇒ 都沒覆寫，都吃預設值。
3. `'medium'` 迷宮佈局字串在 `maze.py` 只定義一份，ant／humanoid 共用同一個
   `maze_type='medium'` 分支，不是各自一份。
4. 實測佐證：從 `humanoidmaze-medium-stitch-v0-val.npz` 的 qpos xy 全範圍量出
   `MAZE_CENTER=(10.00, 10.01)`（x −1.93~21.93、y −1.90~21.92），跟 ant 對應值
   `(9.42, 10.24)`（`p2_analyze_traces.py:37`）量級幾乎一樣。

⇒ humanoidmaze-medium 跟 antmaze-medium 是**同一個迷宮、同一個座標尺度**，只是站在裡面
走的軀體不同。`DELTA_SUB=7.5`（沿路標弧長，原始 xy 單位）代表的仍是同一段物理迷宮距離，
不需要因為「humanoid 比較慢」重新校——humanoid 慢的是**時間尺度**（量尺包：步速 0.048
vs ant 0.131，只有 37%），不是**空間尺度**，時間尺度的差異已經由 N_data／N_table 兩把
限時尺各自處理（humanoid 自己的 distance→steps 表本身就量得比 ant 慢，d=2 時 p50=43 步
vs ant 15 步）。**本次沿用 ant 數值：`DELTA_SUB=7.5`、`rho=1.875`（=0.25×DELTA_SUB），
未改動任何數字。**

副作用（如實記，不是漏做）：humanoid 一個 episode 的平均總弧長比 ant 短（步速慢 37%、
episode 只長 2 倍，0.37×2=0.74 倍），200 條題目量出的路標數 `M`：**min=1, p50=2, max=3,
總計 412 leg**（ant 同款是 min=1, p50=3, max=4, 總計 595 leg）——每題路標數略少，符合
預期方向，不是異常。

---

## 三、Block 1｜段獨立重放（三組對照 × 兩顆字典，1000 視窗，H=80，seed=20260908）

協定：對每個視窗每個 4 步 chunk 都 `set_state` 重置到真起點，跟 ant P1 的
`dict_indep`／`true_chunkcmp`／`dict_randcode` 三組同款（用 ant 的原名以便對照）。
一手 log：`logs/seg_G1K32.log`、`logs/seg_G16K16.log`；json：
`results/seg_{G1K32,G16K16}_summary.json`。

**末端誤差（第 gs 步時累積合成位移離「真人形同一時刻位置」的距離，公尺；gs=40 對齊 ant
報告用的那一欄）：**

```
                    gs=8    gs=12   gs=20   gs=40   ← 對照：ant gs=40
G1K32  真(true_chunkcmp)   0.0000  0.0000  0.0000  0.0000        0.121
       字典(dict_indep)    0.0134  0.0169  0.0222  0.0330        0.199
       亂字(dict_randcode) 0.0258  0.0333  0.0454  0.0655        0.655
G16K16 真(true_chunkcmp)   0.0000  0.0000  0.0000  0.0000        0.121
       字典(dict_indep)    0.0103  0.0128  0.0169  0.0250        0.199
       亂字(dict_randcode) 0.0399  0.0502  0.0672  0.0985        0.655
```

**每 chunk 位移誤差（公尺，僅 dict 組；真對照組定義上恆為極小值不列）：**

```
G1K32   dict_indep  mean 0.0094  p50 0.0079    dict_randcode  mean 0.0169  p50 0.0134
G16K16  dict_indep  mean 0.0071  p50 0.0061    dict_randcode  mean 0.0242  p50 0.0203
```

**判讀**：

1. **「真」（true_chunkcmp）在 humanoid 上末端誤差幾乎恆為 0**（精確值 ~1e-7~1e-5，非
   顯示位數的錯覺，見 §八 #1），**跟 ant 的 0.121 明顯不同**——ant 那邊 0.121 來自
   float32 儲存 qpos/qvel 起手誤差被 8 個關節、高頻接觸的螞蟻步態放大；humanoid 在
   同一個「每 4 步重置一次」的協定下，這個誤差沒有被放大到看得見的程度。**這代表兩件
   事**：(a) 這個結果不是 bug——已用完整 float32 精度值核對過（§八 #1），且
   `hr_common.sim_obs` 已對過資料集的 `observations` 逐位元一致（差在 float32 round-trip
   量級 ~1e-7）；(b) 但這也代表**「dict_indep 絕對誤差比 ant 小」不能直接讀成「humanoid
   字典比 ant 字典準」**——humanoid 這邊的天花板（真動作重放）本身就幾乎零誤差，兩邊
   起跑點不同，公平的比法是下面的「字典/亂字」比值。
2. **字典 vs 亂字的比值（分辨力）：G16K16＝3.9×（0.0985/0.0250）、G1K32＝2.0×
   （0.0655/0.0330），對照 ant＝3.3×（0.655/0.199）。** G16K16 在這把「調過起手誤差」
   的尺上跟 ant 同量級甚至略高，G1K32 明顯較弱——**G16K16 在 block 1 是比 G1K32 更好
   的字典**（絕對誤差更低：0.025 vs 0.033；分辨力更強：3.9× vs 2.0×）。
3. 兩顆字典的 `fall(win)`（1000 視窗裡任一步跌破 fall_line 的視窗比例）都落在
   8.0~8.7%，三組（真/字典/亂字）幾乎相同——這是資料本身「蹲/跌」regime 的基底比例
   （`docs/NOTE-2026-09-08-humanoid-dict.md` 已指出這批資料軀幹 z 左偏），不是字典造成的
   額外跌倒，符合預期（跟 block 2 的高跌倒率成對比，見 §四）。
4. 選字使用率兩顆都健康：G1K32 32/32 全活、perplexity 31.1~32.0；G16K16 每組（16 組）
   最少也全活（`codes_active_min_over_groups`=16=K），perplexity 均值 15.3~16.0（滿分
   16）——不是靠塌陷取巧達到的低誤差。

---

## 四、Block 2｜教材接力（schedule／nn 兩版 × 兩顆字典，200 題，seed=20260908）

協定逐項照抄 ant `run_teacher_relay.py`：`schedule`＝純開環照這條 episode 自己的
hindsight 真字播；`nn`＝每 chunk 先看模擬中人形當下 xy，在同一條 episode 的 chunk
格點裡找最近點、播那一點的真字；兩版解碼都用**模擬中人形的當下 obs**（連續執行，不重
置）。一手 log：`logs/relay_{G1K32,G16K16}.log`；json：
`results/relay_{G1K32,G16K16}_summary.json`。

**四關全表：**

```
關                                  G1K32-schedule  G1K32-nn   G16K16-schedule  G16K16-nn   │  ant 對照
──────────────────────────────────────────────────────────────────────────────────────────┤
① per-leg 不限時到達率              0.1216          0.0481     0.0524           0.0293      │  0.554/0.555
   legs reached/attempted           27/222          10/208     11/210           6/205       │  613/9761(P2)
① per-leg 限時1.25N/1.5N(N_data)    8.8%/8.9%       4.8%/4.9%  4.8%/4.9%        2.9%/3.0%    │  （ant 未印per-leg限時,見附註）
① per-leg 限時1.25N/1.5N(N_table)   2.3%/4.1%       2.4%/2.4%  2.9%/3.3%        2.4%/2.4%    │
   完整走完比例                     0.025 (5/200)   0.010      0.005            0.005        │  0.125/0.150
② 步速 p50                          0.0140          0.0017     0.0078           0.0017       │  真humanoid p50=0.0478
③ 跌倒率：逐步／逐集(fall_task)     23.5%／98.0%    37.3%／79.0%  60.0%／100.0% 49.3%／75.5% │  真資料逐步線定義=1.00%
④ 平滑度adiff p50                   0.759           0.726      1.579            1.476       │  真humanoid p50=2.165
   選字使用率(active_min/perplex)   32/31.1         32/29.7    16/15.3          16/14.7      │
   nn回頭跳字比例                   —               3.1%       —                3.0%        │
```

（ant 對照行取自 `docs/NOTE-2026-09-08-teacher-relay.md` §二，一手引用非本次量測；
ant 版沒有另外印 per-leg 限時到達率，故該格留白。）

**判讀（跟 ant 的關鍵差異）：**

1. **per-leg 到達率全面遠低於 ant**：最好的一格（G1K32-schedule 0.122）也只有 ant 的
   22%；最差一格（G16K16-nn 0.029）只有 ant 的 5%。**四格沒有一格接近 ant 的 .554/.555**。
2. **跌倒率是最大的質變**：block 1（chunk 獨立重置）的跌倒視窗比例只有 ~8%，block 2
   （連續執行）的**逐集跌倒比例飆到 75.5%~100.0%**——四種設定裡有一種（G16K16-schedule）
   是**全部 200 集都至少跌倒一次**。這代表「接上真實接力、讓狀態自己往前滾動」這件事
   本身，對 humanoid 是壓力測試的主因，跟 block 1 顯示的「字典本身能準確複述單段」
   形成明顯反差——問題出在**連續執行下的狀態漂移**，不是字典的表達力（跟 ant 的
   walk-verify 判讀「壞的不是字典，是選字/執行方式」同方向，但 humanoid 這裡崩得
   嚴重很多）。
3. **`nn`（閉環修正）在 humanoid 上反而比 `schedule`（純開環）更差**（G1K32：0.048 <
   0.122；G16K16：0.029 < 0.052）——**跟 ant 的「schedule≈nn」（.554 vs .555，幾乎打平）
   方向不同**。推論（⚠️ 未獨立量測機制，是合理但未證實的解讀）：`nn` 只用 xy 位置找
   最近的參考點，不管當下人形的姿態／朝向／平衡相位是否跟那個參考點吻合；一旦 xy
   位置漂移，`nn` 可能會挑到一個「位置近但姿態脈絡完全不同」的字（例如不同跨步相位），
   對需要連續相位才能維持平衡的雙足步態是更大的干擾，反而不如 `schedule` 至少保持
   時間上連貫的字序列。回頭跳字比例（3.0~3.1%）本身不高，不足以完整解釋差距——跟
   ant note 當時「schedule/nn 打平、機制沒查到底」的誠實態度一致，這裡也沒有深挖到底。
4. **選字使用率健康**（跟 block 1 一致，兩顆字典在 block 2 也都全活、perplexity 接近
   滿分）⇒ **排除「字典塌陷」當低分的解釋**，問題確實在執行協定，不在字典本身塌縮。

### 補充診斷：到達路標時是不是已經跌倒了？（`check_reach_posture.py`，非主流程但直接回答一個關鍵疑點）

抽 gif 檢視時（見 §六）發現 `schedule_success_ep273`／`nn_fail_ep319` 的畫面都出現人形
倒地，一度懷疑「到達」是跌倒後滑到 rho 半徑內湊到的，不是真的走到。**寫了一支獨立診斷
腳本、對全部 200 題重跑 `run_one`、在每個 leg 被判定到達的那個 global_step 精確記錄軀幹
z**（跟主流程完全同一份函式，決定性重跑，一手 log：`logs/check_reach_posture.log`）：

```
                z_at_reach: mean / p10 / p50 / p90   到達時 z<fall_line 的比例
G1K32-schedule   0.925 / 0.340 / 1.011 / 1.493        0/27  = 0.0%
G1K32-nn         1.080 / 0.227 / 1.374 / 1.499        0/10  = 0.0%
G16K16-schedule  0.873 / 0.098 / 1.314 / 1.497        2/11  = 18.2%
G16K16-nn        1.312 / 0.945 / 1.493 / 1.499        0/6   = 0.0%
（fall_line=0.158，真資料軀幹 z p50=1.117，供對照）
```

**修正我自己看 gif 得到的第一印象**：系統性量出來，**到達路標的當下絕大多數（4 組裡
3 組是 100%、G16K16-schedule 是 81.8%）人形其實還站著**（z 中位數 1.0~1.5，接近真實站姿
1.117），**不是普遍靠跌倒滑進去湊到的**——`schedule_success_ep273` 恰好落在
G16K16-schedule 那 18.2%「到達時已跌倒」的少數案例裡，不能代表整體。⇒ 正確的讀法是：
**到達率低（0.03~0.12）主因是「大多數情況下根本沒能往下一個路標前進」，跌倒是接力過程
中另一個常發生但跟「到達那一刻」不完全綁定的現象**（一個任務裡可能先跌倒、之後仍在地上
挪動或重新站起、也可能到達之後才跌倒）——本次沒有進一步拆解「跌倒發生在到達前/後/
從未跌倒」三者的時間順序分布，如實記在 §八。

---

## 五、跟 ant 一句話對照表（判準要求的格式）

```
                    真/天花板    字典        亂字對照      判讀
ant   block1(gs=40)   0.121      0.199       0.655        字典比亂字準3.3×
huma  block1(gs=40)   ~0.000     0.025(G16K16)/0.033(G1K32)  0.099/0.066   字典比亂字準3.9×/2.0×

                    schedule    nn          判讀
ant   block2(P-leg)   0.554      0.555       教材能走到、值得訓M2
huma  block2(P-leg)   0.052~0.122 0.029~0.048  教材在單段可信、接力走不遠，暫不建議直接訓M2
```

---

## 六、視覺化：4 支 gif（`experiments/humanoid_dict/teacher_relay/results/gifs/`，來自 G16K16）

```
schedule_success_ep273.gif   50 幀 (200KB×~5≈916KB)  完整走完 1/1 leg（200 步）
schedule_fail_ep319.gif      100幀 (~1.5MB)          卡在 0/2 legs（跑滿 400 步預算）
nn_success_ep273.gif         13 幀 (~220KB)          完整走完 1/1 leg（只用 52 步，比 schedule 快很多）
nn_fail_ep319.gif            100幀 (~1.2MB)          卡在 0/2 legs（跑滿 400 步預算）
gif 合計 3.7MB（遠低於 100MB 上限）
```

格式同 ant：每支兩格並排（左＝跟拍看步態、右＝全迷宮固定機位看有沒有前進；橘點＝當下
路標）。**只用 G16K16（推薦格）出 gif**——工單「3｜視覺化」寫的是「4 支 gif」（不是
「兩顆字典各 4 支」），block 1/2 的數字兩顆字典都已各自跑完，gif 選跑起來更完整的推薦格
即可，這裡明講理由而非默默決定。跟拍鏡頭參數（lookat z=0.9, distance=5.0, elevation=-20,
azimuth=135）依 `hd_common.py` 既有 render 函式的取景經驗值調整（humanoid 站得比 ant 高，
ant 原始參數 lookat z=0.3 會拍到腳邊）；全迷宮鏡頭沿用 ant 數值不變（§二已證明同一個
maze 座標尺度）。

**我自己抽幀看過**：`schedule_success_ep273` 的首/中/末三幀＋`nn_fail_ep319` 中段一幀
（共 4 幀，另存於 `/tmp/frame_*.png`，非本任務目錄、僅供本次核對用）。首幀為站姿（雙臂
上舉，跟資料集常見起始姿態一致），中/末幀已倒地——後來用 §四補充診斷證實這支剛好是
G16K16-schedule 那 18.2%「到達時已跌倒」的案例，不是全體代表，如實記。`schedule_fail`／
`nn_fail` 兩支沒有逐幀看完，只確認「檔案成功產生、n_frames>0」。⚠️ 人眼關本身要主人自己
看過 4 支才算數（同 ant note 慣例）。

---

## 七、G16K16 vs G1K32：誰好？（沒有單一贏家，依用途分）

- **block 1（單段、給真狀態）：G16K16 更好**——絕對誤差更低（0.025 vs 0.033）、字典/
  亂字分辨力更強（3.9× vs 2.0×），跟 humanoid-dict 偵察報告「G16K16 是膝點推薦格」的
  結論一致。
- **block 2（連續接力）：G1K32 更好**——per-leg 到達率兩版都贏（schedule 0.122 vs
  0.052、nn 0.048 vs 0.029），逐集跌倒率也略低（98%/79% vs 100%/75.5%，惟 nn 版兩者
  相近）。推論（⚠️ 未證實）：G1K32 只有 1 組 8 維連續 latent 量化，比 G16K16 的 16 組
  獨立量化更不容易在「當下 obs 已經漂移」時產生訓練時沒見過的（obs, code）組合，
  對接力這種持續 OOD 壓力測試更穩健。
- ⇒ **兩者都不是「哪個能讓 humanoid 教材接力可用」的答案**——block 2 的絕對數字（兩顆
  都遠低於 ant）才是主要限制，dictionary 選型只影響「差多少」不影響「過不過關」。

---

## 八、沒做到／不確定清單

1. **「真」（true_chunkcmp）末端誤差幾乎恆為 0，跟 ant 的 0.121 形成明顯反差**——已用
   完整精度數值核對非顯示位數錯覺（gs=40 時 p95 量級 ~1e-5~1e-6，見
   `results/seg_*_summary.json` 的 `end_err_q`），也用獨立腳本核對過
   `hr_common.sim_obs()` 在 `set_state` 後／`env.step` 後都跟資料集 `observations` 逐位元
   一致（差在 float32 round-trip 量級 ~1e-7~1e-5）。**但沒有像 ant 當時寫
   `p1_diag_replay_fidelity.py` 那樣做一支專門的「float32 round-trip vs warmstart vs
   base」三變體診斷**——本次的解讀（humanoid 資料集主要落在較穩定的站姿/慢速移動
   regime，短程 4 步內動力學沒有把 float32 起手誤差放大到看得見的程度）是**推論**，
   不是逐項排除法驗證過的結論。
2. **只有 1 個字典 seed**（沿用既有 26 格掃描各自的訓練 seed，未量 run-to-run
   variance）；本次的 200 題／1000 視窗抽樣序列本身是決定性的（固定 `seed=20260908`），
   但字典本身沒有多 seed 對照，跟 ant note 同款保留。
3. **DELTA_SUB／rho 沿用 ant 數值的判斷基於「maze_unit 相同＋佈局字串同源」的間接證據
   ＋ MAZE_CENTER 量出來的量級吻合**，不是直接比對兩份 maze XML 或逐格 BFS 路徑長度
   算出來的直接證據——判斷有依據但不是最強形式的驗證，如實記。
4. **`nn` 版本比 `schedule` 更差的機制沒有查到底**（§四第 3 點）：只量了「回頭跳字比例」
   一項（3.0~3.1%，判斷不足以完整解釋），沒有做「nn 改成只准往前搜尋」的對照實驗，也
   沒有比較 nn 選中的參考點姿態（不只是 xy）跟當下人形姿態的差異——工單範圍是量測，
   不包含這一層機制歸因。
5. **「跌倒」與「到達」兩個事件在時間軸上的先後關係沒有拆解**（§四補充診斷已指出多數
   到達發生在未跌倒時，但沒有進一步統計「先跌倒後到達」「先到達後跌倒」「全程未跌倒」
   三種時間序列各佔多少比例）。
6. **人眼關（4 支 gif）本次只抽看 4 幀（1 支的首/中/末＋另 1 支的中段各一幀）**，其餘
   frames 與另外 2 支 gif（`schedule_fail`／`nn_fail`）沒有逐幀看完，只確認「檔案成功
   產生」——同 ant note 當時的驗證量級，人眼關要主人自己看過才算數。
7. **只在 humanoidmaze-medium-stitch-v0 單一資料集驗證**，沒有測 large/giant 或
   navigate 變體。
8. **`check_reach_posture.py` 是本次任務期間額外寫的補充診斷腳本**，不在原始工單列的
   兩支主腳本（`run_segment_replay.py`／`run_relay.py`）之內，但因為它是回答一個在
   gif 抽查時冒出來的具體疑點所需、且輸出直接進了本報告的判讀，一併留在
   `experiments/humanoid_dict/teacher_relay/` 目錄下並在此點名，不是隱藏的臨時檔。
9. **`results/smoke_*`、`results/smoke_relay_*` 是開發期用小樣本（20 視窗／10 題）驗證
   程式正確性留下的殘留檔案**（tag 前綴跟正式產出 `seg_*`／`relay_*` 清楚分開），照「不
   刪任何東西」規則沒有清掉，留在 `results/` 目錄下，如實記在這裡而非默默留著。
10. 沒做到的「查不到／跑不出」項目：無。兩顆字典、block 1（1000 視窗×3組）、block 2
    （200 題×2版）、4 支 gif、跌倒時機補充診斷全部跑完並留下一手 log（`logs/` 目錄）。
    ⛔ 全程 CPU-only，未訓練任何模型，未修改任何既有檔案，未 git commit。
