# NOTE — 姿態感知選字驗證：nn_state（教材接力最近鄰距離度量消融，2026-09-08）

_對應主人裁示：既有 `nn`（閉環最近鄰，只用 xy 位置找參考點）在 humanoid 上比純開環
`schedule` 更差（見 `docs/NOTE-2026-09-08-humanoid-relay.md` §四），推測病因是「選字不看
姿態，遞給 decoder 姿態不相容的字」。本次把最近鄰距離度量換成「位置＋joint 姿態（完整
RL state）」的正規化歐氏距離，量能救多少。全部新程式與結果都在各自的
`teacher_relay/`（檔名帶 `_nnstate`）；⛔ 沒有修改任何既有檔案、沒有訓練任何模型、
沒有 git commit。_

---

## 〇、一句話結論

**姿態感知選字對 humanoid 有實質幫助、對 ant 沒有幫助；而且「(a) 全 state 含位置」
全面優於或持平「(b) 只姿態排除位置」，跟工單原本『(b) 可能更對症』的假設方向相反。**
humanoid G1K32 上 `nn_state_full` 把 per-leg 從 nn 的 .048（比 schedule 差 61%）救到
.155（反超 schedule 27%、是 nn 的 3.2×）；G16K16 上救到 nn 的 1.3×（.029→.039）但仍
低於 schedule。Ant 上 nn 本來就沒有輸 schedule（.555 vs .554 打平），`nn_state_full`
持平（.553），`nn_state_pose`（排除 root xy）反而更差（.470，−15%）。三個測試點
（ant／G16K16／G1K32）裡 (a) 全 state 版都 ≥ (b) 只姿態版，沒有一個是 (b) 贏。副作用：
兩個 nn_state 版本的「回頭跳字比例」都比原版 nn 高 5～12 倍（16～36% vs 3～6%），但這
個「跳更多」不必然是壞事——humanoid 上跳更多反而換到更高到達率。

---

## 一、方法與設計決策

**沿用（未改）**：`experiments/walk_verify/teacher_relay/run_teacher_relay.py` 與
`experiments/humanoid_dict/teacher_relay/run_relay.py` 的 `build_tasks()`（題目構造：
200 條 val episode、沿真實 xy 路徑每 `DELTA_SUB=7.5` 取路標、`rho=1.875`）、`schedule`／
`nn` 兩版的執行邏輯逐行未動、`COMPARE`／`COMPARE_ANT` 對照常數、字典 ckpt、量尺包。
新檔只 `import` 這些既有模組與函式，不修改一行。

**新增（本次設計，工單要求先想清楚，理由如下）**：

1. **距離度量**：`nn_state_*` 把「找同一條 episode 的 chunk 起點格點裡最近的一點」
   從 xy 歐氏距離換成正規化 state 向量的歐氏距離。state 向量＝decoder 本來就在用的
   `obs`（ant 29 維＝`concat(qpos[:15],qvel[:14])`；humanoid 69 維＝
   `concat(xy(2),joint_angles(21),head_height(1),extremities(12),torso_vert(3),
   com_vel(3),qvel(27))`，一手核對 `ogbench/locomaze/{ant,humanoid}.py` 的 `get_ob`）。
2. **normalize 用的 mean/std**：直接重用字典 ckpt 內建的統計（ant：
   `model.obs_mean`/`obs_std` buffer；humanoid：`hr.load_dict()` 回傳的 `obs_mu`/
   `obs_sd`）——這是「該環境資料的 mean/std」最直接可得的來源，且讓 nn_state 搜尋
   用的正規化空間跟 decoder 內部 `norm_obs()` 完全一致。已用
   `obs_mean[:2]`（ant=9.434,10.237；humanoid=9.606,10.262）核對跟
   `MAZE_CENTER`（ant=9.42,10.24；humanoid=10.00,10.01）同量級，確認索引 0,1 就是
   root xy，佐證這份統計、這個索引判斷都正確。⚠️ 這份統計是訓練時對 **train split**
   算的，本次搜尋用在 **val split** 的距離計算上，沒有對 val 重新算一份——設計選擇，
   見 §八#1。
3. **子版 (a) `nn_state_full`**：正規化 state 全 29／69 維都算進距離（含 root xy）。
   **子版 (b) `nn_state_pose`**：排除索引 `(0,1)`（root xy）之後剩下 27／67 維——
   保留軀幹高度／朝向（quaternion 或 torso_vertical_orientation）、全部關節角、
   全部速度（qvel，humanoid 另含 head_height/extremities/com_vel），因為這些才是
   「這個參考點的動作字適不適合套用到當下身體姿態」的姿態訊號；只拿掉「在迷宮
   哪裡」這一項純位置資訊。兩版都跑，才看得出「排除位置」這個動作本身有沒有差異
   （見 §六）。
4. **參考點的 state 向量**：直接切 `raw["observations"][s0+grid_local]`（資料集自己
   存的 obs），不重新呼叫模擬取值——`wv_common`/`hr_common` 的文件與既有程式已驗證
   這逐位元對應 `sim_obs()` 在歷史時刻的值，也是訓練時 decoder 條件輸入的同一個
   取法（`wv_common.build_seg_tensors` 用 `obs[seg_starts]`），不是新發明的捷徑。
5. **schedule／nn 一併重跑（不是只跑新版）**：目的是同一次執行裡拿到「重現性檢查」
   ——用同一份 `build_tasks(seed=20260908, n_traj=200)` 重新跑一次 schedule／nn，
   跟 `docs/NOTE-2026-09-08-{teacher-relay,humanoid-relay}.md` 的歷史數字逐一核對。
   結果：**三個測試點的 schedule/nn 差距全部在 ±0.00005 以內**（ant: −0.0004/−0.0000；
   G16K16: −0.00002/−0.00003；G1K32: +0.00002/−0.00002）——證明新腳本忠實重現既有
   方法論，沒有在搬過程中不小心改動任何既有邏輯，本次的 nn_state 數字可信。

---

## 二、結果全表：ant（`antmaze-medium-stitch-v0`，200 題，seed=20260908）

一手 log：`experiments/walk_verify/teacher_relay/logs/run_nnstate.log`；
json：`experiments/walk_verify/teacher_relay/results/teacher_relay_nnstate_summary.json`。

```
關                              schedule      nn            nn_state_full  nn_state_pose
────────────────────────────────────────────────────────────────────────────────────────
per-leg 不限時到達率（官方口徑）  0.554         0.555         0.553          0.470
  legs reached/attempted        217/392       212/382       213/385        162/345
限時 1.25N/1.5N (N_data)        41.7%/45.7%   30.1%/34.0%   30.8%/33.8%    23.5%/28.8%
限時 1.25N/1.5N (N_table)       13.0%/20.2%   6.4%/10.9%    8.9%/16.3%     6.9%/13.4%
完整走完比例                    0.125         0.150         0.140          0.085
步速 p50 / mean                 0.1033/0.1075 0.0783/0.0869 0.1006/0.1062  0.1000/0.1048
  （真螞蟻 p50/mean 參考值＝0.132/0.131）
翻倒率（逐步／逐集）             0.03%/3.50%   0.02%/2.00%   0.01%/2.00%    0.02%/2.50%
平滑度 adiff p50                1.236         1.210         1.265          1.251
選字 active/perplexity          32/31.09      32/29.56      32/30.45       32/30.44
nn 回頭跳字比例                 —             6.0%(n=9395)  36.2%(n=9369)  35.1%(n=9530)
重現性差（vs 歷史 .554/.555）    −0.0004       −0.0000       —              —
```

對照行（COMPARE，一手引用自 `NOTE-2026-09-08-walk-verify.md` §3.4）：
`貪心=0.063  主頭=0.320  lo頭=0.124`。四版全部遠贏貪心／主頭，這不是本次新發現
（跟既有 NOTE 結論一致），本次的焦點是四版**互相**怎麼比。

---

## 三、結果全表：humanoid（`humanoidmaze-medium-stitch-v0`，200 題，seed=20260908，兩顆字典）

一手 log：`logs/relay_nnstate_{G16K16,G1K32}.log`；
json：`results/relay_nnstate_{G16K16,G1K32}_summary.json`。

```
G16K16                          schedule      nn            nn_state_full  nn_state_pose
────────────────────────────────────────────────────────────────────────────────────────
per-leg 不限時到達率             0.0524        0.0293        0.0386         0.0386
  legs reached/attempted        11/210        6/205         8/207          8/207
限時 1.25N/1.5N (N_data)        4.8%/4.9%     2.9%/3.0%     3.9%/3.9%      3.9%/3.9%
限時 1.25N/1.5N (N_table)       2.9%/3.3%     2.4%/2.4%     2.4%/2.9%      2.4%/2.9%
完整走完比例                    0.005         0.005         0.005          0.005
步速 p50                        0.0078        0.0017        0.0071         0.0069
翻倒率（逐步／逐集）             60.04%/100.0% 49.30%/75.50% 55.65%/94.50%  54.63%/95.50%
平滑度 adiff p50                1.579         1.476         1.542          1.535
選字 active_min/perplex_mean    16/15.31      16/14.68      16/14.95       16/14.94
nn 回頭跳字比例                 —             3.0%(n=19713) 21.0%(n=19738) 20.2%(n=19738)
重現性差（vs 歷史 .0524/.0293）  −0.00002      −0.00003      —              —

G1K32                           schedule      nn            nn_state_full  nn_state_pose
────────────────────────────────────────────────────────────────────────────────────────
per-leg 不限時到達率             0.1216        0.0481        0.1552         0.1161
  legs reached/attempted        27/222        10/208        36/232         26/224
限時 1.25N/1.5N (N_data)        8.8%/8.9%     4.8%/4.9%     10.4%/11.0%    8.3%/9.3%
限時 1.25N/1.5N (N_table)       2.3%/4.1%     2.4%/2.4%     2.3%/4.1%      2.7%/3.7%
完整走完比例                    0.025         0.010         0.020          0.010
步速 p50                        0.0140        0.0017        0.0100         0.0106
翻倒率（逐步／逐集）             23.52%/98.00% 37.28%/79.00% 26.85%/93.50%  24.72%/93.50%
平滑度 adiff p50                0.759         0.726         0.766          0.768
選字 active_min/perplex_mean    32/31.12      32/29.66      32/30.66       32/30.50
nn 回頭跳字比例                 —             3.1%(n=19623) 16.3%(n=19564) 16.4%(n=19628)
重現性差（vs 歷史 .1216/.0481）  +0.00002      −0.00002      —              —
```

真 humanoid 步速參考值 p50=0.0478（`ruler_pack.json`）——四版都遠低於真值（跟既有
NOTE 一致，humanoid 接力本來就慢，不是本次新發現）。三個測試點（ant/G16K16/G1K32）
的字典選字使用率在四版裡都健康（active 全滿、perplexity 接近滿分），**排除了「nn_state
版數字變化是靠字典塌陷取巧」的可能**——差異確實來自參考點怎麼選，不是字典退化。

---

## 四、nn 回頭跳字比例：三個測試點一致的側面現象

```
測試點      nn(xy)   nn_state_full   nn_state_pose
ant         6.0%     36.2%           35.1%
G16K16      3.0%     21.0%           20.2%
G1K32       3.1%     16.3%           16.4%
```

**兩個 nn_state 版本的回頭跳字比例都比原版 nn 高 5～12 倍**，三個測試點方向一致。
推論（⚠️ 未獨立量測機制，合理但未證實）：姿態／速度這類維度在一個 episode 內會
隨步態週期反覆震盪（同一個關節角度、朝向組合在一集裡出現不只一次），而 xy 位置
在一集內大致單調累積前進；把姿態也算進距離，會讓搜尋偶爾被「姿態相似但其實是
本集中另一個時間點」的候選點吸走，造成回頭跳字比例大增。**但這個「跳更多」不必然
是壞事**——G1K32 上 nn_state_full 回頭跳字比原版 nn 高 5×，per-leg 到達率反而是
nn 的 3.2×；ant 上 nn_state_full 回頭跳字比 nn 高 6×，per-leg 到達率卻只是打平沒有
變差；只有 ant 的 nn_state_pose 是「跳更多且到達率更差」兩者同時發生。這代表「跳字
頻率」本身不是判斷好壞的可靠指標，要看跳到的點是不是真的姿態相容（本次沒有再深挖
「跳到的點姿態相容度」這一層，見 §八#3）。

---

## 五、gif（4 支，成功/失敗各一 × ant/humanoid）

選 `nn_state_pose` 出 gif（不是數字上更好的 `nn_state_full`）——理由：`nn_state_pose`
是工單原本假說（「排除位置、只看姿態才對症」）直接對應的那個子版，人眼檢視它的實際
選字行為最能回答「這個假說是否成立」這個問題本身；`nn_state_full` 的數字已經在
§二/§三列出，不需要靠 gif 佐證。跟既有 NOTE 慣例一樣，humanoid 只用推薦格 G16K16。

```
ant:      nn_state_pose_success_ep423.gif   38 幀   完整走完 2/2 legs
          nn_state_pose_fail_ep319.gif      50 幀   卡在 0/3 legs（跑滿預算）
humanoid: nn_state_pose_success_ep273.gif   38 幀   完整走完 1/1 leg
          nn_state_pose_fail_ep319.gif      100幀   卡在 0/2 legs（跑滿預算）
```

存放：`experiments/walk_verify/teacher_relay/results/gifs/`、
`experiments/humanoid_dict/teacher_relay/results/gifs/`，4 支合計 4.1MB。格式同既有
gif（左＝跟拍看步態、右＝全迷宮固定機位、橘點＝當下路標）。渲染時已用
`assert r["completed"]==... and r["n_steps_run"]==...` 核對重跑跟量測那次逐位元一致
（同既有慣例）。**⚠️ 我自己沒有逐幀播放看完 4 支**——只確認檔案成功產生、幀數 >0；
人眼關要主人自己看過才算數（同 ant/humanoid 既有 NOTE 慣例）。

---

## 六、判讀

**證據鏈**：

1. §二/§三：**humanoid 上姿態感知選字有實質幫助，尤其 G1K32**——`nn_state_full`
   把 per-leg 從 nn 的 .0481 救到 .1552（3.2×），還反超 schedule 的 .1216（+27%）；
   G16K16 救到 nn 的 1.32×（.0293→.0386），但仍低於 schedule 的 .0524（−26%）。
   **ant 上沒有幫助**——ant 的 nn 本來就沒有輸 schedule（§〇既有結論：.555 vs
   .554 打平），`nn_state_full` 持平在 .553，`nn_state_pose` 反而退步到 .470（−15%
   vs nn）。
2. 這個「因環境而異」的結果，跟工單背景描述的病因假說（「選字不看姿態，遞給
   decoder 姿態不相容的字」）方向一致但程度不同：humanoid 的既有 NOTE 已指出
   humanoid 是雙足步態、對相位連續性更敏感（`docs/NOTE-2026-09-08-humanoid-relay.md`
   §四第3點的推論），本次量到的結果支持這個方向——姿態訊號對 humanoid 這種平衡
   敏感的軀體確實有用。ant 是四足、既有 P1/P2 分析顯示 ant 對「哪個字」本來就沒
   那麼挑剔（.554/.555 打平代表選字方式在 ant 上影響有限），姿態訊號加進去自然
   看不到提升，符合「病因程度因環境而異」的方向。
3. §三/§四：**排除「字典塌陷」跟「量測管道錯誤」兩種混淆因子**——三個測試點的
   選字使用率在四版裡都健康（active 全滿），且 schedule/nn 的重現性檢查全部在
   ±0.00005 內，代表 nn_state 版本的數字差異確實反映「距離度量換了」這個唯一變因，
   不是程式碼路徑或字典退化造成的假訊號。

**回答工單的問題——「姿態感知選字能不能改善接力」**：**能，但只在 humanoid 上
明顯成立，且只有 (a) 全 state（含位置）版本才有效**；ant 上這個修正沒有用（甚至
(b) 版本讓 ant 變差）。這代表「選字不看姿態」在 ant 上很可能**不是**當初 nn≈schedule
打平現象的解釋（因為 ant 上 nn 本來就沒有輸），而在 humanoid 上「選字不看姿態」
（更精確地說：只看 xy 位置）確實是 nn 比 schedule 差的部分成因——姿態感知選字
把這個差距顯著縮小（G1K32）甚至扭轉（G1K32 反超 schedule）。

---

## 七、(a) 全 state 含位置 vs (b) 只姿態排除位置：誰好？

```
測試點      (a) nn_state_full   (b) nn_state_pose   誰好
ant         0.553               0.470               (a) 明顯贏（+0.083）
G16K16      0.0386              0.0386              打平
G1K32       0.1552              0.1161              (a) 明顯贏（+0.039）
```

**(a) 全 state 含位置在三個測試點裡都 ≥ (b) 只姿態版，沒有一個測試點是 (b) 贏**——
跟工單原本「(b) 排除位置可能更對症，因為位置像但姿態不像正是要避免的」這個假說
方向**相反**。推論（⚠️ 未獨立量測機制，合理但未證實）：完全拿掉位置後，搜尋在
「哪個時間點姿態最像」上可能找到一個位置上離得很遠、純粹因為步態相位巧合而相似
的參考點，這樣選出來的字雖然姿態相容，卻可能把 ant/humanoid 導向錯誤的方向或
距離尺度，反而干擾接力；保留位置（即使跟其他維度一起正規化、只占 29/69 或
69/69 維中的 2 維）似乎足以避免搜尋跑到「時間上很遠但姿態剛好像」的錯誤候選點，
同時仍然受益於其餘 27/67 維帶來的姿態資訊。**結論：如果要在這兩個子版中選一個
繼續用，選 (a) 全 state（含位置），不要排除位置。**

---

## 八、沒做到／不確定清單

1. **normalize 用的 mean/std 是 ckpt 內建的 train-split 統計，沒有對 val split
   重新計算一份**——這是本檔案 §一#2 明講的設計決策，不是遺漏；沒有做「val 重新
   估計 mean/std 是否會讓 nn_state 數字不同」這個對照實驗。
2. **`nn_state_full`／`nn_state_pose` 比 `nn`（xy-only）在 humanoid 上更好、在 ant
   上打平或更差的機制沒有查到底**——§六第2點的解讀（雙足對相位更敏感、四足對
   選字方式沒那麼挑剔）是**推論**，不是逐項排除法驗證過的結論；也沒有比較「nn
   選中的參考點」跟「nn_state 選中的參考點」在同一個 chunk 上具體差在哪裡（例如
   時間距離、姿態夾角）。
3. **回頭跳字比例大增（16～36% vs 3～6%）背後「跳到的點姿態相容度」沒有拆解**——
   只量了跳字的方向性（往前/往後），沒有量「跳去的點的姿態/state 距離」分佈，
   無法直接驗證 §四提出的「步態週期性讓姿態搜尋偶爾跳到同集中另一個時間點」這個
   推論。
4. **只驗證了 antmaze-medium 與 humanoidmaze-medium 的 stitch-v0 val split，200 題
   一組**——跟兩份既有 NOTE 同款保留項，沒有測 large/giant、沒有測多個隨機 seed
   的 run-to-run variance（本流程如既有 NOTE 所述是全確定性的，200 題內部沒有隨機
   性，但抽哪 200 題本身只試了 `seed=20260908` 一種）。
5. **只有 1 個字典 ckpt/seed**（ant 沿用 `p0_dict_v1_L4K32_50k.pt`；humanoid 沿用
   `G16_K16.pt`／`G1_K32.pt`）——跟既有 NOTE 同一個開放項，未量字典本身的
   run-to-run variance。
6. **gif 只對 `nn_state_pose` 出**（§五已講理由），`nn_state_full`（三個測試點裡
   數字上更好的子版）**沒有對應的 gif 可以人眼檢視**——如果之後要更進一步確認
   `nn_state_full` 的實際步態行為，需要另外跑 `--gif-versions nn_state_full`（腳本
   已支援，未執行，避免超出「每個環境 2 支 gif」的預算）。
7. **`nn_state_pose` 在 humanoid 兩顆字典上跟 `nn_state_full` 差距不大**
   （G16K16 打平、G1K32 差距 0.039，遠小於 ant 上的差距 0.083）——沒有進一步拆解
   為什麼「排除位置」在 ant 上代價明顯大於 humanoid，只在 §七標出現象，未深挖
   原因。
8. **開發期 smoke test 殘留檔案**（`results/smoke_nnstate*`、
   `results/smoke_nnstate_G16K16*`，皆用 5 題跑的小樣本驗證程式正確性）
   照「不刪任何東西」規則沒有清掉，留在各自 `results/` 目錄下，檔名前綴跟正式
   產出（`teacher_relay_nnstate_*`／`relay_nnstate_{G16K16,G1K32}_*`）清楚分開，
   如實記在這裡而非默默留著。
9. 沒做到的「查不到／跑不出」項目：無。ant 四版 200 題、humanoid 兩顆字典×四版
   各 200 題、四項度量（到達率含限時/步速/摔倒率/回頭跳字）、4 支 gif、重現性
   核對全部跑完並留下一手 log（各自 `logs/` 目錄）。⛔ 全程 CPU-only，未訓練任何
   模型，未修改任何既有檔案，未 git commit。
