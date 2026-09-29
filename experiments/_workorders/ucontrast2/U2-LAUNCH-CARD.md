# ucontrast2 發射單 v2（F5；u 線兩補充臂＝n64 折衷＋s35 重現）

日期 2026-09-29。主人 18:02 裁「艦隊可以準備發射」。⓪官五查 4 PASS＋選機
jasmine GO（out-device-check-u2.md）；lead 補驗全過〔實測 9/29 晚〕：
ada-lite MaxTime=12:00:00（scontrol）、jasmine 3×4060Ti util 全 0／0 MiB、
/archive 5.2T 餘、五個 OUT 目錄全不存在（零舊產物）、frozen worktree
validate_inputs()=READY（provenance_sha=dd759e3feb49f51c、
flagoff_sha=b4f6f5cab7afd1da、jasmine 親跑）。

_v1→v2（①預檢 NO-GO 五必改全修、U2-PREFLIGHT-REPORT.md）：select 命令改
jasmine 一行完整版（--outdir 即 calibration root——select branch 不用
--calibration-root、v1 寫法會找錯地方）；計數表改 README 真義（n_tasks=
(task,episode) 題對數）；鏈序聲明降級為「⑤官人肉序列 enforce」；六支 sbatch
加 PYTHONDONTWRITEBYTECODE=1（封 NFS home 的 __pycache__ 寫入）；ckpt/seed
映射明寫入契約。_

## 在證明什麼

臂甲（n64）：u 的 oracle 優勢是否隨 draw 數放大過門檻（A−B≥+5pp 且 McNemar
p<.05、50 題×64 draws）。臂乙（s35）：9/28 u 判決是否在第二顆凍結 ckpt 重現
（200 題×8 draws、方向與解鎖結構照實記、無通過門檻）。預註冊判準全文=
WORKORDER.md（單一事實源、⛔ 不重抄）。

## 跑哪台＋部署

- jasmine（⓪官 GO；其他節點可見性未證=NO-GO）。
- 全部 sbatch `cd $HOME/Projects/lacot-u-frozen`（凍結 worktree；主樹碼會被
  hash gate 拒=設計）。
- venv=/archive/cymaxwelllee/LaCoT/.venv；資料=OGBENCH_DATA_DIR=/archive/
  cymaxwelllee/data/ogbench；s33/s35 ckpt SHA 由 run.py gate 釘（README）。
- **ckpt/seed 映射（契約、run.py:178-179 固定映射）**：n64 臂=s33 ckpt／
  seed=33（u 判決原 ckpt 的 N=64 放大）；s35 臂=s35 ckpt／seed=35。
- 六支 sbatch 均 export PYTHONDONTWRITEBYTECODE=1（含 child subprocess）——
  frozen worktree（NFS home）零 bytecode 寫入。

## 指令全文（六支 sbatch＝experiments/_workorders/ucontrast2/*.sbatch）

從 zeldajr 提交（sbatch 檔即指令全文、逐字不改）：

```
鏈 A（s35 側、序執行）:
  1. sbatch s35-mutant.sbatch          # σ0 mutant 單格（校準）
  2. sbatch s35-sweep.sbatch           # array 0-3：A σ0＋B σ=.05/.10/.20
  3. CPU select（只在 jasmine 跑；venv 在該台 /archive）、一行完整命令:
     ssh jasmine 'cd "$HOME/Projects/lacot-u-frozen" && /archive/cymaxwelllee/LaCoT/.venv/bin/python -B experiments/_workorders/ucontrast2/run.py s35 --s35-stage select --outdir /archive/cymaxwelllee/ucontrast1/s35-calibration'
     （⚠️ select branch 的 --outdir 就是 calibration root、不用
     --calibration-root——run.py:262-276 契約。）
  4. sbatch s35-smoke.sbatch           # array 0-1 兩臂、ep 40-44
  5. （smoke 收割過）sbatch s35.sbatch  # 正式 200 題×8 draws×兩臂
鏈 B（n64 側、序執行；與鏈 A 並行）:
  1. sbatch n64-smoke.sbatch           # array 0-1 兩臂、ep 40-44
  2. （smoke 收割過）sbatch n64.sbatch  # 正式 50 題×64 draws×兩臂
```

**鏈序 enforce 聲明（誠實版）**：code 對「前置產物缺席」fail-closed（sweep 驗
mutant gate、select 驗四 verified、s35 formal/smoke-B 讀 selection），但
⛔ **不 enforce 提交順序**（無 --dependency；s35 smoke-A 不驗 selection、
formal 不驗 smoke receipt）。順序由 lead／⑤官人肉序列 enforce：**每步收割過
才發下一步**（⑤官逐字執行單上順序）。已知盲點：s35 smoke-A 的 PASS 不證
selection 鏈；亂序提交會產生 stranded partial OUT（exist_ok=False 拒重跑、
需人工 quarantine）——按序即避免。

## 預期輸出＋schema＋計數

- 正式：`/archive/cymaxwelllee/ucontrast1/n64/`、`…/s35/`；校準
  `…/s35-calibration/`；smoke `…/smoke2-n64/`、`…/smoke2-s35/`；log
  `…/logs/`（既存目錄）。schema 照 ucontrast1/README（provenance＋結果
  JSON 末三欄 `n_tasks,n_draws,n_success`；exclusive write、既有檔=拒 started）。
- 計數（README 真義：n_tasks=(task,episode) 題對數、n_draws=raw draw rows
  總數=n_tasks×draws_per_task；run.py:234-238 硬驗）：

```
stage                n_tasks  draws_per_task  n_draws
n64 正式（每臂）        50         64           3200
s35 正式（每臂）       200          8           1600
n64 smoke（每臂）       25          2             50
s35 smoke（每臂）       25          8            200
s35 mutant/sweep 每格   25          8            200
```

  50 題單=n64-questions.json（seed=20260929、逐題入結果 JSON）。

## 寫入面清單（全 jasmine 本機 /archive）

上列六個目錄＋logs/，別無其他。⛔ 不碰凍結資產（s33/s35 ckpt 唯讀）、不寫
NFS home、不碰 exp1/refine_v3 產物。

## ETA×死點

〔實測推算 6.9s/rollout、9/28 主跑；①官修訂帳〕mutant ~23 分；sweep 四格三卡
=兩波 ~46 分（⛔ 不是「四格同時 23 分」）；n64 smoke ~5.8 分/臂、s35 smoke
~23 分/臂；s35 正式 ~3.07h/臂、n64 正式 ~6.13h/臂（各 array 兩臂並行、
jasmine 3 卡夠）。全鏈過夜完。--time=12h 餘量：s35 ~3.9×、n64 ~1.96×。
mem：16G host RAM vs MaxRSS 1.22G（1600 rows）~13×；n64 3200 rows 線性外推
~2.4G 仍 ~6.6×——但 n64/derived/GPU 峰值無實測、OOM 不能寫已排除。
**死點（①官點名）**：result 只在 rollout loop 全完才寫（collector 尾寫）——
逾時/OOM/搶佔=只剩 preflight/partial OUT、無 checkpoint 續跑、reserve 拒同
arm 重跑 ⇒ 失敗=換新 OUT＋人工調查，⛔ 不自動重提原 arm。

## smoke 蓋不到的面（誠實聲明、①請咬）

- n64 smoke 不走 rollout_n64.py 的 64-draw 衍生路徑（smoke=25 題對×2 draws
  縮版、questions=None 走原 frozen rollout）＝正式 n64 是 derived 路徑首次
  GPU 執行（⓪官點名、①官評「聲明誠實、緩解部分足夠、殘餘風險中高」）。
  緩解：rollout_n64.py 驗原 source PIN＋三唯一替換錨＋DERIVED_PIN；formal
  完成後 run.py:234-259 重驗題數/draw shape/題單/schema；CPU 側已驗 50 題
  抽樣重現性（sol 交付兩態證據）。收割判讀時列為正式首航風險、⛔ 不把
  n64-smoke PASS 升格成全路徑 PASS。
- s35 正式 B 臂的 σ 由 select 產物餵入——smoke 跑在 select 之後、同鏈序即
  蓋到讀取路徑。

## 依賴

frozen worktree（READY 親驗）／s33+s35 ckpt（gate 釘 SHA）／venv torch（exp1
慣例同套）／OGBENCH 資料（9/28 主跑同源）。

## 發射權

smoke＋校準（分鐘級、寫 archive 專屬新目錄）=⑤發射官代發；正式兩臂（3.1h／
6.2h 長跑）=呈主人摘要過目後發（lead 9/29 承諾）。⑤發射官管制五道照
FLEET-LAUNCH（單張授權、配額=本單七次提交、逐字執行、ETA×2、記 LEDGER）。
