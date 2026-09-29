# 在量什麼：主線訓練迴圈（主人裁的首航標的；sacct 實帳單顆 0.5–2 小時的大戶）在未來訓練主場 lady 上的 S0 端到端基線＋S1 時間分解。判準：S0＝固定輸入／seed／步數、基線 N≥5 帶方差（目標 CV<5%）、golden 輸出存檔；S1＝data／H2D／forward／backward／optimizer／eval／ckpt-IO 各段佔比、合計約 100%、卡／獨佔／重複數／方差全記錄。

2026-09-28｜F4 量測官｜**F5 預檢靶，尚未發射**。主線是 `experiments/scratch_lacot_rollout.py` 的從零訓練、官方 rollout、checkpoint 寫盤全程；不是單獨 GEMM microbench。主人指定此項為首航標的，既有 sacct 指出單顆完整訓練約 0.5–2 小時。本卡只規劃，沒有 GPU 實測值。

## 1. 場景與判準

| 欄位 | 凍結值／判準 |
| --- | --- |
| 來源配置 | `results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt`；用其**配方**從零訓練，不載入該 ckpt。檔案本地存在；它是配置參照，不是本次輸入權重。 |
| 資料 | lady `/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz`；在發射前驗存在、記 SHA256、筆數與檔案大小。此路徑沿用既有 SLURM 腳本，lady 當日是否仍有此檔尚未驗。 |
| 固定場景〔拍〕 | seed `33`，stage 1 `1500` 步、stage 2 `2000` 步，`K=8, COND=256, CHUNK=4, TCAP=128`；官方 5 task × 每 task 2 episode、只取 `R=0`，所以預期 rollout `10` 集。`STEPS2=2000` 是原 8000 的 1/4，越過 500 步 warmup 後尚有 1500 步穩態；S0 結論僅屬此縮時場景，原 8000 步總時不可直接乘四。 |
| dtype〔拍〕 | `LACOT_AMP=0, LACOT_COMPILE=0`，即現有 s33 ckpt 的預設 FP32 訓練路徑；先保存原配方數值與效能錨。Turing 2080 Ti 已有實測 `fp16 56.57 / bf16 7.38 / fp32 12.22 TFLOP/s`，bf16 比 fp16 慢 7.66×；`torch.cuda.is_bf16_supported()` 回 True 不足以選 bf16。後續若試 fp16，使用腳本已有的 `LACOT_AMP=1` autocast **及 GradScaler**，記錄 skip 數與品質差，當獨立 ablation，不混入 S0 方差。lady 只需覆核卡型、sm_75、dtype 數量級，不重跑完整 dtype 掃描。 |
| 卡數 | S0 單卡，S1 同場景也先單卡。後續多卡獨立成案，先讀 `nvidia-smi topo -m`、NUMA/PHB/SYS，再決定同 NUMA 卡組與 DDP 尺；既有 2→3 卡吞吐反降案例禁止假設八卡線性。 |
| S0 完成門檻 | 同一固定場景先完整暖機 `r00`，**丟棄其統計值**；再序列跑 `r01`–`r05` 共 `N=5`。每次外層 wall 秒，記均值、樣本 SD、CV=SD/mean，目標 CV<5%；若超標保留全部原始值、查噪聲與佔卡，不挑數重算。指定 `r01` 的 rollout JSON、checkpoint、SHA256、環境與程式版本為 golden。 |
| S1 完成門檻 | `data / H2D / forward / backward / optimizer / eval / ckpt-IO` 的 wall 秒、佔全程百分比；另列 setup/other，不把它偷塞進 GPU kernel。**所有互斥段合計約 100%**（四捨五入容差 ±1 pp）；反推全程秒數須與外層 wall 差 ≤5%，超過則分解無效。完整記 GPU UUID、卡型、獨佔狀況、CPU/NUMA、重複數、方差。 |

`STEPS1` 的 1500 步也在 E2E 之內；不能只 profile stage 2 後聲稱覆蓋全程。`LACOT_EVAL_RS=0` 故 golden 是固定的 R0 觀測，並非原 s33 完整品質判決。種子固定只控制已實作 RNG，CUDA 算子是否逐位元重現待驗；golden 要存原始輸出，不能預設 5 份 ckpt hash 相同。

## 2. lady 卡位與 F5 預檢

- **時段〔拍〕**：過 F5 預檢後，選 lady 上其他工作穩定佔用且不搶本卡的連續 8 小時窗口；本單序列跑一個 job，目標物理 **GPU index 2、僅一張**。今天已有別人的 job 佔 1 卡，該卡由即時 PID/UUID 核對後保留給對方；本單不疊卡。若 index 2 被佔、Slurm 實際配到別的卡、或蒸餾 fine-tune 改在同卡點火，本單不啟動訓練，先重排卡位與時段。
- **硬體帳矛盾**：題面稱 lady `8×RTX 2080 Ti`；`luna/cluster-inventory/inventory.db` 的 2026-08-13 快照卻記 **3×RTX 2080 Ti、每卡 11 GiB**。這份快照不能證明 9/28 現況。發射前用 lady `nvidia-smi -L`、`nvidia-smi topo -m`、Slurm GRES 實帳覆核實卡數、index↔UUID、NUMA；差異記入 manifest。此卡不以「八張空卡」安排任何 job。
- **雙量閘**：排程者先 `squeue -w lady`；上機取得卡後查 `nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv`、`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv`，再對目標卡採 `nvidia-smi dmon -i 2 -s u -c 10`。目標卡須無他人 PID、VRAM 僅驅動底量（暫訂 <512 MiB）、10 秒 SM 利用率中位 <5%；任何一格不符就停，不以單次 0% 當空卡。記錄其他工作的卡與時段，S0 六跑之間重查；CPU/IO 爭用也記錄。
- **唯讀查詢現況**：本規劃環境的 `squeue/sinfo` 連 Slurm controller 被 sandbox 拒絕，未取得 9/28 即時佔用、Slurm GRES/卡號映射；因此 GPU 2 是明示目標，**不是已驗可用卡**。lady `torch 2.6.0+cu124 cuda=True` 已由先遣驗過，但此卡仍檢查該 venv、資料與 `torch.cuda.get_device_capability()`。

## 3. F5 發射單（**執行段才複製執行**）

- **在證明什麼**：同配方、同 seed、同資料與單卡隔離下，主線短時 E2E 的可靠 wall 基線；接著在同一場景對 wall 做 S1 分解。先 profile 與診斷，暫不改模型、batch、dtype、worker 或優化旋鈕。
- **提交機／執行機**：從可連 Slurm 的登入機提交；`--nodelist=lady -p admin -A it -q great-mage --gres=gpu:1 --cpus-per-task=8`，job 實際跑 lady。Slurm 是否給物理 GPU 2 必須以 `SLURM_JOB_GPUS`、UUID 與進程佔卡三者核對；不符停機。`admin` 參數沿用 repo 的既有 lady 訓練腳本，執行前確認帳號權限。
- **完整提交命令〔拍〕**（以下是待執行文字，**本段未提交**）：

```bash
cd /home/cymaxwelllee/Projects/lacot
mkdir -p results/opt1/s0/logs
sbatch --partition=admin --account=it --qos=great-mage --nodelist=lady \
  --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 \
  --job-name=opt1-s0 --output=results/opt1/s0/logs/s0-%j.out <<'BATCH'
#!/bin/bash
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
# 清掉提交殼殘留的 LACOT_*；下面 env 指令列出本場景全部顯式設定。
while IFS='=' read -r KEY _; do
  case "$KEY" in LACOT_*) unset "$KEY" ;; esac
done < <(env)
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
DATA=/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz
ROOT=results/opt1/s0
test -x "$PY"
test -f "$DATA"
test "${SLURM_JOB_GPUS:-}" = 2 || { echo "GPU 2 未獲分配：${SLURM_JOB_GPUS:-unset}" >&2; exit 20; }
"$PY" -B -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
nvidia-smi -L > "$ROOT/gpu-list.txt"
nvidia-smi topo -m > "$ROOT/topo.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$ROOT/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$ROOT/gpu-processes-before.csv"
nvidia-smi dmon -i 2 -s u -c 10 > "$ROOT/gpu2-dmon-before.txt"
GPU_UUID=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
test -n "$GPU_UUID"
if grep -Fq "$GPU_UUID" "$ROOT/gpu-processes-before.csv"; then
  echo 'GPU 2 有其他 compute PID，停止' >&2; exit 21
fi
nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits | \
  awk '{if ($1 >= 512) exit 1}' || { echo 'GPU 2 VRAM 超過 512 MiB，停止' >&2; exit 22; }
awk '$1 == 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 >= 5) busy++}
     END {exit !(n == 10 && busy <= 4)}' "$ROOT/gpu2-dmon-before.txt" || \
  { echo 'GPU 2 SM 採樣未過，停止' >&2; exit 23; }
sha256sum "$DATA" > "$ROOT/dataset.sha256"
git rev-parse HEAD > "$ROOT/git-head.txt"
for R in 00 01 02 03 04 05; do
  OUT="$ROOT/r$R"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$ROOT/gpu-processes-pre-r$R.csv"
  if grep -Fq "$GPU_UUID" "$ROOT/gpu-processes-pre-r$R.csv"; then
    echo "r$R 前 GPU 2 有其他 compute PID，停止" >&2; exit 24
  fi
  nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits | \
    awk '{if ($1 >= 512) exit 1}' || { echo "r$R 前 VRAM 未過，停止" >&2; exit 25; }
  nvidia-smi dmon -i 2 -s u -c 10 > "$ROOT/gpu2-dmon-pre-r$R.txt"
  awk '$1 == 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 >= 5) busy++}
       END {exit !(n == 10 && busy <= 4)}' "$ROOT/gpu2-dmon-pre-r$R.txt" || \
    { echo "r$R 前 SM 採樣未過，停止" >&2; exit 26; }
  mkdir -p "$OUT"
  env OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    LACOT_ENV=pointmaze-large-stitch-v0 LACOT_SEED=33 LACOT_DATA_SEED=-1 \
    LACOT_CONS=self LACOT_K=8 LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128 \
    LACOT_STEPS1=1500 LACOT_STEPS2=2000 LACOT_ENC_OBJ=recon_ictr \
    LACOT_TEACHER_MIX=0.5 LACOT_EMA_W=0.999 LACOT_WARMUP=500 \
    LACOT_DEC_START=soft LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1 \
    LACOT_AMP=0 LACOT_COMPILE=0 LACOT_LOAD_CKPT= LACOT_CONT_TRAIN=0 LACOT_S1_FROM= \
    LACOT_GRPO_W=0 LACOT_DEV_EVAL=0 LACOT_PREREQ=0 LACOT_FLOW_PROBE=0 \
    LACOT_DIAG_DUMP=0 LACOT_DIAG_TRAIN=0 LACOT_BON_N=0 \
    LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_LOG_EVERY=1000 \
    LACOT_OUT_DIR="$OUT" \
    /usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S maxrss_kb=%M' -o "$OUT/time.txt" \
    "$PY" -u experiments/scratch_lacot_rollout.py > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  sha256sum "$OUT"/rollout_*.json "$OUT"/ckpt_*.pt > "$OUT/artifacts.sha256"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$OUT/gpu-after.csv"
done
BATCH
```

**發射前人工閘**：批次檔內已含 PID／VRAM／10 秒 SM 的自動阻擋，且每跑前複查；F5 還須先確認 `SLURM_JOB_GPUS=2` 的站點語義及 index↔UUID 映射，核對其他工作與本單時段沒有同卡重疊。若索引映射不同，以實際 UUID 為準修訂提交命令。這兩格是預檢 blocker，不能用「大概空」放行。

- **輸出路徑與 schema**：`results/opt1/s0/r{00..05}/` 每跑各一個 `rollout_large-stitch_self_K8_c256_ch4_st2000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.json`、同 tag 的 `ckpt_*.pt`、`time.txt`、`stdout.log`、`stderr.log`、`artifacts.sha256`、`gpu-after.csv`。JSON 至少驗 `env,seed,K,cond,chunk,steps2,tcap,episodes,rates,enc_obj,teacher_mix,ema_w,learned_refine,cond_drop,bc_indep`；`episodes=10`、`steps2=2000`；ckpt `cfg` 至少驗 `SEED=33, STEPS2=2000, ENC_OBJ=recon_ictr, EMA_W=0.999` 且有 `ema`。`r01` 為 golden；完整檔案與 hash 原地凍結。`r00` 保留作暖機紀錄但不進統計。
- **預期計數**：六跑＝`6 JSON + 6 PT + 6 time.txt + 6 hash 清單`；統計有效 `N=5`；預期每 JSON 10 episode、R0 欄。若任何跑中止、輸出少一份或 step/cfg 不符，該次標失敗並保留；重新補一個全新 run ID，不覆寫。
- **ETA〔拍〕**：依原 8000 步 sacct 的 0.5–2 小時粗估，2000 步訓練本體約 7.5–30 分，加 stage 1、環境初始化與 10 集 rollout/IO；保守抓 **20–60 分/跑，六跑 2–6 小時**。8 小時 wall limit 是暫定，實際以 `r00` 的 `time.txt` 外推；若預估超限，先停機改時間額度，不讓後五跑缺樣本。此估算尚未量 lady 本次 eval/IO，不能當實測。
- **寫入面**：只在執行段寫 `results/opt1/s0/`（六個獨立 run 目錄、log、GPU/拓樸/資料 hash/版本快照）；主線腳本因 `LACOT_OUT_DIR` 所寫 JSON/PT 也只落該處。沒有覆蓋既有 `results/` ckpt。若執行器需要另存 S1 profiler trace，限定 `results/opt1/s1/`。本規劃段只寫本卡。
- **依賴**：原 repo commit SHA、lady 本地 CUDA venv `/archive/cymaxwelllee/LaCoT/.venv/bin/python`、資料 NPZ、`ogbench`/MuJoCo `osmesa`、可用的 `/usr/bin/time`、Slurm admin 權限與 **獨佔目標卡**。先遣只驗過 torch `2.6.0+cu124 cuda=True`，不等於資料/卡位/全腳本均已驗。

上方先清除繼承的 `LACOT_*`，再逐項顯式設定本場景的 `LACOT_*`；未列的旋鈕一律採所記 Git SHA 的腳本預設。保留 Slurm 提供的 `CUDA_VISIBLE_DEVICES` 與 CUDA/venv 系統環境，避免繞開排程的 GPU 隔離。執行段另將非敏感的 CUDA/SLURM 環境鍵記入 summary，不把完整殼環境（可能含 token）散落到結果檔。

## 4. 執行序、golden 與 S1 宏觀分解

1. F5 預檢：讀 `optimize/hardware/nvidia.md` 與 `ACTIVE-INSIGHTS` 既有 Turing 數字、inventory；執行段再核 topology、即時卡數與雙量；清點 `results/opt1/s0` 不能有同名既有檔。記 torch/CUDA、GPU UUID、driver、CPU/NUMA、Git SHA、資料 hash。S0 只用 GPU 2 一張；六次**序列**跑，不併發，且每次開始前核對獨佔。避免同節點其他工作動態搶 SM/IO；若期間變動，標註受污染區間。
2. `r00` 全程暖機，保留所有輸出但丟棄 wall 統計；`r01`–`r05` 原序執行。每次是新 Python process，起始條件一致；報告外層 E2E wall 與每步均值時明確分開。收割時產 `results/opt1/s0/summary.json`〔執行段〕：`scenario, node, gpu_index, gpu_uuid, exclusive, seed, steps1, steps2, episodes, dtype, git_sha, data_sha256, runs[{id,wall_s,artifact_hashes,status}], n, mean_s, sd_s, cv, golden_run, notes`。`summary.json` 的方差只算 r01–r05，SD 用樣本分母 `N−1`。
3. golden 保存 r01 的原始 rollout JSON、ckpt、hash 清單、時間與 stdout；驗證 JSON 鍵與 ckpt cfg/EMA、訓練日誌 1500+2000 步走完、rollout 10 集。另記 stage 1 recon/effective dim、stage 2 loss 的已印點，供之後品質比對；不同 dtype 的精度改動須走品質 gate，不能拿 bit-exact 模板直接宣稱等價。
4. **S1 工具〔拍〕：先 PyTorch `torch.profiler`**（CPU + CUDA activities、schedule `wait=2,warmup=2,active=20,repeat=3`，只取代表性 stage 1、stage 2 穩態窗口；`record_function` 標注 data/H2D/forward/backward/optimizer）。這使用應用同版 CUDA ABI，先得 operator/kernel 排名與 CPU↔GPU 空檔；避免全程 trace 龐大且 profiler 自身開銷污染 S0。主線目前沒有這些互斥 region marker，執行段須先在隔離副本或獨立受控 instrumentation 加上標記並做數值/流程對照，**本卡不改 repo**。若 torch profiler 看不清 CUDA 排隊/NUMA 傳輸，再用 `nsys` 單次短窗口定位；版本相容性先驗證，絕不以 nsys 全程 trace 當 S0 wall。
5. **S1 wall 帳**：另用外層 `/usr/bin/time` 包完整同場景；在 instrumented run 用 `perf_counter` + 必要 CUDA sync/CUDA event 給互斥區間計時，stage 1 與 stage 2 各自計數，再按實際步數 `1500/2000` 加權。`data`=batch 建立/CPU 前處理，`H2D`=明確拷貝，`forward`=含 loss，`backward`=反傳，`optimizer`=zero/clip/step/GradScaler/EMA；`eval`=環境建立與 10 集 rollout，`ckpt-IO`=序列化＋寫盤（必要時分出 JSON IO），`setup/other`=imports、資料讀取、模型建置及未歸屬成本。同步點開銷另報；上述互斥 wall 合計對外層 wall，不能把各 kernel duration 直接相加當百分比。S1 至少 3 次同卡短窗口/完整段複量，報各段均值、樣本 SD/CV、profiler overhead（與未 profile S0 同場景比較）；若 S1 改變程式或量測場景，先標差異，絕不把它的 wall 混進 S0 N=5。

## 5. 自我懷疑／未驗格

- **卡數/卡位**：題面 8 卡與 inventory 3 卡矛盾；目前環境連不上 Slurm controller，也看不到 lady 即時 PID、VRAM、SM、topology。GPU 2 的獨佔性與 `SLURM_JOB_GPUS` 映射未證；這是發射前阻擋項。
- **時間與資源**：2000 步、兩集 eval 的 wall、11 GiB VRAM 是否足夠，以及 8 小時 job limit/32 GiB RAM 是否合適，都只是估。`r00` 先校 ETA，OOM 或超限時另立新場景，不能私自縮 batch/步數混入既定 N=5。
- **代表性**：縮短 8000→2000 改變訓練後段權重與 eval 難度；`R=0` 也縮了 eval 工作。這把尺主要測主線運算組成；若要推原 8000 步 E2E 或全 R rollout，需另用實帳校準，不拿此值直接替代。
- **數值**：FP32 S0 是歷史配方的錨，尚未驗 5 次是否 bitwise 相同；fp16 的 GradScaler skip、品質與 Turing matmul 數量級只在執行段覆核。資料檔實際 hash、任務數 5 與輸出 tag 依現有腳本推導，仍須上機驗。
- **profiling**：現有腳本沒有足以閉合七段 wall 的內建標記；執行段的受控 instrumentation 與 profiler overhead 尚未驗。若區間合計無法對齊全程，先修量測，不進優化結論。

方法與輪子：`experiments/_workorders/opt1/KO-INJECT-0928.md` 主題③；`/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/.claude/skills/optimize/` 的 `workflow.md`（8 步法）、`diagnose.md`、`levers.md`、`harness/ablation_template.py`、`harness/bit_exact_check.py`。Step 0 已讀 `hardware/nvidia.md`；該 skill 沒有 LaCoT 專屬 architecture 文件，不能假裝已存在。Turing 實測來自 luna `ACTIVE-INSIGHTS.md:37-42`，NVIDIA 筆記目前尚無 Turing 實測列。完成 S1 診斷、進入 ablation/品質 gate 後才用 harness；首航結果日後應回填 GPU/NVIDIA case study。

STATUS: DONE
