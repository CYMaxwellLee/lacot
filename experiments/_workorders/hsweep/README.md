這張單只交「能正確產生 H 掃描訓練資料的程式」。證明：新增的開關 `LACOT_TEACHER_H` 不設時，訓練資料與 M9 **逐位元相同**；設成 H 時，teacher 樣本的軌跡＝原最短路「沿弧長前 4H 原始單位」那一段、**終點條件 g 仍是整條路的終點**，而且亂數消耗與不設時完全相同。判準＝§6 的六個自測（T1-T6）全過，而且兩個殺手輸入（T1、T3）都真的 FAIL 過。

回鍋單 `hsweep-trainer-v1-r1` 已完成：T1 等價及亂數殺手沿用上一輪原文，不重跑；本輪 T2–T6 全過，新增的 `4H+2` 殺手及 T3 近端 g 殺手均確實 FAIL。只修改上一輪自測與交件文件，`train_hsweep.py` 保持逐位元不變；M9 也保持 SHA256 不變；本任務未寫入同目錄另單四檔。唯讀核對發現 `harness_sg.py` 與 `selfsub_probe_ckpt.py` 期間有外部變更，前後 hash 記於自測原文末段；保留其目前內容。未上 GPU、送 Slurm 或正式訓練；T6 僅 CPU stage1／stage2 各兩步，到存 ckpt 所需的原腳本短 rollout。五份交件檔案均在本目錄；`/tmp/hsweep-selftest-f9o48f9q`、`/tmp/hsweep-cpu-smoke-8sni4ssk` 產物保留、不刪除。

來源是 `experiments/_workorders/ucontrast1/smoke/mutants/M9/scratch_lacot_rollout.py`，SHA256 為 `276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc`。上一輪核對見 [selftest-output.txt:56](selftest-output.txt#L56)，本輪送件核對見該檔末段。訓練程式以 M9 為底，並非 s33 的 9/2 主檔；正式掃描的 H 未設對照仍需另行訓練，本單不執行。

開關及行為：

| 開關 | 定義 |
| --- | --- |
| `LACOT_TEACHER_H` 未設 | 不截斷，保留 M9 題庫、取樣、內插及 batch 資料輸出。 |
| `LACOT_TEACHER_H=H` | 正整數；一格為原始座標 4 單位，沿平移後整條 teacher 折線截到 4H。0、負數拒絕，非整數由 `int()` 拒絕。 |
| `LACOT_TEACHER_MIX` | 沿用 M9；本次測 teacher 用 0.5，也測 mix=0 的等價。 |
| `LACOT_OUT_DIR` | 沿用 M9；自測強制設 `/tmp` 新建目錄，不碰 repo 的 `results/`。 |

原題庫建置保留 M9:476–507；挑路 `rng.integers`、平移 `rng.uniform(-0.5,0.5,size=(1,2))` 原碼保留。新碼 [train_hsweep.py:593](train_hsweep.py#L593) 計算 `norm(diff(p) * SD_XY)`，因此量的是原始座標弧長，沒有把細格步數當迷宮格，也沒有新增亂數呼叫。總長 ≤4H 原樣返回；超過時在所在段線性內插截點。保存平移後完整終點，再用 M9 原本的 normalized 弧長 `np.linspace`／`np.interp` 內插為 `T_CAP=128` 點。batch 的 s 仍取軌跡起點，g 改取保存的完整終點（[train_hsweep.py:714](train_hsweep.py#L714)）；端點保持 float32→float64 的原轉型順序。real-mask／act 零佔位原碼保留。

`_BOOT` 不是題庫最短路，沿用 M9 不 jitter、不內插、不截斷的分支，回傳自舉樣本原末點。此額外分支只有靜態審查，本次未測；H 的截斷契約針對最短路 teacher。

檔名組装使用原 `_tag_extra`、`tag` 段落，只在 `_tch...` 後加 `_th{H}`，H 未設不加。T5 實測 H=4 檔名：

```text
ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_th4_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt
```

目標 ckpt 已存在時，全新訓練及續訓都 `SystemExit`；第一道在 tag／dst 組裝完成後、rollout JSON 寫入前（[train_hsweep.py:3731](train_hsweep.py#L3731)），第二道在 `torch.save` 前（[train_hsweep.py:3849](train_hsweep.py#L3849)）。沿用 M9 檔名段落的位置，因此第一道仍在訓練及原腳本 rollout 之後，並非啟動前 preflight；只評估模式沿用 M9 不存 ckpt。這是存在檢查，沒有承諾並行程序的原子排他寫入。T5 已用 `/tmp` 空檔實測 H 未設及 H=4 的 fresh guard 與 save-time recheck；空檔未變、未寫 rollout JSON。

CPU 自測指令（在 repo 根目錄；缺套件應停止，不安裝）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  /home/cymaxwelllee/Projects/lacot/.venv/bin/python -u \
  experiments/_workorders/hsweep/selftest_hsweep.py --start-at T2
```

腳本在 import torch／M9 前清除外部 `LACOT_*`、設 CPU 裝置隱藏及環境；資料為 `~/.ogbench/data/pointmaze-large-stitch-v0.npz`。`MUJOCO_GL=osmesa`、`ENC_OBJ=recon_ictr`、`LEARNED_REFINE=0`、`COND_DROP=0.1`、`BC_INDEP=1`、teacher mix、EMA、warmup、soft 起點配方參考唯讀 [night_0902_overnight2.sh:11](../../../slurm/night_0902_overnight2.sh#L11)。該模板及 M9 沒有明列 `CUBLAS_WORKSPACE_CONFIG`；自測在 import 前明設 `:4096:8`，CPU 不會驗證 CUDA 效果。`PYTHONPATH` 顯式加入 repo root，因複製 M9 的既有 `sys.path` 算式未指到新位置的 repo root。Python 3.11.15、NumPy 2.4.6、PyTorch 2.6.0+cu124（CUDA 不可見）、CPU threads=2。

T1–T5 使用 AST 執行兩支實際程式的資料引擎，停在 `sota_mlp` 前；載入完整本機資料、各自建立 4096 條真題庫，不 mock 資料／內插／batch。T2 基準以逐條索引的 RNG adapter 把所有 4096 條路交給 M9 原 `_teacher_traj`，平移仍是 seed=33 的原 `uniform`，內插及 float32 cast 完全沿用 M9；分母是相同平移後原折線的 raw 弧長。H=3、4、8 各自固定 seed=33，連續抽 256 條一組直到至少 200 條被截；所有已抽樣本都檢查，未換 seed 或排除失敗樣本。T5 執行真實 filename／guard AST。T6 完整 trainer 子程序各 stage 兩步、MAXH=1／每 task 一 seed／R=0 到存 ckpt，保留原腳本必經的 20 次 env step；沒有新增 trainer smoke 分支，也不以這些短 rollout 數字主張性能。省略 `--start-at T2` 可執行整套 T1–T6；本次依回鍋單從 T2 起跑。

上一輪自測原文第 1–57 行逐字保留（包含歷史失敗）；本輪原文自 [selftest-output.txt:59](selftest-output.txt#L59) 追加，process exit code=0。上輪第一輪的 RNG mutant 未 rebind caller，唯一修復輪已在原 harness 修好；本次保留那輪 T1 PASS 與 RNG killer FAIL 證據，並以 trainer 的同一 SHA256 驗證仍可沿用。

| 自測 | 實際結果 |
| --- | --- |
| T1 等價（沿用） | PASS：seeds 33、20261004，各 mix=0、0.5，各 8 批，共 32 批；traj、mask、s、g、act 及 real-mask 全 `np.array_equal`。原文第 32 行。 |
| T1 亂數殺手（沿用） | FAIL：共同的平移後插入點多吃一次 `rng.random()`，seed=33 mix=0.5 batch=0 的 traj 不相等。原文第 33–45 行。 |
| T2 截斷長度 | PASS：所有樣本原折線截斷弧長誤差 <1e-6；每個被截樣本的 `1−L(128)/L(prefix)` ≤ B+1e-9。另檢查完整 g、獨立截點 oracle、短路不變及重複點／恰好邊界。 |
| T2 `4H+2` 殺手 | FAIL：T2 第一句直接報 `T2 H=3 i=0 raw length error=2.0`（第 81–91 行）；殺手必須命中 raw length error，不能以其他斷言失敗代替。 |
| T3 遠終點 | PASS：g 等於平移後完整路終點，距 prefix 末点最大 raw 距離 30.628622 >4，s／real-mask／act 原規則保留。 |
| T3 近端 g 殺手 | FAIL：`T3: g differs from full translated endpoint`（第 93–103 行）。 |
| T4 RNG／真資料 | PASS：8 批各 32 列真資料全部逐位元相等，rng state 及全部 1029 次呼叫的名稱／順序／參數相同（第 104 行）。 |
| T5 檔名與覆蓋 | PASS：H 未設檔名同 M9、H=4 含 `_th4`；兩種 fresh 目標及 save-time guard 拒絕既有空檔（第 105–109 行）。 |
| T6 CPU 微訓練 | PASS：H=3、soft、各 stage 兩步，子程序 exit=0；存出唯一 `_th3` ckpt，required 模組浮點 tensors 有限，cfg 的 STEPS2=2、T_CAP=128。完整輸出第 112–139 行，最後 20 行第 141–160 行，keys 第 162 行。 |

T2 基準全題庫 4096 條：B=`0.0218364525264`（2.18364525264%），中位數=`0.00549410800779`（0.549410800779%），p99=`0.0162034385522`（1.62034385522%）。短少公式固定為 `1−弧長(128點)/弧長(原折線)`，H 樣本分母為截斷後原折線；raw 弧長以 `SD_XY` 還原單位。基準與 H 都量 M9 原本 float32 輸出，未換內插算法、未改截斷定義。H=4 的歷史 1.029941% 反例仍包含於本輪樣本；回鍋單以量得的 B 取代原 `<1%` 門檻。

| H | 抽樣總數 | 被截數 | 最大原折線弧長誤差 | 最大相對短少 | 基準 B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 256 | 221 | 3.553e-15 | 0.761158702835% | 2.18364525264% |
| 4 | 256 | 200 | 5.329e-15 | 1.0299410136% | 2.18364525264% |
| 8 | 512 | 229 | 1.066e-14 | 1.40237971888% | 2.18364525264% |

T6 必要欄位按本配方與實際 save 決定：`cond_enc`、`cond_head`、`flow`、`refine`、`ahead`、`bc_head`、`traj_enc`、`e_pooler` 是無條件存的模組（[train_hsweep.py:3852](train_hsweep.py#L3852)）；`u_dec` 因 `ENC_OBJ=recon_ictr` 而存；`ema` 因 `EMA_W=0.999` 而存；`cfg` 無條件存。`s_embed` 只在 hard 時要求；本次 soft 不要求它，且驗證 key 不存在。建立條件原文在 [train_hsweep.py:960](train_hsweep.py#L960)：

```python
s_embed = nn.Linear(XY_DIM, D_MODEL).to(device) if DEC_START == "hard" else None
```

實際存檔条件 [train_hsweep.py:3860](train_hsweep.py#L3860) 是 `if s_embed is not None`，同時存 `s_embed` 與 `dec_start`；因此 hard 配方 required 會加兩者，soft 都不加。未啟用的 vq／intent／bc_own／lo／續訓 optimizer 等 conditional keys 不列必要欄位。`torch.load(..., map_location="cpu", weights_only=False)` 後的 key 清單原文：

```text
T6 torch.load keys: ['cond_enc', 'cond_head', 'flow', 'refine', 'ahead', 'bc_head', 'traj_enc', 'e_pooler', 'u_dec', 'ema', 'cfg']
```

T6 ckpt（110527256 bytes，保留於暫存目錄）：

```text
/tmp/hsweep-cpu-smoke-8sni4ssk/ckpt_large-stitch_self_K8_c256_ch4_st2_T128_ep1_gu_eorecon_ictr_tch0.5_th3_emw0.999_wu500_s12_dssoft_norf_cd0.1_bci_s33.pt
```

輸出契約：

| 成功路徑必須保留的輸出 | 允許變更的行為 |
| --- | --- |
| H 未設：make_batch 的 traj、mask、s、g、act 逐位元同 M9；real-mask、題庫、取樣／平移 RNG 呼叫不變。 | H 設定：最短路 teacher 的軌跡改為 raw 弧長前 4H，再沿用原內插；g 保持完整遠終點。 |
| 真資料前 n_r 列、teacher action 零佔位及 loss mask 原規則。 | `_teacher_traj` 內部回傳改為 `(trajs, goals)`；唯一 batch 呼叫點同步接收。 |
| H 未設：既有 filename 組裝、ckpt payload keys、rollout JSON schema 及正常 stdout 不新增欄位／行。未主張 ckpt 檔案 bytes 相等。 | H 設定：檔名加 `_th{H}`；現有 ckpt 全新訓練也拒覆蓋並非零 exit。 |
| 沿用原 M9 訓練、probe、rollout、存檔程式；全新目標成功路徑產出原格式 JSON 與 ckpt（T6 已存出）。 | 非正整數 H 拒絕（靜態審查，未另加測試）。 |

測試環境差異聲明：CPU 無法涵蓋 GPU kernel 的浮點累加／TF32、CUDA/CuBLAS 決定性設定、AMP GradScaler 跳步、torch.compile CUDA 路徑、GPU 記憶體／效能／驅動差異。本次 CPU threads=2、AMP=0、COMPILE=0、各 stage 兩步、每 task 一 seed、MAXH=1；完整步數的收斂、H 掃描成效、正式 rollout 成功率及 s33 與 M9 基底差異都未評估。hard 的 s_embed 分支與 BOOT 非本配方，僅依原 save 條件審查，沒有另跑訓練。

SECOND PASS: 最沒把握的兩處是 (1) T2 的 B 是固定題庫與固定 jitter 的 M9 實測值，各 H 雖達 ≥200 個被截樣本，仍是抽樣驗證，沒有主張所有未抽到路線／平移都已驗證；(2) T5 的 guard 沿原 tag 組裝位置，雖已證明拒絕覆蓋且不寫 JSON，仍在訓練／短 rollout 之後，也不提供並行程序原子排他保證。這兩處均未擴張成工單外改動。

n_tests=6 n_killers=3（總交件：沿用 T1 與 RNG killer 1／1，本輪實跑 T2–T6 與兩個殺手 5／2。）

STATUS: DONE
