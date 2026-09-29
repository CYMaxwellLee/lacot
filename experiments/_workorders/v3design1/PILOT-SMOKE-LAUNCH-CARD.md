# pilot 拒啟 smoke 發射單 v4（F5；三輪預檢 #2#4 補閉）

日期 2026-09-29。v1 NO-GO（六 blocker）→v2→二輪 NO-GO（四殘留）→v3（PATH／
fail-fast／symlink／hash 執行門）→三輪 NO-GO（#1#3 CLOSED、#2#4 NOT-CLOSED、
PILOT-SMOKE-PREFLIGHT-V3.md）→v4 修訂面：in-batch batch_status append 全
fail-closed（寫失敗轉 exit 45）／smoke.log 跑前預開（封「redirection 失敗回
rc=1、偽裝成 gate 拒啟」）／pre-batch vs in-batch ABORT 兩類分界明文／
dataset_hash 升無條件硬錨（程式契約引註）。本發射=**CPU 拒啟 smoke**。
拒啟是預期結果。

## 在證明什麼

production builder 對真 exp1 ckpt 的載入鏈成立＋授權 gate 正確拒啟，且
**「gate 拒啟」與「環境/資料/schema 炸掉」可判讀地分開**。判準＝完成錨（exact
契約版）全中。

## 跑哪台＋部署（同 v1、補 probe 實測）

- jasmine、直跑。code=`/home/cymaxwelllee/Projects/lacot`（NFS home 共享、hash
  兩機一致〔實測 9/29〕：scratch=b7d3a0d9…、builder=c1d63cdd…）。
- PY=`/archive/cymaxwelllee/LaCoT/.venv/bin/python` ＋ `-B`。
- ⭐ **clean-env import probe 已過**〔實測 9/29、jasmine〕：`env -i PATH=/usr/bin:/bin
  CUDA_VISIBLE_DEVICES='' …` 下 numpy/torch/ogbench＋lacot.{e_target,nf_head,model,
  refine_training,dev_eval,refine_service_builder,refine_models,refine_quality} 全
  import OK、`torch.cuda.is_available()==False`、無 HOME 不炸。
- 真檔錨值〔實測 9/29 撈自 oracle.pt〕：
  - `split_hash = 9aec9ac9a22e7fe4e77d829ae19b7ab462e0b9acf4ff3d17f2de33dab01fa75e`
  - `readings_digest = 840b40766d1ffd31e360bef48921bfd955108d2657fab2b1e3954f53555b178f`
  - `dataset_hash = 9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8`
  - `domain=pointmaze`、`source_kind=offline_dataset`、`ckpt sha256=de6338b6d513…`

## 指令全文 v3（批次契約＋執行時 hash 門版）

```bash
set -u
cd /home/cymaxwelllee/Projects/lacot || exit 40
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
NPZ=/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz
CKPT=/archive/cymaxwelllee/refine_v3/exp1/pointmaze-s0/oracle.pt
BASE=/archive/cymaxwelllee/refine_v3/pilot-builder-smoke
NPZ_SHA=9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8
CKPT_SHA=de6338b6d5132795f44fbf540614a6b2346d845631f368ad1fc6d2e059804b17
test -x "$PY"   || { echo "ABORT MISSING PY" >&2;   exit 41; }
test -r "$NPZ"  || { echo "ABORT MISSING NPZ" >&2;  exit 42; }
test -r "$CKPT" || { echo "ABORT MISSING CKPT" >&2; exit 43; }
[ "$(sha256sum "$NPZ"  | cut -d' ' -f1)" = "$NPZ_SHA" ]  || { echo "ABORT NPZ SHA MISMATCH" >&2;  exit 46; }
[ "$(sha256sum "$CKPT" | cut -d' ' -f1)" = "$CKPT_SHA" ] || { echo "ABORT CKPT SHA MISMATCH" >&2; exit 47; }
mkdir -p "$BASE" && test -w "$BASE" || { echo "ABORT BASE UNWRITABLE" >&2; exit 44; }
RUN_ID="run-$(date +%Y%m%dT%H%M%S)"
echo "$RUN_ID START" > "$BASE/batch_status" || exit 45
summary=""
for pilot_seed in 0 1 2; do
  out="$BASE/s${pilot_seed}"
  [ ! -L "$out" ] || { echo "$RUN_ID ABORT SYMLINK s${pilot_seed}" >> "$BASE/batch_status" || exit 45; echo "ABORT SYMLINK $out" >&2; exit 48; }
  if [ -d "$out" ]; then
    listing=$(ls -A "$out" 2>&1) || { echo "$RUN_ID ABORT UNREADABLE s${pilot_seed}" >> "$BASE/batch_status" || exit 45; echo "ABORT UNREADABLE $out: $listing" >&2; exit 49; }
    if [ -n "$listing" ]; then
      echo "$RUN_ID STALE s${pilot_seed}" >> "$BASE/batch_status" || exit 45
      echo "STALE $out" >&2; exit 24
    fi
  fi
  mkdir -p "$out" || exit 45
  printf "%s\n" "$RUN_ID" > "$out/run_id" || exit 45
  : >"$out/smoke.log" || { echo "$RUN_ID ABORT LOGWRITE s${pilot_seed}" >> "$BASE/batch_status" || exit 45; echo "ABORT LOGWRITE $out" >&2; exit 45; }
  rc=0
  env -i PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
    OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench \
    LACOT_ENV=pointmaze-large-stitch-v0 \
    LACOT_REFINE_QUALITY_CKPT="$CKPT" \
    LACOT_LEARNED_REFINE=1 LACOT_CONS=ema LACOT_EMA_M=0.996 \
    LACOT_INTENT='' LACOT_INTENT_GUID_W=0 \
    LACOT_SEED="$pilot_seed" LACOT_DATA_SEED=-1 LACOT_BOOT_SEED=-1 \
    LACOT_ENC_OBJ=recon_ictr LACOT_W_ICTR=0.2 LACOT_ICTR_SIGMA=0.05 \
    LACOT_DEC_START=hard LACOT_K=4 LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128 \
    LACOT_STEPS1=0 LACOT_STEPS2=1 LACOT_WARMUP=0 LACOT_LR_SCALE=1 \
    LACOT_LOAD_CKPT='' LACOT_CONT_TRAIN=0 LACOT_S1_FROM='' \
    LACOT_COND_DROP=0.1 LACOT_TEACHER_MIX=0 LACOT_EMA_W=0 \
    LACOT_AMP=0 LACOT_COMPILE=0 LACOT_DIV_W=0 LACOT_GRPO_W=0 LACOT_LO_W=0 \
    LACOT_BC_OWN=0 LACOT_BC_INDEP=0 LACOT_BON_N=0 LACOT_ORACLE_ARM='' \
    LACOT_EVAL_EPISODES=2 LACOT_DEV_EVAL=0 LACOT_PREREQ=0 \
    LACOT_OUT_DIR="$out" \
    "$PY" -B experiments/scratch_lacot_rollout.py >"$out/smoke.log" 2>&1 || rc=$?
  printf "%s\n" "$rc" > "$out/exit_code" || exit 45
  summary="$summary s${pilot_seed}:rc=$rc"
  if [ "$rc" -ne 1 ]; then
    echo "$RUN_ID ABORT rc=$rc s${pilot_seed}" >> "$BASE/batch_status" || exit 45
    echo "ABORT unexpected rc=$rc seed=$pilot_seed (expected gate rc=1)" >&2
    exit 50
  fi
done
echo "$RUN_ID DONE$summary" >> "$BASE/batch_status" || exit 45
echo "BATCH-SUMMARY$summary gate-class=judged-by-harvest"
```

（v4 契約：wrapper 只認 rc=1 為可繼續。ABORT 兩類——**40-47=pre-batch**：
batch_status 尚未建立或本身不可信，原因行只保證在 stderr／sbatch log；
**24/48/49/50=in-batch**：batch_status 必有原因行、append 失敗即轉 exit 45
（wrapper 寫入失敗類）。smoke.log 跑前 `: >` 預開驗可寫——否則 bash
redirection 失敗會令該行回 rc=1、恰好偽裝成 gate 拒啟。gate class 語義判讀
=④收割官按錨執行、wrapper 不冒判。）

（env block 逐字同 v1＝builder 草稿、預檢官已逐字核 PASS；v2 只動 wrapper 骨架。）

## 完成錨 v2（exact 契約；預檢官 5 條照納）

每一 seed 同時滿足：
1. `exit_code` 檔內容 == `1`（預期 Python gate rc）。`24`=stale FAIL、`137`=
   signal/OOM FAIL、其他非 1 非 0=crash FAIL、`0`=**大 FAIL**（gate 沒攔）。
   wrapper ABORT 碼＝整批 FAIL，兩類：**40-47=pre-batch**（原因行在 stderr／
   sbatch log、batch_status 可能不存在）；**24/48/49/50=in-batch**（batch_status
   必有原因行、append fail-closed 轉 45）。rc 非 1 時 wrapper 立即 ABORT 不續跑
   後續 seed（v4 契約）。
2. `smoke.log` 順序：先 `device: cpu`（硬錨；出現 `device: cuda`=FAIL）→ 再
   `refine quality metadata:`，逐 key 對：`domain: pointmaze`、
   `source_kind: offline_dataset`、`split_hash` 全文=上方錨值、`readings_digest`
   全文=上方錨值、`dataset_hash` 全文=上方錨值（**無條件硬錨、缺欄=FAIL**。
   程式契約：builder 的 provenance dict 無條件寫入 `dataset_hash=cal.
   dataset_hash`、且 ckpt load 時對該欄驗 64-hex 非法即 raise——
   `lacot/refine_service_builder.py` 構造處＋欄位驗證 loop〔實測 9/29 親讀〕）、
   `exposure_available: False`、fingerprint 欄非空（首載產出、收割官記錄全文
   ——首載無先驗參照、記錄值成為後續跑的錨）。metadata 必須在 traceback 之前。
3. traceback 最後一行 marker：`ValueError: oracle qualification failed:` 前綴
   且訊息含 generated validity／coverage／`A action NMSE` 字樣（現況真檔預期）。
   ⭐ 補強（二輪預檢）：**三 seed 的 traceback 尾行必須逐字一致**（同 ckpt 同
   gate＝同訊息；不一致=FAIL）；尾行全文由④收割官記錄並人工判讀語義（本 smoke
   =首航、人工判讀明文化，記錄值成為日後 exact 錨）。僅當人工核對 qualification
   blockers 全空時，才接受替代 marker `qualified same-plan action teacher/
   exposure unavailable; R>0 refused`。**無 marker 時 exit_code=1 也算 FAIL。**
4. marker 之後：無 backward／opt2 update／rollout JSON／ckpt 寫出；`$out` 內
   只有 `smoke.log`＋`exit_code`＋`run_id`。
5. 批次級：`batch_status` 有本次 `RUN_ID` 的 `START`＋`DONE` 行、無 `STALE` 行；
   三 seed 的 `run_id` 檔=同一 RUN_ID（⛔ 不得用舊檔湊批）。

## 收割規矩

④收割官照上錨逐格；M1 限定照舊（fingerprint／digest／split hash 對真檔＝
本單錨值）。任何一格不中=SMOKE FAIL 回 lead，⛔ 不算「拒啟成功」。

## ETA×死點

每 seed 〔拍〕2-6 分（np.load 全量 NPZ＋CPU 建模是 gate 前主要成本——預檢官
格 8 指正；非 ckpt schema）；三 seed <20 分；⑤發射官 TTL=40 分（ETA×2）。

## 依賴（全 [實測] 9/29）

exact NPZ=`pointmaze-large-stitch-v0.npz`（在地）／oracle.pt（sha de6338b6…）／
clean-env import 全鏈 probe 過（含動態 builder 鏈＋numpy）／`/archive`=ext4／
venv torch 2.6.0+cu124。
