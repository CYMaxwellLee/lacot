#!/bin/bash
# 2026-09-06 日批：①WS 療程擴八顆（p=0.3 續訓 s41~47 新跑＋s40 沿用深夜）
#                  ②p=1 對照臂（單顆 s40 續訓、drop=1.0）
#
# ⭐ 機制來源＝9/5 深夜實測 job（24417 WS-idp0.3 train / 24418 on / 24419 z0，
#   sacct -j 24417 --parsable2 --format=SubmitLine 逐字讀出），不是 eve_0905_idp_dose.sh
#   的 S1_FROM 機制 —— 那支是「凍 s27 stage1、fresh stage2 訓 STEPS2 步」的另一種實驗（idp_dose）。
#   這批是【續訓 warm-start】：LACOT_LOAD_CKPT=f27n 全模型 ckpt（st8000）＋LACOT_CONT_TRAIN=1，
#   續 LACOT_STEPS2=4000 步，檔名長成 _st0..._ct4000（scratch_lacot_rollout.py:1218-1222,2734-2735）。
#   env 全串已跟 sacct 的 SubmitLine token-diff 過（排序後完全一致，只有 LACOT_OUT_DIR 不同——
#   OUT_DIR 只餵 os.path.join 存檔路徑，不影響訓練，見腳本 2835-2836）。
#
# s40（p=0.3）判定：深夜 job 24417 的 env 跟本批 s41~47 模板逐 token diff 乾淨，
#   唯一差異是 LACOT_OUT_DIR（深夜 results/day_0906/ws40 vs 本批 results/day_0906/ws8_idp0.3）
#   —— 非訓練相關 ⇒ 沿用深夜產物，不重跑 s40 的續訓，只補 eval 兩腿。
#
# 判讀先釘（送出前就寫死，不能等看到數字再挑）：
#   ① 八顆（s40~47）平均 zero-probe 站不站得住 .5 級 ⇒ 「療程」敘事 go/no-go
#   ② p=1（s40, drop=1.0）zero-probe 爬到 .55 ⇒ 續訓本身就會回、療程無功勞；
#      停在 .3~.4 ⇒ dropout 課程（p=0.3）真的有教到東西，不是續訓的功勞
#
# 紀律：先 DRYRUN=1 跑過、echo 每條 sbatch 全參數自檢，再無 DRYRUN 真送。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0

OUT03=results/day_0906/ws8_idp0.3
OUTZ03=results/day_0906/ws8_idp0.3_zero
OUTP1=results/day_0906/ws_p1
OUTZP1=results/day_0906/ws_p1_zero
mkdir -p slurm/logs "$OUT03" "$OUTZ03" "$OUTP1" "$OUTZP1"

DRYRUN=${DRYRUN:-0}
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN] sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 --job-name=$name -o slurm/logs/%x-%j.out $depflag --wrap \"cd ~/Projects/lacot && env $*\"" >&2
    echo "DRY-$name"; return 0   # ⚠️ 每次呼叫都經 $(...) 子殼、全域計數器在子殼裡不會累加——改用 job 名本身當假 id（本來就唯一）
  fi
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

NODES=(jasmine lady moana pocahontas); i=0

# f27n 起點（st8000、8 顆共用同一批來源；⛔ 手寫全名——下面先逐顆驗存在）
f27n_ck() { echo "results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$1.pt"; }
# 續訓輸出 ckpt 全名（照 _tag_extra 的 :g 格式；drop=1.0 會印成 idp1，不是 idp1.0——已用 python3 -c 驗過）
ws_ck() { # $1=outdir $2=seed $3=drop(:g 格式化後的字串)
  echo "$1/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp$3_ct4000_dssoft_norf_cd0.1_bci_s$2.pt"
}

echo "=== 起點 ckpt 存在性檢查（f27n s40~47）==="
for S in 40 41 42 43 44 45 46 47; do
  f=$(f27n_ck $S)
  [ -f "$f" ] || { echo "⛔ 缺 s$S 起點 ckpt: $f" >&2; exit 1; }
done
WS40_CK=results/day_0906/ws40/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct4000_dssoft_norf_cd0.1_bci_s40.pt
[ -f "$WS40_CK" ] || { echo "⛔ 深夜 s40 續訓產物不存在，無法沿用: $WS40_CK" >&2; exit 1; }
echo "✓ 8 顆 f27n 起點都在，深夜 s40(p=0.3) 續訓產物也在"

# 續訓一顆（LOAD_CKPT + CONT_TRAIN warm-start；照 24417 實測 env，只換 JT/SEED/DROP/OUT_DIR）
# ⚠️ 不能用 $(train_ws ...) 呼叫——那會逼 bash fork 子殼，子殼裡 i=$((i+1)) 累加完就跟著子殼死掉，
#    父殼的 i 永遠停在 0 ⇒ 八顆全落在 NODES[0]（DRYRUN 第一輪就是這樣炸出來的，已修正）。
#    改成設全域變數，呼叫端用「純函式呼叫」（不包 $(...)）保留在同一個殼裡執行。
train_ws() { # $1=jobtag $2=seed $3=drop(env寫法, 如 0.3 / 1.0) $4=outdir ; 設 TRW_NODE / TRW_JOB
  local JT=$1 S=$2 DP=$3 OD=$4
  TRW_NODE=${NODES[$((i%4))]}; i=$((i+1))
  local FROM; FROM=$(f27n_ck $S)
  TRW_JOB=$(sub "$TRW_NODE" $JT-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=f27n LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$FROM LACOT_CONT_TRAIN=1 LACOT_STEPS2=4000 LACOT_INTENT_DROP=$DP LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OD $APY -u experiments/scratch_lacot_rollout.py")
}

# 成對 eval（on + z0）；CK 一律手寫全名傳進來，不用萬用字元
eval_pair() { # $1=jobtag $2=seed $3=drop(env寫法) $4=ck_full $5=on_dir $6=zero_dir $7=deps
  local JT=$1 S=$2 DP=$3 CK=$4 OND=$5 ZOD=$6 DEPS=$7
  local EVBASE="$BASE0 OGBENCH_DATA_DIR=$ZDATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=$DP"
  local JON; JON=$(sub zeldajr $JT-s$S-on "$DEPS" "$EVBASE LACOT_OUT_DIR=$OND $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py")
  local JZ; JZ=$(sub zeldajr $JT-s$S-z0 "$DEPS" "$EVBASE LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$ZOD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py")
  echo "  eval $JT s$S(drop=$DP) ck=$(basename "$CK") -> on=$JON z0=$JZ"
}

echo "=== ① WS8 p=0.3（s40 沿用深夜、s41~47 新續訓）==="
echo "s40：沿用深夜 job 24417 產物（$WS40_CK）—— env token-diff 乾淨、唯一差異 LACOT_OUT_DIR（見腳本頭註解），不重跑"
eval_pair WS8 40 0.3 "$WS40_CK" "$OUT03" "$OUTZ03" -

for S in 41 42 43 44 45 46 47; do
  train_ws WS8 $S 0.3 "$OUT03"           # 純呼叫、不包 $(...) —— i 才會在同一個殼裡累加
  CK=$(ws_ck "$OUT03" $S 0.3)
  echo "WS8 s$S 續訓 -> $TRW_NODE (train job=$TRW_JOB)"
  eval_pair WS8 $S 0.3 "$CK" "$OUT03" "$OUTZ03" "$TRW_JOB"
done

echo "=== ② p=1 對照臂（單顆 s40, drop=1.0）==="
train_ws WSP1 40 1.0 "$OUTP1"
CKP1=$(ws_ck "$OUTP1" 40 1)   # ⚠️ :g 格式化：1.0 -> "1"（idp1，不是 idp1.0——python3 -c 已驗）
echo "WSP1 s40 續訓 -> $TRW_NODE (train job=$TRW_JOB)"
eval_pair WSP1 40 1.0 "$CKP1" "$OUTP1" "$OUTZP1" "$TRW_JOB"

echo "=== 送出完畢（DRYRUN=$DRYRUN）==="
