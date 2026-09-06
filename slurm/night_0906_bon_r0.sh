#!/bin/bash
# 2026-09-06 夜：推論期 BoN 的 R0 錶（rung 0.5 設計卡 §1 的 c 格）—— eval only、12 支
#
# 量什麼：把 rung 0.5 探針（experiments/probe_bon_rung05.py）的 plan 層 BoN 接進【完整
#   rollout】的規劃步驟（LACOT_BON_N），看它在官方協定的成功率上兌現多少。
#   3 顆 ckpt × on/zero 兩個 conditioning × N∈{8,16} ＝ 12 支。
#
# 🚨🚨 開跑前先讀這一段（09-06 CPU 實測，⛔ 不是推測）：
#   C8 乘法閘是在【CHUNK 長度的資料短窗】上校準的（ρ_len=2.595、δ_step=0.1040 ——
#   本腳本跑出來的數字跟探針報告逐位元相同，⇒ 校準本身沒問題）。但 rollout 的規劃題是
#   【現在位置 → 最終目標】，L_BFS 平均 37.5 格（探針的量測窗 L_BFS 中位數是 1）。
#   ⇒ arclen 門檻要 12.50 而解碼計畫只有 6.57（T_CAP=128 × δ_step 的物理上限≈13.3，
#     而且「要夠長」跟「每步要夠小」在固定點數下互相衝突）
#   ⇒ 實測閘過率 0.01（arclen 那道 0.01／步長那道 0.08）、98.6% 的規劃事件 N 條全同分
#   ⇒ BoN 退化成「取第 0 條」＝【等於沒開】。
#   ⛔ 所以這 12 支若跑出「沒有增益」，那不是「BoN 沒用」，是【尺套錯尺度】。
#   ⭐ json 的 out["bon"] 已經把兩道閘、L_BFS、arclen 門檻 vs 實際分開落地 ⇒ 讀表時先看那格。
#   ⇒ 真要量 c 格，得先把 ρ_len/δ_step 改在【rollout 尺度的 (s,g)】上校準（設計決定，
#     ⛔ 不在本次施工範圍）。這 12 支目前的定位＝把上面那句話用真機數字釘死。
#
# 節點政策（沿用 09-06 深夜裁定）：eval 只用 Turing 三台 lady／moana／pocahontas，
#   明確 --nodelist 綁定＋輪流；⛔ 不用 zeldajr（drain）、不碰既有隊伍。
# ⛔ 本檔只【新增】job，既有過夜批＋修復批一個都不動。
#
# 紀律：DRYRUN=1 先自檢（ckpt 在不在、節點清單、檔名不互撞），過了才 DRYRUN=0 真送。
set -euo pipefail
cd ~/Projects/lacot

APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
OUT=results/day_0906/bon_r0
OUTZ=results/day_0906/bon_r0_zero
mkdir -p slurm/logs "$OUT" "$OUTZ"

# ── 三顆量測對象 ────────────────────────────────────────────────────────────
# ⚠️ 三顆【都是 s40】⇒ eval 檔名的 _s 段跟 ckpt 的 seed 走（全是 _s40）
#    ⇒ ⛔ 光靠 seed 分不開。⇒ 用 LACOT_BOOT_TAG 把身份寫進檔名（_bt<tag>），
#      再加 INTENT_DROP 的差別 ⇒ 同一個 OUT_DIR 裡六個檔互不撞。
#      （BOOT_TAG 在沒有 LACOT_BOOT_DATA 時【只進檔名】，⛔ 不改任何行為。）
# ⚠️ INTENT_DROP 也只進檔名／json 血緣（它是訓練期旋鈕）⇒ 照各顆【訓練時的真值】填，
#    ⛔ 不要為了統一而給 f27n 填 0.3 —— 那會在檔名上把它標成別的顆。
CK1=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s40.pt
CK2=results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s40.pt
CK3=results/day_0906/ws40/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct4000_dssoft_norf_cd0.1_bci_s40.pt

echo "=== 起點 ckpt 存在性檢查 ==="
for f in "$CK1" "$CK2" "$CK3"; do
  [ -f "$f" ] || { echo "⛔ 缺 ckpt: $f" >&2; exit 1; }
  echo "✓ $(basename "$f")"
done

DRYRUN=${DRYRUN:-0}
NODES=(lady moana pocahontas)
NI=0
JOBS=()

sub() { # $1=node $2=name ; 其餘＝env
  local node=$1 name=$2; shift 2
  if [ "$DRYRUN" = "1" ]; then
    # ⛔ 印到 stderr：這個函式的 stdout 是 job id（被 $( ) 接走）——
    #    印到 stdout 會讓 dry-run 的訊息假裝成 job id，整份自檢就看不懂了
    { echo "[DRY] $name -> $node"; echo "      env $*" | head -c 260; echo; } >&2
    echo "DRY"
    return 0
  fi
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist="$node" --gres=gpu:1 \
    --job-name="$name" -o slurm/logs/%x-%j.out \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

ep() { # $1=標籤(BOOT_TAG) $2=ckpt $3=INTENT_DROP $4=BON_N
  local TAG=$1 CK=$2 IDP=$3 BN=$4
  local EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 \
LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft \
LACOT_INTENT=embed LACOT_BOOT_TAG=$TAG LACOT_INTENT_DROP=$IDP LACOT_INTENT_ARMS=0 \
LACOT_BON_N=$BN"
  local NODE=${NODES[$((NI % 3))]}; NI=$((NI + 1))
  local J1; J1=$(sub "$NODE" "BON-$TAG-n$BN-on" \
    "$EVB LACOT_OUT_DIR=$OUT $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE=${NODES[$((NI % 3))]}; NI=$((NI + 1))
  local J2; J2=$(sub "$NODE" "BON-$TAG-n$BN-z0" \
    "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  JOBS+=("$J1" "$J2")
  echo "  $TAG N=$BN  on=$J1  zero=$J2"
}

echo "=== 送 12 支（3 顆 × on/zero × N∈{8,16}）==="
for BN in 8 16; do
  ep f27n      "$CK1" 0.0 "$BN"
  ep idpxm     "$CK2" 0.3 "$BN"
  ep ws40ct4k  "$CK3" 0.3 "$BN"
done

echo
echo "job ids: ${JOBS[*]}"
echo "產物：$OUT（on）／$OUTZ（zero）"
echo "⚠️ 讀表先看 json 的 bon.mean_gate_arclen —— < 0.1 就代表閘恆關、這支的 BoN 等於沒開。"
