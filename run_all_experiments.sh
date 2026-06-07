#!/usr/bin/env bash
# ============================================================
# NaviPath-MoE 完整實驗腳本（給同學用）
# 用法：
#   bash run_all_experiments.sh paper  1   # fold 1, paper order
#   bash run_all_experiments.sh reverse 2  # fold 2, reverse order
#   bash run_all_experiments.sh paper  3   # fold 3, paper order
#
# 建議：同時開 2-3 個 RunPod 機器，各跑不同的 fold
# 跑完後把 outputs/ 裡的 .json 和 .log 傳回 Mac
# ============================================================

ORDER=${1:-paper}
FOLD=${2:-1}
OUT="outputs"
QCFG="QPMIL-VL/configs/main.yaml"

echo "============================================"
echo " NaviPath-MoE Experiments"
echo " Order: $ORDER  |  Fold: $FOLD"
echo "============================================"
mkdir -p $OUT

# ── M1: QPMIL baseline ─────────────────────────────────────────────────────
echo "[Step 1/7] M1: QPMIL baseline ($ORDER, fold $FOLD)"
[ ! -f $OUT/qpmil_${ORDER}_fold${FOLD}.pt ] && \
  python train_qpmil_runner.py --order $ORDER --fold $FOLD --save-ckpt \
      2>&1 | tee $OUT/m1_${ORDER}_f${FOLD}.log || \
  echo "[Skip] M1 checkpoint already exists"

# ── CL Baselines: EWC ──────────────────────────────────────────────────────
echo "[Step 2/7] EWC baseline ($ORDER, fold $FOLD)"
python train_cl_baselines.py --method ewc --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/ewc_${ORDER}_f${FOLD}.log

# ── CL Baselines: LwF ──────────────────────────────────────────────────────
echo "[Step 3/7] LwF baseline ($ORDER, fold $FOLD)"
python train_cl_baselines.py --method lwf --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/lwf_${ORDER}_f${FOLD}.log

# ── Ablation A: Router Only ─────────────────────────────────────────────────
echo "[Step 4/7] Ablation A: Router Only ($ORDER, fold $FOLD)"
python train_navipath.py --config configs/navipath_router_only.yaml \
    --backbone-ckpt $OUT/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/abl_router_only_${ORDER}_f${FOLD}.log

# ── Ablation B: w/o MacroRouter & Consolidation ─────────────────────────────
echo "[Step 5/7] Ablation B: No MacroRouter & Consolidation ($ORDER, fold $FOLD)"
python train_navipath.py --config configs/navipath_no_macro.yaml \
    --backbone-ckpt $OUT/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/abl_no_macro_${ORDER}_f${FOLD}.log

# ── Ablation C: w/o Consolidation ───────────────────────────────────────────
echo "[Step 6/7] Ablation C: No Consolidation ($ORDER, fold $FOLD)"
python train_navipath.py --config configs/navipath_no_consol.yaml \
    --backbone-ckpt $OUT/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/abl_no_consol_${ORDER}_f${FOLD}.log

# ── M8: Full NaviPath-MoE ───────────────────────────────────────────────────
echo "[Step 7/7] M8: Full NaviPath-MoE ($ORDER, fold $FOLD)"
python train_navipath.py --config configs/navipath_full.yaml \
    --backbone-ckpt $OUT/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --save-ckpt \
    2>&1 | tee $OUT/m8_${ORDER}_f${FOLD}.log

# ── 結果摘要 ────────────────────────────────────────────────────────────────
echo ""
echo "=== RESULTS SUMMARY ($ORDER fold $FOLD) ==="
grep -E "CL summary|method\] CL|go/no-go" $OUT/m1_${ORDER}_f${FOLD}.log 2>/dev/null \
    | head -5 | sed 's/^/  M1: /'
grep -E "CL summary" $OUT/ewc_${ORDER}_f${FOLD}.log 2>/dev/null \
    | head -2 | sed 's/^/  EWC: /'
grep -E "CL summary" $OUT/lwf_${ORDER}_f${FOLD}.log 2>/dev/null \
    | head -2 | sed 's/^/  LwF: /'
grep -E "NaviPath\] CL" $OUT/abl_router_only_${ORDER}_f${FOLD}.log 2>/dev/null \
    | sed 's/^/  Router-only: /'
grep -E "NaviPath\] CL" $OUT/abl_no_macro_${ORDER}_f${FOLD}.log 2>/dev/null \
    | sed 's/^/  No-Macro: /'
grep -E "NaviPath\] CL" $OUT/abl_no_consol_${ORDER}_f${FOLD}.log 2>/dev/null \
    | sed 's/^/  No-Consol: /'
grep -E "NaviPath\] CL" $OUT/m8_${ORDER}_f${FOLD}.log 2>/dev/null \
    | sed 's/^/  M8-Full: /'
echo "=== ALL DONE ==="
