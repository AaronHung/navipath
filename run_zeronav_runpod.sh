#!/usr/bin/env bash
# run_zeronav_runpod.sh — ZeroNav RunPod execution script
#
# Project: ZeroNav (ZeroSlide-inspired backbone-agnostic navigation)
# Outputs: outputs/zeronav/  (does NOT touch existing outputs/)
#
# ──────────────────────────────────────────────────────────────────────────
# WORKFLOW OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
#   [RunPod] git pull → symlink setup → train → eval → analyze → git push
#   [Mac]    git pull → inspect JSON results, plot, report
#
# ──────────────────────────────────────────────────────────────────────────
# BACKBONE TERMINOLOGY
# ──────────────────────────────────────────────────────────────────────────
#   The frozen diagnostic backbone is accessed ONLY as a feature extractor:
#     • Z   = patch embeddings via CONCH vision encoder (frozen)
#     • f_txt = class text embeddings via CONCH text encoder (frozen)
#   No prototype features. No backbone internals beyond these two.
#   Referred to as:  outputs/backbone_reverse_fold1.pt  (symlink)
#
# ──────────────────────────────────────────────────────────────────────────
# EXPERIMENT GOALS
# ──────────────────────────────────────────────────────────────────────────
#   New clean architecture: TextNavRouter (no prototype_features, 514-dim)
#   Comparison:
#     zeronav_oneshot   — top-64 patches at once (λ=0, 1 round)
#     zeronav_multishot — 4 rounds × 16 patches, MMR diversity (SBO)
#     zeroshot_oneshot  — ZeroSlide: max text-patch cosine, top-64
#     zeroshot_multishot— ZeroSlide score + SBO (no training)
#   λ sweep: 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0
#   ONE fold only (fold 1, reverse order)
#
# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: INFERENCE ONLY after Phase 1 training
#   run_zeronav.py eval    → loads skill bank, ZERO retraining
#   run_zeronav.py analyze → loads skill bank, ZERO retraining
# ──────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="/workspace/src/navipath"
FOLD=1
ORDER="reverse"
BACKBONE="outputs/backbone_reverse_fold1.pt"   # symlink to frozen diagnostic backbone
LAMBDAS="0.0,0.1,0.25,0.5,0.75,1.0,1.5,2.0"
BUDGET=64
STEP_SIZE=16    # 4 rounds × 16 = 64 patches total

mkdir -p "${REPO_DIR}/logs"

echo "[zeronav] Creating tmux session 'zeronav' with 4 windows..."
tmux new-session  -d -s zeronav -x 240 -y 50
tmux rename-window -t zeronav:0 "main"
tmux new-window    -t zeronav:1 -n "eval"
tmux new-window    -t zeronav:2 -n "analyze"
tmux new-window    -t zeronav:3 -n "git"

# ── Window 0: Phase 0+1 — setup + train ──────────────────────────────────
tmux send-keys -t zeronav:0 "
cd ${REPO_DIR} && source /workspace/bootstrap/env.sh

echo '=== Phase 0: git pull + backbone symlink ==='
git pull origin main

# Create neutral-named symlink for the frozen diagnostic backbone
ln -sf outputs/qpmil_reverse_fold1.pt outputs/backbone_reverse_fold1.pt
echo 'backbone symlink ready: outputs/backbone_reverse_fold1.pt'

echo ''
echo '=== Phase 1: TRAIN TextNavRouter (fold ${FOLD}, order ${ORDER}) ==='
echo 'Backbone used as FROZEN feature extractor only (Z + f_txt).'
echo 'No prototype features accessed.'
echo ''
python run_zeronav.py train \\
    --backbone-ckpt ${BACKBONE} \\
    --order ${ORDER} --fold ${FOLD} \\
    --epochs 5 --top-k 64 --lr 5e-4 \\
    2>&1 | tee logs/zeronav_train_f${FOLD}.log

echo ''
echo '=== Phase 1 DONE ==='
echo 'Skill bank: outputs/zeronav/skill_bank_${ORDER}_f${FOLD}.pt'
echo 'Switch to Window 1 (Ctrl-b 1) and paste the eval command.'
" Enter

# ── Window 1: Phase 2 — Inference-only λ sweep ───────────────────────────
tmux send-keys -t zeronav:1 "
cd ${REPO_DIR} && source /workspace/bootstrap/env.sh
echo 'Window 1: Phase 2 — INFERENCE ONLY λ sweep'
echo 'Wait for Phase 1 to complete, then paste:'
echo ''
echo 'python run_zeronav.py eval \\'
echo '    --backbone-ckpt ${BACKBONE} \\'
echo '    --order ${ORDER} --fold ${FOLD} \\'
echo '    --lambdas ${LAMBDAS} \\'
echo '    --budget ${BUDGET} --step-size ${STEP_SIZE} \\'
echo '    2>&1 | tee logs/zeronav_eval_f${FOLD}.log'
" Enter

# ── Window 2: Phase 3 — Router analysis ──────────────────────────────────
tmux send-keys -t zeronav:2 "
cd ${REPO_DIR} && source /workspace/bootstrap/env.sh
echo 'Window 2: Phase 3 — Router analysis (INFERENCE ONLY)'
echo 'Wait for Phase 1 to complete, then paste:'
echo ''
echo 'python run_zeronav.py analyze \\'
echo '    --backbone-ckpt ${BACKBONE} \\'
echo '    --order ${ORDER} --fold ${FOLD} \\'
echo '    2>&1 | tee logs/zeronav_analyze_f${FOLD}.log'
" Enter

# ── Window 3: Phase 4 — git push results ─────────────────────────────────
tmux send-keys -t zeronav:3 "
cd ${REPO_DIR} && source /workspace/bootstrap/env.sh
echo 'Window 3: Phase 4 — git push results'
echo 'Run AFTER Phases 2 and 3 are complete:'
echo ''
echo 'git add outputs/zeronav/'
echo 'git add logs/zeronav_train_f1.log logs/zeronav_eval_f1.log logs/zeronav_analyze_f1.log 2>/dev/null || true'
echo 'git commit -m \"results(zeronav): fold1 reverse lambda-sweep 0.0-2.0\"'
echo 'git push origin main'
echo ''
echo 'Then on Mac: git pull origin main'
" Enter

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " TMux session 'zeronav' created."
echo " Attach: tmux attach -t zeronav"
echo ""
echo " Window 0 (main):    RUNNING — git pull + train"
echo " Window 1 (eval):    WAITING — paste after training done"
echo " Window 2 (analyze): WAITING — paste after training done"
echo " Window 3 (git):     WAITING — paste after all done"
echo "═══════════════════════════════════════════════════════════════"
