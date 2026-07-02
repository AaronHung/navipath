#!/usr/bin/env bash
# run_zeronav_runpod.sh — ZeroNav RunPod execution script
#
# Project: ZeroNav (ZeroSlide-inspired backbone-agnostic navigation)
# Outputs: outputs/zeronav/  (does NOT touch existing outputs/)
#
# ──────────────────────────────────────────────────────────────────────────
# WORKFLOW OVERVIEW
# ──────────────────────────────────────────────────────────────────────────
#   [RunPod] git pull → train → eval → analyze → git push outputs/zeronav/
#   [Mac]    git pull → inspect JSON results, plot, report
#   [Mac→RunPod tmux] extra inference with saved skill bank (no retraining!)
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
#   ONE fold only (fold 1, reverse order) for tonight's deadline
#
# ──────────────────────────────────────────────────────────────────────────
# IMPORTANT: INFERENCE ONLY after Phase 1 training
#   run_zeronav.py eval  → loads skill bank, ZERO retraining
#   run_zeronav.py analyze → loads skill bank, ZERO retraining
# ──────────────────────────────────────────────────────────────────────────
#
# Expected total time on RunPod GPU: ~2–3 hours
#   Phase 0 (git pull):  ~1 min
#   Phase 1 (train):    ~60 min  (4 tasks × 5 epochs, TextNavRouter only)
#   Phase 2 (eval):     ~30 min  (λ sweep × 4 tasks, INFERENCE ONLY)
#   Phase 3 (analyze):  ~15 min  (router similarity + cross-task acc)
#   Phase 4 (git push): ~2 min
#
# Usage:
#   bash run_zeronav_runpod.sh          # launches tmux session
#   tmux attach -t zeronav              # attach and watch progress
#
# ──────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="/workspace/src/navipath"
VENV="${REPO_DIR}/.venv/bin/activate"
FOLD=1
ORDER="reverse"
CKPT="outputs/qpmil_reverse_fold1.pt"
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

# ── Window 0: Phase 0+1 — git pull + train ───────────────────────────────
# Pulls latest code from GitHub, then trains TextNavRouter on 4 tasks.
# prototype_features() is NEVER called. Text-patch alignment only.
# Saves skill bank checkpoints every task + final to outputs/zeronav/
tmux send-keys -t zeronav:0 "
cd ${REPO_DIR} && source ${VENV}

echo '=== Phase 0: git pull ==='
git pull origin main
echo 'git pull done'

echo ''
echo '=== Phase 1: TRAIN TextNavRouter (fold ${FOLD}, order ${ORDER}) ==='
echo 'NOTE: Only TextNavRouter trains. CONCH backbone stays frozen.'
echo 'Saves: outputs/zeronav/skill_bank_${ORDER}_f${FOLD}.pt'
echo ''
python run_zeronav.py train \\
    --backbone-ckpt ${CKPT} \\
    --order ${ORDER} --fold ${FOLD} \\
    --epochs 5 --top-k 64 --lr 5e-4 \\
    2>&1 | tee logs/zeronav_train_f${FOLD}.log

echo ''
echo '=== Phase 1 DONE ==='
echo 'Skill bank: outputs/zeronav/skill_bank_${ORDER}_f${FOLD}.pt'
echo ''
echo 'Now run Phase 2 (eval) in Window 1 — switch with Ctrl-b 1'
" Enter

# ── Window 1: Phase 2 — Inference-only λ sweep ───────────────────────────
# INFERENCE ONLY. Loads skill bank, runs 4 eval modes × 8 λ values × 4 tasks.
# zeronav_oneshot   : top-64 patches at once (one-shot baseline)
# zeronav_multishot : 4 rounds × 16 patches with MMR diversity (our method)
# zeroshot_oneshot  : ZeroSlide score + top-64 (no training)
# zeroshot_multishot: ZeroSlide score + SBO (no training)
# !! NO RETRAINING !! skill_bank is loaded from checkpoint !!
tmux send-keys -t zeronav:1 "
cd ${REPO_DIR} && source ${VENV}
echo 'Window 1: Phase 2 — INFERENCE ONLY λ sweep'
echo 'Paste this command AFTER Phase 1 training is complete:'
echo ''
echo '─────────────────────────────────────────────────────────'
echo 'python run_zeronav.py eval \\'
echo '    --backbone-ckpt ${CKPT} \\'
echo '    --order ${ORDER} --fold ${FOLD} \\'
echo '    --lambdas ${LAMBDAS} \\'
echo '    --budget ${BUDGET} --step-size ${STEP_SIZE} \\'
echo '    2>&1 | tee logs/zeronav_eval_f${FOLD}.log'
echo '─────────────────────────────────────────────────────────'
echo ''
echo 'This loads skill_bank_${ORDER}_f${FOLD}.pt and runs inference ONLY.'
echo 'budget=64, step_size=16 => 4 rounds of 16 patches (our multi-step SBO)'
echo '!! Do NOT add --epochs or any training flags here !!'
" Enter

# ── Window 2: Phase 3 — Router analysis (inference only) ─────────────────
# Loads skill bank, computes:
#   1. Router weight cosine similarity matrix (task_i vs task_j weight vectors)
#   2. Cross-task accuracy matrix (router_i on task_j slides)
# Addresses teacher: "拿Router的Output去和TASK_ID算Similarity"
# !! NO RETRAINING !!
tmux send-keys -t zeronav:2 "
cd ${REPO_DIR} && source ${VENV}
echo 'Window 2: Phase 3 — Router analysis (INFERENCE ONLY)'
echo 'Paste this command AFTER Phase 1 training is complete:'
echo ''
echo '─────────────────────────────────────────────────────────'
echo 'python run_zeronav.py analyze \\'
echo '    --backbone-ckpt ${CKPT} \\'
echo '    --order ${ORDER} --fold ${FOLD} \\'
echo '    2>&1 | tee logs/zeronav_analyze_f${FOLD}.log'
echo '─────────────────────────────────────────────────────────'
echo ''
echo 'Expected result: diagonal=high, off-diagonal=low in acc matrix'
echo '(shows each task router is task-specific — Router vs TASK_ID)'
" Enter

# ── Window 3: Phase 4 — git push results ─────────────────────────────────
# After all phases complete, push outputs/zeronav/ to GitHub.
# Mac can then git pull to get all results.
tmux send-keys -t zeronav:3 "
cd ${REPO_DIR} && source ${VENV}
echo 'Window 3: Phase 4 — git push results to GitHub'
echo 'Run AFTER Phases 2 and 3 are complete:'
echo ''
echo '─────────────────────────────────────────────────────────'
echo 'cd /workspace/src/navipath'
echo 'git add outputs/zeronav/'
echo 'git add logs/zeronav_train_f1.log logs/zeronav_eval_f1.log logs/zeronav_analyze_f1.log 2>/dev/null || true'
echo 'git commit -m \"results(zeronav): ZeroNav fold1 reverse λ-sweep 0.0-2.0\"'
echo 'git push origin main'
echo '─────────────────────────────────────────────────────────'
echo ''
echo 'Then on Mac: git pull origin main'
echo 'Results will be in outputs/zeronav/eval/*.json'
" Enter

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " TMux session 'zeronav' created."
echo " Attach with: tmux attach -t zeronav"
echo ""
echo " Window 0 (main):    RUNNING — git pull + training TextNavRouter"
echo " Window 1 (eval):    WAITING — paste eval command after training"
echo " Window 2 (analyze): WAITING — paste analyze command after training"
echo " Window 3 (git):     WAITING — paste git push after all done"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "IMPORTANT: eval and analyze are INFERENCE ONLY (no retraining)"
echo "Skill bank checkpoint: outputs/zeronav/skill_bank_reverse_f1.pt"
echo ""
echo "After Mac git pull, extra inference on RunPod (no retrain):"
echo "  python run_zeronav.py eval \\"
echo "      --backbone-ckpt ${CKPT} \\"
echo "      --skill-bank-in skill_bank_reverse_f1.pt \\"
echo "      --order ${ORDER} --fold ${FOLD} \\"
echo "      --lambdas 0.0,0.5,1.0  --eval-tasks 0,1,2,3"
echo "  (loads saved checkpoint, runs specified λ values, NO training)"
