# NaviPath-MoE

**Agentic Patch Routing for Continual Whole Slide Image Classification**

Research codebase — COMPAYL @ MICCAI 2026 (workshop submission target).  
Backbone: [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL) with frozen [CONCH](https://github.com/mahmoodlab/CONCH) features.

---

> ### ⚠️ Read this first — current state (Jun 2026)
> The project **pivoted** from a MoE/"zero-forgetting" framing to a focused
> analysis paper on **selection forgetting** (a trainable patch router forgets
> *what to look at* for old tasks, even when the frozen predictor cannot forget).
> The sections **below (M0–M9) are kept as historical record** and still describe
> the old MoE narrative and "guarantees zero forgetting" angle.
>
> **For the authoritative, current story, use these instead:**
> - 📄 Paper: [`paper/paper_body.tex`](paper/paper_body.tex) (+ `references.bib`, `figs/`)
> - 🏃 How to run / re-run anything: [`ONBOARDING_runbook.md`](ONBOARDING_runbook.md)
> - 🛡️ Rebuttal ammo: [`paper/REVIEW_rebuttal.md`](paper/REVIEW_rebuttal.md)
> - ⚖️ ConSlide→QPMIL fairness rationale: [`FAIRNESS_sanity_check_zh.md`](FAIRNESS_sanity_check_zh.md)
>
> **Important honesty note:** the paper reports `Forgetting = 0` as a *structural
> identity* of the decoupled frozen backbone (not a contribution). Where the text
> below sells "zero forgetting", read it through the paper's framing.
> **Setup:** QPMIL-VL is now vendored in this repo; CONCH weights are **not**
> (download separately, see `RUNPOD_SETUP.md`).

---

## Overview

Computational pathology models are increasingly deployed in clinical workflows that span multiple cancer types, but standard sequential fine-tuning suffers from **catastrophic forgetting**: accuracy on previously learned tasks degrades as new tasks are added. This project explores whether adding a **task-aware routing layer** on top of a frozen foundation model (CONCH/QPMIL-VL) can provide patch-level efficiency and specialization without forgetting.

### Core Research Question

> Can a trainable routing mechanism (MicroRouter + MacroRouter + ExpertBank) placed *above* a frozen pathology FM provide per-task patch selection efficiency while the frozen backbone itself guarantees zero forgetting?

---

## Architecture

```
Whole Slide Image (WSI)
        ↓
[Frozen CONCH encoder] ──→ Z: [N, 512]   (patch features, never modified)
        ↓
[MicroRouter]  scalar saliency score per patch  →  rank patches
[MacroRouter]  expert assignment weights        →  per-expert routing
        ↓
  top-K patch selection (K = 64/128/256/all)
        ↓ ←─────────────────────────────────┐
[ExpertBank]  4×MLP residual experts           │
 Z_exp = z + MLP_e(z)  (auxiliary only)        │
        ↓                                       │
  L_exp (cosine-sim loss)    ← trains experts  │
  L_soft_route (diff. path) ← trains router   │
  L_sem  (semantic anchor)                     │
  L_bal  (load balance)                        │
                                               │
[Frozen backbone.aggregate_and_predict(Z[idx])] ← original Z, not Z_exp
        ↓
      logits  →  L_C (classification)
```

**Dual-path design (key architectural decision):** The frozen backbone always receives original patch features `Z[idx]`. Expert-transformed features `Z_exp` are used only for auxiliary training losses and never affect the prediction path. This was introduced after discovering that feeding expert outputs into the frozen backbone caused severe catastrophic forgetting (Forgetting = 0.73).

---

## Milestones: Plan & Results

### M0 — Environment & Codebase Mapping
**Goal:** Set up reproducible environment, map QPMIL-VL internals, build adapter hooks.

| Item | Status |
|---|---|
| QPMIL-VL adapter (`navipath_moe/qpmil_adapter.py`) | ✅ |
| CONCH checkpoint integration | ✅ |
| Device-agnostic runner (CUDA > MPS > CPU) | ✅ |
| `CODEBASE_MAP.md` | ✅ |

---

### M1 — QPMIL-VL Continual Learning Baseline
**Goal:** Reproduce QPMIL-VL under class-incremental continual learning (4 tasks, 2 task orders), establishing the baseline forgetting numbers.

**Tasks:** `tcga_lung → tcga_brca → tcga_rcc → tcga_esca` (paper order)  
**Reverse:** `tcga_esca → tcga_rcc → tcga_brca → tcga_lung`

| Order | Fold | ACC | Forgetting | BWT | Notes |
|---|---|---|---|---|---|
| paper | 1 | **0.887** | 0.018 | −0.018 | Full run, 12 ep/task |
| reverse | 1 | 0.594 | 0.042 | −0.042 | Smoke test (1 ep, 32 slides) |

**Accuracy matrix (paper order, fold 1):**
```
         lung   brca   rcc   esca
task 1  [0.863   —      —     — ]
task 2  [0.853  0.925   —     — ]
task 3  [0.853  0.925  0.947  — ]
task 4  [0.811  0.925  0.947  0.867]
```

Forgetting on `lung` after 4 tasks: 0.863 → 0.811 (−0.052). Real but modest.

**Script:**
```bash
python train_qpmil_runner.py --order paper   --fold 1 --save-ckpt
python train_qpmil_runner.py --order reverse --fold 1 --save-ckpt
```

---

### M2 — CONCH Feature Extraction Verification
**Goal:** Confirm frozen CONCH features are identical across calls (no stochasticity), validate `encode_patches` shapes.

| Check | Result |
|---|---|
| Feature shape | `[N, 512]` ✅ |
| Determinism | ✅ |
| MPS/CUDA compatibility | ✅ |

---

### M3 — Patch-Budget Baseline Evaluation
**Goal:** Establish how well **random / prototype / semantic** patch selection performs at various budgets K ∈ {32, 64, 128, 256, All} using the frozen QPMIL-VL backbone. This is the baseline the router must beat.

**Script:**
```bash
python run_patch_budget.py --ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --task-index 0
```

---

### M4 — MicroRouterV0 (Scalar Patch Scoring)
**Goal:** Train a lightweight scalar router that assigns saliency scores to patches. First test of whether a trainable router can outperform random selection.

**Architecture:** CONCH feature → linear → sigmoid → scalar score per patch.  
**Training signal:** `L_sem` (cosine similarity to class text features) + differentiable soft-weighted aggregation.

**Patch-budget eval (paper order, fold 1, lung task only — single task probe):**

| Method | All | @256 | @128 | @64 | @32 |
|---|---|---|---|---|---|
| **Router V0** | 0.875 | **1.000** | **1.000** | **1.000** | **1.000** |
| Random | 0.875 | 0.875 | 1.000 | 0.875 | 0.750 |
| Prototype | 0.875 | 0.875 | 0.875 | 0.750 | 0.750 |
| Semantic | 0.875 | 1.000 | 0.750 | 0.750 | 0.750 |

**Result: GO ✓** (router consistently matches or beats random across budgets on the lung probe set).

**⚠️ Caveat:** This is a single-task evaluation on a small test set (~8 slides). The signal is encouraging but not generalizable until verified across all 4 tasks and multiple folds.

**Script:**
```bash
python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1 --epochs 5
```

---

### M5 — NaviPath-MoE v1 (MicroRouter + ExpertBank)
**Goal:** Add `ExpertBank` (4 × MLP residual experts) and `MicroRouter` to the continual learning loop.

#### Critical Bug: Expert-Backbone Interference → Catastrophic Forgetting

**First implementation (wrong):** Expert-transformed features `Z_exp` were fed into `backbone.aggregate_and_predict()`.

```
WRONG: Z → Expert → Z_exp → frozen_backbone(Z_exp) → logits
```

The backbone was trained on raw CONCH features. Expert transformations from task T distorted task (T−1) features → backbone misclassified old tasks → **fake catastrophic forgetting**.

| Run | ACC | Forgetting | Notes |
|---|---|---|---|
| M5 paper (buggy) | 0.378 | **0.735** | Expert corruption |
| M5 reverse (buggy) | 0.218 | **0.950** | Expert corruption |

**Fix — Dual-path Architecture:**
```
CORRECT: Z ──→ frozen_backbone(Z[idx]) ──→ logits (L_C)
              └→ Expert → Z_exp ──────────→ L_exp (auxiliary only)
```

After fix (full 4-task run on RunPod):

| Config | Order | ACC | Forgetting | BWT |
|---|---|---|---|---|
| M5 (micro only) | paper | 0.857 | 0.000 | 0.000 |
| M5 (micro only) | reverse | 0.852 | 0.000 | 0.000 |

---

### M6 — MacroRouter
**Goal:** Add a `MacroRouter` that assigns patches to specific experts based on morphological features, enabling task-specialized routing.

Controlled by `use_macro: true` in `configs/navipath_full.yaml`.

---

### M7 — Semantic Anchor Loss (`L_sem`)
**Goal:** Train the router using CONCH's text embeddings as supervision — patches that are semantically close to the task's class descriptions should receive higher scores.

Loss: `L_sem = −cos(router_weighted_aggregate, f_txt)`  
Controlled by `gamma` in `loss_weights`.

---

### M8 — Replay-Free Momentum Consolidation
**Goal:** Protect expert parameters from task-to-task drift using importance-weighted consolidation (no stored samples required).

After task T, expert importance `m_e[k]` is computed as the mean activation magnitude of expert k. New expert weights are consolidated toward the previous snapshot weighted by importance.

**Full NaviPath-MoE results (M5+M6+M7+M8, RunPod, 4-task):**

| Config | Order | ACC | Forgetting | BWT |
|---|---|---|---|---|
| NaviPath-MoE (full) | paper | 0.857 | 0.000 | 0.000 |
| NaviPath-MoE (full) | reverse | 0.852 | 0.000 | 0.000 |

M5 and M8 produce **identical** CL summaries (adding 3 modules changed nothing). See Critical Findings below.

**Script:**
```bash
python train_navipath.py --config configs/navipath_full.yaml \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1 --save-ckpt
```

---

### M9 — Routing Drift Visualization
**Goal:** Visualize whether different tasks activate different experts (evidence of specialization).

**Routing weight table — paper order, fold 1:**

| Task | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| tcga_lung | 0.257 | 0.155 | 0.166 | **0.422** |
| tcga_brca | 0.123 | 0.071 | 0.135 | **0.671** |
| tcga_rcc | 0.245 | 0.258 | 0.220 | 0.278 |
| tcga_esca | 0.208 | 0.142 | 0.170 | **0.479** |

Mean inter-task L1 drift: **0.290** (paper) / **0.356** (reverse) — signal PRESENT in final run.

Expert E4 dominates for lung and brca, while rcc shows more uniform routing. This pattern suggests the router does learn task-differentiated patch selection behavior, even if it doesn't yet translate to ACC improvement.

**Script:**
```bash
python viz_routing_drift.py \
    --ckpt outputs/navipath_full_paper_fold1.pt \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1
```

---

## Critical Self-Diagnosis

### Finding: Forgetting = 0 is a Mathematical Identity, Not a Method Result

After completing M9, a systematic code-level analysis revealed that the `Forgetting = 0.000` numbers reported by M5–M8 are **guaranteed by the architecture**, not earned by any learning mechanism.

**Proof chain:**

1. **Backbone is frozen** (`requires_grad_(False)` after loading M1 checkpoint).
2. **Eval uses `top_k=0`** → `top_k_select(score, 0)` returns all patches → router scores have zero effect during evaluation.
3. **Expert outputs (`Z_exp`) never enter `logits`** → experts are purely auxiliary.
4. → Backbone input is identical across all tasks at eval time → `R[t, i] ≡ R[t', i]` for all `t, t'` → **Forgetting ≡ 0 by definition.**

**Smoking gun:** M5 (micro router only) and M8 (micro + macro + L_sem + consolidation) produce byte-for-byte identical CL summaries. Three additional architectural modules → zero change in metrics = textbook no-op signature.

**Implication for ACC:** The accuracy numbers (0.857 paper, 0.852 reverse) reflect the frozen QPMIL-VL checkpoint evaluated on all 4 tasks — they are not influenced by NaviPath routing or experts.

### Finding: Patch-Budget Router Does Not Consistently Beat Random

In the 4-task full evaluation (RunPod), the router underperformed random at most budgets:

| Budget | Router | Random | Δ |
|---|---|---|---|
| @256 | 0.733 | 0.933 | −0.200 |
| @128 | 0.733 | 0.800 | −0.067 |
| @64 | 0.667 | 0.733 | −0.067 |

The M4 single-task probe showed a strong GO signal (router = 1.000 at @32–256), but this did not generalize to the full 4-task continual setting. The router trained in a single-task probe environment may not have learned to discriminate patches well enough for multi-task evaluation.

### Root Cause: Evaluation-Training Mismatch

The `L_soft_route` loss (soft-weighted cosine classification during training) provides a differentiable gradient signal to the router. However, at inference time the router's top-K selection is evaluated against the frozen backbone using all patches at `top_k=0`. This mismatch means the router never "practices" the exact inference task it needs to solve.

---

## What Genuinely Stands Up

Despite the above findings, three results are real and worth reporting:

1. **Expert-backbone interference is a real failure mode.** Feeding trainable expert outputs into a frozen FM backbone causes severe catastrophic forgetting (Forgetting = 0.735–0.950). This is a concrete, reproducible finding for the community.

2. **Frozen-FM continual learning is already very robust.** QPMIL-VL (frozen CONCH + trainable prompt/key) achieves ACC = 0.887 with only Forgetting = 0.018 on the 4-task paper order — sequential fine-tuning of the FM is not even needed.

3. **Task-differentiated routing behavior exists.** The routing drift signal (mean L1 = 0.29–0.36) shows the router learns distinct patch-selection strategies per task. Whether this translates to efficiency gains requires further validation with `top_k > 0` at eval.

---

## Future Directions

### Near-Term (needed for any publication)

**1. Honest eval: use `top_k=K` at inference time**

Change `eval_accuracy(..., top_k=0)` → `eval_accuracy(..., top_k=128)`. This makes the router's patch selection consequential during evaluation, allowing a real measurement of routing quality. The claim then becomes: *"can K selected patches approximate full-slide accuracy?"* — an efficiency claim, not a forgetting claim.

**2. Fair baseline: match epochs**

Rerun QPMIL-VL baseline with the same `epochs=12/task` used for NaviPath. Current M1 reverse numbers used `epochs=1, max_train=32` (smoke test settings). A fair comparison requires identical training budgets.

**3. Router training fix: train in multi-task setting**

Router currently trained per-task independently. Retrain the router jointly across tasks, or use the CL training loop with `top_k=128` at eval to get a proper gradient signal. The single-task probe success (M4: all budgets WIN) suggests the router architecture is capable — it needs better multi-task training.

### Medium-Term

**4. Expert-in-prediction with parameter isolation**

Instead of the dual-path no-op, use per-task expert copies:
```
Z_exp_t → frozen when task T' ≠ T
Z[idx] → backbone(Z[idx] + gate_t * Z_exp_t)
```
With `gate_t` frozen for old tasks, experts contribute to prediction and forgetting is controlled. This is the original intended architecture.

**5. Multi-fold validation**

Run 3 folds × 2 orders = 6 complete runs for both M1 and NaviPath before any publication.

**6. Honest negative-result paper**

Current findings are publishable as an analysis paper:
> *"On Adding Trainable Routing to Frozen-Foundation-Model Continual WSI Classification: Interference, Decoupling, and the Limits of Patch Selection."*

The three findings above (interference failure mode, FM robustness, routing behavior) are honest and scientifically interesting. COMPAYL format suits this style.

---

## Repository Structure

```
.
├── navipath_moe/
│   ├── device.py           CUDA > MPS > CPU auto-select
│   ├── routers.py          MicroRouterV0, MicroRouter, MacroRouter
│   ├── experts.py          ExpertBank (4 × MLP residual)
│   ├── consolidate.py      Replay-free momentum consolidation (M8)
│   ├── losses.py           L_sem, L_bal, L_route
│   ├── qpmil_adapter.py    QPMIL-VL backbone hooks
│   └── qpmil_bootstrap.py  Env setup, param init
├── eval/
│   ├── metrics.py          ACC / Forgetting / BWT / UpperBoundRatio
│   └── patch_budget_eval.py random / prototype / semantic / router budget eval
├── configs/
│   ├── navipath_micro.yaml      MicroRouter + ExpertBank (M5)
│   ├── navipath_full.yaml       + MacroRouter + L_sem + Consolidation (M8)
│   ├── navipath_router_only.yaml  Ablation A: router only
│   ├── navipath_no_macro.yaml     Ablation B: no MacroRouter / Consolidation
│   └── navipath_no_consol.yaml    Ablation C: no Consolidation
├── QPMIL-VL/               Submodule (upstream backbone)
├── train_qpmil_runner.py   M1: QPMIL-VL continual learning runner
├── train_navipath.py       M5–M8: NaviPath-MoE training loop
├── train_cl_baselines.py   EWC and LwF baselines
├── train_router_v0.py      M4: scalar router pre-training
├── run_patch_budget.py     M3: patch-budget baseline eval
├── viz_routing_drift.py    M9: routing drift heatmap
├── run_all_experiments.sh  One-shot script for all experiments (for students)
├── outputs/                Results (JSON + checkpoints)
├── outputs_history/        Timestamped run archives
└── RUNPOD_SOP.md           RunPod setup guide
```

---

## Reproducing Results

### Environment (RunPod, RTX 4090 or later)

```bash
git clone https://github.com/AaronHung/navipath.git
cd navipath
# Follow RUNPOD_SOP.md for conda env setup
conda activate pt-exp
```

### Full Experiment Suite (one fold)

```bash
bash run_all_experiments.sh paper   1   # fold 1, paper order
bash run_all_experiments.sh reverse 1   # fold 1, reverse order
```

This runs: M1 → EWC → LwF → Ablation A/B/C → M8 NaviPath-MoE.

### Individual Scripts

```bash
# M1 baseline
python train_qpmil_runner.py --order paper --fold 1 --save-ckpt

# M8 NaviPath-MoE
python train_navipath.py --config configs/navipath_full.yaml \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt --order paper --fold 1 --save-ckpt

# EWC baseline
python train_cl_baselines.py --method ewc --order paper --fold 1 --save-ckpt

# Routing drift visualization (M9)
python viz_routing_drift.py \
    --ckpt outputs/navipath_full_paper_fold1.pt \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt \
    --order paper --fold 1
```

### Local Dev / Smoke Test (Mac MPS)

```bash
# Quick pipeline check (~5 min on M1 Mac)
python train_qpmil_runner.py --order paper --fold 1 --epochs 1 --max-train 40 --max-eval 20
python train_navipath.py --config configs/navipath_micro.yaml \
    --backbone-ckpt outputs/qpmil_paper_fold1.pt --order paper --fold 1 \
    --epochs 1 --max-train 32 --max-eval 16
```

---

## Key Configuration Parameters

| Parameter | Location | Meaning |
|---|---|---|
| `use_experts` | navipath_*.yaml | Enable ExpertBank (MLP residual) |
| `use_macro` | navipath_*.yaml | Enable MacroRouter (expert assignment) |
| `use_consolidation` | loss_weights in yaml | Enable momentum consolidation (M8) |
| `train_top_k` | navipath_*.yaml | Patches selected per slide during training |
| `gamma` | loss_weights | Weight of L_sem (semantic anchor) |
| `eta` | loss_weights | Weight of L_bal (expert load balance) |
| `zeta` | loss_weights | Weight of L_soft_route (differentiable routing) |
| `num_experts` | navipath_*.yaml | Number of experts in ExpertBank |

---

## CL Metrics Reference

| Metric | Formula | Meaning |
|---|---|---|
| **ACC** | mean of diagonal of final R | Average accuracy after learning all T tasks |
| **Forgetting** | mean of (max over training − final) per task | How much accuracy was lost on old tasks |
| **BWT** | mean of (final − when-first-learned) | Negative = forgetting; Positive = backward transfer |
| **UpperBoundRatio** | ACC / joint_train_acc | How close to joint-training upper bound |

---

## Citation / Acknowledgements

Built on top of:
- [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL) — Quantile-based Prompt MIL with Vision-Language alignment
- [CONCH](https://github.com/mahmoodlab/CONCH) — Pathology foundation model
- [TCGA](https://www.cancer.gov/tcga) datasets: NSCLC (lung), BRCA, RCC, ESCA

*Research conducted at [Institution], 2026.*
