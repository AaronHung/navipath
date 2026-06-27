# NaviPath — Selection Forgetting in Continual WSI Classification

**Continual Patch Routing Forgets Old Tasks in Frozen-Foundation-Model
Whole-Slide Image Classification**

Research codebase — target venue: **COMPAYL @ MICCAI 2026** (workshop).
Backbone: [QPMIL-VL](https://github.com/can-can-ya/QPMIL-VL) (vendored) on frozen
[CONCH](https://github.com/mahmoodlab/CONCH) features.

---

## TL;DR

Freezing a pathology foundation model removes the classic source of catastrophic
forgetting (the predictor cannot drift). But under a patch **budget**, a model must
still learn *which* patches to look at — and **that learned selector forgets**.

We attach a lightweight trainable **patch router** to a frozen CONCH+QPMIL backbone.
Because the predictor never consumes router-modified features (backbone forgetting
is `0` *by construction*), any degradation is attributable to **selection** alone.
We find:

- **Recent tasks:** the router beats random / prototype / semantic heuristics at
  tight budgets (**+2.5–6.7 pp @ K=64; 6/6 GO** across folds and orders).
- **Old tasks:** its selection quality collapses **below random** (**6/6 NO-GO**) —
  we call this **selection forgetting**.
- **Cause is recency, not difficulty/size:** a *same-task recency-flip* test drops
  accuracy from ~0.9 to 0.33–0.40, and holds even for the sample-abundant lung task.
- **Mechanism:** the forgotten router *confidently mis-prioritizes* (selects
  later-task-salient patches) rather than degrading to noise.
- **Recoverable in principle, but not by EWC:** a per-task router restores old-task
  selection **0.33 → 0.93 (3/3 GO)**; EWC-on-router does not (**0.40, 0/3**).

> **Honesty note.** `Forgetting = 0` is a *structural identity* of the decoupled
> frozen backbone, **not a contribution**. We do not claim an accuracy win over
> QPMIL (QPMIL ACC ≥ ours). Our contribution is the **analysis of the selection
> path**. See `paper/` and `FAIRNESS_sanity_check_zh.md` / `paper/FAIRNESS_sanity_check_en.md`.

---

## Key results

**Router selection (accuracy @ budget K=64, mean over 3 folds).** GO = router beats
the best heuristic at K=64.

| Task | Recency | router@64 | best heuristic@64 | verdict |
|---|---|---|---|---|
| esca | recent (learned last) | **0.956** | 0.889 | GO (3/3) |
| lung | recent (learned last) | **0.922** | 0.897 | GO (3/3) |
| esca | old (learned first)   | **0.333** | 0.822 | NO-GO (0/3) |
| lung | old (learned first)   | **0.397** | 0.813 | NO-GO (0/3) |

**Mitigation on the oldest task (esca, 3 folds).**

| Router consolidation | @64 | verdict |
|---|---|---|
| none (continual)             | 0.333 | NO-GO (0/3) |
| EWC-on-router (replay-free)   | 0.400 | NO-GO (0/3) |
| per-task router (upper bound) | **0.933** | **GO (3/3)** |

Full curves/tables and figures: `paper/paper_body.tex`, `outputs/figs/`.

---

## What to read first

| You want to… | Go to |
|---|---|
| Read the paper | `paper/paper_body.tex` (+ `references.bib`, `figs/`) |
| Run / re-run any experiment | **`ONBOARDING_runbook.md`** |
| Rebuttal ammunition | `paper/REVIEW_rebuttal.md` |
| Why we pivoted ConSlide→QPMIL (fairness) | `FAIRNESS_sanity_check_zh.md` / `paper/FAIRNESS_sanity_check_en.md` |
| Original project history (MoE/M0–M9) | `git log`, `Navipath_moe_plan_v01.md` |

---

## Repository structure

```
paper/                     Overleaf-ready paper: paper_body.tex, references.bib, figs/,
                           REVIEW_rebuttal.md, FAIRNESS_sanity_check_en.md
ONBOARDING_runbook.md      How to run everything + task chain + known cleanups
FAIRNESS_sanity_check_zh.md  ConSlide->QPMIL fairness rationale (zh)
QPMIL-VL/                  Vendored backbone (upstream can-can-ya/QPMIL-VL @ 3a7a769)
navipath_moe/             Router / experts / losses / qpmil_adapter (our modules)
configs/*.yaml            NaviPath ablation configs
train_qpmil_runner.py     Step A: QPMIL baseline + frozen backbone ckpt
train_navipath.py         Step B: NaviPath decoupled
train_router_v0.py        Step C/D: router training + patch-budget eval + Plan B
tools/                    collect_results.py, plot_results.py, draw_arch.py
outputs/                  Tracked results (*.json/*.pt) and figs/
outputs_history/          Historical evidence (incl. early buggy runs)
```

---

## Setup

QPMIL-VL is **vendored** in this repo, so `git clone` gives you the code directly.
Two things are **not** in the repo and must be obtained separately:

1. **CONCH weights** (~1 GB, gitignored as `*.bin`): download from
   [`MahmoodLab/CONCH`](https://huggingface.co/MahmoodLab/CONCH) (HF access required)
   and place at the path set in `QPMIL-VL/configs/main.yaml: conch_ckpt_path`.
2. **TCGA CONCH features / data**: the prepared per-cohort features; point
   `dataset_root_dir` (and `class_ensemble_path`) in `main.yaml` to them.

> ⚠️ `QPMIL-VL/configs/main.yaml` contains **absolute paths**
> (`dataset_root_dir`, `conch_ckpt_path`, `class_ensemble_path`). **Edit these three
> for your machine.** Do **not** change hyperparameters (`epochs`, `adam_lr`,
> `adam_weight_decay`, `pool_size`, …) — they are the fairness baseline.

Python deps (RunPod PyTorch image already has torch):

```bash
pip install "transformers>=4.40,<5" huggingface-hub==0.36.2 \
    timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml
# do NOT use QPMIL-VL/requirements.txt (it pins conda-only packages)
```

See `RUNPOD_SETUP.md` for a one-line setup after reboots.

---

## Quickstart (one fold)

```bash
ORDER=reverse; FOLD=1
# A. QPMIL baseline + frozen backbone ckpt  -> Table 1, Fig 6
python train_qpmil_runner.py --order $ORDER --fold $FOLD --epochs 12 --save-ckpt
# B. NaviPath decoupled                     -> Table 1
python train_navipath.py --config configs/navipath_full.yaml --order $ORDER --fold $FOLD --save-ckpt
# C. Router + patch-budget (recent + oldest)-> Table 2, Figs 2-4
python train_router_v0.py --backbone-ckpt outputs/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --eval-tasks="-1,0" --epochs 5
# D. Plan B (mitigation)                    -> Table 3
python train_router_v0.py --backbone-ckpt outputs/qpmil_${ORDER}_fold${FOLD}.pt \
    --order $ORDER --fold $FOLD --eval-tasks="-1,0" --epochs 5 --router-consol pertask
```

Full details, all folds/orders, aggregation and plotting: **`ONBOARDING_runbook.md`**.

---

## Method (one paragraph)

A slide is a bag of frozen CONCH patch features. A two-layer MLP **MicroRouter**
(~132K params) scores each patch from its feature plus four task-count-invariant
similarity statistics; we keep the Top-K. It is trained continually with a
differentiable soft-route objective while the QPMIL prediction head stays **frozen**
and only ever sees *original* features. We compare the router against training-free
selectors (random / prototype / semantic) under budgets K ∈ {All,256,128,64,32},
over two task orders (paper / reverse) and 3 folds, reporting Top-1 accuracy and a
GO/NO-GO criterion. For mitigation we add replay-free **EWC-on-router** and a
**per-task router** upper bound. See `paper/paper_body.tex` §3 for equations.

---

## Acknowledgements & citation

This work builds directly on:

- **QPMIL-VL** — Gou, Ji, Liu, Ye. *Queryable Prototype Multiple Instance Learning
  with Vision-Language Models for Incremental WSI Classification.* AAAI 2025.
  (Vendored under `QPMIL-VL/`; see its `LICENSE`.)
- **CONCH** — Lu et al. *A visual-language foundation model for computational
  pathology.* Nature Medicine 2024.

Please cite these upstream works when using this repository. A `references.bib` with
all related citations is in `paper/`.

---

*Project note:* this codebase originally explored a MoE / "zero-forgetting" routing
framework (see git history and `Navipath_moe_plan_v01.md`). It pivoted to the focused
**selection forgetting** analysis above; the paper in `paper/` is the authoritative
current statement of the work.
