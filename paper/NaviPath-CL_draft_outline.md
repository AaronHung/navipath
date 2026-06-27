# NaviPath-CL — paper rewrite outline (NaviPath-CL framing)

> Markdown 大綱（SPEC-05）。**不覆寫** `main.tex` / `paper_body.tex`；此檔供 7/3–7/15 移植回 LaTeX。
> Paper framing = continual learning of the observation policy under budgeted WSI inference.
> Internal note: do NOT put North Star / project-identifying details in the paper.
> 數字/圖來自凍結的 Phase-0 pilot；**TODO** = 7/15 前補。

**Title**: NaviPath-CL: A Continual Navigation Layer for Agentic Whole-Slide Image Diagnosis

---

## Abstract

Gigapixel WSIs contain thousands of patches; practical, physician-like diagnosis observes only
a small *budget* of patches via an **observation/navigation policy** (where to look). We study an
overlooked problem: under a continual stream of WSI tasks, this navigation policy itself suffers
catastrophic forgetting — a model that learns to look well for a new cancer type "forgets how to
look" at old ones, independent of the diagnostic backbone. We propose a **Continual Navigation
Layer (CNL)** — a backbone-agnostic layer learning a budgeted patch-selection policy on a frozen
VL diagnostic backbone — and a **Navigation Skill Memory (NSM)** preserving per-task navigation
skills. With QPMIL-VL as a frozen prompt/prototype-based backbone and weak signal, a
mechanism-selection study on TCGA shows a single shared policy and EWC regularization are
insufficient to retain old-task navigation, while modular NSM recovers it, surpassing strong
training-free selectors. **TODO**: task-free gating + consolidation results.

## 1. Introduction

- WSI MIL/VL; gigapixel → budgeted patch observation is intrinsic, reframed as a navigation policy
  (not a compute-saving trick).
- WSI continual learning has targeted the **classifier**; we show the **navigation policy** forgets.
- Contributions: (1) formalize continual learning of the observation policy under budgeted WSI
  inference; (2) CNL (backbone-agnostic) + NSM (per-task skill memory); (3) mechanism-selection
  study — modular skill memory recovers old-task navigation.

## 2. Related Work

- WSI MIL + VL (CONCH; prompt/prototype-based VL, QPMIL-VL).
- CL families (QPMIL-VL taxonomy): regularization (EWC, LwF), rehearsal/replay (ER, DER++,
  ConSlide), prompt/prototype-based (L2P, S-Prompts, QPMIL-VL — rehearsal-free).
  **Positioning**: QPMIL-VL is a backbone *instance*, not a competitor; CNL is orthogonal,
  backbone-agnostic.
- Budgeted/agentic visual inference, hard attention, glimpse policies. **TODO** refs.

## 3. Method

- **3.1 Backbone interface (backbone-agnostic)**: `encode(WSI)->Z`, `predict(subset)->logits`,
  optional `task_query(WSI)->q`. Instantiated with QPMIL-VL (frozen CONCH 512-d; prototype-agg +
  class-text classifier). Maps to `navipath_moe/qpmil_adapter.py` 4 hooks.
- **3.2 CNL**: navigation policy φ: patch (+ text/proto summary) → score → Top-K under budget K;
  φ is the only trained part (`navipath_moe/routers.py::MicroRouterV0`).
- **3.3 NSM**: per-task navigation skills + context gate (oracle = upper bound here; task-free via
  `task_query` is future). Mechanisms compared: shared / EWC / per-task NSM
  (`navipath_moe/continual_agent.py`, `eval_continual_agent.py`).

## 4. Experiments

- **4.1 Setup**: TCGA tasks, two orders (paper/reverse), k folds, budgets {32,64,128,256,All};
  frozen QPMIL-VL backbone; baselines = training-free selectors (random/prototype/semantic) feeding
  the SAME frozen backbone (so any gap is attributable to selection).
- **4.2 Mechanism-selection (main)** — old ESCA, reverse order, budget 64, mean over folds:

  | Mechanism | old-task ACC@64 | GO |
  |---|---|---|
  | shared (single policy) | 0.333 | 0/3 |
  | EWC (weight reg.) | 0.400 | 0/3 |
  | per-task NSM | **0.933** | 3/3 |
  | best heuristic | 0.844 | — |

  Decision tree: Q1 shared cannot retain all tasks; Q2 EWC insufficient; Q3 old skill not lost
  (NSM recovers > heuristic) ⇒ bottleneck = continual navigation memory, not diagnostic signal.
  Figures: `outputs/figs/Fig1_arch.png`, `Fig_mechanism.png`, `Fig_budget_curve.png`.
  **TODO**: paper-order symmetry; end-to-end agent reproduction (0.933) on RunPod; optional
  task-free gate / consolidation.

## 5. Discussion / Limitations

- Oracle context gate (upper bound); task-free gating future.
- Per-task NSM grows with #tasks; consolidation / parameter-merging ongoing.
- Phase-0 abstracts navigation as patch selection over precomputed features; move/zoom/trajectory
  and RLHF out of scope. No compute-saving claim (budget = observation constraint).

## 6. Conclusion

Navigation policies forget under WSI continual learning; modular NSM recovers old-task observation
behavior. CNL is a backbone-agnostic basis for future agentic WSI diagnosis.

---

*Provenance: `specs/` (ADR/SPEC/WORKLOG), `STORYLINE.md`, `reports/bimonthly_2026-07-03.md`.*
