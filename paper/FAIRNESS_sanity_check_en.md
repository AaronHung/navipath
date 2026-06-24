# Fairness & Reproducibility Statement (Supplementary)

*On the ConSlide → QPMIL-VL pivot and the controlled comparison protocol.*

This note documents (i) why we pivoted the experimental backbone from ConSlide to
QPMIL-VL, (ii) why this pivot **strengthens** rather than weakens fairness, and
(iii) the concrete sanity checks that establish a controlled comparison. It is
intended as supplementary/rebuttal material.

---

## 1. The fairness concern, stated precisely
Fairness in a comparative study reduces to a single requirement: **the baseline
and the proposed method must be produced under identical, controllable, and
reproducible conditions.** If a baseline number is obtained by *our* re-implementation
of a method whose original execution conditions we cannot faithfully reproduce, then
that number conflates the method's true capability with the quality of our
re-implementation. Such a comparison is not defensible.

Our advisor raised exactly this concern about comparing against **ConSlide**: we
could not fully replicate its execution conditions, and we lacked the required data
preparation. Any ConSlide number we reported would therefore be an unverifiable
artifact of our own re-implementation.

---

## 2. Why a ConSlide comparison is not a fair (or feasible) anchor
Three independent reasons (recorded in our internal pivot note,
`Navipath_moe_plan_v01.md` §1.4):

1. **Reproducibility / verifiability gap.** ConSlide uses a hierarchical
   architecture (HIT) whose region-level attention requires attention rollout.
   We cannot faithfully reproduce this, and a reviewer can reasonably challenge
   whether our reproduced region-level signal is faithful. An unverifiable
   re-implementation cannot serve as a fair anchor.
2. **Different problem setting (buffer-based vs. replay-free).** ConSlide is
   **buffer-based** (it stores past data); our method is **replay-free** (it stores
   nothing). These are different rules of the game; a head-to-head comparison would
   be apples-to-oranges.
3. **Missing data / unaligned conditions.** The data and preprocessing needed to
   match ConSlide's setup are not available to us, whereas QPMIL-VL provides
   official public code and pre-computed CONCH features, enabling an exact,
   official-config run.

**Decision.** Rather than report an unverifiable ConSlide number, we retain
ConSlide as a **cited baseline** (discussed in Related Work, with the setting
difference made explicit) and move the experimental backbone to QPMIL-VL, where
every condition is under our control. This *raises* rigor; it does not avoid
comparison.

---

## 3. Why the QPMIL-VL setting is fair: a controlled comparison
Fairness is operationalized as a **controlled experiment**: hold every factor fixed
and vary only the factor under study. In our study the **only** variable is *who
selects the patches* (the learned router vs. random / prototype / semantic
heuristics). Everything else is identical:

| Factor | Baseline vs. ours | Why this guarantees fairness |
|---|---|---|
| Image encoder | Same: frozen CONCH; features extracted once and shared | No stronger encoder is silently substituted |
| Predictor (backbone head) | Same: QPMIL prompt-MIL head, **frozen** | All selectors feed the **same** frozen head; any accuracy gap is attributable to selection only |
| Data and splits | Same: four TCGA cohorts, official fold splits, both task orders, 3 folds | No favorable-split cherry-picking |
| Training budget | Same: **12 epochs/task, Adam lr 1e-3, weight decay 5e-4** | No "under-trained baseline" advantage |
| Evaluation protocol | Same: identical test sets, budgets $K$, and GO/NO-GO criterion | All selectors judged identically |

Because only the selector changes, the observed effect—the router selects well on
the most-recent task (GO) but falls below random on old tasks (NO-GO)—is
attributable to selection alone. This is the definition of a fair comparison.

---

## 4. Sanity checks performed (with verifiable evidence)
Each item maps to an artifact in the repository.

1. **Matched training budget (most important).**
   `QPMIL-VL/configs/main.yaml`: `epochs: [12, 12, 12, 12]`, `adam_lr: 0.001`,
   `adam_weight_decay: 0.0005`. The **same config** drives both the QPMIL baseline
   and the NaviPath backbone. This directly forecloses the most common reviewer
   objection (epoch/optimizer mismatch).
2. **Same features and data.** `main.yaml`:
   `dataset_names: [tcga_lung, tcga_brca, tcga_rcc, tcga_esca]`,
   `dataset_label_shift: [0,2,4,6]`, CONCH `feats-l1-s256` features. Baseline and
   ours read the identical feature tensors.
3. **Same frozen predictor for all selectors.** (Paper §3.5 / Fig. S1.) The router
   and the three training-free heuristics each select a Top-$K$ subset and feed the
   *same* frozen backbone; only the selected set differs.
4. **Honesty checks (self-disclosed).** We report `Forgetting = 0` as a *structural
   identity* of the decoupled frozen backbone (Paper §4.2, §5, Fig. 6), not as a
   contribution; and we do **not** claim an accuracy win over QPMIL (QPMIL ACC
   0.924/0.917 $\ge$ ours 0.879/0.886). We therefore have no incentive to weaken the
   baseline—our contribution is the selection analysis, not accuracy.

---

## 5. Anticipated questions and responses
- **Q: Did you switch backbones to avoid a hard comparison with ConSlide?**
  No. ConSlide is buffer-based (stores past data), a different setting from our
  replay-free method, and its region-level attention is not reliably reproducible
  by us. Reporting a self-reimplemented ConSlide number would be *less* fair. We
  keep ConSlide as a cited baseline and move to a setting (QPMIL-VL) where the
  comparison is fully controlled and reproducible.
- **Q: Did you handicap QPMIL to win?**
  No. We use the official config and features; the baseline uses the same 12
  epochs/task and the same lr/wd as our method. We also do not compete on accuracy
  (QPMIL's ACC exceeds ours).
- **Q: Is `Forgetting = 0` inflated?**
  It is an identity, not a result: a frozen predictor that never consumes
  router-modified features has flat per-task accuracy by construction. We surface
  this explicitly (Fig. 6); the contribution is the selection analysis.
- **Q: With only ~15 esca test slides, is the conclusion reliable?**
  We added a same-task recency-flip test and show the same collapse (0.92→0.40) on
  the sample-abundant lung task (~760 slides/task), excluding sample-size confounds;
  it replicates across all 6 fold×order runs.
- **Q: Why not just include ConSlide in the table anyway?**
  We can include it as a "cited / best-effort reproduction" with the setting
  difference flagged, but it cannot serve as our core controlled anchor without
  reopening the fairness problem above.

---

## 6. Summary
We pivoted from ConSlide to QPMIL-VL to convert an unverifiable comparison into a
controlled, reproducible one: identical encoder, frozen predictor, data splits,
**training budget (12 epochs, lr 1e-3, wd 5e-4)**, and evaluation—varying only the
patch selector. We do not compete on accuracy and we disclose `Forgetting = 0` as
an identity. ConSlide is retained as a cited baseline with its replay-free vs.
buffer-based setting difference made explicit.

*Evidence:* `QPMIL-VL/configs/main.yaml` (epochs/lr/wd); paper `paper_body.tex`
§3.5/§4.0/§4.2; `Navipath_moe_plan_v01.md` §1.4 (pivot record);
`ONBOARDING_runbook.md` (reproducible pipeline).
