# NaviPath-CL — Paper Outline (v2.0, 2026-06-30)

> **Status**: Active. v2.0 extends v1.x with Sequential Budgeted Observer (Route A),
> zero-shot navigation baseline (ZeroSlide-inspired), λ sweep results, and LoRA
> future work.  `main.tex` targets a venue-agnostic full paper; adapt header for
> MICCAI/COMPAYL submission.
>
> **Internal notes**:
> - Do NOT name QPMIL directly in the title/abstract/method — use
>   "prompt/prototype-based VL diagnosis backbone" or "frozen VL diagnosis backbone".
> - QPMIL-VL is cited as a backbone *instance*, not a baseline competitor.
> - Numbers from N2/N3 pilot (reverse order, fold 1-3, budget 64); λ sweep from N6.
> - ZeroSlide reference: `zeroslide` BibTeX key — verify final citation before submission.

---

## Title (current)
**NaviPath-CL: Continual Navigation of Whole-Slide Images via Budgeted Sequential Observation and Skill Memory**

*Alternative (shorter for workshop)*: **Selection Forgetting in Continual WSI Diagnosis: A Budgeted Sequential Observer with Navigation Skill Memory**

---

## Abstract (≈250 words)

Key claims to hit:
1. Selection forgetting problem — patch router forgets old tasks even when backbone is frozen.
2. Proposed solution: CNL = MicroRouter + Sequential Budgeted Observer (SBO) + NSM.
3. Zero-shot navigation baseline (frozen FM text-patch similarity, no training).
4. NSM (per-task skill memory) = upper bound, recovers mACC 0.595 → 0.935, Forgetting 0.
5. Sequential observation with MMR diversity (Route A) makes the selection truly adaptive;
   optimal λ reflects spatial clustering of diagnostic regions.
6. Future: LoRA cheap NSM closes memory gap.

---

## 1. Introduction  (~1 page)

- WSI → gigapixel → MIL; budgeted patch observation is intrinsic (*physician-like*).
- Frozen FM + prompt/prototype VL backbone: classification can be made CL-safe (frozen
  encoder → no representation drift). But **selection is still trainable and forgets**.
- Gap: selection forgetting under continual WSI tasks is unstudied.
- Our contributions:
  1. Formalize *selection forgetting* + show it is distinct from classifier forgetting.
  2. CNL framework: MicroRouter + SBO + NSM (backbone-agnostic).
  3. Sequential Budgeted Observer: truly adaptive multi-step selection via MMR.
  4. Zero-shot navigation baseline (ZeroSlide-inspired, training-free).
  5. NSM as principled upper bound; LoRA-NSM as scalable future direction.
  6. Empirical study on 4 TCGA tasks, 2 orders, 3 folds.

---

## 2. Related Work  (~0.75 page)

### 2.1 WSI MIL and Foundation Models
- ABMIL [Ilse 2018], CLAM [Lu 2021], TransMIL, DSMIL.
- Frozen encoders: CONCH [Lu 2024], UNI [Chen 2024], PLIP [Huang 2023], CLIP [Radford 2021].
- Prompt/prototype VL for WSI: cite VL diagnosis backbone (QPMIL-VL).

### 2.2 Continual Learning
- Regularization: EWC [Kirkpatrick 2017].
- Replay: iCaRL [Rebuffi 2017], A-GEM [Chaudhry 2019], DER++ [Buzzega 2020].
- Prompt CL: L2P [Wang 2022], DualPrompt [Wang 2022], CODA-Prompt [Smith 2023].
- Continual MIL: AKDPMP [Li 2025], ConSlide (if published).
- **Gap**: none study the patch *selector* as a forgetting locus.

### 2.3 Budgeted / Agentic Inference for WSI
- IPS [Bergner 2023], RLogist [Zhao 2023 AAAI] — RL-based observation.
- ZeroSlide [cite zeroslide] — zero-shot patch scoring via FM text similarity.
- Hard attention / glimpse policies.
- **Distinction**: we study *continual learning* of the selection policy, not efficiency alone.

### 2.4 Mixture-of-Experts and Parameter-Efficient Fine-Tuning
- MoE routing [Shazeer 2017, Fedus 2022] — routing inspiration.
- LoRA [Hu 2022] — planned efficient NSM; each task adds ≪1K params.

---

## 3. Method  (~2 pages)

### 3.1 Problem Setup
- Class-incremental tasks $t=1,\dots,T$ (WSI subtyping). Budget $K \ll n$.
- Decouple: *what to predict* (frozen backbone) vs. *what to look at* (router, trained).
- Goal: learn selection policies that don't forget old tasks.

### 3.2 Backbone Interface (Backbone-Agnostic)
- 4 hooks: `encode(WSI) → Z ∈ R^{n×D}`, `predict(S) → logits`,
  `class_text_features() → T`, `prototype_features() → P`.
- Instantiated with frozen VL diagnosis backbone (CONCH+prompt/prototype head).
- **Key identity**: backbone frozen ⇒ Forgetting ≡ 0 by construction (not a contribution).

### 3.3 MicroRouter (Navigation Policy φ)
- Input per patch: $[z_i; s_i]$ where $s_i$ = 4 task-count-invariant summary features
  (max text sim, entropy of text sim, max proto sim, mean proto sim).
- Architecture: Linear(516→256) + GELU + Linear(256→1); ~132K params.
- Training objective: soft-route loss — differentiable Top-K via softmax weighting.
- **Formulas** (Eqs. 1–4 in paper body).

### 3.4 Sequential Budgeted Observer (SBO) ← NEW
- Replaces static top-K; makes selection truly adaptive to already-seen evidence.
- Algorithm loop (pseudocode + formal definition).
- **z-score normalization** of base scores (Eq. 5): scale-invariant, monotone preserving.
- **MMR redundancy penalty** (Eq. 6): $a_i^{(t)} = \tilde{r}_i - \lambda \cdot \max_{j \in \mathcal{S}^{<t}} \cos(z_i, z_j)$.
- **Confidence-based early stopping** (Route B, Eq. 7).
- λ as diversity rotor: λ=0 degrades to static top-K; optimal λ=0.5–1.0 (empirical).

### 3.5 Navigation Skill Memory (NSM)
- Stores: $\phi^{(t)} = $ router state_dict after training on task $t$.
- Size: ~533 KB per task (float32); 4 tasks ≈ 2.1 MB.
- Context gate: *oracle* (task ID known) = upper bound; task-free gate = future work.
- **Why NSM ≠ "just storing"**: stores *keys* (routing policy for which patches to look),
  not raw signals; frozen backbone retains signals permanently.

### 3.6 Zero-Shot Navigation Baseline ← NEW
- Score: $r_i^{zs} = \max_c \hat{z}_i^\top \hat{t}_c$ (frozen FM text-patch similarity).
- No training, no task-specific parameters. Inspired by ZeroSlide [cite].
- Evaluated as `zeroshot_seq` and `zeroshot_oneshot` modes.

### 3.7 Consolidation and Baselines
- Naive sequential: shared router, fine-tuned across all tasks (mACC 0.595, Forgetting 0.454).
- EWC-on-router: Fisher diagonal; insufficient (0.40; 0/3).
- Per-task NSM: upper bound (mACC 0.935, Forgetting 0).
- Zero-shot navigator: strong training-free (mACC 0.858) — still < NSM.
- *Planned (N7)*: LoRA-NSM — low-rank adapters per task, shared frozen MLP base.

---

## 4. Experiments  (~2 pages)

### 4.0 Experimental Setup
- **Data**: 4 TCGA binary subtyping tasks — esca (ESCA), rcc (RCC), brca (BRCA), lung (NSCLC).
  Label shifts 0/2/4/6; patches encoded once by frozen CONCH (512-d).
- **Orders**: *reverse* (esca→rcc→brca→lung) and *paper* (lung→brca→rcc→esca); 3 folds each.
  Reverse makes esca the oldest (hardest CL test), paper makes lung oldest.
- **Budgets**: {All, 128, 64, 32, 16}; primary metric budget = 64.
- **Metrics**: mACC (mean accuracy over all tasks), Forgetting (average accuracy drop), BWT.

### 4.1 Backbone Accuracy and Decoupling
- Matched training: frozen backbone ACC 0.879/0.886 (paper/reverse), Forgetting = 0 (identity).
- Non-decoupled variant: 0.378/0.218 accuracy, 0.735/0.950 forgetting → decoupling is necessary.

### 4.2 Static Selection: Router vs. Heuristics (Recent Tasks)
- At budget 64, router: esca (recent) 0.956, lung (recent) 0.922 vs. best heuristic 0.889/0.897.
- 6/6 GO across folds and orders.

### 4.3 Selection Forgetting (Old Tasks)
- Router@64: esca (old) 0.333, lung (old) 0.397 vs. best heuristic 0.822/0.813.
- 6/6 NO-GO. Falls below random.

### 4.4 Same-Task Recency Flip (Core Causal Test)
- Identical task/test set: lung 0.922→0.397, esca 0.956→0.333 at K=64.
- Holds for lung (large sample), ruling out size/difficulty confounds.

### 4.5 Mechanism: Confident Mis-Prioritization
- t-SNE visualization: forgotten router keeps structured scores but concentrates on different subpopulation.
- Explains sub-random behavior (actively wrong, not random).

### 4.6 Mitigation Ablation (Table 3)
| Method | esca@64 | Forgetting | mACC | GO |
|---|---|---|---|---|
| Naive (shared router) | 0.333 | 0.454 | 0.595 | 0/3 |
| EWC-on-router | 0.400 | — | — | 0/3 |
| Zero-shot navigator | — | 0 | **0.858** | — |
| Per-task NSM (ours) | **0.911** | **0** | **0.935** | 3/3 |

### 4.7 Sequential Observation: λ Sweep (NEW, N6 results)
- Route A (MMR) confirms seq ≠ oneshot when λ > 0.
- λ=0: seq ≡ oneshot (mechanism verified — no penalty = no diversity).
- λ=1.0: small divergence, acc near-unchanged (esca 0.867).
- λ=2.0: moderate divergence, slight acc cost (esca 0.800@64).
- λ=4.0: large divergence, large acc drop (esca 0.267@64).
- **Insight**: optimal λ=0.5–1.0; WSI tumor patches cluster spatially, over-penalization forces agent off diagnostic regions.
- Visualization: t-SNE of 4-round selection trace colored by round (Figure N_seq_trace).

### 4.8 Comparison Table (summary)
| Method | mACC | Forgetting | seq ≠ oneshot | params/task |
|---|---|---|---|---|
| Naive continual | 0.595 | 0.454 | ✗ | shared |
| Zero-shot | 0.858 | 0 | ✓ | 0 |
| NSM (ours, oracle) | **0.935** | **0** | ✓ (route A) | ~533KB |
| NSM+LoRA (planned) | — | — | ✓ | ~4KB |

---

## 5. Discussion  (~0.75 page)

### 5.1 Selection Forgetting vs. Classifier Forgetting
- Distinct: backbone frozen → forgetting = 0 by construction. Router is the unique locus.
- Broader implication: any learned selector front-ending a frozen FM can exhibit this.

### 5.2 Why EWC Fails, and What Would Work Better
- The router's ranking (a global nonlinear function of weights) is not protected by weight-space penalties.
- Needed: distillation over old-task patch score *rankings*, or function-space regularization.

### 5.3 Sequential Observation and Spatial Clustering
- λ finding reveals that tumor patches cluster spatially — router has learned this geometry.
- Strong diversity (large λ) forces exploration of non-diagnostic regions → accuracy drop.
- Implication: SBO should respect learned spatial priors; pure MMR-diversity is suboptimal.

### 5.4 LoRA-NSM: Scalable Navigation Skill Memory (Planned)
- Per-task NSM grows O(#tasks × 533KB). At scale, undesirable.
- LoRA on router MLP: each task adds A·B low-rank adapters (~4K params, ~30× smaller).
- Shared frozen base retains common navigation priors; task adapters store per-task differences.
- 7/20 target: LoRA-NSM approaching oracle NSM accuracy.

### 5.5 Limitations
- 4 TCGA tasks from one source; longer sequences needed.
- Oracle context gate (task-free gate is an open problem).
- Sequential observation studied at inference; joint training with SBO is future work.

---

## 6. Conclusion  (~0.25 page)

- Selection forgetting: a new forgetting locus in frozen-FM continual WSI classification.
- CNL + NSM recovers old-task selection to all-patch level (mACC 0.935, Forgetting 0).
- Sequential Budgeted Observer makes selection adaptive; optimal λ reveals spatial structure.
- Future: LoRA-NSM, task-free gate, RL-based observation, multi-resolution navigation.

---

## Figures Plan

| # | Key | Content | Status |
|---|---|---|---|
| 1 | `Fig1_arch.pdf` | Architecture: frozen backbone + router + NSM | ✅ existing |
| 2 | `P0b_recency_flip.pdf` | Recency flip (core) | ✅ existing |
| 3 | `P2contrast_esca_fold1.pdf` | t-SNE mis-prioritization | ✅ existing |
| 4 | `Tab_mitigation` | Mitigation ablation table | in main.tex |
| 5 | `Fig_seq_trace.pdf` | Sequential observation 4-round trace (t-SNE) | **TODO** (viz script) |
| 6 | `Fig_lambda_sweep.pdf` | λ vs. acc/diversity tradeoff | **TODO** (from sweep JSONs) |
| S1 | `FigS1_arch.pdf` | Evaluation protocol (appendix) | ✅ existing |

---

## Appendix

- A: Full implementation details (MicroRouterV0 architecture, hyperparameters).
- B: 3-fold results per task per order (extended tables).
- C: ZeroSlide baseline details.
- D: Sequential observation algorithm (formal pseudocode).
- E: LoRA-NSM design sketch.

---

## Open TODO (before camera-ready)

- [ ] Verify ZeroSlide citation (`zeroslide` BibTeX key).
- [ ] Run LoRA-NSM experiment (N7, target 7/20).
- [ ] Task-free gate experiment.
- [ ] 3-fold full results for sequential observation (fold 2, 3).
- [ ] Fig_seq_trace: run `viz_sequential_trace.py` on RunPod (needs features).
- [ ] Fig_lambda_sweep: generate from `outputs/routeA_sweep/` JSONs.

---

*Provenance: `specs/` (ADR/SPEC/WORKLOG), `STORYLINE.md`, N2/N3/N4/N5 reports,
N6 λ sweep (`outputs/routeA_sweep/`). Updated 2026-06-30.*
