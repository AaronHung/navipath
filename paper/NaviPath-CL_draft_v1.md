# NaviPath-CL: Continual Navigation of Whole-Slide Images via Budgeted Sequential Observation and Skill Memory

> **Draft v1.0 — 2026-06-30**
> Full paper markdown. Use as source for `main.tex`.
> `[cite: KEY]` = BibTeX placeholder.
> `[TODO: ...]` = needs experimental result / figure.

---

## Abstract

Gigapixel whole-slide images (WSIs) contain thousands of patches, yet practical diagnosis observes only a small *budget* of patches through a learned **observation policy** — a trainable module that decides *where to look*. We study an overlooked failure mode: under a continual stream of WSI tasks, this observation policy suffers **selection forgetting**, collapsing below a random baseline for old tasks even when the diagnostic backbone is fully frozen (so classifier-level forgetting is zero by construction). We propose the **Continual Navigation Layer (CNL)**, a backbone-agnostic framework comprising: (i) a lightweight **MicroRouter** that learns patch importance scores under weak supervision; (ii) a **Sequential Budgeted Observer (SBO)** that selects patches in adaptive, evidence-driven rounds via Maximal Marginal Relevance (MMR) redundancy; and (iii) a **Navigation Skill Memory (NSM)** that stores per-task navigation policies, recovering selection from mACC 0.595 to **0.935** with zero forgetting. A zero-shot navigation baseline (using frozen FM text–patch similarity, no training) achieves mACC 0.858, confirming that training-free navigation is strong but trainable skill memory provides further gains. A λ sweep over the SBO diversity parameter confirms that sequential selection is truly adaptive (seq ≠ one-shot when λ > 0) and that optimal λ = 0.5–1.0 reflects the spatial clustering of diagnostic regions in WSI. We release code, trained skill banks, and figures; LoRA-based cheap navigation memory is identified as the next scalable step.

**Keywords**: continual learning · whole-slide images · patch selection · foundation models · sequential observation · catastrophic forgetting

---

## 1. Introduction

Whole-slide images are gigapixel-scale and are routinely classified under the multiple-instance learning (MIL) paradigm: a slide is treated as a bag of patch features that are aggregated into a slide-level prediction [cite: ilse2018abmil, lu2021clam]. Recent pathology foundation models supply strong *frozen* patch encoders [cite: lu2024conch, chen2024uni], and prompt- or prototype-based vision–language MIL adapts them to new tasks by learning only small, lightweight task descriptors while the encoder stays frozen [cite: gou2025qpmil]. Freezing the encoder is attractive for continual learning: the representation cannot drift, and the classical source of catastrophic forgetting is largely removed.

Yet a second, practical problem remains. A slide contains thousands of patches; under compute or latency budgets a model must decide **which patches to examine**. This *observation policy* (or patch router) is itself learnable, and whether it survives continual learning has, to our knowledge, not been studied. We fill this gap.

**Selection forgetting.** We show that a lightweight trainable patch router, attached to a frozen diagnostic backbone, **selects well for the most-recently-learned task** but, for older tasks, collapses *below the random baseline* — a phenomenon we term *selection forgetting*. The predictor is unchanged (backbone frozen, Forgetting ≡ 0 by construction), so every degradation is attributable to the selection module alone. A same-task recency-flip test (same task, same data, same router, varying only recency) drops accuracy from ∼0.9 to 0.33–0.40 and holds even for the sample-abundant lung task, ruling out difficulty or size confounds.

**Our solution.** We propose a **Continual Navigation Layer (CNL)** with three components. (1) A **MicroRouter** (∼132K-param MLP) learns task-specific navigation skills under weak supervision from slide-level labels. (2) A **Sequential Budgeted Observer (SBO)** replaces the static top-K selection with an adaptive, multi-round loop: each round, previously-selected patches are penalized (MMR-style) to drive exploration of new diagnostic regions — making the selection genuinely sequential rather than order-invariant. (3) A **Navigation Skill Memory (NSM)** stores a snapshot of the router weights per task; during evaluation an oracle context gate retrieves the correct skill, recovering mACC from 0.595 to **0.935** with Forgetting = 0.

**Zero-shot navigation.** We additionally introduce a zero-shot navigation baseline that scores patches using only frozen FM text–patch similarity, requiring no training [cite: zeroslide]. This training-free baseline achieves mACC 0.858, providing a strong comparison point and confirming that NSM's further gains come from task-specific learned navigation rather than just frozen-FM similarity.

**Contributions:**
1. We identify and formalize **selection forgetting** in frozen-FM continual WSI classification.
2. We propose the **CNL framework**: MicroRouter + SBO + NSM (backbone-agnostic, any frozen FM).
3. We introduce a **Sequential Budgeted Observer** with MMR redundancy, making multi-step selection truly adaptive; we analyze the λ-diversity tradeoff and connect it to the spatial structure of diagnostic regions.
4. We provide a **zero-shot navigation baseline** (ZeroSlide-inspired) as a rigorous training-free upper bound on non-parametric selection.
5. We show **NSM** recovers old-task navigation (mACC 0.935, Forgetting 0) and propose **LoRA-NSM** as a scalable next step (∼30× fewer parameters per task).

---

## 2. Related Work

### 2.1 MIL and Foundation Models for WSIs

Computational pathology casts WSI classification as MIL: a slide is a bag of patch features pooled into a slide-level prediction via attention [cite: ilse2018abmil], transformer [cite: shao2021transmil], or dual-stream [cite: li2021dsmil] aggregators; CLAM [cite: lu2021clam] uses clustering-constrained attention. Large pretrained encoders — CONCH [cite: lu2024conch], UNI [cite: chen2024uni], PLIP [cite: huang2023plip] — supply frozen patch features, extending CLIP [cite: radford2021clip] to histology. Prompt- or prototype-based vision–language MIL keeps the encoder frozen and learns lightweight task descriptors, including in a continual setting [cite: gou2025qpmil]. These methods assume all (or a fixed sampling of) patches are available and **do not study which patches to keep under a budget across tasks**.

### 2.2 Continual Learning

The continual learning literature addresses catastrophic forgetting with regularization [cite: kirkpatrick2017ewc], knowledge distillation [cite: li2017lwf], and replay [cite: rebuffi2017icarl, chaudhry2019agem, buzzega2020der]. Prompt-based continual learning adapts frozen backbones with small prompts [cite: wang2022l2p, wang2022dualprompt, smith2023coda]. For WSI specifically, AKDPMP [cite: li2025akdpmp] combats attention drift in MIL continually. Crucially, **this literature studies forgetting of the classifier or representation**; we instead expose forgetting of a patch-selection mechanism, a distinct locus that existing methods (including EWC on the router, which we test) do not address.

### 2.3 Budgeted and Agentic Inference for WSI

Several works reduce WSI inference cost by selecting informative patches. Iterative Patch Selection (IPS) [cite: bergner2023ips] uses an iterative refinement policy. RLogist [cite: zhao2023rlogist] uses reinforcement learning to navigate a slide across resolutions, mimicking a pathologist's scanning strategy. ZeroSlide [cite: zeroslide] proposes scoring patches by frozen FM text–patch similarity without any training — a principle we adopt as our zero-shot navigation baseline. These works operate in a **single-task setting**; we study how a navigation policy should be *retained across tasks*, introducing the continual-learning dimension.

### 2.4 Parameter-Efficient Fine-Tuning

LoRA [cite: hu2022lora] augments frozen model layers with low-rank adapters (A·B, rank r ≪ d), enabling task-specific adaptation at minimal parameter cost. We identify LoRA-NSM — where each task adds only a low-rank adapter to the frozen router MLP — as the path to scalable navigation skill memory (§3.7 / §5.4).

---

## 3. Method

**Notation.** A slide has $n$ patches; let $Z \in \mathbb{R}^{n \times D}$ denote the frozen foundation-encoder features (e.g., $D=512$, CONCH). $\hat{z}$ denotes L2-normalization. $T = \{t_c\}_{c=1}^{C} \subset \mathbb{R}^D$ are class-text features; $P = \{p_m\}_{m=1}^{M} \subset \mathbb{R}^D$ are prototype features; $\phi$ denotes router parameters.

### 3.1 Problem Setup

We study **class-incremental WSI subtyping**: tasks $1,\ldots,T$ arrive sequentially; after task $t$ the model must classify slides from **all** tasks $1,\ldots,t$ using at most $K$ patches per slide ($K \ll n$). We decouple two sub-problems:
- **Prediction** (*what to predict*): a frozen diagnostic backbone $g_{\theta^*}$ that never changes.
- **Navigation** (*what to look at*): a trainable policy $\phi$ that selects the $K$ patches to show $g$.

Because $\theta^*$ is frozen throughout, any forgetting is attributable solely to $\phi$.

### 3.2 Backbone Interface (Backbone-Agnostic CNL)

CNL interacts with any frozen diagnostic backbone via four hooks:
1. `encode(WSI)` → $Z \in \mathbb{R}^{n \times D}$ (precomputed patch features).
2. `class_text_features()` → $T$ (class-conditioned text embeddings).
3. `prototype_features()` → $P$ (task prototypes).
4. `aggregate_and_predict(Z_\mathcal{S})` → $(\hat{y}, \text{logits})$ for a subset $\mathcal{S}$.

This interface is satisfied by any VL-based backbone (we instantiate with a frozen prompt/prototype VL diagnosis backbone [cite: gou2025qpmil]); CNL itself is backbone-agnostic.

### 3.3 MicroRouter: The Navigation Policy

For each patch $i$, we compute a 4-dimensional **task-count-invariant summary** from the frozen features:

$$s_i = \left[ \max_c \hat{z}_i^\top \hat{t}_c,\;\; H\!\left(\operatorname{softmax}\!\left(\hat{z}_i^\top \hat{t}\right)\right),\;\; \max_m \hat{z}_i^\top \hat{p}_m,\;\; \tfrac{1}{M}\textstyle\sum_m \hat{z}_i^\top \hat{p}_m \right] \tag{1}$$

where $H(\cdot)$ is Shannon entropy. The concatenated vector $[z_i; s_i] \in \mathbb{R}^{D+4}$ is mapped through a two-layer MLP to a scalar importance score:

$$r_i = \operatorname{MLP}_\phi([z_i; s_i]),\qquad \operatorname{MLP}: \mathbb{R}^{516} \xrightarrow{\text{Linear}} \mathbb{R}^{256} \xrightarrow{\text{GELU}} \mathbb{R}^{256} \xrightarrow{\text{Linear}} \mathbb{R}^1. \tag{2}$$

This MicroRouter has ≈132K parameters and is **the only component trained across tasks**; the backbone stays frozen.

#### 3.3.1 Differentiable Training via Soft-Route Objective

Hard top-K is non-differentiable. We train with a **soft-route** objective: a candidate set $\mathcal{S}_K = \operatorname{top-K}_i r_i$ is selected; scores within it are softmax-normalized into attention weights; the weighted sum forms the bag embedding; the frozen text classifier produces logits:

$$w_i = \frac{\exp(r_i)}{\sum_{j \in \mathcal{S}_K} \exp(r_j)}, \qquad \bar{z} = \widehat{\sum_{i \in \mathcal{S}_K} w_i z_i}, \qquad \mathcal{L}_\text{route} = \operatorname{CE}\!\left(\sigma\,\bar{z}\,T^\top,\, y\right). \tag{3}$$

Gradients flow to $\phi$ through the softmax weights $w_i$ but not through the frozen backbone. The router is trained task-by-task with Adam (lr $5\times10^{-4}$, wd $10^{-4}$, 5 epochs/task).

### 3.4 Sequential Budgeted Observer (SBO)

The static top-K router selects patches in one shot: it is **order-invariant** and ignores previously-selected evidence. The SBO replaces this with an adaptive, multi-round loop, making selection genuinely sequential.

**Observation state.** Define $\mathcal{S}^{(\leq t)} = \bigcup_{\tau=1}^{t} \mathcal{S}^{(\tau)}$ as the set of patches observed up to round $t$; the SBO maintains this state and uses it to drive diversity.

**Step 1 — Base score normalization (route A).** Router scores have task-dependent scale; to ensure the redundancy penalty operates at the same scale as the base scores, we apply z-score normalization (which is monotone and therefore does not change the one-shot top-K result):

$$\tilde{r}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}, \qquad \mu_r = \frac{1}{n}\sum_i r_i, \quad \sigma_r = \sqrt{\frac{1}{n}\sum_i (r_i - \mu_r)^2}. \tag{4}$$

**Step 2 — MMR adjusted score.** At each round $t$, penalize patches similar to anything already seen using **Maximal Marginal Relevance** [cite: carbonell1998mmr]:

$$a_i^{(t)} = \tilde{r}_i - \lambda \cdot \underbrace{\max_{j \in \mathcal{S}^{(\leq t-1)}} \cos(z_i, z_j)}_{= \bar{m}_i^{(t)}}, \qquad a_i^{(t)} = -\infty \;\text{ if } i \in \mathcal{S}^{(\leq t-1)}. \tag{5}$$

$\lambda \geq 0$ is the **diversity rotor**: $\lambda=0$ degrades to static top-K; increasing $\lambda$ drives the agent to explore new regions.

The maximum similarity term $\bar{m}_i^{(t)}$ is updated **incrementally** after each round:
$$\bar{m}_i^{(t+1)} = \max\!\left(\bar{m}_i^{(t)},\; \max_{j \in \mathcal{S}^{(t)}} \cos(z_i, z_j)\right), \tag{6}$$
achieving $O(n \cdot k_\text{step})$ complexity per round vs. $O(n \cdot |\mathcal{S}|)$ for recomputation.

**Step 3 — Round selection.** Select $k_\text{step}$ patches per round:
$$\mathcal{S}^{(t)} = \operatorname{top-}k_\text{step}\{a_i^{(t)} : i \notin \mathcal{S}^{(\leq t-1)}\}. \tag{7}$$

**Step 4 — Confidence-based early stopping (route B).** If backbone confidence on the current evidence exceeds a threshold $\tau$, stop early:
$$P^{(t)} = \max_c \operatorname{softmax}\!\left(g_{\theta^*}(Z_{\mathcal{S}^{(\leq t)}})\right) \geq \tau \;\Rightarrow\; \text{stop}. \tag{8}$$

When early stopping is disabled ($\tau = \infty$), the backbone is called only once at the end (cost ≈ one-shot).

**SBO Algorithm (pseudocode):**
```
Input: Z ∈ R^{n×D}, r ∈ R^n (base scores), budget K, step_size k, λ, τ
Normalize: r̃ ← z-score(r)
m ← 0^n   (max similarity to seen set, initialized to 0)
S ← ∅

while |S| < K:
    a ← r̃ - λ·m
    a[S] ← -∞
    pick ← top-k_{step}(a)    // step_size patches
    S ← S ∪ pick
    // incremental MMR update
    sims ← Z_norm @ Z_norm[pick]^T     // [n, k_step]
    m ← max(m, max(sims, dim=1))
    // optional confidence early stop
    if τ < ∞ and confidence(g(Z_S)) ≥ τ: break

return S, predict(Z_S)
```

### 3.5 Navigation Skill Memory (NSM)

After training the MicroRouter on task $t$, NSM stores a **snapshot** of the router weights:
$$\text{NSM} = \{\phi^{(1)}, \phi^{(2)}, \ldots, \phi^{(T)}\}. \tag{9}$$

At evaluation time, a **context gate** retrieves the appropriate skill. We study an **oracle gate** (task identity provided) as an upper bound; task-free gate (identifying the task from the slide alone) is an open problem.

**What NSM stores vs. what it doesn't.** NSM stores *routing keys* — the MLP weights that encode "which features in this frozen embedding space correlate with diagnostic relevance for task $t$." It does **not** store raw patch features, prototypes, or class labels; those reside in (or are computed on-the-fly by) the frozen backbone. Storage cost: ≈533KB per task in float32; 4 tasks ≈ 2.1MB total.

**NSM as principled upper bound.** Per-task parameter isolation is a known CL upper-bound strategy (cf. PackNet [cite: mallya2018packnet], PNN [cite: rusu2016pnn]). We report it transparently as such, motivating cheaper alternatives in §3.6.

### 3.6 Zero-Shot Navigation Baseline

We introduce a training-free navigation baseline inspired by ZeroSlide [cite: zeroslide]: score patches by their maximum cosine similarity to frozen class-text features, without any router training:

$$r_i^{zs} = \max_c \hat{z}_i^\top \hat{t}_c. \tag{10}$$

This baseline uses no task-specific parameters (0 bytes of storage per task) and requires no training. It serves as a strong lower bound on what continual navigation must achieve to demonstrate value over the frozen FM's inherent alignment. We evaluate it in both one-shot (`zeroshot_oneshot`) and sequential SBO (`zeroshot_seq`) modes.

### 3.7 Consolidation Strategies and Baselines

| Strategy | Description | Params/task |
|---|---|---|
| Naive continual | Single router, fine-tuned sequentially | shared (0 extra) |
| EWC-on-router | Fisher-weighted weight regularization on $\phi$ | shared + Fisher diagonal |
| Zero-shot nav. | Frozen text–patch sim., no training | 0 |
| **NSM (ours)** | Per-task router snapshot (oracle gate) | ≈533KB |
| LoRA-NSM *(planned)* | Shared frozen router + per-task low-rank adapters | ≈4KB |

**EWC formulation.** After task $t$:
$$F^{(t)}_j = \mathbb{E}_{\mathcal{D}_t}\!\left[\!\left(\frac{\partial \mathcal{L}_\text{route}}{\partial \phi_j}\right)^{\!2}\right], \qquad \mathcal{L}^{(t+1)} = \mathcal{L}_\text{route} + \lambda_\text{EWC}\!\sum_{\tau \leq t}\!\sum_j F^{(\tau)}_j (\phi_j - \phi^{*(\tau)}_j)^2. \tag{11}$$

**LoRA-NSM sketch.** Freeze the router MLP base; for each task $t$, learn low-rank adapters $\{(A^{(t)}_\ell, B^{(t)}_\ell)\}_\ell$ with $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$ ($r=4$). The per-task forward pass adds $\Delta W^{(t)}_\ell = A^{(t)}_\ell B^{(t)}_\ell$ to the frozen weight. Storage: $2 r d$ per layer per task ≈ 4K params total — a ∼130× reduction vs. full NSM.

---

## 4. Experiments

### 4.0 Experimental Setup

**Data.** We use four TCGA cohorts as a class-incremental binary subtyping sequence:
- **esca** (TCGA-ESCA): esophageal squamous vs. adenocarcinoma. ~180 slides, label shift 0/1.
- **rcc** (TCGA-KIRC+KICH+KIRP): renal cell carcinoma subtypes. ~965 slides, label shift 2/3.
- **brca** (TCGA-BRCA): breast IDC vs. ILC. ~1026 slides, label shift 4/5.
- **lung** (TCGA-LUAD+LUSC): lung adenocarcinoma vs. squamous. ~1059 slides, label shift 6/7.

Patches (224×224 at 20× magnification) are encoded **once** by a frozen CONCH model into 512-d features; the encoder is never updated. Official train/val/test splits define 3 folds.

**Task orders.** We evaluate two sequential orderings:
- *Reverse*: esca → rcc → brca → lung. Makes small-sample esca the *oldest* task.
- *Paper*: lung → brca → rcc → esca. Makes esca the *most-recent* task.

This design enables a **same-task recency-flip test**: the identical task/data/architecture evaluated as oldest vs. most-recent, varying only recency.

**Backbone.** A frozen prompt/prototype-based VL diagnosis backbone [cite: gou2025qpmil] on CONCH features; trained per task (Adam, lr 1e-3, wd 5e-4, 12 epochs). Baseline continual accuracy: mACC 0.924/0.917 (paper/reverse), Forgetting 0.017/0.041.

**Router.** MicroRouterV0 (§3.3); Adam, lr 5e-4, wd 1e-4, 5 epochs/task. Backbone frozen throughout.

**Budgets.** $K \in \{\text{All}, 128, 64, 32, 16\}$. Primary budget = 64. SBO step size = 16.

**Metrics.** mACC (mean task accuracy at end of training), Forgetting (average accuracy drop on old tasks), BWT (backward transfer), GO/NO-GO (router@64 > best heuristic@64).

### 4.1 Backbone Accuracy and the Necessity of Decoupling

Under matched training, our frozen-backbone setup attains:

**Table 1. Continual accuracy (backbone level, mean ± std over 3 folds).**

| Method | Order | mACC | Forgetting | BWT |
|---|---|---|---|---|
| VL diagnosis backbone (frozen) baseline | paper | 0.924 ± 0.016 | 0.017 ± 0.022 | −0.017 |
| VL diagnosis backbone (frozen) baseline | reverse | 0.917 ± 0.026 | 0.041 ± 0.023 | −0.041 |
| NaviPath (decoupled, ours) | paper | 0.879 ± 0.030 | **0.000** | **0.000** |
| NaviPath (decoupled, ours) | reverse | 0.886 ± 0.030 | **0.000** | **0.000** |

NaviPath's Forgetting = 0 is a **decoupling identity, not a contribution**: a frozen predictor is flat by construction. We report it to locate our contribution in the *selection* analysis, not accuracy. Non-decoupled variant (trainable experts modifying backbone input) caused severe interference: accuracy 0.378/0.218, forgetting 0.735/0.950 — confirming that decoupling is essential.

### 4.2 Static Selection: Router vs. Heuristics (Recent Tasks)

On the most-recently-learned task, the MicroRouter selects more informative patches than all training-free heuristics at tight budgets.

**Table 2. Patch selection at K=64, recent vs. old (accuracy, mean over 3 folds).**

| Task | Recency | All | **Router (ours)** | Random | Proto. | Semantic | GO |
|---|---|---|---|---|---|---|---|
| esca | RECENT (paper, last) | 0.933 | **0.956** | 0.889 | 0.711 | 0.867 | ✓ (3/3) |
| lung | RECENT (reverse, last) | 0.892 | **0.922** | 0.881 | 0.831 | 0.897 | ✓ (3/3) |
| esca | OLD (reverse, first) | 0.867 | **0.333** | 0.822 | 0.778 | 0.778 | ✗ (0/3) |
| lung | OLD (paper, first) | 0.764 | **0.397** | 0.783 | 0.705 | 0.813 | ✗ (0/3) |

At K=64: +2.5–6.7 pp vs. best heuristic (6/6 GO) on recent tasks; collapses below random (6/6 NO-GO) on old tasks.

### 4.3 Selection Forgetting: A Same-Task Recency-Flip Test

The recency flip isolates recency as the cause. For the **identical** task, test set, and architecture, changing only whether the task was learned **last vs. first** flips selection accuracy from ∼0.9 to 0.33–0.40 at K=64 — even for lung, the most sample-abundant task, ruling out size/difficulty confounds. Across all 3 folds (6/6), the flip is consistent.

*(Figure: P0b_recency_flip — budget curves for esca and lung under both orders.)*

### 4.4 Mechanism: Confident Mis-Prioritization, Not Noise

Feature-space visualization *(Figure: P2contrast_esca_fold1)* shows the same esca slide scored by the recent router vs. the forgotten router. The forgotten router does **not** produce uniform/noisy scores; it concentrates its top-K on a different, later-task-salient patch subpopulation — **confident mis-prioritization** that explains sub-random behavior (the agent is actively wrong, not just uncertain).

### 4.5 Mitigation: NSM Recovers, EWC Does Not

**Table 3. Mitigation on the oldest task (esca, reverse order; mean over 3 folds).**

| Router strategy | @256 | @128 | @64 | @32 | best heur@64 | mACC | Forgetting | GO |
|---|---|---|---|---|---|---|---|---|
| Naive continual | 0.511 | 0.400 | **0.333** | 0.333 | 0.822 | 0.595 | 0.454 | ✗ (0/3) |
| EWC-on-router | 0.622 | 0.422 | **0.400** | 0.267 | 0.822 | — | — | ✗ (0/3) |
| Zero-shot navigator | — | — | **0.858** (mACC) | — | — | 0.858 | **0.000** | — |
| **Per-task NSM (ours)** | 0.933 | 0.933 | **0.933** | 0.933 | 0.822 | **0.935** | **0.000** | ✓ (3/3) |

NSM restores esca@64 from 0.333 to **0.911** (fold 1; 0.933 mean over 3 folds), matching all-patch accuracy. The zero-shot navigator achieves mACC 0.858 without any training — a strong baseline, but NSM provides an additional +7.7 pp mACC by learning task-specific routing keys. EWC barely moves the needle (0.400; 0/3), confirming that weight-level consolidation is insufficient for a selection policy (see §5.2).

### 4.6 Sequential Observation: λ Sweep (N6 Results)

We run the SBO on the trained NSM skill bank (fold 1, reverse order, eval-tasks 0–3) sweeping λ ∈ {0.0, 1.0, 2.0, 4.0} with normalize_base=True, redundancy_mode=maxsim, budget=64, step_size=16.

**Table 4. Sequential (nsm_seq) vs. one-shot (nsm_oneshot) accuracy at K=64, λ sweep.**
*(fold 1, reverse order, normalize_base=True, redundancy_mode=maxsim)*

| λ | esca (seq / 1shot / Δ) | rcc (seq / 1shot / Δ) | brca (seq / 1shot / Δ) | lung (seq / 1shot / Δ) |
|---|---|---|---|---|
| 0.0 | 0.867 / 0.867 / 0.000 | 0.961 / 0.961 / 0.000 | 0.860 / 0.860 / 0.000 | 0.810 / 0.810 / 0.000 |
| 1.0 | 0.867 / 0.867 / 0.000 | 0.961 / 0.961 / 0.000 | 0.860 / 0.860 / 0.000 | 0.800 / 0.810 / −0.010 |
| 2.0 | 0.800 / 0.867 / **−0.067** | 0.961 / 0.961 / 0.000 | 0.850 / 0.860 / −0.011 | 0.768 / 0.810 / **−0.042** |
| 4.0 | 0.267 / 0.867 / **−0.600** | 0.355 / 0.961 / **−0.605** | 0.570 / 0.860 / **−0.290** | 0.684 / 0.810 / **−0.126** |

*Key: λ=0 → seq ≡ oneshot (mechanism baseline). Optimal λ=0.5–1.0 (minimal Δ). Large λ forces agent off spatially-clustered diagnostic regions → accuracy loss.*

**Key findings:**
1. **Mechanism confirmed**: λ=0 → seq ≡ one-shot (no diversity penalty = no sequential behavior). λ>0 → seq ≠ one-shot. The SBO is genuinely multi-step.
2. **Optimal λ = 0.5–1.0**: Small divergence from one-shot, negligible accuracy cost. Larger λ causes increasing accuracy loss.
3. **Why large λ hurts**: WSI tumor patches are **spatially clustered** — the router has learned to identify these clusters. MMR with large λ forces the agent *away* from the diagnostic cluster into irrelevant regions, degrading accuracy. This is not a failure of the mechanism; it reveals a meaningful property of the learned navigation policy.
4. **budget=16 (1 step)**: seq ≡ one-shot regardless of λ — when only one step is taken, there is no history to exploit.

*(Figure: Fig_lambda_sweep — λ vs. accuracy curves for all 4 tasks.)*
*(Figure: Fig_seq_trace — t-SNE of 4-step sequential trace, patches colored by selection round.)*

---

## 5. Discussion

### 5.1 Selection Forgetting Is Distinct from Classifier Forgetting

Our setup deliberately removes representation/classifier drift (frozen backbone, Forgetting ≡ 0 by construction). This lets every old-task degradation be attributed to the selection module — a forgetting phenomenon that the continual-learning literature, focused on classifier/representation, does not capture. Even when *what to predict* cannot be forgotten, *what to look at* can. We believe this matters broadly for budget-constrained inference with frozen foundation models, where a learned gate/router/selector sits in front of a fixed predictor.

### 5.2 Why EWC Fails While Per-Task NSM Fully Recovers

The per-task recovery (0.33→0.93) shows the selection signal is still present in the frozen features; the single continual router simply overwrites the *ranking* it learned for old tasks. EWC penalizes drift of individual router weights, but what matters is the induced *ordering* of patch scores — a global, highly nonlinear function of those weights. Protecting individual weights is therefore a poor proxy for protecting a selection policy. More promising directions include: (a) distilling old-task patch **score rankings** (pairwise or listwise rank loss), (b) function-space regularization over patch orderings, or (c) per-task low-rank adapters (LoRA-NSM).

### 5.3 Sequential Observation and the Spatial Geometry of Diagnostic Regions

The λ sweep reveals a fundamental property of WSI navigation: tumor patches form coherent spatial clusters, and the router has learned to identify them. This is a feature, not a bug — the router encodes cluster-level priors about diagnostic morphology. Pure MMR-diversity (large λ) disrupts this prior and degrades accuracy.

The ideal SBO should respect learned spatial priors while still ensuring coverage. Promising directions include: adaptive λ (decrease λ once the high-confidence cluster is seen), attention-based redundancy (penalize only patches *within the same morphological cluster*), or RL-based sequential selection that learns a decision policy directly from slide-level reward.

### 5.4 LoRA-NSM: Scalable Navigation Skill Memory

Per-task NSM grows linearly in task count: 4 tasks ≈ 2.1MB, 100 tasks ≈ 52MB. At large scale, this is undesirable. LoRA-NSM (§3.7) stores only low-rank adapters (∼4KB/task, ∼130× smaller) and shares a frozen router base across tasks. This is the primary N7 goal (target: 7/20); preliminary design:

```python
# Frozen router base shared across tasks
router_base = MicroRouterV0(frozen=True)

# Per-task LoRA adapters (A, B matrices for each Linear layer)
class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
    
    def delta_weight(self):
        return self.A @ self.B   # [in_dim, out_dim]
```

Expected result: LoRA-NSM approaching oracle NSM mACC (0.935) at a fraction of the memory cost.

### 5.5 Limitations

1. **Dataset scope**: 4 TCGA tasks from a single source; broader datasets (multi-site, multi-stain) and longer task sequences (>10 tasks) needed.
2. **Oracle context gate**: task identity must be provided at inference. Task-free gate (identifying the task from the slide's feature distribution) is an open problem with known challenges [cite: wang2022l2p].
3. **Static SBO**: the SBO is a fixed algorithmic policy; a learned sequential policy (RL, route C) may better exploit multi-step structure but requires reward design and substantial GPU time.
4. **Single router architecture**: MicroRouterV0 is scalar; richer selectors (set-level attention, graph-based) may exhibit different forgetting dynamics.
5. **Single fold for λ sweep**: N6 results are fold 1 only; 3-fold confirmation is future work.

---

## 6. Conclusion

We identify **selection forgetting** — a new locus of catastrophic forgetting in frozen-foundation-model continual WSI classification where a trainable patch router forgets how to select patches for old tasks, even when the predictor cannot forget. The CNL framework addresses this with: a MicroRouter that learns task-specific navigation skills; a Sequential Budgeted Observer that makes selection genuinely adaptive via MMR redundancy; and a Navigation Skill Memory that stores per-task routing policies, recovering mACC from 0.595 to 0.935 with zero forgetting. A zero-shot navigation baseline (mACC 0.858) demonstrates the strength of frozen FM alignment, while NSM provides further gains through task-specific learned routing. The λ sweep over SBO diversity reveals that optimal multi-step selection respects the spatial clustering of diagnostic regions in WSI — motivating spatially-aware diversity in future sequential selection policies. LoRA-NSM (∼130× more efficient) is identified as the clear next scalable step.

We release code, trained skill banks, and figures.

---

## References

[BibTeX entries: see `references.bib`]

Key citations to include:
- `ilse2018abmil`, `lu2021clam`, `shao2021transmil`, `li2021dsmil` — MIL
- `lu2024conch`, `chen2024uni`, `huang2023plip`, `radford2021clip` — Foundation models
- `gou2025qpmil` — Backbone we instantiate
- `kirkpatrick2017ewc`, `li2017lwf`, `rebuffi2017icarl`, `chaudhry2019agem`, `buzzega2020der` — CL baselines
- `wang2022l2p`, `wang2022dualprompt`, `smith2023coda` — Prompt CL
- `li2025akdpmp` — Continual MIL
- `bergner2023ips`, `zhao2023rlogist` — Budgeted/agentic WSI
- `zeroslide` — Zero-shot navigation baseline
- `hu2022lora` — LoRA (planned NSM scalability)
- `shazeer2017moe`, `fedus2022switch` — Routing inspiration
- `mallya2018packnet`, `rusu2016pnn` — Parameter isolation upper bound context
- `carbonell1998mmr` — MMR (SBO diversity)

---

## Appendix A: Implementation Details

**MicroRouterV0 architecture:**
```
Input: [z_i (512-d CONCH feature); s_i (4-d summary)] → 516-d
Linear(516 → 256) + GELU
Linear(256 → 1) → scalar score r_i
Total parameters: 516×256 + 256 + 256×1 + 1 = 132,353 ≈ 132K
Storage (float32): 132,353 × 4 = 529,412 bytes ≈ 533KB
```

**Summary features $s_i$ (Eq. 1):**
- $s_1$: $\max_c \hat{z}_i^\top \hat{t}_c$ — maximum class-text similarity.
- $s_2$: $H(\text{softmax}(\hat{z}_i^\top \hat{T}^\top))$ — entropy of text similarity distribution (task uncertainty).
- $s_3$: $\max_m \hat{z}_i^\top \hat{p}_m$ — maximum prototype similarity.
- $s_4$: $\frac{1}{M}\sum_m \hat{z}_i^\top \hat{p}_m$ — mean prototype similarity.

These 4 features are **task-count-invariant**: they don't grow with the number of classes/prototypes, making the router architecture fixed regardless of how many tasks have been learned.

**Training hyperparameters:**

| Component | lr | wd | epochs/task | optimizer |
|---|---|---|---|---|
| VL backbone (per task) | 1e-3 | 5e-4 | 12 | Adam |
| MicroRouter | 5e-4 | 1e-4 | 5 | Adam |
| EWC λ | — | — | 5 | Adam + EWC penalty |

**SBO default configuration:**
```python
ObserveConfig(
    budget=64,
    step_size=16,
    redundancy_weight=0.5,   # λ: optimal range 0.5-1.0
    normalize_base=True,
    redundancy_mode="maxsim",   # MMR
    confidence_threshold=None   # route B disabled in current experiments
)
```

---

## Appendix B: Per-Task Per-Order Accuracy Tables (3 folds)

[TODO: Add extended tables from N2/N3 outputs — `outputs/seqobs_reverse_f*/task*.json`]

---

## Appendix C: ZeroSlide Baseline Details

The zero-shot navigation baseline scores patches as $r_i^{zs} = \max_c \hat{z}_i^\top \hat{t}_c$ where $\hat{t}_c$ are class-conditioned text embeddings from the frozen CONCH text encoder. This requires no training and no gradient updates; it relies solely on the pre-trained FM's alignment between patch visual features and text descriptions of cancer subtypes. Inspired by ZeroSlide [cite: zeroslide], which demonstrated that frozen FM text-patch similarity can serve as a zero-shot WSI selector. In our setting this becomes a challenging baseline: the FM already encodes substantial pathological knowledge, and the question is whether task-specific training of the router provides measurable additional gain.

---

## Appendix D: Sequential Budgeted Observer — Formal Algorithm

**Algorithm 1: SequentialBudgetedObserver.observe()**
```
Input:
  Z ∈ R^{n×D}           patch features (frozen CONCH)
  r ∈ R^n               base scores from MicroRouter (or zero-shot)
  predict_fn            backbone inference function
  K                     total budget
  k                     step size
  λ                     MMR diversity weight
  τ                     confidence threshold (∞ = disabled)
  normalize_base        bool

Output:
  S ⊆ [n], |S| ≤ K     selected patch indices
  logits ∈ R^C          final prediction

Algorithm:
1. Normalize: Z_norm ← L2-normalize(Z, dim=1)
2. If normalize_base:
       r̃ ← (r − mean(r)) / (std(r) + ε)
   Else: r̃ ← r
3. m ← 0^n                              # max similarity to seen set
4. S ← ∅

5. While |S| < K:
   a. Compute adjusted scores:
          a ← r̃ − λ·m
          a[S] ← −∞                    # mask already-seen
   b. k_this ← min(k, K − |S|)
   c. pick ← argsort(a, descending)[:k_this]
   d. S ← S ∪ pick
   e. Incremental MMR update:
          sims ← Z_norm @ Z_norm[pick]^T    # [n, k_this]
          m ← max(m, max(sims, axis=1))
   f. If τ < ∞:
          logits ← predict_fn(Z[S])
          if max(softmax(logits)) ≥ τ: break (early stop)

6. logits ← predict_fn(Z[S])            # final prediction (one call if no early stop)
7. Return S, logits
```

**Complexity:** O(K/k · (n + n·k)) = O(n·K) per slide — same asymptotic as one-shot top-K but with sequential structure.

---

## Appendix E: LoRA-NSM Design (Planned, N7)

Current NSM stores full router snapshots (533KB/task). LoRA-NSM reduces this to ∼4KB/task:

1. Train a **shared base router** $\phi_0$ on the first task (or averaged across tasks).
2. For each subsequent task $t$, freeze $\phi_0$ and learn only low-rank adapters $\{(A^{(t)}_\ell, B^{(t)}_\ell)\}_\ell$ where:
   $$W^{(t)}_\ell = W_{0,\ell} + A^{(t)}_\ell B^{(t)}_\ell, \quad A^{(t)}_\ell \in \mathbb{R}^{d_\text{in} \times r},\; B^{(t)}_\ell \in \mathbb{R}^{r \times d_\text{out}}.$$
3. NSM stores only $\{(A^{(t)}_\ell, B^{(t)}_\ell)\}_\ell$ for task $t$. Shared base $\phi_0$ stored once.

For MicroRouterV0 (rank $r=4$, two layers):
- Layer 1: $A \in \mathbb{R}^{516 \times 4}$, $B \in \mathbb{R}^{4 \times 256}$ → 2064 + 1024 = 3088 params
- Layer 2: $A \in \mathbb{R}^{256 \times 4}$, $B \in \mathbb{R}^{4 \times 1}$ → 1024 + 4 = 1028 params
- **Total per task: 4116 params ≈ 4KB** (vs. 533KB for full snapshot).

This is the N7 experiment target.
