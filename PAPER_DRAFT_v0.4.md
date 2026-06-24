# NaviPath — Paper Draft (v0.4, COMPAYL)

> 單一主稿檔；逐節填。各節來源：Method=`METHOD_v0.4.md`、Related=`RELATED_WORK_v0.4.md`、
> 圖表/caption/Tables=`PAPER_OUTLINE_v0.4.md`。`[cite: ...]` 為引用佔位，定稿時補 BibTeX。
> 進度：✅Step1 Intro ｜ ⬜其餘見 todo。

**Title**：Selection Forgetting: Continual Patch Routing Forgets Old Tasks in
Frozen-Foundation-Model Whole-Slide Image Classification

**Abstract.**
Frozen pathology foundation models with prompt-based multiple-instance learning
make continual whole-slide image (WSI) classification attractive: the encoder
cannot drift, removing the classic source of catastrophic forgetting. Yet under
compute budgets a model must still decide *which* of a slide's thousands of patches
to examine—a learnable step whose behavior under continual learning has not been
studied. We attach a lightweight trainable patch router to a frozen CONCH+QPMIL
backbone and, because the predictor never consumes router-modified features
(backbone forgetting is zero by construction), isolate the selection mechanism. On
the most-recently-learned task the router selects more informative patches than
random, prototype, and semantic heuristics (+2.5–6.7 pp at a 64-patch budget; 6/6
across folds and orders). On *old* tasks, however, its selection quality collapses
**below random** (6/6), a phenomenon we call **selection forgetting**. A same-task
recency-flip test—varying only whether a task was learned last vs first—drops
accuracy from ~0.9 to 0.33–0.40 and holds even for the sample-abundant lung task,
ruling out difficulty/size confounds; a feature-space analysis shows the forgotten
router *confidently mis-prioritizes* rather than degrading to noise. Finally,
selection forgetting is recoverable in principle—a per-task router restores
old-task selection from 0.33 to 0.93 (all-patch level; 3/3 folds)—but a standard
replay-free fix (EWC-on-router) does not (0.40; 0/3), motivating selection-aware
continual learning. We release code, configs, and figures.

---

## 1. Introduction

Whole-slide images (WSIs) are gigapixel and are usually classified under the
multiple-instance learning (MIL) paradigm, where a slide is treated as a bag of
many patch features that are aggregated into a slide-level prediction
[cite: Ilse ICML 2018; Lu NatBME 2021]. Recent pathology foundation models supply
strong **frozen** patch encoders [cite: Lu NatMed 2024 (CONCH); Chen NatMed 2024 (UNI)],
and prompt-based vision–language MIL adapts them to new tasks by learning only a
small set of prompts/prototypes while the encoder stays frozen
[cite: QPMIL-VL (verify)]. Freezing the encoder is attractive for continual
learning: the representation cannot drift, so the classic source of catastrophic
forgetting is largely removed. Yet a second, practical problem remains: a slide
contains thousands of patches, and under compute or latency budgets a model must
decide **which patches to look at**. This selection step is itself learnable, and
whether it survives continual learning has, to our knowledge, not been examined.

Most continual-learning research targets forgetting of the **classifier or
representation**, and mitigates it with regularization, distillation, or replay
[cite: Kirkpatrick PNAS 2017 (EWC); Li TPAMI 2017 (LwF); Rebuffi CVPR 2017 (iCaRL)],
including prompt-based variants for frozen backbones
[cite: Wang CVPR 2022 (L2P); Wang ECCV 2022 (DualPrompt)]. In contrast, the
**patch-selection / attention mechanism** that decides what the model attends to
has not been studied as a locus of forgetting. Our setting isolates exactly this
question: because the prediction backbone is frozen and never consumes
router-modified features, backbone-level forgetting is zero **by construction**,
so any degradation we observe must come from the selection module alone.

We add a lightweight, continually-trained patch router on top of a frozen
CONCH+QPMIL backbone: it scores each patch and keeps the Top-$K$, which are then
classified by the unchanged backbone. We first verified that such a router is
useful—on the **most-recently-learned** task it selects more informative patches
than random, prototype, and semantic heuristics at tight budgets (e.g.\ +2.5–6.7
pp at $K{=}64$, 6/6 GO across folds and orders). However, when we evaluate the
router on **old** tasks after it has been trained on later ones, its selection
quality **collapses below the random baseline** (6/6 NO-GO). We call this
phenomenon **selection forgetting**. A clean, single-variable test confirms the
cause is recency rather than task difficulty or sample size: for the *same* task
and test set, merely changing whether it was learned **last** vs **first** flips
router accuracy from ~0.9 to 0.33–0.40, and this holds even for the
sample-abundant lung task. A feature-space visualization shows the forgotten
router does not become random noise but **confidently mis-prioritizes**, selecting
a patch population made salient by later tasks—explaining why it can fall *below*
random.

Finally, we ask whether selection forgetting is recoverable. Storing a separate
router per task (an upper bound) **fully restores** old-task selection (the oldest
task's $K{=}64$ accuracy returns from 0.33 to 0.93, matching all-patch accuracy and
beating all heuristics; 3/3 folds), proving the signal is intact and the failure is
genuine forgetting. A standard replay-free fix—Elastic Weight Consolidation applied
to the router—does **not** suffice (0.40; 0/3), indicating that weight-level
regularization is inadequate for selection forgetting and that selection-aware
continual methods are needed. This question—how an autonomous selection/routing
policy should persist across tasks—connects to the emerging interest in
\emph{agentic AI for pathology} highlighted by the venue.

**Contributions.**
1. We identify and name **selection forgetting**: in frozen-foundation-model
   continual WSI classification, a trainable patch router forgets how to select
   patches for old tasks, even when the predictor cannot forget.
2. We provide a **clean causal demonstration** via a same-task recency-flip test
   (controlling task, data, and router; varying only recency) and **rule out**
   sample-size/difficulty confounds using the sample-abundant lung task (6/6).
3. We give a **mechanistic** account: the forgotten router mis-prioritizes
   (selecting later-task-salient patches) rather than degrading to noise, which
   explains sub-random behavior.
4. We show selection forgetting is **recoverable in principle** (per-task router
   upper bound) but **not fixed by EWC-on-router**, motivating selection-aware
   continual learning as future work.

---

## 2. Related Work

**MIL for whole-slide images.** Computational pathology typically casts WSI
classification as multiple-instance learning (MIL): a slide is a bag of patch
features pooled into a slide-level prediction [cite: Ilse ICML 2018 (ABMIL)].
Attention- and transformer-based aggregators improve this pooling
[cite: Lu NatBME 2021 (CLAM); Shao NeurIPS 2021 (TransMIL); Li CVPR 2021 (DSMIL)].
These methods assume all (or a fixed sampling of) patches are available at
inference and do not study *which* patches to keep under a budget across tasks.

**Foundation models and vision–language MIL in pathology.** Large pretrained
encoders provide strong frozen patch features
[cite: Chen NatMed 2024 (UNI); Lu NatMed 2024 (CONCH); Huang NatMed 2023 (PLIP)],
extending image–text pretraining [cite: Radford ICML 2021 (CLIP)] to histology.
Prompt-based vision–language MIL keeps the encoder frozen and learns lightweight
prompts/prototypes for classification, including continually
[cite: QPMIL-VL (verify)]. We build on a frozen CONCH + prompt-based QPMIL head and
add a patch-selection module on top.

**Continual learning.** Methods against catastrophic forgetting fall into
regularization [cite: Kirkpatrick PNAS 2017 (EWC)], distillation
[cite: Li TPAMI 2017 (LwF)], and replay
[cite: Rebuffi CVPR 2017 (iCaRL); Chaudhry ICLR 2019 (A-GEM); Buzzega NeurIPS 2020 (DER)].
Prompt-based continual learning adapts frozen backbones with small prompts
[cite: Wang CVPR 2022 (L2P); Wang ECCV 2022 (DualPrompt); Smith CVPR 2023 (CODA-Prompt)].
Crucially, this literature studies forgetting of the **classifier/representation**;
we instead expose forgetting of a **patch-selection** mechanism and test whether a
standard replay-free regularizer (EWC, on the router) can mitigate it.

**Budget-constrained inference and patch selection.** To cut compute, prior work
selects informative patches/regions [cite: Bergner ICLR 2023 (IPS)] or uses
attention as saliency [cite: Ilse ICML 2018]. We compare a learned router against
training-free selectors (random / prototype / semantic) under explicit budgets.

**Mixture-of-Experts / routing.** Conditional computation routes inputs to experts
[cite: Shazeer ICLR 2017; Fedus JMLR 2022 (Switch Transformer)], with
load-balancing to prevent expert collapse. We borrow only the routing idea (a
per-patch scalar router) and report MoE variants as ablations.

**Gap.** To our knowledge, no prior work studies whether a *trainable patch
selector* itself forgets across tasks in frozen-FM continual WSI classification.
We name and quantify this **selection forgetting**, give a clean same-task
recency-flip causal test, and show it is recoverable in principle (per-task upper
bound) yet not fixed by weight-level consolidation.

## 3. Method

**Notation.** A slide has $n$ patches with frozen CONCH features
$z_i\in\mathbb{R}^{512}$; $\hat{(\cdot)}$ denotes L2-normalization,
$\{t_c\}_{c=1}^{C}$ class-text features, $\{p_m\}_{m=1}^{M}$ prototype features,
and $\phi$ the router parameters.

### 3.1 Problem setup
We study class-incremental WSI classification under a fixed feature budget. Tasks
$t=1,\dots,T$ arrive sequentially; after task $t$ the model classifies slides from
all seen tasks using at most $K$ patches per slide ($K\ll n$). We separate two
sub-problems usually conflated: *what to predict* (classifier) and *what to look
at* (selection).

### 3.2 Decoupled prediction backbone
Prediction reuses a prompt-based continual MIL head (QPMIL) on the frozen CONCH
features. The prediction path never consumes router- or expert-transformed
features:
```latex
\hat{y} = \arg\max\; g_{\theta^\ast}\!\big(\{z_i\}_{i\in\mathcal{S}}\big),
\qquad \theta^\ast \text{ frozen}\;\Rightarrow\; \mathrm{Forgetting}\equiv 0 . \tag{4}
```
Hence backbone-level forgetting is zero **by construction** (Fig. 6), an identity
we report transparently rather than as a contribution.

### 3.3 Trainable patch router
For each patch we compute four task-count-invariant summary statistics (max and
entropy of class-text similarity; max and mean prototype similarity) and map
$[z_i;s_i]$ through a two-layer MLP to a scalar importance $r_i$:
```latex
s_i = \Big[\max_c \hat{z}_i^\top \hat{t}_c,\;
           H(\mathrm{softmax}_c\,\hat{z}_i^\top \hat{t}),\;
           \max_m \hat{z}_i^\top \hat{p}_m,\;
           \tfrac{1}{M}\sum_m \hat{z}_i^\top \hat{p}_m\Big],
\qquad r_i = \mathrm{MLP}_\phi([z_i;s_i]). \tag{1}
```
At inference we keep the Top-$K$ patches:
```latex
\mathcal{S}_K = \operatorname*{Top\text{-}K}_i r_i,\qquad |\mathcal{S}_K|=K. \tag{2}
```
The router (≈132K params) is the only component trained across tasks; the backbone
stays frozen.

### 3.4 Differentiable training
Because hard Top-$K$ is non-differentiable, we train with a soft-route objective:
scores over the selected set are softmax-normalized into weights, aggregated into a
bag embedding, and classified by the frozen text head; gradients reach $\phi$ only
through the weights.
```latex
w_i=\frac{\exp(r_i)}{\sum_{j\in\mathcal{S}_K}\exp(r_j)},\quad
\bar{z}=\widehat{\textstyle\sum_{i\in\mathcal{S}_K} w_i z_i},\quad
\mathcal{L}_{\text{route}}=\mathrm{CE}(\sigma\,\bar{z}F_{txt}^\top,\,y). \tag{3}
```

### 3.5 Selection baselines
We compare the router against three training-free selectors that pick the Top-$K$
patches by (i) random; (ii) prototype similarity $\max_m \hat z_i^\top\hat p_m$;
(iii) semantic similarity $\max_c \hat z_i^\top\hat t_c$. All feed the same frozen
backbone, so any accuracy gap is due to selection alone (Fig. S1).

### 3.6 Router consolidation (mitigation)
To test recoverability we add replay-free consolidation on the router: after each
task we estimate a diagonal Fisher and penalize drift of important weights on later
tasks (EWC-on-router); we also report a **per-task router** (one stored per task)
as an empirical upper bound.
```latex
F^{(t)}_j=\mathbb{E}_{\mathcal{D}_t}\!\big[(\partial \mathcal{L}_{\text{route}}/\partial \phi_j)^2\big],\;
\phi^{\ast(t)}=\phi,\quad
\mathcal{L}^{(t+1)}=\mathcal{L}_{\text{route}}+\lambda\!\sum_{\tau\le t}\!\sum_j F^{(\tau)}_j(\phi_j-\phi^{\ast(\tau)}_j)^2. \tag{5}
```

### 3.7 Evaluation protocol
Four TCGA tasks, two orders (paper/reverse), three folds, budgets
$K\in\{\text{All},256,128,64,32\}$. We report Top-1 accuracy and a GO/NO-GO
criterion (router beats the best heuristic at a finite budget), evaluating both the
most-recent task and each old task to expose recency-dependent selection forgetting
(Fig. 4).

## 4. Experiments and Results

### 4.0 Experimental setup
**Data and tasks.** We use four TCGA cohorts as a class-incremental sequence of
binary subtyping tasks—lung (NSCLC), breast (BRCA), renal (RCC), and esophageal
(ESCA)—each contributing two subtypes, for eight classes in total (label shifts
$0,2,4,6$) [cite: TCGA]. Patches are encoded once by a frozen CONCH foundation
model into 512-d features [cite: Lu NatMed 2024 (CONCH)]; the encoder is never
updated. We evaluate two task orders, **paper** (lung→brca→rcc→esca) and
**reverse** (esca→rcc→brca→lung), each over **3 folds** (official train/val/test
splits), and report the mean. The reverse order makes the small-sample esca the
*oldest* task and the large-sample lung the *most-recent*, and vice-versa for
paper—enabling the same-task recency comparison of §4.4.

**Backbone (prediction path).** We adopt a prompt-based vision–language MIL head
(QPMIL) on the frozen CONCH features [cite: QPMIL-VL (verify)], trained per task
with Adam (lr $1\mathrm{e}{-3}$, weight decay $5\mathrm{e}{-4}$, 12 epochs/task).
The same optimizer/epoch budget is used for the QPMIL baseline and our setup, so
backbone-accuracy comparisons (Table 1) are matched.

**Router (selection path).** The MicroRouter is a two-layer MLP (≈132K parameters)
mapping each patch's feature concatenated with four task-count-invariant summary
statistics to a scalar score; it is trained continually with Adam (lr
$5\mathrm{e}{-4}$, weight decay $1\mathrm{e}{-4}$, 5 epochs/task) using the
differentiable soft-route objective (§3), with the backbone frozen throughout.

**Budgets and selectors.** At inference we keep the Top-$K$ patches for
$K\in\{\text{All},256,128,64,32\}$ and feed them to the frozen head. We compare the
learned router against three training-free selectors: **random**, **prototype**
(max cosine similarity to the QPMIL prototype pool), and **semantic** (max cosine
similarity to class-text features). All selectors feed the identical backbone, so
accuracy differences reflect selection only (Fig. S1).

**GO/NO-GO criterion.** For a given task and budget we say the router is **GO** if
it exceeds the best of the three heuristics at $K{=}64$
(router@64 $>$ $\max$\{random, prototype, semantic\}@64), and **NO-GO** otherwise.
We additionally report full budget curves. (Note: esca has a small test set,
motivating the lung-based confound check in §4.4.)

### 4.1 Backbone accuracy and the necessity of decoupling
Under matched training (12 epochs/task, Adam lr=1e-3, wd=5e-4) our frozen-backbone
setup attains continual accuracy on par with the QPMIL baseline (Table 1): QPMIL
reaches ACC 0.924/0.917 (paper/reverse) with mild forgetting (0.017/0.041), while
the decoupled NaviPath reaches 0.879/0.886 with **Forgetting = 0**. We stress that
this zero is an **identity, not a contribution**: because the prediction path is a
frozen backbone that never consumes router- or expert-modified features, per-task
accuracy is flat by construction (Fig. 6, R-matrix). Decoupling is *necessary*:
an earlier non-decoupled variant, in which trainable experts transformed the
features fed back to the frozen backbone, induced severe interference: accuracy
collapsed to 0.378/0.218 and forgetting rose to 0.735/0.950 (paper/reverse), far
worse than the QPMIL baseline. We therefore fix the predictor and study the only
component that learns continually—the patch router.

### 4.2 The router helps on recently-learned tasks
On the **most-recently-learned** task, the learned router selects more informative
patches than all training-free heuristics at tight budgets (Fig. 2, Table 2). At
$K{=}64$ it reaches 0.956 (esca, recent) and 0.922 (lung, recent), versus the best
heuristic 0.889 and 0.897 respectively, i.e.\ +2.5–6.7 pp; using only 64 of
thousands of patches nearly matches all-patch accuracy. This holds for both
"recent" tasks across all folds and orders (6/6 GO), establishing that the router
*can* learn a useful, budget-efficient selection policy.

### 4.3 Selection forgetting on old tasks
The picture reverses for **old** tasks—those learned early and then followed by
other tasks. Here the router's selection quality drops **below the random
baseline** at tight budgets (Fig. 3, Table 2): $K{=}64$ accuracy falls to 0.333
(esca, old) and 0.397 (lung, old), against best heuristics of 0.822 and 0.813
(6/6 NO-GO). Because the predictor is unchanged, this degradation is attributable
to the selection module alone. We term this **selection forgetting**: the router
forgets how to choose patches for tasks it learned earlier.

### 4.4 Same-task recency flip: a clean causal test
To rule out that old tasks are simply harder or smaller, we compare the *identical*
task and test set under the two orders, changing only whether the task was learned
**last** (recent) or **first** (old). Selection accuracy flips sharply (Fig. 4):
lung 0.922→0.397 and esca 0.956→0.333 at $K{=}64$. Holding task, data, and router
architecture fixed and varying only recency isolates recency as the cause.
Critically, the flip also occurs for **lung**, the most sample-abundant task,
excluding sample-size/difficulty confounds. The effect replicates across all
folds (6/6).

### 4.5 Mechanism: confident mis-prioritization, not noise
A shared-embedding feature-space visualization (Fig. 5) scores the *same* esca
slide with the router that learned esca recently vs the router that learned it long
ago. The forgotten router does **not** collapse to uniform/noisy scores; it still
produces structured scores but concentrates its Top-$K$ on a *different* patch
sub-population—one made salient by the later tasks. This **confident
mis-prioritization** explains why old-task selection can fall *below* random
(actively wrong) rather than merely matching it (no signal).

### 4.6 Mitigation: recoverable in principle, but not by EWC
Finally, we test recovery on the oldest task (Table 3, 3-fold). Storing a
**separate router per task** (an empirical upper bound) **fully restores** old-task
selection: $K{=}64$ accuracy rises from 0.333 to **0.933**—flat across all budgets
(matching all-patch accuracy) and above every heuristic (GO 3/3). This proves the
selection signal remains intact and the failure is genuine forgetting. In contrast,
a standard replay-free regularizer, **EWC-on-router**, barely helps (0.333→0.400 at
$K{=}64$, still below the 0.822 best heuristic; NO-GO 0/3), while leaving
recent-task selection unharmed (e.g.\ reverse-lung remains GO). Weight-level
consolidation is thus insufficient for selection forgetting, motivating
selection-aware approaches (Sec. 5).

### Tables

**Table 1.** Continual accuracy (backbone level, mean±std over 3 folds). QPMIL
baseline and NaviPath use matched training (12 epochs/task, Adam lr 1e-3, wd 5e-4).
NaviPath's Forgetting$=0$ is a decoupling identity (Fig. 6), not a contribution;
QPMIL ACC $\ge$ NaviPath, so our contribution is the selection analysis, not ACC.

| Method | Order | ACC | Forgetting | BWT |
|---|---|---|---|---|
| QPMIL baseline | paper | 0.924±0.016 | 0.017±0.022 | −0.017±0.022 |
| QPMIL baseline | reverse | 0.917±0.026 | 0.041±0.023 | −0.041±0.023 |
| NaviPath (decoupled) | paper | 0.879±0.030 | 0.000 | 0.000 |
| NaviPath (decoupled) | reverse | 0.886±0.030 | 0.000 | 0.000 |

**Table 2.** Router patch selection, recent vs old (router accuracy at budget $K$,
mean over 3 folds). "best heur" $=\max$\{random, prototype, semantic\}; GO if
router@64 $>$ best heur@64. The router is GO on both recent tasks (6/6) and NO-GO
on both old tasks (6/6).

| Task | Recency | @256 | @128 | @64 | @32 | best heur@64 | GO |
|---|---|---|---|---|---|---|---|
| esca | RECENT (paper, last) | 0.956 | 0.956 | **0.956** | 0.933 | 0.889 | ✓ (3/3) |
| esca | OLD (reverse, first) | 0.511 | 0.400 | **0.333** | 0.333 | 0.822 | ✗ (0/3) |
| lung | RECENT (reverse, last) | 0.904 | 0.915 | **0.922** | 0.918 | 0.897 | ✓ (3/3) |
| lung | OLD (paper, first) | 0.512 | 0.453 | **0.397** | 0.353 | 0.813 | ✗ (0/3) |

**Table 3.** Mitigation on the oldest task (esca, reverse order; router accuracy at
budget $K$, mean over 3 folds). The per-task router (upper bound) fully restores
old-task selection to recent-task levels (GO 3/3), whereas EWC-on-router barely
moves the needle (NO-GO 0/3): weight-level consolidation is insufficient for
selection forgetting.

| Router consolidation | @256 | @128 | @64 | @32 | best heur@64 | GO |
|---|---|---|---|---|---|---|
| none (continual) | 0.511 | 0.400 | **0.333** | 0.333 | 0.822 | ✗ (0/3) |
| EWC-on-router (replay-free) | 0.622 | 0.422 | **0.400** | 0.267 | 0.822 | ✗ (0/3) |
| per-task router (upper bound) | 0.933 | 0.933 | **0.933** | 0.933 | 0.822 | ✓ (3/3) |

### Figure captions

**Fig. 1 (`Fig1_arch`).** NaviPath's decoupled design. A frozen CONCH + QPMIL
prediction path (blue) shares patch features with a lightweight, continually-trained
patch router (orange) that scores patches and keeps the Top-$K$; the selected
patches are classified by the unchanged backbone. Because the predictor never
consumes router-modified features, backbone forgetting is zero by construction, so
the router is the sole locus of continual learning. Plan B (purple, dashed)
consolidates the router (per-task / EWC).

**Fig. 2 (`P0_router_v0`).** Patch-budget accuracy on the most-recently-learned
task. The learned router matches or exceeds random, prototype, and semantic
selectors across budgets, with +2.5–6.7 pp at $K{=}64$ (6/6 GO across folds and
orders); 64 of thousands of patches nearly match all-patch accuracy.

**Fig. 3 (`P0_oldtask_budget`).** Patch-budget accuracy on an old task (learned
first, then overwritten). The router falls below random at tight budgets
(router@64 $<$ random@64; 6/6 NO-GO), i.e. selection forgetting; training-free
heuristics are unaffected.

**Fig. 4 (`P0b_recency_flip`, core).** Same-task recency flip. For the identical
task and test set, the router selects well when the task was learned last
(solid, ~0.9) but collapses below random when the same task was learned first and
overwritten (dashed, 0.33–0.40). Holds for both lung and esca, excluding
sample-size/difficulty confounds.

**Fig. 5 (`P2contrast_esca_fold1`).** Mechanism. The same esca slide in CONCH
feature space (shared t-SNE), scored by the router when esca was recent (left) vs
old/overwritten (right). The forgotten router keeps structured scores but
concentrates its Top-$K$ on a different, later-task-salient sub-population—confident
mis-prioritization, explaining sub-random behavior.

**Fig. 6 (`P1_r_matrix`).** Per-task accuracy $R[i,j]$ (accuracy on task $j$ after
learning task $i$). NaviPath columns are flat by construction (decoupled frozen
backbone $\Rightarrow$ Forgetting$=0$ is an identity), while the QPMIL baseline
shows mild genuine drift; we surface this to locate our contribution in the
selection analysis.

**Fig. S1 (`FigS1_arch`, appendix).** Evaluation protocol. The router and three
training-free selectors each pick a Top-$K$ subset from the same frozen features
and feed the same frozen head; we report accuracy at budget $K$ and a GO/NO-GO
criterion. Only the selected set differs, so accuracy gaps reflect selection alone.

## 5. Discussion and Limitations

**Selection forgetting is distinct from classifier forgetting.** Our setup
deliberately removes representation/classifier drift (the backbone is frozen, so
Forgetting$=0$ by construction), which lets us attribute every old-task
degradation to the selection module. The result is a forgetting phenomenon that
the continual-learning literature, focused on classifier/representation, does not
capture: even when *what to predict* cannot be forgotten, *what to look at* can.
We believe this matters broadly for budget-constrained inference with frozen
foundation models, where a learned gate/router/selector sits in front of a fixed
predictor.

**Why does EWC fail while a per-task router fully recovers?** The per-task
upper bound restoring old-task accuracy to the all-patch level (0.33→0.93) shows
the information needed to select well is still present in the frozen features;
the single continual router simply overwrites the *ranking* it had learned for old
tasks. EWC penalizes drift of individual router weights, but the quantity that
matters here is the induced *ordering* of patch scores, a global, highly nonlinear
function of those weights. Protecting individual weights is therefore a poor proxy
for protecting a selection policy—suggesting that selection-aware objectives
(e.g.\ distilling old-task score rankings, or function-space/rank-preserving
regularization) are a more promising direction than weight-space penalties.

**On the honesty of Forgetting$=0$.** We report this as an identity, not a
contribution (Fig. 6), precisely to avoid over-claiming: the decoupled design buys
zero backbone forgetting at the cost of the experts/router contributing nothing to
the main accuracy metric. Our contribution is the analysis of the selection path,
not a new state-of-the-art accuracy.

**Limitations.** (i) We study four TCGA cohorts from a single source; broader
datasets and longer task sequences would strengthen generality. (ii) The oldest
task in the reverse order (esca) has a small test set; we mitigate this with the
sample-abundant lung task, which shows the same recency flip, but larger cohorts
remain desirable. (iii) The router is a scalar v0; richer selectors (set-level or
attention-based) may exhibit different dynamics. (iv) Our mitigation study (Table 3)
is demonstrated on the oldest task over all 3 folds; extending the per-task/EWC
comparison to every task and order is left as future work, though the upper-bound
recovery and EWC insufficiency are highly consistent across folds (3/3).

## 6. Conclusion
We show that adding a trainable patch router to a frozen-foundation-model continual
WSI classifier introduces **selection forgetting**: the router selects well for the
most-recent task but, for older tasks, falls below a random baseline. A same-task
recency-flip test establishes recency as the cause and excludes sample-size
confounds, and a feature-space analysis reveals confident mis-prioritization rather
than noise. The phenomenon is recoverable in principle—a per-task router restores
old-task selection to the all-patch level—yet a standard replay-free regularizer
(EWC-on-router) does not fix it. We argue that **selection-aware continual
learning** is needed when a learned selector front-ends a frozen predictor, and we
release our code, configs, and figures to support follow-up work.

## References
[Step 8：web search 補 BibTeX]
