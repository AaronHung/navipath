# Related Work — draft + citation shortlist (v0.4)

> 正文用 `[cite: ...]` 標記。這些都是**真實、知名**的論文；定稿前我會用 web search
> 把每條補成正確 BibTeX（作者全名 / 卷期 / DOI）。**你不必自己找。**
> Workshop 目標引用數 ~20–30 篇即足夠。

---

## Draft (English, ~1 page)

**MIL for whole-slide images.** Computational pathology typically casts WSI
classification as multiple-instance learning (MIL): a slide is a bag of patch
features pooled into a slide-level prediction [cite: Ilse ICML 2018 (ABMIL)].
Attention- and transformer-based aggregators improve this pooling
[cite: Lu NatBME 2021 (CLAM); Shao NeurIPS 2021 (TransMIL); Li CVPR 2021 (DSMIL)].
These methods assume all (or a fixed sampling of) patches are available at
inference and do not study *which* patches to keep under a budget across tasks.

**Foundation models and vision–language MIL in pathology.** Large pretrained
encoders now provide strong frozen patch features
[cite: Chen NatMed 2024 (UNI); Lu NatMed 2024 (CONCH); Huang NatMed 2023 (PLIP)],
extending image–text pretraining [cite: Radford ICML 2021 (CLIP)] to histology.
Prompt-based vision–language MIL keeps the encoder frozen and learns lightweight
prompts/prototypes for classification, including in a continual setting
[cite: QPMIL-VL (verify)]. We build directly on a frozen CONCH + prompt-based
QPMIL head and add a patch-selection module on top.

**Continual learning.** Methods to combat catastrophic forgetting fall into
regularization [cite: Kirkpatrick PNAS 2017 (EWC)], knowledge distillation
[cite: Li TPAMI 2017 (LwF)], and replay/rehearsal
[cite: Rebuffi CVPR 2017 (iCaRL); Chaudhry ICLR 2019 (A-GEM); Buzzega NeurIPS 2020 (DER)].
Prompt-based continual learning adapts frozen backbones with small learnable
prompts [cite: Wang CVPR 2022 (L2P); Wang ECCV 2022 (DualPrompt); Smith CVPR 2023 (CODA-Prompt)].
Crucially, this literature studies forgetting of the **classifier/representation**;
we instead expose forgetting of a **patch-selection** mechanism, and test whether a
standard replay-free regularizer (EWC, applied to the router) can mitigate it.

**Budget-constrained / efficient WSI inference and patch selection.** To cut
compute, prior work selects informative patches or regions
[cite: Bergner ICLR 2023 (IPS)] or uses attention as a saliency signal
[cite: Ilse ICML 2018]. We compare a learned router against training-free
selectors (random / prototype / semantic similarity) under explicit patch budgets.

**Mixture-of-Experts / routing.** Conditional computation routes inputs to experts
[cite: Shazeer ICLR 2017 (sparsely-gated MoE); Fedus JMLR 2022 (Switch Transformer)],
with load-balancing losses to prevent expert collapse. We borrow only the routing
idea (a per-patch scalar router) and report MoE variants as ablations.

**Gap.** To our knowledge, no prior work studies whether a *trainable patch
selector* itself forgets across tasks in frozen-FM continual WSI classification.
We name and quantify this **selection forgetting**, give a clean same-task
recency-flip causal test, and show it is fully recoverable in principle
(per-task upper bound) yet not fixed by weight-level consolidation.

---

## 可引用清單（grouped；真實論文，BibTeX 定稿前再補）

**Pathology foundation / VLM**
- CONCH — Lu et al., Nature Medicine 2024.
- UNI — Chen et al., Nature Medicine 2024.
- PLIP — Huang et al., Nature Medicine 2023.
- CLIP — Radford et al., ICML 2021.
- QPMIL-VL — (我們的 backbone；確認正確出處/年份)。

**MIL / WSI aggregation**
- ABMIL — Ilse, Tomczak, Welling, ICML 2018.
- CLAM — Lu et al., Nature Biomedical Engineering 2021.
- TransMIL — Shao et al., NeurIPS 2021.
- DSMIL — Li, Li, Eliceiri, CVPR 2021.

**Continual learning**
- EWC — Kirkpatrick et al., PNAS 2017.
- LwF — Li & Hoiem, TPAMI 2017.
- iCaRL — Rebuffi et al., CVPR 2017.
- A-GEM — Chaudhry et al., ICLR 2019.
- DER — Buzzega et al., NeurIPS 2020.

**Prompt-based continual learning**
- L2P — Wang et al., CVPR 2022.
- DualPrompt — Wang et al., ECCV 2022.
- CODA-Prompt — Smith et al., CVPR 2023.

**Patch selection / efficient inference**
- IPS (Iterative Patch Selection) — Bergner et al., ICLR 2023.

**MoE / routing**
- Sparsely-gated MoE — Shazeer et al., ICLR 2017.
- Switch Transformer — Fedus, Zoph, Shazeer, JMLR 2022.

> 小計 ~19 篇核心；寫作時每段再各補 1–2 篇（如病理 CL、TCGA dataset 出處、t-SNE）
> 即可達 ~25 篇，對 workshop 充足。
