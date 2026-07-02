#!/usr/bin/env python3
"""Two-panel architecture figure: Training phase vs Inference phase.

Addresses teacher Q2/Q3: the figure must show BOTH phases, with the training
panel containing the skill bank and the training loss (cf. Pin-Zhen's figure).

Output: experiment_visualize/figs/fig_arch_train_infer.{pdf,png}
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-navipath")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUTDIR, exist_ok=True)

# Colors
C_FROZEN = "#cfe8ff"   # frozen = light blue
C_FROZEN_E = "#4a90d9"
C_TRAIN = "#ffe0b3"    # trained = orange
C_TRAIN_E = "#e08a1e"
C_LOSS = "#ffd6d6"     # loss = red
C_LOSS_E = "#d73027"
C_BANK = "#d9f0d3"     # skill bank = green
C_BANK_E = "#4daf4a"
C_SEL = "#f0e0ff"      # selection = purple
C_SEL_E = "#9467bd"


def box(ax, x, y, w, h, text, fc, ec, fs=9, fw="normal", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                       linewidth=1.6, facecolor=fc, edgecolor=ec, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=fw, zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, text="", color="#333333", fs=8, ls="-", rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.5, color=color, zorder=1,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.015, text, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=3)


fig, (axT, axI) = plt.subplots(1, 2, figsize=(15, 6.2))

# ══════════════════════════════════════════════════════════════════════════
# LEFT PANEL: TRAINING PHASE
# ══════════════════════════════════════════════════════════════════════════
ax = axT
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(a) Training Phase  —  learn navigation skill for task $t$",
             fontsize=12, fontweight="bold", pad=10)

# WSI -> frozen encoder
box(ax, 0.02, 0.72, 0.16, 0.13, "WSI\n(task $t$)", "#f0f0f0", "#888888", fs=9)
box(ax, 0.02, 0.50, 0.16, 0.14, "Frozen\nEncoder\n(CONCH)", C_FROZEN, C_FROZEN_E, fs=8.5)
arrow(ax, 0.10, 0.72, 0.10, 0.645)
ax.text(0.10, 0.44, r"$Z\in\mathbb{R}^{n\times512}$", ha="center", fontsize=8.5)

# frozen text + prototype (context)
box(ax, 0.02, 0.24, 0.16, 0.11, "class text $T$\n+ prototype $P$\n(frozen)", C_FROZEN, C_FROZEN_E, fs=7.5)

# MicroRouter (trained)
box(ax, 0.30, 0.50, 0.20, 0.16, "MicroRouter $\\phi$\n(2-layer MLP,\nTRAINED)", C_TRAIN, C_TRAIN_E, fs=8.5, fw="bold")
arrow(ax, 0.18, 0.57, 0.30, 0.58, r"$z_i$")
arrow(ax, 0.18, 0.29, 0.29, 0.51, r"$s_i$ (Eq.1)", rad=0.15)
ax.text(0.40, 0.46, r"score $r_i=\mathrm{MLP}_\phi([z_i;s_i])$ (Eq.2)", ha="center", fontsize=7.5)

# top-K soft aggregation
box(ax, 0.58, 0.50, 0.18, 0.16, "Top-K +\nsoftmax weights\n$\\bar z=\\sum w_i z_i$", C_SEL, C_SEL_E, fs=8)
arrow(ax, 0.50, 0.58, 0.58, 0.58)

# frozen text head -> logits
box(ax, 0.80, 0.50, 0.17, 0.16, "Frozen text\nclassifier\n$\\sigma\\,\\bar z\\,T^\\top$", C_FROZEN, C_FROZEN_E, fs=8)
arrow(ax, 0.76, 0.58, 0.80, 0.58)

# Loss
box(ax, 0.80, 0.26, 0.17, 0.13, "$\\mathcal{L}_{route}=$\nCE(logits, $y$)\n(Eq.3)", C_LOSS, C_LOSS_E, fs=8, fw="bold")
arrow(ax, 0.885, 0.50, 0.885, 0.39, color=C_LOSS_E)

# gradient back to router (dashed red)
arrow(ax, 0.80, 0.30, 0.40, 0.49, color=C_LOSS_E, ls="--", rad=-0.25)
ax.text(0.55, 0.34, "gradient (only via $w_i$)", ha="center", fontsize=7.5,
        color=C_LOSS_E, style="italic")

# Skill bank (store after task)
box(ax, 0.30, 0.05, 0.30, 0.14, "Navigation Skill Memory (NSM)\nstore $\\phi^{(t)}$ after task $t$\n(Eq.9)",
    C_BANK, C_BANK_E, fs=8.5, fw="bold")
arrow(ax, 0.40, 0.50, 0.44, 0.195, color=C_BANK_E, ls="-", rad=0.2)
ax.text(0.30, 0.30, "snapshot\non task end", ha="center", fontsize=7.5, color=C_BANK_E, style="italic")

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: INFERENCE PHASE
# ══════════════════════════════════════════════════════════════════════════
ax = axI
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("(b) Inference Phase  —  everything frozen, no loss",
             fontsize=12, fontweight="bold", pad=10)

# WSI -> encoder
box(ax, 0.02, 0.72, 0.16, 0.13, "WSI\n(test slide)", "#f0f0f0", "#888888", fs=9)
box(ax, 0.02, 0.50, 0.16, 0.14, "Frozen\nEncoder\n(CONCH)", C_FROZEN, C_FROZEN_E, fs=8.5)
arrow(ax, 0.10, 0.72, 0.10, 0.645)
ax.text(0.10, 0.44, r"$Z\in\mathbb{R}^{n\times512}$", ha="center", fontsize=8.5)

# skill bank + context gate
box(ax, 0.02, 0.05, 0.24, 0.15, "NSM skill bank\n$\\{\\phi^{(1)},\\dots,\\phi^{(T)}\\}$", C_BANK, C_BANK_E, fs=8.5, fw="bold")
box(ax, 0.31, 0.05, 0.17, 0.15, "Context Gate\n(oracle:\ntask id)", C_SEL, C_SEL_E, fs=8)
arrow(ax, 0.26, 0.125, 0.31, 0.125, color=C_BANK_E)

# MicroRouter (frozen, loaded)
box(ax, 0.30, 0.50, 0.20, 0.16, "MicroRouter $\\phi^{(t)}$\n(loaded, FROZEN)", C_FROZEN, C_FROZEN_E, fs=8.5)
arrow(ax, 0.18, 0.57, 0.30, 0.58, r"$z_i,s_i$")
arrow(ax, 0.40, 0.20, 0.40, 0.49, color=C_SEL_E, ls="--", rad=0)
ax.text(0.52, 0.34, "retrieve skill\nfor this task", ha="center", fontsize=7.5, color=C_SEL_E, style="italic")

# SBO / top-K selection
box(ax, 0.58, 0.50, 0.18, 0.16, "Top-K select\n(optional SBO\nmulti-round, Eq.4-8)", C_SEL, C_SEL_E, fs=7.8)
arrow(ax, 0.50, 0.58, 0.58, 0.58, r"$r_i$")

# frozen aggregate & predict
box(ax, 0.80, 0.50, 0.17, 0.16, "Frozen\naggregate &\npredict $\\hat y$", C_FROZEN, C_FROZEN_E, fs=8)
arrow(ax, 0.76, 0.58, 0.80, 0.58, r"$Z_\mathcal{S}$")

# prediction out
ax.text(0.885, 0.44, "prediction $\\hat y$", ha="center", fontsize=9, fontweight="bold")

# legend (shared)
legend_elems = [
    FancyBboxPatch((0,0),1,1, facecolor=C_FROZEN, edgecolor=C_FROZEN_E, label="Frozen (no grad)"),
    FancyBboxPatch((0,0),1,1, facecolor=C_TRAIN, edgecolor=C_TRAIN_E, label="Trained (router $\\phi$)"),
    FancyBboxPatch((0,0),1,1, facecolor=C_LOSS, edgecolor=C_LOSS_E, label="Loss / gradient"),
    FancyBboxPatch((0,0),1,1, facecolor=C_BANK, edgecolor=C_BANK_E, label="Skill bank (NSM)"),
    FancyBboxPatch((0,0),1,1, facecolor=C_SEL, edgecolor=C_SEL_E, label="Selection / gate"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("NaviPath-CL Architecture: Training vs. Inference\n"
             "(backbone frozen throughout; only the MicroRouter is trained, then snapshotted into NSM)",
             fontsize=12.5, y=1.02)
fig.tight_layout(rect=(0, 0.03, 1, 1))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"fig_arch_train_infer.{ext}"),
                bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"[saved] {OUTDIR}/fig_arch_train_infer.{{pdf,png}}")
