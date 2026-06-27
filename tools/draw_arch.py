"""Fig 1 — NaviPath architecture (vector PDF/PNG, no PowerPoint).

Produces TWO figures in one run:
  outputs/figs/Fig1_arch.{pdf,png}   - MAIN: clean two-path view (decoupling)
  outputs/figs/FigS1_arch.{pdf,png}  - APPENDIX: evaluation protocol —
                                       router vs training-free selectors feeding
                                       the SAME frozen backbone -> GO/NO-GO

Edit boxes/text below and re-run `python tools/draw_arch.py`; output is vector
PDF for the paper. No PowerPoint needed.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.linewidth": 0,
})

FROZEN, FROZEN_E = "#cfe3f7", "#3776c0"      # blue
TRAIN, TRAIN_E = "#fde0c7", "#e08a37"        # orange
PLANB, PLANB_E = "#ece7f7", "#8a6fc0"        # purple
NEUT, NEUT_E = "#ffffff", "#444444"          # white
GREY, GREY_E = "#f1f1f1", "#9a9a9a"


def box(ax, x, y, w, h, text, fc, ec, dashed=False, fs=9.5, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.6, edgecolor=ec, facecolor=fc,
                       linestyle="--" if dashed else "-", mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=5, weight=weight)


def arrow(ax, p1, p2, ec="#444444", dashed=False, rad=0.0, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=ec,
        connectionstyle=f"arc3,rad={rad}", linestyle="--" if dashed else "-",
        shrinkA=2, shrinkB=2, zorder=3))


def banner(ax, x, y, w, h):
    box(ax, x, y, w, h, "", GREY, GREY_E, fs=9)
    ax.text(x + w / 2, y + h * 0.66,
            "Continual task stream:  tasks $t = 1 \\dots T$",
            ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + h * 0.30,
            "navigation policy trained sequentially "
            "$\\Rightarrow$ navigation forgetting on old tasks",
            ha="center", va="center", fontsize=9)


def legend(ax, x, y):
    items = [(FROZEN, FROZEN_E, "frozen backbone", False),
             (TRAIN, TRAIN_E, "navigation policy (CNL)", False),
             (PLANB, PLANB_E, "Navigation Skill Memory", True)]
    for i, (fc, ec, lab, dash) in enumerate(items):
        yy = y - i * 0.62
        box(ax, x, yy, 0.5, 0.4, "", fc, ec, dashed=dash)
        ax.text(x + 0.68, yy + 0.2, lab, fontsize=8.8, va="center")


# ---------------------------------------------------------------------------
def fig_a(path):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # input row
    box(ax, 0.3, 6.85, 2.3, 1.3, "WSI\npatches $x_i$", NEUT, NEUT_E)
    box(ax, 3.0, 6.85, 2.6, 1.3, "CONCH encoder\n(frozen)", FROZEN, FROZEN_E)
    box(ax, 6.0, 6.85, 2.6, 1.3, "patch feats\n$Z=\\{z_i\\}$  $n{\\times}512$",
        NEUT, NEUT_E)
    arrow(ax, (2.6, 7.5), (3.0, 7.5))
    arrow(ax, (5.6, 7.5), (6.0, 7.5))

    # diagnostic backbone (frozen)
    ax.text(10.5, 8.42, "Diagnostic backbone  (frozen; QPMIL-VL instance)",
            fontsize=10, color=FROZEN_E, weight="bold", ha="center")
    box(ax, 9.6, 6.55, 3.1, 1.65,
        "QPMIL head $\\theta^\\ast$\nprompts · prototypes\ntext classifier $F_{txt}$",
        FROZEN, FROZEN_E, fs=9)
    box(ax, 13.4, 6.85, 2.0, 1.3, "bag logits\n$\\hat{y}$", NEUT, NEUT_E)
    arrow(ax, (12.7, 7.4), (13.4, 7.5))
    arrow(ax, (8.6, 7.5), (9.6, 7.5), ec="#9a9a9a", dashed=True)  # all-patch
    ax.text(9.1, 7.75, "All", fontsize=8, color="#9a9a9a", ha="center")

    # navigation policy (trainable, Continual Navigation Layer)
    ax.text(0.3, 5.7, "Navigation policy  (CNL; trained continually over the task stream)",
            fontsize=10, color=TRAIN_E, weight="bold")
    box(ax, 0.3, 3.7, 2.6, 1.4, "summary $s_i$\ntext / proto sim", NEUT, NEUT_E)
    box(ax, 3.3, 3.7, 3.0, 1.4, "navigation policy $\\phi$\nMLP $\\to$ score $r_i$",
        TRAIN, TRAIN_E)
    box(ax, 6.9, 3.7, 2.3, 1.4, "Top-$K$\nselect $\\mathcal{S}_K$", NEUT, NEUT_E)
    arrow(ax, (7.0, 6.85), (1.6, 5.1), rad=-0.12)     # Z -> summary
    arrow(ax, (7.4, 6.85), (4.8, 5.1), rad=-0.04)     # Z -> router
    arrow(ax, (2.9, 4.4), (3.3, 4.4))
    arrow(ax, (6.3, 4.4), (6.9, 4.4))
    arrow(ax, (9.2, 4.7), (10.4, 6.55), ec=TRAIN_E, rad=0.16)
    ax.text(9.35, 5.75, "$K$ patches", fontsize=8.6, color=TRAIN_E, ha="left")

    # Navigation Skill Memory (NSM)
    box(ax, 3.3, 1.55, 3.0, 1.35,
        "Navigation Skill Memory\nper-task / EWC (replay-free)", PLANB, PLANB_E,
        dashed=True, fs=8.8)
    arrow(ax, (4.8, 2.9), (4.8, 3.7), ec=PLANB_E, dashed=True)

    legend(ax, 10.6, 4.55)
    banner(ax, 0.3, 0.2, 15.1, 1.0)
    _save(fig, path)


# ---------------------------------------------------------------------------
def fig_b(path):
    """Variant: emphasize that router AND training-free selectors feed the SAME
    frozen backbone, so any accuracy gap is attributable to selection only."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # input row (top-left)
    box(ax, 0.3, 6.95, 2.2, 1.25, "WSI\npatches $x_i$", NEUT, NEUT_E)
    box(ax, 2.9, 6.95, 2.3, 1.25, "CONCH\n(frozen)", FROZEN, FROZEN_E)
    box(ax, 5.6, 6.95, 2.2, 1.25, "patch feats\n$Z=\\{z_i\\}$", NEUT, NEUT_E)
    arrow(ax, (2.5, 7.58), (2.9, 7.58))
    arrow(ax, (5.2, 7.58), (5.6, 7.58))

    # two selectors feeding a shared Top-K (left-middle)
    ax.text(0.3, 5.75, "Patch selectors  (all feed the SAME frozen backbone)",
            fontsize=10, color="#333333", weight="bold")
    box(ax, 0.3, 4.0, 3.5, 1.4,
        "navigation policy $\\phi$  (ours)\n$[z_i; s_i]\\to$ MLP $\\to r_i$",
        TRAIN, TRAIN_E, fs=8.8)
    box(ax, 0.3, 2.0, 3.5, 1.4,
        "training-free baselines\nrandom · prototype · semantic",
        GREY, GREY_E, fs=8.4)
    box(ax, 4.5, 2.95, 2.1, 1.5, "Top-$K$\nselect $\\mathcal{S}_K$",
        NEUT, NEUT_E)
    arrow(ax, (6.4, 6.95), (2.0, 5.4), rad=-0.14)         # Z -> router
    arrow(ax, (6.5, 6.95), (2.0, 3.4), rad=-0.24)         # Z -> baselines
    arrow(ax, (3.8, 4.5), (4.5, 4.05), ec=TRAIN_E)        # router -> topk
    arrow(ax, (3.8, 2.9), (4.5, 3.35), ec=GREY_E)         # baselines -> topk

    # shared frozen backbone (middle-right) -> ACC -> GO/NO-GO
    box(ax, 7.6, 4.3, 3.0, 1.6,
        "QPMIL head $\\theta^\\ast$\n(frozen)\ntext classifier $F_{txt}$",
        FROZEN, FROZEN_E, fs=9)
    arrow(ax, (6.6, 3.9), (7.6, 4.7), ec="#333333", rad=0.10)
    ax.text(7.0, 4.95, "selected\n$K$ patches", fontsize=8.4, ha="center")
    box(ax, 11.5, 4.65, 1.9, 1.25, "ACC @ $K$", NEUT, NEUT_E)
    arrow(ax, (10.6, 5.1), (11.5, 5.27))
    box(ax, 8.7, 2.0, 4.7, 1.5,
        "GO / NO-GO:\nrouter@$K$  vs  best heuristic@$K$",
        "#eef6ee", "#5a9c5a", fs=9)
    arrow(ax, (12.45, 4.65), (11.7, 3.5), ec="#5a9c5a", rad=0.1)

    # compact legend (top-right, empty area)
    items = [(TRAIN, TRAIN_E, "ours (trainable)", False),
             (GREY, GREY_E, "baseline", False),
             (FROZEN, FROZEN_E, "frozen", False)]
    for i, (fc, ec, lab, dash) in enumerate(items):
        yy = 7.9 - i * 0.6
        box(ax, 11.6, yy, 0.45, 0.38, "", fc, ec, dashed=dash)
        ax.text(12.15, yy + 0.19, lab, fontsize=8.6, va="center")

    banner(ax, 0.3, 0.2, 15.1, 1.0)
    _save(fig, path)


def _save(fig, path):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}.{{pdf,png}}")


def main():
    figdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "outputs", "figs")
    os.makedirs(figdir, exist_ok=True)
    fig_a(os.path.join(figdir, "Fig1_arch"))
    fig_b(os.path.join(figdir, "FigS1_arch"))


if __name__ == "__main__":
    main()
