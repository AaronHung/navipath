"""SPEC-03 — Core figures (problem / mechanism / budget-curve / roadmap).

數據圖（mechanism / budget-curve）從 outputs/oldtask_budget_*.json 動態重算
（重用 mechanism_table.load_records），不重抄數字。概念圖（problem / roadmap）
用 matplotlib box/arrow。架構圖見 tools/draw_arch.py。

用法: python tools/make_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechanism_table import load_records, mech_row, best_heur_row, _budget_keys  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "outputs", "figs")

GO_C, NOGO_C, HEUR_C = "#5a9c5a", "#c0504d", "#9a9a9a"
NSM_C, SHARED_C, EWC_C = "#3776c0", "#c0504d", "#e08a37"


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}.{{pdf,png}}")


def _pick_reverse_esca(groups):
    """挑 reverse / tcga_esca 那組；找不到就回第一組。"""
    for key, by_mech in groups.items():
        if key[0] == "reverse" and "esca" in key[1]:
            return key, by_mech
    return (next(iter(groups)) if groups else (None, None)), \
        (next(iter(groups.values())) if groups else None)


def fig_mechanism(groups):
    key, by_mech = _pick_reverse_esca(groups)
    if not by_mech:
        return
    bkeys = _budget_keys(by_mech[next(iter(by_mech))][0][2])
    vals, labels, colors = [], [], []
    for mech, disp in (("shared", "shared\n(single policy)"),
                       ("ewc", "EWC\n(weight reg.)"),
                       ("pertask", "per-task NSM\n(skill memory)")):
        if mech in by_mech:
            row, _n, go = mech_row(by_mech[mech], bkeys)
            v = row.get("64")
            if v is not None:
                vals.append(v)
                labels.append(disp)
                colors.append(GO_C if go > 0 else NOGO_C)
    bh = best_heur_row(by_mech, bkeys).get("64")
    if bh is not None:
        vals.append(bh)
        labels.append("best\nheuristic")
        colors.append(HEUR_C)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    if bh is not None:
        ax.axhline(bh, ls="--", lw=1, color=HEUR_C, zorder=0)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("old-task ACC @ budget 64")
    ax.set_ylim(0, 1.02)
    ax.set_title("Which continual mechanism fits a WSI navigation policy?\n"
                 "(old ESCA, reverse order, mean over folds)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "Fig_mechanism")


def fig_budget_curve(groups):
    key, by_mech = _pick_reverse_esca(groups)
    if not by_mech:
        return
    bkeys = _budget_keys(by_mech[next(iter(by_mech))][0][2])
    # x 軸用數值 budget（All 放最右）
    xs = [k for k in bkeys if k != "All"]
    xs_num = sorted(int(x) for x in xs)
    xlabels = [str(x) for x in xs_num] + (["All"] if "All" in bkeys else [])
    xpos = list(range(len(xlabels)))

    def series(row):
        ys = [row.get(str(x)) for x in xs_num]
        if "All" in bkeys:
            ys.append(row.get("All"))
        return ys

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    style = {"shared": (SHARED_C, "single policy (shared)"),
             "ewc": (EWC_C, "EWC (weight reg.)"),
             "pertask": (NSM_C, "per-task NSM (skill memory)")}
    for mech, (c, lab) in style.items():
        if mech in by_mech:
            row, _n, _go = mech_row(by_mech[mech], bkeys)
            ax.plot(xpos, series(row), marker="o", color=c, label=lab, lw=1.8)
    bh = best_heur_row(by_mech, bkeys)
    ax.plot(xpos, series(bh), marker="s", color=HEUR_C, ls="--",
            label="best heuristic", lw=1.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("observation budget (#patches)")
    ax.set_ylabel("old-task ACC")
    ax.set_ylim(0, 1.02)
    ax.set_title("Navigation forgetting vs budget (old ESCA, reverse order)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "Fig_budget_curve")


def _box(ax, x, y, w, h, text, fc, ec, fs=9.5, weight="normal", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.6, edgecolor=ec, facecolor=fc,
                 linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=5)


def _arrow(ax, p1, p2, ec="#444", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                 lw=1.5, color=ec, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=2, shrinkB=2, zorder=3))


def fig_problem():
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _box(ax, 0.3, 4.6, 3.0, 1.6, "WSI (gigapixel)\nthousands of patches", "#ffffff", "#444")
    _box(ax, 4.0, 4.6, 3.2, 1.6, "budgeted observation\n(look at only K patches)",
         "#fde0c7", "#e08a37")
    _box(ax, 8.0, 4.6, 3.2, 1.6, "navigation policy\n(where to look)", "#cfe3f7", "#3776c0")
    _box(ax, 12.0, 4.6, 3.4, 1.6, "diagnosis\n(frozen backbone)", "#eef6ee", "#5a9c5a")
    _arrow(ax, (3.3, 5.4), (4.0, 5.4))
    _arrow(ax, (7.2, 5.4), (8.0, 5.4))
    _arrow(ax, (11.2, 5.4), (12.0, 5.4))
    # task stream + forgetting
    _box(ax, 0.3, 1.4, 11.0, 1.6,
         "continual task stream  $t=1\\dots T$  →  policy trained sequentially",
         "#f1f1f1", "#9a9a9a", fs=9.5)
    _box(ax, 12.0, 1.4, 3.4, 1.6,
         "navigation forgetting\non old tasks", "#fdecec", "#c0504d", fs=9.5, weight="bold")
    _arrow(ax, (8.0, 4.6), (6.0, 3.0), ec="#9a9a9a", rad=-0.1)
    _arrow(ax, (11.3, 2.2), (12.0, 2.2), ec="#c0504d")
    ax.set_title("Problem: continual learning of the observation policy under budgeted WSI inference",
                 fontsize=10.5, weight="bold")
    _save(fig, "Fig_problem")


def fig_roadmap():
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    phases = [
        ("Phase-0 (this work)\nbudgeted patch selection\n+ Navigation Skill Memory", "#cfe3f7", "#3776c0"),
        ("Phase-1\ntask-free context gate\n+ skill consolidation", "#fff7d6", "#c9a227"),
        ("Phase-2\nmulti-scale move/zoom\n+ physician trajectories", "#fde0c7", "#e08a37"),
        ("North Star\nphysician-like WSI\nnavigation agent (RLHF)", "#eef6ee", "#5a9c5a"),
    ]
    w, gap = 3.4, 0.45
    x = 0.3
    centers = []
    for txt, fc, ec in phases:
        _box(ax, x, 2.0, w, 2.0, txt, fc, ec, fs=8.8)
        centers.append(x + w)
        x += w + gap
    for cx in centers[:-1]:
        _arrow(ax, (cx, 3.0), (cx + gap, 3.0))
    ax.set_title("Roadmap: Phase-0 prototype → North Star navigation agent", fontsize=10.5,
                 weight="bold")
    _save(fig, "Fig_roadmap")


def main():
    groups = load_records("outputs")
    fig_mechanism(groups)
    fig_budget_curve(groups)
    fig_problem()
    fig_roadmap()


if __name__ == "__main__":
    main()
