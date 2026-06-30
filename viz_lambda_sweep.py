"""Visualization: λ sweep accuracy curves (Fig_lambda_sweep).

Reads outputs/routeA_sweep/lambda_*/seqobs_*.json (already on Mac after git pull).
Produces paper/figs/Fig_lambda_sweep.pdf + .png.

Run: python viz_lambda_sweep.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
SWEEP_DIR = pathlib.Path("outputs/routeA_sweep")
OUT_DIR   = pathlib.Path("paper/figs")
LAMBDAS   = [0.0, 1.0, 2.0, 4.0]
TASKS     = ["task0", "task1", "task2", "task3"]
TASK_NAMES = {"task0": "esca", "task1": "rcc", "task2": "brca", "task3": "lung"}
BUDGETS   = [16, 32, 64, 128]
FOLD      = 1

COLORS = {0.0: "#888888", 1.0: "#2196F3", 2.0: "#FF9800", 4.0: "#E53935"}
TASK_COLOR = {"esca": "#9C27B0", "rcc": "#2196F3", "brca": "#4CAF50", "lung": "#FF5722"}


def load_results(lam: float, task: str, fold: int = 1) -> dict | None:
    p = SWEEP_DIR / f"lambda_{lam}" / f"seqobs_reverse_f{fold}_{task}.json"
    if not p.exists():
        print(f"  [warn] not found: {p}", file=sys.stderr)
        return None
    return json.loads(p.read_text())["results"]


def get_acc(results: dict, mode: str, budget: int) -> float:
    return results.get(mode, {}).get(str(budget), float("nan"))


# ── Figure 1: per-task λ vs. acc@64 (seq vs oneshot) ────────────────────────
def fig_lambda_vs_acc():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)
    for ax, task in zip(axes, TASKS):
        name = TASK_NAMES[task]
        seq_accs, one_accs = [], []
        for lam in LAMBDAS:
            res = load_results(lam, task, FOLD)
            if res is None:
                seq_accs.append(np.nan); one_accs.append(np.nan)
            else:
                seq_accs.append(get_acc(res, "nsm_seq", 64))
                one_accs.append(get_acc(res, "nsm_oneshot", 64))

        ax.plot(LAMBDAS, seq_accs,  "o-", color="#2196F3", lw=2, ms=7,
                label="seq (SBO)")
        ax.plot(LAMBDAS, one_accs,  "s--", color="#888888", lw=2, ms=7,
                label="one-shot")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("λ (diversity)", fontsize=11)
        ax.set_ylabel("Acc@64", fontsize=11)
        ax.set_xticks(LAMBDAS)
        ax.set_ylim(0.0, 1.05)
        ax.axvline(x=1.0, ls=":", color="#FF9800", alpha=0.7, label="optimal λ range")
        ax.axvline(x=0.5, ls=":", color="#FF9800", alpha=0.4)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("SBO λ sweep: sequential vs. one-shot accuracy at K=64\n"
                 "(fold 1, reverse order, NSM oracle gate)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


# ── Figure 2: budget curves for λ=0 vs λ=1 (seq only, all tasks) ────────────
def fig_budget_curves():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=False)
    for ax, task in zip(axes, TASKS):
        name = TASK_NAMES[task]
        for lam, ls, lbl in [(0.0, "--", "λ=0 (=oneshot)"),
                              (1.0, "-",  "λ=1 (SBO)")]:
            res = load_results(lam, task, FOLD)
            if res is None:
                continue
            accs = [get_acc(res, "nsm_seq", b) for b in BUDGETS]
            ax.plot(BUDGETS, accs, ls=ls, marker="o", ms=6,
                    color=COLORS[lam], lw=2, label=lbl)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Budget K", fontsize=11)
        ax.set_ylabel("Acc (nsm_seq)", fontsize=11)
        ax.set_xticks(BUDGETS)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Budget curves: SBO λ=0 vs λ=1 across tasks\n"
                 "(fold 1, reverse order)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Fig_lambda_sweep (λ vs acc@64) …")
    f1 = fig_lambda_vs_acc()
    f1.savefig(OUT_DIR / "Fig_lambda_sweep.pdf", bbox_inches="tight")
    f1.savefig(OUT_DIR / "Fig_lambda_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(f1)
    print("  → paper/figs/Fig_lambda_sweep.{pdf,png}")

    print("Generating Fig_lambda_budget_curves …")
    f2 = fig_budget_curves()
    f2.savefig(OUT_DIR / "Fig_lambda_budget.pdf", bbox_inches="tight")
    f2.savefig(OUT_DIR / "Fig_lambda_budget.png", dpi=150, bbox_inches="tight")
    plt.close(f2)
    print("  → paper/figs/Fig_lambda_budget.{pdf,png}")

    print("Done.")
