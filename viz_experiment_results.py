#!/usr/bin/env python3
"""Generate story-focused experiment figures for NaviPath-CL paper.

Figures produced:
  Fig_main_comparison.{pdf,png}   — NSM vs naive vs zero-shot per task @K=64
  Fig_budget_curves.{pdf,png}     — Router vs baselines, budget vs accuracy per task
  Fig_lambda_analysis.{pdf,png}   — λ sweep: seq vs one-shot at K=64
  Fig_method_summary.{pdf,png}    — Consolidated bar: mACC across all 4 tasks + methods
  Fig_seqobs_budgets.{pdf,png}    — NSM sequential observation at multiple budgets

Usage:
    python viz_experiment_results.py
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FIGDIR = "outputs/figs"
os.makedirs(FIGDIR, exist_ok=True)

TASKS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
TASK_LABELS = ["ESCA", "RCC", "BRCA", "Lung"]
BUDGETS_MAIN = [16, 32, 64, 128]

# Colour palette (colour-blind friendly)
C_NSM      = "#2166ac"   # blue — our method
C_NAIVE    = "#d73027"   # red — naive continual
C_ZERO     = "#4dac26"   # green — zero-shot
C_RANDOM   = "#7f7f7f"   # grey — random
C_PROTO    = "#ff7f0e"   # orange — prototype
C_SEMANTIC = "#9467bd"   # purple — semantic

def _savefig(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIGDIR, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {FIGDIR}/{name}.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_seqobs(order="reverse", folds=(1,)):
    """Returns {task: {method: {budget: [vals across folds]}}}"""
    data = {}
    for fold in folds:
        for ti, task in enumerate(TASKS):
            # router-based (NSM + naive)
            f_r = f"outputs/seqobs_{order}_f{fold}_task{ti}.json"
            # zero-shot
            f_z = f"outputs/seqobs_{order}_f{fold}_task{ti}_policy-zeroshot.json"
            for fpath in (f_r, f_z):
                if not os.path.exists(fpath):
                    continue
                d = json.load(open(fpath))
                res = d.get("results", {})
                if task not in data:
                    data[task] = {}
                for method, bvals in res.items():
                    if method not in data[task]:
                        data[task][method] = {}
                    if isinstance(bvals, dict):
                        for bstr, v in bvals.items():
                            b = bstr
                            if b not in data[task][method]:
                                data[task][method][b] = []
                            if isinstance(v, (float, int)) and not np.isnan(float(v)):
                                data[task][method][b].append(float(v))
    return data


def load_router_budget(order="reverse", folds=(1,2,3)):
    """Returns {task: {method: {budget_int: [vals]}}}"""
    data = {}
    for fold in folds:
        f = f"outputs/router_v0_{order}_fold{fold}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        task = d.get("eval_task", d.get("last_task", ""))
        res = d.get("results", {})
        if task not in data:
            data[task] = {}
        for method, bvals in res.items():
            if method not in data[task]:
                data[task][method] = {}
            if isinstance(bvals, dict):
                for bstr, v in bvals.items():
                    if b := _to_int(bstr):
                        data[task][method].setdefault(b, []).append(float(v))
    return data


def _to_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def mean_err(vals):
    if not vals:
        return np.nan, 0.0
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def load_lambda_sweep(order="reverse", fold=1):
    """Returns list of {lambda, task, nsm_seq@64, nsm_oneshot@64}"""
    rows = []
    for ldir in sorted(glob.glob(f"outputs/routeA_sweep/lambda_*")):
        if not os.path.isdir(ldir):
            continue
        try:
            lam = float(ldir.split("lambda_")[-1])
        except ValueError:
            continue
        for ti, task in enumerate(TASKS):
            fp = os.path.join(ldir, f"seqobs_{order}_f{fold}_task{ti}.json")
            if not os.path.exists(fp):
                continue
            d = json.load(open(fp))
            res = d.get("results", {})
            seq64 = res.get("nsm_seq", {}).get("64", np.nan)
            one64 = res.get("nsm_oneshot", {}).get("64", np.nan)
            rows.append({"lambda": lam, "task": task, "seq64": seq64, "one64": one64})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Main comparison — NSM vs Naive vs Zero-shot @ K=64
# ─────────────────────────────────────────────────────────────────────────────

def fig_main_comparison():
    """NSM vs naive vs zero-shot per task @K=64.
    NOTE: These results are from pre-RouteA seqobs (no normalize_base).
    seq == oneshot here; we correctly label as 'top-K @64' not 'sequential'.
    The KEY comparison is NSM vs naive vs zero-shot, not seq vs oneshot.
    """
    seqobs = load_seqobs(order="reverse", folds=(1, 2, 3))
    seqobs1 = load_seqobs(order="reverse", folds=(1,))

    # Use nsm_seq == nsm_oneshot (pre-RouteA); label as top-K @64
    methods = [
        ("nsm_seq",       "NSM — ours (@K=64)",  C_NSM,   "o"),
        ("nonsm_seq",     "Naive continual",      C_NAIVE, "s"),
        ("zeroshot_seq",  "Zero-shot (no train)", C_ZERO,  "^"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(TASKS))
    width = 0.22
    offsets = [-1, 0, 1]

    for i, (mkey, mlabel, col, _mrk) in enumerate(methods):
        means, errs = [], []
        for task in TASKS:
            vals = seqobs.get(task, {}).get(mkey, {}).get("64", [])
            if not vals:
                vals = seqobs1.get(task, {}).get(mkey, {}).get("64", [])
            m, e = mean_err(vals)
            means.append(m)
            errs.append(e)
        bars = ax.bar(x + offsets[i] * width, means, width,
                      label=mlabel, color=col, alpha=0.85,
                      yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})
        for rect, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # mACC annotation
    means_nsm = [seqobs1.get(t, {}).get("nsm_seq", {}).get("64", [None])[0] or 0 for t in TASKS]
    means_nai = [seqobs1.get(t, {}).get("nonsm_seq", {}).get("64", [None])[0] or 0 for t in TASKS]
    means_zer = [seqobs1.get(t, {}).get("zeroshot_seq", {}).get("64", [None])[0] or 0 for t in TASKS]
    macc_nsm = np.mean([v for v in means_nsm if v > 0])
    macc_nai = np.mean([v for v in means_nai if v > 0])
    macc_zer = np.mean([v for v in means_zer if v > 0])
    ax.text(0.01, 0.97, f"mACC:  NSM={macc_nsm:.3f}  Naive={macc_nai:.3f}  Zero-shot={macc_zer:.3f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=12)
    ax.set_ylabel("Accuracy  (top-K patch selection, K = 64)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Navigation Accuracy @K=64 patches across 4 cancer tasks\n"
                 "(reverse task order, fold 1;  NSM = stored per-task navigation skill)",
                 fontsize=11)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "Fig_main_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Budget efficiency curves — router vs heuristics on LAST task
# ─────────────────────────────────────────────────────────────────────────────

def fig_budget_efficiency():
    """Router vs Random vs Prototype vs Semantic across all available folds."""
    import tools.collect_results as cr

    records = cr.load_all("outputs")
    agg, folds, _ = cr._aggregate_budget(records, "router_v0")
    if not agg:
        print("[skip] no router_v0 data for budget efficiency")
        return

    # pick reverse order entries (we have 3 folds)
    keys_rev = [k for k in agg if k[0] == "reverse"]
    if not keys_rev:
        keys_rev = list(agg.keys())[:4]

    ARM_STYLE = {
        "router":    dict(color=C_NSM,      marker="o", lw=2.2, ls="-",  zorder=5),
        "random":    dict(color=C_RANDOM,   marker="s", lw=1.5, ls="--"),
        "prototype": dict(color=C_PROTO,    marker="^", lw=1.5, ls="--"),
        "semantic":  dict(color=C_SEMANTIC, marker="D", lw=1.5, ls="--"),
    }

    ncol = min(len(keys_rev), 2)
    nrow = (len(keys_rev) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.2 * nrow), squeeze=False)

    for idx, key in enumerate(sorted(keys_rev)):
        ax = axes[idx // ncol][idx % ncol]
        order, eval_task = key
        results = agg[key]
        bks = sorted(int(b) for b in cr.budget_keys(results))
        n = len(folds[key])
        for arm in ("router", "random", "prototype", "semantic"):
            ys, es = [], []
            for b in bks:
                m, s = cr.mean_std(results.get(arm, {}).get(str(b), []))
                ys.append(np.nan if m is None else m)
                es.append(0.0 if s is None else s)
            st = ARM_STYLE[arm]
            ax.errorbar(bks, ys, yerr=es, label=arm, capsize=3,
                        marker=st["marker"], color=st["color"],
                        lw=st["lw"], ls=st["ls"],
                        zorder=st.get("zorder", 3))
        # All-patch reference
        m_all, _ = cr.mean_std(results.get("router", {}).get("All", []))
        if m_all is not None:
            ax.axhline(m_all, color="black", lw=0.9, ls=":", alpha=0.6)
            ax.text(bks[-1], m_all + 0.01, "All-patch baseline",
                    ha="right", fontsize=7.5, color="black", alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(bks)
        ax.set_xticklabels([str(b) for b in bks])
        ax.set_xlabel("Patch budget K")
        ax.set_ylabel("Accuracy")
        task_label = eval_task.replace("tcga_", "").upper()
        ax.set_title(f"{task_label} — {order} order (n={n} folds)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    for j in range(len(keys_rev), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Budget efficiency: patch selection strategy vs accuracy\n"
                 "(evaluated on the most recently learned task)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, "Fig_budget_efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: λ sweep — sequential vs one-shot
# ─────────────────────────────────────────────────────────────────────────────

def fig_lambda_sweep():
    """SBO Route A λ sweep — with normalize_base=True (Route A), seq truly ≠ oneshot.
    At λ=0,1: seq==oneshot (tumor patches are spatially clustered → diversity penalty irrelevant).
    At λ≥2: forced off-cluster → accuracy degrades.
    Key insight: SBO mechanism confirmed; optimal λ∈[0,1] preserves accuracy.
    """
    rows = load_lambda_sweep(order="reverse", fold=1)
    if not rows:
        print("[skip] no routeA_sweep data")
        return

    lambdas = sorted({r["lambda"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: per-task accuracy (seq vs oneshot)
    ax = axes[0]
    task_colors = [C_NSM, "#e31a1c", "#ff7f00", "#6a3d9a"]
    for task, label, col in zip(TASKS, TASK_LABELS, task_colors):
        ys_seq, ys_one = [], []
        for lam in lambdas:
            match = [r for r in rows if r["lambda"] == lam and r["task"] == task]
            ys_seq.append(match[0]["seq64"] if match else np.nan)
            ys_one.append(match[0]["one64"] if match else np.nan)
        ax.plot(lambdas, ys_seq, "-o", color=col, lw=2.2, ms=7, label=f"{label} (SBO seq)")
        ax.plot(lambdas, ys_one, "--", color=col, lw=1.2, alpha=0.4, ms=4,
                marker="s", label=f"{label} (one-shot)" if label == "ESCA" else "")

    # annotations
    ax.annotate("λ=0,1: seq≈oneshot\n(tumor clustered)", xy=(1, 0.93),
                xytext=(1.5, 0.70), fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))
    ax.annotate("λ≥2: forced off-cluster\n→ accuracy drops", xy=(2, 0.82),
                xytext=(2.3, 0.5), fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.axvspan(-0.1, 1.2, alpha=0.07, color="green")
    ax.text(0.5, 0.17, "Optimal λ ∈ [0, 1]", color="green", fontsize=9,
            ha="center", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Redundancy weight λ  (Route A / SBO)", fontsize=11)
    ax.set_ylabel("Accuracy @ K=64", fontsize=11)
    ax.set_title("Per-task: SBO sequential vs one-shot accuracy\n"
                 "(solid=SBO sequential, dashed=one-shot; normalize_base=True)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.1, 1.08)
    ax.set_xlim(-0.2, 4.3)

    # Panel B: mean mACC + Δ(seq - oneshot)
    ax2 = axes[1]
    mean_seq, mean_one, mean_diff = [], [], []
    for lam in lambdas:
        sq = [r["seq64"] for r in rows if r["lambda"] == lam and not np.isnan(r["seq64"])]
        on = [r["one64"] for r in rows if r["lambda"] == lam and not np.isnan(r["one64"])]
        ms = np.mean(sq) if sq else np.nan
        mo = np.mean(on) if on else np.nan
        mean_seq.append(ms); mean_one.append(mo)
        mean_diff.append(ms - mo if not np.isnan(ms) and not np.isnan(mo) else np.nan)

    ax2.plot(lambdas, mean_seq, "-o", color=C_NSM, lw=2.5, ms=9,
             label="SBO sequential (NSM, Route A)")
    ax2.plot(lambdas, mean_one, "--s", color="#888888", lw=2, ms=7,
             label="One-shot top-K (NSM, no diversity)")
    ax2_r = ax2.twinx()
    ax2_r.bar(lambdas, mean_diff, width=0.25, color="red", alpha=0.35,
              label="Δ seq−oneshot")
    ax2_r.axhline(0, color="red", lw=0.8, ls=":")
    ax2_r.set_ylabel("Δ (seq − one-shot)", fontsize=10, color="red")
    ax2_r.tick_params(axis="y", colors="red")
    ax2_r.set_ylim(-0.45, 0.05)

    ax2.set_xlabel("Redundancy weight λ", fontsize=11)
    ax2.set_ylabel("Mean accuracy across 4 tasks", fontsize=11)
    ax2.set_title("Mean mACC vs λ  (left axis)\nΔ seq−oneshot (right axis, bars)",
                  fontsize=10)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.3, 1.05)

    fig.suptitle("SBO (Route A) λ sweep — normalize_base=True, mode=maxsim\n"
                 "Tumor patches are spatially clustered → large λ forces off-target exploration",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, "Fig_lambda_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Summary bar — mACC across methods
# ─────────────────────────────────────────────────────────────────────────────

def fig_method_summary():
    """Consolidated: all methods at @64 + All-patch reference."""
    seqobs = load_seqobs(order="reverse", folds=(1,))

    tasks = TASKS
    methods_cfg = [
        ("nsm_seq",      "NSM (ours, @64)",       C_NSM,      "///"),
        ("nonsm_seq",    "Naive (@64)",            C_NAIVE,    "xxx"),
        ("zeroshot_seq", "Zero-shot (@64)",        C_ZERO,     "..."),
    ]

    fig, axes = plt.subplots(1, len(tasks), figsize=(13, 4.8), sharey=True)
    for ti, (task, label) in enumerate(zip(tasks, TASK_LABELS)):
        ax = axes[ti]
        x_pos, labels_local, colors_local = [], [], []
        xidx = 0
        for mkey, mlabel, col, hatch in methods_cfg:
            vals = seqobs.get(task, {}).get(mkey, {}).get("64", [])
            m, e = mean_err(vals)
            bar = ax.bar(xidx, m, 0.7, color=col, alpha=0.82, hatch=hatch,
                         yerr=e, capsize=4, error_kw={"elinewidth": 1.2},
                         label=mlabel if ti == 0 else "")
            if not np.isnan(m):
                ax.text(xidx, m + e + 0.02, f"{m:.3f}", ha="center", fontsize=7.5,
                        fontweight="bold", color=col)
            x_pos.append(xidx)
            labels_local.append(mlabel.split(" ")[0])
            colors_local.append(col)
            xidx += 1

        # All-patch reference (from seqobs All)
        vals_all = seqobs.get(task, {}).get("nsm_seq", {}).get("All", [])
        if vals_all:
            m_all = np.mean(vals_all)
            ax.axhline(m_all, color="black", lw=1.2, ls="--", alpha=0.6)
            ax.text(xidx - 0.5, m_all + 0.015, f"All: {m_all:.3f}",
                    ha="center", fontsize=7, color="black", alpha=0.7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["NSM", "Naive", "Zero-shot"], fontsize=9)
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.25)
        if ti == 0:
            ax.set_ylabel("Accuracy", fontsize=11)

    axes[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("NaviPath-CL — Performance Comparison @K=64\n"
                 "(reverse task order, fold 1; dashed = All-patch reference)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, "Fig_method_summary")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Sequential observation budget curves (NSM vs naive vs zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

def fig_seqobs_budgets():
    """Show NSM / naive / zero-shot accuracy curves across budgets for all 4 tasks."""
    seqobs = load_seqobs(order="reverse", folds=(1,))

    budgets = [16, 32, 64, 128]  # from seqobs JSON keys

    methods_cfg = [
        ("nsm_seq",       "NSM (ours)",     C_NSM,      "o",  2.2),
        ("nonsm_seq",     "Naive",          C_NAIVE,    "s",  1.6),
        ("zeroshot_seq",  "Zero-shot",      C_ZERO,     "^",  1.6),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ti, (task, label) in enumerate(zip(TASKS, TASK_LABELS)):
        ax = axes[ti]
        for mkey, mlabel, col, mrk, lw in methods_cfg:
            ys = []
            for b in budgets:
                vals = seqobs.get(task, {}).get(mkey, {}).get(str(b), [])
                m, _ = mean_err(vals)
                ys.append(m)
            ax.plot(budgets, ys, f"-{mrk}", color=col, lw=lw,
                    label=mlabel if ti == 0 else "", ms=6)
        # All-patch reference
        vals_all = seqobs.get(task, {}).get("nsm_seq", {}).get("All", [])
        if vals_all:
            ax.axhline(np.mean(vals_all), color="black", lw=0.9, ls=":", alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Patch budget K")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 1.05)
        if ti == 0:
            ax.set_ylabel("Accuracy")

    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("NaviPath-CL — Accuracy vs Patch Budget\n"
                 "(NSM continual vs Naive vs Zero-shot, reverse order fold 1)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, "Fig_seqobs_budgets")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Continual R-matrix summary (NaviPath vs QPMIL across 3 folds)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cl_performance_summary():
    """mACC + forgetting comparison: NaviPath NSM vs QPMIL (naive CL backbone)."""

    def load_summary_stats(pattern):
        accs, forgets = [], []
        for f in sorted(glob.glob(pattern)):
            d = json.load(open(f))
            s = d.get("summary", {})
            accs.append(s.get("ACC", np.nan))
            forgets.append(abs(s.get("Forgetting", 0.0)))
        return accs, forgets

    configs = [
        ("NaviPath-NSM\n(paper order)",  "outputs/navipath_full_paper_fold*.json",   C_NSM),
        ("NaviPath-NSM\n(reverse order)","outputs/navipath_full_reverse_fold*.json",  C_NSM),
        ("QPMIL baseline\n(paper order)","outputs/qpmil_paper_fold*.json",            C_NAIVE),
        ("QPMIL baseline\n(reverse)",    "outputs/qpmil_reverse_fold*.json",          C_NAIVE),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(configs))
    width = 0.55

    for panel_ax, metric_idx, ylabel, title in [
        (ax1, 0, "Final-state mACC", "Mean Accuracy (final after all tasks)"),
        (ax2, 1, "Forgetting", "Forgetting (|BWT|, lower is better)"),
    ]:
        for i, (label, pattern, col) in enumerate(configs):
            accs, forgets = load_summary_stats(pattern)
            vals = accs if metric_idx == 0 else forgets
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                continue
            m, e = np.mean(vals), np.std(vals)
            bar = panel_ax.bar(i, m, width, color=col, alpha=0.78,
                               yerr=e, capsize=5, error_kw={"elinewidth": 1.5},
                               hatch="///" if "NSM" in label else "xxx")
            panel_ax.text(i, m + e + 0.005, f"{m:.3f}±{e:.3f}",
                          ha="center", fontsize=8, fontweight="bold")

        panel_ax.set_xticks(x)
        panel_ax.set_xticklabels([c[0] for c in configs], fontsize=8.5)
        panel_ax.set_ylabel(ylabel, fontsize=10)
        panel_ax.set_title(title, fontsize=10)
        panel_ax.grid(axis="y", alpha=0.3)
        if metric_idx == 0:
            panel_ax.set_ylim(0.7, 1.05)
        else:
            panel_ax.set_ylim(0, 0.12)

    # Add NSM=0 forgetting annotation
    ax2.axhline(0, color=C_NSM, lw=1.5, ls="--", alpha=0.7, label="NSM target: 0 forgetting")
    ax2.legend(fontsize=8)

    fig.suptitle("NaviPath-NSM vs QPMIL baseline: accuracy and forgetting\n"
                 "(3-fold cross-validation, both task orders)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _savefig(fig, "Fig_cl_performance_summary")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Generating NaviPath-CL experiment figures...")
    fig_main_comparison()
    fig_budget_efficiency()
    fig_lambda_sweep()
    fig_method_summary()
    fig_seqobs_budgets()
    fig_cl_performance_summary()
    print(f"\n[done] all figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
