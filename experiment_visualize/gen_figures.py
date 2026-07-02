#!/usr/bin/env python3
"""Generate presentation-quality figures for experiment_visualize/.

Produces 4 figures tailored for slides/paper:
  1. fig_budget_efficiency.png   — Router vs baselines budget curves
  2. fig_main_comparison.png     — NSM vs Naive vs Zero-shot bar chart
  3. fig_mechanism_tsne.png      — (copy from existing P2contrast)
  4. fig_lambda_sweep.png        — λ sweep with dual axis

Run: python experiment_visualize/gen_figures.py
"""
import json, glob, os, sys
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-navipath")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUTDIR, exist_ok=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

C_OURS   = "#2166ac"
C_NAIVE  = "#d73027"
C_ZERO   = "#4dac26"
C_RANDOM = "#7f7f7f"
C_PROTO  = "#ff7f0e"
C_SEM    = "#9467bd"

TASKS = ['tcga_esca', 'tcga_rcc', 'tcga_brca', 'tcga_lung']
LABELS = ['ESCA', 'RCC', 'BRCA', 'Lung']


def savefig(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"  [saved] {OUTDIR}/{name}.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Budget Efficiency
# ─────────────────────────────────────────────────────────────────────────────
def fig_budget_efficiency():
    budgets = [16, 32, 64, 128, 256]
    methods = [
        ('router',    'Learned Router (ours)', C_OURS,   'o', 2.5, '-'),
        ('random',    'Random sampling',       C_RANDOM, 's', 1.5, '--'),
        ('prototype', 'Prototype-based',       C_PROTO,  '^', 1.5, '--'),
        ('semantic',  'Semantic (text-cos)',    C_SEM,    'D', 1.5, '--'),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mkey, mlabel, col, mrk, lw, ls in methods:
        means, stds = [], []
        for b in budgets:
            vals = []
            for fold in [1, 2, 3]:
                fp = os.path.join(REPO, f'outputs/router_v0_reverse_fold{fold}.json')
                if not os.path.exists(fp): continue
                d = json.load(open(fp))
                v = d['results'].get(mkey, {}).get(str(b))
                if v is not None: vals.append(float(v))
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(budgets, means, yerr=stds, label=mlabel,
                    color=col, marker=mrk, lw=lw, ls=ls, capsize=4, ms=7)

    # All-patch reference
    all_vals = []
    for fold in [1, 2, 3]:
        fp = os.path.join(REPO, f'outputs/router_v0_reverse_fold{fold}.json')
        if os.path.exists(fp):
            all_vals.append(float(json.load(open(fp))['results']['router']['All']))
    ax.axhline(np.mean(all_vals), color='black', lw=1.2, ls=':', alpha=0.7)
    ax.text(260, np.mean(all_vals)+0.005, f"All patches: {np.mean(all_vals):.3f}",
            fontsize=9, color='black', alpha=0.7, ha='right')

    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('Patch budget K (log scale)', fontsize=11)
    ax.set_ylabel('Classification accuracy', fontsize=11)
    ax.set_title('Budget Efficiency: Patch Selection Strategy vs. Accuracy\n'
                 '(Lung cancer, 3-fold CV; learned router surpasses all-patch at K=64)',
                 fontsize=10.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.25)
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    savefig(fig, 'fig_budget_efficiency')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Main Comparison (NSM vs Naive vs Zero-shot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_comparison():
    methods = [
        ('nsm_seq',      'NSM (ours)',   C_OURS),
        ('nonsm_seq',    'Naive',        C_NAIVE),
        ('zeroshot_seq', 'Zero-shot',    C_ZERO),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TASKS))
    width = 0.23
    offsets = [-1, 0, 1]

    for i, (mkey, mlabel, col) in enumerate(methods):
        means, stds = [], []
        for ti in range(4):
            vals = []
            for fold in [1, 2, 3]:
                if 'zeroshot' in mkey:
                    fp = os.path.join(REPO, f'outputs/seqobs_reverse_f{fold}_task{ti}_policy-zeroshot.json')
                else:
                    fp = os.path.join(REPO, f'outputs/seqobs_reverse_f{fold}_task{ti}.json')
                if not os.path.exists(fp): continue
                d = json.load(open(fp))['results']
                v = d.get(mkey, {}).get('64')
                if v is not None: vals.append(float(v))
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x + offsets[i]*width, means, width, label=mlabel,
                      color=col, alpha=0.85, yerr=stds, capsize=4,
                      error_kw={'elinewidth': 1.3})
        for rect, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # mACC text box
    nsm_m = np.mean([float(json.load(open(os.path.join(REPO, f'outputs/seqobs_reverse_f{fold}_task{ti}.json')))['results']['nsm_seq']['64']) for fold in [1,2,3] for ti in range(4)])
    naive_m = np.mean([float(json.load(open(os.path.join(REPO, f'outputs/seqobs_reverse_f{fold}_task{ti}.json')))['results']['nonsm_seq']['64']) for fold in [1,2,3] for ti in range(4)])
    zero_m = np.mean([float(json.load(open(os.path.join(REPO, f'outputs/seqobs_reverse_f{fold}_task{ti}_policy-zeroshot.json')))['results']['zeroshot_seq']['64']) for fold in [1,2,3] for ti in range(4)])
    ax.text(0.02, 0.97,
            f'mACC across 4 tasks:\n  NSM (ours) = {nsm_m:.3f}\n  Naive = {naive_m:.3f}\n  Zero-shot = {zero_m:.3f}',
            transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(['ESCA\n(1st, oldest)', 'RCC\n(2nd)', 'BRCA\n(3rd)', 'Lung\n(4th, newest)'], fontsize=10)
    ax.set_ylabel('Accuracy @K=64 patches', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('Continual Navigation Accuracy: NSM vs. Naive vs. Zero-shot\n'
                 '(4 tasks sequential, reverse order, 3-fold CV, K=64)',
                 fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, 'fig_main_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Copy P2contrast
# ─────────────────────────────────────────────────────────────────────────────
def fig_mechanism_tsne():
    src = os.path.join(REPO, 'paper/figs/P2contrast_esca_fold1.png')
    dst = os.path.join(OUTDIR, 'fig_mechanism_tsne.png')
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        # also copy pdf
        src_pdf = src.replace('.png', '.pdf')
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, os.path.join(OUTDIR, 'fig_mechanism_tsne.pdf'))
        print(f"  [copied] {dst}")
    else:
        print(f"  [skip] {src} not found")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: λ Sweep
# ─────────────────────────────────────────────────────────────────────────────
def fig_lambda_sweep():
    lambdas = [0.0, 1.0, 2.0, 4.0]
    mean_seq, mean_one = [], []
    for lam in lambdas:
        sq, on = [], []
        for ti in range(4):
            fp = os.path.join(REPO, f'outputs/routeA_sweep/lambda_{lam}/seqobs_reverse_f1_task{ti}.json')
            if not os.path.exists(fp): continue
            res = json.load(open(fp))['results']
            sq.append(float(res['nsm_seq']['64']))
            on.append(float(res['nsm_oneshot']['64']))
        mean_seq.append(np.mean(sq) if sq else np.nan)
        mean_one.append(np.mean(on) if on else np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(lambdas, mean_seq, '-o', color=C_OURS, lw=2.5, ms=9,
            label='Sequential (SBO)', zorder=5)
    ax.plot(lambdas, mean_one, '--s', color=C_RANDOM, lw=2, ms=7,
            label='One-shot (top-K)', alpha=0.7)

    # Fill degradation region
    ax.fill_between(lambdas, mean_seq, mean_one,
                    where=[s < o for s, o in zip(mean_seq, mean_one)],
                    color='red', alpha=0.10, label='SBO degradation')

    # Optimal region
    ax.axvspan(-0.1, 1.2, alpha=0.06, color='green')
    ax.text(0.5, 0.50, 'Optimal\nλ ∈ [0, 1]', color='green', fontsize=10,
            ha='center', fontweight='bold', alpha=0.8)

    # Annotations
    ax.annotate('λ=4: forced off-cluster\n(catastrophic)', xy=(4, 0.469),
                xytext=(3.2, 0.60), fontsize=8.5, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('Diversity weight λ  (SBO redundancy penalty)', fontsize=11)
    ax.set_ylabel('Mean accuracy across 4 tasks (@K=64)', fontsize=11)
    ax.set_title('Sequential Budgeted Observer: λ vs. Accuracy\n'
                 '(Higher λ = more diversity forcing; pathology favors focused selection)',
                 fontsize=10.5)
    ax.legend(fontsize=9.5, loc='lower left')
    ax.grid(alpha=0.25)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlim(-0.3, 4.5)
    fig.tight_layout()
    savefig(fig, 'fig_lambda_sweep')


def main():
    print("Generating experiment figures...")
    fig_budget_efficiency()
    fig_main_comparison()
    fig_mechanism_tsne()
    fig_lambda_sweep()
    print(f"\nDone. All figures in: {OUTDIR}/")


if __name__ == '__main__':
    main()
