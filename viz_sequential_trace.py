"""Visualization: Sequential Budgeted Observer multi-round trace (Fig_seq_trace).

Shows WHERE the SBO selects patches across rounds, color-coded by round.
Two modes:
  --mode synthetic   (Mac OK)  : generates synthetic 4-cluster slide, runs SBO,
                                 plots t-SNE colored by selection round.
  --mode real        (RunPod)  : loads a real slide's CONCH features + skill bank,
                                 runs SBO, plots t-SNE.

Usage:
  # Mac (synthetic demo):
  python viz_sequential_trace.py --mode synthetic

  # RunPod (real data):
  python viz_sequential_trace.py --mode real \
      --skill-bank outputs/skill_bank_reverse_f1.pt \
      --slide-idx 0 --task-id 0 \
      --qpmil-config QPMIL-VL/configs/main.yaml
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = pathlib.Path("paper/figs")
ROUND_COLORS = ["#E53935", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ── helpers ──────────────────────────────────────────────────────────────────

def run_tsne(Z: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000).fit_transform(Z)


def plot_trace(xy: np.ndarray, trace: list[list[int]], title: str,
               ax: plt.Axes, n_total: int) -> None:
    """Plot t-SNE with patches colored by selection round."""
    # All patches: grey background
    ax.scatter(xy[:, 0], xy[:, 1], c="#CCCCCC", s=10, alpha=0.4, zorder=1)

    # Selected patches, colored by round
    legend_handles = []
    all_selected = []
    for round_idx, picks in enumerate(trace):
        if not picks:
            continue
        color = ROUND_COLORS[round_idx % len(ROUND_COLORS)]
        x_picks = xy[picks, 0]
        y_picks = xy[picks, 1]
        ax.scatter(x_picks, y_picks, c=color, s=50, zorder=3,
                   edgecolors="white", linewidths=0.5, alpha=0.9)
        all_selected.extend(picks)
        legend_handles.append(
            mpatches.Patch(color=color, label=f"Round {round_idx + 1} ({len(picks)} patches)")
        )

    # Mark unselected with smaller dots
    unselected = [i for i in range(n_total) if i not in set(all_selected)]
    if unselected:
        ax.scatter(xy[unselected, 0], xy[unselected, 1],
                   c="#AAAAAA", s=8, alpha=0.25, zorder=1)

    ax.legend(handles=legend_handles, fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")


# ── Mode 1: SYNTHETIC ────────────────────────────────────────────────────────

def synthetic_demo(lambdas: list[float] = (0.0, 0.5, 1.0, 2.0),
                   n_patches: int = 200,
                   budget: int = 64,
                   step_size: int = 16,
                   D: int = 64,
                   seed: int = 42) -> None:
    """Synthetic 4-cluster slide. Router scores = distance to cluster 0 (simulated).
    High-score cluster = cluster 0 (diagnostic). Shows how λ drives exploration."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from navipath_moe.sequential_observation import ObserveConfig, SequentialBudgetedObserver

    rng = np.random.default_rng(seed)
    n_clusters = 4
    n_per = n_patches // n_clusters
    centers = rng.normal(0, 4.0, (n_clusters, D))

    Z_np = np.vstack([rng.normal(centers[k], 0.6, (n_per, D))
                      for k in range(n_clusters)])
    Z_np = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-8)
    Z = torch.tensor(Z_np, dtype=torch.float32)

    # Simulated base scores: patches in cluster 0 score higher (diagnostic cluster)
    cluster_id = np.repeat(np.arange(n_clusters), n_per)
    base_score = torch.zeros(len(Z_np))
    base_score[:n_per] = torch.tensor(  # cluster 0: high score
        rng.normal(2.0, 0.3, n_per).astype(np.float32))
    for k in range(1, n_clusters):     # other clusters: low score
        base_score[k*n_per:(k+1)*n_per] = torch.tensor(
            rng.normal(-0.5, 0.3, n_per).astype(np.float32))

    dummy_predict = lambda S: torch.zeros(2)  # noqa: E731

    print("Running t-SNE on synthetic features …")
    xy = run_tsne(Z_np)

    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4.5))
    if len(lambdas) == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambdas):
        cfg = ObserveConfig(budget=budget, step_size=step_size,
                            redundancy_weight=lam, normalize_base=True,
                            redundancy_mode="maxsim")
        obs = SequentialBudgetedObserver(cfg)
        result = obs.observe(Z, base_score, dummy_predict)
        plot_trace(xy, result.trace,
                   title=f"λ = {lam}\n({result.n_rounds} rounds, {len(result.selected)} patches)",
                   ax=ax, n_total=len(Z_np))
        # Mark cluster boundaries (visual annotation)
        for k in range(n_clusters):
            idx = np.where(cluster_id == k)[0]
            cx, cy = xy[idx, 0].mean(), xy[idx, 1].mean()
            ax.annotate(f"C{k}", (cx, cy), fontsize=9, color="#333333",
                        ha="center", va="center", fontweight="bold")

    fig.suptitle("Sequential Budgeted Observer: patch selection trace\n"
                 "(synthetic 4-cluster slide; C0 = high-score diagnostic cluster)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "Fig_seq_trace_synthetic.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig_seq_trace_synthetic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → paper/figs/Fig_seq_trace_synthetic.{pdf,png}")


# ── Mode 2: REAL (RunPod) ─────────────────────────────────────────────────────

def real_trace(skill_bank_path: str, slide_idx: int, task_id: int,
               qpmil_config: str, budget: int = 64, step_size: int = 16,
               lambda_vals: list[float] = (0.0, 0.5, 1.0),
               fold: int = 1, order: str = "reverse") -> None:
    """Load real CONCH features for one slide and run SBO trace visualization."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    print(f"Loading skill bank from {skill_bank_path} …")
    from navipath_moe.continual_agent import NavigationSkillBank, ContextGate
    from navipath_moe.sequential_observation import (
        ObserveConfig, ContinualSequentialNavigationAgent)
    from navipath_moe.qpmil_adapter import QPMILBackboneAdapter
    import yaml

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dict = yaml.safe_load(open(qpmil_config))

    print("Loading backbone …")
    backbone = QPMILBackboneAdapter(cfg_dict, order=order, fold=fold,
                                    device=device)
    bank = NavigationSkillBank.load(skill_bank_path, map_location=device)

    print("Loading slide features …")
    # Get the test loader for the specified task
    from eval_sequential_observation import build_loaders, TASK_ORDERS
    task_order = TASK_ORDERS[order]
    task_name = task_order[task_id]
    loaders = build_loaders(cfg_dict, task_order, fold)
    test_loader = loaders[task_name]["test"]

    # Pick a specific slide
    slides = list(test_loader.dataset)
    if slide_idx >= len(slides):
        print(f"slide_idx {slide_idx} out of range ({len(slides)} slides); using 0")
        slide_idx = 0
    feats, label = slides[slide_idx]
    Z = torch.tensor(feats, dtype=torch.float32, device=device)
    print(f"  Slide: {len(Z)} patches, label={label}, task={task_name}")

    print("Running t-SNE …")
    Z_np = F.normalize(Z, dim=-1).cpu().numpy()
    xy = run_tsne(Z_np, perplexity=min(30.0, len(Z_np) / 3))

    fig, axes = plt.subplots(1, len(lambda_vals),
                              figsize=(5 * len(lambda_vals), 4.5))
    if len(lambda_vals) == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambda_vals):
        obs_cfg = ObserveConfig(budget=budget, step_size=step_size,
                                redundancy_weight=lam, normalize_base=True,
                                redundancy_mode="maxsim")
        agent = ContinualSequentialNavigationAgent(
            backbone, bank, ContextGate("oracle"), obs_cfg, device=device)
        result = agent.observe(Z, task_id=task_id)
        plot_trace(Z_np, result.trace,
                   title=f"λ = {lam}\n{task_name} slide {slide_idx}, label={label}",
                   ax=ax, n_total=len(Z))

    fig.suptitle(f"SBO trace on {task_name} (fold {fold}, reverse order)\n"
                 f"budget={budget}, step={step_size}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"Fig_seq_trace_real_{task_name}_slide{slide_idx}"
    fig.savefig(OUT_DIR / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → paper/figs/{fname}.{{pdf,png}}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--step-size", type=int, default=16)
    ap.add_argument("--lambdas", default="0.0,0.5,1.0,2.0",
                    help="comma-separated λ values to sweep")
    # real mode only
    ap.add_argument("--skill-bank", default="outputs/skill_bank_reverse_f1.pt")
    ap.add_argument("--slide-idx", type=int, default=0)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--order", default="reverse")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]

    if args.mode == "synthetic":
        print("=== Mode: SYNTHETIC (Mac-compatible) ===")
        synthetic_demo(lambdas=lambdas, budget=args.budget, step_size=args.step_size)
    else:
        print("=== Mode: REAL (requires RunPod + features) ===")
        real_trace(
            skill_bank_path=args.skill_bank,
            slide_idx=args.slide_idx,
            task_id=args.task_id,
            qpmil_config=args.qpmil_config,
            budget=args.budget,
            step_size=args.step_size,
            lambda_vals=lambdas,
            fold=args.fold,
            order=args.order,
        )
