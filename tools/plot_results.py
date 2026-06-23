#!/usr/bin/env python3
"""Generate paper figures for NaviPath-MoE.

P0  Patch-budget curves   (router vs random/prototype/semantic, per order/task)
P1  R-matrix heatmaps      (continual-learning behavior, QPMIL vs NaviPath)
P2  (lite) Feature-space scatter colored by router score  [optional, needs model]

P0 + P1 read ONLY the result JSONs (no torch) and always run.
P2-lite lazily imports the model stack; if anything is missing it is skipped
with a warning, so the must-have figures still get produced.

Usage:
  # must-haves only (fast, no GPU/model):
  python tools/plot_results.py --outputs outputs --figdir outputs/figs

  # also try the qualitative P2-lite figure (auto-picks a slide):
  python tools/plot_results.py --outputs outputs --figdir outputs/figs --p2 \
      --p2-order paper --p2-fold 1
"""
from __future__ import annotations

import argparse
import os
import sys

# matplotlib cache dir may be unwritable in some envs — point it at /tmp.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "mpl-navipath"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reuse parsing/aggregation from the collector (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_results as cr  # noqa: E402

ARMS = ["router", "random", "prototype", "semantic"]
ARM_STYLE = {
    "router":    dict(color="#d62728", marker="o", lw=2.2, zorder=5),
    "random":    dict(color="#7f7f7f", marker="s", lw=1.5, ls="--"),
    "prototype": dict(color="#1f77b4", marker="^", lw=1.5, ls="--"),
    "semantic":  dict(color="#2ca02c", marker="D", lw=1.5, ls="--"),
}


def _finite_budgets(results):
    return sorted(int(b) for b in cr.budget_keys(results))


def _savefig(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(figdir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {os.path.join(figdir, name)}.{{pdf,png}}")


# ----------------------------------------------------------------------------
# P0 — patch-budget curves
# ----------------------------------------------------------------------------
def plot_budget(records, kind, figdir, title_prefix):
    agg, folds, _ = cr._aggregate_budget(records, kind)
    if not agg:
        print(f"[P0] no '{kind}' data — skip")
        return
    keys = sorted(agg.keys())
    ncol = min(len(keys), 3)
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow),
                             squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        order, eval_task = key
        results = agg[key]
        bks = _finite_budgets(results)
        n = len(folds[key])
        for arm in ARMS:
            ys, es = [], []
            for b in bks:
                m, s = cr.mean_std(results.get(arm, {}).get(str(b), []))
                ys.append(np.nan if m is None else m)
                es.append(0.0 if s is None else s)
            style = ARM_STYLE[arm]
            ax.errorbar(bks, ys, yerr=es, label=arm, capsize=3,
                        marker=style.get("marker"), color=style["color"],
                        lw=style["lw"], ls=style.get("ls", "-"),
                        zorder=style.get("zorder", 3))
        # full-bag (All) reference line
        m_all, _ = cr.mean_std(results.get("router", {}).get("All", []))
        if m_all is not None:
            ax.axhline(m_all, color="black", lw=0.8, ls=":", alpha=0.6)
            ax.text(bks[-1], m_all, " All-patch", va="bottom", ha="right",
                    fontsize=7, color="black", alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(bks)
        ax.set_xticklabels([str(b) for b in bks])
        ax.set_xlabel("patch budget (K)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{order} / {eval_task} (n={n})", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    # blank leftover axes
    for j in range(len(keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _savefig(fig, figdir, f"P0_{kind}")


# ----------------------------------------------------------------------------
# P1 — R-matrix heatmaps
# ----------------------------------------------------------------------------
def _mean_R(records, method, order):
    Rs, tasks = [], None
    for r in records:
        if r["kind"] == method and r["meta"]["order"] == order:
            R = r["data"].get("R")
            if R:
                Rs.append(np.array(R, dtype=float))
                tasks = r["data"].get("tasks", tasks)
    if not Rs:
        return None, None
    shp = Rs[0].shape
    Rs = [R for R in Rs if R.shape == shp]
    return np.mean(Rs, axis=0), tasks


def plot_recent_vs_old(records, figdir):
    """Same-task recency flip: router@K when the task was learned LAST (recent,
    from router_v0_*) vs FIRST then overwritten (old, from oldtask_budget_*).
    The killer figure: only recency differs, yet router flips GO -> crash."""
    from collections import defaultdict
    recent = defaultdict(lambda: defaultdict(list))
    old = defaultdict(lambda: defaultdict(list))
    old_rand = defaultdict(lambda: defaultdict(list))
    for r in records:
        d = r["data"]; res = d.get("results")
        if not res:
            continue
        task = d.get("eval_task", "?")
        if r["kind"] == "router_v0":
            for b, v in res.get("router", {}).items():
                recent[task][b].append(v)
        elif r["kind"] == "oldtask_budget":
            for b, v in res.get("router", {}).items():
                old[task][b].append(v)
            for b, v in res.get("random", {}).items():
                old_rand[task][b].append(v)
    tasks = [t for t in recent if t in old]
    if not tasks:
        print("[flip] need both router_v0 and oldtask_budget for a task — skip")
        return
    ncol = len(tasks)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2), squeeze=False)
    for i, task in enumerate(sorted(tasks)):
        ax = axes[0][i]
        bks = sorted({int(b) for b in recent[task] if b not in ("All", "0")}
                     & {int(b) for b in old[task] if b not in ("All", "0")})
        def curve(src):
            return [cr.mean_std(src[task].get(str(b), []))[0] for b in bks]
        ax.plot(bks, curve(recent), "-o", color="#d62728", lw=2.4,
                label=f"router — {task.replace('tcga_','')} as RECENT (last learned)")
        ax.plot(bks, curve(old), "--s", color="#d62728", lw=2.0, alpha=0.55,
                label=f"router — {task.replace('tcga_','')} as OLD (learned first)")
        ax.plot(bks, curve(old_rand), ":", color="#7f7f7f", lw=1.5,
                label="random (no-skill baseline)")
        ax.set_xscale("log", base=2); ax.set_xticks(bks)
        ax.set_xticklabels([str(b) for b in bks])
        ax.set_xlabel("patch budget (K)"); ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"{task.replace('tcga_','')}: same task, recency flip", fontsize=10)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("Same-task recency flip: router selection is forgotten on old tasks",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, figdir, "P0b_recency_flip")


def plot_r_matrix(records, figdir):
    methods = [("qpmil", "QPMIL baseline"), ("navipath_full", "NaviPath (full)")]
    orders = ["paper", "reverse"]
    panels = [(m, lbl, o) for (m, lbl) in methods for o in orders]
    have = [(m, lbl, o, *_mean_R(records, m, o)) for (m, lbl, o) in panels]
    have = [h for h in have if h[3] is not None]
    if not have:
        print("[P1] no R matrices — skip")
        return
    ncol = 2
    nrow = (len(have) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow),
                             squeeze=False)
    for i, (m, lbl, o, R, tasks) in enumerate(have):
        ax = axes[i // ncol][i % ncol]
        masked = np.ma.masked_where(np.triu(np.ones_like(R), k=1) > 0, R)
        im = ax.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
        T = R.shape[0]
        for a in range(T):
            for b in range(a + 1):
                ax.text(b, a, f"{R[a, b]:.2f}", ha="center", va="center",
                        color="white" if R[a, b] < 0.6 else "black", fontsize=8)
        labs = tasks if tasks else [f"t{j}" for j in range(T)]
        labs = [t.replace("tcga_", "") for t in labs]
        ax.set_xticks(range(T)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(T)); ax.set_yticklabels([f"after {labs[j]}" for j in range(T)], fontsize=8)
        ax.set_xlabel("evaluated task"); ax.set_title(f"{lbl} — {o}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(have), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("R-matrix: accuracy on task j after learning task i", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, figdir, "P1_r_matrix")


# ----------------------------------------------------------------------------
# P2-lite — feature-space scatter colored by router score (optional)
# ----------------------------------------------------------------------------
def plot_p2_lite(figdir, repo_root, data_root, order, fold, feat_file, topk):
    try:
        sys.path.insert(0, repo_root)
        import glob
        import torch
        from navipath_moe import MicroRouterV0, top_k_select, get_device, setup_mps
        from navipath_moe.qpmil_adapter import build_backbone_from_ckpt
        from train_qpmil_runner import TASK_ORDERS
        from sklearn.manifold import TSNE
    except Exception as e:  # noqa: BLE001
        print(f"[P2-lite] skipped (import failed: {e})")
        return

    try:
        setup_mps()
        device = get_device("auto")
        bb_ckpt = os.path.join(repo_root, "outputs", f"qpmil_{order}_fold{fold}.pt")
        rt_ckpt = os.path.join(repo_root, "outputs", f"router_v0_{order}_fold{fold}.pt")
        for p in (bb_ckpt, rt_ckpt):
            if not os.path.exists(p):
                print(f"[P2-lite] skipped (missing ckpt: {p})")
                return
        last_ds = TASK_ORDERS[order][-1]
        if not feat_file:
            cand = sorted(glob.glob(os.path.join(
                data_root, last_ds, "feats-l1-s256_CONCH", "pt_files", "*.pt")))
            if not cand:
                print(f"[P2-lite] skipped (no feature .pt under {data_root}/{last_ds})")
                return
            feat_file = cand[0]
        print(f"[P2-lite] backbone={os.path.basename(bb_ckpt)} "
              f"router={os.path.basename(rt_ckpt)} slide={os.path.basename(feat_file)}")

        # make portable: ckpts trained on RunPod embed /workspace/src/navipath
        # absolute paths; remap that prefix to this repo root so P2-lite runs
        # on any machine that has the repo + CONCH weights.
        backbone = build_backbone_from_ckpt(
            bb_ckpt, device,
            path_remap=("/workspace/src/navipath", repo_root))
        for p in backbone.parameters():
            p.requires_grad_(False)
        router = MicroRouterV0(feat_dim=512, hidden=256).to(device)
        router.load_state_dict(torch.load(rt_ckpt, map_location=device))
        router.eval()

        feats = torch.load(feat_file, map_location="cpu")
        if isinstance(feats, dict):  # be defensive
            feats = feats.get("features", next(iter(feats.values())))
        with torch.no_grad():
            Z = backbone.encode_patches(feats).to(device)
            f_txt = backbone.class_text_features().to(device)
            F_p = backbone.prototype_features().to(device)
            score, _ = router(Z, f_txt, F_p)
        score = score.detach().float().view(-1).cpu().numpy()
        Znp = Z.detach().float().cpu().numpy()
        n = Znp.shape[0]

        # subsample for t-SNE speed but always keep the selected top-K
        k = min(topk, n)
        sel = set(top_k_select(torch.from_numpy(score), k).cpu().numpy().tolist())
        rng = np.random.default_rng(0)
        cap = 3000
        if n > cap:
            keep = set(sel)
            extra = [i for i in range(n) if i not in keep]
            keep |= set(rng.choice(extra, size=cap - len(keep), replace=False).tolist())
            idx = np.array(sorted(keep))
        else:
            idx = np.arange(n)
        emb = TSNE(n_components=2, init="pca", perplexity=min(30, len(idx) - 1),
                   random_state=0).fit_transform(Znp[idx])
        s_sub = score[idx]
        is_sel = np.array([i in sel for i in idx])

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        sc = ax.scatter(emb[~is_sel, 0], emb[~is_sel, 1], c=s_sub[~is_sel],
                        cmap="viridis", s=10, alpha=0.6)
        ax.scatter(emb[is_sel, 0], emb[is_sel, 1], facecolors="none",
                   edgecolors="red", s=42, linewidths=1.2,
                   label=f"router top-{k} selected")
        fig.colorbar(sc, ax=ax, label="router score")
        ax.set_title(f"P2-lite: patch feature space — {order}/{last_ds}\n"
                     f"slide {os.path.basename(feat_file)[:18]}…  (n={n})", fontsize=10)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)
        _savefig(fig, figdir, f"P2lite_{order}_fold{fold}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[P2-lite] skipped (runtime error: {e})")
        traceback.print_exc()


def plot_p2_contrast(figdir, repo_root, data_root, fold, slide_ds, topk):
    """Mechanism contrast (Fig 3): the SAME slide scored by the router that learned
    `slide_ds` RECENTLY vs the router that learned it long ago (OLD). One shared
    t-SNE; two colorings + top-K. Shows old-router scores degenerate -> mis-select.
    `slide_ds` must be a task that is LAST in one order and FIRST in the other
    (e.g. tcga_esca: paper-last=recent, reverse-first=old)."""
    try:
        sys.path.insert(0, repo_root)
        import glob
        import torch
        from navipath_moe import MicroRouterV0, top_k_select, get_device, setup_mps
        from navipath_moe.qpmil_adapter import build_backbone_from_ckpt
        from train_qpmil_runner import TASK_ORDERS
        from sklearn.manifold import TSNE
    except Exception as e:  # noqa: BLE001
        print(f"[P2-contrast] skipped (import failed: {e})")
        return

    # which order has slide_ds last (recent) vs first (old)?
    recent_order = old_order = None
    for o, seq in TASK_ORDERS.items():
        if seq[-1] == slide_ds:
            recent_order = o
        if seq[0] == slide_ds:
            old_order = o
    if recent_order is None or old_order is None:
        print(f"[P2-contrast] {slide_ds} is not both last & first across orders — skip")
        return

    try:
        setup_mps()
        device = get_device("auto")
        cand = sorted(glob.glob(os.path.join(
            data_root, slide_ds, "feats-l1-s256_CONCH", "pt_files", "*.pt")))
        if not cand:
            print(f"[P2-contrast] no slide under {data_root}/{slide_ds} — skip")
            return
        feat_file = cand[0]
        feats = torch.load(feat_file, map_location="cpu")
        if isinstance(feats, dict):
            feats = feats.get("features", next(iter(feats.values())))

        def score_with(order):
            bb = os.path.join(repo_root, "outputs", f"qpmil_{order}_fold{fold}.pt")
            rt = os.path.join(repo_root, "outputs", f"router_v0_{order}_fold{fold}.pt")
            if not (os.path.exists(bb) and os.path.exists(rt)):
                raise FileNotFoundError(f"missing ckpt for order={order}")
            backbone = build_backbone_from_ckpt(
                bb, device, path_remap=("/workspace/src/navipath", repo_root))
            for p in backbone.parameters():
                p.requires_grad_(False)
            router = MicroRouterV0(feat_dim=512, hidden=256).to(device)
            router.load_state_dict(torch.load(rt, map_location=device))
            router.eval()
            with torch.no_grad():
                Z = backbone.encode_patches(feats).to(device)
                s, _ = router(Z, backbone.class_text_features().to(device),
                              backbone.prototype_features().to(device))
            return Z.detach().float().cpu().numpy(), s.detach().float().view(-1).cpu().numpy()

        print(f"[P2-contrast] slide={os.path.basename(feat_file)} "
              f"recent={recent_order} old={old_order}")
        Znp, s_recent = score_with(recent_order)
        _, s_old = score_with(old_order)
        n = Znp.shape[0]
        k = min(topk, n)

        rng = np.random.default_rng(0)
        sel_r = set(top_k_select(torch.from_numpy(s_recent), k).tolist())
        sel_o = set(top_k_select(torch.from_numpy(s_old), k).tolist())
        cap = 3000
        if n > cap:
            keep = set(sel_r) | set(sel_o)
            extra = [i for i in range(n) if i not in keep]
            keep |= set(rng.choice(extra, size=cap - len(keep), replace=False).tolist())
            idx = np.array(sorted(keep))
        else:
            idx = np.arange(n)
        emb = TSNE(n_components=2, init="pca", perplexity=min(30, len(idx) - 1),
                   random_state=0).fit_transform(Znp[idx])

        # shared color scale (z-score each panel for comparability of structure)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
        tag = slide_ds.replace("tcga_", "")
        for ax, s_all, sel, title in [
            (axes[0], s_recent, sel_r, f"{tag} as RECENT (order={recent_order}): GO"),
            (axes[1], s_old, sel_o, f"{tag} as OLD (order={old_order}): forgotten")]:
            s_sub = s_all[idx]
            is_sel = np.array([i in sel for i in idx])
            sc = ax.scatter(emb[~is_sel, 0], emb[~is_sel, 1], c=s_sub[~is_sel],
                            cmap="viridis", s=10, alpha=0.6)
            ax.scatter(emb[is_sel, 0], emb[is_sel, 1], facecolors="none",
                       edgecolors="red", s=42, linewidths=1.2,
                       label=f"router top-{k}")
            fig.colorbar(sc, ax=ax, label="router score", fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
            ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.2)
        fig.suptitle(f"P2 mechanism: same {tag} slide, recent vs forgotten router "
                     f"(slide {os.path.basename(feat_file)[:16]}…, n={n})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _savefig(fig, figdir, f"P2contrast_{tag}_fold{fold}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[P2-contrast] skipped (runtime error: {e})")
        traceback.print_exc()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--figdir", default="outputs/figs")
    ap.add_argument("--p2", action="store_true", help="also try P2-lite (needs model stack)")
    ap.add_argument("--p2-order", default="paper", choices=["paper", "reverse"])
    ap.add_argument("--p2-fold", type=int, default=1)
    ap.add_argument("--p2-feat", default="", help="explicit slide .pt; default auto-pick")
    ap.add_argument("--p2-topk", type=int, default=64)
    ap.add_argument("--p2-contrast", action="store_true",
                    help="Fig 3: same slide scored by recent vs forgotten router")
    ap.add_argument("--p2-contrast-task", default="tcga_esca",
                    help="task that is last in one order & first in the other")
    ap.add_argument("--data-root", default="/Users/aaron/research/can_dataset")
    args = ap.parse_args()

    if not os.path.isdir(args.outputs):
        raise SystemExit(f"[error] outputs dir not found: {args.outputs}")
    records = cr.load_all(args.outputs)
    if not records:
        raise SystemExit(f"[error] no recognised result JSONs in {args.outputs}")
    print(f"[plot] {len(records)} result files from {args.outputs}")

    # P0
    plot_budget(records, "router_v0", args.figdir,
                "P0: Patch-budget on LAST task (router vs heuristics)")
    plot_budget(records, "oldtask_budget", args.figdir,
                "P0: Patch-budget on OLD tasks (router forgetting)")
    # P0b — same-task recency flip (the key defense figure)
    plot_recent_vs_old(records, args.figdir)
    # P1
    plot_r_matrix(records, args.figdir)
    # P2-lite / P2-contrast (optional, need model stack)
    if args.p2 or args.p2_contrast:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.p2:
            plot_p2_lite(args.figdir, repo_root, args.data_root,
                         args.p2_order, args.p2_fold, args.p2_feat, args.p2_topk)
        if args.p2_contrast:
            plot_p2_contrast(args.figdir, repo_root, args.data_root,
                             args.p2_fold, args.p2_contrast_task, args.p2_topk)

    print(f"[done] figures in {args.figdir}/")


if __name__ == "__main__":
    main()
