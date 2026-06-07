"""M9 — Routing Drift Visualization.

在 inference 時錄下每個任務 test set 的 expert 路由分布，
用 heatmap 顯示各任務用了哪些 expert（drift = 任務間偏好轉移）。

用法：
  python viz_routing_drift.py \
      --ckpt outputs/navipath_full_paper_fold1.pt \
      --backbone-ckpt outputs/qpmil_paper_fold1.pt \
      --order paper --fold 1 --max-eval 30

輸出：
  outputs/routing_drift_<order>_fold<fold>.png  （4×4 heatmap）
  outputs/routing_drift_<order>_fold<fold>.json （raw data）
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from navipath_moe import get_device, setup_mps, MicroRouter, MacroRouter, fuse
from navipath_moe.qpmil_adapter import build_backbone_from_ckpt
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS
from train_navipath import NaviPathMoE


# ── collect routing weights ────────────────────────────────────────────────────

@torch.no_grad()
def collect_routing(model, loader, device, task_pos, max_eval=50):
    """回傳 [N, E]：所有 slide 的 patch 平均 expert 路由權重（每 slide 一行）。"""
    model.eval()
    model.invalidate_cache()
    f_txt = model._get_f_txt(device)
    F_p   = model._get_F_p(device)
    shift = 2 * task_pos
    rows = []
    n = 0
    for _idx, feats, label in loader:
        Z = model.backbone.encode_patches(feats).to(device)
        w_micro, _, _ = model.micro(Z, f_txt, F_p)     # [n_patch, E]
        if model.use_macro and model.macro is not None:
            w = fuse(model.macro(Z), w_micro, model.beta)  # [n_patch, E]
        else:
            w = w_micro
        rows.append(w.mean(dim=0).cpu().numpy())       # [E]
        n += 1
        if max_eval and n >= max_eval:
            break
    return np.stack(rows, axis=0) if rows else np.zeros((1, 4))


def drift_table(routing_per_task: list[np.ndarray]) -> np.ndarray:
    """T × E：每個任務的平均 expert 使用率（已 normalize）。"""
    table = np.stack([r.mean(axis=0) for r in routing_per_task], axis=0)  # [T, E]
    table /= table.sum(axis=1, keepdims=True) + 1e-8
    return table


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(table, task_names, out_path):
    """table:[T,E]  — 如果 matplotlib 沒裝，fallback 到 ASCII heatmap。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(6, len(task_names)), 4))
        sns.heatmap(table.T, ax=ax, annot=True, fmt=".2f",
                    xticklabels=[n.replace("tcga_", "") for n in task_names],
                    yticklabels=[f"E{i}" for i in range(table.shape[1])],
                    cmap="Blues", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="white")
        ax.set_title("Expert Routing Drift Across Tasks", fontsize=13, pad=12)
        ax.set_xlabel("Task"); ax.set_ylabel("Expert")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[M9] heatmap saved: {out_path}")
    except ImportError:
        _ascii_heatmap(table, task_names)
        print(f"[M9] matplotlib/seaborn not found; ASCII heatmap shown above.")


def _ascii_heatmap(table, task_names):
    E = table.shape[1]
    print("\n[M9] Routing Drift Table  (row=expert, col=task)")
    header = "       " + "".join(f"{n.replace('tcga_',''):>10s}" for n in task_names)
    print(header)
    for e in range(E):
        row = f"  E{e}  " + "".join(f"{table[t,e]:>10.3f}" for t in range(len(task_names)))
        print(row)
    print()


# ── delta drift (inter-task similarity) ──────────────────────────────────────

def print_drift_summary(table, task_names):
    T = len(task_names)
    if T < 2:
        return
    print("[M9] Inter-task routing distance (L1):")
    for i in range(T):
        for j in range(i + 1, T):
            d = np.abs(table[i] - table[j]).sum()
            print(f"  {task_names[i].replace('tcga_','')} vs "
                  f"{task_names[j].replace('tcga_','')}: {d:.4f}")
    overall = np.mean([np.abs(table[i]-table[j]).sum()
                       for i in range(T) for j in range(i+1, T)])
    print(f"  mean drift L1 = {overall:.4f}")
    signal = overall > 0.05
    print(f"[M9] routing drift signal: {'PRESENT ✓' if signal else 'ABSENT (all experts used equally)'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="navipath_full checkpoint（含 micro/macro/experts）")
    ap.add_argument("--backbone-ckpt", required=True,
                    help="M1 QPMIL backbone checkpoint")
    ap.add_argument("--config", default="configs/navipath_full.yaml")
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--max-eval", type=int, default=50)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    import yaml
    setup_mps()
    device = get_device("auto")
    nav_cfg   = yaml.safe_load(open(args.config))
    qpmil_cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    T = len(order)

    print(f"[M9] device={device} order={args.order} fold={args.fold} "
          f"tasks={order}", flush=True)

    backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    model = NaviPathMoE(
        backbone,
        feat_dim     = nav_cfg["feat_dim"],
        num_experts  = nav_cfg["num_experts"],
        expert_hidden= nav_cfg["expert_hidden"],
        beta         = nav_cfg["beta"],
        use_experts  = nav_cfg["use_experts"],
        use_macro    = nav_cfg["use_macro"],
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.micro.load_state_dict(ckpt["micro"])
    if nav_cfg["use_macro"] and ckpt.get("macro"):
        model.macro.load_state_dict(ckpt["macro"])
    if nav_cfg["use_experts"] and ckpt.get("experts"):
        model.experts.load_state_dict(ckpt["experts"])
    print(f"[M9] loaded navipath ckpt: {args.ckpt}", flush=True)

    loaders = build_loaders(qpmil_cfg, order, args.fold)
    routing_per_task = []
    for t, ds in enumerate(order):
        r = collect_routing(model, loaders[ds]["test"], device, t, args.max_eval)
        routing_per_task.append(r)
        print(f"[M9] task {t+1}/{T} {ds}: collected {len(r)} slides, "
              f"mean expert w = {r.mean(axis=0).round(3).tolist()}", flush=True)

    table = drift_table(routing_per_task)
    print_drift_summary(table, order)

    os.makedirs(args.out, exist_ok=True)
    out_png  = os.path.join(args.out, f"routing_drift_{args.order}_fold{args.fold}.png")
    out_json = os.path.join(args.out, f"routing_drift_{args.order}_fold{args.fold}.json")
    plot_heatmap(table, order, out_png)
    with open(out_json, "w") as f:
        json.dump({"order": args.order, "fold": args.fold, "tasks": order,
                   "table": table.tolist()}, f, indent=2)
    print("[M9] json saved:", out_json)


if __name__ == "__main__":
    main()
