"""Continual training loop skeleton for NaviPath-MoE.

device-agnostic（Mac MPS 開發 / RunPod CUDA 跑重活）。標 TODO 處接 QPMIL-VL。
用法:
  python train_continual.py --config configs/navipath_full.yaml --order paper
  python train_continual.py --config configs/navipath_full.yaml --order reverse
"""
from __future__ import annotations

import argparse, copy
import numpy as np, torch, yaml

from navipath_moe import (
    NaviPathMoE, ExpertImportance, consolidate, snapshot_experts,
    l_sem, l_balance, l_route, total_loss, get_device, setup_mps,
)
from eval.metrics import summarize, patch_budget_table

# 任務序用 feature 目錄名。paper forward = NSCLC(lung) 先、ESCA 最後；reverse 相反。
TASK_ORDERS = {
    "paper":   ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
}


def build_backbone(cfg, device):
    # TODO(坤倫): 回傳已載入 CONCH features 的 QPMIL-VL backbone（4 個 hook 見 model.py）。
    raise NotImplementedError("接上 QPMIL-VL backbone")

def get_loader(task_name, split, cfg):
    # TODO(珈鋒): 回傳該 cohort dataloader（每筆 = 一張切片 [n,512] CONCH 特徵 + label）。
    raise NotImplementedError("接上 data loader")


def train_one_task(model, importance, loader, cfg, device, teacher=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w = cfg["loss_weights"]; model.train()
    for _ in range(cfg["epochs"]):
        for slide, label in loader:
            out = model(slide, top_k=cfg.get("train_top_k", 0))
            importance.observe(out["w"])
            sem = l_sem(out["score"], out["sim_txt"], tau=cfg.get("tau", 1.0))
            bal = l_balance(out["w"])
            route = None
            if teacher is not None and w.get("xi", 0.0) > 0:
                with torch.no_grad():
                    route = l_route(out["w"], teacher(slide, top_k=cfg.get("train_top_k", 0))["w"])
            loss = total_loss(out["qpmil_losses"], sem, bal, route, w)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_task(model, loader, top_k=0):
    model.eval(); correct = total = 0
    for slide, label in loader:
        pred = model(slide, top_k=top_k)["logits"].argmax(-1)
        correct += int((pred == label).sum()); total += len(label)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    setup_mps(); device = get_device(cfg.get("device", "auto"))
    print("device:", device)

    tasks = TASK_ORDERS[args.order]; T = len(tasks)
    backbone = build_backbone(cfg, device)
    model = NaviPathMoE(backbone, feat_dim=512, num_experts=cfg["num_experts"],
                        expert_hidden=cfg["expert_hidden"], beta=cfg["beta"],
                        use_experts=cfg["use_experts"], use_macro=cfg["use_macro"]).to(device)
    importance = ExpertImportance(cfg["num_experts"], ema_decay=cfg["ema_decay"])

    R = np.zeros((T, T)); teacher = None
    old_state = snapshot_experts(model.experts) if model.use_experts else None

    for t, task in enumerate(tasks):
        train_one_task(model, importance, get_loader(task, "train", cfg), cfg, device, teacher)
        importance.end_task()
        if model.use_experts and cfg["loss_weights"].get("use_consolidation", True) and t > 0:
            m = consolidate(model.experts, importance, old_state,
                            a=cfg["consolidate_a"], b=cfg["consolidate_b"])
            print(f"[task {t} {task}] consolidation m_e = {[round(x,3) for x in m.tolist()]}")
        if model.use_experts:
            old_state = snapshot_experts(model.experts)
        teacher = copy.deepcopy(model).eval()
        for i in range(t + 1):
            R[t, i] = eval_task(model, get_loader(tasks[i], "test", cfg))

    print("Accuracy matrix R:\n", np.round(R, 4))
    print("CL summary:", summarize(R, joint_train_acc=cfg.get("joint_train_acc")))
    last = get_loader(tasks[-1], "test", cfg)
    print("Patch-budget ACC:", patch_budget_table(lambda k: eval_task(model, last, top_k=k)))


if __name__ == "__main__":
    main()
