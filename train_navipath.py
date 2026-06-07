"""M5-M8 — NaviPath-MoE 統一訓練腳本。

用 config flag 漸進開啟各元件（v1→v4）：
  v1 (navipath_micro.yaml)      : MicroRouter v1 + ExpertBank + L_bal
  v2 (navipath_macro_micro.yaml): + MacroRouter + fusion
  v3 (同上 + gamma>0)           : + L_sem
  v4 (navipath_full.yaml)       : + replay-free momentum consolidation

QPMIL backbone 凍結；只訓練 router / expert / macro（依 config 決定）。

用法：
  python train_navipath.py --config configs/navipath_micro.yaml --order paper   --fold 1
  python train_navipath.py --config configs/navipath_full.yaml  --order reverse --fold 1 --save-ckpt
  # smoke（CPU/MPS 快速驗證，2 任務限切片）：
  python train_navipath.py --config configs/navipath_micro.yaml --order paper --fold 1 \
      --epochs 1 --tasks 2 --max-train 12 --max-eval 8
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from navipath_moe import (
    get_device, setup_mps,
    MicroRouter, MacroRouter, fuse, top_k_select,
    ExpertBank,
    ExpertImportance, consolidate, snapshot_experts,
    l_sem, l_balance, l_route,
)
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from eval.metrics import summarize
from eval.patch_budget_eval import select_indices
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS

# ── Model ────────────────────────────────────────────────────────────────────

class NaviPathMoE(nn.Module):
    """M5-M8 主模型：frozen backbone + router(s) + optional experts。"""

    def __init__(self, backbone, feat_dim, num_experts, expert_hidden,
                 beta, use_experts, use_macro):
        super().__init__()
        self.backbone = backbone
        self.micro = MicroRouter(feat_dim, num_experts, expert_hidden)
        self.macro = MacroRouter(feat_dim, num_experts, expert_hidden) if use_macro else None
        self.experts = ExpertBank(feat_dim, num_experts, expert_hidden) if use_experts else None
        self.beta = beta
        self.use_experts = use_experts
        self.use_macro = use_macro
        # cache backbone static outputs
        self._f_txt = None
        self._F_p = None

    def _get_f_txt(self, device):
        if self._f_txt is None:
            self._f_txt = self.backbone.class_text_features().to(device).detach()
        return self._f_txt

    def _get_F_p(self, device):
        if self._F_p is None:
            self._F_p = self.backbone.prototype_features().to(device).detach()
        return self._F_p

    def invalidate_cache(self):
        self._f_txt = None
        self._F_p = None

    def forward(self, Z, device, top_k=0):
        """Z:[n,512] → dict with logits, logits_soft, w, score, sim_txt, Z_exp.

        Two parallel paths:
        1. Backbone path  : Z[idx] (original) → frozen backbone → logits (L_C)
        2. Soft-route path: softmax(score[idx]) ⊙ Z[idx] aggregate → cosine logits
                            (L_soft_route) — DIFFERENTIABLE, teaches router which
                            patches help classification, gradient flows to router.
        """
        f_txt = self._get_f_txt(device)   # [C,512]
        F_p   = self._get_F_p(device)     # [M,512]

        w_micro, score, sim_txt = self.micro(Z, f_txt, F_p)  # [n,E],[n],[n]
        w = w_micro
        if self.use_macro and self.macro is not None:
            w = fuse(self.macro(Z), w_micro, self.beta)

        idx = top_k_select(score, top_k)

        # ── Soft-route path (differentiable) ──────────────────────────────────
        # Weighted average of selected patches → cosine-sim classification.
        # Unlike hard top-k, softmax weights keep gradients alive so the router
        # learns "selecting patch i → better prediction" directly from L_soft_route.
        logits_soft = None
        if top_k > 0:
            logit_scale = self.backbone.model.logit_scale.exp().detach()
            w_soft = F.softmax(score[idx], dim=0)          # [K] – differentiable
            z_soft = (w_soft.unsqueeze(-1) * Z[idx]).sum(0)  # [D]
            logits_soft = (z_soft @ f_txt.T) * logit_scale   # [C]

        # ── Expert path (auxiliary) ────────────────────────────────────────────
        # Experts refine selected patch features; used only for L_exp, NOT fed
        # into the backbone (prevents task-specific feature corruption = forgetting).
        Z_exp = None
        if self.use_experts and self.experts is not None:
            Z_exp = self.experts(Z[idx], w[idx])   # [n_sel, D]

        # ── Backbone path (original features, frozen) ──────────────────────────
        logits, _ = self.backbone.aggregate_and_predict(Z[idx], no_grad=False)
        return {"logits": logits, "logits_soft": logits_soft,
                "w": w, "score": score, "sim_txt": sim_txt, "Z_exp": Z_exp}


# ── Data helpers ──────────────────────────────────────────────────────────────

def iter_slides(loader, shift, limit=0):
    n = 0
    for _idx, feats, label in loader:
        yield feats, int(label.view(-1)[0]) + shift
        n += 1
        if limit and n >= limit:
            break


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_task(model, importance, loader, device, task_pos,
                   epochs, top_k, lr, wd, wt, gamma, eta, max_train,
                   teacher=None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    shift = 2 * task_pos
    model.invalidate_cache()
    for epoch in range(epochs):
        seen = total_l = 0
        for feats, label in iter_slides(loader, shift, max_train):
            Z = model.backbone.encode_patches(feats).to(device)
            out = model(Z, device, top_k)

            gt = torch.tensor([label], device=device)
            logit_scale = model.backbone.model.logit_scale.exp().detach()
            f_txt = model._get_f_txt(device)

            # L_C: backbone sees ORIGINAL patch features — no expert corruption
            L_C = F.cross_entropy(out["logits"], gt)

            # L_soft_route: differentiable routing loss — soft-weighted patch avg
            # → cosine-sim classification.  Gradient flows back through softmax
            # weights to the router score, teaching it which patches matter.
            zeta = wt.get("zeta", 0.0)
            if out["logits_soft"] is not None and zeta > 0:
                L_soft_route = F.cross_entropy(out["logits_soft"].unsqueeze(0), gt)
            else:
                L_soft_route = torch.tensor(0., device=device)

            # L_exp: auxiliary cosine-sim loss on expert-transformed features
            if out["Z_exp"] is not None:
                z_exp_agg = out["Z_exp"].mean(dim=0)
                exp_logits = (z_exp_agg @ f_txt.T) * logit_scale
                L_exp = F.cross_entropy(exp_logits.unsqueeze(0), gt)
            else:
                L_exp = torch.tensor(0., device=device)

            # router losses
            w = out["w"]
            s_sem = l_sem(out["score"], out["sim_txt"]) if gamma > 0 else torch.tensor(0.)
            s_bal = l_balance(w) if eta > 0 else torch.tensor(0.)

            s_route = torch.tensor(0.)
            if teacher is not None and wt.get("xi", 0) > 0:
                with torch.no_grad():
                    t_out = teacher(Z, device, top_k)
                s_route = l_route(w, t_out["w"])

            loss = (L_C
                    + zeta  * L_soft_route
                    + 0.5   * L_exp
                    + gamma * s_sem.to(device)
                    + eta   * s_bal.to(device)
                    + wt.get("xi", 0) * s_route.to(device))

            if model.use_experts and importance is not None:
                importance.observe(w)

            opt.zero_grad(); loss.backward(); opt.step()
            total_l += loss.item(); seen += 1

        print(f"    [M5 task_pos {task_pos}] epoch {epoch+1}/{epochs} "
              f"loss={total_l/max(seen,1):.4f} slides={seen}", flush=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(model, loader, device, task_pos, top_k, max_eval):
    model.eval()
    shift = 2 * task_pos
    correct = total = 0
    model.invalidate_cache()
    for feats, label in iter_slides(loader, shift, max_eval):
        Z = model.backbone.encode_patches(feats).to(device)
        out = model(Z, device, top_k)
        pred = out["logits"].argmax(-1)
        correct += int((pred == torch.tensor([label], device=device)).item())
        total += 1
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def eval_budget(model, loader, device, task_pos, budgets, max_eval):
    """Router vs heuristics ACC@K（M5 後接替 M4 的 budget 評估）。"""
    model.eval()
    shift = 2 * task_pos
    f_txt = model._get_f_txt(device)
    F_p   = model._get_F_p(device)
    methods = ("router", "random", "prototype", "semantic")
    res = {m: {("All" if k==0 else k): 0 for k in budgets} for m in methods}
    cnt = {m: {("All" if k==0 else k): 0 for k in budgets} for m in methods}
    model.invalidate_cache()
    for feats, label in iter_slides(loader, shift, max_eval):
        Z = model.backbone.encode_patches(feats).to(device)
        _, score, _ = model.micro(Z, f_txt, F_p)
        gt = torch.tensor([label], device=device)
        for k in budgets:
            key = "All" if k == 0 else k
            for m in methods:
                if m == "router":
                    idx = top_k_select(score, k)
                else:
                    idx = select_indices(m, Z, f_txt, F_p, k)
                logits, _ = model.backbone.aggregate_and_predict(Z[idx])
                res[m][key] += int((logits.argmax(-1) == gt).item())
                cnt[m][key] += 1
    for m in methods:
        for k in res[m]:
            res[m][k] = round(res[m][k] / max(cnt[m][k], 1), 4)
    model.train()
    return res


def print_budget_table(results, budgets):
    cols = ["All" if k == 0 else k for k in budgets]
    hdr = f"{'method':12s}" + "".join(f"ACC@{str(c):>5s}" for c in cols)
    print(f"\n{hdr}\n{'-'*len(hdr)}")
    for m in ("router", "random", "prototype", "semantic"):
        row = f"{m:12s}" + "".join(f"{results[m][c]:>9.4f}" for c in cols)
        print(row)
    k64 = 64 if 64 in budgets else budgets[-1]
    k128 = 128 if 128 in budgets else budgets[-1]
    g64  = results["router"][k64]  - results["random"][k64]
    g128 = results["router"][k128] - results["semantic"][k128]
    go = g64 > 0.02 or results["router"][k128] >= results["semantic"][k128] - 0.01
    print(f"\n[go/no-go] router@{k64}-random@{k64}={g64:+.4f}  "
          f"router@{k128}-semantic@{k128}={g128:+.4f}")
    print(f"[go/no-go] {'GO ✓' if go else 'NO-GO'}", flush=True)
    return go


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", default="",
                    help="M1 checkpoint；空白=隨機初始化（smoke only）")
    ap.add_argument("--epochs", type=int, default=0, help="0=用 config")
    ap.add_argument("--tasks", type=int, default=0, help="0=全部；>0=前 N（smoke）")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval",  type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42); np.random.seed(42)

    nav_cfg  = yaml.safe_load(open(args.config))
    qpmil_cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    if args.tasks:
        order = order[:args.tasks]
    T = len(order)

    budgets = (0, 256, 128, 64, 32)
    epochs  = args.epochs or nav_cfg["epochs"]
    top_k   = nav_cfg.get("train_top_k", 0)
    gamma   = nav_cfg["loss_weights"].get("gamma", 0.0)
    eta     = nav_cfg["loss_weights"].get("eta", 0.0)
    use_consolidation = nav_cfg["loss_weights"].get("use_consolidation", False)

    print(f"[NaviPath] device={device} order={args.order} fold={args.fold} "
          f"use_experts={nav_cfg['use_experts']} use_macro={nav_cfg['use_macro']} "
          f"consolidation={use_consolidation}", flush=True)

    # backbone
    backbone_ckpt = args.backbone_ckpt or \
        f"outputs/qpmil_{args.order}_fold{args.fold}.pt"
    if os.path.exists(backbone_ckpt):
        print(f"[NaviPath] backbone from ckpt: {backbone_ckpt}")
        backbone = build_backbone_from_ckpt(backbone_ckpt, device)
    else:
        print("[NaviPath] backbone fresh (smoke mode)")
        backbone = build_backbone_fresh(qpmil_cfg, order, device)
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

    # log trainable params
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[NaviPath] trainable params = {trainable_n:,}", flush=True)

    loaders  = build_loaders(qpmil_cfg, order, args.fold)
    importance = ExpertImportance(nav_cfg["num_experts"], nav_cfg["ema_decay"]) \
                 if nav_cfg["use_experts"] else None
    old_state = snapshot_experts(model.experts) if nav_cfg["use_experts"] else None
    teacher   = None

    R = np.zeros((T, T))

    for t, ds in enumerate(order):
        print(f"\n[NaviPath] task {t+1}/{T} {ds}", flush=True)
        if importance:
            importance.reset_current()

        train_one_task(
            model, importance, loaders[ds]["train"],
            device, t, epochs, top_k,
            lr=nav_cfg["lr"], wd=nav_cfg["weight_decay"],
            wt=nav_cfg["loss_weights"],
            gamma=gamma, eta=eta,
            max_train=args.max_train,
            teacher=teacher,
        )

        # M8: replay-free consolidation（task 邊界）
        if nav_cfg["use_experts"] and use_consolidation and t > 0:
            I_cur = importance.end_task()
            m_e = consolidate(model.experts, importance, old_state,
                              a=nav_cfg["consolidate_a"], b=nav_cfg["consolidate_b"])
            print(f"[consolidate] m_e = {[round(x,3) for x in m_e.tolist()]}", flush=True)
        elif importance:
            importance.end_task()

        if nav_cfg["use_experts"]:
            old_state = snapshot_experts(model.experts)
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # eval
        for i in range(t + 1):
            R[t, i] = eval_accuracy(model, loaders[order[i]]["test"],
                                    device, i, top_k=0, max_eval=args.max_eval)
        print(f"[task {t+1}] R[{t}] = {np.round(R[t, :t+1], 4).tolist()}", flush=True)

    summary = summarize(R, joint_train_acc=qpmil_cfg.get("joint_train_acc", 0.908))
    print("\n[NaviPath] Accuracy matrix R:")
    print(np.round(R, 4))
    print("[NaviPath] CL summary:", summary)

    # budget eval（最後一個任務的 test set）
    print("\n[NaviPath] Patch-budget eval (last task):")
    bres = eval_budget(model, loaders[order[-1]]["test"],
                       device, T-1, budgets, args.max_eval)
    print_budget_table(bres, budgets)

    # save
    os.makedirs(args.out, exist_ok=True)
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out_json = os.path.join(args.out, f"{cfg_name}_{args.order}_fold{args.fold}.json")
    with open(out_json, "w") as f:
        json.dump({"config": cfg_name, "order": args.order, "fold": args.fold,
                   "tasks": order, "R": R.tolist(), "summary": summary,
                   "budget": bres}, f, indent=2)
    print("[NaviPath] saved:", out_json)

    if args.save_ckpt:
        out_pt = os.path.join(args.out, f"{cfg_name}_{args.order}_fold{args.fold}.pt")
        torch.save({"micro": model.micro.state_dict(),
                    "macro": model.macro.state_dict() if model.macro else None,
                    "experts": model.experts.state_dict() if model.experts else None,
                    "config": nav_cfg, "order": args.order, "fold": args.fold}, out_pt)
        print("[NaviPath] saved ckpt:", out_pt)


if __name__ == "__main__":
    main()
