"""CL Baseline Methods on top of QPMIL-VL backbone.

EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017):
  After each task, compute Fisher Information Matrix (diagonal approximation)
  for the trainable parameters (key, prompt, tunable_v). Add regularization
  loss penalizing changes to important parameters in subsequent tasks.

LwF (Learning without Forgetting, Li & Hoiem 2017):
  After each task, save the model's soft predictions as "teacher". On next task,
  add a distillation loss ensuring new model outputs match teacher on current data.

Usage:
  python train_cl_baselines.py --method ewc --order paper   --fold 1 --save-ckpt
  python train_cl_baselines.py --method lwf --order reverse --fold 1 --save-ckpt
  # smoke:
  python train_cl_baselines.py --method ewc --order paper --fold 1 \\
      --epochs 1 --max-train 32 --max-eval 20
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from navipath_moe import get_device, setup_mps
from navipath_moe.qpmil_bootstrap import (
    add_qpmil_to_path, load_base_model, build_ensemble_classes, make_trainable_params,
)
from eval.metrics import summarize
from train_qpmil_runner import (
    load_qpmil_cfg, build_loaders, iter_slides, eval_task, save_checkpoint, TASK_ORDERS
)


# ── EWC ──────────────────────────────────────────────────────────────────────

class EWC:
    """Diagonal Fisher-based regularization (per-parameter importance)."""

    def __init__(self, lam: float = 5000.0):
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.optima: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model, loader, device, shift, cfg, n_samples=200):
        """Estimate diagonal Fisher on current task data after training."""
        fishers: dict[str, torch.Tensor] = {}
        model.eval()
        seen = 0
        xs, ys = [], []
        for feats, label in loader:
            xs.append(feats.to(device))
            ys.append(label)
            seen += 1
            if len(xs) == cfg["bp_every_batch"] or seen >= n_samples:
                logits, _ = model(xs, eval=True)
                labels = torch.tensor([y + shift for y in ys],
                                      dtype=torch.long, device=logits.device)
                loss = F.cross_entropy(logits, labels)
                model.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        fishers[name] = fishers.get(name, torch.zeros_like(p.data)) \
                                        + p.grad.data.pow(2)
                xs, ys = [], []
                if seen >= n_samples:
                    break
        n = max(seen, 1)
        # Accumulate Fisher (sum over tasks) and record optimal parameters
        for name, f in fishers.items():
            self.fisher[name] = self.fisher.get(name, torch.zeros_like(f)) + f / n
        for name, p in model.named_parameters():
            self.optima[name] = p.data.clone()

    def penalty(self, model) -> torch.Tensor:
        """EWC regularization loss."""
        loss = torch.tensor(0., device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (p - self.optima[name]).pow(2)).sum()
        return 0.5 * self.lam * loss


# ── LwF ──────────────────────────────────────────────────────────────────────

class LwF:
    """Knowledge distillation from previous task model (teacher)."""

    def __init__(self, temperature: float = 2.0, lam: float = 1.0):
        self.T = temperature
        self.lam = lam
        self.teacher = None
        self.teacher_shift: int = 0

    def update(self, model):
        """Snapshot current model as teacher after task t."""
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def distill_loss(self, student_model, xs, shift: int) -> torch.Tensor:
        """KD loss: student soft predictions vs teacher soft predictions."""
        if self.teacher is None:
            return torch.tensor(0.)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(xs, eval=True)
        student_logits, _ = student_model(xs, eval=True)
        # Only distill over classes the teacher knows
        C_t = teacher_logits.shape[-1]
        student_logits = student_logits[:, :C_t]
        soft_teacher = F.softmax(teacher_logits / self.T, dim=-1)
        log_soft_student = F.log_softmax(student_logits / self.T, dim=-1)
        return self.lam * self.T ** 2 * F.kl_div(log_soft_student, soft_teacher,
                                                   reduction="batchmean")


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_step_cl(model, optimizer, xs, ys, shift, cfg, device,
                   ewc=None, lwf=None):
    logits, loss_dict, indices, _ = model(xs)
    labels = torch.tensor([y + shift for y in ys], dtype=torch.long, device=logits.device)
    clf = F.cross_entropy(logits, labels)
    loss = clf + cfg["lambda"] * loss_dict["matching_loss"] + cfg["beta"] * loss_dict["class_sim_loss"]

    if ewc is not None:
        loss = loss + ewc.penalty(model)
    if lwf is not None and lwf.teacher is not None:
        loss = loss + lwf.distill_loss(model, xs, shift)

    optimizer.zero_grad()
    loss.backward()
    if cfg["max_norm"] != "None":
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_norm"])
    for p in model.parameters():
        if p.grad is not None and torch.all(p.grad == 0):
            p.grad = None
    optimizer.step()
    freq = torch.zeros(cfg["pool_size"], dtype=torch.int)
    for i in indices.detach().cpu().flatten():
        freq[i] += 1
    return freq


def train_one_task_cl(model, optimizer, loader, cfg, device, task_pos, epochs,
                      max_train, ewc=None, lwf=None):
    shift = 2 * task_pos
    bp = cfg["bp_every_batch"]
    model.eval()
    for epoch in range(epochs):
        xs, ys = [], []
        seen = 0
        for feats, label in iter_slides(loader, max_train):
            xs.append(feats.to(device))
            ys.append(label)
            seen += 1
            if len(xs) == bp:
                _train_step_cl(model, optimizer, xs, ys, shift, cfg, device, ewc, lwf)
                xs, ys = [], []
        if xs:
            _train_step_cl(model, optimizer, xs, ys, shift, cfg, device, ewc, lwf)
        print(f"    [task_pos {task_pos}] epoch {epoch+1}/{epochs} done ({seen} slides)",
              flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ewc", "lwf"], required=True)
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--ewc-lambda", type=float, default=5000.0)
    ap.add_argument("--lwf-lambda", type=float, default=1.0)
    ap.add_argument("--lwf-temp", type=float, default=2.0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save-ckpt", action="store_true")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[{args.method.upper()}] device={device} order={args.order} fold={args.fold}",
          flush=True)

    cfg = load_qpmil_cfg(args.qpmil_config)
    cfg["task_num"] = 0
    order = TASK_ORDERS[args.order]
    if args.tasks:
        order = order[:args.tasks]
    T = len(order)

    add_qpmil_to_path()
    from models import QPMIL_VL, set_tunable_v

    base_model, feat_dim, emb_dim = load_base_model(cfg, device)
    dtype = base_model.dtype
    key, prompt, tunable_v = make_trainable_params(cfg, feat_dim, emb_dim, T, device, dtype)
    loaders = build_loaders(cfg, order, args.fold)

    ewc = EWC(lam=args.ewc_lambda) if args.method == "ewc" else None
    lwf = LwF(temperature=args.lwf_temp, lam=args.lwf_lambda) if args.method == "lwf" else None

    train_key_frequency = {name: [] for name in order}
    R = np.zeros((T, T))

    for t, ds in enumerate(order):
        cfg["task_num"] = t + 1
        seen = order[:t + 1]
        ensemble = build_ensemble_classes(seen, cfg["class_ensemble_path"])
        set_tunable_v(tunable_v, cfg["task_num"])
        model = QPMIL_VL(cfg, base_model, device, key, prompt, tunable_v,
                         ensemble, train_key_frequency)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"],
                                     weight_decay=cfg["adam_weight_decay"], eps=cfg["adam_eps"])
        epochs = args.epochs or cfg["epochs"][t]
        print(f"[task {t+1}/{T}] {ds}  (num_cls={2*(t+1)}, epochs={epochs})", flush=True)

        train_one_task_cl(model, optimizer, loaders[ds]["train"], cfg, device,
                          t, epochs, args.max_train, ewc=ewc, lwf=lwf)
        train_key_frequency[ds].append(torch.zeros(cfg["pool_size"], dtype=torch.int))

        # Post-task update for CL methods
        if ewc is not None:
            ewc.update(model, loaders[ds]["train"], device, 2 * t, cfg)
            print(f"[EWC] Fisher updated after task {t+1}", flush=True)
        if lwf is not None:
            lwf.update(model)
            print(f"[LwF] Teacher snapshot saved after task {t+1}", flush=True)

        for i in range(t + 1):
            R[t, i] = eval_task(model, loaders[order[i]]["test"], i, device,
                                 cfg["bp_every_batch"], args.max_eval)
        print(f"[task {t+1}] R[{t}] = {np.round(R[t, :t+1], 4).tolist()}", flush=True)

        if args.save_ckpt:
            # Reuse save_checkpoint with method name in output path
            orig_order = args.order
            args.order = f"{args.method}_{args.order}"
            save_checkpoint(args, cfg, order, key, prompt, tunable_v)
            args.order = orig_order

    summary = summarize(R, joint_train_acc=0.908 if args.order == "paper" else None)
    print(f"\n[{args.method.upper()}] Accuracy matrix R:")
    print(np.round(R, 4))
    print(f"[{args.method.upper()}] CL summary:", summary)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.method}_{args.order}_fold{args.fold}.json")
    with open(out_path, "w") as f:
        json.dump({"method": args.method, "order": args.order, "fold": args.fold,
                   "tasks": order, "R": R.tolist(), "summary": summary}, f, indent=2)
    print(f"[{args.method.upper()}] saved:", out_path)


if __name__ == "__main__":
    main()
