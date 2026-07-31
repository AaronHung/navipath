#!/usr/bin/env python3
"""run_zeronav.py — ZeroNav experiment driver.

Project codename: ZeroNav  (ZeroSlide-inspired backbone-agnostic navigation)
Outputs:          outputs/zeronav/       (never touches existing outputs/)

Architecture:
  - Frozen diagnostic backbone (CONCH) provides patch embeddings Z and class text
    features f_txt. The backbone is treated as a black-box feature extractor.
  - TextNavRouter input: [Z(512); max_text_sim(1); text_entropy(1)] = 514-dim
    → backbone-agnostic: no prototype features, no backbone internals accessed.

Subcommands
-----------
  train   sequential training of TextNavRouter on T tasks → ZeroNavSkillBank
  eval    fine-grained λ sweep (inference only, no retraining):
            • zeronav multishot: TextNavRouter + SBO 4×16 (step_size=16)
            • zeronav oneshot:   TextNavRouter + top-64-at-once (step_size=∞, λ=0)
            • zeroshot multishot: ZeroSlide score + SBO 4×16
            • zeroshot oneshot:  ZeroSlide score + top-64-at-once
  analyze router weight cosine similarity matrix + cross-task accuracy matrix
  smoke   quick end-to-end test on Mac (random backbone, tiny data)

λ sweep (finer resolution vs old [0,1,2,4]):
  0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0

RunPod usage (fold 1, reverse order, ~2–3 h total):
  # backbone_reverse_fold1.pt  = frozen CONCH diagnostic backbone (symlink on RunPod)
  python run_zeronav.py train \\
      --backbone-ckpt outputs/backbone_reverse_fold1.pt \\
      --order reverse --fold 1 --epochs 5 --top-k 64
  python run_zeronav.py eval \\
      --backbone-ckpt outputs/backbone_reverse_fold1.pt \\
      --order reverse --fold 1
  python run_zeronav.py analyze \\
      --backbone-ckpt outputs/backbone_reverse_fold1.pt \\
      --order reverse --fold 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

# ── Existing infrastructure (unchanged) ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS   # noqa: E402
from train_router_v0 import iter_slides                                      # noqa: E402
from navipath_moe import get_device, setup_mps                               # noqa: E402
from navipath_moe.qpmil_adapter import build_backbone_from_ckpt, build_backbone_fresh  # noqa: E402
from navipath_moe.sequential_observation import (                            # noqa: E402
    SequentialBudgetedObserver, ObserveConfig,
)

# ── ZeroNav components ───────────────────────────────────────────────────────
from zeronav.router import TextNavRouter, ZeroNavSkillBank, zeroslide_score  # noqa: E402

ZERONAV_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


# ════════════════════════════════════════════════════════════════════════════
# TRAIN PHASE
# ════════════════════════════════════════════════════════════════════════════

def _train_one_task(router: TextNavRouter, backbone, loader, device,
                    task_pos: int, epochs: int, top_k: int,
                    lr: float, max_train: int) -> None:
    """Train TextNavRouter on one task.

    Loss path: router score [n] → soft top-K weighted sum → CONCH text logit
               → cross-entropy → backprop into router only.
    Identical loss to MicroRouterV0 training, but without F_p in router input.
    backbone.prototype_features() is NEVER called.
    """
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=1e-4)
    shift = 2 * task_pos
    f_txt = backbone.class_text_features().to(device).detach()    # [C, 512]
    logit_scale = backbone.model.logit_scale.exp().detach()

    router.train()
    for epoch in range(epochs):
        seen = total_loss = 0
        for feats, label in iter_slides(loader, shift, max_train):
            Z = backbone.encode_patches(feats).to(device)         # [n, 512]
            score = router(Z, f_txt)                              # [n]  — no F_p

            n = Z.shape[0]
            k = min(top_k, n) if top_k > 0 else n
            topk_score, topk_idx = torch.topk(score, k)
            w = F.softmax(topk_score, dim=0).unsqueeze(-1)        # [k, 1]
            Z_w = F.normalize(
                (w * Z[topk_idx].detach()).sum(0, keepdim=True), dim=-1
            )                                                      # [1, 512]

            logits = logit_scale * (Z_w @ f_txt.t())              # [1, C]
            gt = torch.tensor([label], device=device)
            loss = F.cross_entropy(logits, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            seen += 1

        avg = total_loss / max(seen, 1)
        print(f"  [zeronav t{task_pos}] epoch {epoch + 1}/{epochs}"
              f"  loss={avg:.4f}  slides={seen}", flush=True)


def run_train(args) -> None:
    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42)

    out_dir = os.path.join(args.out, "zeronav")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    cfg = load_qpmil_cfg(args.qpmil_config)
    order = args.order
    task_order = TASK_ORDERS[order]

    if args.backbone_ckpt:
        print(f"[zeronav train] backbone: {args.backbone_ckpt}")
        backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    else:
        print("[zeronav train] random backbone (smoke/debug)")
        backbone = build_backbone_fresh(cfg, task_order, device)

    loaders = build_loaders(cfg, task_order, args.fold)
    bank = ZeroNavSkillBank()

    for task_pos, task_name in enumerate(task_order):
        print(f"\n[zeronav] ▶ task {task_pos}: {task_name}")
        router = TextNavRouter().to(device)
        _train_one_task(
            router, backbone, loaders[task_name]["train"], device, task_pos,
            epochs=args.epochs, top_k=args.top_k,
            lr=args.lr, max_train=args.max_train,
        )
        bank.add_skill(task_pos, router)
        # checkpoint after each task (safe resumption)
        ckpt_path = os.path.join(out_dir,
                                 f"skill_bank_{order}_f{args.fold}_t{task_pos}.pt")
        bank.save(ckpt_path)

    final_path = os.path.join(out_dir, f"skill_bank_{order}_f{args.fold}.pt")
    bank.save(final_path)
    print(f"\n[zeronav train] ✓ done → {final_path}")


# ════════════════════════════════════════════════════════════════════════════
# EVAL PHASE
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_one_task(backbone, bank: ZeroNavSkillBank, loader, device,
                   eval_task: int, lambdas: list[float],
                   step_size: int, budget: int,
                   max_eval: int, normalize_base: bool = True) -> dict:
    """Lambda sweep for one eval task.

    Returns dict:
      {
        "lambda_X.XX": {
          "zeronav_multishot": acc,   TextNavRouter + SBO 4×16
          "zeronav_oneshot":   acc,   TextNavRouter + top-K at once
          "zeroshot_multishot": acc,  ZeroSlide score + SBO 4×16
          "zeroshot_oneshot":  acc,   ZeroSlide score + top-K at once
        },
        ...
        "n_slides": int,
      }
    """
    shift = 2 * eval_task
    f_txt = backbone.class_text_features().to(device)
    router = bank.build_router(eval_task, device)

    slides = list(iter_slides(loader, shift, max_eval))
    n_slides = len(slides)
    if n_slides == 0:
        print(f"  [warn] no slides for eval_task={eval_task}")
        return {"n_slides": 0}

    def predict_fn(S):
        return backbone.aggregate_and_predict(S)[0]

    result: dict = {}

    for lam in lambdas:
        cfg_multi = ObserveConfig(
            budget=budget, step_size=step_size,
            redundancy_weight=lam, normalize_base=normalize_base,
            redundancy_mode="maxsim",
        )
        cfg_one = ObserveConfig(
            budget=budget, step_size=10 ** 9,
            redundancy_weight=0.0, normalize_base=normalize_base,
            redundancy_mode="maxsim",
        )

        c_zn_m = c_zn_o = 0    # zeronav multishot / oneshot
        c_zs_m = c_zs_o = 0    # zeroshot multishot / oneshot

        for feats, label in slides:
            Z = backbone.encode_patches(feats).to(device)
            gt = int(label)

            # ZeroNav router scores (no F_p)
            base_nav = router(Z, f_txt)
            # ZeroSlide zero-shot scores (text-patch cosine similarity)
            base_zs = zeroslide_score(Z, f_txt)

            # multishot (sequential SBO, λ-dependent)
            obs_m = SequentialBudgetedObserver(cfg_multi)
            r = obs_m.observe(Z, base_nav, predict_fn)
            c_zn_m += int(int(r.logits.reshape(-1).argmax()) == gt)

            r = obs_m.observe(Z, base_zs, predict_fn)          # reuse same obs
            c_zs_m += int(int(r.logits.reshape(-1).argmax()) == gt)

            # oneshot (single round, no redundancy, λ=0)
            obs_o = SequentialBudgetedObserver(cfg_one)
            r = obs_o.observe(Z, base_nav, predict_fn)
            c_zn_o += int(int(r.logits.reshape(-1).argmax()) == gt)

            r = obs_o.observe(Z, base_zs, predict_fn)
            c_zs_o += int(int(r.logits.reshape(-1).argmax()) == gt)

        acc = lambda c: round(c / n_slides, 4)
        key = f"lambda_{lam:.2f}"
        result[key] = {
            "zeronav_multishot":  acc(c_zn_m),
            "zeronav_oneshot":    acc(c_zn_o),
            "zeroshot_multishot": acc(c_zs_m),
            "zeroshot_oneshot":   acc(c_zs_o),
        }
        print(
            f"  λ={lam:.2f}  "
            f"nav_multi={result[key]['zeronav_multishot']:.3f}  "
            f"nav_one={result[key]['zeronav_oneshot']:.3f}  "
            f"zs_multi={result[key]['zeroshot_multishot']:.3f}  "
            f"zs_one={result[key]['zeroshot_oneshot']:.3f}",
            flush=True,
        )

    result["n_slides"] = n_slides
    return result


def run_eval(args) -> None:
    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42)

    order = args.order
    task_order = TASK_ORDERS[order]
    cfg = load_qpmil_cfg(args.qpmil_config)

    out_dir = os.path.join(args.out, "zeronav", "eval")
    os.makedirs(out_dir, exist_ok=True)

    if args.backbone_ckpt:
        backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    else:
        backbone = build_backbone_fresh(cfg, task_order, device)

    bank_filename = args.skill_bank_in or f"skill_bank_{order}_f{args.fold}.pt"
    bank_path = os.path.join(args.out, "zeronav", bank_filename)
    print(f"[zeronav eval] skill bank: {bank_path}")
    bank = ZeroNavSkillBank.load(bank_path, map_location=str(device))

    loaders = build_loaders(cfg, task_order, args.fold)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    eval_tasks = (
        [int(x) for x in args.eval_tasks.split(",")]
        if args.eval_tasks else list(range(len(task_order)))
    )

    for eval_task in eval_tasks:
        if not bank.has(eval_task):
            print(f"  [warn] no skill for task {eval_task}, skipping")
            continue
        task_name = task_order[eval_task]
        print(f"\n[zeronav eval] ▶ task {eval_task}: {task_name}")

        res = _eval_one_task(
            backbone, bank, loaders[task_name]["test"], device, eval_task,
            lambdas=lambdas, step_size=args.step_size, budget=args.budget,
            max_eval=args.max_eval, normalize_base=True,
        )

        out_file = os.path.join(
            out_dir,
            f"task{eval_task}_{task_name}_{order}_f{args.fold}.json",
        )
        with open(out_file, "w") as f:
            json.dump({
                "task": eval_task, "task_name": task_name,
                "order": order, "fold": args.fold,
                "budget": args.budget, "step_size": args.step_size,
                "lambdas": lambdas, "normalize_base": True,
                "results": res,
            }, f, indent=2)
        print(f"  ✓ saved → {out_file}")


# ════════════════════════════════════════════════════════════════════════════
# ANALYZE PHASE — Router vs TASK_ID similarity
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_analyze(args) -> None:
    """Router weight cosine similarity + cross-task accuracy matrix.

    Experiment: show that each task's TextNavRouter learned distinct
    navigation weights (different weight vectors → distinct task navigation).
    This addresses: "拿Router的Output去和TASK_ID算Similarity".

    The cross-task accuracy matrix uses the same SBO selection policy as
    eval (args.budget / args.step_size / args.redundancy_weight). Since
    --step-size defaults to 16, running analyze without extra flags now
    performs multi-step selection, not one-shot.
    """
    order = args.order
    task_order = TASK_ORDERS[order]
    print(f"[zeronav analyze] selection policy: budget={args.budget} "
          f"step_size={args.step_size} redundancy_weight={args.redundancy_weight}")

    bank_filename = args.skill_bank_in or f"skill_bank_{order}_f{args.fold}.pt"
    bank_path = os.path.join(args.out, "zeronav", bank_filename)
    print(f"[zeronav analyze] loading: {bank_path}")
    bank = ZeroNavSkillBank.load(bank_path)
    task_ids = bank.task_ids()

    # ── 1. Router weight cosine similarity matrix ────────────────────────────
    vecs = bank.weight_vectors()       # {task_id: [P]}
    print("\n── Router weight cosine similarity (task_i vs task_j weight vectors) ──")
    header = "       " + "  ".join(f"T{i}:{task_order[i][:5]}" for i in task_ids)
    print(header)
    cos_matrix: dict[int, dict[int, float]] = {}
    for i in task_ids:
        vi = F.normalize(vecs[i].unsqueeze(0), dim=-1)
        row: dict[int, float] = {}
        for j in task_ids:
            vj = F.normalize(vecs[j].unsqueeze(0), dim=-1)
            row[j] = round(float((vi @ vj.t()).item()), 4)
        cos_matrix[i] = row
        vals = "  ".join(f"{row[j]:6.3f}" for j in task_ids)
        print(f"  T{i}: {vals}")

    # ── 2. Cross-task accuracy matrix (router_i evaluated on task_j) ─────────
    if args.backbone_ckpt or not args.skip_acc_matrix:
        device = get_device("auto")
        cfg = load_qpmil_cfg(args.qpmil_config)
        if args.backbone_ckpt:
            backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
        else:
            backbone = build_backbone_fresh(cfg, task_order, device)

        loaders = build_loaders(cfg, task_order, args.fold)

        cfg_eval = ObserveConfig(
            budget=args.budget, step_size=args.step_size,
            redundancy_weight=args.redundancy_weight,
            normalize_base=True, redundancy_mode="maxsim",
        )

        print("\n── Cross-task accuracy matrix: acc[router_i][task_j] ──")
        print("  (diagonal = correct task skill; off-diagonal = skill mismatch)")
        header2 = "         " + "  ".join(f"Task{j}({task_order[j][:4]})" for j in task_ids)
        print(header2)
        acc_matrix: dict[int, dict[int, float]] = {}

        for i in task_ids:
            router_i = bank.build_router(i, device)
            f_txt = backbone.class_text_features().to(device)
            row_acc: dict[int, float] = {}
            for j in task_ids:
                shift = 2 * j
                slides = list(iter_slides(loaders[task_order[j]]["test"], shift, args.max_eval))
                if not slides:
                    row_acc[j] = -1.0
                    continue
                correct = 0
                obs = SequentialBudgetedObserver(cfg_eval)
                for feats, label in slides:
                    Z = backbone.encode_patches(feats).to(device)
                    base = router_i(Z, f_txt)
                    res = obs.observe(Z, base,
                                      lambda S: backbone.aggregate_and_predict(S)[0])
                    correct += int(int(res.logits.reshape(-1).argmax()) == int(label))
                row_acc[j] = round(correct / len(slides), 4)
            acc_matrix[i] = row_acc
            vals = "  ".join(f"{row_acc[j]:8.3f}" for j in task_ids)
            print(f"  R{i}({task_order[i][:4]}): {vals}")
    else:
        acc_matrix = {}

    # ── save ────────────────────────────────────────────────────────────────
    out_dir = os.path.join(args.out, "zeronav")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = os.path.join(out_dir, f"router_analysis_{order}_f{args.fold}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "order": order, "fold": args.fold,
            "task_names": {str(i): task_order[i] for i in task_ids},
            "weight_cosine_similarity": {
                str(i): {str(j): v for j, v in row.items()}
                for i, row in cos_matrix.items()
            },
            "cross_task_accuracy": {
                str(i): {str(j): v for j, v in row.items()}
                for i, row in acc_matrix.items()
            },
        }, f, indent=2)
    print(f"\n  ✓ saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# SMOKE TEST (Mac, random backbone)
# ════════════════════════════════════════════════════════════════════════════

def run_smoke(args) -> None:
    """Quick end-to-end smoke test on Mac (random backbone, no CONCH weights)."""
    print("[zeronav smoke] running with random backbone — no CONCH weights needed")
    args.backbone_ckpt = ""
    args.epochs = 1
    args.max_train = 6
    args.max_eval = 4
    args.top_k = 16
    args.lambdas = "0.0,0.5,1.0"
    args.budget = 16
    args.step_size = 4
    args.eval_tasks = "0,1"
    args.skip_acc_matrix = True

    print("\n── Phase 1: train ──")
    run_train(args)
    print("\n── Phase 2: eval ──")
    run_eval(args)
    print("\n── Phase 3: analyze (weight cosine only) ──")
    run_analyze(args)
    print("\n[zeronav smoke] ✓ all phases completed successfully")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ZeroNav: ZeroSlide-inspired backbone-agnostic navigation"
    )
    ap.add_argument("cmd", choices=["train", "eval", "analyze", "smoke"])
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="reverse")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", default="",
                    help=".pt checkpoint; empty=random backbone (smoke/debug)")
    # train options
    ap.add_argument("--epochs", type=int, default=5,
                    help="training epochs per task")
    ap.add_argument("--top-k", type=int, default=64,
                    help="top-K patches for soft aggregation during training")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-train", type=int, default=0,
                    help="max slides per task for training (0=all)")
    # eval options
    ap.add_argument("--budget", type=int, default=64,
                    help="total patch budget K (default 64)")
    ap.add_argument("--step-size", type=int, default=16,
                    help="patches per SBO round (default 16 → 4 rounds for K=64); "
                         "also used by analyze's cross-task matrix (default = multi-step, not one-shot)")
    ap.add_argument("--lambdas",
                    default=",".join(str(x) for x in ZERONAV_LAMBDAS),
                    help="comma-separated redundancy weights for λ sweep")
    ap.add_argument("--eval-tasks", default="",
                    help="comma-separated task indices (empty=all)")
    ap.add_argument("--max-eval", type=int, default=0,
                    help="max slides per task for eval (0=all)")
    ap.add_argument("--skill-bank-in", default="",
                    help="override skill bank filename (in outputs/zeronav/)")
    # analyze options
    ap.add_argument("--skip-acc-matrix", action="store_true",
                    help="analyze: skip cross-task accuracy (just do weight cosine)")
    ap.add_argument("--redundancy-weight", type=float, default=0.0,
                    help="analyze: lambda for cross-task matrix (0 = no diversity penalty)")
    ap.add_argument("--tag", type=str, default="",
                    help="analyze: suffix appended to the output json filename")
    # output
    ap.add_argument("--out", default="outputs",
                    help="base output directory (default: outputs/)")
    args = ap.parse_args()

    dispatch = {
        "train":   run_train,
        "eval":    run_eval,
        "analyze": run_analyze,
        "smoke":   run_smoke,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
