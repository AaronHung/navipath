"""Weight-averaged router baseline (teacher Q9).

Motivation (reviewer/advisor request): between the two extremes
  - naive continual router (single router overwritten each task) = LOWER bound
  - per-task NSM (one router per task, oracle-selected)          = UPPER bound
we add an intermediate baseline: element-wise AVERAGE of the per-task router
weights into a single router,
        phi_bar = (1/T) * sum_t phi^(t),
and evaluate this single averaged router on every task under the identical
protocol as eval_sequential_observation.py.

This is inference-only and cheap: the per-task weights already live in the
saved NSM skill bank (outputs/skill_bank_*.pt). No retraining.

Usage (RunPod, reverse order, 3 folds):
  for FOLD in 1 2 3; do
    python eval_weight_avg_baseline.py \
      --backbone-ckpt outputs/qpmil_reverse_fold${FOLD}.pt \
      --order reverse --fold ${FOLD} --eval-tasks 0,1,2,3 \
      --skill-bank-in outputs/skill_bank_reverse_f${FOLD}.pt \
      --budgets 0,128,64,32,16 --step-size 16 \
      --out outputs/weight_avg
  done
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from navipath_moe import get_device, setup_mps
from navipath_moe.continual_agent import NavigationSkillBank
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS
from eval_sequential_observation import eval_grid


def average_state_dicts(states: list[dict]) -> dict:
    """Element-wise mean of a list of state_dicts (identical keys/shapes)."""
    avg = copy.deepcopy(states[0])
    for k in avg:
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


def build_weight_avg_bank(src_bank: NavigationSkillBank,
                          eval_tasks: list[int]) -> NavigationSkillBank:
    """Return a bank whose skill for EVERY eval task is the averaged router."""
    all_states = [src_bank.get_state(t) for t in src_bank.task_ids()]
    phi_bar = average_state_dicts(all_states)
    avg_bank = NavigationSkillBank(feat_dim=src_bank.feat_dim, hidden=src_bank.hidden)
    for t in eval_tasks:
        avg_bank.add_skill(t, copy.deepcopy(phi_bar))
    return avg_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="reverse")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", required=True)
    ap.add_argument("--skill-bank-in", required=True,
                    help="已存的 NSM skill bank（.pt）；取其各 task 權重做平均")
    ap.add_argument("--budgets", default="0,128,64,32,16", help="0=All")
    ap.add_argument("--step-size", type=int, default=16)
    ap.add_argument("--redundancy", type=float, default=0.0,
                    help="baseline 預設 one-shot（λ=0）")
    ap.add_argument("--normalize-base", type=lambda x: x.lower() != "false", default=True)
    ap.add_argument("--redundancy-mode", choices=["maxsim", "centroid"], default="maxsim")
    ap.add_argument("--eval-tasks", default="0,1,2,3")
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--out", default="outputs/weight_avg")
    ap.add_argument("--repo-remap", default="",
                    help="若 ckpt 內嵌 RunPod 絕對路徑，於本機以此 repo root 重映射 "
                         "(remap '/workspace/src/navipath' -> this path). RunPod 上留空。")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    budgets = tuple(int(x) for x in args.budgets.split(","))
    eval_list = [int(x) % len(order) for x in args.eval_tasks.split(",")]

    from navipath_moe.qpmil_adapter import build_backbone_from_ckpt
    print(f"[wavg] backbone <- {args.backbone_ckpt}", flush=True)
    remap = ("/workspace/src/navipath", args.repo_remap) if args.repo_remap else None
    backbone = build_backbone_from_ckpt(args.backbone_ckpt, device, path_remap=remap)
    for p in backbone.parameters():
        p.requires_grad_(False)

    loaders = build_loaders(cfg, order, args.fold)

    src_bank = NavigationSkillBank.load(args.skill_bank_in, map_location=device)
    print(f"[wavg] loaded skill bank tasks={src_bank.task_ids()}; "
          f"averaging into a single router", flush=True)
    avg_bank = build_weight_avg_bank(src_bank, eval_list)

    os.makedirs(args.out, exist_ok=True)
    for eval_t in eval_list:
        eval_ds = order[eval_t]
        # For the weight-avg baseline both "nsm" and "nonsm" slots use the same
        # averaged router; we report it under key "wavg_*" for clarity.
        print(f"\n[wavg] eval on {eval_ds} (task_index={eval_t})", flush=True)
        results, n_slides = eval_grid(
            backbone, avg_bank, avg_bank,
            loaders[eval_ds]["test"], device, eval_t,
            budgets, args.step_size, args.redundancy, args.max_eval,
            policy_mode="router",
            normalize_base=args.normalize_base,
            redundancy_mode=args.redundancy_mode)
        # rename nsm_* -> wavg_* (nonsm_* duplicates, drop)
        renamed = {}
        for m, d in results.items():
            if m.startswith("nsm_"):
                renamed[m.replace("nsm_", "wavg_")] = d
        for m, d in renamed.items():
            print(f"[wavg]   {m:>14}: {d}", flush=True)

        out_path = os.path.join(
            args.out, f"wavg_{args.order}_f{args.fold}_task{eval_t}.json")
        with open(out_path, "w") as f:
            json.dump({
                "order": args.order, "fold": args.fold, "eval_task": eval_ds,
                "task_index": eval_t, "baseline": "weight_averaged_router",
                "src_skill_bank": args.skill_bank_in,
                "budgets": list(budgets), "step_size": args.step_size,
                "redundancy": args.redundancy,
                "normalize_base": args.normalize_base,
                "redundancy_mode": args.redundancy_mode,
                "n_eval_slides": n_slides,
                "results": renamed,
            }, f, indent=2)
        print(f"[wavg] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
