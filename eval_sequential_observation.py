"""N2 — Sequential Budgeted Observation: end-to-end eval driver.

在 frozen QPMIL backbone + 真資料上，跑 N2 三組實驗的格點：
  1. Budget 效率：acc@K vs acc@All（recent / eval task）。
  2. CL 保留：old-task acc@K，有 NSM（per-task skill）vs 無 NSM（最終連續訓練的單一 router）。
  3. Agentic：sequential vs one-shot（同 budget）。

沿用既有主流程（train_qpmil_runner / train_router_v0），不重寫。
backbone 凍結；逐任務訓練單一 router：
  - 每學完一個 task 把 snapshot 存進 skill bank（= per-task NSM skill）。
  - 迴圈結束時的 router 狀態 = 無 NSM 的連續訓練 policy（會忘舊任務）。

用法：
  # Mac pipeline smoke（fresh backbone、小資料；只驗管路）
  python eval_sequential_observation.py --order reverse --fold 1 --eval-task 0 \
      --epochs 1 --max-train 8 --max-eval 4 --budgets 0,16,8 --step-size 4 --out outputs/_smoke

  # RunPod（真數字）
  python eval_sequential_observation.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt \
      --order reverse --fold 2 --eval-task 0 --epochs 5 \
      --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
      --skill-bank-out outputs/skill_bank_reverse_f2.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from navipath_moe import get_device, setup_mps, MicroRouterV0
from navipath_moe.continual_agent import NavigationSkillBank, ContextGate
from navipath_moe.sequential_observation import (
    ObserveConfig, ContinualSequentialNavigationAgent,
)
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS
from train_router_v0 import train_router_one_task, iter_slides


def _make_agent(backbone, bank, *, step_size, redundancy, device):
    """建一個 agent；mode 由 ObserveConfig 控制（per-budget 重設 budget）。"""
    cfg = ObserveConfig(budget=64, step_size=step_size, redundancy_weight=redundancy)
    return ContinualSequentialNavigationAgent(backbone, bank, ContextGate("oracle"),
                                              cfg, device=device)


@torch.no_grad()
def eval_grid(backbone, nsm_bank, nonsm_bank, loader, device, eval_task,
              budgets, step_size, redundancy, max_eval):
    """回傳 4 個 mode 的 {budget: acc}：{nsm_seq, nsm_oneshot, nonsm_seq, nonsm_oneshot}。"""
    shift = 2 * eval_task
    modes = {
        "nsm_seq":      (nsm_bank,   step_size, redundancy),
        "nsm_oneshot":  (nsm_bank,   10 ** 9,   0.0),
        "nonsm_seq":    (nonsm_bank, step_size, redundancy),
        "nonsm_oneshot": (nonsm_bank, 10 ** 9,  0.0),
    }
    agents = {m: _make_agent(backbone, bank, step_size=ss, redundancy=rw, device=device)
              for m, (bank, ss, rw) in modes.items()}

    correct = {m: {("All" if k == 0 else k): 0 for k in budgets} for m in modes}
    n_slides = 0
    for feats, label in iter_slides(loader, shift, max_eval):
        Z = backbone.encode_patches(feats).to(device)
        gt = int(label)
        n_slides += 1
        for m, agent in agents.items():
            for k in budgets:
                key = "All" if k == 0 else k
                agent.observer.cfg.budget = k
                logits, _idx = agent.predict(Z, task_id=eval_task)
                pred = int(logits.reshape(-1).argmax().item())
                correct[m][key] += int(pred == gt)
    out = {m: {key: round(c / max(n_slides, 1), 4) for key, c in d.items()}
           for m, d in correct.items()}
    return out, n_slides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="reverse")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", default="",
                    help="M1 .pt；空白=隨機初始化 backbone（pipeline smoke）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=64, help="router 訓練時的 Top-K")
    ap.add_argument("--budgets", default="0,128,64,32,16", help="0=All")
    ap.add_argument("--step-size", type=int, default=16, help="sequential 每輪看幾個")
    ap.add_argument("--redundancy", type=float, default=0.5, help="sequential 冗餘懲罰權重")
    ap.add_argument("--eval-task", type=int, default=0,
                    help="要評估的（舊）任務 index；oracle gate 用此 id 選 skill")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--skill-bank-out", default="")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"[seqobs] device={device} order={args.order} fold={args.fold} "
          f"eval_task={args.eval_task} step={args.step_size} gamma={args.redundancy}",
          flush=True)

    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    budgets = tuple(int(x) for x in args.budgets.split(","))

    if args.backbone_ckpt:
        print(f"[seqobs] backbone from ckpt: {args.backbone_ckpt}")
        backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    else:
        print("[seqobs] backbone fresh (random init) — pipeline smoke only")
        backbone = build_backbone_fresh(cfg, order, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    loaders = build_loaders(cfg, order, args.fold)

    # 逐任務訓練單一 router：snapshot -> NSM；最終狀態 -> 無 NSM policy
    nsm_bank = NavigationSkillBank(feat_dim=512, hidden=256)
    router = MicroRouterV0(feat_dim=512, hidden=256).to(device)
    for t, ds in enumerate(order):
        print(f"\n[seqobs] train task {t+1}/{len(order)} {ds}", flush=True)
        train_router_one_task(router, backbone, loaders[ds]["train"],
                              device, t, args.epochs, args.top_k,
                              args.lr, args.max_train, ewc=None)
        nsm_bank.add_skill(t, copy.deepcopy(router.state_dict()))

    # 無 NSM：用「學完所有任務後的最終 router」對舊任務導覽（預期會忘）
    nonsm_bank = NavigationSkillBank(feat_dim=512, hidden=256)
    nonsm_bank.add_skill(args.eval_task, copy.deepcopy(router.state_dict()))

    if args.skill_bank_out:
        os.makedirs(os.path.dirname(args.skill_bank_out) or ".", exist_ok=True)
        nsm_bank.save(args.skill_bank_out)
        print(f"[seqobs] saved skill bank -> {args.skill_bank_out} (tasks={nsm_bank.task_ids()})")

    eval_ds = order[args.eval_task]
    print(f"\n[seqobs] eval grid on {eval_ds} test (task_index={args.eval_task})", flush=True)
    results, n_slides = eval_grid(backbone, nsm_bank, nonsm_bank,
                                  loaders[eval_ds]["test"], device, args.eval_task,
                                  budgets, args.step_size, args.redundancy, args.max_eval)
    for m, d in results.items():
        print(f"[seqobs]   {m:>14}: {d}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(
        args.out, f"seqobs_{args.order}_f{args.fold}_task{args.eval_task}.json")
    with open(out_path, "w") as f:
        json.dump({
            "order": args.order, "fold": args.fold, "eval_task": eval_ds,
            "task_index": args.eval_task, "gate": "oracle",
            "budgets": list(budgets), "step_size": args.step_size,
            "redundancy": args.redundancy, "n_eval_slides": n_slides,
            "results": results,
        }, f, indent=2)
    print(f"[seqobs] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
