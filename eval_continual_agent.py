"""SPEC-06 — Continual WSI navigation agent: end-to-end eval.

用 ContinualWSINavigationAgent（oracle gate + NavigationSkillBank）跑 end-to-end，
重現既有 per-task 數字（證明 agent 包裝正確 = navigate→predict 等價於 per-task eval）。

重用既有主流程：train_router_v0.train_router_one_task / iter_slides，不重寫。
backbone 凍結，逐任務訓練單一 router，每學完一個 task 把 snapshot 存進 skill bank。

用法（RunPod 重現）：
  python eval_continual_agent.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt \
      --order reverse --fold 2 --eval-task 0 --epochs 5 \
      --skill-bank-out outputs/skill_bank_reverse_f2.pt

  # Mac pipeline smoke（fresh backbone，小資料）
  python eval_continual_agent.py --order reverse --fold 1 --eval-task 0 \
      --epochs 1 --max-train 8 --max-eval 4 --out outputs/_smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from navipath_moe import get_device, setup_mps, MicroRouterV0
from navipath_moe.continual_agent import (
    NavigationSkillBank, ContextGate, ContinualWSINavigationAgent,
)
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS
from train_router_v0 import train_router_one_task, iter_slides


@torch.no_grad()
def eval_agent_on_task(agent, backbone, loader, device, eval_task, budgets, max_eval):
    """用 agent.predict 在指定 (old) task 上算各 budget 的 acc。"""
    shift = 2 * eval_task
    results = {("All" if k == 0 else k): 0 for k in budgets}
    counts = {("All" if k == 0 else k): 0 for k in budgets}
    for feats, label in iter_slides(loader, shift, max_eval):
        Z = backbone.encode_patches(feats).to(device)
        gt = torch.tensor([label], device=device)
        for k in budgets:
            key = "All" if k == 0 else k
            logits, _idx = agent.predict(Z, k, task_id=eval_task)
            results[key] += int((logits.argmax(-1) == gt).item())
            counts[key] += 1
    for key in results:
        results[key] = round(results[key] / max(counts[key], 1), 4)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="reverse")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", default="",
                    help="M1 .pt；空白=隨機初始化 backbone（pipeline smoke）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--budgets", default="0,256,128,64,32")
    ap.add_argument("--eval-task", type=int, default=0,
                    help="要評估的舊任務 index（oracle gate 用此 id 選 skill）")
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
    print(f"[agent] device={device} order={args.order} fold={args.fold} "
          f"eval_task={args.eval_task}", flush=True)

    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    budgets = tuple(int(x) for x in args.budgets.split(","))

    if args.backbone_ckpt:
        print(f"[agent] backbone from ckpt: {args.backbone_ckpt}")
        backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    else:
        print("[agent] backbone fresh (random init) — pipeline smoke only")
        backbone = build_backbone_fresh(cfg, order, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    loaders = build_loaders(cfg, order, args.fold)

    # 逐任務訓練單一 router；每學完一個 task 把 snapshot 存進 skill bank（= per-task skill）
    skill_bank = NavigationSkillBank(feat_dim=512, hidden=256)
    router = MicroRouterV0(feat_dim=512, hidden=256).to(device)
    for t, ds in enumerate(order):
        print(f"\n[agent] train task {t+1}/{len(order)} {ds}", flush=True)
        train_router_one_task(router, backbone, loaders[ds]["train"],
                              device, t, args.epochs, args.top_k,
                              args.lr, args.max_train, ewc=None)
        skill_bank.add_skill(t, copy.deepcopy(router.state_dict()))

    if args.skill_bank_out:
        os.makedirs(os.path.dirname(args.skill_bank_out) or ".", exist_ok=True)
        skill_bank.save(args.skill_bank_out)
        print(f"[agent] saved skill bank -> {args.skill_bank_out} "
              f"(tasks={skill_bank.task_ids()})")

    # 用 oracle gate 的 agent 評估指定舊任務
    agent = ContinualWSINavigationAgent(backbone, skill_bank,
                                        ContextGate("oracle"), device=device)
    eval_ds = order[args.eval_task]
    print(f"\n[agent] eval (oracle gate) on {eval_ds} test (task_index={args.eval_task})",
          flush=True)
    results = eval_agent_on_task(agent, backbone, loaders[eval_ds]["test"],
                                 device, args.eval_task, budgets, args.max_eval)
    print(f"[agent] router-via-agent acc: {results}", flush=True)

    # 一致性檢查：對照既有 pertask json（若存在）
    ref_path = os.path.join(
        args.out, f"oldtask_budget_{args.order}_f{args.fold}_task{args.eval_task}__pertask.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)["results"]["router"]
        print("[agent] consistency vs per-task json (budget: agent vs ref):")
        for k in budgets:
            key = "All" if k == 0 else k
            a = results[key]
            r = ref.get(str(key) if key != "All" else "All")
            flag = "OK" if (r is not None and abs(a - r) <= 0.01) else "DIFF"
            print(f"          @{key:>3}: {a} vs {r}  [{flag}]")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(
        args.out, f"agent_oldtask_{args.order}_f{args.fold}_task{args.eval_task}.json")
    with open(out_path, "w") as f:
        json.dump({"order": args.order, "fold": args.fold, "eval_task": eval_ds,
                   "task_index": args.eval_task, "gate": "oracle",
                   "budgets": list(budgets), "results": {"agent_router": results}}, f, indent=2)
    print(f"[agent] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
