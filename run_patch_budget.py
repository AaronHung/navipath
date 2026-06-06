"""M3 — Patch-budget 導航評估（第一張表，不訓練）。

對某個任務的 test set，用 random / prototype-sim / semantic-sim 三種選法各選 Top-K patch，
只用這些 patch 跑 QPMIL 推論，報 ACC@{All,256,128,64,32}。

用法（用 M1 訓練後的 checkpoint 最有意義）：
  python run_patch_budget.py --ckpt outputs/qpmil_paper_fold1.pt --order paper --task-index -1
  python run_patch_budget.py --order paper --fold 1            # 無 ckpt：隨機初始化（僅驗流程）

判準：K 小時 prototype/semantic 是否明顯贏 random；random 的 ACC 應隨 K 單調上升（sanity）。
"""
from __future__ import annotations

import argparse

import torch

from navipath_moe import get_device, setup_mps
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS
from eval.patch_budget_eval import patch_budget_eval


def wrap_loader(loader, task_pos, limit=0):
    """把 QPMIL (idx,feats,label) loader 轉成 (slide, global_label)。"""
    shift = 2 * task_pos
    n = 0
    for _idx, feats, label in loader:
        yield feats, int(label.view(-1)[0]) + shift
        n += 1
        if limit and n >= limit:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--task-index", type=int, default=-1, help="評估第幾個任務的 test set（-1=最後）")
    ap.add_argument("--max-eval", type=int, default=0, help="最多評估切片數（0=全部）")
    ap.add_argument("--budgets", default="0,256,128,64,32", help="K 列表，0=All")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    task_pos = args.task_index % len(order)
    budgets = tuple(int(x) for x in args.budgets.split(","))

    if args.ckpt:
        print(f"[budget] backbone from ckpt: {args.ckpt}")
        backbone = build_backbone_from_ckpt(args.ckpt, device)
        order = backbone.model.cfg.get("dataset_names", order) if hasattr(backbone.model, "cfg") else order
    else:
        print(f"[budget] backbone fresh (random init) — 數字僅供驗流程，非真實效能")
        backbone = build_backbone_fresh(cfg, order, device)

    ds = TASK_ORDERS[args.order][task_pos]
    print(f"[budget] order={args.order} eval task[{task_pos}]={ds} test set, K={budgets}")
    loaders = build_loaders(cfg, TASK_ORDERS[args.order], args.fold)
    test_loader = list(wrap_loader(loaders[ds]["test"], task_pos, args.max_eval))
    print(f"[budget] n_slides={len(test_loader)}")

    results = patch_budget_eval(backbone, test_loader,
                                methods=("random", "prototype", "semantic"), budgets=budgets)

    # 印表
    cols = ["All" if k == 0 else k for k in budgets]
    header = f"{'selection':12s}" + "".join(f"ACC@{str(c):>5s}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for m in ("random", "prototype", "semantic"):
        row = f"{m:12s}" + "".join(f"{results[m][c]:>9.4f}" for c in cols)
        print(row)

    # sanity：random 是否隨 K 單調上升
    rand = [results["random"][c] for c in cols][::-1]  # 從 @32 -> @All
    mono = all(rand[i] <= rand[i + 1] + 1e-9 for i in range(len(rand) - 1))
    print(f"\n[sanity] random ACC monotonic in K (@32<=...<=@All): {mono}  -> {rand}")


if __name__ == "__main__":
    main()
