"""M4 — MicroRouterV0 訓練 + go/no-go 評估。

MicroRouterV0：每個 patch → 純量 importance score → Top-K 選擇。
QPMIL backbone 完全凍結，只訓練 router（~130K params）。
訓練方式：用 router 選 Top-K patch → 送 QPMIL inference → 切片分類 loss 反傳到 router。

go/no-go 判準（任一綠燈）：
  router@64 明顯 > random@64（>2 pp）
  router@128 ≳ semantic@128

用法（先用 M1 checkpoint；若無用 --no-ckpt 隨機初始化 backbone）：
  python train_router_v0.py --order paper --fold 1 \\
      --backbone-ckpt outputs/qpmil_paper_fold1.pt \\
      --epochs 3 --top-k 64
  python train_router_v0.py --order paper --fold 1 \\
      --backbone-ckpt outputs/qpmil_paper_fold1.pt \\
      --epochs 1 --max-train 32 --max-eval 16          # quick smoke on CPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navipath_moe import get_device, setup_mps, MicroRouterV0, top_k_select
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from eval.patch_budget_eval import select_indices
from train_qpmil_runner import load_qpmil_cfg, build_loaders, TASK_ORDERS


# ── data helper ──────────────────────────────────────────────────────────────

def iter_slides(loader, shift, limit=0):
    n = 0
    for _idx, feats, label in loader:
        yield feats, int(label.view(-1)[0]) + shift
        n += 1
        if limit and n >= limit:
            break


# ── training ──────────────────────────────────────────────────────────────────

def train_router_one_task(router, backbone, loader, device, task_pos,
                          epochs, top_k, lr, max_train):
    """凍結 backbone，只訓練 router。

    梯度路徑：router 輸出 score[n] → softmax → soft weight Z_w（differentiable
    加權聚合）→ QPMIL 的 text-based logit → cross-entropy loss → 反傳回 router。
    評估時才用 hard top-K。
    """
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=1e-4)
    shift = 2 * task_pos
    f_txt_cached = backbone.class_text_features().to(device).detach()   # [C,512]
    router.train()
    for epoch in range(epochs):
        seen = total_loss = 0
        for feats, label in iter_slides(loader, shift, max_train):
            Z = backbone.encode_patches(feats).to(device)               # [n,512]
            F_p = backbone.prototype_features().to(device).detach()     # [M,512]

            score, _ = router(Z, f_txt_cached, F_p)                    # [n]  has grad

            # soft weighted aggregation（可微分）：取 top_k 後 softmax weight sum
            n = Z.shape[0]
            k = min(top_k, n) if top_k > 0 else n
            topk_score, topk_idx = torch.topk(score, k)                # [k]
            w = F.softmax(topk_score, dim=0).unsqueeze(-1)             # [k,1]
            Z_sel = Z[topk_idx]                                         # [k,512] detach OK；
            # Z_sel 不需要 grad；grad 流回 router 是透過 w，w 來自 score
            Z_w = (w * Z_sel.detach()).sum(0, keepdim=True)            # [1,512]
            Z_w = F.normalize(Z_w, dim=-1)

            # text-based logit（對齊 QPMIL 的 logit_scale * bag@f_txt）
            logit_scale = backbone.model.logit_scale.exp().detach()
            logits = logit_scale * (Z_w @ f_txt_cached.t())           # [1,C]

            gt = torch.tensor([label], device=device)
            loss = F.cross_entropy(logits, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); seen += 1
        avg = total_loss / max(seen, 1)
        print(f"    [router task_pos {task_pos}] epoch {epoch+1}/{epochs}"
              f"  loss={avg:.4f}  slides={seen}", flush=True)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_router_vs_heuristics(router, backbone, loader, device, task_pos,
                               budgets, max_eval):
    """比較 router / random / prototype / semantic 在各 K 下的 ACC。"""
    router.eval()
    shift = 2 * task_pos

    results = {m: {("All" if k == 0 else k): 0 for k in budgets}
               for m in ("router", "random", "prototype", "semantic")}
    counts = {m: {("All" if k == 0 else k): 0 for k in budgets}
              for m in ("router", "random", "prototype", "semantic")}

    f_txt = backbone.class_text_features().to(device)
    F_p = backbone.prototype_features().to(device)

    for feats, label in iter_slides(loader, shift, max_eval):
        Z = backbone.encode_patches(feats).to(device)
        gt = torch.tensor([label], device=device)

        # router score
        score, _ = router(Z, f_txt, F_p)

        for k in budgets:
            key = "All" if k == 0 else k
            n = Z.shape[0]
            # router
            idx_r = top_k_select(score, k)
            logits_r, _ = backbone.aggregate_and_predict(Z[idx_r])
            results["router"][key] += int((logits_r.argmax(-1) == gt).item())
            counts["router"][key] += 1
            # heuristics
            for m in ("random", "prototype", "semantic"):
                idx_h = select_indices(m, Z, f_txt, F_p, k)
                logits_h, _ = backbone.aggregate_and_predict(Z[idx_h])
                results[m][key] += int((logits_h.argmax(-1) == gt).item())
                counts[m][key] += 1

    for m in results:
        for k in results[m]:
            results[m][k] = round(results[m][k] / max(counts[m][k], 1), 4)
    return results


def print_table(results, budgets):
    cols = ["All" if k == 0 else k for k in budgets]
    header = f"{'method':12s}" + "".join(f"ACC@{str(c):>5s}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for m in ("router", "random", "prototype", "semantic"):
        row = f"{m:12s}" + "".join(f"{results[m][c]:>9.4f}" for c in cols)
        print(row)
    # go/no-go 判斷
    k64 = 64 if 64 in budgets else budgets[-1]
    k128 = 128 if 128 in budgets else budgets[-1]
    g64 = results["router"][k64] - results["random"][k64]
    g128 = results["router"][k128] - results["semantic"][k128]
    go = g64 > 0.02 or results["router"][k128] >= results["semantic"][k128] - 0.01
    print(f"\n[go/no-go] router@{k64} - random@{k64} = {g64:+.4f}  "
          f"router@{k128} - semantic@{k128} = {g128:+.4f}")
    print(f"[go/no-go] {'GO ✓' if go else 'NO-GO — 退安全稿'}", flush=True)
    return go


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--backbone-ckpt", default="",
                    help="M1 產出的 .pt；空白=隨機初始化 backbone（僅驗 pipeline）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=64,
                    help="訓練時送幾個 patch 進 QPMIL（固定預算）")
    ap.add_argument("--budgets", default="0,256,128,64,32")
    ap.add_argument("--tasks", type=int, default=0, help="0=跑所有任務；>0=前 N 個")
    ap.add_argument("--task-to-eval", type=int, default=-1,
                    help="go/no-go 評估哪個任務（-1=最後學完的任務）")
    ap.add_argument("--eval-tasks", default="",
                    help="逗號分隔要評估的 task index（如 \"-1,0\"）；空=用 --task-to-eval。"
                         "最後一個任務 -> router_v0_*.json；其餘 -> oldtask_budget_*.json")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(42); np.random.seed(42)
    print(f"[M4] device={device}  order={args.order}  fold={args.fold}", flush=True)

    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]
    if args.tasks:
        order = order[:args.tasks]
    budgets = tuple(int(x) for x in args.budgets.split(","))

    # backbone（凍結）
    if args.backbone_ckpt:
        print(f"[M4] backbone from ckpt: {args.backbone_ckpt}")
        backbone = build_backbone_from_ckpt(args.backbone_ckpt, device)
    else:
        print("[M4] backbone fresh (random init) — pipeline smoke only")
        backbone = build_backbone_fresh(cfg, order, device)

    for p in backbone.parameters():
        p.requires_grad_(False)

    # router（只有這個有 grad）
    router = MicroRouterV0(feat_dim=512, hidden=256).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"[M4] router params = {n_params:,}", flush=True)

    loaders = build_loaders(cfg, order, args.fold)

    # 逐任務訓練 router（continual，backbone 不動）
    for t, ds in enumerate(order):
        print(f"\n[M4] task {t+1}/{len(order)} {ds}", flush=True)
        train_router_one_task(router, backbone, loaders[ds]["train"],
                              device, t, args.epochs, args.top_k,
                              args.lr, args.max_train)

    # 解析要評估的 task 清單（--eval-tasks 優先，否則用單一 --task-to-eval）
    if args.eval_tasks.strip():
        eval_list = [int(x) for x in args.eval_tasks.split(",")]
    else:
        eval_list = [args.task_to_eval]

    os.makedirs(args.out, exist_ok=True)
    last_pos = len(order) - 1

    # 存 router checkpoint（防覆蓋：canonical 已存在就不動）
    ckpt_path = os.path.join(args.out, f"router_v0_{args.order}_fold{args.fold}.pt")
    if os.path.exists(ckpt_path):
        print(f"[skip] exists: {ckpt_path}")
    else:
        torch.save(router.state_dict(), ckpt_path)
        print(f"[M4] saved {ckpt_path}")

    # 逐一評估指定的 task；最後一任務寫 canonical router_v0，其餘寫 oldtask_budget
    for et in eval_list:
        eval_t = et % len(order)
        eval_ds = order[eval_t]
        if eval_t == last_pos:
            json_path = os.path.join(args.out, f"router_v0_{args.order}_fold{args.fold}.json")
        else:
            json_path = os.path.join(
                args.out, f"oldtask_budget_{args.order}_f{args.fold}_task{eval_t}.json")
        if os.path.exists(json_path):
            print(f"[skip] exists: {json_path}")
            continue
        print(f"\n[M4] eval on {eval_ds} test set (task_index={eval_t})", flush=True)
        results = eval_router_vs_heuristics(
            router, backbone, loaders[eval_ds]["test"],
            device, eval_t, budgets, args.max_eval)
        go = print_table(results, budgets)
        with open(json_path, "w") as f:
            json.dump({"order": args.order, "fold": args.fold, "eval_task": eval_ds,
                       "task_index": eval_t, "budgets": list(budgets),
                       "results": results, "go": go}, f, indent=2)
        print(f"[M4] saved {json_path}")


if __name__ == "__main__":
    main()
