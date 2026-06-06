"""Patch-budget 導航評估（研究計劃 主表 B / Milestone 2）— 不需訓練 MoE，最早出表。

對每張 WSI：用不同方法選 Top-K patch，只用這些 patch 跑 QPMIL inference，報 ACC@K。
比較 4 種 selection：random / prototype-sim / semantic-sim / router(之後加)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _topk_idx(score, k, n, device):
    if k <= 0 or k >= n:
        return torch.arange(n, device=device)
    return torch.topk(score, k).indices


def select_indices(method, Z, f_txt, F_p, k, router=None):
    """回傳被選中的 patch index。Z:[n,512] f_txt:[C,512] F_p:[N,512]。"""
    n = Z.shape[0]; dev = Z.device
    if method == "random":
        if k <= 0 or k >= n:
            return torch.arange(n, device=dev)
        return torch.randperm(n, device=dev)[:k]
    z = F.normalize(Z, dim=-1)
    if method == "prototype":
        score = (z @ F.normalize(F_p, dim=-1).t()).amax(-1)
    elif method == "semantic":
        score = (z @ F.normalize(f_txt, dim=-1).t()).amax(-1)
    elif method == "router":
        assert router is not None
        score = router(Z, f_txt, F_p)[1] if hasattr(router, "experts") else router(Z, f_txt, F_p)[0]
    else:
        raise ValueError(method)
    return _topk_idx(score, k, n, dev)


@torch.no_grad()
def patch_budget_eval(backbone, loader, methods=("random", "prototype", "semantic"),
                      budgets=(0, 256, 128, 64, 32), router=None):
    """回傳 {method: {K: ACC}}。backbone 需提供 prototype_features / class_text_features /
    aggregate_and_predict（同 model.py 的 hook）。"""
    results = {m: {} for m in methods}
    counts = {m: {} for m in methods}
    f_txt = backbone.class_text_features()
    F_p = backbone.prototype_features()
    for m in methods:
        for k in budgets:
            corr = tot = 0
            for slide, label in loader:
                Z = backbone.encode_patches(slide)
                idx = select_indices(m, Z, f_txt, F_p, k, router)
                logits, _ = backbone.aggregate_and_predict(Z[idx], f_txt)
                corr += int((logits.argmax(-1) == label).sum()); tot += 1
            key = "All" if k == 0 else k
            results[m][key] = round(corr / max(tot, 1), 4)
    return results
