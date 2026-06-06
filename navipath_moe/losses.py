"""NaviPath-MoE loss terms (研究計劃 §5.6，L_sem 已改為維度乾淨版).

總損失:
  L = L_C + lambda*L_M + beta_S*L_S        # QPMIL 原損失（在 backbone 算）
      + gamma*L_sem + eta*L_bal + xi*L_route

關鍵修正（對方修正 2）：L_sem 定義在「patch 選擇分布」上，避免 expert 維度 [E] 與
class 維度 [C] 不匹配。兩邊都是 patch 上的分布（長度 n），KL 才合理。
全部 buffer-free：L_route 用凍結 teacher 在「當前資料」上算，不存任何舊切片。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l_sem(patch_score: torch.Tensor, sim_txt_max: torch.Tensor,
          tau: float = 1.0) -> torch.Tensor:
    """語義錨損失 KL(q || pi)，兩者皆為 patch 上的分布（長度 n）。

    patch_score : [n] micro router 給每個 patch 的重要性分數（v0 的純量輸出）
    sim_txt_max : [n] 每個 patch 對類別文字的 max cosine 相似度
    pi = softmax(patch_score)  router 認為的 patch 重要度分布
    q  = softmax(sim_txt_max)  CONCH 語義認為的 patch 重要度分布（target）
    """
    log_pi = F.log_softmax(patch_score / tau, dim=0)     # [n]
    q = F.softmax(sim_txt_max / tau, dim=0)              # [n]
    return F.kl_div(log_pi, q, reduction="batchmean")    # KL(q || pi)


def l_balance(router_w: torch.Tensor) -> torch.Tensor:
    """MoE 負載平衡（Switch-Transformer 式），防止 expert collapse。router_w:[n,E]。"""
    E = router_w.shape[-1]
    P = router_w.mean(dim=0)                              # [E] 平均機率
    hard = F.one_hot(router_w.argmax(dim=-1), num_classes=E).float()
    f = hard.mean(dim=0)                                  # [E] top-1 比例
    return E * torch.sum(f * P)


def l_route(student_w: torch.Tensor, teacher_w: torch.Tensor) -> torch.Tensor:
    """路由穩定損失（AKD 思想，buffer-free）。teacher 在當前資料上算、不含梯度。"""
    t = teacher_w.detach()
    return F.kl_div(F.log_softmax(student_w, dim=-1),
                    F.softmax(t, dim=-1), reduction="batchmean")


def total_loss(qpmil_losses: dict, sem, bal, route, weights: dict) -> torch.Tensor:
    L = (qpmil_losses["L_C"]
         + weights["lambda"] * qpmil_losses["L_M"]
         + weights["beta_S"] * qpmil_losses["L_S"]
         + weights["gamma"] * sem
         + weights["eta"] * bal)
    if route is not None and weights.get("xi", 0.0) > 0:
        L = L + weights["xi"] * route
    return L
