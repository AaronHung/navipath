"""Lightweight MLP experts for NaviPath-MoE (研究計劃 §5.4，已從 LoRA 改為 MLP).

為什麼是 MLP 而不是 LoRA：你們不是 fine-tune 大 transformer，而是在 precomputed
CONCH 512 維 patch features 上加幾個小 expert。最簡單可跑的 expert 就是作用在
512 維上的殘差小 MLP：e_j(z) = z + MLP_j(z)。最後一層初始化為 0，訓練前等於恆等，
不破壞 QPMIL backbone。

設計約束（§5.4 修正二）：expert 數量 E 要少（先 4，再試 6）。ESCA 只有 158 張，
太多 expert 會 collapse / overfit。E 是第一級超參數，寧少勿多。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPExpert(nn.Module):
    """殘差小 MLP：e(z) = z + W2(act(W1 z))，最後一層零初始化（起步等於恆等）。"""

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, feat_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ExpertBank(nn.Module):
    """E 個小 MLP expert，依路由權重做加權混合（per-patch soft MoE）。

    forward(x, w):
      x : [n, D] patch 特徵
      w : [n, E] 每個 patch 對各 expert 的權重（已 softmax）
      回傳 [n, D]：sum_e w[:, e] * expert_e(x)
    """

    def __init__(self, feat_dim: int = 512, num_experts: int = 4, hidden: int = 256):
        super().__init__()
        self.experts = nn.ModuleList(
            [MLPExpert(feat_dim, hidden) for _ in range(num_experts)]
        )
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        outs = torch.stack([e(x) for e in self.experts], dim=1)   # [n, E, D]
        return (w.unsqueeze(-1) * outs).sum(dim=1)                # [n, D]
