"""Agentic macro/micro routing for NaviPath-MoE (研究計劃 §5.2/5.3/5.4).

關鍵修正（對方）：
- micro router 輸入用「固定維度的 summary 統計」而非 concat 原始 per-class 相似度，
  因為 class 數 C 會隨任務增加，concat 會變維。summary = [z_i ; 4 個固定統計] -> 516 維。
- micro router 分兩版：
    v0 (score 模式)：輸出每個 patch 的純量重要度，用於 Top-K 選擇（先驗證能不能贏
                      random/semantic/prototype heuristic，再決定要不要做 MoE）。
    v1 (expert 模式)：輸出 [n, E] expert 權重，接 ExpertBank。
- macro router 最後才加。

本檔不依賴 QPMIL 內部，可單獨 import 與單元測試（device-agnostic）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def summary_feats(Z: torch.Tensor, f_txt: torch.Tensor,
                  F_p: torch.Tensor) -> torch.Tensor:
    """每個 patch 的固定維度 summary 統計（與 class 數 C、prototype 數 N 無關）。

    Z:[n,D]  f_txt:[C,D] 類別文字特徵  F_p:[N,D] 選中的 prototype 特徵
    回傳 [n, 4]: [max_text_sim, entropy(text_sim 分布), max_proto_sim, mean_proto_sim]
    """
    z = F.normalize(Z, dim=-1)
    t = F.normalize(f_txt, dim=-1)
    p = F.normalize(F_p, dim=-1)
    txt = z @ t.t()                                  # [n, C]
    pro = z @ p.t()                                  # [n, N]
    txt_prob = F.softmax(txt, dim=-1)                # [n, C]
    txt_ent = -(txt_prob * (txt_prob + 1e-9).log()).sum(-1, keepdim=True)  # [n,1]
    feats = torch.cat([
        txt.amax(-1, keepdim=True),                  # max text sim
        txt_ent,                                     # entropy of text sim
        pro.amax(-1, keepdim=True),                  # max proto sim
        pro.mean(-1, keepdim=True),                  # mean proto sim
    ], dim=-1)                                       # [n, 4]
    return feats, txt.amax(-1)                       # 同時回傳 sim_txt_max[n] 給 L_sem


class MicroRouterV0(nn.Module):
    """v0：每個 patch -> 純量重要度 score（用於 Top-K 選擇，先不做 expert）。"""

    def __init__(self, feat_dim: int = 512, hidden: int = 256, n_summary: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + n_summary, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, Z, f_txt, F_p):
        feats, sim_txt_max = summary_feats(Z, f_txt, F_p)      # [n,4], [n]
        u = torch.cat([Z, feats], dim=-1)                       # [n, D+4]
        score = self.mlp(u).squeeze(-1)                         # [n]
        return score, sim_txt_max


class MicroRouter(nn.Module):
    """v1：每個 patch -> [n, E] expert 權重；score = max_e w（給 Top-K 用）。"""

    def __init__(self, feat_dim: int = 512, num_experts: int = 4,
                 hidden: int = 256, n_summary: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + n_summary, hidden), nn.GELU(),
            nn.Linear(hidden, num_experts)
        )

    def forward(self, Z, f_txt, F_p):
        feats, sim_txt_max = summary_feats(Z, f_txt, F_p)
        u = torch.cat([Z, feats], dim=-1)
        w_micro = F.softmax(self.mlp(u), dim=-1)                # [n, E]
        score = w_micro.amax(dim=-1)                            # [n]
        return w_micro, score, sim_txt_max


class MacroRouter(nn.Module):
    """整張切片 pooled 特徵 -> [E] 策略 expert 權重（最後才加）。"""

    def __init__(self, feat_dim: int = 512, num_experts: int = 4, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, num_experts)
        )

    def forward(self, Z):
        return F.softmax(self.mlp(Z.mean(0)), dim=-1)           # [E]


def fuse(w_macro, w_micro, beta: float = 0.3):
    """w_i = beta * w_macro + (1-beta) * w_micro_i。w_macro:[E], w_micro:[n,E]->[n,E]。"""
    return beta * w_macro.unsqueeze(0) + (1.0 - beta) * w_micro


def top_k_select(score, k: int):
    """選分數最高的 K 個 patch index；k<=0 或 k>=n 表示全選。"""
    n = score.shape[0]
    if k <= 0 or k >= n:
        return torch.arange(n, device=score.device)
    return torch.topk(score, k).indices
