"""NaviPath-MoE model wrapper (研究計劃 §5).

把 QPMIL-VL backbone 與我們的 macro/micro router + MLP expert bank 接起來。
支援分階段開關：
  use_experts=False, use_macro=False -> 等價「micro router 只做 patch 選擇」(v0/budget 階段)
  use_experts=True                   -> 接 MLP MoE，做特徵轉換 (v1)
  use_macro=True                     -> 加 macro router 融合

QPMIL 整合 4 個 hook（給坤倫，見 README）：
  backbone.encode_patches(slide)           -> Z   [n,512]
  backbone.prototype_features()            -> F_p [N,512]
  backbone.class_text_features()           -> f_txt [C,512]
  backbone.aggregate_and_predict(Z, f_txt) -> (logits, {"L_C","L_M","L_S"})
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .routers import MicroRouter, MacroRouter, fuse, top_k_select
from .experts import ExpertBank


class NaviPathMoE(nn.Module):
    def __init__(self, backbone, feat_dim: int = 512, num_experts: int = 4,
                 expert_hidden: int = 256, beta: float = 0.3,
                 use_experts: bool = True, use_macro: bool = False):
        super().__init__()
        self.backbone = backbone
        self.micro = MicroRouter(feat_dim, num_experts, expert_hidden)
        self.macro = MacroRouter(feat_dim, num_experts, expert_hidden) if use_macro else None
        self.experts = ExpertBank(feat_dim, num_experts, expert_hidden) if use_experts else None
        self.beta = beta
        self.use_experts = use_experts
        self.use_macro = use_macro

    def forward(self, slide, top_k: int = 0):
        Z = self.backbone.encode_patches(slide)        # [n,512]
        F_p = self.backbone.prototype_features()       # [N,512]
        f_txt = self.backbone.class_text_features()    # [C,512]

        w_micro, score, sim_txt = self.micro(Z, f_txt, F_p)   # [n,E],[n],[n]
        w = w_micro
        if self.use_macro:
            w = fuse(self.macro(Z), w_micro, self.beta)

        idx = top_k_select(score, top_k)               # agent action: 選 Top-K patch
        Z_sel = Z[idx]
        if self.use_experts:
            Z_sel = self.experts(Z_sel, w[idx])        # MLP MoE 特徵轉換

        logits, qpmil_losses = self.backbone.aggregate_and_predict(Z_sel, f_txt)
        return {
            "logits": logits, "qpmil_losses": qpmil_losses,
            "w": w, "score": score, "sim_txt": sim_txt, "selected_idx": idx,
        }
