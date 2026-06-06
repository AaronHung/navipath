"""NaviPath-MoE: Agentic Macro/Micro Routing for Continual WSI Classification."""

from .device import get_device, setup_mps
from .routers import (
    MicroRouterV0, MicroRouter, MacroRouter, fuse, top_k_select, summary_feats,
)
from .experts import ExpertBank, MLPExpert
from .consolidate import ExpertImportance, consolidate, snapshot_experts
from .losses import l_sem, l_balance, l_route, total_loss
from .model import NaviPathMoE

__all__ = [
    "get_device", "setup_mps",
    "MicroRouterV0", "MicroRouter", "MacroRouter", "fuse", "top_k_select", "summary_feats",
    "ExpertBank", "MLPExpert",
    "ExpertImportance", "consolidate", "snapshot_experts",
    "l_sem", "l_balance", "l_route", "total_loss",
    "NaviPathMoE",
]
