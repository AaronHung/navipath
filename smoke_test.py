"""不需 backbone / GPU 的形狀自測。執行: python smoke_test.py"""
import torch
from navipath_moe import (
    MicroRouterV0, MicroRouter, MacroRouter, fuse, top_k_select, ExpertBank,
    ExpertImportance, consolidate, snapshot_experts, l_sem, l_balance, l_route,
    get_device, setup_mps,
)

setup_mps(); print("device:", get_device())
N, D, E, C, Np = 3000, 512, 4, 6, 5
Z, f_txt, F_p = torch.randn(N, D), torch.randn(C, D), torch.randn(Np, D)

s0, sim0 = MicroRouterV0(D)(Z, f_txt, F_p)
assert s0.shape == (N,) and sim0.shape == (N,)

w_micro, score, sim = MicroRouter(D, E)(Z, f_txt, F_p)
w = fuse(MacroRouter(D, E)(Z), w_micro, 0.3)
assert w_micro.shape == (N, E) and w.shape == (N, E) and score.shape == (N,)

idx = top_k_select(score, 64); assert idx.shape[0] == 64
bank = ExpertBank(D, E, hidden=256)
assert bank(Z[idx], w[idx]).shape == (64, D)

imp = ExpertImportance(E); imp.observe(w); old = snapshot_experts(bank); imp.end_task()
m = consolidate(bank, imp, old); assert m.shape == (E,)

assert all(x.ndim == 0 and torch.isfinite(x)
           for x in (l_sem(score, sim), l_balance(w), l_route(w, w.clone())))

print("OK — NaviPath-MoE standalone components pass (MLP experts).")
print(f"  micro_v0 score {tuple(s0.shape)} | micro_v1 w {tuple(w_micro.shape)} | "
      f"fused {tuple(w.shape)} | top-64 {idx.shape[0]} | m_e {[round(x,3) for x in m.tolist()]}")
