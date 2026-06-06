"""Shape / sanity tests（對方 Test 1）。執行: pytest tests/ -q  或  python tests/test_shapes.py"""
import torch
from navipath_moe import (
    MicroRouterV0, MicroRouter, MacroRouter, fuse, top_k_select,
    ExpertBank, ExpertImportance, consolidate, snapshot_experts,
    l_sem, l_balance, l_route,
)

N, D, E, C, Np = 3000, 512, 4, 6, 5          # 用真實量級 patch 數 (~3000)
Z = torch.randn(N, D); f_txt = torch.randn(C, D); F_p = torch.randn(Np, D)


def test_micro_v0():
    score, sim = MicroRouterV0(D)(Z, f_txt, F_p)
    assert score.shape == (N,) and sim.shape == (N,)

def test_micro_v1():
    w, score, sim = MicroRouter(D, E)(Z, f_txt, F_p)
    assert w.shape == (N, E) and score.shape == (N,)
    assert torch.allclose(w.sum(-1), torch.ones(N), atol=1e-4)

def test_macro_and_fuse():
    w_macro = MacroRouter(D, E)(Z)
    w_micro = MicroRouter(D, E)(Z, f_txt, F_p)[0]
    w = fuse(w_macro, w_micro, 0.3)
    assert w_macro.shape == (E,) and w.shape == (N, E)

def test_topk_and_experts():
    score = torch.randn(N)
    idx = top_k_select(score, 64); assert idx.shape[0] == 64
    bank = ExpertBank(D, E)
    w = torch.softmax(torch.randn(64, E), -1)
    assert bank(Z[idx], w).shape == (64, D)

def test_losses():
    score = torch.randn(N); sim = torch.randn(N)
    w = torch.softmax(torch.randn(N, E), -1)
    for L in (l_sem(score, sim), l_balance(w), l_route(w, w.clone())):
        assert L.ndim == 0 and torch.isfinite(L)

def test_consolidation():
    bank = ExpertBank(D, E); imp = ExpertImportance(E)
    imp.observe(torch.softmax(torch.randn(N, E), -1)); imp.end_task()
    m = consolidate(bank, imp, snapshot_experts(bank))
    assert m.shape == (E,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
    print("OK — all shape tests pass.")
