"""Smoke tests for N1 — Sequential Budgeted Observation.

只驗管路（正確性），不驗成績：形狀對、budget 被遵守、trace 順序一致、
sequential vs one-shot 行為不同、信心早停可觸發、agent end-to-end 能 navigate->predict。
CPU + 合成輸入，不需 GPU / QPMIL / 真資料。

跑法：PYTHONPATH=. python tests/test_sequential_observation.py
"""
from __future__ import annotations

import torch

from navipath_moe.routers import MicroRouterV0, top_k_select
from navipath_moe.continual_agent import NavigationSkillBank, ContextGate
from navipath_moe.sequential_observation import (
    ObserveConfig, SequentialBudgetedObserver, ContinualSequentialNavigationAgent,
)

D, C, P = 16, 3, 4


class DummyBackbone:
    """frozen backbone 的最小鴨子型別：text/proto 特徵 + subset 預測。"""

    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.f_txt = torch.randn(C, D, generator=g)
        self.F_p = torch.randn(P, D, generator=g)
        self.W = torch.randn(D, C, generator=g)

    def class_text_features(self):
        return self.f_txt

    def prototype_features(self):
        return self.F_p

    def aggregate_and_predict(self, Zsub):
        return Zsub.mean(0) @ self.W, None


def _make(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, D, generator=g)
    base_score = torch.randn(n, generator=g)
    bb = DummyBackbone(seed)
    predict_fn = lambda S: bb.aggregate_and_predict(S)[0]  # noqa: E731
    return Z, base_score, predict_fn


def test_budget_and_trace_consistency():
    Z, score, pf = _make()
    obs = SequentialBudgetedObserver(ObserveConfig(budget=24, step_size=8, redundancy_weight=0.5))
    res = obs.observe(Z, score, pf)
    assert res.selected.numel() == 24, res.selected.numel()
    assert len(set(res.selected.tolist())) == 24, "no duplicates"
    flat = [i for r in res.trace for i in r]
    assert flat == res.selected.tolist(), "trace order == selected order"
    assert res.n_rounds == 3, res.n_rounds
    assert 0.0 <= res.confidence <= 1.0
    assert res.logits is not None and res.logits.shape == (C,)


def test_one_shot_equals_topk():
    Z, score, pf = _make()
    obs = SequentialBudgetedObserver(ObserveConfig(budget=16, step_size=16, redundancy_weight=0.0))
    res = obs.observe(Z, score, pf)
    expected = top_k_select(score, 16)
    assert set(res.selected.tolist()) == set(expected.tolist()), "one-shot == plain Top-K"
    assert res.n_rounds == 1


def test_sequential_differs_from_one_shot():
    Z, score, pf = _make()
    one = SequentialBudgetedObserver(
        ObserveConfig(budget=24, step_size=24, redundancy_weight=0.0)).observe(Z, score, pf)
    seq = SequentialBudgetedObserver(
        ObserveConfig(budget=24, step_size=6, redundancy_weight=1.0)).observe(Z, score, pf)
    assert one.selected.tolist() != seq.selected.tolist(), "redundancy-aware seq 應不同於 one-shot"
    assert seq.selected.numel() == 24


def test_early_stop_triggers():
    Z, score, pf = _make()
    # 門檻設很低 -> 第一輪就該停
    res = SequentialBudgetedObserver(
        ObserveConfig(budget=32, step_size=8, confidence_threshold=0.0)).observe(Z, score, pf)
    assert res.stopped_early
    assert res.selected.numel() == 8, res.selected.numel()


def test_budget_caps_at_n():
    Z, score, pf = _make(n=10)
    res = SequentialBudgetedObserver(ObserveConfig(budget=64, step_size=4)).observe(Z, score, pf)
    assert res.selected.numel() == 10, "budget 不超過 patch 數"


def test_agent_end_to_end():
    Z, _, _ = _make(n=50)
    bb = DummyBackbone(0)
    bank = NavigationSkillBank(feat_dim=D, hidden=32)
    bank.add_skill(0, MicroRouterV0(feat_dim=D, hidden=32))   # task 0 的 navigation skill
    agent = ContinualSequentialNavigationAgent(
        bb, bank, ContextGate("oracle"),
        ObserveConfig(budget=20, step_size=5, redundancy_weight=0.5))
    res = agent.observe(Z, task_id=0)
    assert res.selected.numel() == 20
    logits, idx = agent.predict(Z, task_id=0)
    assert logits.shape == (C,) and idx.numel() == 20
    assert res.state.summary()["n_seen"] == 20


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n[PASS] {len(tests)} sequential-observation smoke tests")


if __name__ == "__main__":
    _run_all()
