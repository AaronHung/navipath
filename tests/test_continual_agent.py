"""SPEC-01 smoke / shape tests for the Continual Navigation Layer agent.

device-agnostic；用 stub backbone（鴨子型別）+ 隨機 tensor，不需 QPMIL/CONCH。
執行: pytest tests/test_continual_agent.py -q  或  python tests/test_continual_agent.py
"""
import os
import tempfile

import torch

from navipath_moe import (
    MicroRouterV0, NavigationSkillBank, ContextGate, ContinualWSINavigationAgent,
)

N, D, C, M = 3000, 512, 6, 5
Z = torch.randn(N, D)


class _StubBackbone:
    """提供 CNL backbone interface 的最小替身（隨機輸出，僅驗 shape/流程）。"""

    def __init__(self, C=C, M=M, D=D):
        self.C, self.M, self.D = C, M, D

    def class_text_features(self):
        return torch.randn(self.C, self.D)

    def prototype_features(self):
        return torch.randn(self.M, self.D)

    def aggregate_and_predict(self, Z_sub, f_txt=None, no_grad=True):
        return torch.randn(1, self.C), {}


def _make_agent(mode="oracle"):
    bank = NavigationSkillBank(feat_dim=D, hidden=256)
    bank.add_skill(0, MicroRouterV0(feat_dim=D, hidden=256))
    bank.add_skill(3, MicroRouterV0(feat_dim=D, hidden=256))
    agent = ContinualWSINavigationAgent(_StubBackbone(), bank, ContextGate(mode))
    return agent, bank


def test_navigate_budget():
    agent, _ = _make_agent()
    idx = agent.navigate(Z, 64, task_id=0)
    assert idx.shape[0] == 64
    # budget 0 或 >= n -> 全選
    assert agent.navigate(Z, 0, task_id=0).shape[0] == N
    assert agent.navigate(Z, N + 10, task_id=3).shape[0] == N


def test_predict_shape():
    agent, _ = _make_agent()
    logits, idx = agent.predict(Z, 64, task_id=3)
    assert logits.shape == (1, C)
    assert torch.isfinite(logits).all()
    assert idx.shape[0] == 64


def test_skill_bank_save_load():
    agent, bank = _make_agent()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "skill_bank_test.pt")
    bank.save(p)
    bank2 = NavigationSkillBank.load(p)
    assert bank2.task_ids() == [0, 3]
    # round-trip：同一 Z 下兩個 bank 的 router score 一致
    r1 = bank.build_router(0)
    r2 = bank2.build_router(0)
    f_txt, F_p = torch.randn(C, D), torch.randn(M, D)
    s1, _ = r1(Z, f_txt, F_p)
    s2, _ = r2(Z, f_txt, F_p)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_oracle_requires_task_id():
    agent, _ = _make_agent()
    try:
        agent.navigate(Z, 64)  # 沒給 task_id
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_infer_gate_not_implemented():
    agent, _ = _make_agent(mode="infer")
    try:
        agent.navigate(Z, 64)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("OK — continual agent smoke tests pass.")
