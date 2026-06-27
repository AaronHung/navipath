"""SPEC-01 — Continual Navigation Layer (CNL) agent wrapper.

把既有 MicroRouterV0 navigation policy 包成具 Navigation Skill Memory (NSM) 與
Context Gate 的 continual WSI navigation agent。不重寫既有訓練/評估主流程
(train_router_v0.py)，只在其上加一層 agent 介面。

設計原則（同 routers.py）：本檔不依賴 QPMIL 內部、也不 import train_router_v0，
可單獨 import 與單元測試（device-agnostic）。backbone 以鴨子型別注入，只需提供
class_text_features() / prototype_features() / aggregate_and_predict()。

對應 spec：specs/features/SPEC-01-continual-agent.md
"""
from __future__ import annotations

import copy
from typing import Optional

import torch

from .routers import MicroRouterV0, top_k_select


class NavigationSkillBank:
    """task_id -> MicroRouterV0 state_dict 的可持久化記憶（NSM）。

    把 train_router_v0.py 的 in-memory `router_states` 抽成可存取/持久化的 skill bank，
    每個 learned task 一個 navigation skill。
    """

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        self.feat_dim = feat_dim
        self.hidden = hidden
        self._skills: dict[int, dict] = {}

    def add_skill(self, task_id: int, router_or_state) -> None:
        """router_or_state 可為 nn.Module（取其 state_dict）或 state_dict 本身。"""
        state = router_or_state.state_dict() if hasattr(router_or_state, "state_dict") \
            else router_or_state
        self._skills[int(task_id)] = copy.deepcopy(state)

    def has(self, task_id: int) -> bool:
        return int(task_id) in self._skills

    def get_state(self, task_id: int) -> dict:
        return self._skills[int(task_id)]

    def task_ids(self) -> list[int]:
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def build_router(self, task_id: int, device=None) -> MicroRouterV0:
        """建一個 MicroRouterV0，載入指定 task 的 skill，設為 eval。"""
        router = MicroRouterV0(feat_dim=self.feat_dim, hidden=self.hidden)
        router.load_state_dict(self._skills[int(task_id)])
        if device is not None:
            router = router.to(device)
        router.eval()
        return router

    def save(self, path: str) -> None:
        torch.save({"feat_dim": self.feat_dim, "hidden": self.hidden,
                    "skills": self._skills}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "NavigationSkillBank":
        blob = torch.load(path, map_location=map_location)
        bank = cls(blob.get("feat_dim", 512), blob.get("hidden", 256))
        bank._skills = blob["skills"]
        return bank


class ContextGate:
    """決定推論時用哪個 navigation skill。

    本次只啟用 oracle（已知 task id = upper bound）。task-free infer（用 QPMIL
    MaxPooling query 對 prototype-match / full-patch logits 推 task）列 future
    work（ADR-0002 / ADR-0003）。
    """

    def __init__(self, mode: str = "oracle"):
        if mode not in ("oracle", "infer"):
            raise ValueError(f"unknown gate mode: {mode}")
        self.mode = mode

    def select(self, *, task_id: Optional[int] = None, Z=None, backbone=None) -> int:
        if self.mode == "oracle":
            if task_id is None:
                raise ValueError("oracle gate 需要 task_id")
            return int(task_id)
        return self.infer(Z, backbone)

    def infer(self, Z, backbone) -> int:
        raise NotImplementedError(
            "task-free context gate 為 future work（ADR-0003）：用 QPMIL "
            "MaxPooling query 對 prototype-match 頻率 / full-patch logits 推 task。")


class ContinualWSINavigationAgent:
    """CNL：frozen backbone + Navigation Skill Memory + Context Gate。

    navigate(Z, budget, task_id) -> 選出的 patch index；
    predict(Z, budget, task_id) -> (WSI logits, 選出的 index)。
    """

    def __init__(self, backbone, skill_bank: NavigationSkillBank,
                 gate: Optional[ContextGate] = None, device=None):
        self.backbone = backbone
        self.skill_bank = skill_bank
        self.gate = gate or ContextGate("oracle")
        self.device = device
        self._router_cache: dict[int, MicroRouterV0] = {}

    def _router_for(self, task_id: int) -> MicroRouterV0:
        if task_id not in self._router_cache:
            self._router_cache[task_id] = self.skill_bank.build_router(task_id, self.device)
        return self._router_cache[task_id]

    @torch.no_grad()
    def navigate(self, Z, budget: int, *, task_id: Optional[int] = None):
        skill_id = self.gate.select(task_id=task_id, Z=Z, backbone=self.backbone)
        router = self._router_for(skill_id)
        f_txt = self.backbone.class_text_features()
        F_p = self.backbone.prototype_features()
        if self.device is not None:
            f_txt = f_txt.to(self.device)
            F_p = F_p.to(self.device)
        score, _ = router(Z, f_txt, F_p)
        return top_k_select(score, budget)

    @torch.no_grad()
    def predict(self, Z, budget: int, *, task_id: Optional[int] = None):
        idx = self.navigate(Z, budget, task_id=task_id)
        logits, _ = self.backbone.aggregate_and_predict(Z[idx])
        return logits, idx
