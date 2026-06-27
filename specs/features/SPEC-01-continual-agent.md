# SPEC-01 — Continual Navigation Layer agent (`navipath_moe/continual_agent.py`)

- Status: Done (2026-06-27)
- Milestone: M1 (6/28)
- Related: ADR-0002, ADR-0003, ADR-0004

## 1. 目標

把既有 `MicroRouterV0` navigation policy 包成具 **Navigation Skill Memory (NSM)** 與 **Context Gate** 的 continual WSI navigation agent。**不重寫**既有訓練/評估主流程（`train_router_v0.py`），只在其上加一層 agent 介面。

模組必須 **backbone-agnostic**（鴨子型別）：只要 backbone 提供 `class_text_features()` / `prototype_features()` / `aggregate_and_predict()`，agent 即可運作（對齊 CNL 的 backbone interface）。本檔不依賴 `train_router_v0.py`，可單獨 import 與單元測試（沿用 `routers.py` 的設計原則）。

## 2. 介面

```python
class NavigationSkillBank:
    def __init__(self, feat_dim=512, hidden=256)
    def add_skill(self, task_id: int, router_or_state) -> None   # 接受 nn.Module 或 state_dict
    def has(self, task_id) -> bool
    def get_state(self, task_id) -> dict
    def task_ids(self) -> list[int]
    def build_router(self, task_id, device=None) -> MicroRouterV0  # 建 router 並 load state, eval()
    def save(self, path) -> None
    @classmethod
    def load(cls, path, map_location="cpu") -> "NavigationSkillBank"

class ContextGate:
    def __init__(self, mode="oracle")                # "oracle"（本次） | "infer"（future）
    def select(self, *, task_id=None, Z=None, backbone=None) -> int
    def infer(self, Z, backbone) -> int              # raise NotImplementedError（future, ADR-0003）

class ContinualWSINavigationAgent:
    def __init__(self, backbone, skill_bank, gate=None, device=None)
    def navigate(self, Z, budget, *, task_id=None) -> torch.Tensor   # idx [k]
    def predict(self, Z, budget, *, task_id=None) -> tuple[torch.Tensor, torch.Tensor]  # (logits [1,C], idx)
```

資料流：`gate.select(task_id)` → `skill_bank.build_router(skill_id)` → `score = router(Z, f_txt, F_p)` → `top_k_select(score, budget)` → `backbone.aggregate_and_predict(Z[idx])`。

## 3. 依賴（已確認介面）

- `navipath_moe.routers.MicroRouterV0`（`forward(Z,f_txt,F_p)->(score,sim)`）、`top_k_select(score,k)`。
- backbone 介面：`navipath_moe.qpmil_adapter.QPMILBackbone.{class_text_features, prototype_features, aggregate_and_predict}`（real backbone 由 `build_backbone_from_ckpt` 提供，agent 不直接依賴它）。
- 不在模組頂層 import `train_router_v0`（避免與 `navipath_moe` 循環）；需重用訓練時於 M3 的 root 腳本延遲 import。

## 4. 驗收標準 (acceptance criteria)

- [x] `python -c "import navipath_moe.continual_agent"` 成功。
- [x] `tests/test_continual_agent.py` 全綠（5 passed；device-agnostic，stub backbone + 隨機 N=3000,D=512）：
  - `navigate(Z, 64, task_id=0)` 回傳 idx，`idx.shape[0]==64`；budget 0/≥n 時回全選。
  - `predict(...)` 回傳 `logits.shape==(1,C)`、`torch.isfinite(logits).all()`。
  - `NavigationSkillBank.save/load` round-trip 後 `build_router` 輸出與原 router 一致。
  - `ContextGate("oracle").select()` 無 task_id → `ValueError`；`mode="infer"` → `NotImplementedError`。
- [x] `ruff check` All checks passed；全套件 `pytest tests/ -q` 11 passed（無回歸）。

## 5. 不做的事（本 SPEC 範圍外）

- 不實作 task-free `infer`（future）。
- 不在此跑真實 backbone / 真資料的數字重現（屬 M3 / 另一支 root 腳本）。
- 不改 `train_router_v0.py`、`routers.py`、`qpmil_adapter.py`。

## 6. Changelog

- 2026-06-27：建立 SPEC（M1 動工前）。
