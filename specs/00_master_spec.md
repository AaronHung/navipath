# 00 — Master Spec（技術索引）

> 單一入口。敘事/基調的真相在 [`../STORYLINE.md`](../STORYLINE.md)，本檔只做技術索引與 scope/roadmap 的可追蹤摘要，不複製敘事。

---

## 1. 一句話

NaviPath-CL = **North Star**（physician-like WSI navigation agent 長期願景）的 **Phase-0 CL 原型**：在無醫師軌跡前，用 WSI label / QPMIL-VL prototype-text signal 作 weak supervision，學一個 budgeted WSI navigation policy，並研究該 policy 在 continual learning 下的遺忘與記憶機制。

詳見 [`../STORYLINE.md`](../STORYLINE.md)。

## 2. 命名（固定）

- 整體方法：**NaviPath-CL**
- 貢獻層：**Continual Navigation Layer (CNL)**
- 核心模組：**Navigation Skill Memory (NSM)**
- 上位願景代稱：**North Star**

決策見 [ADR-0004](decisions/ADR-0004-naming-cnl-nsm-northstar.md)。

## 3. 框架（backbone-agnostic）

CNL 定義在抽象 backbone 介面上；QPMIL-VL 為一個 prompt/prototype-based instance。介面：

- `encode(WSI) -> Z`：patch features（QPMIL：frozen CONCH 512-d）。
- `predict(subset) -> logits`：stable predictor（QPMIL：prototype-agg + CFE）。
- `task_query(WSI) -> q`（optional）：domain descriptor（QPMIL：MaxPooling query / prototype-match）。

實作對應（已驗介面，見 worklog 引用）：

- backbone：`navipath_moe/qpmil_adapter.py::QPMILBackbone`（4 hooks：`encode_patches` / `class_text_features` / `prototype_features` / `aggregate_and_predict`）+ `build_backbone_from_ckpt`。
- navigation policy：`navipath_moe/routers.py::MicroRouterV0`（patch→score→Top-K）。
- NSM / CL update：本次新建 `navipath_moe/continual_agent.py`（SPEC-01）。
- consolidation 工具：`navipath_moe/consolidate.py`（dual-importance，未來 parameter-merging pilot）。

## 4. Scope（本次 Phase-0）

In-scope：budgeted patch selection、navigation skill memory、oracle context gate（upper bound）、mechanism probe（shared / EWC / per-task）。

Out-of-scope（future）：task-free gate、真 RL/PPO、zoom/move/多尺度、醫師軌跡、order-aware reward、完整 RLHF、parameter-merging solved、compute-saving 宣稱。

完整邊界與 do-not-claim 見 [`../STORYLINE.md`](../STORYLINE.md) §10。

## 5. Roadmap / 里程碑

對應主計畫 `.cursor/plans/navipath-cl_sdd_plan_11f257cb.plan.md`：

- M0 6/27 SDD 骨架（本批）
- M1 6/28 [SPEC-01](features/SPEC-01-continual-agent.md) continual_agent.py
- M2 6/29 [SPEC-02](features/SPEC-02-evidence-and-symmetry-runs.md) 證據彙整
- M3 6/30 ★ agent end-to-end + mechanism-selection 表 + 決策樹
- M4 7/1 [SPEC-03](features/SPEC-03-core-figures.md) 5 張圖
- M5 7/2 [SPEC-04](features/SPEC-04-report-0703.md) 凍結 + 雙月報告
- M6 7/3 報告
- M7 7/3–7/15 [SPEC-05](features/SPEC-05-paper-draft.md) 論文初稿；7/20 年度報告

## 6. 證據現況（數字已驗，見 outputs/）

- shared policy：recent GO（ESCA@64 0.956 / lung 0.922）；old NO-GO（reverse old ESCA@64 0.133 ≪ random 0.8）。
- per-task skill bank：reverse old ESCA fold2 0.933 / fold3 1.0（恢復）。
- EWC：reverse old ESCA ~0.40（不足）。
- consolidation（parameter-merging）：尚無 navigation 版數字 → ongoing。
