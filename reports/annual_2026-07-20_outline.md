# NaviPath-CL 年度報告大綱（2026-07-20）

> 大綱（SPEC-05）。**保密**：上位計畫以 **North Star** 代稱。引用雙月報告與 7/15 前進展，避免重抄。

## 1. 年度摘要
- 從 selector 診斷 pivot 至 **NaviPath-CL**（North Star Phase-0 CL 原型）。
- 主成果：navigation policy 的 continual forgetting 與 **Navigation Skill Memory** 的恢復證據。

## 2. 研究問題與定位
- budgeted/agentic WSI 下的 observation policy continual learning。
- North Star 對齊（Phase-0 → 後續 task-free / consolidation / move-zoom / RLHF）。

## 3. 方法
- CNL（backbone-agnostic）+ NSM；QPMIL-VL 作 frozen backbone/weak signal。

## 4. 成果
- Mechanism-selection 證據（雙月報告 §6；Fig_mechanism / Fig_budget_curve）。
- 論文進度（`paper/NaviPath-CL_draft_outline.md` → LaTeX；7/15 老師 review 狀態）。
- **TODO（7/15 前補入）**：paper-order 對稱、agent 真數字重現、可選 task-free/consolidation pilot。

## 5. 後續規劃（North Star roadmap）
- Phase-1 task-free gate + consolidation；Phase-2 move/zoom + 醫師軌跡；North Star RLHF。

## 6. 附錄
- SDD 記錄：`specs/`（README/ADR/SPEC/WORKLOG）；實驗執行：`outputs/PROGRESS.md`。
- 雙月報告：`reports/bimonthly_2026-07-03.md`。

---

*待 7/15–7/20 依論文與補跑結果填實數據與圖。*
