# specs/ — NaviPath-CL SDD 記錄體系

> 這是我們的 **spec-driven development (SDD)** 工作區與契約。
> 原則：**spec-first、決策可追、過程留痕**。寫任何 code / 跑任何實驗前，先有 spec 與驗收標準；做完即回寫結果與筆記。

---

## 0. 三類文件 + 與既有檔的分工

| 路徑 | 角色 | 變動規則 |
|---|---|---|
| `specs/00_master_spec.md` | 技術索引：問題 / 框架 / scope / roadmap 的單一入口（指回 `STORYLINE.md`，不複製敘事） | 隨主軸演進更新 |
| `specs/decisions/ADR-*.md` | **決策記錄**（Architecture Decision Record）：每個重要選擇一條 | **append-only**，翻案開新 ADR 標 supersedes |
| `specs/features/SPEC-*.md` | **可交付物 spec**：目標 / 介面 / 依賴 / 驗收標準 / 不做的事 | 動工前先寫，實作中可改但要記 changelog |
| `specs/worklog/WORKLOG.md` | **開發過程**：每段 output / 數字 / smoke / 踩雷 / commit | **append-only**，日期區塊 |

既有檔（不搬進 specs/，交叉引用）：

- `STORYLINE.md` = 對外敘事/基調的真相。
- `SESSION_CONTEXT.md` = 換 session 接手文件。
- `COLLAB_PLAYBOOK.md` = 合作規矩（device / tmux / git / 測試 / 成本）。
- `outputs/PROGRESS.md` = **實驗執行**紀錄（RunPod 跑了什麼、產出哪些 json）。

> 分界：`WORKLOG.md` 記「開發過程（code/設計/smoke）」；`outputs/PROGRESS.md` 記「實驗執行（跑批/結果 json）」。兩者交叉引用，不重抄。

---

## 1. SDD 工作流程（每個 milestone 都照這個走）

1. **SPEC**：在 `features/SPEC-xx` 寫清楚 目標 / 介面 (signatures) / 依賴 / 驗收標準 (acceptance criteria) / 明確不做的事。
2. **(必要時) ADR**：若過程中做了影響架構或敘事的選擇，補一條 ADR。
3. **實作 / 執行**：小步 patch；附 shape/smoke test（見 `COLLAB_PLAYBOOK.md` §3）。
4. **驗收**：對照 SPEC 的 acceptance criteria 逐項打勾。
5. **回寫**：在 `WORKLOG.md` append 日期區塊（做了什麼、數字、smoke 結果、commit、下一步）。
6. **commit**：訊息對應 SPEC/ADR 編號（例 `feat(SPEC-01): continual agent skill bank + oracle gate`）。

失敗即止：任一步失敗就停、貼 error、提最小修補（`COLLAB_PLAYBOOK.md` §0.5）。不捏造實驗數字。

---

## 2. 索引

### Master
- [00_master_spec.md](00_master_spec.md)

### Decisions (ADR)
- [ADR-0001 pivot to NaviPath-CL](decisions/ADR-0001-pivot-to-navipath-cl.md)
- [ADR-0002 QPMIL-VL as backbone, not replay](decisions/ADR-0002-qpmil-as-backbone-not-replay.md)
- [ADR-0003 mechanism-selection framing](decisions/ADR-0003-mechanism-selection-framing.md)
- [ADR-0004 naming: CNL / NSM / North Star](decisions/ADR-0004-naming-cnl-nsm-northstar.md)
- [ADR-0005 SDD workflow & records](decisions/ADR-0005-sdd-workflow-and-records.md)

### Features (SPEC)
- [SPEC-01 continual agent](features/SPEC-01-continual-agent.md)
- [SPEC-02 evidence & symmetry runs](features/SPEC-02-evidence-and-symmetry-runs.md)
- [SPEC-03 core figures](features/SPEC-03-core-figures.md)
- [SPEC-04 report 0703](features/SPEC-04-report-0703.md)
- [SPEC-05 paper draft](features/SPEC-05-paper-draft.md)
- [SPEC-06 continual agent end-to-end eval](features/SPEC-06-continual-agent-eval.md)

### Worklog
- [worklog/WORKLOG.md](worklog/WORKLOG.md)

### Plan
- 主計畫：`.cursor/plans/navipath-cl_sdd_plan_11f257cb.plan.md`（不在此 commit；唯讀參考）。

---

## 3. 保密

上位計畫一律以 **North Star** 代稱（= physician-like WSI navigation agent 的長期願景）。任何 specs/ 文件不得出現計畫名稱、單位、主持人、頁碼等可識別資訊。
