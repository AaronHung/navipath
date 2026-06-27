# SPEC-03 — Core figures（5 張核心圖）

- Status: Done (2026-06-27)
- Milestone: M4 (7/1)
- Related: STORYLINE §3–§5, ADR-0003

## 1. 目標

產 7/3 報告 + 論文用的 5 張核心圖，對齊新敘事（CNL/NSM/budgeted navigation），數據圖用真實 pilot 數字（reverse old ESCA，跨 3 fold mean）。

## 2. 5 張圖（檔名 + 來源 + 內容）

1. **Fig1_arch**（`tools/draw_arch.py`，改名）：CNL 架構——frozen diagnostic backbone（QPMIL-VL instance）+ navigation policy（CNL，continually trained）+ NSM（per-task / EWC）。移除舊「Forgetting=0 / compute-saving」宣稱措辭。
2. **Fig_problem**（`tools/make_figures.py`）：問題圖——budgeted WSI observation + 任務串流下 navigation policy 的 catastrophic forgetting。
3. **Fig_mechanism**（`make_figures.py`，數據）：mechanism bar @budget 64——shared / EWC / per-task(NSM) / best-heuristic，GO 綠 / NO-GO 紅。
4. **Fig_budget_curve**（`make_figures.py`，數據）：各 budget（32→All）三機制 + best-heuristic 曲線（EWC negative + NSM 恢復同圖呈現）。
5. **Fig_roadmap**（`make_figures.py`）：Phase-0 prototype → North Star physician-like navigation agent 的階段圖。

## 3. 依賴

- matplotlib（已用於 `draw_arch.py`）。
- 數據來源：`tools/mechanism_table.load_records(outputs)`（重用，不重抄數字）。

## 4. 驗收標準

- [x] `tools/draw_arch.py` 改名後 ruff pass、產 `outputs/figs/Fig1_arch.{pdf,png}`、`FigS1_arch`，移除 "Forgetting=0"/compute-saving 措辭，改用 CNL/NSM/navigation policy。
- [x] `tools/make_figures.py` ruff pass、產 Fig_problem / Fig_mechanism / Fig_budget_curve / Fig_roadmap 的 {png,pdf}。
- [x] Fig_mechanism @64 條形值 = shared 0.333 / ewc 0.400 / pertask 0.933 / best-heur 0.844（視覺驗證一致；per-task 綠/GO，shared+EWC 紅/NO-GO）。

## 5. 不做的事

- 不追求論文級美術微調（Aaron 後續微調）；先確保數字正確、敘事對齊。

## 6. Changelog

- 2026-06-27：建立 SPEC（M4 動工）。
