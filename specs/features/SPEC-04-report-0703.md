# SPEC-04 — Bimonthly / 7-3 report

- Status: In progress
- Milestone: M5 (7/2) 寫稿 + 凍結；M6 (7/3) 定稿呈現
- Related: STORYLINE（全部）, SPEC-02/03/06

## 1. 目標

產 7/3 雙月報告稿：呈現 pivot 後的架構（CNL/NSM）、Phase-0 pilot 證據（mechanism-selection）、與 pivot plan。**不假裝 paper 已完成**。對外以 North Star 代稱上位計畫。

## 2. 交付物

`reports/bimonthly_2026-07-03.md`，章節：
1. 一頁摘要（pivot + 主發現 + Phase-0 定位）。
2. 背景與 pivot（舊 selector → NaviPath-CL；為何）。
3. 問題定位（budgeted WSI observation + navigation policy 的 continual forgetting）。
4. North Star 對齊（Phase-0 prototype；不洩漏可識別資訊）。
5. 方法（CNL + NSM + QPMIL-VL 作 frozen backbone/weak signal；backbone-agnostic）。
6. Pilot 證據（mechanism-selection 表 + 決策樹 + 圖 Fig_mechanism/Fig_budget_curve）。
7. 結論與 pivot plan（future：task-free gate、consolidation、move/zoom、RLHF）。
8. 誠實邊界（do-not-claim；哪些待 RunPod/未做）。

## 3. 凍結結果 (frozen snapshot)

報告引用的結果在 M5 凍結，不再加新實驗：
- 數據：`outputs/MECHANISM_SELECTION.md`、`outputs/RESULTS_SUMMARY.md`。
- 圖：`outputs/figs/{Fig1_arch,Fig_problem,Fig_mechanism,Fig_budget_curve,Fig_roadmap}`。
- 核心數字：reverse old ESCA@64 = shared 0.333 / EWC 0.400 / per-task NSM 0.933 / best-heur 0.844。

## 4. 驗收標準

- [x] `reports/bimonthly_2026-07-03.md` 八章齊備、引用 Fig1/problem/mechanism/budget_curve/roadmap 與核心數字、North Star 代稱無可識別資訊。
- [x] §8 do-not-claim 明列未完成項（paper-order 對稱、agent 真數字重現、task-free/consolidation 為 future）+ frozen snapshot 宣告。

## 5. 不做的事

- 不宣稱 paper 完成、不宣稱 compute-saving、不宣稱 task-free/consolidation solved。

## 6. Changelog

- 2026-06-27：建立 SPEC（M5 動工）。
