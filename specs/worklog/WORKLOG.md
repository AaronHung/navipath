# WORKLOG — NaviPath-CL 開發過程（append-only）

> 每段做完 append 一個區塊：做了什麼 / 數字 / smoke / 踩雷 / commit / 下一步。
> 實驗執行細節在 [`../../outputs/PROGRESS.md`](../../outputs/PROGRESS.md)，此處不重抄。

---

## 2026-06-27 — M0：建立 SDD 記錄骨架

做了什麼：
- 建 `specs/` 體系：`README.md`（流程規則+索引）、`00_master_spec.md`（技術索引）。
- ADR-0001~0005 回填本 session 已拍板決策（pivot / QPMIL 定位 / mechanism-selection / 命名 / SDD workflow）。
- 開本 WORKLOG。

重要事項：
- 保密規則生效：上位計畫一律 North Star，文件不得有可識別資訊。
- 兩套語言紀律（內部 vs paper）寫入 ADR-0004。
- consolidation/parameter-merging 無 navigation 數字 → 一律標 ongoing，不 claim solved（ADR-0003）。

smoke / 驗收：
- 文件結構建立完成；README 索引連結涵蓋全部 ADR/SPEC/worklog。

下一步：M1 — 寫 [SPEC-01](../features/SPEC-01-continual-agent.md) 再實作 `navipath_moe/continual_agent.py`。

---

## 2026-06-27 — M1：continual_agent.py（CNL agent wrapper）

做了什麼：
- 寫 [SPEC-01](../features/SPEC-01-continual-agent.md)（介面 + 驗收標準 + 不做的事），再實作。
- 新增 `navipath_moe/continual_agent.py`：`NavigationSkillBank`（NSM，可持久化 task→router state）、`ContextGate`（oracle 啟用 / infer 留 future）、`ContinualWSINavigationAgent`（navigate→predict）。backbone-agnostic（鴨子型別），不依賴 train_router_v0。
- `navipath_moe/__init__.py` 加 3 個 export。
- 新增 `tests/test_continual_agent.py`（stub backbone + 隨機 tensor）。

數字 / 驗收：
- ruff All checks passed（修了 test 的 E401）。
- `import navipath_moe.continual_agent` OK。
- `pytest tests/test_continual_agent.py -q` → 5 passed；全套件 `pytest tests/ -q` → 11 passed（動了 __init__.py 無回歸）。

重要事項 / 設計決策：
- agent 不在頂層 import train_router_v0（避免循環）；M3 的數字重現放 root 腳本延遲 import。
- 真實 backbone 數字重現（用 ckpt + 真資料）不在 M1 範圍，屬 M3。
- Python 環境：用 repo 內 `.venv`（torch 2.12.0）；`ruff` 不在系統 PATH，需 `source .venv/bin/activate`。

下一步：M2 — [SPEC-02](../features/SPEC-02-evidence-and-symmetry-runs.md) 證據彙整（先用既有 outputs/*.json 跑 collect_results）。

---

## 2026-06-27 — M2：證據彙整（mechanism-selection 主表）

做了什麼：
- 跑既有 `tools/collect_results.py` → `outputs/RESULTS_SUMMARY.md` + `outputs/csv/*.csv`（既有工具只認 shared，不做機制對比）。
- 新增 `tools/mechanism_table.py`（純 stdlib），補 shared/ewc/pertask 機制對比 → `outputs/MECHANISM_SELECTION.md`。

數字（reverse old ESCA, router acc，跨 3 fold mean）：
- @64：shared **0.333** / ewc **0.400** / pertask **0.933**；best-heuristic 0.844。
- GO：shared 0/3、ewc 0/3、pertask 3/3 → 完整支撐決策樹（ADR-0003）：共用會忘、EWC 不足、per-task skill memory 恢復。
- paper-order ewc/pertask = no data（stretch 待 RunPod，指令見 SPEC-02）。

驗收：ruff pass；MECHANISM_SELECTION.md 產出且數字對齊 STORYLINE §9。

下一步：M3 — agent end-to-end 重現 + 決策樹 argument（核心交付）。

---

## 2026-06-27 — M3 ★：agent end-to-end + 決策樹 argument

做了什麼：
- 寫 [SPEC-06](../features/SPEC-06-continual-agent-eval.md)，新增 root 腳本 `eval_continual_agent.py`：重用 `train_router_one_task` 逐任務訓練、每 task snapshot 進 `NavigationSkillBank`，用 `ContinualWSINavigationAgent`(oracle gate) 評估舊任務，並對照既有 pertask json 做一致性檢查。
- ruff pass；`--help`/import OK；Mac pipeline smoke（fresh backbone + 極小資料）端到端跑通（loss 1.87→0.06，skill bank 4 skill，navigate→predict OK）。真數字（重現 0.933）指令備好交 RunPod（SPEC-06 §6）。

M3 三項交付狀態：
1. agent e2e：腳本完成 + Mac smoke 通過；真數字待 RunPod。
2. mechanism-selection 表：M2 完成（`outputs/MECHANISM_SELECTION.md`）。
3. 決策樹 argument（數字已撐，reverse old ESCA，router acc 跨 3 fold mean）：
   - Q1 shared 能學所有任務？→ No：@64 **0.333** < best-heur 0.844（GO 0/3）。
   - Q2 EWC 能修？→ Not sufficiently：@64 **0.400** < best-heur（GO 0/3）。
   - Q3 舊 skill 丟了嗎？→ No：per-task skill memory @64 **0.933** > best-heur（GO 3/3）。
   - 結論：問題不在缺診斷訊號，而在缺 continual navigation memory（NSM）。

重要事項：
- fresh backbone 的 agent acc 是隨機值（backbone 未訓練），不可當結果；僅證 pipeline。真數字一律 RunPod 用 trained ckpt。

下一步：M4 — [SPEC-03](../features/SPEC-03-core-figures.md) 5 張核心圖。

---

## 2026-06-27 — M4：5 張核心圖

做了什麼：
- 寫 [SPEC-03](../features/SPEC-03-core-figures.md)。
- 改 `tools/draw_arch.py` 命名對齊新敘事（Selection path→Navigation policy/CNL、Plan B→Navigation Skill Memory、移除 "Forgetting=0"、Prediction path→Diagnostic backbone (QPMIL-VL instance)）。
- 新增 `tools/make_figures.py`（重用 `mechanism_table.load_records`，數字不重抄）：Fig_problem / Fig_mechanism / Fig_budget_curve / Fig_roadmap。

產出（`outputs/figs/`）：Fig1_arch、FigS1_arch、Fig_problem、Fig_mechanism、Fig_budget_curve、Fig_roadmap（各 {png,pdf}）。

驗收：ruff All checks passed（順手修掉 draw_arch 既有 E702）；Fig_mechanism 視覺驗證 @64 = shared 0.333(紅) / EWC 0.400(紅) / per-task NSM 0.933(綠,唯一過 best-heur 0.844 虛線)。

下一步：M5 — [SPEC-04](../features/SPEC-04-report-0703.md) 凍結結果 + 雙月報告稿。

---

## 2026-06-27 — M5：凍結結果 + 雙月報告稿

做了什麼：
- 寫 [SPEC-04](../features/SPEC-04-report-0703.md)，產 `reports/bimonthly_2026-07-03.md`（zh-TW，八章：摘要/背景pivot/問題/North Star對齊/方法/pilot證據/pivot plan/誠實邊界）。
- 凍結 snapshot：報告引用 `outputs/MECHANISM_SELECTION.md` + `outputs/RESULTS_SUMMARY.md` + `outputs/figs/`，宣告報告期間不加新實驗。

驗收：八章齊備、引用 5 張圖與核心數字、North Star 代稱、§8 do-not-claim + frozen 宣告齊。

下一步：M6 — 報告定稿（架構+pilot+pivot plan 三要素確認）。

---

## 2026-06-27 — M6：報告（架構 + pilot + pivot plan）

做了什麼：
- 確認 `reports/bimonthly_2026-07-03.md` 已涵蓋 M6 三要素：架構（§5 CNL/NSM + Fig1）、pilot（§6 mechanism-selection 表 + 決策樹 + Fig_mechanism/budget_curve）、pivot plan（§7 future：task-free gate / consolidation / RLHF）。
- 不假裝 paper 已完成（§8 明確標示）。

狀態：報告稿就緒，待 7/3 由 Aaron 對外呈現（簡報轉換為人工步驟）。無新增實驗（凍結期）。

下一步：M7 — [SPEC-05](../features/SPEC-05-paper-draft.md) 論文初稿（paper framing）。

---

## 2026-06-27 — M7：論文初稿（paper framing）骨架 + 年度報告大綱

做了什麼：
- 寫 [SPEC-05](../features/SPEC-05-paper-draft.md)。
- 發現既有論文資產為 LaTeX（`paper/main.tex`、`paper_body.tex`，舊 framing）→ 決定**不覆寫 .tex**，改交付 markdown 重寫大綱供移植。
- 新增 `paper/NaviPath-CL_draft_outline.md`（NaviPath-CL framing：Abstract/Intro/Related/Method/Experiments/Discussion/Conclusion，填入 mechanism-selection 證據與圖、標 TODO）。
- 新增 `reports/annual_2026-07-20_outline.md`（年度報告大綱）。

踩雷：先前兩次 Write（SPEC-05、paper/draft.md）的工具回應被異常內容污染、實際未寫入；已重做並用 `ls` 驗證。改用 `NaviPath-CL_draft_outline.md` 命名避免與既有 LaTeX 混淆。

狀態（M7 為 7/3–7/15 的多週里程碑，依賴 Aaron + RunPod + review）：
- 我這端可完成者：SPEC-05 + 論文大綱 + 年度報告大綱 → **已完成**。
- 待人工/外部：移植回 `.tex`、老師 review、paper-order 對稱與 agent 真數字（RunPod）、可選 task-free/consolidation pilot。

---

## 凍結 / 交接摘要（2026-06-27 本批）

- 全部 SDD 骨架 + M1–M5 程式/數據/圖/報告交付物已完成並驗證；M6 報告稿就緒；M7 論文/年報大綱就緒。
- **待 Aaron / RunPod** 的外部相依（非本機可完成）集中於：
  1. RunPod：`eval_continual_agent.py` 重現 reverse old ESCA@64≈0.933（SPEC-06 §6 指令）。
  2. RunPod：paper-order per-task/EWC 對稱補跑（SPEC-02 指令）。
  3. 7/3 對外簡報、7/15 老師 review、7/20 年度報告填實。
- 未自動 git commit（依規矩等 Aaron 指示）。建議 commit 訊息：`feat(SDD): NaviPath-CL specs + continual agent + evidence/figures/report (M0–M7 skeleton)`。
