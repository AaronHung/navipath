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

---

## 2026-06-27（晚）N0 — 架構鎖定 + 大掃除（對齊老師 bottom line）

老師定調：在現有 WSI navigation agent 之上擴增 CL；7/3 報 agent+CL 架構（含圖）+ 到 7/2 的實驗結果。

- **新 SOP**：`specs/01_sop_navipath-cl_phase0.md`（N0–N8，取代 `legacy/SOP_v0.4.md`）。看板新增 `site/sop.html`（SOP Tab）。
- **架構圖（vector SVG，可轉 PDF）**：
  - 總圖 `site/figs/arch_navipath_cl.svg`（頂刊寬幅，frozen/CNL/CL-memory/future 四色）。
  - 細部 `site/figs/arch_navipath_cl_detail.svg`（CNL → Backbone Interface → QPMIL instance，源自 STORYLINE §6）。
  - 兩張皆掛上 `site/architecture.html`；`index.html#arch` 舊 `Fig1_arch.png`（被打槍那張）已換成新總圖。
  - 修正：SVG 內 `…`/`—` 等標點造成 XML 破損 → 全改 ASCII，xml parse 驗證 OK。
- **wiki 04**：`docs/wiki/04_generalization-and-attribution.md`（兩條軸：backbone CL 家族 vs 我們的 CNL/NSM；歸因＝凍結同一 backbone 只動 navigation 層）。
- **README 重寫**；14 個舊/v0.4 MD 歸檔 `legacy/`（保留 STORYLINE/COLLAB_PLAYBOOK/SESSION_CONTEXT/RUNPOD_*）。
- **下一步 N1**：實作 `navipath_moe/sequential_observation.py`（Observation State + 多步 budgeted observation）。

---

## 2026-06-27（晚）N1 — Sequential Budgeted Observation（Mac，code 完成）

- 新增 `navipath_moe/sequential_observation.py`：
  - `ObservationState`：累積 seen trace / 聚合特徵 / coverage / backbone 信心（Agent 短期記憶）。
  - `SequentialBudgetedObserver`：多輪、每輪 top-k；用 redundancy penalty（與已看區域相似度）讓「下一步看哪」取決於已看到的 → 真正序列決策；可信心達標早停。純函式、可測，不依賴 backbone 內部。
  - `ContinualSequentialNavigationAgent`：frozen backbone + NSM(per-task skill) + oracle Gate + 序列觀察；產出 trace。重用 `routers.py` / `continual_agent.py` 零件，不重造。
  - **one-shot 模式**（redundancy_weight=0 且 step_size>=budget）退化成舊 Top-K → 即 N2 的 sequential vs one-shot ablation。
- `__init__.py` 匯出新類別。
- 新增 `tests/test_sequential_observation.py`：6 個 smoke（budget 上限、trace 順序一致、one-shot==Top-K、seq≠one-shot、早停、agent e2e）。
- 驗證：`ruff check` 乾淨；`PYTHONPATH=. python tests/test_sequential_observation.py` → **6/6 PASS**（CPU、合成輸入，未碰 canonical outputs）。
- **下一步 N2（RunPod）**：掛 frozen QPMIL backbone + 真資料，跑 acc@K vs acc@All、有/無 NSM old-task 保留、sequential vs one-shot。

---

## 2026-06-28 — N2：RunPod 第一批實驗（reverse 3-fold）

做了什麼：
- 加 `eval_sequential_observation.py` 的 `--skill-bank-out/in`、`--eval-tasks`、`--policy-mode`：訓練一次存 NSM bank，之後純 inference 補滿 4 任務 retention + zero-shot（避免重訓）。
- RunPod 跑 reverse fold 1/2/3：每 fold 訓 4 任務 router、存 `outputs/skill_bank_reverse_f{1,2,3}.pt`；評估 nsm/nonsm（router）與 zero-shot navigator。

產出：`outputs/seqobs_reverse_f*_task{0,1,2,3}.json` + `*_policy-zeroshot.json` + skill banks，全 push（commit 56dae04 / f7f8672）。

踩雷：RunPod 的 `QPMIL-VL/configs/main.yaml` 等 config 是本機路徑、反覆擋 pull/push → 改用 `git update-index --skip-worktree` 一勞永逸。

下一步：N3 Mac 分析。

---

## 2026-06-28 — N3：Mac 分析 + 結果圖

做了什麼：
- 新增 `analyze_seqobs_n3.py`：彙整 24 個 JSON → `outputs/RESULTS_seqobs_20260628.md` + `site/figs/n3_retention_bar.png`、`n3_esca_budget_curve.png`。
- 看板新增「D · N2/N3 Pilot 結果」。

數字（reverse 3-fold, oracle, seq, budget=64）：
- mACC：continual+NSM **0.935±0.017** / naive **0.595±0.035** / zero-shot **0.858±0.038**。
- Forgetting：NSM **0** / naive **0.454**。esca@64：我們 0.911 vs naive 0.333。
- 三層故事：naive 嚴重遺忘 → NSM 修復 → zero-shot 強但仍輸我們（且贏 naive）。
- 誠實：seq == oneshot（差 0），policy 尚靜態。

commit 09fa11c。

下一步：N4 報告稿。

---

## 2026-06-28 — N4：雙月報告稿改寫 + 答辯底稿 wiki

做了什麼：
- `reports/bimonthly_2026-07-03.md` 從舊 mechanism-selection/GO-NO-GO 敘事**改寫**為 agent+CL 通用敘事 + 真實 N2/N3 數字（commit 0fc909a）。
- 新增 wiki 10（機制防禦＋多步路線圖）、wiki 11（機制逐層拆解白話 + budget 省算力其他表示 RLogist/Cordonnier + 術語對照）；看板答辯筆記擴到六區（藍/綠/紫/琥珀/青/灰）。架構頁加兩個「讀圖必懂」。commit fb0f0b8。

重要釐清（寫進 wiki，供答辯）：
- NSM = parameter-isolation **上界**（不是賣點）；賣點＝便宜記憶逼近上界 + navigation 需 CL 這條軸 + task-free gate 難題。
- 多步靠**結果回饋**（label-only），不需醫師 trajectory；存的是 policy 不是步驟。

下一步：N5 口頭報告收尾。

---

## 2026-06-28 — N5：7/3 口頭報告收尾

做了什麼：
- 新增 `reports/talk_2026-07-03.md`（5–7 分鐘講稿大綱 + 7 張投影片極簡版 + Q&A 速查表，對應答辯筆記顏色）。
- 看板 kanban 更新為 N-phase（N0–N4 done / N5 in progress / N6–N8 next）；hero 進度改 ~75%；交付物清單補 N2/N3 結果、講稿、wiki 答辯底稿。

狀態：**N6 之前全部完成、雙邊 repo 同步。** 7/3 由 Aaron 對外呈現（簡報轉換為人工步驟）。

下一步（N6，7/3 之後）：便宜記憶（prompt/LoRA 擇一）逼近 NSM 上界 + 多步路線 A（Mac 可做）；論文初稿。RunPod 可關，需要時重開做 paper-order 對稱 / EWC baseline。
