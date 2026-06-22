# NaviPath v0.4 執行進度

> 逐步執行紀錄，可追可溯。每個 TASK 結束後更新。
> 流程定義見 [SOP_v0.4.md](../SOP_v0.4.md)。

---

## v0.3 分界點
- commit `dd914af` — docs(v0.3): summary checkpoint + v0.4 experiment SOP
- 內容：SOP_v0.4.md + fold2/3 歷史封存（outputs_history/0607_5th_fold_2_3）

---

## TASK 0 — Mac：開啟追蹤 + 改 code + smoke + push  [進行中]

時間：2026-06-22

### [1] .gitignore
- 移除 `outputs/`、`*.pt`、`*.pth`；保留 `checkpoints/`、`*.bin`（保護 CONCH）。
- 新增 `outputs/_smoke/`（smoke 暫存不入 git）。
- 驗證：`git check-ignore outputs/qpmil_paper_fold1.pt` → 無輸出（不再忽略）✅
- 驗證：`checkpoints/conch/pytorch_model.bin` 仍被 `checkpoints/` 忽略 ✅

### [2] train_router_v0.py
- 新增 `--eval-tasks`（逗號字串，如 `"-1,0"`；空字串 fallback `--task-to-eval`）。
- 訓練後依清單逐一評估：最後任務→`router_v0_{order}_fold{f}.json`；其餘→`oldtask_budget_{order}_f{f}_task{t}.json`。
- 防覆蓋：目標 json/.pt 已存在則 `[skip] exists`。json 多存 `task_index`。
- 未動 run_patch_budget.py / eval / backbone / metrics。lint 通過 ✅

### [3] outputs/PROGRESS.md
- 本檔，建立。

### [4] smoke test
- ruff：train_router_v0.py 有 7 個 pre-existing 警告（lines 24/28/86/87/119/185，皆未用 import 等），非本次新增；不處理。
- `--help`：`--eval-tasks` 出現 ✅；`import train_router_v0` OK ✅
- 重要修正：`--eval-tasks "-1,0"`（空白形式）會被 argparse 誤判為旗標 → 改用 `=` 形式 `--eval-tasks="-1,0"`。SOP 已同步更新。
- micro run（reverse f1, 1 epoch, 16 slides, --out outputs/_smoke）成功，產出：
  - `router_v0_reverse_fold1.json`（task_index=3, tcga_lung）
  - `oldtask_budget_reverse_f1_task0.json`（task_index=0, tcga_esca）
  - 兩者 schema 皆含 `task_index` ✅；數字為 smoke 垃圾值（不採用）。
- `outputs/_smoke` 已刪除 ✅

### [5] commit + push
- commit：[待填 hash]
- 內容：.gitignore + train_router_v0.py + SOP_v0.4.md（=修正）+ outputs/PROGRESS.md

---

## TASK 1 — RunPod：同步 + 體檢  [未開始]

## TASK 2 — RunPod：router 補格 + old-task  [未開始]

## TASK 3 — RunPod：彙整結果  [未開始]

## TASK 4 — RunPod：結果推回 GitHub  [未開始]
