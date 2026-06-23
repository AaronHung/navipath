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
- commit：`6f56a26` feat(v0.4): track outputs/, add --eval-tasks for old-task budget ✅
- 內容：.gitignore + train_router_v0.py + SOP_v0.4.md（=修正）+ outputs/PROGRESS.md
- 另外也加了 `outputs_history/**/*.pt` 到 .gitignore（避免 500MB 冗餘 .pt 備份進 repo；報告需要的 json/log/png 已在 git）。

**TASK 0 完成。** 下一步：去 RunPod 貼 TASK 1（git pull + 體檢）。

---

## SYNC — RunPod outputs 上 GitHub（2026-06-22）  [完成]

- 起因：RunPod 才是結果來源，Mac 的 outputs/ 只是 smoke 殘留。
- RunPod：`git pull --rebase`（快轉到 6f56a26）→ `git add outputs/`（57 檔，0 非 outputs/）→ commit + push。
  - 過程修了兩個 RunPod 容器問題：git 身分未設（`git config user.email/name`）、HTTPS 需 PAT token（`git remote set-url` 帶 token）。
  - commit `b18feb1` results(v0.4): RunPod authoritative outputs（47 MiB）✅
- Mac：`git clean -fd outputs/`（刪 14 個本地重複）→ `git pull --ff-only` 到 `b18feb1` ✅
- 從此 **RunPod 為 outputs/ 唯一來源**，Mac 不再 push 結果（PROGRESS.md 例外）。

### 既有結果盤點（b18feb1，已驗證為真實非 smoke）

- baseline：qpmil paper f1/2/3 + reverse f1/2/3 → **6/6** ✅
- router_v0：paper f1/2/3 + reverse f2/3 → **5/6**（缺 reverse f1）；5 個皆 GO ✅
  - reverse f2/f3 eval=lung(95 張) 數字細緻（如 0.9263=88/95）→ 確為完整評估
  - paper f2/f3 eval=esca(15 張) → 真實但小樣本
- oldtask_budget：**0/3**（缺 reverse f1/f2/f3 task0）
- bonus：navipath_full / micro 各 fold 的 M5-M9 也都在

---

## TASK 1 — RunPod：同步 + 體檢  [可略/已等價完成]
SYNC 已完成 git pull 與檔案盤點；正式 TASK 1 體檢可選跑（環境 nvidia-smi / torch）。

## TASK 2 — RunPod：補 router reverse f1 + 三個 oldtask  [待跑，僅 3 條]

> 開跑前先 `cd /workspace/src/navipath && git pull --ff-only`（拿這份 PROGRESS 更新）。
> 防覆蓋已內建：router_v0_reverse_fold2/3 已存在 → 自動 [skip]。

```bash
# (1) reverse f1：一次補 router_v0 reverse f1 + oldtask reverse f1
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold1.pt --order reverse --fold 1 --eval-tasks="-1,0" --epochs 5 2>&1 | tee outputs/router_v0_reverse_fold1.log
# (2) reverse f2：只補 oldtask（router_v0 已存在→skip）
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt --order reverse --fold 2 --eval-tasks="0" --epochs 5 2>&1 | tee outputs/oldtask_budget_reverse_f2_task0.log
# (3) reverse f3：只補 oldtask
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold3.pt --order reverse --fold 3 --eval-tasks="0" --epochs 5 2>&1 | tee outputs/oldtask_budget_reverse_f3_task0.log
```

跑完 13/13：baseline 6 + router 6 + oldtask 3（注意 router 第 6 格 = reverse f1 由 (1) 產生）。

## TASK 3 — 彙整結果  [工具就緒，待 TASK 2 完成後跑]

時間：2026-06-22（Mac 端先把工具寫好，各開各的工）

### 工具：`tools/collect_results.py`
- 純 stdlib（無外部相依），掃 `outputs/*.json` 自動分類聚合（跨 fold mean±std）。
- 認得的檔名：`qpmil_* / navipath_full_* / navipath_micro_*`（ACC/Forgetting/BWT）、
  `router_v0_*`（last-task budget GO/NO-GO）、`oldtask_budget_*`（舊任務生死表）、`routing_drift_*`。
- 產出三張 Markdown 表 + 可選 CSV；缺資料的區塊顯示 `(no data found)` 不會壞。
- 已用現有 21 個 JSON 實測通過 ✅（oldtask 區塊現為空，待 TASK 2 產出後自動填）。

跑法（TASK 2 的 3 條 oldtask 補完後執行）：
```bash
cd /workspace/src/navipath
python tools/collect_results.py --outputs outputs -o outputs/RESULTS_SUMMARY.md --csv outputs/csv
```
- `RESULTS_SUMMARY.md`：論文用三表（人看）。
- `outputs/csv/{accuracy,router_budget,oldtask_budget}.csv`：畫圖/貼表用。

### 繪圖：`tools/plot_results.py`（論文圖，Mac 已驗證）
- **P0** patch-budget 曲線（router vs random/prototype/semantic，逐 order/task，含 fold std error bar + All-patch 參考線）→ `P0_router_v0.*`、`P0_oldtask_budget.*`。
- **P1** R-matrix heatmap（QPMIL vs NaviPath × paper/reverse，跨 fold 平均）→ `P1_r_matrix.*`。
  - 重要敘事：NaviPath 每個 column 完全持平（= F=0 是 decouple 恆等式）；QPMIL baseline 有微幅變動。這張就是誠實面對 reviewer 的圖。
- **P2-lite**（質化，需 model stack）feature-space t-SNE，色=router score，紅圈=top-K 選中 → `P2lite_{order}_fold{f}.*`。
  - 跨機器移植：`build_backbone_from_ckpt` 新增 `path_remap`，自動把 ckpt 內嵌的 `/workspace/src/navipath` 改寫成本機 repo root（RunPod 存的 ckpt 在 Mac 也能跑）。
- P0/P1 純讀 JSON、無 torch，一定能跑；P2-lite 缺 model/ckpt 會自動 `[skip]` 不報錯。
- Mac 已實測三張圖皆正常產出（demo 圖未進 git；由 RunPod 跑權威版）。

跑法（TASK 3 之後，oldtask 已補齊）：
```bash
cd /workspace/src/navipath
# 必出圖（P0+P1）
python tools/plot_results.py --outputs outputs --figdir outputs/figs
# 加質化圖 P2-lite（paper/fold1，自動挑一張 esca slide）
python tools/plot_results.py --outputs outputs --figdir outputs/figs --p2 --p2-order paper --p2-fold 1
```

## ⚠️ 重大發現（2026-06-23）：Router catastrophic forgetting

TASK 2 reverse f1 跑出 oldtask_budget（esca，最舊任務）= **NO-GO 且崩潰**：
`router@64 = 0.133`，遠輸 random 0.80。對照 paper order 同一個 esca（剛學完）router@64=0.933。
→ 判定為 **router 自身的 catastrophic forgetting**（非 bug；All budget 四法皆 0.867、heuristics 同條件正常、訓練 log 顯示 esca 先學好後被覆寫）。

**完整紀錄 + Plan B 設計 + 實驗/寫稿計畫見 [../ROUTER_FORGETTING_v0.4.md](../ROUTER_FORGETTING_v0.4.md)。**

決議：(a) 不中斷，等 reverse f2/f3 確認複現；(b) 設計草稿已寫（暫不動 code）；
(c) 敘事三 fold 後二選一：A 誠實分析稿（預設）/ B 加 router consolidation 救。

**2026-06-23 更新：reverse f1/f2/f3 全部 pull 完成，三 fold 全崩 GO=0/3**（esca router@64
mean=0.333，輸 random 0.822 / prototype 0.778 / semantic 0.778）。複現確認，效應強。

**2026-06-23 對照完成：paper oldtask（最舊=lung，樣本多）三 fold 也全崩**（router@64
mean=0.397，GO 0/3）→ **混淆排除，崩因是 recency 非樣本少**。總計舊任務 6/6 NO-GO、
近期任務 6/6 GO，且 lung/esca 都有「最近→GO、最舊→崩」翻轉。發現坐實。下一步：出圖/寫稿 或 Plan B。

## TASK 4 — RunPod：結果推回 GitHub  [未開始]
