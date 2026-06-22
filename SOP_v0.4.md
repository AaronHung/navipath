# SOP v0.4 — Router Budget + Old-task Budget

> v0.3 已完成（baseline + 架構自我診斷，見 [README.md](README.md)）。
> v0.4 目標：補滿 router patch-budget 6 格 + 新增 old-task 生死表，湊齊 13/13 canonical JSON。
> 結果存 `outputs/`（進 git）；`outputs_history/` 為暫存 / 報告封存。
> Mac 改 code、RunPod 跑實驗。每個 TASK 結束更新 `outputs/PROGRESS.md`。

---

## 為什麼這樣設計（一句話）

`run_patch_budget.py` 不含 router 也不產 JSON，無法回答「router 還選得對嗎」的生死表。
改用 `train_router_v0.py` 一次連續訓練，同時產出「最後任務表（router_v0）」＋「task0 老任務表（oldtask_budget）」，把舊 SOP 的 PROMPT 1+2 合併。

---

## 期望檔案矩陣（= 完成標準，13/13）

| 類別 | order | fold | canonical 檔 | 狀態 |
|---|---|---|---|---|
| baseline | paper | 1/2/3 | `qpmil_paper_fold{1,2,3}.json` | DONE |
| baseline | reverse | 1/2/3 | `qpmil_reverse_fold{1,2,3}.json` | DONE |
| router | paper | 1 | `router_v0_paper_fold1.json` | DONE |
| router | reverse | 3 | `router_v0_reverse_fold3.json` | DONE |
| router | paper | 2/3 | `router_v0_paper_fold{2,3}.json` | TASK 2 |
| router | reverse | 1/2 | `router_v0_reverse_fold{1,2}.json` | TASK 2 |
| old-task | reverse | 1/2/3 | `oldtask_budget_reverse_f{1,2,3}_task0.json` | TASK 2 |

TASK 2 跑完應新增 7 個 canonical json（router x4 + old-task x3）。

---

## 命名規則

- baseline：`qpmil_{order}_fold{f}.json`
- router（最後任務）：`router_v0_{order}_fold{f}.json` + `.pt` + `.log`
- old-task（回看 task t）：`oldtask_budget_{order}_f{f}_task{t}.json` + `.log`
- 重跑：副檔名前加 `__rerun_{YYYYMMDD_HHMM}`，canonical 不動。

---

# TASK 0 — Mac：開啟追蹤 + 改 code + smoke + push

```text
你在我的 Mac repo (/Users/aaron/research/01_navipath) 協助 v0.4 第一步。逐項做，任一失敗就停、貼 error、提最小修補等我批准。

[1] 改 .gitignore：移除 outputs/、*.pt、*.pth 三行；保留 checkpoints/ 與 *.bin。
    驗證：git check-ignore outputs/qpmil_paper_fold1.pt 應「無輸出」。

[2] 改 train_router_v0.py 的 main()：
    - 新增 --eval-tasks（逗號字串，如 "-1,0"；空字串時 fallback 用 --task-to-eval）。預設行為不變。
    - 訓練迴圈結束後，依 --eval-tasks 逐一評估：
        eval_t = et % len(order)；跑既有 eval_router_vs_heuristics + print_table
        檔名：最後一個任務 -> router_v0_{order}_fold{f}.json
              其他任務     -> oldtask_budget_{order}_f{f}_task{eval_t}.json
        防覆蓋：目標 json 已存在 -> [skip] exists 跳過；.pt 同樣防覆蓋。
        json 多存 task_index 欄位。
    - 不動 run_patch_budget.py / eval/patch_budget_eval.py / backbone / metrics。

[3] 建 outputs/PROGRESS.md，寫入 TASK 0 起始紀錄。

[4] smoke（不可碰 canonical，全寫到丟棄目錄）：
    ruff check .
    python train_router_v0.py --help
    python -c "import train_router_v0"
    python train_router_v0.py --order reverse --fold 1 \
      --backbone-ckpt outputs/qpmil_reverse_fold1.pt \
      --eval-tasks="-1,0" --epochs 1 --max-train 16 --max-eval 8 --out outputs/_smoke
    # 注意：--eval-tasks 值以 - 開頭，必須用 = 形式，否則 argparse 會誤判為旗標。
    確認 outputs/_smoke/router_v0_reverse_fold1.json 與
         outputs/_smoke/oldtask_budget_reverse_f1_task0.json 都生成且含 task_index，
    然後 rm -rf outputs/_smoke。

[5] commit code + .gitignore + PROGRESS.md（不含 _smoke）：
    git add .gitignore train_router_v0.py outputs/PROGRESS.md
    git commit -m "feat(v0.4): track outputs/, add --eval-tasks for old-task budget"
    git pull --rebase && git push
```

---

# TASK 1 — RunPod：同步 + 體檢（不跑實驗、不改 code）

```text
你在 RunPod /workspace/src/navipath 協助我。只檢查，不跑訓練/評估、不改 code。

[0] cd /workspace/src/navipath && git pull --rebase
    印 pwd、git log --oneline -3、git status（clean）、HEAD == origin。
[1] 寫/更新 outputs/SOP_RULES.md（結果存 outputs/ 且進 git；RunPod 用 git add outputs/）。
[2] python -c "import torch;print(torch.__version__, torch.cuda.is_available())"；nvidia-smi
[3] 盤點：outputs/qpmil_{paper,reverse}_fold{1,2,3}.pt（6）+ 6 baseline json + router json（paper f1、reverse f3）。
[4] python train_router_v0.py --help | grep -iE "eval-tasks|task-to-eval|fold|order" -> 確認 --eval-tasks 存在。
[5] 更新 outputs/PROGRESS.md。給 GO/NO-GO for TASK 2。
```

---

# TASK 2 — RunPod：router 補格 + old-task（5 條指令）

```text
先讀 outputs/SOP_RULES.md。pwd = /workspace/src/navipath。既有檔唯讀、防覆蓋、失敗即止、不捏造。

[防覆蓋] ls 這 7 個 json，已存在那格跳過（重跑加 __rerun_YYYYMMDD_HHMM）：
  router_v0_paper_fold2.json / router_v0_paper_fold3.json
  router_v0_reverse_fold1.json / router_v0_reverse_fold2.json
  oldtask_budget_reverse_f1_task0.json / _f2_task0.json / _f3_task0.json

[執行] 逐條跑，全程 tee：
  python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold2.pt   --order paper   --fold 2 --epochs 5 2>&1 | tee outputs/router_v0_paper_fold2.log
  python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold3.pt   --order paper   --fold 3 --epochs 5 2>&1 | tee outputs/router_v0_paper_fold3.log
  python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold1.pt --order reverse --fold 1 --eval-tasks="-1,0" --epochs 5 2>&1 | tee outputs/router_v0_reverse_fold1.log
  python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt --order reverse --fold 2 --eval-tasks="-1,0" --epochs 5 2>&1 | tee outputs/router_v0_reverse_fold2.log
  python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold3.pt --order reverse --fold 3 --eval-tasks="0"    --epochs 5 2>&1 | tee outputs/oldtask_budget_reverse_f3_task0.log

# 注意：--eval-tasks 值以 - 開頭必須用 = 形式（--eval-tasks="-1,0"），否則 argparse 報錯。

每條跑完 ls 確認 json 非空；任一失敗就停。抓 router/random/prototype/semantic 各 K 的 ACC 貼我。更新 PROGRESS.md。
```

---

# TASK 3 — RunPod：彙整結果

```text
先讀 outputs/SOP_RULES.md。pwd = /workspace/src/navipath。不准捏造，只用實際 json，缺格標 [MISSING]。

[1] cat 一個 router_v0_*.json、一個 oldtask_budget_*.json、一個 qpmil_*.json 確認 schema。
[2] 寫 tools/collect_results.py（冪等、不覆蓋），glob outputs/ 產 outputs/RESULTS_<ts>.md：
    表 A：QPMIL baseline 3 折 ACC/Forgetting + mean±std
    表 B：router budget 6 格 + mean±std + GO 判定 (router@64-random@64>0 且 router@128>=semantic@128)
    表 C：old-task（reverse task0=esca, 3 折）各 K ACC + mean±std；印 router vs random / vs semantic delta（不設門檻）
    用 task_index 區分 B/C。
[3] append outputs/MANIFEST.md。
[4] 表 B、C 轉乾淨 LNCS markdown 貼回（填 §4.3/§4.4）。更新 PROGRESS.md。
```

---

# TASK 4 — RunPod：結果推回 GitHub

```text
cd /workspace/src/navipath
git add outputs/
git status     # 只該有 outputs/，不該有 code
git commit -m "results(v0.4): router paper f2/f3 + reverse f1/f2, old-task reverse f1-3 task0 (<日期>)"
git pull --rebase && git push
```

---

## 模型建議

- TASK 0（Mac，動 code）/ TASK 3（寫 collector）：Opus 4.8 High
- TASK 1 / 2 / 4（pull、跑指令、commit）：Sonnet 4.6 medium 亦可，省 token
