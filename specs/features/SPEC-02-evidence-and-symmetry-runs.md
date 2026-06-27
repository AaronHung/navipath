# SPEC-02 — Evidence consolidation & symmetry runs

- Status: MVP done (2026-06-27); stretch 待 RunPod
- Milestone: M2 (6/29)
- Related: ADR-0003（mechanism-selection framing）

## 1. 目標

把既有 `outputs/*.json` 聚成 **mechanism-selection 證據**（STORYLINE §4 主表）：對 old task，比較 shared / EWC / per-task 三種 continual 機制的 navigation 表現。stretch：RunPod 補 paper-order 對稱。

## 2. 交付物

- MVP-1：跑既有 `tools/collect_results.py` → `outputs/RESULTS_SUMMARY.md` + `outputs/csv/*.csv`（已驗存在）。
- MVP-2：新工具 `tools/mechanism_table.py`（純 stdlib），掃 `outputs/oldtask_budget_{order}_f{fold}_task{t}{suffix}.json`（suffix: ""=shared / `__ewc` / `__pertask`），輸出 `outputs/MECHANISM_SELECTION.md`：每個 (order, eval_task) 一張表，rows = {shared, ewc, pertask, best-heuristic}，cols = budgets，值 = router acc 跨 fold mean。
- stretch（RunPod tmux，需 Aaron 跑）：補 paper-order per-task + EWC（SESSION_CONTEXT §7 指令），讓兩 order 對稱。

## 3. 資料 schema（已確認）

`oldtask_budget_*.json`：`{order, fold, eval_task, task_index, consol, budgets:[0,256,128,64,32], results:{router|random|prototype|semantic:{"All"|"<K>": acc}}, go}`。budget key 為字串（"All" 或 "64"）。mechanism 由**檔名後綴**判定（json `consol` 欄：none/ewc/pertask）。

## 4. 驗收標準

- [x] `tools/mechanism_table.py` 純 stdlib、ruff pass、可跑、缺資料顯示 `(no data)` 不報錯。
- [x] `outputs/MECHANISM_SELECTION.md` 產出，reverse old ESCA router@64 跨 fold mean = shared **0.333** / ewc **0.400** / pertask **0.933**，best-heuristic 0.844（對齊 STORYLINE §9）。
- [x] WORKLOG 記錄三機制數字。
- [ ] stretch（RunPod）：補 paper-order ewc/pertask，讓兩 order 對稱（目前 paper-order ewc/pertask = no data）。

## RunPod stretch 指令（tmux，跑完自動 push）

```bash
cd /workspace/src/navipath && git pull --ff-only && set -o pipefail && \
for F in 1 2 3; do \
  python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold${F}.pt \
    --order paper --fold $F --eval-tasks="0" --epochs 5 --router-consol pertask \
    2>&1 | tee outputs/oldtask_budget_paper_f${F}_task0__pertask.log ; \
  python train_router_v0.py --backbone-ckpt outputs/qpmil_paper_fold${F}.pt \
    --order paper --fold $F --eval-tasks="0" --epochs 5 --router-consol ewc --consol-lam 1000 \
    2>&1 | tee outputs/oldtask_budget_paper_f${F}_task0__ewc.log ; \
done && \
git add outputs/ && git commit -m "results(SPEC-02): paper-order per-task/EWC symmetry" && git push
```
跑完 Mac `git pull` 後重跑 `python tools/mechanism_table.py` 即補齊 paper-order 兩列。

## 5. 不做的事

- 不在 Mac 重跑訓練（per-task/EWC 補跑屬 RunPod stretch，本 SPEC 只備指令）。
- 不改既有 `collect_results.py` / `plot_results.py`。

## 6. Changelog

- 2026-06-27：建立 SPEC（M2 動工）。
