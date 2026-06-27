# SPEC-06 — Continual agent end-to-end eval (`eval_continual_agent.py`)

- Status: Mac done (2026-06-27); 真數字待 RunPod
- Milestone: M3 (6/30) ★ 核心交付
- Related: SPEC-01, ADR-0003

## 1. 目標

用 `ContinualWSINavigationAgent`（oracle gate + NavigationSkillBank）跑 end-to-end，**重現既有 per-task 數字**（reverse old ESCA@64 ≈ 0.933），證明 agent 包裝正確：agent 的 navigate→predict 等價於 train_router_v0 的 per-task eval（oracle gate 選 skill_bank[old_task]）。

## 2. 行為

root 腳本（與 `train_router_v0.py` 同層，可延遲 import），流程：
1. 建 frozen backbone（`build_backbone_from_ckpt` 或 fresh smoke）。
2. 逐任務用既有 `train_router_one_task` 訓練單一 router；**每學完一個 task 即把 router snapshot 存進 `NavigationSkillBank`**（= per-task skill）。
3. 可選 `--skill-bank-out` 持久化 skill bank。
4. 對 `--eval-task` 用 `ContinualWSINavigationAgent(backbone, bank, ContextGate("oracle"))`：遍歷該 task test loader，對每張 slide `agent.predict(Z, k, task_id=eval_task)`，算各 budget 的 agent router acc。
5. 輸出 `outputs/agent_oldtask_{order}_f{fold}_task{t}.json`；若同條件的 `oldtask_budget_*__pertask.json` 存在，印**一致性檢查**（agent acc vs pertask json）。

## 3. 介面（CLI）

沿用 train_router_v0 風格：`--qpmil-config --order --fold --backbone-ckpt --epochs --top-k --budgets --lr --max-train --max-eval --out`，新增 `--eval-task`（要評估的舊任務 index）、`--skill-bank-out`。

## 4. 驗收標準

- [x] Mac：`--help` OK、import OK、ruff All checks passed。
- [x] Mac pipeline smoke（fresh backbone，`--max-train 4 --max-eval 2 --epochs 1`）：4 task 逐一訓練（loss 1.87→0.06）、skill bank 存 4 skill、oracle gate agent navigate→predict 端到端成功（fresh backbone 數字為隨機值，僅驗 pipeline）。
- [ ] RunPod：用 `outputs/qpmil_reverse_fold2.pt` 重現 old ESCA@64 ≈ 0.933，與 `oldtask_budget_reverse_f2_task0__pertask.json` 一致（±0.01）。指令見 §6。

## 5. 不做的事

- 不實作 task-free gate（future）。
- 不改既有 `train_router_v0.py` / `routers.py` / `qpmil_adapter.py`（只 import 重用）。

## 6. RunPod 重現指令（tmux）

```bash
cd /workspace/src/navipath && git pull --ff-only && \
python eval_continual_agent.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt \
  --order reverse --fold 2 --eval-task 0 --epochs 5 \
  --skill-bank-out outputs/skill_bank_reverse_f2.pt \
  2>&1 | tee outputs/agent_oldtask_reverse_f2_task0.log
```

## 7. Changelog

- 2026-06-27：建立 SPEC（M3 動工）。
