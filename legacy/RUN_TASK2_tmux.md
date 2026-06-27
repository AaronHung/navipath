# TASK 2 — tmux 跑法（出差不斷線版）

> 目的：在 RunPod 用 tmux 跑，detach 後可關電腦/斷網/出差，程式繼續跑、跑完自己 push。
> 安全：防覆蓋已內建（檔案存在即 skip），重跑不會弄亂資料。

## 步驟（WezTerm SSH 進 RunPod 後，依序貼）

### 1) 確保 tmux 存在 + 開 session
```bash
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
tmux new -s task2
```

### 2) 在 tmux 裡貼這整段（會自己跑完並 push）
```bash
cd /workspace/src/navipath && git pull --ff-only && set -o pipefail && \
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold1.pt --order reverse --fold 1 --eval-tasks="-1,0" --epochs 5 2>&1 | tee outputs/router_v0_reverse_fold1.log && \
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt --order reverse --fold 2 --eval-tasks="0" --epochs 5 2>&1 | tee outputs/oldtask_budget_reverse_f2_task0.log && \
python train_router_v0.py --backbone-ckpt outputs/qpmil_reverse_fold3.pt --order reverse --fold 3 --eval-tasks="0" --epochs 5 2>&1 | tee outputs/oldtask_budget_reverse_f3_task0.log && \
echo "=== RUNS DONE, pushing ===" && \
git add outputs/ && \
git commit -m "results(v0.4): router reverse f1 + old-task budget reverse f1-3 task0" && \
git push && \
echo "=== TASK 2 PUSHED ==="
```

### 3) Detach（讓它在背景跑，然後可關電腦）
按 `Ctrl+b`，放開，再按 `d`。回到一般畫面後即可關電腦出門。

---

## 回來後

```bash
# 重連看狀態
tmux attach -t task2
# 看到 "=== TASK 2 PUSHED ===" 就完成了；Ctrl+b d 再離開
```
或直接在 Mac `git pull`（結果已在 GitHub），跟 Cursor 說「跑完了」即可。

---

## 萬一（少見）

- **git push 要帳號/密碼**（換了新 pod）：
  ```bash
  git config user.email "you@example.com" && git config user.name "Aaron"
  git remote set-url origin https://<YOUR_GITHUB_TOKEN>@github.com/AaronHung/navipath.git
  ```
  設好後在 tmux 裡單獨補一次 `git add outputs/ && git commit -m "results(v0.4) task2" && git push`。
- **想中途看進度**：`tmux attach -t task2`（看完 `Ctrl+b d` 離開，別直接關視窗裡的程式）。
- **跑完想清掉 session**：`tmux kill-session -t task2`。
