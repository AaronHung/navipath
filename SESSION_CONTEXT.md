# NaviPath-MoE 研究上下文（給新 session 讀）

## 論文目標
COMPAYL 2026（MICCAI satellite），截稿 2026-07-01。
方法：NaviPath-MoE = QPMIL-VL backbone + Agentic Macro/Micro Routing + Replay-free Consolidation。
主賣點：逆序任務（reverse-order）+ patch budget + replay-free + routing drift。
完整計劃見：`Navipath_moe_plan_v01.md`、`KICKOFF_PLAYBOOK.md`

## 當前進度（M0-M4 全部完成，已 commit）
- M0：QPMIL codebase map → `CODEBASE_MAP.md`
- M1：thin runner → `train_qpmil_runner.py`（paper/reverse 兩方向，存 ckpt）
- M2：backbone adapter → `navipath_moe/qpmil_adapter.py`（4 hook + forward_internals）
- M3：patch-budget eval → `run_patch_budget.py`（random/prototype/semantic ACC@K）
- M4：router training → `train_router_v0.py`（MicroRouterV0，soft-weighted 訓練，hard top-K 評估）

## 現在正在跑（MPS，Terminal，整晚）
```bash
python train_qpmil_runner.py --order paper   --fold 1 --save-ckpt 2>&1 | tee outputs/run_paper_f1.log && \
python train_qpmil_runner.py --order reverse --fold 1 --save-ckpt 2>&1 | tee outputs/run_reverse_f1.log && \
python run_patch_budget.py --ckpt outputs/qpmil_paper_fold1.pt --order paper --task-index 0 2>&1 | tee outputs/budget_lung.log && \
python train_router_v0.py  --backbone-ckpt outputs/qpmil_paper_fold1.pt --order paper --fold 1 --epochs 5 2>&1 | tee outputs/router_v0_f1.log
```

## 明天要做的事
1. 把 `outputs/run_paper_f1.log`、`outputs/router_v0_f1.log` 最後幾行貼給新 session。
2. 新 session 確認 signal 後，決定：
   - GO → 繼續 M5（MoE expert + L_bal）
   - NO-GO → 退安全稿（QPMIL + semantic router + budget 分析）

## 關鍵設計決策（供新 session 參考）
- 不用 QPMIL 的 Manager（綁 wandb/cuda/np.Inf），自建 thin runner。
- transformers 鎖在 4.x（CONCH tokenizer 需 batch_encode_plus，5.x 已移除）。
- Router 訓練用 soft weighted aggregation（可微分），評估用 hard top-K。
- qpmil_adapter 的 prototype_features() 用 `torch.cat([pl.prompt[i] ...])` 而非 merge_parameter（後者需 opt_name key）。
- label shift = 2 * task_pos（每任務 2 類，class-incremental）。
- cumulative ensemble 自建（QPMIL 原本依 JSON 字典序，reverse order 會對錯）。

## 關鍵檔案路徑
- 資料：`data/` → symlink to `/Users/aaron/research/can_dataset`
- CONCH 權重：`checkpoints/conch/pytorch_model.bin`（765MB）
- QPMIL repo：`QPMIL-VL/`（不入 git）
- 我們的 code：`navipath_moe/`、`eval/`、`configs/`、`tests/`
- RunPod SOP：`RUNPOD_SOP.md`

## 資料規模
lung 1054 / brca 1133 / rcc 937 / esca 158 slides，patch ~3000/slide，512 dim CONCH feat。
Task order paper = lung→brca→rcc→esca；reverse = esca→rcc→brca→lung。
Label shift = [0,2,4,6]（class-incremental，不給 task id）。

## 下一步 Milestone（M5 起）
M5：MicroRouter v1（expert weights）+ ExpertBank（MLP residual）+ L_bal → `navipath_moe/experts.py` 已 scaffold。
M6：MacroRouter + fusion（`beta * macro + (1-beta) * micro`）。
M7：L_sem（KL 讓 router 被 CONCH 語義空間約束）。
M8：replay-free momentum consolidation（`navipath_moe/consolidate.py` 已 scaffold）。
M9：routing drift 可視化。
