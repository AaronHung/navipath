# SPEC-07 — Zero-shot navigator baseline（`policy_mode="zero_shot"`）

- Status: Mac done（2026-06-28，smoke 通過）；真數字待 RunPod
- Milestone: 後續（回應 ZeroSlide / 老師 challenge 的關鍵 baseline）
- Related: SPEC-06, ADR-0006, docs/wiki/09

## 1. 目標（一句話）
在**同一個 CNL 架構**下，新增一條**不訓練 router** 的 navigation policy：patch 分數直接用 frozen backbone 的 CONCH patch-text 相似度（`sim_txt_max`）。用來回答老師/ZeroSlide 的問題：

> 「navigation（看哪）也能像 ZeroSlide 的分類那樣 zero-shot 就夠嗎？還是非得訓練 + 記憶（continual navigator）？」

天生**零訓練、零遺忘**，是 continual navigator 的對照組（強 baseline）。

## 2. 概念定位（兩條獨立軸）
- 訓練軸：**zero-shot**（不訓練，分數＝CONCH 文字相似度）↔ **continual**（訓練 router + NSM）。
- Agent 軸：**one-shot**（一次選 K）↔ **sequential**（多輪選 K）。
- 兩軸可自由組合 → 2×2。ZeroSlide 本人＝不訓練 + 不選（吃全部 patch）+ 只做分類，在此表之外（連 budget 都不要）。
- 重點：`zero-shot` 的「shot」＝訓練樣本數（0＝不訓練）；`one-shot`/`sequential` 的「shot」＝一張片內觀察輪數。**兩個 shot 不同義。**

## 3. 行為（最小改動，不是重寫）
**同一台車換引擎**：Observation State / budget / sequential 迴圈 / Context Gate 全部沿用，只換 `_base_score` 的來源。

1. `ContinualSequentialNavigationAgent` 新增建構參數 `policy_mode: str = "router"`（可選 `"zero_shot"`）。
2. `_base_score(Z, task_id)` 分支：
   - `"router"`（現狀）：`router(Z, f_txt, F_p)[0]`（訓練過的 MicroRouterV0 分數）。
   - `"zero_shot"`：**不查 skill bank、不建 router**，直接
     ```python
     from .routers import summary_feats
     f_txt = self.backbone.class_text_features().to(self.device)
     F_p = self.backbone.prototype_features().to(self.device)
     _, sim_txt_max = summary_feats(Z, f_txt, F_p)   # [n]
     return sim_txt_max
     ```
   （`summary_feats` 已回傳 `sim_txt_max = txt.amax(-1)`，即每 patch 對各類文字相似度的最大值。）
3. driver `eval_sequential_observation.py` 新增 `--policy-mode {router,zero_shot}`：
   - `zero_shot` 時**跳過所有 router 訓練**（不呼叫 `train_router_one_task`、不需 skill bank），直接對各 `--eval-task` 跑 observe→predict 算 acc。
   - 輸出檔名加 `policy=zeroshot` 標記，**不覆蓋** router 版結果。

## 4. 介面（CLI）
沿用 `eval_sequential_observation.py` 既有旗標（`--backbone-ckpt --order --fold --budgets --step-size --redundancy-weight --eval-task ...`），新增：
- `--policy-mode {router,zero_shot}`（預設 `router`）。

## 4b. RunPod 指令（真數字，不訓練 → 很快）
```bash
cd /workspace/src/navipath && git pull --ff-only && \
for t in 0 1 2 3; do \
  python eval_sequential_observation.py --backbone-ckpt outputs/qpmil_reverse_fold2.pt \
    --order reverse --fold 2 --eval-task $t --policy-mode zero_shot \
    --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
    2>&1 | tee outputs/seqobs_reverse_f2_task${t}_policy-zeroshot.log ; \
done
```
（與 router-mode 同 backbone/同 fold 並列比較；zero_shot 不需 `--epochs`/`--skill-bank-out`。）

## 5. 驗收標準
- [x] Mac：import OK、ruff/lints 通過；fresh backbone smoke：zero_shot 模式跳過訓練、observe→predict 端到端、輸出 `seqobs_*_policy-zeroshot.json`（fresh backbone 數字為隨機，僅驗管路）。
- [ ] RunPod/Mac：對 esca/rcc/brca/lung 各 budget 產 `outputs/seqobs_*_policy-zeroshot_*.json`。
- [ ] 對比表：同 budget、同 fold 下 zero-shot navigator vs continual navigator（router+NSM）acc 並列。
- [ ] 不覆蓋既有 router-mode 結果。

## 6. 解讀（寫進報告/論文）
- continual > zero-shot → 證明「訓練 + 記憶的 navigation」有價值，遺忘問題值得解。
- 打平 → 誠實報告；仍貢獻「navigation 維度的 CL 分析」＋「一個強 zero-shot baseline」，且把問題問清楚。
- 因為 zero-shot navigator **天生零遺忘**，它同時是 retention 表的「無遺忘參考線」。

## 7. 不做的事
- 不改 `routers.py` / `train_router_v0.py`（只 import `summary_feats`）。
- 不實作 task-free gate（zero-shot navigator 與 gate 正交；Phase-1 再合）。
- 不在 zero-shot 模式做任何反向傳播。

## 8. Changelog
- 2026-06-28：建立 SPEC（概念定版 + 最小改動實作路徑）。
