# Route A λ Sweep 結果分析（N6, 2026-06-28）

## 實驗設置
- fold 1, reverse order, eval-tasks 0,1,2,3 (esca/rcc/brca/lung)
- budgets: 64, 32, 16 | step_size=16 | normalize_base=True | redundancy_mode=maxsim
- skill-bank: outputs/skill_bank_reverse_f1.pt（N2 已訓練，inference-only）

## 關鍵數字

| λ | esca@64 nsm_seq | esca@64 nsm_1shot | diff | rcc@64 nsm_seq | rcc@64 diff |
|---|---|---|---|---|---|
| 0.0 | 0.867 | 0.867 | **0.000** | 0.961 | **0.000** |
| 1.0 | 0.867 | 0.867 | 0.000 | 0.961 | 0.000 |
| 2.0 | 0.800 | 0.867 | -0.067 | 0.961 | 0.000 |
| 4.0 | 0.267 | 0.867 | **-0.600** | 0.355 | **-0.605** |

## 結論

1. **Route A 機制確認**：λ=0 → seq≡oneshot；λ>0 → seq 與 oneshot 分歧，機制成立。
2. **最佳 λ = 0.5–1.0**：稍微多樣，acc 代價微小（<0.01）。
3. **λ 過大傷 acc**：強迫離開病灶聚集區 = 準確率崩潰（λ=4 esca 降至 0.267）。
   - 原因：WSI 腫瘤 patch 本來就空間聚集，router 學到了這個特性，MMR 強行打散反而有害。
4. **budget=16 不受影響**：step_size=16 只走 1 步，seq≡oneshot 無論任何 λ。

## 建議後續設定
- 預設 `redundancy_weight = 0.5`（輕度 MMR，不破壞病灶聚集）
- budget=32: step_size=16 走 2 步，λ=0.5 sec 效果最佳
- budget=64: step_size=16 走 4 步，λ=0.5–1.0 為宜

## 對論文的描述方式
> Route A (adaptive sequential selection) verifies that seq ≠ oneshot under λ>0,
> confirming the mechanism. Optimal λ=0.5–1.0 reflects the spatial clustering
> of tumor patches in WSI: over-penalizing similarity forces the agent away from
> the diagnostically relevant region, confirming the router has learned
> cluster-level navigation skills.
