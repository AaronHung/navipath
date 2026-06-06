# NaviPath-MoE — repo skeleton

Agentic Macro/Micro Routing for Continual WSI Classification（COMPAYL 2026）。
骨架：可獨立 import/測試的部分已實作並通過自測；標 `TODO` 處接
[`can-can-ya/QPMIL-VL`](https://github.com/can-can-ya/QPMIL-VL) 的 backbone 與 data loader。

## 重點設定（本版）
- **Expert = 小 MLP 殘差**（`e(z)=z+MLP(z)`，非 LoRA）。在 512 維 CONCH 特徵上最簡單可跑。
- **Device-agnostic**：`get_device()` 自動 `cuda > mps > cpu`。Mac M1 開發、RunPod CUDA 跑重活。
  MPS 已自動開 `PYTORCH_ENABLE_MPS_FALLBACK`，並一律 float32。
- **任務序**：`paper`（NSCLC→BRCA→RCC→ESCA，對齊 QPMIL 已發表數字當 sanity）與
  `reverse`（ESCA→RCC→BRCA→NSCLC，主戰場，forgetting 重現）。資料目錄名：
  `tcga_lung / tcga_brca / tcga_rcc / tcga_esca`。
- **分階段開關**：`use_experts` / `use_macro` / `use_consolidation`，對應 v0→v4 開發路線。

## 結構
```
navipath_moe/
  device.py        Mac MPS / CUDA 共用                              ✅
  routers.py       summary_feats + MicroRouterV0/MicroRouter/Macro  ✅
  experts.py       MLP expert bank（殘差、零初始化）                 ✅
  consolidate.py   replay-free 雙重要度動量鞏固                      ✅
  losses.py        L_sem(維度乾淨) / L_bal / L_route                ✅
  model.py         NaviPathMoE：插進 QPMIL aggregation 之前          🔌 接 backbone
eval/
  metrics.py            ACC/Forgetting/BWT/UpperBoundRatio          ✅
  patch_budget_eval.py  random/prototype/semantic/router 選擇        🔌 接 backbone
configs/
  qpmil_sanity.yaml      Milestone 1：純 QPMIL（關掉所有 router/expert）
  navipath_micro.yaml    v1：micro + MLP MoE（無 macro/momentum）
  navipath_macro_micro.yaml  v3：+ macro
  navipath_full.yaml     v4：+ replay-free momentum
train_continual.py   CL 主迴圈（device-aware + consolidation + accuracy matrix + budget）🔌
tests/test_shapes.py 形狀自測
smoke_test.py        一鍵自測
```

## 接 QPMIL 的 4 個 hook（給坤倫，見 model.py）
```python
backbone.encode_patches(slide)           -> Z   [n, 512]   # 已是 CONCH features
backbone.prototype_features()            -> F_p [N, 512]
backbone.class_text_features()           -> f_txt [C, 512]
backbone.aggregate_and_predict(Z, f_txt) -> (logits, {"L_C":, "L_M":, "L_S":})
```

## 自測（不需 backbone / 不需 GPU）
```bash
pip install -r requirements.txt
python smoke_test.py
PYTHONPATH=. python tests/test_shapes.py     # 或: pytest tests/ -q
python -m eval.metrics
```

## 開發路線（v0→v4，見 KICKOFF_PLAYBOOK.md）
| 版本 | config | 內容 | 用途 |
| -- | -- | -- | -- |
| v0 | qpmil_sanity | QPMIL + patch-budget eval | sanity / 第一張表 |
| v1 | navipath_micro | + MicroRouter + MLP MoE + L_bal | 安全稿 |
| v2 | navipath_micro | + L_sem + routing drift | 可投稿 |
| v3 | navipath_macro_micro | + Macro router | 強方法 |
| v4 | navipath_full | + replay-free momentum | 完整 NaviPath-MoE |
