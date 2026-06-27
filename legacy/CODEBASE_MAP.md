# QPMIL-VL Codebase Map (M0, read-only)

針對 NaviPath-MoE 整合所做的 QPMIL-VL code 導覽。所有行號對應目前 repo 狀態。
特徵已預抽（CONCH image encoder，pre-computed），runtime 只跑 CONCH **text** encoder。

---

## 1. Model forward pass
- `models/model_il.py` → `QPMIL_VL.forward(x_list, eval=False)`（line 365）。
  - 輸入 `x_list = [Tensor(B=1, N, C=512)]`（一個 mini-batch 的多張切片）。
  - Vision branch：`_query_prototype_pool`（選 prototype）→ `_get_bag_feature`（聚合）。
  - Language branch：`tunable_v_learner(prompt_learner.class_ensemble_feature)` → enhanced class feature。
  - logits = `logit_scale * (bag_feature * enhanced_class_feature).sum(-1)`，shape `[MB, num_cls]`。
  - eval=True 回傳 `(logits, indices)`；train 回傳 `(logits, loss_dict, indices, feature_dict)`。

## 2. Prototype selection + prototype-guided aggregation
- 選取：`QPMIL_VL._query_prototype_pool`（line 275）。
  - query 向量 `q_vec = max/mean pool over patches`（`cfg['pooling']`，預設 max），normalize。
  - `merged_key`（[pool_size=M=20, C]）做 cosine sim → top-`match_size`（N=5）。
  - `task_num>1` 時用 `penalty_table`（頻率懲罰，line 259 `_get_penalty_table`）避免每張切片都選同一組。
  - 回傳 `indices [MB, N]`、`matching_loss`。
- 聚合：`QPMIL_VL._get_bag_feature`（line 327）。
  - 選中的 prompt → `text_encoder` → `prototype_features [MB, N, C]`（normalize）。
  - `scaled_cos_sim_matrix = csm_logit_scale * x_norm @ prototype_features.t()` → `[N_patch, N]`。
  - `W = softmax(.., dim=0)`（**這就是 patch→prototype 的指派矩陣 / 類 attention**）。
  - `bag_feature = mean(W.t() @ x)` → `[1, C]`，normalize → `[MB, 1, C]`。

## 3. Class text features / CFE (language branch)
- `PromptLearner`（line 131）：把 class ensemble prompt 經 `CONCHPromptEncoder` 編碼 → `class_ensemble_feature [num_cls, C]`（`__init__` 內算好，line 166）。
- `TunableVLearner`（line 17）：`enhanced = class_ensemble_feature + alpha * tunable_v`（line 33）。
- text encoder：`CONCHPromptEncoder`（line 41），來自 `base_model.text`。
- class ensemble 文字來源：`class_ensemble/class_ensemble.json`，由 `utils/tools.py::get_current_ensemble_classes`（line 109）依 **JSON 字典順序** 累積，遇到 `current_dataset` 才 break。
  - ⚠️ **順序耦合**：label index 與 `dataset_label_shift=[0,2,4,6]` 綁定 JSON 順序（lung,brca,rcc,esca）。做 reverse-order 時不能直接沿用，需自建 cumulative ensemble（runner 已自行處理）。

## 4. Training loop & losses (L_C, L_M, L_S)
- `manager/manager.py` → `Manager._update_network`（line 547）。
  - `total_loss = clf_loss + lambda * matching_loss + beta * class_sim_loss`（line 560）。
  - `clf_loss = F.cross_entropy(logits, label)`（L_C）。
  - `matching_loss`（L_M）來自 `_query_prototype_pool`；`class_sim_loss`（L_S）來自 `_compute_class_sim_loss`（line 356）。
  - mini-batch 累積 `bp_every_batch=16` 張切片才 backward；只有「被選中的 key/prompt」grad 非零，其餘清零（line 576-579）。
- per-epoch 迴圈：`Manager._run_training`（line 340）；單 epoch：`_train_each_epoch`（line 494）。
- ⚠️ 兩個本機相依：`torch.cuda.set_device` / `empty_cache`（line 469/534）在無 CUDA 會炸；`wandb.*` 全程呼叫；`EarlyStopping` 用 `np.Inf`（numpy 2.x 已移除）→ 故 runner 不複用 Manager，改自建 loop。

## 5. Continual task order & CL metrics
- task 序：`configs/main.yaml` → `dataset_names`（forward = lung,brca,rcc,esca）、`dataset_label_shift`、`dataset_subtype_num`。
- 增量迴圈：`Manager.incre_train`（line 75），每個 dataset 一個 task，`task_num` 累加。
- 評估：`eval_model`（line 433，class-IL，full logits argmax）、`_eval_all`（line 258）、masked task-IL（line 220）。
- accuracy 矩陣：`self.test_acc`（list-of-list）寫到 `metrics/test_acc.txt`；Forgetting/BWT 不在 repo 內算，靠 `eval_template/forward-order.xlsx` 彙整。
  - → 我們改用 `eval/metrics.py::summarize`（ACC / Forgetting / BWT / UpperBoundRatio）直接從 R 矩陣算。
- evaluator：`utils/evaluator_clf.py`（binary 用 `acc@mid` 門檻 0.5；⚠️ 用到 `np.long`）。

## 6. Insertion points（不重寫 QPMIL，只加 optional 層 / adapter）
4 個 hook（對應 `navipath_moe/model.py::NaviPathMoE` 的需求），由 `navipath_moe/qpmil_adapter.py::QPMILBackbone` 提供：

| Hook | QPMIL 對應 | shape | 備註 |
| --- | --- | --- | --- |
| `encode_patches(slide)` | dataloader 直給（無 learnable image encoder） | `Z [n,512]` | `slide.squeeze(0)` |
| `class_text_features()` | `tunable_v_learner(prompt_learner.class_ensemble_feature)` normalize | `f_txt [C,512]` | 訓練後 static |
| `prototype_features()` | 全 pool 的 prompt 經 `text_encoder` | `F_p [M=20,512]` | static，給 patch-budget 的 prototype 選法 |
| `aggregate_and_predict(Z_sub, f_txt)` | `_query_prototype_pool`+`_get_bag_feature`+logit | `(logits [1,C], loss_dict)` | 對 patch 子集跑真實 QPMIL 推論 |

其他插入點：
- micro router：`navipath_moe/routers.py::MicroRouterV0 / MicroRouter`（已 scaffolded，輸入 `Z,f_txt,F_p`）。
- MLP experts：`navipath_moe/experts.py::ExpertBank`（residual `e(z)=z+MLP(z)`）。
- semantic-anchor / load-balance loss：`navipath_moe/losses.py::l_sem / l_balance`。
- patch-budget eval：`eval/patch_budget_eval.py`（已對 4-hook 介面寫好）。
- replay-free momentum：`navipath_moe/consolidate.py`（task 邊界呼叫）。
- assignment matrix `W [n,N]`：QPMIL 內部沒回傳；adapter 端用 `prototype_features` 重算 `softmax(csm_logit_scale * z_norm @ F_p_sel.t(), dim=0)`（不改 QPMIL）。

## 7. Step-by-step integration plan (small commits)
1. M1：`train_qpmil_runner.py` thin runner（no wandb / device-agnostic / 自建 ensemble & R 矩陣）。
2. M2：`navipath_moe/qpmil_adapter.py` `QPMILBackbone`（4 hook + `forward_internals`）+ `debug_internals.py`。
3. M3：`run_patch_budget.py` 接 adapter → `eval/patch_budget_eval.py` 出 ACC@K 表。
4. （之後）M4 MicroRouter v0 → M5 v1+MoE+L_bal → M6 macro → M7 L_sem → M8 momentum → M9 viz。
