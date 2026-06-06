# NaviPath-MoE 開工手冊（Cursor / Claude Code prompt playbook）

這份是「拿去貼」的操作手冊。每個 Milestone 都附：① 程式 prompt ② 測試 prompt
③ debug 指令 ④ 跑實驗指令 ⑤ ablation。骨架程式在 `navipath_moe/`，QPMIL 整合
hook 見 `model.py`。**第一優先級不是把 MoE 寫完，而是先跑出 patch-budget 表**——
那張表最早告訴你 agentic navigation 故事有沒有分數。

---

## 0. 開發環境與鐵則

**環境**：Mac M1 16G 開發（MPS），RunPod CUDA 跑重活，同一份 code。

- 一律 `from navipath_moe import get_device, setup_mps; setup_mps(); dev = get_device()`。
- 資料很小（單張切片 ≤8000 patch ×512×4B ≈16MB），**M1 16G 跑單 fold 完全夠**。
- **RunPod 只拿來跑「平行吞吐」**：10-fold CV、多 seed、超參 sweep。開發/除錯都在 M1。
- MPS 坑：float32（勿 float64）、`setup_mps()` 已開 fallback。

**和 Cursor/Claude 協作的鐵則**（每個任務都要求）：
1. 先讀 code、不要改；2. 說清楚改哪些檔案；3. 小步 patch；4. 附 shape test；
5. 跑 debug 指令；6. **不要重寫 QPMIL 主流程**，只加 optional router/MoE layer 並能 return debug outputs。

**開發路線 v0→v4**（每一級都能止損成稿）：
`v0` QPMIL+budget eval → `v1` +MicroRouter+MLP MoE+L_bal → `v2` +L_sem+drift →
`v3` +Macro → `v4` +replay-free momentum。

---

## M0 — Codebase map（只讀，不改）

**Prompt（貼給 Cursor/Claude）：**
```
You are helping me adapt the existing QPMIL-VL codebase (can-can-ya/QPMIL-VL) for
continual WSI classification. Do NOT edit any files yet. Inspect the repo and produce
a concise codebase map covering:
1. Where the model forward pass is implemented.
2. Where prototype selection and prototype-guided aggregation are implemented.
3. Where class text features / CFE are implemented.
4. Where the training loop computes losses (L_C, L_M, L_S).
5. Where the continual task order and the CL metrics (ACC, Forgetting, BWT) live.
6. Minimal insertion points to add, without rewriting QPMIL: an optional micro router,
   MLP experts, semantic-anchor loss, load-balance loss, patch-budget evaluation,
   replay-free momentum consolidation.
For each insertion point give the exact file path, function/class name, and expected
tensor shapes. End with a step-by-step plan of small commits.
```

---

## M1 — 跑通 QPMIL backbone（paper + reverse sanity）

目標：確認資料、訓練、metric 都對，對齊論文數字（paper ~0.890 / reverse ~0.859）。

**Code prompt：**
```
Make the QPMIL-VL repo runnable on my data and verify it reproduces the paper trend.
My CONCH features live under data_root with dirs: tcga_lung, tcga_brca, tcga_rcc,
tcga_esca (feature dim 512; ~3000 patches/slide; counts 1054/1133/937/158).
Tasks: paper order = lung->brca->rcc->esca (NSCLC first, ESCA last);
reverse = esca->rcc->brca->lung. Each task adds 2 classes (class-incremental).
Add a thin runner that (a) loads one fold/one seed, (b) trains QPMIL sequentially,
(c) outputs a per-task accuracy matrix R[t,i] and dumps ACC/Forgetting/BWT.
Make device selection automatic (cuda>mps>cpu) and use float32 for MPS compatibility.
Do not change QPMIL's model logic; only add the runner + data adapter.
```

**Debug 指令：**
```bash
# 先驗單張切片特徵 shape，再跑 1 fold
python -c "import torch; x=torch.load('data_root/tcga_esca/<some_slide>.pt'); print(x.shape, x.dtype)"
python train_qpmil_runner.py --order paper   --fold 0 --seed 0   # 你/Cursor 建立的 runner
python train_qpmil_runner.py --order reverse --fold 0 --seed 0
```

**判準**：per-task accuracy matrix 印得出來；paper-order ACC 落在 ~0.85–0.90、
reverse 略低；code 可改不崩。對不上先查 task order / class 累加 / metric 公式。

---

## M2 — QPMIL forward 回傳 internals（所有創新的地基）

**Code prompt：**
```
Add an optional `return_internals=True` mode to the QPMIL forward. Default behavior
must be byte-for-byte unchanged. When enabled, also return a dict with as many of:
logits, bag_feature f_b, patch_features Z [n,512], query_vector,
selected_prototype_indices, selected_prototype_features F_p [N,512],
patch_prototype_similarity, assignment_matrix W [n,N], class_text_features f_txt [C,512],
and loss components {L_C, L_M, L_S}. Add shape comments. Add a debug script that runs
one mini-batch and prints every returned key with its shape. If something isn't directly
available, return the closest equivalent and note where it's computed. Keep it minimal
and backward compatible.
```

**Debug 指令：**
```bash
python debug_internals.py   # 印出 {key: shape}；確認 Z[n,512]、F_p[N,512]、f_txt[C,512]、W[n,N]
```

**判準**：拿得到 `Z / F_p / f_txt / W`。拿不到這四個，後面 micro router / patch-budget /
routing drift 全做不了。

---

## M3 — Patch-budget eval（最早出表，不訓練、最安全）★第一優先

目標：用我們 `eval/patch_budget_eval.py` 的 random / prototype / semantic 三種選法，
只用 Top-K patch 跑 QPMIL inference，出第一張表。

**Code prompt：**
```
Wire navipath_moe/eval/patch_budget_eval.py to the QPMIL backbone (it needs
encode_patches, prototype_features, class_text_features, aggregate_and_predict — map
these to the QPMIL functions from M2). For each WSI, select Top-K patches by
{random, prototype-sim, semantic-sim}, run QPMIL inference on only those patches, report
ACC@K for K in {All,256,128,64,32}. Print a table: method, K, ACC, n_slides, avg_patches.
Do NOT implement MoE or any router yet. Reuse QPMIL inference; do not retrain.
```

**跑實驗指令：**
```bash
python -m eval.patch_budget_eval --config configs/qpmil_sanity.yaml --order paper
```

**判準（決定 patch-budget 故事有沒有戲）**：K 小時 prototype/semantic 是否明顯贏 random？
若 ACC@64(semantic) ≫ ACC@64(random)，agentic selective observation 故事成立。

**Sanity（對方 Test 5）**：random 的 ACC 應隨 K 單調上升（@32<@64<@128<@256<@All）。
不單調 → Top-K indexing 或 patch 子集沒正確送進 QPMIL。

---

## M4 — MicroRouter v0（純 patch score）+ go/no-go

目標：訓練一個 patch 評分器，看它能不能比 heuristic 更會選 patch。**這一步是 Day-16 的 go/no-go。**

**Code prompt：**
```
Implement MicroRouterV0 (already scaffolded in navipath_moe/routers.py) end-to-end:
input per patch = [z_i (512) ; summary_feats (4): max_text_sim, entropy(text_sim),
max_proto_sim, mean_proto_sim] -> scalar importance score. Use it to select Top-K
patches, run QPMIL on the selected patches, and train ONLY the router + (optionally)
the QPMIL head with the slide-level classification loss. Keep QPMIL backbone frozen
where possible. Add a config flag to compare router-Top-K against the M3 heuristics.
Do NOT add MoE experts or macro router yet.
```

**測試 prompt / shape test：**
```bash
PYTHONPATH=. python tests/test_shapes.py    # test_micro_v0 必過
```

**跑實驗指令：**
```bash
python train_router_v0.py --config configs/navipath_micro.yaml --order paper --fold 0
```

**go/no-go 判準（任一綠燈即可往下）**：
- Router@64 明顯 > Random@64；
- Router@128 ≳ Semantic@128；
- 或 reverse-order 下 Router 的 Forgetting 明顯較低。

連 patch selection 都沒 signal → 退安全稿（QPMIL + 此 router + budget + 分析），不做 MoE。

---

## M5 — MicroRouter v1 + MLP MoE experts + L_bal

**Code prompt：**
```
Upgrade to MicroRouter (v1): output [n,E] expert weights (softmax). Use ExpertBank
(navipath_moe/experts.py, MLP residual experts e_j(z)=z+MLP_j(z), hidden=256, E=4).
Fuse: z_tilde_i = sum_j w[i,j]*expert_j(z_i); feed z_tilde back into QPMIL aggregation.
Add load-balance loss l_balance (navipath_moe/losses.py): loss = qpmil_loss + eta*l_bal.
Ensure the optimizer includes micro router + experts params. Add shape tests and an
overfit-4-slides debug. Do NOT add macro router or momentum yet.
```

**測試（對方 Test 1/2/3/4）：**
```bash
PYTHONPATH=. python tests/test_shapes.py            # test_micro_v1, test_topk_and_experts
# overfit-4：train on 4 slides for 200 steps，loss 應明顯下降、train ACC→~100%
# param check：印 requires_grad 參數，需看到 micro.* 與 experts.*
# collapse check：每 task 後印 w.mean(0)；不應是 [0.99,0,0,0.01]，要分散
```
**Overfit debug prompt：**
```
Add a debug routine: train NaviPath-MoE (micro+experts) on 4 slides for 200 steps and
print loss + train ACC each 20 steps. Also print all requires_grad params with shapes,
and per-step expert usage = w.mean(dim=0). Expected: loss clearly decreases, ACC->~100%,
expert usage not collapsed to one expert.
```

**跑實驗指令：**
```bash
python train_continual.py --config configs/navipath_micro.yaml --order paper --fold 0
```
collapse 時：調高 `eta`(L_bal)、降 router lr、減 `num_experts`。

---

## M6 — Macro router + fusion

**Code prompt：**
```
Add MacroRouter (slide-level mean-pooled feature -> [E]) and fuse with micro:
w_i = beta*w_macro + (1-beta)*w_micro_i (beta=0.3, fixed first; sweep later).
Set use_macro=true. Keep everything else identical. Add a config navipath_macro_micro.
Report whether macro improves over micro-only on reverse-order; if not, keep micro-only.
```
**跑實驗指令：**
```bash
python train_continual.py --config configs/navipath_macro_micro.yaml --order reverse --fold 0
```

---

## M7 — Semantic anchor loss L_sem

**Code prompt：**
```
Add semantic-anchor loss l_sem (navipath_moe/losses.py): on the patch importance
distribution. pi = softmax(router score over patches); q = softmax(max-class-text-sim
over patches); loss += gamma * KL(q || pi). gamma=0.5. Verify it's a 0-dim finite scalar
and that turning it on does not collapse routing. Add an ablation flag gamma in config.
```
**Ablation：** `gamma ∈ {0, 0.25, 0.5, 1.0}`，看 reverse-order Forgetting 與 routing drift。

---

## M8 — Replay-free momentum consolidation（最後加）★主戰場

**Code prompt：**
```
Wire replay-free dual-importance consolidation (navipath_moe/consolidate.py) into the
between-task boundary in train_continual.py (already scaffolded). After each task:
I_cur = normalized avg router usage; I_old = EMA(I_cur history); m_e = sigmoid(a*I_old -
b*I_cur); theta_e <- m_e*theta_e_old + (1-m_e)*theta_e_new. Store NO past slides/patches.
Keep it as a standalone function called between tasks, not inside the training step.
Log m_e per task. Provide a config flag use_consolidation.
```
**跑實驗指令（主戰場 reverse）：**
```bash
python train_continual.py --config configs/navipath_full.yaml --order reverse --fold 0
python train_continual.py --config configs/navipath_full.yaml --order paper   --fold 0
```
**子 fallback**：momentum 沒幫助 → 退 v3（macro+micro+L_bal+L_sem），momentum 退成 ablation 一列。

---

## M9 — 可視化：routing drift + expert usage

**Code prompt：**
```
Add two analysis scripts. (1) routing_drift: for fixed old-task slides, record the
router's selected Top-K patch set after training task 1 vs after task T; report overlap
/ rank correlation, and overlay selected patches on the slide thumbnail (ours vs QPMIL
vs fine-tune). (2) expert_usage: bar chart of per-task expert usage w.mean(0) showing
specialization (not collapse). Save figures to outputs/figs/.
```

---

## 測試策略（每步必跑，省 debug 時間）

1. **Shape test**：`PYTHONPATH=. python tests/test_shapes.py`（先測形狀，別先跑 full train）。
2. **Overfit-4**：4 張切片 200 步，loss 要降、ACC→~100%。降不了 = loss 沒接 / label 錯 /
   features 錯 / optimizer 沒含新參數 / router 無梯度。
3. **Param check**：印 `requires_grad` 參數，要看到 `micro.* / experts.* / macro.*`。
4. **Collapse check**：每 task 後 `w.mean(0)`，不應 `[0.99,0,0,0.01]`。
5. **Budget 單調**：random ACC 隨 K 上升；不單調 = indexing / 子集 / label / eval-mode 問題。

---

## 實驗矩陣與 RunPod

| 項目 | 在哪跑 |
| -- | -- |
| 開發、單 fold、overfit、shape test、第一張 budget 表 | **Mac M1**（足夠） |
| 10-fold CV × 多 seed（對齊 QPMIL 的 10-fold 協定） | **RunPod CUDA**（平行） |
| 超參 sweep：`num_experts∈{4,6}`、`beta`、`gamma`、`eta`、`K` | **RunPod** |
| 主表（paper sanity + reverse 主戰場）、patch-budget 表、ablation | M1 出初版 → RunPod 補滿 fold/seed |

**主要 ablation（對齊研究計劃 §7.6）**：
```
QPMIL → +Micro → +Macro → +L_sem → +L_bal → +Momentum
另：num_experts∈{4,6}、beta sweep、gamma sweep、patch budget K sweep
```

---

## 今天先做 4 件事（不要先寫 NaviPath）

1. **跑 QPMIL debug run**：1 fold / 1 seed / paper order，輸出 per-task accuracy matrix（M1）。
2. **讓 QPMIL forward 回傳 internals**（M2）——所有創新的地基。
3. **寫 patch-budget eval**：random / prototype / semantic 三選法，出第一張表（M3）。
4. **寫 MicroRouter v0**：先只輸出 patch score、接進 budget eval，不接訓練（M4 的前半）。

> 第 3 步很可能今天就能出第一張表，最早告訴你 agentic navigation 故事有沒有分數。
