# HANDOFF PROMPT — NaviPath-CL 答辯思路深入 session

> **怎麼用**：開一個新的 Cursor Agent chat，把「下面分隔線之間的全部內容」貼進去當第一則訊息即可。
> 新 agent 會據此接手，陪你一步一步深入老師的問題、逐條核對公式、整理答辯（rebuttal）思路。

---

（以下整段複製）

我是 NaviPath-CL 專案的作者，正在準備雙月報告 / 論文答辯。這個 session 的目標是：**逐條深入指導教授提出的問題，把我們的公式、algorithm、實驗一步一步問清楚，整理出可上台自洽的答辯思路。** 請你當我的答辯教練，用蘇格拉底式一問一答，不要一次倒完；每次聚焦一個點，等我回應再往下。

回覆語言規則：我用繁體中文你就用繁體中文答；英文專有名詞保留原文，必要時括號補中文。公式要精確，教授非常在意每個符號怎麼來的。

## 先讀這些檔案（湯底 + 事實來源，請務必先讀再回答）

**背景湯底（先讀，理解專案基調與禁區）**
- `SESSION_CONTEXT.md` — 專案接手文件（pivot 到 NaviPath-CL 的基調、命名、Do-not-claim 清單、oracle gate 風險）
- `COLLAB_PLAYBOOK.md` — 合作規矩（Mac MPS 構思 → RunPod CUDA tmux 跑重活、git/成本紀律）
- `STORYLINE.md` — 目前拍板的故事線

**核心答辯素材（本輪剛做好，最重要）**
- `paper/Teacher_QA_and_Formulas.md` — 老師 10 個問題的逐題回答（每題【說人話】+【正式 ZH/EN】+ 精確公式），**這是本 session 的主軸，優先精讀**
- `paper/NaviPath-CL_draft_v1.md` — 論文全文（§3 Method 含 Eq. 1–11；§4 Experiments 含所有表格），**公式的權威來源**
- `experiment_visualize/experiments.md` — 3 slides 版實驗故事（含 talking points）
- `experiment_visualize/tables.tex` — 5 張 LaTeX 表格（每張有 presenter 說明）
- `experiment_visualize/figs/fig_arch_train_infer.pdf` — training/inference 雙 panel 架構圖

**程式碼（公式 ↔ 實作對照的事實來源，回答公式問題前請對照）**
- `navipath_moe/routers.py` — `MicroRouterV0`（Eq.2 的 MLP）與 `summary_feats`（Eq.1 的 4 維摘要）
- `train_router_v0.py::train_router_one_task` — soft-route 訓練（Eq.3 的 CE loss、梯度只走 softmax weight）；`RouterEWC`（Eq.11）
- `navipath_moe/sequential_observation.py` — SBO（Eq.4 z-score、Eq.5 MMR、Eq.6 增量更新、Eq.7 選取、Eq.8 早停）
- `navipath_moe/continual_agent.py` — `NavigationSkillBank`（Eq.9 NSM）、`ContextGate`（**注意：只有 oracle，`infer()` 是 NotImplementedError**）
- `navipath_moe/qpmil_adapter.py` — 4 個 backbone hooks（`class_text_features`=Eq 的 T、`prototype_features`=P）
- `eval_sequential_observation.py` — seqobs 評估主程式
- `eval_weight_avg_baseline.py` — 本輪新增的 weight-averaged baseline（老師 Q9 要求的中間對照）

## 公式速覽（權威版見 paper §3；符號務必一致）

- $Z\in\mathbb{R}^{n\times512}$：凍結 CONCH patch features；$\hat{z}$=L2-normalize。$T=\{t_c\}_{c=1}^{C}$ class-text features；$P=\{p_m\}_{m=1}^{M}$ prototypes；$\phi$ router 參數；$g_{\theta^*}$ 凍結診斷 backbone。

- **Eq.1（4 維摘要）**：$s_i=[\max_c \hat{z}_i^\top\hat{t}_c,\ H(\mathrm{softmax}(\hat{z}_i^\top\hat{t})),\ \max_m \hat{z}_i^\top\hat{p}_m,\ \tfrac1M\sum_m \hat{z}_i^\top\hat{p}_m]$
- **Eq.2（base score）**：$r_i=\mathrm{MLP}_\phi([z_i;s_i])$，MLP: $\mathbb{R}^{516}\to\text{Linear}\to256\to\text{GELU}\to\text{Linear}\to1$（≈132K params）
- **Eq.3（soft-route CE loss）**：$w_i=\mathrm{softmax}_{\mathcal{S}_K}(r_i)$，$\bar z=\widehat{\sum_{i\in\mathcal{S}_K}w_i z_i}$，$\mathcal{L}_{route}=\mathrm{CE}(\sigma\bar z T^\top, y)$；梯度只經 $w_i$，不經凍結 backbone
- **Eq.4（z-score，單調不改 one-shot top-K）**：$\tilde r_i=(r_i-\mu_r)/(\sigma_r+\epsilon)$
- **Eq.5（MMR）**：$a_i^{(t)}=\tilde r_i-\lambda\max_{j\in\mathcal{S}^{(\leq t-1)}}\cos(z_i,z_j)$；已選者設 $-\infty$
- **Eq.6（增量更新，$O(n\cdot k_{step})$/round）**、**Eq.7（每輪選 $k_{step}$）**、**Eq.8（信心早停 route B）**
- **Eq.9（NSM）**：$\text{NSM}=\{\phi^{(1)},\dots,\phi^{(T)}\}$，oracle gate 檢索
- **Eq.10（zero-shot nav）**：$r_i^{zs}=\max_c\hat{z}_i^\top\hat{t}_c$（免訓練）
- **Eq.11（EWC-on-router）**：Fisher 加權權重正則（negative baseline，救不回）

## SBO algorithm（一句話 + pseudocode 在 paper §3.4）

多輪：每輪對 $\tilde r-\lambda\cdot(\text{對已選集合的最大 cos 相似度})$ 取 top-$k_{step}$，選完增量更新相似度；$\tau<\infty$ 才每輪呼叫 backbone 早停，否則只在最後預測一次（成本≈one-shot）。$\lambda=0$ 退化成 static top-K。

## 目前實驗數據（真實、已驗；reverse order, 3-fold 除非註明）

- **Budget 效率（Lung, recent task, @K）**：router@64=**0.922±0.020** > all-patch 0.892 > semantic 0.897 > random 0.881 > prototype 0.831
- **NSM vs naive vs zero-shot @64（4-task mACC）**：NSM **0.935±0.031** / naive 0.595±0.228 / zero-shot 0.858±0.073
  - 逐 task：ESCA NSM 0.911 / naive 0.333 / zero 0.800；RCC 0.965/0.576/0.904；BRCA 0.944/0.549/0.841；Lung 0.922/0.922/0.888（Lung 最後學，naive=NSM）
- **舊 task ESCA（forgetting 焦點）**：NSM(per-task) 0.933 / naive 0.333（< random 0.800，主動選錯）/ EWC ~0.40
- **系統級 mACC（all-patch level）**：NaviPath reverse 0.886±0.024（F=0）、paper 0.879±0.024（F=0）；backbone-only reverse 0.917（F=0.041）、paper 0.924（F=0.017）
- **λ sweep（fold1，mean over 4 task, @64）**：λ0→seq 0.874 / λ1→0.872 / λ2→0.845 / λ4→0.469；one-shot 恆 0.874
- **weight-avg baseline（fold-1 preview）**：ESCA 0.467、RCC 0.947（介於 naive 與 NSM，符合中間對照預期；完整 3-fold 待 GPU）

## ⚠️ 本輪發現的關鍵 CAVEAT（答辯時務必守住，別自打臉）

1. **所有主 seqobs 檔（`outputs/seqobs_reverse_f*`）都是 pre-Route-A**：沒有 `normalize_base`，λ=0.5 對排序無效 → **seq ≡ one-shot top-K**。所以 3-fold 的「NSM vs naive vs zero-shot」數字本質是 **one-shot top-K 導航**，不是 sequential。真正的 sequential 只有 `outputs/routeA_sweep/lambda_*`（僅 fold1，normalize_base=True）。**不要宣稱主結果是 sequential observation 的成果。**
2. **0.922 ≠ 0.935**：0.922 是 Lung 單 task budget@64；0.935 是 4-task mACC。scope 不同，slide 要標清楚。
3. **oracle gate only**：`ContextGate.infer()` 未實作。現況＝每 task 獨立 MLP + oracle 載入 → **「0 forgetting」是 decoupling identity / upper-bound reference，不是 CL 貢獻**。老師這點完全正確，**照單全收、不要辯**；roadmap 轉向 learned selection gate + LoRA。
4. **forward-order seqobs 沒跑**：只有 reverse。forward 只有 budget 曲線 + 系統 R-matrix（mACC 0.879）。
5. **λ 大會掉分的真因**：腫瘤 patch 在 feature space 空間聚集，MMR 強制分散 → 選到 stroma。SBO 機制成立（λ≥2 時 seq≠oneshot），但 optimal λ∈[0,1]。
6. **淡化 forgetting 敘事**：reviewer 可能爭論遺忘來自 prompt/prototype classifier 而非 navigation。策略：多比「成績」，少主打 forgetting 因果；backbone frozen ⇒ classifier forgetting≡0 是我們的 control。
7. **命名禁區**：paper/slides **不得出現 "QPMIL"**；一律稱 "frozen diagnostic backbone / diagnosis model"。上位計畫一律以 **North Star** 代稱，勿寫可識別資訊。

## 老師 10 個問題（詳答在 `paper/Teacher_QA_and_Formulas.md`）

A. 訓練機制：Q1 input/output/loss；Q2/Q3 綠色部分是 inference、圖要含 training＋inference（skill bank+loss，已補雙 panel 圖）；Q4 MicroRouter 位置（encoder 後、aggregate 前）。
B. 定義：Q5 class text feature vs prototype；Q6 base score（Eq.1-2，非單純 cosine）與 redundancy（Eq.5 MMR）。
C. 實驗：Q7 forward-order（誠實：只有 reverse）；Q8 0.922 vs 0.935；Q9 加 weight-avg baseline（naive=lower、NSM=upper）；Q10 別宣稱 0 forgetting（independent model=upper bound，需 selection mechanism）。

## 待辦（尚未跑，指令在 `paper/Teacher_QA_and_Formulas.md` D 節）

- forward-order seqobs（router + zero-shot policy，3 folds）
- weight-avg baseline 完整 3-fold × 4-task（目前只有 fold1 preview 2 個 task）
- （可選）reverse λ sweep 補 fold2/3 做 error bar

## 我這個 session 想做的事（請引導我，一次一題）

我要**逐條把老師的問題想透、把公式的每一步搞懂、把答辯講稿練到能上台**。請先幫我確認你已讀過上述檔案與 caveat，然後問我：想從哪一題／哪條公式開始深入？之後就一問一答，一次一個重點，並在我講得不精確時糾正（尤其公式符號與上面 caveat 的邊界）。

（複製結束）

---

## 給作者（Aaron）的備註 — 不用貼進新 chat

- 這份 handoff 是「答辯深入」專用；若新 session 是要**繼續寫 code / 跑實驗**，改用 `SESSION_CONTEXT.md` + `COLLAB_PLAYBOOK.md` 當入口即可。
- 若之後 forward-order / weight-avg 完整數據跑完，記得回來更新本檔「目前實驗數據」與「待辦」兩節，保持 handoff 與事實同步。
- caveat #1（seq≡one-shot）是最容易被老師/reviewer 抓的點，新 session 第一步建議先把這條的講法練順。
