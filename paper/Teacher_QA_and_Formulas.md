# 雙月報告 — 老師提問回覆 & 公式精確說明

> **用途**：老師針對 0701 投影片提出的問題，逐題回答。
> 每題提供【說人話】（幫助理解）＋【正式回答 ZH／EN】（可直接放進報告或回信）。
> 所有公式對齊 `paper/NaviPath-CL_draft_v1.md` 的 Eq. (1)–(11)，並與實際程式碼核對。
>
> **符號**（與 paper §3 Notation 一致）：
> - 一張 slide 有 $n$ 個 patch；$Z \in \mathbb{R}^{n \times D}$ 為凍結 backbone 的 patch features（$D=512$，CONCH）。
> - $\hat{z}$ 表示 L2-normalize。
> - $T = \{t_c\}_{c=1}^{C}$：class-text features（類別文字特徵）。
> - $P = \{p_m\}_{m=1}^{M}$：prototype features（原型特徵）。
> - $\phi$：router（MicroRouter）參數。
> - $g_{\theta^*}$：凍結的診斷 backbone（frozen diagnostic backbone），$\theta^*$ 全程不變。

---

## 目錄

- [A. 訓練機制與架構圖](#a-訓練機制與架構圖)
  - [Q1. 如何 continually 訓練？input／output／loss？](#q1)
  - [Q2 & Q3. 綠色下半是 inference 還是 training？圖要含兩個 phase](#q2q3)
  - [Q4. MicroRouter 放在架構圖哪裡？](#q4)
- [B. 概念定義](#b-概念定義)
  - [Q5. class text feature vs prototypes 的差別？](#q5)
  - [Q6. base score 怎麼算？redundancy 是什麼、怎麼得到？](#q6)
- [C. 實驗與結果](#c-實驗與結果)
  - [Q7. Forward-order 表現？](#q7)
  - [Q8. page 6 的 0.922 ± 0.020 是什麼？為何不是 0.935？](#q8)
  - [Q9. 應加上 weight-averaged MLP baseline；naive 是 lower bound](#q9)
  - [Q10. 不要宣稱 0 forgetting（independent model = upper bound）](#q10)
- [D. 待補實驗的 RunPod 指令](#d-待補實驗的-runpod-指令)

---

## A. 訓練機制與架構圖

<a id="q1"></a>
### Q1. It's still very unclear how to train the navigation module continually. Inputs/outputs? Loss function?

#### 【說人話】

MicroRouter 是一個很小的 2-layer MLP（約 132K 參數）。它的工作是：看過一張 slide 的所有 patch，替每個 patch 打一個「值不值得看」的分數。

- **輸入**：所有 patch 的 features $Z$（來自凍結 backbone）＋ 兩組參考向量（class text features、prototype features）。
- **輸出**：每個 patch 一個純量分數 $r_i$。
- **怎麼學（loss）**：我們**沒有**逐 patch 的標註。我們只有 slide 的診斷 label。所以訓練方式是：讓 router 挑出 top-K patch → 用分數做 softmax 加權平均，聚合成一個 slide 向量 → 拿去和 class text 算 logit → 和真 label 做 **cross-entropy**。分數好不好，完全由「挑出來的 patch 能不能診斷正確」來反推。
- **continual 的部分**：一個 task 訓練完，把這個 MLP 的權重**存進 skill bank（NSM）**，再換下一個 task 重新訓練。目前每個 task 各存一份。

關鍵一句：**backbone 全程凍結，唯一被訓練的只有這個小 MLP，而且它是靠 slide-level label 弱監督學到「哪裡該看」。**

#### 【正式回答 — 中文】

導航模組（MicroRouter）為一個 2-layer MLP（約 132K 參數），是全流程中**唯一**被訓練的元件；診斷 backbone $g_{\theta^*}$ 全程凍結。

**輸入／輸出。** 對每個 patch $i$，先由凍結特徵計算一個 task 數無關（task-count-invariant）的 4 維摘要 $s_i$（paper Eq. 1）：

$$s_i = \Big[\; \max_c \hat{z}_i^\top \hat{t}_c,\;\; H\big(\mathrm{softmax}(\hat{z}_i^\top \hat{t})\big),\;\; \max_m \hat{z}_i^\top \hat{p}_m,\;\; \tfrac{1}{M}\sum_m \hat{z}_i^\top \hat{p}_m \;\Big]$$

其中 $H(\cdot)$ 為 Shannon entropy。將 $[z_i; s_i] \in \mathbb{R}^{516}$ 送入 MLP 得到純量重要度分數（Eq. 2）：

$$r_i = \mathrm{MLP}_\phi([z_i; s_i]),\qquad \mathrm{MLP}:\ \mathbb{R}^{516}\xrightarrow{\text{Linear}}\mathbb{R}^{256}\xrightarrow{\text{GELU}}\mathbb{R}^{256}\xrightarrow{\text{Linear}}\mathbb{R}^{1}$$

（程式：`MicroRouterV0`，`nn.Linear(516,256)→GELU→nn.Linear(256,1)`。）

**Loss（soft-route 目標，Eq. 3）。** Hard top-K 不可微，故採 soft-route：先取候選集 $\mathcal{S}_K=\text{top-}K_i\, r_i$，在集合內對分數做 softmax 得 attention weight，加權聚合成 bag embedding，再由凍結 text classifier 產生 logit：

$$w_i = \frac{\exp(r_i)}{\sum_{j\in\mathcal{S}_K}\exp(r_j)},\qquad \bar{z}=\widehat{\sum_{i\in\mathcal{S}_K} w_i z_i},\qquad \mathcal{L}_{\text{route}}=\mathrm{CE}\big(\sigma\,\bar{z}\,T^\top,\; y\big)$$

其中 $\sigma$ 為 backbone 的 logit scale，$y$ 為 slide-level label。**梯度只透過 softmax weight $w_i$ 回傳到 $\phi$**，不經過凍結 backbone。訓練設定：Adam, lr $5\times10^{-4}$, weight decay $10^{-4}$, 每 task 5 epochs。

**Continual 協定。** Task $1,\dots,T$ 依序到來；學完 task $t$ 後，將 router 權重快照存入 NSM（Eq. 9）：$\text{NSM}=\{\phi^{(1)},\dots,\phi^{(T)}\}$。（關於「這是否算嚴格 continual learning」見 Q10。）

#### 【正式回答 — English】

The navigation module (MicroRouter) is a 2-layer MLP (~132K params) and is **the only trained component**; the diagnostic backbone $g_{\theta^*}$ is frozen throughout.

**Inputs/outputs.** For each patch $i$, we first compute a task-count-invariant 4-D summary $s_i$ from frozen features (Eq. 1):

$$s_i = \Big[\; \max_c \hat{z}_i^\top \hat{t}_c,\;\; H\big(\mathrm{softmax}(\hat{z}_i^\top \hat{t})\big),\;\; \max_m \hat{z}_i^\top \hat{p}_m,\;\; \tfrac{1}{M}\sum_m \hat{z}_i^\top \hat{p}_m \;\Big]$$

with $H(\cdot)$ the Shannon entropy. The concatenation $[z_i; s_i]\in\mathbb{R}^{516}$ is mapped to a scalar importance score (Eq. 2):

$$r_i=\mathrm{MLP}_\phi([z_i;s_i]),\qquad \mathbb{R}^{516}\to\mathbb{R}^{256}\ (\text{Linear})\to \text{GELU}\to\mathbb{R}^{1}\ (\text{Linear}).$$

**Loss (soft-route objective, Eq. 3).** Since hard top-K is non-differentiable, we select a candidate set $\mathcal{S}_K=\text{top-}K_i\,r_i$, softmax-normalize scores within it into attention weights, aggregate into a bag embedding, and classify with the frozen text head:

$$w_i=\frac{\exp(r_i)}{\sum_{j\in\mathcal{S}_K}\exp(r_j)},\quad \bar{z}=\widehat{\textstyle\sum_{i\in\mathcal{S}_K}w_i z_i},\quad \mathcal{L}_{\text{route}}=\mathrm{CE}(\sigma\,\bar{z}\,T^\top,\,y).$$

Gradients reach $\phi$ **only through the softmax weights $w_i$**, never through the frozen backbone. Training: Adam, lr $5\times10^{-4}$, wd $10^{-4}$, 5 epochs/task. After task $t$, the router snapshot is stored in NSM (Eq. 9).

---

<a id="q2q3"></a>
### Q2 & Q3. Is the green (lower) part inference or training? The figure should show BOTH phases (skill bank + training loss, cf. Pin-Zhen's figure).

#### 【說人話】

老師說得對。目前的圖只畫了**推論**（載入已存好的 skill、挑 patch、做診斷），沒有畫**訓練**。而兩者行為不同，所以一張圖蓋不住：

- **訓練時**：有 cross-entropy loss、梯度回傳到 router、訓練完把權重存進 skill bank。
- **推論時**：backbone 和 router 都凍結，只挑 patch 做預測，沒有 loss、沒有梯度。

所以要拆成兩個 panel。我已經畫好一張新的雙 panel 架構圖（見下方檔案位置），訓練 panel 有標出 **skill bank** 和 **training loss**。

#### 【正式回答 — 中文】

現行架構圖僅呈現**推論階段**：由 NSM 取出對應 task 的 skill（router 權重），router 對 patch 評分、選 top-K，交給凍結 backbone 預測，無 loss、無梯度。

我們將架構圖拆為兩個 panel（對齊品臻的圖）：

- **Training panel**：slide features → router 評分 → top-K soft-weighted aggregation（Eq. 3）→ text logit → **cross-entropy loss $\mathcal{L}_{\text{route}}$** → 梯度回傳 router；task 結束後將 $\phi^{(t)}$ 寫入 **skill bank（NSM）**。
- **Inference panel**：凍結 backbone ＋ 凍結 router（skill 由 context gate 從 NSM 取出）→ top-K 選擇 →（可選 SBO 多步）→ 預測。

新圖檔：`experiment_visualize/figs/fig_arch_train_infer.pdf`／`.png`。

#### 【正式回答 — English】

The current figure shows only the **inference phase** (retrieve a stored skill, score patches, select top-K, predict — no loss/gradient). Because training and inference differ, we split the figure into two panels (following Pin-Zhen's layout):

- **Training panel**: slide features → router scoring → top-K soft-weighted aggregation (Eq. 3) → text logit → **cross-entropy loss** → gradient to router; on task completion, write $\phi^{(t)}$ into the **skill bank (NSM)**.
- **Inference panel**: frozen backbone + frozen router (skill retrieved from NSM by the context gate) → top-K selection → (optional multi-round SBO) → prediction.

New figure: `experiment_visualize/figs/fig_arch_train_infer.{pdf,png}`.

---

<a id="q4"></a>
### Q4. Where should the MicroRouter be placed in the (a) proposed-method figure?

#### 【說人話】

放在「backbone 抽完 patch features」和「送進去做 MIL 診斷聚合」**中間**。backbone 先把所有 patch 編碼成 features，MicroRouter 讀這些 features 給分數、挑 top-K，只有被挑中的 patch 才進入後面的診斷聚合。它就是一個「該看哪裡」的閘門。

#### 【正式回答 — 中文】

MicroRouter 位於**凍結 patch encoder 之後、診斷聚合（aggregate_and_predict）之前**。資料流：

$$\text{frozen encoder} \to Z \in \mathbb{R}^{n\times D} \to \underbrace{\text{MicroRouter}}_{\text{選 where to look}} \to \text{top-}K \to \underbrace{g_{\theta^*}.\mathrm{aggregate\_and\_predict}(Z_{\mathcal{S}})}_{\text{凍結診斷頭}} \to \hat{y}$$

backbone 提供「如何表徵（how to represent）」與「如何分類（how to classify）」；MicroRouter 只負責「看哪裡（where to look）」。此為 backbone-agnostic：任何滿足 §3.2 四個 hooks 的 VL backbone 皆可接。

#### 【正式回答 — English】

The MicroRouter sits **between the frozen patch encoder and the diagnostic aggregation head**. The backbone provides "how to represent" and "how to classify"; the MicroRouter only decides "where to look." It is backbone-agnostic: any VL backbone satisfying the four hooks in §3.2 (`encode`, `class_text_features`, `prototype_features`, `aggregate_and_predict`) can be plugged in.

---

## B. 概念定義

<a id="q5"></a>
### Q5. What are the differences between 'class text feature' and 'prototypes'?

#### 【說人話】

兩者都是 512 維向量、都由凍結 backbone 的 text 分支產生，但來源與角色不同：

- **class text feature $t_c$**：把**類別名稱的文字描述**編碼出來的向量（例如把 "esophageal carcinoma" 這段文字丟進 text encoder）。數量 = 類別數 $C$，一個類別一個。給 patch 一個「像不像某個類別」的訊號。
- **prototype $p_m$**：一組**可學習 prompt 模板**編碼出來的向量池（pool），數量 = pool_size $M$。它不直接對應某個類別名稱，是模型自己學到的「代表性樣板」。給 patch 一個「像不像典型組織樣板」的訊號。

#### 【正式回答 — 中文】

兩者皆為凍結 backbone text 分支輸出的 512 維向量：

- **Class text features** $T=\{t_c\}_{c=1}^{C}$：類別名稱文字描述的編碼（經 context-feature enhancement 後 normalize），一類一個，共 $C$ 個。對應程式 `class_text_features()`。提供 **class-alignment** 訊號。
- **Prototype features** $P=\{p_m\}_{m=1}^{M}$：整個**可學習 prompt-prototype pool** 的編碼（$M=$ pool_size），不與類別名稱一一對應，捕捉代表性子模式（sub-patterns）。對應程式 `prototype_features()`。提供 **representative-pattern** 訊號。

在 router 中，兩者僅用於計算每個 patch 的 4 維摘要 $s_i$（Eq. 1）：text 貢獻前 2 維（max text-sim 與 text-sim 分布 entropy），prototype 貢獻後 2 維（max/mean proto-sim）。

#### 【正式回答 — English】

Both are 512-D vectors from the frozen backbone's text branch:

- **Class text features** $T=\{t_c\}_{c=1}^{C}$: encodings of the **class-name text descriptions** (context-enhanced, then normalized); one per class, $C$ total. → a *class-alignment* signal.
- **Prototype features** $P=\{p_m\}_{m=1}^{M}$: encodings of the entire **learnable prompt-prototype pool** ($M$ = pool_size); not tied one-to-one to class names; they capture representative sub-patterns. → a *representative-pattern* signal.

In the router they are used only to compute each patch's 4-D summary $s_i$ (Eq. 1): text contributes dims 1–2 (max text-sim, entropy of text-sim), prototypes contribute dims 3–4 (max/mean proto-sim).

---

<a id="q6"></a>
### Q6. How are the base scores computed? What is redundancy and how is it obtained?

#### 【說人話】

**base score**（單步、靜態分數）= 前面 Q1 講的 $r_i$。兩步：
1. 每個 patch 和 text、prototype 算相似度，濃縮成 4 個數字（Eq. 1）。
2. 把 patch 原始 512 維 ＋ 這 4 維 = 516 維，丟進 MLP → 1 個分數（Eq. 2）。
**注意：不是單純的 cosine similarity。** text/prototype 相似度只佔 516 維輸入裡的 4 維，其餘 512 維是 patch 本身的 feature，最後由 learned MLP 綜合判斷。

**redundancy**（只有做「序列多步」SBO 時才用）：挑第二批、第三批 patch 時，如果某候選 patch 和「已經挑過的 patch」太像，就扣分，避免一直擠在同一區。做法是：算每個候選 patch 對「已選集合」的**最大 cosine 相似度**，再乘上權重 $\lambda$ 當懲罰。$\lambda=0$ 就退回普通 top-K；$\lambda$ 越大越強迫分散。

#### 【正式回答 — 中文】

**Base score（Eq. 1–2）。** 對 patch $i$，先算 4 維摘要 $s_i$（Eq. 1），再 concat patch 原始特徵成 $[z_i;s_i]\in\mathbb{R}^{516}$，經 2-layer MLP 得純量 $r_i$（Eq. 2）。它是**學習得到的函數**，非原始相似度；text/prototype 相似度僅為輸入的 4/516 維。

**Redundancy（僅用於 SBO 序列選取）。** SBO 為多輪流程。令 $\mathcal{S}^{(\leq t-1)}$ 為前 $t-1$ 輪已觀察的 patch 集合。第 $t$ 輪的調整分數採 Maximal Marginal Relevance（MMR，Eq. 5）：

$$a_i^{(t)} = \tilde{r}_i - \lambda\cdot\underbrace{\max_{j\in\mathcal{S}^{(\leq t-1)}}\cos(z_i,z_j)}_{\bar{m}_i^{(t)}\ \text{（redundancy 項）}},\qquad a_i^{(t)}=-\infty\ \text{若}\ i\in\mathcal{S}^{(\leq t-1)}$$

其中 $\tilde{r}_i$ 為 z-score 正規化後的 base score（Eq. 4，單調轉換，不改變 one-shot top-K 結果）：

$$\tilde{r}_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon},\quad \mu_r=\tfrac{1}{n}\textstyle\sum_i r_i,\ \sigma_r=\sqrt{\tfrac{1}{n}\sum_i(r_i-\mu_r)^2}.$$

**redundancy 項 $\bar{m}_i^{(t)}$ = 候選 patch $i$ 與「已選集合」中任一 patch 的最大 cosine 相似度**，並以增量方式更新（Eq. 6），複雜度每輪 $O(n\cdot k_{\text{step}})$：

$$\bar{m}_i^{(t+1)}=\max\!\Big(\bar{m}_i^{(t)},\ \max_{j\in\mathcal{S}^{(t)}}\cos(z_i,z_j)\Big).$$

每輪選 $k_{\text{step}}$ 個（Eq. 7）：$\mathcal{S}^{(t)}=\text{top-}k_{\text{step}}\{a_i^{(t)}: i\notin\mathcal{S}^{(\leq t-1)}\}$。$\lambda$ 為 diversity rotor：$\lambda=0$ 退化為 static top-K。

#### 【正式回答 — English】

**Base score (Eq. 1–2).** Compute the 4-D summary $s_i$ (Eq. 1), concatenate with the raw feature to $[z_i;s_i]\in\mathbb{R}^{516}$, and map through the 2-layer MLP to scalar $r_i$ (Eq. 2). It is a **learned function**, not a raw similarity; text/prototype similarities are only 4 of the 516 input dims.

**Redundancy (SBO only).** Let $\mathcal{S}^{(\leq t-1)}$ be the patches observed before round $t$. The MMR-adjusted score (Eq. 5) is

$$a_i^{(t)}=\tilde{r}_i-\lambda\cdot\underbrace{\max_{j\in\mathcal{S}^{(\leq t-1)}}\cos(z_i,z_j)}_{\text{redundancy }\bar{m}_i^{(t)}},\qquad a_i^{(t)}=-\infty\ \text{if}\ i\in\mathcal{S}^{(\leq t-1)},$$

with $\tilde{r}_i$ the z-score-normalized base score (Eq. 4, monotone → does not change one-shot top-K). The redundancy term is each candidate's **maximum cosine similarity to the already-selected set**, updated incrementally (Eq. 6) at $O(n\cdot k_{\text{step}})$/round. Each round selects $k_{\text{step}}$ patches (Eq. 7). $\lambda$ is the diversity rotor; $\lambda=0$ recovers static top-K.

---

## C. 實驗與結果

<a id="q7"></a>
### Q7. Forward-order performance?

#### 【說人話】

老實說：**NSM vs naive vs zero-shot 的 seqobs 對照，目前只跑了 reverse order（壓力測試方向），還沒跑 forward（paper）order。** 但 forward order 我們有兩份數據：(1) budget 效率曲線（`router_v0_paper`，3 folds）；(2) 系統級 R-matrix（`navipath_full_paper`，mACC = 0.879 ± 0.030，forgetting = 0）。要補齊 forward-order 的 NSM 對照，需再上 RunPod 跑一輪（指令見 [D 節](#d-待補實驗的-runpod-指令)）。

#### 【正式回答 — 中文】

目前 NSM／naive／zero-shot 的序列導航對照僅於 **reverse order**（壓力測試順序，使小樣本 esca 成為最舊 task）完成。Forward（paper）order 已有：(i) budget 效率曲線 `router_v0_paper`（3 folds）；(ii) 系統級 R-matrix `navipath_full_paper`（mACC $0.879\pm0.030$、Forgetting $0$）。為對稱性，將補跑 forward-order 的 seqobs 對照（指令見 D 節）。

#### 【正式回答 — English】

The NSM/naive/zero-shot sequential comparison is currently available only in **reverse order** (the stress-test order making small-sample esca the oldest task). For the forward (paper) order we have (i) budget-efficiency curves (`router_v0_paper`, 3 folds) and (ii) the system-level R-matrix (`navipath_full_paper`, mACC $0.879\pm0.030$, Forgetting $0$). We will run the forward-order seqobs comparison for symmetry (commands in Section D).

---

<a id="q8"></a>
### Q8. Page 6: what does 0.922 ± 0.020 mean? Why not 0.935 ± 0.031 (the average)?

#### 【說人話】

這兩個數字來自**不同的表格與情境**，難怪會混淆：

- **0.922 ± 0.020**：是 budget-efficiency 表裡 **Lung 這一個 task** 的 router @K=64。Lung 是 reverse order 最後學的 task，沒有遺忘問題，用來單純證明「選 patch 的效率」。
- **0.935 ± 0.031**：是 continual 表裡 **4 個 task 平均**的 NSM mACC，用來證明「跨 task 不遺忘」。

建議在 slide 上明確標註「single-task @K=64」還是「mACC over 4 tasks」，避免混淆。

#### 【正式回答 — 中文】

兩數字來自不同表格：**0.922 ± 0.020** 為 budget-efficiency 實驗中 **Lung 單一 task** 的 router@K=64（Lung 為 reverse order 最新 task，用以隔離選取效率、排除遺忘干擾）；**0.935 ± 0.031** 為 continual 實驗中 **4 個 task 平均**的 NSM mACC。兩者 scope 不同（per-task vs mACC），將於投影片明確標示。

#### 【正式回答 — English】

They come from different tables. **0.922 ± 0.020** is the Lung-task router accuracy @K=64 in the budget-efficiency experiment (Lung is the most-recent task in reverse order, chosen to isolate selection efficiency without forgetting confounds). **0.935 ± 0.031** is the mean over all 4 tasks (mACC) of NSM in the continual experiment. We will label each number's scope (per-task vs mACC).

---

<a id="q9"></a>
### Q9. Should compare with a baseline using the AVERAGED MLP weights. Naive continual router is the lower bound.

#### 【說人話】

老師的意思是對照組要更完整。目前有兩端：
- **NSM（我們）**：每 task 一份 MLP → 偏 upper bound。
- **naive continual**：一份 MLP 一直被新 task 覆蓋 → **lower bound**。

老師建議中間再加一個「把各 task 的 MLP 權重**直接平均**成一份」的 baseline。這很便宜（skill bank 裡已有各 task 權重），我已經寫好 script（見下方），到 RunPod／本機跑一下就有數字。

#### 【正式回答 — 中文】

同意。我們將明確標定：naive continual（單一持續被覆蓋的 router）為 **lower bound**，per-task NSM（oracle gate）接近 **upper bound**。並依老師建議新增 **weight-averaged baseline**：將 skill bank 中各 task 的 MLP 權重逐元素平均為單一 router $\bar{\phi}=\tfrac{1}{T}\sum_{t}\phi^{(t)}$，於相同評估協定下比較。此 baseline 成本極低（權重已存於 NSM）。腳本：`eval_weight_avg_baseline.py`。

**初步結果（fold-1 preview，@K=64；完整 3-fold 待 GPU 跑完）：**

| Task | Naive（lower bound） | **Weight-avg（新增中間對照）** | NSM（upper bound） |
|---|---|---|---|
| ESCA（最舊） | 0.133 | **0.467** | 0.867 |
| RCC | 0.684 | **0.947** | 0.947 |

已可見 weight-avg 為名副其實的中間值：在最易遺忘的 ESCA 上，權重平均（0.467）優於 naive（0.133）但遠低於 NSM（0.867），支持「per-task 隔離才是關鍵」的論點。完整 3-fold × 4-task 數據待 D.2 指令於 GPU 跑完補上。

#### 【正式回答 — English】

Agreed. We will label naive continual (a single continuously-overwritten router) as the **lower bound** and per-task NSM (oracle gate) as close to the **upper bound**, and add the suggested **weight-averaged baseline**: element-wise average of the per-task MLP weights into one router, $\bar{\phi}=\tfrac{1}{T}\sum_t\phi^{(t)}$, evaluated under the identical protocol. This is inexpensive since the per-task weights already reside in the NSM. Script: `eval_weight_avg_baseline.py`.

---

<a id="q10"></a>
### Q10. Don't claim "0 forgetting" — one MLP per task ≈ independent models (upper bound, not真正 CL). Adapter/LoRA CL papers have a selection mechanism.

#### 【說人話】

**這題老師完全正確，建議照單全收、不要辯。**

現況：每個 task 各訓練一個獨立 MLP，推論時由 **oracle**（外部告知 task id）決定載哪一個。既然是不同模型、又有 oracle 告知，本來就不會互相干擾 ——「0 forgetting」是理所當然，不能當賣點。程式裡的 `ContextGate` 目前也只有 `oracle` 模式，自動判斷 task 的 `infer()` 還是 `NotImplementedError`（標為 future work）。

真正的 continual learning（像 adapter/LoRA 的 CL 論文）需要一個「**自動選 adapter/skill**」的機制，不能靠 oracle。所以建議說法轉向：
- 把 per-task NSM 定位成 **upper-bound reference**，不宣稱 0 forgetting 是貢獻。
- 明確把「**learned skill-selection gate（免 oracle）＋ LoRA 化 skill bank**」列為下一步。

#### 【正式回答 — 中文】

我們接受此意見。目前每 task 一個獨立 MLP，並以 **oracle gate**（外部提供 task id，Eq. 9 之檢索）於推論時載入對應 router；因模型彼此獨立且有 oracle 指派，task 間本質上不干擾，「zero forgetting」為 **decoupling identity（結構恆等式）**，屬 **upper-bound reference**，非 continual-learning 貢獻。程式 `ContextGate` 目前僅實作 oracle 模式；task-free 的 `infer()` 尚未實作（future work）。

依 adapter/LoRA continual-learning 文獻慣例（parameter-efficient module bank + 一個 selection mechanism），下一步為：(i) **learned skill-selection gate**，免 oracle 由 slide 自身辨識 task／skill；(ii) 以 **LoRA 低秩 skill**（$r=4$，約每 task 4K 參數，較 full NSM 約 130× 縮減，Eq. §3.6 LoRA-NSM）取代 per-task 完整 MLP、共享凍結 router。唯有具備自動選取機制，方構成嚴格意義的 continual learning；我們將據此重新定位現有結果。

#### 【正式回答 — English】

We accept this. Currently we train one independent MLP per task and load the corresponding router at inference via an **oracle gate** (task id provided externally; retrieval in Eq. 9). Because the per-task models are independent and oracle-assigned, they do not interfere by construction, so "zero forgetting" is a **decoupling identity / upper-bound reference**, not a continual-learning contribution. Our `ContextGate` implements only the oracle mode; the task-free `infer()` is unimplemented (future work).

Following the adapter/LoRA CL literature (a parameter-efficient module bank paired with a **selection mechanism**), our next steps are: (i) a **learned skill-selection gate** that identifies the task/skill from the slide alone (no oracle); and (ii) replacing per-task full MLPs with **LoRA low-rank skills** ($r=4$, ≈4K params/task, ~130× smaller than full NSM; §3.6 LoRA-NSM) over a shared frozen router. Only with automatic selection does the setting become genuine continual learning; we will reposition the results accordingly.

---

## D. 待補實驗的 RunPod 指令

> 以下指令**先給你，之後再找時間跑**。皆為 inference-only（載入已存 skill bank，不重新訓練），數分鐘即可完成。
> 前置：`cd $REPO && source .venv/bin/activate`（RunPod 上 `$REPO=/workspace/src/navipath`）。

### D.1 Forward-order（paper order）seqobs 對照（補 Q7）

paper order = `lung → brca → rcc → esca`。需先確認 forward-order 的 skill bank 是否存在；若無，先訓練一次存檔，再 inference。

```bash
# (1) 若尚無 forward-order skill bank，先訓練一次並存檔（每 task 5 epochs）
python eval_sequential_observation.py \
  --backbone-ckpt outputs/qpmil_paper_fold1.pt \
  --order paper --fold 1 --eval-tasks 0,1,2,3 \
  --epochs 5 --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
  --skill-bank-out outputs/skill_bank_paper_f1.pt \
  --out outputs

# (2) router policy（NSM + naive 對照），純 inference（載入 skill bank）
python eval_sequential_observation.py \
  --backbone-ckpt outputs/qpmil_paper_fold1.pt \
  --order paper --fold 1 --eval-tasks 0,1,2,3 \
  --skill-bank-in outputs/skill_bank_paper_f1.pt \
  --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
  --policy-mode router --out outputs

# (3) zero-shot policy（免訓練 baseline）
python eval_sequential_observation.py \
  --backbone-ckpt outputs/qpmil_paper_fold1.pt \
  --order paper --fold 1 --eval-tasks 0,1,2,3 \
  --budgets 0,128,64,32,16 --step-size 16 \
  --policy-mode zero_shot --out outputs

# fold 2、fold 3 同理，改 --fold 與 --backbone-ckpt / --skill-bank-* 檔名。
```

### D.2 Weight-averaged baseline（補 Q9）

新腳本 `eval_weight_avg_baseline.py`（已建於 repo 根目錄）：載入既有 skill bank，將各 task 權重平均為單一 router，於各 task 評估。

```bash
# reverse order（已有 skill bank：outputs/skill_bank_reverse_f{1,2,3}.pt）
for FOLD in 1 2 3; do
  python eval_weight_avg_baseline.py \
    --backbone-ckpt outputs/qpmil_reverse_fold${FOLD}.pt \
    --order reverse --fold ${FOLD} --eval-tasks 0,1,2,3 \
    --skill-bank-in outputs/skill_bank_reverse_f${FOLD}.pt \
    --budgets 0,128,64,32,16 --step-size 16 \
    --out outputs/weight_avg
done

# 之後彙整三個對照：naive(lower) / weight-avg(mid) / NSM(upper)
```

### D.3 （可選）reverse-order λ sweep 補 fold 2、3

目前 `routeA_sweep` 只有 fold 1。若要 error bar：

```bash
for FOLD in 2 3; do
  for LAMBDA in 0.0 1.0 2.0 4.0; do
    python eval_sequential_observation.py \
      --backbone-ckpt outputs/qpmil_reverse_fold${FOLD}.pt \
      --order reverse --fold ${FOLD} --eval-tasks 0,1,2,3 \
      --skill-bank-in outputs/skill_bank_reverse_f${FOLD}.pt \
      --budgets 0,64,32,16 --step-size 16 \
      --redundancy ${LAMBDA} --normalize-base true --redundancy-mode maxsim \
      --out outputs/routeA_sweep/lambda_${LAMBDA}_f${FOLD}
  done
done
```

---

## 附：現有可用數據速查（供報告引用）

| 指標 | 數字 | 來源 | 用途 |
|---|---|---|---|
| Router@64（Lung, recent, 單 task） | 0.922 ± 0.020 | `router_v0_reverse_fold*` | budget 效率（Q8） |
| NSM mACC @64（4 task 平均，reverse） | 0.935 ± 0.031 | `seqobs_reverse_f*` | continual 導航（Q8） |
| Naive mACC @64（reverse） | 0.595 | `seqobs_reverse_f*` | lower bound（Q9） |
| Zero-shot mACC @64（reverse） | 0.858 | `seqobs_*_policy-zeroshot` | 免訓練 baseline |
| 系統 mACC（reverse, all-patch level） | 0.886 ± 0.030 | `navipath_full_reverse_*` | 系統級 |
| 系統 mACC（paper, all-patch level） | 0.879 ± 0.030 | `navipath_full_paper_*` | forward-order 現有數據（Q7） |
| Weight-avg baseline @64（ESCA/RCC, fold1 preview） | 0.467 / 0.947 | `eval_weight_avg_baseline.py` | 中間對照（Q9）；完整 3-fold 待跑 |
| Forward-order seqobs | 待跑 | D.1 指令 | 對稱性（Q7） |
