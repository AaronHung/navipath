# NaviPath-MoE 研究計劃 v1（應該是0.9，下列未改）

> ### 計劃修改部分：
>
> 先把幾個決策定下來,再改 code,最後給你完整的開工 prompt 手冊。
>
> 我同意對方那份的工程修正——它非常實在,而且都是「不修會卡」的點。我把它和你這次的四個新條件(任務序自由、MoE 改 MLP、Mac M1 + CUDA 共用、RunPod 跑重活、真實 CONCH 統計)整合,逐項定論:
>
> **任務序(你說不綁 BRCA→正好簡化)。** 既然不綁,最聰明的選法是**直接用 QPMIL paper order 當主序、reverse 當主戰場**,理由是這樣 baseline 數字能直接對齊 QPMIL 已發表的結果(sanity 最穩),而且 reverse 正是 QPMIL 自己會掉分的地方(0.890→0.859)。所以:
>
> - 主序(sanity + baseline 對齊):`tcga_lung(NSCLC) → tcga_brca → tcga_rcc → tcga_esca`
> - 主戰場(forgetting 重現):`tcga_esca → tcga_rcc → tcga_brca → tcga_lung`(小的 ESCA 先學、最大的 NSCLC 最後學,干擾最大)
> - **砍掉自訂的 BRCA-first 序**(不再需要,少跑一套省算力)。
>
> **MoE 改 MLP expert——同意,我把 skeleton 改掉。** 你們是在 precomputed 512 維特徵上做,不是 fine-tune 大 transformer,LoRA 沒意義。第一版 expert 就是 `e_j(z)=z+MLP_j(z)`,殘差初始化為 0(訓練前等於不動 backbone)。
>
> **Mac M1 16G 開發 + CUDA 共用 + RunPod 跑重活——完全可行,而且比你想的寬鬆。** 關鍵認知:你們的資料很小——單張切片最多 ~8000 patch × 512 × 4 byte ≈ 16MB,batch 16 張也才 ~100MB,**16G 統一記憶體綽綽有餘**。瓶頸不是記憶體而是「10-fold × 多序 × ablation」的總時數。所以策略是:**M1/MPS 做開發 + 單 fold 驗證(完全夠),RunPod 只拿來跑 10-fold CV、多 seed、超參 sweep 這種「平行吞吐」**。Code 一律 device-agnostic(`cuda→mps→cpu` 自動選),並注意 MPS 兩個坑:`float32`(MPS 不支援 float64)、設 `PYTORCH_ENABLE_MPS_FALLBACK=1`。
>
> **真實資料統計已納入:** lung 1054 / brca 1133 / rcc 937 / esca 158,patch mean ~3000–3600。ESCA 只有 158 張(最小)再次印證「expert 數要少、防 collapse」;patch mean ~3000 代表 patch-budget K∈{32,64,128,256,512} 是有意義的大幅縮減,**patch-budget 故事很有戲**。
>
> **對方最關鍵的工程修正我全採納:** patch-budget eval 最先做(不訓練、最安全、最早出第一張表)、QPMIL forward 先回傳 internals(所有創新的地基)、MicroRouter 分 v0(純 patch score)→v1(expert weights)、micro router 輸入用固定維度 summary 統計(因為 class 數會隨任務變,不能 concat 原始 per-class sim)、macro 與 momentum 都最後才加。
>
> 先改 code(MLP expert、v0/v1 router、乾淨 L_sem、device util、真實資料 config)。全部通過(6/6 shape tests + smoke + metrics,MLP expert、device-agnostic 都正常)。更新 README,然後寫你要的開工 prompt 手冊。最後是核心——開工的 prompt 手冊,按 milestone 排序,每個都有「程式 / 測試 / debug / 跑實驗 / ablation」的 prompt。計劃改完、code 改完、開工手冊也好了。三件事都按你的新條件落地,而且 skeleton 重跑過自測全綠(6/6 shape test + smoke + metrics,MLP expert / device-agnostic 都正常)。
>
> **改了什麼(對應你的四個條件):**
>
> 任務序鬆綁後我直接定成 `paper`(NSCLC→BRCA→RCC→ESCA,對齊 QPMIL 已發表數字當 sanity)+ `reverse`(ESCA→RCC→BRCA→NSCLC,主戰場),砍掉 BRCA-first。少跑一套、baseline 還能直接對上。
>
> MoE 從 LoRA 改成殘差小 MLP `e(z)=z+MLP(z)`、最後一層零初始化(訓練前等於不動 backbone),`experts.py` 已換。
>
> `device.py` 做了 `cuda>mps>cpu` 自動選 + MPS fallback + float32;`train_continual.py` 全程 device-aware。手冊裡寫清楚 **M1 16G 跑單 fold 完全夠(資料一張切片才 ~16MB),RunPod 只拿來跑 10-fold/seed/sweep 這種平行吞吐**。
>
> 真實統計(lung1054/brca1133/rcc937/esca158,patch~3000)寫進 config 與測試;ESCA 只有 158 張再次支持「expert 數先設 4」、patch-budget K∈{32…256} 是有意義的縮減。
>
> **也把對方那份的工程修正全採納了**:patch-budget eval 最先做(`eval/patch_budget_eval.py` 已寫好 random/prototype/semantic 三選法)、QPMIL 先回傳 internals、MicroRouter 拆 v0(純 score)/v1(expert weights)、輸入用固定維度 summary 統計(max/entropy/mean,因為 class 數會變)、L_sem 改成 patch 分布上的乾淨 KL(`losses.py`)、macro 與 momentum 最後加。
>
> **開工就照 `KICKOFF_PLAYBOOK.md` 走**:M0(codebase map,只讀)→M1(QPMIL 跑通)→M2(回傳 internals)→**M3(patch-budget 表,今天就能出第一張)**→M4(MicroRouter v0 + Day-16 go/no-go)→M5(v1+MLP MoE+L_bal)→M6(macro)→M7(L_sem)→M8(replay-free momentum,主戰場 reverse)→M9(可視化)。每個 milestone 都附:可直接貼的 Cursor prompt、shape/overfit/collapse 測試、debug 指令、跑實驗指令、ablation。
>
> config 對應 v0→v4 五級,每級都能止損成稿:`qpmil_sanity`(純 backbone)→`navipath_micro`(+MoE)→`navipath_macro_micro`→`navipath_full`(+momentum)。
>
> **今天先做 4 件事(手冊最後一節)**:① 跑 QPMIL 1 fold 出 accuracy matrix ② 讓 forward 回傳 internals ③ 寫 patch-budget eval 出第一張表 ④ MicroRouter v0 先接 budget、不接訓練。第 3 步今天就能告訴你 agentic navigation 故事有沒有分數。
>
> 一個提醒:`tests/` 用 `PYTHONPATH=. python tests/test_shapes.py` 或裝 `pytest` 跑(直接 `python tests/test_shapes.py` 會找不到套件,這是 import path 不是 bug)。
>
> 要不要我下一步把這套包成給教授的**簡報**(放兩張架構圖 + v0→v4 路線 + 主表骨架),或先針對 M1/M2 產一個能直接對著 QPMIL repo 跑的 `data adapter + runner` 草稿?

---

---

## 原來計劃如下：（可說是v0.9）



### 架構分兩張比較不會擠：第一張是「單張切片怎麼被處理」（也就是 agent 本身），第二張是「跨任務怎麼學」（CL 機制）。先看第一張。

![image-20260606180607664](/Users/aaron/Library/Application Support/typora-user-images/image-20260606180607664.png)

![image-20260606180709891](/Users/aaron/Library/Application Support/typora-user-images/image-20260606180709891.png)

### 以上的架構，細節有錯誤，因為做的比較早，例如task的次序就不是BRCA開始的，必須依照QPMIL以及一個倒序。WSI CLAM的shape可能有誤，需要小心，再確認。目前CONCH出來的shapes，不是這個shape。





### Agentic Macro/Micro Routing for Continual Whole Slide Image Classification

> **投稿目標：** COMPAYL 2026（MICCAI 2026 satellite workshop）
> **截稿：** 2026-07-01 AOE（OpenReview 投稿；正文 8 頁 + 最多 2 頁 references；MICCAI 格式；錄取收錄於 MICCAI Satellite Events Springer LNCS proceedings）
> **目前日期：** 2026-06-06，距截稿約 25 天
> **硬體：** 單卡 RTX 4090 / 5090（不依賴大型 GPU）
> **本文件目的：** 讓全組與指導教授一次看懂方向、方法、實驗、分工與時程，看完即可開工。

---

## 0. 一頁總結（先讀這段）

我們做的任務是：**在一連串依序到來的癌症資料集上，做 whole slide image（WSI，全切片影像）的亞型分類，並且要求模型學新任務時不要忘記舊任務**（這叫 continual learning / 持續學習）。任務順序固定為 **BRCA → NSCLC → RCC → ESCA**，每來一個資料集就新增兩個分類類別。

我們的方法叫 **NaviPath-MoE**，一句話定義：

> 以 **QPMIL-VL** 作為 patch-level 的 vision-language MIL backbone，把它原本的「prototype 查詢」升級成一個 **agentic 的 macro/micro 雙層路由器（router）**：macro 路由器看整張切片決定「用哪組診斷策略」，micro 路由器看每個 patch 決定「這塊組織值不值得看、屬於哪種證據」；並用一個 **不需要儲存任何舊切片（replay-free）** 的動量鞏固機制，讓承載舊知識的 expert 在學新任務時被保護。

**這同時滿足國科會病理 AI 組這一年新增的四個關鍵詞：CL（持續學習）+ Agent（代理人路由決策）+ WSI（巨像素切片）+ VLM（病理視覺-語言模型）。** 而且 COMPAYL 2026 今年的官方徵稿特別點名鼓勵「agentic AI in pathology」，方向與 venue 完全對齊。

**我們的勝負手不是在標準設定上把準確率刷得比 QPMIL 高**（理由見 §1.3，QPMIL 在標準設定已接近天花板）。我們贏的地方是：**逆序任務（reverse-order）、有限觀察預算（patch budget）、路由穩定性（routing drift）、以及不存舊資料（replay-free）** 這四個維度。

---

## 1. 背景與動機

### 1.1 問題：病理 AI 的模型不會「持續學習」

現實中的病理 AI 模型，是在某一個固定資料集上訓練好的。但臨床上資料是**動態**的：新的癌種、新的院區資料、新的染色條件會陸續出現。傳統做法是把舊資料 + 新資料全部混在一起重新訓練一次，成本高、且常常因隱私／法規無法保留舊病患資料。

**Continual Learning（CL，持續學習）** 就是要解決這件事：讓模型一個任務接一個任務地學，學新任務時盡量不忘舊任務（避免所謂的 catastrophic forgetting / 災難性遺忘）。

### 1.2 病理 CL 的特殊難點：忘記的不只是「分類器」，還有「該看哪裡」

WSI 是十億像素等級的影像，標準做法是 **Multiple Instance Learning（MIL，多實例學習）**：把一張切片切成成千上萬個 patch，每個 patch 抽特徵，再用一個聚合機制（attention 或 prototype）決定哪些 patch 重要，聚合成切片層級的特徵去分類。

近期一個重要發現（AKD-PMP, 2025）是：在 attention MIL 的持續學習裡，**遺忘主要不是發生在最後的分類器，而是發生在「決定該看哪裡」的 attention／聚合層**。也就是說，模型還記得「乳癌長什麼樣」，但任務切換後它「看錯地方」了。

**這個發現天然地把問題帶到 Agent 的視角：** WSI 模型在做的事，本質上就是「在巨大的視野裡選擇要觀察哪些區域，再據此下判斷」——這就是一個 perceive → decide → act 的代理人行為。持續學習的核心難題，因此可以重新表述為：**當疾病領域演進時，模型能不能維持一個穩定的「觀察策略（navigation/routing policy）」。**

### 1.3 為什麼不直接「刷贏」現有最好方法？（關鍵策略，請教授特別看這段）

我們的 backbone QPMIL-VL 是目前 incremental WSI classification 的最強方法之一。在標準的 forward-order（BRCA→NSCLC→RCC→ESCA）設定下，它的平均準確率已達 **0.890**，而其理論上界（把所有資料一次混合訓練的 JointTrain）是 **0.908**，遺忘指標低到 **0.027**。

**這代表標準設定已經幾乎沒有「遺忘」可修了。** 任何方法（包括我們的）想在這個設定上把 ACC 再往上推 1–2 個百分點，都會碰到天花板，且 reviewer 會質疑提升不顯著。

因此我們的策略是：**不在已飽和的標準設定上硬碰，而是把問題推到「遺忘會重新出現」的更難設定**——

- **逆序任務（reverse-order）**：QPMIL 在逆序下會從 0.890 掉到約 0.859，任務順序干擾讓遺忘重新浮現。
- **有限觀察預算（patch budget）**：強迫模型只看 K 個 patch（K = 256 / 128 / 64 / 32）。被丟掉的證據會讓遺忘重新出現，這正好給我們的方法舞台，**同時也是 Agent「選擇性觀察」最直觀的證據**。
- **路由穩定性（routing drift）**：量化舊任務切片在學完新任務後，模型的觀察策略漂移多少。
- **不存舊資料（replay-free）**：我們完全不儲存任何過去的切片或 patch，這在隱私上是強賣點。

> **一句話定位給 reviewer：** 現有持續學習 WSI 方法聚焦於「保住分類器或 prototype」，但沒有顯式地建模「當疾病領域演進時，模型該繼續看哪裡、怎麼看」。我們把持續式 WSI 分類重新形式化為一個 agentic routing 問題，並提出一個 macro/micro expert routing 框架，在不儲存任何過去切片的前提下，保住模型的診斷觀察行為。

### 1.4 與先前 ConSlide 路線的關係（給組員銜接用）

本組先前曾以 **ConSlide** 為 backbone。我們現在**正式 pivot 到 QPMIL-VL**，原因：

1. ConSlide 的階層式架構（HIT）要拿到 region-level attention 需要 attention rollout，復現困難、且 reviewer 容易挑戰「你的 region-level 是真的嗎」。
2. ConSlide 是 buffer-based（要存資料），而 QPMIL 天生 buffer-free，與我們 replay-free 的賣點一致。
3. QPMIL 是**純 patch-level**、原生使用 CONCH、且**官方有公開 code 與我們已下載的 prepared CONCH features**（即 `can_dataset`），可省下大量特徵抽取時間，幾天內就能把 backbone 跑起來。

**因此 ConSlide 降級為「可選的 cited baseline」，不再是工程主線；坤倫先前的 ConSlide 整合工作可保留為 baseline 對照，但主力轉向 QPMIL backbone 與我們的 router。**

---

## 2. 任務定義（精確版）

### 2.1 任務流

| Task | Cohort（TCGA） | 分類 | 約略切片數 |
| --- | --- | --- | --- |
| Task 1 | **BRCA**（乳癌） | IDC vs ILC | 多 |
| Task 2 | **NSCLC**（非小細胞肺癌） | LUAD vs LUSC | 多（最多） |
| Task 3 | **RCC**（腎細胞癌） | CCRCC vs PRCC | 中 |
| Task 4 | **ESCA**（食道癌） | Adeno vs Squamous | 少（約 150，最少） |

- **Class-incremental（類別增量）**：每來一個 cohort 新增 2 個類別，推論時不給 task id。
- **BRCA 起步** 的原因：與詹老師國科會計劃的 TCGA-BRCA 對齊。
- **逆序（reverse-order）** 即 ESCA → RCC → NSCLC → BRCA，是我們的主戰場之一。

### 2.2 資料與標註現況

| 層級 | 是否有標註 |
| --- | --- |
| Slide-level subtype | ✅ 有 |
| Patch-level | ❌ 無 ground truth |
| Lesion / bounding box | ❌ 無 |
| 病理醫師 navigation 軌跡 | ❌ 無 |

→ 因此只能做 **weakly-supervised**（弱監督，只有切片層級標籤），不能假設有逐塊或逐區域的監督訊號。我們的 patch 特徵已用 **CONCH**（凍結的病理 VLM）預先抽好：CLAM 在 10× 倍率切非重疊 256×256 patch，每個 patch → 512 維 CONCH 特徵。

---

## 3. 為什麼這是「Agent」，而不是換個名字

我們不做強化學習（RL / PPO）。但 Agent 的本質是 **perceive → decide → act**，RL 只是訓練 agent 的方式之一，不是必要條件（LLM agent 推論時也不是 RL）。

**重點：我們的 Agent 不是「把 attention 改名叫 agent」這種包裝，而是一個真的會做決策的路由模組。** 對應如下：

| Agent 組件 | 在 NaviPath-MoE 中是什麼 |
| --- | --- |
| **State（狀態）** | 一張 WSI 的 patch 特徵集合 `{z_i}` + 類別文字特徵 + prototype pool |
| **Decision（決策）** | macro 路由器選診斷策略 expert；micro 路由器對每個 patch 選感知 expert / 決定是否值得看 |
| **Action（動作）** | 用路由權重選 Top-K patch、選 prototype、加權聚合成切片特徵 |
| **Outcome（結果）** | 切片層級亞型預測 |
| **Continual forgetting** | 任務切換後，路由策略漂移，模型不再看原本該看的證據 |

我們在 COMPAYL 版本做 **single-step 路由 agent**（一次性決定看哪裡），未來再升級到 multi-step 導航 agent（見 §11 roadmap）。

---

## 4. 核心創新（三個貢獻）

**Contribution 1 — Agentic Macro/Micro Expert Routing。**
把 QPMIL 的 prototype 查詢機制升級成雙層路由器：macro 看整張切片決定策略 expert，micro 看每個 patch 決定感知 expert。**micro 路由器同時就是 patch 選擇器**——它的輸出可直接用來選 Top-K patch，這讓「Agent 路由」與「有限預算下的選擇性觀察」變成同一個機制的兩個出口。

**Contribution 2 — Replay-free Dual-Importance Expert Consolidation。**
一個**不存任何舊切片**的持續學習鞏固機制：根據每個 expert 對「舊任務」與「新任務」的重要度，決定它在學新任務時要被保護還是放開更新。這是把 VLN 領域 M³E 的動量更新思想，針對 WSI／小資料改造後的版本。

**Contribution 3 — Semantic-Anchored, Collapse-free Routing。**
用凍結的 CONCH 影像-文字語義空間當作**跨任務不變的錨點（anchor）**來約束路由器，避免它亂跑（`L_sem`）；並用負載平衡損失避免所有 patch 擠到同一兩個 expert（`L_bal`）。

> **概念層級賣點（higher-level framing）：** Anchor-based, replay-free routing regularization for continual WSI agents——用凍結病理 VLM 當穩定錨點，約束一個持續學習的觀察策略，全程不儲存過去切片。

**AKD-PMP 的定位：** 只保留它的洞見（「routing/attention drift 是 WSI CL 的核心」）當作 motivation，並做一個 **buffer-free 的 routing-stability ablation**（`L_route`，見 §5.5）。**不重建完整 AKD-PMP、不做 buffer/pseudo-bag memory**，因為那與我們 replay-free 的主張衝突。

---

## 5. 方法設計（NaviPath-MoE 完整技術規格）

整體資料流：

```
WSI ──CLAM patching──▶ {256×256 patches}
                          │ (CONCH frozen image encoder，已預抽)
                          ▼
              patch 特徵 Z = {z_i}, z_i ∈ R^512
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   Macro Router      Micro Router        QPMIL Prototype Pool
 (看整張切片)      (看每個 patch)        (查詢 + 聚合)
        └────────┬────────┘                  │
                 ▼ fusion                     ▼
          expert 權重 w_i ───▶ 選 Top-K patch / 加權聚合 ──▶ bag 特徵 f_b
                                                              │
class text prompt ──CONCH text encoder──▶ 類別文字特徵 f'_txt ─┘
                                                              ▼
                                          prediction p(y) = softmax(τ·⟨f_b, f'_txt⟩)
```

### 5.1 Backbone：QPMIL-VL（不更動，先跑通）

給一張切片，patch 特徵為 $Z=\{z_i\}_{i=1}^{n}, z_i\in\mathbb{R}^{512}$。

1. 查詢向量 $\bar z = \mathrm{MaxPool}(Z)\in\mathbb{R}^{512}$。
2. **Prototype pool**：$M$ 組 (key, prompt) 配對 $\{(k_m, \text{prompt}_m)\}_{m=1}^{M}$（QPMIL 設 $M{=}20$）。用 $\bar z$ 與各 key 距離選出 top-$N$（$N{=}5$）個 prototype，並加上「頻率懲罰」避免所有切片都選同一組。
3. 選中的 prompt 經 CONCH 文字編碼器 → prototype 特徵 $F_p$。
4. 算 $Z$ 與 $F_p$ 的 cosine 相似度矩陣 $S\in\mathbb{R}^{n\times N}$，沿 column 做 softmax → 聚合矩陣 $W\in\mathbb{R}^{n\times N}$；加權後取平均 → bag 特徵 $f_b\in\mathbb{R}^{512}$。
5. **CFE 語言分支**：類別文字經 class ensemble + tunable vector 強化 → $f'_{txt}$。
6. 預測 $p(y\mid \bar z)=\mathrm{softmax}(\tau\cdot\langle f_b, f'_{txt}\rangle)$。
7. QPMIL 原損失：$\mathcal{L}_C$（分類）、$\mathcal{L}_M$（prototype 匹配）、$\mathcal{L}_S$（類別相似度），$\mathcal{L}_{\text{QPMIL}}=\mathcal{L}_C+\lambda\mathcal{L}_M+\beta\mathcal{L}_S$。

> 註：QPMIL 裡那個「像 attention 的物件」就是指派矩陣 $W$（patch→prototype）。我們的路由器與穩定性分析都圍繞 $W$ 與路由權重展開。

### 5.2 Macro Router（看整張切片，選策略 expert）

$$s=\mathrm{Pool}(Z)\quad(\text{mean / max / attention pooling 或 prototype usage histogram})$$
$$w^{\text{macro}}=\mathrm{softmax}\big(\mathrm{MLP}_{\text{macro}}(s)\big)\in\Delta^{E}$$

角色：判斷「這張切片整體像哪一類病理任務、該啟動哪組全局策略 expert」。

### 5.3 Micro Router（看每個 patch，選感知 expert，**同時是 patch 選擇器**）

$$u_i=\big[\,z_i\;;\;\cos(z_i, f'_{txt})\;;\;\cos(z_i, F_p)\,\big]$$
$$w^{\text{micro}}_i=\mathrm{softmax}\big(\mathrm{MLP}_{\text{micro}}(u_i)\big)\in\Delta^{E}$$

角色：判斷「這個 patch 像不像某類證據、靠不靠近某些 prototype、值不值得用來診斷」。其純量分數（如 $\max_e w^{\text{micro}}_{i,e}$）可直接排序選 **Top-K patch**，用於有限預算評估。

### 5.4 Fusion + Experts

$$w_i=\beta\,w^{\text{macro}}+(1-\beta)\,w^{\text{micro}}_i$$

- $w_i$ 用途：(a) 選 expert、(b) 選 prototype、(c) 選 Top-K patch、(d) 作為聚合權重。這就是 **Agent action**。
- **Experts 設計（重要工程約束）：** expert 數量 $E$ **要少**（先設 $E{=}4\sim6$，可先綁定 cohort 數），且用 **LoRA / 小 adapter** 實作（低參數）。**原因：四個 cohort 很小（ESCA 僅約 150 張），expert 太多會 collapse 或 overfit。$E$ 視為第一級超參數，寧少勿多。**

### 5.5 Replay-free Dual-Importance Consolidation（持續學習鞏固，不存舊資料）

每個任務結束後，**不存任何切片或 patch**，只統計 expert 的使用度：

- 當前任務重要度：$\displaystyle I^{\text{cur}}_e=\frac{\sum_{x\in D_t} w_e(x)}{\sum_j\sum_{x\in D_t} w_j(x)}$
- 歷史重要度（跨舊任務）：$I^{\text{old}}_e=\mathrm{EMA}\big(I^{1:t-1}_e\big)$
- 動量（保護程度）：$m_e=\sigma\big(a\,I^{\text{old}}_e - b\,I^{\text{cur}}_e\big)$
- 鞏固更新：$\theta_e^{t}=m_e\,\theta_e^{t-1}+(1-m_e)\,\phi_e^{t}$

直覺：

| Expert 狀態 | 策略 |
| --- | --- |
| 舊任務重要、新任務不重要 | 保護舊知識，少更新（$m_e$ 大） |
| 舊任務不重要、新任務重要 | 放開更新（$m_e$ 小） |
| 新舊都重要 | 小心更新，必要時 split / adapter |
| 新舊都不重要 | 幾乎不動 |

> 這比照搬 M³E（只看當前使用度）更穩，因為它同時考慮舊任務的重要度，避免一更新就忘舊、或一凍結就學不動。

### 5.6 最終 Loss Stack（鎖定版）

$$
\mathcal{L}=\underbrace{\mathcal{L}_C+\lambda\mathcal{L}_M+\beta\mathcal{L}_S}_{\text{QPMIL 原損失（不動）}}
+\gamma\,\mathcal{L}_{sem}
+\eta\,\mathcal{L}_{bal}
+\xi\,\mathcal{L}_{route}
$$

- **語義錨損失（防路由亂跑）：**
$$\mathcal{L}_{sem}=D_{KL}\Big(\mathrm{softmax}(w_i)\;\big\|\;\mathrm{softmax}\big(\cos(z_i, f'_{txt})\big)\Big)$$
讓路由器選 patch／expert 的傾向，被 CONCH 語義空間約束。

- **負載平衡損失（防 expert collapse）：**
$$\mathcal{L}_{bal}=E\sum_{e=1}^{E} f_e\,P_e$$
（$f_e$＝路由到 expert $e$ 的比例，$P_e$＝router 對 $e$ 的平均機率；Switch-Transformer 式）。這也對應 QPMIL 原本觀察到的「不同資料集容易選到同一組 prototype」問題。

- **路由穩定損失（可選、嚴格 buffer-free）：**
$$\mathcal{L}_{route}=D_{KL}\big(\mathrm{sg}(w^{\text{teacher}})\;\big\|\;w^{\text{new}}\big)$$
這是 AKD 思想的輕量化。**鐵律：teacher 是凍結的舊模型快照，在「當前任務資料」上算（LwF 式），或只存純路由統計量；絕不儲存任何舊切片或 patch。** 第一階段不一定開，視 Day-16 結果決定。

---

## 6. Borrow / Modify / Build（我們各部分從哪來）

| 來源 | 我們怎麼用 |
| --- | --- |
| **QPMIL-VL（官方 repo + `can_dataset` CONCH features）** | 直接當 backbone 與資料，先跑通 |
| **CONCH（凍結 VLM）** | 影像編碼器（特徵已抽）+ 文字編碼器（語義錨） |
| **M³E（VLN 的 macro/micro MoE + 動量更新）** | 借**思想**，改造成 WSI 版 router 與 replay-free 鞏固；**不碰 7B LLM / VLN simulator** |
| **AKD-PMP** | 只借洞見（routing drift 是核心）+ 一個 buffer-free routing-stability ablation；**不重建完整方法、不做 buffer** |
| **ConSlide** | 降級為可選 cited baseline |
| **NaviPath-MoE（我們自己 build）** | macro router、micro router、fusion、replay-free 鞏固、$\mathcal{L}_{sem}$、$\mathcal{L}_{bal}$、patch-budget 評估、routing-drift 分析、main loop |

---

## 7. 實驗設計

> **表的優先級（重要）：難設定是主表，標準 forward 只是 sanity check。** 因為 forward 已飽和（§1.3），把它當主表會被 reviewer 質疑。

### 7.1 主表 A — 逆序持續分類（reverse-order，主戰場）

| Method | ACC ↑ | Upper-bound Ratio ↑ | Forgetting ↓ | BWT ↑ | Params ↓ | Buffer |
| --- | --- | --- | --- | --- | --- | --- |
| FineTune（下界） | | | | | | No |
| EWC / LwF | | | | | | No |
| ER / DER++ | | | | | | **Yes** |
| ConSlide（cited） | | | | | | Yes |
| QPMIL-VL | | | | | | No |
| **NaviPath-MoE（ours）** | | | | | | **No** |

### 7.2 主表 B — 有限預算導航（patch budget，Agent 最直觀的證據）

| Selection | ACC@All | ACC@256 | ACC@128 | ACC@64 | ACC@32 |
| --- | --- | --- | --- | --- | --- |
| Random | | | | | |
| Attention / Prototype（QPMIL heuristic） | | | | | |
| Semantic（CONCH cos-sim） | | | | | |
| **NaviPath-MoE router（ours）** | | | | | |

### 7.3 主圖 — Routing Drift

至少兩張圖：
1. **Expert usage over tasks**：不同任務啟動不同 expert，沒有全部 collapse。
2. **Old-task routing drift**：學到 Task 4 後，回頭看 Task 1 切片，被選中的診斷證據是否仍穩定（ours vs QPMIL／fine-tune 對比）。

### 7.4 Sanity 表 — forward-order

放正文或附錄，證明「我們在標準設定上沒有退步」即可，不當主賣點。

### 7.5 Metrics

ACC（平均準確率）、Upper-bound Ratio、Forgetting、BWT（backward transfer）、Masked ACC（task-IL 參考用）、Routing/Selection drift（我們自定義）。**全部沿用 QPMIL 已定義的算法以保持公平。**

### 7.6 Ablation（逐元件加上去）

| 變體 | Macro | Micro | $\mathcal{L}_{sem}$ | $\mathcal{L}_{bal}$ | Momentum | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| QPMIL（backbone） | ✗ | ✗ | ✗ | ✗ | ✗ | 基準 |
| + Micro | ✗ | ✓ | ✗ | ✗ | ✗ | patch-level agent |
| + Macro | ✓ | ✓ | ✗ | ✗ | ✗ | slide-level 策略 |
| + $\mathcal{L}_{sem}$ | ✓ | ✓ | ✓ | ✗ | ✗ | 語義錨 |
| + $\mathcal{L}_{bal}$ | ✓ | ✓ | ✓ | ✓ | ✗ | 防 collapse |
| + Momentum（full） | ✓ | ✓ | ✓ | ✓ | ✓ | CL 鞏固 |

另做：$\beta$（fusion 權重）sweep、$E$（expert 數）sweep、$\gamma/\eta$ sweep。

---

## 8. 25 天計劃（含 go/no-go 與雙重 fallback）

> 核心紀律：**先把安全的東西做完做穩，再往上加風險高的東西，每一級都可停損。**

| 區間 | 目標 | 交付物 | 停損／判準 |
| --- | --- | --- | --- |
| **Day 1–7** | QPMIL backbone 跑通（forward + reverse）+ 乾淨 eval harness | `qpmil_forward.json`、`qpmil_reverse.json`、per-task accuracy matrix、prototype usage、`eval_continual.py`、`metrics.py`、configs | 能輸出五大指標 + per-task 矩陣；trend 與論文接近；code 可改不崩 |
| **Day 8–11** | **只**做 Micro Router（先不做 macro / momentum） | micro router + patch-budget 評估結果 | **綠燈條件 = router 選的 Top-K 在 ACC@{64,128} 明顯贏 random/attention/semantic 選擇**；崩則止損 |
| **Day 12–15** | 加 Macro Router + fusion + $\mathcal{L}_{bal}$ | macro/micro 完整 router、expert usage 圖 | macro 沒幫助就砍 macro、保 micro |
| **Day 16** | **Go / No-Go 決策日** | go/no-go 報告 | 見下方 |
| **Day 17–20** | **僅 Go 後**：replay-free 動量鞏固 | 逆序 / 更長序列 / few-shot 結果 | **子 fallback**：Day 19 momentum 沒幫助 → 以「macro/micro + $\mathcal{L}_{bal}$ + $\mathcal{L}_{sem}$」成稿，momentum 退為 ablation 一列 |
| **Day 21–23** | Ablation + 兩張可視化圖 | ablation tables、routing-drift 圖、expert-usage 圖 | — |
| **Day 24–26** | 寫稿 + 排版 + 投稿 | 8 頁 draft、figures、references、OpenReview 投出 | 預留排版（MICCAI margin/spacing 違規會 desk reject） |

**Day 16 Go 條件（任一即可）：**
1. 逆序設定贏 QPMIL；或
2. Forgetting 明顯低於 QPMIL；或
3. ACC@64 / ACC@128 明顯優於 QPMIL heuristic；或
4. Routing-drift 可視化明顯較穩。

**Day 16 No-Go fallback（安全稿，本身已可投）：**
> QPMIL backbone + Semantic Micro Router + $\mathcal{L}_{sem}$ + patch-budget 評估 + routing 分析。不再追 macro/momentum。

---

## 9. 分工

### 希仁（洪希仁）— Lead / 方法設計 / 核心實作 / 寫稿
- 方向掌舵、story 與 claim 設計、與教授溝通、組員技術 review。
- 親自實作：Micro/Macro Router、$\mathcal{L}_{sem}$、replay-free 鞏固機制。
- 寫稿：Intro、Related Work、Method 主體、Discussion。
- 長線 owner：M³E / multi-step agent 的 2027 follow-up。

### 潘坤倫 — Method Implementation Owner
- QPMIL-VL backbone 整合與跑通（forward + reverse），讀 `can_dataset` CONCH features。
- 與希仁協作把 router 接進 QPMIL 的聚合流程；實作 expert（LoRA）與 fusion。
- 跑主實驗（BRCA→NSCLC→RCC→ESCA 與逆序）與 ablation。
- （ConSlide 先前工作 → 轉為 cited baseline 對照，不再是主線。）

### 廖珈峰 — Data / Baselines / Evaluation / Visualization
- Data verify：QPMIL prepared CONCH features 下載與檢查（四 cohort 完整性）。
- Baseline runs：FineTune / EWC / LwF / ER / DER++ / QPMIL（cited + reproduced 兩種數字都備）。
- Eval pipeline：一鍵輸出 ACC / Upper-bound Ratio / Forgetting / BWT / Masked ACC / per-task 矩陣 / patch-budget。
- Visualization：expert usage 圖、routing-drift 圖、結果彙整（tables / logs / final figures）。

---

## 10. 風險與應對

| 風險 | 影響 | 應對 |
| --- | --- | --- |
| CONCH features 不完整 | 主線 | 優先用 QPMIL prepared features；必要時先跑已驗證 cohort |
| Micro router 在 Day 8–11 沒贏 heuristic | 主方法 | 退安全稿（QPMIL + semantic router + 分析） |
| MoE expert collapse（小資料） | 方法穩定性 | $E$ 設小（4–6）、LoRA experts、$\mathcal{L}_{bal}$ 從第一天開 |
| Momentum 鞏固沒提升 | 創新強度 | 子 fallback：退為 ablation 一列，主稿用 router + $\mathcal{L}_{bal}$ + $\mathcal{L}_{sem}$ |
| 時間不夠 | 完整度 | 優先 main 表 + patch-budget + 一張 drift 圖；其餘進 supplementary / future work |
| reverse-order 仍贏不了 | 主賣點 | 改打 patch-budget + replay-free + 可解釋性（仍是完整故事） |

---

## 11. Paper Outline（8 頁）

```
1. Introduction
   - WSI continual learning 的重要性與難點
   - 關鍵洞見：遺忘發生在「該看哪裡」，不只在分類器
   - 標準設定已飽和 → 把問題推到 routing/budget/replay-free
   - 將持續式 WSI 分類形式化為 agentic routing 問題
   - 三個貢獻
2. Related Work
   - Continual Learning for WSI（ConSlide, AKD-PMP, QPMIL-VL）
   - MIL for WSI（ABMIL, CLAM, TransMIL）
   - Pathology VLM（CONCH）
   - Continual agents / MoE routing / VLN（M³E）— 點明我們是 single-step routing agent
3. Method — NaviPath-MoE
   3.1 Problem formulation：continual WSI classification as agentic routing
   3.2 QPMIL-VL backbone
   3.3 Agentic Macro/Micro Expert Routing
   3.4 Replay-free Dual-Importance Consolidation
   3.5 Losses（L_C, L_M, L_S, L_sem, L_bal, L_route）
   3.6 Training algorithm（一個 Algorithm box）
4. Experiments
   4.1 Setup（datasets, task stream, metrics, implementation）
   4.2 Main results — reverse-order（主表 A）
   4.3 Patch-budget navigation（主表 B）
   4.4 Routing drift analysis（主圖）
   4.5 Ablation
   4.6 Forward-order sanity（附錄或精簡）
5. Discussion / Limitation / Future work（single-step → multi-step）
6. Conclusion
```

---

## 12. 最終定位與 Roadmap

```
COMPAYL 2026（本篇）：
  NaviPath-MoE
  = QPMIL-VL backbone
  + Agentic Macro/Micro Expert Routing
  + Replay-free Dual-Importance Consolidation
  + Semantic Anchor (L_sem) + Load Balance (L_bal)
  主打：reverse-order / patch-budget / routing-drift / replay-free
  涵蓋：CL + Agent + WSI + VLM，對齊 COMPAYL「agentic AI in pathology」主題

AAAI / MICCAI 2027（下一篇，AgentPath）：
  = hierarchical / multi-step WSI navigation agent
  = 完整 M³E-style macro/micro routing + momentum
  = （可選）buffer-free 與 RL navigation 的進階版
```

> **NaviPath-MoE 是第一步：把 CL + Agent + WSI + VLM 用一個 25 天內可完成、單卡可跑、replay-free、且有自定義勝負軸的形式落地。AgentPath 是下一步：把 single-step 路由 agent 升級成多步導航 agent。**

---

### 附：開工前 checklist（Day 0）
- [ ] QPMIL-VL 官方 repo clone + 環境建好（單卡可跑）
- [ ] `can_dataset` 四 cohort CONCH features 完整性確認（每張切片 `[n, 512]`）
- [ ] monorepo 結構：`backbone/`、`router/`、`consolidate/`、`losses/`、`eval/`、`configs/`、`experiments/`
- [ ] eval harness 規格凍結（五大指標 + per-task 矩陣 + patch-budget）
- [ ] 教授 sign-off：方向、主賣軸（不打 forward-ACC）、分工、時程

---

---

---





> ## 以下是我啟動的手順：

一步一步帶你從零開始。目標:結束後你有一個可以跑 `smoke_test.py` 的開發環境,然後用 Cursor 做第一件事(M0 codebase map)。

------

## 第一步:建立專案目錄

開 Terminal,跑下面這幾行:

```bash
mkdir -p ~/xk8/01_navipath
cd ~/xk8/01_navipath
git init
```

這就是你的**專案根目錄**。之後所有東西都在這裡面。

------

## 第二步:Clone QPMIL-VL

```bash
git clone https://github.com/can-can-ya/QPMIL-VL.git
```

Clone 完你會有:

```
01_navipath/
  QPMIL-VL/    ← 作者原始 code
```

把它加進 `.gitignore`(我們不 track 作者的 code,只 track 自己的):

```bash
echo "QPMIL-VL/" >> .gitignore
echo "outputs/"  >> .gitignore
echo ".env"      >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc"     >> .gitignore
```

------

## 第三步:建立 Python 環境(uv)

```bash
# 建虛擬環境(Python 3.11)
uv venv .venv --python 3.11
source .venv/bin/activate      # 每次開新 Terminal 都要跑這行
```

確認環境啟動了(看到 `(.venv)` 在 prompt 前面):

```bash
python --version    # 應該顯示 Python 3.11.x
```

------

## 第四步:安裝套件

```bash
# PyTorch(Mac M1 MPS 支援在 2.x 已內建)
uv pip install torch torchvision torchaudio

# 其他基礎工具
uv pip install numpy pyyaml pytest ruff

# 看 QPMIL 有沒有自己的 requirements
cat QPMIL-VL/requirements.txt   # 先看一下

# 如果有,就裝
uv pip install -r QPMIL-VL/requirements.txt
```

驗證 PyTorch 跑得起來(特別確認 MPS):

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
"
```

M1 Mac 應該顯示 `MPS available: True`。

------

## 第五步:把 NaviPath-MoE skeleton 放進來

你在這次對話裡下載到了 `navipath_moe` 資料夾(包含 `routers.py`、`experts.py` 等)。把它移進專案根目錄:

```bash
# 假設你下載到 ~/Downloads/navipath_moe/
cp -r ~/Downloads/navipath_moe/ ~/xk8/01_navipath/

# 確認結構正確
ls 01_navipath/
# 應該看到: QPMIL-VL/  navipath_moe/  eval/  configs/  tests/  ...
```

此時你的目錄長這樣:

```
01_navipath/
  QPMIL-VL/             ← 作者原始 code(不動)
  navipath_moe/         ← 我們的 Python 套件
    device.py
    routers.py
    experts.py
    ...
  eval/
    metrics.py
    patch_budget_eval.py
  configs/
    qpmil_sanity.yaml
    navipath_micro.yaml
    ...
  tests/
    test_shapes.py
  train_continual.py
  smoke_test.py
  KICKOFF_PLAYBOOK.md
  requirements.txt
```

------

## 第六步:設定資料路徑

你的 CONCH features 已經在硬碟某個地方了。**用 symlink 連進來**,不要複製(檔案大):

```bash
# 假設你的特徵在 /path/to/conch_features/,裡面有 tcga_lung/ tcga_brca/ tcga_rcc/ tcga_esca/
ln -s /path/to/conch_features data

# 確認 symlink 正確
ls data/
# 應該看到: tcga_lung/  tcga_brca/  tcga_rcc/  tcga_esca/

# 確認格式
python -c "
import os, torch
# 隨便找一個檔案驗 shape
root = 'data/tcga_esca'
files = [f for f in os.listdir(root) if f.endswith('.pt')][:1]
x = torch.load(os.path.join(root, files[0]), map_location='cpu')
print('shape:', x.shape, '  dtype:', x.dtype)
# 期望: shape: [n, 512]  dtype: torch.float32
"
```

如果 shape 是 `[n, 512]`、dtype 是 `float32`,就對了。

然後更新 configs 裡的資料路徑:

```bash
# 把 configs 裡的 /path/to/conch_features 換成你的實際路徑
# 最簡單:用 sed 批次替換
sed -i '' 's|/path/to/conch_features|/Users/aaron/xk8/01_navipath/data|g' configs/*.yaml
```

------

## 第七步:設定環境變數

```bash
cat > .env << 'EOF'
PYTORCH_ENABLE_MPS_FALLBACK=1
EOF
```

然後在你的 `~/.zshrc`(或 `~/.bash_profile`)加一行,讓每次 Terminal 都自動載入:

```bash
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc
source ~/.zshrc
```

------

## 第八步:加一個 Makefile(省得記指令)

```bash
cat > Makefile << 'EOF'
.PHONY: smoke test clean

smoke:
	python smoke_test.py

test:
	PYTHONPATH=. python tests/test_shapes.py

metrics:
	python -m eval.metrics

budget:
	python -m eval.patch_budget_eval --config configs/qpmil_sanity.yaml --order paper

train-sanity:
	python train_continual.py --config configs/qpmil_sanity.yaml --order paper

train-reverse:
	python train_continual.py --config configs/navipath_full.yaml --order reverse

clean:
	find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
EOF
```

------

## 第九步:第一次驗證——骨架能跑嗎

```bash
# 在 01_navipath/ 根目錄
make smoke     # 或: python smoke_test.py
make test      # 或: PYTHONPATH=. python tests/test_shapes.py
make metrics
```

期望輸出:

```
device: mps                              ← Mac M1 成功用 MPS
OK — NaviPath-MoE standalone components pass (MLP experts).
  micro_v0 score (3000,) | micro_v1 w (3000, 4) | ...
PASS test_micro_v0
PASS test_micro_v1
...OK — all shape tests pass.
{'ACC': 0.8475, 'Forgetting': 0.0633, ...}
```

全部過了才進下一步。有任何 import error 先貼給我。

------

## 第十步:在 Cursor 開啟專案

```bash
# 在 01_navipath/ 根目錄
cursor .
```

如果 `cursor` 指令不存在:打開 Cursor App → `Cmd+Shift+P` → 搜 `Shell Command: Install 'cursor' command` → 裝好後重開 Terminal 再試。

Cursor 開啟後,先做這個確認:底部狀態列確認 Python 解譯器是 `.venv`(你的虛擬環境)。如果不是:按右下角 Python 版本 → 選 `.venv/bin/python`。

------

## 第十一步:M0 — 用 Cursor 做 QPMIL codebase map(只讀、不改)

在 Cursor 裡按 `Cmd+L` 開 AI chat,把下面這段 **完整貼上去**:

```
You are helping me adapt the existing QPMIL-VL codebase (in ./QPMIL-VL/) for
continual WSI classification. Do NOT edit any files yet.

Please inspect the QPMIL-VL repository and produce a concise codebase map covering:
1. Where the model forward pass is implemented.
2. Where prototype selection and prototype-guided aggregation are implemented.
3. Where class text features / CFE are implemented.
4. Where the training loop computes losses (L_C, L_M, L_S).
5. Where continual task order and evaluation metrics (ACC, Forgetting, BWT) live.
6. The minimal insertion points to add — without rewriting QPMIL — for:
   - micro router (navipath_moe/routers.py MicroRouterV0)
   - MLP experts (navipath_moe/experts.py ExpertBank)
   - semantic-anchor loss (navipath_moe/losses.py l_sem)
   - load-balance loss (navipath_moe/losses.py l_balance)
   - patch-budget evaluation (eval/patch_budget_eval.py)
   - replay-free momentum consolidation (navipath_moe/consolidate.py)

For each insertion point give the exact file path, function/class name, and the
expected tensor shapes at that point.

End with a numbered step-by-step integration plan (small commits, one thing at a time).
```

Cursor 會掃描 `QPMIL-VL/` 的所有檔案,給你一份 codebase map。**把它的輸出存起來**(貼到 `QPMIL-VL/CODEBASE_MAP.md` 或截圖)。這份 map 是 M1-M9 所有 prompt 的前提。

------

## 現在你的狀態

```
01_navipath/
  ✅ QPMIL-VL/         已 clone
  ✅ navipath_moe/      skeleton 可跑(smoke + shape test 全過)
  ✅ data/              CONCH features 已 symlink
  ✅ .env               MPS fallback 已設
  ✅ Makefile           常用指令
  ⏳ QPMIL-VL/CODEBASE_MAP.md   (M0 跑完後產生)
```

做完 M0 codebase map 之後,就把 Cursor 給你的 map 貼過來,我幫你確認插入點正不正確,然後帶你做 M1(讓 QPMIL 跑通第一個 fold)。

------

**最常見的卡點預告:**

- `smoke_test.py` 出現 `ModuleNotFoundError: navipath_moe` → 確認你在 `01_navipath/` 根目錄跑,不是在子目錄。
- MPS not available → `pip install --upgrade torch` 確認 2.x 以上。
- data symlink 裡的 `.pt` shape 不是 `[n, 512]` → 可能是用 `.h5` 或 `.npy` 格式,告訴我實際格式我幫你改 data loader。
- QPMIL 需要 CONCH 文字編碼器 model weights → M0 codebase map 做完後確認,如果需要我帶你申請 HuggingFace token 並下載。

有任何一步卡住,把 terminal 輸出貼給我就好。



> #### 執行·錯誤：
>
> ❯ uv pip install -r QPMIL-VL/requirements.txt  × No solution found when resolving dependencies:  ╰─▶ Because there is no version of conda-build==3.21.7 and you require      conda-build==3.21.7, we can conclude that your requirements are unsatisfiable.
>
> Show less
>
> 那個 `requirements.txt` 是**把整個 conda 環境 dump 出來的**,裡面混了 `conda-build`、`conda` 這些 conda 私有套件,用 pip/uv 裝一定炸。**不要裝它**。
>
> 直接裝真正需要的 ML 核心套件就好,而且 torch 版本要用 2.x(原檔寫的 1.11 沒有 MPS 支援,2.x 向下相容):
>
> ```bash
> # 一行裝完所有真正需要的東西
> uv pip install "torch>=2.1" torchvision torchaudio \
>   numpy transformers timm einops \
>   h5py scikit-learn matplotlib seaborn tqdm \
>   pyyaml huggingface-hub safetensors wandb \
>   pytest ruff
> ```
>
> 裝完驗一下:
>
> ```bash
> python -c "
> import torch
> print('torch:', torch.__version__)
> print('MPS:', torch.backends.mps.is_available())
> import transformers, timm, einops, h5py
> print('all imports OK')
> "
> ```
>
> 應該看到 `torch: 2.x.x`、`MPS: True`、`all imports OK`。
>
> 之後做 M0 codebase map 時,Cursor 掃完 QPMIL-VL 的 import 會告訴你還差什麼——到時候缺什麼補什麼就行。跳過的套件(`openslide`、`staintools`、`spams`、jupyter 系列)是 WSI 原始影像處理用的,你已經有預抽好的 CONCH features,不需要那些。
>
> ---
>
> 問題清楚了。是我沒說清楚造成混亂，原因是**同名兩層**：
>
> - 外層 `navipath_moe/` = 整個專案資料夾（應放在 `01_navipath/` 根目錄）
> - 內層 `navipath_moe/navipath_moe/` = 真正的 Python 套件（`import navipath_moe` 那個）
>
> 下載的檔案因為是一個一個分開的，目錄結構會消失，所以你看到的是平的。現在直接用指令重建正確結構：
>
> ```bash
> cd ~/xk8/01_navipath
> 
> # 建好需要的資料夾
> mkdir -p navipath_moe eval configs tests outputs
> 
> # ── Python 套件 (navipath_moe/ 裡的 .py) ──
> # __init__.py, device.py, routers.py, experts.py,
> # consolidate.py, losses.py, model.py
> # → 這些留在 navipath_moe/ 不用動
> 
> # ── eval/ ──
> mv navipath_moe/metrics.py          eval/  2>/dev/null
> mv navipath_moe/patch_budget_eval.py eval/ 2>/dev/null
> 
> # ── configs/ ──
> mv navipath_moe/qpmil_sanity.yaml       configs/ 2>/dev/null
> mv navipath_moe/navipath_micro.yaml     configs/ 2>/dev/null
> mv navipath_moe/navipath_macro_micro.yaml configs/ 2>/dev/null
> mv navipath_moe/navipath_full.yaml      configs/ 2>/dev/null
> 
> # ── tests/ ──
> mv navipath_moe/test_shapes.py  tests/ 2>/dev/null
> 
> # ── 根目錄 ──
> mv navipath_moe/train_continual.py  . 2>/dev/null
> mv navipath_moe/smoke_test.py       . 2>/dev/null
> mv navipath_moe/KICKOFF_PLAYBOOK.md . 2>/dev/null
> mv navipath_moe/README.md           . 2>/dev/null
> mv navipath_moe/requirements.txt    . 2>/dev/null
> ```
>
> 跑完之後驗一下結構和 smoke test：
>
> ```bash
> # 確認結構（不看 QPMIL-VL 和 git）
> find . -not -path './QPMIL-VL/*' -not -path './.git/*' \
>        -not -name '*.pyc' -type f | sort
> 
> # 執行
> python smoke_test.py
> ```
>
> 期望看到的 `find` 結果（大概長這樣）：
>
> ```
> ./KICKOFF_PLAYBOOK.md
> ./README.md
> ./configs/navipath_full.yaml
> ./configs/navipath_macro_micro.yaml
> ./configs/navipath_micro.yaml
> ./configs/qpmil_sanity.yaml
> ./eval/metrics.py
> ./eval/patch_budget_eval.py
> ./navipath_moe/__init__.py
> ./navipath_moe/consolidate.py
> ./navipath_moe/device.py
> ./navipath_moe/experts.py
> ./navipath_moe/losses.py
> ./navipath_moe/model.py
> ./navipath_moe/routers.py
> ./requirements.txt
> ./smoke_test.py
> ./tests/test_shapes.py
> ./train_continual.py
> ```
>
> 貼一下你的 `find` 輸出跟 `smoke_test.py` 結果，我幫你確認有沒有缺漏。
