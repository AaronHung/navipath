# 10 · 機制底層原理 + 多步路線圖（What / Why / How，append-only）

> 對應老師最會 challenge 的兩點：(1)「你們不就是把每個任務存下來，當然不忘？」(2)「多步沒做、沒有醫師看片 trajectory 怎麼訓練？」
> 這篇用人話講清楚：機制到底是什麼、和「純存檔不忘」差在哪、多步怎麼來、每個結果對應哪個主張。
> 對照實作：`navipath_moe/continual_agent.py`（NSM/Gate）、`navipath_moe/sequential_observation.py`（序列觀察 + policy_mode）、`train_router_v0.py`、`eval_sequential_observation.py`、`analyze_seqobs_n3.py`。
> 規則：只增不刪。

---

## 1. 機制三件套（人話版架構）

凍結的 backbone（CONCH encoder + QPMIL 文字分類頭）負責「**看得懂**」；我們外掛的 Continual Navigation Layer 負責「**看哪裡**」。三個可動零件：

| 零件 | 白話 | 可訓練？ | 存什麼 |
|---|---|---|---|
| **Navigation Policy π**（router） | 「看哪」的大腦：對每個 patch 打「對這個診斷多重要」的分數 | ✅ 唯一可訓練 | — |
| **NSM**（Navigation Skill Memory） | 技能櫃：存「每個任務的 π 該長怎樣」 | — | 每任務一份技能 |
| **Context Gate** | 現在這張片該用哪份技能（Phase-0 = oracle 已知 task id；未來 task-free） | future 才學 | — |

外加 **Observation State + Budgeted Sequential Observation**：在 budget K 下多輪挑 patch、累積已看證據，輸出可稽核的 **Navigation Trace**。

---

## 2. ★ 老師 challenge：「不就是把東西存下來？」——誠實拆解

### 2.1 先承認哪裡對
pilot 的 NSM 確實是「**每任務存一份 router 權重（533KB），oracle 挑對的那份用**」。所以：
- `nsm` 條件 = **no-interference upper bound**：不共享參數 → 不互相干擾 → **定義上**零遺忘。
- 這跟「per-task 獨立模型 + oracle」幾乎等價。
- **⇒「我們零遺忘」本身不是貢獻。老師完全對。** 若宣稱「貢獻＝NSM 不忘」會被打穿。

### 2.2 那貢獻是什麼？（三點，每點都不是「存下來」）

1. **問題的成立與定位（最核心）**
   ZeroSlide 已證「**分類**」可 frozen/zero-shot、天生不忘。但只要你要做 budgeted / agentic 的「**看哪**」，就需要一個 trainable navigator，而它一旦 sequential train 就崩（mACC 0.595、Forgetting 0.454）。
   → **我們是第一個把「navigation 也需要 CL」這條軸立起來並量化的人。** 這是 ZeroSlide 沒碰、卻是任何 agent 必經的維度。

2. **NSM 是「量問題大小」的上界，不是終點方法**
   pilot 用 per-task 權重把**上界**（nsm 0.935）和**下界**（naive 0.595）框出來。**這個 gap = 任何真正 CL navigation 方法要填的空間。** 科學在「**用便宜記憶填這個 gap**」：把每任務的 full router 換成 **prompt / prototype / low-rank / replay**（從 533KB → 更小、且共享 backbone 表徵）。那時就不是「存全部」，而是「**存一個小技能、共享主幹**」——正是 prompt-based CL（L2P / DualPrompt）的精神，我們把它帶到 navigation。

3. **Gate（部署取代 oracle）才是非 trivial 的真問題**
   oracle 是上界假設。真實部署沒有 task id，要**從片子本身推出**「現在像哪個任務、該調哪份技能」。這是 Phase-1 的硬骨頭，也是「存下來」說法**完全沒回答**的部分。

### 2.3 一句話 defend
> 我們不主張「存下來所以不忘」。我們主張：**(a)** navigation 這條臨床關鍵能力會遺忘、需要 CL（**新軸**）；**(b)** per-task NSM 是**量化問題上界的工具**，真正方法是用**便宜記憶**（prompt/低秩/replay）逼近上界；**(c)** 部署要的是 **task-free gate**，那才是難題。**零遺忘是起點，不是賣點。**

---

## 3. ★ 多步（sequential）：為什麼這版沒出來、將來怎麼做

### 3.1 為什麼 seq == oneshot（不是沒時間，是設計還沒到）
seq 有跑，但跟 oneshot 完全一樣（差 0）。原因：**現在的 policy 是「靜態」的**——router 給每個 patch 的分數**只看 patch 自己**，不看「我已經看過什麼」。所以不管一次選 64、還是分 4 輪每輪 16，選出的集合一樣（分數不變 → top-k 不變）。redundancy 懲罰太弱，沒真的改排序。
→ **多步要有意義，policy 必須「狀態相依」**：看過一些 patch 後，下一步分數要改變。這是**設計增量，不是時間問題、不是 bug。**

### 3.2 關鍵澄清：多步 ≠ 存醫師的步驟
你的疑惑：「沒給你醫師看片 trajectory，不知道哪種病該上下左右還是放大，怎麼訓練多步？」
**答：我們不模仿醫師路徑（不是 imitation learning），也不需要 gaze trajectory。** 多步是靠**結果回饋**自己長出來的：唯一監督訊號仍是「**整張片診斷對不對**」（跟現在的弱監督一樣）。policy 學的是一套**搜尋策略**——在最少 budget 下把診斷信心拉到最高。「該放大 / 該轉向」是這個目標的**副產品**，不是抄來的標註。
- **我們存的不是「步驟」，是「policy（小參數）」。** eval 時 policy **當場生成** Navigation Trace。所以「存」依然很小，**步驟是「跑出來」不是「背下來」。**

### 3.3 把多步做出來的三條路（由易到難，標清楚要不要 GPU）

| 路線 | 原理 | 要 GPU？ | 狀態 |
|---|---|---|---|
| **A. 推論期自適應選擇** | 下一 patch 分數＝`norm(relevance) − λ·(與已選『任一』patch 最大相似度, MMR)`。step t 依賴 step 1..t-1 → seq 真的 ≠ oneshot | ❌ Mac 即可 | **✅ 已實作**（`sequential_observation.py`：`normalize_base` + `redundancy_mode="maxsim"`）；待 RunPod 驗 acc |
| **B. 信心驅動早停 / 自適應 budget** | 邊看邊算 backbone 信心 margin，超過 τ 就停。trajectory 由停止規則 emergent（看到夠確定就停，臨床語意強） | ❌ 無需新訓練 | 已有 `confidence_threshold` 鉤子，待調 |
| **C. RL / policy-gradient 學搜尋** | 選 patch 當序列決策，reward＝最小 budget 下診斷正確（**label-only，仍無 trajectory**）。policy 吃 Observation State 決定下一步，REINFORCE/bandit 訓練 | ✅ 需 GPU | North Star（真正可學的 agent） |

> **2026-06-28 實作筆記（路線 A）**：把 base_score 做 **z-score 正規化**（單調 → one-shot top-K 不變），redundancy 從「與已看平均（centroid）」改成 **MMR「與已選任一 patch 的最大相似度」**。合成驗證（4 群、高分群故意冗餘）：λ=0 全擠高分群；**λ=2 開始分散、λ=4 覆蓋四群** → 機制正確。**關鍵：正規化後預設 `redundancy=0.5` 太小，λ 應 sweep ~1–4。** 這就是「seq==oneshot 是調參問題」的具體解。

### 3.4 「調參」到底在調什麼（原理）
- **λ / redundancy**：控「探索新區域 vs 深挖高分區」。太小→退化成 oneshot；太大→亂跑漏病灶。**這版 seq==oneshot 的直接原因之一就是它相對 router 分數尺度太小。**
- **base_score 正規化**：把 router 分數與 redundancy 拉到同尺度，否則懲罰被淹沒。
- **step_size / 最大步數**：每輪看幾個、最多幾輪 → 決定 trace 顆粒度與多步是否真觸發。
- **τ（早停閾值）**：信心多高就停 → 控 budget 自適應（簡單片少看、難片多看）。
- **RL reward 係數**：`acc − cost·步數` 的權衡 → 決定 agent 願不願意多看。

---

## 4. 結果 ↔ 主張對應表（哪個數字證明哪句話）

> 來源：`outputs/RESULTS_seqobs_20260628.md`（reverse, 3 folds, oracle, seq, budget=64）。

| 你想講的話 | 看哪個結果 | 數字 |
|---|---|---|
| navigation **會遺忘**（問題成立） | nonsm retention / Forgetting | mACC **0.595**、Forgetting **0.454**、esca@64 **0.333** |
| 遺忘**可被記憶救回**（問題可解、上界） | nsm retention | mACC **0.935**、Forgetting **0**、esca@64 **0.911** |
| budget 下**少少 patch 就抓到重點** | esca budget 曲線 | nsm @16/32/64 ≈ **0.91**，已達/超過 @All 0.867 |
| zero-shot navigation **強但不夠**（回應 ZeroSlide/老師） | zero-shot vs nsm/nonsm | zero **0.858** < nsm 0.935；但 zero 0.858 **> naive 0.595** |
| 多步 agent 增益（**還沒出來，誠實**） | seq vs oneshot | 差 **0** → 標「policy 尚靜態，見 §3 路線」 |

---

## 5. 一頁 What / Why / How（報告口袋版）
- **What**：在**凍結診斷 backbone** 上，做 budgeted、可持續學習的「看哪」導覽層（CNL）。
- **Why**：分類已能 zero-shot 不忘（ZeroSlide），但 agentic「看哪」需 trainable navigator，而它**會遺忘**——這條臨床關鍵軸沒人處理。
- **How**：router 用**弱監督**學 where-to-look；**NSM 框出問題上界**；gate 路由技能；序列觀察輸出**可稽核 trace**。pilot 證明問題**真實＋可解**；下一步＝**便宜記憶 + 自適應/RL 多步 + task-free gate**。

## 更新日誌
- 2026-06-28：建立本篇（機制防禦「不只是存下來」＋多步路線圖＋結果對應表），同步看板答辯筆記與架構頁說明。
