# 07 · 我們在做什麼（深入）— N2 實驗在驗證什麼（append-only）

> 被老師/reviewer 問「你們到底在做什麼、經得起推敲嗎」時的完整底稿。
> 對照實作：`train_router_v0.py`（`train_router_one_task`）、`navipath_moe/routers.py`、`navipath_moe/losses.py`、`navipath_moe/sequential_observation.py`、`eval_sequential_observation.py`。
> 規則：只增不刪；舊認知要改用「更新註記」標日期。

---

## 0. 一句話
在一個**凍結**的病理 VLM backbone（QPMIL-VL，內含 CONCH patch encoder + 文字分類頭）之上，外掛一層獨立的 **Continual Navigation Layer**。它回答：

> 當模型「一個癌症接一個癌症」依序學會 4 種癌症後，對**最早學的那個癌症**，還能不能正確地「在整張 WSI 上知道該看哪些 patch」？

我們發現「看哪」這個能力會被後面的任務覆蓋（遺忘），並用 per-task skill memory（NSM）把它救回來。

---

## 1. seq vs oneshot；單步還是多步？
- **budget**：一張 WSI 最多准看幾個 patch（16/32/64/128/All）。
- **one-shot（單步）**：router 一次對所有 patch 打分，直接挑最高的 K 個。一步到位。
- **sequential（多步）**：分多輪，每輪挑 `step_size` 個；挑完更新 **Observation State**（已看 patch 的彙整特徵），下一輪在「已看內容」的基礎上對剩餘 patch 重新評估（加 redundancy 懲罰，避免一直看雷同區域），再挑下一批。這就是 agentic 的「邊看邊決定下一步」。

→ oneshot = 單步選擇；sequential = 多步、有狀態的選擇。兩個都跑，是要證明「多步＋狀態」有沒有額外好處。

---

## 2. 這是 Agent 還是 CL？→ 兩條軸的交叉

| 軸 | 管什麼 | 模組 |
|---|---|---|
| **Agent** | 「**一張片內**怎麼看、何時停」 | Sequential Budgeted Observation + Observation State |
| **CL** | 「學了新癌症後，**舊癌症**還會不會看」 | Navigation Skill Memory (NSM) + Context Gate |

一句話：**Agent 管「一張片內怎麼看」，CL 管「跨癌症別還記不記得怎麼看」。**

---

## 3. router 跟以前 per-task / selector 一樣嗎？
- **零件沿用**：router 本體是 `MicroRouterV0`（小 MLP，輸入 = patch 特徵 + 4 個 summary 統計，輸出每 patch 一個分數）。
- **架構全新**：舊 "selector" 只做 one-shot top-K（單步、無記憶）；現在把同一個 router 包進 **Sequential Observer（多步＋狀態）＋ NSM（每任務存一份權重）＋ Context Gate**。

→ 定位（擋 reviewer）：**舊 selector 是我們新框架的退化特例（oneshot + no-memory）；貢獻是外面那層 sequential + skill memory。**

---

## 4. CL 怎麼設計？核心結果撐得住嗎？
設計：
1. 依序訓練 4 個任務的 router；**每學完一個任務，把該 router 權重拍快照存進 NSM**。
2. 推論時 **Context Gate**（Phase-0 用 oracle = 已知任務 id）取出對應 skill 再 navigate。
3. 對照組 nonsm：只留「學完 lung 之後」那份 router（無記憶），拿去看 esca。

核心證據（fold 1，看最舊的 esca）：

| budget | nonsm（無記憶） | nsm（有記憶） |
|---|---|---|
| 64 | **0.133**（主動選錯＝災難性遺忘） | **0.867**（救回） |
| 16 | 0.20 | 0.867 |

**問題存在（遺忘）＋ 我們的 memory 解決它** → CL 軸站得住。
（`nonsm@64=0.133` 與舊版獨立驗過的數字一致，代表重現無誤。）

---

## 5. 為什麼只測 esca？lung 會慢幾十倍嗎？
- **訓練階段：4 種癌症全部都訓練了**（log: train task 1/4 esca … 4/4 lung，各 5 epoch；rcc/brca/lung 各幾百到七百多 slides）。**一個 fold 花 ~3 小時幾乎全在訓練這 4 個 router。**
- **評估階段：json 只測 esca**，因為 esca 最早學、最容易被後面覆蓋，是遺忘最關鍵的觀察點；且 esca test 只有 15 張，eval 很快。
- **inference 很便宜**（打分 → 選 K → 一次前傳），測 lung 不會慢幾十倍。慢的是訓練，而訓練已做完、存進 skill bank。
- **N3 直接用同一個 skill bank**，把 `task_index` 換 1/2/3 分別 eval rcc/brca/lung，**不必重訓**，就能補成「完整 4 任務 retention 表」。

---

## 6. 最關鍵：沒有「醫生看哪裡」標註，router 怎麼學會看哪？（弱監督原理）
**我們完全沒有 patch-level / 醫生 gaze 標註，只有「整張片的診斷 label」。** 訓練流程（對照 `train_router_one_task`）：
1. router 對一張片所有 patch 打分。
2. 取最高 K 個，用 softmax 權重把特徵**加權聚合**成一個 bag 向量。
3. bag 丟進**凍結的 QPMIL 文字分類頭**（bag 對各類別文字特徵算相似度）得 logits。
4. 跟**整張片的 label** 算 cross-entropy，**梯度只回傳到 router**。

關鍵在第 4 步：router 挑的 patch 聚合後若讓分類**正確**，loss 低；挑錯則高。**為降 loss，router 被迫學會「把對診斷有判別力的 patch 打高分」**——也就是病灶區。沒人告訴它哪裡是腫瘤，它是**從「我挑了這些→診斷對不對」的回饋裡自己學出來**（weakly-supervised / attention-MIL 標準原理）。

**Policy 直觀例子：**
- 學 esca 時，router 把「esca 病灶紋理區」打高分 → 聚合後分類成功 → 這類區域的高分被強化。久了 policy =「看到 esca 那種組織就給高分」。
- 換看 rcc/brca/lung，判別區不同 → 需要**不同的 skill（不同 router 權重）**。用 lung 的 policy 看 esca 會選錯 → 崩到 0.133 → **這就是為什麼需要 NSM。**

> 補充：`losses.py` 另有 `l_sem`（KL 對齊 CONCH 的文字相似度，可給「語義錨」訊號），但 **N2 這版的訓練迴圈只用 cross-entropy**，沒開 `l_sem`。即「目前唯一的監督訊號 = slide-level label」。

---

## 7. 經得起推敲嗎？（誠實版）
- **站得住**：CL 軸（問題＋解法）有乾淨證據；「無標註卻能 navigate」有明確弱監督原理；零件沿用、框架創新的定位誠實。
- **要主動承認的弱點**：
  1. Agent 軸 sequential 這版**還沒贏 oneshot**（seq==oneshot，原因見下）→ 報告講「進行中」。
  2. 目前只測 esca 單任務、單 fold、test set 小（15 張，顆粒度 6.67%）。
- **seq==oneshot 的原因**：(a) budget=16、step=16 只有單輪，redundancy 根本沒啟動；(b) 15 張顆粒太粗；(c) redundancy 權重 0.5 相對 router 分數尺度太小，幾乎不改變排序。**這是調參/設定問題，不是 bug。** 修法：base_score 正規化 + 加大 redundancy + 縮小 step，且在大一點 test set（lung/brca）上測。

## 更新日誌
- 2026-06-27（深夜）：建立本篇（N2 深入解釋，對照實作核對過）。
