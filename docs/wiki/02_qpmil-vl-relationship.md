# 02 · 與 QPMIL-VL 的關係（必被問）

> 來源：QPMIL-VL = Gou et al., AAAI 2025, "Queryable Prototype MIL with VLMs for **Incremental** WSI Classification"（arXiv:2410.10573）。

## 先承認事實：QPMIL-VL 本身就是一個 CL 方法

它的官方定位是「**第一個**為 incremental WSI classification 設計的 VL 框架，常**顯著超越**他法、達 **SOTA**」。
它有兩條分支：
1. **prototype-guided aggregation**：把 instance 特徵聚成 bag-level 特徵。
2. **class feature 強化**：class ensemble + tunable vector + class similarity loss。
並借鏡 L2P / AttriCLIP 的 **prompt-query** 機制來**緩解 catastrophic forgetting**。

➡️ 所以 **QPMIL 的 prototype/prompt 確實是 CL 記憶——但那是給「分類器」用的**（記住「每個 class 長什麼樣」、怎麼聚合特徵）。

## 那我們和它差在哪？（一張表講完）

| 維度 | QPMIL-VL | NaviPath-CL（我們） |
|---|---|---|
| CL 作用在 | **分類器 / class 表徵** | **navigation policy（怎麼看）** |
| 記憶存什麼 | prototype/prompt = class 長相 | NSM = 各任務「該看哪」的技能 |
| 觀察方式 | prototype-guided **聚合（全看）** | **budgeted、多步、序列**觀察 |
| 有沒有「where-to-look」決策 | ❌ 沒有 | ✅ 有（Agent 核心） |
| 可解釋性 | bag 特徵 | **Navigation Trace（看的順序）** |
| budget / 效率 | 不處理 | ✅ 核心設定 |

**一句話**：QPMIL 解「**記住怎麼分類**」；我們解「**記住怎麼看**」。兩者不同層、不衝突——我們把 QPMIL 當 **frozen 診斷 backbone**疊在底下。

## 我們「用了 QPMIL 什麼」（回答老師）
1. **frozen 診斷 backbone**：prototype-guided aggregation + class text classifier，把「一組被選到的 patch」變成 slide 預測。
2. **navigation 訊號**：prototype/text 相似度 → 餵進 Observation State。
3. **弱監督 / proxy reward**：slide label / QPMIL confidence（在醫師軌跡資料出現前的代理目標）。

> 寫法紀律：QPMIL-VL 是 **prompt/prototype-based diagnostic backbone**，**不是** replay、**不是**我們要打敗的對手。

## 關鍵釐清：QPMIL 有 recency 遺忘問題嗎？
- **它的分類器**：被設計成**不太忘**（這正是它 SOTA 的賣點）。所以我們**不能**主打「我們解分類器遺忘」——那它早做了，我們會輸。
- **budgeted navigation 這一層**：QPMIL **根本沒有**。一旦你加上「只能看 K 個 patch、且要持續學這個挑選行為」，**navigation policy 會遺忘**（我們的舊證據：reverse old ESCA 的 navigation 崩到 0.133）。
- 結論：**我們解的遺忘，發生在 QPMIL 沒碰過的 navigation 層**；全看（no budget）時根本沒有 navigation 可忘。**這就是我們存在的縫隙。**

→ 詳見 [03 成功判準](03_success-criteria-experiments.md) 的「為何不用贏 QPMIL」。
