# 01 · 架構與名詞（Agent 在哪、CL 在哪）

> 來源：2026-06-27 對話拍板。authoritative 架構圖見 `specs/decisions/ADR-0006-*.md`。

## 一句話定位

NaviPath-CL = 在 frozen 診斷 backbone 之上，加一層 **Continual Navigation Layer (CNL)**：
讓模型在**有限 observation budget** 下，像醫師一樣**多步決定「下一步看哪裡」**，
並且這個「怎麼看」的能力能**跨任務流持續學習而不遺忘**。

> 既有 WSI-CL 只讓**分類器**持續學習；我們第一個讓「**怎麼看**(navigation policy)」持續學習。

## 五個核心名詞（Q&A 版）

### Q1. 新架構和 per-task 一樣嗎？
不一樣。per-task **不再是主張**，它降格成一個**零件**——是 **NSM 的 Phase-0 實作方式**。
我們不再「比較 per-task / shared / EWC」（那是分析型思維），而是「**用** per-task skill bank 來**實作**記憶模組」。

### Q2. per-task 的 replay 整個丟棄嗎？ —— per-task ≠ replay
兩個不同模組，別混：
- **NSM（Navigation Skill Memory，技能記憶）**：存「**怎麼看**的技能」（navigation policy 參數）。Phase-0 = per-task skill bank。**← 這次做**
- **FRM（Feature Replay Memory，特徵回放）**：存「**代表性 patch 特徵**」拿來 rehearse。**← future（Phase-2），這次不做**

所以 per-task 留下（當 NSM），FRM 本來就還沒做、列 future。沒有「丟棄」問題。

### Q3. Memory 設計在哪？ —— 兩層記憶
- **長期記憶 = NSM（跨 slide / 跨 task）**：學完一個任務（癌種/器官），把該任務的 navigation 技能存進 bank。**這是 CL 的記憶。**
- **短期記憶 = Observation State（單張 slide 內）**：讀這張片子時，累積「已經看到什麼證據」。**這是 Agent 的記憶。**

### Q4. State 是什麼？
**Observation State** = agent 讀**單張** slide 時的累積信念：
已看 patch 的聚合特徵 ＋ running 的 prototype/text 相似度摘要 ＋ 目前 backbone 預測信心 ＋ 覆蓋率。
它讓「下一步看哪」**取決於已經看到的東西** → 這就是序列決策，也產出 Navigation Trace。

### Q5. Gate 是什麼？
**Context/Task Gate** = 決定這張 slide 要從 NSM 取**哪一個**技能。
Phase-0 = oracle（已知 task id，當 upper bound）；future = task-free（從 slide 自己推 task/domain）。
它是 CL 裡「在多個技能之間路由」的開關。

## Agent 在哪、CL 在哪（老師必釘）

| 問題 | 答案 | 架構模組 |
|---|---|---|
| **Agent 在哪？** | 有限 budget 下、依累積 Observation State、**多步決定下一步看哪**，並輸出可解釋 Navigation Trace | Observation State + Navigation Policy + Sequential Budgeted Observation |
| **CL 在哪？** | 面對**任務流**時，navigation 技能被持續累積/路由/更新，舊任務不退化 | NSM + Context Gate + CL Update |
| **WSI 機制怎麼接？** | WSI 是 gigapixel → budgeted navigation 是 WSI 專屬必要機制；CL 加在**navigation policy 上**（非分類器） | Feature Interface(CONCH) → 整條 navigation |

## 價值鏈（一條龍）

```text
WSI 是 gigapixel  →  必須 budgeted 觀察(不能全看)              [WSI 機制]
→ 用會累積證據的 Observation State 做多步「下一步看哪」         [Agent]
→ frozen QPMIL-VL 對選到的證據做診斷推理 + 當弱監督訊號          [Backbone]
→ 面對任務流，navigation 技能存進 NSM、由 Gate 路由、CL Update 維護  [CL]
→ 舊癌種「該看哪」不退化，新癌種也學得會                       [成果]
```

## 新肉 = Sequential Budgeted Observation（這次甩開「單步選擇器」的關鍵）
不再一次性挑 K 個，而是分 T 回合：每回合依 Observation State 決定「下一小批看哪」→ 更新 state → 直到用完 budget → 輸出 Navigation Trace（看的順序＝可解釋）。在 precomputed CONCH 特徵上即可實作（不需重新編碼）。
