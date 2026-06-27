# 05 · 介面契約 與 相容性範圍（versioned · append-only）

> 來源：2026-06-27 Aaron 校正「通用 module」定義。
> **核心原則**：我們的 CNL（Agent/CL 本體）是獨立通用的貢獻 module。它的「通用性」**由它對 backbone 的 input 需求（＝介面契約）界定**。backbone 可換，但必須滿足契約；契約**隨版本演進**，每次改設計就在此更新，避免偷偷擴大宣稱。
>
> 受眾：① reviewer 會問「你說 general，到底什麼能插、什麼不能？」② 老師/內部要知道路線可達與邊界 ③ 我們自己改設計時要清楚邊界。

---

## 0. 我們是什麼（重申，別再混）
通用 module ＝ **CNL** ＝ Observation State + Navigation Policy + Budgeted Sequential Observation + NSM + Context Gate + CL Update。
這是**論文貢獻本體**。QPMIL-VL 只是 Phase-0 插進 backbone 槽的一個例子，**不是比較對象**。

---

## 介面契約 v0（Phase-0）

### backbone 必須提供（Required）
| 代號 | 契約 | 為什麼我們的 module 需要 |
|---|---|---|
| **R1 per-patch features** | `encode(WSI) → Z ∈ R^{n×d}`，固定維度 per patch | State/Policy 以 **patch 為決策單位**；backbone 必須暴露 patch-level 特徵，不能只給單一 slide embedding |
| **R2 subset-defined prediction** | `predict(S) → logits`，對**任意 patch 子集 S** 有定義且穩定（理想：permutation-invariant 聚合，如 prototype / MIL pooling） | 我們做 **budgeted 子集觀察**；若 backbone 必須吃「全部 patch / 全域 self-attention」才能預測，子集化會壞 |
| **R3 relevance / confidence 訊號** | per-patch relevance（prototype/text similarity）或可隨已觀察子集更新的 confidence | 供 **Observation State** 累積證據、驅動序列決策 |

### 可選（Optional）
| 代號 | 契約 | 用途 |
|---|---|---|
| **O1 task_query** | `task_query(WSI) → q` | task-free Context Gate（future） |

### 本版「我們沒有的」＝相容性的硬邊界
- 內部**沒有 transformer**、policy 是輕量 scoring、**不吃 attention map 當 input** → 以**全域 self-attention 為核心、預測與全域 attention 綁死**的 backbone 此版**收不進來**（無法 clean 子集化、也無處消化 attention 訊號）。
- backbone 視為 **frozen、不反傳** → 必須 end-to-end 訓練 encoder 才能用的方法此版**收不進來**。

---

## 相容性範圍 v0（Phase-0）
| 判定 | backbone 類型 | 條件 |
|---|---|---|
| ✅ in-envelope | prototype/prompt-based（**QPMIL-VL**）、replay-based、regularization-based 診斷 CL | 能提供 (R1,R2,R3) |
| ⚠️ 邊界（逐一驗證） | attention-based MIL（ABMIL / CLAM 類） | attention pooling 在**子集上仍 well-defined** 才收；若預測對子集極不穩定則否 |
| ❌ out-of-envelope (v0) | 需要全 bag / 全域 self-attention 才能預測、或必須訓練 encoder 的 backbone | 違反 R2 或 frozen 假設 |

---

## 版本演進（持續更新 changelog）
| 版本 | 我們 module 的關鍵設計 | input 需求變化 | 相容範圍變化 |
|---|---|---|---|
| **v0 (Phase-0, 現在)** | 輕量 scoring policy；no transformer；backbone frozen | R1+R2+R3，O1 optional | prototype/prompt · replay · regularization；attention-based 看子集穩定性；需訓練 encoder 者排除 |
| v1 (future, PEFT/LoRA) | backbone 上加 trainable adapter/LoRA；State 可吃 attention 訊號 | R2 的「frozen」鬆綁；可吃 attention-based | **擴大**：attention-based、可 end-to-end 微調者納入 |

> 規則：**每次改 module 的 input 需求 / 內部結構 → 在此表新增一列（日期 + 改了什麼 + 範圍變化）**，舊列不刪。

---

## 一句話 reviewer 答覆（safe statement）
> Our contribution is a backbone-agnostic continual navigation module. It is general **with respect to a defined interface contract**: any backbone exposing patch-level features, subset-defined (permutation-invariant) prediction, and a per-patch relevance signal can be plugged in. In this work we instantiate it with QPMIL-VL; replay-based and regularization-based diagnostic backbones satisfy the same contract. Backbones whose prediction is inseparable from global self-attention over all patches are out of scope in this version, and become compatible once PEFT/LoRA adaptation is introduced.

## 更新日誌
- 2026-06-27：建立；定義 v0 介面契約（R1/R2/R3,O1）+ 相容範圍 + v0→v1 演進表。
