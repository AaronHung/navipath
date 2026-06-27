# 04 · 通用化定位 與 歸因（attribution）

> 來源：2026-06-27 老師回饋 + Aaron 對齊。

## 1. 兩條軸，不要混

| 軸 | 是什麼 | 我們的立場 |
|---|---|---|
| **軸 1：診斷 backbone（CL 家族）** | prototype/prompt-based（QPMIL-VL）、replay-based、regularization-based… | CNL **backbone-agnostic**，可疊在任一家之上；QPMIL 只是**插在 backbone 槽的一個 instance** |
| **軸 2：我們的貢獻 module ＝ CNL / NSM** | navigation 記憶層（怎麼看 + 跨任務不忘） | 這才是**獨立、通用**的 module；其記憶可實例化成 per-task / adapter / prototype-prompt / replay / regularization |

**關鍵措辭**：我們**不是泛化 QPMIL**（那是別人分類器的工作）。
我們提出一個**獨立通用的 Navigation 記憶層**；QPMIL-VL 只是實驗時的一個 backbone 例子。

> **更新註記（2026-06-27，Aaron 校正「通用 module」定義）**
> 「通用 module」＝**我們論文貢獻的本體**＝CNL 這套 Agent 骨架＋肉（Observation State + Navigation Policy + Sequential Observation + NSM + Gate + CL Update）。它**概念上獨立通用**，**不是拿來跟 QPMIL 比較的東西**。
> backbone **可換**（未來經小型接口改造即可換 replay-based / prototype-prompt / 其他），但**能不能插進來，取決於「我們這個 module 需要什麼 input」**＝介面契約。
> 通用性**有邊界、且隨版本演進**：本版內部沒有 transformer、backbone frozen → 預測與全域 attention 綁死的 backbone 進不來；未來加 PEFT/LoRA 契約放寬。
> ⇒ 這份「介面契約 + 相容範圍」要**版本化、持續記錄**，見 **[05](05_interface-contract-and-compatibility.md)**。reviewer 一定會問「你說 general，到底什麼能插、什麼不能」。

> 敘事紀律：**不以 QPMIL 開頭**。開頭講通用問題（agentic + continual WSI navigation），QPMIL 只在 method/experiment 當 Phase-0 instance 出現。大氣度、可被任何讀者理解。

## 2. 歸因（老師：贏了憑什麼是你的機制，不是 QPMIL 的 CL？）

**設計上保證 attribution 乾淨：**
- **全程凍結同一個 QPMIL backbone**（含它自己的 CL），在所有對照組完全一致。
- **只變動我們的 navigation 層**：有/無 NSM、序列/單步、State 組成。
- ⇒ 任何效能差異**只能歸因於我們的 navigation 記憶**，與 QPMIL 的分類器 CL 無關。

這也是為什麼 backbone 必須 frozen——不只是省事，而是**因果隔離**。

## 3. EWC 等的定位（Aaron 已同意）
EWC / naive finetune 等 = **新問題的對照 baseline**（套在 navigation 層上），合乎本次設計即可。
**不是**「比較哪種機制好」的分析型主張。

## 4. 老師 bottom line（鎖定方向）
> 在現有的 WSI navigation agent 之上，擴增 CL 能力。
- 7/3 報告：不報 selector 進度；改報「agent + CL 的預計架構（含架構圖）」+「到 7/2 為止跑出的實驗結果」。
- 承認的舊弱點（不再犯）：Top-K 在 encode-all 下省不了算力（動機弱）；無 CL 的 selector 會忘不算貢獻；過度 defensive 的 decoupling framing 不耐檢視。
