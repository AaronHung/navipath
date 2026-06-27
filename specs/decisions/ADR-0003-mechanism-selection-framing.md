# ADR-0003 — Mechanism-Selection framing as the main experimental logic

- Status: Accepted
- Date: 2026-06-27

## Context

我們已有 shared / EWC / per-task 三組 router 結果。若只當「四個 variant 的對齊表」，敘事偏防守。North Star 的 continual 機制設計光譜（共用 module / weight 正則 / per-task LoRA / parameter merging）正好對應我們的 variant。

## Decision

把實驗主邏輯定為一個設計問題：**Which continual mechanism is suitable for a WSI navigation agent?** 並以決策樹呈現：

- Q1 shared policy 能學所有任務？→ No（recent GO / old NO-GO）。
- Q2 weight 正則（EWC）能修？→ Not sufficiently（old ~0.40）。
- Q3 舊 navigation skill 真的丟了？→ No（per-task skill memory 恢復 0.933/1.0）。
- 結論：問題不在缺診斷訊號，而在缺 continual navigation memory。

機制對應 North Star：shared↔單一導覽 agent；EWC↔weight 正則/alignment；per-task↔PEFT per-task LoRA；consolidation↔parameter merging。

## Consequences

- EWC 是 **negative baseline**（它失敗反而說明 navigation forgetting 更結構性），不包裝成主解法。
- consolidation / parameter-merging **只講 ongoing，不 claim solved**（無 navigation 版數字）。
- 安全措辭：mechanism *probe*，非 *solved*。
- 細節見 [`../../STORYLINE.md`](../../STORYLINE.md) §4–§5。
