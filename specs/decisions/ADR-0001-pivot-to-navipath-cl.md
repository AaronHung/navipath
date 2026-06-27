# ADR-0001 — Pivot to NaviPath-CL

- Status: Accepted
- Date: 2026-06-27
- Supersedes: 舊主軸「selector forgetting 診斷 + Top-K budget」

## Context

舊主軸（trainable patch selector 會忘、Top-K 省算力）被老師打掉，理由成立：
- encode-all-then-aggregate 下，貴的 CONCH 抽取已完成 → 「省算力」站不住。
- 沒有 CL component 的 selector 會忘是預期，不構成貢獻。
- decoupled backbone 的 Forgetting=0 是結構恆等，不是成果。

老師 bottom line：拿出「WSI navigation agent + CL 能力」的正面架構。

## Decision

主軸 pivot 為 **NaviPath-CL**：在 budgeted/agentic WSI 設定下，研究 **navigation policy 本身的 continual learning**。核心主張：WSI 持續學習不只學「怎麼分類」，也學「怎麼看」；observation policy 也會 catastrophic forgetting。

舊結果不丟，降級為 motivation + ablation。

## Consequences

- 命名與框架見 ADR-0003 / ADR-0004。
- 不再主打 compute-saving；budget 改述為 agentic observation constraint。
- EWC 改定位為 negative baseline（ADR-0003）。
- 敘事真相落在 [`../../STORYLINE.md`](../../STORYLINE.md)。
