# ADR-0005 — SDD workflow & records

- Status: Accepted
- Date: 2026-06-27

## Context

Aaron 要求：不要「想一個做一個」，要 spec-driven development；所有計畫、過程、output、筆記、重要事項都要記錄好（明定為契約）。

## Decision

採用 `specs/` 體系（見 [`../README.md`](../README.md)）：
- `00_master_spec.md`（技術索引）、`decisions/ADR-*`（決策，append-only）、`features/SPEC-*`（可交付物 spec）、`worklog/WORKLOG.md`（過程，append-only）。
- 流程：SPEC → (ADR) → 實作/執行 → 驗收 → 回寫 WORKLOG → commit（訊息帶 SPEC/ADR 編號）。
- 分工：`WORKLOG.md`（開發過程）vs `outputs/PROGRESS.md`（實驗執行），交叉引用不重抄。

## Consequences

- 每個 milestone 動工前必有 SPEC 與 acceptance criteria。
- 翻案開新 ADR，不改舊的。
- 主計畫檔（`.cursor/plans/*.plan.md`）為唯讀參考，不在 specs/ 內維護。
