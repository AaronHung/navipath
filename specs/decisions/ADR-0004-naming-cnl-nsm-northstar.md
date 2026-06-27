# ADR-0004 — Naming: NaviPath-CL / CNL / NSM / North Star

- Status: Accepted
- Date: 2026-06-27

## Context

需要固定對內/對外用詞，避免後續 session 漂回 "selector" framing 或洩漏上位計畫。

## Decision

- 整體方法：**NaviPath-CL**。
- 貢獻層：**Continual Navigation Layer (CNL)**。
- 核心模組：**Navigation Skill Memory (NSM)**。
- 上位願景代稱：**North Star**（= physician-like WSI navigation agent 長期願景）。
- 題目：*NaviPath-CL: A Continual Navigation Layer for Agentic Whole-Slide Image Diagnosis*。

兩套語言（不可混用）：
- 對內 / North Star：Phase-0 prototype for physician-like WSI navigation agent。
- 對 paper reviewer：continual learning for budgeted WSI observation policies。

## Consequences

- 「selector」不再當主詞，一律 navigation policy。
- **保密**：所有文件以 North Star 代稱，不出現計畫名/單位/主持人/頁碼。
- 對外論文不過度講 North Star（內部合理性 ≠ paper 賣點）。
