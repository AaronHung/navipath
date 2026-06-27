# ADR-0002 — QPMIL-VL is a backbone, not a replay method (and not a competitor)

- Status: Accepted
- Date: 2026-06-27

## Context

讀 QPMIL-VL 論文（arXiv:2410.10573）後釐清其 CL taxonomy 定位。論文自分三類 IL：
- Regularization-based（EWC, LwF）—— 在 WSI 幾乎失效。
- Rehearsal/Replay-based（A-GEM, ER, DER++, ConSlide）—— 需 buffer。
- **Prompt/Prototype-based（QPMIL-VL 本身，靈感自 L2P/S-Prompts）—— rehearsal-free**，論文強調 "without relying on extra buffer"。

QPMIL 內部的 prototype key-query matching + matching penalty 本質是一個 **task-free domain routing 機制，作用在表徵/分類層**——與我們要在 navigation policy 層做的 continual 機制概念同構。

## Decision

- QPMIL-VL 定位為 **prompt/prototype-based diagnostic backbone + weak supervisory signal**。
- **不寫 replay**；**不當作要打敗的對手**。
- 我們的 CNL 是 backbone-agnostic 的一層，QPMIL-VL 只是本次唯一實例化的 backbone instance。
- QPMIL 的 query vector / prototype-match frequency 是未來 task-free context gate 的現成輸入訊號（future，見 ADR-0003 / SPEC-01 的 infer stub）。

## Consequences

- 本次只實例化一個 backbone（prompt/prototype-based）；跨 CL 家族（replay / regularization）驗證列 future。
- 固定寫法見 [`../../STORYLINE.md`](../../STORYLINE.md) §7。
