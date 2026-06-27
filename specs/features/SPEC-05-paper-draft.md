# SPEC-05 — Paper draft (paper framing)

- Status: In progress（骨架就緒；7/3–7/15 填實，7/15 老師 review，7/20 年度報告）
- Milestone: M7 (7/3–7/15)
- Related: STORYLINE（§1 兩套語言、§6 contributions）, SPEC-02/03/06

## 1. 目標

把 Phase-0 pilot 寫成獨立 scientific paper。**Paper framing**：continual learning of the
observation policy under budgeted WSI inference。**不過度講 North Star**（內部合理性來源、且需保密）。

## 2. 現況與交付物

- 既有論文資產為 LaTeX：`paper/main.tex`、`paper/paper_body.tex`、`paper/references.bib`（反映**舊** framing）。
- **不直接覆寫 .tex**。本 SPEC 交付 `paper/NaviPath-CL_draft_outline.md`：NaviPath-CL framing 的章節重寫大綱 + 現有證據填入 + TODO，供 7/3–7/15 移植回 LaTeX。
- `reports/annual_2026-07-20_outline.md`：年度報告大綱。

## 3. 驗收標準

- [x] `paper/NaviPath-CL_draft_outline.md` 章節齊備（Abstract/Intro/Related/Method/Experiments/Discussion/Conclusion）、Method 用 CNL/NSM/backbone-interface、Experiments 引用 mechanism-selection 證據與圖、Limitations 列 oracle gate / future。
- [x] 不含 North Star 可識別資訊；framing 為獨立 WSI-CL 問題。
- [x] `reports/annual_2026-07-20_outline.md` 建立。
- [ ] （7/3–7/15 人工）移植回 `paper/*.tex` + 老師 review + RunPod 補數字。

## 4. 待 7/15 前補（人工 + RunPod）

- paper-order per-task/EWC 對稱補跑（SPEC-02 stretch）。
- agent 真數字重現（SPEC-06 RunPod）。
- 可選：task-free context gate / consolidation pilot 補強 contribution。
- 將大綱移植回 `paper/*.tex` 並依老師 review 修訂。

## 5. 不做的事

- 不覆寫既有 `.tex`、不宣稱 SOTA、不偽造未跑數字、不寫 North Star 細節。

## 6. Changelog

- 2026-06-27：建立 SPEC + 大綱（M7 動工）。
