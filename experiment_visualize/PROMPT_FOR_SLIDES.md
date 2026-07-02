# Slide Generation Prompt

> Copy the entire content below (between the `---` lines) and paste it into Claude.AI or Gemini.
> Point the AI to read from: `/Users/aaron/research/01_navipath/experiment_visualize/`

---

## Instructions

You are helping me create a presentation slide deck (Google Slides / PowerPoint style) for a bimonthly research progress report. The audience is a committee of deep learning experts who may not be familiar with computational pathology or whole-slide image (WSI) analysis.

I have a directory at `/Users/aaron/research/01_navipath/experiment_visualize` containing all the materials you need:

- `experiments.md` — Full experiment writeup with tables, findings, and "Presentation talking points"
- `tables.tex` — LaTeX tables (5 tables, each with presenter context comments)
- `figs/` — 4 figures in PDF and PNG format:
  - `fig_budget_efficiency.png` — Budget efficiency curves
  - `fig_main_comparison.png` — NSM vs Naive vs Zero-shot bar chart
  - `fig_mechanism_tsne.png` — t-SNE visualization of router mechanism
  - `fig_lambda_sweep.png` — Sequential observation λ sweep

Please read all files in that directory, then produce a **7-slide** presentation deck with the following structure:

---

### Slide 1: Title + One-sentence summary

- Title: "NaviPath-CL: Continual Navigation for Whole-Slide Image Diagnosis"
- Subtitle: One sentence — "A backbone-agnostic framework that learns WHERE to look in gigapixel pathology slides, and remembers it across tasks."
- My name / affiliation / date

### Slide 2: Problem Statement (30 seconds)

- What is a whole-slide image? (gigapixel, thousands of patches)
- The budget problem: we can only look at K << N patches
- The continual learning setting: tasks arrive sequentially
- KEY INSIGHT: the diagnosis model doesn't forget (frozen backbone), but the navigation module (WHERE to look) forgets catastrophically
- Use simple language. Define "patch", "budget K", "frozen backbone" for non-pathology audience.

### Slide 3: Method Overview (30 seconds)

- 3 components: MicroRouter (learns where to look) + NSM (stores skills per task) + SBO (sequential multi-step selection)
- Emphasize: backbone-agnostic — our layer plugs onto ANY frozen foundation model
- Emphasize: only 132K parameters per task (lightweight)

### Slide 4: Experiment 1 — Navigation Works (data slide)

- Show Table 1 or `fig_budget_efficiency.png`
- Key message: "Router @K=64 (0.922) > All patches (0.892)"
- Talking point from `experiments.md`: "Our router learns WHERE to look. Using only 64 out of thousands of patches, it not only matches but exceeds the accuracy of examining the entire slide."

### Slide 5: Experiment 2 — Navigation Forgetting & Recovery (data slide, MOST IMPORTANT)

- Show `fig_main_comparison.png` or Table 2
- Key numbers: NSM mACC = 0.935, Naive = 0.595, Zero-shot = 0.858
- ESCA (oldest task): NSM 0.911 vs Naive 0.333 — catastrophic
- Naive is WORSE THAN RANDOM (0.333 < 0.800) — active mis-navigation
- Talking point: "The backbone never forgets HOW to classify. But the router forgets WHERE to look. NSM stores navigation skills per task → zero forgetting, +34 points over naive."

### Slide 6: Experiment 3 — Mechanism + SBO (data slide)

- Left: `fig_mechanism_tsne.png` — same slide, recent router vs forgotten router (t-SNE)
  - "Same feature space. Only routing scores differ. The information is preserved; the navigation skill is lost."
- Right: `fig_lambda_sweep.png` — λ sweep confirming SBO mechanism
  - "Sequential observation works; but tumors cluster spatially → optimal λ ∈ [0,1]"

### Slide 7: Summary & Next Steps

- Summary claims (from experiments.md "Summary of Claims" table):
  - Router > all-patch at K=64
  - NSM: 0.595 → 0.935, zero forgetting
  - Backbone-agnostic (frozen FM, any encoder)
  - SBO mechanism confirmed
- Next steps: LoRA-based cheap memory (30× fewer params/task), learned λ, multi-scale zoom

---

### Style Requirements

- Clean, modern academic style (white background, minimal text)
- Each data slide: ONE figure/table + 2-3 bullet points max
- Numbers in bold when they are key results
- Use the "Presentation talking point" quotes from experiments.md as speaker notes
- Avoid jargon where possible; when unavoidable, add a parenthetical definition
- Do NOT use the term "QPMIL" anywhere — say "frozen diagnostic backbone" or "diagnosis model" instead

### Output Format

Please produce the slide content as structured text I can copy into Google Slides:
- For each slide: Title, Body (bullets), Figure placement instruction, Speaker notes
- If you can produce actual slide markup (e.g., Marp markdown, reveal.js, or LaTeX Beamer), that's even better.

---
