# Overleaf SOP — 第一次投 COMPAYL @ MICCAI 2026

> 給第一次用 Overleaf / 第一次投稿的你。照著做就好。
> 官方規定(已查證 2026/06):8 頁正文(含圖表)+ 最多 2 頁 references;**不可改 template**
> (改 margin/行距/字體 = desk reject);**匿名**;經 **OpenReview** 投稿;
> **supplementary 只能放影片,禁止 PDF/文字/分析**;featured 主題含 *agentic AI in pathology*。

---

## 0. 名詞先懂(30 秒)
- **template(範本)**:出版社規定的排版格式。MICCAI 用 **Springer LNCS**。你**必須**用它,且不能改格式參數。
- **`main.tex`**:Overleaf 專案的「主檔」,LaTeX 從這裡開始編譯。template 自帶一份。
- **preamble(前導區)**:`\documentclass...` 到 `\begin{document}` 之間,放 `\usepackage`、巨集設定。
- **body(正文)**:`\begin{document}...\end{document}` 之間,放真正的內容(章節、圖表)。
- **`.bib`**:參考文獻資料庫(我已備好 `references.bib`)。
- **bibtex**:把 `.bib` 編成文末 reference 清單的工具。

---

## 1. 「main.tex wrapper」到底是什麼、你需要嗎?
- 「wrapper」= 一個只放 preamble、再用 `\input{paper_body.tex}` 把正文拉進來的 `main.tex`。好處是內容與設定分開。
- **你不需要它。** 兩條更簡單的路:
  - **路 A(推薦,合規最安全)**:用官方 LNCS template 的 `main.tex` 當主檔,把我的內容貼進去。
  - **路 B(最省事,但要自己確認合規)**:直接把我的 `paper_body.tex` 當 `main.tex`(它本身就能編譯)。風險:我的 preamble 不是官方那份,投稿前要核對格式沒被改到。
- 結論:**走路 A**。template 自帶的 `main.tex` 就是你的 wrapper,不用我另外做。

---

## 2. 第一次完整流程(路 A,一步步)

### Step 1 — 註冊 + 拿官方 template
1. 到 overleaf.com 註冊(免費帳號夠用)。
2. 拿 MICCAI/LNCS template,二選一:
   - 從 **COMPAYL 網站 / OpenReview 投稿頁**找官方提供的 Overleaf 連結(最準,直接「Open in Overleaf」會複製一份到你帳號)。
   - 或 Overleaf 首頁 **New Project → Templates → 搜尋 "Springer LNCS"** → 用最新版(含 *Disclosure of Interests* 段落那版)。
3. 開啟後你會看到一個專案,主檔通常叫 `main.tex` 或 `samplepaper.tex`。

### Step 2 — 放進我們的素材
1. 左側檔案列 **Upload** → 上傳 `references.bib`。
2. 新增資料夾 `figs/`(或直接上傳到根目錄)→ 上傳 `paper/figs/` 裡 7 個 `.pdf`。
3. 確認 template 主檔最後是 `\bibliographystyle{splncs04}` + `\bibliography{references}`(我們的 bib 檔名就是 `references`)。

### Step 3 — 把內容搬進 template(用 `paper_body.tex` 的標記)
打開我做的 `paper/paper_body.tex`,裡面有標記:
- 把 `[[PREAMBLE TO COPY]] ... [[END PREAMBLE TO COPY]]` 之間的 `\usepackage`/`\newcommand`
  **貼到 template `main.tex` 的 preamble**(`\begin{document}` 之前)。這些只是「加套件/巨集」,
  **不是改格式**,合規。
- 把 `[[BODY START]]`(`\begin{document}`)到 `[[BODY END]]`(`\end{document}`)之間的內容,
  **取代** template body 裡的範例內容。**保留** template 原本的:
  - 匿名 `\author{}` / `\institute{}` 區塊(別填真名,投稿要匿名);
  - 文末 **Disclosure of Interests**(`\begin{credits}...\end{credits}`,我也已寫一份,可對齊)。
- `\graphicspath{{figs/}}`:若你的圖放在根目錄就改成 `\graphicspath{{./}}` 或拿掉。

### Step 4 — 編譯
1. 上方選 **Recompile**。編譯器用 **pdfLaTeX**(Menu → Compiler 確認)。
2. 第一次跑完文獻可能顯示 `[?]`;Overleaf 會自動跑 bibtex,**再按一次 Recompile** 就會出現引用編號。
3. 看右側 PDF。常見錯誤:
   - 圖找不到 → 檢查 `figs/` 路徑與檔名大小寫。
   - `Undefined control sequence \discintname` → template 太舊,換最新版 LNCS。
   - 引用變 `[?]` → 確認 `\bibliography{references}` 檔名對、且有 `\cite{}` 用到。

### Step 5 — 檢查頁數合規
- 正文(到 conclusion/acknowledgement)**≤ 8 頁**;references 另計 **≤ 2 頁**。
- **絕對不要**為了塞下而改行距/margin/字體 → 會被 desk reject。要縮就**刪字、縮圖**。

### Step 6 — 邀請 team 共編(不同顏色 review)
1. 右上 **Share** → 用 email 邀請,或開 **link sharing**(Can edit)。大家同一專案即時共編。
2. 想要「不同顏色的修改痕跡」:用 **Review** 面板的 **Track Changes**(每人一色)+ **Comment**(留言討論)。
   - 註:Track Changes 在免費版有限制,通常需要其中一人有付費/機構方案才能全開;留言功能免費可用。

### Step 7 — 投稿
1. Menu → **Download → PDF** 拿到投稿 PDF;再 **Download → Source (zip)**(camera-ready 或被要求原始碼時用)。
2. 到 **OpenReview** COMPAYL 2026 投稿頁上傳 PDF。確認 PDF 是 **searchable**(文字可選取)、**匿名**。

---

## 3. 關於 supplementary(重要,直接回答你的疑問)
- COMPAYL/MICCAI 2026:**supplementary 只能是多媒體(avi/mp4/wmv),嚴禁 PDF/文字/proof/分析/額外結果**,違反 = desk reject。
- 所以我們的 **fairness 報告不能當 supplementary 投出去**。
- 正確用法:
  1. **公平性的核心一句話已在正文 §4.1**(matched 12 epochs/lr/wd、所有 selector 餵同一 frozen head),
     是**正面陳述**,不是防衛性辯解 → reviewer 讀正文就懂,不會更被 challenge。
  2. `FAIRNESS_sanity_check_zh.md` / `paper/FAIRNESS_sanity_check_en.md` → **給教授看 + 留作 rebuttal 彈藥**。
     真的被 reviewer 質疑公平性時,再用裡面的 Q&A 回。**先不要主動寫一段防衛**(預先過度解釋反而像心虛)。

---

## 4. 一句話總結
走官方 LNCS template(路 A)→ 貼我的 preamble + body → 上傳 `references.bib` + `figs/` →
pdfLaTeX 編兩次 → 控制 8+2 頁、不動格式 → Share 給 team 共編 → OpenReview 投。
fairness 留給教授與 rebuttal,**不**進 supplementary。
