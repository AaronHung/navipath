# NaviPath-CL

**A Continual Navigation Layer for Agentic Whole-Slide Image (WSI) Diagnosis.**

WSI 持續學習不只是學「怎麼分類」，也是學「**怎麼看**」。在有限 observation budget 下，agent 學了新癌種後，會忘記舊癌種「該看哪裡」——既有 WSI-CL 只處理 classifier forgetting，沒人處理 **navigation-policy forgetting**。NaviPath-CL 提出一個 **backbone-agnostic 的 Continual Navigation Layer (CNL)** 來解這個新問題。

> QPMIL-VL 只是我們實驗時插在 backbone 槽上的**一個 instance**，不是主角，也不是要打敗的對手。

---

## 核心概念（30 秒版）

| 模組 | 角色 |
|---|---|
| **CNL**（Continual Navigation Layer） | 我們的貢獻層，backbone-agnostic |
| **Observation State** | 單張 slide 內累積證據（Agent 短期記憶）→ 序列決策 |
| **Navigation Policy + Budgeted Sequential Observation** | 有限 budget 下多步決定「下一步看哪」 |
| **Navigation Skill Memory (NSM)** | 跨任務的 navigation 技能記憶（CL 長期記憶） |
| **Context / Task Gate** | 從 NSM 取對的技能（現在 oracle，未來 task-free） |
| **Frozen Diagnostic Backbone** | 凍結的診斷模型；Phase-0 instance = QPMIL-VL + CONCH |

- **Agent 在哪**：Observation State + Policy + Sequential Observation（怎麼看）。
- **CL 在哪**：Gate + NSM + CL Update，沿任務流不忘（怎麼不忘）。
- **歸因**：全程凍結同一 backbone，只動 navigation 層 → 效能差異只能歸因於我們的記憶。

---

## 看板與文件入口

| 想看什麼 | 去哪裡 |
|---|---|
| 架構圖（總圖 + 細部）、Agent/CL 在哪 | `site/architecture.html` |
| Storyline（敘事錨、論文基調） | `STORYLINE.md` |
| 設計 / 技術 / 架構 wiki | `docs/wiki/` |
| SOP（6/27→7/20 里程碑） | `specs/01_sop_navipath-cl_phase0.md` · 看板 `site/sop.html` |
| RunPod 操作（登入/setup/tmux/同步/踩坑） | `RUNPOD_RUNBOOK.md` |
| 進度看板 / 日程 | `site/board.html` · `site/schedule.html` |
| 架構決策（authoritative） | `specs/decisions/ADR-0006-*.md` |
| 規格（SDD） | `specs/` |
| 實驗結果 | `outputs/` |

### 怎麼看 HTML 看板
```bash
open -a Safari site/index.html      # 或 architecture.html / board.html / schedule.html / sop.html
```
（自包含 HTML，圖在 `site/figs/`，Safari `file://` 可直接開。）

---

## 開發原則（SDD）
- **Mac 改 code + smoke；RunPod 跑實驗。** 結果存 `outputs/`、不覆蓋舊檔。
- 每個里程碑更新 `specs/worklog/WORKLOG.md` 與 `outputs/PROGRESS.md`。
- 重大架構決策寫 ADR（`specs/decisions/`）。

## 目錄
```
STORYLINE.md            敘事錨
README.md               本檔
specs/                  SDD 規格、ADR、SOP、worklog
docs/wiki/              設計 / 技術 / 架構 wiki
site/                   HTML 看板（storyline / 架構 / 看板 / 日程 / SOP）
navipath_moe/           核心程式（agent / backbone adapter / NSM …）
outputs/                實驗結果（進 git）
reports/                雙月 / 年度報告
paper/                  論文
legacy/                 舊 v0.4 文件（待整理，勿污染新脈絡）
```

> 舊版 v0.4 文件已歸檔於 `legacy/`（含舊 README）。
