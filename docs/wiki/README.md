# NaviPath-CL 知識庫 Wiki（append-only）

> 這是我們的**設計 / 技術 / 架構 wiki**——隨時可讀、持續累積。
> 規則：**只增不刪**，舊認知要改時用「更新註記」標日期，不直接覆蓋，避免遺忘脈絡。
> 對外敘事禁用詞：mechanism probe / selection、selector、GO-NO-GO、「比較 shared/EWC/per-task」、compute-saving、forgetting 診斷。

## 索引

| 編號 | 主題 | 一句話 |
|---|---|---|
| [01](01_architecture.md) | 架構與名詞（Agent/CL/State/Gate/NSM） | 我們到底做什麼、Agent 在哪、CL 在哪 |
| [02](02_qpmil-vl-relationship.md) | 與 QPMIL-VL 的關係 | 我們用了它什麼、跟它差在哪、它的 prototype/prompt 為何不是我們的 policy |
| [03](03_success-criteria-experiments.md) | 成功判準與實驗設計 | 什麼數字算 POC 成功、跟誰比、為何不用贏 QPMIL |
| [04](04_generalization-and-attribution.md) | 通用化定位 與 歸因 | 通用 module＝我們的貢獻本體；贏了憑什麼是我們的機制 |
| [05](05_interface-contract-and-compatibility.md) | 介面契約 與 相容性範圍（versioned） | backbone 要給什麼才能插進來、什麼能插什麼不能、隨版本演進 |
| [06](06_glossary.md) | 架構 block 白話詞彙表 | 圖上每個 block 在說什麼（看板 `site/glossary.html`） |

## 相關正式文件
- 架構決策：`specs/decisions/ADR-0006-*.md`（authoritative 架構圖 + 標題 + 對應表）
- pivot 緣由：`specs/decisions/ADR-0001-*.md`
- 敘事錨（待改寫為新架構）：`STORYLINE.md`

## 更新日誌
- 2026-06-27：建立 wiki，新增 01/02/03。
- 2026-06-27（晚）：新增 04（通用化/歸因）、05（介面契約/相容範圍, versioned）；校正「通用 module＝我們的貢獻本體」。
- 2026-06-27（晚）：新增 06（架構 block 白話 glossary），同步看板 `site/glossary.html`。
