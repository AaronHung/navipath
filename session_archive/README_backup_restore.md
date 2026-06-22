# Session Archive — Cursor 對話備份與還原

這個資料夾用來保存 **Cursor Agent 的完整對話逐字稿（transcript）**，避免 context 被壓縮後，中間的說明、指令、判讀等有用訊息永久消失。

- `*.jsonl` / `*.zip`：被 `.gitignore` 忽略（內容大且屬私人對話，不進 GitHub）。
- 本 `README` 有進 git，所以備份/還原方法不會遺失。

---

## 背景：為什麼要手動備份

- Cursor 的對話會隨 context 變長而**自動壓縮（summarize）**，舊訊息會被裁掉，之後**再也看不到原文**。
- 對話逐字稿存在本機（不在 repo）：
  ```
  ~/.cursor/projects/<專案雜湊>/agent-transcripts/<session-id>/<session-id>.jsonl
  ```
  本專案的雜湊目錄：
  ```
  /Users/aaron/.cursor/projects/Users-aaron-research-01-navipath/agent-transcripts/
  ```
- 每個 session 一個資料夾（名字是 session UUID），裡面一個同名 `.jsonl`，**一行一則訊息**（user / assistant）。

---

## 備份（把目前 session 複製進來）

### 方法 A：手動一次性備份

```bash
cd /Users/aaron/research/01_navipath

# 1) 找出最近更新的 transcript（最上面那個就是最新 session）
ls -lat ~/.cursor/projects/Users-aaron-research-01-navipath/agent-transcripts/*/*.jsonl | head

# 2) 複製進來，檔名加日期 + session 短碼
SRC="<上一步看到的完整路徑>"
cp "$SRC" "session_archive/$(date +%Y-%m-%d)_navipath_$(basename "$SRC" .jsonl | cut -c1-8).jsonl"

# 3) 確認
ls -lh session_archive/
```

### 方法 B：一鍵備份「最新」session

```bash
cd /Users/aaron/research/01_navipath
SRC=$(ls -t ~/.cursor/projects/Users-aaron-research-01-navipath/agent-transcripts/*/*.jsonl | head -1)
cp "$SRC" "session_archive/$(date +%Y-%m-%d_%H%M)_navipath_$(basename "$SRC" .jsonl | cut -c1-8).jsonl"
echo "backed up: $SRC"
ls -lh session_archive/
```

> 建議：在每次「重要 session 結束前」或「感覺 context 快滿」時做一次。

---

## 還原 / 閱讀（把備份內容看回來）

`.jsonl` 是純文字，一行一則訊息。幾種讀法：

```bash
# 看有幾則訊息
wc -l session_archive/<檔名>.jsonl

# 純文字逐則瀏覽（用 jq 美化；沒裝 jq 就直接 less）
jq -r '.role + ": " + (.content|tostring)' session_archive/<檔名>.jsonl 2>/dev/null | less
less session_archive/<檔名>.jsonl

# 搜尋某個關鍵字（例如某個指令或結論）
grep -i "old-task" session_archive/<檔名>.jsonl
```

### 餵回給新的 Cursor / Claude session

context 壓縮後若要讓新 session「接上」：
1. 把對應 `.jsonl` 用 `less` 打開，找出需要的段落。
2. 把關鍵段落貼進新 session，或
3. 直接告訴新 Agent：「請讀 `session_archive/<檔名>.jsonl` 的第 N–M 行」（Agent 可用 Read 工具讀本機檔）。

---

## 進階：壓縮存檔（多個 session 打包）

```bash
cd /Users/aaron/research/01_navipath
zip -j session_archive/navipath_sessions_$(date +%Y-%m-%d).zip session_archive/*.jsonl
# .zip 也被 gitignore，純本機保存；要長期保存可另外複製到雲端硬碟。
```

---

## 注意事項

- 這些 `.jsonl` **不會上 GitHub**（已 gitignore），屬本機私人備份。要異地保存請自行複製到雲端 / 外接碟。
- 若換電腦，`~/.cursor/projects/...` 路徑中的專案雜湊會不同，但 `agent-transcripts/<uuid>/<uuid>.jsonl` 的結構相同。
- 一行就是一則訊息；檔案可能很大，避免整檔貼進對話，用 `grep` / 行號範圍擷取。
