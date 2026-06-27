# RunPod 重開後如果系統沒掉但是library蹦掉了—— setup（一行）

RunPod 重置環境後，跑任何實驗前先貼這行（裝缺的套件 + 確認 GPU）：

```bash
cd /workspace/src/navipath && pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

- 看到 `cuda available: True` 就可以開跑。
- 程式與結果都在 git，重裝只補這幾個套件（timm/transformers/pandas/pyyaml）。
- 之後接 `git pull` 取最新 code/結果，再跑 `train_router_v0.py` 等指令。

---

---

---

> # 更新版
>
> ## ——每次開機執行的RunPod恢復 dotfiles 和 links 系統程式安裝
>
> 做好了。你的**開機後手順**現在補上了工具和 claude CLI，更新到 [開機關機手順.md](vscode-webview://0kqoa3umob4q1ovdubp8soj50bh5gurmo2vl5urhb1sp2qhkdeth/開機關機手順.md)，也新做了一個一鍵腳本 [scripts/install_system_tools.sh](vscode-webview://0kqoa3umob4q1ovdubp8soj50bh5gurmo2vl5urhb1sp2qhkdeth/scripts/install_system_tools.sh)。

## A. 🍁 關機前手順（power off / Stop 之前一定要做）

```bash
# 1) 備份 /root 重要資料到 /workspace（含 Claude 對話歷史 / memory / 設定 / 憑證）
bash /workspace/scripts/backup_home.sh backup
```

看到綠字 **`✓ 已驗證：包含 N 個 Claude 對話歷史(.jsonl)`** 才算成功。
若看到紅字警告「找不到對話歷史」→ **先別關機**，回 §F 排查。

```bash
# 2) 再確認一次（看時間是現在、jsonl 數量合理）
bash /workspace/scripts/backup_home.sh snapshots
```

確認最新快照時間正確、`jsonl=` 數量合理後，再到 RunPod 介面關機 / Stop。

> ⚠️ 沒做第 1 步就關機 = 這次的對話歷史與設定永久消失。‼️

給你最新的完整版，重開機後照這個跑：

## 🦀 最新開機手順 —— 🟢 重開機後（一行一行貼）

**第 1 步**：載入環境（prompt 變 `(pt-exp): workspace#`）

```bash
source /workspace/bootstrap/env.sh
```

**第 2 步**：救回 Claude 對話/設定

```bash
bash /workspace/scripts/backup_home.sh restore
```

**第 3 步**：讓新終端自動就緒

```bash
grep -q 'source /workspace/bootstrap/env.sh' /root/.bashrc 2>/dev/null || echo 'source /workspace/bootstrap/env.sh' >> /root/.bashrc
```

**第 4 步**：重裝系統工具（git-lfs / ffmpeg / tmux / rsync / unzip…）

```bash
bash /workspace/scripts/install_system_tools.sh
```

**第 5 步**：重裝終端版 `claude` 指令（只在你要在終端打 claude 時）

```bash
curl -fsSL https://claude.ai/install.sh | bash

# 如果 claude --help 不能使用，就是PATH沒接上
# Native installation exists but ~/.local/bin is not in your PATH. Run:

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

claude 可以用了之後，進vscode安裝extention

- Claude
- Lynx
- Jupyter/Python（打開ipynb的程式，會自動安裝）

---

### 重點觀念（記這 一條關機 / 三條開機 就好）

- # `第0步` 每次關機必須

  # **scripts/backup_home.sh backup**

- **1～3 步**：每次開機**必做**（環境 + 對話 + 自動載入）。

- **4～5 步**：用到才做。它們裝在 `/usr` 和 `/root`（揮發層），所以每次開機都會不見，重跑即可。

- **VSCode/Cursor 的擴充套件**也在揮發層、會掉，**只能手動重裝**（沒有一鍵還原）—— 這是 RunPod 的限制，不是設定壞掉。

要驗證少了什麼，手冊 §D 有對照清單；最快一句：

```bash
command -v git-lfs ffmpeg tmux rsync claude
```

有印路徑就是在，沒印就是還沒裝。

---

---

# RunPod 開機 / 關機手順（手動版）

> **背景**：RunPod 是 network volume，**重開機後只有 `/workspace` 會保留**，
> `/root` 底下全部重建（含 Claude Code 的對話歷史 / memory / 設定 / 憑證、conda 是否在 PATH、prompt 是否載入）。
> 本手順「不依賴開機自動腳本」，全部手動貼指令、邊貼邊看，確定可控。
> 確認穩定後再談自動化。

> ⚠️ **2026-05-27 慘痛教訓**：舊版 `backup_home.sh` 用 `rsync`，但本機**沒裝 rsync**，
> 且備份會先 `rm -rf` 舊備份再 rsync → 結果舊備份被刪、新備份是空的，丟了一整天的 Claude 對話。
> 現在的 `backup_home.sh` 已改用 `tar`（系統一定有），**每次存成帶時間戳的快照、永不刪上一份好的**，
> 而且備份完會**驗證對話歷史(.jsonl)真的有打包進去**。詳見 §F。

---

## 0. 東西分別存在哪（搞清楚就不會慌）

| 東西                             | 實際位置                                                | 重開機後            | 怎麼救回                     |
| -------------------------------- | ------------------------------------------------------- | ------------------- | ---------------------------- |
| 短 prompt「設定本體」            | `/workspace/bootstrap/env.sh`                           | ✅ 在（/workspace） | 不用救，source 它即可        |
| 關掉 conda 長路徑                | `/workspace/miniconda3/.condarc`（`changeps1: false`）  | ✅ 在               | 不用救                       |
| conda 本體 / 環境                | `/workspace/miniconda3`、`/workspace/conda-envs/pt-exp` | ✅ 在               | 不用救，source env.sh 即可用 |
| prompt 在「目前終端」生效        | 記憶體（source 後才有）                                 | ❌ 要重新 source    | 手順 B-1                     |
| **對話歷史 / memory / 工具過程** | `/root/.claude/projects/-workspace/*.jsonl`             | ❌ **消失**         | 手順 B-2（restore）          |
| Claude 設定 / 憑證 / MCP         | `/root/.claude/*`、`/root/.claude.json`                 | ❌ 消失             | 手順 B-2（restore）          |
| 新終端自動載入                   | `/root/.bashrc` 末尾那行 source                         | ❌ 每次重建         | 手順 B-3（append）           |

**一句話：prompt/conda 的「設定」永遠在 /workspace 不會掉；會掉的只是「目前終端有沒有載入」和「/root 下的 Claude 資料」。後者一定要靠關機前備份。**

---

## A. 關機前手順（power off / Stop 之前一定要做）

```bash
# 1) 備份 /root 重要資料到 /workspace（含 Claude 對話歷史 / memory / 設定 / 憑證）
bash /workspace/scripts/backup_home.sh backup
```

看到綠字 **`✓ 已驗證：包含 N 個 Claude 對話歷史(.jsonl)`** 才算成功。
若看到紅字警告「找不到對話歷史」→ **先別關機**，回 §F 排查。

```bash
# 2) 再確認一次（看時間是現在、jsonl 數量合理）
bash /workspace/scripts/backup_home.sh snapshots
```

確認最新快照時間正確、`jsonl=` 數量合理後，再到 RunPod 介面關機 / Stop。

> ⚠️ 沒做第 1 步就關機 = 這次的對話歷史與設定永久消失。

---

## B. 開機後手順（重開機後，逐步貼，邊貼邊看）

開機後開一個 terminal，**依序**貼下面三步：

> ### Aaron 注意⚠️：
>
> - 到 VSCode 的 Extention 安裝 Claude和Lynx等主題。
> - 到 src 找一個 .ipynb python的程式，讓extention自動安裝
> - Claude 需要重新安裝？

```bash
# 1) 讓「這個終端」有 conda + 短 prompt（設定本體一直都在 env.sh）
source /workspace/bootstrap/env.sh
```

做完 prompt 應立刻變成 `(pt-exp): workspace# `（青色 env 名 + 綠色目錄）。
若沒變 → env.sh 有問題，記下來，回 §C / §E。

```bash
# 2) 把 /root 下的 Claude 資料（對話歷史 / 設定 / 憑證 / MCP）還原回去
bash /workspace/scripts/backup_home.sh restore
```

做完會印出「目前 projects 有 N 個 .jsonl」。**重啟 Claude Code** 即可看到先前對話歷史。
（想還原某一份較舊快照：先 `bash /workspace/scripts/backup_home.sh snapshots` 看清單，
　再 `bash /workspace/scripts/backup_home.sh restore 20260527-140051`。）

```bash
# 3) 讓「之後新開的終端」也自動有 prompt + conda（冪等，重複貼不會重複寫）
grep -q 'source /workspace/bootstrap/env.sh' /root/.bashrc 2>/dev/null \
  || echo 'source /workspace/bootstrap/env.sh' >> /root/.bashrc
```

做完第 3 步，**新開**的 terminal 會自動有 prompt + conda（不做也行，只是每個新終端要自己跑一次 B-1）。

> 💡 也可把 jupyter 寫入點導回 /workspace（某些工具會寫 `/root/.local/share/jupyter`）：
>
> ```bash
> mkdir -p /root/.local/share && rm -rf /root/.local/share/jupyter 2>/dev/null
> ln -s /workspace/.jupyter /root/.local/share/jupyter
> ```

---

## C. prompt 設定「在哪、怎麼重建」

短 prompt 分兩個檔（都在 /workspace，重開機不掉）：

1. **prompt 本體**：`/workspace/bootstrap/env.sh` 的 `# >>> rp short prompt >>>` 區塊
   （函式 `__rp_env` 取 conda 環境名最後一段；`PS1` 青色 env + 綠色 `\W`）
2. **關掉 conda 加長路徑**：`/workspace/miniconda3/.condarc` 的 `changeps1: false`

被改壞 / 消失時，一鍵重建（冪等）：

```bash
bash /workspace/scripts/setup_prompt.sh
```

跑完開新終端，或 `source /workspace/bootstrap/env.sh`。
改顏色：編 `env.sh` 的 `PS1`（`36`青 `32`綠 `31`紅 `33`黃 `34`藍 `35`紫；`\W`只末層、`\w`全路徑）。

---

## D. 重開機後「先觀察少什麼」對照清單

重開機後**先別跑 B**，先觀察、記下少了什麼，再跑 B 補回：

- [ ] 新終端 prompt 是短的 `(pt-exp): workspace#`？（是長的/預設 → env.sh 沒自動 source → B-1/B-3）
- [ ] `conda env list` 有自動在 pt-exp？（沒有 → env.sh 沒 source）
- [ ] Claude Code 打開後對話歷史在不在？（不在 → B-2 restore）
- [ ] `tail -3 /root/.bashrc` 有那行 source？（沒有 → 開機自動化這次沒跑）
- [ ] 其他發現少的 → 記下來，加進 `backup_home.sh` 的 `SRC` 陣列

---

## E. 相關檔案一覽

| 檔案                                   | 作用                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `/workspace/scripts/backup_home.sh`    | 備份/還原 /root 重要資料（`backup`/`restore`/`list`/`snapshots`/`verify`） |
| `/workspace/scripts/setup_prompt.sh`   | 一鍵重建短 prompt（冪等）                                                  |
| `/workspace/bootstrap/env.sh`          | 每個終端要 source：conda hook + 自動 activate pt-exp + 短 prompt           |
| `/workspace/bootstrap/00-bootstrap.sh` | 開機自動化腳本（實驗中，目前用手順 B 手動代替）                            |
| `/workspace/backups/root-home/`        | 快照存放處（帶時間戳，保留最近 10 份，`latest` 指向最新）                  |
| `/workspace/從零建置手順.md`           | 全新 network volume「從零建置」一條龍手順                                  |

---

## F. backup_home.sh 細節 / 排查

**指令一覽**

```bash
bash /workspace/scripts/backup_home.sh list        # 看會備份哪些（不動東西）
bash /workspace/scripts/backup_home.sh backup      # 建立新快照（關機前）
bash /workspace/scripts/backup_home.sh snapshots   # 列出所有快照（→ 標示最新）
bash /workspace/scripts/backup_home.sh verify      # 驗證最新快照（或 verify 快照名）
bash /workspace/scripts/backup_home.sh restore     # 還原最新快照（或 restore 快照名）
```

**設計重點（為什麼這版安全）**

- 用 `tar`，不用 `rsync`（本機無 rsync，且 rsync 在揮發層 /usr，重開機就沒）。
- 每次備份 = 一個新的時間戳資料夾 `backups/root-home/YYYYmmdd-HHMMSS/home.tgz`，
  **永不刪掉上一份好的**；只在超過 `KEEP=10` 份時刪「最舊的」。
- 備份完自動驗證 `.jsonl` 對話歷史有沒有進去，沒有就印**紅字警告**。
- 還原是「覆蓋疊上去」，不會刪 /root 其他檔；還原後自動修 `.credentials.json`/ssh 權限。

**想多備份其他 dotfile**：編 `backup_home.sh`，往 `SRC=( … )` 陣列加一行絕對路徑即可（不存在會自動略過）。

**排查：backup 說「找不到對話歷史」**

- 確認 Claude Code 這次真的有跑過（`ls /root/.claude/projects/-workspace/*.jsonl`）。
- 若 `.jsonl` 確實存在卻沒打包 → 檢查 `SRC` 是否含 `/root/.claude`。

**為什麼不備份 `/root/.bashrc`**：它每次開機由 RunPod 重建（含官方 banner），備份回去可能蓋成過時版本；
那行 `source env.sh` 改在手順 B-3 用 append 補回，較安全。
