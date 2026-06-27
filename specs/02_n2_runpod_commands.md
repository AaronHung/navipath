# N2 · RunPod 執行指令（step-by-step）

> 目標：在 frozen QPMIL backbone + 真資料上跑出 N2 三組數字。
> 驅動腳本：`eval_sequential_observation.py`（已 push）。操作環境見 `RUNPOD_RUNBOOK.md`。
> 同步契約：RunPod `git add outputs/ && commit && push` → Aaron 通知 → Mac `git pull` 做 N3 check。

```bash
REPO=/workspace/src/navipath        # 依你的 pod 調整
```

---

## Step 0 — 進 pod、setup、同步
```bash
cd $REPO
# 環境若重置過，先補套件（看到 cuda available: True 才繼續）
pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 timm einops h5py \
  openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && \
python -c "import torch; print('cuda', torch.cuda.is_available())"
git pull --ff-only
```
> 第一次 pull 若撞 `untracked QPMIL-VL` → 照 `RUNPOD_RUNBOOK.md §6` 的安全步驟處理。

## Step 1 — 體檢（確認檔案/腳本/GPU 就緒）
```bash
ls -1 outputs/qpmil_reverse_fold{1,2,3}.pt          # 6 個 backbone ckpt 中的 reverse 3 個
python eval_sequential_observation.py --help | head -3   # 確認腳本存在
nvidia-smi | head -12
```

## Step 2 — GPU 快速 smoke（先確認端到端會動，再跑真的）
```bash
python eval_sequential_observation.py --backbone-ckpt outputs/qpmil_reverse_fold1.pt \
  --order reverse --fold 1 --eval-task 0 \
  --epochs 1 --max-train 8 --max-eval 4 --budgets 0,16,8 --step-size 4 \
  --out outputs/_smoke
# 應印出 4 個 mode 的 {budget: acc}；確認有 outputs/_smoke/seqobs_reverse_f1_task0.json
rm -rf outputs/_smoke
```
任一報錯就停、把 error 貼給我（不要硬跑）。

## Step 3 — 正式跑（reverse 3 折，eval 最舊任務 esca=task0）— tmux
```bash
command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)
tmux new -s n2
```
在 tmux 裡貼這整段（跑完自己 push）：
```bash
cd $REPO && git pull --ff-only && set -o pipefail && \
for F in 1 2 3; do \
  python eval_sequential_observation.py \
    --backbone-ckpt outputs/qpmil_reverse_fold${F}.pt \
    --order reverse --fold ${F} --eval-task 0 \
    --epochs 5 --budgets 0,128,64,32,16 --step-size 16 --redundancy 0.5 \
    --skill-bank-out outputs/skill_bank_reverse_f${F}.pt \
    2>&1 | tee outputs/seqobs_reverse_f${F}_task0.log || break; \
done && \
echo "=== N2 RUNS DONE, pushing ===" && \
git add outputs/ && \
git commit -m "results(N2): seqobs reverse f1-3 task0 (seq/oneshot x nsm/nonsm) $(date +%F)" && \
git pull --rebase && git push && \
echo "=== N2 PUSHED ==="
```
然後 `Ctrl+b` 放開、按 `d` detach（可關電腦）。

## Step 4 — 回來確認 + 通知
```bash
tmux attach -t n2          # 看到 "=== N2 PUSHED ===" 即完成；Ctrl+b d 離開
```
跟我說「N2 跑完了」→ 我在 Mac `git pull` 做 N3（彙整 + 出圖 + check）。

---

## 防呆 / 規則
- **不覆蓋**：`seqobs_*` canonical 已存在就先改名或加 `__rerun_YYYYMMDD_HHMM`。
- 失敗即止（`|| break`），不硬跑、不捏造。
- 想省時間：先只跑 `F=1`（拿掉 for 迴圈），確認數字合理我再喊跑 2、3。

## 怎麼看數字算成功（N3 會正式整理）
每個 json 的 `results` 有 4 個 mode：`nsm_seq / nsm_oneshot / nonsm_seq / nonsm_oneshot`，各含 `{All,128,64,32,16}` 的 acc。
- **CL 保留（核心）**：old task 上 `nsm_*` 應明顯 > `nonsm_*`（有技能記憶能恢復「該看哪」）。
- **Budget 效率**：`acc@K`（K=64/32）接近 `acc@All`，代表少看也夠準。
- **Agentic**：`*_seq` vs `*_oneshot` 同 budget 的差異（序列觀察是否更好/相當）。

## 預估時間
- 每折：4 task router 訓練（5 epochs）+ 4-mode×5-budget eval。約 15–30 分/折（視 GPU、test 集大小）。
- 3 折 ≈ 1–1.5 小時。想更快：`--epochs 3` 或 `--max-eval 60`。

## （可選 stretch）paper order
```bash
# 把上面 for 段的 --order reverse 改成 --order paper，backbone-ckpt 換 qpmil_paper_fold${F}.pt
# eval-task 0 在 paper order = lung（最舊）。
```
