# RunPod 重開後 setup（一行）

RunPod 重置環境後，跑任何實驗前先貼這行（裝缺的套件 + 確認 GPU）：

```bash
cd /workspace/src/navipath && pip install -q "transformers>=4.40,<5" huggingface-hub==0.36.2 timm einops h5py openpyxl wandb scikit-learn tqdm seaborn pandas pyyaml && python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

- 看到 `cuda available: True` 就可以開跑。
- 程式與結果都在 git，重裝只補這幾個套件（timm/transformers/pandas/pyyaml）。
- 之後接 `git pull` 取最新 code/結果，再跑 `train_router_v0.py` 等指令。
