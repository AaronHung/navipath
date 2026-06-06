"""QPMIL-VL 接線共用工具（M1 runner 與 M2 adapter 共用）。

職責：
- 把 ./QPMIL-VL 加進 import path（QPMIL 內部用相對 import，需 cwd 在該目錄或加 path）。
- 載入並凍結 CONCH base model。
- 依「任務序」建立 cumulative class ensemble（支援 reverse；不依賴 QPMIL 內部
  依 JSON 字典序 break 的 get_current_ensemble_classes）。
- 提供建立 QPMIL 可訓練參數（key / prompt / tunable_v）的 helper。

刻意不複用 QPMIL 的 Manager（其綁死 wandb / torch.cuda.* / np.Inf，無法在 Mac CPU/MPS 跑）。
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn as nn

QPMIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "QPMIL-VL")


def add_qpmil_to_path() -> str:
    if QPMIL_DIR not in sys.path:
        sys.path.insert(0, QPMIL_DIR)
    return QPMIL_DIR


def load_base_model(cfg, device):
    """載入 CONCH 並凍結。回傳 (base_model, feature_dim, embedding_dim)。"""
    add_qpmil_to_path()
    from models.conch import create_model_from_pretrained
    from models import freeze_weight

    base_model, _ = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint_path=cfg["conch_ckpt_path"], device=device
    )
    base_model.eval()
    base_model.dtype = base_model.logit_scale.dtype
    freeze_weight(base_model, cfg["base_model_arch"])
    embedding_dim = base_model.text.ln_final.weight.shape[0]   # 768
    feature_dim = base_model.visual.proj_contrast.shape[1]     # 512
    return base_model, feature_dim, embedding_dim


def build_ensemble_classes(seen_order, class_ensemble_path):
    """依「任務序」(seen_order) 累積 class ensemble。

    回傳 {'ensemble_classes': [...], 'count': [...]}，row 順序即 label 順序，
    故 label_shift[i] = 2*i（每任務 2 類）。forward-order 時與 QPMIL 原結果一致。
    """
    with open(class_ensemble_path) as f:
        prompts = json.load(f)["0"]
    classnames = prompts["classnames"]
    templates = prompts["templates"]

    out = {"ensemble_classes": [], "count": []}
    for ds in seen_order:
        for _subtype, names in classnames[ds].items():
            out["count"].append(len(names) * len(templates))
            for name in names:
                out["ensemble_classes"].extend(
                    [t.replace("CLASSNAME", name) for t in templates]
                )
    return out


def make_trainable_params(cfg, feature_dim, embedding_dim, num_tasks, device, dtype):
    """建立 QPMIL 的可訓練參數：Prototype Pool (key, prompt) 與 Tunable Vector。"""
    key = nn.ParameterList([
        nn.Parameter(0.02 * torch.randn(1, feature_dim, dtype=dtype, device=device))
        for _ in range(cfg["pool_size"])
    ])
    prompt = nn.ParameterList([
        nn.Parameter(0.02 * torch.randn(1, cfg["prompt_length"], embedding_dim, dtype=dtype, device=device))
        for _ in range(cfg["pool_size"])
    ])
    tunable_v = nn.ParameterList([
        nn.Parameter(torch.zeros(cfg["dataset_subtype_num"][0], feature_dim, dtype=dtype, device=device))
        for _ in range(num_tasks)
    ])
    return key, prompt, tunable_v
