"""M2 debug — 驗證 QPMILBackbone.forward_internals 的回傳 keys 與 shape。

用一張真實切片跑一次，印出每個中間張量的 shape。確認拿得到 Z / F_p / f_txt / W。
（用隨機初始化參數即可驗 shape；不需訓練。）

用法：
  python debug_internals.py
  python debug_internals.py --slide-dir data/tcga_esca/feats-l1-s256_CONCH/pt_files
"""
from __future__ import annotations

import argparse
import os

import torch

from navipath_moe import get_device, setup_mps
from navipath_moe.qpmil_adapter import build_backbone_fresh, build_backbone_from_ckpt
from train_qpmil_runner import load_qpmil_cfg, TASK_ORDERS


def _first_slide(slide_dir):
    files = sorted(f for f in os.listdir(slide_dir) if f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"no .pt under {slide_dir}")
    x = torch.load(os.path.join(slide_dir, files[0]), map_location="cpu").float()
    return x, files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--slide-dir", default="data/tcga_lung/feats-l1-s256_CONCH/pt_files")
    ap.add_argument("--ckpt", default="", help="若給，載入訓練後 backbone；否則隨機初始化")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    cfg = load_qpmil_cfg(args.qpmil_config)
    order = TASK_ORDERS[args.order]

    if args.ckpt:
        print(f"[debug] backbone from ckpt: {args.ckpt}")
        backbone = build_backbone_from_ckpt(args.ckpt, device)
    else:
        print(f"[debug] backbone fresh (random init), order={args.order}, C={2 * len(order)}")
        backbone = build_backbone_fresh(cfg, order, device)

    slide, name = _first_slide(args.slide_dir)
    print(f"[debug] slide={name}  Z_in shape={tuple(slide.shape)}")

    internals = backbone.forward_internals(slide)
    print("\n[debug] forward_internals returned keys / shapes:")
    for k, v in internals.items():
        if torch.is_tensor(v):
            shp = tuple(v.shape) if v.dim() > 0 else "scalar"
            print(f"  {k:38s} {str(shp):16s} dtype={v.dtype}")
        else:
            print(f"  {k:38s} {v}")

    # 關鍵 4 件：Z / F_p / f_txt / W
    Z = internals["patch_features_Z"]
    Fp = internals["selected_prototype_features_Fp"]
    ftxt = internals["class_text_features_ftxt"]
    W = internals["assignment_matrix_W"]
    n = Z.shape[0]
    assert Z.shape[1] == 512
    assert Fp.shape == (cfg["match_size"], 512)
    assert ftxt.shape == (2 * len(order), 512)
    assert W.shape == (n, cfg["match_size"])
    print("\n[debug] OK — Z[n,512], F_p[N,512], f_txt[C,512], W[n,N] all present and correctly shaped.")


if __name__ == "__main__":
    main()
