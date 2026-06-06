"""M1 — QPMIL-VL thin continual runner（不依賴 wandb / 不碰 torch.cuda.* / device-agnostic）。

跑通 QPMIL backbone（forward 與 reverse 任務序），輸出 per-task accuracy matrix R[t,i]
與 ACC/Forgetting/BWT，對齊論文趨勢（paper ~0.890 / reverse ~0.859）。

用法：
  python train_qpmil_runner.py --order paper   --fold 1 --seed 1
  python train_qpmil_runner.py --order reverse --fold 1 --seed 1
快速 smoke（縮 epoch / 限切片數，僅驗證 pipeline）：
  python train_qpmil_runner.py --order paper --fold 1 --epochs 1 --max-train 40 --max-eval 30

不更動 QPMIL model 邏輯，只新增 runner + data adapter（重用 QPMIL_VL / WSIClf / CONCH）。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from navipath_moe import get_device, setup_mps
from navipath_moe.qpmil_bootstrap import (
    add_qpmil_to_path, load_base_model, build_ensemble_classes, make_trainable_params,
)
from eval.metrics import summarize

TASK_ORDERS = {
    "paper":   ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
}


def save_checkpoint(args, cfg, order, key, prompt, tunable_v):
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"qpmil_{args.order}_fold{args.fold}.pt")
    torch.save({
        "order": args.order, "fold": args.fold, "tasks": order,
        "qpmil_cfg": {k: cfg[k] for k in (
            "pool_size", "prompt_length", "match_size", "csm_logit_scale",
            "alpha", "pooling", "base_model_arch", "dataset_subtype_num", "conch_ckpt_path",
            "class_ensemble_path", "opt_name")},
        "key": [p.detach().cpu() for p in key],
        "prompt": [p.detach().cpu() for p in prompt],
        "tunable_v": [p.detach().cpu() for p in tunable_v],
    }, ckpt_path)
    print("[result] saved checkpoint:", ckpt_path, flush=True)


def load_qpmil_cfg(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # 還原 QPMIL set_config 的路徑拼接（dataset_root_dir + 相對樣板）。
    cfg["path_split"] = cfg["dataset_root_dir"] + cfg["path_split"]
    cfg["path_feat"] = cfg["dataset_root_dir"] + cfg["path_feat"]
    cfg["path_table"] = cfg["dataset_root_dir"] + cfg["path_table"]
    return cfg


def build_loaders(cfg, order, fold, num_workers=0):
    add_qpmil_to_path()
    from dataset import get_data_loaders
    sub_cfg = dict(cfg)
    sub_cfg["dataset_names"] = order
    sub_cfg["data_split_seed"] = fold
    sub_cfg["num_workers"] = num_workers
    return get_data_loaders(sub_cfg)


def iter_slides(loader, limit=0):
    n = 0
    for idx, feats, label in loader:
        yield feats, int(label.view(-1)[0])
        n += 1
        if limit and n >= limit:
            break


def train_one_task(model, optimizer, loader, cfg, device, task_pos, epochs, max_train):
    shift = 2 * task_pos
    bp = cfg["bp_every_batch"]
    pool_size = cfg["pool_size"]
    last_freq = torch.zeros(pool_size, dtype=torch.int)
    model.eval()  # 與 QPMIL 一致：base 凍結，learnable 為純 Parameter，無 dropout/BN
    for epoch in range(epochs):
        xs, ys = [], []
        epoch_freq = torch.zeros(pool_size, dtype=torch.int)
        seen = 0
        for feats, label in iter_slides(loader, max_train):
            xs.append(feats.to(device))
            ys.append(label)
            seen += 1
            if len(xs) == bp:
                epoch_freq += _train_step(model, optimizer, xs, ys, shift, cfg, device)
                xs, ys = [], []
        if xs:
            epoch_freq += _train_step(model, optimizer, xs, ys, shift, cfg, device)
        last_freq = epoch_freq
        print(f"    [task_pos {task_pos}] epoch {epoch + 1}/{epochs} done ({seen} slides)")
    return last_freq


def _train_step(model, optimizer, xs, ys, shift, cfg, device):
    logits, loss_dict, indices, _ = model(xs)
    labels = torch.tensor([y + shift for y in ys], dtype=torch.long, device=logits.device)
    clf = torch.nn.functional.cross_entropy(logits, labels)
    loss = clf + cfg["lambda"] * loss_dict["matching_loss"] + cfg["beta"] * loss_dict["class_sim_loss"]
    optimizer.zero_grad()
    loss.backward()
    if cfg["max_norm"] != "None":
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_norm"])
    for p in model.parameters():           # 只更新被匹配到的 key/prompt（grad 全 0 者跳過）
        if p.grad is not None and torch.all(p.grad == 0):
            p.grad = None
    optimizer.step()
    freq = torch.zeros(cfg["pool_size"], dtype=torch.int)
    for i in indices.detach().cpu().flatten():
        freq[i] += 1
    return freq


@torch.no_grad()
def eval_task(model, loader, task_pos, device, bp, max_eval):
    shift = 2 * task_pos
    correct = total = 0
    xs, ys = [], []

    def flush():
        nonlocal correct, total, xs, ys
        if not xs:
            return
        logits, _ = model(xs, eval=True)
        pred = logits.argmax(-1).cpu()
        gt = torch.tensor([y + shift for y in ys])
        correct += int((pred == gt).sum())
        total += len(ys)
        xs, ys = [], []

    for feats, label in iter_slides(loader, max_eval):
        xs.append(feats.to(device))
        ys.append(label)
        if len(xs) == bp:
            flush()
    flush()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpmil-config", default="QPMIL-VL/configs/main.yaml")
    ap.add_argument("--order", choices=list(TASK_ORDERS), default="paper")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=0, help="0=用 config 的 epochs；>0 覆寫")
    ap.add_argument("--tasks", type=int, default=0, help="0=全部任務；>0 只跑前 N 個")
    ap.add_argument("--max-train", type=int, default=0, help="每任務最多訓練切片數（0=全部）")
    ap.add_argument("--max-eval", type=int, default=0, help="每任務最多評估切片數（0=全部）")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save-ckpt", action="store_true", help="存訓練後的 key/prompt/tunable_v 供 M2/M3 載入")
    args = ap.parse_args()

    setup_mps()
    device = get_device("auto")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[runner] device={device} order={args.order} fold={args.fold} seed={args.seed}")

    cfg = load_qpmil_cfg(args.qpmil_config)
    cfg["task_num"] = 0
    order = TASK_ORDERS[args.order]
    if args.tasks:
        order = order[:args.tasks]
    T = len(order)

    add_qpmil_to_path()
    from models import QPMIL_VL, set_tunable_v

    base_model, feat_dim, emb_dim = load_base_model(cfg, device)
    dtype = base_model.dtype
    key, prompt, tunable_v = make_trainable_params(cfg, feat_dim, emb_dim, T, device, dtype)

    loaders = build_loaders(cfg, order, args.fold)
    train_key_frequency = {name: [] for name in order}

    R = np.zeros((T, T))
    for t, ds in enumerate(order):
        cfg["task_num"] = t + 1
        seen = order[: t + 1]
        ensemble = build_ensemble_classes(seen, cfg["class_ensemble_path"])
        set_tunable_v(tunable_v, cfg["task_num"])
        model = QPMIL_VL(cfg, base_model, device, key, prompt, tunable_v, ensemble, train_key_frequency)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"],
                                     weight_decay=cfg["adam_weight_decay"], eps=cfg["adam_eps"])
        epochs = args.epochs or cfg["epochs"][t]
        print(f"[task {t + 1}/{T}] {ds}  (num_cls={2 * (t + 1)}, epochs={epochs})", flush=True)

        freq = train_one_task(model, optimizer, loaders[ds]["train"], cfg, device, t, epochs, args.max_train)
        train_key_frequency[ds].append(freq)

        for i in range(t + 1):
            R[t, i] = eval_task(model, loaders[order[i]]["test"], i, device,
                                cfg["bp_every_batch"], args.max_eval)
        print(f"[task {t + 1}] R[{t}] = {np.round(R[t, : t + 1], 4).tolist()}", flush=True)
        if args.save_ckpt:                       # 每任務後存檔，背景被切斷也保有可用 backbone
            save_checkpoint(args, cfg, order, key, prompt, tunable_v)

    summary = summarize(R, joint_train_acc=0.908 if args.order == "paper" else None)
    print("\n[result] Accuracy matrix R (R[t,i] = acc on task i after training task t):")
    print(np.round(R, 4))
    print("[result] CL summary:", summary)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"qpmil_{args.order}_fold{args.fold}.json")
    with open(out_path, "w") as f:
        json.dump({
            "order": args.order, "fold": args.fold, "seed": args.seed,
            "tasks": order, "R": R.tolist(), "summary": summary,
            "epochs_override": args.epochs, "max_train": args.max_train, "max_eval": args.max_eval,
        }, f, indent=2)
    print("[result] saved:", out_path)


if __name__ == "__main__":
    main()
