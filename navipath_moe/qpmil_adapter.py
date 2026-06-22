"""M2 — QPMIL-VL backbone adapter。

把 QPMIL_VL 包成 NaviPath-MoE 需要的 4-hook 介面（見 model.py），並提供
forward_internals() 取出所有創新會用到的中間張量（Z / F_p / W / f_txt / bag）。

設計原則：**不改 QPMIL model 邏輯**，只在 adapter 端呼叫其既有 method
（_query_prototype_pool / _get_bag_feature / text_encoder / prompt_learner），
W（patch→prototype 指派矩陣）在 adapter 端用相同公式重算（QPMIL 內部沒回傳）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qpmil_bootstrap import (
    add_qpmil_to_path, load_base_model, build_ensemble_classes, make_trainable_params,
)


def _build_qpmil(cfg, base_model, device, key, prompt, tunable_v, seen_order, train_key_frequency=None):
    add_qpmil_to_path()
    from models import QPMIL_VL
    cfg = dict(cfg)
    cfg["task_num"] = len(seen_order)
    ensemble = build_ensemble_classes(seen_order, cfg["class_ensemble_path"])
    if not train_key_frequency:
        # penalty_table（task_num>1 時用）需 key frequency；推論用 eval=True 不影響排序，
        # 故補 uniform 頻率讓 penalty_table 良好定義（均勻 = 不偏好任何 prototype）。
        train_key_frequency = {
            ds: [torch.ones(cfg["pool_size"], dtype=torch.int)] for ds in seen_order
        }
    model = QPMIL_VL(cfg, base_model, device, key, prompt, tunable_v, ensemble,
                     train_key_frequency)
    return model


class QPMILBackbone(nn.Module):
    """單一 QPMIL_VL 實例的 4-hook 介面 + internals。對應 model.py 的 backbone 約定。"""

    def __init__(self, model, cfg, device):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.device = device
        self._f_txt = None
        self._F_p = None

    # ---- 4 hooks ----------------------------------------------------------
    def encode_patches(self, slide):
        """slide: [1,n,512] 或 [n,512] -> Z [n,512] (float, on device)。"""
        Z = slide
        if Z.dim() == 3:
            Z = Z.squeeze(0)
        return Z.to(self.device).float()

    def class_text_features(self):
        """f_txt [C,512]：CFE 增強後的類別文字特徵（normalize）。訓練後 static，快取。"""
        if self._f_txt is None:
            enh = self.model.tunable_v_learner(self.model.prompt_learner.class_ensemble_feature)
            self._f_txt = (enh / enh.norm(dim=-1, keepdim=True)).detach()
        return self._f_txt

    def prototype_features(self):
        """F_p [M,512]：整個 prototype pool 的 prompt 文字特徵（normalize）。static，快取。

        重用 QPMIL 的 prompt_learner.embedding_prefix/suffix 與 text_encoder，
        一次編碼全部 pool_size 個 prototype（給 patch-budget 的 prototype 選法）。
        """
        if self._F_p is None:
            pl = self.model.prompt_learner
            M = self.cfg["pool_size"]
            # merge_parameter 依賴 cfg['opt_name']；直接 cat 更通用
            merged = torch.cat([pl.prompt[i] for i in range(M)], dim=0)  # [M, L, emb]
            prefix = pl.embedding_prefix.repeat(M, 1, 1)                  # [M, X1, emb]
            suffix = pl.embedding_suffix.repeat(M, 1, 1)                  # [M, X2, emb]
            emb = torch.cat([prefix, merged, suffix], dim=1)             # [M, X3, emb]
            tok = pl.tokenized_prompts.repeat(M, 1)                       # [M, X3]
            with torch.no_grad():
                F_p = self.model.text_encoder(emb, tok)                   # [M, 512]
            self._F_p = (F_p / F_p.norm(dim=-1, keepdim=True)).detach()
        return self._F_p

    def aggregate_and_predict(self, Z_sub, f_txt=None, no_grad=True):
        """對 patch 子集 Z_sub [k,512] 跑真實 QPMIL 推論 -> (logits [1,C], loss_dict)。

        no_grad=True（預設）：eval / budget 評估用，節省記憶體。
        no_grad=False：router 訓練時用，logits 保留 grad_fn（grad 流回 Z_sub 的選取）。
        """
        x_list = [Z_sub.to(self.device).float().unsqueeze(0)]
        if no_grad:
            with torch.no_grad():
                logits, _indices = self.model(x_list, eval=True)
        else:
            logits, _indices = self.model(x_list, eval=True)
        return logits, {}

    # ---- internals (M2 地基) ---------------------------------------------
    @torch.no_grad()
    def forward_internals(self, slide):
        Z = self.encode_patches(slide)                                   # [n,512]
        x_list = [Z.unsqueeze(0)]
        # query vector（同 _query_prototype_pool 的 pooling）
        if self.cfg["pooling"] == "max":
            q_vec = Z.amax(0)
        else:
            q_vec = Z.mean(0)
        q_vec = q_vec / q_vec.norm()
        # prototype 選取 + matching loss
        indices, matching_loss = self.model._query_prototype_pool(x_list, 1, eval=False)
        embedding, tok = self.model.prompt_learner(indices, 1)
        F_p_sel = self.model.text_encoder(embedding, tok)
        F_p_sel = F_p_sel / F_p_sel.norm(dim=-1, keepdim=True)
        F_p_sel = F_p_sel.view(self.cfg["match_size"], -1)               # [N,512]
        z_norm = Z / Z.norm(dim=-1, keepdim=True)
        S = self.cfg["csm_logit_scale"] * z_norm @ F_p_sel.t()          # [n,N]
        W = torch.softmax(S, dim=0)                                      # [n,N]
        # bag feature + logits + class text
        bag = self.model._get_bag_feature(x_list, indices, 1).squeeze()  # [512]
        f_txt = self.class_text_features()                              # [C,512]
        logits, _ = self.model(x_list, eval=True)
        enh = self.model.tunable_v_learner(self.model.prompt_learner.class_ensemble_feature)
        enh = (enh / enh.norm(dim=-1, keepdim=True)).unsqueeze(0)
        class_sim_loss = self.model._compute_class_sim_loss(enh)
        return {
            "logits": logits,                                # [1,C]
            "bag_feature": bag,                              # [512]
            "patch_features_Z": Z,                           # [n,512]
            "query_vector": q_vec,                           # [512]
            "selected_prototype_indices": indices.squeeze(0),# [N]
            "selected_prototype_features_Fp": F_p_sel,       # [N,512]
            "assignment_matrix_W": W,                        # [n,N]
            "class_text_features_ftxt": f_txt,               # [C,512]
            "prototype_pool_features_Fp_all": self.prototype_features(),  # [M,512]
            "loss_L_M_matching": matching_loss,              # scalar
            "loss_L_S_class_sim": class_sim_loss,            # scalar
        }


def build_backbone_fresh(qpmil_cfg, order, device, seed=0):
    """用隨機初始化的可訓練參數建 backbone（給 shape/internals 驗證，不需訓練）。"""
    torch.manual_seed(seed)
    base_model, feat_dim, emb_dim = load_base_model(qpmil_cfg, device)
    key, prompt, tunable_v = make_trainable_params(
        qpmil_cfg, feat_dim, emb_dim, len(order), device, base_model.dtype)
    model = _build_qpmil(qpmil_cfg, base_model, device, key, prompt, tunable_v, order)
    return QPMILBackbone(model, qpmil_cfg, device)


def build_backbone_from_ckpt(ckpt_path, device, conch_ckpt_path=None,
                             path_remap=None):
    """從 M1 runner 存的 checkpoint 載入訓練後的 backbone（全任務 seen）。

    conch_ckpt_path: 可覆蓋 ckpt 內嵌的 CONCH 權重路徑。
    path_remap: (old_prefix, new_prefix) 跨機器移植用——把 qcfg 內所有以
        old_prefix 開頭的絕對路徑（如 RunPod 的 /workspace/src/navipath）
        改寫成本機 new_prefix。conch_ckpt_path 若提供則優先。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    qcfg = ckpt["qpmil_cfg"]
    # 舊版 ckpt 可能缺 opt_name，補上 default
    qcfg.setdefault("opt_name", "adam")
    if path_remap:
        old_p, new_p = path_remap
        for k, v in list(qcfg.items()):
            if isinstance(v, str) and v.startswith(old_p):
                qcfg[k] = new_p + v[len(old_p):]
    if conch_ckpt_path:
        qcfg["conch_ckpt_path"] = conch_ckpt_path
    order = ckpt["tasks"]
    base_model, _, _ = load_base_model(qcfg, device)
    dtype = base_model.dtype
    key = nn.ParameterList([nn.Parameter(p.to(device=device, dtype=dtype)) for p in ckpt["key"]])
    prompt = nn.ParameterList([nn.Parameter(p.to(device=device, dtype=dtype)) for p in ckpt["prompt"]])
    tunable_v = nn.ParameterList([nn.Parameter(p.to(device=device, dtype=dtype)) for p in ckpt["tunable_v"]])
    model = _build_qpmil(qcfg, base_model, device, key, prompt, tunable_v, order)
    return QPMILBackbone(model, qcfg, device)
