# build_vectors.py (DROP-IN)
import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from .config import cfg
from .models import CrossAttention
from .utils_io import get_client_reps

# ---------- Hungarian helpers ----------
def _cosine_cost(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32); B = B.astype(np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    sim = A @ B.T
    return 1.0 - sim

def hungarian_match_centroids(img_cent: np.ndarray, txt_cent: np.ndarray, metric: str = "cosine"):
    if metric != "cosine":
        raise NotImplementedError("Only 'cosine' is supported.")
    Ka, Kb = img_cent.shape[0], txt_cent.shape[0]
    K = min(Ka, Kb)
    img_use = img_cent[:K]
    txt_use = txt_cent[:K]

    cost = _cosine_cost(img_use, txt_use)
    try:
        from scipy.optimize import linear_sum_assignment
        r_idx, c_idx = linear_sum_assignment(cost)
        # 보장된 정렬
        order = np.argsort(r_idx)
        r_idx = r_idx[order]; c_idx = c_idx[order]
    except Exception:
        # 폴백: 간단 그리디
        used = set()
        c_idx = [None] * K
        row_order = np.argsort(cost.min(axis=1))
        for r in row_order:
            for c in np.argsort(cost[r]):
                if c not in used:
                    used.add(c); c_idx[r] = c; break
        for r in range(K):
            if c_idx[r] is None:
                c_idx[r] = int(np.argmin(cost[r]))
        r_idx = np.arange(K, dtype=int)
        c_idx = np.array(c_idx, dtype=int)

    img_ordered = img_use[r_idx]
    txt_ordered = txt_use[c_idx]
    return img_ordered, txt_ordered

# ---------- grouping (기존 보존) ----------
def split_groups_by_quantile(metrics: dict) -> Tuple[list, list, list]:
    vals = np.array([v for k,v in metrics.items() if k in cfg.FUSION_CLIENTS and not np.isnan(v)])
    q1, q2 = np.quantile(vals, 1/3), np.quantile(vals, 2/3)
    low = [cid for cid,v in metrics.items() if cid in cfg.FUSION_CLIENTS and v <= q1]
    mid = [cid for cid,v in metrics.items() if cid in cfg.FUSION_CLIENTS and q1 < v <= q2]
    top = [cid for cid,v in metrics.items() if cid in cfg.FUSION_CLIENTS and v >  q2]
    return top, mid, low

def stack_reps(clients: List[int], split="train", max_samples=1024):
    imgs, txts = [], []
    for cid in clients:
        img, txt = get_client_reps(cid, split=split, max_samples=max_samples)
        if len(img): imgs.append(img)
        if len(txt): txts.append(txt)
    return np.vstack(imgs), np.vstack(txts)

def kmeans_centroids(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=cfg.KMEANS_N_INIT, random_state=cfg.SEED).fit(X)
    return km.cluster_centers_

# ---------- Cross-Attention with Hungarian matching ----------
def build_global_vectors(img_cent: np.ndarray, txt_cent: np.ndarray) -> Dict[str, torch.Tensor]:
    dev = torch.device("cpu")
    ca_it = CrossAttention(cfg.IMG_DIM, cfg.TXT_DIM, cfg.TXT_DIM, cfg.D_MODEL).to(dev)
    ca_ti = CrossAttention(cfg.TXT_DIM, cfg.IMG_DIM, cfg.IMG_DIM, cfg.D_MODEL).to(dev)

    # 1) 1:1 매칭
    img_matched, txt_matched = hungarian_match_centroids(img_cent, txt_cent, metric="cosine")

    # 2) 정렬된 입력 텐서
    Qimg = torch.tensor(img_matched, dtype=torch.float32, device=dev)  # Query = img
    Ktxt = torch.tensor(txt_matched, dtype=torch.float32, device=dev)  # Key/Value = text
    Vtxt = Ktxt.clone()

    Qtxt = torch.tensor(txt_matched, dtype=torch.float32, device=dev)  # 역방향: Query = text
    Kimg = torch.tensor(img_matched, dtype=torch.float32, device=dev)  # 역방향: Key/Value = img
    Vimg = Kimg.clone()

    # 3) Cross-Attention (양방향)
    Z_mm      = ca_it(Qimg, Ktxt, Vtxt).detach().cpu()
    Z_img2txt = Z_mm.clone()
    Z_txt2img = ca_ti(Qtxt, Kimg, Vimg).detach().cpu()
    return {"Z_mm": Z_mm, "Z_img2txt": Z_img2txt, "Z_txt2img": Z_txt2img}
