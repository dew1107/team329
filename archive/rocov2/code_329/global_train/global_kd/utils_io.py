import os, json, random
import numpy as np
import torch
from typing import Optional, Tuple
from .config import cfg

# ----- basic utils -----
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def client_dir(cid: int) -> str: return os.path.join(cfg.BASE_DIR, f"client_{cid}")
def ckpt_path(cid: int) -> str:  return os.path.join(client_dir(cid), cfg.CKPT_NAME.format(cid=cid))

# ----- metrics -----
def load_client_metric(cid: int) -> Optional[float]:
    path = os.path.join(client_dir(cid), f"client_{cid}_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d.get(cfg.METRIC_NAME, np.nan))
    return np.nan

# ----- reps loader (★ 네 환경에 맞게 구현) -----
def get_client_reps(cid: int, split: str = "train", max_samples: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    반환: (img_reps[N_i, IMG_DIM], txt_reps[N_t, TXT_DIM])
    - 미리 저장한 *.npy가 있으면 읽고, 없으면 NotImplementedError를 던져서
      네 인코더 경로를 연결하도록 함.
    """
    base = client_dir(cid)
    # 1) KD-후 벡터가 있으면 우선 사용
    kd_img = os.path.join(base, f"repr_img_kd.npy")
    kd_txt = os.path.join(base, f"repr_txt_kd.npy")
    img_path = kd_img if os.path.exists(kd_img) else os.path.join(base, f"{split}_img_reps.npy")
    txt_path = kd_txt if os.path.exists(kd_txt) else os.path.join(base, f"{split}_txt_reps.npy")
    if not (os.path.exists(img_path) or os.path.exists(txt_path)):
        raise NotImplementedError("get_client_reps: ...")

    img_path = os.path.join(client_dir(cid), f"{split}_img_reps.npy")
    txt_path = os.path.join(client_dir(cid), f"{split}_txt_reps.npy")
    if not (os.path.exists(img_path) and os.path.exists(txt_path)):
        raise NotImplementedError("get_client_reps: 네 인코딩 파이프라인과 연결해 구현하세요.")
    img = np.load(img_path); txt = np.load(txt_path)

    def sample_rows(x, k):
        if len(x) <= k: return x
        idx = np.random.choice(len(x), size=k, replace=False)
        return x[idx]
    return sample_rows(img, max_samples), sample_rows(txt, max_samples)

# ----- persistence -----
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def save_payload_for_client(cid: int, payload: dict):
    out_dir = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid}")
    ensure_dir(out_dir)
    torch.save(payload, os.path.join(out_dir, "global_payload.pt"))
