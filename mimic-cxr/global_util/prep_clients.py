# -*- coding: utf-8 -*-
"""
prep_clients.py
- (1) 검증셋으로 f1_micro/f1_macro/auc_macro 계산 → outputs/client_{cid}/client_{cid}_metrics.json
- (2) train 90% split에서 이미지/텍스트 임베딩 추출(L2 정규화) → outputs/client_{cid}/repr_img.npy / repr_txt.npy + index.csv

train_local.py의 Dataset/모델/라벨 설정을 그대로 재사용합니다.
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

# ---- local/train_local.py 에서 필요한 구성요소 import ----
from local_util.train_local import (
    CLIENT_CSV_DIR,          # r".\client_splits"
    ClientDataset,
    LABEL_COLUMNS,
    load_label_table,
    build_image_picker_from_metadata,
    LABEL_CSV,
    METADATA_CSV,
    IMG_ROOT,
    ImageHead,
    TextHead,
    MultiModalLateFusion,
)

SEED = 42
BATCH_DEFAULT = 64  # 임베딩 추출 배치 (메모리 상황에 따라 조정)

# =========================
# 유틸
# =========================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def client_dir(cid: int) -> str:
    d = Path(f"./outputs/client_{cid:02d}")
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

def decide_mode_for_cid(cid: int) -> str:
    if 1 <= cid <= 16: return "multimodal"
    if cid in [17, 18]: return "image_only"
    if cid in [19, 20]: return "text_only"
    raise ValueError(f"invalid client id: {cid}")

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)

# =========================
# 체크포인트 로드
# =========================
def build_model_from_ckpt(cid: int, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[client_{cid:02d}] checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = ckpt.get("mode", "multimodal")
    n_out = len(LABEL_COLUMNS)

    if mode == "multimodal":
        model = MultiModalLateFusion(n_out)
    elif mode == "image_only":
        model = ImageHead(n_out)
    else:
        model = TextHead(n_out)

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, mode

# =========================
# DataLoader 구성 (client CSV → Dataset → train/val 90/10)
# =========================
def build_dataloaders(cid: int, batch_size: int):
    csv_path = Path(CLIENT_CSV_DIR) / f"client_{cid:02d}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[client_{cid:02d}] CSV not found: {csv_path}")

    mode = decide_mode_for_cid(cid)
    label_table = load_label_table(LABEL_CSV)
    meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)

    ds_full = ClientDataset(csv_path, label_table, mode=mode, meta_picker=meta_picker)

    n = len(ds_full)
    n_tr = int(n * 0.9)
    n_va = n - n_tr
    tr_set, va_set = random_split(
        ds_full, [n_tr, n_va],
        generator=torch.Generator().manual_seed(SEED)
    )

    # 재현성 & 임베딩-인덱스 매칭을 위해 둘 다 shuffle=False
    pin = torch.cuda.is_available()
    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(va_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader, mode

# =========================
# 멀티라벨 메트릭
# =========================
def safe_multilabel_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(all_logits).cpu().numpy()
    y_true = all_labels.cpu().numpy().astype(np.int32)

    # F1
    y_pred = (probs >= 0.5).astype(np.int32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUROC (정의 불가 클래스 제외)
    auc_list = []
    C = y_true.shape[1]
    for c in range(C):
        col = y_true[:, c]
        if len(np.unique(col)) < 2:
            continue
        try:
            auc_c = roc_auc_score(col, probs[:, c])
            auc_list.append(auc_c)
        except Exception:
            pass
    auc_macro = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")
    return {"f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

# =========================
# 로짓(검증) & 임베딩(훈련) 추출
# =========================
def forward_logits(model, batch, mode: str, device: torch.device):
    y = batch["labels"].to(device)
    if mode == "multimodal":
        img = batch["image"].to(device)
        txt = {k: v.to(device) for k, v in batch["text"].items()}
        logits = model(image=img, text=txt)
    elif mode == "image_only":
        logits = model(batch["image"].to(device))
    else:  # text_only
        txt = {k: v.to(device) for k, v in batch["text"].items()}
        logits = model(**txt)
    return logits, y

@torch.no_grad()
def evaluate_on_loader(model, val_loader, mode: str, device: torch.device) -> Dict[str, float]:
    logits_all, labels_all = [], []
    for batch in val_loader:
        logits, y = forward_logits(model, batch, mode, device)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    if len(logits_all) == 0:
        return {"f1_micro": 0.0, "f1_macro": 0.0, "auc_macro": float("nan")}

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return safe_multilabel_metrics(logits_all, labels_all)

@torch.no_grad()
def extract_reps_from_batch(model, batch, mode: str, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    train_local.py 구조에 맞춰 인코더 출력(분류기 이전)을 임베딩으로 사용.
      - 이미지: (멀티모달) model.img.backbone(x)  / (이미지전용) model.backbone(x)
      - 텍스트: (멀티모달) model.txt.enc(...).last_hidden_state[:,0] / (텍스트전용) model.enc(...).last_hidden_state[:,0]
    """
    img_rep, txt_rep = None, None

    if mode in ["multimodal", "image_only"] and "image" in batch:
        x = batch["image"].to(device)
        if mode == "multimodal" and hasattr(model, "img") and hasattr(model.img, "backbone"):
            img_rep = model.img.backbone(x)          # [B, D_img]
        elif mode == "image_only" and hasattr(model, "backbone"):
            img_rep = model.backbone(x)              # [B, D_img]

    if mode in ["multimodal", "text_only"] and "text" in batch:
        t = {k: v.to(device) for k, v in batch["text"].items()}
        if mode == "multimodal" and hasattr(model, "txt") and hasattr(model.txt, "enc"):
            out = model.txt.enc(**t)
            txt_rep = out.last_hidden_state[:, 0]    # [B, D_txt]
        elif mode == "text_only" and hasattr(model, "enc"):
            out = model.enc(**t)
            txt_rep = out.last_hidden_state[:, 0]    # [B, D_txt]

    return img_rep, txt_rep

def rows_for_index(batch, count: int, has_img: bool, has_txt: bool) -> List[Dict[str, str]]:
    sids  = batch["subject_id"]  # list[str]
    stids = batch["study_id"]    # list[str]
    rows = []
    for i in range(count):
        rows.append({
            "subject_id": sids[i],
            "study_id": stids[i],
            "has_img": int(has_img),
            "has_txt": int(has_txt),
        })
    return rows

def dump_train_reps(model, train_loader, mode: str, device: torch.device, out_dir: str):
    img_buf, txt_buf = [], []
    index_rows: List[Dict[str, str]] = []

    for batch in train_loader:
        img_rep, txt_rep = extract_reps_from_batch(model, batch, mode, device)
        bsz = len(batch["subject_id"])

        has_img = img_rep is not None
        has_txt = txt_rep is not None
        index_rows.extend(rows_for_index(batch, bsz, has_img, has_txt))

        if img_rep is not None:
            img_buf.append(img_rep.detach().cpu())
        if txt_rep is not None:
            txt_buf.append(txt_rep.detach().cpu())

    # 저장
    if img_buf:
        img_np = torch.cat(img_buf, dim=0).numpy().astype(np.float32)
        img_np = l2_normalize_rows(img_np)
        np.save(os.path.join(out_dir, "repr_img.npy"), img_np)

    if txt_buf:
        txt_np = torch.cat(txt_buf, dim=0).numpy().astype(np.float32)
        txt_np = l2_normalize_rows(txt_np)
        np.save(os.path.join(out_dir, "repr_txt.npy"), txt_np)

    # 임베딩 순서와 동일한 인덱스 CSV
    if index_rows:
        import pandas as pd
        pd.DataFrame(index_rows).to_csv(
            os.path.join(out_dir, "index.csv"),
            index=False, encoding="utf-8-sig"
        )

# =========================
# 클라이언트 단위 실행
# =========================
def run_for_client(cid: int, batch_size: int, device: torch.device):
    cdir = client_dir(cid)
    ensure_dir(cdir)

    ckpt_path = os.path.join(cdir, "best.pt")
    model, mode = build_model_from_ckpt(cid, ckpt_path, device)
    train_loader, val_loader, mode = build_dataloaders(cid, batch_size)

    # (1) 검증 메트릭 저장
    metrics = evaluate_on_loader(model, val_loader, mode, device)
    metrics["num_classes"] = len(LABEL_COLUMNS)
    with open(os.path.join(cdir, f"client_{cid:02d}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[client_{cid:02d}] metrics:", metrics)

    # (2) 훈련 임베딩 덤프
    dump_train_reps(model, train_loader, mode, device, cdir)
    print(f"[client_{cid:02d}] reps saved → {cdir}")

# =========================
# CLI
# =========================
def parse_ids(text: str) -> List[int]:
    text = text.strip()
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", type=str, default="1-20", help='예: "1-20" 또는 "1,3,5"')
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ids = parse_ids(args.cids)

    for cid in ids:
        try:
            run_for_client(cid, args.batch, device)
        except Exception as e:
            print(f"[client_{cid:02d}] ERROR: {e}")

if __name__ == "__main__":
    main()
