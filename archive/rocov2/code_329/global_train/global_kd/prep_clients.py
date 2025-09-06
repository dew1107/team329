# -*- coding: utf-8 -*-
"""
클라이언트 준비 스크립트:
- (1) 검증 셋으로 f1_macro/auc_macro 계산 → client_{cid}_metrics.json 저장
- (2) train 셋에서 이미지/텍스트 임베딩을 추출 → train_img_reps.npy / train_txt_reps.npy 저장

👉 반드시 아래 3개 훅을 네 코드베이스에 맞게 구현해줘:
   - build_model_from_ckpt(cid, ckpt_path, device)
   - build_dataloaders(cid, batch_size)
   - extract_reps_from_batch(model, batch, device)

Author: you + ChatGPT
"""
from __future__ import annotations
import os, argparse, math, json
from typing import Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .config import cfg
from .utils_io import ensure_dir, client_dir

import sys
import os

from ...local_train import utils_fusion as uf

# ===== prep_clients.py 안에 붙여 넣기 =====
import os, glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# uf = local_train.utils_fusion (위에서 이미 import 되어 있다고 가정)
# from .config import cfg  (orchestrator/prep_clients 에서 이미 import 되어 있다고 가정)

# ---------- 헬퍼들 ----------
def _find_csv(base: str, cid: int, split: str, mod: str):
    """
    base/client_{cid} 아래에서 CSV 자동 탐색
      split: 'train' | 'valid' | 'val'
      mod  : 'image' | 'img' | 'text' | 'caption'
    """
    d = os.path.join(base, f"client_{cid}")
    pats = [
        os.path.join(d, f"{split}_{mod}.csv"),
        os.path.join(d, f"{split}_{'image' if mod in ['image','img'] else 'text'}.csv"),
        os.path.join(d, f"client_{cid}_{split}_{mod}.csv"),
        os.path.join(d, f"*{split}*{mod}*.csv"),
        os.path.join(d, f"*{split}*{'image' if mod in ['image','img'] else 'text'}*.csv"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

def _read_csv_or_none(path: str):
    return pd.read_csv(path) if (path and os.path.exists(path)) else None

def _guess_img_dirs(base: str, cid: int):
    """
    MMClientDataset 가 resolve_image_path()에서
      - 기본적으로 'train' 베이스를 쓰고,
      - ID 문자열에 'valid'/'test' 가 들어 있으면 해당 베이스로 바꿉니다.
    그래서 split별 디렉토리를 최대한 유추해 채워줍니다.
    """
    d = os.path.join(base, f"client_{cid}")
    # 우선순위 높은 후보들
    candidates = {
        "train": ["images_train", "train_images", "train", "images", "imgs"],
        "valid": ["images_valid", "valid_images", "val_images", "valid", "val"],
        "test":  ["images_test",  "test_images",  "test"],
    }
    out = {}
    for split, names in candidates.items():
        found = None
        for name in names:
            p = os.path.join(d, name)
            if os.path.isdir(p):
                # 이미지가 실제로 있는지 간단히 점검
                if glob.glob(os.path.join(p, "*.jpg")) or glob.glob(os.path.join(p, "*.png")):
                    found = p; break
        # 그래도 못 찾으면 client 폴더 자체를 베이스로 둔다(파일명이 ID.jpg 로 바로 놓여있는 경우를 대비)
        out[split] = found if found else d
    return out  # {"train": "...", "valid": "...", "test": "..."}

# ========== TODO 1: 모델 로드 ==========
def _safe_load_state_dict(ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file as safe_load
        return safe_load(str(p), device=str(device))  # state_dict(dict[str,Tensor])
    else:
        # torch>=2.6 권고. 가급적 weights_only=True로 state_dict만 읽기
        return torch.load(str(p), map_location=device, weights_only=True)

def build_model_from_ckpt(cid: int, ckpt_path: str, device: torch.device):
    # 1) 네 로컬 모델 아키텍처 생성 (필요시 수정)
    model = uf.FusionClassifier(num_classes=cfg.NUM_CLASSES)
    model.to(device)

    # 2) state_dict 로드 (safetensors 우선)
    sd = _safe_load_state_dict(ckpt_path, device)
    if isinstance(sd, dict):
        # 마지막 분류기 계층 키 제외
        sd = {k: v for k, v in sd.items() if not k.startswith("fuse.3.")}
        if any(isinstance(v, torch.Tensor) for v in sd.values()):
            model.load_state_dict(sd, strict=False)

    # if isinstance(sd, dict) and any(isinstance(v, torch.Tensor) for v in sd.values()):
    #     model.load_state_dict(sd, strict=False)
    else:
        # 드물게 전체 모델이 저장된 경우 대비(2.6 이상에서만 허용됨)
        if torch.__version__ < "2.6":
            raise RuntimeError(
                "PyTorch >= 2.6 필요: pickle 모델 객체 로드는 차단됩니다. "
                "체크포인트를 state_dict(.safetensors) 형식으로 저장하세요."
            )
        model = sd  # 전체 모델 객체
        model.to(device)

    model.eval()
    return model

# ========== TODO 2: 데이터로더 ==========
def build_dataloaders(cid: int, batch_size: int):
    """
    utils_fusion 의 Dataset/Collate 규약에 맞춰 로더 생성.
    - CSV 자동 탐색: train/valid x image/text
    - label space 구성
    - tokenizer 준비 후 mm_collate 사용
    - 이미지 경로 베이스 자동 유추
    반환: (train_loader, val_loader)
    """
    base = cfg.BASE_DIR  # ex) ...\archive\rocov2\_prepared
    # CSV 찾기 (val/valid 둘 다 지원)
    img_train_csv = _find_csv(base, cid, "train", "image") or _find_csv(base, cid, "train", "img")
    txt_train_csv = _find_csv(base, cid, "train", "text")  or _find_csv(base, cid, "train", "caption")
    img_val_csv   = (_find_csv(base, cid, "valid", "image") or _find_csv(base, cid, "valid", "img")
                     or _find_csv(base, cid, "val", "image") or _find_csv(base, cid, "val", "img"))
    txt_val_csv   = (_find_csv(base, cid, "valid", "text")  or _find_csv(base, cid, "valid", "caption")
                     or _find_csv(base, cid, "val", "text")  or _find_csv(base, cid, "val", "caption"))

    df_img_tr = _read_csv_or_none(img_train_csv)
    df_txt_tr = _read_csv_or_none(txt_train_csv)
    df_img_va = _read_csv_or_none(img_val_csv)
    df_txt_va = _read_csv_or_none(txt_val_csv)

    if df_img_tr is None and df_txt_tr is None:
        raise FileNotFoundError(
            f"[client_{cid}] train CSV를 찾을 수 없습니다. 탐색 경로 예시:\n"
            f" - {os.path.join(base, f'client_{cid}', 'train_image.csv')} / train_text.csv\n"
            f" - 또는 client_{cid}_train_image.csv / client_{cid}_train_text.csv"
        )
    if df_img_va is None and df_txt_va is None:
        # val이 없다면 train을 복제해서라도 최소 동작 (메트릭이 조금 부정확할 수 있음)
        df_img_va, df_txt_va = df_img_tr, df_txt_tr

    # 라벨 공간 구성
    label2idx, _ = uf.build_label_space(
        *(x for x in [df_img_tr, df_txt_tr, df_img_va, df_txt_va] if x is not None)
    )
    num_classes = len(label2idx)
    if num_classes == 0:
        # 레이블이 CSV에 전혀 없다면, 1클래스라도 만들어서 동작하게
        label2idx = {"__dummy__": 0}
        num_classes = 1

    # 토크나이저 (모델 기본값과 동일)
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # 이미지 디렉토리 유추
    img_dirs = _guess_img_dirs(base, cid)  # {'train': ..., 'valid': ..., 'test': ...}

    # Dataset
    ds_train = uf.MMClientDataset(df_img_tr, df_txt_tr, label2idx=label2idx, img_dirs=img_dirs, transform=None, max_len=128)
    ds_val   = uf.MMClientDataset(df_img_va, df_txt_va, label2idx=label2idx, img_dirs=img_dirs, transform=None, max_len=128)

    # Collate
    def _collate(b):
        d = uf.mm_collate(b, tokenizer, max_len=128)
        y = d.get("labels", None)
        if y is not None:
            B, C = y.shape
            T = cfg.NUM_CLASSES  # 230
            if C < T:
                import torch
                pad = torch.zeros(B, T - C, dtype=y.dtype)
                d["labels"] = torch.cat([y, pad], dim=1)
            elif C > T:
                d["labels"] = y[:, :T]
        return d

    # DataLoader
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False, collate_fn=_collate)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=_collate)

    # 모델 생성에 필요한 클래스 수를 외부에서 쓰고 싶다면 속성으로 매달아두기(옵션)
    train_loader.num_classes = num_classes
    val_loader.num_classes   = num_classes

    return train_loader, val_loader

# ========== TODO 3: 임베딩 추출 ==========
def extract_reps_from_batch(model, batch, device):
    """
    utils_fusion.FusionClassifier 내부 인코더를 직접 호출해 임베딩만 추출.
    반환: (img_rep[B, img_out] or None, txt_rep[B, txt_out] or None, labels[B, C])
    """
    # batch 키: pixel_values, input_ids, attention_mask, img_mask, txt_mask, labels
    pv   = batch.get("pixel_values", None)
    ids  = batch.get("input_ids", None)
    am   = batch.get("attention_mask", None)
    imsk = batch.get("img_mask", None)
    tmsk = batch.get("txt_mask", None)
    y    = batch.get("labels", None)

    # 필수 이동/보정
    if pv   is not None: pv   = pv.to(device, non_blocking=True)
    if ids  is not None: ids  = ids.to(device, non_blocking=True)
    if am   is not None: am   = am.to(device, non_blocking=True)
    if imsk is None:     imsk = torch.ones(pv.size(0) if pv is not None else ids.size(0), 1, device=device)
    else:                imsk = imsk.to(device, non_blocking=True).float()
    if tmsk is None:     tmsk = torch.ones(pv.size(0) if pv is not None else ids.size(0), 1, device=device)
    else:                tmsk = tmsk.to(device, non_blocking=True).float()
    y = y.to(device, non_blocking=True).float()

    model.eval()
    with torch.no_grad():
        img_rep = None
        txt_rep = None
        if pv is not None and hasattr(model, "img_enc"):
            zi = model.img_enc(pv)          # [B, img_out]
            zi = zi * imsk                  # 마스크 반영
            img_rep = zi
        if ids is not None and hasattr(model, "txt_enc"):
            zt = model.txt_enc(ids, am)     # [B, txt_out]
            zt = zt * tmsk
            txt_rep = zt

    return img_rep, txt_rep, y

# ========== 멀티라벨 메트릭 (안전판 포함) ==========
def safe_multilabel_metrics(all_logits: torch.Tensor, all_labels: torch.Tensor) -> Dict[str, float]:
    """
    BCEWithLogits 기준:
      - f1_micro / f1_macro: threshold=0.5
      - auc_macro: 클래스별 ROC AUC의 평균, 정의 불가 클래스는 제외하고 평균
    """
    probs = torch.sigmoid(all_logits).cpu().numpy()
    y_true = all_labels.cpu().numpy().astype(np.int32)

    # F1
    y_pred = (probs >= 0.5).astype(np.int32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # AUC (정의 불가 클래스 제외)
    auc_list = []
    C = y_true.shape[1]
    for c in range(C):
        col = y_true[:, c]
        if len(np.unique(col)) < 2:
            continue  # 스킵
        try:
            auc_c = roc_auc_score(col, probs[:, c])
            auc_list.append(auc_c)
        except Exception:
            pass
    auc_macro = float(np.mean(auc_list)) if len(auc_list) > 0 else float("nan")

    return {"f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

# ========== 메트릭 계산 ==========
def evaluate_on_loader(model, val_loader, device):
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            # 모델 최종 로짓을 얻는 네 코드 (아래는 예시)
            # logits = model.forward_for_logits(batch)  # ← 네 코드
            logits = None  # TODO: 위 줄로 교체

            if logits is None:
                # 임시: 인코더만 있는 경우, 분류헤드가 없다면 스킵
                raise NotImplementedError("evaluate_on_loader: 최종 로짓 산출 경로를 연결해 주세요.")
            labels = batch["labels"] if isinstance(batch, dict) else batch[-1]
            logits_all.append(logits.detach().to("cpu"))
            labels_all.append(labels.detach().to("cpu"))
    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return safe_multilabel_metrics(logits_all, labels_all)

# ========== 임베딩 덤프 ==========
def dump_train_reps(model, train_loader, device, out_dir: str):
    img_buf, txt_buf = [], []
    with torch.no_grad():
        for batch in train_loader:
            img_rep, txt_rep, _ = extract_reps_from_batch(model, batch, device)
            if img_rep is not None:
                img_buf.append(img_rep.detach().to("cpu"))
            if txt_rep is not None:
                txt_buf.append(txt_rep.detach().to("cpu"))

    if len(img_buf) == 0 and len(txt_buf) == 0:
        raise RuntimeError("dump_train_reps: 추출된 임베딩이 없습니다. extract_reps_from_batch 구현을 확인하세요.")

    if len(img_buf) > 0:
        img_np = torch.cat(img_buf, dim=0).numpy().astype("float32")
        np.save(os.path.join(out_dir, "train_img_reps.npy"), img_np)
    if len(txt_buf) > 0:
        txt_np = torch.cat(txt_buf, dim=0).numpy().astype("float32")
        np.save(os.path.join(out_dir, "train_txt_reps.npy"), txt_np)

# ========== 엔트리 ==========
def run_for_client(cid: int, batch_size: int, device: torch.device):
    cdir = client_dir(cid)
    ensure_dir(cdir)

    # ckpt 파일명 추정: config.CKPT_NAME 우선
    ckpt_candidates = [
        os.path.join(cdir, cfg.CKPT_NAME.format(cid=cid)),
        os.path.join(cdir, f"client_{cid}_local_fusion_best.pt"),
        os.path.join(cdir, f"client_{cid}_fusion_best.pt"),
        os.path.join(cdir, f"client_{cid}_image_best.pt"),
        os.path.join(cdir, f"client_{cid}_text_best.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError(f"[client_{cid}] 체크포인트(.pt)를 찾을 수 없습니다. 후보: {ckpt_candidates}")

    model = build_model_from_ckpt(cid, ckpt, device)
    train_loader, val_loader = build_dataloaders(cid, batch_size)

    # (1) 메트릭 저장
    try:
        metrics = uf.evaluate(model, val_loader, device)
        with open(os.path.join(cdir, f"client_{cid}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[client_{cid}] metrics saved:", metrics)
    except NotImplementedError as e:
        print(f"[client_{cid}] 평가 스킵 (구현 필요): {e}")

    # (2) 임베딩 저장
    dump_train_reps(model, train_loader, device, cdir)
    print(f"[client_{cid}] reps saved → {cdir}")

def parse_ids(text: str):
    # "0-25" 또는 "0,1,2,3" 형태 지원
    text = text.strip()
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", type=str, default="0-25", help='예: "0-25" 또는 "0,1,2"')
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ids = parse_ids(args.cids)
    for cid in ids:
        try:
            run_for_client(cid, args.batch, device)
        except Exception as e:
            print(f"[client_{cid}] ERROR:", e)

if __name__ == "__main__":
    main()
