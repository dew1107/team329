# -*- coding: utf-8 -*-
"""
evaluate_baseline.py
- 각 클라이언트의 기존 로컬 모델(best.pt)을 '그대로' 사용해 테스트셋을 평가합니다.
- 원본 파일(이미지/텍스트)을 로드하여 추론하고, 멀티모달은 0.5*(img+txt)로 late fusion.
- Z나 updated heads를 전혀 사용하지 않습니다.

사용 예:
  # 전 클라(1~20), 기본 test.csv + 기본 라벨 CSV(NEGBIO/CHEXPERT)
  python -u evaluate_baseline.py --clients 1-20

  # 특정 클라만, 테스트 split/라벨 CSV를 명시
  python -u evaluate_baseline.py --clients 1,3,7 \
    --test_csv .\client_splits\test.csv \
    --label_csv .\mimic-cxr-2.0.0-negbio.csv
"""

import os, csv, json, argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score
from PIL import Image, ImageFile

# --------------------------
# 설정
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_TEST_CSV = r".\client_splits\test.csv"

# 라벨 소스(기본값: train_local.py와 동일)
USE_LABEL = "negbio"  # "negbio" 또는 "chexpert"
LABEL_CSV_NEGBIO  = r".\mimic-cxr-2.0.0-negbio.csv"
LABEL_CSV_CHEXPERT= r".\mimic-cxr-2.0.0-chexpert.csv"
DEFAULT_LABEL_CSV = LABEL_CSV_NEGBIO if USE_LABEL=="negbio" else LABEL_CSV_CHEXPERT

LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

TEXT_MODEL_NAME = "prajjwal1/bert-mini"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary_baseline.csv"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# --------------------------
# 유틸
# --------------------------
def id_to_int(subject_id: str, study_id: str) -> Tuple[int,int]:
    return int(str(subject_id)[1:]), int(str(study_id)[1:])

def load_label_table(label_csv_path: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"라벨 CSV가 없습니다: {label_csv_path}")

    df = pd.read_csv(label_csv_path)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(
            "라벨 CSV에 필요한 칼럼이 없습니다: "
            f"{miss}\n- train_local.py에서 쓰던 NEGBIO/CHEXPERT CSV를 권장합니다.\n"
            "- 만약 test-set CSV(2.1.0)를 쓰려면 per-class 컬럼을 가진 CSV를 지정하세요."
        )
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)
    table = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        vec = [int(row[c]) for c in LABEL_COLUMNS]
        table[key] = vec
    return table

# --------------------------
# 데이터셋 (원본 파일 로드)
# --------------------------
class TestDataset(Dataset):
    def __init__(self, csv_path: str, label_table):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"테스트 CSV가 없습니다: {csv_path}")
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        if len(self.rows) == 0:
            raise RuntimeError(f"테스트 CSV에 데이터가 없습니다: {csv_path}")
        self.label_table = label_table
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self): return len(self.rows)

    def _load_image(self, image_dir: str):
        p = Path(image_dir)
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")])
        if not imgs:
            # 이미지가 없으면 zero-tensor 반환 (멀티모달에서 평균 시 자동 보정)
            return torch.zeros(3,224,224)
        try:
            return IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB"))
        except Exception:
            return torch.zeros(3,224,224)

    def _load_text(self, text_path: str):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""
        enc = self.tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = id_to_int(r["subject_id"], r["study_id"])
        y = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(LABEL_COLUMNS)), dtype=torch.float)
        sample = {"labels": y}
        sample["image"] = self._load_image(r["image_dir"])
        sample["text"]  = self._load_text(r["text_path"])
        return sample

# --------------------------
# 모델 (train_local.py와 동형)
# --------------------------
class ImageHead(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_f, n_out)
    def forward(self, x): return self.head(self.backbone(x))

class TextHead(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, n_out)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:,0]
        return self.cls_head(pooled)

class MultiModalLateFusion(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = ImageHead(n_out)
        self.txt = TextHead(n_out)
    def forward(self, image, text):
        li = self.img(image)
        lt = self.txt(**text)
        return 0.5*(li+lt)

def load_ckpt_as_model(ckpt_path: str, n_out: int, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mode = ckpt.get("mode", "multimodal")
    if mode == "multimodal":
        model = MultiModalLateFusion(n_out)
    elif mode == "image_only":
        model = ImageHead(n_out)
    elif mode == "text_only":
        model = TextHead(n_out)
    else:
        model = MultiModalLateFusion(n_out)
    model.load_state_dict(ckpt["model"])
    return model.to(device), mode

# --------------------------
# 평가
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    y_true, y_prob = [], []
    for batch in loader:
        y = batch["labels"].to(device)
        if isinstance(model, MultiModalLateFusion):
            img = batch["image"].to(device)
            txt = {k:v.to(device) for k,v in batch["text"].items()}
            logits = model(image=img, text=txt)
        elif isinstance(model, ImageHead):
            logits = model(batch["image"].to(device))
        else:
            txt = {k:v.to(device) for k,v in batch["text"].items()}
            logits = model(**txt)
        loss = crit(logits, y)
        total_loss += loss.item() * y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)

    # 지표
    f1_micro = f1_score(y_true, (y_prob>=0.5).astype(int), average="micro", zero_division=0)
    f1_macro = f1_score(y_true, (y_prob>=0.5).astype(int), average="macro", zero_division=0)
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:,j]
        if len(np.unique(col))<2:
            aucs.append(float("nan"))
        else:
            try:
                aucs.append(roc_auc_score(col, y_prob[:,j]))
            except Exception:
                aucs.append(float("nan"))
    macro_auc = float(np.nanmean(aucs))
    return total_loss/len(loader.dataset), f1_micro, f1_macro, macro_auc, aucs

# --------------------------
# 메인
# --------------------------
def parse_clients(s: str) -> List[int]:
    s = s.strip().lower()
    if s in ("all",""): return list(range(1,21))
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            out.extend(list(range(int(a), int(b)+1)))
        else:
            out.append(int(part))
    # unique + sort
    return sorted(list(dict.fromkeys(out)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20", help='예: "1-20" 또는 "1,3,5"')
    ap.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    # 데이터
    label_table = load_label_table(args.label_csv)
    ds = TestDataset(args.test_csv, label_table)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)
    n_out = len(LABEL_COLUMNS)

    clients = parse_clients(args.clients)

    rows = []
    for cid in clients:
        ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (no best.pt)")
            continue

        model, mode = load_ckpt_as_model(str(ckpt_path), n_out, DEVICE)
        loss, f1_micro, f1_macro, macro_auc, aurocs = evaluate(model, dl, DEVICE)

        print(f"[Client {cid:02d} | {mode:11s}] "
              f"loss={loss:.4f}  f1_micro={f1_micro:.4f}  f1_macro={f1_macro:.4f}  macro_auc={macro_auc:.4f}")

        # per-client JSON 저장
        client_dir = Path(f"./outputs/client_{cid:02d}")
        client_dir.mkdir(parents=True, exist_ok=True)
        with open(client_dir/"test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "client_id": cid, "mode": mode,
                "loss": loss, "f1_micro": f1_micro, "f1_macro": f1_macro,
                "macro_auroc": macro_auc,
                "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, aurocs)}
            }, f, indent=2, ensure_ascii=False)

        row = {"client_id": cid, "mode": mode, "loss": loss,
               "f1_micro": f1_micro, "f1_macro": f1_macro, "macro_auroc": macro_auc}
        for c,a in zip(LABEL_COLUMNS, aurocs): row[f"AUROC_{c}"] = a
        rows.append(row)

    # 전체 요약 CSV
    if rows:
        rows.sort(key=lambda r: (r["macro_auroc"] if r["macro_auroc"]==r["macro_auroc"] else -1), reverse=True)
        headers = ["client_id","mode","loss","f1_micro","f1_macro","macro_auroc"] + [f"AUROC_{c}" for c in LABEL_COLUMNS]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
        print("Top-5 by macro_auroc:")
        for r in rows[:5]:
            print(f"  client_{r['client_id']:02d} [{r['mode']}]: {r['macro_auroc']:.4f}")
    else:
        print("[WARN] no evaluated clients; check paths.")

if __name__ == "__main__":
    main()
