# evaluate_all_clients_on_test.py
import os, csv
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
import numpy as np

# =========================
# 경로/설정
# =========================
TEST_CSV_PATH  = r".\client_splits\test.csv"

# 라벨 소스 선택
USE_LABEL = "negbio"  # "negbio" 또는 "chexpert"
LABEL_CSV_NEGBIO  = r"C:\Users\user\PycharmProjects\team329\mimic-cxr\mimic-cxr-2.0.0-negbio.csv"
LABEL_CSV_CHEXPERT= r"C:\Users\user\PycharmProjects\team329\mimic-cxr\mimic-cxr-2.0.0-chexpert.csv"
LABEL_CSV = LABEL_CSV_NEGBIO if USE_LABEL=="negbio" else LABEL_CSV_CHEXPERT

# 라벨 컬럼(학습과 동일해야 함)
LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# (선택) 메타데이터 기반 대표 이미지 선택
METADATA_CSV = r"C:\HJHJ0808\김희진\연구\졸업프로젝트\mimic-cxr-project\mimic-cxr-2.0.0-metadata.csv"  # 없으면 ""로
IMG_ROOT     = r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files"

TEXT_MODEL_NAME = "prajjwal1/bert-mini"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"

# =========================
# 유틸
# =========================
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def id_to_int(subject_id: str, study_id: str) -> Tuple[int,int]:
    return int(subject_id[1:]), int(study_id[1:])

def build_image_picker_from_metadata(metadata_csv: str, img_root: str):
    import pandas as pd
    try:
        if not metadata_csv or not os.path.exists(metadata_csv):
            return {}
        df = pd.read_csv(metadata_csv)
    except Exception as e:
        print(f"[WARN] Skip metadata ({e})")
        return {}
    path_col = "path" if "path" in df.columns else None
    view_col = "ViewPosition" if "ViewPosition" in df.columns else ("view" if "view" in df.columns else None)
    if "subject_id" not in df.columns or "study_id" not in df.columns or not path_col:
        return {}
    df["subject_id"] = df["subject_id"].astype(int)
    df["study_id"]   = df["study_id"].astype(int)
    if view_col:
        df["_prio"] = df[view_col].fillna("").map(lambda v: 0 if v in ("PA","AP") else 1)
    else:
        df["_prio"] = 1
    df = df.sort_values(["subject_id","study_id","_prio"])
    pick = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        if key in pick: continue
        pick[key] = str(Path(img_root) / row[path_col])
    return pick

def load_label_table(label_csv_path: str) -> Dict[Tuple[int,int], List[float]]:
    import pandas as pd
    df = pd.read_csv(label_csv_path)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"라벨 CSV에 칼럼이 없습니다: {miss}")
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)  # -1/0 -> 0, 1 -> 1
    table = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        vec = [int(row[c]) for c in LABEL_COLUMNS]
        table[key] = vec
    return table

# =========================
# 데이터셋
# =========================
class TestDataset(Dataset):
    def __init__(self, csv_path: str, label_table, meta_picker=None):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.label_table = label_table
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.meta_picker = meta_picker

    def __len__(self):
        return len(self.rows)

    def _load_image(self, image_dir: str, subject_id: str=None, study_id: str=None):
        if self.meta_picker and subject_id and study_id:
            key = (int(subject_id[1:]), int(study_id[1:]))
            p = self.meta_picker.get(key)
            if p and os.path.exists(p):
                try:
                    return IMG_TRANSFORM(Image.open(p).convert("RGB"))
                except: pass
        p = Path(image_dir)
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")])
        if not imgs:
            return torch.zeros(3,224,224)
        return IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB"))

    def _load_text(self, text_path: str):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            text = ""
        enc = self.tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = id_to_int(r["subject_id"], r["study_id"])
        y = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(LABEL_COLUMNS)), dtype=torch.float)
        sample = {"labels": y}
        sample["image"] = self._load_image(r["image_dir"], r["subject_id"], r["study_id"])
        sample["text"]  = self._load_text(r["text_path"])
        return sample

# =========================
# 모델들 (train_local.py와 동일 구조)
# =========================
class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(pretrained=False)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_f, n_out)
    def forward(self, x): return self.head(self.backbone(x))

class TextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.enc = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        hid = self.enc.config.hidden_size
        self.cls_head = nn.Linear(hid, n_out)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:,0]
        return self.cls_head(pooled)

class MultiModalLateFusion(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.img = ImageHead(n_out)
        self.txt = TextHead(n_out)
    def forward(self, image, text):
        li = self.img(image); lt = self.txt(**text)
        return 0.5*(li+lt)

@torch.no_grad()
def evaluate_model_on_test(model, loader, criterion, device):
    model.eval()
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
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)
    aurocs = []
    for j in range(y_true.shape[1]):
        try: aurocs.append(roc_auc_score(y_true[:,j], y_prob[:,j]))
        except ValueError: aurocs.append(float("nan"))
    return total_loss/len(loader.dataset), aurocs

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

# =========================
# 메인
# =========================
def main():
    label_table = load_label_table(LABEL_CSV)
    meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)

    dataset = TestDataset(TEST_CSV_PATH, label_table, meta_picker=meta_picker)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    criterion = nn.BCEWithLogitsLoss()
    n_out = len(LABEL_COLUMNS)

    rows = []
    for cid in range(1, 21):
        ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (checkpoint not found)")
            continue
        model, mode = load_ckpt_as_model(str(ckpt_path), n_out, DEVICE)
        loss, aurocs = evaluate_model_on_test(model, loader, criterion, DEVICE)
        macro = float(np.nanmean(aurocs))
        print(f"[Client {cid:02d} | {mode:11s}] Loss={loss:.4f}  Macro-AUROC={macro:.4f}")
        row = {"client_id": cid, "mode": mode, "loss": loss, "macro_auroc": macro}
        for c,a in zip(LABEL_COLUMNS, aurocs): row[f"AUROC_{c}"] = a
        rows.append(row)

    # 저장
    if rows:
        # 정렬: macro_auroc 내림차순
        rows.sort(key=lambda r: (r["macro_auroc"] if r["macro_auroc"]==r["macro_auroc"] else -1), reverse=True)
        headers = ["client_id","mode","loss","macro_auroc"] + [f"AUROC_{c}" for c in LABEL_COLUMNS]
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
        # 상위 5개 출력
        print("\nTop-5 by Macro-AUROC:")
        for r in rows[:5]:
            print(f"  client_{r['client_id']:02d} [{r['mode']}]: {r['macro_auroc']:.4f}")
    else:
        print("[WARN] No results. Check checkpoints exist under ./outputs/client_xx/best.pt")

if __name__ == "__main__":
    main()
