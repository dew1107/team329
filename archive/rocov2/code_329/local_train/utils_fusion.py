# utils_fusion.py
# Self-contained utilities for a multimodal (image+text) classifier that tolerates
# missing modalities per sample. Encoders are separate and fused for classification.
from __future__ import annotations
import os, ast, logging, sys
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import AutoTokenizer, AutoModel

LOGGER = logging.getLogger('fusion')
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    LOGGER.addHandler(ch)

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).lower().replace(' ','').replace('_',''): c for c in df.columns}
    for c in candidates:
        k = c.lower().replace(' ','').replace('_','')
        if k in cols: return cols[k]
    return None

def normalize_image_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idc = _pick_col(df, ['ID','ImageID','image_id','id'])
    if idc is None: raise ValueError('Image CSV must include ID')
    if idc != 'ID': df.rename(columns={idc:'ID'}, inplace=True)
    lab = _pick_col(df, ['Disease_Labels','labels','label','disease_labels','Diseases'])
    if lab and lab != 'Disease_Labels': df.rename(columns={lab:'Disease_Labels'}, inplace=True)
    if 'Disease_Labels' in df.columns: df['Disease_Labels'] = df['Disease_Labels'].fillna('[]').astype(str)
    keep = ['ID'] + (['Disease_Labels'] if 'Disease_Labels' in df.columns else [])
    return df[keep]

def normalize_text_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idc = _pick_col(df, ['ID','TextID','text_id','id'])
    if idc is None: raise ValueError('Text CSV must include ID')
    if idc != 'ID': df.rename(columns={idc:'ID'}, inplace=True)
    cap = _pick_col(df, ['Caption','caption','Text','text','Report','report'])
    if cap and cap != 'Caption': df.rename(columns={cap:'Caption'}, inplace=True)
    lab = _pick_col(df, ['Disease_Labels','labels','label','disease_labels','Diseases'])
    if lab and lab != 'Disease_Labels': df.rename(columns={lab:'Disease_Labels'}, inplace=True)
    if 'Disease_Labels' in df.columns: df['Disease_Labels'] = df['Disease_Labels'].fillna('[]').astype(str)
    if 'Caption' in df.columns: df['Caption'] = df['Caption'].fillna('').astype(str)
    keep = ['ID']
    if 'Caption' in df.columns: keep.append('Caption')
    if 'Disease_Labels' in df.columns: keep.append('Disease_Labels')
    return df[keep]

def parse_labels_cell(cell) -> List[str]:
    if isinstance(cell, list):
        return [str(x) for x in cell]
    s = str(cell).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list): return [str(x) for x in v]
    except Exception:
        pass
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    return [t.strip().strip('\'').strip('"') for t in s.split(',') if t.strip()]

def build_label_space(*dfs: pd.DataFrame) -> Tuple[Dict[str,int], Dict[int,str]]:
    labels = set()
    for df in dfs:
        if df is None: continue
        if 'Disease_Labels' in df.columns:
            for v in df['Disease_Labels'].values:
                labels.update(parse_labels_cell(v))
    labels = sorted(labels)
    label2idx = {l:i for i,l in enumerate(labels)}
    idx2label = {i:l for l,i in label2idx.items()}
    return label2idx, idx2label

def encode_labels(labels: List[str], label2idx: Dict[str,int], num_classes: int) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    for l in labels:
        if l in label2idx: y[label2idx[l]] = 1.0
    return y

def resolve_image_path(id_str: str, img_dirs: Dict[str,str]) -> Optional[str]:
    id_l = id_str.lower()
    base = img_dirs.get('train','')
    if 'valid' in id_l: base = img_dirs.get('valid', base)
    elif 'test' in id_l: base = img_dirs.get('test', base)
    for ext in ('.jpg','.png','.jpeg','.webp'):
        p = os.path.join(base, id_str + ext)
        if os.path.isfile(p): return p
    return None

class MMClientDataset(Dataset):
    def __init__(self, df_img: Optional[pd.DataFrame], df_txt: Optional[pd.DataFrame],
                 label2idx: Dict[str,int], img_dirs: Dict[str,str],
                 transform: Optional[object]=None, max_len: int = 128):
        self.df_img = normalize_image_df(df_img) if df_img is not None else None
        self.df_txt = normalize_text_df(df_txt) if df_txt is not None else None
        self.label2idx = label2idx
        self.num_classes = len(label2idx)
        self.img_dirs = img_dirs
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])
        self.max_len = max_len
        ids = set()
        if self.df_img is not None: ids.update(self.df_img['ID'].tolist())
        if self.df_txt is not None: ids.update(self.df_txt['ID'].tolist())
        self.ids = sorted(ids)
        self.img_map = {r.ID: r for r in self.df_img.itertuples()} if self.df_img is not None else {}
        self.txt_map = {r.ID: r for r in self.df_txt.itertuples()} if self.df_txt is not None else {}

    def __len__(self): return len(self.ids)

    def _open_image(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            return None

    def __getitem__(self, idx):
        sid = self.ids[idx]
        labels: List[str] = []
        if sid in self.img_map and hasattr(self.img_map[sid], 'Disease_Labels'):
            labels += parse_labels_cell(getattr(self.img_map[sid], 'Disease_Labels'))
        if sid in self.txt_map and hasattr(self.txt_map[sid], 'Disease_Labels'):
            labels += parse_labels_cell(getattr(self.txt_map[sid], 'Disease_Labels'))
        labels = sorted(set(labels))
        y = torch.from_numpy(encode_labels(labels, self.label2idx, self.num_classes))
        pix = None; img_mask = 0.0
        if sid in self.img_map:
            p = resolve_image_path(sid, self.img_dirs)
            if p:
                im = self._open_image(p)
                if im is not None:
                    pix = self.transform(im)
                    img_mask = 1.0
        cap = '';
        txt_mask = 0.0
        if sid in self.txt_map and hasattr(self.txt_map[sid], 'Caption'):
            cap = getattr(self.txt_map[sid], 'Caption') or ''
            if len(cap.strip())>0: txt_mask = 1.0
        return {
            "id": sid,
            "pixel_values": pix,  # (3,224,224) 또는 None -> collate에서 0으로 채움
            "caption": cap,  # str
            "labels": y,  # (C,) float tensor
            "img_mask": torch.tensor([img_mask], dtype=torch.float32),  # (1,)
            "txt_mask": torch.tensor([txt_mask], dtype=torch.float32),  # (1,)
        }

def mm_collate(batch, tokenizer, max_len: int = 128):
    # images → (B,3,224,224)
    pix = [b["pixel_values"] for b in batch]
    C,H,W = (3,224,224) if pix[0] is None else pix[0].shape
    pix = [(p if p is not None else torch.zeros(C,H,W)) for p in pix]
    pixel_values = torch.stack(pix, 0).float()

    # text → (B,L), (B,L)
    caps = [b["caption"] if isinstance(b["caption"], str) else "" for b in batch]
    enc = tokenizer(caps, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    # labels → (B,C)
    labels = torch.stack([b["labels"] for b in batch], 0).float()

    # ✅ masks → (B,1)  ← 여기서 “단 한 번만” 만들고 shape 고정
    img_mask = torch.tensor([b["img_mask"] for b in batch], dtype=torch.float32).unsqueeze(1)
    txt_mask = torch.tensor([b["txt_mask"] for b in batch], dtype=torch.float32).unsqueeze(1)

    return {
        "pixel_values": pixel_values,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
        "img_mask": img_mask,
        "txt_mask": txt_mask,
    }

class ImageEncoder(nn.Module):
    def __init__(self, trainable: bool = True, out_dim: int = 256):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.proj = nn.Linear(576, out_dim)
        if not trainable:
            for p in self.parameters(): p.requires_grad_(False)
    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        z = self.proj(h)
        return z

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = 'prajjwal1/bert-tiny', out_dim: int = 256, trainable: bool = True):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hid = self.base.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)
        if not trainable:
            for p in self.base.parameters(): p.requires_grad_(False)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        s = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        z = self.proj(s)
        return z

class FusionClassifier(nn.Module):
    def __init__(self, num_classes: int, img_out: int = 256, txt_out: int = 256, hidden: int = 256, dropout: float = 0.2,
                 text_model_name: str = 'prajjwal1/bert-tiny', image_trainable: bool = True, text_trainable: bool = True):
        super().__init__()
        self.img_enc = ImageEncoder(trainable=image_trainable, out_dim=img_out)
        self.txt_enc = TextEncoder(model_name=text_model_name, out_dim=txt_out, trainable=text_trainable)
        self.fuse = nn.Sequential(
            nn.Linear(img_out + txt_out, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, pixel_values, input_ids, attention_mask, img_mask, txt_mask):
        # 계약 확인 (디버깅에만 도움, 배포 시 빼도 됨)
        assert img_mask.ndim == 2 and img_mask.shape[1] == 1, f"img_mask shape={img_mask.shape} (expected B,1)"
        assert txt_mask.ndim == 2 and txt_mask.shape[1] == 1, f"txt_mask shape={txt_mask.shape} (expected B,1)"

        zi = self.img_enc(pixel_values)               # (B,256)
        zt = self.txt_enc(input_ids, attention_mask)  # (B,256)

        # 여기서 브로드캐스트가 자연스럽게 (B,256)로 됨
        zi = zi * img_mask.to(zi.dtype)
        zt = zt * txt_mask.to(zt.dtype)

        z = torch.cat([zi, zt], dim=1)                # (B,512)
        return self.fuse(z)

@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    ys, ps = [], []

    for b in loader:
        for k in ("pixel_values","input_ids","attention_mask","img_mask","txt_mask","labels"):
            b[k] = b[k].to(device)
        logits = model(b["pixel_values"], b["input_ids"], b["attention_mask"], b["img_mask"], b["txt_mask"])
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = b["labels"].detach().cpu().numpy()
        ys.append(y); ps.append(prob)

    Y = np.concatenate(ys, axis=0)
    P = np.concatenate(ps, axis=0)

    # --- 멀티라벨 평가 ---
    y_pred = (P >= threshold).astype(int)

    # F1
    f1_micro = f1_score(Y, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y, y_pred, average="macro", zero_division=0)

    # AUC: 클래스별로 positive/negative 둘 다 존재하는 경우만
    aucs = []
    for j in range(Y.shape[1]):
        if len(np.unique(Y[:, j])) == 2:   # 양성과 음성이 모두 있는 경우만
            aucs.append(roc_auc_score(Y[:, j], P[:, j]))
    auc_macro = float(np.mean(aucs)) if aucs else 0.0

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "auc_macro": auc_macro,
    }

def compute_pos_weight(loader: DataLoader, device: torch.device):
    total = None; n = 0
    for b in loader:
        y = b['labels']
        n += y.shape[0]
        total = y.sum(0) if total is None else total + y.sum(0)
    if total is None: return None
    P = total.float(); N = (n - total).float().clamp(min=1.0)
    w = (N / P.clamp(min=1.0)).clamp(max=50.0)
    return w.to(device)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, log_every: int = 50):
    model.train(); running = 0.0
    for i, b in enumerate(loader, 1):
        for k in ('pixel_values','input_ids','attention_mask','img_mask','txt_mask','labels'):
            b[k] = b[k].to(device)
        logits = model(b['pixel_values'], b['input_ids'], b['attention_mask'], b['img_mask'], b['txt_mask'])
        loss = criterion(logits, b['labels'])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        running += float(loss.detach().cpu())
        if (i % log_every) == 0:
            LOGGER.info(f"[HB train] i={i}/{len(loader)} loss={running/i:.4f}")
    return running / max(1, len(loader))
