# -*- coding: utf-8 -*-
"""
evaluate_dirs_compare_updated.py
- 폴더 트리(예: ./data/p10, ./data/p11)를 재귀 스캔하여 p*/s* 스터디를 수집
- 라벨 CSV에서 (subject_id, study_id)로 라벨 매칭
- 각 클라이언트의 baseline 모델(./outputs/client_xx/best.pt)과
  Z 정렬로 업데이트된 헤드(./outputs/client_xx/updated_heads.npz)를
  같은 테스트셋에서 각각 평가하여 성능을 비교
- 훈련/업데이트 수행 없음(순수 평가)

예)
  python -u evaluate_dirs_compare_updated.py --clients 1-20 \
    --label_csv .\mimic-cxr-2.1.0-test-set-labeled.csv \
    --roots ".\data\p10,.\data\p11"
"""

import os, re, csv, json, argparse, glob
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from PIL import Image, ImageFile

# --------------------------
# 설정
# --------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_WORKERS = 0
TEXT_MODEL_NAME = "prajjwal1/bert-mini"

LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

OUT_DIR = Path("./eval_results"); OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary_dirs_compare_updated.csv"

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# --------------------------
# 유틸
# --------------------------
def cdir(cid: int) -> Path:
    p = Path(f"./outputs/client_{cid:02d}")
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_roots(roots_arg: str) -> List[str]:
    if not roots_arg:
        return ["./data/p10", "./data/p11"]
    return [r.strip() for r in roots_arg.split(",") if r.strip()]

def find_study_dirs(roots: List[str]) -> List[str]:
    study_dirs: List[str] = []
    for root in roots:
        if not os.path.exists(root): continue
        study_dirs.extend(glob.glob(os.path.join(root, "**", "s*"), recursive=True))
    study_dirs = [d for d in study_dirs if os.path.isdir(d)]
    kept = []
    for d in study_dirs:
        if re.search(r"[\\/]p\d+[\\/]", d) and re.search(r"[\\/]s\d+[\\/]?$", d):
            kept.append(d)
    return sorted(list(dict.fromkeys(kept)))

def parse_ids_from_path(study_dir: str) -> Tuple[int, int]:
    # 예: .../data/p10/p100032/s50414267 -> subject_id=100032, study_id=50414267
    # 윈도우/유닉스 경로 모두 지원
    p_matches = re.findall(r"[\\/]p(\d+)(?=[\\/])", study_dir)
    s_matches = re.findall(r"[\\/]s(\d+)(?=[\\/]?$)", study_dir)
    if not p_matches or not s_matches:
        raise ValueError(f"경로에서 subject/study 파싱 실패: {study_dir}")

    # 가장 안쪽 p#### 를 선택 (또는 길이가 가장 긴 것)
    subject_id = int(p_matches[-1])
    # subject_id = int(max(p_matches, key=len))  # 더 안전하게 가려면 이 라인 사용

    study_id = int(s_matches[-1])
    return subject_id, study_id

def load_label_table(label_csv: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"라벨 CSV가 없습니다: {label_csv}")
    df = pd.read_csv(label_csv)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"라벨 CSV에 필요한 칼럼이 없습니다: {miss}")
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)
    table = {}
    for _, r in df.iterrows():
        key = (int(r["subject_id"]), int(r["study_id"]))
        table[key] = [int(r[c]) for c in LABEL_COLUMNS]
    return table

def safe_per_class_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """클래스별 AUROC와 각 클래스의 유효 샘플 수(=해당 열의 샘플 수)를 반환.
       양/음성 모두 존재할 때만 AUROC를 계산."""
    C = y_true.shape[1]
    aurocs, supports = [], []
    for j in range(C):
        col = y_true[:, j]
        if len(np.unique(col)) < 2:
            aurocs.append(np.nan)
            supports.append(0)
            continue
        aurocs.append(roc_auc_score(col, y_prob[:, j]))
        supports.append(len(col))
    return np.array(aurocs, dtype=float), np.array(supports, dtype=float)

def micro_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """전 클래스/샘플 평탄화 후 ROC-AUC. 전체에 양/음성이 모두 있어야 함."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true.ravel(), y_prob.ravel()))

def micro_ap(y_true: np.ndarray, y_prob: np.ndarray):
    """전 클래스/샘플 평탄화 후 평균정밀도(AP, PR-AUC)."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(average_precision_score(y_true.ravel(), y_prob.ravel()))

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """불균형 친화 지표로 재구성."""
    y_pred = (y_prob >= threshold).astype(int)

    # F1
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUROC들
    per_class_auc, supports = safe_per_class_auc(y_true, y_prob)
    valid = ~np.isnan(per_class_auc) & (supports > 0)

    macro_auc_valid = float(np.nanmean(per_class_auc[valid])) if valid.any() else float('nan')
    macro_auc_weighted = float(np.average(per_class_auc[valid], weights=supports[valid])) if valid.any() else float('nan')
    micro_roc_auc = micro_auc(y_true, y_prob)
    ap_micro = micro_ap(y_true, y_prob)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "micro_auroc": micro_roc_auc,
        "macro_auroc_valid": macro_auc_valid,       # 유효 클래스 평균
        "macro_auroc_weighted": macro_auc_weighted, # 유효 클래스 가중 평균
        "micro_ap": ap_micro,
        "per_class_auroc": per_class_auc.tolist()
    }

# --------------------------
# 데이터셋 (원본 파일 로드, 텍스트 없으면 마스크=0)
# --------------------------
class TestDirsDataset(Dataset):
    def __init__(self, study_dirs, label_table):
        self.rows = []
        missing = 0
        pos_any = 0
        for d in study_dirs:
            try:
                sid, stid = parse_ids_from_path(d)
            except Exception:
                continue
            y = label_table.get((sid, stid))
            if y is None:
                missing += 1
                y = [0] * len(LABEL_COLUMNS)  # 없으면 0으로
            else:
                if any(y):
                    pos_any += 1
            self.rows.append({"dir": d, "sid": sid, "stid": stid, "y": y})

        n = len(self.rows)
        if n == 0:
            raise RuntimeError("테스트 폴더에서 유효한 스터디가 하나도 없습니다.")

        print(f"[INFO] 라벨 매칭 커버리지: matched={n - missing}/{n} "
              f"({(n-missing)/n*100:.1f}%) | any-positive={pos_any}/{n} ({pos_any/n*100:.1f}%)")

        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)


    def __len__(self): return len(self.items)

    def _load_image(self, d: str):
        imgs = sorted([*Path(d).glob("*.jpg"), *Path(d).glob("*.jpeg"), *Path(d).glob("*.png")])
        if not imgs:
            return torch.zeros(3,224,224), 0.0
        try:
            return IMG_TRANSFORM(Image.open(imgs[0]).convert("RGB")), 1.0
        except Exception:
            return torch.zeros(3,224,224), 0.0

    def _find_text(self, d: str) -> Optional[Path]:
        cands = list(Path(d).glob("*.txt"))
        return cands[0] if cands else None

    def _load_text(self, p: Optional[Path]):
        if p is None:
            text = ""
        else:
            try:
                text = Path(p).read_text(encoding="utf-8")
            except Exception:
                text = ""
        enc = self.tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}, (0.0 if text=="" else 1.0)

    def __getitem__(self, idx):
        it = self.items[idx]
        y = torch.tensor(np.array(it["y"], dtype=np.float32))
        img, has_img = self._load_image(it["dir"])
        txt_path = self._find_text(it["dir"])
        txt, has_txt = self._load_text(txt_path)
        return {
            "labels": y,
            "image": img,
            "text": txt,
            "has_img": torch.tensor([has_img], dtype=torch.float32).squeeze(0),
            "has_txt": torch.tensor([has_txt], dtype=torch.float32).squeeze(0),
        }

# --------------------------
# 모델 (train_local.py와 동형)
# --------------------------
class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
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

def load_ckpt_as_model(ckpt_path: str, n_out: int):
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
    return model.to(DEVICE), mode

# --------------------------
# updated_heads 적용
# --------------------------
def maybe_apply_updated_heads(model, cid: int, n_out: int) -> bool:
    """
    outputs/client_xx/updated_heads.npz가 있으면, 가능한 헤드에 가중치를 주입.
    - 키: W_img, b_img, W_txt, b_txt
    - shape가 맞지 않으면 해당 모달은 건너뜀.
    반환: 하나라도 적용되면 True
    """
    npz_path = cdir(cid) / "updated_heads.npz"
    if not npz_path.exists():
        return False

    try:
        data = np.load(npz_path)
    except Exception:
        return False

    applied = False
    with torch.no_grad():
        if hasattr(model, "img") and "W_img" in data and "b_img" in data:
            W = torch.tensor(data["W_img"], dtype=model.img.head.weight.dtype, device=model.img.head.weight.device)
            b = torch.tensor(data["b_img"], dtype=model.img.head.bias.dtype, device=model.img.head.bias.device)
            if tuple(W.shape) == tuple(model.img.head.weight.shape) and tuple(b.shape) == tuple(model.img.head.bias.shape):
                model.img.head.weight.copy_(W); model.img.head.bias.copy_(b); applied = True
        if hasattr(model, "txt") and "W_txt" in data and "b_txt" in data:
            W = torch.tensor(data["W_txt"], dtype=model.txt.cls_head.weight.dtype, device=model.txt.cls_head.weight.device)
            b = torch.tensor(data["b_txt"], dtype=model.txt.cls_head.bias.dtype, device=model.txt.cls_head.bias.device)
            if tuple(W.shape) == tuple(model.txt.cls_head.weight.shape) and tuple(b.shape) == tuple(model.txt.cls_head.bias.shape):
                model.txt.cls_head.weight.copy_(W); model.txt.cls_head.bias.copy_(b); applied = True
        # 싱글 모달 모델 대응
        if isinstance(model, ImageHead) and "W_img" in data and "b_img" in data:
            W = torch.tensor(data["W_img"], dtype=model.head.weight.dtype, device=model.head.weight.device)
            b = torch.tensor(data["b_img"], dtype=model.head.bias.dtype, device=model.head.bias.device)
            if tuple(W.shape) == tuple(model.head.weight.shape) and tuple(b.shape) == tuple(model.head.bias.shape):
                model.head.weight.copy_(W); model.head.bias.copy_(b); applied = True
        if isinstance(model, TextHead) and "W_txt" in data and "b_txt" in data:
            W = torch.tensor(data["W_txt"], dtype=model.cls_head.weight.dtype, device=model.cls_head.weight.device)
            b = torch.tensor(data["b_txt"], dtype=model.cls_head.bias.dtype, device=model.cls_head.bias.device)
            if tuple(W.shape) == tuple(model.cls_head.weight.shape) and tuple(b.shape) == tuple(model.cls_head.bias.shape):
                model.cls_head.weight.copy_(W); model.cls_head.bias.copy_(b); applied = True

    return applied

# --------------------------
# 평가 (모달 마스크 기반 late fusion)
# --------------------------
@torch.no_grad()
def evaluate_mmasked(model, loader):
    loss = 0.0
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    tot = 0.0; y_true=[]; y_prob=[]
    for batch in loader:
        y = batch["labels"].to(DEVICE)
        if isinstance(model, MultiModalLateFusion):
            mi = batch["has_img"].to(DEVICE).unsqueeze(1)
            mt = batch["has_txt"].to(DEVICE).unsqueeze(1)
            li = model.img(batch["image"].to(DEVICE))
            lt = model.txt(**{k:v.to(DEVICE) for k,v in batch["text"].items()})
            logits = (li*mi + lt*mt) / (mi+mt).clamp(min=1.0)
        elif isinstance(model, ImageHead):
            logits = model(batch["image"].to(DEVICE))
        else:
            logits = model(**{k:v.to(DEVICE) for k,v in batch["text"].items()})
        loss = crit(logits, y)
        tot += loss.item()*y.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(y_true); y_prob = np.vstack(y_prob)
    f1_micro = f1_score(y_true, (y_prob>=0.5).astype(int), average="micro", zero_division=0)
    f1_macro = f1_score(y_true, (y_prob>=0.5).astype(int), average="macro", zero_division=0)
    aucs = []
    valid_mask = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if len(np.unique(col)) < 2:
            aucs.append(float("nan"))
            valid_mask.append(False)
        else:
            try:
                aucs.append(roc_auc_score(col, y_prob[:, j]))
                valid_mask.append(True)
            except Exception:
                aucs.append(float("nan"))
                valid_mask.append(False)

    macro_auc_valid = float(np.nanmean([a for a, ok in zip(aucs, valid_mask) if ok])) if any(valid_mask) else float(
        "nan")
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    print(
        f"loss={loss:.4f}  "
        f"F1_micro={metrics['f1_micro']:.4f}  "
        f"F1_macro={metrics['f1_macro']:.4f}  "
        f"AUROC_micro={metrics['micro_auroc'] if metrics['micro_auroc'] == metrics['micro_auroc'] else float('nan'):.4f}  "
        f"AUROC_macro(valid)={metrics['macro_auroc_valid'] if metrics['macro_auroc_valid'] == metrics['macro_auroc_valid'] else float('nan'):.4f}  "
        f"AUROC_macro(weighted)={metrics['macro_auroc_weighted'] if metrics['macro_auroc_weighted'] == metrics['macro_auroc_weighted'] else float('nan'):.4f}  "
        f"AP_micro={metrics['micro_ap'] if metrics['micro_ap'] == metrics['micro_ap'] else float('nan'):.4f}"
    )

    return {
        "loss": loss / len(loader.dataset),
        "f1_micro": metrics["f1_micro"],
        "f1_macro": metrics["f1_macro"],
        "micro_auroc": metrics["micro_auroc"],  # NaN 가능
        "macro_auroc_valid": metrics["macro_auroc_valid"],  # 유효 클래스만 평균
        "macro_auroc_weighted": metrics["macro_auroc_weighted"],  # 라벨 빈도 가중
        "micro_ap": metrics["micro_ap"],  # micro-AP
        "per_class_auroc": metrics["per_class_auroc"],  # 길이=C, NaN 포함 가능
    }

# --------------------------
# 메인
# --------------------------
def parse_clients(s: str) -> List[int]:
    s = s.strip().lower()
    if s in ("all",""): return list(range(1,21))
    out=[]
    for part in s.split(","):
        part=part.strip()
        if "-" in part:
            a,b = part.split("-",1)
            out.extend(list(range(int(a), int(b)+1)))
        else:
            out.append(int(part))
    return sorted(list(dict.fromkeys(out)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20")
    ap.add_argument("--label_csv", type=str, required=True)
    ap.add_argument("--roots", type=str, default="./data/p10,./data/p11",
                    help="쉼표로 여러 루트. 각 루트에서 p*/s* 디렉터리를 재귀 검색")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()



    roots = parse_roots(args.roots)
    study_dirs = find_study_dirs(roots)
    if not study_dirs:
        raise RuntimeError(f"스터디 폴더를 찾지 못했습니다. roots={roots}")
    label_table = load_label_table(args.label_csv)
    ds = TestDirsDataset(study_dirs, label_table)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS)
    n_out = len(LABEL_COLUMNS)

    # 1) 테스트 폴더 수집 후
    print(f"[INFO] #studies in test set: {len(study_dirs)}")
    print("   sample 3:", study_dirs[:3])

    # 2) 첫 20개에 대해 라벨 lookup 시도
    miss = 0;
    pos = 0
    for p in study_dirs[:20]:
        try:
            sid, stid = parse_ids_from_path(p)
            y = label_table.get((sid, stid))
            if y is None:
                miss += 1
            elif any(y):
                pos += 1
        except Exception:
            miss += 1
    print(f"[QUICK CHECK] first20: matched={20 - miss}/20, any-pos={pos}/20")

    clients = parse_clients(args.clients)
    rows=[]
    for cid in clients:
        ckpt_path = cdir(cid)/"best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] skip client {cid:02d} (no best.pt)")
            continue

        # 1) Baseline
        base_model, mode = load_ckpt_as_model(str(ckpt_path), n_out)
        b_loss, b_f1_micro, b_f1_macro, b_macro_auc, b_aucs = evaluate_mmasked(base_model, dl)

        # 2) Updated (updated_heads.npz 주입)
        upd_model, _ = load_ckpt_as_model(str(ckpt_path), n_out)  # 동일 백본
        applied = maybe_apply_updated_heads(upd_model, cid, n_out)
        if not applied:
            print(f"[INFO] client_{cid:02d}: updated_heads.npz 없음 또는 shape 불일치 → baseline과 동일로 간주")
        u_loss, u_f1_micro, u_f1_macro, u_macro_auc, u_aucs = evaluate_mmasked(upd_model, dl)

        print(f"[Client {cid:02d} | {mode:11s}] "
              f"BASE(mAUC={b_macro_auc:.4f}) vs UPD(mAUC={u_macro_auc:.4f})  "
              f"ΔmAUC={u_macro_auc - b_macro_auc:+.4f}")

        # per-client JSON 저장
        outj = {
            "client_id": cid, "mode": mode,
            "baseline": {
                "loss": b_loss, "f1_micro": b_f1_micro, "f1_macro": b_f1_macro,
                "macro_auroc": b_macro_auc,
                "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, b_aucs)}
            },
            "updated": {
                "loss": u_loss, "f1_micro": u_f1_micro, "f1_macro": u_f1_macro,
                "macro_auroc": u_macro_auc,
                "per_class_auroc": {c: (None if (a!=a) else float(a)) for c,a in zip(LABEL_COLUMNS, u_aucs)}
            },
            "delta": {
                "loss": u_loss - b_loss,
                "f1_micro": u_f1_micro - b_f1_micro,
                "f1_macro": u_f1_macro - b_f1_macro,
                "macro_auroc": u_macro_auc - b_macro_auc
            }
        }
        with open(cdir(cid)/"test_compare_dirs.json", "w", encoding="utf-8") as f:
            json.dump(outj, f, indent=2, ensure_ascii=False)

        row = {
            "client_id": cid, "mode": mode,
            "baseline_loss": b_loss, "updated_loss": u_loss, "delta_loss": u_loss - b_loss,
            "baseline_f1_micro": b_f1_micro, "updated_f1_micro": u_f1_micro, "delta_f1_micro": u_f1_micro - b_f1_micro,
            "baseline_f1_macro": b_f1_macro, "updated_f1_macro": u_f1_macro, "delta_f1_macro": u_f1_macro - b_f1_macro,
            "baseline_macro_auroc": b_macro_auc, "updated_macro_auroc": u_macro_auc,
            "delta_macro_auroc": u_macro_auc - b_macro_auc
        }
        rows.append(row)

    # 요약 CSV
    if rows:
        headers = list(rows[0].keys())
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
        print(f"\n[INFO] Saved summary → {SUMMARY_CSV} ({len(rows)} rows)")
    else:
        print("[WARN] no evaluated clients; check paths/label_csv/updated_heads.npz")

if __name__ == "__main__":
    main()
