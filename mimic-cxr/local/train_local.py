# train_local.py
import os
import csv
import math
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from transformers import AutoTokenizer, AutoModel
from torchvision.models import MobileNet_V3_Small_Weights

# =========================
# 경로/설정
# =========================
CLIENT_CSV_DIR = r".\client_splits"
TEST_CSV_PATH  = r".\client_splits\test.csv"

# ===== 라벨 소스 선택 =====
USE_LABEL = "negbio"  # "negbio" 또는 "chexpert"
LABEL_CSV_NEGBIO  = r"C:\Users\user\PycharmProjects\Mimic-cxr\team329\mimic-cxr\mimic-cxr-2.0.0-negbio.csv"
LABEL_CSV_CHEXPERT= r"C:\Users\user\PycharmProjects\Mimic-cxr\team329\mimic-cxr\mimic-cxr-2.0.0-chexpert.csv"

LABEL_CSV = LABEL_CSV_NEGBIO if USE_LABEL=="negbio" else LABEL_CSV_CHEXPERT

# === 라벨 컬럼 ===
# 필요에 따라 원하는 라벨 집합으로 조정하세요.
LABEL_COLUMNS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Pleural Effusion",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
    # "No Finding"는 보통 제외
]

# 메타데이터(대표 이미지 선택용)
METADATA_CSV = r"C:\Users\user\PycharmProjects\Mimic-cxr\team329\mimic-cxr\mimic-cxr-2.0.0-metadata.csv"
IMG_ROOT = r"D:\mimic-cxr-jpg\mimic-cxr-jpg\2.1.0\files"

# 훈련 설정
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
NUM_WORKERS= 4  # Windows 안전
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)

# 텍스트 모델
TEXT_MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 256

# 이미지 전처리
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# =========================
# 메타데이터 기반 대표 이미지 선택
# =========================
def build_image_picker_from_metadata(metadata_csv: str, img_root: str):
    """
    metadata.csv에 'path' 컬럼이 있으면 그걸 사용.
    'ViewPosition' or 'view' 컬럼이 있으면 PA/AP(정면)를 우선 선택.
    반환: {(subject_id:int, study_id:int) -> 절대경로(str)}
    """
    import pandas as pd

    if not metadata_csv or not os.path.exists(metadata_csv):
        return {}

    df = pd.read_csv(metadata_csv)
    path_col = "path" if "path" in df.columns else None
    view_col = "ViewPosition" if "ViewPosition" in df.columns else ("view" if "view" in df.columns else None)

    df["subject_id"] = df["subject_id"].astype(int)
    df["study_id"]   = df["study_id"].astype(int)

    if view_col:
        df["_prio"] = df[view_col].fillna("").map(lambda v: 0 if v in ("PA","AP") else 1)
    else:
        df["_prio"] = 1

    df = df.sort_values(["subject_id","study_id","_prio"])
    pick = {}
    if path_col:
        for _, row in df.iterrows():
            key = (int(row["subject_id"]), int(row["study_id"]))
            if key in pick:
                continue
            abs_path = str(Path(img_root) / row[path_col])
            pick[key] = abs_path
    return pick

# =========================
# 라벨 로딩/매핑
# =========================
def load_label_table(label_csv_path: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd, time, shutil
    from pathlib import Path

    df = None
    last_err = None

    # 1) 최대 3번 재시도 (PermissionError면 대기 후 재시도, 그 외 에러도 재시도)
    for attempt in range(3):
        try:
            df = pd.read_csv(label_csv_path)
            break  # 성공
        except PermissionError as e:
            last_err = e
            print(f"[WARN] PermissionError on '{label_csv_path}' (attempt {attempt+1}/3). Retrying...")
            time.sleep(1.0)
        except Exception as e:
            last_err = e
            print(f"[WARN] Read failed (attempt {attempt+1}/3): {e}")
            time.sleep(0.5)

    # 2) 여전히 실패면 임시 복사본으로 시도
    if df is None:
        try:
            tmp_dir = Path("./_tmp"); tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / Path(label_csv_path).name
            print(f"[INFO] Trying temp copy -> {tmp_path}")
            shutil.copy2(label_csv_path, tmp_path)
            df = pd.read_csv(tmp_path)
        except Exception as e:
            raise RuntimeError(f"라벨 CSV를 읽지 못했습니다: {label_csv_path} ({last_err or e})")

    required = {"subject_id","study_id", *LABEL_COLUMNS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"라벨 CSV에 칼럼이 없습니다: {missing}")

    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)

    table = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        vec = [int(row[c]) for c in LABEL_COLUMNS]
        table[key] = vec
    return table


def id_to_int(subject_id: str, study_id: str) -> Tuple[int,int]:
    # 'p10000032' -> 10000032, 's50414267' -> 50414267
    return int(subject_id[1:]), int(study_id[1:])

# =========================
# 데이터셋
# csv_path(client_X.csv)에서 현재 모달리티에 맞는 행만 읽어온다.
# __getitem__:
# (subject_id, study_id)로 라벨 벡터(LABEL_COLUMNS 순서)를 만든다.
# 이미지:
# meta_picker에 대표 이미지가 있으면 그 파일을, 없으면 스터디 폴더의 첫 이미지를 로드
# PIL로 RGB 변환 → ImageNet mean/std로 정규화(MobileNet 사전학습과 호환)
# 텍스트:
# 해당 study의 .txt를 읽어 BERT 토크나이저로 input_ids, attention_mask 등 텐서 생성
# max_length=256으로 고정 길이 패딩/트렁케이션
# 반환 예시(멀티모달):
# {"labels": y, "image": image_tensor, "text": {"input_ids":..., "attention_mask":...}, "subject_id":..., "study_id":...}
# (image_only/text_only는 해당 키만 포함)
# 변경 포인트:
# 이미지 해상도/크롭/증강(aug) 바꾸려면 IMG_TRANSFORM 수정
# 텍스트 길이/토크나이저 모델 바꾸려면 TEXT_MODEL_NAME, MAX_LEN 수정
# =========================
class ClientDataset(Dataset):
    def __init__(self, csv_path: str, label_table: Dict[Tuple[int,int], List[float]], mode: str, meta_picker=None):
        """
        mode: 'multimodal', 'image_only', 'text_only', 'test_mix'
        """
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["modality"] != mode and not (mode=="test_mix" and row["modality"]=="test_mix"):
                    continue
                self.rows.append(row)

        self.label_table = label_table
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.meta_picker = meta_picker

    def __len__(self):
        return len(self.rows)

    def _load_image(self, image_dir: str, subject_id: str=None, study_id: str=None) -> torch.Tensor:
        if self.meta_picker and subject_id is not None and study_id is not None:
            key = (int(subject_id[1:]), int(study_id[1:]))
            meta_choice = self.meta_picker.get(key)
            if meta_choice and os.path.exists(meta_choice):
                img = Image.open(meta_choice).convert("RGB")
                return IMG_TRANSFORM(img)
        # fallback: 스터디 폴더 첫 이미지
        p = Path(image_dir)
        imgs = sorted([*p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.png")])
        if not imgs:
            return torch.zeros(3,224,224)
        img = Image.open(imgs[0]).convert("RGB")
        return IMG_TRANSFORM(img)

    def _load_text(self, text_path: str) -> Dict[str, torch.Tensor]:
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            text = ""
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        return {k: v.squeeze(0) for k,v in enc.items()}

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid_int, stid_int = id_to_int(r["subject_id"], r["study_id"])
        y = torch.tensor(self.label_table.get((sid_int, stid_int), [0]*len(LABEL_COLUMNS)), dtype=torch.float)

        sample = {"labels": y, "subject_id": r["subject_id"], "study_id": r["study_id"]}

        if self.mode in ["multimodal","image_only","test_mix"]:
            sample["image"] = self._load_image(r["image_dir"].strip(), r["subject_id"], r["study_id"])
        if self.mode in ["multimodal","text_only","test_mix"]:
            sample["text"] = self._load_text(r["text_path"].strip())

        return sample

# =========================
# 모델
# =========================
class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        base = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        in_f = base.classifier[0].in_features
        base.classifier = nn.Identity()
        self.backbone = base
        self.head = nn.Linear(in_f, n_out)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

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
        logit_img = self.img(image)
        logit_txt = self.txt(**text)
        return 0.5 * (logit_img + logit_txt)

# =========================
# 학습/평가 루프
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        labels = batch["labels"].to(device)
        if "image" in batch and "text" in batch:
            image = batch["image"].to(device)
            text = {k: v.to(device) for k,v in batch["text"].items()}
            logits = model(image=image, text=text)
        elif "image" in batch:
            image = batch["image"].to(device)
            logits = model(image)
        else:
            text = {k: v.to(device) for k,v in batch["text"].items()}
            logits = model(**text)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        labels = batch["labels"].to(device)
        if "image" in batch and "text" in batch:
            image = batch["image"].to(device)
            text = {k: v.to(device) for k,v in batch["text"].items()}
            logits = model(image=image, text=text)
        elif "image" in batch:
            image = batch["image"].to(device)
            logits = model(image)
        else:
            text = {k: v.to(device) for k,v in batch["text"].items()}
            logits = model(**text)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

# =========================
# 진입점: 특정 client_id 학습
# =========================
def run_client(client_id: int):
    import random
    import numpy as np

    # ================== [DEBUG] 로그 추가 시작 ==================
    print(f"[DEBUG] run_client 시작 (client_id: {client_id})")

    # 재현성
    torch.manual_seed(42);
    random.seed(42);
    np.random.seed(42)

    try:
        print(f"[DEBUG] 라벨 테이블 로딩 시작: {LABEL_CSV}")
        label_table = load_label_table(LABEL_CSV)
        print("[DEBUG] 라벨 테이블 로딩 완료")

        print(f"[DEBUG] 메타데이터 로더 생성 시작: {METADATA_CSV}")
        meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)
        print("[DEBUG] 메타데이터 로더 생성 완료")

        if client_id == 0:
            csv_path = TEST_CSV_PATH
        else:
            csv_path = os.path.join(CLIENT_CSV_DIR, f"client_{client_id:02d}.csv")
        print(f"[DEBUG] 클라이언트 CSV 경로: {csv_path}")

        # 모달리티 결정
        if 1 <= client_id <= 16:
            mode = "multimodal"
        elif client_id in [17, 18]:
            mode = "image_only"
        elif client_id in [19, 20]:
            mode = "text_only"
        elif client_id == 0:
            mode = "test_mix"
        else:
            raise ValueError("invalid client_id")
        print(f"[DEBUG] 클라이언트 모드: {mode}")

        print("[DEBUG] ClientDataset 생성 시작...")
        dataset = ClientDataset(csv_path, label_table, mode=mode, meta_picker=meta_picker)
        print(f"[DEBUG] ClientDataset 생성 완료 (데이터 수: {len(dataset)})")

        # 모델 구성
        n_out = len(LABEL_COLUMNS)
        print(f"[DEBUG] 모델 생성 시작 (mode: {mode}, 출력 클래스 수: {n_out})")
        if mode == "multimodal":
            model = MultiModalLateFusion(n_out)
        elif mode == "image_only":
            model = ImageHead(n_out)
        else:
            model = TextHead(n_out)
        print("[DEBUG] 모델 생성 완료")

        print(f"[DEBUG] 모델을 디바이스로 이동: {DEVICE}")
        model.to(DEVICE)
        print("[DEBUG] 모델 이동 완료")

        # 데이터 분할 (train/val 90/10)
        n = len(dataset)
        n_train = int(n * 0.9)
        n_val = n - n_train
        print(f"[DEBUG] 데이터 분할 시작 (train: {n_train}, val: {n_val})")
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val],
                                                           generator=torch.Generator().manual_seed(42))
        print("[DEBUG] 데이터 분할 완료")

        print("[DEBUG] DataLoader 생성 시작...")
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print("[DEBUG] DataLoader 생성 완료")

        # 손실 함수 및 옵티마이저
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        best_val = math.inf
        save_dir = Path(f"./outputs/client_{client_id:02d}")
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] 결과 저장 디렉토리: {save_dir}")

        print("\n[DEBUG] <<<<<<<<<< 훈련 루프 시작 >>>>>>>>>>")
        for epoch in range(1, EPOCHS + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss = evaluate(model, val_loader, criterion, DEVICE)
            print(f"[Client {client_id:02d}] Epoch {epoch}/{EPOCHS} | train {tr_loss:.4f} | val {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                print(f"[DEBUG] 최고 성능 갱신. 모델 저장 시작 -> {save_dir / 'best.pt'}")
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "label_columns": LABEL_COLUMNS,
                    "mode": mode,
                }, save_dir / "best.pt")
                print("[DEBUG] 모델 저장 완료")

        print("[DEBUG] <<<<<<<<<< 훈련 루프 종료 >>>>>>>>>>")

    except Exception as e:
        # 에러가 발생하면 여기서 잡아서 좀 더 상세히 출력
        import traceback
        print(f"\n[FATAL ERROR] run_client (client_id: {client_id})에서 심각한 오류 발생!")
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        print("--- Traceback ---")
        traceback.print_exc()
        print("-----------------")
        # 에러를 다시 발생시켜 상위 로직에서 처리하게 함
        raise e

    print(f"[Client {client_id:02d}] Done. Best val loss: {best_val:.4f}  (saved at {save_dir / 'best.pt'})")


# ================== [DEBUG] 로그 추가 종료 ==================

def run_many(client_ids: List[int]):
    for cid in client_ids:
        print("\n" + "="*70)
        print(f"▶ Start training client {cid:02d}")
        print("="*70)
        try:
            run_client(cid)
        except Exception as e:
            print(f"[ERROR] client {cid:02d} failed: {e}")
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_id",
        type=str,
        default="all",
        help="단일 클라 ID(1~20), 'all'이면 1~20 전부. 0은 test_mix 로더."
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="건너뛸 클라 ID CSV (예: '3,7,12')"
    )
    args = parser.parse_args()

    # 파싱
    if args.client_id.lower() == "all" or args.client_id == "":
        client_ids = list(range(1, 21))
    else:
        # 단일 ID 또는 범위 문자열 지원 (예: "5" 또는 "1-16")
        s = args.client_id
        if "-" in s:
            a, b = s.split("-", 1)
            client_ids = list(range(int(a), int(b) + 1))
        else:
            client_ids = [int(s)]

    # 스킵 적용
    if args.skip.strip():
        skip_set = {int(x) for x in args.skip.split(",") if x.strip().isdigit()}
        client_ids = [cid for cid in client_ids if cid not in skip_set]

    # 실행
    if client_ids == [0]:
        run_client(0)   # test_mix 전용
    else:
        run_many(client_ids)
