# D:\ROCO\prep\config_roco.py
import os

# ===== 경로 =====
ROOT = r"C:\Users\user\PycharmProjects\ROCO\archive\rocov2"
TRAIN_IMG_DIR = os.path.join(ROOT, "train_images", "train")
TEST_IMG_DIR  = os.path.join(ROOT, "test_images", "test")

TRAIN_CAP_CSV   = os.path.join(ROOT, "train_captions.csv")
VALID_CAP_CSV   = os.path.join(ROOT, "valid_captions.csv")
TEST_CAP_CSV    = os.path.join(ROOT, "test_captions.csv")
TRAIN_CONC_CSV  = os.path.join(ROOT, "train_concepts.csv")
VALID_CONC_CSV  = os.path.join(ROOT, "valid_concepts.csv")
TEST_CONC_CSV   = os.path.join(ROOT, "test_concepts.csv")
CUI_MAP_CSV = os.path.join(ROOT, "cui_mapping_diseases.csv")
VALID_IMG_DIR = os.path.join(ROOT, "valid_images", "valid")

OUT_ROOT = os.path.join(ROOT, "_prepared")
os.makedirs(OUT_ROOT, exist_ok=True)

# ===== 전역 split =====
GLOBAL_VAL_RATIO  = 0.10
GLOBAL_TEST_RATIO = 0.10
RANDOM_STATE = 42

# 라벨 소스: 'concept' or 'modality'
LABEL_SOURCE = 'concept'

# ===== 분배(클라이언트) =====
NUM_CLIENTS = 10
CLIENTS = list(range(NUM_CLIENTS))
MODALITY = {cid: ("both" if cid <= 5 else "image" if cid <= 7 else "text") for cid in CLIENTS}

TARGET_PER_CLIENT_MIN = 5000
TARGET_PER_CLIENT_MAX = 10000
LOCAL_VAL_RATIO = 0.10
WITH_REPLACEMENT = True  # 중복 허용
