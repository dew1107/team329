from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Config:
    BASE_DIR: str = r"C:\Users\user\PycharmProjects\ROCO\archive\rocov2\_prepared"
    OUT_GLOBAL_DIR: str = r"C:\Users\user\PyCharmProjects\ROCO\archive\rocov2\global_train\out_global"

    FUSION_CLIENTS: Tuple[int, ...] = tuple(range(0, 26))
    IMAGE_ONLY: Tuple[int, ...] = (26, 27)
    TEXT_ONLY:  Tuple[int, ...] = (28, 29)

    NUM_CLASSES: int = 230

    CKPT_NAME: str = "client_{cid}_local_fusion_best.pt"

    IMG_DIM: int = 256
    TXT_DIM: int = 256
    FUSED_DIM: int = 704
    D_MODEL: int = 256

    SAMPLE_PER_CLIENT: int = 1024
    K_IMG: int = 8
    K_TXT: int = 8
    KMEANS_N_INIT: str | int = "auto"

    METRIC_NAME: str = "f1_macro"
    KD_REP_WEIGHT: float = 0.2
    KD_TEMP: float = 2.0

    METRICS_SNAPSHOT: str = "metrics_snapshot.json"
    KD_PLAN_JSON: str = "kd_plan.json"
    SEED: int = 42

cfg = Config()
