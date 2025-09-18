# -*- coding: utf-8 -*-
"""
repr_kd.py
- prep_clients가 생성한 임베딩(repr_img.npy / repr_txt.npy)을 사용해
  그룹별(멀티모달, 이미지온리, 텍스트온리) 지식증류(KD)를 벡터 공간에서 수행.
- 교사벡터 풀을 KMeans 프로토타입으로 요약 → 학생벡터를 프로토타입 쪽으로 α-블렌딩.

그룹 규칙:
  * 멀티모달(1~16): test loss 오름차순 정렬 → s1(상위5, 교사) / s2(다음5, 유지) / s3(하위6, 학생)
      - s1 → s3 로 KD (이미지/텍스트 각각)
      ** 일단 이렇게 실험 진행. (이후에 그룹을 두 개로만 나눌 수도 있음)
  * 이미지온리(17~18): 낮은 loss = teacher, 높은 loss = student → s1 → s2
  * 텍스트온리(19~20): 낮은 loss = teacher, 높은 loss = student → s1 → s2

입출력 파일:
  * 입력(필수): ./outputs/client_{cid}/repr_img.npy / repr_txt.npy  (모달리티에 따라 존재)
  * 입력(선택): ./eval_results/summary.csv (열: client_id, loss, ...)
               ./outputs/client_{cid}/client_{cid}_metrics.json (f1_macro 등)
  * 출력:       ./outputs/client_{cid}/repr_img_kd.npy / repr_txt_kd.npy (학생만)
               ./global_outputs/kd_report.csv, kd_groups.json

하이퍼:
  ALPHA=0.5, K_TEACH=32 (필요시 CLI로 오버라이드)
"""

from __future__ import annotations
import os, json, csv, math, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

# -------------------
# 설정/상수
# -------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUTPUTS_DIR = Path("./outputs")
GLOBAL_OUT  = Path("./global_outputs"); GLOBAL_OUT.mkdir(parents=True, exist_ok=True)
EVAL_SUMMARY = Path("./eval_results/summary.csv")

MM_CLIENTS  = list(range(1, 17))   # 멀티모달
IMG_ONLY    = [17, 18]
TXT_ONLY    = [19, 20]

DEFAULT_ALPHA  = 0.5
DEFAULT_KTEACH = 32

# -------------------
# 유틸
# -------------------
def client_dir(cid: int) -> Path:
    d = OUTPUTS_DIR / f"client_{cid:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_summary_losses(summary_csv: Path) -> Dict[int, float]:
    """
    eval_results/summary.csv에서 client별 test loss 읽기.
    없거나 파싱 실패 시 빈 dict 반환.
    """
    if not summary_csv.exists(): return {}
    out: Dict[int, float] = {}
    with open(summary_csv, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["client_id"])
                loss = float(row["loss"])
                out[cid] = loss
            except Exception:
                continue
    return out

def _read_client_metric_json(cid: int) -> Optional[Dict]:
    p = client_dir(cid) / f"client_{cid}_metrics.json"
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def _metric_for_grouping(cid: int, losses: Dict[int, float]) -> Tuple[float, str]:
    """
    그룹핑 스코어 반환. 우선순위:
      1) test loss (낮을수록 좋음)  -> ('loss', value)
      2) f1_macro (높을수록 좋음)  -> ('f1_macro', 1 - f1) 로 변환해 낮을수록 좋은 형태로 정렬
    """
    if cid in losses and not math.isnan(losses[cid]):
        return float(losses[cid]), "loss"
    j = _read_client_metric_json(cid)
    if j and isinstance(j.get("f1_macro", None), (int, float)):
        f1 = float(j["f1_macro"])
        return (1.0 - f1), "f1_macro"  # 낮을수록 좋도록 변환
    # fallback: 큰 값(나쁨)으로 반환
    return float("inf"), "none"

def _load_reprs(cid: int, prefer_kd: bool=False, max_samples: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    repr 파일 로드. prefer_kd=True이면 *_kd.npy 우선 사용.
    반환: (Xi, Xt) 없는 경우 shape=(0,0) 빈 배열
    """
    base = client_dir(cid)
    def _load_one(name_main: str, name_kd: str) -> np.ndarray:
        path = base / (name_kd if prefer_kd and (base / name_kd).exists() else name_main)
        if not path.exists(): return np.zeros((0, 0), dtype=np.float32)
        x = np.load(path)
        if x.ndim != 2:  # 안전장치
            x = x.reshape(x.shape[0], -1)
        if max_samples and x.shape[0] > max_samples:
            idx = np.random.RandomState(SEED).choice(x.shape[0], size=max_samples, replace=False)
            x = x[idx]
        return x.astype(np.float32, copy=False)

    Xi = _load_one("repr_img.npy", "repr_img_kd.npy")
    Xt = _load_one("repr_txt.npy", "repr_txt_kd.npy")
    return Xi, Xt

def _save_kd_reprs(cid: int, Xi_kd: Optional[np.ndarray], Xt_kd: Optional[np.ndarray]):
    base = client_dir(cid)
    if Xi_kd is not None and Xi_kd.size > 0:
        np.save(base / "repr_img_kd.npy", Xi_kd.astype(np.float32))
    if Xt_kd is not None and Xt_kd.size > 0:
        np.save(base / "repr_txt_kd.npy", Xt_kd.astype(np.float32))

def _kmeans_centers(X: np.ndarray, k: int) -> np.ndarray:
    if X is None or X.size == 0: 
        return np.zeros((0, 0), dtype=np.float32)
    # 중복 제거 후 k 조정
    Xu = np.unique(X, axis=0)
    k  = max(1, min(k, Xu.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(Xu)
    return km.cluster_centers_.astype(np.float32)

def _cosine_nearest_blend(X: np.ndarray, P: np.ndarray, alpha: float) -> np.ndarray:
    if X is None or X.size == 0 or P is None or P.size == 0:
        return X
    # cosine 유사도 근접 프로토타입
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    idx = np.argmax(Xn @ Pn.T, axis=1)  # 가장 유사한 프로토타입 index
    return (1.0 - alpha) * X + alpha * P[idx]

def _write_groups_json(groups: Dict):
    (GLOBAL_OUT / "kd_groups.json").write_text(json.dumps(groups, indent=2, ensure_ascii=False), encoding="utf-8")

def _append_report(rows: List[Dict]):
    rep = GLOBAL_OUT / "kd_report.csv"
    write_header = not rep.exists()
    with open(rep, "a", newline="", encoding="utf-8-sig") as f:
        cols = ["modality","teacher_clients","student_client","n_vec_img","n_vec_txt","alpha","K_teacher"]
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)

# -------------------
# 그룹핑
# -------------------
def group_clients_by_loss() -> Dict:
    """
    멀티모달: s1(상위5) / s2(다음5) / s3(하위6)  (loss 오름차순)
    이미지/텍스트: 낮은 loss = s1(teacher), 높은 loss = s2(student)
    """
    losses = _read_summary_losses(EVAL_SUMMARY)

    # 멀티모달 정렬
    scored = []
    for c in MM_CLIENTS:
        v, src = _metric_for_grouping(c, losses)
        scored.append((c, v, src))
    # v(낮을수록 좋음) 오름차순
    scored.sort(key=lambda t: (t[1], t[0]))
    s1 = [c for c,_,_ in scored[:5]]
    s2 = [c for c,_,_ in scored[5:10]]
    s3 = [c for c,_,_ in scored[10:16]]

    # 이미지 2명
    img_vals = []
    for c in IMG_ONLY:
        v,_ = _metric_for_grouping(c, losses)
        img_vals.append((c, v))
    img_vals.sort(key=lambda t: (t[1], t[0]))  # 낮을수록 좋음
    img_s1, img_s2 = img_vals[0][0], img_vals[1][0]

    # 텍스트 2명
    txt_vals = []
    for c in TXT_ONLY:
        v,_ = _metric_for_grouping(c, losses)
        txt_vals.append((c, v))
    txt_vals.sort(key=lambda t: (t[1], t[0]))
    txt_s1, txt_s2 = txt_vals[0][0], txt_vals[1][0]

    groups = {
        "mm":  {"s1": s1, "s2": s2, "s3": s3},
        "img": {"s1": img_s1, "s2": img_s2},
        "txt": {"s1": txt_s1, "s2": txt_s2},
        "source_note": "loss from eval_results/summary.csv if present; fallback to (1-f1_macro) from client_metrics.json",
    }
    _write_groups_json(groups)
    print("[Groups] mm s1:", s1, " s2:", s2, " s3:", s3)
    print("[Groups] img s1->s2:", img_s1, "->", img_s2)
    print("[Groups] txt s1->s2:", txt_s1, "->", txt_s2)
    return groups

# -------------------
# KD 파이프라인
# -------------------
def kd_multimodal(groups: Dict, alpha: float, kteach: int, max_teacher_samples: int = 10000):
    rows: List[Dict] = []
    # 교사 풀(멀티모달 s1)
    XTi_img, XTi_txt = [], []
    for c in groups["mm"]["s1"]:
        Xi, Xt = _load_reprs(c, prefer_kd=True, max_samples=max_teacher_samples)
        if Xi.size: XTi_img.append(Xi)
        if Xt.size: XTi_txt.append(Xt)
    P_img = _kmeans_centers(np.vstack(XTi_img) if XTi_img else np.zeros((0,0),np.float32), kteach)
    P_txt = _kmeans_centers(np.vstack(XTi_txt) if XTi_txt else np.zeros((0,0),np.float32), kteach)

    # 학생: s3만 갱신
    for c in groups["mm"]["s3"]:
        Xi, Xt = _load_reprs(c, prefer_kd=False)
        Xi_kd = _cosine_nearest_blend(Xi, P_img, alpha) if Xi.size else None
        Xt_kd = _cosine_nearest_blend(Xt, P_txt, alpha) if Xt.size else None
        _save_kd_reprs(c, Xi_kd, Xt_kd)
        rows.append({
            "modality": "mm",
            "teacher_clients": ",".join(map(lambda x: f"{x:02d}", groups["mm"]["s1"])),
            "student_client": f"{c:02d}",
            "n_vec_img": int(Xi.shape[0]) if Xi.size else 0,
            "n_vec_txt": int(Xt.shape[0]) if Xt.size else 0,
            "alpha": alpha, "K_teacher": kteach
        })
    _append_report(rows)

def kd_image_only(groups: Dict, alpha: float, kteach: int, max_teacher_samples: int = 10000):
    rows: List[Dict] = []
    cT, cS = groups["img"]["s1"], groups["img"]["s2"]
    XiT, _ = _load_reprs(cT, prefer_kd=True, max_samples=max_teacher_samples)
    P = _kmeans_centers(XiT, kteach)
    XiS, _ = _load_reprs(cS, prefer_kd=False)
    XiS_kd = _cosine_nearest_blend(XiS, P, alpha) if XiS.size else None
    _save_kd_reprs(cS, XiS_kd, None)
    rows.append({
        "modality": "img",
        "teacher_clients": f"{cT:02d}",
        "student_client": f"{cS:02d}",
        "n_vec_img": int(XiS.shape[0]) if XiS.size else 0,
        "n_vec_txt": 0,
        "alpha": alpha, "K_teacher": kteach
    })
    _append_report(rows)

def kd_text_only(groups: Dict, alpha: float, kteach: int, max_teacher_samples: int = 10000):
    rows: List[Dict] = []
    cT, cS = groups["txt"]["s1"], groups["txt"]["s2"]
    _, XtT = _load_reprs(cT, prefer_kd=True, max_samples=max_teacher_samples)
    P = _kmeans_centers(XtT, kteach)
    _, XtS = _load_reprs(cS, prefer_kd=False)
    XtS_kd = _cosine_nearest_blend(XtS, P, alpha) if XtS.size else None
    _save_kd_reprs(cS, None, XtS_kd)
    rows.append({
        "modality": "txt",
        "teacher_clients": f"{cT:02d}",
        "student_client": f"{cS:02d}",
        "n_vec_img": 0,
        "n_vec_txt": int(XtS.shape[0]) if XtS.size else 0,
        "alpha": alpha, "K_teacher": kteach
    })
    _append_report(rows)

# -------------------
# 엔트리
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="KD blend weight: x'=(1-a)x + a*p")
    ap.add_argument("--kteach", type=int, default=DEFAULT_KTEACH, help="#teacher prototypes (KMeans)")
    ap.add_argument("--max_teacher_samples", type=int, default=10000, help="teacher pool 샘플 최대 수")
    args = ap.parse_args()

    groups = group_clients_by_loss()
    kd_multimodal(groups, alpha=args.alpha, kteach=args.kteach, max_teacher_samples=args.max_teacher_samples)
    kd_image_only(groups, alpha=args.alpha, kteach=args.kteach, max_teacher_samples=args.max_teacher_samples)
    kd_text_only(groups,  alpha=args.alpha, kteach=args.kteach, max_teacher_samples=args.max_teacher_samples)

    print("[KD] representation KD done. (repr_*_kd.npy written for students)")
    print(f"[KD] report → {GLOBAL_OUT/'kd_report.csv'}")
    print(f"[KD] groups → {GLOBAL_OUT/'kd_groups.json'}")

if __name__ == "__main__":
    main()
