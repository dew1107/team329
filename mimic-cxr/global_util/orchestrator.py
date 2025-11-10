# -*- coding: utf-8 -*-
"""
orchestrator.py  (self-contained, project-adapted)
- prep_clients.py 가 만든 repr_*.npy (또는 *_kd.npy)를 모아 글로벌 벡터 Z 생성
- eval_results/summary.csv 의 metric (macro_auroc 또는 loss)로 그룹/간선 구성
- 각 클라에 payload(JSON)로 Z 경로 및 역할(teacher/student/solo) 배포

사용 예:
  python -u orchestrator.py --metric macro_auroc --K_img 6 --K_txt 6 --d_model 256
  (loss를 쓰려면 --metric loss)
"""

from __future__ import annotations
import os, csv, json, math, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# 기본 설정
# -------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUTPUTS_DIR = Path("./outputs")
GLOBAL_DIR  = Path("./global_outputs"); GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = Path("./eval_results/summary.csv")

FUSION_CLIENTS = list(range(1, 17))  # 1..16 멀티모달
IMAGE_ONLY     = [17, 18]
TEXT_ONLY      = [19, 20]

# -------------------------------
# 유틸
# -------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def cdir(cid: int) -> Path:
    d = OUTPUTS_DIR / f"client_{cid:02d}"
    ensure_dir(d); return d

def _read_summary(summary_csv: Path, metric: str) -> Dict[int, float]:
    """
    eval_results/summary.csv 에서 metric 열 읽기
      - metric='macro_auroc' (높을수록 좋음) 또는 'loss' (낮을수록 좋음)
    """
    if not summary_csv.exists():
        print(f"[WARN] summary.csv not found: {summary_csv}")
        return {}
    out = {}
    with open(summary_csv, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["client_id"])
                val = float(row[metric])
                out[cid] = val
            except Exception:
                pass
    return out

def _load_reprs_for_client(cid: int, prefer_kd: bool = True, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    repr 파일 로드 (존재 시 *_kd.npy 우선)
    반환 (Xi, Xt) : 없으면 (0,0) shape
    """
    base = cdir(cid)
    def _load(name_main: str, name_kd: str) -> np.ndarray:
        p = base / (name_kd if prefer_kd and (base / name_kd).exists() else name_main)
        if not p.exists():
            return np.zeros((0, 0), dtype=np.float32)
        x = np.load(p)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if max_samples and x.shape[0] > max_samples:
            idx = np.random.RandomState(SEED).choice(x.shape[0], size=max_samples, replace=False)
            x = x[idx]
        # L2 정규화 보정
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return (x / n).astype(np.float32)

    Xi = _load("repr_img.npy", "repr_img_kd.npy")
    Xt = _load("repr_txt.npy", "repr_txt_kd.npy")
    return Xi, Xt

def stack_reps(cids: List[int], split: str = "train", max_samples_per_client: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    멀티모달 클라 집합에서 이미지/텍스트 임베딩 수집 후 스택
    """
    Xi_all, Xt_all = [], []
    for cid in cids:
        Xi, Xt = _load_reprs_for_client(cid, prefer_kd=True, max_samples=max_samples_per_client)
        if Xi.size: Xi_all.append(Xi)
        if Xt.size: Xt_all.append(Xt)
    Xi_all = np.vstack(Xi_all) if Xi_all else np.zeros((0, 0), dtype=np.float32)
    Xt_all = np.vstack(Xt_all) if Xt_all else np.zeros((0, 0), dtype=np.float32)
    return Xi_all, Xt_all

def kmeans_centroids(X: np.ndarray, k: int) -> np.ndarray:
    if X is None or X.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    Xu = np.unique(X, axis=0)
    k  = max(1, min(k, Xu.shape[0]))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    km.fit(Xu)
    C = km.cluster_centers_.astype(np.float32)
    # L2 정규화
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    return C

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A_ = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_ = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.clip(A_ @ B_.T, -1.0, 1.0).astype(np.float32)

def hungarian_pairs(S: np.ndarray) -> List[Tuple[int,int]]:
    """
    유사도 행렬 S (I x J) → 비용 = 1-S 로 최소합 매칭
    scipy가 없으면 그리디 fallback
    """
    I, J = S.shape
    K = min(I, J)
    try:
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - S
        ri, cj = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(ri[:K], cj[:K])]
        return pairs
    except Exception:
        # greedy fallback
        pairs, used_i, used_j = [], set(), set()
        S_flat = [(-S[i, j], i, j) for i in range(I) for j in range(J)]
        S_flat.sort()
        for _, i, j in S_flat:
            if i in used_i or j in used_j: continue
            pairs.append((i, j)); used_i.add(i); used_j.add(j)
            if len(pairs) >= K: break
        return pairs

def project_to_dim(X: np.ndarray, d: int) -> np.ndarray:
    """
    PCA로 d 차원으로 투영 (샘플 부족 시 zero-pad/trim)
    """
    if X.size == 0:
        return np.zeros((0, d), dtype=np.float32)
    D = X.shape[1]
    if d == D:
        return X.astype(np.float32)
    if X.shape[0] >= 2 and D > 1:
        pca = PCA(n_components=min(d, D), random_state=SEED)
        Y = pca.fit_transform(X).astype(np.float32)
    else:
        Y = X.astype(np.float32)
    if Y.shape[1] < d:
        pad = np.zeros((Y.shape[0], d - Y.shape[1]), dtype=np.float32)
        Y = np.concatenate([Y, pad], axis=1)
    elif Y.shape[1] > d:
        Y = Y[:, :d]
    return Y

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V) -> np.ndarray:
    """ NumPy 기반의 단순 스케일드 닷 프로덕트 어텐션 """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V
    return output.astype(np.float32)

def build_global_vectors(img_cent: np.ndarray, txt_cent: np.ndarray, d_model: int) -> Dict[str, np.ndarray]:
    """
    1) 이미지/텍스트 센트로이드를 코사인 유사도 기반으로 매칭
    2) 공통 차원 d_model로 각각 투영
    3) Z_mm: (img_proj + txt_proj)/2
       Z_img2txt: txt_proj
       Z_txt2img: img_proj
    """
    if img_cent.size == 0 or txt_cent.size == 0:
        return {"Z_mm": np.zeros((0, d_model), np.float32),
                "Z_img2txt": np.zeros((0, d_model), np.float32),
                "Z_txt2img": np.zeros((0, d_model), np.float32)}

    # 매칭 전에 같은 차원이 되도록 투영
    img_proj_all = project_to_dim(img_cent, d_model) # (K_img, d_model)
    txt_proj_all = project_to_dim(txt_cent, d_model)

    S = cosine_matrix(img_proj_all, txt_proj_all)   # I x J
    pairs = hungarian_pairs(S)              # 리스트[(i,j)]
    I = len(pairs)

    img_proj_sel = np.stack([img_proj_all[i] for i, _ in pairs], axis=0)
    txt_proj_sel = np.stack([txt_proj_all[j] for _, j in pairs], axis=0)

    # Z_mm      = 0.5 * (img_proj_sel + txt_proj_sel)
    # Z_img2txt = txt_proj_sel.copy()
    # Z_txt2img = img_proj_sel.copy()
    # Z_img2txt: 이미지(Q)가 텍스트(K,V)를 참조
    # Z_img2txt = scaled_dot_product_attention(
    #     Q=img_proj_all, K=txt_proj_all, V=txt_proj_all
    # )
    #
    # # Z_txt2img: 텍스트(Q)가 이미지(K,V)를 참조
    # Z_txt2img = scaled_dot_product_attention(
    #     Q=txt_proj_all, K=img_proj_all, V=img_proj_all
    # )
    #  주의: Z_mm의 최적 퓨전 방식은 실험이 필요
    #    가장 간단한 Z_mm은 Z_img2txt를 재사용하는 것
    #    (혹은 Z_txt2img를 사용. 텍스트가 중요하다면?)
    #    일단 "텍스트가 중요" 의견을 반영해 Z_txt2img를 Z_mm으로 사용

    # Z_img2txt (이미지 Q, 텍스트 K/V)
    Z_img2txt_fused = scaled_dot_product_attention(
        Q=img_proj_all, K=txt_proj_all, V=txt_proj_all
    )  # Shape: (K_img, D)

    # Z_txt2img (텍스트 Q, 이미지 K/V)
    Z_txt2img_fused = scaled_dot_product_attention(
        Q=txt_proj_all, K=img_proj_all, V=img_proj_all
    )  # Shape: (K_txt, D)

    # 3. Z 벡터 할당
    # Z_mm은 img(Q)가 txt(K,V)를 융합한 벡터
    Z_mm = Z_img2txt_fused.copy()
    # 이미지 전용 클라이언트가 쓸 텍스트 프록시
    Z_img2txt = Z_img2txt_fused.copy()

    # 텍스트 전용 클라이언트가 쓸 이미지 프록시
    Z_txt2img = Z_txt2img_fused.copy()

    # 저장: 페어링 CSV
    with open(GLOBAL_DIR / "cluster_pairing.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["img_cluster_idx", "txt_cluster_idx", "cosine"])
        for (i, j) in pairs:
            w.writerow([i, j, float(S[i, j])])

    return {"Z_mm": Z_mm.astype(np.float32),
            "Z_img2txt": Z_img2txt.astype(np.float32),
            "Z_txt2img": Z_txt2img.astype(np.float32)}

def save_payload_for_client(cid: int, payload: Dict):
    (cdir(cid) / "global_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

# -------------------------------
# 그룹/간선: 멀티모달 High/Low, img/txt 전용 1:1
# -------------------------------
def split_median(metrics: Dict[int, float], cids: List[int], higher_is_better: bool) -> Tuple[List[int], List[int]]:
    vals = [metrics.get(c, np.nan) for c in cids]
    xs   = [v for v in vals if not np.isnan(v)]
    if not xs: return [], []
    med = float(np.median(np.array(xs)))
    if higher_is_better:
        high = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] >  med]
        low  = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] <= med]
    else:
        # 낮을수록 좋으면 반대
        high = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] <  med]
        low  = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] >= med]
    return high, low

def best_pair(a: int, b: int, metrics: Dict[int, float], higher_is_better: bool) -> Tuple[int, int]:
    va, vb = metrics.get(a, np.nan), metrics.get(b, np.nan)
    if np.isnan(va) and np.isnan(vb): return (a, b)  # 디폴트
    if np.isnan(vb): return (a, b)
    if np.isnan(va): return (b, a)
    if higher_is_better:
        return (a, b) if va >= vb else (b, a)
    else:
        return (a, b) if va <= vb else (b, a)

def build_fusion_kd_edges(high: List[int], low: List[int], metrics: Dict[int, float], k_per_student: int = 2, higher_is_better: bool = True) -> List[Tuple[int, int]]:
    key = (lambda c: metrics.get(c, float("-inf"))) if higher_is_better else (lambda c: -metrics.get(c, float("inf")))
    high_sorted = sorted(high, key=key, reverse=True)
    edges = []
    for s in low:
        teachers = high_sorted[:k_per_student] if len(high_sorted) >= k_per_student else high_sorted
        for t in teachers:
            edges.append((t, s))
    return edges

# -------------------------------
# 메인
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", type=str, default="macro_auroc", choices=["macro_auroc","loss"],
                    help="클라이언트 성능 구분에 사용할 지표 (macro_auroc: 높을수록 좋음, loss: 낮을수록 좋음)")
    ap.add_argument("--K_img", type=int, default=4)
    ap.add_argument("--K_txt", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--sample_per_client", type=int, default=20000)
    ap.add_argument("--k_per_student", type=int, default=2)
    args = ap.parse_args()

    set_seed(SEED)

    # 1) 메트릭 로드
    metrics = _read_summary(SUMMARY_CSV, metric=args.metric)
    higher_is_better = (args.metric != "loss")
    plan = {
        "metric_name": args.metric,
        "higher_is_better": higher_is_better,
        "K_img": args.K_img, "K_txt": args.K_txt, "d_model": args.d_model,
        "sample_per_client": args.sample_per_client, "k_per_student": args.k_per_student,
    }

    # 2) 멀티모달 High/Low
    fusion_high, fusion_low = split_median(metrics, FUSION_CLIENTS, higher_is_better)

    # 3) KD 간선
    kd_edges = {
        "fusion": build_fusion_kd_edges(fusion_high, fusion_low, metrics,
                                        k_per_student=args.k_per_student, higher_is_better=higher_is_better),
        "image_only": [],
        "text_only": []
    }
    # 이미지 전용
    if len(IMAGE_ONLY) == 2:
        t, s = best_pair(IMAGE_ONLY[0], IMAGE_ONLY[1], metrics, higher_is_better)
        kd_edges["image_only"].append((t, s))
    # 텍스트 전용
    if len(TEXT_ONLY) == 2:
        t, s = best_pair(TEXT_ONLY[0], TEXT_ONLY[1], metrics, higher_is_better)
        kd_edges["text_only"].append((t, s))

        # 4) 글로벌 벡터 (멀티모달 임베딩 전체 → KMeans → 매칭 → Z)
        Xi_all, Xt_all = stack_reps(FUSION_CLIENTS, split="train", max_samples_per_client=args.sample_per_client)

        # 4-1. 이미지(576D)를 텍스트(256D)와 같은 d_model 차원으로 *먼저* 투영
        print(f"[PCA] Projecting Image Reps {Xi_all.shape} -> {args.d_model}D ...")
        #       (N, 576) -> (N, 256) (여기선 N이 6이 아니라 수만 개임)
        Xi_all_proj = project_to_dim(Xi_all, args.d_model)
        print(f"[PCA] Done. New shape: {Xi_all_proj.shape}")

        # 4-2. *투영된* 벡터로 K-Means 실행
        img_cent = kmeans_centroids(Xi_all_proj, args.K_img)

        # 4-3. 텍스트 벡터로 K-Means 실행
        txt_cent = kmeans_centroids(Xt_all, args.K_txt)

        print(f"[KMeans] Centroids created: img={img_cent.shape}, txt={txt_cent.shape}")

        if img_cent.shape[0] == 0 or txt_cent.shape[0] == 0:
            print("[WARN] No centroids (check repr files). Z will be empty.")
        # (참고: txt_cent의 2번째 차원이 d_model과 다르면 여기서도 project_to_dim을 해줘야 함)
        if img_cent.shape[1] != txt_cent.shape[1]:
            print(f"[WARN] Dimension mismatch after KMeans: img={img_cent.shape[1]} vs txt={txt_cent.shape[1]}")

        # 4-4. build_global_vectors 호출 (이제 둘 다 (6, 256)이 입력됨)
        Zs = build_global_vectors(img_cent, txt_cent, d_model=args.d_model)

    # 4-1) Z 저장 (파일로 저장하고 경로를 페이로드에 넣자)
    z_mm_path      = GLOBAL_DIR / "Z_mm.npy"
    z_img2txt_path = GLOBAL_DIR / "Z_img2txt.npy"
    z_txt2img_path = GLOBAL_DIR / "Z_txt2img.npy"
    np.save(z_mm_path,      Zs["Z_mm"])
    np.save(z_img2txt_path, Zs["Z_img2txt"])
    np.save(z_txt2img_path, Zs["Z_txt2img"])

    # 5) 플랜 저장
    plan.update({
        "fusion_groups": {"high": fusion_high, "low": fusion_low},
        "kd_edges": kd_edges,
        "image_only": IMAGE_ONLY,
        "text_only": TEXT_ONLY,
        "Z_paths": {"Z_mm": str(z_mm_path), "Z_img2txt": str(z_img2txt_path), "Z_txt2img": str(z_txt2img_path)},
    })
    (GLOBAL_DIR / "orchestrator_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6) 클라별 페이로드 배포
    fusion_teachers = {t for (t, s) in kd_edges["fusion"]}
    fusion_students = {s for (t, s) in kd_edges["fusion"]}
    teachers_by_student: Dict[int, List[int]] = {}
    for (t, s) in kd_edges["fusion"]:
        teachers_by_student.setdefault(s, []).append(t)

    # FUSION
    for cid in FUSION_CLIENTS:
        role = "teacher" if cid in fusion_teachers else ("student" if cid in fusion_students else "solo")
        payload = {
            "client_id": cid,
            "type": "fusion",
            "Z_path": str(z_mm_path),
            "d_global": args.d_model,
            "role": role,
            "teacher_ids": teachers_by_student.get(cid, []),
            "group": "high" if cid in fusion_high else ("low" if cid in fusion_low else "unknown"),
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    # IMAGE ONLY
    tpair = kd_edges["image_only"][0] if len(kd_edges["image_only"]) else None
    for cid in IMAGE_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "image_only",
            "Z_proxy_text_path": str(z_img2txt_path),
            "d_global": args.d_model,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    # TEXT ONLY
    tpair = kd_edges["text_only"][0] if len(kd_edges["text_only"]) else None
    for cid in TEXT_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "text_only",
            "Z_proxy_image_path": str(z_txt2img_path),
            "d_global": args.d_model,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "metric": metrics.get(cid, None),
        }
        save_payload_for_client(cid, payload)

    print("[GLOBAL] Done ->", str(GLOBAL_DIR))
    print("Fusion groups:", {"high": fusion_high, "low": fusion_low})
    print("KD edges:", kd_edges)

if __name__ == "__main__":
    main()
