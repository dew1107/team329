# orchestrator.py (DROP-IN)
import os
from .config import cfg
from .utils_io import set_seed, ensure_dir, load_client_metric, save_json, save_payload_for_client
from .build_vectors import stack_reps, kmeans_centroids, build_global_vectors
import numpy as np

def _split_median(metrics, cids):
    vals = [metrics.get(c, np.nan) for c in cids]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return [], []
    med = float(np.median(np.array(vals)))
    high = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] >  med]
    low  = [c for c in cids if not np.isnan(metrics.get(c, np.nan)) and metrics[c] <= med]
    return high, low

def _best_pair(a, b, metrics):
    va, vb = metrics.get(a, np.nan), metrics.get(b, np.nan)
    if np.isnan(va) and np.isnan(vb): return None
    if np.isnan(vb): return (a, b)
    if np.isnan(va): return (b, a)
    return (a, b) if va >= vb else (b, a)

def _build_fusion_kd_edges(high, low, metrics, k_per_student=2):
    high_sorted = sorted(high, key=lambda c: metrics.get(c, float("-inf")), reverse=True)
    edges = []
    for s in low:
        teachers = high_sorted[:k_per_student] if len(high_sorted) >= k_per_student else high_sorted
        for t in teachers:
            edges.append((t, s))
    return edges

def main():
    set_seed(cfg.SEED)
    ensure_dir(cfg.OUT_GLOBAL_DIR)

    # 1) 모든 관련 클라이언트 메트릭 로드 (fusion+image/text-only)
    cids_all = list(cfg.FUSION_CLIENTS) + list(cfg.IMAGE_ONLY) + list(cfg.TEXT_ONLY)
    metrics = {cid: load_client_metric(cid) for cid in cids_all}
    save_json({"metric_name": cfg.METRIC_NAME, "metrics": metrics},
              os.path.join(cfg.OUT_GLOBAL_DIR, cfg.METRICS_SNAPSHOT))

    # 2) FUSION High/Low(중앙값)
    fusion_high, fusion_low = _split_median(metrics, list(cfg.FUSION_CLIENTS))

    # 3) KD 간선
    kd_edges = {
        "fusion": _build_fusion_kd_edges(fusion_high, fusion_low, metrics, k_per_student=2),
        "image_only": [],
        "text_only": []
    }
    # 이미지 전용(26,27)
    if len(cfg.IMAGE_ONLY) == 2:
        bp = _best_pair(cfg.IMAGE_ONLY[0], cfg.IMAGE_ONLY[1], metrics)
        if bp: kd_edges["image_only"].append(bp)
    # 텍스트 전용(28,29)
    if len(cfg.TEXT_ONLY) == 2:
        bp = _best_pair(cfg.TEXT_ONLY[0], cfg.TEXT_ONLY[1], metrics)
        if bp: kd_edges["text_only"].append(bp)

    # 4) 글로벌 벡터: 멀티모달 전체 reps → K-means → 헝가리안 매칭 포함 Cross-Attention
    img_all, txt_all = stack_reps(list(cfg.FUSION_CLIENTS), split="train", max_samples=cfg.SAMPLE_PER_CLIENT)
    img_cent = kmeans_centroids(img_all, cfg.K_IMG)
    txt_cent = kmeans_centroids(txt_all, cfg.K_TXT)
    if img_cent.shape[0] != txt_cent.shape[0]:
        print(f"[warn] K mismatch: K_IMG={img_cent.shape[0]} vs K_TXT={txt_cent.shape[0]} → min K 사용")
    Zs = build_global_vectors(img_cent, txt_cent)

    # 5) KD 플랜 저장
    kd_plan = {
        "metric_name": cfg.METRIC_NAME,
        "kd_rep_weight": cfg.KD_REP_WEIGHT,
        "kd_temperature": cfg.KD_TEMP,
        "fusion_groups": {"high": fusion_high, "low": fusion_low},
        "kd_edges": kd_edges,
        "image_only": list(cfg.IMAGE_ONLY),
        "text_only": list(cfg.TEXT_ONLY),
    }
    save_json(kd_plan, os.path.join(cfg.OUT_GLOBAL_DIR, cfg.KD_PLAN_JSON))

    # 6) 클라이언트별 payload 저장
    fusion_teachers = {t for (t, s) in kd_edges["fusion"]}
    fusion_students = {s for (t, s) in kd_edges["fusion"]}
    teachers_by_student = {}
    for (t, s) in kd_edges["fusion"]:
        teachers_by_student.setdefault(s, []).append(t)

    # FUSION
    for cid in cfg.FUSION_CLIENTS:
        role = "teacher" if cid in fusion_teachers else ("student" if cid in fusion_students else "solo")
        payload = {
            "client_id": cid,
            "type": "fusion",
            "Z": Zs["Z_mm"],
            "d_global": cfg.D_MODEL,
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
            "role": role,
            "teacher_ids": teachers_by_student.get(cid, []),
            "group": "high" if cid in fusion_high else ("low" if cid in fusion_low else "unknown"),
        }
        save_payload_for_client(cid, payload)

    # IMAGE ONLY
    tpair = kd_edges["image_only"][0] if len(kd_edges["image_only"]) else None
    for cid in cfg.IMAGE_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "image_only",
            "Z_proxy_text": Zs["Z_img2txt"],
            "d_global": cfg.D_MODEL,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
        }
        save_payload_for_client(cid, payload)

    # TEXT ONLY
    tpair = kd_edges["text_only"][0] if len(kd_edges["text_only"]) else None
    for cid in cfg.TEXT_ONLY:
        t_for_me = (tpair[0] if tpair and tpair[1] == cid else None)
        payload = {
            "client_id": cid,
            "type": "text_only",
            "Z_proxy_image": Zs["Z_txt2img"],
            "d_global": cfg.D_MODEL,
            "role": "student" if t_for_me is not None else "teacher",
            "teacher_ids": [t_for_me] if t_for_me is not None else [],
            "kd_rep_weight": cfg.KD_REP_WEIGHT,
            "kd_temperature": cfg.KD_TEMP,
        }
        save_payload_for_client(cid, payload)

    print("[GLOBAL] Done ->", cfg.OUT_GLOBAL_DIR)
    print("Fusion groups:", {"high": fusion_high, "low": fusion_low})
    print("KD edges:", kd_edges)

if __name__ == "__main__":
    main()
