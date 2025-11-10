# k_sweep_kstudy.py
# - orchestrator.py에서 함수들을 가져와서
# - 여러 K 값(K_img, K_txt)에 대해 클러스터 정렬 점수 및 관련 지표를 기록
#
# 사용 예:
#   python -u k_sweep_kstudy.py --metric macro_auroc --Ks 4 6 32 --d_model 256
#
# 결과:
#   ./global_outputs/k_sweep_results.csv 에 각 K별 정렬 점수 및 요약 통계가 append됨.

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


import argparse
import csv
from pathlib import Path
import numpy as np

import global_util.orchestrator as orch

GLOBAL_DIR = Path("./global_outputs")
GLOBAL_DIR.mkdir(parents=True, exist_ok=True)

def compute_alignment_stats(img_cent, txt_cent, d_model: int):
    """
    주어진 이미지/텍스트 클러스터 센트로이드로부터
    - 정렬(헝가리안 매칭) 후 코사인 유사도 통계 계산
    """
    if img_cent.size == 0 or txt_cent.size == 0:
        return {
            "align_mean": np.nan,
            "align_std": np.nan,
            "align_min": np.nan,
            "align_max": np.nan,
            "num_pairs": 0,
        }

    # d_model 차원으로 투영
    img_proj = orch.project_to_dim(img_cent, d_model)
    txt_proj = orch.project_to_dim(txt_cent, d_model)

    # 코사인 유사도 행렬 + 헝가리안 매칭
    S = orch.cosine_matrix(img_proj, txt_proj)  # (K_img, K_txt)
    pairs = orch.hungarian_pairs(S)

    if len(pairs) == 0:
        return {
            "align_mean": np.nan,
            "align_std": np.nan,
            "align_min": np.nan,
            "align_max": np.nan,
            "num_pairs": 0,
        }

    sims = np.array([float(S[i, j]) for (i, j) in pairs], dtype=np.float32)
    return {
        "align_mean": float(sims.mean()),
        "align_std": float(sims.std()),
        "align_min": float(sims.min()),
        "align_max": float(sims.max()),
        "num_pairs": int(len(pairs)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric", type=str, default="macro_auroc",
        choices=["macro_auroc", "loss"],
        help="클라이언트 성능 지표 (summary.csv 기준)"
    )
    parser.add_argument(
        "--Ks", type=int, nargs="+", default=[4, 6, 32],
        help="실험할 K 값 목록 (예: --Ks 4 6 8 16 32)"
    )
    parser.add_argument(
        "--d_model", type=int, default=256,
        help="글로벌 임베딩 차원 (orchestrator와 동일)"
    )
    parser.add_argument(
        "--sample_per_client", type=int, default=20000,
        help="각 클라이언트당 샘플링할 최대 repr 개수"
    )
    parser.add_argument(
        "--out_csv", type=str, default=str(GLOBAL_DIR / "k_sweep_results.csv"),
        help="결과를 저장할 CSV 경로"
    )
    args = parser.parse_args()

    # seed 설정
    orch.set_seed(orch.SEED)

    # 1) 클라이언트 성능 메트릭 로드
    metrics = orch._read_summary(orch.SUMMARY_CSV, metric=args.metric)
    higher_is_better = (args.metric != "loss")

    # 2) 멀티모달 클라이언트(1..16) high/low split
    fusion_high, fusion_low = orch.split_median(metrics, orch.FUSION_CLIENTS, higher_is_better)
    # high/low 그룹 메트릭 평균
    def avg_metric(cids):
        vals = [metrics.get(c, np.nan) for c in cids]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    fusion_high_mean = avg_metric(fusion_high)
    fusion_low_mean = avg_metric(fusion_low)

    # 3) 멀티모달 repr 전체 스택 (이미지/텍스트)
    Xi_all, Xt_all = orch.stack_reps(orch.FUSION_CLIENTS, split="train", max_samples_per_client=args.sample_per_client)
    print(f"[Repr] Xi_all shape={Xi_all.shape}, Xt_all shape={Xt_all.shape}")

    # 4) 결과 CSV 준비
    out_path = Path(args.out_csv)
    header = [
        "metric_name", "higher_is_better",
        "K_img", "K_txt", "d_model",
        "align_mean", "align_std", "align_min", "align_max", "num_pairs",
        "fusion_high_mean", "fusion_low_mean",
        "num_fusion_high", "num_fusion_low",
    ]
    write_header = not out_path.exists()

    with out_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)

        # 5) 여러 K 값에 대해 반복
        for K in args.Ks:
            print(f"\n[K={K}] Clustering and alignment evaluation...")

            # 5-1) 이미지 repr를 d_model로 먼저 투영 후 KMeans
            Xi_proj = orch.project_to_dim(Xi_all, args.d_model)
            img_cent = orch.kmeans_centroids(Xi_proj, K)

            # 5-2) 텍스트 repr로 KMeans
            txt_cent = orch.kmeans_centroids(Xt_all, K)

            print(f"[KMeans] img_cent shape={img_cent.shape}, txt_cent shape={txt_cent.shape}")

            # 5-3) 정렬(헝가리안) 및 코사인 유사도 통계
            align_stats = compute_alignment_stats(img_cent, txt_cent, d_model=args.d_model)

            row = [
                args.metric,
                int(higher_is_better),
                K,       # K_img
                K,       # K_txt (여기서는 동일하게 K 사용)
                args.d_model,
                align_stats["align_mean"],
                align_stats["align_std"],
                align_stats["align_min"],
                align_stats["align_max"],
                align_stats["num_pairs"],
                fusion_high_mean,
                fusion_low_mean,
                len(fusion_high),
                len(fusion_low),
            ]
            w.writerow(row)
            print(f"[K={K}] align_mean={align_stats['align_mean']:.4f}, "
                  f"align_min={align_stats['align_min']:.4f}, "
                  f"align_max={align_stats['align_max']:.4f}, "
                  f"num_pairs={align_stats['num_pairs']}")

    print(f"\n[Done] K-sweep results saved to: {out_path}")

if __name__ == "__main__":
    main()
