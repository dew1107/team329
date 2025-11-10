# fl_baselines.py
# 전통적인 FedAvg / FedProx 연합학습 베이스라인
# - 기존 best.pt를 초기값으로 사용
# - 모달리티별 그룹(multimodal / image_only / text_only) 안에서 FL 수행
# - 최종 per-client 성능을 CSV로 저장하여 우리 알고리즘 결과와 비교 가능

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score

# 프로젝트 루트 기준 경로 추가 (update_local_v2.py와 동일 패턴)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from local_util.train_local import (
    CLIENT_CSV_DIR,
    ClientDataset,
    LABEL_COLUMNS,
    load_label_table,
    build_image_picker_from_metadata,
    LABEL_CSV as DEFAULT_LABEL_CSV,
    METADATA_CSV,
    IMG_ROOT,
)

# prep_clients / update_local_v2와 동일 API 사용
from global_util.prep_clients import (
    build_model_from_ckpt,   # outputs/client_{cid}/best.pt 로부터 모델 생성
    decide_mode_for_cid,     # cid -> "multimodal" / "image_only" / "text_only"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 유틸 함수
# ---------------------------

def parse_client_ids(text: str) -> List[int]:
    """
    "1-20" 또는 "1,3,5" 형식 문자열을 [1, 2, ..., 20] 리스트로 파싱
    """
    text = text.strip().lower()
    ids = []
    if text in ("all", ""):
        # 기본: 1~20 모두
        return list(range(1, 21))
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(list(range(int(a), int(b) + 1)))
        else:
            ids.append(int(part))
    # 중복 제거 + 정렬
    ids = sorted(list(dict.fromkeys(ids)))
    return ids


def build_dataloaders(cid: int, label_csv_path: str, batch_size: int):
    """
    update_local_v2.py와 동일한 90/10 분할 로직 사용.
    """
    csv_path = Path(CLIENT_CSV_DIR) / f"client_{cid:02d}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[client_{cid:02d}] CSV not found: {csv_path}")

    mode = decide_mode_for_cid(cid)
    label_table = load_label_table(label_csv_path)
    meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)

    ds_full = ClientDataset(str(csv_path), label_table, mode=mode, meta_picker=meta_picker)
    n = len(ds_full)
    n_tr = int(n * 0.9)
    n_va = n - n_tr

    tr_set, va_set = random_split(
        ds_full,
        [n_tr, n_va],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(va_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[client_{cid:02d}] Data loaded for FL: mode={mode}, total={n} (train={n_tr}, val={n_va})")
    return train_loader, val_loader, mode, n_tr, n_va


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, mode: str, device: torch.device) -> Dict[str, float]:
    """
    update_local_v2.evaluate_v2와 동일한 멀티라벨 평가.
    """
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    logits_all, labels_all = [], []
    total_loss = 0.0

    for batch in loader:
        y = batch["labels"].to(device)

        if mode == "multimodal":
            img = batch["image"].to(device)
            txt = {k: v.to(device) for k, v in batch["text"].items()}
            logits = model(image=img, text=txt)
        elif mode == "image_only":
            logits = model(batch["image"].to(device))
        else:  # text_only
            txt = {k: v.to(device) for k, v in batch["text"].items()}
            logits = model(**txt)

        loss = crit(logits, y)
        total_loss += loss.item() * y.size(0)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    if len(logits_all) == 0:
        return {"loss": float("nan"), "f1_micro": 0.0, "f1_macro": 0.0, "auc_macro": float("nan")}

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    probs = torch.sigmoid(logits_all).numpy()
    y_true = labels_all.numpy().astype(np.int32)
    y_pred = (probs >= 0.5).astype(np.int32)

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if len(np.unique(col)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(col, probs[:, c]))
        except Exception:
            pass
    auc_macro = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

    return {
        "loss": total_loss / len(loader.dataset),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "auc_macro": auc_macro,
    }


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    FedAvg/FedProx baseline에서는 전체 모델(백본+헤드)을 학습 대상으로 둔다.
    필요하면 여기서 특정 모듈만 선택하도록 조정할 수 있음.
    """
    return [p for p in model.parameters() if p.requires_grad]


def state_dict_clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def fedavg_aggregate(local_states: Dict[int, Dict[str, torch.Tensor]],
                     weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
    """
    표준 FedAvg: 파라미터를 데이터 개수 비율로 가중 평균.
    """
    client_ids = list(local_states.keys())
    global_state = {}

    for k in local_states[client_ids[0]].keys():
        acc = None
        for cid in client_ids:
            w = weights[cid]
            v = local_states[cid][k]
            if acc is None:
                acc = w * v
            else:
                acc += w * v
        global_state[k] = acc
    return global_state


def local_train_one_round(
    cid: int,
    global_state: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    mode: str,
    algo: str,
    mu: float,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    한 클라이언트에서 한 round 동안 FedAvg/FedProx local update 수행.
    - algo: "fedavg" or "fedprox"
    - global_state: round 시작 시점의 글로벌 모델 state_dict
    """
    # best.pt에서 초기 모양(아키텍처) 가져오기 위해 한 번 로드.
    # (이미 global_state를 덮어쓰므로 가중치는 글로벌 모델 기준이 됨)
    ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
    model, _ = build_model_from_ckpt(cid, str(ckpt_path), device)
    model.load_state_dict(global_state)
    model.to(device)

    params = get_trainable_params(model)
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()

    # FedProx용 글로벌 파라미터 복사
    global_params = None
    if algo.lower() == "fedprox":
        global_params = [p.detach().clone() for p in params]

    for ep in range(local_epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            y = batch["labels"].to(device)

            if mode == "multimodal":
                img = batch["image"].to(device)
                txt = {k: v.to(device) for k, v in batch["text"].items()}
                logits = model(image=img, text=txt)
            elif mode == "image_only":
                logits = model(batch["image"].to(device))
            else:
                txt = {k: v.to(device) for k, v in batch["text"].items()}
                logits = model(**txt)

            loss = crit(logits, y)

            # FedProx proximal term
            if algo.lower() == "fedprox" and global_params is not None:
                prox = 0.0
                for p, pg in zip(params, global_params):
                    prox = prox + torch.norm(p - pg) ** 2
                loss = loss + 0.5 * mu * prox

            loss.backward()
            opt.step()

    return model.state_dict()


# ---------------------------
# 그룹 FL 실행 (multimodal / image_only / text_only)
# ---------------------------

def run_fl_for_group(
    group_name: str,
    cids: List[int],
    args: argparse.Namespace,
    summary_rows: List[Dict],
):
    """
    같은 모달리티 그룹 내에서 FedAvg / FedProx 실행.
    - group_name: "multimodal" / "image_only" / "text_only"
    - cids: 해당 그룹에 포함된 client id 리스트
    """
    if not cids:
        return

    print("\n" + "=" * 80)
    print(f"[FL-{group_name}] clients = {cids}")
    print("=" * 80)

    # 1) 초기 global model: 첫 클라이언트의 best.pt 기반
    first_cid = cids[0]
    ckpt_path = Path(f"./outputs/client_{first_cid:02d}/best.pt")
    model, mode = build_model_from_ckpt(first_cid, str(ckpt_path), DEVICE)
    global_state = model.state_dict()

    # 2) 클라이언트별 DataLoader, val_loader, 데이터 수
    loaders: Dict[int, Tuple[DataLoader, DataLoader, str, int, int]] = {}
    n_train_total = 0
    for cid in cids:
        train_loader, val_loader, mode_c, n_tr, n_va = build_dataloaders(
            cid,
            args.label_csv,
            args.batch_size,
        )
        loaders[cid] = (train_loader, val_loader, mode_c, n_tr, n_va)
        n_train_total += n_tr

    # 3) FedAvg / FedProx rounds
    for rnd in range(1, args.rounds + 1):
        print("\n" + "-" * 60)
        print(f"[{group_name}] Round {rnd}/{args.rounds}")
        print("-" * 60)

        local_states: Dict[int, Dict[str, torch.Tensor]] = {}
        weights: Dict[int, float] = {}

        # (1) 각 클라이언트 local update
        for cid in cids:
            train_loader, val_loader, mode_c, n_tr, _ = loaders[cid]
            w_k = n_tr / n_train_total
            weights[cid] = w_k

            print(f"[{group_name}] Client {cid:02d} local update (w={w_k:.4f}) ...")
            local_state = local_train_one_round(
                cid=cid,
                global_state=global_state,
                train_loader=train_loader,
                mode=mode_c,
                algo=args.algo,
                mu=args.mu,
                local_epochs=args.local_epochs,
                lr=args.lr,
                device=DEVICE,
            )
            local_states[cid] = local_state

        # (2) FedAvg aggregation
        global_state = fedavg_aggregate(local_states, weights)

        # (3) Round 끝에서 글로벌 모델 성능 로그 (선택사항)
        #     여기서는 첫 번째 클라이언트의 val셋 기준으로만 간단히 확인
        ref_cid = cids[0]
        _, val_loader_ref, mode_ref, _, _ = loaders[ref_cid]
        model.load_state_dict(global_state)
        metrics_ref = evaluate_model(model, val_loader_ref, mode_ref, DEVICE)
        print(f"[{group_name}] Round {rnd} global eval on client {ref_cid:02d}: "
              f"loss={metrics_ref['loss']:.4f}, "
              f"auc_macro={metrics_ref['auc_macro']:.4f}")

    # 4) 최종 global_state를 각 클라이언트에 배포 후, 개별 val 성능 기록
    for cid in cids:
        train_loader, val_loader, mode_c, n_tr, n_va = loaders[cid]
        ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
        model_c, _ = build_model_from_ckpt(cid, str(ckpt_path), DEVICE)
        model_c.load_state_dict(global_state)
        model_c.to(DEVICE)

        metrics = evaluate_model(model_c, val_loader, mode_c, DEVICE)
        print(f"[{group_name}] FINAL | client {cid:02d} | "
              f"loss={metrics['loss']:.4f}, auc_macro={metrics['auc_macro']:.4f}, "
              f"f1_micro={metrics['f1_micro']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

        summary_rows.append({
            "algo": args.algo,
            "group": group_name,
            "client_id": cid,
            "mode": mode_c,
            "n_train": n_tr,
            "n_val": n_va,
            "loss": metrics["loss"],
            "f1_micro": metrics["f1_micro"],
            "f1_macro": metrics["f1_macro"],
            "auc_macro": metrics["auc_macro"],
        })


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20",
                    help='예: "1-20" 또는 "1,3,5" (default: 1-20)')
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--algo", type=str, default="fedavg",
                    choices=["fedavg", "fedprox"],
                    help="연합 학습 알고리즘 (FedAvg 또는 FedProx)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="통신(aggregation) 라운드 수")
    ap.add_argument("--local_epochs", type=int, default=1,
                    help="각 라운드에서 클라이언트 로컬 에폭 수")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="로컬 학습 learning rate")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mu", type=float, default=0.01,
                    help="FedProx proximal 계수(mu)")
    ap.add_argument("--out_csv", type=str, default="./outputs/fl_baseline_summary.csv")
    args = ap.parse_args()

    all_ids = parse_client_ids(args.clients)

    # 모달리티별 그룹화 (prep_clients.py 로직과 동일)
    multimodal_ids = [cid for cid in all_ids if 1 <= cid <= 16]
    image_only_ids = [cid for cid in all_ids if cid in [17, 18]]
    text_only_ids  = [cid for cid in all_ids if cid in [19, 20]]

    summary_rows: List[Dict] = []

    # 그룹별 FL 실행
    run_fl_for_group("multimodal", multimodal_ids, args, summary_rows)
    run_fl_for_group("image_only", image_only_ids, args, summary_rows)
    run_fl_for_group("text_only",  text_only_ids,  args, summary_rows)

    # 결과 저장
    if summary_rows:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[FL-Baseline] DONE. Summary saved → {out_csv} (rows={len(summary_rows)})")
    else:
        print("\n[FL-Baseline] WARN: no clients processed; nothing saved.")


if __name__ == "__main__":
    main()
