# fl_fedprox.py
# 이미 학습된 best.pt를 초기값으로 사용해서
# FedProx 방식으로 한 번의 FL round를 수행하고, 각 클라이언트 성능을 평가해 CSV로 저장.

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score

# 프로젝트 구조에 맞게 import (이전 코드랑 동일하게 가정)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from global_util.prep_clients import build_model_from_ckpt, decide_mode_for_cid
from local_util.train_local import (
    CLIENT_CSV_DIR,
    ClientDataset,
    load_label_table,
    build_image_picker_from_metadata,
    LABEL_CSV as DEFAULT_LABEL_CSV,
    METADATA_CSV,
    IMG_ROOT,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def build_dataloaders(cid: int, label_csv_path: str, batch_size: int):
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
    val_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[client_{cid:02d}] data loaded for FedProx: mode={mode}, total={n} (train={n_tr}, val={n_va})")
    return train_loader, val_loader, mode, n_tr, n_va


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, mode: str, device: torch.device) -> Dict[str, float]:
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


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def fedavg_aggregate(local_states: Dict[int, Dict[str, torch.Tensor]],
                     weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
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


def local_train_fedprox(
    cid: int,
    global_state: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    mode: str,
    mu: float,
    local_epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    FedProx local update:
      L_i = BCE + (mu/2) * ||w_i - w_global||^2
    """
    ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
    model, _ = build_model_from_ckpt(cid, str(ckpt_path), device)
    model.load_state_dict(global_state)
    model.to(device)

    params = get_trainable_params(model)
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()

    global_params = [p.detach().clone() for p in params]  # for prox term

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
            prox = 0.0
            for p, pg in zip(params, global_params):
                prox = prox + torch.norm(p - pg) ** 2
            loss = loss + 0.5 * mu * prox

            loss.backward()
            opt.step()

    return model.state_dict()


def run_fedprox_for_group(
    group_name: str,
    cids: List[int],
    label_csv: str,
    batch_size: int,
    mu: float,
    local_epochs: int,
    lr: float,
    summary_rows: List[Dict],
):
    if not cids:
        return

    print("\n" + "=" * 80)
    print(f"[FedProx-{group_name}] clients = {cids}")
    print("=" * 80)

    # 1) 초기 global: 첫 클라이언트 best.pt에서 가져옴
    first_cid = cids[0]
    ckpt_path = Path(f"./outputs/client_{first_cid:02d}/best.pt")
    model, mode = build_model_from_ckpt(first_cid, str(ckpt_path), DEVICE)
    global_state = model.state_dict()

    # 2) 데이터 로더 로딩
    loaders: Dict[int, Tuple[DataLoader, DataLoader, str, int, int]] = {}
    n_train_total = 0
    for cid in cids:
        train_loader, val_loader, mode_c, n_tr, n_va = build_dataloaders(cid, label_csv, batch_size)
        loaders[cid] = (train_loader, val_loader, mode_c, n_tr, n_va)
        n_train_total += n_tr

    # 3) FedProx round = 1 (논문 비교를 위해 1회 global update만)
    local_states: Dict[int, Dict[str, torch.Tensor]] = {}
    weights: Dict[int, float] = {}

    for cid in cids:
        train_loader, _, mode_c, n_tr, _ = loaders[cid]
        w_k = n_tr / n_train_total
        weights[cid] = w_k

        print(f"[{group_name}] FedProx local update client {cid:02d} (w={w_k:.4f}) ...")
        local_state = local_train_fedprox(
            cid=cid,
            global_state=global_state,
            train_loader=train_loader,
            mode=mode_c,
            mu=mu,
            local_epochs=local_epochs,
            lr=lr,
            device=DEVICE,
        )
        local_states[cid] = local_state

    # 4) FedAvg aggregation (FedProx도 서버에서는 평균)
    global_state = fedavg_aggregate(local_states, weights)

    # 5) 최종 global_state를 각 클라에 적용해서 val 성능 평가
    for cid in cids:
        _, val_loader, mode_c, n_tr, n_va = loaders[cid]
        ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
        model_c, _ = build_model_from_ckpt(cid, str(ckpt_path), DEVICE)
        model_c.load_state_dict(global_state)
        model_c.to(DEVICE)

        metrics = evaluate_model(model_c, val_loader, mode_c, DEVICE)
        print(f"[FedProx-{group_name}] FINAL | client {cid:02d} | "
              f"loss={metrics['loss']:.4f}, auc_macro={metrics['auc_macro']:.4f}, "
              f"f1_micro={metrics['f1_micro']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

        summary_rows.append({
            "algo": "fedprox",
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=str, default="1-20",
                    help='예: "1-20" 또는 "17-20" (default: 1-20)')
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_csv", type=str, default="./outputs/fl_fedprox_summary.csv")
    args = ap.parse_args()

    # clients 파싱 (매우 단순 버전)
    text = args.clients.strip().lower()
    ids: List[int] = []
    if text in ("all", "", "1-20"):
        ids = list(range(1, 21))
    else:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                ids.extend(list(range(int(a), int(b) + 1)))
            else:
                ids.append(int(part))
        ids = sorted(list(dict.fromkeys(ids)))

    # 모달리티 그룹 분리 (너 구조에 맞게)
    multimodal_ids = [cid for cid in ids if 1 <= cid <= 16]
    image_only_ids = [cid for cid in ids if cid in [17, 18]]
    text_only_ids  = [cid for cid in ids if cid in [19, 20]]

    summary_rows: List[Dict] = []

    run_fedprox_for_group("multimodal", multimodal_ids,
                          label_csv=args.label_csv,
                          batch_size=args.batch_size,
                          mu=args.mu,
                          local_epochs=args.local_epochs,
                          lr=args.lr,
                          summary_rows=summary_rows)

    run_fedprox_for_group("image_only", image_only_ids,
                          label_csv=args.label_csv,
                          batch_size=args.batch_size,
                          mu=args.mu,
                          local_epochs=args.local_epochs,
                          lr=args.lr,
                          summary_rows=summary_rows)

    run_fedprox_for_group("text_only", text_only_ids,
                          label_csv=args.label_csv,
                          batch_size=args.batch_size,
                          mu=args.mu,
                          local_epochs=args.local_epochs,
                          lr=args.lr,
                          summary_rows=summary_rows)

    # CSV 저장
    if summary_rows:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[FedProx] DONE. Summary saved → {out_csv} (rows={len(summary_rows)})")
    else:
        print("\n[FedProx] WARN: no clients processed; nothing saved.")


if __name__ == "__main__":
    main()
