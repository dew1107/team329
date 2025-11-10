# fl_eval_baseline.py
# FedAvg baseline으로 생성된 global 모델을 각 클라이언트 데이터로 평가

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

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

@torch.no_grad()
def evaluate_model(model, loader, mode):
    model.eval()
    crit = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    logits_all, labels_all = [], []

    for batch in loader:
        y = batch["labels"].to(DEVICE)

        if mode == "multimodal":
            img = batch["image"].to(DEVICE)
            txt = {k: v.to(DEVICE) for k, v in batch["text"].items()}
            logits = model(image=img, text=txt)
        elif mode == "image_only":
            logits = model(batch["image"].to(DEVICE))
        else:
            txt = {k: v.to(DEVICE) for k, v in batch["text"].items()}
            logits = model(**txt)

        loss = crit(logits, y)
        total_loss += loss.item() * y.size(0)
        logits_all.append(logits.detach().cpu())
        labels_all.append(y.detach().cpu())

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
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
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
    }

def evaluate_global_baseline(global_path, client_ids, label_csv=DEFAULT_LABEL_CSV):
    print(f"Evaluating baseline → {global_path}")
    results = []
    state = torch.load(global_path, map_location=DEVICE)

    for cid in client_ids:
        mode = decide_mode_for_cid(cid)
        ckpt_path = Path(f"./outputs/client_{cid:02d}/best.pt")
        model, _ = build_model_from_ckpt(cid, str(ckpt_path), DEVICE)

        # load averaged weights
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[client_{cid:02d}] loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        # build validation dataloader
        from torch.utils.data import random_split
        from local_util.train_local import LABEL_COLUMNS

        csv_path = Path(CLIENT_CSV_DIR) / f"client_{cid:02d}.csv"
        label_table = load_label_table(label_csv)
        meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)
        ds_full = ClientDataset(str(csv_path), label_table, mode=mode, meta_picker=meta_picker)

        n = len(ds_full)
        n_tr = int(n * 0.9)
        _, val_set = random_split(ds_full, [n_tr, n - n_tr])
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        metrics = evaluate_model(model, val_loader, mode)
        print(f"[client_{cid:02d}] AUC={metrics['auc_macro']:.4f}, F1micro={metrics['f1_micro']:.4f}")
        results.append({"cid": cid, **metrics})

    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = Path(global_path).with_suffix(".eval.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results → {out_csv}")

if __name__ == "__main__":
    # 모달리티별 FedAvg 결과 파일 평가
    evaluate_global_baseline("./outputs/global_fedavg_multimodal.pt", list(range(1, 17)))
    evaluate_global_baseline("./outputs/global_fedavg_imgonly.pt", [17, 18])
    evaluate_global_baseline("./outputs/global_fedavg_txtonly.pt", [19, 20])
