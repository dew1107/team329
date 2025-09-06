import os, argparse, json, torch
from torch.optim import AdamW

from ..global_train.global_kd.config import cfg
from ..global_train.global_kd.prep_clients import build_dataloaders  # 데이터로더 재사용
from ..local_train import utils_fusion as uf  # FusionClassifier 인코더 재사용

from .kd_utils import compute_pos_weight
from .model_head import LocalClassifierWithAugment

# 일단 26~29 ( 모달리티 결손된 clients 만)

def load_payload_for_client(cid: int, map_location="cpu"):
    import torch
    import os
    from ..global_train.global_kd.config import cfg

    path = os.path.join(cfg.OUT_GLOBAL_DIR, f"client_{cid}", "global_payload.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location=map_location)

    # 1) 프록시 Z 우선 (이미지전용: Z_proxy_text / 텍스트전용: Z_proxy_image)
    Z = None
    if "Z_proxy_text" in payload and payload["Z_proxy_text"] is not None:
        Z = payload["Z_proxy_text"]
    elif "Z_proxy_image" in payload and payload["Z_proxy_image"] is not None:
        Z = payload["Z_proxy_image"]
    elif "Z" in payload and payload["Z"] is not None:
        # (보험) 멀티모달용 Z가 있을 때 사용할 수 있도록
        Z = payload["Z"]

    if Z is None:
        raise ValueError("No proxy Z in payload (expect Z_proxy_text or Z_proxy_image)")

    # 2) torch.Tensor/np.ndarray 모두 대응
    if not isinstance(Z, torch.Tensor):
        import numpy as np
        assert isinstance(Z, (np.ndarray, list)), f"Unexpected Z type: {type(Z)}"
        Z = torch.tensor(Z, dtype=torch.float32)
    else:
        Z = Z.detach().float()

    T = float(payload.get("kd_temperature", 2.0))
    W = float(payload.get("kd_rep_weight", 0.2))
    return Z, T, W, payload

def build_backbone(device):
    model = uf.FusionClassifier(num_classes=cfg.NUM_CLASSES).to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

@torch.no_grad()
def extract_reps(backbone, batch, device):
    pv  = batch.get("pixel_values"); ids = batch.get("input_ids"); am = batch.get("attention_mask")
    to = (lambda x: x.to(device, non_blocking=True) if x is not None else None)
    pv, ids, am = to(pv), to(ids), to(am)
    zi = backbone.img_enc(pv) if pv is not None and hasattr(backbone, "img_enc") else None
    zt = backbone.txt_enc(ids, am) if ids is not None and hasattr(backbone, "txt_enc") else None
    return zi, zt



def train_one_epoch(backbone, head, loader, Z, device, bce, optimizer):
    head.train(); Z = Z.to(device); total = 0.0
    import torch.nn.functional as F
    for b in loader:
        y = b["labels"].to(device).float()
        with torch.no_grad():
            zi, zt = extract_reps(backbone, b, device)

        logits, R = head(img_rep=zi, txt_rep=zt, Z_global=Z)

        # ★ pos_weight 적용된 BCE 사용
        loss = bce(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)  # 선택: 안정화
        optimizer.step()

        total += float(loss.detach().cpu())
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(backbone, head, loader, Z, device):
    from sklearn.metrics import f1_score, roc_auc_score
    head.eval(); Z = Z.to(device)
    import torch as _t
    Ls, Ys, Ps = [], [], []
    for b in loader:
        y = b["labels"].to(device).float()
        zi, zt = extract_reps(backbone, b, device)
        logits, _ = head(img_rep=zi, txt_rep=zt, Z_global=Z)
        Ls.append(logits.detach().cpu()); Ys.append(y.detach().cpu())
    L = _t.cat(Ls, 0); Y = _t.cat(Ys, 0)
    P = _t.sigmoid(L).numpy(); Y = Y.numpy().astype("int32")
    yhat = (P >= 0.5).astype("int32")
    from sklearn.metrics import f1_score, roc_auc_score
    f1_micro = f1_score(Y, yhat, average="micro", zero_division=0)
    f1_macro = f1_score(Y, yhat, average="macro", zero_division=0)
    aucs = []
    for c in range(Y.shape[1]):
        col = Y[:, c]
        if len(set(col.tolist())) < 2: continue
        try: aucs.append(roc_auc_score(col, P[:, c]))
        except: pass
    auc_macro = float(sum(aucs)/len(aucs)) if aucs else float("nan")
    return {"f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # 0) 프록시 Z 로드 (26–29 전용)
    Z, T, W, payload = load_payload_for_client(args.cid, map_location="cpu")

    # 1) 데이터
    train_loader, val_loader = build_dataloaders(args.cid, args.batch)  # 기존 로더 재사용

    # pos_weight 계산
    pw = compute_pos_weight(train_loader, cfg.NUM_CLASSES).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    # 2) 백본 인코더(고정), 결손 보충 헤드
    backbone = build_backbone(device)
    head = LocalClassifierWithAugment(
        d_img=cfg.IMG_DIM, d_txt=cfg.TXT_DIM, d_local_fused=cfg.FUSED_DIM,
        d_global=cfg.D_MODEL, n_labels=cfg.NUM_CLASSES, use_hallucinate=False # img 만 있는 경우 false 가 좀 더 잘 나오긴 하는데 f1은 여전히 0 임
    ).to(device)
    global opt; opt = AdamW(head.parameters(), lr=args.lr)

    best = 1e9; best_m = None
    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(backbone, head, train_loader, Z, device, bce, opt)
        m = evaluate(backbone, head, val_loader, Z, device)
        print(f"[client_{args.cid}] ep{ep} train_loss={tr:.4f}  val f1_micro={m['f1_micro']:.4f} macro={m['f1_macro']:.4f} auc={m['auc_macro']:.4f}")
        if tr < best:
            best, best_m = tr, m
            out = os.path.join(cfg.BASE_DIR, f"client_{args.cid}", f"client_{args.cid}_local_gated_best.pt")
            torch.save({"head": head.state_dict(), "payload_meta": {"cid": args.cid, "d_global": payload.get("d_global", None)}}, out)
            print(f"[client_{args.cid}] saved -> {out}")

    # 메트릭 기록
    outj = os.path.join(cfg.BASE_DIR, f"client_{args.cid}", f"client_{args.cid}_local_gated_metrics.json")
    with open(outj, "w", encoding="utf-8") as f:
        json.dump(best_m, f, indent=2, ensure_ascii=False)
    print(f"[client_{args.cid}] metrics saved -> {outj}")

if __name__ == "__main__":
    main()

# txt only 는 결과가 잘 나옴 - hallucination true 일 때