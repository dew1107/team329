# -*- coding: utf-8 -*-
"""
update_local.py (refined eval)
- 원본 파일을 다시 읽지 않고, prep_clients.py가 만든 임베딩(repr_img.npy, repr_txt.npy) + index.csv + 라벨 CSV만으로
  로컬 분류 헤드들을 소규모 업데이트(Z 정렬 + 게이팅 포함)하고,
  '동일한 임베딩 검증 분할'에서 baseline(best.pt의 헤드) vs updated 헤드를 공정 비교합니다.
- 두 임베딩이 모두 있으면 멀티모달로 간주하여 late fusion 평가(모달 마스크 기반).
- Z(글로벌 벡터) 정렬 손실과 게이팅 알고리즘은 유지됩니다.

사용 예:
  python -u update_local.py --client_id 1-20 --epochs 1 --z_weight 0.1 --tau 0.1
  python -u update_local.py --client_id 1 --label_csv .\\mimic-cxr-2.0.0-negbio.csv
"""

import os, json, argparse, csv
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score

# ---- 기존 설정 재사용(라벨 컬럼/기본 라벨 CSV) ----
from train_local import (
    LABEL_COLUMNS, LABEL_CSV as DEFAULT_LABEL_CSV
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------
# 경로/도움 함수
# ---------------------------
def cdir(cid: int) -> Path:
    d = Path(f"./outputs/client_{cid:02d}")
    d.mkdir(parents=True, exist_ok=True)
    return d

def fmt4(x: float) -> str:
    return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

# ---------------------------
# 데이터(임베딩 + 라벨)
# ---------------------------
def load_embeddings_and_index(cid: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Dict[str,str]]]]:
    base = cdir(cid)
    Xi, Xt, rows = None, None, None
    p_img = base/"repr_img.npy"
    p_txt = base/"repr_txt.npy"
    p_idx = base/"index.csv"
    if p_img.exists(): Xi = np.load(p_img)
    if p_txt.exists(): Xt = np.load(p_txt)
    if p_idx.exists():
        import pandas as pd
        rows = pd.read_csv(p_idx).to_dict("records")
    return Xi, Xt, rows

def load_label_table(label_csv: str) -> Dict[Tuple[int,int], List[int]]:
    import pandas as pd
    if not os.path.exists(label_csv):
        raise FileNotFoundError(f"라벨 CSV가 없습니다: {label_csv}")
    df = pd.read_csv(label_csv)
    required = {"subject_id","study_id", *LABEL_COLUMNS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"라벨 CSV 칼럼 누락: {missing}")
    for c in LABEL_COLUMNS:
        df[c] = df[c].fillna(0)
        df[c] = (df[c] >= 1).astype(int)
    table = {}
    for _, r in df.iterrows():
        key = (int(r["subject_id"]), int(r["study_id"]))
        table[key] = [int(r[c]) for c in LABEL_COLUMNS]
    return table

def idstr_to_ints(p: str, s: str) -> Tuple[int,int]:
    return int(str(p)[1:]), int(str(s)[1:])

def build_embed_dataset(cid: int, label_csv: str):
    """
    index.csv: subject_id, study_id, has_img, has_txt
    repr_img.npy, repr_txt.npy: 각 모달이 있는 샘플만 순서대로 쌓임
    """
    Xi, Xt, rows = load_embeddings_and_index(cid)
    if rows is None or (Xi is None and Xt is None):
        raise FileNotFoundError(f"[client_{cid:02d}] 임베딩/인덱스 누락")

    i_ptr = 0; t_ptr = 0
    Ximg_list, Xtxt_list, Y_list, mask_img, mask_txt = [], [], [], [], []

    label_table = load_label_table(label_csv)

    for r in rows:
        sid, stid = idstr_to_ints(r["subject_id"], r["study_id"])
        y = label_table.get((sid, stid), [0]*len(LABEL_COLUMNS))
        has_img = int(r.get("has_img", 0)) == 1 and Xi is not None
        has_txt = int(r.get("has_txt", 0)) == 1 and Xt is not None

        if has_img and i_ptr < (0 if Xi is None else Xi.shape[0]):
            Ximg_list.append(Xi[i_ptr]); i_ptr += 1; mask_img.append(1)
        else:
            Ximg_list.append(np.zeros((0,), dtype=np.float32)); mask_img.append(0)

        if has_txt and t_ptr < (0 if Xt is None else Xt.shape[0]):
            Xtxt_list.append(Xt[t_ptr]); t_ptr += 1; mask_txt.append(1)
        else:
            Xtxt_list.append(np.zeros((0,), dtype=np.float32)); mask_txt.append(0)

        Y_list.append(y)

    Y = torch.tensor(np.array(Y_list, dtype=np.float32), device=DEVICE)   # [N, C]
    N = len(rows)
    Di = 0 if Xi is None else Xi.shape[1]
    Dt = 0 if Xt is None else Xt.shape[1]

    def pad_stack(lst, D):
        if D == 0: return torch.zeros((N, 0), device=DEVICE)
        X = np.zeros((N, D), dtype=np.float32)
        for idx in range(N):
            v = lst[idx]
            if v.shape == (0,): continue
            X[idx] = v
        return torch.tensor(X, device=DEVICE)

    Ximg = pad_stack(Ximg_list, Di)   # (N, Di)
    Xtxt = pad_stack(Xtxt_list, Dt)   # (N, Dt)
    m_img = torch.tensor(np.array(mask_img, dtype=np.float32), device=DEVICE)  # (N,)
    m_txt = torch.tensor(np.array(mask_txt, dtype=np.float32), device=DEVICE)  # (N,)

    # 현재 임베딩 보유 상태로 동적 모드 판정
    has_any_img = (Di > 0) and (m_img.sum().item() > 0)
    has_any_txt = (Dt > 0) and (m_txt.sum().item() > 0)
    if has_any_img and has_any_txt:
        mode = "multimodal"
    elif has_any_img:
        mode = "image_only"
    elif has_any_txt:
        mode = "text_only"
    else:
        raise RuntimeError(f"[client_{cid:02d}] 사용 가능한 임베딩이 없습니다.")

    return Ximg, Xtxt, Y, m_img, m_txt, mode

# ---------------------------
# Z 로드 & 정렬 손실(게이팅)
# ---------------------------
def load_Z_for_client(cid: int) -> Optional[torch.Tensor]:
    p = cdir(cid) / "global_payload.json"
    if not p.exists(): return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    z_path = payload.get("Z_path") or payload.get("Z_proxy_text_path") or payload.get("Z_proxy_image_path")
    if not z_path or not os.path.exists(z_path): return None
    Z = torch.from_numpy(np.load(z_path)).float().to(DEVICE)
    Z = torch.nn.functional.normalize(Z, dim=1)
    return Z

def _pad_to_dim(X: torch.Tensor, D: int) -> torch.Tensor:
    Dx = X.shape[1]
    if Dx == D: return X
    pad = torch.zeros((X.size(0), D - Dx), device=X.device, dtype=X.dtype)
    return torch.cat([X, pad], dim=1)

def _fused_feature(Fi: Optional[torch.Tensor],
                   Ft: Optional[torch.Tensor],
                   mi: torch.Tensor, mt: torch.Tensor) -> Optional[torch.Tensor]:
    if Fi is None and Ft is None: return None
    if Fi is None: return Ft
    if Ft is None: return Fi
    D = max(Fi.shape[1], Ft.shape[1])
    Fi_p = torch.nn.functional.normalize(_pad_to_dim(Fi, D), dim=1)
    Ft_p = torch.nn.functional.normalize(_pad_to_dim(Ft, D), dim=1)
    denom = (mi + mt).clamp(min=1.0)
    return (Fi_p * mi + Ft_p * mt) / denom

def alignment_loss_with_gating(F: torch.Tensor, Z: torch.Tensor,
                               base: float = 0.1, tau: float = 0.1) -> torch.Tensor:
    if (Z is None) or (F is None) or (F.numel() == 0):
        dev = F.device if isinstance(F, torch.Tensor) else (Z.device if isinstance(Z, torch.Tensor) else "cpu")
        return torch.tensor(0.0, device=dev)
    D = max(F.shape[1], Z.shape[1])
    Fp = torch.nn.functional.normalize(_pad_to_dim(F, D), dim=1)
    Zp = torch.nn.functional.normalize(_pad_to_dim(Z, D), dim=1)
    sim = Fp @ Zp.t()                 # (B, K)
    sim_max, _ = sim.max(dim=1)       # (B,)
    gate = torch.sigmoid((1.0 - sim_max) / max(tau, 1e-6))
    loss_align = (1.0 - sim_max) * gate
    return base * loss_align.mean()

# ---------------------------
# baseline 헤드 로딩 & 로짓 계산
# ---------------------------
def load_baseline_heads(cid: int, Di: int, Dt: int):
    """
    best.pt에서 분류 헤드 가중치를 로드.
    - 멀티모달/싱글모달 저장 케이스 모두 커버
    - 입력 임베딩 차원(Di, Dt)과 가중치 shape이 맞는 경우에만 채택
    - 하나라도 안 맞으면 해당 모달은 scratch 초기화로 진행
    """
    ckpt_path = cdir(cid) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"[client_{cid:02d}] best.pt 없음: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"]

    W_img = b_img = W_txt = b_txt = None

    # ----- image head 후보들 -----
    # 멀티모달 저장형
    if "img.head.weight" in sd and "img.head.bias" in sd:
        if Di > 0 and sd["img.head.weight"].shape[1] == Di:
            W_img = sd["img.head.weight"]; b_img = sd["img.head.bias"]
    # 이미지 전용 저장형
    if W_img is None and "head.weight" in sd and "head.bias" in sd:
        if Di > 0 and sd["head.weight"].shape[1] == Di:
            W_img = sd["head.weight"]; b_img = sd["head.bias"]

    # ----- text head 후보들 -----
    # 멀티모달 저장형
    if "txt.cls_head.weight" in sd and "txt.cls_head.bias" in sd:
        if Dt > 0 and sd["txt.cls_head.weight"].shape[1] == Dt:
            W_txt = sd["txt.cls_head.weight"]; b_txt = sd["txt.cls_head.bias"]
    # 텍스트 전용 저장형
    if W_txt is None and "cls_head.weight" in sd and "cls_head.bias" in sd:
        if Dt > 0 and sd["cls_head.weight"].shape[1] == Dt:
            W_txt = sd["cls_head.weight"]; b_txt = sd["cls_head.bias"]

    # 디바이스로 이동
    def to_dev(x): return None if x is None else x.to(DEVICE).float()
    W_img, b_img, W_txt, b_txt = map(to_dev, (W_img, b_img, W_txt, b_txt))

    # 디버그: 둘 다 못 불러오면 키 목록을 한 번 출력해 줌
    if W_img is None and Di > 0:
        print(f"[WARN] client_{cid:02d} image head 불일치 → scratch (Di={Di}). "
              f"가능키: {[k for k in sd.keys() if 'head' in k]}")
    if W_txt is None and Dt > 0:
        print(f"[WARN] client_{cid:02d} text head 불일치 → scratch (Dt={Dt}). "
              f"가능키: {[k for k in sd.keys() if 'cls_head' in k or 'head' in k]}")

    return W_img, b_img, W_txt, b_txt


def logits_from_heads(Ximg, Xtxt, m_img, m_txt, W_img, b_img, W_txt, b_txt):
    """
    모달 마스크 기반 late fusion:
      logits = (li*mi + lt*mt) / clamp(mi+mt, min=1)
    가중치/특징 차원이 안맞으면 해당 모달은 자동 무시.
    """
    N = Ximg.shape[0] if Ximg is not None else Xtxt.shape[0]
    C = len(LABEL_COLUMNS)
    logits = torch.zeros((N, C), device=DEVICE)

    li = None
    lt = None
    if (W_img is not None) and (Ximg is not None) and (Ximg.shape[1] == W_img.shape[1]):
        li = Ximg @ W_img.t() + b_img
    if (W_txt is not None) and (Xtxt is not None) and (Xtxt.shape[1] == W_txt.shape[1]):
        lt = Xtxt @ W_txt.t() + b_txt

    mi = m_img.unsqueeze(1) if m_img is not None else torch.zeros((N,1), device=DEVICE)
    mt = m_txt.unsqueeze(1) if m_txt is not None else torch.zeros((N,1), device=DEVICE)

    if li is not None: logits = logits + li * mi
    if lt is not None: logits = logits + lt * mt

    denom = (mi + mt).clamp(min=1.0)
    logits = logits / denom
    return logits

# ---------------------------
# 학습용 선형 헤드 + 평가 유틸
# ---------------------------
class LinearHead(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_out)
    def forward(self, X):
        return self.fc(X)

def make_heads(Di, Dt, C, init_from=None):
    W_img0, b_img0, W_txt0, b_txt0 = init_from or (None, None, None, None)
    head_img = LinearHead(Di, C).to(DEVICE) if Di > 0 else None
    head_txt = LinearHead(Dt, C).to(DEVICE) if Dt > 0 else None
    with torch.no_grad():
        if head_img and (W_img0 is not None) and (b_img0 is not None) and (head_img.fc.weight.shape == W_img0.shape):
            head_img.fc.weight.copy_(W_img0); head_img.fc.bias.copy_(b_img0)
        if head_txt and (W_txt0 is not None) and (b_txt0 is not None) and (head_txt.fc.weight.shape == W_txt0.shape):
            head_txt.fc.weight.copy_(W_txt0); head_txt.fc.bias.copy_(b_txt0)
    return head_img, head_txt

def forward_heads(Ximg, Xtxt, m_img, m_txt, head_img, head_txt):
    N = Ximg.shape[0] if Ximg is not None else Xtxt.shape[0]
    C = len(LABEL_COLUMNS)
    logits = torch.zeros((N, C), device=DEVICE)
    li = head_img(Ximg) if head_img is not None else None
    lt = head_txt(Xtxt) if head_txt is not None else None
    mi = m_img.unsqueeze(1) if m_img is not None else torch.zeros((N,1), device=DEVICE)
    mt = m_txt.unsqueeze(1) if m_txt is not None else torch.zeros((N,1), device=DEVICE)
    if li is not None: logits = logits + li * mi
    if lt is not None: logits = logits + lt * mt
    denom = (mi + mt).clamp(min=1.0)
    return logits / denom

def evaluate_heads(Ximg, Xtxt, Y, m_img, m_txt, ids, head_img, head_txt):
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = forward_heads(Ximg[ids], Xtxt[ids], m_img[ids], m_txt[ids], head_img, head_txt)
        y = Y[ids]
        loss = crit(logits, y).item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy().astype(np.int32)
        y_pred = (probs >= 0.5).astype(np.int32)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:,c]
            if len(np.unique(col))<2: continue
            try:
                aucs.append(roc_auc_score(col, probs[:,c]))
            except:
                pass
        auc_macro = float(np.mean(aucs)) if len(aucs)>0 else float("nan")
        return {"loss": loss, "f1_micro": float(f1_micro), "f1_macro": float(f1_macro), "auc_macro": auc_macro}

# ---------------------------
# 학습 루프(임베딩 1~몇 에폭, Z 정렬 포함)
# ---------------------------
def train_heads(
    Ximg, Xtxt, Y, m_img, m_txt,
    head_img, head_txt,
    Z: Optional[torch.Tensor], z_weight: float, tau: float,
    tr_idx: torch.Tensor, va_idx: torch.Tensor,
    lr: float = 5e-4, epochs: int = 1
):
    params = []
    if head_img: params += list(head_img.parameters())
    if head_txt: params += list(head_txt.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()

    # 특징 융합(정렬 손실용) 준비
    Di = 0 if Ximg is None else Ximg.shape[1]
    Dt = 0 if Xtxt is None else Xtxt.shape[1]

    for ep in range(1, epochs+1):
        opt.zero_grad()
        y = Y[tr_idx]
        logits = forward_heads(Ximg[tr_idx], Xtxt[tr_idx], m_img[tr_idx], m_txt[tr_idx], head_img, head_txt)
        loss = crit(logits, y)

        # F: late-fused feature for alignment (입력 임베딩 기반)
        Fi = Ximg[tr_idx] if Di>0 else None
        Ft = Xtxt[tr_idx] if Dt>0 else None
        mi = m_img[tr_idx].unsqueeze(1)
        mt = m_txt[tr_idx].unsqueeze(1)
        F = _fused_feature(Fi, Ft, mi, mt)
        if Z is not None and F is not None:
            loss = loss + alignment_loss_with_gating(F, Z, base=z_weight, tau=tau)

        loss.backward()
        opt.step()

        va = evaluate_heads(Ximg, Xtxt, Y, m_img, m_txt, va_idx, head_img, head_txt)
        print(f"[Ep {ep}/{epochs}] train_loss={loss.item():.4f} | val {va}")

    return evaluate_heads(Ximg, Xtxt, Y, m_img, m_txt, va_idx, head_img, head_txt)

# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=str, default="1-20", help='예: "1-20" 또는 "1,3,5"')
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--z_weight", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    # client id 파싱
    text = args.client_id.strip().lower()
    ids = []
    if text in ("all",""):
        ids = list(range(1,21))
    else:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a,b = part.split("-",1)
                ids.extend(list(range(int(a), int(b)+1)))
            else:
                ids.append(int(part))
        ids = sorted(list(dict.fromkeys(ids)))

    out_csv = Path("./outputs/update_embed_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for cid in ids:
        print("\n" + "="*70)
        print(f"▶ Client {cid:02d} | build dataset from embeddings")
        print("="*70)

        try:
            Ximg, Xtxt, Y, m_img, m_txt, mode_now = build_embed_dataset(cid, args.label_csv)
        except Exception as e:
            print(f"[SKIP client {cid:02d}] {e}")
            continue

        # baseline 헤드 로딩(있으면 로딩, 없으면 None)
        try:
            Di = Ximg.shape[1]
            Dt = Xtxt.shape[1]
            W_img, b_img, W_txt, b_txt = load_baseline_heads(cid, Di, Dt)
        except Exception as e:
            print(f"[WARN] best.pt 로딩 실패 → scratch 초기화로 진행 ({e})")
            W_img = b_img = W_txt = b_txt = None

        # 동일한 분할(10% val)에서 baseline vs updated 비교
        N = Y.shape[0]
        idx = np.arange(N); rng = np.random.RandomState(SEED); rng.shuffle(idx)
        va_idx = torch.tensor(idx[:max(1, int(N*0.1))], device=DEVICE)
        tr_idx = torch.tensor(idx[max(1, int(N*0.1)):], device=DEVICE)

        # Baseline 평가 (best.pt 헤드 그대로)
        base_head_img, base_head_txt = make_heads(
            Ximg.shape[1], Xtxt.shape[1], len(LABEL_COLUMNS),
            init_from=(W_img, b_img, W_txt, b_txt)
        )
        baseline_val = evaluate_heads(Ximg, Xtxt, Y, m_img, m_txt, va_idx, base_head_img, base_head_txt)
        print("[Baseline on val] ", baseline_val)

        # Updated 학습(Z 정렬 + 게이팅 유지, late fusion 유지)
        Z = load_Z_for_client(cid)
        upd_head_img, upd_head_txt = make_heads(
            Ximg.shape[1], Xtxt.shape[1], len(LABEL_COLUMNS),
            init_from=(W_img, b_img, W_txt, b_txt)
        )
        updated_val = train_heads(
            Ximg, Xtxt, Y, m_img, m_txt,
            upd_head_img, upd_head_txt,
            Z=Z, z_weight=args.z_weight, tau=args.tau,
            tr_idx=tr_idx, va_idx=va_idx,
            lr=args.lr, epochs=args.epochs
        )
        print("[Updated on val]", updated_val)

        # 업데이트된 헤드 저장
        save_path = cdir(cid)/"updated_heads.npz"
        out_npz = {}
        if upd_head_img is not None:
            out_npz["W_img"] = upd_head_img.fc.weight.detach().cpu().numpy()
            out_npz["b_img"] = upd_head_img.fc.bias.detach().cpu().numpy()
        if upd_head_txt is not None:
            out_npz["W_txt"] = upd_head_txt.fc.weight.detach().cpu().numpy()
            out_npz["b_txt"] = upd_head_txt.fc.bias.detach().cpu().numpy()
        if out_npz:
            np.savez(save_path, **out_npz)
            print(f"[client {cid:02d}] updated heads saved → {save_path}")

        rows.append({
            "client_id": cid,
            "mode_now": mode_now,
            "baseline_loss": baseline_val["loss"], "updated_loss": updated_val["loss"],
            "baseline_f1_micro": baseline_val["f1_micro"], "updated_f1_micro": updated_val["f1_micro"],
            "baseline_f1_macro": baseline_val["f1_macro"], "updated_f1_macro": updated_val["f1_macro"],
            "baseline_auc_macro": baseline_val["auc_macro"], "updated_auc_macro": updated_val["auc_macro"],
        })

    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[DONE] Summary saved → {out_csv} (rows={len(rows)})")
    else:
        print("\n[WARN] no clients processed; nothing saved.")

if __name__ == "__main__":
    main()
