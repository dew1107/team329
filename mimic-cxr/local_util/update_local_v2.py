# -*- coding: utf-8 -*-
"""
update_v2.py (All-in-One: Re-load, Align, Update)

- [파일 불일치 문제 해결]
  'repr_*.npy' 파일을 더 이상 사용하지 않습니다.
  'train_local.py'가 만든 'best.pt'와 'client_splits/*.csv' 원본 데이터를 직접 로드합니다.
  특징(F)과 헤드(W)가 항상 100% 동일한 버전을 사용하도록 보장합니다.

- [훈련]
  1. best.pt 모델의 '몸통(backbone)'은 동결(freeze)합니다.
  2. '머리(head)'만 Z 벡터와 정렬(alignment)하며 재훈련합니다.
  3. 차원 불일치(576D vs 256D) 문제를 PCA 투영으로 자동 해결합니다.

- [평가]
  'prep_clients.py'와 동일한 90/10 분할을 사용하여
  'val_loader' (10% 검증 세트)에서 베이스라인과 최종 성능을 측정합니다.
  (이제 Baseline loss가 train_local/prep_clients 로그와 일치해야 합니다.)
"""

import os, json, argparse, csv, random
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA

# --- GPU 모니터링 ---
import threading, time

try:
    from pynvml import *
except ImportError:
    print("[WARN] pynvml not installed. GPU monitoring will be disabled.")
    NVML_TEMPERATURE_GPU = 0;
    pass
# --------------------
# --- train_local.py, prep_clients.py에서 설정/함수 가져오기 ---
from local_util.train_local import (
    CLIENT_CSV_DIR,
    ClientDataset,
    LABEL_COLUMNS,
    load_label_table,
    build_image_picker_from_metadata,
    LABEL_CSV as DEFAULT_LABEL_CSV,
    METADATA_CSV,
    IMG_ROOT
)
from global_util.prep_clients import (
    build_model_from_ckpt,  # best.pt 로더
    decide_mode_for_cid
)

# ---------------------------
# 전역 상수/설정
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED);
np.random.seed(SEED);
random.seed(SEED)
N_CLASSES = len(LABEL_COLUMNS)


# ---------------------------
# GPU 모니터
# ---------------------------
class GpuMonitor:
    # ... (이전과 동일한 GpuMonitor 클래스 코드) ...
    def __init__(self, log_file_path: str, device_index: int = 0, interval_sec: int = 2):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.log_file_path = log_file_path
        self.active = False
        self.thread = None
        self.handle = None
        self.csv_file = None
        self.csv_writer = None
        self.pynvml_available = False

    def _init_pynvml(self):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(self.device_index)
            self.pynvml_available = True
            print(f"[GpuMonitor] pynvml initialized. Monitoring GPU {self.device_index}.")
            return True
        except Exception as e:
            print(f"[GpuMonitor] Error initializing pynvml: {e}. Monitoring disabled.")
            self.pynvml_available = False
            return False

    def _get_stats(self) -> Dict:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        try:
            util = nvmlDeviceGetUtilizationRates(self.handle)
            mem = nvmlDeviceGetMemoryInfo(self.handle)
            power_mw = nvmlDeviceGetPowerUsage(self.handle)
            temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
            return {
                "timestamp": timestamp, "gpu_util_pct": util.gpu, "mem_util_pct": util.memory,
                "mem_used_mb": mem.used / (1024 ** 2), "mem_total_mb": mem.total / (1024 ** 2),
                "power_w": power_mw / 1000.0, "temp_c": temp
            }
        except Exception as e:
            return {"timestamp": timestamp, "gpu_util_pct": -1, "mem_util_pct": -1, "mem_used_mb": -1,
                    "mem_total_mb": -1, "power_w": -1, "temp_c": -1}

    def _monitor_loop(self):
        try:
            headers = ["timestamp", "gpu_util_pct", "mem_util_pct", "mem_used_mb", "mem_total_mb", "power_w", "temp_c"]
            self.csv_file = open(self.log_file_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
            self.csv_writer.writeheader()
            while self.active:
                stats = self._get_stats()
                self.csv_writer.writerow(stats);
                self.csv_file.flush()
                time.sleep(self.interval_sec)
        except Exception as e:
            print(f"[GpuMonitor] Error in monitor loop: {e}")
        finally:
            if self.csv_file: self.csv_file.close()
            try:
                nvmlShutdown();
                print("[GpuMonitor] pynvml shutdown.")
            except:
                pass

    def start(self):
        if self.active: return
        if not self._init_pynvml(): return
        if not self.pynvml_available: return
        self.active = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"[GpuMonitor] Monitor thread started. Logging to: {self.log_file_path}")

    def stop(self):
        if not self.active or not self.pynvml_available: return
        print("[GpuMonitor] Stopping monitor thread...")
        self.active = False
        if self.thread:
            self.thread.join(timeout=self.interval_sec + 1)
        print("[GpuMonitor] Monitor stopped.")


# ---------------------------
# 경로/데이터 로더
# ---------------------------
def cdir(cid: int) -> Path:
    d = Path(f"./outputs/client_{cid:02d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_dataloaders(cid: int, label_csv_path: str, batch_size: int):
    """prep_clients.py와 동일한 로직으로 90/10 분할"""
    csv_path = Path(CLIENT_CSV_DIR) / f"client_{cid:02d}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[client_{cid}] CSV not found: {csv_path}")

    mode = decide_mode_for_cid(cid)
    label_table = load_label_table(label_csv_path)
    meta_picker = build_image_picker_from_metadata(METADATA_CSV, IMG_ROOT)

    ds_full = ClientDataset(str(csv_path), label_table, mode=mode, meta_picker=meta_picker)
    n = len(ds_full)
    n_tr = int(n * 0.9)
    n_va = n - n_tr
    tr_set, va_set = random_split(ds_full, [n_tr, n_va],
                                  generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[client_{cid:02d}] Data loaded: mode={mode}, total={n} (train={n_tr}, val={n_va})")
    return train_loader, val_loader, mode


# ---------------------------
# Z 로드 & 정렬 손실(게이팅)
# ---------------------------
def load_Z_for_client(cid: int) -> Optional[torch.Tensor]:
    p = cdir(cid) / "global_payload.json"
    if not p.exists():
        print(f"[client_{cid:02d}] global_payload.json not found. Z-Align disabled.")
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    z_path = payload.get("Z_path") or payload.get("Z_proxy_text_path") or payload.get("Z_proxy_image_path")
    if not z_path or not os.path.exists(z_path):
        print(f"[client_{cid:02d}] Z_path not found in payload. Z-Align disabled.")
        return None

    Z = torch.from_numpy(np.load(z_path)).float().to(DEVICE)
    Z = torch.nn.functional.normalize(Z, dim=1)
    print(f"[client_{cid:02d}] Z loaded: shape={tuple(Z.shape)}")
    return Z


def _pad_to_dim(X: torch.Tensor, D: int) -> torch.Tensor:
    Dx = X.shape[1]
    if Dx == D: return X
    pad = torch.zeros((X.size(0), D - Dx), device=X.device, dtype=X.dtype)
    return torch.cat([X, pad], dim=1)


def alignment_loss_with_gating(F: torch.Tensor, Z: torch.Tensor,
                               base: float = 0.1, tau: float = 0.1) -> torch.Tensor:
    if (Z is None) or (F is None) or (F.numel() == 0):
        return torch.tensor(0.0, device=DEVICE)
    D = max(F.shape[1], Z.shape[1])
    Fp = torch.nn.functional.normalize(_pad_to_dim(F, D), dim=1)
    Zp = torch.nn.functional.normalize(_pad_to_dim(Z, D), dim=1)
    sim = Fp @ Zp.t()  # (B, K)
    sim_max, _ = sim.max(dim=1)  # (B,)
    gate = torch.sigmoid((1.0 - sim_max) / max(tau, 1e-6))
    loss_align = (1.0 - sim_max) * gate
    return base * loss_align.mean()


# ---------------------------
# 차원 투영 (PCA)
# ---------------------------
def project_tensor(X: torch.Tensor, d: int) -> torch.Tensor:
    if X.numel() == 0: return torch.zeros((0, d), dtype=X.dtype, device=X.device)
    X_np = X.detach().cpu().numpy()
    N, D = X_np.shape
    if d == D: return X
    Y_np = None
    if N >= 2 and D > 1:
        n_components_for_pca = min(d, min(N, D))
        try:
            pca = PCA(n_components=n_components_for_pca, random_state=SEED)
            Y_np = pca.fit_transform(X_np).astype(np.float32)
        except Exception as e:
            Y_np = None
    if Y_np is None:
        if D > d:
            Y_np = X_np[:, :d]
        else:
            Y_np = np.concatenate([X_np, np.zeros((N, d - D), dtype=np.float32)], axis=1)
    Y = torch.from_numpy(Y_np).to(X.device)
    if Y.shape[1] < d:
        Y = torch.cat([Y, torch.zeros((Y.shape[0], d - Y.shape[1]), device=Y.device, dtype=Y.dtype)], axis=1)
    elif Y.shape[1] > d:
        Y = Y[:, :d]
    return Y


# ---------------------------
# 평가/훈련 루프 (V2)
# ---------------------------
@torch.no_grad()
def evaluate_v2(model, loader, mode: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    logits_all, labels_all = [], []
    total_loss = 0.0

    for batch in loader:
        y = batch["labels"].to(device)
        logits = None
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
        return {"loss": float('nan'), "f1_micro": 0.0, "f1_macro": 0.0, "auc_macro": float("nan")}

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    # metrics
    probs = torch.sigmoid(logits_all).numpy()
    y_true = labels_all.numpy().astype(np.int32)
    y_pred = (probs >= 0.5).astype(np.int32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if len(np.unique(col)) < 2: continue
        try:
            aucs.append(roc_auc_score(col, probs[:, c]))
        except Exception:
            pass
    auc_macro = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

    return {
        "loss": total_loss / len(loader.dataset),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "auc_macro": auc_macro
    }


def train_v2(cid: int, model, train_loader, val_loader, mode: str, Z: torch.Tensor, args: argparse.Namespace):
    # 1. 몸통(Backbone) 동결, 머리(Head)만 훈련 대상으로 설정
    params = []
    if mode == "multimodal":
        if hasattr(model, "img") and hasattr(model.img, "backbone"): model.img.backbone.requires_grad_(False)
        if hasattr(model, "txt") and hasattr(model.txt, "enc"): model.txt.enc.requires_grad_(False)
        if hasattr(model, "img") and hasattr(model.img, "head"): params += list(model.img.head.parameters())
        if hasattr(model, "txt") and hasattr(model.txt, "cls_head"): params += list(model.txt.cls_head.parameters())
    elif mode == "image_only":
        if hasattr(model, "backbone"): model.backbone.requires_grad_(False)
        if hasattr(model, "head"): params += list(model.head.parameters())
    elif mode == "text_only":
        if hasattr(model, "enc"): model.enc.requires_grad_(False)
        if hasattr(model, "cls_head"): params += list(model.cls_head.parameters())

    if not params:
        print(f"[WARN client_{cid}] No head parameters found to train. Skipping.")
        return

    opt = torch.optim.AdamW(params, lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    d_z = Z.shape[1] if Z is not None else 0

    # 2. 훈련 루프
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss_main = 0.0
        total_loss_z = 0.0

        for batch in train_loader:
            opt.zero_grad()
            y = batch["labels"].to(DEVICE)
            logits = None
            if mode == "multimodal":
                img = batch["image"].to(DEVICE)
                txt = {k: v.to(DEVICE) for k, v in batch["text"].items()}
                logits = model(image=img, text=txt)
            elif mode == "image_only":
                logits = model(batch["image"].to(DEVICE))
            else:
                txt = {k: v.to(DEVICE) for k, v in batch["text"].items()}
                logits = model(**txt)

            # 3. 손실 계산
            loss_main = crit(logits, y)
            loss_z = torch.tensor(0.0, device=DEVICE)

            # 4. Z-Loss (PCA 투영 + head.weight 비교)
            if Z is not None:
                if mode == "multimodal":
                    w_img_proj = project_tensor(model.img.head.weight, d_z)
                    w_txt_proj = project_tensor(model.txt.cls_head.weight, d_z)
                    loss_z_img = alignment_loss_with_gating(w_img_proj, Z, base=args.z_weight, tau=args.tau)
                    loss_z_txt = alignment_loss_with_gating(w_txt_proj, Z, base=args.z_weight, tau=args.tau)
                    loss_z = 0.5 * (loss_z_img + loss_z_txt)
                elif mode == "image_only":
                    w_img_proj = project_tensor(model.head.weight, d_z)
                    loss_z = alignment_loss_with_gating(w_img_proj, Z, base=args.z_weight, tau=args.tau)
                elif mode == "text_only":
                    w_txt_proj = project_tensor(model.cls_head.weight, d_z)
                    loss_z = alignment_loss_with_gating(w_txt_proj, Z, base=args.z_weight, tau=args.tau)

            loss = loss_main + loss_z
            loss.backward()
            opt.step()

            total_loss_main += loss_main.item() * y.size(0)
            total_loss_z += loss_z.item() * y.size(0)

        # 5. 에포크마다 검증(val)
        val_metrics = evaluate_v2(model, val_loader, mode, DEVICE)
        print(f"[Ep {ep}/{args.epochs}] "
              f"train_loss_main={total_loss_main / len(train_loader.dataset):.4f} | "
              f"train_loss_z={total_loss_z / len(train_loader.dataset):.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_f1_macro={val_metrics['f1_macro']:.4f}")


# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_id", type=str, default="1-20", help='예: "1-20" 또는 "1,3,5"')
    ap.add_argument("--label_csv", type=str, default=DEFAULT_LABEL_CSV)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--z_weight", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    # client id 파싱
    text = args.client_id.strip().lower()
    ids = []
    if text in ("all", ""):
        ids = list(range(1, 21))
    else:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1); ids.extend(list(range(int(a), int(b) + 1)))
            else:
                ids.append(int(part))
        ids = sorted(list(dict.fromkeys(ids)))

    out_csv = Path("./outputs/update_v2_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    for cid in ids:
        print("\n" + "=" * 70)
        print(f"▶ Client {cid:02d} | V2 Update starting...")
        print("=" * 70)

        monitor = None
        if DEVICE.type == "cuda":
            log_path = cdir(cid) / f"client_{cid:02d}_gpu_update_v2_log.csv"
            monitor = GpuMonitor(str(log_path), device_index=0, interval_sec=2)
            monitor.start()

        try:
            # 1. 데이터 로더 (원본 CSV 90/10 분할)
            train_loader, val_loader, mode = build_dataloaders(cid, args.label_csv, args.batch_size)

            # 2. best.pt 모델 로드
            ckpt_path = str(cdir(cid) / "best.pt")
            model, mode_ckpt = build_model_from_ckpt(cid, ckpt_path, DEVICE)

            # 3. Z 로드
            Z = load_Z_for_client(cid)

            # 4. 베이스라인 성능 측정 (파일 꼬임 없는 '진짜' 성능)
            baseline_val = evaluate_v2(model, val_loader, mode, DEVICE)
            print(f"[client_{cid:02d}] Baseline (on 10% val): {baseline_val}")

            # 5. Z 정렬 훈련 (몸통 동결, 머리만)
            train_v2(cid, model, train_loader, val_loader, mode, Z, args)

            # 6. 최종 성능 측정
            updated_val = evaluate_v2(model, val_loader, mode, DEVICE)
            print(f"[client_{cid:02d}] Updated (on 10% val): {updated_val}")

            # 7. 업데이트된 헤드 저장
            save_path = cdir(cid) / "updated_heads_v2.pt"
            torch.save(model.state_dict(), save_path)
            print(f"[client {cid:02d}] updated full model (head only trained) saved → {save_path}")

            rows_out.append({
                "client_id": cid,
                "mode_now": mode,
                "baseline_loss": baseline_val["loss"], "updated_loss": updated_val["loss"],
                "baseline_f1_micro": baseline_val["f1_micro"], "updated_f1_micro": updated_val["f1_micro"],
                "baseline_f1_macro": baseline_val["f1_macro"], "updated_f1_macro": updated_val["f1_macro"],
                "baseline_auc_macro": baseline_val["auc_macro"], "updated_auc_macro": updated_val["auc_macro"],
            })

        except Exception as e:
            print(f"[SKIP client {cid:02d}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        finally:
            if monitor:
                monitor.stop()
            print("=" * 70)

    if rows_out:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader();
            w.writerows(rows_out)
        print(f"\n[DONE] Summary saved → {out_csv} (rows={len(rows_out)})")
    else:
        print("\n[WARN] no clients processed; nothing saved.")


if __name__ == "__main__":
    main()