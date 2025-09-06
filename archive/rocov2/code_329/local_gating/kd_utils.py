import torch
import torch.nn.functional as F

def bce_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels)

def kd_logits_loss(student_logits, teacher_logits, T: float):
    if teacher_logits is None:
        return torch.tensor(0.0, device=student_logits.device)
    p = F.log_softmax(student_logits / T, dim=-1)
    q = F.softmax(teacher_logits / T, dim=-1)
    return (T * T) * F.kl_div(p, q, reduction="batchmean")

def kd_repr_loss(R_local, Z_global, weight: float):
    if Z_global is None or weight <= 0.0:
        return torch.tensor(0.0, device=R_local.device if isinstance(R_local, torch.Tensor) else "cpu")
    # Z_global: [d_model] 또는 [B, d_model]
    if Z_global.dim() == 1:
        Z_global = Z_global.unsqueeze(0).expand(R_local.size(0), -1)
    return weight * F.mse_loss(R_local, Z_global)

def compute_pos_weight(dataloader, num_labels):
    pos = torch.zeros(num_labels, dtype=torch.float64)
    neg = torch.zeros(num_labels, dtype=torch.float64)
    for b in dataloader:
        y = b["labels"].float()
        pos += y.sum(0)
        neg += (1.0 - y).sum(0)
    pw = torch.where(pos>0, neg / (pos + 1e-8), torch.ones_like(neg))
    return pw.float()
