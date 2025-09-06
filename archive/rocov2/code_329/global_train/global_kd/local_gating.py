import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import GlobalLocalGatedFusion

class LocalClassifierWithGate(nn.Module):
    def __init__(self, d_img, d_txt, d_local_fused, d_global, n_labels):
        super().__init__()
        self.proj_img = nn.Linear(d_img, 256)
        self.proj_txt = nn.Linear(d_txt, 128)
        self.fuse = nn.Linear(256+128, d_local_fused)
        self.gate = GlobalLocalGatedFusion(d_local_fused, d_global, d_local_fused)
        self.cls = nn.Linear(d_local_fused, n_labels)

    def forward(self, img_rep=None, txt_rep=None, Z_global=None):
        if img_rep is None and txt_rep is None:
            raise ValueError("at least one modality must be provided")
        if img_rep is None:
            img_feat = torch.zeros(txt_rep.size(0), 256, device=txt_rep.device)
        else:
            img_feat = F.relu(self.proj_img(img_rep))
        if txt_rep is None:
            txt_feat = torch.zeros(img_feat.size(0), 128, device=img_feat.device)
        else:
            txt_feat = F.relu(self.proj_txt(txt_rep))
        R = F.relu(self.fuse(torch.cat([img_feat, txt_feat], dim=-1)))
        H = self.gate(R, Z_global) if Z_global is not None else R
        return self.cls(H), R
