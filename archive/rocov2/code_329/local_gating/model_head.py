# local_gating/model_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalLocalGatedFusion(nn.Module):
    def __init__(self, d_local, d_global, d_out):
        super().__init__()
        self.proj_local  = nn.Linear(d_local,  d_out)
        self.proj_global = nn.Linear(d_global, d_out)
        self.Wi = nn.Linear(d_out * 2, d_out)
        self.Wf = nn.Linear(d_out * 2, d_out)

    def forward(self, R, Z):
        if Z.dim() == 1:
            Z = Z.unsqueeze(0).expand(R.size(0), -1)
        r = self.proj_local(R); z = self.proj_global(Z)
        cat = torch.cat([r, z], dim=-1)
        i = torch.sigmoid(self.Wi(cat))
        f = torch.sigmoid(self.Wf(cat))
        return f * z + i * r

class LocalClassifierWithAugment(nn.Module):
    """
    모달리티 결손 보충:
    - 이미지 전용: Z → 텍스트 feature(128)로 hallucinate
    - 텍스트 전용: Z → 이미지 feature(256)로 hallucinate
    """
    def __init__(self, d_img, d_txt, d_local_fused, d_global, n_labels, use_hallucinate=False):
        super().__init__()
        self.use_hallucinate = use_hallucinate # hallucination true/false 비교 필요
        self.proj_img = nn.Linear(d_img, 256)
        self.proj_txt = nn.Linear(d_txt, 128)
        # Z로부터 결손 모달 feature를 직접 생성
        self.z2img = nn.Linear(d_global, 256)
        self.z2txt = nn.Linear(d_global, 128)

        self.fuse = nn.Linear(256+128, d_local_fused)
        self.gate = GlobalLocalGatedFusion(d_local_fused, d_global, d_local_fused)
        self.cls  = nn.Linear(d_local_fused, n_labels)

    def forward(self, img_rep=None, txt_rep=None, Z_global=None, mode="auto"):
        if Z_global is None:
            raise ValueError("Z_global is required")

        if img_rep is None and txt_rep is not None:
            # 이미지 결손: txt만 있음
            img_feat = (F.relu(self.z2img(Z_global))
                        if self.use_hallucinate else
                        torch.zeros(txt_rep.size(0), 256, device=txt_rep.device))
            if img_feat.dim() == 1: img_feat = img_feat.unsqueeze(0).expand(txt_rep.size(0), -1)
            txt_feat = F.relu(self.proj_txt(txt_rep))
        elif txt_rep is None and img_rep is not None:
            # 텍스트 결손: img만 있음
            txt_feat = (F.relu(self.z2txt(Z_global))
                        if self.use_hallucinate else
                        torch.zeros(img_rep.size(0), 128, device=img_rep.device))
            if txt_feat.dim() == 1: txt_feat = txt_feat.unsqueeze(0).expand(img_rep.size(0), -1)
            img_feat = F.relu(self.proj_img(img_rep))
        elif img_rep is not None and txt_rep is not None:
            img_feat = F.relu(self.proj_img(img_rep))
            txt_feat = F.relu(self.proj_txt(txt_rep))
        else:
            raise ValueError("at least one modality must be provided")

            # L2 정규화(스케일 폭주 방지)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        Zg = Z_global
        if Zg.dim() == 1: Zg = Zg.unsqueeze(0).expand(img_feat.size(0), -1)
        Zg = F.normalize(Zg, dim=-1)

        R = self.fuse(torch.cat([img_feat, txt_feat], dim=-1))
        H = self.gate(R, Zg)
        return self.cls(H), R
