import math
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_q, d_model, bias=False)
        self.Wk = nn.Linear(d_k, d_model, bias=False)
        self.Wv = nn.Linear(d_v, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, Q, K, V):
        Qp = self.Wq(Q); Kp = self.Wk(K); Vp = self.Wv(V)
        A = torch.softmax(Qp @ Kp.T / self.scale, dim=-1)
        Z = (A @ Vp).mean(dim=0)          # [d_model]
        return Z

class GlobalLocalGatedFusion(nn.Module):
    """R(local)와 Z(global)를 LSTM 게이트 컨셉으로 결합."""
    def __init__(self, d_local, d_global, d_out):
        super().__init__()
        self.proj_local  = nn.Linear(d_local,  d_out)
        self.proj_global = nn.Linear(d_global, d_out)
        self.Wi = nn.Linear(d_out * 2, d_out)  # input gate (R 비중)
        self.Wf = nn.Linear(d_out * 2, d_out)  # forget gate (Z 비중)

    def forward(self, R, Z):
        if Z.dim() == 1:
            Z = Z.unsqueeze(0).expand(R.size(0), -1)
        r = self.proj_local(R); z = self.proj_global(Z)
        cat = torch.cat([r, z], dim=-1)
        i = torch.sigmoid(self.Wi(cat))
        f = torch.sigmoid(self.Wf(cat))
        return f * z + i * r
