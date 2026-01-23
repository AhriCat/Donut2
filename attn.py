# ===============================
# Focused Attention Group (Kronecker-friendly)
# ===============================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple

def _closest_factor_pair_int(n: int) -> Tuple[int, int]:
    root = int(math.sqrt(n))
    for delta in range(0, root + 1):
        b = root + delta
        if b > 0 and n % b == 0:
            return (b, n // b)
        b = root - delta
        if b > 0 and n % b == 0:
            return (b, n // b)
    return (1, n)
    
class FocusedAttentionGroup(nn.Module):
    """
    Kronecker-separable focused attention.
    Each head's rank `r` is factorized r = r1 * r2.
    We compute two attention logits A1 (via r1) and A2 (via r2),
    combine them elementwise, apply bias/softmax and compute attn @ V.
    """
    def __init__(self, dim, heads=8, groups=4, rank=32, dropout=0.1):
        super().__init__()
        assert heads % groups == 0
        self.dim = dim
        self.heads = heads
        self.groups = groups
        self.rank = rank

        # factorize rank into r1 * r2 (close to sqrt)
        self.r1, self.r2 = _closest_factor_pair_int(rank)
        self.scale = (self.r1 * self.r2) ** -0.5

        # projections: project to (heads, r1) and (heads, r2) separately
        # We spectral-norm the linear layers for stability (like original)
        self.q1_proj = spectral_norm(nn.Linear(dim, heads * self.r1, bias=False))
        self.k1_proj = spectral_norm(nn.Linear(dim, heads * self.r1, bias=False))

        self.q2_proj = spectral_norm(nn.Linear(dim, heads * self.r2, bias=False))
        self.k2_proj = spectral_norm(nn.Linear(dim, heads * self.r2, bias=False))

        # v uses full head value dim (like before)
        self.v_proj = spectral_norm(nn.Linear(dim, dim, bias=False))
        self.out_proj = spectral_norm(nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias=None):
        # x: [B, N, C]
        B, N, C = x.shape
        H = self.heads

        # q/k factor 1
        q1 = self.q1_proj(x).view(B, N, H, self.r1)   # B,N,H,r1
        k1 = self.k1_proj(x).view(B, N, H, self.r1)   # B,N,H,r1

        # q/k factor 2
        q2 = self.q2_proj(x).view(B, N, H, self.r2)   # B,N,H,r2
        k2 = self.k2_proj(x).view(B, N, H, self.r2)   # B,N,H,r2

        # v: split per-head value dim
        head_dim = C // H
        v = self.v_proj(x).view(B, N, H, head_dim)    # B,N,H,head_dim

        # compute small attn matrices
        # A1: B,H,N,N  <- einsum over r1
        A1 = torch.einsum('bnhd,bmhd->bhnm', q1, k1)  # (B, H, N, N) using r1
        # A2: B,H,N,N  <- einsum over r2
        A2 = torch.einsum('bnhd,bmhd->bhnm', q2, k2)  # (B, H, N, N) using r2

        logits = (A1 * A2) * self.scale

        if bias is not None:
            # bias expected [N, N] or [H, N, N]; expand to B,H,N,N
            if bias.ndim == 2:
                logits = logits + bias.unsqueeze(0).unsqueeze(0)
            elif bias.ndim == 3 and bias.shape[0] == H:
                logits = logits + bias.unsqueeze(0)
            else:
                logits = logits + bias.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)  # B,H,N,head_dim
        out = out.reshape(B, N, C)
        return self.out_proj(out)
