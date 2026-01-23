import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple

class KroneckerTransform(nn.Module):
    """
    Learnable Kronecker-like linear transform:
        Y = B * X * A^T
    where A_lin: (b -> a), B_lin: (d -> c)
    Each wrapped with spectral_norm for stability.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.b, self.d = _closest_factor_pair(dim)
        self.a, self.c = self.b, self.d  # symmetric choice keeps output dim = input dim

        self.A_lin = nn.Linear(self.b, self.a, bias=False)
        self.B_lin = nn.Linear(self.d, self.c, bias=False)


        # fallback linear if shapes mismatch (rare)
        self.res_proj = spectral_norm(nn.Linear(self.a * self.c, dim)) if (self.a * self.c) != dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, dim]  (dim = b * d)
        returns: [B, N, dim]
        """
        B, N, D = x.shape
        assert D == self.dim, f"Input dim {D} != expected {self.dim}"

        # reshape each sequence element to matrix form [d, b]
        x_mat = x.reshape(B * N, self.d, self.b)          # (BN, d, b)

        # apply Kronecker factor maps
        # 1. apply A^T on columns (b -> a)
        XA = torch.matmul(x_mat, self.A_lin.weight.t())   # (BN, d, a)
        # 2. apply B on rows (d -> c)
        Y  = torch.matmul(self.B_lin.weight, XA)          # (c, d) @ (BN, d, a) -> (BN, c, a)
        # 3. flatten back
        Y  = Y.permute(0, 2, 1).contiguous().view(B, N, self.a * self.c)

        # ensure output dim matches input dim
        if self.res_proj is not None:
            Y = self.res_proj(Y)
        return Y
