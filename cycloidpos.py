
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple

# ===============================
# Cycloidal Positional Bias
# ===============================
class CycloidPositionalBias(nn.Module):
    def __init__(self, max_seq_len, learnable=True, init_r=1.0, init_alpha=0.4, init_sigma=1.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.r = nn.Parameter(torch.tensor(init_r), requires_grad=learnable)
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=learnable)
        self.sigma = nn.Parameter(torch.tensor(init_sigma), requires_grad=learnable)
        self.phase = nn.Parameter(torch.zeros(1))  # <-- new learnable phase offset

    def forward(self, seq_len, device=None, window=None):
        device = device or next(self.parameters()).device
        i = torch.arange(seq_len, device=device).float()
        t = self.alpha * i + self.phase  # add learnable offset
        if window is not None:
            return self._relative_window_bias(t, window)
        else:
            x = self.r * (t - torch.sin(t))
            y = self.r * (1 - torch.cos(t))
            diff = torch.cdist(torch.stack([x, y], dim=1), torch.stack([x, y], dim=1)) ** 2
            return -diff / (2 * self.sigma ** 2)

    def _relative_window_bias(self, t, window):
        seq_len = t.size(0)
        bias = torch.zeros(seq_len, seq_len, device=t.device)
        for offset in range(-window, window + 1):
            if offset == 0: continue
            dt = (t[window] - t[0]) * (offset / window)
            dx = self.r * (dt - torch.sin(dt))
            dy = self.r * (1 - torch.cos(dt))
            val = -(dx**2 + dy**2) / (2 * self.sigma**2)
            i = torch.arange(max(0, -offset), min(seq_len, seq_len - offset), device=t.device)
            bias[i, i + offset] = val
        return bias

