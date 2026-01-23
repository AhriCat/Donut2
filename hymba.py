import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# HyMBA Block: State-space + RNN reasoning
# ===============================
class HyMBA_Block(nn.Module):
    def __init__(self, dim, ssm_dim=64, rnn_dim=128, dropout=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(ssm_dim, ssm_dim) * 0.02)
        self.B = nn.Linear(dim, ssm_dim)
        self.C = nn.Linear(ssm_dim, dim)
        self.D = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.GRUCell(dim, rnn_dim)
        self.rnn_proj = nn.Linear(rnn_dim, dim)
        self.gate = nn.Linear(dim, 2)
        self.register_buffer("_h_ssm", None, persistent=False)
        self.register_buffer("_h_rnn", None, persistent=False)

    def reset_state(self, B=1, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype  = dtype or next(self.parameters()).dtype
        self._h_ssm = torch.zeros(B, self.B.out_features, device=device, dtype=dtype)
        self._h_rnn = torch.zeros(B, self.rnn.hidden_size, device=device, dtype=dtype)

    def get_state(self):
        return (self._h_ssm, self._h_rnn)

    def set_state(self, state):
        self._h_ssm, self._h_rnn = state

    def forward(self, x, use_streaming=False):
        B, N, D = x.shape
        if use_streaming:
            h_ssm = self._h_ssm
            h_rnn = self._h_rnn
        else:
            h_ssm = torch.zeros(B, self.B.out_features, device=x.device, dtype=x.dtype)
            h_rnn = torch.zeros(B, self.rnn.hidden_size, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(N):
            inp = x[:, t]
            h_ssm = F.silu(h_ssm @ self.A.T + self.B(inp))   # use SiLU
            y_ssm = self.C(h_ssm) + self.D(inp)
            h_rnn = self.rnn(inp, h_rnn)
            y_rnn = self.rnn_proj(h_rnn)
            g = torch.sigmoid(self.gate(inp))
            fused = g[:, :1] * y_ssm + g[:, 1:2] * y_rnn
            outputs.append(fused.unsqueeze(1))
        out = torch.cat(outputs, dim=1)

        if use_streaming:
            self._h_ssm = h_ssm.detach()
            self._h_rnn = h_rnn.detach()

        return out
