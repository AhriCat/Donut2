import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
from cycloidpos import CycloidPositionalBias
from attn import FocusedAttentionGroup
from transform import KroneckerTransform
from logic import LogicBias
from hymba import HyMBA_Block
# ===============================
# Full Tatochromic Hybrid Model
# ===============================
class TatochromicHybridModel(nn.Module):
    def __init__(self, vocab_size, dim=512, depth=6, heads=8, groups=4, rank=32,
                 ssm_dim=64, rnn_dim=128, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.cycloid_bias = CycloidPositionalBias(max_seq_len)
        self.kronecker = KroneckerTransform(dim)      # <-- learnable Kronecker transform
        self.logic_bias = LogicBias(dim, strength=0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout)
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.res_scales = nn.Parameter(torch.ones(depth))
        self.out = nn.Linear(dim, vocab_size)
        # weight tying
        try:
            self.out.weight = self.embed.weight
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N] token ids
        returns logits: [B, N, V]
        """
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, N = x.shape
        bias = self.cycloid_bias(N, device=x.device)
        x = self.embed(x)                     # [B, N, D]
        x = self.kronecker(x)                 # learnable structured transform
        x = self.logic_bias(x)                # small logical inductive bias

        for i, layer in enumerate(self.layers):
            attn_out = layer["attn"](layer["norm"](x), bias)
            ssm_out  = layer["hybrid"](x)
            scale = self.res_scales[i]
            x = x + scale * (attn_out + ssm_out)

        return self.out(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        prompt,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token: Optional[int] = None,
        device: Optional[torch.device] = None,
        return_ids: bool = False,
    ):
        device = device or next(self.parameters()).device
        if isinstance(prompt, str):
            ids = tokenizer.encode(prompt, return_torch=True, device=device)
        elif isinstance(prompt, list):
            ids = torch.tensor(prompt, dtype=torch.long, device=device)
        elif isinstance(prompt, torch.Tensor):
            ids = prompt.to(device)
        else:
            raise TypeError("Expected str, list[int], or torch.Tensor")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        generated = ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                top_vals, top_idx = torch.topk(next_token_logits, top_k)
                probs = torch.zeros_like(next_token_logits).scatter_(-1, top_idx, F.softmax(top_vals, dim=-1))
            else:
                probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break
        gen_ids = generated.squeeze(0).tolist()
        decoded = tokenizer.decode(gen_ids)
        if return_ids:
            return decoded, gen_ids
        return decoded
