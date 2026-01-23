"""
Tatochromic Hybrid Model with mHC (Manifold-Constrained Hyper-Connections)
==========================================================================

This is an enhanced version of the original model that replaces the simple
scalar residual scaling with DeepSeek's mHC architecture for improved
training stability and expressiveness.

Key changes from original:
1. Residual connections use doubly-stochastic mixing matrices
2. Multiple parallel residual streams (n_streams parameter)
3. Sinkhorn-Knopp projection ensures bounded signal propagation
4. H_pre mixing applied before layer computations (per paper spec)

Reference: https://arxiv.org/abs/2512.24880
"""

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
from mhc import (
    mHCResidual,
    mHCDualPathResidual,
    sinkhorn_knopp,
    check_doubly_stochastic,
    compute_composite_gain,
)


class TatochromicHybridModel_mHC(nn.Module):
    """
    Tatochromic Hybrid Model with Manifold-Constrained Hyper-Connections.
    
    Architecture combines:
    - Cycloid positional bias
    - Kronecker transform
    - Focused attention groups
    - HyMBA (SSM + RNN) blocks
    - **mHC residual connections** (new)
    
    The mHC component widens the residual stream into multiple parallel paths
    and constrains their mixing to preserve signal magnitude across depth.
    
    This version correctly applies H_pre before layer computations.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        # mHC parameters
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            dim: Model dimension (must be divisible by n_streams)
            depth: Number of layers
            heads: Attention heads
            groups: Attention groups
            rank: Attention rank
            ssm_dim: State-space model dimension
            rnn_dim: RNN hidden dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            n_streams: Number of mHC residual streams
            sinkhorn_iters: Sinkhorn-Knopp iterations for doubly-stochastic projection
            mhc_alpha_init: Initial off-diagonal mixing strength (small = more identity-like)
        """
        super().__init__()
        
        assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"
        
        self.dim = dim
        self.depth = depth
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        
        # Embeddings and pre-processing
        self.embed = nn.Embedding(vocab_size, dim)
        self.cycloid_bias = CycloidPositionalBias(max_seq_len)
        self.kronecker = KroneckerTransform(dim)
        self.logic_bias = LogicBias(dim, strength=0.02)
        
        # Main transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            }) for _ in range(depth)
        ])
        
        # mHC residual modules - one per layer
        self.mhc_residuals = nn.ModuleList([
            mHCResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])
        
        # H_pre matrices for each layer (applied before layer computation)
        # Per paper: F(H_pre @ x)
        self.H_pre_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams, n_streams))
            for _ in range(depth)
        ])
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        
        # Weight tying
        try:
            self.out.weight = self.embed.weight
        except Exception:
            pass
    
    def get_H_pre(self, layer_idx: int) -> torch.Tensor:
        """Get non-negative input mixing matrix for a layer."""
        return F.softmax(self.H_pre_logits[layer_idx], dim=-1)
    
    def _apply_H_pre(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply H_pre mixing to input before layer computation."""
        B, N, D = x.shape
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        H_pre = self.get_H_pre(layer_idx)
        x_mixed = torch.einsum('bnsd,st->bntd', x_streams, H_pre)
        return x_mixed.view(B, N, D)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mHC residual connections.
        
        Implements: x_{l+1} = H_res @ x_l + H_post.T @ F(H_pre @ x_l)
        
        Args:
            x: Token IDs [B, N]
            
        Returns:
            Logits [B, N, V]
        """
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        B, N = x.shape
        bias = self.cycloid_bias(N, device=x.device)
        
        # Embedding and pre-processing
        x = self.embed(x)              # [B, N, D]
        x = self.kronecker(x)          # Learnable structured transform
        x = self.logic_bias(x)         # Small logical inductive bias
        
        # Main layers with mHC residuals
        for i, layer in enumerate(self.layers):
            # Apply H_pre mixing before layer computation
            x_pre = self._apply_H_pre(layer["norm"](x), i)
            
            # Compute attention on H_pre-mixed input
            attn_out = layer["attn"](x_pre, bias)
            
            # Compute hybrid on H_pre-mixed input
            x_pre_hybrid = self._apply_H_pre(x, i)  # Hybrid doesn't use norm
            ssm_out = layer["hybrid"](x_pre_hybrid)
            
            # Combined layer output
            layer_output = attn_out + ssm_out
            
            # Apply mHC residual: H_res @ x + H_post.T @ layer_output
            x = self.mhc_residuals[i](x, layer_output)
            
        return self.out(self.norm(x))
    
    def get_mhc_diagnostics(self) -> dict:
        """
        Get diagnostic information about mHC matrices.
        
        Returns:
            dict with per-layer H_res matrices and their properties
        """
        diagnostics = {
            'per_layer': {},
            'composite_gain': None,
        }
        
        H_res_list = []
        for i, mhc in enumerate(self.mhc_residuals):
            H_res = mhc.get_H_res()
            H_res_list.append(H_res)
            diagnostics['per_layer'][f'layer_{i}'] = {
                'H_res': H_res.detach().cpu().tolist(),
                'doubly_stochastic_check': check_doubly_stochastic(H_res),
                'forward_gain': H_res.abs().sum(dim=-1).max().item(),
                'backward_gain': H_res.abs().sum(dim=-2).max().item(),
            }
        
        # Compute composite gain across all layers
        diagnostics['composite_gain'] = compute_composite_gain(H_res_list)
        
        return diagnostics
    
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
        """Generate text from prompt."""
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
                probs = torch.zeros_like(next_token_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
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


# =============================================================================
# Alternative: Dual-Path mHC (separate constraints for attention and hybrid)
# =============================================================================

class TatochromicHybridModel_mHC_DualPath(nn.Module):
    """
    Variant with separate mHC residuals for attention and hybrid paths.
    
    This allows the attention and SSM/RNN paths to have independently
    learned doubly-stochastic mixing, which may capture different
    information routing patterns.
    
    Each path has its own:
    - H_pre (input mixing)
    - H_res (residual mixing, doubly stochastic)
    - H_post (output mixing)
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        super().__init__()
        
        assert dim % n_streams == 0
        
        self.dim = dim
        self.depth = depth
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.cycloid_bias = CycloidPositionalBias(max_seq_len)
        self.kronecker = KroneckerTransform(dim)
        self.logic_bias = LogicBias(dim, strength=0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            }) for _ in range(depth)
        ])
        
        # Separate mHC for attention and hybrid paths
        self.mhc_attn = nn.ModuleList([
            mHCResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])
        self.mhc_hybrid = nn.ModuleList([
            mHCResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])
        
        # Separate H_pre for each path at each layer
        self.H_pre_attn_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams, n_streams))
            for _ in range(depth)
        ])
        self.H_pre_hybrid_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams, n_streams))
            for _ in range(depth)
        ])
        
        # Learnable blend between attention and hybrid mHC outputs
        self.path_blend = nn.Parameter(torch.ones(depth) * 0.5)
        
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        
        try:
            self.out.weight = self.embed.weight
        except Exception:
            pass
    
    def get_H_pre_attn(self, layer_idx: int) -> torch.Tensor:
        return F.softmax(self.H_pre_attn_logits[layer_idx], dim=-1)
    
    def get_H_pre_hybrid(self, layer_idx: int) -> torch.Tensor:
        return F.softmax(self.H_pre_hybrid_logits[layer_idx], dim=-1)
    
    def _apply_H_pre(self, x: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        x_mixed = torch.einsum('bnsd,st->bntd', x_streams, H_pre)
        return x_mixed.view(B, N, D)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        B, N = x.shape
        bias = self.cycloid_bias(N, device=x.device)
        
        x = self.embed(x)
        x = self.kronecker(x)
        x = self.logic_bias(x)
        
        for i, layer in enumerate(self.layers):
            normed = layer["norm"](x)
            
            # Attention path with H_pre
            H_pre_attn = self.get_H_pre_attn(i)
            attn_input = self._apply_H_pre(normed, H_pre_attn)
            attn_out = layer["attn"](attn_input, bias)
            x_attn = self.mhc_attn[i](x, attn_out)
            
            # Hybrid path with H_pre
            H_pre_hybrid = self.get_H_pre_hybrid(i)
            hybrid_input = self._apply_H_pre(x, H_pre_hybrid)
            ssm_out = layer["hybrid"](hybrid_input)
            x_hybrid = self.mhc_hybrid[i](x, ssm_out)
            
            # Blend the two mHC-transformed paths
            alpha = torch.sigmoid(self.path_blend[i])
            x = alpha * x_attn + (1 - alpha) * x_hybrid
            
        return self.out(self.norm(x))
    
    def get_mhc_diagnostics(self) -> dict:
        """Get diagnostics for both attention and hybrid mHC paths."""
        diagnostics = {
            'attention_path': {},
            'hybrid_path': {},
            'composite_gain_attn': None,
            'composite_gain_hybrid': None,
        }
        
        H_res_attn_list = []
        H_res_hybrid_list = []
        
        for i in range(self.depth):
            H_res_attn = self.mhc_attn[i].get_H_res()
            H_res_hybrid = self.mhc_hybrid[i].get_H_res()
            H_res_attn_list.append(H_res_attn)
            H_res_hybrid_list.append(H_res_hybrid)
            
            diagnostics['attention_path'][f'layer_{i}'] = {
                'H_res': H_res_attn.detach().cpu().tolist(),
                'doubly_stochastic_check': check_doubly_stochastic(H_res_attn),
                'forward_gain': H_res_attn.abs().sum(dim=-1).max().item(),
            }
            diagnostics['hybrid_path'][f'layer_{i}'] = {
                'H_res': H_res_hybrid.detach().cpu().tolist(),
                'doubly_stochastic_check': check_doubly_stochastic(H_res_hybrid),
                'forward_gain': H_res_hybrid.abs().sum(dim=-1).max().item(),
            }
        
        diagnostics['composite_gain_attn'] = compute_composite_gain(H_res_attn_list)
        diagnostics['composite_gain_hybrid'] = compute_composite_gain(H_res_hybrid_list)
        
        return diagnostics
    
    @torch.no_grad()
    def generate(self, prompt, tokenizer, **kwargs):
        device = kwargs.pop('device', None) or next(self.parameters()).device
        
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
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 40)
        eos_token = kwargs.get('eos_token', None)
        return_ids = kwargs.get('return_ids', False)
        
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if top_k is not None and top_k > 0:
                top_vals, top_idx = torch.topk(next_token_logits, top_k)
                probs = torch.zeros_like(next_token_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
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


# =============================================================================
# Simplified variant using mHCDualPathResidual
# =============================================================================

class TatochromicHybridModel_mHC_Simple(nn.Module):
    """
    Simplified mHC model using shared H_res for both paths.
    
    This variant uses a single doubly-stochastic residual matrix per layer,
    but allows separate H_post mixing for attention and hybrid outputs.
    
    Fewer parameters than DualPath, may be better for smaller models.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        super().__init__()
        
        assert dim % n_streams == 0
        
        self.dim = dim
        self.depth = depth
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.cycloid_bias = CycloidPositionalBias(max_seq_len)
        self.kronecker = KroneckerTransform(dim)
        self.logic_bias = LogicBias(dim, strength=0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            }) for _ in range(depth)
        ])
        
        # Unified mHC with shared H_res but separate H_post
        self.mhc = nn.ModuleList([
            mHCDualPathResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        
        try:
            self.out.weight = self.embed.weight
        except Exception:
            pass
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        B, N = x.shape
        bias = self.cycloid_bias(N, device=x.device)
        
        x = self.embed(x)
        x = self.kronecker(x)
        x = self.logic_bias(x)
        
        for i, layer in enumerate(self.layers):
            normed = layer["norm"](x)
            attn_out = layer["attn"](normed, bias)
            ssm_out = layer["hybrid"](x)
            
            # Apply unified mHC with dual-path mixing
            x = self.mhc[i](x, attn_out, ssm_out)
            
        return self.out(self.norm(x))
    
    @torch.no_grad()
    def generate(self, prompt, tokenizer, **kwargs):
        device = kwargs.pop('device', None) or next(self.parameters()).device
        
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
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 40)
        eos_token = kwargs.get('eos_token', None)
        return_ids = kwargs.get('return_ids', False)
        
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if top_k is not None and top_k > 0:
                top_vals, top_idx = torch.topk(next_token_logits, top_k)
                probs = torch.zeros_like(next_token_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
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
