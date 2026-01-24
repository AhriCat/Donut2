"""
Recursive Tatochromic Hybrid Model (Donut-R)
=============================================

Combines:
- Relaxed Recursive Transformer architecture (weight sharing + LoRA)
- mHC residual connections (manifold-constrained hyper-connections)
- Toroidal geometry (cycloid positions, ternary embeddings)

The recursive structure naturally aligns with the toroidal manifold:
information cycles through the same layers multiple times, refining
representations on each pass — like tracing loops on a torus.

Key benefits:
1. Parameter efficiency: K×N effective depth with only K unique blocks
2. Iterative refinement: Multiple passes allow progressive certainty
3. Stable depth scaling: mHC ensures bounded propagation through recursions
4. Memory efficiency: Shared weights reduce footprint significantly

Reference: "Relaxed Recursive Transformers" (ICLR 2025, Bae et al.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from cycloidpos import CycloidPositionalBias
from attn import FocusedAttentionGroup
from transform import KroneckerTransform
from logic import LogicBias
from hymba import HyMBA_Block
from mhc import mHCResidual, sinkhorn_knopp, check_doubly_stochastic
from recursive import LoRAAdapter, RecursiveLayerBlock, RecursiveDonutConfig


class TatochromicHybridModel_Recursive(nn.Module):
    """
    Recursive Donut: Toroidal transformer with weight-shared recursive layers.
    
    Architecture:
        Input → Embed → Kronecker → LogicBias
                            ↓
                   ┌───────────────┐
                   │  Block 1      │ ←─┐
                   │  Block 2      │   │ × N iterations
                   │  ...          │   │ (with per-iteration LoRA)
                   │  Block K      │ ──┘
                   └───────────────┘
                            ↓
                   Output (tied weights)
    
    Effective depth = K blocks × N iterations
    Parameters ≈ K blocks + small LoRA overhead
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_blocks: int = 3,
        n_iterations: int = 2,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        # Recursion params
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        use_iteration_lora: bool = True,
        # mHC params
        use_mhc: bool = True,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        super().__init__()
        
        # Validate
        if use_mhc:
            assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"
        
        self.dim = dim
        self.n_blocks = n_blocks
        self.n_iterations = n_iterations
        self.effective_depth = n_blocks * n_iterations
        self.use_mhc = use_mhc
        
        # Embeddings and preprocessing
        self.embed = nn.Embedding(vocab_size, dim)
        self.cycloid_bias = CycloidPositionalBias(max_seq_len)
        self.kronecker = KroneckerTransform(dim)
        self.logic_bias = LogicBias(dim, strength=0.02)
        
        # Shared layer blocks (the K unique blocks)
        self.blocks = nn.ModuleList([
            RecursiveLayerBlock(
                dim=dim,
                attn_module=FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                hybrid_module=HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
                n_iterations=n_iterations,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_iteration_lora=use_iteration_lora,
            )
            for _ in range(n_blocks)
        ])
        
        # mHC residuals - one per (block, iteration) pair
        if use_mhc:
            self.mhc_residuals = nn.ModuleList([
                mHCResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
                for _ in range(n_blocks * n_iterations)
            ])
        else:
            self.mhc_residuals = None
            
        # Per-iteration embeddings (helps differentiate recursion passes)
        self.iteration_embed = nn.Parameter(
            torch.randn(n_iterations, dim) * 0.02
        )
        
        # Simple residual scaling (used if mHC disabled)
        self.res_scales = nn.Parameter(torch.ones(n_blocks * n_iterations))
        
        # Output
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        
        # Weight tying
        try:
            self.out.weight = self.embed.weight
        except Exception:
            pass
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with recursive layer application.
        
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
        
        # Embedding and preprocessing
        x = self.embed(x)
        x = self.kronecker(x)
        x = self.logic_bias(x)
        
        # Recursive layer application
        layer_idx = 0
        for iteration in range(self.n_iterations):
            # Add iteration embedding
            iter_emb = self.iteration_embed[iteration].view(1, 1, self.dim)
            x = x + iter_emb
            
            # Process through all blocks
            for block_idx, block in enumerate(self.blocks):
                attn_out, hybrid_out = block(x, iteration, bias)
                layer_out = attn_out + hybrid_out
                
                # Residual connection
                if self.mhc_residuals is not None:
                    x = self.mhc_residuals[layer_idx](x, layer_out)
                else:
                    x = x + self.res_scales[layer_idx] * layer_out
                    
                layer_idx += 1
                
        return self.out(self.norm(x))
    
    def get_recursion_info(self) -> dict:
        """Get information about the recursive structure."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        
        # Estimate what a non-recursive model would need
        # (rough: effective_depth / n_blocks times more block params)
        block_params = sum(
            p.numel() for block in self.blocks for p in block.parameters()
        )
        equivalent_non_recursive = block_params * self.n_iterations
        
        return {
            'n_blocks': self.n_blocks,
            'n_iterations': self.n_iterations,
            'effective_depth': self.effective_depth,
            'total_params': total_params,
            'block_params': block_params,
            'equivalent_non_recursive_estimate': equivalent_non_recursive,
            'param_savings_estimate': 1 - (block_params / equivalent_non_recursive),
        }
    
    def get_mhc_diagnostics(self) -> Optional[dict]:
        """Get mHC stability diagnostics if enabled."""
        if self.mhc_residuals is None:
            return None
            
        diagnostics = {'per_layer': {}}
        H_res_list = []
        
        for i, mhc in enumerate(self.mhc_residuals):
            H_res = mhc.get_H_res()
            H_res_list.append(H_res)
            iteration = i // self.n_blocks
            block = i % self.n_blocks
            diagnostics['per_layer'][f'iter{iteration}_block{block}'] = {
                'forward_gain': H_res.abs().sum(dim=-1).max().item(),
                'is_doubly_stochastic': check_doubly_stochastic(H_res)['is_doubly_stochastic'],
            }
            
        # Composite gain through all iterations
        composite = H_res_list[0].clone()
        for H in H_res_list[1:]:
            composite = composite @ H
        diagnostics['composite_gain'] = composite.abs().max().item()
        
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
# Factory functions for common configurations
# =============================================================================

def create_recursive_donut_nano(vocab_size: int) -> TatochromicHybridModel_Recursive:
    """
    Donut-R Nano: ~25M params, effective depth 12
    3 blocks × 4 iterations
    """
    return TatochromicHybridModel_Recursive(
        vocab_size=vocab_size,
        dim=384,
        n_blocks=3,
        n_iterations=4,
        heads=6,
        groups=3,
        rank=24,
        ssm_dim=48,
        rnn_dim=96,
        dropout=0.1,
        lora_rank=4,
        use_mhc=True,
        n_streams=4,
    )


def create_recursive_donut_base(vocab_size: int) -> TatochromicHybridModel_Recursive:
    """
    Donut-R Base: ~100M params, effective depth 24
    6 blocks × 4 iterations
    """
    return TatochromicHybridModel_Recursive(
        vocab_size=vocab_size,
        dim=512,
        n_blocks=6,
        n_iterations=4,
        heads=8,
        groups=4,
        rank=32,
        ssm_dim=64,
        rnn_dim=128,
        dropout=0.1,
        lora_rank=8,
        use_mhc=True,
        n_streams=4,
    )


def create_recursive_donut_large(vocab_size: int) -> TatochromicHybridModel_Recursive:
    """
    Donut-R Large: ~400M params, effective depth 48
    8 blocks × 6 iterations
    """
    return TatochromicHybridModel_Recursive(
        vocab_size=vocab_size,
        dim=1024,
        n_blocks=8,
        n_iterations=6,
        heads=16,
        groups=4,
        rank=64,
        ssm_dim=128,
        rnn_dim=256,
        dropout=0.1,
        lora_rank=16,
        use_mhc=True,
        n_streams=8,
    )
