"""
mHC: Manifold-Constrained Hyper-Connections
============================================
Implementation based on DeepSeek's arxiv:2512.24880

Core insight: Widen residual streams into n parallel paths, but constrain
the mixing matrix H_res to be doubly stochastic (Birkhoff polytope) via
Sinkhorn-Knopp. This prevents signal explosion while allowing richer
information routing.

Key equations:
    x_{l+1} = H_res @ x_l + H_post.T @ F(H_pre @ x_l)
    
Where:
    H_res: doubly stochastic (rows/cols sum to 1, entries ≥ 0)
    H_pre, H_post: non-negative mixing maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable


def sinkhorn_knopp(
    logits: torch.Tensor,
    iterations: int = 20,
    eps: float = 1e-8,
    stable: bool = True
) -> torch.Tensor:
    """
    Project a matrix onto the doubly stochastic manifold using Sinkhorn-Knopp.
    
    Args:
        logits: Raw learnable parameters [n_streams, n_streams]
        iterations: Number of row/column normalization iterations (20 is typical)
        eps: Small constant for numerical stability
        stable: If True, use fp32 for accumulation even if input is fp16
        
    Returns:
        Doubly stochastic matrix where rows and columns each sum to 1
    """
    orig_dtype = logits.dtype
    
    # Use fp32 for numerical stability during Sinkhorn iterations
    if stable and logits.dtype == torch.float16:
        logits = logits.float()
    
    # Exponentiate to ensure non-negativity (subtract max for stability)
    P = torch.exp(logits - logits.max())
    
    for _ in range(iterations):
        # Row normalization
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        # Column normalization  
        P = P / (P.sum(dim=-2, keepdim=True) + eps)
    
    # Cast back to original dtype
    if stable and orig_dtype == torch.float16:
        P = P.to(orig_dtype)
    
    return P


class mHCResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection residual block.
    
    Implements: x_{l+1} = H_res @ x + H_post.T @ layer_output
    
    Where H_res is constrained to be doubly stochastic via Sinkhorn-Knopp.
    
    Note: H_pre should be applied BEFORE passing to layer_fn. This module
    handles the residual mixing (H_res) and output mixing (H_post).
    """
    
    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.1,
    ):
        """
        Args:
            dim: Hidden dimension (must be divisible by n_streams)
            n_streams: Number of parallel residual streams
            sinkhorn_iters: Iterations for doubly-stochastic projection
            alpha_init: Initial scaling for off-diagonal mixing (small = more identity-like)
        """
        super().__init__()
        assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"
        
        self.dim = dim
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # Learnable logits for H_res (will be projected to doubly stochastic)
        # Initialize close to identity: strong diagonal, weak off-diagonal
        init_logits = torch.zeros(n_streams, n_streams)
        init_logits.fill_diagonal_(1.0)  # Favor identity initially
        init_logits += alpha_init * torch.randn(n_streams, n_streams)
        self.H_res_logits = nn.Parameter(init_logits)
        
        # H_post: non-negative mixing map for layer output
        # Use softmax over streams to ensure non-negativity and normalization
        self.H_post_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        
        # Optional: learnable scaling per stream
        self.stream_scales = nn.Parameter(torch.ones(n_streams))
        
    def get_H_res(self) -> torch.Tensor:
        """Get the doubly stochastic residual mixing matrix."""
        return sinkhorn_knopp(self.H_res_logits, self.sinkhorn_iters)
    
    def get_H_post(self) -> torch.Tensor:
        """Get non-negative output mixing."""
        return F.softmax(self.H_post_logits, dim=-1)
        
    def forward(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mHC residual connection.
        
        Args:
            x: Input tensor [B, N, D]
            layer_output: Output from layer function F(H_pre @ x) [B, N, D]
            
        Returns:
            Updated hidden state [B, N, D]
        """
        B, N, D = x.shape
        assert D == self.dim
        
        # Reshape to streams: [B, N, n_streams, stream_dim]
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        layer_streams = layer_output.view(B, N, self.n_streams, self.stream_dim)
        
        # Get constrained matrices
        H_res = self.get_H_res()    # [n_streams, n_streams], doubly stochastic
        H_post = self.get_H_post()  # [n_streams, n_streams], non-negative
        
        # Apply H_res to residual path: mix streams
        # x_streams: [B, N, n_streams, stream_dim]
        # H_res: [n_streams, n_streams]
        # Result: [B, N, n_streams, stream_dim]
        residual = torch.einsum('bnsd,st->bntd', x_streams, H_res)
        
        # Apply H_post.T to layer output (transpose for proper mixing direction)
        # Paper: H_post.T @ F(H_pre @ x)
        layer_mixed = torch.einsum('bnsd,ts->bntd', layer_streams, H_post)
        
        # Combine with per-stream scaling
        scales = self.stream_scales.view(1, 1, self.n_streams, 1)
        out_streams = residual + scales * layer_mixed
        
        # Reshape back to [B, N, D]
        return out_streams.view(B, N, D)


class mHCLayer(nn.Module):
    """
    Complete mHC-enhanced layer wrapper.
    
    Wraps any layer function with full mHC: H_res @ x + H_post.T @ F(H_pre @ x)
    
    This correctly applies H_pre before the layer computation.
    """
    
    def __init__(
        self,
        dim: int,
        layer_fn: nn.Module,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.1,
        use_pre_norm: bool = True,
    ):
        """
        Args:
            dim: Hidden dimension
            layer_fn: The actual layer computation (attention, FFN, etc.)
            n_streams: Number of residual streams
            sinkhorn_iters: Sinkhorn-Knopp iterations
            alpha_init: Initial off-diagonal mixing strength
            use_pre_norm: Whether to apply LayerNorm before layer_fn
        """
        super().__init__()
        assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"
        
        self.dim = dim
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        
        self.layer_fn = layer_fn
        self.mhc_residual = mHCResidual(dim, n_streams, sinkhorn_iters, alpha_init)
        
        if use_pre_norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = None
            
        # H_pre: non-negative mixing for input to layer_fn
        self.H_pre_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        
    def get_H_pre(self) -> torch.Tensor:
        """Get non-negative input mixing."""
        return F.softmax(self.H_pre_logits, dim=-1)
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with full mHC: H_res @ x + H_post.T @ F(H_pre @ x)
        
        Args:
            x: Input [B, N, D]
            *args, **kwargs: Passed to layer_fn
        """
        B, N, D = x.shape
        
        # Optional pre-norm
        if self.norm is not None:
            normed = self.norm(x)
        else:
            normed = x
            
        # Apply H_pre mixing before layer computation
        # This is the key part: F(H_pre @ x)
        x_streams = normed.view(B, N, self.n_streams, self.stream_dim)
        H_pre = self.get_H_pre()
        x_mixed = torch.einsum('bnsd,st->bntd', x_streams, H_pre)
        x_mixed = x_mixed.view(B, N, D)
        
        # Compute layer output on H_pre-mixed input
        layer_out = self.layer_fn(x_mixed, *args, **kwargs)
        
        # Apply mHC residual: H_res @ x + H_post.T @ layer_out
        return self.mhc_residual(x, layer_out)


class mHCBlock(nn.Module):
    """
    Dual-path mHC block for hybrid architectures like Donut.
    
    Combines attention and SSM/RNN outputs with manifold-constrained mixing.
    Each path gets its own H_pre, H_res, and H_post matrices.
    """
    
    def __init__(
        self,
        dim: int,
        attn_module: nn.Module,
        hybrid_module: nn.Module,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.1,
    ):
        """
        Args:
            dim: Hidden dimension
            attn_module: Attention layer
            hybrid_module: SSM/RNN hybrid layer (like HyMBA)
            n_streams: Number of residual streams
            sinkhorn_iters: Sinkhorn iterations
            alpha_init: Initial off-diagonal mixing strength
        """
        super().__init__()
        assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"
        
        self.dim = dim
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        
        self.norm = nn.LayerNorm(dim)
        self.attn = attn_module
        self.hybrid = hybrid_module
        
        # Separate mHC components for attention and hybrid paths
        self.mhc_attn = mHCResidual(dim, n_streams, sinkhorn_iters, alpha_init)
        self.mhc_hybrid = mHCResidual(dim, n_streams, sinkhorn_iters, alpha_init)
        
        # H_pre for each path
        self.H_pre_attn_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_pre_hybrid_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        
        # Cross-path mixing (optional): allows attention and hybrid to inform each other
        self.cross_mix = nn.Parameter(torch.tensor(0.5))  # Learnable blend
    
    def get_H_pre_attn(self) -> torch.Tensor:
        return F.softmax(self.H_pre_attn_logits, dim=-1)
    
    def get_H_pre_hybrid(self) -> torch.Tensor:
        return F.softmax(self.H_pre_hybrid_logits, dim=-1)
        
    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward with dual mHC paths.
        
        Args:
            x: Input [B, N, D]
            bias: Optional attention bias
        """
        B, N, D = x.shape
        normed = self.norm(x)
        
        # Reshape for stream mixing
        normed_streams = normed.view(B, N, self.n_streams, self.stream_dim)
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        
        # Attention path with H_pre
        H_pre_attn = self.get_H_pre_attn()
        attn_input = torch.einsum('bnsd,st->bntd', normed_streams, H_pre_attn).view(B, N, D)
        attn_out = self.attn(attn_input, bias)
        x_attn = self.mhc_attn(x, attn_out)
        
        # Hybrid path with H_pre
        H_pre_hybrid = self.get_H_pre_hybrid()
        hybrid_input = torch.einsum('bnsd,st->bntd', x_streams, H_pre_hybrid).view(B, N, D)
        ssm_out = self.hybrid(hybrid_input)
        x_hybrid = self.mhc_hybrid(x, ssm_out)
        
        # Blend the two mHC-transformed paths
        alpha = torch.sigmoid(self.cross_mix)
        return alpha * x_attn + (1 - alpha) * x_hybrid


# =============================================================================
# Unified mHC module for simpler integration
# =============================================================================

class mHCDualPathResidual(nn.Module):
    """
    Simplified mHC for dual-path architectures.
    
    Takes two layer outputs (e.g., attention and SSM) and applies mHC
    residual connections with a learnable blend.
    
    This is useful when you want to keep the layer computations separate
    but still benefit from mHC's stability properties.
    """
    
    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        assert dim % n_streams == 0
        
        self.dim = dim
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams
        self.sinkhorn_iters = sinkhorn_iters
        
        # Shared H_res for both paths (ensures consistent residual behavior)
        init_logits = torch.zeros(n_streams, n_streams)
        init_logits.fill_diagonal_(1.0)
        init_logits += alpha_init * torch.randn(n_streams, n_streams)
        self.H_res_logits = nn.Parameter(init_logits)
        
        # Separate H_post for each path
        self.H_post_attn_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_post_hybrid_logits = nn.Parameter(torch.zeros(n_streams, n_streams))
        
        # Per-stream scales
        self.stream_scales_attn = nn.Parameter(torch.ones(n_streams))
        self.stream_scales_hybrid = nn.Parameter(torch.ones(n_streams))
        
        # Blend factor
        self.blend = nn.Parameter(torch.tensor(0.5))
        
    def get_H_res(self) -> torch.Tensor:
        return sinkhorn_knopp(self.H_res_logits, self.sinkhorn_iters)
    
    def get_H_post_attn(self) -> torch.Tensor:
        return F.softmax(self.H_post_attn_logits, dim=-1)
    
    def get_H_post_hybrid(self) -> torch.Tensor:
        return F.softmax(self.H_post_hybrid_logits, dim=-1)
        
    def forward(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        hybrid_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mHC to dual-path outputs.
        
        Args:
            x: Original input [B, N, D]
            attn_out: Attention layer output [B, N, D]
            hybrid_out: Hybrid/SSM layer output [B, N, D]
            
        Returns:
            mHC-combined output [B, N, D]
        """
        B, N, D = x.shape
        
        # Reshape to streams
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        attn_streams = attn_out.view(B, N, self.n_streams, self.stream_dim)
        hybrid_streams = hybrid_out.view(B, N, self.n_streams, self.stream_dim)
        
        # Get matrices
        H_res = self.get_H_res()
        H_post_attn = self.get_H_post_attn()
        H_post_hybrid = self.get_H_post_hybrid()
        
        # Shared residual
        residual = torch.einsum('bnsd,st->bntd', x_streams, H_res)
        
        # Attention path
        attn_mixed = torch.einsum('bnsd,ts->bntd', attn_streams, H_post_attn)
        scales_attn = self.stream_scales_attn.view(1, 1, self.n_streams, 1)
        x_attn = residual + scales_attn * attn_mixed
        
        # Hybrid path
        hybrid_mixed = torch.einsum('bnsd,ts->bntd', hybrid_streams, H_post_hybrid)
        scales_hybrid = self.stream_scales_hybrid.view(1, 1, self.n_streams, 1)
        x_hybrid = residual + scales_hybrid * hybrid_mixed
        
        # Blend
        alpha = torch.sigmoid(self.blend)
        out_streams = alpha * x_attn + (1 - alpha) * x_hybrid
        
        return out_streams.view(B, N, D)


# =============================================================================
# Diagnostic utilities
# =============================================================================

def compute_gain_magnitude(H: torch.Tensor) -> Tuple[float, float]:
    """
    Compute forward and backward gain magnitudes (Amax metrics from paper).
    
    For a doubly stochastic matrix, both should be close to 1.0.
    
    Returns:
        (forward_gain, backward_gain): Max row sum and max column sum
    """
    forward_gain = H.abs().sum(dim=-1).max().item()   # Max row sum
    backward_gain = H.abs().sum(dim=-2).max().item()  # Max column sum
    return forward_gain, backward_gain


def compute_composite_gain(H_list: list, depth: Optional[int] = None) -> float:
    """
    Compute composite gain magnitude through multiple layers.
    
    This is the key stability metric from the paper. For mHC, this should
    remain bounded even at large depths. For unconstrained HC, it explodes.
    
    Args:
        H_list: List of H_res matrices from each layer
        depth: Number of layers to compose (default: all)
        
    Returns:
        Composite gain magnitude (should be ~1.0 for stable training)
    """
    if depth is None:
        depth = len(H_list)
    
    composite = H_list[0].clone()
    for i in range(1, min(depth, len(H_list))):
        composite = composite @ H_list[i]
    
    return composite.abs().max().item()


def check_doubly_stochastic(H: torch.Tensor, tol: float = 1e-5) -> dict:
    """
    Verify a matrix is doubly stochastic.
    
    Returns:
        dict with row_sums, col_sums, and whether constraints are satisfied
    """
    row_sums = H.sum(dim=-1)
    col_sums = H.sum(dim=-2)
    
    row_ok = (row_sums - 1.0).abs().max().item() < tol
    col_ok = (col_sums - 1.0).abs().max().item() < tol
    non_neg = (H >= -tol).all().item()
    
    return {
        'row_sums': row_sums.tolist(),
        'col_sums': col_sums.tolist(),
        'row_constraint_satisfied': row_ok,
        'col_constraint_satisfied': col_ok,
        'non_negative': non_neg,
        'is_doubly_stochastic': row_ok and col_ok and non_neg,
    }


def visualize_H_res(model: nn.Module, layer_idx: int = 0) -> torch.Tensor:
    """
    Extract H_res matrix from a model for visualization.
    
    Args:
        model: Model with mHC residuals
        layer_idx: Which layer to extract from
        
    Returns:
        H_res matrix [n_streams, n_streams]
    """
    if hasattr(model, 'mhc_residuals'):
        return model.mhc_residuals[layer_idx].get_H_res().detach()
    elif hasattr(model, 'mhc_attn'):
        return model.mhc_attn[layer_idx].get_H_res().detach()
    else:
        raise ValueError("Model does not have mHC residuals")
