# Donut 🍩

A toroidal transformer architecture with geometric epistemics.

## Overview

Donut is a novel neural architecture that embeds epistemic uncertainty directly into its geometric structure. Unlike standard transformers where all representations exist in undifferentiated Euclidean space, Donut operates on a toroidal manifold where **certainty and uncertainty are geometric properties**.

The core insight: the zero vector isn't just an arbitrary point — it's the **uncertainty pole**. Representations near zero mean "I don't know." Representations near the ±1 boundary mean high confidence. The architecture enforces this through every component.

## Architecture

```
Input → Ternary Tokenizer → Embedding (tied weights)
                               ↓
                      Kronecker Transform
                               ↓
                         Logic Bias
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
            Focused Attention         HyMBA (SSM+RNN)
              (with cycloid            (parallel path)
               positional bias)              
                    ↓                     ↓
                    └──────────┬──────────┘
                               ↓
                      mHC Residual Connection
                      (doubly-stochastic mixing)
                               ↓
                         [repeat × depth]
                               ↓
                      Output Projection (tied weights)
```

### Components

| Component | Purpose |
|-----------|---------|
| **Ternary Tokenizer** | BPE with ternary merges; embeddings in [-1, 1] with 0 = uncertainty |
| **Kronecker Transform** | Factorized linear transform exploiting matrix structure |
| **Cycloid Positional Bias** | Periodic positions on cycloid curve; encodes toroidal geometry |
| **Focused Attention** | Kronecker-factorized attention with spectral normalization |
| **HyMBA Block** | Parallel SSM + GRU with learned gating; captures long-range state |
| **Logic Bias** | Soft AND/OR operations as inductive bias |
| **mHC Residuals** | Manifold-constrained hyper-connections (DeepSeek arxiv:2512.24880) |

## Key Ideas

### 1. Geometric Epistemics

The ternary representation space creates three regimes:

| Region | Meaning | Geometric Location |
|--------|---------|-------------------|
| Positive (+1) | Confident assertion | Boundary |
| Negative (-1) | Confident negation | Boundary |
| Zero (0) | Uncertainty / "I don't know" | Center |

With tied weights, output logits inherit this geometry. Uncertain hidden states → flat output distributions → calibrated "I don't know" responses.

### 2. Toroidal Manifold

The cycloid positional encoding wraps — there's no edge. Combined with the mHC residual constraints, information circulates without escaping to infinity or collapsing. This matches physical intuitions about closed systems.

### 3. Bounded Signal Propagation

mHC constrains residual mixing matrices to be **doubly stochastic** (rows and columns sum to 1). This guarantees:
- No signal explosion through depth
- No gradient explosion during backprop
- Composite gain stays ~1.0 regardless of layer count

Critical for scaling to very deep networks.

### 4. Parallel Hybrid Paths

Attention and SSM/RNN run in parallel, not sequential. Attention handles sparse, long-range dependencies. HyMBA handles dense, sequential state. The model learns to route information through whichever path suits it.

## Installation

```bash
git clone https://github.com/AhriCat/donut.git
cd donut
pip install torch numpy
```

Optional for training:
```bash
pip install datasets
```

## Usage

### Basic Inference

```python
from model import TatochromicHybridModel
from tokenizer import TernaryTokenizer

# Load or create tokenizer
tokenizer = TernaryTokenizer(vocab_size=50000)
tokenizer.train(your_texts, min_frequency=2)
tokenizer.freeze()

# Create model
model = TatochromicHybridModel(
    vocab_size=len(tokenizer.token_to_id),
    dim=512,
    depth=6,
    heads=8,
    groups=4,
    rank=32,
    ssm_dim=64,
    rnn_dim=128,
    dropout=0.1,
    max_seq_len=512,
)

# Generate
output = model.generate(
    "The meaning of life is",
    tokenizer,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
)
print(output)
```

### With mHC (Donut v2)

```python
from model_mhc import TatochromicHybridModel_mHC

model = TatochromicHybridModel_mHC(
    vocab_size=len(tokenizer.token_to_id),
    dim=512,
    depth=6,
    heads=8,
    n_streams=4,          # mHC parallel streams
    sinkhorn_iters=20,    # doubly-stochastic projection iterations
    mhc_alpha_init=0.1,   # initial off-diagonal mixing
)

# Check mHC stability
diagnostics = model.get_mhc_diagnostics()
print(f"Composite gain: {diagnostics['composite_gain']:.4f}")  # Should be ~1.0
```

### Training

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    for batch in dataloader:
        logits = model(batch["input_ids"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=tokenizer.pad_id,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Model Variants

| Variant | Class | Description |
|---------|-------|-------------|
| Donut (base) | `TatochromicHybridModel` | Original with scalar residual scales |
| Donut-mHC | `TatochromicHybridModel_mHC` | With manifold-constrained hyper-connections |
| Donut-mHC-DualPath | `TatochromicHybridModel_mHC_DualPath` | Separate mHC for attention and SSM paths |
| Donut-mHC-Simple | `TatochromicHybridModel_mHC_Simple` | Shared H_res, separate H_post (fewer params) |
| **Donut-R** | `TatochromicHybridModel_Recursive` | Recursive with weight sharing + LoRA |

## Recursive Architecture (Donut-R)

Donut supports two types of recursion serving different purposes:

### 1. Parameter-Sharing Recursion (Efficiency)

Based on "Relaxed Recursive Transformers" (ICLR 2025, Bae et al., arxiv:2410.20672).

Instead of L unique layers, use K shared layer blocks repeated N times:

```
Standard:   L = 24 unique layers     → 24× layer parameters
Recursive:  K = 6 blocks × N = 4 iterations = 24 effective depth
            → 6× layer parameters + small LoRA overhead
            → ~75% parameter reduction at same effective depth
```

**Purpose**: Compression — same computation with fewer parameters.

### 2. Context-Battling Recursion (Handling Complexity)

Based on "Recursive Language Models" (Zhang et al., arxiv:2512.24601).

When context is too long or complex:

```
Input (long/complex)
        ↓
   ┌────┴────┐
   ↓    ↓    ↓
Chunk1 Chunk2 Chunk3   ← Decompose
   ↓    ↓    ↓
 Model Model Model     ← Recursive calls
   ↓    ↓    ↓
   └────┬────┘
        ↓
   Aggregate           ← Integrate into coherent whole
        ↓
    Output
```

**Purpose**: Adaptation — handle arbitrarily complex contingency structures.

This directly instantiates Donut's philosophy:
> "Intelligence is hierarchical, self-consistent, and adaptive modeling of contingency"

- **Hierarchical**: Complex structure decomposed into sub-structures
- **Self-consistent**: Aggregation enforces coherence across chunks  
- **Adaptive**: Recursion depth adjusts to input complexity

### Why Both?

| Type | When to Use | What it Does |
|------|-------------|--------------|
| Parameter-sharing | Always (training/inference) | Efficiency — more depth, fewer params |
| Context-battling | Long/complex inputs | Adaptation — prevents context rot |

### Usage

```python
# Parameter-sharing recursion (built-in)
from model_recursive import TatochromicHybridModel_Recursive

model = TatochromicHybridModel_Recursive(
    vocab_size=len(tokenizer.token_to_id),
    dim=512,
    n_blocks=6,           # K: unique layer blocks
    n_iterations=4,       # N: recursion count
)

# Add context-battling on top
from context_recursion import wrap_with_context_battling

model = wrap_with_context_battling(
    model,
    dim=512,
    context_threshold=512,    # Recurse when sequence > 512
    max_recursion_depth=4,    # Maximum decomposition depth
)
```

### Philosophical Alignment

The context-battling recursion mirrors how humans handle complexity:

1. **Decompose**: "This is too much — let me break it into parts"
2. **Model each part**: Understand the contingencies within each chunk
3. **Integrate**: "Now how do these parts relate to each other?"

This is hierarchical modeling of contingency in action. When the contingency structure (what follows from what) is too dense for a single pass, decompose it into sub-structures, model each, then synthesize a coherent understanding.

## File Structure

```
donut/
├── model.py              # Base Tatochromic Hybrid Model
├── model_mhc.py          # mHC-enhanced variants
├── model_recursive.py    # Parameter-sharing recursive variants
├── recursive.py          # Parameter-sharing recursion components
├── context_recursion.py  # Context-battling recursion (RLM-style)
├── mhc.py                # Manifold-constrained hyper-connections
├── attn.py               # Focused attention with Kronecker factorization
├── hymba.py              # HyMBA block (SSM + RNN hybrid)
├── cycloidpos.py         # Cycloid positional bias
├── transform.py          # Kronecker transform
├── logic.py              # Logic bias module
├── tokenizer.py          # Ternary BPE tokenizer
├── train.py              # Basic training example
├── LICENSE               # CC BY-NC 4.0
└── README.md
```

## Theoretical Foundation

Donut embodies a specific definition of intelligence:

> **Intelligence is hierarchical, self-consistent, and adaptive modeling of contingency — what follows from what, across modalities and time.**

This definition:
- **Hierarchical**: Contingencies nest (local → global, fast → slow, concrete → abstract)
- **Self-consistent**: The model must cohere across domains and timescales
- **Adaptive**: Updates when contingencies surprise you
- **Contingency**: What follows from what — not reward, not value, just structure

The architecture instantiates this directly:

| Principle | Architectural Element |
|-----------|----------------------|
| Hierarchical | Ternary token merges, nested mHC streams, recursive decomposition |
| Self-consistent | Toroidal manifold (no edges), tied weights, zero = uncertainty |
| Adaptive | mHC residuals adjust mixing, context-battling recursion |
| Contingency modeling | Autoregressive prediction, SSM dynamics |

The ternary tokenizer's uncertainty encoding means calibration is **structural**, not learned post-hoc. A model that doesn't know should naturally produce flat output distributions because its hidden state is near zero.

## Scaling Targets

| Variant | Parameters | Layers | n_streams | Use Case |
|---------|------------|--------|-----------|----------|
| Donut-Nano | ~100M | 12 | 4 | Edge, fine-tuning |
| Donut-Base | ~1-3B | 24 | 4 | Consumer GPU |
| Donut-Large | ~7-13B | 32 | 8 | Single A100 |
| Donut-XL | ~70B | 48 | 8 | Multi-GPU |
| Donut-Ultra | ~400B+ | 64+ | 16 | Cluster |

## Citation

If you use Donut in your research, please cite:

```bibtex
@software{donut2025,
  title={Donut: A Toroidal Transformer with Geometric Epistemics},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-username]/donut}
}
```

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

- ✅ Academic research, personal projects, education
- ❌ Commercial use without separate agreement

See [LICENSE](LICENSE) for details. For commercial licensing, contact [your email].

## Acknowledgments

- mHC implementation based on DeepSeek's "Manifold-Constrained Hyper-Connections" (arxiv:2512.24880)
- Parameter-sharing recursion based on "Relaxed Recursive Transformers" (ICLR 2025, Bae et al., arxiv:2410.20672)
- Context-battling recursion based on "Recursive Language Models" (Zhang et al., arxiv:2512.24601)
- HyMBA inspired by state-space model literature (Mamba, S4, etc.)
