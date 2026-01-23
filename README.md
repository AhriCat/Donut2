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
git clone https://github.com/[your-username]/donut.git
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

## File Structure

```
donut/
├── model.py          # Base Tatochromic Hybrid Model
├── model_mhc.py      # mHC-enhanced variants
├── mhc.py            # Manifold-constrained hyper-connections
├── attn.py           # Focused attention with Kronecker factorization
├── hymba.py          # HyMBA block (SSM + RNN hybrid)
├── cycloidpos.py     # Cycloid positional bias
├── transform.py      # Kronecker transform
├── logic.py          # Logic bias module
├── tokenizer.py      # Ternary BPE tokenizer
├── train.py          # Basic training example
└── README.md
```

## Theoretical Foundation

Donut embodies a specific hypothesis about intelligence:

> **Intelligence is hierarchical information processing in pursuit of self-consistent predictive modeling across all modalities and time.**

The architecture instantiates this:

- **Hierarchical**: Ternary token merges, nested mHC streams, parallel attention/SSM paths
- **Self-consistent**: Toroidal manifold prevents edge artifacts; inconsistency collapses to uncertainty
- **Predictive**: Autoregressive decoding, SSM dynamics for state evolution
- **Bounded**: mHC ensures stable propagation at any depth

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
  author={Ahri Steele},
  year={2025},
  url={https://github.com/AhriCat/donut}
}
```

## License

[Your chosen license]

## Acknowledgments

- mHC implementation based on DeepSeek's "Manifold-Constrained Hyper-Connections" (arxiv:2512.24880)
- HyMBA inspired by state-space model literature (Mamba, S4, etc.)
