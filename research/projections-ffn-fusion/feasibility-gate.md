# Feasibility Gate: Case #14 Baseline Memory Analysis

**Date**: 29 August 2026
**Status**: BLOCKER
**Scope**: Determines whether `torch_transformer_benchmark.py` baseline can physically execute Case #14 on any single GPU.

## Case #14 Parameters

| Parameter | Value |
|-----------|-------|
| Batch size (B) | 32 |
| Sequence length (S) | 100,000 |
| Model dim (D) | 1,024 |
| Heads (H) | 16 |
| Head dim (d_k) | 64 |
| Layers (L) | 2 |
| Causal | TRUE |
| FFN dim | 1,024 |
| dtype | FP16 (2 bytes/element) |

## Memory Calculations

### 1. Input tensor x: [B, S, D] in FP16

```
32 × 100,000 × 1,024 = 3,276,800,000 elements
× 2 bytes = 6,553,600,000 bytes = 6.10 GiB
```

### 2. Q/K/V projection outputs: [B, S, D] each in FP16

Each of `q_proj(x)`, `k_proj(x)`, `v_proj(x)` produces a [B, S, D] tensor:

```
32 × 100,000 × 1,024 = 3,276,800,000 elements × 2 bytes = 6.10 GiB each
All three simultaneously alive: 3 × 6.10 = 18.31 GiB
```

### 3. Q/K/V after split_heads: [B, H, S, d_k] each in FP16

```
32 × 16 × 100,000 × 64 = 3,276,800,000 elements × 2 bytes = 6.10 GiB each
All three simultaneously alive: 3 × 6.10 = 18.31 GiB
```

### 4. Score tensor: [B, H, S, S] in FP16 — THE BLOCKER

Produced at `torch_transformer_benchmark.py:97`:
```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
```

```
32 × 16 × 100,000 × 100,000 = 5,120,000,000,000 elements
× 2 bytes = 10,240,000,000,000 bytes = 9.31 TiB
```

### 5. Causal mask: [S, S] in bool

Produced at `torch_transformer_benchmark.py:100-102`:
```python
causal_mask = torch.ones(
    (seq_len, seq_len), device=x.device, dtype=torch.bool
).triu(diagonal=1)
```

```
100,000 × 100,000 = 10,000,000,000 elements
× 1 byte = 9.31 GiB
```

### 6. Softmax probs: [B, H, S, S] in FP16

Produced at `torch_transformer_benchmark.py:111`:
```python
probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
```

This line also creates a temporary FP32 copy of scores:

```
scores.float() FP32: 5,120,000,000,000 × 4 bytes = 18.63 TiB (temporary)
softmax output FP32: 5,120,000,000,000 × 4 bytes = 18.63 TiB (temporary)
probs FP16:          5,120,000,000,000 × 2 bytes = 9.31 TiB
```

### 7. Context and other intermediates (per layer)

| Tensor | Shape | Size (FP16) |
|--------|-------|-------------|
| k.transpose(-2,-1) | [B,H,d_k,S] | 6.10 GiB |
| context = matmul(probs, v) | [B,H,S,d_k] | 6.10 GiB |
| context reshaped | [B,S,D] | 6.10 GiB |
| out_proj(context) | [B,S,D] | 6.10 GiB |
| norm1(x) | [B,S,D] | 6.10 GiB |
| FFN intermediate (ffn_in) | [B,S,ffn_dim] | 6.10 GiB |
| FFN gelu output | [B,S,ffn_dim] | 6.10 GiB |
| FFN output (ffn_out) | [B,S,D] | 6.10 GiB |

### 8. Model weights (negligible)

```
Per layer: 4 × (D² + D) + 2 × (D × ffn_dim + ffn_dim) + 2 × (2D)
         = 4 × (1,049,600) + 2 × (1,049,600) + 4,096
         = 6,297,600 + 2,099,200 + 4,096
         = 8,400,896 elements × 2 bytes × 2 layers ≈ 32 MiB total
```

## Peak Memory Summary

### Attention block peak (single layer)

Alive simultaneously at the point of `scores` computation:

| Tensor | Size |
|--------|------|
| x (input, needed for residual) | 6.10 GiB |
| q [B,H,S,d_k] | 6.10 GiB |
| k [B,H,S,d_k] | 6.10 GiB |
| v [B,H,S,d_k] | 6.10 GiB |
| **scores [B,H,S,S]** | **9.31 TiB** |

**Subtotal: ~9.33 TiB** (dominated by scores)

During softmax (line 111), the FP32 cast temporarily requires an additional 18.63 TiB, pushing peak to **~27.94 TiB** momentarily.

### Even ignoring the score tensor

The non-score intermediates alone peak at approximately:

```
x + q + k + v + norm1 + probs(if S×S were small) + context ≈ 30-40 GiB
```

This would be feasible on an 80 GB GPU but **not** on a 24 GB GPU for the non-score parts.

## Can the Baseline Run on Any Single GPU?

| GPU | VRAM | Score tensor (9.31 TiB) | Verdict |
|-----|------|--------------------------|---------|
| RTX 4090 / A6000 | 24 GB | 397× over capacity | **IMPOSSIBLE** |
| A100 | 80 GB | 119× over capacity | **IMPOSSIBLE** |
| H100 | 80 GB | 119× over capacity | **IMPOSSIBLE** |
| H100 (80GB) × 8 NVLink | 640 GB total | 119× over per-GPU | **IMPOSSIBLE** without tensor parallelism |
| Any foreseeable single GPU | ≤100 GB | >93× over capacity | **IMPOSSIBLE** |

**No.** The baseline cannot run Case #14 on any single GPU, nor even on an 8-GPU node, without explicit tensor parallelism or algorithmic changes to avoid materializing the full [B,H,S,S] score tensor.

## Root Cause

The O(S²) score materialization is the fundamental issue. With S=100,000:

```
B × H × S² × sizeof(FP16) = 32 × 16 × 10¹⁰ × 2 = 10.24 TB
```

This is not a matter of optimization or kernel efficiency. The baseline algorithm — as implemented in `torch_transformer_benchmark.py:97` — allocates a 9.31 TiB tensor that physically cannot exist in GPU memory.

## Classification: Task-Level Issue, Not Person 3 Optimization

This is **not** an optimization opportunity for Person 3 (or any participant). It is a **task-level constraint** that must be addressed by the competition organizers or treated as an expected condition:

1. **The benchmark script is the immutable reference** (per `TASK.md:93`). Participants cannot modify `torch_transformer_benchmark.py`.
2. **The baseline must be runnable** to establish correctness. If the baseline OOMs, there is no reference output to compare against.
3. **Case #14 specifically tests long-sequence behavior** (S=100,000). The O(S²) materialization was always going to fail at this scale.
4. **Flash Attention / memory-efficient attention** eliminates the S² score tensor, but that is the *optimized* implementation, not the baseline.

### Implications for the competition

- Case #14 baseline execution will OOM on any available hardware.
- The optimized implementation (using Flash Attention or equivalent) can likely run, but correctness validation against the baseline is impossible if the baseline cannot execute.
- The organizers may need to: (a) accept that Case #14 only tests the optimized path, (b) provide a reference output file for offline comparison, or (c) reduce S or use chunked evaluation.

### Recommendation

Flag Case #14 as a baseline-infeasible shape. The optimized implementation should be validated against smaller S values where the baseline can run, and Case #14 should be treated as a performance-only benchmark (throughput measurement) rather than a correctness comparison.

## Appendix: All Intermediate Sizes

| # | Tensor | Shape | Elements | FP16 (GiB) | Notes |
|---|--------|-------|----------|------------|-------|
| 1 | x (input) | [32, 100K, 1024] | 3.28B | 6.10 | Kept for residual |
| 2 | norm1(x) | [32, 100K, 1024] | 3.28B | 6.10 | |
| 3 | q_proj(x) | [32, 100K, 1024] | 3.28B | 6.10 | Temporary |
| 4 | k_proj(x) | [32, 100K, 1024] | 3.28B | 6.10 | Temporary |
| 5 | v_proj(x) | [32, 100K, 1024] | 3.28B | 6.10 | Temporary |
| 6 | q split_heads | [32, 16, 100K, 64] | 3.28B | 6.10 | |
| 7 | k split_heads | [32, 16, 100K, 64] | 3.28B | 6.10 | |
| 8 | v split_heads | [32, 16, 100K, 64] | 3.28B | 6.10 | |
| 9 | k.transpose | [32, 16, 64, 100K] | 3.28B | 6.10 | Temporary |
| 10 | **scores** | **[32, 16, 100K, 100K]** | **5.12T** | **9,536** | **BLOCKER** |
| 11 | causal_mask | [100K, 100K] | 10B | 9.31 | bool |
| 12 | scores.float() | [32, 16, 100K, 100K] | 5.12T | 19,073 | FP32 temp |
| 13 | softmax(fp32) | [32, 16, 100K, 100K] | 5.12T | 19,073 | FP32 temp |
| 14 | **probs** | **[32, 16, 100K, 100K]** | **5.12T** | **9,536** | **BLOCKER** |
| 15 | context | [32, 16, 100K, 64] | 3.28B | 6.10 | |
| 16 | context reshaped | [32, 100K, 1024] | 3.28B | 6.10 | |
| 17 | out_proj | [32, 100K, 1024] | 3.28B | 6.10 | |
| 18 | norm2(x) | [32, 100K, 1024] | 3.28B | 6.10 | |
| 19 | ffn_in | [32, 100K, 1024] | 3.28B | 6.10 | |
| 20 | gelu(ffn_in) | [32, 100K, 1024] | 3.28B | 6.10 | |
| 21 | ffn_out | [32, 100K, 1024] | 3.28B | 6.10 | |

**Non-score peak: ~40 GiB** (feasible on 80 GB GPU)
**With score tensor: >9.31 TiB** (impossible on any single GPU)
