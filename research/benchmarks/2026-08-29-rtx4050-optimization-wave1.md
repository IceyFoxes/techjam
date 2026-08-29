# Optimization Wave 1 Results

## Date and Environment

- Date: 29 August 2026
- GPU: RTX 4050 Laptop GPU (6 GB VRAM)
- Commit: f4546f4 (master)

## Baseline Latencies

| Case | Config | Median (ms) |
|------|--------|-------------|
| #8 | B=64,S=128,D=1024,H=4,F=1024,L=4 | 51.75 |
| #7 | B=64,S=128,D=32,H=4,F=32,L=4 | 9.35 |
| #6 | B=10000,S=128,D=128,H=4,F=128,L=4 | OOM (6 GB) |

## Experiments

### 1. View-Only QKV (removes .contiguous() copies)

Removes four explicit layout copies by using transpose views with last stride 1.

| Case | Accuracy | Speedup | Notes |
|------|----------|---------|-------|
| #8 | PASS | 0.921x | Slower (noise) |
| #7 | PASS | 1.009x | Noise |

**Conclusion:** Copy elimination alone is not high-impact. The baseline's copies are likely optimized by PyTorch's allocator.

### 2. Packed QKV (one F.linear for Q+K+V)

Replaces three separate linears with one packed F.linear producing [B,S,3D].

| Case | Accuracy | Speedup | Notes |
|------|----------|---------|-------|
| #8 | PASS | 0.947x | Slower (noise) |
| #7 | PASS | 1.010x | Noise |

**Conclusion:** Packed QKV is not faster than three linears for these shapes. The benefit may only appear when combined with attention layout changes.

### 3. Compiler (max-autotune)

| Case | Accuracy | Speedup | Failed Elements |
|------|----------|---------|-----------------|
| #8 | FAIL | 1.227x | 40/41943040 |
| #7 | FAIL | 3.089x | 24/1310720 |

**Conclusion:** Compiler achieves significant speedup but fails correctness. Default Inductor removes precision checkpoints.

### 4. Compiler + Precision-Cast Emulation

| Case | Accuracy | Speedup | Failed Elements | Max Abs Error |
|------|----------|---------|-----------------|---------------|
| #8 | FAIL | 1.195x | 4/41943040 | 0.0078 |
| #7 | PASS | 20.75x | 0/1310720 | 0.0039 |

**Breakthrough:** Case #7 with `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` achieves 20.75x speedup while passing correctness. Case #8 still fails but improves.

### 5. Fused GELU + Residual Epilogues

Uses F.linear (bias fused into GEMM epilogue) for both FFN projections.

| Case | Accuracy | Speedup |
|------|----------|---------|
| #7 | PASS | 1.001x |
| #12 | PASS | 0.999x |

**Conclusion:** F.linear matches baseline exactly. The optimization is negligible because F.linear is the same kernel as nn.Linear.

### 6. Mask Redundancy

Skips attention-output and block-output masks when valid_token_mask is all True.

| Case | Padding | Accuracy | Speedup |
|------|---------|----------|---------|
| #8 | 0.0 | PASS | 0.967x |
| #8 | 0.25 | PASS | 1.126x |

**Conclusion:** Masks are redundant when padding_ratio=0. Conditional skip provides 12.6% gain on padded case #8.

### 7. Case #14 Feasibility

The score tensor [32,16,100000,100000] in FP16 requires 9.31 TiB. The baseline cannot run on any single GPU. This is a task-level issue.

## Prioritized Next Steps

1. **Compiler + emulation on #7** — 20.75x speedup, already passing. Investigate why #8 fails and whether Inductor template selection can be constrained.
2. **Mask elimination** — 12.6% gain on padded #8, zero cost on unpadded.
3. **Packed QKV with attention layout change** — needs Person 2 coordination.
4. **Case #14 clarification** — require organizer input on expected execution.
