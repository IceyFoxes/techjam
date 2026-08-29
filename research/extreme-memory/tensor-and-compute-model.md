# Tensor And Compute Model

## Scope And Units

This model uses the internally assumed #14 dimensions only:

```text
B=32, H=16, N=100000, D=1024, d_h=D/H=64
```

It is not an official test shape. Byte counts are exact decimal values where
shown. MiB and GiB use powers of two. FP16/BF16 use 2 bytes, FP32 uses 4 bytes,
and CUDA boolean tensors use 1 byte.

The local L4 reports 23,034 MiB (22.494 GiB) total memory through `nvidia-smi`.
NVIDIA specifies the L4 as a 24 GB Ada GPU with 300 GB/s memory bandwidth and
242 sparse FP16/BF16 Tensor Core TFLOP/s. Dense peak is one half of the starred
sparse figure, approximately 121 TFLOP/s. See the [source catalog](sources.md).

## Core Formulas

For scalar size `s` bytes:

```text
activation bytes       = B * N * D * s
QKV bytes              = 3 * B * N * D * s
score bytes            = B * H * N * N * s
one-sample K+V bytes   = 2 * N * D * s
non-causal attn FLOPs  = 4 * B * N^2 * D
causal attn FLOPs      = 2 * B * N * (N + 1) * D
four projection FLOPs  = 8 * B * N * D^2
```

The attention FLOP convention counts the `QK^T` and `PV` fused multiply-adds
as two operations each. It excludes masking, maximum reductions, exponentials,
sums, scaling, and normalization.

## Assumed #14 Tensor Sizes

| Object | Elements | Type | Size |
| --- | ---: | --- | ---: |
| Input or output `[B,N,D]` | 3,276,800,000 | FP16/BF16 | 6,553,600,000 bytes = 6,250 MiB = 6.104 GiB |
| Same activation | 3,276,800,000 | FP32 | 12,500 MiB = 12.207 GiB |
| Output-sized mask | 3,276,800,000 | bool | 3,125 MiB = 3.052 GiB |
| Valid-token mask `[B,N]` | 3,200,000 | bool | 3.052 MiB |
| Full Q+K+V | 9,830,400,000 | FP16/BF16 | 18,750 MiB = 18.311 GiB |
| One sample's K+V | 204,800,000 | FP16/BF16 | 390.625 MiB |
| Scores `[B,H,N,N]` | 5,120,000,000,000 | FP16/BF16 | 10.24 TB = 9.313 TiB |
| Scores or probabilities | 5,120,000,000,000 | FP32 | 20.48 TB = 18.626 TiB |
| Causal mask `[N,N]` | 10,000,000,000 | bool | 10.00 GB = 9.313 GiB |
| Default-harness FFN intermediate `[B,N,2048]` | 6,553,600,000 | FP16/BF16 | 12,500 MiB = 12.207 GiB |

The input plus full QKV is 25,000 MiB before context, scores, masks, weights,
or workspace, so replacing only the score operation is insufficient.

## Compute Lower Bound

| Work per layer | FLOPs | L4 dense FP16/BF16 peak lower bound |
| --- | ---: | ---: |
| Non-causal attention | 1,310,720,000,000,000 | 10.83 s |
| Causal attention | 655,366,553,600,000 | 5.42 s |
| Q, K, V, and output projections | 26,843,545,600,000 | 0.22 s |

These times assume an impossible 100% of peak and omit all non-matmul work.
FlashAttention-2 reports that attention forward reaches less than GEMM peak,
and the L4's 300 GB/s bandwidth and 72 W power limit further separate real
latency from this bound.

At the benchmark defaults, each model receives 20 warmups and 300 timed calls.
That is 320 calls per model. Even the causal attention-only peak bound is about
28.9 minutes per layer per model, or 2.9 hours for six layers. This excludes
the five untimed accuracy calls per model and all omitted work.

## Baseline Active-Set Bounds

With the root benchmark's default six layers and `ffn_dim=2048`, one FP16 model
contains about 96.1 MiB of parameters. Both baseline and candidate stay on the
GPU, so approximately 192.3 MiB is resident before activations.

Let `A=6,250 MiB`, the size of one full FP16 activation.

| Phase | Visible live set | Lower-bound size | Result on 23,034 MiB |
| --- | ---: | ---: | --- |
| Input scaling | 2A + models | 12,692 MiB | Fits |
| Block input + norm + Q projection | 3A + models | 18,942 MiB | Fits narrowly |
| Add transposed-Q contiguous copy | 4A + models | 25,192 MiB | OOM |
| Input + normalized input + full QKV | at least 5A + models | 31,442 MiB | OOM |
| One raw FP16 score | score alone | 9.313 TiB | OOM |
| Residual + norm2 + default FFN input | 4A + models | 25,192 MiB | OOM |
| Residual + FFN input + GELU output | 5A + models | 31,442 MiB | OOM |

The exact allocator request that fails can move earlier due to the CUDA
context, cuBLAS workspace, another process, or fragmentation. The structural
classification does not change.

## Analytic Attention-Tile Scratch

For a one-sample schedule retaining full sample K/V, a conservative
attention-only scratch estimate for one query tile is:

```text
Q FP16             = Bq * D * 2
context FP16       = Bq * D * 2
FP32 accumulator   = Bq * D * 4
score/prob tile    = H * Bq * Bk * 2
FP32 max and sum   = 2 * H * Bq * 4
```

Real fused kernels keep per-CTA score tiles in registers/shared memory rather
than allocating all heads' tiles in HBM, so this table is a conservative
comparison, not a measured PyTorch peak.

| `Bq` | `Bk` | Conservative scratch | Query-head CTAs for `N=100000` |
| ---: | ---: | ---: | ---: |
| 64 | 64 | 0.633 MiB | 25,008 |
| 128 | 64 | 1.266 MiB | 12,512 |
| 128 | 128 | 1.516 MiB | 12,512 |
| 256 | 128 | 3.031 MiB | 6,256 |
| 256 | 256 | 4.031 MiB | 6,256 |

The fixed per-sample K/V cost, 390.625 MiB, dominates these tile sizes. All
listed query sizes provide abundant grid parallelism over 16 heads. Selection
should therefore optimize occupancy and register/shared-memory use, not choose
the smallest tile merely to save a few MiB.

## Assumed #6

Only an approximate 1.3 GB FP16 score size is known. One such tensor consumes
about 1,240 MiB if `GB` is decimal. The eager baseline can transiently require:

- About twice that size for raw and scaled scores.
- Up to roughly five FP16-score equivalents around the FP32 softmax path,
  approximately 6.5 GB, before QKV and other activations.
- Further score-sized outputs for causal or key-padding `masked_fill` calls.

This does not prove an OOM on the L4. Full shape dimensions are required to
model QKV, FFN, masks, model weights, and the checker.
