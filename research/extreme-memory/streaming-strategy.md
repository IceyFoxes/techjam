# Exact Streaming Strategy

## Online Attention Recurrence

For one query row, maintain FP32 state over processed key tiles:

```text
m = running maximum score
l = sum_j exp(score_j - m)
A = sum_j exp(score_j - m) * value_j
```

For a new tile with scores `s`:

```text
tile_m = max(s)
new_m  = max(m, tile_m)
alpha  = exp(m - new_m)
p       = exp(s - new_m)
new_l   = alpha * l + sum(p)
new_A   = alpha * A + p @ V_tile
```

After all valid keys, `context=A/l`. In real arithmetic this is the same dense
softmax attention, not an approximation. It needs only a score tile plus
per-query maximum, normalizer, and output accumulator. FlashAttention and
FlashAttention-2 apply this idea while keeping the tile on chip.

Initialize `m=-inf`, `l=0`, and `A=0`. Skip a fully masked tile or explicitly
set its exponential contribution to zero so the implementation never evaluates
`-inf - -inf`.

## Recommended Schedule For Assumed #14

1. Process one batch sample through the complete layer at a time. The benchmark
   operations have no cross-sample dependence.
2. Compute that sample's normalized input in chunks if a full sample temporary
   would make the global input/output live set unsafe.
3. Generate and retain full K and V for that sample. Together they cost only
   390.625 MiB in FP16/BF16.
4. Generate Q for the current query tile only.
5. Run all 16 heads through fused online attention over K/V tiles. Keep `m`,
   `l`, and the context accumulator in FP32.
6. Finalize all heads for the query tile, concatenate to width 1024, cast at the
   intended dtype boundary, and issue one normal output projection.
7. Add the residual and write the completed tile. Release Q/context scratch.
8. Chunk the second LayerNorm and FFN over tokens; these operations do not mix
   sequence positions.
9. Complete all layers for the current sample before moving to the next sample
   when this avoids another global activation buffer.

All heads should remain together unless measured evidence requires head
chunking. A 128-token, all-head context is only 256 KiB in FP16. One full-width
output GEMM is faster and numerically closer to the baseline than accumulating
16 head-slice GEMMs.

## Why Not Generate K/V For Every Query Tile

A query-outer Flash-style kernel scans K and V once per query tile. If K/V
projections are regenerated during every scan, `Bq=128` gives 782 query tiles.
Across assumed #14, one K or V projection costs about 6.71 TFLOPs. Regenerating
both for every query tile costs about 10.5 PFLOPs, around eight times the full
non-causal attention core.

Retaining one sample's K/V is the better time-memory tradeoff. A key-outer
alternative avoids K/V regeneration but repeatedly reads and writes FP32 state
for all query rows, creating a large HBM-bandwidth cost. Use it only if
per-sample K/V cannot fit on the confirmed judging device.

## Causal And Padding Semantics

The root reference applies a top-left square causal mask: query position `i`
may attend to key positions `j <= i`. A custom tiled kernel must compare global
indices, skip future key tiles, and mask only the tile intersecting the causal
diagonal.

For right-padded benchmark samples:

- Valid query rows attend only to keys in `[0, valid_length)`.
- Causal valid row `i` attends to `[0, i]`.
- Padded query rows can be skipped, but the attention output, block output, and
  final normalized output must be written as zero exactly as the root harness
  does at lines 120-121, 143-144, and 170-171.
- Do not allow output-projection bias to survive on padded rows.

PyTorch Flash SDPA currently rejects a non-null `attn_mask`. The root generator
always returns a mask, even when every token is valid. Pass `attn_mask=None`
only after proving that the sample has no invalid keys. For right-padded
samples, use actual valid lengths and per-sample or packed variable-length
execution rather than constructing a dense causal-plus-padding mask.

There is a subtle query-chunk issue. A query tile from the middle of a sequence
cannot blindly call a non-square causal kernel against all `N` keys because
causal alignment differs by API/version. A custom kernel should use global
positions. For FlashAttention implementations with bottom-right non-square
alignment, query chunk `[q0:q1)` can use `K[:q1]` and `V[:q1]`; this must be
verified against the installed version.

## PyTorch SDPA First, Custom Kernel Second

The current PyTorch source considers L4's SM89, FP16/BF16, four-dimensional
Q/K/V, and `head_dim=64` eligible for Flash attention when `attn_mask=None`.
No explicit 100,000-token upper bound was found, but square 100K support is not
documented or demonstrated by the upstream test suite.

Eligibility and execution must both be tested on the installed wheel:

```python
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

params = SDPAParams(q, k, v, None, 0.0, True, False)
assert can_use_flash_attention(params, debug=True)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
```

Forcing the backend is important: an unnoticed fallback to math attention can
allocate a quadratic score/probability tensor and OOM.

## Numerical-Correctness Risk

"Exact attention" means algebraically exact, not bitwise equivalent to the
root FP16/BF16 reference. The reference performs these observable boundaries:

1. Q, K, and V projections are rounded to input dtype.
2. `QK^T` and score scaling produce an input-dtype score tensor.
3. Scores are converted to FP32 for softmax.
4. Probabilities are rounded back to input dtype.
5. Low-precision probabilities are multiplied by V.

Flash implementations commonly accumulate `QK^T`, online softmax state, and
context in FP32, use a different reduction order and exponential formulation,
and normalize only after the key loop. PyTorch documents that fused SDPA
backends, batched versus sliced operations, and changed reduction order can
produce different floating-point results.

The repository requires every element to satisfy absolute error at most 0.002
or error at most `0.02 * abs(reference)`. Billions of output values make rare
tails important. Validation must include maximum errors and near-zero outputs,
not only mean error.

If one-pass online attention fails, the closest practical fallback is a
two-pass query-tile kernel:

1. First pass computes the final FP32 row maximum and denominator.
2. Second pass recomputes scores, normalizes by the final denominator, casts
   each probability to input dtype, and performs `P @ V`.

This better matches the reference's probability cast but raises attention
matmul work from two to three matmuls, approximately 50%, and still does not
guarantee the same GEMM/reduction order.

## Chunk Sweep

Start with `Bq,Bk` in `{64,128}` as motivated by FlashAttention-2, then include
`Bq=256` only if register/shared-memory occupancy allows it. For each supported
dtype and causal mode:

1. Reject configurations that fail correctness on feasible explicit-reference
   lengths.
2. Run each configuration in an isolated process.
3. Record allocated and reserved peak VRAM, median latency, p90 latency, and
   backend/kernel identity.
4. Increase batch chunk size after query/key tuning and select the largest
   chunk that stays below 85% of total memory across repeated trials.
5. Retain at least 10-15% safety margin for CUDA context, library workspace,
   allocator variance, and integration with the full layer.

Do not hardcode the smallest chunk. The analytic scratch table shows that tile
scratch is tiny relative to retained activations and per-sample K/V; overly
small tiles mainly sacrifice Tensor Core utilization and add loop overhead.
