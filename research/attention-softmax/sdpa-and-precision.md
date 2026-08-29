# SDPA Backends and the Numerical Tolerance Budget

Why this project targets **float32 with `scaled_dot_product_attention`**, why
float16 was rejected, and how much of the tolerance budget is already spent
before we optimize anything.

Status: current as of 29 August 2026. Environment and method in
[`measurements.md`](measurements.md).

## The pass criterion is a budget, not a formality

The reference checks every output element against

```
abs_error <= 0.002  OR  rel_error <= 0.02 * |reference|
```

and passes only when **zero** elements fail
([`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py),
`compare_outputs`, lines 289-353; `AccuracyResult.passed` is
`failed_elements == 0`).

Two consequences that shaped every decision below:

1. **A single bad element out of eight million fails the whole case.** There is
   no averaging and no percentile.
2. **Deviation is measured from the reference, not from the true value.** An
   implementation that is *more* numerically accurate than the reference still
   fails if it differs from it. This is not hypothetical — see below.

## Backend availability on the target GPU

Measured on the RTX 4060 Laptop GPU (sm_89, Ada), PyTorch 2.6.0+cu124, by
attempting each backend under `torch.nn.attention.sdpa_kernel` for every head
dimension the official cases require:

| dtype | FlashAttention | memory-efficient | math |
| --- | --- | --- | --- |
| float16 / bfloat16 | available, all `d_h` in {8,32,64,128,256} | available | available |
| **float32** | **rejected** | available | available |

The flash backend refuses non-half inputs outright: *"Expected query, key and
value to all be of dtype: {Half, BFloat16}."* So in float32 we get the
memory-efficient (xFormers-derived) kernel, which still implements lever L1 — no
`N x N` materialization — but not FlashAttention-2's warp specialization.

## Why float16 was rejected

A float16 model fails the criterion for **any** attention change that reassociates
arithmetic. Tested variants, all mathematically identical to the reference:

| Variant | Result |
| --- | --- |
| identical arithmetic (control) | PASS, `max_abs = 0` |
| cached causal mask | PASS, `max_abs = 0` |
| 3-D `bmm` reshape instead of 4-D `matmul` | PASS, `max_abs = 0` |
| **fold `scale` into `Q`** | **FAIL** (cases 7, 13) |
| SDPA — flash | FAIL |
| SDPA — memory-efficient | FAIL |
| SDPA — **math** | FAIL |
| fp32 probabilities (*more* accurate than reference) | FAIL |

Seed robustness of float16 SDPA, 8 seeds per case:

| Case | seeds passing | worst error / allowance |
| --- | --- | --- |
| 13 (N=1024) | **0/8** | 1.30x |
| 9 (d_h=128) | 1/8 | 1.37x |
| 1, 7, 11 | 2/8 | 1.25-1.39x |
| 12 (N=32) | 5/8 | 1.19x |

**Mechanism.** The reference computes softmax in float32 and then rounds the
probabilities back to float16 before the `PV` matmul
(`probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)`, line 111).
Any fused attention keeps float32 accumulators throughout and never performs that
rounding, so it is *more accurate* and therefore deviates. Failures cluster on
outputs where the residual stream nearly cancels — `|reference|` at failing
elements is 0.003-0.09 against an overall median of 0.68. At those magnitudes the
effective tolerance is about 2.5% relative and the observed deviation is about
2.8%.

Passing in float16 would require **reproducing the reference's precision loss**,
which is exactly what fusion exists to remove. The margin is only 1.19-1.39x, so
this is a near miss rather than a blowout — but it is a near miss we cannot steer,
and it fails most of the time.

bfloat16 is far worse: 3.4-5.3% of elements fail, as expected from its 8-bit
mantissa.

**This constrains the whole team, not just attention.** Any transformation that
reassociates float16 arithmetic carries the same risk, including `torch.compile`
fusion (Person 1) and elementwise/GEMM fusion (Person 3).

## float32 passes, and TF32 has already spent most of the budget

With true float32 matmuls (`matmul_precision="highest"`, TF32 off), SDPA deviates
from the reference by `max_abs ~1.2e-06` — a **~1600x** margin under the 0.002
threshold.

But the harness defaults to `--allow-tf32` enabled and
`--matmul-precision high`, which routes float32 matmuls through TF32 tensor
cores (10-bit mantissa). Measured with TF32 on, `max_abs` rises to
**~9.0e-04 to 1.3e-03**, leaving only about **1.5-2x** margin.

So most of the tolerance budget is consumed by the harness's own default TF32
setting before Person 2 changes anything. This is worth stating plainly because
it bounds how aggressive any further precision reduction can be.

## Reduced-precision attention internals under a float32 model

Because the outputs stay float32, the residual stream is never quantized, and the
near-cancellation failures that sink the float16 model do not occur. Running only
the attention core in reduced precision is therefore viable.

Seed sweep, 10 seeds per case, float32 model, TF32 on, `padding_ratio=0`:

| Case | `d_h` | N | `sdpa_fp32` | `sdpa_fp16` internals |
| --- | --- | --- | --- | --- |
| 13 | 32 | 1024 | 10/10 pass, `max_abs` 1.10e-03 | 10/10 pass, `max_abs` 1.16e-03 |
| 1 | 32 | 128 | 10/10, 1.21e-03 | 10/10, 1.13e-03 |
| 7 | 8 | 128 | 10/10, 1.33e-03 | 10/10, 1.27e-03 |
| 9 | 128 | 128 | 10/10, 1.08e-03 | 10/10, 1.18e-03 |
| 12 | 32 | 32 | 10/10, 1.21e-03 | 10/10, 1.04e-03 |
| 11 | 8 | 128 | 10/10, 1.03e-03 | 10/10, 1.04e-03 |

Both are consistent across seeds, unlike the float16 model. bfloat16 internals
**fail** (`max_abs ~5e-03`, above the 0.002 absolute threshold) and are rejected.

Notably `sdpa_fp16` internals are not *more accurate* than `sdpa_fp32` here: at
these magnitudes the error is dominated by TF32 in the surrounding projections,
not by the attention core.

### But it is not faster, so the lever is not worth taking

Correctness was never the deciding factor. Clean whole-model timing (single
settle, shared baseline, interleaved) shows float16 internals **losing** to plain
float32 SDPA on every case with a trustworthy noise floor:

| Case | `sdpa_fp32` | `sdpa_fp16` internals |
| --- | --- | --- |
| 13 | **6.41x** ±3.5% | 9.51x ±110% (floor too wide to trust) |
| 7 | **1.68x** ±17.3% | 1.18x ±15.0% |
| 1 | **1.46x** ±9.7% | 1.23x ±8.8% |
| 12 | **1.41x** ±9.9% | 1.23x ±8.9% |

Casting Q/K/V down and the output back costs more than the tensor-core throughput
it buys at these sizes. The only case where float16 leads is 13, and that
measurement's ±110% noise floor makes it unusable as evidence.

**Lever L8 is therefore dropped.** It carries extra precision risk (margin ~1.5x
rather than 1600x, and untested sensitivity to `--input-scale` and
`--padding-ratio`) in exchange for a measured slowdown. Plain float32 SDPA is
both safer and faster. Revisit only if a future shape shows the attention core
dominating enough for tensor-core throughput to outweigh the cast overhead.

## Decision

1. **Target float32.** It is the harness default and the only dtype where fused
   attention provably passes.
2. **Use SDPA (memory-efficient backend) for lever L1** on every in-scope case,
   not a hand-written kernel. The full sweep in
   [`measurements.md`](measurements.md) shows all twelve cases passing and all
   twelve gaining, geometric mean ≈1.94x.
3. **Apply the bitwise-exact levers (L6, L7) everywhere**, including shapes where
   SDPA is not profitable, because they cannot affect correctness in any dtype.
4. **Do not pursue float16 internals (L8).** Measured slower than plain float32
   SDPA on every trustworthy case, for added precision risk.
5. **Raise the float16 finding with the team and the organizer.** If float16 is
   scored, the reference's own rounding makes fused attention close to
   unpassable, which is arguably a property of the test rather than of any
   submission.

## Sources

- **PyTorch, `torch.nn.functional.scaled_dot_product_attention` documentation.**
  <https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html>.
  Accessed 29 August 2026. Documents the three backends, the `scale=` parameter,
  and `is_causal=True` semantics (lower-triangular for square masks, mutually
  exclusive with `attn_mask`). States the math backend keeps intermediates in
  float32 for half-precision inputs, which is why even the math backend does not
  reproduce the reference's explicit float16 rounding of probabilities.

- **PyTorch, `torch.backends.cuda.matmul.allow_tf32` / TF32 semantics.**
  <https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices>.
  Accessed 29 August 2026. TF32 uses a 10-bit mantissa for float32 matmuls on
  Ampere and later. This explains the measured drop in tolerance headroom from
  ~1600x to ~1.5-2x under the harness's default `--allow-tf32` setting.

- **Dao, T., "FlashAttention-2", ICLR 2024.**
  <https://arxiv.org/abs/2307.08691>. Accessed 29 August 2026. The flash backend
  PyTorch selects for half-precision inputs; unavailable in float32, which is why
  the float32 path uses the memory-efficient backend instead.

- **Rabe, M. N. and Staats, C., "Self-attention Does Not Need O(n²) Memory".**
  <https://arxiv.org/abs/2112.05682>. Accessed 29 August 2026. The algorithm
  behind the memory-efficient backend that carries lever L1 in float32.

- **Reference implementation:** this repository's
  [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py) at
  commit `7eb8fb1`. Relevant symbols: `BaselineSelfAttention.forward` line 111
  for the float32 softmax and float16 round-trip, and `compare_outputs`
  lines 289-353 for the exact `abs OR rel` criterion and the
  `failed_elements == 0` pass rule.
