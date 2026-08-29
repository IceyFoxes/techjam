# Decomposing the Attention Block into Optimizable Sub-Problems

Person 2 owns `QK^T`, the causal softmax, and multiplication by `V`. This
document subdivides that region into the smallest units that can be reasoned
about and optimized independently, states the cost of each, and identifies the
algorithmic lever that applies to it.

Status: current as of 29 August 2026. Measurements were taken on the RTX 4060
Laptop GPU described in [`measurements.md`](measurements.md).

## The stage boundary

Per layer, the reference
([`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py),
`BaselineSelfAttention.forward`, lines 85-122) performs:

| # | Stage | Shape | Owner |
| --- | --- | --- | --- |
| 0 | `norm1(x)` LayerNorm | `[B, N, d]` | Person 3 |
| 1 | `q_proj`, `k_proj`, `v_proj` | `[B, N, d]` | Person 3 |
| 2 | `_split_heads` reshape + `.contiguous()` | `[B, H, N, d_h]` | boundary |
| **3** | **`S = Q K^T`** | `[B, H, N, N]` | **Person 2** |
| **4** | **`S' = S * d_h^{-1/2}`** | `[B, H, N, N]` | **Person 2** |
| **5** | **`S'' = masked_fill(S', causal ∪ padding)`** | `[B, H, N, N]` | **Person 2** |
| **6** | **`P = softmax(S'')`** in fp32, cast back | `[B, H, N, N]` | **Person 2** |
| **7** | **`O = P V`** | `[B, H, N, d_h]` | **Person 2** |
| 8 | transpose + `.contiguous()` + view | `[B, N, d]` | boundary |
| 9 | `out_proj` | `[B, N, d]` | Person 3 |

Stages 3-7 are the optimization surface. Stages 2 and 8 are shared with Person 3
and should be negotiated rather than changed unilaterally.

## Why this is a memory problem, not a compute problem

Let `S` denote the score tensor with `B·H·N²` elements. Stages 4, 5, and 6 all
read and write `S` with **zero arithmetic reuse** — every element is touched a
constant number of times and then discarded. Stages 3 and 7 are the only ones
with reuse (each does `2·B·H·N²·d_h` FLOPs).

The arithmetic intensity of the middle is therefore `O(1)` FLOP/byte, while the
GEMMs are `O(d_h)`. With `d_h ∈ {8, 32, 64, 128, 256}` the middle is between 8x
and 256x less efficient per byte moved.

For case 13 (`B=64, H=4, N=1024, d_h=32`), `S` is 268,435,456 elements = **1.00 GB
in fp32**, and the eager path moves it roughly twelve times per layer:

| Stage | Traffic |
| --- | --- |
| scale (read + write) | 2.00 GB |
| causal mask fill (read + write) | 2.00 GB |
| padding mask fill (read + write) | 2.00 GB |
| softmax row max (read) | 1.00 GB |
| softmax exp (read + write) | 2.00 GB |
| softmax row sum (read) | 1.00 GB |
| softmax divide (read + write) | 2.00 GB |
| **total per layer** | **~12 GB** |
| **x 4 layers** | **~48 GB** |

Against the measured hardware (~272 GB/s, ~20 TFLOP/s TF32):

- memory time: 48 GB / 272 GB/s = **~176 ms**
- attention GEMM: 137 GFLOP / 20 TFLOP/s = **~6.9 ms**

**Attention on this shape is memory-bound by roughly 25x.** The measured eager
baseline (~200-400 ms depending on clock state) is consistent with the memory
estimate, not the compute estimate.

This is the single most important fact for Person 2: **making the matmuls faster
is nearly worthless; removing `S` from memory is nearly everything.**

Operator-level profiling agrees. On case 13 the actual attention matmul
(`aten::bmm`) is the *smallest* attention cost — see
[`measurements.md`](measurements.md).

## Score-tensor size per case

Cases 6 and 14 are excluded — extreme-shape memory strategy is Person 4's scope
per [`four-way-team-split.md`](../team-coordination/four-way-team-split.md).

| Case | B | H | N | d_h | `S` elements | `S` fp32 | attn GFLOP/layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 4 | 128 | 32 | 4,194,304 | 0.016 GB | 0.54 |
| 2 | 1 | 4 | 128 | 32 | 65,536 | ~0 GB | 0.01 |
| 3 | 4 | 4 | 128 | 32 | 262,144 | 0.001 GB | 0.03 |
| 4 | 16 | 4 | 128 | 32 | 1,048,576 | 0.004 GB | 0.13 |
| 5 | 128 | 4 | 128 | 32 | 8,388,608 | 0.031 GB | 1.07 |
| 7 | 64 | 4 | 128 | 8 | 4,194,304 | 0.016 GB | 0.13 |
| 8 | 64 | 4 | 128 | 256 | 4,194,304 | 0.016 GB | 4.29 |
| 9 | 64 | 1 | 128 | 128 | 1,048,576 | 0.004 GB | 0.54 |
| 10 | 64 | 2 | 128 | 64 | 2,097,152 | 0.008 GB | 0.54 |
| 11 | 64 | 16 | 128 | 8 | 16,777,216 | 0.062 GB | 0.54 |
| 12 | 64 | 4 | 32 | 32 | 262,144 | 0.001 GB | 0.03 |
| 13 | 64 | 4 | 1024 | 32 | 268,435,456 | **1.000 GB** | **34.36** |

Case 13 has a score tensor **16x larger than every other case combined**, and is
the only one where `N` exceeds 128. It is the primary target. Case 8 is the only
compute-heavy one (`d_h=256`), and there the projections dominate anyway.

## The algorithmic levers

### L1. Never materialize `S` (fuses stages 3-7)

The dominant lever. Tile the computation so each block of `S` is produced in
registers/SRAM, consumed by the softmax, multiplied into `V`, and discarded.
Traffic drops from `O(B·H·N²)` to `O(B·H·N·d_h)`.

This requires L4 (online softmax) to be correct, because a tile cannot know the
row maximum until every tile in that row has been seen.

Established by Rabe & Staats and made practical on GPUs by FlashAttention. In
this project we get it from `scaled_dot_product_attention` rather than writing
it; see [`sdpa-and-precision.md`](sdpa-and-precision.md).

### L2. Fold the scale into `Q` (removes stage 4)

`(Q·s)K^T = s(QK^T)` mathematically, so the separate `* self.scale` pass over
`S` is unnecessary — a full read+write of `S` removed for free. `SDPA` exposes
this via its `scale=` argument.

**Precision caveat, measured:** this reassociation is *not* free in a float16
model — folding the scale alone flips cases 7 and 13 to FAIL. In float32 it is
safe. See [`sdpa-and-precision.md`](sdpa-and-precision.md).

### L3. Exploit causal structure to skip half the work

Every official case has `causal: true`. The reference computes all `N²` scores
and then discards the strict upper triangle with `masked_fill`. A tiled kernel
can skip whole blocks above the diagonal entirely, saving close to **50% of both
the GEMM FLOPs and the score traffic**. FlashAttention-2 lists this as one of its
three contributions.

This lever is *unavailable* to the eager path and is a direct argument for a
fused kernel over any amount of eager tuning.

### L4. Online (single-pass) softmax

Safe softmax normally needs three passes: row max, exponentiate and sum, then
divide. Milakov & Gimelshein show a single streaming pass maintaining a running
max `m` and running normalizer `d`:

```
m_i = max(m_{i-1}, x_i)
d_i = d_{i-1} · exp(m_{i-1} - m_i) + exp(x_i - m_i)
```

The rescaling factor `exp(m_{i-1} - m_i)` corrects the running sum whenever a new
maximum appears. This is what makes L1 possible, and it removes two full passes
over `S` even on its own.

### L5. Defer the normalization

Accumulate the unnormalized `exp(S)V` and the normalizer `l` separately, then
divide once at the end on the `[B,H,N,d_h]` output instead of on the `[B,H,N,N]`
probabilities. The divide moves from `N²` elements to `N·d_h` — for case 13 that
is 1024 vs 32 per row, a **32x reduction** on that stage. FlashAttention-2
describes this as reducing non-matmul FLOPs.

### L6. Never build the mask tensor

The reference allocates `torch.ones((N,N), bool).triu(1)` **per layer, per
forward call** (line 100-102). For case 13 that is a 1 MB boolean allocation plus
a full `N²` write and read, four times per forward, for a matrix that never
changes. Caching it is bitwise-exact; computing the predicate from tile indices
inside a kernel removes it entirely.

### L7. Skip masking that provably does nothing

At `padding_ratio=0` — the harness default — `valid_token_mask` is all-true, so
the padding `masked_fill` is a no-op that still costs a full read+write of `S`.
Skipping it is bitwise-exact. Measured worth up to 1.43x whole-model, but the
gain disappears when padding is present, and the `.all()` test must be hoisted to
one host sync per forward rather than one per layer or it costs more than it
saves. See [`measurements.md`](measurements.md).

### L8. Spend the numerical tolerance budget deliberately — tested and rejected

The pass criterion is `abs_err <= 0.002 OR rel_err <= 2%`, which is a *budget*
that could in principle buy tensor-core throughput for stages 3 and 7.

Measured: a float32 model whose attention core runs internally in float16 does
pass, 10/10 seeds on every in-scope case (`max_abs ≈ 1.2e-03`), and bfloat16 does
not. **But it is slower than plain float32 SDPA** on every case with a
trustworthy noise floor — the Q/K/V down-cast and output up-cast cost more than
the throughput they buy at these sizes.

**Lever rejected.** Kept here because the negative result is worth not
rediscovering. Note also that the harness's default TF32 has already spent most
of the budget: headroom drops from ~1600x to ~1.5-2x with `--allow-tf32` on. See
[`sdpa-and-precision.md`](sdpa-and-precision.md).

## Lever-to-stage map

| Lever | Removes / changes | Stages | Safe in fp32 | Safe in fp16 |
| --- | --- | --- | --- | --- |
| L1 no `S` materialization | ~10 GB/layer traffic | 3-7 | yes | no (fails tolerance) |
| L2 fold scale | one `S` read+write | 4 | yes | **no** |
| L3 causal block skip | ~50% FLOPs + traffic | 3,5,7 | yes | no |
| L4 online softmax | two `S` passes | 6 | yes | no |
| L5 deferred normalize | `N²` divide -> `N·d_h` | 6 | yes | no |
| L6 cached / implicit mask | mask alloc + `N²` pass | 5 | yes | **yes (bitwise)** |
| L7 skip no-op padding mask | one `S` read+write | 5 | yes | **yes (bitwise)** |
| ~~L8 low-precision internals~~ | ~~GEMM throughput~~ | 3,7 | rejected (slower) | n/a |

Only L6 and L7 are bitwise-exact. Everything else reassociates arithmetic, which
float32 tolerates comfortably and float16 does not.

## Order of work

1. **L1 via SDPA in float32** — largest win, lowest risk. Measured **6.41x
   whole-model on case 13** (±3.5%), 1.4-1.7x on cases 1, 7, 12.
2. **L6 + L7** — bitwise-exact, apply everywhere including shapes where SDPA
   loses. Must hoist the all-true test to one host sync per forward.
3. **L3 via a custom kernel** — only if profiling after steps 1-2 still shows the
   upper triangle being computed. This is the main reason to consider Triton, and
   the only lever SDPA does not already deliver.
4. Leave case 8 (`d_h=256`) on the eager path; SDPA regresses to 0.64x there and
   the projections dominate that case anyway (Person 3).

L8 was tested and rejected. Cases 6 and 14 are Person 4's.

## Sources

- **Rabe, M. N. and Staats, C., "Self-attention Does Not Need O(n²) Memory".**
  <https://arxiv.org/abs/2112.05682>. Accessed 29 August 2026. Shows attention
  needs only `O(log n)` memory in principle and `O(√n)` in a practical numerically
  stable implementation, by chunking the sequence and using lazy softmax
  normalization; reports 59x memory reduction at sequence length 16,384. This is
  the theoretical basis for lever L1 and for PyTorch's memory-efficient backend,
  which is the only non-math SDPA backend available in float32.

- **Dao, T., "FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning", ICLR 2024.** <https://arxiv.org/abs/2307.08691>. Accessed
  29 August 2026. Three contributions relevant here: reducing non-matmul FLOPs
  (lever L5), parallelizing across thread blocks to raise occupancy, and
  distributing work between warps to cut shared-memory traffic. Reports ~2x over
  FlashAttention-1 and 50-73% of theoretical peak on A100. Its causal-masking
  block-skipping is lever L3.

- **Milakov, M. and Gimelshein, N., "Online normalizer calculation for softmax",
  2018.** <https://arxiv.org/abs/1805.02867>. Accessed 29 August 2026. Introduces
  the single-pass running-max/running-normalizer recurrence that reduces softmax
  memory accesses from 4 to 3 per element and enables fusing softmax into a
  streaming kernel. This is lever L4 and the precondition for L1.

- **PyTorch, `torch.nn.functional.scaled_dot_product_attention` documentation.**
  <https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html>.
  Accessed 29 August 2026. Documents three backends (FlashAttention-2,
  memory-efficient/xFormers, and a C++ math fallback), the `scale=` argument that
  implements lever L2, and `is_causal=True` semantics (lower-triangular for square
  masks; cannot be combined with an explicit `attn_mask`). Notes the math backend
  keeps intermediates in float32 when inputs are half precision.

- **Reference implementation:** this repository's
  [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py) at
  commit `7eb8fb1`, symbol `BaselineSelfAttention.forward` (lines 85-122) for
  stages 3-7, and `compare_outputs` (lines 289-353) for the pass criterion.
