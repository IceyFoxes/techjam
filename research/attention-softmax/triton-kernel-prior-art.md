# Prior Art for the Case-14 Triton Kernel

Status: current as of 30 August 2026. Design input for the fused kernel proposed
in [`long-sequence-attention.md`](long-sequence-attention.md) section 6.

Question asked: before writing a custom Triton kernel for order-2 polynomial
feature-map linear attention at case 14's shapes, what already exists in
libraries and literature, and how much of it can we reuse?

## Summary

**The algorithm we validated is already implemented, in production quality, by
`fla-org/flash-linear-attention` — and it refuses our head dimension.** Every
published implementation of second-order Taylor linear attention assumes a head
dimension of 16, by design, because the second-order state is `O(d^2 * d)` and
`d = 16` is the largest value that keeps that state in registers. Our case-14
head dimension is 64, which puts us outside the regime all of this code targets.

That is a useful result rather than a blocked one: it explains the memory-bound
behaviour measured in `long-sequence-attention.md` section 4.4, tells us the
off-the-shelf kernels cannot be reused as-is, and identifies feature-dimension
tiling as the specific thing our kernel must do that theirs do not.

## 1. The decisive constraint

`fla-org/flash-linear-attention` implements Based at
`fla/ops/based/fused_chunk.py`, which begins with:

```python
assert q.shape[-1] <= 16, 'only support feature dimension up to 16.'
```

with block sizes `BT = 16`, `BK = 16`, `BV in [16, 32]`, and a state carried as
three accumulators — zero-order scalar, first-order `BK x BV`, second-order
`BK^2 x BV`. The arithmetic is `1 + s + 0.5 * s * s`, i.e. the plain Taylor
constant rather than the `sigma`-fitted one we measured to be 2.3x more accurate.

The Based authors state the reason for `d' = 16` directly: it keeps the KV state
*in thread registers*, and "the latency of matrix multiplication for 16x16 vs.
64x64 matrices are roughly equal" because tensor cores saturate either way. Their
IO-aware kernel reduces HBM-to-SRAM movement by `O(N d'^2)` bytes and
SRAM-to-register movement by `O(N d'^2 d)` bytes — both terms that explode at
`d' = 64`.

### Why this matters for us, in bytes

Measured on the target GPU (RTX 4060 Laptop, sm_89): 24 SMs, **100 KiB shared
memory per SM**, 48 KiB per block by default.

| tensor | size at `d_h = 64` | fits in SRAM? |
| --- | ---: | :--- |
| order-2 state `S [d^2, d]`, fp16 | **512 KiB / head** | no, 5x over a whole SM |
| materialised feature tile `[C=512, d^2]`, fp16 | 4.0 MiB / head | no |
| feature *block* tile `[C=512, 256]`, fp16 | 256 KiB | no, but streams |
| state slice `[256, d]`, fp16 | **32 KiB** | **yes, in a 48 KiB block** |

At `d' = 16` the entire state is `16*16*16*2 = 8 KiB` and lives in registers,
which is the whole basis of the Based/FLA kernel design. At `d_h = 64` it is
512 KiB and cannot. **A direct port is not possible; the state must be tiled over
the feature dimension.**

## 2. What our kernel must do that published kernels do not

The last row of the table is the design. Split the `d^2 = 4096` feature dimension
into blocks of 256; for each block, generate the corresponding slice of
`a (x) a` in registers and multiply it against a `[256, 64]` slice of the state
held in shared memory. This is a K-dimension tiling of the state GEMM, with the
feature map fused into the tile's prologue so it never reaches HBM.

Estimated traffic, per sample-layer at `N = 100000, H = 16` (arithmetic, not yet
measured):

| design | dominant traffic | note |
| --- | ---: | --- |
| current PyTorch order-2 | **~51 GB** | writes and re-reads `[C, 4096]` features |
| fused, feature-tiled | **~3.2 GB** | streams the state instead; features never materialise |

That is the ~16x traffic reduction which should convert the measured 1.19x into
something near the compute bound. The compute is ~1.7 TFLOP against the exact
path's ~20.5 TFLOP; at the exact path's realised 28 TFLOPS that is ~61 ms
against 604 ms measured today. Treat 2-4x as the honest expectation and the
remainder as headroom, not as a forecast.

## 3. Directly relevant prior art

### 3.1 Reusable as design reference

- **`fla-org/flash-linear-attention`** — MIT licensed, Triton, the canonical
  chunkwise-parallel linear attention library. Implements GLA, RetNet, DeltaNet,
  Gated DeltaNet, **Based and Rebased**, RWKV6/7, Mamba/Mamba2. The Based
  directory carries `naive.py`, `fused_chunk.py` and `parallel.py`. **Not usable
  at `d_h = 64`** per section 1, but `naive.py` is a correctness oracle for a
  reduced-`d` configuration, and the chunkwise structure is the one to copy.
  The MIT license permits reuse with attribution.

- **Tiled Flash Linear Attention (TFLA)** — the most directly applicable paper.
  It adds a *second* level of sequence parallelism inside each chunk, explicitly
  to fix the problem that "limited chunks force intermediate states into GPU
  memory, reducing computational efficiency," and to allow "arbitrary large chunk
  sizes and high arithmetic intensity." That is our problem restated. Reference
  Triton implementation at `NX-AI/mlstm_kernels`.

- **The Anatomy of a Triton Attention Kernel** — practical guidance: block sizes
  drive both correctness and performance, register pressure and shared-memory
  capacity are the binding constraints, and autotuning beats manual tuning for
  portability. Directly relevant given we must pick a feature-block size.

### 3.2 Attacks our exact bottleneck, but unverified

- **PolySketchFormer (Kacham et al., ICML 2024)** — sketches the polynomial
  kernel's tensor-product feature map down to a smaller dimension using
  techniques from randomised numerical linear algebra, with approximation
  guarantees, reporting 2.5-4x over FlashAttention at 32k context with no quality
  degradation. If feature-dimension tiling underperforms, sketching `d^2 = 4096`
  to a few hundred dimensions is the obvious fallback, and our error budget is
  unusually forgiving: the quadratic term carries only a small part of the
  attention output, so a sketch with substantial relative error on *that term*
  may still land inside tolerance.

  **Caveat: the sketch construction, the guarantee's dependence on sketch size,
  and the sketch dimensions used experimentally could not be retrieved.** The
  OpenReview page returned a verification interstitial and the arXiv PDF did not
  extract. Only the abstract-level claims above are sourced. Do not build on this
  until the construction is read from the paper.

### 3.3 Considered and not applicable

- **ThunderKittens** (HazyResearch) — CUDA tile-primitive framework used for
  Based's own fused linear-attention plus sliding-window kernel. CUDA rather than
  Triton, and it inherits the same `d' = 16` register-residency assumption.
- **Mamba / selective-scan kernels** — the parallel-scan machinery is relevant to
  the chunk-state recurrence, but the SSM parameterisation is not ours.

## 4. Consequences for the plan

1. **Do not port; write.** No published kernel covers `d_h = 64` order-2, so the
   kernel is genuinely new work rather than an adaptation. Keep `fla`'s
   `naive.py` as an oracle at reduced `d`.
2. **Feature-dimension tiling is mandatory**, not an optimization. The state does
   not fit in SRAM at our head dimension. This is the single structural decision.
3. **Two-pass parallel scan**, per the A6 reasoning in
   [`fast-attention-survey.md`](fast-attention-survey.md): the graded path streams
   1-2 samples at a time, so a one-program-per-(batch, head) kernel would occupy
   16-32 of 24 SMs. TFLA is the reference for doing this well.
4. **Keep the `sigma`-fitted constant.** FLA's Based uses the plain Taylor
   constant; our measured Gauss-Hermite constant was 2.3x more accurate at no
   cost, and is what makes order 2 pass where order 1 fails.
5. **Sketching is the fallback, not the plan**, until section 3.2's caveat is
   resolved.

## 5. Sources

- **`fla-org/flash-linear-attention`.**
  <https://github.com/fla-org/flash-linear-attention>. Accessed 30 August 2026,
  `main` branch. MIT licensed Triton library of chunkwise-parallel linear
  attention kernels. Relevant symbol:
  `fla/ops/based/fused_chunk.py`, which asserts
  `q.shape[-1] <= 16, 'only support feature dimension up to 16.'` and uses
  `BT = 16`, `BK = 16`, `BV in [16, 32]` with zero-, first- and second-order
  state accumulators of shapes scalar, `BK x BV` and `BK^2 x BV`. This assert is
  the reason a direct port is impossible at our `d_h = 64`.

- **Arora, S. et al., "Simple linear attention language models balance the
  recall-throughput tradeoff" (Based), ICML 2024.** Blog:
  <https://hazyresearch.stanford.edu/blog/2024-03-03-based>. Accessed
  30 August 2026. States the feature dimension `d' = 16`, a recurrent state of
  `O(N d'^2 d)`, and the rationale: 16x16 and 64x64 matmuls have roughly equal
  latency because tensor cores saturate, so `d' = 16` keeps the KV state in
  thread registers. Their IO-aware kernel saves `O(N d'^2)` HBM-to-SRAM bytes and
  `O(N d'^2 d)` SRAM-to-register bytes. Paper:
  <https://arxiv.org/abs/2402.18668>.

- **Beck, M. et al., "Tiled Flash Linear Attention: More Efficient Linear RNN and
  xLSTM Kernels".** <https://arxiv.org/abs/2503.14376>. Accessed
  30 August 2026. Two levels of sequence parallelism — chunkwise across the
  sequence, plus intra-chunk tiling — enabling arbitrary chunk sizes and high
  arithmetic intensity, explicitly to avoid forcing intermediate states into GPU
  memory. Reports outperforming tuned FlashAttention, linear attention and Mamba
  kernels. Implementation: <https://github.com/NX-AI/mlstm_kernels>. The closest
  published solution to our state-does-not-fit problem.

- **"The Anatomy of a Triton Attention Kernel".**
  <https://arxiv.org/abs/2511.11581>. Accessed 30 August 2026. Block sizes affect
  correctness and performance; register pressure and shared-memory capacity are
  the binding constraints; autotuning is preferred over manual tuning for
  cross-platform portability. Claims cross-platform state-of-the-art LLM
  attention using only Triton.

- **Kacham, P., Mirrokni, V., Zhong, P., "PolySketchFormer: Fast Transformers via
  Sketching Polynomial Kernels", ICML 2024.**
  <https://arxiv.org/abs/2310.01655>, proceedings
  <https://proceedings.mlr.press/v235/kacham24a.html>. Accessed 30 August 2026.
  Sketching techniques from randomised numerical linear algebra give linear-time
  polynomial attention with approximation guarantees; reports 2.5-4x training
  speedup over FlashAttention at 32k context with no observed quality
  degradation. **Only abstract-level claims are sourced** — the sketch
  construction, guarantee, and experimental sketch dimensions could not be
  retrieved (OpenReview interstitial; PDF did not extract) and must be read
  before use.

- **`HazyResearch/ThunderKittens`.**
  <https://github.com/HazyResearch/ThunderKittens>. Accessed 30 August 2026.
  CUDA tile-primitive framework; hosts Based's fused linear-attention plus
  sliding-window kernel. Not Triton, and inherits the `d' = 16` assumption.
