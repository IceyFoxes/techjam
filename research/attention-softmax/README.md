# Attention and Softmax (Person 2)

Research for the `QK^T` / causal softmax / `PV` region of the Transformer layer,
owned by Person 2 per
[`four-way-team-split.md`](../team-coordination/four-way-team-split.md).

Status: current as of 29 August 2026.

## Documents

- [`safe-optimization-spec.md`](safe-optimization-spec.md) — the implementation
  design for the optimizations that gain on every in-scope case at no correctness
  cost: route table, module layout, validation matrix, and acceptance criteria.
  Contains the section 3 finding that corrects the padding conclusions below.
- [`review.md`](review.md) — **start here.** Decision-ready synthesis: what works,
  what does not and why, the complete per-case table, residual opportunities, and
  handoffs to the other streams.
- [`decomposition.md`](decomposition.md) — subdivides attention into the smallest
  independently optimizable stages, gives the roofline argument for why this
  region is memory-bound, and enumerates the algorithmic levers (L1-L8) with the
  stages each one removes.
- [`sdpa-and-precision.md`](sdpa-and-precision.md) — why the target is float32 +
  `scaled_dot_product_attention`, why float16 was rejected on correctness, and how
  much of the tolerance budget TF32 has already spent.
- [`measurements.md`](measurements.md) — environment, operator-level attribution,
  and every measured speedup with its noise floor.
- [`fast-attention-survey.md`](fast-attention-survey.md) — idea catalogue of ~20
  fast-attention methods from the literature, each filtered against this task's
  constraints, plus the measurement showing SDPA already performs causal
  block-skipping. **Partially superseded for case 14 (30 August 2026)**: its
  Tier C blanket exclusion of approximate methods is re-tested in
  `long-sequence-attention.md` and does not hold at N=100000.
- [`triton-kernel-spec.md`](triton-kernel-spec.md) — implementation spec for the
  fused kernel: phasing, the two Triton kernels, precision rules, the runtime
  `sigma` guard, and the acceptance thresholds.
- [`triton-kernel-prior-art.md`](triton-kernel-prior-art.md) — what already
  exists for the proposed fused kernel. Key finding: `flash-linear-attention`
  implements this exact algorithm but asserts head dimension <= 16, and Based
  chose `d'=16` deliberately for register residency, so no published kernel
  covers our `d_h=64`.
- [`long-sequence-attention.md`](long-sequence-attention.md) — case 14
  (`N=100000`) only. Measures the score distribution, shows the error budget at
  that scale admits approximation, and reports a validated order-2 polynomial
  feature-map linear attention at 1.19x, plus four negative results.

## Scope of these conclusions

Conclusions 1-8 below were measured at `N <= 1024` and govern cases 1-5 and 7-13.
**They do not transfer to case 14** (`N=100000`), where attention is
compute-bound rather than memory-bound and the tolerance budget is far larger.
See [`long-sequence-attention.md`](long-sequence-attention.md), which restates
conclusions 1 and 8 for that case.

## Conclusions so far (N <= 1024)

1. **Attention here is memory-bound by roughly 25x**, not compute-bound. On
   case 13 the eager path moves the `N x N` score tensor about twelve times per
   layer (~48 GB across 4 layers, ~176 ms at 272 GB/s) to perform ~137 GFLOP of
   actual matmul (~6.9 ms). Removing the score tensor from memory is the whole
   problem; faster matmuls are nearly worthless.

2. **float32 is the only viable dtype.** In float16 the criterion fails for *any*
   arithmetic reassociation — even folding the `scale` into `Q` breaks cases 7
   and 13, and SDPA fails on 0/8 to 5/8 seeds depending on case. The cause is
   that the reference rounds softmax probabilities back to float16 before the
   `PV` matmul, so a fused kernel is *more* accurate and therefore deviates.

3. **float32 SDPA gains on all twelve in-scope cases, geomean ≈1.94x**, and all
   twelve pass the official criterion. Best: case 13 at 6.34x (±2.6%), case 11 at
   4.96x (±8.4%), case 5 at 2.92x (±4.5%). Weakest: case 8 at 1.047x (±0.3%),
   still a gain. An earlier draft excluded case 8 based on an attention-only
   microbenchmark; that measurement was flawed and is corrected in
   [`measurements.md`](measurements.md).

4. **Two bitwise-exact levers are worth ~1.25x on their own** (cached causal mask,
   skipping all-true padding masks) but only when `padding_ratio=0`.
   **Corrected 29 August 2026 — the `padding_ratio=0` restriction is wrong.**
   Under causal attention the padding key mask is dead code for *any*
   right-padded mask, not only an all-true one, because causal masking already
   sets `-inf` everywhere the padding mask would. Verified bitwise identical
   (`0.000e+00`) over 4 cases x 4 padding ratios x 4 seeds. See
   [`safe-optimization-spec.md`](safe-optimization-spec.md) section 3.

5. **Reduced-precision attention internals were tested and rejected** — they pass
   correctness but are slower than plain float32 SDPA.

6. **SDPA already performs causal block-skipping.** Measured `is_causal=True`
   against `is_causal=False`: ratio 0.522 at N=2048 and 0.537 at N=1024, against a
   theoretical maximum of 0.5. Lever L3 is therefore already delivered, which
   removes the main argument for FlexAttention or a hand-written Triton kernel.
   The shortfall at N=128 (0.771) is block granularity and would affect any
   block-masked implementation equally.

7. **Dropping the `.contiguous()` copies adds a further 6-12% on top of SDPA**
   (case 13 6.379x -> 6.908x, case 8 1.044x -> 1.119x). SDPA accepts strided
   inputs, so `_split_heads`'s copies are avoidable. The Q/K/V side is inside
   Person 2's module and needs no coordination.

8. **The surveyed efficient-attention literature is almost entirely inapplicable
   at these shapes.** **Scoped 30 August 2026:** true for `N <= 1024`; at case
   14's `N=100000` the kernel/feature-map branch becomes viable and is measured
   in [`long-sequence-attention.md`](long-sequence-attention.md). Selection-based
   sparse methods remain excluded there too, now by measurement rather than by
   argument.
   Every approximate method (Performer, Linformer, Reformer, Longformer, BigBird,
   Nyströmformer, cosFormer, linear attention, NSA) changes the mathematical
   result and cannot satisfy a per-element tolerance; Linformer and MQA/GQA
   additionally break `strict=True` weight loading. Most also target N in the
   thousands, while our largest in-scope N is 1024.

## Open questions

- Which dtype and `padding_ratio` does the official evaluation use? Both change
  the conclusions materially: float16 would make fused attention close to
  unpassable, and padding removes most of the bitwise-exact gain.
- Does the upper-triangle work still show up after SDPA lands? That determines
  whether a Triton kernel exploiting causal block-skipping (lever L3) is worth
  building.

## Scope boundaries

- Case 6 (`B=10000`) is **Person 4's** extreme-shape memory scope and is excluded
  from these documents. Case 14's *memory strategy* is likewise Person 4's
  (see PR #13), but its *attention algorithm* is analysed in
  [`long-sequence-attention.md`](long-sequence-attention.md), since at N=100000
  the attention core is ~95% of all FLOPs.
- **For Person 4 (30 August 2026):** the section 3 result applies to cases 6 and
  14 as well. Under causal attention with right padding, no padding mask need
  ever be built, which removes a `B x N` term from any chunked design and one
  full read+write of the score block per chunk.
- Stages 2 and 8 (head reshape and `.contiguous()`) are shared with Person 3 and
  should be negotiated rather than changed unilaterally.
- Case 8 is projection-bound (`aten::addmm` 31.5%, `bmm` 3.1%) and belongs to
  Person 3 in practice.
