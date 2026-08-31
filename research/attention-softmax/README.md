# Attention and Softmax (Person 2)

Research for the `QK^T` / causal softmax / `PV` region of the Transformer layer,
owned by Person 2 per
[`four-way-team-split.md`](../team-coordination/four-way-team-split.md).

Status: current as of 31 August 2026.

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
  much of the tolerance budget TF32 has already spent. **Its blanket dtype
  conclusion is narrowed as of 31 August 2026**: the float16-reference rejection
  remains valid, but it does not prove that FP16 internal compute behind an FP32
  interface always fails. See
  [`fp16-interface-followup.md`](fp16-interface-followup.md).
- [`fp16-interface-followup.md`](fp16-interface-followup.md) — separates the
  float16-reference contract from FP32-interface mixed precision, records the
  Case-14 five-trial pass and a one-seed cases 1-13 diagnostic, and defines the
  new per-shape promotion rule.
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
  `sigma` guard, and the acceptance thresholds. **Section 8 superseded
  30 August 2026** by [`integrated-kernel-spec.md`](integrated-kernel-spec.md):
  its Phase 2 sketch proposed a two-level sequence-parallel scan to fix an
  occupancy problem ("16 programs against 24 SMs") that the shipped kernels do
  not have — they launch 256 and 2048 programs at B=2. Sections 1-7 and 8.2
  remain current and are inherited by the Phase 2 spec unchanged.
- [`integrated-kernel-spec.md`](integrated-kernel-spec.md) — Phase 2 spec: fold
  the remaining per-chunk PyTorch work into the kernels. Staged as six
  individually A/B'd redundancy fixes (led by a causal-tiled diagonal block),
  then two complete per-chunk kernels, with a persistent-slab scan specified but
  gated on measurement. Opens with a blocking noise-floor task, because four
  identical profiles on this hardware spread 2.17x.
- [`kernel-integration-notes.md`](kernel-integration-notes.md) — cross-stream
  integration contract for the polynomial Case 14 route: rollback control,
  memory and synchronization constraints, correctness boundary, and measured
  RTX 4060/5080 evidence.
- [`triton-kernel-prior-art.md`](triton-kernel-prior-art.md) — what already
  exists for the proposed fused kernel. Key finding: `flash-linear-attention`
  implements this exact algorithm but asserts head dimension <= 16, and Based
  chose `d'=16` deliberately for register residency, so no published kernel
  covers our `d_h=64`.
- [`long-sequence-attention.md`](long-sequence-attention.md) — case 14
  (`N=100000`) only. Section 5.5 measures what happens if the method is applied
  outside that scope: it is 2-186x *slower* on cases 1-13 and OOMs on case 8,
  because it trades `O(N^2 d)` for `O(N d^3)` and those shapes sit below the
  crossover. Measures the score distribution, shows the error budget at
  that scale admits approximation, and reports a validated order-2 polynomial
  feature-map linear attention at 1.19x, plus four negative results. **Integration
  status disputed 30 August 2026:** the later fused kernel is 1.561x faster than
  exact Flash in an RTX 5080 attention-core microbenchmark, but the submitted PR
  does not select it and a locally wired full Case 14 run is 16.886x slower with
  2.58 GiB more peak allocation at the 16 GiB VRAM cliff. See the
  [`b9506f3` A/B record](../benchmarks/2026-08-30-rtx5080-b9506f3/README.md).

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

2. **Narrowed 31 August 2026: float32 remains the accepted default for these
   cases, but is not proven to be the only viable internal dtype.** The original
   float16-reference test still rejects arithmetic reassociation — even folding
   the `scale` into `Q` breaks cases 7 and 13, and SDPA passes only 0/8 to 5/8
   seeds depending on case — because the reference rounds probabilities back to
   float16 before `PV`. A separate FP32-reference/FP16-internal diagnostic passes
   one seed on cases 2, 3, 4, 9, 11, and 12, but fails the other seven. See
   [`fp16-interface-followup.md`](fp16-interface-followup.md); no new short-case
   route is accepted without multi-seed correctness and performance evidence.

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
