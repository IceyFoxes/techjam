# Attention and Softmax (Person 2)

Research for the `QK^T` / causal softmax / `PV` region of the Transformer layer,
owned by Person 2 per
[`four-way-team-split.md`](../team-coordination/four-way-team-split.md).

Status: current as of 29 August 2026.

## Documents

- [`decomposition.md`](decomposition.md) — subdivides attention into the smallest
  independently optimizable stages, gives the roofline argument for why this
  region is memory-bound, and enumerates the algorithmic levers (L1-L8) with the
  stages each one removes.
- [`sdpa-and-precision.md`](sdpa-and-precision.md) — why the target is float32 +
  `scaled_dot_product_attention`, why float16 was rejected on correctness, and how
  much of the tolerance budget TF32 has already spent.
- [`measurements.md`](measurements.md) — environment, operator-level attribution,
  and every measured speedup with its noise floor.

## Conclusions so far

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

3. **float32 SDPA is measured at 6.41x whole-model on case 13** (±3.5%) and
   1.4-1.7x on cases 1, 7, 12, passing 10/10 seeds. Case 8 (`d_h=256`) must stay
   on the eager path; SDPA regresses to 0.64x there.

4. **Two bitwise-exact levers are worth ~1.25x on their own** (cached causal mask,
   skipping all-true padding masks) but only when `padding_ratio=0`.

5. **Reduced-precision attention internals were tested and rejected** — they pass
   correctness but are slower than plain float32 SDPA.

## Open questions

- Which dtype and `padding_ratio` does the official evaluation use? Both change
  the conclusions materially: float16 would make fused attention close to
  unpassable, and padding removes most of the bitwise-exact gain.
- Does the upper-triangle work still show up after SDPA lands? That determines
  whether a Triton kernel exploiting causal block-skipping (lever L3) is worth
  building.

## Scope boundaries

- Cases 6 (`B=10000`) and 14 (`N=100000`) are **Person 4's** extreme-shape memory
  scope and are excluded from these documents.
- Stages 2 and 8 (head reshape and `.contiguous()`) are shared with Person 3 and
  should be negotiated rather than changed unilaterally.
- Case 8 is projection-bound (`aten::addmm` 31.5%, `bmm` 3.1%) and belongs to
  Person 3 in practice.
