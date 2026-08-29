# Focused Research Review — Attention and Softmax

A decision-ready synthesis of the Person 2 stream. Detail lives in
[`decomposition.md`](decomposition.md),
[`sdpa-and-precision.md`](sdpa-and-precision.md),
[`measurements.md`](measurements.md) and
[`fast-attention-survey.md`](fast-attention-survey.md); this document states what
was concluded and on what evidence.

Status: current as of 29 August 2026. In scope: cases 1-5 and 7-13. Cases 6 and
14 are Person 4's extreme-shape memory work.

## Bottom line

**One intervention captures essentially all of the available gain in this stream:
replace the eager attention core with `scaled_dot_product_attention` in float32.**
All twelve in-scope cases pass the official criterion and all twelve get faster,
geometric mean ≈**1.94x** whole-model.

Every other idea investigated was either (a) already implemented inside SDPA,
(b) numerically impossible under the pass criterion, or (c) measured slower.
**Person 2's optimization surface is close to exhausted by a single library
call**, and the remaining time in this region is dominated by layout copies that
sit on the Person 3 boundary.

That is a useful conclusion, not a disappointing one: it says where the team's
remaining effort should *not* go.

## 1. What the constraints permit

Three constraints, established early, eliminate most of the literature:

1. **Weights are fixed.** `copy_model_weights` loads the reference `state_dict`
   with `strict=True`. Anything changing parameter shapes cannot load — this
   alone removes Linformer's K/V projections and MQA/GQA's shared heads.
2. **Per-element tolerance, zero failures allowed.** `abs <= 0.002 OR rel <= 2%`
   with `failed_elements == 0`. Every *approximate* attention method fails by
   construction, not by a tuning margin.
3. **Wrong regime.** Our N is 32-1024 with large batches. The efficient-attention
   literature targets N in the thousands; Reformer's gains reportedly only appear
   past N=2048.

What survives is the exact, IO-aware, kernel-level branch — which is precisely
what SDPA already implements.

## 2. Where the time actually is

**The region is memory-bound by roughly 25x.** On case 13 the eager path moves
the `N x N` score tensor about twelve times per layer — ~48 GB across 4 layers,
~176 ms at 272 GB/s — to perform ~137 GFLOP of matmul, ~6.9 ms at 20 TFLOP/s.
Operator attribution agrees: `aten::bmm`, the actual attention math, is the
*smallest* attention cost, while softmax, masking and dtype casts are each 2-3x
it and the `.contiguous()` copies are 7.1x it.

This single fact determines everything else. Faster matmuls are worthless here;
removing the score tensor from memory is the whole problem. It is independently
corroborated by Ivanov et al., *Data Movement Is All You Need* (MLSys 2021),
which reached the same conclusion for transformer training.

## 3. What works — the complete measured table

float32, TF32 on (harness default), `padding_ratio=0`, correctness over 3 seeds,
paired timing with 30 repeats and a reported noise floor.

| Case | B | H | N | `d_h` | correct | base ms | SDPA ms | speedup | floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 64 | 4 | 1024 | 32 | PASS | 385.747 | 60.885 | **6.336x** | ±2.6% |
| 11 | 64 | 16 | 128 | 8 | PASS | 24.881 | 5.013 | **4.964x** | ±8.4% |
| 5 | 128 | 4 | 128 | 32 | PASS | 15.555 | 5.331 | **2.918x** | ±4.5% |
| 2 | 1 | 4 | 128 | 32 | PASS | 7.805 | 5.207 | 1.499x | ±14.7% |
| 9 | 64 | 1 | 128 | 128 | PASS | 8.176 | 5.605 | 1.459x | ±18.9% |
| 3 | 4 | 4 | 128 | 32 | PASS | 7.875 | 5.408 | 1.456x | ±7.8% |
| 10 | 64 | 2 | 128 | 64 | PASS | 6.996 | 4.853 | 1.441x | ±10.4% |
| 4 | 16 | 4 | 128 | 32 | PASS | 7.727 | 5.371 | 1.439x | ±8.6% |
| 12 | 64 | 4 | 32 | 32 | PASS | 7.026 | 4.943 | 1.421x | ±11.8% |
| 7 | 64 | 4 | 128 | 8 | PASS | 8.962 | 6.718 | 1.334x | ±13.6% |
| 1 | 64 | 4 | 128 | 32 | PASS | 31.648 | 26.369 | 1.200x | ±0.1% |
| 8 | 64 | 4 | 128 | 256 | PASS | 61.549 | 58.788 | 1.047x | ±0.3% |

The gain tracks attention's share of the work, as expected: cases 13 and 11 have
the largest score tensors relative to everything else and gain most; case 8 is
projection-bound (attention only ~16% of device time) and gains least.

Cases 2 and 9 carry wide noise floors (±14.7%, ±18.9%) and should be re-measured
with more repeats before either number is quoted on its own.

**Also available, and bitwise-exact:** caching the causal mask and skipping
all-true padding masks, worth ~1.25x geometric mean on its own with `max_abs = 0`
— but only when `padding_ratio=0`, and only if the all-true test is hoisted to one
host sync per forward rather than one per layer.

**And stacking on top of SDPA:** feeding SDPA strided views instead of contiguous
copies adds a further 6-12% (case 13 6.336x -> 6.908x, case 8 1.047x -> 1.119x).
See section 5.

## 4. What does not work — negative results and their evidence

These are the substance of the review. Each was measured, not assumed.

| Idea | Verdict | Evidence |
| --- | --- | --- |
| **float16 anything that reassociates** | Fails correctness | SDPA passes 0/8 seeds on case 13, 1-2/8 elsewhere. Even folding `scale` into `Q` breaks cases 7 and 13. The reference rounds probabilities to float16 before `PV`, so a fused kernel is *more accurate* and therefore deviates |
| **bfloat16** | Fails badly | 3.4-5.3% of elements fail |
| **float16 attention internals under float32** | Passes but slower | 1.18x vs 1.68x on case 7; cast overhead exceeds the tensor-core benefit |
| **FlexAttention / custom causal block mask** | Redundant | SDPA already skips upper-triangle blocks: causal/non-causal ratio 0.522 at N=2048, 0.537 at N=1024, against a theoretical floor of 0.5 |
| **Hand-written Triton kernel** | Not justified | SDPA already delivers levers L1, L3, L4, L5. A custom kernel would have to beat a mature CUTLASS FMHA implementation at its own game |
| **Split-K / Flash-Decoding parallelism** | Not applicable | Occupancy analysis against 24 SMs: only case 2 is starved (`B*H`=4, 0.2x SMs), and case 2 is launch-bound, so extra splits add launches to a case limited by launches |
| **All approximate attention** (Performer, Linformer, Reformer, Longformer, BigBird, Nyströmformer, cosFormer, linear attention, SOFT, ReLU/sigmoid, NSA) | Excluded by construction | Changes the mathematical result; cannot satisfy a per-element criterion |
| **MQA / GQA** | Cannot load | Changes parameter count, breaking `strict=True`; also targets KV-cache bandwidth, which this benchmark does not exercise |

### Two corrections worth recording

**Case 8 does not regress.** An attention-only microbenchmark showed SDPA at
0.64x for `d_h=256` and an earlier draft recommended excluding case 8. The
whole-model measurement shows **1.047x ±0.3%** — a small gain. The microbenchmark
built the causal mask once outside the timed region, while the real reference
rebuilds it per layer per forward; it therefore gave the baseline a free mask and
understated SDPA everywhere. **Whole-model numbers should drive routing.**

**Case 9 is not occupancy-starved.** An earlier draft assumed `H=1` implied low
occupancy. With `B=64`, `B*H` = 64 against 24 SMs — comfortable. Only case 2 is
genuinely starved.

## 5. Residual opportunities, ranked

1. **Drop the `.contiguous()` copies — measured, and it works.** `aten::copy_`
   plus `Memcpy DtoD` are **7.1x the attention matmul** on case 13. SDPA accepts
   strided inputs, so feeding it `.transpose(1,2)` views instead of contiguous
   copies (and `.reshape()` on the way out) is a further **6-12% on top of SDPA**,
   correct on all cases tested:

   | Case | SDPA + `.contiguous()` | SDPA + strided |
   | --- | --- | --- |
   | 13 | 6.379x | **6.908x** |
   | 11 | 5.039x | **5.353x** |
   | 1 | 1.407x | **1.577x** |
   | 8 | 1.044x | **1.119x** |

   The input side is inside `BaselineSelfAttention`, so **this is Person 2's to
   take**, not a Person 3 negotiation. Only the output reshape touches stage 8.
   Individual floors are wide on cases 1 and 13; the consistent direction across
   all four is what makes it credible. Worth re-measuring on the full sweep before
   claiming a specific number.
2. **Re-measure cases 2 and 9** with more repeats to tighten their floors.
3. **Confirm `padding_ratio` and dtype with the organizer** — both change the
   conclusions materially (see section 6).
4. Nothing else. The attention core itself is done.

## 6. Open questions that could change the conclusions

- **Which dtype does the official evaluation use?** If float16 is scored, fused
  attention is close to unpassable and the whole plan changes. Worth raising with
  the organizer, since the cause is the reference's own rounding rather than any
  submission's error.
- **Which `padding_ratio`?** The bitwise-exact 1.25x collapses to ~0.95x once
  padding is present, because nearly all of it comes from skipping an all-true
  mask.
- **Which `input_scale`?** Untested. It moves output magnitudes and therefore
  which branch of the `abs OR rel` criterion applies.

## 7. Handoffs

- **Person 1:** cases 2, 3 and 12 are launch-bound — case 12's GPU work is
  ~0.6x its kernel-launch floor — and cannot be fixed by attention math. They
  need CUDA graphs or compilation. Also: the float16 fragility applies to
  `torch.compile`, which reassociates freely.
- **Person 3:** case 8 is projection-bound (attention ~16% of device time), and
  the stage 2/8 layout copies are the largest remaining attention-adjacent cost.
- **Person 4:** cases 6 and 14 untouched by this stream.
- **Everyone:** the measurement harness reports a noise floor and labels results
  `WITHIN NOISE`. Several conclusions here reversed once measured properly;
  treat any speedup inside its floor as no change.

## Sources

All literature citations, with URLs and access dates, are in
[`fast-attention-survey.md`](fast-attention-survey.md). Reference-implementation
citations (commit `7eb8fb1`, symbols and line ranges) are in
[`decomposition.md`](decomposition.md) and
[`sdpa-and-precision.md`](sdpa-and-precision.md). Environment and method are in
[`measurements.md`](measurements.md).
