# FP16 Interface Follow-up: Reference Dtype vs Internal Compute

Status: current as of 31 August 2026. This note narrows, without deleting, the
earlier float16 conclusion in [`sdpa-and-precision.md`](sdpa-and-precision.md).

## The earlier conclusion mixed two different contracts

The original rejection compared a **float16 reference model** with an optimized
float16 attention implementation. In that contract the reference explicitly
rounds the FP32 softmax probabilities back to FP16 before `PV`. Fused SDPA does
not reproduce that intermediate rounding, so even a more accurate fused result
can fail the executable comparison. That evidence remains valid.

Case 14 now uses a different contract: the oracle is the **float32 reference**,
while the candidate accepts FP32 input and parameters but uses FP16 internally
before returning FP32 output. The official `absolute OR relative` comparison is
then applied to the two FP32 outputs. Passing this contract does not contradict
the earlier float16-reference result.

The previous shorthand that “the other thirteen cases cannot quantize” was
therefore too broad. It generalized a result about float16-reference arithmetic
to every possible FP32-interface/mixed-precision candidate.

## Exploratory cross-case check

To test the broader claim directly, cases 1-13 were run once with the immutable
FP32 baseline as reference and the current optimized candidate converted to
FP16 internally, with FP32 input/output casts at the candidate boundary. This
was a one-seed diagnostic, not a preserved acceptance benchmark and not a
performance measurement.

| Case | One-seed result | Failed elements | Maximum absolute error |
| ---: | :---: | ---: | ---: |
| 1 | FAIL | 4 | 0.00607729 |
| 2 | PASS | 0 | 0.00448513 |
| 3 | PASS | 0 | 0.00551963 |
| 4 | PASS | 0 | 0.00607729 |
| 5 | FAIL | 10 | 0.00666022 |
| 6 | FAIL | 306 | 0.00910378 |
| 7 | FAIL | 5 | 0.00566053 |
| 8 | FAIL | 9 | 0.00977659 |
| 9 | PASS | 0 | 0.00618649 |
| 10 | FAIL | 3 | 0.01015282 |
| 11 | PASS | 0 | 0.00620031 |
| 12 | PASS | 0 | 0.00607729 |
| 13 | FAIL | 12 | 0.00839067 |

Six cases pass this one seed, while seven have rare tail failures. Therefore:

1. Whole-model FP16 behind an FP32 interface is not universally invalid.
2. It is also not universally safe; the official zero-failure rule makes even
   three or four tail elements decisive.
3. A one-seed pass is only a promotion lead. Each case still needs a multi-seed
   gate covering its padding and input-scale risks, plus a paired performance
   comparison.

## Why Case 14 passes robustly

Measured facts are that Case 14 passes five trials and 160 full-length samples
with zero failures, uses only two Transformer layers, and has per-layer
polynomial sigma around `0.334`, below the calibrated `0.40` guard. Its maximum
absolute error is about `0.0103`, but affected values are large enough to pass
the 2% relative arm.

The likely numerical explanation is the combination of fewer layers and the
averaging behavior of attention over `N=100000`: FP16 perturbations contribute
less relative error to most outputs than in the shorter cases. This is an
inference from the measured error distribution, not a proof that sequence
length alone guarantees safety. In the failing short cases, errors cluster in
the near-cancellation tail where reference outputs are small, leaving only the
`0.002` absolute allowance.

The performance economics differ too. At Case 14, attention dominates and its
enormous workload amortizes the FP32/FP16 boundary conversions. Earlier tests
of FP16 **attention-only** internals in otherwise FP32 short-case models passed
representative accuracy sweeps but were slower than plain FP32 SDPA because
cast overhead dominated. Whole-model FP16 may change that tradeoff, but it has
not yet earned a general dispatcher route.

## Decision

- Keep the Case-14 FP32-facing, FP16-compute route: it passes its dedicated
  five-trial oracle gate and makes the shape tractable.
- Preserve the original float16-reference rejection as a distinct result.
- Replace “float32 is the only viable dtype” with the narrower statement:
  float32 is the accepted default for cases 1-13; mixed precision is a
  shape-specific optimization requiring independent correctness and speed
  evidence.
