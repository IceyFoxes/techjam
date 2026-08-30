# Final Controls and Branch Base — RTX 5080

This directory preserves the immutable-reference A/A controls and the dispatcher
at branch base `3e6a3ea32b838dfac17281018f6379ec69094590` before the final
Case-14 FP32-facing implementation commits are applied.

## Method

- Device/dtype: NVIDIA GeForce RTX 5080, CUDA, float32.
- Accuracy: five seeds beginning at 1234, default input scale, zero padding.
- Timing: paired interleaved CUDA events, 10 seconds settle per arm, 100 paired
  samples, automatic block targeting 50 ms.
- Numerical flags: high float32 matmul precision, TF32 enabled.
- Case 14: FP32 dispatcher is **unsupported** at this base commit and is
  deliberately not allocated.

## Branch-Base Dispatcher

| Case | Baseline ms | Base candidate ms | Speedup | Accuracy |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1.9240 | 0.6985 | 2.754x | PASS 5/5 |
| 2 | 1.1211 | 0.1587 | 7.064x | PASS 5/5 |
| 3 | 0.9733 | 0.1673 | 5.817x | PASS 5/5 |
| 4 | 1.1766 | 0.2622 | 4.488x | PASS 5/5 |
| 5 | 3.0165 | 1.2671 | 2.381x | PASS 5/5 |
| 6 | 503.4646 | 210.2879 | 2.394x | PASS 5/5 |
| 7 | 1.5302 | 0.4093 | 3.738x | PASS 5/5 |
| 8 | 17.3501 | 15.6273 | 1.110x | PASS 5/5 |
| 9 | 1.4974 | 0.6836 | 2.190x | PASS 5/5 |
| 10 | 1.7851 | 0.6414 | 2.783x | PASS 5/5 |
| 11 | 7.6543 | 1.3725 | 5.577x | PASS 5/5 |
| 12 | 1.0978 | 0.2361 | 4.650x | PASS 5/5 |
| 13 | 116.7104 | 13.5784 | 8.595x | PASS 5/5 |
| 14 | unsupported | unsupported FP32 | N/A | UNSUPPORTED |

All 65 executable accuracy trials have zero failed elements.

## Immutable A/A Controls

| Case | A/A ratio | Noise floor | Interpretation |
| ---: | ---: | ---: | --- |
| 1 | 1.106x | 7.34% | nominally significant drift |
| 2 | 0.942x | 9.42% | within noise |
| 3 | 0.935x | 7.89% | within noise |
| 4 | 0.927x | 7.66% | within noise |
| 5 | 0.995x | 4.28% | within noise |
| 6 | 0.977x | 1.87% | nominally significant drift |
| 7 | 1.014x | 9.93% | within noise |
| 8 | 1.003x | 0.84% | within noise |
| 9 | 1.001x | 6.21% | within noise |
| 10 | 0.997x | 7.71% | within noise |
| 11 | 1.001x | 0.56% | within noise |
| 12 | 1.011x | 6.89% | within noise |
| 13 | 1.002x | 0.37% | within noise |

The controls compare two separately instantiated but bitwise-equivalent
immutable models. They quantify run-order, cache, boost, and launch variation;
they are not performance gains.

## Evidence Files

Immutable A/A controls:
[`1`](base-reference-case1.json),
[`2`](base-reference-case2.json),
[`3`](base-reference-case3.json),
[`4`](base-reference-case4.json),
[`5`](base-reference-case5.json),
[`6`](base-reference-case6.json),
[`7`](base-reference-case7.json),
[`8`](base-reference-case8.json),
[`9`](base-reference-case9.json),
[`10`](base-reference-case10.json),
[`11`](base-reference-case11.json),
[`12`](base-reference-case12.json), and
[`13`](base-reference-case13.json).

Branch-base dispatcher:
[`1`](base-dispatcher-case1.json),
[`2`](base-dispatcher-case2.json),
[`3`](base-dispatcher-case3.json),
[`4`](base-dispatcher-case4.json),
[`5`](base-dispatcher-case5.json),
[`6`](base-dispatcher-case6.json),
[`7`](base-dispatcher-case7.json),
[`8`](base-dispatcher-case8.json),
[`9`](base-dispatcher-case9.json),
[`10`](base-dispatcher-case10.json),
[`11`](base-dispatcher-case11.json),
[`12`](base-dispatcher-case12.json), and
[`13`](base-dispatcher-case13.json).

Each JSON preserves its exact command, timestamp, Git state, environment, raw
timing samples, accuracy trials, GPU state, and memory snapshots.

The recorded dirty flag comes from pre-existing untracked benchmark artifacts
and the new output directory accumulating files during the matrix. Tracked
source remained fixed at `3e6a3ea` for every run.
