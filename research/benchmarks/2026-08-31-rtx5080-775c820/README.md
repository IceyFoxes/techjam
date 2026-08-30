# Final Fourteen-Case Submission Matrix — RTX 5080

## Result

Latest implementation:
`775c82004fd31d5a2203619f671ee214f444411c` on
`impl/case14-fp32-streamed-oracle`.

- Cases 1-13: **PASS 65/65**, zero failed elements, all paired improvements
  significant, **3.611x geometric-mean speedup**.
- Case 14: **PASS 5/5**, zero failures across 16.384 billion elements,
  `max_abs=0.010294`, 8.972x diagnostic oracle/candidate ratio, and
  3,643.988 MiB peak allocation.
- Latest versus branch base: no accuracy regression and no latency regression
  above 5%; range `-1.82%` to `+2.14%`.
- CUDA-aware unit suite: 210 tests pass.

## Directly Comparable Cases

| Case | Baseline ms | Candidate ms | Speedup | Noise | Max abs | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.9350 | 0.6975 | 2.774x | 25.30% | 0.001054 | PASS 5/5 |
| 2 | 1.1953 | 0.1609 | 7.428x | 62.49% | 0.000819 | PASS 5/5 |
| 3 | 0.9788 | 0.1643 | 5.959x | 34.53% | 0.000908 | PASS 5/5 |
| 4 | 1.1567 | 0.2678 | 4.319x | 17.64% | 0.000908 | PASS 5/5 |
| 5 | 3.0159 | 1.2513 | 2.410x | 8.85% | 0.001181 | PASS 5/5 |
| 6 | 504.1028 | 210.5723 | 2.394x | 0.52% | 0.001347 | PASS 5/5 |
| 7 | 1.5418 | 0.4060 | 3.797x | 33.12% | 0.001316 | PASS 5/5 |
| 8 | 17.3913 | 15.5889 | 1.116x | 0.82% | 0.001218 | PASS 5/5 |
| 9 | 1.5037 | 0.6881 | 2.185x | 11.03% | 0.001078 | PASS 5/5 |
| 10 | 1.8052 | 0.6464 | 2.793x | 20.44% | 0.001089 | PASS 5/5 |
| 11 | 7.6778 | 1.3724 | 5.595x | 7.27% | 0.001063 | PASS 5/5 |
| 12 | 1.0901 | 0.2335 | 4.668x | 41.84% | 0.001128 | PASS 5/5 |
| 13 | 116.7147 | 13.6078 | 8.577x | 9.12% | 0.001123 | PASS 5/5 |

## Case 14

| Reference | Revision | Accuracy | Timing | Peak allocation |
| --- | --- | --- | ---: | ---: |
| Immutable dense | immutable root | full shape infeasible | unavailable | unavailable |
| Reduced dense oracle gate | immutable vs oracle at `N=4096` | PASS, 0/4,194,304; `max_abs=0.00060248` | validation only | bounded |
| Branch base | `3e6a3ea` | FP32 dispatcher unsupported | unavailable | unavailable |
| Latest streamed comparison | `775c820` | PASS 5/5, 0/16,384,000,000; `max_abs=0.010294` | oracle 601.975 s; candidate 67.095 s; 8.972x diagnostic | 3,643.988 MiB |

The Case-14 ratio compares two sample-streamed, linear-memory paths. It is not
an immutable dense-baseline speedup and is not included in the 3.611x geometric
mean. Maximum absolute error can exceed 0.002 because the same elements pass the
2% relative arm of the official rule.

## Memory Highlights

| Case | Baseline peak MiB | Candidate peak MiB | Reduction |
| ---: | ---: | ---: | ---: |
| 5 | 171.105 | 83.059 | 51.5% |
| 6 | 10,672.719 | 2,312.512 | 78.3% |
| 8 | 544.369 | 416.338 | 23.5% |
| 11 | 199.082 | 75.051 | 62.3% |
| 13 | 2,372.229 | 227.104 | 90.4% |
| 14 | dense unavailable | 3,643.988 combined proxy run | N/A |

Small launch-bound cases may have a higher absolute process peak because the
measurement includes resident compiler caches. Complete memory rows are in the
[final report](../../final-submission/README.md#memory-results).

## Exact Method

Cases 1-13 use five seeds, paired interleaved CUDA-event timing, 10-second
settling per arm, 100 paired samples, auto-selected blocks targeting 50 ms,
input scale 1.0, no padding, high matmul precision, and TF32 enabled. The exact
command appears inside each JSON.

Case 14 uses the same five input seeds and exact correctness thresholds. Its
special runner ignores steady-state repeat flags because it performs 160
full-length oracle/candidate sample pairs directly.

## Evidence Files

[`case1`](latest-dispatcher-case1.json),
[`case2`](latest-dispatcher-case2.json),
[`case3`](latest-dispatcher-case3.json),
[`case4`](latest-dispatcher-case4.json),
[`case5`](latest-dispatcher-case5.json),
[`case6`](latest-dispatcher-case6.json),
[`case7`](latest-dispatcher-case7.json),
[`case8`](latest-dispatcher-case8.json),
[`case9`](latest-dispatcher-case9.json),
[`case10`](latest-dispatcher-case10.json),
[`case11`](latest-dispatcher-case11.json),
[`case12`](latest-dispatcher-case12.json),
[`case13`](latest-dispatcher-case13.json), and
[`case14`](latest-dispatcher-case14.json).

The [branch-base controls](../2026-08-31-rtx5080-3e6a3ea/README.md) provide the
other two reference points. The [submission report](../../final-submission/README.md)
contains specifications, implementation decisions, the regression table,
reproduction commands, AI usage, contributions, and limitations.

The JSON dirty flag reflects pre-existing untracked benchmark artifacts plus
the final output directories. No tracked source changed during this matrix;
every latest file records commit `775c820`.
