# Case-3 Packed-QKV Integration — RTX 5080 — 30 August 2026

## Outcome

PR #15 passes every merge gate at merge commit
`ce3f7f2c426211f552baf47c84cac21a6be74708`.

- Case 3 improves from `0.156653` ms on branch-base `fecf994` to `0.128575`
  ms, an incremental **1.218x** gain and a **17.92% latency reduction**.
- Case 3 passes the complete 60-trial stress matrix with zero failed elements.
- Case 2 improves from `0.122745` ms to `0.118854` ms, so there is no
  greater-than-5% regression; latency is 3.17% lower in this comparison.
- The CUDA-aware unit suite passes all 128 tests. `compileall` and
  `git diff --check` also pass.

The fresh results agree with the earlier same-process Case-3 direct A/B at
`12a37c6` (1.192x with a +/-0.84% floor). The new 1.218x result clears PR #15's
required 1.15x incremental gate.

## Three Reference Points

| Case | Reference | Commit | Accuracy | Max abs error | Median latency | Paired speedup |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 3 | Immutable baseline | `fecf994` | PASS 10/10 | 0 | 0.867775 ms | 1.000x |
| 3 | Branch-base dispatcher | `fecf994` | PASS 10/10 | 0.000947773 | 0.156653 ms | 5.171x |
| 3 | Latest packed dispatcher | `ce3f7f2` | PASS 10/10 | 0.000947773 | 0.128575 ms | 6.533x |
| 2 | Immutable baseline | `fecf994` | PASS 10/10 | 0 | 0.895467 ms | 1.000x |
| 2 | Branch-base dispatcher | `fecf994` | PASS 10/10 | 0.000819206 | 0.122745 ms | 8.443x |
| 2 | Latest packed dispatcher | `ce3f7f2` | PASS 10/10 | 0.000819206 | 0.118854 ms | 6.753x |

Paired speedup is the value reported within each benchmark process against its
co-resident immutable baseline. Cross-commit regression decisions use candidate
median latency directly, avoiding baseline clock drift between processes.

## Case-3 Correctness Stress Matrix

Timing in these two-repeat, zero-settle stress cells is diagnostic only. Their
acceptance purpose is numerical coverage.

| Input scale | Padding ratio | Accuracy | Failed elements | Max abs error |
| ---: | ---: | --- | ---: | ---: |
| 0.125 | 0.00 | PASS 10/10 | 0 | 0.00236814 |
| 0.125 | 0.25 | PASS 10/10 | 0 | 0.00236814 |
| 1 | 0.00 | PASS 10/10 | 0 | 0.000947773 |
| 1 | 0.25 | PASS 10/10 | 0 | 0.000947773 |
| 8 | 0.00 | PASS 10/10 | 0 | 0.000124633 |
| 8 | 0.25 | PASS 10/10 | 0 | 0.000124633 |

Errors above the absolute threshold remain valid where they satisfy the
independent 2% relative-error branch of the executable OR criterion.

## Environment

- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB; driver 616.56.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs.
- PyTorch / CUDA runtime / Triton: 2.13.0+cu130 / 13.0 / 3.7.1.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, x86-64, glibc 2.39.
- Dtype and numerical flags: float32, high matmul precision, TF32 enabled.
- Timing: 10 seconds settling and 40 balanced paired samples for acceptance
  runs; automatic blocks targeting 50 ms.

## Records

- Case 3 acceptance: [`candidate-case3-default.json`](candidate-case3-default.json)
- Case 2 regression: [`candidate-case2-regression.json`](candidate-case2-regression.json)
- Case 3 stress matrix:
  [`scale=0.125,padding=0`](case3-scale0p125-padding0.json),
  [`scale=0.125,padding=0.25`](case3-scale0p125-padding0p25.json),
  [`scale=1,padding=0`](case3-scale1-padding0.json),
  [`scale=1,padding=0.25`](case3-scale1-padding0p25.json),
  [`scale=8,padding=0`](case3-scale8-padding0.json), and
  [`scale=8,padding=0.25`](case3-scale8-padding0p25.json).
- Branch-base and immutable controls:
  [`../2026-08-30-rtx5080-fecf994/`](../2026-08-30-rtx5080-fecf994/README.md).

Every JSON records the exact command, UTC timestamp, clean Git revision,
environment, raw samples, official shape, and per-trial correctness details.
