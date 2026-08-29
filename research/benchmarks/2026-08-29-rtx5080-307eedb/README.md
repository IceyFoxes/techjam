# Hardened Twelve-Case Dispatcher — RTX 5080 — 29 August 2026

## Scope and Revision

- Implementation branch: `impl/person1-dispatcher-hardening`.
- Git revision under test: `307eedb2777c483befe7eadaecf1a7a9f5aff6be`.
- Pull request: #6, `Harden dispatcher execution contracts`.
- Candidate: `src.dispatcher`, with internal lazy compilation; the benchmark
  did not pass `--compile-user`.
- Covered official cases: 1-5 and 7-13, the twelve cases with an implemented
  memory-safe route.
- Explicitly unsupported: cases 6 and 14, rejected before model/input allocation
  pending an extreme-shape backend.

Cases 1-5 and 7-12 use float32 SDPA, strided Q/K/V views, and
`torch.compile(mode="reduce-overhead")`. Case 13 uses the same SDPA/view path
with ordinary/default compilation. The route is enabled only for the preserved
RTX 5080, PyTorch 2.13.0+cu130, high-matmul-precision, TF32-enabled contract.

## Environment and Method

- GPU: NVIDIA GeForce RTX 5080; driver 616.56; CUDA runtime 13.0.
- PyTorch: 2.13.0+cu130; Python 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, glibc 2.39.
- Timing: paired CUDA events, 10-second settling, 40 paired samples, and an
  automatically selected block targeting 50 ms.
- Correctness: five seeds, input scale 1.0, zero padding, and zero failed output
  elements under the executable absolute-error-at-most-0.002 **or**
  relative-error-at-most-2% rule.
- Memory: after lazy compilation was warm, one baseline and candidate forward
  each recorded pre-forward, absolute peak, and incremental peak allocated and
  reserved CUDA bytes. Both models and compiler caches remained resident, so
  absolute peaks describe the actual benchmark process rather than standalone
  model storage.

Every JSON contains the exact command, timestamp, official shape, raw latency
samples, accuracy trials, environment, Git state, compiler contract metadata,
and CUDA memory measurements. Files after case 1 report a dirty tree only
because preceding immutable JSON records existed untracked; source code remained
fixed at `307eedb` throughout.

## Results

| Case | Baseline ms | Dispatcher ms | Speedup | Noise floor | Accuracy |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.4160 | 0.5361 | 2.641x | ±26.73% | PASS 5/5; max abs 0.001054 |
| 2 | 1.3027 | 0.1737 | 7.498x | ±127.86% | PASS 5/5; max abs 0.000770 |
| 3 | 0.7964 | 0.1815 | 4.389x | ±31.30% | PASS 5/5; max abs 0.000908 |
| 4 | 0.8466 | 0.2075 | 4.081x | ±37.81% | PASS 5/5; max abs 0.000908 |
| 5 | 2.8440 | 0.9747 | 2.918x | ±21.33% | PASS 5/5; max abs 0.001181 |
| 7 | 1.2455 | 0.3381 | 3.684x | ±25.25% | PASS 5/5; max abs 0.001316 |
| 8 | 14.6409 | 13.0909 | 1.118x | ±3.90% | PASS 5/5; max abs 0.001218 |
| 9 | 1.1347 | 0.5243 | 2.164x | ±30.14% | PASS 5/5; max abs 0.001078 |
| 10 | 1.4694 | 0.5631 | 2.610x | ±16.14% | PASS 5/5; max abs 0.001089 |
| 11 | 6.0541 | 1.0864 | 5.573x | ±24.47% | PASS 5/5; max abs 0.001063 |
| 12 | 0.7666 | 0.1798 | 4.263x | ±59.94% | PASS 5/5; max abs 0.001128 |
| 13 | 92.6065 | 13.3312 | 6.947x | ±12.36% | PASS 5/5; max abs 0.001123 |

The geometric-mean speedup across all twelve supported cases is **3.548x**.
Every improvement clears the harness's run-specific significance rule. Exact
values for cases 2 and 12 remain particularly noisy; their direction is useful,
but their point estimates are not final score predictions. Case 8 has the
smallest gain but a comparatively tight decision margin.

Each result links to its immutable JSON record:

- [`case1`](case1-fp32-dispatcher.json),
  [`case2`](case2-fp32-dispatcher.json),
  [`case3`](case3-fp32-dispatcher.json),
  [`case4`](case4-fp32-dispatcher.json),
  [`case5`](case5-fp32-dispatcher.json),
  [`case7`](case7-fp32-dispatcher.json),
  [`case8`](case8-fp32-dispatcher.json),
  [`case9`](case9-fp32-dispatcher.json),
  [`case10`](case10-fp32-dispatcher.json),
  [`case11`](case11-fp32-dispatcher.json),
  [`case12`](case12-fp32-dispatcher.json), and
  [`case13`](case13-fp32-dispatcher.json).

## CUDA Memory Checkpoints

| Case | Baseline peak allocated | Candidate peak allocated | Candidate incremental peak | Process peak reserved |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 171.11 MiB | 107.06 MiB | 64.00 MiB | 334 MiB |
| 8 | 544.37 MiB | 428.34 MiB | 172.00 MiB | 820 MiB |
| 13 | 2,372.23 MiB | 1,219.10 MiB | 1,152.00 MiB | 2,438 MiB |

The candidate does not increase the warmed-process peak allocated memory on
these three risk-bearing supported cases. No run required an increase beyond
the already reserved CUDA allocator pool during the measured memory forward.

## Limits

- Compilation/cold-start time is excluded; compilation and one replay are
  deliberately completed before a route enters the runtime cache.
- The final timing suite uses zero padding. Padded CUDA execution was separately
  checked on case 1 at 25% padding with three passing seeds.
- Input-scale and adversarial near-zero sweeps remain future acceptance work.
- Cases 6 and 14 have no runtime result because the candidate now refuses their
  unsafe dense path before allocation.
- Case 8 still uses the reference projection/FFN layout; Person 3's packed-QKV
  backend remains unimplemented.

No recorded file has been overwritten.
