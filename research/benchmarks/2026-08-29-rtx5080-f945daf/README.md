# Case 2 Packed-QKV Merge Check — RTX 5080 — 29 August 2026

## Scope

- Git revision: `f945dafefb6fbfafe889e96db0f4a2cbbf52f5f0`.
- Official case 2: batch 1, sequence 128, model/QKV dimension 128, four
  heads, four layers, FFN dimension 128, causal, float32.
- Candidate: merged `src.dispatcher` route with one packed Q/K/V projection
  per layer, strided views, SDPA, and `reduce-overhead` compilation.
- Timestamp: 29 August 2026 at 13:26:32 UTC.

## Environment

- GPU: NVIDIA GeForce RTX 5080, capability 12.0, driver 616.56.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs.
- PyTorch / CUDA runtime: 2.13.0+cu130 / 13.0; Python 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, x86-64, glibc 2.39.
- Disk at collection: 1,081,101,176,832 bytes total and approximately
  665,934,749,696 bytes free.
- Numerical contract: float32, high matmul precision, TF32 enabled.

## Official Baseline Comparison

Exact command:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 2 --device cuda --dtype float32 \
  --accuracy-trials 5 --repeats 40 --settle-seconds 10 \
  --sample-target-ms 50 \
  --output /tmp/techjam-case2-packed-f945daf.json
```

The output was copied without modification to
[`case2-packed-dispatcher.json`](case2-packed-dispatcher.json). The tree was
source-clean; its dirty flag records two unrelated pre-existing untracked
benchmark directories.

| Baseline median | Packed dispatcher median | Speedup | Noise floor | Correctness |
| ---: | ---: | ---: | ---: | --- |
| 0.790769 ms | 0.127719 ms | **6.191x** | ±28.34% | PASS 5/5; max abs 0.0008095 |

The immutable baseline comparison confirms that the merged route remains a
large, significant end-to-end improvement. It does not isolate packed QKV from
the already-integrated SDPA and compiler improvements.

## Packed-QKV Incremental Diagnostic

A same-process diagnostic instantiated two otherwise-identical dispatchers with
strictly copied reference weights. The control replaced Case 2's packed
attention modules with the prior three-projection `StridedSDPASelfAttention`;
the candidate retained `PackedQKVSDPASelfAttention`. Both used the dispatcher's
normal Case-2 lazy `reduce-overhead` compilation. After five correctness seeds,
each side settled for 10 seconds and ran 40 balanced paired CUDA-event samples
of 200 forwards each. Input generation used seed 101234, scale 1, and no
padding. Raw samples are in
[`case2-packed-vs-strided-diagnostic.json`](case2-packed-vs-strided-diagnostic.json).

| Three-projection control | Packed QKV | Incremental speedup | Noise floor | Correctness |
| ---: | ---: | ---: | ---: | --- |
| 0.159885 ms | 0.129521 ms | **1.234x** | ±1.99% | Both PASS 5/5 |

This isolates a clear RTX 5080 packed-projection win. The diagnostic used a
temporary orchestration script rather than the loadable-candidate harness, so
the official baseline JSON above remains the submission-facing benchmark
record. The paired raw samples and complete comparison construction are retained
to make that distinction explicit.

