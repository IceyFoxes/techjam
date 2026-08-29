# Integrated SDPA + Compiler Probe — RTX 5080 — 29 August 2026

## Scope and Revision

- Implementation branch: `impl/person1-integrated-probe`.
- Git revision under test: `330cf602d331a9e577e7260ff056ddd97d707fc7`.
- Candidate: `IntegratedProbeTransformer` in
  `src/implementations/integrated_probe.py` at that revision.
- Optimization: float32 `scaled_dot_product_attention`, strided Q/K/V head
  views, and whole-candidate `torch.compile(mode="reduce-overhead")`.
- Covered official cases: 1, 3, 4, 5, 7, 9, 10, 11, and 12.
- Deliberately excluded here: cases 2, 8, and 13 already have Person 1 compiler
  controls; cases 6 and 14 require a memory-safe extreme-shape backend.

The probe preserves all projection, normalization, residual, FFN, and output
masking operations and state-dict keys. Only the reference attention score,
softmax, dropout, and value-product sequence is replaced by PyTorch SDPA; Q/K/V
remain non-contiguous transposed views. Unit tests on the implementation branch
cover state-dict compatibility, all-valid inputs, and padded-token semantics.

## Environment and Method

- GPU: NVIDIA GeForce RTX 5080; driver 616.56; CUDA runtime 13.0.
- PyTorch: 2.13.0+cu130; Python 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, glibc 2.39.
- CPU: AMD Ryzen 7 9800X3D, x86_64.
- Timing: paired CUDA events, 10-second settling, 40 paired samples, and an
  automatically selected timing block targeting 50 ms.
- Correctness: five seeds per case under the executable benchmark rule of
  absolute error at most 0.002 **or** relative error at most 2%, with zero failed
  elements required.

Every JSON contains the exact command, UTC timestamp, official shape, dtype,
tolerances, per-seed correctness, raw latency samples, speedup, run-specific
noise floor, GPU state, environment, and Git state. Files after case 1 report a
dirty tree only because the preceding immutable JSON records were untracked in
this new directory; the implementation and benchmark source stayed fixed at
`330cf60` throughout.

The command pattern was:

```bash
.venv/bin/python -m src.benchmark \
  --candidate integrated_probe --case CASE --device cuda --dtype float32 \
  --compile-user --compile-mode reduce-overhead --accuracy-trials 5 \
  --repeats 40 --settle-seconds 10 --sample-target-ms 50 \
  --output research/benchmarks/2026-08-29-rtx5080-330cf60/RESULT.json
```

## Results

| Case | Shape difference | Baseline ms | Candidate ms | Speedup | Noise floor | Accuracy | Record |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ordinary control | 1.8277 | 0.7346 | 2.488x | ±32.12% | PASS 5/5; max abs 0.001054 | [`case1`](case1-fp32-sdpa-strided-reduce-overhead.json) |
| 3 | batch 4 | 0.9096 | 0.1836 | 4.955x | ±53.56% | PASS 5/5; max abs 0.000908 | [`case3`](case3-fp32-sdpa-strided-reduce-overhead.json) |
| 4 | batch 16 | 1.1079 | 0.2356 | 4.702x | ±93.73% | PASS 5/5; max abs 0.000908 | [`case4`](case4-fp32-sdpa-strided-reduce-overhead.json) |
| 5 | batch 128 | 2.6553 | 1.0863 | 2.444x | ±27.56% | PASS 5/5; max abs 0.001181 | [`case5`](case5-fp32-sdpa-strided-reduce-overhead.json) |
| 7 | model/FFN 32; head dim 8 | 1.4667 | 0.4328 | 3.389x | ±21.91% | PASS 5/5; max abs 0.001316 | [`case7`](case7-fp32-sdpa-strided-reduce-overhead.json) |
| 9 | 1 head; head dim 128 | 1.1761 | 0.5527 | 2.128x | ±45.86% | PASS 5/5; max abs 0.001078 | [`case9`](case9-fp32-sdpa-strided-reduce-overhead.json) |
| 10 | 2 heads; head dim 64 | 1.3927 | 0.5374 | 2.592x | ±30.82% | PASS 5/5; max abs 0.001089 | [`case10`](case10-fp32-sdpa-strided-reduce-overhead.json) |
| 11 | 16 heads; head dim 8 | 6.2062 | 1.1544 | 5.376x | ±10.34% | PASS 5/5; max abs 0.001063 | [`case11`](case11-fp32-sdpa-strided-reduce-overhead.json) |
| 12 | sequence 32 | 0.8274 | 0.1957 | 4.227x | ±21.76% | PASS 5/5; max abs 0.001128 | [`case12`](case12-fp32-sdpa-strided-reduce-overhead.json) |

The geometric-mean speedup across these nine cases is **3.397x**. All nine clear
the harness's significance rule. Cases 3, 4, and 9 have wide run-specific noise
floors because their sub-millisecond paths are launch-bound, so their exact
speedup values should not be presented as final score predictions. Case 11 is
the strongest and least ambiguous result: 5.376x with a ±10.34% floor.

## Decision and Limits

This is direct evidence to promote the float32 SDPA + strided views +
`reduce-overhead` composition as the current leading implementation route for
all nine covered tuples. It is not yet final acceptance evidence:

- padded inputs were covered by unit tests and an earlier case-1 exploratory
  benchmark, but this nine-case timing sweep used zero padding;
- only ordinary Gaussian input scale 1.0 and five seeds were measured;
- compile time, graph recompiles, and peak retained memory were not measured;
- case 5 needs a peak-memory check before CUDA Graph replay is accepted; and
- the probe does not include Person 3 packed projection/FFN work.

No recorded file has been overwritten.
