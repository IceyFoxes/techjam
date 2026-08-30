# NVIDIA L4 Extreme-Memory Checkpoint

## Status

Current Person 4 implementation evidence for commit
`f128f6ee6a40619d5b92bc92b91d60d13d266f15`, collected 29 August 2026.

This directory preserves one minimal official Case 6 comparison and one
candidate-only official Case 14 allocation smoke test. The low repeat counts
make latency preliminary rather than final competition evidence.

## Environment

| Component | Value |
| --- | --- |
| CPU | x86_64 on AMD EPYC 7R13 host |
| GPU | NVIDIA L4, 23,034 MiB |
| OS | Linux 6.1.158-178.288.amzn2023.x86_64 |
| Driver | 580.95.05 |
| CUDA runtime | 13.0 |
| CUDA toolkit | 12.9 |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cu130 |
| Disk | 579,740,872,704 bytes total; about 535 GB free |

## Case 6

Command:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 6 --device cuda --dtype float32 \
  --accuracy-trials 1 --timing legacy --warmup 1 --repeats 1 \
  --benchmark-rounds 1 --settle-seconds 0 \
  --output research/benchmarks/2026-08-29-l4-f128f6e/case6.json
```

Shape: `B=10000, N=128, D=128, H=4, F=128, L=4`, causal, FP32.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Correctness | Reference | PASS, 0 / 163,840,000 failures |
| Maximum absolute error | N/A | 0.00140804 |
| Median latency | 1129.343 ms | 454.020 ms |
| Incremental peak allocated | 9.776 GiB | 2.612 GiB |

Preliminary speedup: `2.487x`. The result contains one timed sample after one
warmup and is useful as an implementation checkpoint, not a noise-qualified
final result.

## Case 14

Command:

```bash
.venv/bin/python -m src.extreme_smoke \
  --case 14 --dtype float16 \
  --output research/benchmarks/2026-08-29-l4-f128f6e/case14-smoke.json
```

Shape: `B=32, N=100000, D=1024, H=16, F=1024, L=2`, causal, FP16.

| Metric | Candidate |
| --- | ---: |
| Route | `extreme-memory` |
| Selected batch chunk | 2 |
| Output shape | `[32,100000,1024]` |
| Full output finite | Yes |
| Elapsed wall time | 25.230 s |
| Peak allocated | 15.270 GiB |
| Peak reserved | 15.314 GiB |

Case 14 is candidate-only because the immutable baseline requires a 9.313 TiB
FP16 score tensor and cannot run on this GPU. The smoke result proves memory-safe
completion and full-output finiteness, not target-scale numerical correctness.

Reduced representative tests at `D=1024, H=16, L=2` pass the executable
tolerance for FP16 at the official default input scale. BF16 and FP16 at input
scale 0.1 failed reduced-shape tests and are therefore explicitly outside the
candidate contract.

## Verification

```bash
.venv/bin/python -m unittest discover -s src/tests -v
.venv/bin/python -m compileall -q src
git diff --check
```

All 79 tests passed, including a CUDA reduced-shape Flash-versus-reference
correctness test, adaptive chunking, OOM backoff, right-padding, preflight
contracts, and unsafe-fallback prevention.
