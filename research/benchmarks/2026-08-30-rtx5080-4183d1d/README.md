# Case 14 FP32 Streamed Reference — RTX 5080 — 30 August 2026

## Purpose

This is a validation-oracle feasibility record, not a candidate benchmark and
not a competition speedup claim. The official Appendix does not specify dtype,
while the immutable harness defaults to float32. The immutable Case 14 baseline
cannot execute at full shape because its first FP32 attention-score tensor alone
would require 18.626 TiB.

The implementation at `4183d1d` leaves the reference projections, LayerNorm,
FFN, residuals, parameter values, and FP32 dtype unchanged. It replaces only the
explicit score/softmax/probability materialization with algebraically equivalent
SDPA, forces a linear-memory CUDA backend, and processes one batch sample at a
time. Outputs are reduced immediately rather than retained as a full batch.

## Three Reference Points

| Reference | Commit | Correctness | Runtime | Peak allocation | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Immutable dense baseline, full `N=100000` | immutable | UNSUPPORTED: 18.626 TiB score tensor | N/A | N/A | 1.000x definition only |
| Branch-base dispatcher | `7fc4e8d` | UNSUPPORTED: Case 14 FP32 rejected before allocation | N/A | N/A | N/A |
| Latest streamed oracle | `4183d1d` | PASS at dense `N=4096`; full shape finite | 90.581 s | 3,595.902 MiB | N/A: oracle only |

Branch base: `7fc4e8d060349e8ffc00095c4d45c1fafa1988bd`, committed
2026-08-30T21:19:10+08:00. Latest implementation:
`4183d1d510f131d3230a6f078b267b3afbef7cd9`, committed
2026-08-30T23:29:05+08:00.

## Numerical Validation

Before the full run, the streamed oracle and immutable dense
`BaselineTransformer` evaluated the same FP32 model and input at `B=1`,
`N=4096`, `D=1024`, `H=16`, `F=1024`, and two causal layers.

- Criterion: `abs_error <= 0.002 OR abs_error <= 0.02 * abs(reference)`.
- Result: PASS, zero failures across 4,194,304 output elements.
- Maximum absolute error: `0.0006024837493896484`.
- Mean absolute error: `4.2655925655693484e-05`.
- The large maximum relative error occurs at reference values near zero; those
  elements pass the benchmark's absolute-error arm.

The full `B=32`, `N=100000` run then completed all 32 samples with finite FP32
outputs. Its elapsed time was 90.581 seconds and its peak allocated GPU memory
was 3,770,576,896 bytes (3,595.902 MiB).

## Environment

- CPU: AMD Ryzen 7 9800X3D host, reported to Python as `x86_64`.
- GPU: NVIDIA GeForce RTX 5080.
- Driver / CUDA runtime: 616.56 / 13.0.
- PyTorch: 2.13.0+cu130.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 under WSL2, glibc 2.39.
- Matmul precision / TF32: `high` / enabled.
- Git worktree: implementation commit `4183d1d`; dirty only because unrelated
  pre-existing untracked benchmark records were present.

## Exact Command and Result

```bash
.venv/bin/python -m src.case14_fp32_reference \
  --device cuda --batch-size 32 --seq-len 100000 \
  --validate-dense-n 4096 --seed 1234 \
  --output research/benchmarks/2026-08-30-rtx5080-4183d1d/case14-fp32-reference.json
```

- [`case14-fp32-reference.json`](case14-fp32-reference.json) contains the exact
  command, UTC timestamp, Git state, environment, configuration, dense accuracy,
  output fingerprint, elapsed time, and peak allocation.

## Limitations

- This proves a tractable FP32 reference oracle, not an FP32 candidate.
- Fused SDPA is algebraically exact but has a different floating-point reduction
  order from the explicit baseline. The reduced dense comparison measures that
  difference within available memory.
- Sample-wise CUDA random generation is deterministic but not bitwise identical
  to one monolithic `[32, 100000, 1024]` `torch.randn` call. A candidate checked
  by this oracle must consume the same generated sample tensors.
- Timing includes sequential execution of the oracle and output fingerprint
  reductions. It must not be used as the baseline latency in a speedup claim.
