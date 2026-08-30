# Case 14 FP32 Dispatcher vs Streamed FP32 Oracle — RTX 5080

## Result

The editable benchmark now recognizes official Case 14 with the dispatcher and
`--dtype float32`, substitutes the validated linear-memory FP32 oracle for the
infeasible dense reference, and feeds the same FP32 samples directly to the
dispatcher. The dispatcher presents an FP32 interface, performs the existing
Case-14 route in FP16, and returns FP32 output.

- Shape: `B=32, N=100000, D=1024, H=16, F=1024`, two causal layers.
- Accuracy: **PASS 5/5 trials**, zero failures across `16,384,000,000`
  elements.
- Maximum absolute error: `0.010293960571289062`.
- Mean absolute error: `0.00037412972828601294`.
- Per-trial maximum absolute errors: `0.01029396`, `0.00993299`,
  `0.00998259`, `0.00990772`, and `0.01000071`.
- Oracle time accumulated over the samples: `449.6634 s`.
- Candidate time accumulated over the same samples: `52.3967 s`.
- Diagnostic oracle/candidate ratio: **8.5819x**.
- Peak allocation: `3,820,998,656` bytes (`3,643.988 MiB`).
- Polynomial guard sigma by layer: `0.333984375`, `0.33447265625`; both are
  below the validated `0.40` ceiling.
- CUDA-aware regression suite: 210 tests passed in 41.155 seconds.

The 8.5819x ratio compares two linear-memory proxy implementations executed
sample by sample. It is useful for judging this FP32-compatible route, but it is
not an official dense-baseline speedup: the immutable full-shape reference is
not executable at Case 14.

Although maximum absolute error exceeds `atol=0.002`, every such element passes
the other arm of the official rule: `error <= 0.02 * abs(reference)`. Relative
errors reported near a reference value of zero are correspondingly huge, but
those elements pass the absolute arm.

## Three Reference Points

| Reference | Revision | Accuracy | Timing |
| --- | --- | --- | --- |
| Immutable dense baseline | immutable root harness | Full Case 14 is infeasible; at `N=4096` the streamed oracle passes with 0/4,194,304 failures and `max_abs=0.00060248` | unavailable at full shape |
| Branch base | `3c87546` | oracle exists, but dispatcher rejects FP32 Case 14 | unavailable |
| Latest tested result | `27af97d` | **PASS 5/5**, 0/16,384,000,000 failures, `max_abs=0.01029396` | oracle `449.6634 s`; candidate `52.3967 s`; diagnostic 8.5819x |

## Exact Command and Environment

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 14 --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --warmup 0 --repeats 1 \
  --benchmark-rounds 1 \
  --output research/benchmarks/2026-08-30-rtx5080-27af97d/case14-fp32-dispatcher.json
```

- Tested commit: `27af97d1f9064e0fff9b03b12bd5dfcd96dee6e8`.
- Timestamp: `2026-08-30T16:06:05.313694+00:00`.
- GPU: NVIDIA GeForce RTX 5080.
- Driver / CUDA runtime: 616.56 / 13.0.
- PyTorch: 2.13.0+cu130.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 under WSL2, glibc 2.39.
- Matmul precision / TF32: `high` / enabled.
- [`case14-fp32-dispatcher.json`](case14-fp32-dispatcher.json) preserves the
  command, environment, Git state, all 160 sample comparisons, per-trial
  results, timings, output fingerprint, sigma values, and memory peak.

## Scope and Limitations

- The root `torch_transformer_benchmark.py` remains unmodified. Only the
  editable `src.benchmark` dispatcher path substitutes the oracle.
- The dense `N=4096` gate validates the oracle against the immutable reference;
  the full run then compares the dispatcher against that oracle.
- Five trials vary the input seed (`1234` through `1238`) while retaining one
  model initialization, matching the benchmark's normal trial semantics.
- This route is deliberately Case-14-specific. Extending whole-model FP16
  compute to another case requires that case's own multi-seed accuracy and
  performance gate.
