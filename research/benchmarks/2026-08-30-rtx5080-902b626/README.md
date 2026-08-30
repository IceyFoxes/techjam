# Case 14 Current FP16 Candidate vs FP32 Oracle — RTX 5080

## Result

The current Case-14 FP16/polynomial backend passes the streamed FP32 reference
oracle at the complete official shape.

- Shape: `B=32, N=100000, D=1024, H=16, F=1024`, two causal layers.
- Shared input: each FP32 oracle sample is explicitly cast to FP16 for the
  unchanged candidate backend.
- Result: **PASS**, zero failures across 3,276,800,000 output elements.
- Maximum absolute error: `0.01025533676147461`.
- Mean absolute error: `0.0003741441356491592`.
- Polynomial guard sigma by layer: `0.33338398`, `0.33422178`; both are below
  the validated `0.40` ceiling.
- Complete validation-run wall time: 102.360 seconds.
- Candidate-forward time accumulated inside that run: 10.752 seconds. This is a
  diagnostic, not an official paired speedup measurement.
- Peak allocation with both models resident and one sample compared at a time:
  3,795,787,776 bytes (3,619.945 MiB).

Although maximum absolute error exceeds `atol=0.002`, those elements pass the
other arm of the official rule: `error <= 0.02 * abs(reference)`. The very large
reported maximum relative error occurs at reference values near zero, where the
absolute arm passes; it is not a failed element.

## Three Reference Points

| Reference | Commit | Accuracy | Runtime | Speedup |
| --- | --- | --- | ---: | ---: |
| Immutable dense baseline | immutable | Full shape unsupported; oracle agrees at `N=4096`, 0/4,194,304 failures | N/A | 1.000x definition only |
| Branch base | `4276b5b` | FP32 dispatcher unsupported; no candidate-vs-FP32 comparison runner | N/A | N/A |
| Latest comparison | `902b626` | PASS 32/32 samples, 0/3,276,800,000 failures | 102.360 s complete validation | N/A: correctness run |

The implementation branch originally forked from `7fc4e8d`; local master's
`4276b5b` adds only the earlier FP32-oracle research record. Neither base changes
the candidate code being checked. Latest comparison commit:
`902b626f8faf32dcfea5a7a65d0854e77ed68ffa`, committed
2026-08-30T23:42:33+08:00.

## Method

1. Construct the FP32 streamed reference and current Case-14 candidate with the
   same official configuration and strictly copy one state dict.
2. Keep the oracle in FP32 and convert the candidate model to FP16, matching its
   current validated execution route.
3. Generate one deterministic FP32 input sample at a time.
4. Run the FP32 oracle, cast that exact input to FP16, run the unchanged
   polynomial candidate, and cast its output back to FP32.
5. Apply `abs_error <= 0.002 OR abs_error <= 0.02 * abs(reference)` in bounded
   token chunks, then discard both outputs.

Before the target-scale comparison, the FP32 oracle itself passed against the
immutable dense `BaselineTransformer` at `N=4096`: zero failures, maximum
absolute difference `0.0006024837`, mean absolute difference `0.0000426559`.

## Exact Command and Environment

```bash
.venv/bin/python -m src.case14_fp32_reference \
  --device cuda --batch-size 32 --seq-len 100000 \
  --validate-dense-n 4096 --seed 1234 \
  --compare-current-candidate \
  --output research/benchmarks/2026-08-30-rtx5080-902b626/case14-fp16-vs-fp32-oracle.json
```

- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0.
- Driver / CUDA runtime: 616.56 / 13.0.
- PyTorch: 2.13.0+cu130.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 under WSL2, glibc 2.39.
- Matmul precision / TF32: `high` / enabled.
- [`case14-fp16-vs-fp32-oracle.json`](case14-fp16-vs-fp32-oracle.json)
  preserves the exact command, per-sample results, environment, Git state,
  output fingerprint, timings, sigma values, and memory peak.

## Scope and Limitations

- This validates the current Case-14 backend's numerical output against a FP32
  reference oracle. It does not make the submitted dispatcher accept FP32 input;
  the comparison runner deliberately performs the input/output casts around the
  unchanged backend.
- The run covers one model seed and 32 independently generated full-length input
  samples. It is stronger than a one-sample smoke check but is not a five-model-
  seed adversarial sweep.
- Streamed CUDA random generation is deterministic but not bitwise identical to
  one monolithic `[32,100000,1024]` random call. Oracle and candidate consume the
  same tensors, so this does not weaken their numerical comparison.
- The immutable full-shape dense reference remains impossible; fused FP32 SDPA
  differs in reduction order. Its reduced dense validation bounds that oracle
  difference within the executable tolerance.
