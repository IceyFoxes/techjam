# Direct cuBLASLt Case-8 Algorithm Probe

## Status

- Date: 30 August 2026.
- Implementation base: `c585cfa8`, branch `impl/person3-cublaslt-probe`.
- Environment: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9,
  driver 610.57.04, PyTorch 2.13.0+cu130, CUDA runtime 13.0, and cuBLASLt
  13.1.1 (`cublasLtGetVersion() == 130101`).
- Shape: FP32/TF32 `(M,N,K) = (8192,1024,1024)`, matching all 24 dense
  operations in official Case #8.
- Decision: **reject**. No tested legal route has a 5% whole-model opportunity.
- These are local exploratory selection measurements, not final RTX 5080
  benchmark claims. Raw working outputs remain under `/tmp/opencode/`.

## Probe

The host-only PyTorch C++ extension calls cuBLASLt directly and uses the current
PyTorch CUDA stream. It obtains every FP32/TF32 algorithm ID with
`cublasLtMatmulAlgoGetIds`, expands advertised tile, stage, custom-option, and
swizzle settings, and tests practical split-K counts `{2,3,4,5,6,8,12,16,32}`
with each supported reduction scheme. Every retained configuration must pass
`cublasLtMatmulAlgoCheck` for both default and bias descriptors, including the
workspace limit.

Two weight layouts were searched independently:

- normal contiguous PyTorch `[N,K]`, represented as transposed column-major A;
- one-time pretransposed contiguous `[K,N]`, represented as non-transposed A.

Both use the column-major transpose interpretation of the row-major activation
and output. This makes semantic D `[N,M]`, so the cuBLASLt bias length is the
required output width `N=1024`. Prepacking is outside timed execution.

The search found 727 valid configurations for each layout. All attempted valid
configurations launched. The selected normal-layout mainloop was:

```text
algorithm=21, tile=20, stages=4, split_k=1,
reduction=none, swizzle=0, custom=0, workspace=0
```

The same explicit `cublasLtMatmulAlgo_t` configuration was used without
substitution for plain GEMM, bias, `beta*C` residual, and bias plus residual.

## Measurements

The final comparison used 10 seconds of settling and 100 interleaved AB/BA CUDA
event samples. Reported speedup is the median of paired baseline/candidate ratios.

| Operation | PyTorch median | Explicit cuBLASLt median | Paired speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| GEMM | 2.1253 ms | 2.0972 ms | 1.0146x | Neutral |
| GEMM + bias | 2.1412 ms | 2.1125 ms | 1.0134x | Neutral |
| GEMM + residual, no bias | 2.7500 ms | 2.1862 ms | 1.2595x | Illegal for benchmark linear |
| GEMM + bias + residual | 2.7428 ms | 2.8646 ms | 0.9555x | Reject, regression |

The normal biased route is the relevant replacement for all 24 dense calls. With
their measured 75.49% Case-8 device-time share, its optimistic Amdahl projection
is only:

```text
1 / ((1 - 0.7549) + 0.7549 / 1.0134) = 1.010x
```

This is far below the required 1.0526x speedup corresponding to a 5% latency
reduction. Residual-only fusion could have a static whole-model opportunity only
for the eight output GEMMs, but it omits their mandatory bias. Supplying that bias
through the supported fused epilogue changes the measured result from a 25.95%
isolated gain to a 4.45% regression. A separate bias pass would restore a full
activation traversal and is not a new dataflow win over the compiled control.

The prepacked search did not produce a better finalist. Its leading biased result
was approximately 2.12 ms in the expanded run, versus approximately 1.65-2.11 ms
for normal finalists depending on thermal position; the paired normal result
above is the accepted local comparison.

## Correctness

The selected configuration was tested over three seeds, input and residual scales
`{0.125,1,8}`, and all four epilogue variants, for 36 comparisons. Every comparison
passed the executable `abs_error <= 0.002 OR relative_error <= 0.02` rule with zero
failed elements. The worst maximum absolute error was `7.6293945e-6`.

Plain, bias-only, and residual-only outputs were bitwise identical to their
PyTorch controls in the initial full-shape screen. Bias plus residual differed by
at most `9.536743e-7` there because fusion removes the materialized biased-linear
checkpoint, but it remained well inside tolerance.

## Commands

```bash
MAX_JOBS=2 .venv/bin/python src/cublaslt_probe.py \
  --repeats 5 --warmup 2 --shortlist 8 --settle-seconds 5

PYTHONPATH=. .venv/bin/python src/cublaslt_validate.py \
  --repeats 100 --settle-seconds 10
```

## Conclusion

The direct API closes the uncertainty left by the earlier indirect weight-layout
screen on this RTX 4050. Explicit algorithm selection does not materially improve
the required biased mainloop, physical pretransposition does not win, and adding
the mandatory bias to the promising residual epilogue makes it slower. No
whole-model candidate is integrated because the isolated evidence proves the 5%
acceptance threshold unreachable for the tested mechanisms.

RTX 5080 algorithm IDs and schedules are architecture-specific, so this does not
replace an RTX 5080 rerun if that machine is available. It does establish a
reproducible direct probe for that rerun and provides no local evidence supporting
integration.

## Sources

- NVIDIA cuBLAS 13.3 documentation, accessed 30 August 2026,
  <https://docs.nvidia.com/cuda/cublas/index.html>. Relevant symbols are
  `cublasLtMatmulAlgoGetIds`, `cublasLtMatmulAlgoInit`,
  `cublasLtMatmulAlgoCheck`, `cublasLtMatmul`, and
  `CUBLASLT_EPILOGUE_BIAS`.
- Installed NVIDIA header `nvidia/cu13/include/cublasLt.h`, cuBLASLt 13.1.1.
  It supplies the exact configuration and capability attributes used by the
  executable probe.
- Immutable local benchmark `torch_transformer_benchmark.py`, symbols
  `BaselineTransformerBlock.forward` and `compare_outputs`, and current
  dispatcher `src/dispatcher.py`.
