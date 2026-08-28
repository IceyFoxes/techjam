# Extreme-Shape Experiment Protocol

## Current Environment Snapshot

Captured 28 August 2026 before any experiment:

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA L4, 23,034 MiB, no active process |
| Driver | 580.95.05 |
| Driver-advertised maximum CUDA version | 13.0 |
| CUDA toolkit | 12.9, `nvcc` 12.9.86 |
| CPU | AMD EPYC 7R13, 8 cores / 16 threads |
| OS kernel | Amazon Linux, Linux 6.1.158-178.288.amzn2023.x86_64 |
| Workspace disk | 540 GB total, 504 GB available |
| Python | `/usr/local/bin/python` |
| PyTorch | Not installed (`ModuleNotFoundError: No module named 'torch'`) |

No benchmark run is recorded because the dependency needed to execute the
authoritative harness is absent. Installing dependencies or changing runtime
configuration is implementation work and must occur on a separate branch under
the repository rules.

## Isolated Failure Classification

Each official extreme shape must run in a fresh process. The parent runner
should capture exit status, stderr, elapsed wall time, and the last completed
phase. Classify the result as:

| Class | Evidence |
| --- | --- |
| Input-allocation OOM | Failure before model entry while creating/scaling/masking `x` |
| QKV/layout OOM | Failure in normalization, projection, reshape, transpose, or contiguous copy before score creation |
| Attention-score OOM | Failure creating/scaling/masking scores or probabilities |
| FFN OOM | Attention completed; failure in norm2, `ffn_in`, GELU, or `ffn_out` |
| Checker OOM | Both forwards completed; failure in `compare_outputs` |
| Timeout | Child exceeds the declared wall-time limit without CUDA OOM |
| Numerical failure | Process completes but at least one element violates the executable OR tolerance |

Use `CUDA_LAUNCH_BLOCKING=1` for diagnostic classification only, not latency
measurement, because PyTorch documents that CUDA execution is asynchronous.

## Preflight Before Allocation

For each full official tuple:

1. Compute input, output, QKV, score, causal mask, FFN, model, and checker sizes.
2. Compare each single requested tensor and each known live-set lower bound to
   total and currently free VRAM.
3. If one allocation exceeds total VRAM, record the deterministic blocker and
   skip the destructive allocation attempt.
4. If only the active set exceeds capacity, run the phase-specific child once
   to confirm the exact first allocation, then stop repeated attempts.
5. Never treat a score-only estimate as a full-model peak.

## Required Measurements

Once PyTorch and official dimensions are available, preserve only meaningful
baseline, accepted checkpoint, regression, and final runs under the benchmark
directory required by `AGENTS.md`.

| Run | Purpose | Required fields |
| --- | --- | --- |
| Explicit root baseline | Identify first failure for each extreme shape | Exact command, commit, timestamp, tuple, dtype, exit class, elapsed time, stderr excerpt, environment |
| Forced Flash SDPA microbenchmark | Prove backend eligibility and memory behavior | Backend identity, `Bq/Bk` or API path, correctness, peak allocated/reserved VRAM, latency |
| Query/key tile sweep | Select tile sizes | At least `(64,64)`, `(128,64)`, `(128,128)`, `(256,128)`, `(256,256)` when launchable |
| Batch-chunk sweep | Select largest safe batch chunk | Chunks 1, 2, 4, 8, then larger only while under the memory guardrail |
| Whole-layer run | Catch QKV/FFN integration peaks | Accuracy fields, peak VRAM by phase, median/p90/min latency |
| Repeated regression run | Detect allocator growth or nondeterministic tails | Multiple seeds, padding ratios, input scales, stable peak reserved memory |

## Memory Guardrail

Use the lower of 85% of total VRAM and total VRAM minus a measured fixed
reserve for context, both models, and library workspace. On the local L4, 85%
of 23,034 MiB is 19,579 MiB. A configuration that succeeds once above this
level is not accepted as robust.

Record both:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

`allocated` represents live tensor storage tracked by PyTorch. `reserved`
includes cached allocator segments and is relevant to repeatability. A large
gap should trigger a memory snapshot or summary before allocator tuning.

## Correctness Progression

1. Validate recurrence and mask semantics in FP32 at small lengths.
2. Compare FP16 and BF16 one-pass online attention with the exact root
   `compare_outputs()` on feasible shapes.
3. Sweep causal/non-causal, no padding, right padding, input scales, and seeds.
4. Test long reductions at the largest length for which the explicit reference
   fits.
5. If one-pass fails, test the two-pass probability-cast strategy.
6. Use a streamed checker only for target scale, and cross-validate it against
   the root checker on small tensors before trusting it.

No latency or VRAM number should enter the result table until the kernel backend
has been identified and correctness passes.
