# Final Submission Technical Report

Status: final validation completed 31 August 2026. Implementation under test:
`impl/case14-fp32-streamed-oracle` at
`775c82004fd31d5a2203619f671ee214f444411c`.

## Executive Summary

The final implementation is a strict shape-aware dispatcher rather than one
universal kernel. It integrates the fastest numerically accepted route for each
of the fourteen disclosed Transformer shapes while retaining safe fallbacks and
explicitly bounded extreme-shape execution.

On the target RTX 5080:

- Cases 1-13 pass 65/65 accuracy trials with zero failed elements and a
  **3.611x geometric-mean speedup** over the immutable dense reference.
- All thirteen paired improvements are significant under the harness's
  run-specific criterion. Speedups range from **1.116x** on projection-bound
  Case 8 to **8.577x** on long-sequence Case 13.
- Case 6 reduces peak allocated memory from 10,672.719 MiB to 2,312.512 MiB
  while improving latency by **2.394x**.
- Case 14 passes five full-shape FP32-oracle trials with zero failures across
  16,384,000,000 elements. The streamed oracle takes 601.975 s and the candidate
  67.095 s, a **diagnostic 8.972x ratio**, at 3,643.988 MiB peak allocation.
- The latest implementation changes only Case 14 relative to branch base
  `3e6a3ea`. Cases 1-13 reproduce without accuracy regression, and candidate
  latency changes remain between `-1.82%` and `+2.14%`.

Case 14 is not included in the 3.611x geometric mean because its immutable dense
baseline cannot run. Its timing compares two linear-memory proxy
implementations and is not represented as an official dense-baseline score.

## Problem and Correctness Contract

The benchmark applies two causal Transformer residual blocks per Case 14 and
four blocks per every other case, with Q/K/V projection, multi-head attention,
LayerNorm, GELU FFN, residual additions, and a final LayerNorm. Each output
element passes if either:

```text
abs(candidate - reference) <= 0.002
OR
abs(candidate - reference) <= 0.02 * abs(reference)
```

One failed element fails a trial. The root
[`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py) is
immutable and remains the authority for model semantics, inputs, and the
checker.

## Final Machine Specifications

| Component | Final benchmark machine |
| --- | --- |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| CPU cache | 384 KiB L1d, 256 KiB L1i, 8 MiB L2, 96 MiB L3 |
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0 |
| GPU memory | 16,303 MiB |
| GPU limits | 360 W; queried maximum SM clock 3,090 MHz; memory clock 15,001 MHz |
| Driver | NVIDIA 616.56 |
| CUDA runtime | 13.0 |
| cuDNN | 9.2.0 (`92000`) |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| Python | 3.12.3 |
| OS | WSL2, Linux 6.6.114.1, x86-64, glibc 2.39 |
| Disk | 1,007 GiB filesystem; 616 GiB available before the final run |

The benchmark explicitly sets float32 matmul precision to `high`, enables TF32,
uses input scale 1.0 and zero padding, and executes on CUDA.

## Final Benchmark Method

For Cases 1-13, three reference points were measured:

1. An immutable-reference A/A control at branch-base commit `3e6a3ea`.
2. The dispatcher at branch base `3e6a3ea`.
3. The latest dispatcher at `775c820`.

Every accuracy result uses five seeds (`1234` through `1238`). Timing uses
paired, interleaved CUDA events, 10 seconds of settling for each arm, 100 paired
samples, and an automatically selected block targeting 50 ms. Memory is probed
after route compilation and replay are warm. The candidate is self-compiling;
the benchmark does not wrap it in an external compiler.

The A/A controls expose the machine's launch-bound timing floor. Most are within
noise; the two nominally significant controls are Case 1 at 1.106x and Case 6
at 0.977x. These controls are retained to prevent interpreting small timing
movement as an implementation effect.

For Case 14, the immutable dense attention would materialize tensors on the
order of terabytes, and its full FP32 input plus output alone is about 24.4 GiB.
The final harness therefore:

1. Validates the linear-memory FP32 oracle against the immutable dense model at
   `B=1, N=4096`: 0/4,194,304 failures, `max_abs=0.00060248`.
2. Generates one deterministic FP32 sample at a time at the complete
   `B=32, N=100000, D=1024` target shape.
3. Passes the same sample directly to the FP32 oracle and FP32-facing candidate.
4. Reduces the official comparison in bounded token chunks and immediately
   discards each sample's outputs.

## Complete Result Table

| Case | Immutable baseline ms | Latest candidate ms | Speedup | Max abs error | Accuracy | Noise floor |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1.9350 | 0.6975 | 2.774x | 0.001054 | PASS 5/5 | 25.30% |
| 2 | 1.1953 | 0.1609 | 7.428x | 0.000819 | PASS 5/5 | 62.49% |
| 3 | 0.9788 | 0.1643 | 5.959x | 0.000908 | PASS 5/5 | 34.53% |
| 4 | 1.1567 | 0.2678 | 4.319x | 0.000908 | PASS 5/5 | 17.64% |
| 5 | 3.0159 | 1.2513 | 2.410x | 0.001181 | PASS 5/5 | 8.85% |
| 6 | 504.1028 | 210.5723 | 2.394x | 0.001347 | PASS 5/5 | 0.52% |
| 7 | 1.5418 | 0.4060 | 3.797x | 0.001316 | PASS 5/5 | 33.12% |
| 8 | 17.3913 | 15.5889 | 1.116x | 0.001218 | PASS 5/5 | 0.82% |
| 9 | 1.5037 | 0.6881 | 2.185x | 0.001078 | PASS 5/5 | 11.03% |
| 10 | 1.8052 | 0.6464 | 2.793x | 0.001089 | PASS 5/5 | 20.44% |
| 11 | 7.6778 | 1.3724 | 5.595x | 0.001063 | PASS 5/5 | 7.27% |
| 12 | 1.0901 | 0.2335 | 4.668x | 0.001128 | PASS 5/5 | 41.84% |
| 13 | 116.7147 | 13.6078 | 8.577x | 0.001123 | PASS 5/5 | 9.12% |
| 14 | unsupported dense | 67.0945 s | 8.972x diagnostic | 0.010294 | PASS 5/5 oracle | N/A |

The small-case noise floor is large because the point latency is below one
millisecond, but the gains remain far outside the corresponding A/A drift. Case
8 is the narrowest accepted speedup and has a tight 0.82% noise floor.

## Three-Reference Regression Check

| Case | Branch-base candidate ms | Latest candidate ms | Latest delta | Accuracy regression |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.6985 | 0.6975 | -0.15% | none |
| 2 | 0.1587 | 0.1609 | +1.40% | none |
| 3 | 0.1673 | 0.1643 | -1.82% | none |
| 4 | 0.2622 | 0.2678 | +2.14% | none |
| 5 | 1.2671 | 1.2513 | -1.25% | none |
| 6 | 210.2879 | 210.5723 | +0.14% | none |
| 7 | 0.4093 | 0.4060 | -0.82% | none |
| 8 | 15.6273 | 15.5889 | -0.25% | none |
| 9 | 0.6836 | 0.6881 | +0.65% | none |
| 10 | 0.6414 | 0.6464 | +0.78% | none |
| 11 | 1.3725 | 1.3724 | -0.01% | none |
| 12 | 0.2361 | 0.2335 | -1.09% | none |
| 13 | 13.5784 | 13.6078 | +0.22% | none |
| 14 | FP32 unsupported | 67.0945 s proxy | newly supported | none |

The branch base is `3e6a3ea32b838dfac17281018f6379ec69094590`,
committed `2026-08-31T00:11:10+08:00`. The latest result is
`775c82004fd31d5a2203619f671ee214f444411c`, committed eight seconds
later after rebasing the Case-14 implementation onto the research checkpoint.

## Memory Results

| Case | Baseline peak MiB | Candidate peak MiB | Change |
| ---: | ---: | ---: | ---: |
| 1 | 103.082 | 75.051 | -27.2% |
| 2 | 36.878 | 67.924 | +84.2% |
| 3 | 40.066 | 68.299 | +70.5% |
| 4 | 52.064 | 69.045 | +32.6% |
| 5 | 171.105 | 83.059 | -51.5% |
| 6 | 10,672.719 | 2,312.512 | -78.3% |
| 7 | 73.270 | 66.238 | -9.6% |
| 8 | 544.369 | 416.338 | -23.5% |
| 9 | 79.082 | 75.051 | -5.1% |
| 10 | 87.082 | 75.051 | -13.8% |
| 11 | 199.082 | 75.051 | -62.3% |
| 12 | 46.050 | 69.045 | +49.9% |
| 13 | 2,372.229 | 227.104 | -90.4% |
| 14 | dense unavailable | 3,643.988 combined proxy run | N/A |

Small-case absolute peaks include resident compiler caches and are not a
standalone model-footprint measurement. The material risks are Cases 6, 13,
and 14; their memory-safe routes are decisive.

## Optimization Decisions

### Compiled float32 SDPA

Cases 1, 4, 5, 7-12 replace explicit score materialization with strided-view
SDPA and use `reduce-overhead` compilation. Case 13 uses default compilation,
which performs better for its larger compute/memory-bound graph. The dispatcher
caches compiled callables only after an initial replay and invalidates them on
weight or device changes.

### Packed QKV

Cases 2 and 3 combine three projection GEMMs into one packed projection. The
packing is rebuilt outside timed execution, preserves the reference parameter
names, and passed a 60-trial scale/padding stress gate before integration. It is
not enabled on Case 8 because its end-to-end screen was neutral.

### Mask and layout elimination

SDPA consumes strided head views, removing avoidable `.contiguous()` copies.
Under causal attention and right padding, the padding-key mask is redundant for
every valid query and is removed after a cached prefix-mask classification.

### Extreme Case 6

Case 6 streams the huge batch through a bounded float32 SDPA path. This avoids
the dense attention peak while retaining the ordinary model semantics. The
final paired run is 2.394x faster and reduces measured peak allocation by 78.3%.

### Extreme Case 14

Case 14 streams one valid prefix at a time, retains FP32 parameters and an FP32
interface, and computes internally in FP16. Attention selects the polynomial
Triton kernel only when its sampled score standard deviation stays below the
validated `0.40` ceiling; final-run layer values are approximately `0.3340` and
`0.3345`. Otherwise it forces exact Flash SDPA rather than permitting a
quadratic fallback.

The FP16 result is shape-specific. Controlled factor tests found that FP16
attention alone passes the shorter cases when their residual stream remains
FP32, but repeatedly rounding the full residual stream across four layers
creates rare failures around near-zero outputs. Case 14 has only two layers and
passes both exact Flash and polynomial routes. Reduced precision is therefore
not enabled universally.

## Reproduction Commands

Direct cases use the same template with `CASE=1` through `13`:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case CASE --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --timing paired --settle-seconds 10 \
  --output research/benchmarks/DATE-GPU-COMMIT/result.json
```

Case 14:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 14 --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --timing paired --settle-seconds 10
```

The exact commands, timestamps, Git status, environment, raw latency samples,
accuracy trials, memory snapshots, and output fingerprints are in the
[latest benchmark directory](../benchmarks/2026-08-31-rtx5080-775c820/README.md).

## AI-Assisted Development

Codex agents were used as engineering collaborators for repository inspection,
task decomposition, source and profiler analysis, hypothesis generation,
implementation, conflict review, benchmark orchestration, and technical
documentation. The workflow deliberately treated AI suggestions as hypotheses:
each optimization was compared against the immutable baseline, its branch base,
and the latest implementation, with negative results preserved rather than
silently discarded.

Examples of evidence-driven corrections include rejecting universal packed QKV,
rejecting Case-14 kernel configurations tuned for the wrong GPU architecture,
distinguishing attention-core speedup from full-model performance, and narrowing
the original blanket FP16 conclusion after controlled residual/layer-count
experiments.

## Team Contributions

- Person 1: benchmark infrastructure, compilation strategy, final dispatcher,
  integration, and full-matrix validation.
- Person 2: attention/softmax decomposition, SDPA selection, Case-14 polynomial
  attention research, kernel design, and numerical guard calibration.
- Person 3: projection/FFN profiling and packed-QKV implementation for Cases 2
  and 3.
- Person 4: extreme-shape memory model, batch/prefix streaming, Case-6 route,
  and Case-14 tractability work.

The authoritative role split and handoffs are preserved in
[`four-way-team-split.md`](../team-coordination/four-way-team-split.md).

## Limitations and Future Work

- Case 14 cannot be timed against the immutable dense implementation. Its
  oracle is numerically validated at a tractable length, but the 8.972x ratio is
  not an official dense-baseline score.
- Timing excludes compilation and cold-start latency. A long-lived inference
  workload benefits most from the compiled routes.
- Final timing covers the default zero-padding/input-scale contract. Separate
  stress evidence exists, but not every combination is part of this final
  matrix.
- Small cases are sensitive to launch scheduling and boost state. Their large
  speedups are directionally robust, but exact point estimates may move.
- FP16 whole-model execution is rejected for the four-layer shapes. A custom
  fusion that preserves an FP32 residual stream without repeated cast kernels
  is the clearest remaining mixed-precision opportunity.
- Case 8 remains projection/FFN-bound and has the smallest accepted gain.
- Performance portability beyond the RTX 5080 and pinned PyTorch/CUDA build
  requires a fresh correctness and timing gate.
