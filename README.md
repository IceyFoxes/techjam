# Shape-Aware Transformer GPU Kernel

This repository contains an AI-assisted, shape-aware PyTorch/Triton
implementation of the fixed Transformer benchmark in
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py). The final
dispatcher selects compiled FP32 SDPA, packed QKV projections, streamed
extreme-shape execution, or guarded polynomial attention according to the full
official model tuple and runtime contract.

## Final Result

On an NVIDIA GeForce RTX 5080, commit `775c820` passes every disclosed test
case under the executable `absolute error <= 0.002 OR relative error <= 2%`
criterion.

- Cases 1-13: **PASS 65/65 accuracy trials**, zero failed elements, and a
  **3.611x geometric-mean speedup** over the immutable dense reference.
- Case 14: **PASS 5/5 streamed FP32-oracle trials**, zero failures across
  16.384 billion output elements, with an **8.972x diagnostic oracle/candidate
  ratio** and 3.644 GiB peak allocation.
- Latest versus branch base: no accuracy regression and no candidate latency
  regression above 5%; all measured changes are within `-1.82%` to `+2.14%`.
- Test suite: 210 tests, including CUDA-aware dispatcher and kernel coverage.

Case 14's ratio is deliberately not folded into the geometric mean. Its
immutable dense FP32 reference would require multi-terabyte attention tensors;
the comparison uses a separately validated linear-memory FP32 oracle and labels
its timing as diagnostic.

See the [final technical report](research/final-submission/README.md) and
[preserved benchmark record](research/benchmarks/2026-08-31-rtx5080-775c820/README.md)
for the complete result table, machine specifications, commands, memory data,
three-reference comparison, methodology, and limitations.

## Implementation

- Cases 1, 4, 5, 7-12: strided-view float32 SDPA with shape-specific
  `torch.compile(mode="reduce-overhead")` routing.
- Cases 2 and 3: the same route plus packed QKV projection.
- Case 13: strided-view float32 SDPA with default compilation.
- Case 6: batch-streamed float32 SDPA, eliminating the dense path's extreme
  peak allocation.
- Case 14: sample/prefix streaming, FP32 interface and master state, FP16
  internal compute, forced-Flash exact fallback, and guarded Triton polynomial
  attention for the validated score distribution.
- All applicable routes remove the redundant padding-key mask under causal,
  right-padded attention.

The dispatcher is conservative: unrecognized shapes, unsupported software or
device contracts, unsafe dtypes, compiler failures, and out-of-range Case-14
score distributions fall back or reject before unsafe execution.

## Setup

The final environment used Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, and
Triton 3.7.1.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Confirm CUDA discovery:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
```

## Reproduce

Run any directly comparable official case:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 6 --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --repeats 100 \
  --timing paired --settle-seconds 10
```

Run the full Case-14 FP32-oracle comparison:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 14 --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234
```

Run the regression suite:

```bash
.venv/bin/python -m unittest discover -q src/tests
.venv/bin/python -m compileall -q src
```

## Limitations

- Performance is tuned and validated on the RTX 5080 with the pinned PyTorch
  build. Other eligible GPUs may select the same route but have different gains.
- Compilation is completed before timed steady-state execution; cold-start
  compilation is not included in latency.
- Final timing uses input scale 1.0, zero padding, TF32 enabled, and high float32
  matmul precision. Separate records cover padding and stress tests.
- The full dense Case-14 reference cannot execute. Its oracle is validated
  against the immutable dense model at `N=4096` before every target-scale run.
- Whole-model FP16 is not accepted for the four-layer cases. FP16 attention
  behind an FP32 residual stream passes, but its cast overhead did not improve
  their measured runtime.

## Team and AI Assistance

The work was divided across framework/dispatcher integration, attention
kernels, projection and FFN fusion, and extreme-shape memory strategy. The
role-level split is recorded in
[`four-way-team-split.md`](research/team-coordination/four-way-team-split.md).
Codex agents assisted with source inspection, experiment design, implementation,
profiling interpretation, A/B validation, conflict review, and documentation.
All promoted results were independently gated by the immutable executable
correctness rule and preserved benchmark evidence.
