# Collaboration and Benchmark Infrastructure

This directory lets each optimization stream develop a full-model candidate in
isolation and compare it with the immutable root
[`torch_transformer_benchmark.py`](../torch_transformer_benchmark.py). The
harness imports the root benchmark's configuration, model, random-case generator,
exact absolute-error **OR** relative-error checker, warmup, and timing functions.

The 14 disclosed cases from the organizer Appendix screenshot
[`task_shapes.png`](../task_shapes.png) are transcribed into the validated
[`cases/task_shapes.json`](cases/task_shapes.json) manifest. The image remains the
human-readable source; the JSON is its executable representation.

## Ownership

| Stream | Candidate selector | Primary file |
| --- | --- | --- |
| Framework fast paths and integration | `compiler` | `implementations/compiler.py` |
| Attention and softmax | `attention` | `implementations/attention.py` |
| Projections, FFN, normalization, fusion | `projections` | `implementations/projections.py` |
| Extreme-shape memory strategy | `extreme` | `implementations/extreme.py` |
| Accepted integrated backends | `src.dispatcher` | `dispatcher.py` |
| Infrastructure smoke candidate | `dummy` | `implementations/dummy.py` |

Each scaffold initially executes the reference implementation, so candidate
loading and correctness work before any optimization lands. Contributors should
change only their owned module and any uniquely named helper modules. They do not
need to edit a central registry.

## Environment

The verified local environment uses Python 3.12, NumPy 2.5.2, PyTorch
2.13.0+cu130, Triton 3.7.1, and the RTX 5080. Reproduce it from the repository
root with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Debian or Ubuntu, `python3 -m venv` normally requires the matching
`python3-venv` OS package. This workspace lacked that package and sudo access, so
its local venv was created with `--without-pip` and bootstrapped using PyPA's
official `get-pip.py`. Teammates with normal host administration should install
the OS package instead.

Confirm the environment before collecting GPU numbers:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())'
```

Run the dependency-free contract tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s src/tests -v
.venv/bin/python -m compileall -q src
```

The full benchmark requires PyTorch. A small CPU smoke test is:

```bash
python3 -m src.benchmark \
  --candidate reference \
  --device cpu \
  --batch-size 1 --seq-len 8 --d-model 16 --heads 4 \
  --ffn-dim 32 --layers 1 \
  --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1
```

Run an optimization on the target GPU by changing the selector and supplying the
official case number. For example:

```bash
python3 -m src.benchmark \
  --candidate attention \
  --device cuda --dtype float16 \
  --case 13
```

`--case 1` through `--case 14` fix batch size, model/QKV dimension, head count,
sequence length, layer count, causal mode, and FFN dimension from the manifest.
Conflicting explicit shape flags are rejected rather than silently overriding the
official case. Explicit shape flags remain available for exploratory cases.

The harness refuses to benchmark a numerically incorrect candidate unless
`--benchmark-on-failure` is supplied. `--compile-user`, `--compile-baseline`,
`--compile-mode`, masking, causal mode, TF32, tolerances, warmup, and repeat flags
match the root benchmark's behavior and defaults.

### Smart dispatcher

The integrated selector is `src.dispatcher`. It lazily compiles its own callable
after the harness moves weights to their final device and dtype. The harness
rejects `--compile-user` for this self-compiling candidate rather than nesting
two compiler wrappers:

```bash
.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case 2 --device cuda --dtype float32
```

On CUDA GPUs with compute capability 8.0 or newer, under the PyTorch
2.13.0+cu130, float32, high-matmul-precision, TF32-enabled contract, official
cases 1-5 and 7-12 use strided-view SDPA inside `reduce-overhead`; case 13 uses
the same SDPA path with ordinary/default compilation. Performance evidence is
currently specific to the RTX 5080; other eligible GPUs may produce different
speedups. The complete official tuple and runtime contract are matched before
selecting a route. Non-float32 inputs, CPU, older/unknown CUDA capabilities,
other software contracts, non-official configurations, mismatched runtime
shapes, unavailable compilation, and compiler failures use exact reference
arithmetic.

Cases 6 and 14 are different: dense reference execution is itself unsafe at
their extreme sizes, so the harness rejects them before model or input
allocation until a memory-safe backend exists. Compiled callables are exercised
through initial compilation and one replay before caching; cached-call failures
demote that runtime key to reference. Caches remain per model instance and are
invalidated if parameters are loaded, moved, or converted.

Case 8 currently retains the reference projection/FFN layout around SDPA. Its
planned packed-QKV specialization remains a separate integration step because
the Person 3 packed-QKV implementation is not yet available.

### End-to-end dummy check

The `dummy` candidate runs the exact reference path, then performs a bounded and
discarded diagnostic workload containing a GEMM, repeated sine/cosine operations,
and softmax before cloning its output. The returned clone is bitwise exact, so
correctness should pass, while the candidate trace has a clearly labelled
`dummy_extra_work` region and several additional GPU kernels. This makes candidate
loading, weight copying, correctness, timing, and trace visualization
independently observable:

```bash
python3 -m src.benchmark \
  --candidate dummy --case 2 \
  --device cuda --dtype float16 \
  --accuracy-trials 2 --warmup 5 --repeats 20 --benchmark-rounds 2
```

Do not use `dummy` as an optimization baseline or with `--compile-user`; it is
only an eager-mode infrastructure fixture, and a compiler may remove deliberately
discarded work.

## Visual Profiling

PyTorch Profiler support is built into PyTorch, so no separate visualization
library is required to collect a trace. Generate separate baseline and candidate
traces with:

```bash
python3 -m src.profile \
  --candidate dummy --case 2 \
  --device cuda --dtype float16
```

By default, each run creates a unique timestamped directory under the Git-ignored
`artifacts/profiles/` directory in the repository. Pass `--output-dir PATH` to
override it. The output directory is created exclusively and contains:

- `baseline-trace.json` and `candidate-trace.json`, which can be opened in
  [Perfetto](https://ui.perfetto.dev);
- operator tables sorted by self CUDA or CPU time; and
- `metadata.json` with the command, Git state, environment, candidate, and shape.

For a reusable graphical overview, import
[`perfetto/transformer_profile_dashboard.json`](perfetto/transformer_profile_dashboard.json)
from Perfetto's **Data Explorer**. It provides scorecards, grouped kernel bars,
a duration histogram, a launch scatter plot, and a CDF. See
[`perfetto/README.md`](perfetto/README.md) for the import workflow.

Add `--with-stack` for source stacks or `--with-flops` for supported operator
FLOP estimates. Both add overhead. Traces can be large, so keep them outside Git
and link important ones from a preserved benchmark record as required by
`AGENTS.md`.

For an NVIDIA Nsight Systems timeline, use the regular benchmark's lightweight
NVTX mode instead of nesting Nsight with PyTorch Profiler:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --output=/tmp/techjam-case2-dummy \
  python3 -m src.benchmark \
    --candidate dummy --case 2 \
    --device cuda --dtype float16 \
    --accuracy-trials 1 --warmup 5 --repeats 2 --benchmark-rounds 1 \
    --nvtx
```

`--nvtx` labels baseline and candidate timed rounds in the Nsight timeline. Do
not use profiler-instrumented latency as a competition result; collect official
latency with the ordinary benchmark command and profiling disabled.

## Preserving a Run

Do not preserve every exploratory run. For a baseline, accepted optimization
checkpoint, regression, or final run, pass a new JSON path under the repository's
required directory convention:

```bash
python3 -m src.benchmark \
  --candidate attention \
  --device cuda --dtype float16 \
  --case 13 \
  --output research/benchmarks/2026-08-29-GPU-COMMIT/attention.json
```

Replace `GPU` and `COMMIT` with the actual GPU identifier and Git commit. The
result includes the exact command, timestamp, Git state, shape, dtype, thresholds,
per-trial correctness details, raw latency samples, speedup, CPU, GPU, OS, driver,
CUDA, PyTorch, and disk information. Output uses exclusive creation: an existing
record is never overwritten. Update `research/benchmarks/README.md` when
preserving a run, including invalidation or supersession links required by
`AGENTS.md`.

## Adding a Candidate

Add a uniquely named module under `implementations/` with a `CANDIDATE` export:

```python
from torch_transformer_benchmark import BaselineTransformer
from src.infra import CandidateSpec


class MyCandidate(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        return super().forward(x, valid_token_mask)


CANDIDATE = CandidateSpec(
    name="my-candidate",
    model_factory=MyCandidate,
    owner="name",
    description="Optimization hypothesis.",
)
```

Run it with `--candidate my_module`. A fully qualified
`module.path[:attribute]` is also accepted. Candidates that retain reference
parameter names receive a strict `state_dict` copy. If parameter packing changes,
provide a `weight_loader(baseline, candidate, strict)` in the `CandidateSpec` and
document why it remains a fair comparison.

## Branch and Integration Flow

After this infrastructure branch is merged, each contributor should create a
separate optimization branch from the latest `master`, following the sync and
incoming-commit review procedure in `AGENTS.md`. Keep exploratory results local;
commit focused candidate checkpoints and only the benchmark records worth
preserving.

An optimization is ready for integration only when it:

1. passes the executable correctness criterion on its claimed shape family;
2. includes a repeatable command and target-GPU result record;
3. shows a meaningful whole-model gain rather than only a microbenchmark gain;
4. documents supported dtype, causal and padding behavior, fallback, and memory
   assumptions; and
5. can be selected centrally from `dispatcher.py` without scattering shape checks.

The integration MR should report the alternatives, expected performance impact,
risks, correctness evidence, benchmark environment, and verification commands
required by `AGENTS.md`.
