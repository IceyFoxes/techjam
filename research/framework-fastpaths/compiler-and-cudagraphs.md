# Compiler and CUDA Graph Analysis

## Status and Reference

- Date: 29 August 2026.
- Repository benchmark revision: `6bde871dd65051fcace36971b27a86771365ba1e`.
- PyTorch: 2.13.0+cu130, source revision
  `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Evidence: immutable benchmark analysis, installed PyTorch source inspection,
  official documentation, one `torch._dynamo.explain` capture, and GPU logging
  with `TORCH_LOGS=graph_breaks,recompiles,perf_hints`.

## What Each Compile Mode Actually Changes

The installed `torch._inductor.list_mode_options()` reports:

| Mode | Inductor changes | Expected use here |
| --- | --- | --- |
| `default` | `{}` | General fusion without forcing CUDA Graphs or GEMM autotuning |
| `reduce-overhead` | `triton.cudagraphs=True` | Static, launch-bound shapes |
| `max-autotune` | `max_autotune=True`, `coordinate_descent_tuning=True`, `triton.cudagraphs=True` | Only a shape-specific candidate after accuracy validation |

This matters because `max-autotune` is not simply a stronger
`reduce-overhead`: it may select a different GEMM implementation and therefore a
different numerical result. Case 8 demonstrates that distinction executably.

## Graph Capture Result

`torch._dynamo.explain` on the exact case 2 model and all-true boolean mask
reported:

```text
graph_count       1
graph_break_count 0
op_count          91
ops_per_graph     [91]
break_reasons     []
```

The GPU diagnostic likewise logged no graph break. It logged one recompile of
`BaselineTransformer.forward` because tensor `x` changed dispatch-key set from
`CUDA, BackendSelect` to `CUDA, BackendSelect, ADInplaceOrView`. This occurred
during the benchmark lifecycle and did not recur during timed samples. The
implementation must nevertheless compile and warm up under the same
`torch.inference_mode()` context used for timing; a compile cache warmed under a
different autograd context must not be assumed reusable.

## Why Static Per-Case Compilation Fits This Task

Every disclosed case fixes batch, sequence, model width, head count, layer count,
causal mode, and FFN width. The benchmark constructs a new model for one fixed
configuration and times repeated calls with the same input addresses and shapes.
That is unusually favorable to specialization and CUDA Graph replay.

The dispatcher should not compile one dynamic graph for all 14 cases. PyTorch
assumes static shapes initially, and a guard failure may trigger recompilation.
More importantly, the fixed table lets us use a better strategy: construct one
specialized implementation for the complete tuple and cache its compiled callable
after the module has been moved to its final device and dtype.

Padding does not require a new shape when only mask values change. An exploratory
case 1 float32 run at `padding_ratio=0.25` passed five seeds and measured 2.775x
with `reduce-overhead`. However, any future optimization that branches on
`bool(valid_token_mask.all())` adds data-dependent host behavior and must be kept
outside the captured region or represented as a separate preselected route.

## CUDA Graph Benefits and Constraints

CUDA Graphs reduce CPU launch cost by recording a fixed operation graph and
replaying it with a single launch. This matches cases 2, 3, and 12, where many
small kernels make host launch overhead material. It does little for case 13,
where a single forward already takes tens of milliseconds, and only about 9.5%
for wide case 8 in the preserved run.

The constraints are material:

- operation topology, shapes, kernel launch parameters, and referenced addresses
  must be stable;
- graph workspace is cached, increasing retained device memory;
- output lifetimes must respect CUDA Graph Tree semantics, or a subsequent replay
  may overwrite an earlier output;
- data-dependent host control flow cannot live inside the graph; and
- dynamic or extreme cases may be better served by eager or chunked execution.

The current benchmark consumes each candidate output before the next candidate
trial and reuses fixed shapes, so replay is suitable. A reusable demo or API that
retains outputs across invocations must validate PyTorch's automatic iteration
heuristic and, if required, call `torch.compiler.cudagraph_mark_step_begin()` at
the outer request boundary or clone the retained output. Neither action should be
inserted blindly into the hot inner graph.

## Benchmark Lifecycle Consequence

`src.benchmark` runs correctness before performance. `torch.compile` is lazy, so
compilation and CUDA Graph recording occur during correctness/warmup, outside the
timed samples. This matches the competition's latency objective but means the
research numbers do not include cold-start latency. The final report must state
that distinction.

The immutable root flag `--compile-user` defaults to false. Therefore measured
gains cannot depend on an evaluator silently enabling it. The implementation
branch must either:

1. establish that the official command enables compilation; or
2. create a lazy compiled callable inside the candidate after device/dtype
   placement, with eager fallback if compilation is unavailable.

This evaluator contract is an open integration question, not a reason to discard
the measured fast path.

## Diagnostic Commands

```bash
TORCH_LOGS=graph_breaks,recompiles,perf_hints \
  .venv/bin/python -m src.benchmark \
  --candidate compiler --case 2 --device cuda --dtype float32 \
  --compile-user --compile-mode default
```

For a small model, use `torch._dynamo.explain(model)(x, mask)` to count graphs,
operators, and break reasons. For integration, use `TORCH_LOGS=recompiles,guards`
across multiple invocations and `TORCH_LOGS=perf_hints` to confirm whether CUDA
Graph capture was skipped.

## Sources

All public sources were accessed 29 August 2026.

- **PyTorch 2.13, `torch.compile` documentation.**
  <https://docs.pytorch.org/docs/stable/generated/torch.compile>. Describes the
  three modes, static/dynamic behavior, graph-break behavior, guard-based
  recompilation, CUDA Graph memory tradeoff, and mode-specific intent.
- **PyTorch 2.13, `torch/_inductor/__init__.py`, tag `v2.13.0`, revision
  `cf30153c4c131c8164ee7798e5022d810682e2cb`, symbol `list_mode_options`.**
  <https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/__init__.py>.
  Defines the exact configuration patches summarized in the mode table.
- **PyTorch 2.13, `torch/_inductor/config.py`, same tag and revision, symbols
  `emulate_precision_casts`, `shape_padding`, `max_autotune`, and
  `triton.cudagraphs`.**
  <https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/config.py>.
  Documents eager precision-checkpoint emulation and compiler controls relevant
  to the measured numerical failures.
- **PyTorch 2.13, Dynamic Shapes.**
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html>.
  Documents initial static specialization, automatic dynamism after a recompile,
  and why forced `dynamic=True` is not the preferred first choice.
- **PyTorch 2.13, CUDA Graph Trees.**
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>.
  Documents shared graph memory pools, output lifetime hazards, and
  `cudagraph_mark_step_begin()`.
- **PyTorch compiler troubleshooting.**
  <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html>.
  Documents `TORCH_LOGS` categories used here: `graph_breaks`, `guards`,
  `recompiles`, and `dynamic`.
- **NVIDIA CUDA Programming Guide, CUDA Graphs, release 13.2, section 4.2.**
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>.
  Explains graph definition/instantiation/replay, reduced launch overhead, fixed
  topology, and graph memory behavior.
- **Local immutable benchmark**, revision
  `6bde871dd65051fcace36971b27a86771365ba1e`, symbols
  `BaselineTransformer.forward`, `maybe_compile`, `run_accuracy_tests`, and
  `benchmark_models`. Accessed 29 August 2026. Defines the executable graph,
  correctness-before-timing lifecycle, and compile flags.
