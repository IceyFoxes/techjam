# Industry Techniques Against the Benchmark

## Status

- Date: 29 August 2026.
- Purpose: select research directions before launching focused source studies.
- Scope: techniques that could improve the fixed Transformer formula and 14
  disclosed shapes while retaining the benchmark's dtypes and numerical rule.

## Benchmark-First Filter

`TASK.md` and `torch_transformer_benchmark.py` are the optimization authority.
The decisive questions are whether a technique preserves the exact graph, passes
the per-element absolute-or-relative error rule, and lowers end-to-end latency on
the disclosed shapes. Hardware-specific techniques are implementation options,
not the organizing principle, and cannot be ranked without benchmark evidence.

## Frontier Map

| Direction | August 2026 state | Relevance | Main limitation |
| --- | --- | --- | --- |
| CUTLASS 4.8 / CuTe DSL | Generated GEMMs, custom epilogues, and back-to-back GEMM | Potentially high for #8 and exact epilogues | Integration cost; examples may not support the execution environment |
| Triton persistent GEMM | Persistent, TMA, warp-specialized, and CLC variants with FP16 support | Medium-high for shape-specialized experiments | Official tutorial warns of shared-memory limits and shows persistent can trail cuBLAS |
| PyTorch 2.13 Inductor | Max-autotuned GEMMs, epilogue fusion, precision-cast emulation controls | Highest first step | Fusion can alter rounding or fail to select the target layout |
| TileLang 0.1.13/main | Persistent scheduling, producer-consumer specialization, hierarchical reductions | Medium as rapid research vehicle | Young, fast-moving dependency; integration and correctness burden |
| ThunderKittens 2.0 | Tile primitives, load-compute-finish pipelines, persistent grids | Medium for a narrow megakernel prototype | C++20 integration and kernel-by-kernel build |
| DeepGEMM | High-performance low-precision and megakernel designs | Conceptual only | Hardware restrictions and predominantly out-of-contract low precision |
| FP8/FP4 block scaling | Strong current industry focus | Low initially | Benchmark accepts FP32/FP16/BF16 input contracts; conversion and accumulated error may erase gains or fail correctness |

## Techniques Worth Transferring

### Persistent Tile Scheduling

Persistent grids cap the number of resident programs and let each program process
multiple output tiles. Potential benefits include lower scheduling overhead,
better load balance, and reuse of descriptors or state. This is most plausible
for narrow case #7 and shape-fixed routes. It is not inherently faster: Triton's
own example reports configurations where a conventional matmul or cuBLAS remains
ahead.

Research gate: compare against the best library kernel and include epilogue or
layout value. Do not retain persistence merely because it is newer.

### Warp-Specialized Producer-Consumer Pipelines

Recent CUTLASS, TileLang, Triton, and ThunderKittens work separates asynchronous
loads from tensor-core compute and stores. This can hide memory latency and make
complex epilogues less disruptive to the mainloop.

Research gate: determine whether exact GELU or target-layout stores fit without
reducing occupancy enough to lose to the benchmark's best library path.

### Back-to-Back GEMM

CUTLASS 4.8 lists back-to-back GEMM examples. This is relevant to the
two-layer FFN, but feasibility depends on intermediate width and tile ownership.
At `D=32`, a complete or large portion of an intermediate can remain on chip. At
`D=1024`, independent output tiles of the second GEMM depend on broad portions of
the first result, creating storage, recomputation, or inter-CTA communication
pressure.

Research gate: prioritize #7. Treat #8 as a mathematical/dataflow study before
writing a kernel.

### Custom Epilogues and Partial Reductions

CUTLASS 4.8 reports custom epilogue fusion with row/column partial reductions.
This may support exact bias, activation, residual, masking, or statistics-related
work while a high-quality GEMM mainloop remains intact.

Research gate: map each reference rounding checkpoint. Exact mathematical fusion
is insufficient if the result differs from eager model-dtype materialization.

### Generated Dense GEMM

CUTLASS and other generators include specialized dense GEMM schedules that
warrant inspection for #8's `(8192,1024,1024)` and packed
`(8192,3072,1024)` shapes.

Research gate: confirm supported dtypes/layouts, epilogue extensibility, PyTorch
integration overhead, and measured superiority over `F.linear`.

### Megakernel Lessons

DeepGEMM and ThunderKittens show an industry move toward kernels that keep
producer-consumer data on chip and overlap stages. The transferable lesson is
not to copy their MoE or low-precision formula. It is to look for shape-specific
pipelines where launch and intermediate traffic dominate, especially #7.

Research gate: preserve the exact dense FFN and GELU formula. Reject MoE, sparse,
SwiGLU, RMSNorm, or low-precision substitutions as direct implementations.

## Mathematical Filters

### I/O Benefit

An idea must reduce at least one of:

- global reads or writes;
- kernel launches or host scheduling;
- synchronization;
- repeated arithmetic;
- peak live storage.

Changing DSLs without improving one of these is not an optimization direction.

### Dependency Width

Fusion feasibility is controlled by dependency width, not source-code adjacency.
Bias, GELU, residual, and masking are output-local. LayerNorm is row-reduction
dependent. A second dense GEMM depends on a broad intermediate dimension. This
explains why epilogue fusion is routine, residual-LayerNorm fusion is plausible,
and complete two-GEMM fusion becomes difficult as `D` grows.

### Numerical Observability

The executable baseline observes model-dtype stores between operations. A kernel
that retains FP32 intermediates computes a more accurate real-number answer but
can be farther from the executable reference. Precision-cast emulation is
therefore a potentially important compiler feature, not merely a debugging aid.

## Focused Research Directions

The next research wave should answer four narrow questions:

1. Does packed QKV reduce total projection-to-attention latency for the disclosed
   shape families after all views or copies are charged?
2. Can compiler or custom fusion preserve explicit model-dtype rounding while
   removing intermediate global-memory traffic?
3. Can case #7 use a persistent or back-to-back pipeline to fuse most of the FFN
   without changing exact GELU?
4. Can packed QKV and attention-native storage remove layout traffic across the
   team boundary without disabling the fastest attention backend?

## Sources

All sources were accessed on 29 August 2026.

- NVIDIA, [CUTLASS 4.8 overview](https://docs.nvidia.com/cutlass/latest/overview.html),
  release 4.8.0 dated August 2026. Relevant sections: CuTe DSL additions,
  back-to-back GEMM, dense GEMM schedules, and custom epilogue partial
  reductions. Architecture-specific examples require separate compatibility
  checks and are not benchmark evidence.
- NVIDIA, [CUTLASS Python overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html),
  updated 27 August 2026. Summary: Python JIT kernel authoring with explicit
  layout, MMA, TMA, pipeline, warp-specialization, and framework integration;
  interfaces remain actively evolving.
- Triton,
  [persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html),
  accessed from current documentation. Symbols include `matmul_kernel_persistent`,
  `matmul_kernel_tma_persistent`, and `matmul_kernel_tma_clc`. Summary: compares
  conventional, persistent, TMA, warp-specialized, and CLC FP16/FP8 kernels with
  cuBLAS and documents capability and shared-memory constraints.
- TileLang, [repository](https://github.com/tile-ai/tilelang), revision
  `a1f8ebd4ed9caa4a579fec1eafe873be452a0860` (28 August 2026). Relevant project
  areas: persistent tile scheduler, producer-consumer warp specialization,
  hierarchical reductions, and CuTe DSL backend. Summary: rapidly evolving
  Python DSL for generated kernels.
- DeepSeek, [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), revision
  `559d79fb6994a58b8a15b4b93bf13ccc16edf247` (15 July 2026). Summary: JIT
  high-performance GEMM and megakernel designs; its restricted hardware and
  low-precision focus make it a design source rather than a direct backend.
- Hazy Research,
  [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), revision
  `be0e7e57e90858dfa2bbeab7296ff252755f8a37` (27 August 2026). Relevant symbols
  and areas: tile primitives, persistent-grid GEMM templates, and
  load-compute-finish pipelines. Summary: modern CUDA tile framework with
  individually compiled kernels.
- PyTorch, [`torch/_inductor/config.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/config.py),
  tag `v2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
  Relevant configuration families: epilogue fusion, GEMM autotuning,
  back-to-back GEMM passes, and precision-cast emulation.
