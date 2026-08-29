# Major Breakdown and Research Directions

## Status

- Date: 29 August 2026.
- Stage: major directions decomposed and focused source validation completed;
  benchmark reproduction is next.
- Scope authority: [`TASK.md`](../../TASK.md),
  [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py), and
  [`MY_ROLE.md`](../../MY_ROLE.md).

## Problem Decomposition

The work should not be divided only by PyTorch operator names. A useful research
direction must connect five layers of reasoning:

1. **Mathematical invariants:** which results and operation order the executable
   reference requires.
2. **Data movement:** which materialized tensors, layout conversions, and global
   memory round trips are avoidable.
3. **Execution shape:** whether a case is limited by launches, bandwidth, dense
   tensor-core throughput, or capacity.
4. **Numerical checkpoints:** where the reference rounds to the model dtype and
   where it deliberately reduces in FP32.
5. **Integration contract:** whether an optimization helps the complete
   projection-attention-FFN path or moves work into another team member's module.

An implementation is interesting only if it removes measurable work without
violating one of the other four layers.

## Mathematical Core

Flatten the token dimensions as `T = batch * sequence_length`. For all disclosed
cases, `ffn_dim = d_model = D`.

| Operation | GEMM shape `(M, N, K)` | FLOPs per layer |
| --- | --- | ---: |
| One Q, K, or V projection | `(T, D, D)` | `2*T*D^2` |
| Packed QKV projection | `(T, 3D, D)` | `6*T*D^2` |
| Output projection | `(T, D, D)` | `2*T*D^2` |
| Two FFN projections | two `(T, D, D)` | `4*T*D^2` |
| All Person 3 dense work | six equivalent square projections | `12*T*D^2` |

Packing QKV does not reduce arithmetic. It can remove two launches, repeated
input reads, allocator work, and downstream layout copies. Fusion of GELU,
residual, masking, or normalization can remove intermediate writes and reads but
may also remove model-dtype rounding points that are observable under the
executable reference.

## Major Breakdown A: Dense Projection Mainloops

### Question

Can the dense arithmetic be presented to vendor or generated kernels in a more
efficient shape without replacing a strong GEMM with a weaker custom one?

### Directions

- Persistent prepacked QKV through one `F.linear` or cuBLAS GEMM.
- Inductor `max-autotune` and template selection after explicit packing.
- cuBLASLt or CUTLASS epilogues when they remove a proven standalone kernel.
- Triton, CUTLASS, CuTe DSL, or persistent kernels only for a disclosed shape
  where library GEMM leaves a measured utilization gap.
- Grouped GEMM only if separate outputs or chunk scheduling make one wide GEMM
  unsuitable.

### Breakthrough Standard

A custom mainloop is not a breakthrough merely because it matches cuBLAS. It
must enable a useful epilogue or target layout and improve the complete producer
to consumer path.

## Major Breakdown B: Fusion Graph and Rounding Boundaries

### Exact graph

```text
x
 -> LayerNorm
 -> QKV projections
 -> attention
 -> output projection
 -> residual add
 -> LayerNorm
 -> FFN input projection
 -> exact GELU
 -> FFN output projection
 -> residual add
 -> padding mask
```

### Legal fusion families

- LayerNorm plus packed QKV producer.
- FFN input projection plus bias plus exact GELU epilogue.
- FFN output projection plus bias plus residual epilogue.
- Residual plus following LayerNorm.
- Padding-mask application folded into the final store of a producer.

### Constraints

- Attention is a dependency barrier between QKV and output projection.
- The second FFN GEMM depends on the complete first-GEMM activation.
- Exact GELU uses `approximate="none"`; a tanh epilogue changes the formula.
- Fusing a residual into an FP32 GEMM accumulator can skip the reference's
  model-dtype store and alter rounding.
- LayerNorm statistics require stable FP32 reduction behavior and `eps=1e-5`.

### Breakthrough Candidates

- Compiler precision-checkpoint emulation that preserves eager rounding while
  retaining fusion.
- A width-32 fused FFN that keeps its intermediate on chip.
- A fused residual-LayerNorm-to-projection pipeline that avoids materializing
  normalized activations without duplicating reduction work excessively.

## Major Breakdown C: Layout as an End-to-End Contract

### Question

Can one projection output layout serve the selected attention backend and permit
zero-copy output projection, rather than optimizing either side in isolation?

### Leading contract

```text
packed QKV storage: [B, S, 3, H, head_dim]
attention output:   physically [B, S, H, head_dim]
block boundary:     contiguous [B, S, D]
```

This is a hypothesis, not yet an accepted interface. PyTorch SDPA can receive
logical `[B,H,S,head_dim]` transpose views, while FlashAttention-style APIs often
accept sequence-major layouts. A custom attention kernel may instead require
physically contiguous head-major storage.

### Breakthrough Standard

Charge every adapter copy to the backend that requires it. A zero-copy producer
that makes attention slower is not an improvement. The decision metric is
projection plus layout plus attention plus output projection.

## Major Breakdown D: Shape Families

| Family | Representative case | Initial bottleneck hypothesis | Promising direction |
| --- | --- | --- | --- |
| Narrow and launch-sensitive | #7: `T=8192,D=32` | launches and elementwise traffic | aggressive graph fusion, packed QKV, possibly persistent full-FFN kernel |
| Wide and compute-heavy | #8: `T=8192,D=1024` | tensor-core GEMMs | packed QKV, vendor/generated epilogues, layout removal |
| Huge-token bandwidth/capacity | #6: `T=1,280,000,D=128` | activation bandwidth and live memory | packed or chunked QKV, fused stores, largest safe chunks |
| Extreme streaming | #14: `T=3,200,000,D=1024` | capacity before projection speed | chunk-local projection consumed by streaming attention |

These classifications are hypotheses until benchmarked. Results must record the
execution environment, but the fixed formula, disclosed shapes, and executable
correctness rule determine the research priorities.

## Major Breakdown E: Numerical Correctness

The executable checker accepts an element only when:

```text
absolute_error <= 0.002 OR error <= 0.02 * abs(reference)
```

The following are independent research variables:

- FP32 with TF32 enabled or disabled.
- FP16 and BF16 GEMM accumulation and reduced-precision reductions.
- exact versus approximate GELU.
- Welford versus stable two-pass LayerNorm.
- model-dtype rounding between a GEMM and a fused residual.
- different reduction trees caused by packed GEMM shapes.

The acceptance metric should include normalized tolerance headroom, not only a
binary pass. Large-output cases make rare per-element failures important.

## Industry and Mathematical Breakthrough Lens

Further research should prioritize recent techniques only when they apply to the
fixed formula and disclosed dtypes:

- **Generated GEMMs:** persistent scheduling, warp specialization, and custom
  epilogues are useful only when they beat the benchmark's library path for a
  disclosed shape.
- **Compiler precision preservation:** modern compilers can fuse graphs but may
  need explicit emulation of low-precision casts to match eager semantics.
- **Producer-consumer layout fusion:** packed QKV and attention-native layouts
  can remove linear traffic that remains even after Flash-style attention.
- **On-chip small-width pipelines:** width 32 may permit fusion across operations
  that are impossible at width 1024.
- **IO lower bounds:** use roofline and red-blue-pebble reasoning to reject ideas
  that do not reduce mainloop work, global traffic, synchronization, or launches.
- **Stable reduction mathematics:** Welford, pairwise reduction, and compensated
  strategies should be evaluated by both accuracy and hardware cost; unstable
  `E[x^2]-E[x]^2` variance is not acceptable.
- **Low-precision innovations:** FP8, FP4, and block-scaled GEMMs are relevant
  industry developments but are outside the declared input dtypes unless they
  can satisfy the executable output criterion without changing the contract.

Architectural substitutions such as RMSNorm, gated FFNs, sparse/MoE layers, or
linear attention change the fixed formula and are not optimization directions.

## Research Questions Before Implementation

1. Does one packed QKV GEMM beat three GEMMs after all required layout adapters?
2. Which attention backend accepts packed sequence-major views without a hidden
   copy, and what physical output layout does it return?
3. Does Inductor preserve model-dtype rounding at residual fusion boundaries?
4. Can exact GELU be fused without a slower GEMM mainloop or tolerance loss?
5. Is LayerNorm materialization a meaningful fraction of case #8, or only #7
   and #6?
6. Which ideas survive end-to-end measurement across the disclosed shape
   families rather than only improving an isolated kernel?
7. Can #6 and #14 consume QKV in chunks without making projection GEMMs too
   narrow or retaining all three projected tensors?

## Evidence and Stop Rules

Every direction must provide:

- an authoritative source or clearly labeled hypothesis;
- exact supported shapes, dtypes, strides, and masks;
- isolated latency and complete-path latency;
- kernel count, memory traffic or peak-memory evidence;
- correctness over multiple seeds, scales, and padding ratios;
- a rerun in the final evaluation environment before acceptance.

Stop when:

- a custom GEMM only matches a library path within noise;
- a removed copy causes a larger downstream regression;
- fusion fails correctness or lacks tolerance headroom;
- an optimization cannot improve whole-model latency;
- maintenance cost is disproportionate to a shape-specific gain.

## Focused Findings

The source-validation wave resolved the initial questions as follows:

- Packed contiguous `[B,S,3,H,d]` is the leading QKV contract, but it must beat
  three view-only linears over projection, attention, merge, and output projection.
- Fusion must preserve model-dtype stores after linear+bias, exact GELU, and
  residual addition for eager-like low-precision behavior.
- A fused width-32 FFN is feasible enough to benchmark, but only after its
  measured whole-model Amdahl ceiling justifies custom implementation.
- Hardware-specific schedules and DSLs are secondary choices; the benchmark
  graph, disclosed shapes, and executable correctness rule determine priority.

Detailed evidence and experiments are in [benchmark graph analysis](benchmark-graph.md),
[numerical fusion boundaries](numerical-fusion.md), [narrow FFN analysis](narrow-ffn.md),
and the [packed QKV layout contract](qkv-layout.md).

## Preliminary Sources

All public sources below were accessed on 29 August 2026. Focused leaf documents
will retain detailed source revisions and symbol references.

- NVIDIA, [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html):
  GEMM arithmetic intensity, tensor-core alignment, tiling, and wave
  quantization establish the dense-operation model.
- PyTorch 2.13,
  [`torch.compile`](https://docs.pytorch.org/docs/2.13/generated/torch.compile.html):
  documents max-autotuned GEMMs, epilogue fusion, shape padding, and CUDA-graph
  tradeoffs.
- PyTorch 2.13,
  [CUDA numerical semantics](https://docs.pytorch.org/docs/2.13/notes/cuda.html):
  documents TF32 and reduced-precision GEMM reductions that can affect
  executable equivalence.
- Triton,
  [matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html):
  demonstrates tiled GEMM, L2-aware scheduling, autotuning, FP32 accumulation,
  and epilogue activation fusion.
- Triton,
  [LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html):
  demonstrates FP32 two-pass statistics and fused affine stores.
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135):
  provides the IO-aware model motivating producer-consumer traffic analysis.
- Hong and Kung,
  [I/O Complexity: The Red-Blue Pebble Game](https://doi.org/10.1145/800076.802486):
  supplies the general lower-bound framework for reasoning about data movement
  rather than arithmetic alone.
