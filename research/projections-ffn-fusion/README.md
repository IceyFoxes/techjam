# Projections, FFN, Layout, and Elementwise Fusion

## Scope

This topic covers Person 3's ownership: Q/K/V and output projections, FFN,
LayerNorm, residual operations, and the layouts joining projections to attention.
The root [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py)
is the executable definition of the required computation and correctness behavior.

## Research Structure

- [Major breakdown and research directions](major-breakdown.md): mathematical
  invariants, bottleneck classes, legal optimization boundaries, breakthrough
  hypotheses, and the evidence required before implementation.
- [Industry techniques against the benchmark](industry-frontier.md): August 2026
  kernel techniques filtered by the fixed graph, shapes, and correctness rule.
- [Benchmark graph analysis](benchmark-graph.md): exact operations, shape-family
  costs, benchmark semantics, and ranked Person 3 opportunities.
- [Numerical fusion boundaries](numerical-fusion.md): eager precision
  checkpoints, fusion safety, and the correctness test matrix.
- [Narrow FFN analysis](narrow-ffn.md): case #7 limits, viable fusion scopes,
  and implementation kill criteria.
- [Packed QKV layout contract](qkv-layout.md): stride analysis, attention
  compatibility, and the shared Person 2/Person 3 interface.
- [Packed QKV cross-case exploration](packed-qkv-exploration.md): RTX 4050
  correctness, layout, profiler, and heuristic timing screen against the current
  SDPA plus strided-view route. Case #2 is the only implementation candidate to
  survive the compiled whole-model gate.
- [Feasibility gate: Case #14 memory analysis](feasibility-gate.md): original
  analysis, **stale in part as of 29 August 2026** because it counts a transpose
  view as an allocation, miscounts model weights, and overstates what baseline
  infeasibility proves.
- [Case #14 memory correction](case14-memory-correction.md): corrected storage
  accounting and the narrower conclusion supported by the supplied baseline.

## Current Status

**BLOCKER (29 Aug 2026, corrected):** The supplied explicit baseline materializes
`[B,H,S,S] = [32,16,100K,100K]`, whose score tensor alone requires 9.31 TiB in
FP16/BF16 or 18.63 TiB in FP32. This blocks the supplied baseline comparison on
the documented contemporary single-GPU environments. It does not prove that an
exact online, chunked, or streaming implementation of Case #14 is inherently
impossible; Person 4 owns that investigation.

Broad reconnaissance and focused source validation were completed on 29 August
2026 across packed QKV, FFN epilogues, residual-LayerNorm fusion, layouts,
precision, and shape modeling. Benchmark reproduction is next. No exploratory
timing is an accepted performance result until preserved under
`research/benchmarks/` with the required environment and correctness metadata.

The Wave-1 view-only QKV and packed-QKV experiments are stale standalone
controls, not accepted optimizations. Their historical measurements were neutral
or slower, but the stale timing method does not support precise regression
claims. The current `src/implementations/projections.py` implementation is also a
reference-equivalent eager control: spelling `nn.Linear` as `F.linear` does not
establish kernel fusion, and GELU and residual addition remain separate eager
operations. Its historical `1.001x` and `0.999x` measurements demonstrate no
accepted gain and miss this stream's 15% isolated-fusion threshold. Packed QKV
remains a future end-to-end layout experiment against the stronger current SDPA
plus strided-view route, especially for Case #8.

The follow-up cross-case screen rejects that Case #8 hypothesis on the measured
RTX 4050: packed QKV is numerically identical to the current SDPA route but its
heuristic whole-model ratio is only 1.003x. Case #2 instead shows repeatable
approximately 1.19x compiled ratios with zero failures over the 60-trial stress
matrix. This is exploratory implementation-selection evidence, not an accepted
benchmark result, because the timing interval remains statistically invalid and
the validated dispatcher environment is the RTX 5080.
