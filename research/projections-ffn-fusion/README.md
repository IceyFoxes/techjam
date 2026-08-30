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
- [Second-pass direction screen](second-pass-after-measurements.md): official
  library/source search after the measured packed-QKV and narrow-FFN decisions.
  It leaves one bounded FP32 output-GEMM-plus-residual epilogue screen and gives
  explicit stop decisions for every official case family.
- [Remaining performance screen](remaining-performance-screen.md): current-dispatcher
  Case #8 attribution, selective GEMM autotuning, weight prepacking, residual
  epilogues, adversarial correctness, and CUDA Graph memory retention. It closes
  the local RTX 4050 routes while leaving selective RTX 5080 tuning open.
- [Direct cuBLASLt Case-8 probe](direct-cublaslt-probe.md): exhaustive explicit
  algorithm/configuration checks for `(8192,1024,1024)`, normal and prepacked
  layouts, and same-mainloop bias/residual epilogues. The required biased route
  has only a 1.010x projected whole-model ceiling and is rejected locally.
- [Paper follow-ups after local measurements](paper-followups-2026-08-30.md):
  benchmark-specific review of ComFuse, CODA, ClusterFusion++, Blockbuster, VTC,
  LAMP, and lower-priority fusion systems, with source-code compatibility,
  quantitative ceilings, numerical risks, and bounded RTX 5080 experiment gates.
- [Feasibility gate: Case #14 memory analysis](feasibility-gate.md): original
  analysis, **stale in part as of 29 August 2026** because it counts a transpose
  view as an allocation, miscounts model weights, and overstates what baseline
  infeasibility proves.
- [Case #14 memory correction](case14-memory-correction.md): corrected storage
  accounting and the narrower conclusion supported by the supplied baseline.

## Current Status

**UPDATE (29 Aug 2026):** The local bounded screen is complete. Case #8 is 75.49%
dense-GEMM time on the RTX 4050, but selective autotuning is neutral and
numerically invalid, pretransposed weights are neutral, and its separate FFN
bias/GELU kernel has only a 2.42% whole-model ceiling. A bundled Case #5 Triton
output-residual/mask/layout candidate improves TF32 latency by about 9% but fails
22,132 stress-matrix elements; TF32x3 and IEEE regress and still fail. See the
[remaining performance screen](remaining-performance-screen.md). Case #2 packed
QKV is the only integrated Person 3 implementation candidate. Exact selective
RTX 5080 tuning remains open.

**UPDATE (30 Aug 2026):** A direct cuBLASLt probe expanded 727 valid algorithm
configurations per normal/prepacked layout on the RTX 4050. The selected biased
mainloop improved by only 1.013x in 100 paired samples; residual-only improved
1.260x, but the mandatory bias-plus-residual form regressed to 0.955x. All 36
stress comparisons passed. The route is rejected before integration because its
relevant Amdahl ceiling is only about 1.010x whole model. RTX 5080 rerun remains
architecture-specific follow-up.

**UPDATE (30 Aug 2026):** Upstream RTX 5080 validation supersedes the local
Case-3 rejection for target-GPU routing. The
[`12a37c6 three-reference validation`](../benchmarks/2026-08-29-rtx5080-12a37c6/README.md)
measures packed QKV at 1.192x over strided Case 3 with a +/-0.84% direct paired
noise floor, PASS 60/60 stress trials, and bitwise packed-versus-strided outputs.
Universal packing remains rejected; Cases 2 and 3 are the accepted promotion
set. The paper follow-up identifies post-GEMM residual-plus-LayerNorm fusion as
the next materially different target, conditional on fresh SM120 profiling.

**UPDATE (30 Aug 2026):** Source inspection leaves CODA as the closest public
mechanism but not a reusable candidate: its current compiler path rejects SM120,
its block API is FP16/BF16, and its implemented normalization is RMSNorm.
ClusterFusion++ supplies RTX 5090 cluster/TMA examples but uses FP16 atomics,
single-pass variance, approximate GELU, and decode-only scheduling. The bounded
Case-8 experiment now starts with an SM120 FP32 epilogue-reduction compile and
requires about 1.156x on the targeted boundary to project a 5% whole-model gain.
Upstream causal key-mask elimination is already complete and is not remaining
Person 3 work.

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

**Superseded status (29 Aug 2026):** The Wave-1 view-only QKV and packed-QKV
experiments are stale standalone
controls, not accepted optimizations. Their historical measurements were neutral
or slower, but the stale timing method does not support precise regression
claims. The current `src/implementations/projections.py` implementation is also a
reference-equivalent eager control: spelling `nn.Linear` as `F.linear` does not
establish kernel fusion, and GELU and residual addition remain separate eager
operations. Its historical `1.001x` and `0.999x` measurements demonstrate no
accepted gain and miss this stream's 15% isolated-fusion threshold. Packed QKV
remains a future end-to-end layout experiment against the stronger current SDPA
plus strided-view route, especially for Case #8.

**Superseded implementation status (29 Aug 2026):** The follow-up cross-case
screen rejects that Case #8 hypothesis on the measured
RTX 4050: packed QKV is numerically identical to the current SDPA route but its
heuristic whole-model ratio is only 1.003x. Case #2 instead shows repeatable
approximately 1.19x compiled ratios with zero failures over the 60-trial stress
matrix. This is exploratory implementation-selection evidence, not an accepted
benchmark result, because the timing interval remains statistically invalid and
the validated dispatcher environment is the RTX 5080.

An RTX 5080 follow-up on 29 August 2026 adds hardware-specific contrary evidence
for Case #3 without changing the historical RTX 4050 conclusion. The
[`12a37c6 three-reference validation`](../benchmarks/2026-08-29-rtx5080-12a37c6/README.md)
measures packed QKV at 1.192x over the existing compiled Case-3 route with a
±0.84% direct paired noise floor, PASS 60/60 stress trials, and bitwise identity
to the strided control. Universal packing remains rejected; only Case #3 should
advance to an RTX-5080-specific dispatcher integration checkpoint.
