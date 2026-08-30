# Paper Follow-ups After Local Measurements

- Status: current research screen, 30 August 2026.
- Scope: techniques materially different from the completed packed-QKV, narrow
  FFN, output-residual, and cuBLASLt experiments.
- Constraint: all proposals must preserve affine LayerNorm, exact `erf` GELU,
  float32 checkpoints, masking, and the executable elementwise tolerance.

## Ranked Findings

### 1. ComFuse

- Source: [ComFuse: Fusing Complex Memory-Intensive Subgraphs with
  Compute-Intensive Kernels For Modern GPU Architectures](https://arxiv.org/abs/2608.03537),
  accessed 30 August 2026.
- Relevant result: its Stage-Stream model overlaps a post-GEMM reduction epilogue
  with GEMM and reports average 1.09x speedup for
  MatMul-BiasAdd-LayerNorm-ReLU. Its cluster-cooperative back-to-back GEMM model
  retains the first GEMM's tiles on chip and reports up to 1.23x for an MLP
  pattern.
- Benchmark mapping: first test `out_proj + bias + residual + norm2`; only then
  consider `ffn_in + exact GELU + ffn_out + residual`.
- Caveats: the paper's setup does not clearly identify its evaluation GPU, the
  method depends on cluster/TMA behavior that must be verified on consumer SM120,
  and it does not establish exact-`erf` GELU behavior.
- Decision: highest-priority new profiler-guided RTX 5080 experiment, but not an
  implementation candidate without target-GPU evidence.

### 2. Blockbuster

- Source: [Blockbuster: An End-to-End Transformer Layer Fusion Approach for
  GPUs](https://arxiv.org/abs/2505.07829), accessed 30 August 2026.
- Relevant result: algebraically combines LayerNorm statistics and a following
  matrix multiplication instead of materializing the normalized activation.
- Benchmark mapping: `norm1 -> packed QKV` is the strongest target because Q, K,
  and V share one normalized input; `norm2 -> ffn_in` is secondary.
- Caveats: no GPU performance result establishes a gain for these shapes, and
  reassociated variance/mean arithmetic may fail near-constant or large-offset
  inputs.
- Decision: profile normalized-activation traffic first; stop before kernel work
  if compilation already removes the write/read or if exact checkpoints cannot
  be retained.

### 3. Virtual Tensor and Layout Propagation

- Source: [VTC: A Virtual Tensor Compiler for Efficient Transformer
  Inference](https://arxiv.org/abs/2604.09558), accessed 30 August 2026.
- Relevant result: represents producer outputs virtually so consumers can use a
  compatible logical layout without physical transposes or copies.
- Benchmark mapping: packed QKV already applies the central idea through aliased
  strided views into SDPA.
- Decision: no new route unless profiling finds another physical layout movement
  after the accepted packed-QKV boundary.

### 4. Selective Precision Recalculation

- Source: [LAMP](https://arxiv.org/abs/2601.21623), accessed 30 August 2026.
- Relevant result: uses selective higher-precision recomputation for components
  that dominate numerical error instead of applying expensive arithmetic
  globally.
- Benchmark mapping: identify which of Case 8's 24 dense GEMMs first consumes
  the tolerance budget, then retain reference arithmetic only for that static
  operation/layer subset.
- Caveat: dynamic per-element correction is likely too expensive and
  input-dependent for this workload.
- Decision: a static per-operation attribution screen is justified; another
  global `max-autotune` attempt is not.

## Lower-Priority Evidence

- Source: [Mirage](https://arxiv.org/abs/2405.05751), accessed 30 August 2026.
  It demonstrates normalization-plus-matmul superoptimization, but its strongest
  published transformer example uses RMSNorm rather than the required affine
  LayerNorm. Use it as mechanism evidence, not a transferable result.
- Source: [FACT](https://arxiv.org/abs/2604.26666), accessed 30 August 2026. Its
  architecture-aware CUTLASS composition supports bounded native-template
  autotuning, but its A100/H100 results do not validate SM120 behavior.
- Source: [Nautilus](https://arxiv.org/abs/2604.14825), accessed 30 August 2026.
  Its RTX 5090 results support Blackwell-specific schedule search, but its main
  opportunity is attention and therefore outside Person 3's primary boundary.
- Source: [Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219), accessed
  30 August 2026. Persistent cross-operator scheduling primarily targets decode;
  this benchmark uses full fixed sequences and does not justify a megakernel.
- Source: [ByteTransformer](https://arxiv.org/abs/2210.03052), accessed 30 August
  2026. Padding-free execution and broad fusion are relevant in principle, but
  changing fixed dense shape behavior is risky and attention dominates the idea.
- Source: [LightSeq](https://arxiv.org/abs/2010.13887), accessed 30 August 2026.
  It establishes conventional transformer operator fusion, now covered more
  directly by the repository's compiler and targeted-kernel experiments.

## Source-Code Compatibility Audit

### CODA

- Sources: [CODA paper](https://arxiv.org/abs/2605.19269) and
  [CODA repository](https://github.com/open-lm-engine/coda-kernels), revision
  `8e90e8617b59f14adb0a8e78939944ef316c3660`, accessed 30 August 2026.
- The useful mechanism is concrete: `residual_sqsum_scaled_epi` keeps a GEMM
  accumulator tile live, adds the residual, emits the residual result, and
  performs a column-vector sum-of-squares reduction. This is the closest public
  implementation found to the required output-GEMM-plus-normalization boundary.
- The repository is not directly usable for this benchmark. Its README targets
  H100; `_compile_gemm` raises `NotImplementedError` unless the device major is
  9 even though the outer interface lists SM120; and the block API accepts only
  FP16/BF16. The supplied forward path implements RMSNorm and SwiGLU, not affine
  LayerNorm with mean subtraction or exact `erf` GELU.
- Decision: use the epilogue/reduction structure as a design reference only.
  The first implementation gate is a minimal SM120 FP32 GEMM-plus-row-reduction
  compile, not an attempted transplant of the full CODA block.

### ClusterFusion++

- Sources: [ClusterFusion++ paper](https://arxiv.org/abs/2604.23553) and
  [ClusterFusion++ repository](https://github.com/superk668/ClusterFusionPlus),
  revision `d020c217cc372bdb5be97984b7fa546f8c5f4a61`, accessed 30 August 2026.
- This is the strongest architecture-transfer evidence because the repository
  contains a dedicated RTX 5090 path using thread-block clusters, distributed
  shared memory, TMA, affine LayerNorm weights and biases, and a full
  projection-to-residual pipeline.
- It is nevertheless incompatible as a candidate. The Pythia kernel is a
  single-token FP16 decoder, accumulates output projection partitions through
  FP16 `atomicAdd`, computes variance as `E[x^2] - E[x]^2`, and uses a PTX tanh
  approximation for GELU. Its own correctness script characterizes bit-level
  non-determinism and validates generated-token agreement rather than this
  benchmark's elementwise tolerance.
- Decision: mine its SM120 cluster/TMA launch and synchronization patterns only.
  Do not port its arithmetic, decode scheduling, atomics, or GELU.

### Incoming Mask Work

- Upstream revision `bcb99ab` already classifies causal prefix masks once per
  runtime key and drops the redundant SDPA key mask. The associated RTX 4060
  matrix covers seeds, padding, and dropout behavior.
- Decision: remove all-true/prefix-mask specialization from Person 3's remaining
  list. It is completed upstream and is independent of projection layout.

## Quantitative Case-8 Ceiling

The measured Case-8 profile totals 62.424 ms. The eight existing fused
residual/mask/LayerNorm kernels consume 6.300 ms, so deleting their entire cost
without changing any GEMM gives an absolute Amdahl ceiling of:

```text
62.424 / (62.424 - 6.300) = 1.112x
```

That is an unattainable upper bound, not a projection. Assuming the 24 equal-shape
dense GEMMs have equal cost, the eight output/down projections consume about
`47.122 * 8 / 24 = 15.707 ms`; together with the reduction kernels, the target
boundary is about 22.007 ms. Applying ComFuse's reported 1.09x pattern result to
that whole boundary projects only about 1.030x whole-model speedup. A 5% whole
model gain would require approximately 1.156x on that boundary; a 10% gain would
require approximately 1.347x.

Therefore this route is worth one bounded SM120 feasibility experiment, but the
paper result alone does not support the original 10-15% Case-8 target. Reaching
that target requires either a materially faster GEMM schedule or fusion across
additional dense boundaries, not launch elimination alone.

## Bounded Execution Plan

1. Compile a minimal CUTLASS/CuTe SM120 FP32 GEMM epilogue that emits both the
   residual value and row `sum`/`sum_sq` partials. Stop if the available toolchain
   cannot compile SM120 reductions without modifying third-party internals.
2. Compare the GEMM output and residual checkpoint directly against the accepted
   Case-8 dispatcher before adding LayerNorm. Reuse the accepted GEMM arithmetic
   where possible; stop on any additional failed element in the stress matrix.
3. Implement affine LayerNorm as a staged row reduction with explicit mean and
   variance, not ClusterFusion++'s `E[x^2] - E[x]^2` shortcut. Test the isolated
   `out_proj + bias + residual + norm2` boundary first.
4. Require at least 1.156x isolated boundary speedup to retain the experiment;
   this is the approximate threshold for a 5% whole-model gain under the current
   profile. Do not proceed to the FFN-down boundary below that threshold.
5. If retained, repeat the exact executable stress matrix, then preserve an RTX
   5080 whole-model record. Promotion still requires no new failures and a
   measured whole-model gain of at least 5%.

Search terms for a replacement implementation if the CODA spike fails:
`CUTLASS SM120 epilogue row reduction`, `CuTe DSL Blackwell EVT reduction`,
`GEMM residual affine LayerNorm fusion FP32`, `post-GEMM Welford LayerNorm`,
`Blackwell thread-block cluster row reduction`, and
`CUTLASS Epilogue Visitor Tree LayerNorm`.

## Experiment Gates

1. Preserve the current dispatcher as the control and profile the exact Case 8
   `out_proj -> residual -> norm2` boundary on RTX 5080.
2. Attempt the post-GEMM LayerNorm route before back-to-back FFN GEMMs.
3. Require all seeds 1234-1243 at scales 0.125, 1, and 8 and padding ratios 0 and
   0.25 to pass before timing.
4. Stop if SM120 lacks the required cluster path, exact LayerNorm checkpoints
   cannot be retained, or the isolated boundary gain is below 1.156x.
5. Preserve a whole-model RTX 5080 record before dispatcher promotion and require
   at least a 5% gain with no additional failed elements.
