# Person 3 Second-Pass Direction Screen

## Status

- Date: 29 August 2026.
- Scope: research-only screen after the packed-QKV and width-32 FFN decisions.
- Evidence: immutable benchmark graph, preserved compiled RTX 5080 whole-model
  records, exploratory RTX 4050 decisions, and official library/source review.
- No code was changed and no GPU timing was run for this screen.

## Starting Decisions

This pass accepts the measured decisions rather than reopening them:

- packed QKV materially survives only for Case #2;
- its compiled gain collapses for Cases #3, #4, and #12, Case #5 regresses,
  Cases #8 and #13 are neutral, and Cases #1 and #9-#11 are too small;
- Case #7 packing adds numerical failures, while its fused FFN improves the
  compiled whole model by only about 1.6% and still fails a low-scale trial; and
- Cases #6 and #14 require Person 4's memory schedule.

SDPA, causal-mask removal, packed QKV, and unspecified custom kernels are not
new Person 3 directions and are excluded below.

## Surviving Direction

### Output GEMM plus bias plus residual through `beta*C`

Both residual branches follow a square output GEMM:

```text
attention context -> out_proj + bias -> add block residual
GELU output       -> ffn_out + bias  -> add block residual
```

A cuBLASLt-style call can express `D = alpha*AB + beta*C`, and NVIDIA
Transformer Engine's `nvte_cublas_gemm_v2` exposes that operation together with
a bias epilogue configuration. This is materially different from packed QKV:
it keeps each `(T,D,D)` GEMM shape unchanged and removes the standalone GEMM
output consumed by the residual add.

For one FP32 activation `A = 4*T*D` bytes, eliminating the separate residual
add can avoid at most `2A` of extra traffic at each of two output GEMMs per
layer. Across four layers the static ceiling is `16A`. This is an upper bound;
epilogue constraints, cache hits, or a weaker GEMM schedule reduce it.

| Cases | `A` | Maximum four-layer traffic removed | Expected compiled whole-model impact | Decision |
| --- | ---: | ---: | --- | --- |
| #2 | 64 KiB | 1 MiB | Launch-scale only; CUDA Graph already dominates | Do not pursue beyond packed QKV |
| #3 | 256 KiB | 4 MiB | Below a material whole-model ceiling | Stop |
| #4, #12 | 1 MiB | 16 MiB | Low, but potentially visible against 0.207/0.180 ms controls | One bounded screen |
| #7 | 1 MiB | 16 MiB | Below the already rejected full-FFN ceiling; numerically fragile | Stop |
| #1, #9, #10, #11 | 4 MiB | 64 MiB | Highest relative opportunity for #1/#9/#10; #11 is more attention-heavy | One bounded screen for #1/#9/#10 only |
| #5 | 8 MiB | 128 MiB | Highest supported-case absolute opportunity | First screen |
| #8, #13 | 32 MiB | 512 MiB | GEMM- or attention-dominated; expected low-single-digit total impact | Stop |
| #6, #14 | 625 MiB / 12.21 GiB | Not actionable as full buffers | Person 4 only |

The implementation target, if tested later, is only Cases #5, #1, #9, #10,
#4, and #12, in that order. One mechanism should cover them; this does not
justify six shape-specific kernels.

The main risk is numerical ordering. The eager reference observes the biased
linear output before residual addition. An epilogue that adds the residual to
the accumulator before reproducing that checkpoint may differ, especially in
FP16/BF16. Therefore this is an FP32-first candidate, not a general dtype route.

## Lower-Ranked Directions And Kills

### 2. FC1 bias plus exact GELU epilogue: kill after paper screen

Transformer Engine has a concrete high-precision GEMM-GELU epilogue, so this is
not merely a generic custom-kernel suggestion. It can remove at most `2A` per
layer, or `8A` over four layers, half the output-residual ceiling. It also skips
the eager linear-output checkpoint, and the official interface does not promise
the exact PyTorch `approximate="none"` arithmetic sequence. The Case #7 full-FFN
experiment already gives a stronger Amdahl test than this narrower fusion.

Decision: do not implement independently. It may only be enabled incidentally
inside the same library experiment as the higher-ranked residual epilogue, and
must be disabled unless it adds at least 5% compiled whole-model improvement
after the residual-only result.

### 3. Residual plus LayerNorm or LayerNorm plus projection: stop

PyTorch 2.13 Inductor explicitly decomposes `aten.native_layer_norm` into
reduction and pointwise operations. The current whole model is already compiled
as one graph, so residual producers and LayerNorm consumers are visible to its
scheduler. A source-level rewrite does not establish a new optimization.

Transformer Engine's `LayerNormLinear` and `LayerNormMLP` names also do not imply
one high-precision kernel: their source calls `apply_normalization`, materializes
`ln_out`, and then calls `general_gemm`. They are useful module compositions,
not evidence that normalized activations avoid global memory in this contract.

Decision: stop unless a generated-kernel/profile audit proves a standalone
residual materialization remains. Even then, require the common experiment to
beat compiled Inductor by 5% whole model; do not write a separate LayerNorm
kernel first.

### 4. Concurrent or grouped separate Q/K/V GEMMs: stop

cuBLAS documents multi-stream overlap for independent small GEMMs, and
Transformer Engine exposes multi-tensor GEMM. This preserves the three original
GEMM shapes and could avoid Case #7's packed-GEMM reassociation. However, it
does not remove arithmetic or activation traffic, complicates CUDA Graph capture
and workspace ownership, and targets the launch/concurrency gap that disappeared
for Cases #3/#4/#12 after compilation. Case #2 already has the stronger packed
route.

Decision: stop for every supported case.

### 5. Alternate projection or attention layout: stop

The measured packed views already retain last-dimension stride one and showed no
Q/K/V or post-SDPA layout copies. Reordering weights to `[B,S,H,3,d]` or physical
head-major output adds setup or consumer constraints without removing a measured
copy. The current strided-view route is the strong control.

Decision: stop unless a profiler first identifies a new hidden copy in the final
dispatcher. Do not speculate from logical non-contiguity.

### 6. New dense mainloops or FP32 emulation: stop

A replacement mainloop that performs the same square GEMM has no dataflow win.
Case #8 `max-autotune` already failed correctness, and CUDA 13.3 documents
BF16x9 FP32 emulation only for compute capabilities 10.0 and 10.3, not the
preserved RTX 5080 compute capability 12.0 environment.

Decision: stop.

## Strict Experiment Gate

Only the output-residual epilogue may receive one short implementation screen,
after Case #2 is validated on the final machine.

Kill it immediately if any condition holds:

1. The selected API cannot express matrix product, original bias, and original
   residual in one GEMM epilogue without an extra full-tensor preparation pass.
2. The selected GEMM algorithm changes the `(T,D,D)` mainloop enough to regress
   isolated GEMM-plus-residual latency.
3. Any element fails the executable rule over the existing seeds, scales, and
   padding matrix; low-precision support additionally requires explicit eager
   checkpoint agreement.
4. Isolated GEMM-plus-residual improvement is below 15%.
5. Compiled whole-model improvement is below 5% on Case #5 and at least one of
   Cases #1/#9/#10, or the gain is inside the corrected run's uncertainty.
6. The route needs a new mandatory dependency or per-case implementation whose
   integration cost is disproportionate to the measured gain.

Cases #4 and #12 are tested only if the same accepted implementation is already
available; they do not justify implementation by themselves.

## Stop Matrix

| Family | Cases | Stop Person 3 research? | Reason |
| --- | --- | --- | --- |
| Tiny batch | #2 | **No, existing direction only** | Finish packed-QKV RTX 5080 validation; stop all second-pass ideas |
| Small/medium batch | #3 | **Yes** | Compiled ceiling is too small |
| Small/medium batch | #4, #12 | **Conditional** | Only free follow-up of the common residual-epilogue screen |
| Ordinary/head sweep | #1, #9, #10 | **No, one final screen** | Best relative residual-traffic opportunity after #5 |
| Ordinary large batch | #5 | **No, one final screen** | Highest supported-case residual-traffic opportunity |
| High-head ordinary | #11 | **Yes** | Same projection traffic as #1/#9/#10 but larger attention share |
| Narrow | #7 | **Yes** | Whole-model and correctness gates already reject stronger fusion |
| Wide | #8 | **Yes** | Packed neutral; epilogue ceiling is low single digit |
| Long sequence | #13 | **Yes** | Attention dominates; projection/layout changes are neutral |
| Extreme | #6, #14 | **Yes for Person 3 alone** | Resume only under Person 4's memory-safe schedule |

Thus Person 3 should stop after the Case #2 final-machine check and one shared
FP32 output-residual epilogue screen. If that screen misses any strict gate,
research stops for all supported case families.

## Sources

All public sources were accessed 29 August 2026.

- PyTorch repository, tag `v2.13.0`, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`,
  [`torch/_inductor/decomposition.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/decomposition.py),
  symbol `_native_layer_norm`. It shows that Inductor decomposes native
  LayerNorm, exposing its reduction and pointwise graph to compilation.
- NVIDIA Transformer Engine, tag `v2.8`,
  [`gemm.h`](https://github.com/NVIDIA/TransformerEngine/blob/v2.8/transformer_engine/common/include/transformer_engine/gemm.h),
  symbols `nvte_cublas_gemm_v2`, `kNVTEMatmulConfigBiasTensor`, and
  `kNVTEMatmulConfigWithGELUEpilogue`. It establishes the concrete
  `alpha*AB + beta*C`, bias, and GELU epilogue capabilities used in this screen.
- NVIDIA Transformer Engine, tag `v2.8`,
  [`layernorm_linear.py`](https://github.com/NVIDIA/TransformerEngine/blob/v2.8/transformer_engine/pytorch/module/layernorm_linear.py),
  `_LayerNormLinear.forward`. It calls normalization and GEMM separately and
  materializes the high-precision LayerNorm output.
- NVIDIA Transformer Engine, tag `v2.8`,
  [`layernorm_mlp.py`](https://github.com/NVIDIA/TransformerEngine/blob/v2.8/transformer_engine/pytorch/module/layernorm_mlp.py),
  `_LayerNormMLP.forward`. It documents and implements the full-precision
  GEMM-GELU epilogue while keeping LayerNorm and the two GEMMs as separate stages.
- NVIDIA cuBLAS 13.3 documentation,
  <https://docs.nvidia.com/cuda/cublas/index.html>. Sections 1.5, 2.1.4,
  2.1.6-2.1.7, and 3.3.2 document FP32 emulation support, multi-stream
  reproducibility/workspaces, small-GEMM concurrency, and cuBLASLt epilogues.
- NVIDIA CUTLASS documentation, updated 27 August 2026,
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#epilogue>.
  It establishes that elementwise operations can consume GEMM accumulators in
  the epilogue, while epilogue work can also affect the GEMM schedule.
- Local immutable benchmark and preserved RTX 5080 dispatcher records at
  implementation revision `307eedb2777c483befe7eadaecf1a7a9f5aff6be`, accessed
  29 August 2026. They supply the exact residual graph and compiled whole-model
  controls used for the ranking.
