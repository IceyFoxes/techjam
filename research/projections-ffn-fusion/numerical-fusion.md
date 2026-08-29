# Numerical Fusion Boundaries

## Status

- Date: 29 August 2026.
- PyTorch source: tag `v2.13.0`, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Evidence type: benchmark and framework source analysis. Generated kernels still
  require executable validation.

## Executable Rule

For every output element, both values must be finite and:

```text
abs(candidate - reference) <= max(0.002, 0.02 * abs(reference))
```

The crossover is `abs(reference)=0.1`. One failed element rejects a trial. This
is an OR rule, not `torch.isclose`'s additive tolerance.

## Eager Checkpoints

For FP16/BF16, the reference FFN is observably structured as:

```text
LayerNorm in FP32 accumulation -> model-dtype store
linear + bias                  -> model-dtype store
exact GELU using erf           -> model-dtype store
linear + bias                  -> model-dtype store
residual add                   -> model-dtype store
mask                           -> exact zero on invalid rows
```

PyTorch's CUDA exact GELU evaluates `x * 0.5 * (1 + erf(x / sqrt(2)))` in the
operation math type; FP16 and BF16 use FP32 operation math. Native CUDA LayerNorm
uses FP32 accumulation and Welford statistics for FP16, BF16, and FP32, followed
by a model-dtype output store.

The benchmark does not use autocast. It explicitly converts the model and input
to the selected dtype. GEMM behavior follows process defaults, including TF32
settings for FP32 and reduced-precision reduction settings for low precision.

## Fusion Assessment

| Fusion | Required behavior | Decision |
| --- | --- | --- |
| Existing linear plus bias | Preserve fused `addmm`/epilogue ordering | Likely safe |
| Linear+bias plus exact GELU | Round linear+bias to model dtype, evaluate exact erf form, round GELU | Safe only with checkpoint emulation |
| Tanh GELU | Changes formula | Reject |
| GELU retained in accumulator | Removes linear-output checkpoint | Reject by default |
| FP32 GELU fed to second GEMM | Changes second-GEMM operands | Reject |
| FFN-out plus residual | Round linear+bias before adding rounded residual, then round result | Safe only with checkpoint emulation |
| Residual plus LayerNorm | Round residual before statistics and validate reduction tree | Research candidate |
| LayerNorm plus projection | Round affine LayerNorm output before GEMM | Research candidate |
| Final mask in producer store | Select exact zero with correct polarity | Safe |
| Key/causal mask after softmax | Changes normalization denominator | Reject |

## Compiler Findings

PyTorch 2.13 Inductor enables epilogue fusion and benchmarked epilogue selection.
It also keeps `addmm` fused for half dtypes to avoid introducing an extra cast
between matrix multiplication and bias.

Precision-cast emulation is disabled by default and can be requested with
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. Its implementation inserts eager-like
downcast/upcast barriers around marked pointwise boundaries and selects PyTorch's
libdevice where relevant. This is evidence of intended eager agreement, not proof
that every GEMM epilogue or reduction is covered.

## Correctness Matrix

Every retained variant must have zero failed elements under:

- dtypes FP32, FP16, and BF16;
- widths 32, 128, and 1024;
- official depths two and four;
- multiple seeds beyond the default five;
- input scales `0.125`, `1`, and `8`, plus controlled FP16 overflow stress;
- padding ratios `0`, `0.25`, and `0.75` with valid lengths 1, half, and full;
- constant and near-constant LayerNorm rows, large offsets with small variance,
  alternating signs, and ordinary Gaussian rows;
- compiler cast emulation both disabled and enabled.

Record failed-element count and
`allowed_error / max(abs_error, tiny)`, not only pass/fail. Large disclosed cases
make rare failures decisive.

## Sources

All public sources were accessed on 29 August 2026.

- PyTorch, [`F.gelu`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.gelu.html),
  tag `v2.13.0`: exact and tanh formulas.
- PyTorch,
  [`ActivationGeluKernel.cu`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/ActivationGeluKernel.cu),
  commit above, symbol `GeluCUDAKernelImpl`: CUDA exact-GELU implementation.
- PyTorch,
  [`OpMathType.h`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/OpMathType.h),
  commit above: FP16/BF16 operation-math mapping.
- PyTorch, [`LayerNorm`](https://docs.pytorch.org/docs/2.13/generated/torch.nn.LayerNorm.html),
  tag `v2.13.0`: biased variance, affine parameters, and epsilon.
- PyTorch,
  [`layer_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu),
  commit above, symbols `LayerNormKernelImpl`, `compute_stats`, and Welford
  helpers: CUDA reduction and accumulation behavior.
- PyTorch, [CUDA semantics](https://docs.pytorch.org/docs/2.13/notes/cuda.html),
  tag `v2.13.0`: TF32 and reduced-precision GEMM reductions.
- PyTorch,
  [`torch/_inductor/config.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/config.py),
  commit above: epilogue fusion, `keep_addmm_fused_for_half_dtypes`, precision-cast
  emulation, and libdevice controls.
- PyTorch,
  [`torch/_inductor/lowering.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/lowering.py),
  commit above, symbols `make_pointwise`, `to_dtype`, and
  `_convert_element_type`: cast emulation lowering.
- Triton, [LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html),
  revision `5d6048aa0a324e090ada215b609ea76620133845`: useful two-pass
  implementation, but its `atol=1e-2` test is not sufficient for this benchmark.
