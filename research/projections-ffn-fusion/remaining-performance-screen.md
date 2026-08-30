# Person 3 Remaining Performance Screen

## Status

- Date: 29 August 2026.
- Repository revision: `f945dafefb6fbfafe889e96db0f4a2cbbf52f5f0`.
- Scope: Case #8 attribution and local bounded screens after Case #2 packed-QKV
  integration.
- Result: no additional route passes both performance and numerical gates on the
  RTX 4050. Exact RTX 5080 selective GEMM tuning remains open; the local results
  do not support another implementation before that access is available.
- These are exploratory selection measurements, not preserved benchmark claims.
  Temporary scripts and profiles remain under `/tmp/opencode/` and are not part
  of the repository.

## Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9, 20 SMs,
  driver 610.57.04, 6,141 MiB reported memory.
- CPU: Intel Core Ultra 7 155H.
- OS: Linux 7.2.2-1-cachyos, x86-64, glibc 2.44.
- Python / PyTorch / CUDA / Triton: 3.14.7 / 2.13.0+cu130 / 13.0 / 3.7.1.
- Numerical contract unless stated otherwise: float32, high matmul precision,
  TF32 enabled, inference mode.
- The worktree contained a pre-existing untracked benchmark directory during
  collection. It was not read, changed, or included in this research.

Exploratory collection ran from approximately 13:30 to 13:49 UTC. The temporary
probes used five seconds of settling. Selective tuning used 40 paired samples of
two calls, weight layout used 80 paired samples of four calls, and residual
epilogues used 80 paired samples of ten calls. The probe entry points were:

```bash
PYTHONPATH=. .venv/bin/python /tmp/opencode/case8_selective_autotune.py --allow-tf32
PYTHONPATH=. .venv/bin/python /tmp/opencode/case8_selective_autotune.py --no-allow-tf32
PYTHONPATH=. .venv/bin/python /tmp/opencode/case8_weight_layout_probe.py
PYTHONPATH=. .venv/bin/python /tmp/opencode/residual_epilogue_whole_model.py --case 5
PYTHONPATH=. .venv/bin/python /tmp/opencode/dispatcher_retention_probe.py --case CASE
```

## Case #8 Current-Dispatcher Attribution

The warmed one-iteration profile used:

```bash
.venv/bin/python -m src.profile \
  --candidate src.dispatcher --case 8 --device cuda --dtype float32 \
  --models candidate --warmup 5 --active 1 --with-stack --with-flops \
  --output-dir /tmp/opencode/case8-current-dispatcher-profile
```

The profile recorded 62.424 ms of self CUDA time:

| Physical work | Calls | CUDA time | Share |
| --- | ---: | ---: | ---: |
| Square CUTLASS/cuBLAS GEMMs | 24 | 47.122 ms | 75.49% |
| Memory-efficient attention | 4 | 6.578 ms | 10.54% |
| Residual, mask, and LayerNorm Triton kernels | 8 | 6.300 ms | 10.09% |
| FFN bias, exact GELU, and view Triton kernels | 4 | 1.510 ms | 2.42% |
| Final LayerNorm | 1 | 0.449 ms | 0.72% |

All 24 dense operations have `(M,K,N) = (8192,1024,1024)`: sixteen Q/K/V or
attention-output projections and eight FFN projections. They use the same
`cutlass_80_tensorop_s1688gemm_64x256_16x4_tn_align4` kernel in this profile.

This corrects the earlier incomplete attribution. Case #8 is not merely 31.5%
projection-bound: all Person 3-owned dense GEMMs consume 75.49% of device time.
However, the trace also shows that the residual adds are already combined with
masking and following LayerNorm work, and FFN bias plus GELU is already one
pointwise kernel. Source-level rewrites cannot claim those launches again.

The separate FFN bias/GELU work has a 2.42% whole-model ceiling. This directly
rejects an independent wide-shape GELU epilogue under the 5% whole-model gate,
even though the earlier Case #7 result alone was not sufficient evidence for
`d_model=1024`.

## Selective GEMM Autotuning

The selective screen set only `torch._inductor.config.max_autotune_gemm=True`
with `ATEN,TRITON` backends while retaining `reduce-overhead`. Inductor normally
disables Triton GEMM templates below 68 SMs, so the temporary probe overrode only
that local hardware gate to expose the same search family available on larger
GPUs.

With TF32 enabled, Inductor benchmarked 22 choices. Its best Triton template used
`BLOCK_M=128`, `BLOCK_N=64`, `BLOCK_K=32`, three stages, and four warps. The
isolated biased GEMM result was only about 2.1% faster than ATen (`1.721 ms`
versus `1.758 ms`), and the whole model was neutral:

| Route | Median | Paired ratio vs current | Stress correctness |
| --- | ---: | ---: | --- |
| Current dispatcher | 67.947 ms | control | reference control |
| Selective GEMM autotuning | 67.975 ms | 0.9996x | **FAIL**, 534,620 elements |

The stress matrix used three seeds, scales `0.125`, `1`, and `8`, and padding
ratios `0` and `0.25`. The tuned route failed every low- and default-scale trial;
worst maximum absolute error was 0.007551. This reproduces the numerical failure
class in the preserved RTX 5080 global `max-autotune` record and localizes it to
alternate GEMM arithmetic rather than pointwise autotuning.

With TF32 disabled through `matmul_precision="highest"`, all 18 trials passed
with worst maximum absolute error `6.97e-6`, but ATen beat every IEEE Triton
template. Selective tuning remained neutral (`125.927 ms` versus `125.635 ms`,
0.9980x) and the current IEEE route was about 1.85x slower than the corresponding
TF32 process. TF32-off execution is therefore a correctness diagnostic, not a
performance route.

Exact RTX 5080 selective-only timing remains unmeasured. The existing RTX 5080
global-autotune record fails 7,899 default-scale elements, but it does not prove
which globally changed choice caused the failures or which selective GEMM the
RTX 5080 would choose. Selective-only RTX 5080 testing therefore remains open,
although the local result provides no reason to implement it now.

## Case #8 Weight Prepacking

Packing each `[N,K]` linear weight once as contiguous `[K,N]` changes cuBLAS from
the `tn` kernel above to
`cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4`. It does not change the
result in this screen, but it also does not materially improve latency:

| Layout | Paired median | One-call profile | Correctness |
| --- | ---: | ---: | --- |
| Standard `F.linear` weight | 2.558 ms | 1.563 ms | control |
| Pretransposed contiguous weight | 2.544 ms | 1.531 ms | bitwise identical |

The median of per-pair ratios is only 1.0033x, far below the isolated and
whole-model gates. It differs from the ratio of displayed medians because the
measurement uses an alternating paired schedule.
Duplicating roughly 96 MiB of Case #8 dense weights for this result is unjustified.
The same experiment also serves as a concrete cuBLAS algorithm/layout screen;
there is no installed Python cuBLASLt binding for a broader heuristic search.

## Output-Residual Epilogue

A temporary `torch.library.triton_op` implemented both output GEMMs for the D128
families. It reads the SDPA output directly from its non-contiguous
BSHD-backed `[B,H,S,D]` stride, computes `linear + bias + residual`, and applies
the validity mask in one launch. Thus the screen jointly covers:

- output-GEMM `beta*C`-style residual fusion;
- padding-mask fusion; and
- zero-copy context consumption by the attention output projection.

After correcting the SDPA output-stride mapping, Case #5 produced:

| Dot precision | Dispatcher median | Candidate median | Paired-ratio median | Stress result |
| --- | ---: | ---: | ---: | --- |
| Triton TF32 | 6.563 ms | 6.026 ms | 1.090x | **FAIL**, 22,132 elements |
| Triton TF32x3 | 6.509 ms | 6.847 ms | 0.948x | **FAIL**, 104 elements |
| Triton IEEE | 6.297 ms | 7.352 ms | 0.860x | **FAIL**, 92 elements |

All three use the same 18-trial scale, padding, and seed matrix. TF32's worst maximum
absolute error was 0.005459; TF32x3's was 0.003284; IEEE's was 0.003237. The
existing dispatcher also failed six low-scale elements in this matrix, but every
fused candidate added failures and cannot be accepted. IEEE still changes the
GEMM reduction tree and fused operation boundaries, so disabling TF32 does not
reproduce the materialized reference checkpoints. Cases #1, #9, and #10 were not
run because the predeclared Case #5 correctness gate rejected the common
implementation.

This result also answers the FP32 precision-emulation question for the actual
D128 residual candidate. TF32x3 materially reduces error but neither clears
correctness nor retains speed; IEEE is slower still and also fails.

## CUDA Graph Memory Retention

Each case ran in a fresh process. Measurements were taken immediately before the
first dispatcher call, after compilation/capture/replay, and after deleting the
output, collecting Python garbage, and calling `torch.cuda.empty_cache()`.

| Case | Mode | Peak allocated | Retained live allocation | Retained reservation |
| ---: | --- | ---: | ---: | ---: |
| 5 | `reduce-overhead` | 97.7 MiB | 2 KiB | 134.0 MiB |
| 8 | `reduce-overhead` | 312.3 MiB | 2 KiB | 214.0 MiB |
| 13 | `default` | 1.166 GiB | 8.125 MiB | 24.0 MiB |

Cases #5 and #8 therefore do not retain material live tensors after output
release, but CUDA Graph/private allocator pools keep substantial reserved memory
that `empty_cache()` cannot return during the model lifetime. Case #13 avoids
CUDA Graph mode and has much lower reservation retention, although its attention
workspace produces the highest transient peak.

## Decisions

| Direction | Decision | Reason |
| --- | --- | --- |
| Case #8 profiler attribution | Complete | Dense GEMMs are 75.49%; fusion kernels are already coalesced |
| Selective GEMM autotuning | Reject locally; RTX 5080 open | Neutral whole model and severe local numerical failure |
| cuBLAS weight prepacking | Reject | 1.0033x isolated ratio with significant duplicate storage |
| Wide FFN GELU epilogue | Reject | Measured whole-model ceiling is 2.42% |
| Output-residual epilogue | Reject | Fast TF32 fails; TF32x3 regresses and still fails |
| FP32 precision emulation | Reject | Error improves, but the speed/correctness intersection is empty |
| Zero-copy context output projection | Reject as implemented | Included in the failed bundled residual candidate; no independent speedup is established |
| Padding-mask epilogue | Reject as implemented | Included in the failed residual candidate and already compiler-fused downstream |
| Back-to-back FFN GEMM | Stop | Exact GELU is a mandatory global intermediate; no measured L2 issue justifies a custom mainloop |

The only integrated Person 3 implementation candidate remains Case #2 packed
QKV. Further work requires one of the following new facts rather than another
source rewrite:

1. RTX 5080 access showing a numerically valid cuBLASLt algorithm unavailable on
   the RTX 4050;
2. an installed, supported GEMM interface that exposes bias plus residual while
   retaining the reference algorithm; or
3. a changed correctness/runtime contract.

Separately, the six low-scale failures in the current Case #5 dispatcher should
be triaged by the integrator because they predate and are independent of the
rejected epilogue candidate.

## Sources

All public sources were accessed 29 August 2026.

- PyTorch repository, tag `v2.13.0`, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`,
  [`torch/_inductor/config.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/config.py),
  symbols `max_autotune_gemm`, `max_autotune_gemm_backends`,
  `epilogue_fusion`, and `b2b_gemm_pass`. These define the selective search and
  show that back-to-back GEMM is disabled by default.
- PyTorch repository at the same revision,
  [`torch/_inductor/utils.py`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/utils.py),
  symbols `is_big_gpu` and `use_triton_template`. These establish the 68-SM
  local template-search gate and its relationship to GEMM autotuning.
- PyTorch `torch.library.triton_op` documentation,
  <https://docs.pytorch.org/docs/2.13/library.html#torch.library.triton_op>.
  It documents graph-visible Triton custom operators used by the residual probe.
- Triton `tl.dot` documentation,
  tag `v3.7.1`,
  [`python/triton/language/core.py`](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/core.py),
  symbol `dot`. It documents the TF32, TF32x3, and IEEE input-precision choices
  screened here.
- Local immutable benchmark `torch_transformer_benchmark.py`, symbols
  `BaselineTransformerBlock.forward` and `compare_outputs`; final dispatcher
  `src/dispatcher.py`, symbols `DispatchingTransformer.forward` and
  `CASE_COMPILE_MODES`; and preserved RTX 5080 records at
  `research/benchmarks/2026-08-29-rtx5080-307eedb/` and
  `research/benchmarks/2026-08-29-rtx5080-6bde871/`. They supply the required
  graph, accepted control, and prior global-autotune failure.
