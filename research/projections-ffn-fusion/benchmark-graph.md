# Benchmark Graph Analysis

## Status

- Date: 29 August 2026.
- Benchmark revision: `a76b37f0b7f62a1fac4b45880b6b031492972611`.
- Evidence type: static analysis. Byte and launch counts are estimates until
  confirmed by a profiler.

## Exact Person 3 Graph

For each layer, the immutable benchmark computes two LayerNorms, Q/K/V/output
projections, two FFN projections, exact GELU, two residual additions, three QKV
layout copies, one context layout copy, and output masks. The relevant source is
`BaselineSelfAttention` and `BaselineTransformerBlock` in
`torch_transformer_benchmark.py:59-145`.

Every linear has bias. LayerNorm uses affine parameters and `eps=1e-5`. There is
no dropout or training. Timed execution uses `torch.inference_mode()`.

Let `T = batch * sequence_length`, `D = d_model`, and `F = ffn_dim`. Every
disclosed case has `F=D`.

| Operation | GEMM `(M,N,K)` | FLOPs per layer |
| --- | --- | ---: |
| Each Q, K, or V | `(T,D,D)` | `2TD^2` |
| Packed QKV | `(T,3D,D)` | `6TD^2` |
| Output projection | `(T,D,D)` | `2TD^2` |
| Two FFN projections | two `(T,D,D)` | `4TD^2` |
| Person 3 dense total | six square projections | `12TD^2` |

## Shape Families

| Family | Cases | Person 3 interpretation |
| --- | --- | --- |
| Batch sweep, `S=D=128` | #1-#6 | Small `T` favors launch removal; #6 makes every full activation pass expensive |
| Narrow, `D=32` | #7 | Only `0.101` GFLOP of Person 3 GEMM work per layer; fusion is more plausible than custom GEMM tuning |
| Wide, `D=1024` | #8 | About `103.1` GFLOP of Person 3 GEMM work per layer; preserve strong GEMM mainloops |
| Head sweep | #9-#11 | Person 3 GEMM shapes are identical; differences expose layout/backend coupling |
| Sequence sweep | #12-#13 | #12 is launch-sensitive; #13's attention dominates but projection copies remain linear traffic |
| Extreme | #14 | Full `[T,D]` low-precision activation is about 6.10 GiB; Person 4 owns streaming feasibility |

## Traffic Opportunities

Let `A = bytes_per_element * T * D`, the size of one activation.

- A packed QKV operator can avoid two repeated logical input reads and two GEMM
  dispatches relative to three linears. This is up to `2A` traffic, subject to
  cache behavior.
- Removing the three QKV transpose-contiguous operations avoids `6A` traffic.
- Removing the context transpose-contiguous operation avoids another `2A`.
- Fusing exact GELU into FFN-in and residual into FFN-out can reduce an idealized
  FFN activation path from `9A` to `5A`, but only if eager rounding is retained.
- Fusing residual followed by LayerNorm while returning both residual and
  normalized values saves about `A`; it cannot discard the residual state.

For FP16/BF16, removing all four explicit layout copies has the following upper
bounds per layer: #7 `4 MiB`, #8 `128 MiB`, #6 `2.44 GiB`, and #14 `48.83 GiB`.
These are traffic calculations, not measured speedups.

## Benchmark Semantics

- Accuracy executes before performance and uses fresh inputs per trial
  (`torch_transformer_benchmark.py:359-424`).
- Performance uses a fixed generated input excluded from timing (`:524-536`).
- Both models warm up before collection (`:463-475`, `:538-540`).
- CUDA timing uses events and alternates model order by round (`:477-583`).
- Setup-time parameter packing is outside timed forward execution.
- The valid-token mask is non-`None` even at zero padding (`:255-259`), so eager
  mask branches execute for the default benchmark.
- The checker converts outputs to FP32 and requires every finite element to meet
  `abs_error <= 0.002 OR abs_error <= 0.02*abs(reference)` (`:289-356`, defaults
  at `:617-620`). The module header's older tolerance text is stale.

## Ranked Decisions

1. Compare packed QKV with both the original baseline and a stronger three-linear
   view-only control. Charge all attention adapters to the complete path.
2. Use compiler or library GEMMs first. A custom mainloop must improve complete
   latency by enabling a layout or epilogue, not merely match `F.linear`.
3. Test exact-GELU and output-residual fusion first on #7/#12, then epilogues on
   #8.
4. Use #1/#9/#10/#11 as a controlled experiment: isolated projection latency
   should be the same because `(T,D)` is identical.
5. Keep #6/#14 projection work chunk-compatible, but leave chunk scheduling and
   feasibility to Person 4.

## Experiments And Stop Rules

- Record isolated and complete-path latency, kernel count, copy kernels, peak
  memory, strides, failed elements, and normalized tolerance headroom.
- Test FP32, FP16, and BF16 independently, multiple seeds, input scales, and
  padding ratios.
- Reject packed QKV where it does not beat three view-only linears end to end.
- Reject a fusion below the team's 15% isolated threshold or with any reproducible
  correctness failure.
- Reject a producer-side layout win that causes a larger attention or output
  projection regression.
- Treat case #14 baseline feasibility as a task-level gate, not a projection
  kernel optimization.

## Sources

- Local immutable benchmark, revision above, accessed 29 August 2026. Relevant
  symbols: `BaselineSelfAttention`, `BaselineTransformerBlock`,
  `compare_outputs`, `benchmark_once`, and `benchmark_models`.
- [`TASK.md`](../../TASK.md), accessed 29 August 2026. Relevant sections: 3.2,
  3.4, and 3.7.
