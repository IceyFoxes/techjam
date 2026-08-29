# Narrow FFN Analysis

## Status

- Date: 29 August 2026.
- Scope: case #7, `B=64`, `S=128`, `D=F=32`, four layers.
- Evidence type: mathematical and graph analysis. Exploratory timings have not
  yet been preserved under `research/benchmarks/` and are not accepted evidence.

## Work And Data

Flattening tokens gives `T=8192`. Each FFN GEMM is
`[8192,32] @ [32,32] -> [8192,32]` and performs about `16.8` MFLOP. Both GEMMs
perform about `33.6` MFLOP per layer; all six Person 3 projections perform about
`100.7` MFLOP per layer.

One low-precision `[T,D]` activation is `0.5 MiB`. Ignoring weight traffic, the
eager FFN's two linears, GELU, and residual add move up to `9A`; exact-GELU and
residual epilogues can reduce this toward `5A`. The reference exposes model-dtype
rounding after FFN-in+bias, GELU, FFN-out+bias, and residual add.

The second GEMM depends on all 32 values of each token's GELU output. That
dependency width is small enough for one program to retain one or several rows
on chip, unlike `D=1024`. It does not eliminate the need to stream both weight
matrices and produce all rows.

## Candidate Order

1. **Compiled subgraph with eager checkpoints preserved.** Lowest implementation
   cost and can fuse pointwise operations or launches. Generated code and cast
   behavior must be inspected.
2. **Library GEMMs with exact-GELU and residual epilogues.** Retains strong
   mainloops while removing activation passes, if the available epilogue supports
   exact `erf` and explicit rounding.
3. **Persistent row/tile FFN kernel.** A fixed grid can process multiple token
   rows and retain the width-32 intermediate. This is plausible because launch
   and traffic costs are large relative to arithmetic.
4. **Whole block fusion.** LayerNorm, attention, and FFN dependencies make this
   much more complex; do not attempt before isolated paths establish a measured
   ceiling.

## Tradeoffs

| Approach | Benefit | Risk |
| --- | --- | --- |
| Compiler | Minimal custom code; broad graph view | May choose slower small GEMMs or remove precision checkpoints |
| Epilogues | Small change around library GEMMs | Exact GELU support and intermediate rounding may be unavailable |
| Persistent fused FFN | Removes launches and intermediate traffic | Register/shared-memory pressure, duplicated weight loads, custom correctness burden |
| Back-to-back tiled GEMM | Keeps a tile intermediate on chip | Second output tile's reduction depends on the full width; scheduling can erase gains |

## Decision

A fused width-32 FFN is technically plausible, but its whole-model value must be
bounded before implementation. Attention and the remaining graph can dominate
case #7 even if the FFN itself improves substantially. The research priority is
therefore:

1. measure isolated eager FFN and whole-model share under the repository protocol;
2. test compiler partitioning and precision-cast emulation;
3. implement a custom persistent path only if the measured Amdahl ceiling can
   affect whole-model ranking.

## Experiments And Kill Criteria

- Preserve baseline runs before accepting any timing.
- Compare eager, compiled, two library GEMMs with epilogues, and one custom fused
  path using identical weights and exact GELU.
- Profile physical kernels, registers, occupancy, and global bytes; source-level
  operator count is not enough.
- Require at least 15% isolated improvement and zero failed elements across all
  correctness stresses in `numerical-fusion.md`.
- Kill the custom path if FFN's measured share caps whole-model benefit below a
  useful threshold, if compiled partitioning is faster, or if checkpoint
  emulation erases the gain.
- Do not extrapolate case #7 results to widths 128 or 1024.

## Sources

All public sources were accessed on 29 August 2026.

- Local immutable benchmark at revision
  `a76b37f0b7f62a1fac4b45880b6b031492972611`, symbols
  `BaselineTransformerBlock.forward` and `compare_outputs`.
- Triton,
  [persistent matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html):
  persistent scheduling patterns and evidence that persistence is not
  automatically faster than conventional or library GEMMs.
- NVIDIA, [CUTLASS 4.8 overview](https://docs.nvidia.com/cutlass/latest/overview.html):
  back-to-back GEMM and custom epilogue directions. Example availability is not
  evidence of benefit for case #7.
- Hong and Kung,
  [I/O Complexity: The Red-Blue Pebble Game](https://doi.org/10.1145/800076.802486):
  framework for requiring a reduction in communication, synchronization, or
  recomputation rather than treating a new kernel language as an optimization.
