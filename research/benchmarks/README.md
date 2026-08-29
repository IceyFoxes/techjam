# Benchmark Run Index

Preserved benchmark runs are immutable evidence. Exploratory runs are not listed
unless promoted to a checkpoint or needed to document a regression.

## Current Runs

- [`2026-08-30-rtx4060-85cfd8d/`](2026-08-30-rtx4060-85cfd8d/README.md): Person 2
  attention mask-route sweep on the pinned cu130 stack. All twelve in-scope
  cases x `padding_ratio` 0.0/0.3 x two routes, 5/5 seeds each with zero failed
  elements. Dropping the causal padding key mask is ahead in 20 comparisons,
  tied in 4, behind in none.
- [`2026-08-29-rtx5080-12a37c6/`](2026-08-29-rtx5080-12a37c6/README.md):
  three-reference packed-QKV cross-case validation. Universal packing is
  rejected; Case 3 is a new RTX 5080 promotion candidate at 1.192x direct gain,
  ±0.84% noise, and PASS 60/60 stress trials.
- [`2026-08-29-rtx5080-f945daf/`](2026-08-29-rtx5080-f945daf/README.md): merged
  Case-2 packed-QKV checkpoint; PASS 5/5 at 6.191x versus the immutable baseline,
  plus a same-process diagnostic showing a 1.234x incremental gain over the
  prior three-projection compiled route.
- [`2026-08-29-rtx5080-307eedb/`](2026-08-29-rtx5080-307eedb/README.md): final
  hardened dispatcher evidence for all twelve supported cases on the RTX 5080;
  PASS 5/5 on every case, 3.548x geometric mean, and CUDA peak-memory records.
- [`2026-08-29-rtx5080-330cf60/`](2026-08-29-rtx5080-330cf60/README.md): Person 1
  integrated float32 SDPA, strided-view, and `reduce-overhead` evidence for
  cases 1, 3, 4, 5, 7, and 9-12; all nine pass 5/5 seeds and show significant
  gains, with a 3.397x geometric mean.
- [`2026-08-29-rtx5080-6bde871/`](2026-08-29-rtx5080-6bde871/README.md): Person 1
  compiler-mode checkpoints and rejected numerical routes on an RTX 5080.

## Invalid, Stale, or Superseded Runs

- 29 August 2026: the RTX 4050 Case-3 packed-QKV rejection in
  [`packed-qkv-exploration.md`](../projections-ffn-fusion/packed-qkv-exploration.md)
  remains valid for that GPU but is **not portable to the RTX 5080**. The
  [`12a37c6 cross-case validation`](2026-08-29-rtx5080-12a37c6/README.md)
  provides contrary hardware-specific evidence: 1.192x, ±0.84%, PASS 60/60.
- 29 August 2026: the integrated probe at
  [`2026-08-29-rtx5080-330cf60/`](2026-08-29-rtx5080-330cf60/README.md) is valid
  historical evidence but **superseded for final routing** by the exact hardened
  dispatcher records at `307eedb`. Preserve it as the pre-dispatch composition
  checkpoint.
- 29 August 2026: [`2026-08-29-rtx4050-optimization-wave1.md`](2026-08-29-rtx4050-optimization-wave1.md)
  is **stale exploratory evidence**. It predates the paired, settled timing
  correction and omits exact commands, timestamps, dtype, raw samples, timing
  parameters, and the complete software and hardware environment. The
  experimental implementations are also not present at its attributed
  `f4546f4` revision. Its `20.75x` compiler result is therefore not an accepted
  breakthrough or routing result, and its `1.126x` padded-mask result cannot be
  attributed to an all-true-mask elimination because padded masks contain
  invalid positions. The same record's `0.967x` unpadded result does not support
  a claim of zero cost. Preserve the leaf as history; use the paired RTX 5080
  records below as contrary methodological evidence, not as same-machine
  numerical replacements.
- 29 August 2026: case 8 float32 `max-autotune` is **invalid for performance
  comparison** because correctness failed on all five trials. Use the valid
  [`reduce-overhead` result](2026-08-29-rtx5080-6bde871/case8-fp32-reduce-overhead.json)
  as the current compiler control; see the run README for the failure record.
- 29 August 2026: case 13 float16 `reduce-overhead` with precision-cast emulation
  is **invalid for performance comparison** because two of eight trials failed.
  There is no accepted compiled float16 replacement; use the eager-compatible
  fallback pending new evidence.
