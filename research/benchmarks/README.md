# Benchmark Run Index

Preserved benchmark runs are immutable evidence. Exploratory runs are not listed
unless promoted to a checkpoint or needed to document a regression.

## Current Runs

- [`2026-08-30-rtx4060-d496539/`](2026-08-30-rtx4060-d496539/README.md): Person 2
  Phase 2 task **F0**, the noise floor. Measures how large a difference this
  machine can resolve, using an A/A control — the same callable timed twice as
  if it were two variants. **Working floor 1.03x**: identical code reproduced to
  0.6% cool and 2.7% warm. Establishes that the Stage 0 and Stage 1 gates are
  decidable, that F6 is not measurable at all, and that **both arms of every A/B
  must run in the same session** — identical code drifted 17.5% between sessions
  minutes apart. The speedups it reports are a by-product and are **not** a
  performance claim; they disagree with each other by 17%.
- [`2026-08-30-rtx4060-6dc9639/`](2026-08-30-rtx4060-6dc9639/README.md): Person 2
  kernel-level **profile** of the Phase 1 polynomial path, taken to decide what
  Phase 2 should target. The two Triton kernels are 51% of GPU time and the
  PyTorch glue around them is the other half; the exact diagonal block is 25-35%
  on its own and computes a full `C x C` score matrix it then masks in half.
  **Its millisecond column is attribution, not latency** — four identical runs
  spread 2.17x while the kernels' share held at 51-55%, which is the evidence
  behind the Phase 2 spec's blocking noise-floor task. Also records two negative
  results: no tile schedule outside the shipped autotune space is materially
  faster, and doing the causal skip by sub-blocking in PyTorch is 2.7x slower
  than not doing it.
- [`2026-08-30-rtx4060-poly/`](2026-08-30-rtx4060-poly/README.md): Person 2 fused
  polynomial attention kernel, Phase 1 acceptance. **4.31x at the real chunk
  shape (B=2, N=100000): 328.1 ms against 1414.0 ms exact flash**, with peak-VRAM
  overhead cut from +6773 MiB to +67 MiB. Zero failed elements against the dense reference at N=4096/8192 and
  against an exact-flash oracle to N=100000 (0 / 102,400,000). Includes the
  sigma-guard calibration sweep. Attention core only; case 14 cannot run end to
  end on an 8 GiB card, and the route is an approximation guarded at runtime.
- [`2026-08-30-rtx4060-85cfd8d/`](2026-08-30-rtx4060-85cfd8d/README.md): Person 2
  attention mask-route sweep on the pinned cu130 stack. All twelve in-scope
  cases x `padding_ratio` 0.0/0.3 x two routes, 5/5 seeds each with zero failed
  elements. Dropping the causal padding key mask is ahead in 20 comparisons,
  tied in 4, behind in none. Includes a case 6 correctness-and-memory record
  whose latency is explicitly **not** a claim: it is `WITHIN NOISE` and ran under
  host-memory oversubscription (10,648 MiB peak on an 8,188 MiB card).
- [`2026-08-29-l4-f128f6e/`](2026-08-29-l4-f128f6e/README.md): Person 4
  extreme-memory checkpoint on NVIDIA L4. Official Case 6 passes 1/1 with zero
  failed elements, 2.487x one-sample speedup, and 2.61 GiB candidate incremental
  peak allocation. Official Case 14 completes candidate-only in FP16 without
  OOM at 15.27 GiB peak allocated memory; target-scale numerical correctness is
  unverified because the immutable dense baseline is not runnable.
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

- 30 August 2026: the `attention-core.json` figure of **342.4 ms / 2.12x** inside
  [`2026-08-30-rtx4060-poly/`](2026-08-30-rtx4060-poly/README.md) is
  **superseded**. It used contiguous inputs where the real module supplies
  strided views, it included a 3-D SDPA fallback costing 2.4 GiB and ~72 ms, and
  its variants were timed back to back on a thermally throttling laptop GPU.
  Replaced by `attention-core-v2.json` (B=1) and `attention-core-b2.json` (B=2)
  in the same directory. The JSON is retained as history; do not quote it.

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
