# Benchmark Run Index

Preserved benchmark runs are immutable evidence. Exploratory runs are not listed
unless promoted to a checkpoint or needed to document a regression.

## Current Runs

- [`2026-08-29-rtx5080-330cf60/`](2026-08-29-rtx5080-330cf60/README.md): Person 1
  integrated float32 SDPA, strided-view, and `reduce-overhead` evidence for
  cases 1, 3, 4, 5, 7, and 9-12; all nine pass 5/5 seeds and show significant
  gains, with a 3.397x geometric mean.
- [`2026-08-29-rtx5080-6bde871/`](2026-08-29-rtx5080-6bde871/README.md): Person 1
  compiler-mode checkpoints and rejected numerical routes on an RTX 5080.

## Invalid, Stale, or Superseded Runs

- 29 August 2026: case 8 float32 `max-autotune` is **invalid for performance
  comparison** because correctness failed on all five trials. Use the valid
  [`reduce-overhead` result](2026-08-29-rtx5080-6bde871/case8-fp32-reduce-overhead.json)
  as the current compiler control; see the run README for the failure record.
- 29 August 2026: case 13 float16 `reduce-overhead` with precision-cast emulation
  is **invalid for performance comparison** because two of eight trials failed.
  There is no accepted compiled float16 replacement; use the eager-compatible
  fallback pending new evidence.
