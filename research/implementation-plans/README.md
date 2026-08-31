# Implementation Plan Index

These documents preserve the task-by-task plans used during implementation.
They are historical research records; current behavior and performance claims
remain defined by the final dispatcher and preserved benchmark records.

- [`2026-08-29-safe-attention-optimizations.md`](2026-08-29-safe-attention-optimizations.md):
  safe SDPA and causal key-mask routing plan. Implemented and superseded by the
  final integrated dispatcher evidence.
- [`2026-08-30-fused-polynomial-attention-kernel.md`](2026-08-30-fused-polynomial-attention-kernel.md):
  Phase 1 fused polynomial-attention plan for Case 14. Implemented; later
  integration and hardware-specific decisions are recorded by Stage 0 and the
  final benchmark matrix.
- [`2026-08-30-integrated-polynomial-kernel-stage0.md`](2026-08-30-integrated-polynomial-kernel-stage0.md):
  executed and accepted Stage 0 plan, including its rejected alternatives and
  links to the corresponding RTX 4060 benchmark evidence.
