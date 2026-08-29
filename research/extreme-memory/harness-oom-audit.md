# Authoritative Harness OOM Audit

## Scope

This is a static audit of root `torch_transformer_benchmark.py` at repository
revision `9488c37883537c6bbb68b9b88ff83882846cbcd7`, the commit that introduced
the authoritative benchmark. The root file was not modified.

The requested assumed #14 geometry is already proven infeasible by tensor-size
lower bounds, so intentionally issuing multi-terabyte allocations would add no
useful evidence. PyTorch is also not installed in the current environment.

## Baseline Forward Failure Order

1. `generate_random_case()` allocates the FP16 input at lines 245-252. Line 253
   performs out-of-place scaling, briefly retaining two 6,250 MiB inputs.
2. `BaselineTransformerBlock.forward()` retains the residual input while
   `norm1(x)` creates another full activation at line 140.
3. `q_proj` creates a third full activation at line 93.
4. `_split_heads()` transposes and then calls `.contiguous()` at lines 77-83.
   The fourth full activation raises the visible set to about 25,192 MiB with
   both models' default FP16 parameters. The local L4 has 23,034 MiB total.
5. If that copy were removed, K/V creation at lines 94-95 would exceed capacity.
6. If QKV somehow fit, line 97 requests the 9.313 TiB FP16 score tensor and then
   creates a separate scaled result.
7. With `causal=True`, lines 100-103 create a 9.313 GiB boolean causal mask and
   a score-sized `masked_fill` result.
8. Even with no padding ratio, lines 255-259 always return an all-true valid
   mask. Lines 105-108 therefore still execute another score-sized
   `masked_fill`.
9. Line 111 converts scores to FP32 for softmax and line 112 casts probabilities
   back. These operations have multi-terabyte active sets.
10. Independently, the default FFN at line 141 cannot retain the residual,
    normalized input, and full `[B,N,2048]` intermediate on this L4.

The first expected classification is **Q head-layout allocation OOM**, not
attention-score OOM. The score and FFN are independent later blockers.

## Accuracy-Checker Failure

`run_accuracy_tests()` retains `x`, `reference`, and `candidate` when it calls
`compare_outputs()` at lines 383-393. Those three FP16 tensors consume 18,750
MiB before models and the mask.

`compare_outputs()` then allocates:

- A full FP32 reference at line 306. The active set reaches about 31,442 MiB,
  so this is the first expected checker OOM.
- A full FP32 candidate at line 307 if the first conversion somehow succeeds.
- Multiple full output-sized finite, failure, and pass masks at lines 309 and
  314-318. Each boolean tensor is 3,125 MiB.
- Full FP32 subtraction/absolute-error storage at line 310.
- Full FP32 denominator and relative error at lines 330-331.

The checker is therefore an independent OOM source even if both model forwards
are perfectly streamed.

At the next accuracy trial, the old `x`, `reference`, and `candidate` names can
remain live while the right-hand side creates the next random input. Explicit
per-trial release is required at this scale.

## Warmup And Timing

- `warmup_model()` at lines 463-474 does not retain returned outputs, so warmup
  iterations do not intentionally accumulate outputs. A per-call OOM remains.
- `benchmark_once()` at lines 477-508 retains CUDA events but not outputs.
  Reducing repeats changes total runtime, not per-call peak memory.
- `benchmark_models()` runs the baseline warmup first at line 539. The explicit
  baseline fails before the candidate can be measured.
- Default work is 20 warmups plus 300 timed calls for each model. This is
  impractical for dense 100K attention even after memory is fixed.

## Failure Matrix

| Phase | Expected classification | Structural? | Solved by `empty_cache()`? |
| --- | --- | --- | --- |
| Input generation | Fits nominally | No | Not applicable |
| First Q head split | QKV/layout allocation OOM | Yes | No |
| Attention scores | Attention-score OOM | Yes | No |
| Default FFN | FFN-allocation OOM | Yes | No |
| Candidate with reference retained | Candidate workspace OOM risk | Yes | No |
| Full output checker | Checker-allocation OOM | Yes | No |
| Full benchmark repetitions | Timeout | Yes for given workload | No |

PyTorch documents that `empty_cache()` releases only unused cached memory; it
does not free live tensors or increase memory available to PyTorch. Allocator
knobs can help borderline fragmentation, not a deterministic live set larger
than device capacity.

## Required Experimental-Harness Design

The immutable root benchmark remains the oracle for feasible small shapes. A
copy under `src/` is required for target-scale experiments and should:

1. Preflight tensor sizes before any allocation and report the predicted first
   failure category.
2. Run baseline, candidate, and each chunk configuration in separate child
   processes so one OOM cannot poison later measurements.
3. Replace explicit target-scale attention with a streamed high-precision
   reference. Do not claim that this changes the immutable competition oracle.
4. Compare output chunks immediately, accumulating only scalar failure count,
   maximum errors, sum of absolute errors, worst index/value, and a
   `D`-element failed-feature mask.
5. Offload reference chunks to host memory, or generate and compare one chunk
   at a time, instead of retaining both full GPU outputs.
6. Chunk LayerNorm and FFN as well as attention.
7. Reset and record `max_memory_allocated()` and `max_memory_reserved()` for
   each phase and synchronize before reading results.
8. Explicitly delete per-trial tensors before generating the next trial.
9. Use shape-appropriate warmup/repeat counts and enforce a subprocess timeout.

These are experimental changes only. They must not be applied to the root
benchmark under the repository rules.
