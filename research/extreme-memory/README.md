# Extreme-Shape Memory Research

## Status

Active as of 28 August 2026. This topic covers Person 4's memory-safe
execution work for internally assumed shapes #6 and #14.

The dimensions below are planning assumptions from
[`four-way-team-split.md`](../team-coordination/four-way-team-split.md), not
organizer-confirmed test cases. `TASK.md` still records that the official test
shape table is unavailable. In particular, assumed #14 is:

```text
B=32, H=16, N=100000, d_model=1024, head_dim=64
```

The dimensions, `ffn_dim`, layer count, dtype, causal setting, timeout, and
scoring status must be confirmed before implementation choices are frozen.

## Conclusions

- The root benchmark's explicit baseline cannot execute assumed #14 on the
  local 23,034 MiB NVIDIA L4. The first expected failure is already the first
  Q head-layout `.contiguous()` allocation, before attention scores are
  created.
- The explicit FP16 attention score contains 5.12 trillion values and requires
  10.24 TB. The subsequent FP32 softmax path is larger. Allocator tuning cannot
  fix a live-set capacity shortfall of this scale.
- A streamed candidate is possible in linear memory: process one sample at a
  time, retain that sample's K and V, generate Q by query tile, apply online
  softmax over key tiles, and immediately output-project each completed query
  tile.
- Linear memory does not make dense attention computationally cheap. Assumed
  #14 requires about 1.311 PFLOPs per non-causal layer or 0.655 PFLOPs per
  causal layer. The corresponding impossible-to-beat L4 dense FP16/BF16 peak
  bounds are about 10.83 and 5.42 seconds per layer, before softmax, projections,
  FFN, memory traffic, and launch overhead.
- The root accuracy checker independently OOMs at this output size because it
  retains full input, reference, and candidate tensors and then creates full
  FP32 copies and boolean/error temporaries. Target-scale correctness therefore
  needs a streamed oracle/checker in an experimental harness under `src/`.
- Assumed #6 cannot yet be classified from its approximate 1.3 GB FP16 score
  tensor alone. The score is individually feasible, but the eager score,
  masking, FP32 softmax, QKV, FFN, and model live sets must be calculated from
  the missing full dimensions.
- No runtime result has been claimed. PyTorch is not installed in the current
  environment, and deliberately requesting a proven multi-terabyte allocation
  would not add useful evidence.

## Documents

- [Tensor and compute model](tensor-and-compute-model.md): formulas, assumed
  #14 sizes, lower bounds, and analytic chunk scratch sizes.
- [Authoritative harness OOM audit](harness-oom-audit.md): tensor lifetimes,
  first-failure classification, and required experimental-harness changes.
- [Exact streaming strategy](streaming-strategy.md): online recurrence,
  scheduling, masks, numerical risks, and chunk selection.
- [Experiment protocol](experiment-protocol.md): isolated-process failure
  classification and the measurements to run when the official shapes and
  PyTorch environment are available.
- [Source catalog](sources.md): public URLs, access dates, repository revisions,
  and task-relevant summaries.

## Organizer Clarifications Needed

1. Is assumed #14 intentional, and what is its exact complete
   `(batch, d_model, heads, sequence_length, layers, causal, ffn_dim, dtype)`
   tuple?
2. Is every disclosed shape scored, or is #14 excluded or informational?
3. What GPU, VRAM capacity, PyTorch/CUDA versions, timeout, warmup count, and
   repeat count are used for judging?
4. Must the submitted candidate run inside the supplied baseline-versus-user
   process, or does judging run the candidate separately against stored or
   streamed expected results?
5. Is preprocessing or weight packing outside the timed region allowed?
6. Are right-padded masks always contiguous prefixes, and can per-sample valid
   lengths be used instead of an explicit attention mask?

## Decision Gate

Do not implement a shape-specific #14 kernel until the shape and judging
harness are confirmed. If confirmed, first copy the harness under `src/`, add a
streamed reference/checker, and validate the ordinary PyTorch Flash SDPA path
on the L4. Build a custom kernel only if Flash SDPA cannot express the required
causal-plus-padding semantics or fails the executable tolerance.
