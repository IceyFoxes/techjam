# Framework Fast Paths and Dispatcher (Person 1)

Research for Person 1's ownership: framework compilation, CUDA Graphs, and the
final shape-aware dispatcher. The immutable benchmark remains
[`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py).

Status: current as of 29 August 2026. The hardened dispatcher implementation is
at branch commit `307eedb` in PR #6. Final preserved evidence covers all twelve
supported cases; Person 3 has no accepted backend to integrate, and cases 6/14
remain explicitly unsupported pending memory-safe execution.

## Documents

- [`compiler-and-cudagraphs.md`](compiler-and-cudagraphs.md) — exact mode
  semantics in PyTorch 2.13, capture and guard behavior, CUDA Graph constraints,
  and consequences for the benchmark lifecycle.
- [`measurements.md`](measurements.md) — RTX 5080 environment, method, preserved
  whole-model measurements, numerical failures, and experimental limits.
- [`dispatcher-strategy.md`](dispatcher-strategy.md) — per-case routing matrix,
  routing key, integration contracts, and the evidence gate for promoting a
  backend.

## Conclusions

1. **Smart dispatch is required.** The disclosed shapes are fixed and may be
   selected explicitly, while the measured best compiler mode and even numerical
   validity vary by shape and dtype. There is no defensible universal route.
2. **The hardened dispatcher covers all twelve supported cases.** Float32 SDPA
   with strided Q/K/V views and shape-specific compilation passes 5/5 seeds on
   cases 1-5 and 7-13. Every improvement clears its run-specific noise floor;
   speedups range from 1.118x to 7.498x with a 3.548x geometric mean.
3. **`reduce-overhead` is the leading launch-bound route.** On case 2 float32 it
   measured 6.581x versus 2.212x for default mode. `max-autotune` measured 6.609x,
   an indistinguishable runtime result with more numerical error and much more
   compilation work.
4. **`max-autotune` is not globally safe.** On case 8 float32 it failed all five
   seeds with 7,899 bad elements in total and `max_abs=0.004585`, despite the
   default tolerance of 0.002 absolute or 2% relative.
5. **Long sequence and wide model need different compiler treatment, but both
   should compose float32 SDPA.** Case 13 gains 3.179x from ordinary compile,
   while CUDA Graph replay adds no credible benefit to that compute/memory-bound
   shape. Case 8 gains only 1.095x from `reduce-overhead`; Person 2's corrected
   whole-model sweep shows SDPA is still a small win there, while the dominant
   remaining opportunity belongs to Person 3's projection path.
6. **Float16 routing must be case-specific and conservative.** Case 2 passes
   eight seeds under `reduce-overhead`, but case 13 still fails with precision-cast
   emulation enabled. Eager-compatible fallback remains mandatory.
7. **The model is compiler-friendly.** `torch._dynamo.explain` captured the full
   case 2 model as one graph with 91 operators and zero graph breaks. One GPU
   recompile was observed from an input dispatch-key mismatch between invocation
   contexts; no repeated graph break or shape recompile was observed.

## Immediate Implementation Direction

- Build a lazy, per-model compiled callable after device and dtype placement.
- Key dispatch on the complete disclosed tuple plus dtype, device, mask presence,
  and relevant numerical flags.
- Use `reduce-overhead` only for measured launch-bound routes; use default compile
  for case 13 until the SDPA composition is measured; reject `max-autotune` by
  default.
- Use the measured hardened dispatcher as the leading implementation for cases
  1-5 and 7-13. Extend its input-scale/padding matrix; peak-memory evidence is
  now present for cases 5, 8, and 13.
- Treat Person 3's current functional FFN as a rejected control. Attempt packed
  QKV only as an end-to-end case-8 experiment against the 1.118x dispatcher
  route, then remeasure if it wins.
