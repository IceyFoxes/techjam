# Shape-Specific Dispatcher Strategy

## Principle

The organizer discloses every test shape and explicitly permits shape checks.
The final dispatcher should therefore select the fastest **validated complete
implementation** for each full tuple. It should not choose a backend from one
dimension alone and should not assume a successful dtype generalizes.

Use this routing key:

```text
(batch, d_model, heads, sequence_length, layers, causal, ffn_dim,
 dtype, device capability, mask presence/route, TF32 and matmul precision)
```

Padding ratio is data rather than a static configuration in the public forward
signature. If an all-valid shortcut requires a host decision, make that decision
once outside the compiled inner model and route to a separately compiled callable
rather than synchronizing once per layer.

## Current Matrix

`Measured` means Person 1 has a preserved or clearly identified exploratory
whole-model result. `Candidate` means the route still requires integrated
benchmark evidence. `Reject` means executable evidence rules the route out.

Final status update, 29 August 2026: the hardened dispatcher at `307eedb`
preserves exact-route evidence for cases 1-5 and 7-13. All pass 5/5 seeds with a
3.548x geometric mean. This resolves the earlier integrated-evidence gaps for
cases 2, 8, and 13. Cases 6 and 14 now fail before allocation rather than enter
an unsafe dense reference fallback.

| Cases | Characteristics | Current preferred route | Status / reason |
| --- | --- | --- | --- |
| 1 | ordinary, B=64, D=128, N=128 | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 2.641x, PASS 5/5; padded smoke also passes |
| 2 | smallest batch, launch-bound | float32 SDPA + strided views inside `reduce-overhead`; dtype fallback | Final dispatcher **measured** at 7.498x, PASS 5/5; exact value has a wide ±127.86% floor |
| 3 | small batch, launch-bound | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 4.389x, PASS 5/5 |
| 4 | medium batch | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 4.081x, PASS 5/5 |
| 5 | large ordinary batch | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 2.918x, PASS 5/5; candidate peak allocated 107.06 MiB |
| 6 | B=10,000 extreme batch | Person 4 memory-safe backend; compile only inside stable chunks | Explicitly unsupported before allocation; no memory-safe backend |
| 7 | D=32, head dim 8 | float32 SDPA + strided views inside `reduce-overhead`; dtype-specific fallback | Final dispatcher **measured** at 3.684x, PASS 5/5; fp16 raw compile remains rejected |
| 8 | D=1024, head dim 256, GEMM-bound | float32 SDPA + strided views inside `reduce-overhead`; future packed-QKV experiment | Final dispatcher **measured** at 1.118x ±3.90%, PASS 5/5; candidate peak allocated 428.34 MiB |
| 9 | one head, head dim 128 | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 2.164x, PASS 5/5 |
| 10 | two heads, head dim 64 | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 2.610x, PASS 5/5 |
| 11 | 16 heads, head dim 8 | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 5.573x, PASS 5/5 |
| 12 | N=32, launch-bound | float32 SDPA + strided views inside `reduce-overhead` | Final dispatcher **measured** at 4.263x, PASS 5/5; exact value has a wide ±59.94% floor |
| 13 | N=1024, memory-bound attention | float32 SDPA + strided views inside default compile | Final dispatcher **measured** at 6.947x, PASS 5/5; candidate peak allocated 1,219.10 MiB |
| 14 | N=100,000 extreme | Person 4 streaming/chunked backend only | Explicitly unsupported before allocation; baseline score alone is 9.313 TiB in FP16 |

## Composition Rules

### Person 2 attention

- Same-machine final-dispatcher evidence now confirms the float32 SDPA +
  strided-view route on all twelve supported cases. All pass five seeds, all
  clear their run-specific noise floors, and their geometric-mean speedup is
  3.548x on the RTX 5080.
- Person 2's corrected whole-model sweep shows float32 SDPA passes and gains on
  every in-scope case: about 1.94x geometric mean on the RTX 4060 Laptop GPU.
  The earlier case 8 exclusion was based on a flawed attention-only control that
  hoisted mask construction; whole-model case 8 is 1.047x, or 1.119x after
  dropping `.contiguous()` copies.
- Case 13's same-machine final route is resolved: float32 SDPA plus strided
  views inside default compile measures 6.947x, with lower peak allocation than
  baseline. CUDA Graph replay remains disabled for this route.
- Case 8 uses the proven float32 SDPA route at 1.118x. Its small gain and
  projection-dominated profile make it the primary future packed-QKV target,
  but Person 3's current functional FFN control must not be integrated.
- Float16 fused attention and whole-model compile have independent numerical
  failures. A pass by one does not authorize the other.

### Person 3 projections and FFN

- Case 8 should prioritize packed QKV and strong library GEMMs. Wrap the complete
  projection-attention-output path in compile only after measuring hidden copies
  and numerical headroom.
- Cases 2, 3, 7, and 12 can benefit from compiler launch and pointwise fusion,
  but explicit eager-rounding checkpoints may be necessary in low precision.
- `max-autotune` must remain disabled unless the exact integrated tuple passes
  the full seed/scale/padding matrix.

### Person 4 extremes

- Do not apply a full-model CUDA Graph to cases 6 or 14 merely because their
  shapes are static. Graph workspace retention can conflict with the primary
  memory-safety objective.
- Compile or graph only stable inner chunks after chunk size, buffer addresses,
  and peak memory are fixed.

## Implementation Shape

The candidate is built on CPU, copied, then moved to device and dtype by the
harness. Compilation must therefore be lazy: on the first forward, determine the
complete key from static config plus `x.device`, `x.dtype`, and mask route; build
or select the specialized callable; then cache it for subsequent calls.

Avoid compiling the routing layer repeatedly. A practical structure is:

```text
dispatcher.forward
  -> resolve static route once
  -> cached eager or compiled backend callable
  -> backend owns all adapters charged to its latency
```

The compiler cache must be per module instance or include parameter identity;
never return a compiled callable bound to another model's weights. Unsupported
devices, dtypes, padding semantics, or compile failures must fall back to the
reference-compatible eager route.

## Promotion Gate

A route enters the dispatcher only when it has:

1. zero failed elements over the required dtype/seed/input-scale/padding matrix;
2. a preserved whole-model benchmark on the target GPU with its noise floor;
3. a meaningful gain over both the immutable reference and the strongest
   simpler integrated route;
4. peak-memory evidence if it uses CUDA Graphs or serves cases 5, 6, 8, 13, or
   14;
5. graph-break/recompile logs showing stable capture after warmup; and
6. an eager-compatible fallback for every key it does not explicitly support.

The `307eedb` checkpoint satisfies the ordinary five-seed, target-contract,
whole-model timing, graph-replay warmup, and supported-case memory gates. It
does not yet satisfy adversarial input-scale/padding coverage. Cases 6 and 14
are outside this promotion because an eager-compatible dense fallback is not
memory-safe at their shapes.

## Sources

- Organizer shape table and permission for shape checks:
  [`TASK.md`](../../TASK.md), sections 3.2 and 3.7, accessed 29 August 2026.
- Team ownership and integration contract:
  [`four-way-team-split.md`](../team-coordination/four-way-team-split.md), current
  status 29 August 2026.
- Person 2 SDPA decisions:
  [`../attention-softmax/README.md`](../attention-softmax/README.md), current
  status 29 August 2026.
- Person 3 layout contract:
  [`../projections-ffn-fusion/qkv-layout.md`](../projections-ffn-fusion/qkv-layout.md),
  current status 29 August 2026.
- Local immutable benchmark at revision
  `6bde871dd65051fcace36971b27a86771365ba1e`, symbols
  `TransformerConfig`, `BaselineTransformer.forward`, `maybe_compile`, and
  `compare_outputs`, accessed 29 August 2026.
