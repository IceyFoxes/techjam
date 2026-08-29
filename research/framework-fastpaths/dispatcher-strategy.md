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

| Cases | Characteristics | Current preferred route | Status / reason |
| --- | --- | --- | --- |
| 1 | ordinary, B=64, D=128, N=128 | float32 SDPA + strided views inside `reduce-overhead` | Candidate composition; compiler-only exploratory 2.713x, padding-safe exploratory check |
| 2 | smallest batch, launch-bound | float32 SDPA + strided views inside `reduce-overhead`; dtype fallback | Compiler route **measured** at 6.581x fp32 and 9.250x fp16; max-autotune gives no runtime advantage and less tolerance headroom; integrated SDPA still unmeasured |
| 3 | small batch, launch-bound | float32 SDPA + strided views inside `reduce-overhead` | Candidate composition; compiler-only exploratory 5.193x |
| 4 | medium batch | float32 SDPA + strided views; compare default vs reduce-overhead | Unmeasured integrated route; do not infer compiler mode from case 3 |
| 5 | large ordinary batch | float32 SDPA + strided views; default vs reduce-overhead with peak-memory check | Unmeasured integrated route; CUDA Graph workspace may matter |
| 6 | B=10,000 extreme batch | Person 4 memory-safe backend; compile only inside stable chunks | Unmeasured; do not graph-capture full buffers by default |
| 7 | D=32, head dim 8 | float32 SDPA + strided views inside compiled fusion; dtype-specific fallback | fp16 raw compile **rejected** by exploratory 8/8 failure; cast emulation promising only for this case |
| 8 | D=1024, head dim 256, GEMM-bound | float32 SDPA + strided views, Person 3 projection backend, optionally `reduce-overhead` | Compiler control **measured** 1.095x; **reject max-autotune** (5/5 failures); corrected Person 2 whole-model result shows SDPA is a small win |
| 9 | one head, head dim 128 | float32 SDPA + strided views, then compiler comparison | Unmeasured integrated route |
| 10 | two heads, head dim 64 | float32 SDPA + strided views, then compiler comparison | Unmeasured integrated route |
| 11 | 16 heads, head dim 8 | float32 SDPA + strided views, then compiler comparison | Unmeasured integrated route; Person 2's SDPA-only route is especially strong |
| 12 | N=32, launch-bound | float32 SDPA + strided views inside `reduce-overhead` | Candidate composition; compiler-only exploratory 4.635x vs 1.850x default |
| 13 | N=1024, memory-bound attention | float32 SDPA + strided views, then compare eager SDPA vs default compile | Compiler control **measured** 3.179x; CUDA Graph adds no credible gain; float16 compiled route rejected |
| 14 | N=100,000 extreme | Person 4 streaming/chunked backend only | Unmeasured; no full-model compile/CUDA Graph until feasibility and memory are proven |

## Composition Rules

### Person 2 attention

- Person 2's corrected whole-model sweep shows float32 SDPA passes and gains on
  every in-scope case: about 1.94x geometric mean on the RTX 4060 Laptop GPU.
  The earlier case 8 exclusion was based on a flawed attention-only control that
  hoisted mask construction; whole-model case 8 is 1.047x, or 1.119x after
  dropping `.contiguous()` copies.
- Case 13's leading route is float32 SDPA plus strided views because Person 2
  measured 6.908x on a different GPU, versus 3.179x for compiler-only on this
  RTX 5080. The final decision needs same-machine A/B of eager SDPA,
  default-compiled SDPA, and reduce-overhead SDPA.
- Case 8 should also use the float32 SDPA route, but its small attention gain and
  projection-dominated profile make Person 3 composition decisive.
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
