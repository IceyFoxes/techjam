# Implementation Spec — Safe Attention Optimizations

Design for the optimizations in the Person 2 stream that gain on **every**
in-scope case and cost no correctness anywhere. Implementation target is
[`src/implementations/attention.py`](../../src/implementations/attention.py).

Status: current as of 29 August 2026. Author: Person 2.
Supersedes the padding contingency in [`review.md`](review.md) section 6 and
conclusion 4 of [`README.md`](README.md); see section 3.

## 1. Scope

**In scope:** official cases 1-13, all dtypes the harness accepts, padding ratios
in `[0, 1)`, and both causal settings.

**Out of scope:** case 14 (`N=100000`), excluded by the repository owner on
29 August 2026 as too large to test on the current hardware. Case 6 (`B=10000`) is
validated for correctness and memory but not for latency; see section 7.3. Compilation and CUDA Graphs remain Person 1's; packed QKV
projections remain Person 3's.

**Definition of safe.** An optimization is safe here when it satisfies all three:

1. it passes the executable criterion (`abs <= 0.002 OR rel <= 2%`, zero failed
   elements) on every in-scope cell of the validation matrix in section 6;
2. it does not slow any in-scope case beyond that case's measured noise floor;
   and
3. its correctness does not depend on an undocumented property of the input.
   Where a fast path does depend on such a property, the property is **tested at
   runtime** and a correct slower route is taken when it does not hold.

Criterion 3 is what makes this spec's central optimization admissible rather
than a gamble; see section 3.

## 2. What is being adopted, and from where

| Lever | Source | Mechanism in this design |
| --- | --- | --- |
| L1 no `S` materialization | [`decomposition.md`](decomposition.md) | `scaled_dot_product_attention` |
| L2 fold scale into `Q` | same | SDPA's internal scaling |
| L3 causal block skip | same | `is_causal=True` |
| L4 online softmax | same | inside SDPA |
| L5 deferred normalization | same | inside SDPA |
| L6 never build the mask tensor | same | `is_causal=True` builds none; eager route caches |
| L7 skip no-op padding mask | same, **generalized in section 3** | route selection |
| L9 strided views, no copies | [`measurements.md`](measurements.md) | `.transpose(1,2)` views, `.reshape()` out |

Levers L8 (reduced-precision internals) and a hand-written Triton kernel are
**not** adopted: both were measured and rejected, L8 as slower and Triton as
redundant against SDPA. Their evidence is in
[`sdpa-and-precision.md`](sdpa-and-precision.md) and
[`fast-attention-survey.md`](fast-attention-survey.md).

## 3. Finding: the padding key mask is dead code under causal attention

This is new to this spec and changes the design materially.

### The argument

`generate_random_case`
([`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py),
line 270) builds `valid_token_mask = positions < lengths[:, None]`. Every
mask it produces is therefore **prefix-valid**: valid tokens form a prefix and
padding is on the right.

Take a valid query row `i`, so `i < length`. Causal masking already sets
`-inf` at every `j > i`. The padding mask sets `-inf` at every `j >= length`.
Since `j >= length > i` implies `j > i`, every entry the padding mask would
write is already `-inf`. The `masked_fill` writes `-inf` onto `-inf` — a no-op,
not an approximation.

Invalid query rows (`i >= length`) genuinely do differ between the two
formulations. They do not matter, because the reference zeroes them itself:
`output.masked_fill(~valid_token_mask[..., None], 0)` in
`BaselineSelfAttention.forward` (line 121), again after the residual in
`BaselineTransformerBlock.forward` (line 144), and once more in
`BaselineTransformer.forward` (line 171). Invalid rows are zero on entry to every
layer, so their divergence cannot propagate.

**Therefore, when the mask is prefix-valid and attention is causal, removing the
padding `masked_fill` is bitwise exact at every element the criterion inspects.**

### The evidence

Measured eager-against-eager, with the padding `masked_fill` as the only
difference, so no reassociation and no SDPA are involved:

| Case | pr=0.0 | pr=0.3 | pr=0.5 | pr=0.9 |
| --- | --- | --- | --- | --- |
| 1 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| 11 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| 12 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| 13 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |

16 configurations, 4 seeds each, up to 51.1% of tokens invalid. `max |reference -
variant|` is exactly zero throughout.

### Why it matters

Setting `is_causal=False` and supplying a full mask forfeits lever L3. Measured
at `padding_ratio=0.3`, paired timing, float32:

| Case | `is_causal=True`, mask dropped | `[B,1,N,N]` bool mask | additive mask |
| --- | --- | --- | --- |
| 13 | **5.234x** ±18.0% | 2.779x ±14.2% | 3.032x ±0.2% |
| 11 | **4.462x** ±1.9% | 3.017x ±13.4% | 2.789x ±2.9% |
| 8 | 1.062x ±0.4% | 1.063x ±0.4% | 1.055x ±3.5% |

All three routes pass with zero failed elements, and `max_abs` is identical
across all three within each case — independent confirmation that the mask
contributes nothing. Case 8 ties because attention is only ~16% of its device
time.

### The real incumbent: `is_causal=True` **with** a broadcast key mask

PyTorch's documentation states `attn_mask` and `is_causal` are mutually
exclusive. **That is not true on either stack this project has used**: on torch
2.6.0+cu124 and on 2.13.0+cu130, passing both is accepted and both are applied.
Verified on each against a hand-computed causal-plus-key-mask reference,
`max_abs` 3.6e-07.

Re-verified on 2.13.0+cu130 with the comparison split by row, which shows the
mechanism above directly: dropping the key mask agrees with ground truth to
`3.3e-07` on **valid** query rows — ordinary SDPA-versus-eager float noise — and
diverges by `1.085` only on the invalid rows the reference zeroes.

This matters because supplying the key mask is what upstream
`src/implementations/sdpa.py` does — both `StridedSDPASelfAttention` (Person 1)
and `PackedQKVSDPASelfAttention` (Person 3). So the comparison that decides this
spec is not against the eager reference but against that module.

### Measured: drop the mask, on every in-scope case

Complete sweep, RTX 4060 Laptop GPU, **torch 2.13.0+cu130 / CUDA 13.0 /
driver 616.56**, float32, TF32 on, paired timing. Candidate milliseconds; "adv"
is how much faster dropping the mask is than keeping it.

| Case | pr=0.0 drop | pr=0.0 keep | adv | pr=0.3 drop | pr=0.3 keep | adv |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.363 | 2.476 | +4.6% | 2.401 | 2.574 | +6.7% |
| 2 | 2.165 | 2.438 | +11.2% | 1.886 | 2.138 | +11.8% |
| 3 | 1.860 | 2.115 | +12.1% | 1.982 | 2.224 | +10.9% |
| 4 | 1.845 | 1.996 | +7.6% | 1.804 | 2.199 | +17.9% |
| 5 | 4.123 | 4.211 | +2.1% | 4.074 | 4.281 | +4.9% |
| 7 | 1.765 | 2.232 | +20.9% | 1.770 | 2.121 | +16.5% |
| 8 | 46.801 | 47.713 | +1.9% | 48.087 | 48.143 | +0.1% |
| 9 | 3.318 | 4.009 | +17.2% | 2.137 | 2.260 | +5.5% |
| 10 | 2.147 | 2.216 | +3.1% | 2.122 | 2.223 | +4.5% |
| 11 | 4.204 | 4.534 | +7.3% | 4.196 | 4.460 | +5.9% |
| 12 | 2.245 | 2.385 | +5.9% | 1.910 | 2.215 | +13.8% |
| 13 | 48.879 | 51.396 | +4.9% | 48.810 | 51.226 | +4.7% |

**Dropping the mask is faster in 24 of 24 comparisons.** Every cell passes with
zero failed elements, and `max_abs` is identical between the two routes within
each case — independent confirmation that the mask contributes nothing.

Read this table with its floors in mind. The trustworthy rows are the large
shapes: case 8 (±0.5-1.0%), case 13 (±1.9-2.2%) and case 5 (±3.0-4.0%), all
consistently positive. Cases 1, 4, 9, 10, 11 and 12 carry floors of ±5-23%, so no
individual figure there should be quoted; their contribution is directional.
Case 8 at `pr=0.3` is a tie (+0.1%), which is expected — attention is only ~16%
of that case's device time.

Cases 2 and 3 initially measured **-68%** and **-12.6%** at 25 repeats. Re-run at
120 repeats with a 20 s settle they became **+11.8%** and **+12.1%**. They are the
launch-bound shapes (`B=1`, `B=4`, ~2 ms) that [`measurements.md`](measurements.md)
already flags as unstable, with recorded floors as wide as ±192%. **Treat any
sub-5 ms case measured at fewer than ~100 repeats as unresolved**, and never
compare figures across separate processes.

### Legacy: the torch 2.6.0+cu124 result does not reproduce

An earlier version of this section reported that the incumbent *collapses* under
padding — case 13 at `padding_ratio=0.3` measuring 2.349x against 5.927x for the
drop-mask route, a 219 ms versus 49 ms candidate gap — and used that 2.5x as the
central argument.

**That was specific to torch 2.6.0+cu124 and does not reproduce on 2.13.0+cu130**,
where the same comparison is 51.23 ms against 48.81 ms. torch 2.13 evidently
handles a broadcast key mask alongside `is_causal` efficiently; 2.6 did not.

The team benchmarks on cu130 only, so the cu124 numbers no longer inform any
decision. They are retained here, marked legacy, because they were real on that
build and a contributor still running it would be misled by their absence. The
conclusion survives the correction — dropping the mask still wins everywhere —
but the margin is roughly 5% on the large shapes rather than 2.5x.

### What this supersedes

[`review.md`](review.md) section 6 and [`README.md`](README.md) conclusion 4 state
that the padding-related gain "collapses to ~0.95x once padding is present" and
holds "only when `padding_ratio=0`". That was measured for a narrower lever —
skipping a mask only when it is entirely all-true. Generalized to prefix-valid
masks, the gain does not collapse: padding is free. Those documents are retained
unedited under the repository's non-destructive research policy; this section is
the correction, and the note is carried in
[`README.md`](README.md).

**Scope limit.** The argument requires causal attention. With `causal=False`
there is no upper-triangular mask to subsume the padding mask, and the padding
mask must be applied. All 14 disclosed cases are causal, but the design must not
assume it.

## 4. Architecture

### 4.1 Route selection

One decision per forward, taken on the host before the layer loop, from
`(dtype, causal, mask_kind)`:

| dtype | causal | mask kind | Route | Attention call |
| --- | --- | --- | --- | --- |
| float32 | any | absent | `SDPA_CAUSAL` | `sdpa(q,k,v, is_causal=causal)` |
| float32 | True | prefix-valid | `SDPA_CAUSAL` | `sdpa(q,k,v, is_causal=True)` — mask dropped; faster in 24/24 measured comparisons (section 3) |
| float32 | False | prefix-valid | `SDPA_KEYMASK` | `sdpa(q,k,v, attn_mask=keep[B,1,1,N])` |
| float32 | True | general | `SDPA_FULLMASK` | `sdpa(q,k,v, attn_mask=keep[B,1,N,N])` |
| float32 | False | general | `SDPA_KEYMASK` | as above |
| non-float32 | any | any | `EXACT_EAGER` | reference arithmetic, L6 + L7 |

`mask_kind` is `absent` when `valid_token_mask is None`, `prefix-valid` when the
mask is non-increasing along the sequence axis, and `general` otherwise. An
all-true mask is prefix-valid, so it needs no separate test: with `causal=True`
it takes `SDPA_CAUSAL` like any other prefix mask, and with `causal=False` it
takes `SDPA_KEYMASK`, where a mask that happens to keep everything is correct and
costs one broadcast `[B,1,1,N]` tensor. Distinguishing it would buy nothing that
the disclosed cases can reach, since all fourteen are causal.

`SDPA_CAUSAL` is the only route the official harness can reach in float32, since
`generate_random_case` emits prefix-valid masks exclusively.
`SDPA_CAUSAL_KEYMASK` computes the same thing and is retained solely as the
incumbent control for A/B measurement, since it reproduces upstream
`StridedSDPASelfAttention` behavior exactly. It is not a routing alternative:
section 3 measured it slower on every in-scope case. The other three
exist so that criterion 3 of section 1 holds: correctness never depends on the
generator's behaviour, only performance does.

`EXACT_EAGER` exists because float16 SDPA fails the criterion — 0/8 seeds on
case 13 — for reasons that are a property of the reference's own rounding, not of
this implementation ([`sdpa-and-precision.md`](sdpa-and-precision.md)). That route
still takes L6 and L7, both bitwise exact in any dtype. L7 applies in its
section-3 generalized form — skip the padding `masked_fill` whenever the mask is
prefix-valid — but **only under `causal=True`**. With `causal=False` the eager
route must apply the padding mask, and L7 narrows back to the all-true case.

### 4.2 The prefix-valid predicate

```
prefix_valid = bool((mask[:, :-1] >= mask[:, 1:]).all())
```

A mask is prefix-valid exactly when it never rises along the sequence axis. One
elementwise comparison and one reduction over `B x (N-1)` booleans, followed by a
**single** device-to-host synchronization per forward.

Cost control is not optional here.
[`measurements.md`](measurements.md) records that evaluating an equivalent
predicate once per layer instead of once per forward turned a 1.15-1.43x gain
into a 0.82-0.95x loss. The predicate is therefore evaluated in the model's
`forward`, above the layer loop, and the resulting route is passed down. It must
never be evaluated inside a layer.

### 4.3 Module layout

Upstream now provides `StridedSDPASelfAttention` in the shared
`src/implementations/sdpa.py` (Person 1) and `PackedQKVSDPASelfAttention`
(Person 3), both wired into a compiling `src/dispatcher.py`. Both **always**
supply the broadcast key mask. This design therefore does not reimplement SDPA;
it subclasses that module and changes only which mask is supplied.
`src/implementations/attention.py` gains four units:

| Unit | Responsibility | Depends on |
| --- | --- | --- |
| `classify_mask(mask) -> MaskKind` | the one host sync; returns `ABSENT`/`PREFIX`/`GENERAL` | torch |
| `select_route(dtype, causal, mask_kind) -> Route` | pure function, the section 4.1 table | nothing |
| `MaskRoutedSDPASelfAttention(StridedSDPASelfAttention)` | per-route attention call; inherits strided views, scale, reshape and output zeroing from upstream | the two above, plus `sdpa.py` |
| `AttentionCandidate(BaselineTransformer)` | classifies once, swaps in `MaskRoutedSDPASelfAttention`, threads the route | all of the above |

`select_route` is a pure function over three scalars, so the routing table is
testable without a GPU. `classify_mask` is the only unit that synchronizes, which
keeps the sync auditable in one place.

Parameter names and module structure are unchanged, so `copy_model_weights`
continues to load with `strict=True` and no custom `weight_loader` is needed.

### 4.4 Dispatcher-ready surface

Per the landing-zone decision, this module is built so Person 1 adopts it by
import rather than by reimplementation. It exports:

- `SdpaTransformer` — a drop-in `BaselineTransformer` subclass;
- `classify_mask` and `select_route` — so the dispatcher can hoist the host
  decision above a compiled region and select a separately compiled callable per
  route, which is exactly the structure
  [`dispatcher-strategy.md`](../framework-fastpaths/dispatcher-strategy.md)
  requires ("make that decision once outside the compiled inner model and route
  to a separately compiled callable"); and
- `CANDIDATE` — the `CandidateSpec` for `--candidate attention`.

No edits to `src/dispatcher.py` or `src/implementations/sdpa.py` are made by
this spec; both are now shared modules owned by Persons 1 and 3.

**Integration constraint.** `sdpa.py` deliberately refuses to inspect mask values
on the host, because that synchronizes and breaks CUDA-graph replay — and the
dispatcher runs cases 1-12 under `reduce-overhead`. `classify_mask` must
therefore be called in the **uncompiled** dispatch layer, before the compiled
callable is invoked, never inside the captured region. The sync costs roughly
10-20 us: noise against case 13 (~25-60 ms), around 10% against case 2
(~0.14 ms). That asymmetry is a further reason the route is selected per case.

### 4.5 Layout at the Person 3 boundary

Inputs use `x.view(B, N, H, d_h).transpose(1, 2)` — a view, no copy, removing
three full `[B,N,D]` copies per layer (lever L9). Output uses
`ctx.transpose(1, 2).reshape(B, N, D)`.

This satisfies the qkv-layout contract's requirement that "no unconditional
`.contiguous()` is permitted at the boundary"
([`qkv-layout.md`](../projections-ffn-fusion/qkv-layout.md)). The output
`reshape` may still copy when SDPA returns head-major storage; that copy is
charged to this backend, and the run records must report the observed output
stride so Person 3 can decide whether a packed producer removes it.

## 5. Expected performance

From [`measurements.md`](measurements.md) at `padding_ratio=0`: float32 SDPA
alone is a measured geometric mean of **1.94x** over the twelve in-scope cases.
Strided views were measured on four of those cases at +6-12%, which implies
roughly **2.1x** overall — an extrapolation from four cases, not a measurement of
twelve, and it is section 7's job to replace it. Measured endpoints with strided
views are 1.119x (case 8) and 6.908x (case 13). Section 3's measurements extend this to
`padding_ratio=0.3` at 5.234x / 4.462x / 1.062x on cases 13 / 11 / 8, and
section 7.3 adds case 6 at 2.032x under memory oversubscription.

These are prior measurements and exploratory spike numbers. They set the
expectation; section 7 defines what must be re-measured before any of it is
claimed.

## 6. Correctness validation matrix

Every cell requires **zero failed elements** under the harness's own checker.

| Axis | Values |
| --- | --- |
| Case | 1-13 |
| dtype | float32 (primary); float16 and bfloat16 for `EXACT_EAGER` |
| `padding_ratio` | 0.0, 0.3 |
| `input_scale` | 0.1, 1.0, 10.0 |
| Seeds | >= 5 per cell |
| TF32 | on (harness default) and off |

Three axes deserve comment:

- **`input_scale`** is listed as untested in [`review.md`](review.md) section 6.
  It moves output magnitudes and therefore which branch of the `abs OR rel`
  criterion applies, so it is measured here rather than assumed.
- **float16 and bfloat16** must be exercised to prove `EXACT_EAGER` is bitwise
  exact (`max_abs == 0`), not merely within tolerance. A non-zero `max_abs` on
  that route is a defect, not a pass.
- **TF32 off** establishes the headroom that TF32 on consumes: ~1600x margin
  drops to ~1.5-2x ([`sdpa-and-precision.md`](sdpa-and-precision.md)). Both are
  recorded so the margin is visible.

Additionally, the section 3 no-op proof is promoted from a spike script into a
regression test: eager reference against eager-without-padding-mask must be
bitwise identical across padding ratios `{0.0, 0.3, 0.5, 0.9}`. If a future
harness change breaks prefix-validity, that test fails loudly and
`classify_mask` routes around it rather than producing a wrong answer.

## 7. Benchmark and acceptance

### 7.1 Method

Paired timing via [`src/infra/timing.py`](../../src/infra/timing.py), >= 30
repeats, >= 10 s settle, with the reported noise floor. As of 30 August 2026 the
measurement stack is Python 3.12.14 / torch 2.13.0+cu130 / CUDA 13.0 on driver
616.56 — identical to Person 1's RTX 5080 box, so route decisions now differ
between the two machines only by GPU, not by software. Per that module's own
rule, **a speedup inside its noise floor is not a speedup**. Cases 2 and 9 carry
floors of ±14.7% and ±18.9% in earlier work and need more repeats before their
numbers are quoted individually.

### 7.2 Acceptance criteria

1. Every cell of section 6 passes with zero failed elements.
2. No in-scope case is slower than the reference beyond its own noise floor.
2b. `SDPA_CAUSAL` is measured against `SDPA_CAUSAL_KEYMASK` (the upstream
   control) on a concurrently interleaved baseline **in the same process**, at
   `padding_ratio` 0.0 and 0.3. Numbers from separate runs are not comparable,
   and any case under ~5 ms needs >= 100 repeats before its result is believed
   (section 3).
3. The `EXACT_EAGER` route reports `max_abs == 0` in every dtype.
4. Every preserved run records the SDPA backend actually selected. If the math
   backend is chosen it materializes the `N x N` scores and defeats lever L1 —
   that is a silent regression and must be caught by inspection, not by timing.
5. Peak allocated memory is recorded for cases 5, 6, 8 and 13.
6. Runs preserved under `research/benchmarks/2026-08-29-rtx4060-<commit>/` with
   the fields `AGENTS.md` requires, and `research/benchmarks/README.md` updated.

### 7.3 Case 6 — measured, with a memory caveat

Case 6 (`B=10000`) is nominally Person 4's. It is validated here because SDPA
addresses exactly its failure mode: the eager path materializes a ~2.6 GB float32
score tensor that SDPA never allocates.

Exploratory result, `padding_ratio=0.3`, float32, route `SDPA_CAUSAL`, 10 paired
repeats:

| Metric | Value |
| --- | --- |
| Correctness | PASS, 0 failed elements, `max_abs` 7.42e-04 |
| Baseline | 6470.0 ms |
| Candidate | 3184.2 ms |
| Speedup | **2.032x**, floor ±2.3% |
| Peak allocated | 11,901 MiB |

**The peak allocation exceeds the device.** This card has 8,188 MiB, so 11.9 GiB
of peak allocation means the run only completed because WSL2 permitted
oversubscription into host memory. Two consequences:

1. The 2.032x is a **valid correctness result but an unreliable latency result**.
   Both sides were paging over PCIe, which compresses the ratio between a
   memory-hungry baseline and a memory-lean candidate. On a card that fits the
   working set, the gap should widen, not narrow — but that is a prediction, not
   a measurement, and must not be quoted as one.
2. Case 6 confirms the *direction* — removing the score tensor is what that shape
   needs — without settling its final route.

Acceptance for case 6 is therefore correctness plus a recorded peak-memory
figure, both now obtained, and it is handed to Person 4 with the oversubscription
caveat attached. It does not gate the other twelve cases. Any latency claim for
case 6 requires a device that holds the working set resident.

## 8. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Organizer evaluates in float16 | High | `EXACT_EAGER` route keeps correctness; gain drops to the bitwise-exact levers. Root cause is the reference's own rounding and is worth raising with the organizer |
| Harness emits non-prefix masks in future | Low | `classify_mask` detects it at runtime and routes to `SDPA_FULLMASK`, still 2.8-3.0x on case 13 |
| SDPA picks the math backend | Medium | Recorded per run (criterion 4); the math backend defeats L1 |
| Host sync hurts launch-bound cases 2, 3, 12 | Medium | One sync per forward, not per layer. If it still costs more than it saves on those cases, they are launch-bound and belong to Person 1's compilation work regardless |
| TF32 leaves only ~1.5-2x headroom | Medium | Matrix includes TF32 on and off; no further precision reduction is taken |
| Case 6 exceeds device memory | Medium | Measured: 11,901 MiB peak on an 8,188 MiB card, completing only via WSL2 host-memory oversubscription. Correctness holds; its latency number is not transferable. See 7.3 |

## 9. Handoffs

- **Person 1:** `StridedSDPASelfAttention` in `src/implementations/sdpa.py`
  always supplies the broadcast key mask. That is correct, but under causal
  attention the mask is dead code, and dropping it measured faster in **24 of 24**
  comparisons on cu130 — every in-scope case at `padding_ratio` 0.0 and 0.3.
  The margin is ~2-5% on the large shapes (case 13: 48.81 ms against 51.23 ms)
  and larger on the small ones. Section 3 proves the change is bitwise safe. The
  proposed one-line diff is in the plan's Task 8; it is a uniform change, not a
  per-case selection.
  `classify_mask` / `select_route` are exported for hoisting above a
  compiled region. The composition of SDPA with `reduce-overhead` and `default`
  remains unmeasured and is the largest open question in
  [`dispatcher-strategy.md`](../framework-fastpaths/dispatcher-strategy.md);
  this module is the object that A/B needs.
- **Person 3:** the output `reshape` may still copy; observed output strides are
  reported per run so the packed-producer decision can be made on evidence.
- **Person 4:** section 3 applies to cases 6 and 14 as well — under causal
  attention with right padding, no padding mask need ever be built, which removes
  a `B x N` term from any chunked design.

## 10. Sources

- **Reference implementation:** [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py)
  at commit `f9a52b6`. Symbols: `BaselineSelfAttention.forward` lines 85-122
  (stages 3-7 and the output zeroing at line 121),
  `BaselineTransformerBlock.forward` line 144, `BaselineTransformer.forward`
  line 171, `generate_random_case` line 270 (prefix-valid mask
  construction), and `compare_outputs` lines 289-353 (the pass criterion).
- **Prior stream measurements:** [`measurements.md`](measurements.md),
  [`sdpa-and-precision.md`](sdpa-and-precision.md),
  [`decomposition.md`](decomposition.md), all current as of 29 August 2026.
- **Boundary contracts:** [`qkv-layout.md`](../projections-ffn-fusion/qkv-layout.md)
  and [`dispatcher-strategy.md`](../framework-fastpaths/dispatcher-strategy.md),
  both 29 August 2026.
- **Section 3 measurements:** exploratory spike runs on this repository, RTX 4060
  Laptop GPU, float32, TF32 on. First pass at commit `f9a52b6`, 29 August 2026,
  torch 2.6.0+cu124. Re-run at commit `a06be92`, 30 August 2026, torch
  2.13.0+cu130 / CUDA 13.0 / driver 616.56, after this host was migrated to the
  team-pinned stack. Scripts were throwaway; their durable replacements are the regression
  test in section 6 and the benchmark commands in section 7.
