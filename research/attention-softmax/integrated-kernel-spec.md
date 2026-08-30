# Implementation Spec — Phase 2, the Integrated Polynomial Attention Kernel

Phase 2 of the fused polynomial attention work for official case 14. Phase 1 —
the two feature-map kernels — is specified in
[`triton-kernel-spec.md`](triton-kernel-spec.md) and shipped at commit `0460d04`
on branch `fused-kernal`, measured at **328.1 ms / 4.31x over exact Flash at
B=2**.

Status: draft as of 30 August 2026. Author: Person 2.
Motivating measurements:
[`../benchmarks/2026-08-30-rtx4060-6dc9639/`](../benchmarks/2026-08-30-rtx4060-6dc9639/).

**This spec supersedes section 8 of [`triton-kernel-spec.md`](triton-kernel-spec.md).**
That section proposed a two-level sequence-parallel scan. Section 7.1 below
explains why that design solves a problem the shipped code does not have, and
why it is not being built. Section 8 of the Phase 1 spec is preserved unchanged
and marked superseded in the topic index, per the repository's non-destructive
research policy.

## 1. Scope

**In scope:** official case 14 only (`B=32, d_model=1024, H=16, N=100000, L=2,
causal, ffn=1024`, head dimension 64), float16, CUDA capability 8.0+. The unit
of work is one sample-pair x one layer at `B=2`, which is what case 14's route
streams.

**Out of scope:** cases 1-13. Commit `6dc9639` measured the route against them
and found it 14-100x *slower*, because the method trades `O(N^2 d)` for
`O(N d^3)` and those shapes all sit below the crossover at `N = 2 d_h^2`. Also
out of scope: changing the approximation itself, the `sigma` guard, the
`SIGMA_CEILING` calibration, and the opt-in/reversibility contract — all four
are inherited from Phase 1 unchanged.

**What Phase 2 changes:** the boundary between Triton and PyTorch. Phase 1 fused
the two operations that materialised a `[C, D*D]` feature tensor and left the
rest of the chunk step in PyTorch. That remaining PyTorch work is now
approximately half of the runtime.

## 2. The measurement this design rests on

Kernel-level profile of the shipped path at `B=2, N=100000`, strided inputs.
Full record and raw output in
[`../benchmarks/2026-08-30-rtx4060-6dc9639/`](../benchmarks/2026-08-30-rtx4060-6dc9639/).

| what | ms | share |
| --- | ---: | ---: |
| `_quad_update_kernel` + `_quad_apply_kernel` | 136.6 | **51.1%** |
| `exp` on the `[M, C, C]` diagonal block | 22.1 | 8.3% |
| `masked_fill_` on the same block | 13.4 | 5.0% |
| all GEMM families (cuBLAS/CUTLASS) | 31.3 | 11.7% |
| dtype conversion and copy | 20.5 | 7.7% |
| scalar multiplies (`c0*`, `c1*`, `c2*`) | 13.4 | 5.0% |
| adds | 12.9 | 4.8% |
| reductions (row sums) | 12.0 | 4.5% |
| divide, flash prefix, mask construction | 5.3 | 2.0% |

**These are profile-scale milliseconds, totalling 267.6 ms. They are not
latency.** Section 2.2 explains why. The authoritative interleaved baseline is
**328.1 ms**, and every target in section 6 is expressed as a *share* or a ratio
so that the two scales are never mixed.

Two facts drive every decision below.

**Fact 1 — half the time is glue.** Phase 1's kernels are 51% of GPU time. The
rest is the PyTorch code around them: ~9700 kernel launches per forward, of which
~7400 are elementwise or reduction work of a few microseconds each. Excluding the
diagonal block, that glue is **~21% of the path**.

**Fact 2 — the diagonal block is the single largest line item after the two
kernels, and most of it is discarded.** It cuts across three lines of the table
above — `exp`, `masked_fill_`, its row sum, and two of the GEMMs. The
confidently attributable part (`exp` + mask + row sum) is 40.1 ms; measured in
isolation at the real per-chunk shape the whole block is 95.7 ms, inflated by
`do_bench`'s cache flushing. **The honest bracket is 65-95 ms, or 25-35% of the
path.** It computes the full `C x C` score matrix and then masks half of it
away, so `exp`, `masked_fill_`, both GEMMs and the row sum all pay for the
discarded upper triangle.

### 2.1 Two negative results that constrain the approach

**The tile schedule has no headroom.** `BC` in {32…512}, `BI` in {1…8},
`num_warps` in {4, 8}, `num_stages` in {2, 3} were swept — far outside the 12-
and 8-config spaces the kernels ship with. Nothing beat the shipped
configuration by a margin that survives this machine's measurement spread. Phase
2's gains must come from structure, not from tuning.

**The causal skip cannot be done in PyTorch.** Sub-blocking the diagonal into
lower-triangular tiles is numerically equivalent (`max|dnum|` = 7.8e-03, one fp16
ulp; `max|dden|` = 0) and **2.7x slower**: 10 tiles x 5 operations is 50 launches
per chunk against 5. Tile-level causality only pays inside a kernel.

### 2.2 The measurement hazard, and what it forces

Four profiles of an identical workload, back to back on this machine, measured
280.3, 263.7, 571.1 and 267.6 ms — a **2.17x spread**. In all four, the two
Triton kernels' *share* was 51-55%.

**Proportional attribution is trustworthy on this hardware; absolute latency is
not.** Phase 1's acceptance run states plainly that no noise floor was ever
established ("256-305 ms across repeats"). A 1.15x improvement is
indistinguishable from a measurement artefact without one. Section 4 makes
establishing that floor the first task of Phase 2, before any A/B runs.

## 3. Structure

Two stages, each with its own gate.

**Stage 0 — the redundancy fixes.** Six changes to the shipped path, each
individually A/B'd and kept only if it pays above the noise floor. Stage 0 is
not a warm-up: it produces the baseline Stage 1 is measured against, and its
first fix (the causal-tiled diagonal block) *is* the component Stage 1A folds
into its query-side kernel.

**Stage 1 — the integrated kernel.** Design A collapses the ~15 operations per
chunk into two complete kernels. Design B, the persistent-slab scan, is
specified but built only if Stage 0's re-profile shows its premise still holds.

Stage 0 must complete and be re-profiled before Stage 1's design is chosen. That
ordering is deliberate: F1 alone may move the diagonal block from 25-35% of the
path to under 10%, which changes what Stage 1 should target.

## 4. Stage 0

### F0 — Establish the noise floor (blocking)

No A/B in this spec is decidable without it. Extend `src/bench_poly.py` to report,
alongside each variant's time, the observed spread over `r` interleaved
repetitions, and record a **minimum detectable effect**. Any Stage 0 or Stage 1
result inside that band is reported as "no measurable change", never as a win.

The protocol is fixed here so later runs are comparable: variants interleaved
one rep at a time (never back to back), at least 5 reps after one warm-up call,
report the minimum and the full spread, and re-establish the floor at the start
of every session rather than reusing an earlier one. The `2026-08-30-rtx4060-poly`
record documents that measuring variants sequentially inflated the last one by
~30%; the section 2.2 profiles show 2.17x. This is the dominant experimental risk
in the whole of Phase 2.

### F1 — Causal-tiled diagonal block

**The largest single fix.** Replace

```python
w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
num = num + (w @ vc).to(state_dtype)
den = den + w.sum(-1, keepdim=True, dtype=torch.float32).to(state_dtype)
```

with a Triton kernel gridded over `(C/BC, M)`. Program `(i, m)` loops over key
tiles `j` in `[0, i]` only, applies the causal mask on the diagonal tile alone,
and accumulates `num` and `den` in registers.

- Visits 62% of the tiles at `BK=128`, 56% at `BK=64`.
- Removes the separate `exp`, `masked_fill_` and row-sum launches, and the
  `[M, C, C]` round trip through HBM between them.
- Removes the cached `blocked_full` mask tensor entirely.

Target: from 25-35% of the path down to under 12% of it — on the profile scale,
65-95 ms down to 25-35 ms.

**Numerics.** Today `exp` is applied to an fp16 score tensor and the resulting
weights stay fp16 into the PV product; only the row sum accumulates in fp32. A
Triton kernel would naturally keep scores in fp32 through `exp`, which is *more*
accurate — and conclusion 2 of the topic index records that "more accurate" is a
real failure mode on cases 1-13, where the reference rounds probabilities back
to fp16 and a better kernel therefore deviates from it. Case 14's tolerance is
far wider and its errors already sit at the fp16 ulp floor, so this should be
safe, but it is not assumed:

- The kernel computes scores with fp32 accumulation, applies `exp` in fp32, then
  **rounds `w` to fp16 before the PV dot**, matching the reference's behaviour.
- `den` accumulates in fp32, as today.
- The full section 6.1 correctness table is re-run.

F1 is written as a standalone `@triton.jit` device function from the outset, so
Stage 1A calls it rather than reimplementing it — the same discipline Phase 1
applied to `_phi_tile`.

### F2 — Skip fully-prefixed chunks

Chunks entirely inside `exact_prefix` have their output overwritten by the
closing SDPA call. At `exact_prefix=4096, C=512` that is chunks 0-7 of 196.

For `t0 + C <= exact_prefix`, skip the numerator, denominator, diagonal block and
output store. **The state update still runs** — later chunks depend on it. Saves
~4% of apply and diagonal work.

Guard it on the actual values rather than the constants, so it stays correct if
`C` changes in Stage 1.

### F3 — The apply-side state re-read

`_quad_apply_kernel` is gridded over `(C/BC, M)`, and **every program reads the
entire `[D*D, V]` state**. At the chosen `BC=64` with `C=512` that is 8 full
reads of 1 MiB per `(batch, head)` per chunk. The state working set is 16 MiB at
B=1 and 32 MiB at B=2, against exactly 32 MiB of L2 — so this is surviving on
cache residency, and it is the first thing to degrade if `B` or `H` grows.

Two independent fixes, A/B'd separately and then together:

1. **Larger `BC`.** `BC=128` halves the re-reads, `BC=256` quarters them. Bounded
   by the `[BC, BV]` fp32 accumulator's register pressure.
2. **An fp16 shadow of `s_quad`,** written by the update kernel and read by the
   apply kernel. This is **bitwise-identical** to today, because the apply kernel
   already does `s.to(tl.float16)` before the dot — the rounding is the same
   fp32-to-fp16 round-to-nearest, just moved. It halves the apply kernel's state
   bytes at the cost of one extra 512 KiB write per `(batch, head)` per chunk.

The master state stays float32. That is non-negotiable and unchanged from Phase
1: an fp16 master passes at N=16384 and fails at N=65536 with 1,064,935 failures.
The shadow is a read-only fp16 *copy* of an fp32 master, which is a different
thing from an fp16 accumulator.

### F4 — Pre-baked configs instead of autotune

The kernels ship with a deliberately narrow autotune space (12 and 8 configs,
`num_stages` pinned at 2). The justification recorded in the module docstring is
sound — case 14 is a single forward pass, so compile time lands directly in the
measured wall clock, and a wider space ran for minutes at 1% GPU. But it treats
a symptom. `restore_value=["o_ptr"]` additionally makes every update-kernel trial
clone the 32 MiB state.

Replace autotune on the shipped path with a table keyed on
`(shape key, device capability)`, populated from an **interleaved** offline
sweep, with a narrow autotune retained only as the unknown-key fallback. This
removes the compile cost *and* the artificial narrowness in one move.

The sweep that produced each entry is recorded under `research/benchmarks/`.
The existing sequential sweep is not usable for this and must be redone —
see section 2.1.

### F5 — Per-chunk conversions and `z_const`

`s_lin`, `z_lin` and `gram` are converted fp32 to fp16 on every one of the 195
chunks (part of the 3326 conversion launches, ~20.5 ms). Keep fp16 shadows
updated alongside the fp32 masters, on the same bitwise argument as F3. Replace
the `z_const` device tensor with the Python scalar `t0`, which is what it holds.

**Lower priority than F1-F4**, because Stage 1A subsumes most of it. Do it only
if it is cheap, and skip it if Stage 1A is reached quickly.

### F6 — The redundant `ai` / `bi` loads

Both kernels re-load `ai` (a column subset of `a`, already resident) from global
memory on every loop iteration. Hoist it if Triton can express the slice cheaply
— `tl.static_range` with a static `offs_i` may allow it.

**Expected to be dropped.** These loads are L1-served and the sweep found no tile
schedule that changes the kernels' cost materially. F6 is listed so the
observation is recorded and the negative result is measured rather than assumed.

### Stage 0 gate

Re-profile after Stage 0 and record the new attribution. Proceed to Stage 1 only
with that profile in hand; it decides between designs A and B.

## 5. Stage 1

### 5.1 Design A — two complete per-chunk kernels

Fifteen operations per chunk become two. PyTorch retains the chunk loop and
nothing else.

```text
poly_chunk_query[(C/BC, M)]              poly_chunk_state[(D/BI, M)]
  num  = c0 * s_const                      acc = 0                     [BI*D, BV]
  den  = c0 * z_const                      for c tiles:
  num += c1 * dot(a, s_lin)                  phi = phi_tile(b, bi)
  den += c1 * sum(a * z_lin)                 acc += dot(phi.T, v)
  den += c2 * sum(dot(a, gram) * a)        s_quad[slab] += acc
  for i0 in range(0, D, BI):               if pid_i == 0:
      phi = phi_tile(a, ai)                  s_const += sum(v)
      num += c2 * dot(phi, s_quad[slab])     s_lin   += dot(b.T, v)
  num, den += causal_diag(a, b, v)          gram    += dot(b.T, b)
  out[t0:t1] = num / den                    z_lin   += sum(b)
```

- `phi_tile` is Phase 1's shared device function, unchanged.
- `causal_diag` is F1's device function, unchanged.
- The small states are computed by the `pid_i == 0` program. This is
  load-imbalanced but cheap; the alternative is a third launch per chunk.

**What this removes:** every dtype conversion, every scalar multiply, every
separate reduction, the `num/den` divide, and the `[M, C, V]` intermediate — the
~62 ms of glue plus whatever F1 left of the diagonal block.

**What it does not change:** the `[B, H, N, D]` interface, the chunked scan, the
`sigma` guard, the opt-in flag. Person 4's memory contract holds unchanged — the
resident state is still `[H, D*D, D]`, independent of `N` and `B`, and the
per-chunk working tensors are still `[H, C, D]`.

**Risk: register pressure.** The query kernel holds a `[BC, BV]` fp32 numerator,
a `[BC]` denominator, an `[BC, D]` query tile and a state slab simultaneously.
The sweep already showed `BI >= 4` exhausting shared memory at 131 KB against
the 101 KB limit. If `BC` is forced down to keep the kernel resident, F3's
re-read problem returns through the back door. This is the main reason section
6.2's Stage 1 target is a range rather than a point.

### 5.2 `C` becomes a tuning knob

Phase 1 fixed `C = 512` and forbade tuning it, on the grounds that changing `C`
changes the approximation rather than the schedule. That was right then and is
wrong now.

- **Upward is accuracy-safe.** A larger chunk means more tokens are covered by
  the exact `exp` diagonal block, so the approximation strictly improves.
- **State passes scale as `1/C`.** Doubling `C` halves the number of times the
  state is read and written.
- **The measurement that rejected `C=1024` no longer applies.** It measured
  423.1 ms against 322.7 ms at `C=512` — but only because the diagonal block
  materialised a `[C, C]` tensor whose cost grows as `C^2`. Once F1 tiles it
  causally, that term grows sub-linearly in wall clock.

A/B `C` in {512, 1024, 2048} after F1 lands, and re-run the **full** section 6.1
correctness table at whichever value wins. `C` remains fixed at runtime; it is
not shape-dependent.

### 5.3 Design B — the persistent-slab scan, and its gate

One program per `(m, feature block)`, scanning all 196 chunks with its state slab
resident in registers, so the quadratic state never reaches HBM:

```text
grid = (D/BI, M)
  for chunk t:
      partial = phi_i0(a_t) @ S_slab     -> reduce across the D/BI programs
      S_slab += phi_i0(b_t)^T @ v_t      -> stays in registers
```

The numerator needs a reduction across the `D/BI` programs, which means either
fp32 atomics into an `[M, N, V]` buffer (**+819 MiB**) or a partial-write scheme
with a second pass.

**Design B is built only if both gates pass on Stage 0's re-profile:**

1. State traffic is still the top cost after F1 and F3, and
2. its projected peak VRAM overhead fits the section 6.3 budget.

Gate 1 is not obviously satisfiable. The state working set is 16-32 MiB against
32 MiB of L2, so the traffic B removes is largely L2 traffic, not HBM traffic —
and section 2.1 found no schedule change that moves the kernels materially, which
is weak evidence that they are not bandwidth-starved. Gate 2 is a hard problem:
+819 MiB against a route whose entire current overhead is +67 MiB, on the
constraint that already forced one correction in this stream.

If either gate fails, **B is recorded as rejected with its evidence** and is not
built. That is a legitimate outcome of Phase 2, not a failure of it.

## 6. Acceptance

### 6.1 Correctness

Unchanged from Phase 1, re-run in full **per stage** and again after any change
to `C`:

| oracle | N | required |
| --- | ---: | --- |
| dense reference (`BaselineTransformer`) | 4096, 8192 | 0 failures |
| exact-flash model | 16384, 32768, 65536, 100000 | 0 failures |

Criterion: `abs <= 0.002 OR rel <= 0.02`, **zero** failing elements. Both oracles
are required; the dense reference is authoritative but cannot exceed ~8192 in
8 GiB, and the exact-flash oracle reaches 100000 but cannot see fp16
reduction-order differences.

Kernel-level equivalence against a dense PyTorch oracle, per Phase 1 section 7.1,
extends to the new kernels: `causal_diag` and `poly_chunk_query` /
`poly_chunk_state` over `d in {16, 32, 64}`, `C in {128, 512}`,
`dv in {32, 64}`, max absolute deviation `<= 1e-3` against the fp32 dense result.

The guard is re-validated unchanged: the `SIGMA_CEILING = 0.45` sweep is re-run
and must reproduce, and the fallback must still engage.

### 6.2 Performance

All figures interleaved, against a noise floor established in the same session
per F0, and reported with their spread.

| gate | criterion |
| --- | --- |
| Stage 0 | **>= 1.15x** over the shipped 328.1 ms at B=2 |
| Stage 1 | **>= 1.4x** over the Stage 0 baseline |
| combined | roughly **<= 200 ms**, about **7x** over exact Flash at B=2 |

The Stage 1 target of 1.4x is the number this spec is least confident in. The
share arithmetic supports it: Stage 1A removes the ~21% of the path that is glue,
worth `1/(1 - 0.21)` = **1.27x** on its own, and it removes it from a Stage 0
baseline where F1 has already shrunk the diagonal block, so the glue's share of
that baseline is *larger* than 21% and the ratio correspondingly better. What it
does not account for is section 5.1's register-pressure risk, which could force
`BC` down far enough to give much of the gain back.

**If Stage 1 measures between 1.15x and 1.4x, record it and keep it.** Below
1.15x over Stage 0 the added kernel complexity is not worth carrying, and Stage
1A is reverted to the Stage 0 baseline — the same rule Phase 1 applied when it
set a "record it and do not promote" band rather than an automatic outcome.

### 6.3 Peak VRAM

Overhead above the exact Flash path must stay **<= +100 MiB** at B=2, against the
+67 MiB the route holds today.

This ceiling is set by this spec rather than inherited; no prior document states
one. It is chosen to leave the existing headroom essentially intact, because
VRAM is the constraint that already produced a +6773 MiB regression in this
stream and case 14 needs a 12.21 GiB floor for its device-resident input and
output alone. Design B is inadmissible under it unless its partial buffer is
restructured.

Pinned by the existing `src/tests/test_poly_attention.py::PolyMemoryTests`, which
must keep using **strided** inputs — with contiguous inputs it measures nothing.

## 7. Rejected alternatives

### 7.1 The two-level sequence-parallel scan (Phase 1 spec section 8.1)

**Not built.** Its premise was occupancy: a fully fused kernel is one program per
`(batch, head)`, "16 programs against 24 SMs", which forces a parallel scan over
`G` sequence groups.

That premise does not describe the shipped code. The two kernels grid over
`(C/BC, M)` and `(D/BI, M)` and launch **256 and 2048 programs** at B=2. Design A
keeps the same structure and launches 128-256. Occupancy is not a problem, so the
scan is complexity bought against a constraint that is not binding — and it would
have to store per-group states to do it.

The Phase 1 spec's section 8.2 decisions still hold and are inherited unchanged:
`S_quad` contiguous in `(i, j, v)` order so a feature block is a contiguous slab,
`phi` generation as a shared device function, and fp32 master state with fp16
tile arithmetic.

### 7.2 CUDA-graph capture of the chunk loop

**Not built.** CPU time is well below GPU time today, and Stage 1A cuts launches
per chunk from ~15 to 2, so launch overhead is not binding. Independently, the
`sigma` guard performs a device-to-host synchronization and Phase 1's integration
contract with Person 1 requires it to stay in the eager dispatch layer, never
inside a graph-replayed region.

### 7.3 Widening the autotune space

**Not built as such** — superseded by F4. Widening the space was already tried and
abandoned during Phase 1 (roughly 400 compilations, minutes at 1% GPU). Pre-baked
configs get the benefit without the cost.

## 8. Risks

| risk | severity | mitigation |
| --- | --- | --- |
| Measurement spread swamps the effects being measured | **high** | F0 is blocking; interleaved protocol; effects inside the floor reported as no change |
| Register pressure forces small `BC`, giving back Stage 1's gain | medium | acceptance band in 6.2 rather than a point target; F3's fp16 shadow reduces the cost of a small `BC` |
| fp32 `exp` in F1 makes the kernel more accurate than the reference and it deviates | medium | round `w` to fp16 before the PV dot; full correctness table per stage |
| Stage 1A regresses peak VRAM | medium | 6.3 ceiling; existing `PolyMemoryTests` with strided inputs |
| Larger `C` changes the approximation | low | changes it in the safe direction; correctness table re-run at the chosen `C` |
| fp16 master state reintroduced via the F3/F5 shadows | **high** | shadows are read-only copies of an fp32 master; the N=65536 regression test stays |
| Triton API drift | low | version pinned; availability probe degrades to the PyTorch path |
| Case 14 may not be gradeable at all | high, external | unchanged from Phase 1; tracked as organizer clarification #4 |
| Single-GPU evidence only | medium | unchanged from Phase 1; F4's config table is keyed on device capability so other GPUs fall back rather than run a bad config |

## 9. Integration

**Nothing in the integration contract changes.** Phase 2 is entirely inside the
attention module.

- `src/implementations/extreme.py` — `POLY_ATTENTION_ENABLED = False` remains the
  default, and setting it False still restores exactly the forced-Flash path.
  Pinned by `PolyRouteToggleTests`.
- `src/dispatcher.py` — untouched, as in Phase 1.
- The `sigma` guard's host synchronization stays in the eager dispatch layer.
- Person 4's memory invariants hold: no resident tensor scales with
  `B*N*d_model`, prefix streaming and OOM backoff still drive execution.

`docs/kernel-integration-notes.md` is updated at the end of each stage with the
new measured numbers, not at the end of Phase 2, so Persons 1 and 4 are never
reading a stale figure.

Whether the route is flipped on by default is **not decided by this spec**. It
stays opt-in throughout Phase 2; promotion is a separate decision that belongs
with the repository owner and Person 4.

## 10. Sources

- [`triton-kernel-spec.md`](triton-kernel-spec.md) — Phase 1 design; its sections
  3 (the derivation), 4.4 (precision rules) and 6 (the guard) are inherited
  unchanged. Its section 8 is superseded by this document.
- [`long-sequence-attention.md`](long-sequence-attention.md) — the measurements
  that justify the method, the fp16-state trap in section 5.2, and section 5.5's
  measurement of what happens outside case 14.
- [`triton-kernel-prior-art.md`](triton-kernel-prior-art.md) — prior art and
  licences; `flash-linear-attention` implements this algorithm but asserts head
  dimension <= 16, so no published kernel covers `d_h = 64`.
- [`../benchmarks/2026-08-30-rtx4060-6dc9639/`](../benchmarks/2026-08-30-rtx4060-6dc9639/)
  — the profile, the diagonal-block isolation, and the measurement-spread
  evidence behind F0.
- [`../benchmarks/2026-08-30-rtx4060-poly/`](../benchmarks/2026-08-30-rtx4060-poly/)
  — Phase 1's acceptance run: 328.1 ms / 4.31x at B=2, the guard sweep, and the
  correctness table this spec re-runs.
