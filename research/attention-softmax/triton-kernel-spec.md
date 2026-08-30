# Implementation Spec — Fused Polynomial Attention Kernel (Case 14)

Design for the Triton kernel proposed in
[`long-sequence-attention.md`](long-sequence-attention.md) section 6.
Implementation target is a new `src/kernels/` module plus a candidate in
`src/implementations/`.

Status: draft as of 30 August 2026. Author: Person 2.
Prior art and the constraints that shape this design are in
[`triton-kernel-prior-art.md`](triton-kernel-prior-art.md).

## 1. Scope

**In scope:** official case 14 only (`B=32, d_model=1024, H=16, N=100000, L=2,
causal, ffn=1024`, head dimension 64), float16, on CUDA capability 8.0+.

**Out of scope:** cases 1-13, where attention is memory-bound and SDPA already
wins ([`README.md`](README.md) conclusions 1-8); case 6, which is Person 4's and
is not sequence-limited; training and backward passes, since the benchmark is
inference-only.

**Why a custom kernel is justified here and nowhere else.** The survey's closing
recommendation was *not* to write one ([`fast-attention-survey.md`](fast-attention-survey.md)).
That holds for `N <= 1024`. At case 14 the order-2 polynomial path already
performs **12x fewer FLOPs** than exact attention (1.7 against 20.5 TFLOP per
sample-layer) yet realises only **1.19x**, because it writes and re-reads roughly
**51 GB of feature tensor per sample-layer**. The gap between 12x of arithmetic
and 1.19x of delivered speed is the entire justification.

## 2. Phasing

**Phase 1 (this spec's deliverable).** Fuse the feature-times-state GEMM. The
`a (x) a` tile is generated in registers and consumed directly, so it never
reaches HBM. PyTorch retains the chunk loop and the exact diagonal block.

**Phase 2 (specified, not built).** Fuse the whole chunked scan, including the
diagonal block and the state update, using a two-level sequence-parallel scan.
Section 8 fixes the decisions Phase 1 must not foreclose.

Phase 1 is chosen first because it targets the measured bottleneck and, crucially,
**it does not have the occupancy problem Phase 2 has**. Phase 1's kernels are
gridded over chunk rows x heads (kernel 1) and feature blocks x heads (kernel 2),
giving 256 and 512 programs per launch at the section 4.5 starting block sizes. A
fully fused kernel is instead one program per `(batch, head)` — 16 programs
against 24 SMs — which is what forces the parallel scan. See section 8.1.

## 3. The computation

Per head, per query chunk `[t0, t1)` of size `C`, with `a = q * sqrt(scale)` and
`b = k * sqrt(scale)` so that `a . b = s`, the scaled score.

Weights use the Gauss-Hermite optimal degree-2 fit to `exp(s)` under the measured
`s ~ N(0, sigma^2)`. The common factor `exp(sigma^2/2)` cancels in the softmax
normalisation, leaving

```text
w(s) ~= c0 + c1*s + c2*s^2      c0 = 1 - sigma^2/2,  c1 = 1,  c2 = 1/2
```

`c0` is the only difference from the plain Taylor constant that `fla`'s Based
kernel uses, and it measured **2.3x more accurate** at no cost.

With `phi2(x) = flatten(x x^T)` in `R^{d^2}`, so that
`<phi2(a), phi2(b)> = (a.b)^2`, the running state over all strictly preceding
chunks is

```text
S_const = sum v_j          in R^{1 x dv}      z_const = count       scalar
S_lin   = sum b_j v_j^T    in R^{d x dv}      z_lin   = sum b_j     in R^d
S_quad  = sum phi2(b_j) v_j^T in R^{d^2 x dv} z_quad  = sum phi2(b_j) in R^{d^2}
```

and each chunk computes

```text
num = c0*S_const + c1*(a @ S_lin) + c2*(phi2(a) @ S_quad)      [C, dv]
den = c0*z_const + c1*(a @ z_lin) + c2*(phi2(a) @ z_quad)      [C, 1]
      + exact diagonal block:  w = tril(exp(a b^T)); num += w@v; den += w.sum
out = num / den
```

then folds itself into the state. The `phi2` terms are the only expensive part
and the only thing Phase 1 fuses.

## 4. Architecture

### 4.1 Modules

| file | contents |
| --- | --- |
| `src/kernels/__init__.py` | exports, Triton availability probe |
| `src/kernels/poly_attention_triton.py` | the two Triton kernels and their Python wrappers |
| `src/implementations/poly_reference.py` | the validated PyTorch chunked implementation, promoted from the spike, as the numerical oracle |
| `src/implementations/poly_attention.py` | attention module + `CandidateSpec` named `poly` |
| `src/tests/test_poly_kernel.py` | kernel-vs-oracle equivalence |
| `src/tests/test_poly_attention.py` | end-to-end criterion and guard behaviour |

Nothing in `src/dispatcher.py` changes in Phase 1. The candidate is selected
explicitly with `--candidate poly`. Promotion into the case-14 route is a
separate change gated on section 7.

### 4.2 Kernel 1 — `poly_quad_apply`

Computes `Y = phi2(A) @ S` for `A` of shape `[C, d]` and `S` of shape
`[d^2, dv]`, without materialising `phi2(A)`.

The `d^2` feature axis is indexed by pairs `(i, j)`. A feature block is a
contiguous range of `i` with all `j`, so it is a contiguous slab of `S` — this is
what makes the tiling work and why section 8.2 fixes the layout.

```text
grid = (ceil(C / BC), H)
for each program:
    a   = load A tile                      [BC, d]      registers
    acc = 0                                [BC, BV]     fp32
    for i0 in range(0, d, BI):
        ai  = a[:, i0:i0+BI]                            [BC, BI]
        phi = reshape(ai[:, :, None] * a[:, None, :],
                      (BC, BI*d))                       [BC, BI*d]  registers
        s   = load S[i0*d : (i0+BI)*d, :]               [BI*d, BV]  SRAM
        acc += dot(phi, s)
    store acc
```

`phi` is never written to HBM. That is the whole point: it removes the ~51 GB.

### 4.3 Kernel 2 — `poly_quad_update`

Computes `S += phi2(B)^T @ V`, a reduction over the chunk's `C` rows. Same
feature-block tiling, with the loop running over `C` tiles instead:

```text
grid = (d / BI, H)
for each program:
    acc = 0                                 [BI*d, BV]  fp32
    for c0 in range(0, C, BC):
        b   = load B tile                   [BC, d]
        phi = reshape(...)                  [BC, BI*d]
        v   = load V tile                   [BC, BV]
        acc += dot(phi.T, v)
    S[i0*d : (i0+BI)*d, :] += acc
```

The two kernels share the `phi` generation, which should be a single
`@triton.jit` device function.

### 4.4 Precision

Non-negotiable, from measurement:

- **The master state is float32.** A float16 state accumulator measured 1.32x and
  **fails at N=65536 with 1,064,935 failures** and a max error of 0.106, while
  passing at N=16384 ([`long-sequence-attention.md`](long-sequence-attention.md)
  section 5.2). It is a trap that only appears at scale.
- **Per-chunk matmuls are float16 with float32 accumulation** (`tl.dot` default).
  State is held fp32 in HBM and converted on load inside the kernel. In PyTorch
  this combination measured 1.19x against 0.73x for fp32 throughout.
- No max-subtraction in the exact diagonal block. Scores are measured bounded to
  `[-2.203, 2.404]`, so `exp` cannot overflow; the online-softmax rescale
  (`fast-attention-survey.md` A3) is deliberately omitted. The section 6 guard is
  what keeps this true.

### 4.5 Block sizes

`BC`, `BI`, `BV` are `triton.autotune` parameters, not constants. The binding
constraint is shared memory: 100 KiB per SM, 48 KiB per block by default on
sm_89. A `[BI*d, BV]` state slice at `BI=2, d=64, BV=64` in fp16 is 16 KiB, and
the `phi` tile at `BC=32` is 8 KiB — comfortable. `BI=4, BC=64` needs ~80 KiB and
will not fit without opting into the larger carveout. Start the autotune space at
`BC in {32, 64}`, `BI in {1, 2, 4}`, `BV in {32, 64}` and let measurement decide.

`C` (the chunk length) stays at **512**, the value validated in
`long-sequence-attention.md`. It is not a tuning parameter in Phase 1 because
changing it changes the approximation, not just the schedule.

## 5. Expected performance

Per sample-layer at `N=100000, H=16, d_h=64`, float16, RTX 4060 Laptop.

| path | measured | note |
| --- | ---: | --- |
| exact flash SDPA (case 14's current route) | 719.8 ms | baseline to beat |
| PyTorch order-2 (best correct config) | 603.9 ms | 1.19x |
| **Phase 1 target** | **<= 360 ms** | **>= 2x**, the acceptance threshold |
| compute bound at the exact path's realised 28 TFLOPS | ~61 ms | not a forecast |

The traffic argument: feature traffic falls from ~51 GB to the state traffic of
roughly 3.2 GB per sample-layer, a ~16x reduction, against ~1.7 TFLOP of
arithmetic. **These are estimates from arithmetic, not measurements.** The honest
expectation is 2-4x; the 61 ms figure is the ceiling, not the goal.

## 6. The runtime guard

The method's accuracy depends on `sigma`, a property of the benchmark's random
initialisation rather than of attention in general. Under trained weights scores
are far larger and a degree-2 fit would be poor. The guard is therefore part of
the deliverable, not an optional extra.

1. Sample 2048 query and key rows, compute their scores, take the standard
   deviation. This recovered `sigma = 0.3338` against the 0.3336 population value.
2. Supply `c0 = 1 - sigma^2/2`.
3. **If `sigma` exceeds a validated ceiling, fall back to exact flash SDPA** and
   record the fallback.

The ceiling must be *measured*, not assumed: sweep `sigma` by scaling Q/K and
find where the end-to-end criterion first fails, then set the ceiling with margin
below it. This sweep is part of Phase 1's acceptance, not a follow-up.

The sample costs one host synchronization per forward. That is acceptable here
because case 14's route is eager and chunked and already reads the mask on the
host for prefix lengths; it must never move into a compiled or graph-replayed
region.

## 7. Correctness validation and acceptance

### 7.1 Kernel equivalence

`poly_quad_apply` and `poly_quad_update` against a PyTorch oracle computing the
same quantity densely, over `d in {16, 32, 64}`, `C in {128, 512}`,
`dv in {32, 64}`, and both dtypes. Tolerance: max absolute deviation `<= 1e-3`
relative to the fp32 dense result. These are the same mathematical quantity in a
different summation order, so anything larger indicates a bug, not rounding.

### 7.2 End-to-end criterion

Official criterion `abs <= 0.002 OR rel <= 0.02`, **zero** failures.

| oracle | N | required |
| --- | ---: | --- |
| dense reference (`BaselineTransformer`) | 4096, 8192 | 0 failures |
| exact-flash model | 16384, 32768, 65536, 100000 | 0 failures |

The dense reference is the authoritative oracle but cannot run beyond ~8192 in
8 GiB. The exact-flash oracle is algebraically exact and isolates approximation
error from fp16 reduction-order noise. Both are required; neither alone is
sufficient. The PyTorch order-2 path already passes every row of this table, so
**the kernel inherits a known-good target** — any failure is the kernel's, not
the method's.

### 7.3 Guard

`sigma` sweep per section 6, recording where the criterion first fails and
confirming the fallback engages below that point.

### 7.4 Acceptance

Phase 1 is accepted when 7.1, 7.2 and 7.3 pass **and** the kernel measures
`<= 360 ms` per sample-layer.

Between 360 ms and 603.9 ms the kernel beats the PyTorch order-2 path but misses
the target. That is a judgment call, not an automatic outcome: record the number,
and promote only if the margin over 603.9 ms justifies carrying a Triton
dependency. Above 603.9 ms it is rejected outright — it would be slower than the
PyTorch path while adding a dependency, and above 719.8 ms it is also slower than
doing nothing at all.

## 8. Phase 2 forward-compatibility

Decisions Phase 1 must make now so Phase 2 is not a rewrite.

### 8.1 Parallelism

Phase 2 fuses the chunk loop, so its natural grid is one program per
`(batch, head)`. Case 14's route streams **1-2 samples at a time** —
`choose_batch_chunk_size` selected 2 on a 24 GB L4 — so with `H=16` that is
**16-32 programs against 24 SMs here and ~58 on an L4**. That is the
occupancy-starvation case, and it is why Phase 2 needs a two-level scan
(`fast-attention-survey.md` A6, and TFLA).

The scheme: split `N` into `G` groups, scan each group sequentially in parallel
across groups, then combine group states with a prefix sum. Storing a per-*chunk*
state is not viable — 195 chunks x 16 heads x 512 KiB is ~1.6 GB — but per-*group*
at `G=8` is ~65 MiB, which is fine. `G` becomes the parallelism knob.

### 8.2 Fixed now, so Phase 2 inherits it

- **State layout.** `S_quad` is `[d, d, dv]` contiguous in `(i, j, v)` order, so a
  feature block is a contiguous slab. Phase 2 uses the same tiling.
- **Chunk length `C = 512`**, so the approximation and the diagonal-block
  structure are unchanged between phases.
- **`phi` generation is a shared `@triton.jit` device function**, so the fused
  kernel reuses it rather than reimplementing it.
- **fp32 master state, fp16 tile arithmetic** (section 4.4) is a property of the
  algorithm at this `N`, not of Phase 1's schedule.

## 9. Risks

| risk | severity | mitigation |
| --- | --- | --- |
| Approximation depends on the benchmark's `sigma` | high | measured guard with fallback, section 6; it is stated plainly in the report rather than hidden |
| Kernel is correct but under 2x | medium | acceptance rule in 7.4 says record and do not promote; the PyTorch path remains |
| Shared-memory pressure forces small tiles and poor efficiency | medium | autotune, section 4.5; `BI=1` always fits |
| fp16 state trap reintroduced during optimization | high | section 4.4 is non-negotiable; a regression test at N=65536 specifically catches it |
| Triton 3.7.1 API drift | low | version is pinned in `requirements.txt`; the availability probe degrades to the PyTorch path |
| Case 14 may not be gradeable at all | high, external | out of our control; tracked as organizer clarification #4 in [`../extreme-memory/README.md`](../extreme-memory/README.md) |

The last one deserves emphasis: the benchmark harness runs the dense baseline
before the candidate, and case 14's dense baseline is not runnable on any GPU. If
that is not resolved, none of this work is scored. It should be pursued in
parallel and does not block Phase 1.

## 10. Handoffs

- **Person 4** owns case 14's memory route. Promotion into that route touches
  `src/implementations/extreme.py` and must be agreed, not landed unilaterally.
- **Person 1** owns compilation and dispatch. The guard's host synchronization
  must stay outside any compiled or graph-replayed region.
- No change to cases 1-13; nothing here affects the merged mask-routing work.

## 11. Sources

Prior art, licences, and the measurements that constrain this design are recorded
in [`triton-kernel-prior-art.md`](triton-kernel-prior-art.md). Measurements
referenced throughout are from
[`long-sequence-attention.md`](long-sequence-attention.md) sections 4 and 5.
