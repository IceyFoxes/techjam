# Polynomial Attention Kernel — Integration Notes

**For Persons 1 and 4.** Person 2 made these changes to files your streams own.
This document says what changed, what contract each change must not break, and
how to turn it off.

Status: 30 August 2026, **updated after RTX 5080 end-to-end integration**. Specs:
[`research/attention-softmax/triton-kernel-spec.md`](../research/attention-softmax/triton-kernel-spec.md)
(Phase 1) and
[`research/attention-softmax/integrated-kernel-spec.md`](../research/attention-softmax/integrated-kernel-spec.md)
(Phase 2).

## The one-line summary

Case 14 uses a degree-2 polynomial approximation by default on the validated
route, guarded by a runtime check that falls back to exact Flash when the
approximation is not valid. `POLY_ATTENTION_ENABLED = False` remains the
one-line exact rollback.

On the RTX 5080 (`sm_120`), the official batch-one stream uses a measured
one-kernel policy: dense PyTorch apply and diagonal blocks plus a static Triton
`quad_update`. Compiling all three Triton kernels made the first complete
forward slower than Flash even though it won after warm-up. The hybrid measured
11.41-11.69 s cold against the fastest 14.68 s exact control, a conservative
1.255x end-to-end speedup. See
`research/benchmarks/2026-08-30-rtx5080-8567f3f-sm120/`.

## How to turn it off

```python
# src/implementations/extreme.py
POLY_ATTENTION_ENABLED = False
```

That restores **exactly** the previous forced-Flash behaviour.
`src/tests/test_poly_attention.py::PolyRouteToggleTests` pins the rollback so it
cannot rot.

## Person 4 — `src/implementations/extreme.py`

**What changed.** A `PolyOrFlashSelfAttention` subclass of your
`FlashOnlySDPASelfAttention`. When the flag is off, or the guard rejects the
input, it calls `super().forward(...)` — your code path, unmodified.

**The contract it must not break, and why it holds:**

| your invariant | why it still holds |
| --- | --- |
| peak VRAM stays close to the Flash path's | measured: **+67 MiB** at B=2, N=100000. An earlier version of this document claimed no tensor scaled with `B*N*d_model`; that was **wrong** and the route peaked **+6773 MiB**, enough to push a 13 GiB run past 16 GiB. Two causes, both fixed and now pinned by a test: a 3-D SDPA call that fell back to the quadratic math backend, and full-length contiguous copies from `reshape` on strided views. |
| the polynomial state itself is shape-independent | `[H, d^2, d]` — 512 KiB per head, independent of `N` and `B`. Per-chunk working tensors are `[H, 512, d]`. This part of the original claim was correct; the problem was everything around it. |
| prefix streaming drives execution | unchanged. The route sits *inside* one chunk's forward; `forward_prefix_chunks` and the OOM backoff are untouched. |
| no attention mask reaches the extreme path | unchanged. It raises on a non-`None` mask exactly as yours does. |
| Flash backend is forced, never quadratic math | unchanged on the fallback path, which is literally your method. |

**What to check in review:** that the `super().forward(...)` fallback really is
reached in the two disable cases, and that nothing in the polynomial path
allocates per-`N`.

## Person 1 — dispatch and compilation

`src/dispatcher.py` now constructs `PolyOrFlashSelfAttention` for Case 14. The
top-level route remains the eager, memory-safe extreme path; the attention
module selects polynomial versus exact Flash through the measured sigma guard.

**The constraint that matters to you:** the guard calls `estimate_sigma`, which
performs **one device-to-host synchronization** per module instance (cached
after the first forward). It must stay in the eager dispatch layer. It is safe
today because case 14's route is eager and chunked, and already reads the mask
on the host to compute prefix lengths — but if case 14 ever gains a compiled or
CUDA-graph route, this sync must not be inside it.

## What the approximation is, and its main risk

`exp(s)` is replaced by the degree-2 polynomial that is L2-optimal under the
measured score distribution. **This is accurate because the benchmark's random
initialisation produces small scores (measured `sigma = 0.3336`), which is a
property of the benchmark rather than of attention.** Under trained weights,
scores are far larger and the fit would be poor.

That is exactly what the guard is for: `estimate_sigma` measures the actual
score spread at runtime and falls back to Flash above a measured ceiling. The
ceiling comes from a sweep, not a guess — see the spec's section 6.

Be aware this is an **approximation**, not an algebraically exact rewrite. It
passes the official criterion with zero failures at every tested `N`, but it is
not the same arithmetic as the reference.

## Evidence

Official criterion `abs <= 0.002 OR rel <= 0.02`, zero failures required.
Full two-layer case-14-shaped model, `B=1`, float16, RTX 4060 Laptop.

| oracle | N | failures | max abs err |
| --- | ---: | ---: | ---: |
| dense reference | 4096 | 0 / 4,194,304 | 5.86e-03 |
| dense reference | 8192 | 0 / 8,388,608 | 5.86e-03 |
| exact flash | 16384 | 0 / 16,777,216 | 7.81e-03 |
| exact flash | 32768 | 0 / 33,554,432 | 7.81e-03 |
| exact flash | 65536 | 0 / 67,108,864 | 7.81e-03 |
| exact flash | 100000 | 0 / 102,400,000 | 7.81e-03 |

`7.81e-03` is one float16 ulp at the output magnitude — the approximation error
is below the representation noise floor. The two dense-oracle rows improved to
`5.86e-03` in Stage 0, because the diagonal-block kernel keeps scores in float32
where the dense block rounded them.

Latency and peak VRAM, attention core only, at `N=100000`, interleaved timing
with strided inputs. **B=2 is the shape your route actually streams.**

| path | B=1 | B=2 | peak MiB (B=2) |
| --- | ---: | ---: | ---: |
| exact flash SDPA (today's route) | 699.8 ms | 1398.7 ms | 404 |
| polynomial, dense PyTorch | 382.1 ms | 790.0 ms | — |
| **polynomial, Stage 0 Triton** | **317.2 ms** | **228.0 ms** | **465** |

**6.135x at B=2, with +61.4 MiB of VRAM overhead** — up from 4.31x and down
from +67 MiB after Phase 2 Stage 0. The A/A control in the same run put the
noise floor at 2.5%, so the ratio is resolvable with wide margin.

The B=1 column is noisy (6.1% floor, 23% spreads) and B=1 is *slower per sample*
than B=2, because `M = B*H` halves and the launch grids shrink with it. B=2 is
the shape your route streams, and it is the one to judge on.

Full record, including the per-fix A/Bs and two rejected changes:
[`research/benchmarks/2026-08-30-rtx4060-stage0/`](../research/benchmarks/2026-08-30-rtx4060-stage0/README.md).

An earlier version of this document quoted 342.4 ms / 2.12x. That figure is
superseded: it was measured with contiguous inputs the real module never
produces, it included a bug worth ~72 ms and 2.4 GiB, and its variants were
timed back to back on a throttling GPU.

**The guard ceiling changed in Stage 0, from `sigma = 0.45` to `0.40`.** Read
this if you review nothing else here.

Stage 0's diagonal-block kernel computes `exp` in float32 where the previous
dense block rounded scores to float16. That shifted the accuracy boundary down:
`sigma 0.4808` passed with zero failures before Stage 0 and fails after it. The
old ceiling of 0.45 was justified as sitting *below* the largest verified pass;
after the change it sat *above* the first failure, so it would have admitted
configurations that fail the criterion.

The new ceiling of 0.40 is below the largest clean pass (0.4188) and about 20%
above the operating point's seed-to-seed spread (0.3327-0.3343), so it can
neither admit a failing configuration nor cause a spurious fallback.

**The transferable lesson: any change to the numerics must re-run this sweep.**
A measured ceiling is only as good as its last re-measurement. Full sweep in
`research/benchmarks/2026-08-30-rtx4060-stage0/`.

**This is the attention core, not whole case 14.** Case 14 cannot run end to end
on the 8 GiB card these numbers come from -- it needs a 12.21 GiB floor for the
device-resident input and output alone -- so no end-to-end case-14 speedup is
claimed here.

## Verification

```bash
.venv/bin/python -m unittest discover -s src/tests
.venv/bin/python -m src.validate_poly --n 8192 --oracle dense
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
.venv/bin/python -m src.bench_poly --n 100000
```
