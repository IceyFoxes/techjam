# Stage 0 Baseline — Integrated Polynomial Kernel, RTX 4060 Laptop

Closing measurement for Stage 0 of
[`../../attention-softmax/integrated-kernel-spec.md`](../../attention-softmax/integrated-kernel-spec.md),
executed from
[`../../../docs/superpowers/plans/2026-08-30-integrated-polynomial-kernel-stage0.md`](../../../docs/superpowers/plans/2026-08-30-integrated-polynomial-kernel-stage0.md).

Date: 30 August 2026. Branch `fused-kernal`, commits `cabeb53` through the guard
fix. Verdict: **Stage 0 ACCEPTED — 1.439x, and Design B rejected with evidence.**

## Headline

| | Phase 1 (shipped) | Stage 0 | change |
| --- | ---: | ---: | ---: |
| latency, B=2, N=100000 | 328.1 ms | **228.0 ms** | **1.439x** |
| speedup over exact Flash | 4.31x | **6.135x** | |
| peak VRAM overhead | +67.1 MiB | **+61.4 MiB** | -5.7 MiB |
| correctness failures | 0 | **0** | |
| `SIGMA_CEILING` | 0.45 | **0.40** | lowered, see below |

Gate was **>= 1.15x**. Met with wide margin, and the ratio is `RESOLVABLE`
against the session's own 2.5% A/A floor.

**On comparing across sessions.** The 328.1 ms figure is from a different
session, and this repository's own rule is that cross-session absolute
comparisons are not measurements. The comparison is anchored instead on exact
Flash, which is unchanged code: Phase 1 measured it at 1414.0 ms and this run at
1398.7 ms, **1.1% apart**. Normalising by it gives 0.2320 of Flash before and
0.1630 after — a ratio of 1.423x, against 1.439x unnormalised. The two agree,
so the session offset is not carrying the result.

## Latency

`stage0-b2.json`, 9 interleaved repetitions after warm-up, strided inputs,
`B=2, H=16, N=100000, d_h=64`, float16. **B=2 is the shape case 14's route
actually streams.**

| variant | min | median | max | spread |
| --- | ---: | ---: | ---: | ---: |
| exact flash SDPA | 1398.7 ms | 1399.1 ms | 1404.4 ms | 0.4% |
| polynomial, dense PyTorch | 790.0 ms | 796.7 ms | 801.5 ms | 1.5% |
| **polynomial, Stage 0 Triton** | **228.0 ms** | 232.8 ms | 237.0 ms | 3.9% |
| A/A control (identical code) | 222.5 ms | 225.6 ms | 230.6 ms | 3.6% |

**A/A discrepancy 2.5%, so the minimum detectable effect is 1.025x.** This was
a cool card; spreads of 0.4-3.9% are the tightest recorded in this stream.

`stage0-b1.json` records B=1 at 317.2 ms against exact Flash's 699.8 ms, 2.206x.
**Treat it with caution:** its A/A floor was 6.1% and its spreads 23-24%,
because the card had just run the B=2 sweep. See "The B=1 anomaly" below.

## Per-fix outcomes

| task | fix | outcome |
| --- | --- | --- |
| F1 | causal-tiled diagonal block | **1.143x, RESOLVABLE** (floor 0.7%) — the largest single gain |
| F2 | skip fully-prefixed chunks | **kept, not measured** — 4% against an 11.2% floor that session |
| F3 | float16 shadow of the quadratic state | **1.054x, RESOLVABLE** (floor 2.4%) — marginal |
| F4 | measured launch configurations | **kept on sweep evidence** — the latency A/B's own control disagreed by 24.9% |
| F5 | float16 shadows of the small states, scalar `z_const` | **1.029x, RESOLVABLE** (floor 1.3%) — marginal |
| F6 | hoist the redundant `ai` load | **rejected: 19x slower** |

F2 is kept on evidence that does not need the floor: a test asserts the diagonal
hook is called 188 times instead of 196, and the profile below confirms it.

F4's sweep found the best configuration **outside the shipped autotune space in
all three kernels** — `BC=128` for both feature-map kernels, where the space held
only 32 and 64, and `num_stages=3` for the diagonal, where it was pinned at 2.

F6 was judged on generated PTX, since F0 established it could never be resolved
by latency. Hoisting needs a compile-time index, so `tl.static_range`, which
fully unrolls the 64-iteration feature loop: 520 `ld.global` and 32 register
spills against 9 and 0, and 8.83 ms against 0.4606 ms. Output was bitwise
identical, so the rewrite was correct and 19x slower.

## Attribution after Stage 0

`profile-b2-stage0.csv`, total self CUDA time 195.3 ms. Compare with
[`../2026-08-30-rtx4060-6dc9639/`](../2026-08-30-rtx4060-6dc9639/README.md),
which totalled 267.6 ms.

| what | ms | share | before |
| --- | ---: | ---: | ---: |
| `_quad_update_kernel` | 69.8 | 35.7% | 25.7% |
| `_quad_apply_kernel` | 59.1 | 30.3% | 25.3% |
| **the two feature-map kernels** | **128.9** | **66.0%** | 51.1% |
| `_causal_diag_kernel` | 9.7 | 5.0% | **25-35%** |
| dtype conversion and copy | 17.8 | 9.1% | 7.7% |
| scalar multiplies | 11.5 | 5.9% | 5.0% |
| GEMM families | 9.1 | 4.7% | 11.7% |
| adds | 7.0 | 3.6% | 4.8% |
| reductions | 5.9 | 3.0% | 4.5% |
| binary elementwise, flash prefix, fill | 5.4 | 2.7% | 2.0% |

**The exact diagonal block went from 25-35% of the path to 5.0%** — 65-95 ms
down to 9.7 ms. That is F1, and it is the largest structural change in Stage 0.

`exp`, `masked_fill_` and the mask-construction rows have vanished entirely: the
kernel does all three in registers. GEMM time more than halved as the diagonal's
two products moved inside it.

The call counts confirm F2 independently: `_causal_diag_kernel` and
`_quad_apply_kernel` are called 188 times where the dense path ran 196 and 195.

## The Stage 1 decision

The spec gates Design B — the persistent-slab scan — on two conditions.

**Condition 1: is state traffic still the top cost? NO.**

The two feature-map kernels are certainly the top cost at 66% of the path. But
Design B exists to remove *state traffic*, and after F3's float16 shadow and
F4's `BC=128` the kernels are no longer traffic-limited:

| kernel | per call | FLOP per call | realised |
| --- | ---: | ---: | ---: |
| `_quad_update_kernel` | 356 us | 8.6 GFLOP | **24.1 TFLOPS** |
| `_quad_apply_kernel` | 314 us | 8.6 GFLOP | **27.4 TFLOPS** |

The exact Flash path realises about 28 TFLOPS on this card. **Both kernels now
run at essentially the machine's achieved throughput, so what remains is
arithmetic, not traffic.** Removing HBM and L2 state traffic cannot buy back time
that is not being spent on it.

**Condition 2: does the VRAM fit? NO.**

Design B needs fp32 atomics into an `[M, N, V]` buffer, **+819 MiB**, against a
route whose entire overhead is +61.4 MiB and a ceiling of +100 MiB.

**Design B is rejected with evidence and will not be built.** That is a
legitimate outcome of Phase 2, not a failure of it. The spec anticipated exactly
this: "the traffic B removes is largely L2 traffic, not HBM traffic".

**Design A is confirmed**, and the profile says what it should prioritise. The
remaining glue — conversions, scalar multiplies, adds, reductions, the divide —
is about 28% of the path, worth `1/(1 - 0.28)` = **1.39x** if fully absorbed
into the two per-chunk kernels. That is the whole of Stage 1's headroom, and it
matches the spec's >= 1.4x target closely enough that the target should be
treated as the optimistic end.

## The guard moved, and the ceiling was lowered

Step 2 re-ran the `sigma` sweep expecting Phase 1's result to reproduce. **It did
not.**

| sigma | before Stage 0 | after Stage 0 |
| ---: | ---: | ---: |
| 0.3339 | 0 failures | 0 failures |
| 0.4040 | 0 failures | 0 failures |
| 0.4188 | not measured | 0 failures — **largest clean pass** |
| 0.4416 | not measured | 1 failure — **first failure** |
| 0.4808 | **0 failures** | **1 failure** |
| 0.5217 | 21 failures | 19 failures |
| 0.7512 | 56,801 failures | 56,758 failures |

`SIGMA_CEILING = 0.45` was justified as sitting below the largest verified pass,
0.4808. After Stage 0 that is false: 0.45 is **above** the first failure, so the
guard would have admitted configurations that fail the criterion.

The cause is Stage 0's own numerics. F1's kernel computes `exp` in float32 where
the dense block rounded scores to float16, and `z_const` became a Python scalar.

**The ceiling is lowered to 0.40**, below the largest clean pass and about 20%
above the operating point's seed-to-seed spread of 0.3327-0.3343, so it can
neither admit a failing configuration nor cause a spurious fallback.
`test_poly_guard` now asserts `SIGMA_CEILING <= 0.4188`.

Near the boundary the failure count is not monotonic — 0.4416 fails with one
element in 8,388,608 while 0.4649 passes — because a handful of elements sit
exactly on the criterion. That is a reason to anchor below the largest clean pass
rather than interpolate a crossing point.

**The general lesson: a measured ceiling is only as good as its last
re-measurement.** Any change to the numerics must re-run this sweep.

## Correctness

Official criterion `abs <= 0.002 OR rel <= 0.02`, **zero** failures required.
Two-layer case-14-shaped model, `B=1`, float16.

| oracle | N | failures | max abs err | rms err |
| --- | ---: | ---: | ---: | ---: |
| dense reference | 4096 | 0 / 4,194,304 | 5.8594e-03 | 2.4607e-04 |
| dense reference | 8192 | 0 / 8,388,608 | 5.8594e-03 | 3.3754e-04 |
| exact flash | 16384 | 0 / 16,777,216 | 7.8125e-03 | 3.3848e-04 |
| exact flash | 32768 | 0 / 33,554,432 | 7.8125e-03 | 3.5096e-04 |
| exact flash | 65536 | 0 / 67,108,864 | 7.8125e-03 | 3.5078e-04 |
| exact flash | 100000 | 0 / 102,400,000 | 7.8125e-03 | 3.4862e-04 |

The max error at N=4096 and 8192 **improved**, from 7.8125e-03 to 5.8594e-03,
because F1's kernel accumulates scores in float32 where the dense block rounded
them to float16. Being more accurate than the reference is not automatically
safe — it is what breaks cases 1-13, where the reference rounds probabilities
back to float16 — so every stage re-ran this table rather than assuming.

N=65536 is the specific regression that catches a float16 master state. It stays
clean, and `quad_update` still raises on a float16 master.

## The B=1 anomaly

B=1 measured 317.2 ms against B=2's 228.0 ms — **B=2 processes twice the work in
less time**, 114 ms per sample against 317 ms.

Part of this is occupancy: `M = B*H` is 16 at B=1 against 32 at B=2, and the
launch grids shrink with it. But part is a real limitation of F4: **the
configuration table is keyed on `(C, D, V)` and not on `M`**, and it was swept at
M=32. At M=16 the `BC=128` entry gives a 64-program grid on 24 SMs. Phase 1's
autotune had the same blind spot — its key was also `(C, D, V)` — so this is a
pre-existing limitation rather than a regression, but Stage 0 made the
configurations more specific and therefore the blind spot sharper.

Carried into Stage 1 as an open item: add `M` to the key, or grid the kernels so
they are less sensitive to it. Case 14's route streams 1-2 samples and selected 2
on a 24 GB L4, so B=2 remains the shape acceptance is judged on.

## Commands

```bash
.venv/bin/python -m unittest discover -s src/tests
for n in 4096 8192; do .venv/bin/python -m src.validate_poly --n $n --oracle dense; done
for n in 16384 32768 65536 100000; do .venv/bin/python -m src.validate_poly --n $n --oracle flash; done
for w in 1.0 1.05 1.1 1.12 1.15 1.18 1.2 1.25 1.3 1.4 1.5; do
  .venv/bin/python -m src.validate_poly --n 8192 --oracle dense --scale-qk-weights $w
done
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 9 \
  --output research/benchmarks/2026-08-30-rtx4060-stage0/stage0-b2.json
.venv/bin/python -m src.bench_poly --n 100000 --batch 1 --reps 9 \
  --output research/benchmarks/2026-08-30-rtx4060-stage0/stage0-b1.json
.venv/bin/python -m src.sweep_poly_configs --rounds 7 \
  --output research/benchmarks/2026-08-30-rtx4060-bfbea79/config-sweep.json
```

Per-fix A/Bs used `--ab-disable {diag,prefix,shadow,configs,smallshadow}`, which
runs the pre-optimization path as a second arm **in the same session**.

## Environment

Full machine-readable environment in each JSON. NVIDIA GeForce RTX 4060 Laptop
GPU, 8.0 GiB, 24 SMs, sm_89, 32 MiB L2, driver 616.56; PyTorch 2.13.0+cu130,
Triton 3.7.1, CUDA 13.0, Python 3.12, Linux 6.18.33.2-microsoft-standard-WSL2.

194 tests pass (`unittest discover -s src/tests`).

## Scope and limitations

- **Attention core only.** Not a whole-case-14 measurement; case 14 cannot run
  end to end on this 8 GiB card, which needs a 12.21 GiB floor for the
  device-resident input and output alone.
- **This is an approximation**, not an algebraically exact rewrite, and its
  accuracy depends on the benchmark's small scores. The guard exists for that,
  and this run is the reason its ceiling moved.
- **Single GPU.** No RTX 5080 or L4 numbers. F4's configuration table is keyed on
  device capability, so other GPUs fall back to autotune rather than run a
  configuration measured elsewhere.
- The B=1 figure is noisy and should not be quoted without its caveat.
