# Noise Floor for Phase 2 A/B Testing — RTX 4060 Laptop

Phase 2 task **F0**, the blocking prerequisite from
[`../../attention-softmax/integrated-kernel-spec.md`](../../attention-softmax/integrated-kernel-spec.md)
section 4. This run does not measure an optimization. It measures **how large a
difference this machine can resolve at all**, so that the Stage 0 and Stage 1
gates are decidable rather than aspirational.

Date: 30 August 2026. Commit: `d496539` (branch `fused-kernal`).
Shapes: `B=2, H=16, N=100000, d_h=64`, float16, causal, strided inputs.

## Method

The harness registers the **same** `poly_triton` callable twice, as
`poly_triton_ms` and `poly_triton_control_ms`, and interleaves all four variants
one repetition at a time after a warm-up. Whatever gap appears between those two
arms is this machine reporting a difference where there is none.

The floor is the **A/A discrepancy between the two minima**, because min-of-N is
the statistic an A/B actually compares.

## Result

Two runs, the second immediately after the first on an already-loaded card.

| | run 1 (`--reps 5`) | run 2 (`--reps 7`) |
| --- | ---: | ---: |
| `exact_flash` min | 1415.0 ms | 1418.7 ms |
| `poly_pytorch` min | 845.3 ms | 837.5 ms |
| `poly_triton` min | **315.5 ms** | **370.7 ms** |
| `poly_triton_control` min | 317.2 ms | 361.0 ms |
| **A/A discrepancy — the floor** | **0.6%** | **2.7%** |
| worst single-rep A/A gap | — | 48.6% |
| worst within-variant spread | 10.3% | 55.6% |
| minimum detectable effect | **1.006x** | **1.027x** |
| speedup vs exact flash | 4.485x | 3.827x |
| peak VRAM overhead | +67.1 MiB | +67.1 MiB |

Raw: `noise-floor-b2.json`, `noise-floor-b2-reps7.json`.

## What this establishes

**1. The working floor is 1.03x.** Take the worse of the two runs. Every Phase 2
gate clears it comfortably: Stage 0 at 1.15x and Stage 1 at 1.4x are decidable
with wide margin, and so is F1, whose target is a 25-35% share of the path.

**2. Small fixes are near the limit, and F6 is below it.** F2 and F5 are each
worth roughly 4% and sit only marginally above a 2.7% floor. F6, expected to be
worth nothing measurable, **cannot be A/B'd at all** — it must be judged by
proxy (kernel and launch counts, profiler share) or dropped. This is the
evidence for the Phase 2 spec's stance that F6 is "expected to be dropped".

**3. Cross-session absolute comparisons are invalid.** `poly_triton`'s own
minimum moved from 315.5 ms to 370.7 ms — **17.5%** — between two runs of
identical code minutes apart. **Both arms of every A/B must run in the same
session, interleaved.** A Stage 0 number compared against a Stage 1 number from
a different session is meaningless, and the 1.15x gate would be swamped by the
session offset alone.

**4. The floor is session-dependent, so it is re-measured every session.** 0.6%
on a cool card and 2.7% on a warm one. The spec's requirement to re-establish it
per session rather than reuse an earlier value is confirmed, not assumed.

**5. One-shot comparisons are worthless here.** The worst single-repetition gap
between identical code was **48.6%**. Interleaving and repetition are not
diligence, they are the difference between a measurement and a coin flip.

## A design correction this run forced

The first implementation defined the floor as the larger of the A/A discrepancy
and the within-variant spread `(max - min) / min`, on the reasoning that a tight
A/A pair should not license trusting a small win on a jittery run.

**The data showed that conflates two different quantities.** In run 1, identical
code reproduced to 0.6% while its own repetitions varied by 10.3% — more than an
order of magnitude apart. The min-of-N estimator already suppresses sample
dispersion; folding the spread in would have set the floor at 1.103x and made
every fix worth less than 10% unmeasurable. Worse, `(max - min) / min` grows
monotonically with repetition count, so the rule would have **penalised
measuring more carefully**: run 2 took more repetitions and would have been
assigned a 1.556x floor.

The floor is now the A/A discrepancy alone. The spread and the worst single-rep
gap are still reported, as dispersion and as one-shot risk respectively, which is
what they actually are. Pinned by
`src/tests/test_bench_stats.py::test_floor_is_the_a_a_gap_not_the_within_variant_spread`.

## Commands

```bash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 5 \
  --output research/benchmarks/2026-08-30-rtx4060-d496539/noise-floor-b2.json
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7 \
  --output research/benchmarks/2026-08-30-rtx4060-d496539/noise-floor-b2-reps7.json
.venv/bin/python -m unittest src.tests.test_bench_stats -v
```

## Environment

Full machine-readable environment in each JSON. RTX 4060 Laptop GPU, 8.0 GiB,
24 SMs, sm_89, driver 616.56; PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0,
Python 3.12, Linux 6.18.33.2-microsoft-standard-WSL2.

## Scope and limitations

- **The 4.485x and 3.827x speedups here are not a new performance claim.** They
  are a by-product of the floor measurement and they disagree with each other by
  17%. The authoritative Phase 1 figure remains **4.31x** from
  [`../2026-08-30-rtx4060-poly/`](../2026-08-30-rtx4060-poly/README.md).
- Two sessions only. The floor should be re-measured, not assumed, each time.
- Single GPU. A different card — particularly a desktop one that does not
  throttle — will have a different and probably much tighter floor.
- Attention core only, one sample-pair x one layer.
