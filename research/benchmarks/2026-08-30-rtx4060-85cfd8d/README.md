# Attention Mask-Route Sweep — RTX 4060 Laptop, cu130

Preserved evidence for the route decision in
[`safe-optimization-spec.md`](../../attention-softmax/safe-optimization-spec.md)
section 3: under causal attention the padding key mask is dead code, so it may
be dropped. This sweep measures whether dropping it is also *faster*.

Status: current as of 30 August 2026.

## What was compared

| Candidate | Selector | Behavior |
| --- | --- | --- |
| drop-mask | `attention` | `sdpa(q,k,v, is_causal=True)`, no attention mask |
| keep-mask (control) | `attention:KEYMASK_CANDIDATE` | `sdpa(q,k,v, attn_mask=keep[B,1,1,N], is_causal=True)` |

The control reproduces upstream `StridedSDPASelfAttention`
(`src/implementations/sdpa.py`) exactly, so this measures Person 2's change
against what the shared module does today — not against the eager reference.
Both are compared to the immutable reference for correctness.

## Environment

| Field | Value |
| --- | --- |
| cpu | x86_64 |
| gpu | {'available': True, 'cuda_runtime': '13.0', 'driver': '616.56', 'name': 'NVIDIA GeForce RTX 4060 Laptop GPU'} |
| os | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| python | 3.12.14 |
| pytorch | 2.13.0+cu130 |
| timestamp | 2026-08-29T17:08:31.366575+00:00 |
| commit | 85cfd8d3177d |
| driver / CUDA runtime | 616.56 / 13.0 |
| Triton | 3.7.1 |
| matmul precision / TF32 | high / enabled (harness defaults) |
| timing | paired interleaved A/B, `src/infra/timing.py` |
| settle | 20 s |
| repeats | 40 (cases 1, 5, 8, 11, 13); 120 (cases 2, 3, 4, 7, 9, 10, 12) |
| accuracy trials | 5 seeds per cell, zero failed elements required |

This host was migrated to the team-pinned stack on 30 August 2026 and now runs
the same driver and CUDA runtime as Person 1's RTX 5080, so results differ from
that machine only by GPU.

## Results

Speedup is each candidate against its own concurrently-interleaved reference
baseline. "Winner" is `tie` when the two differ by less than half the wider
noise floor.

| Case | pr | drop-mask | keep-mask (control) | drop floor | keep floor | winner |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0 | 1.417x | 1.244x | ±10.9% | ±7.2% | **drop** |
| 1 | 0.3 | 1.321x | 1.327x | ±6.7% | ±16.1% | tie |
| 2 | 0.0 | 1.541x | 1.422x | ±9.7% | ±6.1% | **drop** |
| 2 | 0.3 | 1.722x | 1.465x | ±7.7% | ±5.0% | **drop** |
| 3 | 0.0 | 1.578x | 1.457x | ±11.8% | ±5.1% | **drop** |
| 3 | 0.3 | 1.621x | 1.462x | ±6.0% | ±6.4% | **drop** |
| 4 | 0.0 | 1.603x | 1.497x | ±7.7% | ±5.3% | **drop** |
| 4 | 0.3 | 1.612x | 1.484x | ±7.8% | ±6.8% | **drop** |
| 5 | 0.0 | 2.463x | 1.929x | ±19.8% | ±7.2% | **drop** |
| 5 | 0.3 | 2.563x | 2.455x | ±13.3% | ±11.5% | tie |
| 7 | 0.0 | 1.583x | 1.372x | ±7.4% | ±7.5% | **drop** |
| 7 | 0.3 | 1.717x | 1.440x | ±11.0% | ±6.5% | **drop** |
| 8 | 0.0 | 1.071x | 1.062x | ±0.7% | ±0.7% | **drop** |
| 8 | 0.3 | 1.063x | 1.062x | ±1.0% | ±0.6% | tie |
| 9 | 0.0 | 1.194x | 1.147x | ±4.8% | ±3.8% | **drop** |
| 9 | 0.3 | 1.176x | 1.096x | ±5.6% | ±4.4% | **drop** |
| 10 | 0.0 | 1.342x | 1.209x | ±6.9% | ±5.1% | **drop** |
| 10 | 0.3 | 1.537x | 1.259x | ±9.9% | ±7.9% | **drop** |
| 11 | 0.0 | 3.226x | 2.953x | ±11.3% | ±14.8% | **drop** |
| 11 | 0.3 | 3.221x | 3.005x | ±12.5% | ±16.3% | tie |
| 12 | 0.0 | 1.609x | 1.438x | ±7.8% | ±8.0% | **drop** |
| 12 | 0.3 | 1.651x | 1.398x | ±6.3% | ±5.0% | **drop** |
| 13 | 0.0 | 5.364x | 5.152x | ±4.6% | ±3.3% | **drop** |
| 13 | 0.3 | 5.416x | 5.180x | ±4.9% | ±2.8% | **drop** |

drop-mask ahead in 20 comparisons, 4 tie(s)

**All 24 cells pass 5/5 seeds with zero failed elements**, and within each case
`max_abs` is identical between the two routes — confirming again that the mask
contributes nothing numerically.

**Drop-mask is ahead in 20 comparisons, tied in 4, and behind in none.**

## How to read this, and how not to

- **Each route ran in its own process.** Only the speedup ratio is comparable
  between them, because each is paired against a baseline measured in the same
  process at the same clock state. The raw candidate milliseconds are **not**
  comparable across files.
- An in-process A/B of the same two routes, where both candidates share one
  interleaved baseline, gave drop-mask ahead in **24 of 24** comparisons. That
  measurement is the stronger evidence for the route decision; this table is the
  harness-produced confirmation under the official correctness checker.
- Cases 2, 3, 4, 7, 9, 10 and 12 run in ~2-4 ms and were measured at 120
  repeats. At 25 repeats the same comparisons produced swings as large as -68%.
  **Do not believe a sub-5 ms result taken at fewer than ~100 repeats.**
- Wide floors on cases 1, 5 and 11 (±7-20%) mean no single figure there should be
  quoted alone. Cases 8 and 13 carry ±0.6-4.9% and are individually meaningful.

## Reproduce

```bash
.venv/bin/python -m src.benchmark --candidate attention \
  --device cuda --dtype float32 --case 13 --padding-ratio 0.3 \
  --repeats 40 --settle-seconds 20 \
  --output <new-path>.json

.venv/bin/python -m src.benchmark --candidate attention:KEYMASK_CANDIDATE \
  --device cuda --dtype float32 --case 13 --padding-ratio 0.3 \
  --repeats 40 --settle-seconds 20 \
  --output <new-path>.json
```

Each JSON holds the exact command, git state, shape, dtype, thresholds,
per-trial correctness, raw latency samples, noise floor, memory, and full
environment.

## Case 6 — correctness and memory only, not a latency claim

Case 6 (`B=10000`) is nominally Person 4's extreme-shape scope. It is validated
here because SDPA addresses its failure mode directly: the eager path
materializes a ~2.6 GB float32 score tensor that SDPA never allocates.

| Metric | Value |
| --- | --- |
| Correctness | **PASS 2/2 seeds**, 0 failed of 163,840,000 elements per trial |
| `max_abs` | 7.59e-04 / 7.63e-04 |
| Baseline | 12,446.9 ms |
| Candidate | 12,008.5 ms |
| Ratio | 1.037x, floor ±19.42% — **`WITHIN NOISE`** |
| Peak allocated | 11,166,121,984 B (**10,648 MiB**) |
| Peak reserved | 13,123,977,216 B (12,516 MiB) |

**The latency figure is not usable and must not be quoted.** Two independent
reasons:

1. The harness itself classifies it `WITHIN NOISE` at 10 repeats. Per
   `src/infra/timing.py`, a ratio inside its own floor is no change.
2. Peak allocation of 10,648 MiB exceeds this card's 8,188 MiB, so the run
   completed only because WSL2 permitted oversubscription into host memory.
   Both sides were paging over PCIe, which compresses the ratio between a
   memory-hungry baseline and a memory-lean candidate.

What case 6 does establish is that the candidate is **correct** at this shape and
that its memory profile is recorded. A latency claim needs a device that holds
the working set resident. Handed to Person 4 with that caveat.

An earlier exploratory run of the same shape on the legacy torch 2.6.0+cu124
stack reported 2.032x, also under oversubscription and therefore equally
unusable. It is mentioned only so the discrepancy is not mistaken for a
regression.

## Compiled A/B — does the gain survive `reduce-overhead`?

The sweep above measures the **eager** candidate. The open question it left was
whether dropping the key mask still helps once `torch.compile` has run, since
compilation might already hide the mask cost. It does not: the gain is *larger*
compiled than eager.

In-process A/B of `src.dispatcher` against itself, identical in every respect
except `drop_key_mask`. Both sides fully compiled through the routed path
(`reduce-overhead` for cases 1-12, `default` for 13), same process, interleaved.

| Case | B | H | N | d_h | pr | keep mask | drop mask | gain | floor | sig |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | :-: |
| 1 | 64 | 4 | 128 | 32 | 0.0 | 1.6685 ms | 1.4862 ms | **1.123x** | ±0.78% | yes |
| 1 | 64 | 4 | 128 | 32 | 0.3 | 1.6522 ms | 1.4862 ms | **1.112x** | ±0.76% | yes |
| 2 | 1 | 4 | 128 | 32 | 0.0 | 0.3743 ms | 0.3609 ms | 1.037x | ±4.99% | — |
| 2 | 1 | 4 | 128 | 32 | 0.3 | 0.3430 ms | 0.3270 ms | 1.049x | ±6.30% | — |
| 3 | 4 | 4 | 128 | 32 | 0.0 | 0.2658 ms | 0.2602 ms | 1.022x | ±4.47% | — |
| 3 | 4 | 4 | 128 | 32 | 0.3 | 0.2713 ms | 0.2532 ms | **1.071x** | ±4.80% | yes |
| 4 | 16 | 4 | 128 | 32 | 0.0 | 0.5361 ms | 0.5088 ms | **1.054x** | ±0.82% | yes |
| 4 | 16 | 4 | 128 | 32 | 0.3 | 0.5301 ms | 0.5045 ms | **1.051x** | ±1.38% | yes |
| 5 | 128 | 4 | 128 | 32 | 0.0 | 3.5421 ms | 3.2870 ms | **1.078x** | ±0.41% | yes |
| 5 | 128 | 4 | 128 | 32 | 0.3 | 3.5616 ms | 3.3045 ms | **1.078x** | ±0.35% | yes |
| 7 | 64 | 4 | 128 | 8 | 0.0 | 0.9890 ms | 0.8944 ms | **1.106x** | ±0.65% | yes |
| 7 | 64 | 4 | 128 | 8 | 0.3 | 0.9892 ms | 0.8943 ms | **1.106x** | ±0.68% | yes |
| 8 | 64 | 4 | 128 | 256 | 0.0 | 39.2233 ms | 39.0851 ms | **1.004x** | ±0.31% | yes |
| 8 | 64 | 4 | 128 | 256 | 0.3 | 40.0835 ms | 39.8730 ms | **1.005x** | ±0.32% | yes |
| 9 | 64 | 1 | 128 | 128 | 0.0 | 1.4158 ms | 1.3091 ms | **1.081x** | ±0.55% | yes |
| 9 | 64 | 1 | 128 | 128 | 0.3 | 1.4167 ms | 1.3089 ms | **1.082x** | ±0.51% | yes |
| 10 | 64 | 2 | 128 | 64 | 0.0 | 1.4105 ms | 1.2895 ms | **1.094x** | ±0.60% | yes |
| 10 | 64 | 2 | 128 | 64 | 0.3 | 1.4234 ms | 1.2946 ms | **1.099x** | ±0.53% | yes |
| 11 | 64 | 16 | 128 | 8 | 0.0 | 3.9302 ms | 3.5898 ms | **1.095x** | ±0.61% | yes |
| 11 | 64 | 16 | 128 | 8 | 0.3 | 3.7522 ms | 3.4267 ms | **1.095x** | ±0.71% | yes |
| 12 | 64 | 4 | 32 | 32 | 0.0 | 0.4804 ms | 0.4635 ms | **1.036x** | ±0.71% | yes |
| 12 | 64 | 4 | 32 | 32 | 0.3 | 0.5137 ms | 0.4935 ms | **1.041x** | ±0.80% | yes |
| 13 | 64 | 4 | 1024 | 32 | 0.0 | 51.1048 ms | 44.3172 ms | **1.153x** | ±0.34% | yes |
| 13 | 64 | 4 | 1024 | 32 | 0.3 | 51.2876 ms | 44.3976 ms | **1.155x** | ±0.52% | yes |

21 of 24 significant and positive; the rest positive but inside their floors; **none negative**.

Repeats: 60 with a 15 s settle for cases 1, 11, 13; 150 with 20 s for cases 4,
5, 7, 8, 9, 10; 250 with 25 s for the launch-bound cases 2, 3, 12.

An earlier 60-repeat run put case 2 at **0.941x** at `padding_ratio=0`, which
would have read as a regression. At 250 repeats it is 1.037x, and the pr=0.3
direction agreed all along. Its floor was ±14.9%, so the figure never supported
a conclusion. Same lesson as the eager sweep: **sub-millisecond cases need
hundreds of repeats.**

Case 8 gains only 1.004-1.005x, but with a ±0.31% floor that is still
significant. It is the projection-bound shape where attention is ~16% of device
time, so a small attention gain is the expected result, not a disappointment.

### Why compiled beats eager

The eager gain was ~2-5% on large shapes; compiled it is 9.5-15.5%. Inductor
removes surrounding overhead that partly masked the difference eagerly, so the
mask's cost becomes a larger share of what remains. The practical consequence is
that this optimization is worth *more* in the shipped configuration than the
standalone candidate measurement suggested.

### Reproduce

```bash
.venv/bin/python <scratch>/disp_ab.py --case 13 --pr 0.3 --repeats 60 --settle 15
```

The A/B harness subclasses `DispatchingTransformer` and overrides
`_may_drop_key_mask` to force each side, so both paths compile identically and
differ only in the masking decision.
