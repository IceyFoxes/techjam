# Packed-QKV Cross-Case Validation — RTX 5080 — 29 August 2026

## Outcome

Universal packed QKV is rejected. Official Case 3 is the only new promotion
candidate: its full confirmation improves the prior compiled strided-QKV route
from 0.160123 ms to 0.134352 ms (**1.192x**) with a ±0.84% paired noise floor.
It passes all 60 trials across seeds 1234-1243, input scales 0.125, 1, and 8,
and padding ratios 0 and 0.25. Packed and control outputs are bitwise identical
over that matrix; both have zero failed elements against the immutable reference.

Cases 1, 9, 11, 12, and 13 are within direct A/B noise. Cases 4, 5, 7, 8,
and 10 show only 1.9-3.8% direct gains, below the existing 15% isolated-fusion
integration gate. Cases 6 and 14 remain unsupported because dense execution is
unsafe at their official sizes.

## Three Reference Points

- Immutable baseline: root `torch_transformer_benchmark.py`
  `BaselineTransformer`.
- Branch base: `bdf404f0a50fda387f63a052044b41d2c47a68b6`, committed
  29 August 2026 at 21:30:57 +08:00. It uses the deployed strided-SDPA
  dispatcher for every case in this sweep.
- Latest validation candidate: `12a37c621440d3edb2fd6dc9d5affe88cf367f4e`,
  committed 29 August 2026 at 21:34:20 +08:00. It exposes Person 3's packed-QKV
  implementation for all memory-feasible cases on validation branch
  `benchmark/person1-packed-qkv-cross-case`.

The branch base was found with `git merge-base master HEAD` before master gained
this research record. Baseline and base-dispatcher commands ran from a detached
worktree at `bdf404f`; packed commands ran from `12a37c6`.

## Environment and Protocol

- GPU: NVIDIA GeForce RTX 5080, capability 12.0, driver 616.56.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs.
- PyTorch / CUDA runtime / Triton: 2.13.0+cu130 / 13.0 / 3.7.1.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, x86-64, glibc 2.39.
- Official three-point screen: five seeds, float32, high matmul precision, TF32
  enabled, 3 seconds of settling per side, 30 paired CUDA-event samples, and
  automatic blocks targeting 50 ms.
- Direct packed-versus-strided diagnostic: both variants live in one process,
  use identical strictly copied weights and dispatcher compilation, settle for
  3 seconds each, and run 30 balanced paired samples. Case 3's final confirmation
  uses 10 seconds per side, 40 samples, and the 60-trial stress matrix.

Every harness JSON contains its exact command, timestamp, Git state, raw latency
samples, environment, correctness trials, and CUDA memory record. Command
patterns were:

```bash
# Immutable baseline and branch-base dispatcher, cwd at detached bdf404f
.venv/bin/python -m src.benchmark \
  --candidate reference --case CASE --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --repeats 30 --settle-seconds 3 \
  --sample-target-ms 50 --output baseline-caseCASE.json

.venv/bin/python -m src.benchmark \
  --candidate src.dispatcher --case CASE --device cuda --dtype float32 \
  --accuracy-trials 5 --seed 1234 --repeats 30 --settle-seconds 3 \
  --sample-target-ms 50 --output base-dispatcher-caseCASE.json

# Latest packed validation candidate, cwd at 12a37c6
.venv/bin/python -m src.benchmark \
  --candidate projections:PACKED_ALL --compile-user \
  --compile-mode MODE --benchmark-on-failure \
  --case CASE --device cuda --dtype float32 --accuracy-trials 5 --seed 1234 \
  --repeats 30 --settle-seconds 3 --sample-target-ms 50 \
  --output packed-caseCASE.json
```

`MODE` is `reduce-overhead` for cases 1-5 and 7-12, and `default` for case 13,
matching the deployed dispatcher policy.

## Results

The baseline column is the standalone immutable-reference run. Base and packed
cells show candidate median latency and paired speedup versus the immutable
baseline in their own process. `Direct gain` is the decisive same-process
strided-control median divided by packed median. All three reference points pass
5/5 default trials with zero failed elements; the largest default packed max-abs
error is 0.001316.

| Case | Baseline ms | Base dispatcher ms / speedup | Packed ms / speedup | Direct gain | Direct floor | Accuracy | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.3917 | 0.5346 / 3.295x | 0.5407 / 3.179x | 1.004x | ±1.35% | PASS 5/5 | Reject: within noise |
| 3 | 0.8892 | 0.1815 / 4.396x | 0.1331 / 5.233x | **1.192x** | **±0.84%** | PASS 60/60 stress | **Promote on RTX 5080** |
| 4 | 0.8215 | 0.2013 / 4.417x | 0.2290 / 4.009x | 1.026x | ±1.83% | PASS 5/5 | Reject: below gate |
| 5 | 2.4087 | 0.9257 / 2.583x | 0.8807 / 2.617x | 1.038x | ±3.01% | PASS 5/5 | Reject: below gate |
| 7 | 1.2162 | 0.3145 / 3.434x | 0.3003 / 3.738x | 1.035x | ±2.80% | PASS 5/5 | Reject: below gate |
| 8 | 11.9875 | 10.4621 / 1.120x | 10.1346 / 1.158x | 1.019x | ±1.46% | PASS 5/5 | Reject: below gate |
| 9 | 1.0660 | 0.5028 / 2.123x | 0.4963 / 2.145x | 1.019x | ±2.37% | PASS 5/5 | Reject: within noise |
| 10 | 1.3880 | 0.4709 / 2.814x | 0.4868 / 2.705x | 1.026x | ±1.77% | PASS 5/5 | Reject: below gate |
| 11 | 5.6362 | 1.1763 / 5.283x | 1.0033 / 5.495x | 1.002x | ±1.30% | PASS 5/5 | Reject: within noise |
| 12 | 0.8080 | 0.1746 / 4.691x | 0.1716 / 5.925x | 1.004x | ±1.62% | PASS 5/5 | Reject: within noise |
| 13 | 80.0333 | 13.1973 / 6.925x | 11.5513 / 6.975x | 1.008x | ±1.03% | PASS 5/5 | Reject: within noise |
| 6 | — | UNSUPPORTED | UNSUPPORTED | — | — | — | No memory-safe backend |
| 14 | — | UNSUPPORTED | UNSUPPORTED | — | — | — | No memory-safe backend |

The separate-process Case-4 candidates initially implied a greater-than-5%
regression. The required investigation used the direct same-process comparison,
which instead measured a small 1.026x gain outside its ±1.83% floor. This resolves
the apparent regression as inter-process clock/compiler variance, but the gain
is still too small to integrate.

## Raw Records

For each supported case, `baseline-caseN.json`, `base-dispatcher-caseN.json`,
and `packed-caseN.json` hold the skill-required three reference points.
`direct-ab-caseN.json` contains the same-process packed-versus-strided samples.
Case 3 additionally has [`direct-ab-case3-full.json`](direct-ab-case3-full.json)
for its final 10-second, 40-sample, 60-trial confirmation.

