# Fused Polynomial Attention Kernel — RTX 4060 Laptop

Phase 1 acceptance run for the Triton kernel specified in
[`../../attention-softmax/triton-kernel-spec.md`](../../attention-softmax/triton-kernel-spec.md).

Date: 30 August 2026. Commit: see `attention-core.json` (`git` block).
Verdict: **ACCEPTED** — 342.4 ms against the 360 ms acceptance threshold.

## Latency

Attention core only, one sample x one layer, `H=16, N=100000, d_h=64`, float16.
This is the unit of work case 14's route actually executes: it streams 1-2
samples at a time (`choose_batch_chunk_size` selected 2 on a 24 GB L4). Best of
3 timed reps after one warmup, which absorbs Triton autotuning.

| path | time | vs exact | correct |
| --- | ---: | ---: | :---: |
| exact flash SDPA (case 14's current route) | 725.7 ms | 1.00x | yes |
| polynomial, dense PyTorch | 662.8 ms | 1.09x | yes |
| **polynomial, fused Triton** | **342.4 ms** | **2.12x** | **yes** |

Also **1.94x over the dense-PyTorch polynomial path**, which is the figure that
isolates the kernel's own contribution from the algorithm's.

The 725.7 ms and 662.8 ms figures reproduce the 719.8 ms and 603.9 ms recorded
in [`../../attention-softmax/long-sequence-attention.md`](../../attention-softmax/long-sequence-attention.md)
section 4.4 to within run-to-run variation. The PyTorch polynomial path is ~10%
slower than its earlier measurement; it now writes both the numerator and
denominator quadratic states through the same code path, and no attempt was made
to re-tune it.

## Correctness

Official criterion `abs <= 0.002 OR rel <= 0.02`, **zero** failures required.
Full two-layer case-14-shaped model, `B=1`, float16.

| oracle | N | failures | max abs err | rms err |
| --- | ---: | ---: | ---: | ---: |
| dense reference | 4096 | 0 / 4,194,304 | 7.8125e-03 | 2.43e-04 |
| dense reference | 8192 | 0 / 8,388,608 | 7.8125e-03 | 3.36e-04 |
| exact flash | 16384 | 0 / 16,777,216 | 7.8125e-03 | 3.59e-04 |
| exact flash | 32768 | 0 / 33,554,432 | 7.8125e-03 | 3.61e-04 |
| exact flash | 65536 | 0 / 67,108,864 | 7.8125e-03 | 3.56e-04 |
| exact flash | 100000 | 0 / 102,400,000 | 7.8125e-03 | 3.52e-04 |

The max error is flat at 7.8125e-03 across every `N`, which is exactly one
float16 ulp at the output magnitude. The approximation error sits below the
representation noise floor, so what that column measures is fp16 rounding rather
than the polynomial.

Two oracles are used because neither suffices alone: the dense reference is
authoritative but its `N x N` score tensor caps it near N=8192 in 8 GiB, and the
exact-flash oracle reaches N=100000 but cannot see fp16 reduction-order
differences.

## Guard calibration

Swept by scaling the `q_proj`/`k_proj` weights of both models, at N=8192 against
the dense reference.

| sigma | failures | |
| ---: | ---: | --- |
| 0.3339 | 0 | the benchmark's own operating point |
| 0.4040 | 0 | PASS |
| 0.4808 | 0 | **largest verified pass** |
| 0.5217 | 21 | **first observed failure** |
| 0.5642 | 308 | FAIL |
| 0.6544 | 9,566 | FAIL |
| 0.7512 | 56,801 | FAIL |

`SIGMA_CEILING = 0.45`, above the operating point's seed-to-seed spread
(0.3327-0.3343) and below the largest verified pass, so the route never runs in
the untested band. Conservative for the target shape: measured at N=8192, and
attention contributes less to the residual stream as `N` grows.

## Commands

```bash
.venv/bin/python -m src.bench_poly --n 100000 \
  --output research/benchmarks/2026-08-30-rtx4060-poly/attention-core.json
.venv/bin/python -m src.validate_poly --n 4096   --oracle dense
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 16384  --oracle flash
.venv/bin/python -m src.validate_poly --n 32768  --oracle flash
.venv/bin/python -m src.validate_poly --n 65536  --oracle flash
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
for w in 1.0 1.1 1.2 1.25 1.3 1.4 1.5; do
  .venv/bin/python -m src.validate_poly --n 8192 --oracle dense --scale-qk-weights $w
done
.venv/bin/python -m unittest discover -s src/tests
```

## Environment

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB, 24 SMs, sm_89 |
| Driver | 616.56 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| CUDA | 13.0 |
| Python | 3.12 |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 |

Full machine-readable environment is in `attention-core.json`.

## Scope and limitations

- **Attention core only.** This is not a whole-case-14 measurement. Case 14
  cannot run end to end on this 8 GiB card: it needs a 12.21 GiB floor for the
  device-resident input and output alone.
- **This is an approximation**, not an algebraically exact rewrite. It passes the
  criterion with zero failures at every tested `N`, and its accuracy depends on
  the benchmark's small scores — a property of the random initialisation, not of
  attention. The guard exists for exactly that reason.
- **Single-GPU evidence.** No RTX 5080 or L4 numbers.
- Latency is best-of-3; no noise floor was established by repeated runs.
