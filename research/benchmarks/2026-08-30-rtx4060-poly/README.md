# Fused Polynomial Attention Kernel — RTX 4060 Laptop

Phase 1 acceptance run for the Triton kernel specified in
[`../../attention-softmax/triton-kernel-spec.md`](../../attention-softmax/triton-kernel-spec.md).

Date: 30 August 2026. Commit: see the JSON files (`git` block).
Verdict: **ACCEPTED** — 2.41x at the real chunk shape.

## Correction, 30 August 2026 — supersedes `attention-core.json`

The original run in `attention-core.json` recorded **342.4 ms / 2.12x**. That
figure is **superseded and should not be quoted**, for three independent
reasons found while investigating a VRAM overrun:

1. **It used contiguous `[B, H, N, D]` inputs.** The real module hands the
   attention path *strided views* from `_split_heads_view`. With contiguous
   inputs the internal `reshape` is free; with strided ones it copies all of
   q/k/v. On realistic inputs the same code measured **469.5 ms**.
2. **A bug inflated it.** The `exact_prefix` SDPA call passed 3-D tensors, which
   every fused backend rejects, so it silently fell back to the quadratic math
   backend — materialising an `[M, 4096, 4096]` score matrix. That cost
   **2.4 GiB and ~72 ms** on every call.
3. **The variants were timed back to back.** On this laptop GPU the third
   variant runs on a throttled card. Measured sequentially the polynomial path
   read 392 ms; interleaved, and alone, it reads 275-300 ms.

The numbers below are from the corrected harness: strided inputs, the 4-D
prefix fix, the sliced scan, and interleaved timing.

## Latency

Attention core only, one sample x one layer, `H=16, N=100000, d_h=64`, float16.
This is the unit of work case 14's route actually executes: it streams 1-2
samples at a time (`choose_batch_chunk_size` selected 2 on a 24 GB L4). Best of
3 timed reps after one warmup, which absorbs Triton autotuning.

Interleaved A/B/C timing, best of 4 reps after warmup, strided inputs.

**B=1** (`attention-core-v3-b1.json`):

| path | time | vs exact | peak MiB |
| --- | ---: | ---: | ---: |
| exact flash SDPA | 711.9 ms | 1.00x | 201 |
| polynomial, dense PyTorch | 428.1 ms | 1.66x | — |
| **polynomial, fused Triton** | **269.9 ms** | **2.64x** | **233** |

**B=2 — the shape case 14's route actually streams** (`attention-core-v3-b2.json`):

| path | time | vs exact | peak MiB |
| --- | ---: | ---: | ---: |
| exact flash SDPA | 1414.0 ms | 1.00x | 404 |
| polynomial, dense PyTorch | 845.1 ms | 1.67x | — |
| **polynomial, fused Triton** | **328.1 ms** | **4.31x** | **471** |

### Where the 2.41x became 4.31x

A kernel-level profile at B=2 showed the two Triton kernels were only **34%** of
GPU time. The exact diagonal block was 195 ms of 547 -- more than both kernels
combined -- because it was computed in float32:

| change | effect at N=100000, B=2 |
| --- | ---: |
| baseline (v2) | 590.4 ms, 2.41x |
| Gram-matrix denominator | 564.4 ms, 1.046x |
| **float16 diagonal block** | **344.5 ms, 1.714x** |
| both | **322.7 ms, 1.830x** |

- **float16 diagonal block.** Upcasting to float32 forced the PV product onto an
  `ampere_sgemm_128x128_nn` -- off the tensor cores entirely -- and tripled the
  traffic of the `[M, C, C]` block through `exp`, the mask and the row sum. Only
  the row sum still accumulates in float32.
- **Gram-matrix denominator.** `phi2(a) . sum_j phi2(b_j) = a^T G a` with
  `G = sum_j b_j b_j^T` of shape `[D, D]`, so the denominator never needed the
  `[D*D, 1]` feature state. Removes one `quad_apply` and one `quad_update` per
  chunk, and drops peak overhead further (67 MiB at B=2, from 131).

Chunk length was A/B'd at the same time and **512 is confirmed optimal**: 256
measured 530.0 ms and 1024 measured 423.1 ms, against 322.7 ms at 512.

## Peak VRAM — the constraint that prompted the correction

The polynomial route's overhead above exact flash, at `N=100000`:

| | before | after v2 | after v3 | total reduction |
| --- | ---: | ---: | ---: | ---: |
| B=1 | +2834 MiB | +64 MiB | **+32 MiB** | 89x |
| B=2 | +6773 MiB | +131 MiB | **+67 MiB** | 101x |

The +6.8 GiB at B=2 is what pushed a 13 GiB run past 16 GiB. Two causes, both
now fixed: the 3-D SDPA fallback (2.4 GiB, fixed-size), and full-length
contiguous copies from `reshape` on strided views plus the pre-scaled `a_all` /
`b_all` (~2.4 GiB at B=2, O(N)).

Pinned by `src/tests/test_poly_attention.py::PolyMemoryTests`, which asserts the
peak stays under 2x the q/k/v input bytes **using strided inputs** — with
contiguous inputs the test would measure nothing.

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
- Latency is best-of-4 interleaved; no formal noise floor was established. The
  spread seen while investigating was material: 256-305 ms across repeats at
  B=1, so treat 2.4x as approximate rather than precise.
- **This laptop GPU throttles under sustained load.** Any future comparison here
  must interleave variants rather than run them back to back; measuring
  sequentially inflated the last-measured variant by ~30%.
