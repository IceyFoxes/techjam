# Attention and Softmax Measurements

All numbers below were taken on the environment in the first section. They are
exploratory measurements taken during research, not preserved competition runs;
preserved runs belong under `research/benchmarks/<date>-<gpu>-<commit>/` per
[`AGENTS.md`](../../AGENTS.md).

Status: current as of 29 August 2026.

## Environment

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, sm_89 (Ada), 8 GB |
| Driver / CUDA | 566.26 / 12.7 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.10.12 |
| OS | WSL2, Linux 6.18.33.2-microsoft-standard-WSL2 |
| Measured bandwidth budget | ~272 GB/s |
| Measured fp16 matmul throughput | 21.9 TFLOP/s (2048³, sustained) |

This deviates from the team's pinned `requirements.txt` (torch 2.13.0+cu130,
numpy 2.5.2), which needs Python 3.12 and could not be installed on this host.

**Timing method.** All speedups use the paired timing mode added in
[`src/infra/timing.py`](../../src/infra/timing.py): both models are interleaved
sample by sample so boost-clock drift cannot favour either, and every result
carries a noise floor. **Treat any speedup inside its noise floor as no change.**
Absolute latency on this laptop depends on how long the GPU has been loaded and
is not comparable across runs; only within-run ratios are meaningful.

## Where the time actually goes

Operator-level device time, case 13 (`B=64 N=1024 d_h=32`), float16, one forward.
Ratios are against `aten::bmm`, the actual attention matmul.

| Work | Device time | vs. attention matmul |
| --- | --- | --- |
| `aten::bmm` (the real attention math) | 18.6 ms | 1.0x |
| `aten::_softmax` (`softmax_warp_forward<float,...>`) | 36.9 ms | 2.0x |
| `aten::masked_fill_` (causal + padding, n=17) | 38.6 ms | 2.1x |
| dtype casts from `.float()` / `.to(dtype)` | 53.9 ms | 2.9x |
| `aten::mul` (the `* scale` pass) | 18.4 ms | 1.0x |
| `aten::copy_` + `Memcpy DtoD` (`.contiguous()`) | 131.1 ms | 7.1x |

**For every unit of real attention matmul the eager path spends roughly nine
units moving memory around it.** The `<float,...>` template confirms the float32
softmax round-trip.

This differs sharply by shape:

- **Case 8** (`d_h=256`): `aten::addmm` is 31.5% and `bmm` only 3.1%. Projection
  and FFN bound — Person 3's scope, not attention.
- **Case 1** (`N=128`): flat profile, no dominant operator; launch-bound.
- **Case 13**: attention memory traffic dominates.

Caveat: kernel-name heuristics cannot separate attention GEMMs from projection
and FFN GEMMs, since both appear as `cutlass_*` / `ampere_*gemm_*`. The table
above is operator-level (`aten::`) attribution, which does distinguish them.

## Whole-model speedup, float32 + SDPA

Single settle, one shared baseline, variants interleaved. TF32 on
(harness default), `padding_ratio=0`.

| Case | `d_h` | N | `sdpa_fp32` | `sdpa_fp16` internals |
| --- | --- | --- | --- | --- |
| 13 | 32 | 1024 | **6.41x** ±3.5% | 9.51x ±110% (floor too wide to trust) |
| 7 | 8 | 128 | **1.68x** ±17.3% | 1.18x ±15.0% |
| 1 | 32 | 128 | **1.46x** ±9.7% | 1.23x ±8.8% |
| 12 | 32 | 32 | **1.41x** ±9.9% | 1.23x ±8.9% |

All marked SIGNIFICANT against their own noise floor.

**`sdpa_fp32` beats float16 internals on every case with a trustworthy
measurement.** Casting Q/K/V down and the result back costs more than the
tensor-core throughput it buys at these sizes. The one case where float16 leads
(13) has a ±110% floor and cannot be relied on. This removes the main argument
for lever L8 and simplifies the plan: **use plain float32 SDPA.**

## Attention-only microbenchmark

Isolated attention (no projections/FFN), baseline formulation vs SDPA.

float32:

| Case | shape | speedup | tolerance |
| --- | --- | --- | --- |
| 13 | N=1024, `d_h`=32 | 5.75x ±12.2% | PASS, `max_abs` 2.7e-06 |
| 12 | N=32 | 2.09x ±1.7% | PASS |
| 1 | N=128, `d_h`=32 | 1.70x ±0.8% | PASS |
| 11 | H=16, `d_h`=8 | 1.67x ±1.9% | PASS |
| 7 | `d_h`=8 | 1.61x ±6.5% | PASS |
| **8** | **`d_h`=256** | **0.64x — regression** | PASS |

float16 (rejected on correctness, listed for completeness):

| Case | speedup | tolerance |
| --- | --- | --- |
| 13 | 38.8x | FAIL |
| 1 | 6.27x | FAIL |
| 11 | 6.46x | FAIL |
| 7 | 6.23x | FAIL |
| 12 | 0.94x | FAIL |
| 8 | 1.75x | PASS (attention-only) |

**Case 8 must not use SDPA** — it regresses to 0.64x in float32, and attention is
only ~6% of that case's time anyway.

## Bitwise-exact levers (L6 + L7)

Cached causal mask plus skipping all-true padding masks, with the `.all()` test
hoisted to one host sync per forward. Verified `max_abs = 0`, so these are safe
in any dtype.

float16 whole-model, `padding_ratio=0`:

| Case | speedup | noise floor |
| --- | --- | --- |
| 12 | 1.427x | ±9.1% |
| 9 | 1.363x | ±9.3% |
| 13 | 1.214x | ±0.2% |
| 11 | 1.204x | ±0.7% |
| 7 | 1.166x | ±4.7% |
| 1 | 1.150x | ±0.7% |

Geometric mean ≈ **1.25x, bitwise exact**.

**Contingency:** at `padding_ratio=0.3` this collapses to 0.994x / 0.936x /
0.941x (cases 13 / 1 / 11). Almost all of the gain is lever L7 — skipping a
padding mask that does nothing — which only applies when the mask is all-true.
The cached mask alone (L6) is roughly neutral.

An earlier version of this variant was **slower** on small shapes because
`bool(valid_token_mask.all())` was evaluated per layer, forcing a GPU-to-host
sync each time. Hoisting it to one sync per forward turned 0.82-0.95x into
1.15-1.43x. Any implementation of L7 must avoid per-layer syncs.

## Reproduction

Exploratory scripts used for these measurements were kept outside Git. The
durable equivalents are:

```bash
# whole-model A/B with noise floor, official shapes
.venv/bin/python -m src.benchmark --candidate attention \
  --device cuda --dtype float32 --case 13 --repeats 40 --settle-seconds 20

# operator attribution (requires the CUPTI fix below)
.venv/bin/python -m src.profile --candidate reference --case 13 \
  --device cuda --dtype float32 --models baseline
```

**Profiling prerequisite.** PyTorch 2.6.0+cu124 bundles CUPTI 12.4.127 while this
host's driver exposes CUDA 12.7. The mismatch causes the profiler to emit **zero
GPU kernel events** with `self_device_time_total == 0`, silently and with no
error. Raw CUPTI still delivers kernel records, so the fault is at the
CUPTI/kineto version boundary, not a WSL2 platform limit. Fix:

```bash
.venv/bin/python -m pip install nvidia-cuda-cupti-cu12==12.6.80
```

pip reports a pin conflict against torch's declared `==12.4.127`; torch computes
correctly regardless, since CUPTI is only used for profiling. Teammates on
torch 2.13.0+cu130 are unlikely to hit this.

## Sources

- **Reference implementation:** this repository's
  [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py) at
  commit `7eb8fb1`. Symbols used: `BaselineSelfAttention.forward` (lines 85-122)
  for the measured eager path, `compare_outputs` (lines 289-353) for the
  tolerance criterion, and `generate_random_case` (lines 234-272) for the input
  distribution and padding-mask construction.

- **Official shapes:** [`src/cases/task_shapes.json`](../../src/cases/task_shapes.json),
  transcribed from the organizer Appendix screenshot
  [`task_shapes.png`](../../task_shapes.png).

- **PyTorch Profiler / Kineto CUPTI behaviour.**
  <https://docs.pytorch.org/docs/stable/profiler.html>. Accessed 29 August 2026.
  Documents `ProfilerActivity.CUDA` and the CUPTI-backed device-time attribution
  that returns zero under the version mismatch described above.
