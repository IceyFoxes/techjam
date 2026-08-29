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

**Corrected 30 August 2026 — it can now be installed on this host.** Two
obstacles were removed:

1. *Driver.* CUDA 13.x requires driver >= 580; this host was on 566.26
   (CUDA 12.7). Updating the **Windows** driver to 616.56 raised it to CUDA UMD
   13.4 — the same driver Person 1's RTX 5080 runs, so both machines are now on
   identical driver/CUDA stacks. On WSL2 the driver comes from Windows and
   cannot be changed from inside Linux.
2. *Python.* `numpy==2.5.2` requires Python >= 3.12, and this host has only 3.10
   (plus asdf 3.11.9), with no `python3-venv` package and no sudo. Rather than
   compile 3.12, `uv` fetched a prebuilt standalone CPython 3.12.14 in ~17 s
   with no root. `python3 -m venv --without-pip` plus PyPA's `get-pip.py`
   remains necessary for any 3.10 venv here.

The pinned stack now lives in `.venv-cu130` (git-ignored): Python 3.12.14,
torch 2.13.0+cu130, triton 3.7.1, CUDA runtime 13.0. `src/tests` passes 62/62 on
it, and `src.dispatcher` now selects `compiled-sdpa` for the eligible official
cases instead of falling back to reference arithmetic.

**`.venv` (Python 3.10, torch 2.6.0+cu124) is deliberately retained.** Every
measurement in this document was taken on it, and paired-timing ratios are only
comparable within a single build. Any figure re-measured on `.venv-cu130` is a
separate run and must not be compared against the numbers below.

One caution for re-measurement: `uv pip install` failed the first time with a
network timeout during extraction. Use `UV_HTTP_TIMEOUT=600` and
`UV_CONCURRENT_DOWNLOADS=4`.

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

## Whole-model speedup, float32 + SDPA — complete in-scope sweep

All twelve in-scope cases. TF32 on (harness default), `padding_ratio=0`,
correctness checked over 3 seeds under the official criterion, paired timing with
30 repeats.

| Case | B | H | N | `d_h` | correct | attn share | base ms | SDPA ms | speedup | floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 64 | 4 | 1024 | 32 | PASS | 65.1% | 385.747 | 60.885 | **6.336x** | ±2.6% |
| 11 | 64 | 16 | 128 | 8 | PASS | 65.9% | 24.881 | 5.013 | **4.964x** | ±8.4% |
| 5 | 128 | 4 | 128 | 32 | PASS | 50.9% | 15.555 | 5.331 | **2.918x** | ±4.5% |
| 2 | 1 | 4 | 128 | 32 | PASS | 28.3% | 7.805 | 5.207 | 1.499x | ±14.7% |
| 9 | 64 | 1 | 128 | 128 | PASS | 23.8% | 8.176 | 5.605 | 1.459x | ±18.9% |
| 3 | 4 | 4 | 128 | 32 | PASS | 35.3% | 7.875 | 5.408 | 1.456x | ±7.8% |
| 10 | 64 | 2 | 128 | 64 | PASS | 28.4% | 6.996 | 4.853 | 1.441x | ±10.4% |
| 4 | 16 | 4 | 128 | 32 | PASS | 38.8% | 7.727 | 5.371 | 1.439x | ±8.6% |
| 12 | 64 | 4 | 32 | 32 | PASS | 29.1% | 7.026 | 4.943 | 1.421x | ±11.8% |
| 7 | 64 | 4 | 128 | 8 | PASS | 48.5% | 8.962 | 6.718 | 1.334x | ±13.6% |
| 1 | 64 | 4 | 128 | 32 | PASS | 36.4% | 31.648 | 26.369 | 1.200x | ±0.1% |
| 8 | 64 | 4 | 128 | 256 | PASS | 15.7% | 61.549 | 58.788 | 1.047x | ±0.3% |

**Every case passes and every case gains.** Geometric mean ≈ **1.94x**. Cases 2
and 9 have wide floors (±14.7%, ±18.9%) and should be re-measured with more
repeats before being quoted individually.

The "attn share" column is a **lower bound**: it counts only `aten::` operators
inside stages 3-7 and excludes the `.contiguous()` copies and dtype casts that
attention causes but that are attributed to `aten::copy_`. This is why measured
speedups on cases 11 and 13 exceed `1/(1 - share)` — the true attention share is
higher than the column states. Do not use it as an Amdahl ceiling.

### Correction: case 8 does not regress at whole-model level

An earlier version of this document concluded from the attention-only
microbenchmark that case 8 (`d_h=256`) must stay on the eager path, because
isolated attention measured **0.64x** with SDPA. The whole-model measurement
contradicts that: **1.047x ±0.3%**, a small but statistically significant gain.

The microbenchmark was misleading because it constructed the causal mask **once,
outside the timed region**, whereas the real reference rebuilds
`torch.ones((N,N), bool).triu(1)` per layer per forward. The isolated test
therefore gave the baseline a free mask and understated SDPA's benefit
everywhere. **Prefer the whole-model numbers.** The attention-only table below is
retained for the relative picture but should not drive routing decisions.

**`sdpa_fp32` beats float16 internals on every case with a trustworthy
measurement.** Casting Q/K/V down and the result back costs more than the
tensor-core throughput it buys at these sizes. The one case where float16 leads
(13) has a ±110% floor and cannot be relied on. This removes the main argument
for lever L8 and simplifies the plan: **use plain float32 SDPA.**

## Dropping the `.contiguous()` copies

`_split_heads` does `.transpose(1,2).contiguous()` on each of Q, K and V, and the
output path does `.transpose(1,2).contiguous().view(...)`. Since SDPA accepts
strided inputs, the copies may be unnecessary. Tested by replacing them with a
plain `.transpose(1,2)` view and `.reshape()` on the way out.

| Case | SDPA + `.contiguous()` | SDPA + strided view | relative gain |
| --- | --- | --- | --- |
| 13 | 6.379x ±4.9% | **6.908x ±3.0%** | +8.3% |
| 11 | 5.039x ±7.6% | **5.353x ±2.9%** | +6.2% |
| 1 | 1.407x ±10.6% | **1.577x ±10.9%** | +12.1% |
| 8 | 1.044x ±0.3% | **1.119x ±0.2%** | +7.2% |

All four PASS the official criterion. Every case improves, by 6-12%.

Confidence varies: case 8's floors are tight (±0.3% and ±0.2%), so 1.044 -> 1.119
is unambiguous. Case 13's 8.3% improvement is comparable to its ±3-5% floors, and
case 1's floors are ±10%, so those individually are weaker. **The consistency of
the direction across all four cases is the stronger evidence** — a noise artefact
would not favour the same variant every time.

This matters for scoping: the `.contiguous()` copies were previously filed as a
Person 3 boundary issue, but the input side is inside `BaselineSelfAttention` and
therefore **inside Person 2's module**. Only the output-side reshape touches the
stage 8 boundary.

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
| 8 | `d_h`=256 | 0.64x in isolation — **but 1.047x whole-model; see correction above** | PASS |

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
