# RTX 5080 Compiler Measurements

## Status

- Date: 29 August 2026.
- Benchmark revision: `6bde871dd65051fcace36971b27a86771365ba1e`.
- Candidate: the reference-equivalent `compiler` scaffold, with only the
  candidate side passed through `torch.compile` by the harness.
- Timing: paired A/B mode, 10-second settling, CUDA events, compilation and graph
  recording excluded from timed samples.
- Records: [`../benchmarks/2026-08-29-rtx5080-6bde871/`](../benchmarks/2026-08-29-rtx5080-6bde871/README.md).

These are whole-model framework controls, not final integrated results. The
Person 2 SDPA and Person 3 projection implementations were not present in the
candidate during these runs.

## Environment

| Component | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0, 16,303 MiB |
| Driver / CUDA runtime | 616.56 / 13.0 |
| PyTorch / Triton | 2.13.0+cu130 / 3.7.1 |
| Python | 3.12.3 |
| OS | Linux 6.6.114.1 Microsoft WSL2, glibc 2.39 |
| GPU power limit / max SM clock | 360 W / 3,090 MHz |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs, x86_64 |
| Disk | 1,081,101,176,832 bytes total; free space recorded per JSON run |

The original environment lacked `/usr/include/python3.12/Python.h`, so Triton's
first CUDA helper build failed. Installing Ubuntu `python3.12-dev` fixed the
environment without changing the virtual environment.

## Preserved Results

All correctness counts below use the executable absolute-error OR relative-error
rule and require zero failed elements.

| Case | dtype / mode | Baseline ms | Candidate ms | Speedup | Noise floor | Accuracy | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | fp32 / default | 0.7765 | 0.3510 | 2.212x | ±24.63% | PASS 5/5, max abs 0.000484 | Valid control |
| 2 | fp32 / reduce-overhead | 0.9085 | 0.1380 | 6.581x | ±83.26% | PASS 5/5, max abs 0.000484 | Leading fp32 case-2 mode |
| 2 | fp32 / max-autotune | 0.8885 | 0.1345 | 6.609x | ±192.77% | PASS 5/5, max abs 0.000841 | Reject: no gain over reduce-overhead, less headroom |
| 2 | fp16 / reduce-overhead | 0.8210 | 0.0888 | 9.250x | ±137.97% | PASS 8/8, max abs 0.005859 | Promising case-specific route |
| 8 | fp32 / reduce-overhead | 11.8713 | 10.8436 | 1.095x | ±1.08% | PASS 5/5, max abs 0.000879 | Valid but secondary to Person 3 |
| 8 | fp32 / max-autotune | — | — | — | — | **FAIL 5/5**, 7,899 elements, max abs 0.004585 | Reject |
| 13 | fp32 / default | 81.2795 | 25.5641 | 3.179x | ±5.65% | PASS 5/5, max abs 0.000775 | Valid control; compare with SDPA composition |
| 13 | fp16 / reduce-overhead + cast emulation | — | — | — | — | **FAIL 2/8**, 4 elements, max abs 0.006836 | Reject; eager fallback |

The case 2 confidence intervals are wide because occasional baseline samples are
far slower than the modal cluster. The measured improvements remain outside
their run-specific noise floors, but the exact 6.58x and 9.25x values should not
be presented as final scores. The case 8 and 13 conclusions have much tighter
decision margins.

## Numerical Findings

### Max-autotune changes the computation

On case 2, `max-autotune` roughly doubled maximum absolute error relative to
default/reduce-overhead while still passing. During autotuning, PyTorch selected
between ATen and Triton GEMMs and discarded configurations requiring 131,072
bytes shared memory against the GPU's 101,376-byte limit. On case 8, its selected
path exceeded tolerance on every seed. The harness correctly skipped timing.

Therefore `max-autotune` must be an explicit `(shape, dtype, numerical flags)`
route, never a global setting.

### Precision-cast emulation helps but does not make float16 universal

Without `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`, exploratory case 7 float16
failed all eight seeds. Enabling the switch made case 7 pass eight of eight, but
the preserved case 13 run still failed seeds 1238 and 1240 with two elements
each. Triton also warned that `CUDA_HOME` was unset and its bundled libdevice
could introduce minor `pow` differences; this host has no separate CUDA toolkit
libdevice to substitute.

The only safe conclusion is per-case validation. Case 2 float16 is a promising
route; case 13 float16 requires the eager-compatible fallback unless a different
implementation passes a larger adversarial seed matrix.

## Additional Exploratory Routing Evidence

These runs were used to choose what to preserve, not promoted as benchmark
checkpoints:

- case 1 float32: default 1.900x; reduce-overhead 2.713x; with 25% padding,
  reduce-overhead 2.775x and five passing seeds;
- case 3 float32 reduce-overhead: 5.193x, three passing seeds;
- case 12 float32: default 1.850x; reduce-overhead 4.635x;
- case 13 float32: default 3.209x and reduce-overhead 3.197x in the initial
  sweep, showing no CUDA Graph benefit on the long-sequence path; and
- case 7 float16: raw reduce-overhead failed 8/8 seeds; cast emulation passed
  8/8 in that exploratory run but is not a global solution.

These values guide the next experiment matrix but do not replace preserved final
runs for cases 1, 3, 4, 5, 7, 9-12.

## Limits

- Compilation time and peak compiler memory were not measured.
- Cases 4-6, 9-11, 12, and 14 do not yet have preserved Person 1 checkpoints.
- The case 2 fp16 pass is only eight Gaussian seeds at the default input scale
  and zero padding; it is not an acceptance-quality adversarial matrix.
- No measurement yet composes compiler modes with Person 2 SDPA or Person 3
  packed projections.
- CUDA Graph retained-memory cost must be measured before using it on cases 5,
  6, 8, 13, or 14.

## Source

The commands, timestamps, shapes, raw samples, correctness details, Git state,
and environment are stored in the linked JSON records. Benchmark behavior comes
from this repository's immutable `torch_transformer_benchmark.py` and
`src.benchmark` at revision `6bde871dd65051fcace36971b27a86771365ba1e`,
accessed 29 August 2026.
