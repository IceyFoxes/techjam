# Case 14 `sm_120` cold-start promotion — RTX 5080

Date: 2026-08-30 UTC

Implementation commit: `4f02b5f` (based on PR 16 head `8567f3f`).
Branch base: `fecf9943d7cbf4b5e166862087a4a3bb6e21d868`, committed
2026-08-30T10:52:00+08:00.

## Outcome

The warmed three-kernel attention core is faster, but compiling all three
Triton kernels in the first official forward is not. The accepted RTX 5080
policy keeps only the measured static `quad_update` kernel and uses the PyTorch
polynomial apply and diagonal paths. It also skips the dead final state update.

Across two independent fresh-process runs, candidate cold latency was
11.407-11.693 s. Exact Flash was 14.680-21.309 s. Comparing the fastest exact
control with the slowest candidate gives a conservative **1.255x cold speedup**.
Warm latency conservatively improves from 14.189 to 10.308 s, **1.376x**.

## Three reference points

| Reference | Revision | Accuracy | Full Case 14 latency | Peak allocated |
| --- | --- | --- | ---: | ---: |
| Immutable dense baseline | root `torch_transformer_benchmark.py` | UNSUPPORTED at full shape: dense scores need about 9.31 TiB; dense oracle at N=8192 PASS | — | — |
| Branch base exact Flash | `fecf994` | finite output; algebraically exact memory-safe route | 16.886 s | 13,931.858 MiB |
| Latest measured policy | `4f02b5f` | dense N=8192 PASS; Flash N=100000 PASS; full output finite | 11.407-11.693 s cold | 13,931.858 MiB |

The branch-base record is
`research/benchmarks/2026-08-30-rtx5080-b9506f3/full-case14-exact-flash.json`
on `master`. Against that recorded 16.886 s result, the slower candidate cold
run is 1.444x faster. The 1.255x figure above is the stricter same-code control.

Case 14 cannot satisfy the validation skill's normal full immutable-baseline
comparison because constructing the dense score tensor is infeasible. This is
reported as `UNSUPPORTED`, not silently treated as a pass. The two-layer B=1
oracle runs below provide executable numerical evidence for the attention
change itself.

## Correctness

Official criterion: `abs(error) <= 0.002 OR abs(error) <= 0.02 * abs(reference)`.

```text
N=8192 oracle=dense wscale=1.0 sigma=0.3335 failures=0/8388608 max=5.8594e-03 rms=3.3700e-04 PASS
N=100000 oracle=flash wscale=1.0 sigma=0.3339 failures=0/102400000 max=7.8125e-03 rms=3.4854e-04 PASS
Ran 200 tests in 60.754s — OK
```

Commands:

```text
PYTHONPATH=. .../.venv/bin/python src/validate_poly.py --n 8192 --oracle dense --seed 1234 --disable diag,apply
PYTHONPATH=. .../.venv/bin/python src/validate_poly.py --n 100000 --oracle flash --seed 1234 --disable diag,apply
PYTHONPATH=. .../.venv/bin/python -m unittest discover -s src/tests
```

## End-to-end policy A/B

Each candidate cold run used a newly created empty `TRITON_CACHE_DIR`. Each
process executed two full forwards: the first includes first-use compilation,
and the second is warm. No warm-up was hidden outside the timer.

| Policy | Cold | Warm | Verdict |
| --- | ---: | ---: | --- |
| Exact Flash control, fastest session | 14.680 s | 14.189 s | control |
| All three static Triton kernels, canonical ragged chunk | 15.547 s | 7.140 s | reject: 5.9% slower cold |
| Static kernels except diagonal | 14.281 s | 8.060 s | reject: cold margin too small |
| No Triton; PyTorch polynomial | 12.582 s | 11.903 s | valid cold win |
| Apply kernel only | 15.814 s | 10.304 s | reject: compile cost |
| **Update kernel only** | **11.407 s** | **10.308 s** | **accept** |
| **Update kernel only, independent repeat** | **11.693 s** | **9.845 s** | **accept** |

Records:

- `full-case14-exact-control-cold-warm.json`
- `full-case14-poly-static-padded-cold-warm.json`
- `full-case14-poly-static-nodiag-cold-warm.json`
- `full-case14-poly-pytorch-cold-warm.json`
- `full-case14-poly-applyonly-cold-warm.json`
- `full-case14-poly-updateonly-cold-warm.json`
- `final-case14-poly-cold-warm-repeat.json`
- `final-case14-exact-cold-warm-repeat.json`

`full-case14-poly-static-cold-warm.json` is **invalid for peak and warm-time
comparison**: the first version of the two-forward runner left the loop's final
`batch_slice` view alive, retaining the first 6.1 GiB output into the second
forward. The runner was fixed by deleting that view. The first forward still
documents the cost of six separate regular/ragged specializations, but the
record is not used in any accepted ratio.

## Configuration measurement

The official 16 GiB route streams one sample, so the attention kernels see
`M=16`, not PR 16's RTX 4060 `M=32`. The configuration key now includes `M`.

| Shape | Kernel | Selected measured configuration | Best time |
| --- | --- | --- | ---: |
| M16 C512 D64 V64 | quad_apply | BC128 BI2, 4 warps, 3 stages | 0.1000 ms |
| M16 C512 D64 V64 | quad_update | BC32 BI1, 4 warps, 2 stages | 0.1036 ms |
| M16 C512 D64 V64 | causal_diag | BC64 BK64, 4 warps, 3 stages | 0.0562 ms |
| M16 C352 D64 V64 | quad_apply | BC128 BI2, 4 warps, 3 stages | 0.0984 ms |
| M16 C352 D64 V64 | quad_update | BC64 BI2, 4 warps, 2 stages | 0.0877 ms |
| M16 C352 D64 V64 | causal_diag | BC64 BK64, 4 warps, 3 stages | 0.0490 ms |

Records: `config-sweep-m16-c512.json` and `config-sweep-m16-c352.json`, nine
interleaved rounds each. The final C=352 input is padded to C=512 for the apply
and diagonal kernels when those optional paths are enabled, reusing compiled
specializations. The accepted default skips the terminal update entirely, so
only the C=512 update specialization compiles.

The warmed full-fusion attention core remains a real 2.611x speedup over exact
Flash in its session, above the 4.3% A/A floor. It is rejected only as the
default one-shot policy because end-to-end compilation cost reverses that win.
See `attention-core-b1-static.json` and the three `attention-core-b1-ab-*.json`
records.

## Environment

- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 threads
- GPU: NVIDIA GeForce RTX 5080, compute capability 12.0
- Driver / CUDA runtime: 616.56 / 13.0
- PyTorch: 2.13.0+cu130
- Python: 3.12.3
- OS: Linux 6.6.114.1 Microsoft WSL2, glibc 2.39
- Disk: 1,081,101,176,832 bytes total; free space is recorded per JSON

## Reproduction

```text
PYTHONPATH=. .../.venv/bin/python src/sweep_poly_configs.py --m 16 --c 512 --d 64 --v 64 --rounds 9 --output <new-json>
PYTHONPATH=. .../.venv/bin/python src/sweep_poly_configs.py --m 16 --c 352 --d 64 --v 64 --rounds 9 --output <new-json>
TRITON_CACHE_DIR=<new-empty-directory> PYTHONPATH=. .../.venv/bin/python src/extreme_smoke.py --case 14 --dtype float16 --seed 1234 --forwards 2 --output <new-json>
PYTHONPATH=. .../.venv/bin/python src/extreme_smoke.py --case 14 --dtype float16 --seed 1234 --forwards 2 --disable-poly --output <new-json>
```
