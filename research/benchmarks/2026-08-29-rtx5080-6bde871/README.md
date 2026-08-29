# Person 1 Compiler Checkpoints — RTX 5080 — 29 August 2026

## Environment and Revision

- Git revision under test: `6bde871dd65051fcace36971b27a86771365ba1e`.
- GPU: NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0.
- Driver / CUDA runtime: 616.56 / 13.0.
- PyTorch / Triton: 2.13.0+cu130 / 3.7.1.
- Python: 3.12.3.
- OS: Linux 6.6.114.1 Microsoft WSL2, glibc 2.39.
- CPU: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs.
- Timing: paired CUDA-event mode, 10-second settling, fixed official shapes.

Each JSON includes the exact command, UTC timestamp, shape, dtype, tolerances,
per-trial correctness, raw latency samples when valid, speedup, noise floor, GPU
state, CPU/OS/disk information, and Git state. Later files report a dirty tree
only because earlier immutable JSON records already existed in this new run
directory; the benchmark code revision remained the hash above.

## Valid Checkpoints

| File | Result | Status |
| --- | --- | --- |
| [`case2-fp32-default.json`](case2-fp32-default.json) | 2.212x, PASS 5/5 | Default-mode control |
| [`case2-fp32-reduce-overhead.json`](case2-fp32-reduce-overhead.json) | 6.581x, PASS 5/5 | Leading case-2 fp32 compiler route |
| [`case2-fp32-max-autotune.json`](case2-fp32-max-autotune.json) | 6.609x, PASS 5/5 | Valid but rejected: no gain over reduce-overhead, more error and compile work |
| [`case2-fp16-reduce-overhead.json`](case2-fp16-reduce-overhead.json) | 9.250x, PASS 8/8 | Promising case-specific low-precision route; not yet adversarially accepted |
| [`case8-fp32-reduce-overhead.json`](case8-fp32-reduce-overhead.json) | 1.095x ±1.08%, PASS 5/5 | Conservative compiler control for Person 3 integration |
| [`case13-fp32-default.json`](case13-fp32-default.json) | 3.179x ±5.65%, PASS 5/5 | Compiler control for Person 2 SDPA integration |

Case 2's run-specific noise floors are wide, so its exact speedups are not final
scores even though the improvements are significant under the harness rule.

## Invalid Regression Records

| File | Failure | Replacement / action |
| --- | --- | --- |
| [`case8-fp32-max-autotune-invalid.json`](case8-fp32-max-autotune-invalid.json) | FAIL 5/5; 7,899 failed elements; max abs 0.004585 | Use case 8 reduce-overhead control and Person 3 route; never enable max-autotune globally |
| [`case13-fp16-reduce-overhead-emulate-casts-invalid.json`](case13-fp16-reduce-overhead-emulate-casts-invalid.json) | FAIL 2/8; four failed elements even with precision-cast emulation | Use eager-compatible float16 fallback pending a proven backend |

No recorded file has been overwritten.
