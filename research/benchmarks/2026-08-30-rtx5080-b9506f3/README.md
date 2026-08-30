# PR 16 Case 14 A/B validation

Date: 2026-08-30 (UTC)

This run evaluates PR 16 (`fused-kernal`, `b9506f37e5bee2485519eb9174fb2cbc8b6f91da`) against its branch base (`fecf9943d7cbf4b5e166862087a4a3bb6e21d868`) for official Case 14. The immutable dense reference cannot execute this full shape because its attention score tensor would require approximately 9.31 TiB, so full-shape correctness is represented by a finite-output smoke test plus separate dense-oracle and exact-Flash-oracle checks.

Public source: [PR 16 — Fused polynomial attention kernel for case 14](https://github.com/IceyFoxes/techjam/pull/16), accessed 2026-08-30. The PR supplies the guarded degree-2 polynomial reference, Triton kernels, validation tools, and an opt-in Case 14 attention class; this record tests both the submitted default and the intended locally enabled route.

## Environment

- CPU: x86_64
- GPU: NVIDIA GeForce RTX 5080
- OS: Linux 6.6.114.1 Microsoft WSL2, glibc 2.39
- NVIDIA driver: 616.56
- CUDA runtime: 13.0
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- Dtype: FP16
- Seed: 1234

## Full official Case 14

Shape: batch 32, sequence 100,000, model width 1,024, 16 heads, 2 layers, causal, FFN width 1,024. The smoke harness processes one batch element at a time and materializes the full `[32, 100000, 1024]` output.

| Variant | Revision | Actual attention path | Elapsed | Peak allocated | Relative to branch base | Result |
|---|---:|---|---:|---:|---:|---|
| Branch base | `fecf994` | exact Flash SDPA | 16.886 s | 13,931.858 MiB | 1.000x | finite output |
| PR 16 as submitted | `b9506f3` | exact Flash SDPA | 14.305 s | 13,931.858 MiB | not a polynomial A/B | finite output |
| PR 16, locally wired and enabled | `8b48621` | fused polynomial | 285.138 s | 16,575.771 MiB | 16.886x slower | finite output |

The local polynomial run regressed latency by 1,588.6% and added 2,643.913 MiB (19.0%) peak allocated memory versus the branch-base exact-Flash control. It reached the 16 GiB GPU memory cliff and paged, so the attention-core kernel speedup did not translate to this end-to-end shape.

The submitted PR does not wire `PolyOrFlashSelfAttention` into the official Case 14 dispatcher or `ExtremeShapeCandidate`; its default full Case 14 run therefore remains exact Flash. Commit `8b48621` is a local-only benchmark patch that changes the dispatcher to instantiate `PolyOrFlashSelfAttention` and sets `POLY_ATTENTION_ENABLED = True`. It was not pushed to the PR.

The exact local-only wiring used for the polynomial row was:

```diff
--- a/src/dispatcher.py
+++ b/src/dispatcher.py
@@
-    FlashOnlySDPASelfAttention,
+    FlashOnlySDPASelfAttention,
+    PolyOrFlashSelfAttention,
@@
-            attention_type = FlashOnlySDPASelfAttention
+            attention_type = PolyOrFlashSelfAttention
--- a/src/implementations/extreme.py
+++ b/src/implementations/extreme.py
@@
-POLY_ATTENTION_ENABLED = False
+POLY_ATTENTION_ENABLED = True
```

Commands and complete machine-readable results:

- [branch-base exact Flash](full-case14-exact-flash.json)
- [submitted PR default](full-case14-pr-default.json)
- [locally enabled polynomial route](full-case14-poly-local-8b48621.json)

## Attention-core microbenchmark

Command:

```text
/home/kevin/techjam/.venv/bin/python -m src.bench_poly --n 100000 --heads 16 --head-dim 64 --reps 10 --output /home/kevin/techjam/research/benchmarks/2026-08-30-rtx5080-b9506f3/attention-core.json
```

| Implementation | Best of 10 |
|---|---:|
| Exact Flash | 200.006 ms |
| PyTorch polynomial | 235.936 ms |
| Fused Triton polynomial | 128.104 ms |

The fused kernel was 1.561x faster than exact Flash and 1.842x faster than the PyTorch polynomial attention core on this GPU. See [attention-core.json](attention-core.json).

## Correctness and tests

Reduced dense-oracle validation:

```text
/home/kevin/techjam/.venv/bin/python -m src.validate_poly --n 8192 --oracle dense --seed 1234
N=8192 oracle=dense wscale=1.0 sigma=0.3335 failures=0/8388608 max=5.8594e-03 rms=3.3545e-04 PASS
```

Full-length exact-Flash-oracle validation:

```text
/home/kevin/techjam/.venv/bin/python -m src.validate_poly --n 100000 --oracle flash --seed 1234
N=100000 oracle=flash wscale=1.0 sigma=0.3339 failures=0/102400000 max=7.8125e-03 rms=3.5189e-04 PASS
```

PR-head test suite:

```text
/home/kevin/techjam/.venv/bin/python -m unittest discover -s src/tests -v
Ran 165 tests in 49.237s
OK
```

`compileall` and `git diff --check` also passed on the clean PR head.

## Review conclusion

Do not merge PR 16 as-is. Although the standalone fused kernel is correct under the supplied tolerances and faster in the attention-core microbenchmark, the optimization is unreachable from the submitted official dispatcher, and when locally connected it causes a severe full-shape latency and memory regression on the RTX 5080 16 GiB environment. The PR is also currently reported by GitHub as conflicting with the target branch; the observed textual conflict is in the benchmark index.
