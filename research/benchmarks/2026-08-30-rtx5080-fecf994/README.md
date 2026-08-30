# PR #15 Branch-Base Controls — RTX 5080 — 30 August 2026

These immutable-reference and dispatcher controls were collected from clean
detached commit `fecf9943d7cbf4b5e166862087a4a3bb6e21d868`, the current `master`
parent merged into PR #15. They are the baseline and branch-base reference
points for the Case-3 packed-QKV validation at
[`../2026-08-30-rtx5080-ce3f7f2/`](../2026-08-30-rtx5080-ce3f7f2/README.md).

| Case | Reference | Accuracy | Max abs error | Median latency | Paired speedup |
| ---: | --- | --- | ---: | ---: | ---: |
| 3 | Immutable reference control | PASS 10/10 | 0 | 0.867775 ms | 1.000x |
| 3 | Branch-base dispatcher | PASS 10/10 | 0.000947773 | 0.156653 ms | 5.171x |
| 2 | Immutable reference control | PASS 10/10 | 0 | 0.895467 ms | 1.000x |
| 2 | Branch-base dispatcher | PASS 10/10 | 0.000819206 | 0.122745 ms | 8.443x |

The dispatcher speedups are the paired values reported by their own benchmark
processes. Reference-versus-reference controls were within their measured noise
floors, as expected.

## Records

- [`baseline-case3-10trials.json`](baseline-case3-10trials.json)
- [`base-dispatcher-case3-10trials.json`](base-dispatcher-case3-10trials.json)
- [`baseline-case2-10trials.json`](baseline-case2-10trials.json)
- [`base-dispatcher-case2-10trials.json`](base-dispatcher-case2-10trials.json)

Every JSON records the exact command, timestamp, clean Git revision, official
shape, raw timing samples, correctness trials, and environment metadata.

## Case 6 full official comparison

Official Case 6 (`B=10000`, `N=128`, `d_model=128`, 4 heads, 4 layers,
causal, `ffn_dim=128`) was measured in float32 at the same source revision. The
dispatcher uses the memory-safe streamed route instead of materializing the
full dense-attention working set.

| Run | Accuracy | Max abs error | Baseline median | Compared median | Paired speedup | Noise floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Reference control | PASS 5/5 | 0 | 348.209 ms | 344.748 ms | 1.010x | +/-0.74% |
| Dispatcher | PASS 5/5 | 0.00134718 | 354.862 ms | 149.436 ms | **2.375x** | +/-12.91% |

The dispatcher result has zero failed elements across 819,200,000 checked
elements and clears the run-specific significance rule. The reference-control
run confirms the harness comparison is approximately neutral when both sides
execute the immutable implementation.

| Memory in dispatcher comparison | Peak allocated | Incremental peak allocated |
| --- | ---: | ---: |
| Immutable baseline | 10,672.719 MiB | 10,010.457 MiB |
| Dispatcher | 2,312.512 MiB | 1,650.250 MiB |

The dispatcher reduces peak allocated memory by 8,360.207 MiB in this process.

Environment: x86_64 CPU, NVIDIA GeForce RTX 5080, driver 616.56, CUDA runtime
13.0, PyTorch 2.13.0+cu130, Python 3.12.3, Linux 6.6.114.1 under WSL2. Timing
used paired CUDA events, one warmup, ten samples, five seconds of settling, and
automatic one-forward blocks. Correctness used seeds 1234-1238, input scale
1.0, and zero padding. The recorded Git tree is marked dirty only because
untracked benchmark artifacts already existed; source code was fixed at
`fecf994`.

Case 6 records:

- [`baseline-case6.json`](baseline-case6.json) — reference-versus-reference control.
- [`candidate-case6.json`](candidate-case6.json) — immutable baseline versus dispatcher.
