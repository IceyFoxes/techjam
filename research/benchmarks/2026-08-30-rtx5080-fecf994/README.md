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
