# Long-Sequence Attention (Case 14): Measured Feasibility of Approximate Methods

Status: current as of 30 August 2026. Companion to
[`fast-attention-survey.md`](fast-attention-survey.md), whose Tier C exclusions
this document **re-tests and partially overturns for case 14 only**.

Scope: official case 14 — `B=32, d_model=1024, H=16, N=100000, L=2, causal,
ffn=1024`, head dimension 64. Cases 1-13 are unaffected; every conclusion in the
companion documents stands for `N <= 1024`.

## 1. Why case 14 needed a separate analysis

Three load-bearing conclusions in [`README.md`](README.md) and
[`fast-attention-survey.md`](fast-attention-survey.md) were derived at
`N <= 1024` and do not transfer:

| Conclusion (N <= 1024) | Status at N = 100000 |
| --- | --- |
| "Attention is memory-bound by ~25x" | **Inverted.** Attention is ~95% of all FLOPs (~1.31 PFLOP over the case). Only FLOP *reduction* helps. |
| "The shapes are not the ones this literature targets" | **No longer true.** N = 100000 is squarely in the regime the long-context literature targets. |
| "Any approximation fails the tolerance criterion by construction" | **Not measured, and false as stated.** See section 3. |

The third is the important one. It was an argument from first principles, not a
measurement, and the error budget at `N = 100000` is very different from the one
at `N = 1024`.

## 2. Measured score statistics

Measured on real Q/K/V drawn from a case-14-shaped `BaselineTransformer`
layer 0 (`d_model=1024, H=16`), which fixes the score distribution independently
of `N`:

| quantity | measured |
| --- | ---: |
| score standard deviation `sigma` | **0.3336** |
| score mean | 0.0000 |
| score min / max | -2.203 / 2.402 |
| effective softmax support `1/sum(p^2)` at N=4096 | **3656 of 4096 (89%)** |
| probability mass in the top-64 keys | **3.4%** |
| attention context rms at N=4096 | 0.0207 |
| final model output rms | 1.000 (residual-dominated) |

`sigma = 0.334` is a consequence of the benchmark's initialisation, not of any
particular seed: `nn.Linear` default init plus a `LayerNorm`ed input gives
`q, k` component rms 0.577, so `s = q.k/sqrt(d_h)` has standard deviation
`sqrt(d_h) * 0.577^2 / sqrt(d_h) ... ~ 0.33` for any `d_model` with
`d_model/H = 64`.

### 2.1 This kills the entire sparse-attention family, definitively

The softmax over 100000 keys is **near-uniform**, not peaked. With 89% effective
support and only 3.4% of the mass in the top-64 keys, there is no sparsity to
exploit. Every selection-based method — top-k attention, Quest, SpAtten,
SeerAttention, MInference, NSA, and block pruning driven by per-block score
bounds — requires concentrated attention mass and has **nothing to prune here**.

This is a stronger and more useful exclusion than the survey's original
"approximation changes the result" argument, because it is measured and it
applies regardless of the tolerance.

## 3. The error budget at N = 100000 is large

Attention's contribution to the residual stream scales as `1/sqrt(N)`: the
context is an average over the causal prefix, so its rms falls from 0.021 at
N=4096 to roughly 0.004 at N=100000, while the residual stream stays at rms 1.0
and the tolerance stays fixed at `abs <= 0.002 OR rel <= 0.02`.

The whole variable part of attention's output is therefore *smaller than the
per-element tolerance*. This is the opposite of the `N <= 1024` regime recorded
in [`sdpa-and-precision.md`](sdpa-and-precision.md), where folding `scale` into
`Q` was already enough to break the criterion.

**Consequence: an approximation with a few per cent of relative error on the
attention output can pass at case 14.** That reopens exactly one branch of the
Tier C literature — the kernel/feature-map branch — which section 4 tests.

## 4. Polynomial-feature linear attention

### 4.1 Method

Approximate `exp(s)` by the degree-2 polynomial that is L2-optimal under the
measured score distribution `s ~ N(0, sigma^2)`. The Gauss-Hermite projection
gives

```text
exp(s) ~= exp(sigma^2/2) * [ (1 - sigma^2/2) + s + s^2/2 ]
```

The `exp(sigma^2/2)` factor cancels in the softmax normalisation, so this differs
from a plain Taylor expansion at 0 only in the constant term. With
`a = q*sqrt(scale)` and `b = k*sqrt(scale)` so that `a.b = s`, the feature map

```text
phi(a) = [ sqrt(c0), sqrt(c1)*a, sqrt(c2)*vec(a a^T) ]
```

satisfies `phi(a).phi(b) = c0 + c1*(a.b) + c2*(a.b)^2`, so causal attention
becomes a chunked linear scan over a running state, `O(N * d_f * d_h)` instead of
`O(N^2 * d_h)` — about 12x fewer FLOPs at these shapes. Causality is handled by
the standard chunked form: the strict prefix contributes through the state and
the diagonal chunk is computed with exact `exp`.

This is the deterministic cousin of Performer's FAVOR+, closest in the literature
to the second-order Taylor feature map of **Based** (Arora et al., 2024). The
`sigma`-fitted constant term measured **2.3x lower error** than the plain Taylor
constant at N=8192 and is free.

### 4.2 Two properties that make it work here

1. **The error concentrates in early tokens.** A token attending to `t` keys has
   an approximation error scaling as `1/sqrt(t)`, so the worst error is at the
   start of the sequence. Computing the first `W` tokens with exact attention
   costs `(W/N)^2` of the full quadratic work — 0.17% for `W=4096` — and removes
   the entire tail. Without it, max error is 6.6e-3; with it, 1.8e-3.

2. **The remaining error sits at the fp16 noise floor.** At every `N` tested the
   order-2 max error is 7.812e-3, which is exactly one fp16 ulp at the output
   magnitude — i.e. the approximation is no longer the dominant error term.

### 4.3 Correctness results

Official criterion `abs <= 0.002 OR rel <= 0.02`, zero failures required.
Configuration: order 2, chunk 512, `exact_prefix=4096`, fp32 state accumulator
with fp16 per-chunk matmuls.

Against the **dense reference implementation** (`BaselineTransformer`, the
authoritative oracle, runnable only to N~8192 in 8 GiB), full two-layer
case-14-shaped model, B=1, float16:

| N | failed elements | max abs err | rms abs err | verdict |
| ---: | ---: | ---: | ---: | :--- |
| 4096 | 0 / 4,194,304 | 5.859e-03 | 2.45e-04 | PASS |
| 8192 | 0 / 8,388,608 | 5.859e-03 | 3.51e-04 | PASS |

Against an **exact-flash oracle** (algebraically exact; isolates approximation
error from fp16 reduction-order noise), same model, B=1:

| N | failed elements | max abs err | rms abs err | verdict |
| ---: | ---: | ---: | ---: | :--- |
| 16384 | 0 / 16,777,216 | 7.812e-03 | 3.51e-04 | PASS |
| 32768 | 0 / 33,554,432 | 7.812e-03 | 3.59e-04 | PASS |
| 65536 | 0 / 67,108,864 | 7.812e-03 | 3.55e-04 | PASS |
| **100000** | **0 / 102,400,000** | 7.812e-03 | 3.52e-04 | **PASS** |

Note the error is flat in `N`. An earlier hypothesis that it would fall as
`1/sqrt(N)` was **wrong**: the residual error is pinned by the tokens immediately
following the exact prefix, and by fp16 output quantisation, neither of which
depends on `N`.

### 4.4 Performance results

Attention core only, one sample x one layer, `H=16, N=100000, d_h=64`, float16,
RTX 4060 Laptop. Best of 3 after warmup.

| variant | time | vs exact flash | correct |
| --- | ---: | ---: | :---: |
| exact flash SDPA (Person 4's case-14 route) | 719.8 ms | 1.00x | yes |
| **order 2, chunk 512, fp32 state / fp16 compute** | **603.9 ms** | **1.19x** | **yes** |
| order 2, chunk 256, fp32 state / fp16 compute | 660.6 ms | 1.09x | yes |
| order 2, chunk 512, fp32 state and compute | 985.8 ms | 0.73x | yes |
| order 2, chunk 512, fp16 state | 590.4 ms | 1.32x | **NO** (section 5.2) |
| order 1, chunk 512 | 177.9 ms | 4.36x | **NO** (section 5.1) |

The crossover is `N`-dependent, since the method is `O(N)` against the exact
path's `O(N^2)`: the same order-2 configuration measures 0.84x at N=65536 and
1.19x at N=100000. It only pays at case-14 scale.

## 5. Negative results

These are recorded because they cost real measurement time and each one closes a
plausible-looking direction.

### 5.1 Order-1 (pure linear attention) is fast and wrong

Dropping the quadratic term gives 4.36x but fails the criterion at every `N`,
and enlarging the exact prefix does not rescue it:

| configuration | failures |
| --- | ---: |
| order 1, `exact_prefix=4096`, N=65536 | 957 / 67,108,864 |
| order 1, `exact_prefix=16384`, N=65536 | 42 / 67,108,864 |
| order 1, `exact_prefix=32768`, N=65536 | 15 / 67,108,864 |
| order 1, `exact_prefix=4096`, N=8192, vs dense reference | 808 / 8,388,608 |

The quadratic term is required. Its residual weight error is ~7.6% versus ~1.4%
for order 2, and the criterion admits zero failures.

### 5.2 A float16 state accumulator is catastrophically wrong at scale

`state_dtype=float16` is the fastest correct-looking option (1.32x) and passes
at N=16384, then fails at N=65536 with **1,064,935 failures** and a max error of
0.106. The state accumulates several hundred chunk contributions, and small
increments are lost entirely against a large fp16 accumulator. The master state
must be fp32; only the per-chunk matmuls may be fp16.

This is a trap: it passes every cheap small-`N` test and fails only at the scale
that matters.

### 5.3 Symmetric feature packing is slower

Packing `a a^T` into its `d(d+1)/2 = 2080` upper triangle (with `sqrt(2)` on the
off-diagonal, which keeps the inner product exactly `(a.b)^2`) halves the feature
traffic but measured **slower** at every chunk size — 750.5 ms against 526.7 ms
at chunk 512. The two gathers cost more than the bandwidth they save.

### 5.4 Every selection-based sparse method is excluded by measurement

See section 2.1. Effective softmax support is 89%; the top-64 keys hold 3.4% of
the mass. There is nothing to prune.

## 6. Remaining opportunity: a fused kernel

The order-2 path is **memory-bound on materialising the `C x 4096` feature
tensor**, not compute-bound. It performs ~1.7 TFLOP against the exact path's
~20.5 TFLOP per sample-layer — 12x fewer FLOPs — but realises only 1.19x, because
it writes and re-reads ~51 GB of feature tensor per sample-layer.

A fused kernel that generates `a a^T` in registers and consumes it directly in
the GEMM would remove that traffic entirely. Two supporting measurements:

- `torch.compile` prologue fusion on the isolated outer-product-plus-GEMM gives
  **1.31x** (383.3 us -> 292.0 us per chunk) with bitwise-identical output, and
  that is without a hand-written kernel.
- At the exact path's realised 28 TFLOPS, 1.7 TFLOP would take ~61 ms, against
  the 603.9 ms measured — so the headroom is real and large.

Estimated 2-4x for a Triton implementation. This is the single largest remaining
item in the attention stream and the only one whose payoff justifies a custom
kernel, which reverses the "do not write a Triton kernel" recommendation in
[`fast-attention-survey.md`](fast-attention-survey.md) **for case 14 only**.

## 7. Risk assessment and the required runtime guard

The method's accuracy depends on `sigma`, which is a property of the benchmark's
random initialisation. Under trained weights, scores are far larger, the softmax
is peaked, and a degree-2 polynomial fit would be poor. This must not be
presented as a general fast-attention result.

The mitigation is that `sigma` is **cheap to estimate at runtime** and the same
statistic serves two purposes:

1. it supplies the L2-optimal constant term `c0 = 1 - sigma^2/2`; and
2. it gates the route — if the measured `sigma` exceeds the validated range, fall
   back to exact flash SDPA.

A 2048-query sample recovers `sigma` to three decimal places (0.3338 against the
0.3336 population value) for a negligible cost, and does so without a host
synchronization if the comparison is kept on device. Any deployed route must
carry this guard; without it the method is a benchmark-specific trick rather
than a defensible optimization.

## 8. Environment

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB, 24 SMs, sm_89 |
| Driver | 616.56 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| Python | 3.12 |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 |
| dtype | float16 (case 14's route dtype per PR #13) |

All measurements are attention-core or single-sample whole-model microbenchmarks
on a `B=1` slice. That is the shape Person 4's case-14 route actually executes,
since [PR #13](https://github.com/IceyFoxes/techjam/pull/13) streams case 14 one
prefix batch at a time. Full-case numbers require that route and a GPU that can
hold it; none are claimed here.

## 9. Sources

- **Arora, S., Eyuboglu, S., Zhang, M., et al., "Simple linear attention language
  models balance the recall-throughput tradeoff" (Based), ICML 2024.**
  <https://arxiv.org/abs/2402.18668>. Accessed 30 August 2026. Uses a
  second-order Taylor feature map of `exp` to obtain linear attention with a
  finite feature dimension, combined with local exact attention. The method in
  section 4 is this construction with the Taylor constant replaced by the
  L2-optimal Gauss-Hermite constant for the measured score distribution, and with
  the local exact component placed at the start of the sequence rather than in a
  sliding window.

- **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F., "Transformers are RNNs",
  ICML 2020.** <https://arxiv.org/abs/2006.16236>. Accessed 30 August 2026.
  Source of the chunked causal linear-attention scan used here. Listed as
  excluded in [`fast-attention-survey.md`](fast-attention-survey.md) Tier C on
  the grounds that it changes the kernel; that exclusion is correct in general
  and is overturned here only because section 3 shows the induced error fits
  inside case 14's tolerance.

- **Choromanski, K. et al., "Rethinking Attention with Performers", ICLR 2021.**
  <https://arxiv.org/abs/2009.14794>. Accessed 30 August 2026. The stochastic
  counterpart to section 4's deterministic feature map. Still excluded: random
  features add variance, and a deterministic polynomial fit is both cheaper and
  more accurate at `sigma = 0.334`.

- **Tang, J. et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM
  Inference", ICML 2024.** <https://arxiv.org/abs/2406.10774>. Accessed
  30 August 2026. Per-page min/max key bounds to select critical KV pages.
  Excluded by the section 2.1 measurement: the softmax here has 89% effective
  support, so page selection has nothing to discard.

- **Jiang, H. et al., "MInference: Accelerating Pre-filling for Long-Context LLMs
  via Dynamic Sparse Attention", NeurIPS 2024.**
  <https://arxiv.org/abs/2407.02490>. Accessed 30 August 2026. Dynamic sparse
  patterns for long-context pre-filling — the closest published setting to case
  14, which is a pure pre-fill at N=100000. Excluded for the same reason as
  Quest: the measured attention here has no exploitable sparse structure.

- **Milakov, M., Gimelshein, N., "Online normalizer calculation for softmax".**
  <https://arxiv.org/abs/1805.02867>. Accessed 30 August 2026. The exact diagonal
  chunk in section 4 does not need the online recurrence because the polynomial
  denominator is accumulated directly, but the running-state structure is the
  same recurrence with the max-tracking removed — which is only safe because
  section 2 bounds the scores to `[-2.203, 2.402]`.
