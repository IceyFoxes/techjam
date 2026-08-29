# Packed QKV Cross-Case Exploration

Status: exploratory implementation-selection evidence from 29 August 2026. The
timing ratios below are heuristic because the shared harness's confidence
calculation is not statistically valid. They must not be presented as accepted
speedups or generalized beyond this environment.

## Question

Can one prepacked QKV projection replace three independent projections inside
the current float32 SDPA plus strided-view route without adding runtime copies,
changing persistent state-dict names, or violating the executable correctness
criterion?

The experiment concatenates projection rows outside `forward`:

```python
packed_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
packed_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
qkv = F.linear(x, packed_weight, packed_bias)
q, k, v = qkv.view(B, S, 3, H, head_dim).unbind(dim=2)
q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
```

The original Q/K/V parameters remain authoritative. The packed tensors are
detached, non-persistent buffers rebuilt after strict weight loading and device
or dtype conversion. Concatenation is setup work and never appears in the timed
forward.

## Environment

- Repository revision: `0d49340d687b82e75974fff97a4c67d787dad182`.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9,
  6,083,706,880 bytes reported memory, driver 610.57.04, maximum reported SM
  clock 3,105 MHz.
- CPU: Intel Core Ultra 7 155H, 16 cores / 22 logical CPUs.
- OS: Linux 7.2.2-1-cachyos, x86_64, glibc 2.44.
- PyTorch / CUDA runtime / Triton: 2.13.0+cu130 / 13.0 / 3.7.1.
- Disk: 320,841,187,328 bytes total and 183,115,194,368 bytes available at
  collection time.
- Numerical configuration: float32, `matmul_precision=high`, TF32 enabled,
  inference mode.

## Method

An exploratory script under `/tmp/opencode/person3_packed_qkv_probe.py` built
three models with strictly copied weights:

1. the immutable explicit baseline;
2. the current three-projection SDPA plus strided-view control; and
3. the packed-QKV candidate with the same SDPA, output projection, FFN, masks,
   and final normalization.

The full correctness matrix used seeds 1234-1243, input scales 0.125, 1, and 8,
and padding ratios 0 and 0.25. Screens used three seeds, while Case #13 used one
seed because it was immediately neutral. Every trial compared both optimized
routes with the immutable baseline and compared packing directly with the
current SDPA control.

Eager profiles used `torch.profiler` with CUDA activities, shapes, and memory.
Compiled probes used `torch.compile(mode="reduce-overhead")`. Raw timings used
alternating paired CUDA events after continuous settling, but no confidence
interval or significance label is reported. Compilation, packing, and first-call
capture were outside intended steady-state timing; the first few raw samples
still contained visible re-recording outliers and are retained only in temporary
JSON.

Representative commands were:

```bash
PYTHONPATH=. .venv/bin/python /tmp/opencode/person3_packed_qkv_probe.py \
  --case 8 --seeds 10 --repeats 40 --settle-seconds 3 \
  --output /tmp/opencode/person3-case8-eager-full.json

PYTHONPATH=. \
TORCHINDUCTOR_CACHE_DIR=/tmp/opencode/inductor-case2-packed-full \
TRITON_CACHE_DIR=/tmp/opencode/triton-case2-packed-full \
.venv/bin/python /tmp/opencode/person3_packed_qkv_probe.py \
  --case 2 --seeds 10 --repeats 120 --settle-seconds 5 --compile \
  --output /tmp/opencode/person3-case2-compiled-full.json
```

The temporary script and JSON were not promoted to benchmark records. A tracked
candidate and the corrected timing harness are required before preservation.

## Layout and Backend

For Case #8, packed output `[64,128,3072]` has stride
`(393216,3072,1)`. The unbound BHSD Q/K/V views have stride
`(393216,256,3072,1)` and storage offsets 0, 1024, and 2048. All views retain
last-dimension stride 1.

The installed backend selector returned ID 2, `EFFICIENT_ATTENTION`, for the
packed views and broadcast validity mask. Eager profiling showed no Q/K/V or
post-SDPA layout copies. Output masking still produces the same nine clone/copy
pairs as the control, and those are outside the packed projection change.

For Case #2, profiling reduced full-model linear launches from 24 to 16: twelve
Q/K/V launches became four packed launches, while output and FFN projections
were unchanged. The four attention calls retained the same memory-efficient
kernel. Both variants reported a 14,686,208-byte incremental peak in this probe.

## Cross-Case Screen

`Ratio` is the median paired control latency divided by packed latency. It is a
directional statistic, not a valid confidence statement.

| Case | Mode | Trials | Packed failures vs reference | Control ms | Packed ms | Ratio | Decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | eager screen | 18 | 4 | 3.857 | 3.750 | 1.029x | reject: small gain; failures inherited |
| 2 | eager full | 60 | 0 | 1.097 | 0.995 | 1.101x | continue |
| 2 | compiled full | 60 | 0 | 0.281 | 0.236 | 1.191x | **implementation candidate** |
| 3 | eager full | 60 | 0 | 1.157 | 1.049 | 1.098x | compile gate required |
| 3 | compiled screen | 18 | 0 | 0.292 | 0.280 | 1.042x | reject: gain collapses |
| 4 | eager screen | 18 | 0 | 1.094 | 1.031 | 1.057x | compile gate required |
| 4 | compiled screen | 18 | 0 | 0.773 | 0.759 | 1.019x | reject: gain collapses |
| 5 | eager screen | 18 | 4 | 7.833 | 7.960 | 0.983x | reject: regression |
| 7 | eager full | 60 | 8 | 2.262 | 2.163 | 1.045x | reject: packing adds failures |
| 8 | eager full | 60 | 20 | 77.553 | 77.237 | 1.003x | reject: neutral |
| 9 | eager full | 60 | 4 | 3.522 | 3.411 | 1.033x | reject: small gain; failures inherited |
| 10 | eager screen | 18 | 4 | 3.421 | 3.310 | 1.034x | reject: small gain |
| 11 | eager screen | 18 | 2 | 7.109 | 7.007 | 1.014x | reject: neutral |
| 12 | eager full | 60 | 10 | 1.114 | 1.026 | 1.083x | compile gate required |
| 12 | compiled screen | 18 | 2 | 0.716 | 0.702 | 1.019x | reject: gain collapses |
| 13 | eager screen | 6 | 2 | 85.529 | 85.198 | 1.004x | reject: attention dominates |

Cases #6 and #14 were not executed at official scale. Case #6 requires Person
4's batch/memory schedule. Case #14's input alone exceeds this GPU's memory and
full packed QKV would require approximately 18.31 GiB in FP16; a memory-safe
route should instead consider sample/query chunking and packed K/V only.

## Numerical Findings

Case #2 passed all 60 eager and all 60 compiled stress trials with zero failed
elements. The compiled candidate's worst maximum absolute error was
0.0018246174. Packing versus the compiled control also had zero failures.

Case #7 is the only tested family where packing itself crossed the executable
tolerance: two low-scale trials failed packed-versus-control with one element
each. Its packed-versus-reference result failed eight trials, so the route is
rejected regardless of its 1.045x heuristic ratio.

For Cases #8, #9, and #12, packing introduced no executable failures relative to
the SDPA control. Their low-scale failures were inherited from that control.
Case #8 packed and control outputs were bitwise identical over the matrix. This
confirms that packed QKV cannot repair the dispatcher's broader low-scale SDPA
correctness gap.

## Repeated Case #2 Result

Two fresh compiled processes produced paired-ratio medians of 1.191x and 1.189x:

| Run | Control median | Packed median | Raw pairs | Correctness |
| --- | ---: | ---: | ---: | --- |
| full | 0.281440 ms | 0.235552 ms | 120 | PASS 60/60 compiled |
| repeat | 0.276544 ms | 0.232416 ms | 120 | PASS 18/18 compiled |

The direction and magnitude are consistent, and profiler launch removal explains
the latency delta. Nevertheless, serial correlation and the unresolved estimator
prevent a statistical significance claim. The next step is a tracked Case-2-only
candidate followed by the corrected benchmark protocol and exact RTX 5080
validation.

## Rejected FFN Direction

A shape-specialized Case #7 Triton prototype fused LayerNorm, two D32
projections, exact `erf` GELU, residual, and masking. Its TF32x3 variant reduced
isolated median latency from 66.15 microseconds to 23.09 microseconds on this GPU,
but whole-model compiled latency improved only from 1.512 ms to 1.489 ms (1.016x)
and still had one low-scale failure. Plain TF32 was faster but failed 182 elements
at default scale; IEEE regressed whole-model latency. The custom FFN route is
therefore rejected despite clearing the isolated 15% threshold.

## Decision

- Implement packed QKV only for Case #2 as an experimental candidate.
- Preserve original Q/K/V parameter names and strict weight loading; use
  non-persistent derived buffers rebuilt after load and conversion.
- Keep QKV views strided and require profiler evidence of no hidden copies.
- Do not promote Cases #3, #4, or #12 based on eager gains; compilation removes
  almost all benefit.
- Reject universal packing, Case #7 custom FFN, and Case #8 packed QKV.
- Defer Cases #6 and #14 to a joint Person 3/Person 4 chunking design.

## Sources

- PyTorch `Linear.cpp` at installed revision `cf30153c4c131c8164ee7798e5022d810682e2cb`,
  symbols `linear` and `_flatten_nd_linear`:
  <https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/Linear.cpp>.
  Accessed 29 August 2026. It establishes `x @ weight.T + bias`, flattening of
  contiguous higher-rank inputs, and the row-concatenation packing order.
- PyTorch CUDA SDPA selection at the same revision, symbols
  `select_sdp_backend`, `can_use_flash_attention`, and
  `can_use_mem_efficient_attention`:
  <https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp>.
  Accessed 29 August 2026. It explains why FP32 Case #8 selects memory-efficient
  attention and why last-dimension stride 1 is sufficient for these packed views.
- PyTorch 2.13 `F.linear` documentation:
  <https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.linear.html>.
  Accessed 29 August 2026. It documents weight/output dimensions and TF32
  support.
- PyTorch 2.13 numerical-accuracy note:
  <https://docs.pytorch.org/docs/2.13/notes/numerical_accuracy.html>.
  Accessed 29 August 2026. It documents floating-point non-associativity and why
  mathematically equivalent packed and separate GEMMs still require executable
  correctness validation.
- PyTorch 2.13 `torch.compile` documentation:
  <https://docs.pytorch.org/docs/2.13/generated/torch.compile.html>.
  Accessed 29 August 2026. It documents compilation modes, guards, and CUDA Graph
  behavior relevant to the compiled probes.
