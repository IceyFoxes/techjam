# Recommended Four-Way Team Split

## Planning Status

### Current status (29 August 2026)

The numbered shape values and characteristics below match the organizer Appendix
screenshot supplied as [`task_shapes.png`](../../task_shapes.png) and transcribed
in [`TASK.md`](../../TASK.md#37-appendix-test-shapes). The performance targets and
work allocation remain internal team planning decisions, not organizer
requirements.

### Previous status (superseded 29 August 2026)

> The numbered shapes, shape characteristics, dimensions, and performance targets
> below are internal planning assumptions supplied by the repository owner. The
> official test-shape table is not available in this repository; do not treat
> these assumptions as organizer-confirmed requirements.

This status was superseded because the supplied Appendix confirms the shape
values. It is retained here under the repository's non-destructive research
policy.

## Person 1: Framework Fast Paths, Compilation, and Integration

**Ownership:** The strongest low-risk implementation, compilation strategies, and final dispatcher.

**Primary assumed shapes:** #2-#4 small-batch cases, #1-#5 ordinary baseline cases, and eventually all 14 cases through the dispatcher.

**Responsibilities:**

- Reproduce all feasible reference results.
- Replace manual attention with `scaled_dot_product_attention` while preserving causal and padding semantics.
- Compare available SDPA backends by shape with `sdpa_kernel` rather than assuming one universal winner.
- Test eager execution and `torch.compile`, including `reduce-overhead` for small launch-bound cases and `max-autotune` when compilation time is excluded.
- Test CUDA graphs only when inputs and allocations are static.
- Inspect compiler logs for graph breaks.
- Integrate accepted implementations from Persons 2-4 and centralize routing in the final dispatcher.

**Internal targets:**

- Correctness on all ordinary shapes before performance work is accepted.
- At least `1.2x` geometric-mean speedup over the eager reference.
- Stretch target of at least `1.5x` on #2 and #3 when profiling confirms launch overhead.
- Reject a regression greater than roughly 3% unless another shape gains substantially more.

**Deliverables:**

- Reproducible benchmark runner.
- Baseline CSV.
- Final optimized Transformer dispatcher.
- Environment and experiment records.
- Integrated winning backends from the other streams.

**Module ownership:**

- `src/implementations/compiler.py`
- `src/dispatcher.py`

## Person 2: Attention and Softmax Kernels

**Ownership:** `QK^T`, causal softmax, and multiplication by `V`.

**Primary assumed shapes:** #7-#11 head-dimension variants, #12-#14 sequence-length cases, and especially #13 with assumed `N=1024`.

**Responsibilities:**

- Isolate attention from projections and FFN for microbenchmarking.
- Normalize tensors to a contiguous `[B, H, N, head_dim]` layout.
- Test fused PyTorch SDPA before retaining a custom kernel.
- Profile assumed head dimensions `{8, 32, 64, 128, 256}`.
- If SDPA leaves a meaningful gap, adapt a proven fused Triton attention implementation and tune block sizes and warp counts by head-dimension family.
- Apply causal masking inside the kernel without constructing a full mask tensor.
- Never materialize the full `N x N` attention-score matrix for long sequences.

**Internal targets:**

- No `N x N` score allocation for long-sequence cases.
- At least `1.3x` attention-only speedup over reference attention.
- At least 10% whole-layer improvement before accepting a custom kernel.
- Retain SDPA when a custom kernel is not faster.
- Record achieved TFLOP/s and peak VRAM for #13.

**Deliverables:**

- Attention backend keyed by `(head_dim, sequence_length, causal, dtype)`.
- Attention microbenchmark and whole-layer result tables.
- Rationale for each Flash, SDPA, Triton, or fallback route.

**Module ownership:**

- `src/implementations/attention.py`

## Person 3: QKV Projections, FFN, and Elementwise Fusion

**Ownership:** Q/K/V and output projections, FFN, normalization, layouts, and targeted elementwise fusion.

**Primary assumed shapes:** #8 with `d_model=1024, N=128`, #6 as a huge-batch case, #14 with `d_model=1024` if feasible, and #7 with `d_model=32` as a fusion check.

**Responsibilities:**

- Profile Q/K/V projections, output projection, FFN, and normalization separately.
- Combine QKV weights into one GEMM when legal and keep setup or prepacking outside timed execution when the official rules allow it.
- Remove unnecessary `permute(...).contiguous()` copies by retaining a stable layout.
- Keep matmuls in cuBLAS or PyTorch initially.
- Explore bias-plus-activation, residual-plus-LayerNorm, and projection-plus-bias-plus-residual fusion.
- Compare compiler-generated fusion with targeted Triton kernels.
- Test FP16, BF16, and permitted TF32 behavior independently against the executable correctness checker.

**Internal targets:**

- A 10-15% whole-layer improvement on #8.
- Reduce QKV from three GEMMs to one where legal.
- Retain an elementwise fusion only when it beats the unfused sequence by at least 15% in isolation.
- Pass the executable absolute-error OR relative-error criterion.

**Deliverables:**

- Optimized projection and FFN implementation.
- Stable-layout explanation.
- Per-operation profiler comparison before and after fusion.

**Module ownership:**

- `src/implementations/projections.py`

## Person 4: Extreme-Shape Memory Strategy

**Ownership:** Memory-safe execution for assumed extreme shapes #6 and #14.

**Primary assumed shapes:** #6 and #14, investigated immediately.

**Responsibilities:**

- Run each extreme case in an isolated process and classify failures as input-allocation OOM, QKV-allocation OOM, attention-score OOM, or timeout.
- Ask the organizer whether #14 is intentional and whether every listed shape is scored.
- For #14, investigate exact streaming through batch or head chunking, current-chunk QKV generation where possible, online or Flash-style causal softmax, and workspace reuse.
- Avoid retaining all attention-head outputs simultaneously; evaluate accumulation through corresponding output-projection weight slices.
- Sweep chunk sizes and select the largest safe option rather than hardcoding the smallest chunk.
- If #14 is officially excluded, assist Person 2 with attention tuning.

**Internal memory rationale:**

- Assumed #14 has `B=32, H=16, N=100000` and `5.12e12` attention scores, requiring roughly 10.2 TB in FP16 if materialized.
- Its assumed input has 3.28 billion values, roughly 6.1 GiB in FP16, while separately storing full Q, K, and V requires roughly 18.3 GiB.
- Exact causal attention still has an extreme compute requirement even when score materialization is removed.
- Assumed #6 should avoid materializing its roughly 1.3 GB FP16 attention-score tensor.

**Internal targets, in order:**

1. Pass without OOM and meet the executable numerical tolerances.
2. Keep peak allocated memory below roughly 85-90% of available VRAM.
3. Minimize latency by choosing the largest safe chunk.

**Deliverables:**

- Tensor-size memory model.
- Chunk-size versus latency and VRAM table.
- Extreme-shape implementation or a documented blocker from the official harness.

**Module ownership:**

- `src/implementations/extreme.py`

## Dispatcher Integration

Person 1 owns `src/dispatcher.py` and integrates accepted backends from Persons 2-4.

| Internal shape family | Assumed shapes | Preferred strategy |
| --- | --- | --- |
| Small or launch-bound | #2, #3, possibly #12 | Compilation, fusion, or CUDA graphs |
| Ordinary dense | #1, #4, #5 | Library fast paths |
| Huge batch | #6 | Memory-efficient or fused attention; chunking if needed |
| Head-dimension variants | #7-#11 | Attention backend and tensor layout |
| Long sequence | #13 | Flash or online attention |
| Extreme memory or compute | #14 | Streaming and explicit memory planning |

The dispatcher must route on the complete known tuple:

```text
(batch, qkv_dim, heads, sequence_length, layers, causal, ffn_dim, dtype)
```

Keep all shape checks centralized in `src/dispatcher.py` rather than scattering them through kernels.
