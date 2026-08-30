# Packed QKV Layout Contract

## Status

- Date: 29 August 2026.
- Leading hypothesis: contiguous `[B,S,3,H,d]` producer storage with
  sequence-major attention output.
- This is a shared Person 2/Person 3 contract, not yet an accepted implementation.

## Baseline Copies

For a contiguous projection output `[B,S,D]`, the benchmark views
`[B,S,H,d]`, transposes to logical `[B,H,S,d]`, then calls `.contiguous()`.
It does this independently for Q, K, and V
(`torch_transformer_benchmark.py:77-95`). After attention it transposes physical
head-major context and copies to `[B,S,H,d]` before output projection (`:113-118`).

This creates four explicit full-`BSD` copies per layer.

## Leading Layout

One packed linear uses:

```text
Wqkv = cat([Wq, Wk, Wv], dim=0)  # [3D,D]
bqkv = cat([bq, bk, bv], dim=0)  # [3D]
linear(x, Wqkv, bqkv)             # [B,S,3D]
view                              # [B,S,3,H,d]
```

For contiguous `[B,S,3,H,d]`, the base stride is
`(3*S*D, 3*D, D, d, 1)`. Selecting Q, K, or V and transposing S/H gives logical
`[B,H,S,d]` with stride `(3*S*D, d, 3*D, 1)`. The last dimension remains
contiguous without a copy.

The preferred attention result is physical contiguous `[B,S,H,d]`, stride
`(S*D,D,d,1)`. It flattens to `[B,S,D]` without a copy before output projection.

## Alternatives

| Physical storage | Producer | Consumer property | Decision |
| --- | --- | --- | --- |
| Three `[B,S,H,d]` tensors | Three linears | Last stride 1; no explicit QKV copies | Required strong control |
| `[B,S,3,H,d]` | Direct packed-linear view | Last stride 1; proven packed API shape | Leading contract |
| `[B,S,H,3,d]` | Requires reordered weight rows | Last stride 1 but no established benefit | Low priority |
| `[B,S,H,d,3]` | Reordered rows | Q/K/V last stride 3 | Reject for current fused APIs |
| `[3,B,H,S,d]` | Requires physical reorder/custom stores | Head-major contiguous Q/K/V | Test only if attention gain repays producer cost |

Packing does not reduce arithmetic. One wide `(T,3D,D)` GEMM can select a
different algorithm and may lose to three `(T,D,D)` GEMMs, so the view-only
three-linear control is essential.

## API Evidence

PyTorch 2.13 fused SDPA checks that the last Q/K/V dimension has stride one;
other dimensions may be non-contiguous subject to backend constraints. Its CUDA
Flash and memory-efficient paths internally operate on sequence-major storage and
return a logical BHSD transpose view, but public SDPA does not guarantee output
strides. Runtime stride and profiler evidence are therefore required.

FlashAttention's `flash_attn_qkvpacked_func` directly accepts `[B,S,3,H,d]` and
returns `[B,S,H,d]`. This validates the layout as a proven interface, not that an
additional dependency wins this benchmark.

Mask compatibility is independent of layout. The benchmark always passes a
valid-token mask. All-valid causal cases can omit redundant key masking, while
padded causal cases require a backend route preserving both key and causal
semantics. Person 2 owns that decision.

## Ownership Contract

Person 3 owns packed weight lifecycle, one-packed versus three-view-only
projection comparison, producer strides, projection correctness, and flattening
the returned context into output projection.

Person 2 owns attention backend selection, causal and padding semantics, hidden
consumer copies, and physical output layout.

Both owners must measure:

```text
LayerNorm -> QKV projection -> attention -> context merge -> output projection
```

Every adapter is charged to the backend that requires it. No unconditional
`.contiguous()` is permitted at the boundary. Record shape, stride,
`storage_offset`, selected backend, and copy kernels.

Person 4 may override full-buffer layouts for chunked #6/#14 execution.

## Experiments And Kill Criteria

- Compare baseline three-copy, three view-only, packed sequence-major, and a
  forced-copy negative control for representative #2, #7, #8, and #6 shapes.
- Force each available SDPA backend and record rejection reasons, actual output
  strides, hidden clones, complete-path latency, and all three dtypes.
- Test all-valid and padded masks separately.
- Reject packed QKV for a shape if it does not beat three view-only linears in
  complete projection-to-output-projection latency.
- Reject a layout/backend pair that adds hidden copies or regresses complete-path
  latency, regardless of projection-only gains.
- Require zero executable correctness failures over multiple seeds, scales, and
  padding ratios.

## Sources

All public sources were accessed on 29 August 2026.

- PyTorch, [SDPA documentation](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html),
  tag `v2.13.0`: logical shapes, masks, backend selection, and output shape.
- PyTorch repository, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`,
  [`attention.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/attention.cpp):
  `scaled_dot_product_attention`, `qkv_projection`, and `transform_0213`.
- PyTorch,
  [`sdp_utils.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp):
  backend checks and alignment constraints.
- PyTorch,
  [`sdp_utils_cpp.h`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/sdp_utils_cpp.h):
  `check_last_dim_stride_equals_1_dense`.
- PyTorch,
  [`attention.cu`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/transformers/cuda/attention.cu):
  `native_multi_head_attention_cuda` and fused SDPA wrappers.
- FlashAttention, revision `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`,
  [`flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/flash_attn_interface.py):
  `flash_attn_qkvpacked_func` and sequence-major output.
- FlashAttention,
  [`modules/mha.py`](https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/modules/mha.py):
  proven packed projection-to-attention-to-output composition.

## Person 2 note on the key mask (30 August 2026)

`PackedQKVSDPASelfAttention` inherits the always-on broadcast key mask from
`StridedSDPASelfAttention`. Under causal attention that mask is bitwise dead
code and dropping it measured faster on every in-scope case, so whichever route
is chosen for the strided module applies to the packed one unchanged — the
layout decision and the mask decision are independent.

Evidence and the proposed one-line change:
[`../attention-softmax/safe-optimization-spec.md`](../attention-softmax/safe-optimization-spec.md)
section 3 and
[`../framework-fastpaths/dispatcher-strategy.md`](../framework-fastpaths/dispatcher-strategy.md).

Outstanding for the boundary contract: the SDPA output stride, which decides
whether the `context.transpose(1, 2).reshape(...)` on the way out still copies.
Not yet recorded; it needs a run with stride logging rather than an inference
from the timings.
