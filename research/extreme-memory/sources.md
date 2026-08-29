# Source Catalog

All public sources were accessed on 28 August 2026.

## Algorithms And Implementations

### FlashAttention

- URL: https://arxiv.org/html/2205.14135
- Version: arXiv:2205.14135v2, 23 June 2022.
- Relevant details: Algorithm 1 gives tiled online exact attention; Theorem 1
  establishes `O(N^2 d)` FLOPs and `O(N)` additional memory beyond inputs and
  output; Theorem 2 analyzes reduced HBM traffic. It supports the conclusion
  that score materialization can be removed without removing quadratic compute.

### FlashAttention-2

- URL: https://arxiv.org/html/2307.08691
- Version: arXiv:2307.08691v1, 17 July 2023.
- Relevant details: Algorithm 1 maintains unnormalized output, maximum, and
  normalizer by query-row tile; Section 3.2 parallelizes over query rows;
  Section 3.3 reports typical `{64,128} x {64,128}` tiles; causal execution
  skips future blocks and reports about 1.7-1.8x rather than a perfect 2x gain.

### Self-attention Does Not Need O(n^2) Memory

- URL: https://arxiv.org/html/2112.05682
- Version: arXiv:2112.05682v3, 10 October 2022.
- Relevant details: derives stable incremental attention, proves that exact
  attention can use subquadratic auxiliary memory while retaining quadratic
  time, and demonstrates practical query/key chunking and the chunk-size
  performance tradeoff.

### Online Normalizer Calculation For Softmax

- URL: https://arxiv.org/html/1805.02867
- Version: arXiv:1805.02867v2, 28 July 2018.
- Relevant details: derives the stable running maximum and normalizer recurrence,
  bounds the normalizer, and provides an associative parallel reduction.

### NVIDIA Online Softmax Source

- URL: https://github.com/NVIDIA/online-softmax/tree/f4c607fa358789d988b1ff346393c483b457e230
- Revision: `f4c607fa358789d988b1ff346393c483b457e230`, `master` observed 28 August 2026.
- Relevant details: primary CUDA implementation associated with the online
  softmax paper. It is supporting implementation evidence, not a drop-in
  implementation of multi-head attention.

### Official FlashAttention Source

- URL: https://github.com/Dao-AILab/flash-attention/tree/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820
- Revision: `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820`, `main` observed 28 August 2026.
- Relevant files/symbols: `csrc/flash_attn/src/softmax.h` function
  `softmax_rescale_o`; `csrc/flash_attn/src/flash_fwd_kernel.h` function
  `compute_attn_1rowblock`; `csrc/flash_attn/flash_api.cpp` functions `mha_fwd`
  and `mha_varlen_fwd`; `tests/test_flash_attn.py` symbols `attention_ref` and
  `test_flash_attn_splitkv`.
- Relevant details: supports Ada, FP16/BF16, and head dimensions through 256;
  implements FP32 online state with low-precision tensor-core operands. Tests
  include one attention dimension of 100,000 but do not demonstrate square
  100,000 by 100,000 attention.

### Google Memory-Efficient Attention Source

- URL: https://github.com/google-research/google-research/tree/041338718b4e8151372fd63677104c65b73a0a4e/memory_efficient_attention
- Revision: `041338718b4e8151372fd63677104c65b73a0a4e`.
- Relevant symbols: `_query_chunk_attention` and `attention`.
- Relevant details: primary implementation accompanying Rabe and Staats,
  demonstrating independent query and key chunking and rescaled chunk summaries.

### Triton Fused Attention Tutorial

- URL: https://github.com/triton-lang/triton/blob/2db093e69e8c94d718969d4c1693802b6aad974d/python/tutorials/06-fused-attention.py
- Revision: `2db093e69e8c94d718969d4c1693802b6aad974d`.
- Relevant symbols: `_attn_fwd_inner` and `_attn_fwd`.
- Relevant details: compact FlashAttention-2-style implementation useful for
  understanding FP32 running state, low-precision probability tiles, causal
  stages, and block configuration. It must be benchmarked against PyTorch's
  built-in fused backend before adoption.

## PyTorch Behavior

### Scaled Dot Product Attention Documentation

- URL: https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html
- Documentation version: PyTorch 2.13.
- Source tag: `v2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Relevant details: documents Flash, memory-efficient, and math backends;
  boolean-mask polarity; the prohibition on combining `attn_mask` and
  `is_causal`; automatic backend selection; and possible numerical differences
  among fused implementations.

### SDPA Backend Context Manager

- URL: https://docs.pytorch.org/docs/2.13/generated/torch.nn.attention.sdpa_kernel.html
- Documentation version: PyTorch 2.13.
- Source tag: `v2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`.
- Relevant details: documents forcing `SDPBackend.FLASH_ATTENTION` so an
  unsupported input cannot silently use an unsafe math fallback.

### Current SDPA Dispatch Source

- URL: https://github.com/pytorch/pytorch/blob/46b395b5dc2b50f5b4f568c71de806996cb64345/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
- Revision: `46b395b5dc2b50f5b4f568c71de806996cb64345`, `main` observed 28 August 2026.
- Relevant symbols: `can_use_flash_attention`,
  `check_flash_attention_hardware_support`, `check_dtypes_low_precision`, and
  `check_head_dim_size_flash`.
- Relevant details: current implementation evidence that SM89, FP16/BF16, and
  head dimension 64 are eligible in principle; no explicit 100,000 sequence cap
  was found. This is not a stable API guarantee for an installed wheel.

- URL: https://github.com/pytorch/pytorch/blob/46b395b5dc2b50f5b4f568c71de806996cb64345/aten/src/ATen/native/transformers/sdp_utils_cpp.h
- Revision: same PyTorch revision.
- Relevant symbol: `check_for_attn_mask`.
- Relevant details: current Flash backend rejects a non-null attention mask.

### CUDA Memory Management

- URL: https://docs.pytorch.org/docs/2.13/notes/cuda.html#memory-management
- Documentation version: PyTorch 2.13, page last updated 1 June 2026.
- Relevant details: distinguishes allocated from reserved memory, states that
  `empty_cache()` only releases unused cached memory, and describes allocator
  tuning as useful for fragmentation or changing allocation sizes rather than
  active tensors that exceed capacity.

### Numerical Accuracy

- URL: https://docs.pytorch.org/docs/2.13/notes/numerical_accuracy.html
- Documentation version: PyTorch 2.13, page last updated 1 June 2026.
- Relevant details: documents floating-point non-associativity, batched-versus-
  sliced differences, TF32 behavior, reduced-precision GEMM reductions, and
  FP32 upcasting by the SDPA math backend.

## Hardware

### NVIDIA L4 Product Specifications

- URL: https://www.nvidia.com/en-us/data-center/l4/
- Relevant details: official 24 GB memory, 300 GB/s bandwidth, Ada architecture,
  30.3 FP32 TFLOP/s, 242 starred FP16/BF16 Tensor Core TFLOP/s, and 72 W power.
  NVIDIA states starred Tensor Core values include sparsity and are one half
  without sparsity, yielding approximately 121 dense FP16/BF16 TFLOP/s.

## Repository-Local Authorities

- `TASK.md`: task scope, immutable root-benchmark requirement, executable error
  criterion, and current absence of the official shape table.
- `torch_transformer_benchmark.py` at commit
  `9488c37883537c6bbb68b9b88ff83882846cbcd7`: executable reference operations,
  tensor lifetimes, mask semantics, defaults, and checker behavior.
- `research/team-coordination/four-way-team-split.md`: internal assumptions and
  Person 4 ownership. It is planning guidance, not an official source.
