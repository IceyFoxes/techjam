# Survey of Fast-Attention Techniques, Filtered by This Task's Constraints

An idea catalogue for the attention/softmax stream. Each entry records the core
idea, then whether it is usable here and why. Preference was given to work
published at major venues (NeurIPS, ICML, ICLR, ACL, MLSys) as a credibility
proxy.

Status: current as of 29 August 2026. Companion to
[`decomposition.md`](decomposition.md), which defines levers L1-L8.

## The filter: three constraints kill most of the literature

1. **Weights are fixed.** The harness copies the reference `state_dict` with
   `strict=True` (`copy_model_weights`,
   [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py)
   lines 203-213). Anything that changes parameter shapes or count — projecting
   keys/values to a smaller rank, sharing K/V across heads, learned bucketing —
   cannot be loaded, let alone scored.
2. **Output must match the reference numerically.** `abs <= 0.002 OR rel <= 2%`
   with **zero** failing elements. Any method that *approximates* attention
   changes the mathematical result and fails by construction, not by a tuning
   margin.
3. **The shapes are not the ones this literature targets.** Almost all efficient
   attention research assumes long context. Our sequence lengths are 32, 128 and
   1024 with large batches. Reformer's gains reportedly only appear beyond
   N = 2048; Longformer, BigBird, Performer and Linformer are all motivated by
   N in the thousands to hundreds of thousands.

Consequence: **the entire "efficient attention architecture" literature is out of
scope for correctness reasons**, and most of it would not help at our shapes
anyway. It is catalogued below because the task asks for a technical report and
because several of its *mechanisms* (not its approximations) transfer.

What remains usable is the exact, IO-aware, kernel-level branch.

---

## Tier A — Exact and directly usable

These change *how* attention is computed, not *what* it computes.

### A1. FlashAttention / FlashAttention-2 — tiling + online softmax

Fuses score computation, masking, softmax and the `PV` product into one kernel so
the `N x N` matrix never reaches HBM. FlashAttention-2 adds three refinements:
fewer non-matmul FLOPs, parallelization across thread blocks for occupancy, and
warp-level work partitioning to cut shared-memory traffic. Reports ~2x over
FlashAttention-1 and 50-73% of theoretical peak on A100.

**Applies: yes — this is lever L1**, obtained through PyTorch SDPA. Note the flash
backend is float16/bfloat16 only, so our float32 path gets the memory-efficient
backend instead.

### A2. Self-attention Does Not Need O(n²) Memory — chunking + lazy normalization

Rabe & Staats show attention needs only `O(log n)` memory in principle and
`O(sqrt n)` in a practical numerically stable form, by processing keys in chunks
and deferring softmax normalization. Reports 59x memory reduction at N = 16,384.

**Applies: yes.** This is the algorithm behind PyTorch's memory-efficient backend
— the one we actually get in float32. Effectively our production path.

### A3. Online softmax normalizer — the enabling recurrence

Milakov & Gimelshein's single-pass running-max/running-sum recurrence with the
`exp(m_prev - m_new)` rescale. Cuts softmax memory accesses from 4 to 3 per
element on its own, and is the precondition for A1 and A2.

**Applies: yes — lever L4.** Also the thing to reimplement if we ever write a
custom kernel.

### A4. Data Movement Is All You Need (MLSys 2021)

Ivanov et al. build a dataflow graph of a transformer layer, classify every
operator by arithmetic intensity, and show training is **memory-bound**, not
compute-bound. Fusing adjacent operators and systematically searching data
layouts gives 1.30x on a BERT encoder layer and 1.19x end-to-end.

**Applies: yes, and it independently corroborates our central finding.** Our
roofline puts case 13 at ~176 ms of memory traffic against ~6.9 ms of matmul —
memory-bound by ~25x. Their methodology (enumerate operators, measure intensity,
fuse the low-intensity ones) is exactly the decomposition in
[`decomposition.md`](decomposition.md). Their layout-search result also argues for
revisiting the `.contiguous()` copies at stages 2 and 8, which our profiling shows
costing 7.1x the attention matmul.

### A5. FlexAttention (PyTorch, 2024)

A `score_mod` / `mask_mod` API that expresses arbitrary attention masking in a few
lines of PyTorch and lowers it to a fused FlashAttention-style kernel through
`torch.compile`. `create_block_mask` precomputes block sparsity so whole blocks
are skipped, and the same block mask can be reused across layers without
recompiling.

**Applies: measured, and the answer is no — SDPA already does this.** Causal
masking is *exact* structured sparsity, so lever L3 looked like the big remaining
win. Measuring `is_causal=True` against `is_causal=False` on the float32
memory-efficient backend shows it **already skips upper-triangle blocks**:

| shape | full | causal | ratio | noise |
| --- | --- | --- | --- | --- |
| B=64 H=4 N=2048 `d_h`=32 | 59.338 ms | 30.949 ms | **0.522** | ±0.3% |
| B=64 H=4 N=1024 `d_h`=32 | 14.464 ms | 7.763 ms | **0.537** | ±0.8% |
| B=64 H=4 N=512 `d_h`=32 | 3.846 ms | 2.213 ms | 0.575 | ±1.2% |
| B=64 H=4 N=128 `d_h`=32 | 2.672 ms | 2.059 ms | 0.771 | ±0.9% |
| B=64 H=16 N=128 `d_h`=8 | 1.322 ms | 0.998 ms | 0.755 | ±3.4% |

A ratio of ~0.5 is the theoretical maximum for causal skipping, and the backend
reaches it at N >= 1024. **Lever L3 is therefore already delivered by SDPA**, and
FlexAttention would add nothing for plain causal masking.

The shortfall at N=128 (0.77 rather than 0.5) is block granularity, not a missed
optimization: with a block size on the order of 64-128, a 128-length sequence is
one or two blocks, and the partially-masked diagonal blocks must be computed in
full. FlexAttention uses the same block-mask granularity and would hit the same
floor. FlexAttention remains interesting only if we later need a mask pattern
SDPA cannot express — which, with `causal=True` on every official case, we do not.

### A6. Flash-Decoding — split-K over the sequence dimension

Adds a parallelization axis by splitting keys/values into chunks, attending to
each in parallel, storing one log-sum-exp scalar per row per split, then reducing
across splits using those scalars. Reports up to 8x for long-context decoding.

**Applies: no — measured, only one case is occupancy-starved and it is
launch-bound anyway.** The 8x headline is for autoregressive decode (query
length 1), which we do not have. The transferable mechanism is recovering
parallelism when `batch x heads` underfills the GPU, so the question is which of
our cases are starved. Against the measured 24 SMs:

| Case | B | H | `B*H` | vs 24 SMs | verdict |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 4 | 4 | 0.2x | **starved** |
| 3 | 4 | 4 | 16 | 0.7x | marginal |
| 9 | 64 | 1 | 64 | 2.7x | fine |
| 4 | 16 | 4 | 64 | 2.7x | fine |
| all others | | | 128-1024 | 5.3-42.7x | fine |

An earlier draft of this document guessed that case 9 (H=1) would be starved.
It is not: `H=1` is offset by `B=64`. Only case 2 is genuinely starved, and case 2
is also the launch-bound one — its GPU sits at 11-20% utilization drawing 12 W of
a 70 W budget, so the constraint there is CPU dispatch, not GPU parallelism.
Split-K would add launches to a case already limited by launches.

The log-sum-exp rescaled reduction remains the correct way to combine partial
softmax results if we ever write a custom kernel.

### A7. CUTLASS/CuTe fused attention implementations

xFormers' memory-efficient attention is a CUTLASS 2.X FMHA kernel targeting
SM50-SM80, keeping operands in shared memory where FlashAttention keeps them in
registers. There is also a published case study implementing FlashAttention-2 on
Hopper with CUTLASS.

**Applies: as reference material.** We are on sm_89 and getting this kernel via
PyTorch already. Relevant only if we hand-write a kernel and need a proven
structure to copy.

### A8. Neighborhood attention at the threadblock level (NATTEN)

Reduces the `O(n²)` cost by restricting attention to a neighbourhood, implemented
with threadblock-level tiling rather than a masking overlay.

**Applies: mechanism only.** The neighbourhood restriction is an approximation and
is excluded. The threadblock-level tiling discipline is the transferable part.

---

## Tier B — System-level, relevant because our shapes are small

Ten of the twelve in-scope cases have N = 128 or 32. Profiling shows case 1 has a
flat operator profile with no dominant kernel and case 2 runs at 11-20% GPU
utilization — these are **launch-bound**, a regime the attention literature
largely ignores.

### B1. CUDA Graphs / launch-overhead elimination

Reported CPU dispatch overhead of 20-30% of step time at small batch sizes, with
hundreds of launches per forward pass and tens of microseconds of CPU bookkeeping
per launch. CUDA graph capture removes per-step launch overhead entirely.

**Applies: yes, but it is Person 1's territory** (compilation, CUDA graphs,
dispatcher) per the team split. Flagged here because our measurements show the
small-shape cases are launch-bound, which means **Person 2 cannot fix cases 2, 3,
12 by improving attention math at all.** That is a scoping conclusion, not an
optimization.

### B2. Kernel fusion of the surrounding elementwise work

Our operator attribution shows `aten::copy_` + `Memcpy DtoD` at 7.1x the cost of
the attention matmul on case 13, from `_split_heads().contiguous()` and the
context transpose.

**Applies: yes, but at the Person 2/Person 3 boundary** (stages 2 and 8). Worth
raising jointly rather than changing unilaterally.

---

## Tier C — Approximate attention: excluded, catalogued for the report

Every method here changes the mathematical result and therefore **cannot pass the
tolerance criterion**. They are recorded so the technical report can show the
space was considered, and because their *taxonomy* is useful.

The organizing reference is Tay et al., *Efficient Transformers: A Survey* (ACM
Computing Surveys), reported at ~1088 citations, which classifies methods as
Fixed Patterns, Learnable Patterns, Low-Rank, Kernels, Recurrence, Memory, and
Downsampling.

| Method | Idea | Complexity | Why excluded |
| --- | --- | --- | --- |
| **Sparse Transformer** (Child et al., 2019) | Factorized local + strided patterns split across heads | `O(n sqrt n)` | Approximation; drops real score mass |
| **Longformer** | Sliding window + dilated window + task-selected global tokens | `O(n)` | Approximation; also needs chosen global tokens |
| **BigBird** | Random + windowed + global, justified via Erdős-Rényi graph connectivity | `O(n)` | Approximation |
| **Reformer** | LSH buckets approximate nearest-neighbour attention | `O(n log n)` | Approximation; gains reportedly only beyond N=2048 |
| **Linformer** | Projects K and V from `n` to `k`, exploiting low-rank structure of the attention matrix | `O(nk)` | Approximation **and** adds projection parameters that break `strict` weight loading |
| **Performer / FAVOR+** | Unbiased softmax-kernel estimate via positive orthogonal random features | `O(n)` | Stochastic approximation; unbiased in expectation is not per-element tolerance |
| **Linear Transformer** (Katharopoulos et al., ICML 2020) | Replace `exp(q·k)` with `φ(q)·φ(k)`, then reassociate to `φ(Q)(φ(K)^T V)`; causal case becomes an RNN | `O(n)` | Different kernel, different result |
| **Nyströmformer** | Nyström method: approximate the `n x n` matrix from a sampled `m x m` submatrix | `O(n)` | Approximation |
| **cosFormer** | Drop softmax; enforce non-negativity plus a cosine re-weighting | `O(n)` | Changes the operator |
| **SOFT** | Softmax-free self-attention with Gaussian kernel | `O(n)` | Changes the operator |
| **ReLU / sigmoid attention** | Replace softmax with `relu(x)/n` or elementwise sigmoid | `O(n²)` | Changes the operator |
| **Native Sparse Attention** (DeepSeek, ACL 2025) | Hierarchical compression + fine-grained token selection + sliding window, hardware-aligned and natively trainable | sub-quadratic | Approximation; also requires training |
| **MQA / GQA** | Share K/V heads across query heads to shrink the KV cache | — | Changes parameter count; breaks weight loading. Also a *decode-time memory-bandwidth* fix, and we have no KV cache |

### The one transferable idea from Tier C

**Structured sparsity is legitimate when the structure is exact.** Causal masking
is precisely that: the strict upper triangle is mathematically zero after
softmax masking, not approximately so. The block-skipping machinery developed for
Sparse Transformer / Longformer / BigBird / NSA therefore applies to us **in its
exact form** — see lever L3 and idea A5. This is the single place where the
approximate-attention literature pays off here.

---

## Ranked ideas worth prototyping

Revised after measuring causal block-skipping (A5). That measurement removed what
had been the top-ranked idea, which is the main practical result of this survey:
**there is no large algorithmic win left in the attention core beyond adopting
SDPA.** The remaining opportunities are at the boundaries.

1. **Layout work at stages 2 and 8 (A4, B2), jointly with Person 3.**
   `aten::copy_` + `Memcpy DtoD` measure **7.1x the attention matmul** on case 13
   — the `.contiguous()` in `_split_heads` and the context transpose. This is now
   the largest remaining attention-adjacent cost, and Ivanov et al.'s layout-search
   methodology applies directly. Requires negotiating the stage 2/8 boundary.
2. **Accept that cases 2, 3, 12 are launch-bound (B1)** and out of Person 2's
   reach. They need CUDA graphs or compilation, which is Person 1's scope.
3. **Nothing from Tier C**; **nothing from FlexAttention** for plain causal
   masking; and **nothing from split-K (A6)** — occupancy analysis shows only
   case 2 is starved, and it is launch-bound rather than parallelism-bound.

### What this means for a custom Triton kernel

The case for hand-writing one has weakened considerably. SDPA already delivers
L1 (no `N²` materialization), L3 (causal block skipping, at the theoretical 0.5
ratio for N >= 1024), L4 and L5 (online softmax, deferred normalization). A
custom kernel would have to beat a mature CUTLASS FMHA implementation at its own
game. Recommend **not** pursuing one unless a specific measured gap appears.

## Sources

- **Tay, Y., Dehghani, M., Bahri, D., Metzler, D., "Efficient Transformers: A
  Survey", ACM Computing Surveys.** <https://arxiv.org/abs/2009.06732>. Accessed
  29 August 2026. Reported ~1088 citations. Taxonomy of efficient attention into
  Fixed Patterns, Learnable Patterns, Low-Rank, Kernel, Recurrence, Memory and
  Downsampling classes. Used here as the organizing frame for Tier C and to
  confirm that every class in it is an approximation, hence excluded.

- **Dao, T., "FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning", ICLR 2024.** <https://arxiv.org/abs/2307.08691>. Accessed
  29 August 2026. Reduces non-matmul FLOPs, parallelizes across thread blocks for
  occupancy, and partitions work between warps; ~2x over FlashAttention-1 and
  50-73% of A100 peak. Source of levers L3 and L5.

- **Rabe, M. N., Staats, C., "Self-attention Does Not Need O(n²) Memory".**
  <https://arxiv.org/abs/2112.05682>. Accessed 29 August 2026. `O(log n)`
  theoretical and `O(sqrt n)` practical memory via chunking and lazy softmax
  normalization; 59x memory reduction at N=16,384. The algorithm behind PyTorch's
  memory-efficient backend, which is the float32 path we actually use.

- **Milakov, M., Gimelshein, N., "Online normalizer calculation for softmax".**
  <https://arxiv.org/abs/1805.02867>. Accessed 29 August 2026. Single-pass
  running-max/running-normalizer recurrence; softmax memory accesses 4 -> 3 per
  element. Lever L4 and the precondition for all fused attention.

- **Ivanov, A., Dryden, N., Ben-Nun, T., Li, S., Hoefler, T., "Data Movement Is
  All You Need: A Case Study on Optimizing Transformers", MLSys 2021.**
  <https://arxiv.org/abs/2007.00072>. Accessed 29 August 2026. Dataflow-graph
  analysis showing transformer training is memory-bound; operator fusion plus
  systematic data-layout search yields 1.30x on a BERT encoder layer and 1.19x
  end-to-end, reducing data movement up to 22.91%. Independently corroborates
  this project's roofline conclusion and motivates the layout work at stages 2
  and 8.

- **PyTorch team, "FlexAttention: The Flexibility of PyTorch with the Performance
  of FlashAttention", August 2024.** <https://pytorch.org/blog/flexattention/>.
  Accessed 29 August 2026. `score_mod` and `mask_mod` callables lowered to a fused
  FlashAttention kernel via `torch.compile`; `create_block_mask` precomputes block
  sparsity and is reusable across layers without recompilation. The proposed route
  to lever L3. Implementation:
  <https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py>.

- **Together AI / PyTorch, "Flash-Decoding for long-context inference", 2023.**
  <https://pytorch.org/blog/flash-decoding/>. Accessed 29 August 2026. Splits
  keys/values into chunks attended in parallel, storing per-row-per-split
  log-sum-exp and reducing across splits; up to 8x for long-context decoding.
  Relevant here for its parallelism-recovery mechanism on low-occupancy shapes,
  not its decode use case.

- **Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F., "Transformers are RNNs:
  Fast Autoregressive Transformers with Linear Attention", ICML 2020.**
  <https://arxiv.org/abs/2006.16236>. Accessed 29 August 2026. Expresses attention
  as a kernel feature map and reassociates to `O(N)`; causal case becomes a
  constant-memory RNN; up to 4000x on very long autoregressive sequences.
  Excluded: changes the kernel and therefore the result.

- **Choromanski, K. et al., "Rethinking Attention with Performers", ICLR 2021.**
  <https://arxiv.org/abs/2009.14794>. Accessed 29 August 2026. FAVOR+ estimates
  the softmax kernel with positive orthogonal random features, giving unbiased
  linear-complexity attention without sparsity or low-rank priors. Excluded:
  unbiased in expectation does not satisfy a per-element tolerance.

- **Wang, S. et al., "Linformer: Self-Attention with Linear Complexity".**
  <https://arxiv.org/abs/2006.04768>. Accessed 29 August 2026. Observes the
  attention matrix is approximately low-rank and projects K and V from `n` to `k`
  for `O(nk)`. Excluded twice over: approximation, and the added projections break
  `strict=True` weight loading.

- **Xiong, Y. et al., "Nyströmformer: A Nyström-Based Algorithm for Approximating
  Self-Attention", AAAI 2021.** <https://arxiv.org/abs/2102.03902>. Accessed
  29 August 2026. Applies the Nyström method to approximate the `n x n` matrix
  from a sampled submatrix in `O(n)`. Excluded: approximation.

- **Qin, Z. et al., "cosFormer: Rethinking Softmax in Attention", ICLR 2022.**
  <https://arxiv.org/abs/2202.08791>. Accessed 29 August 2026. Identifies
  non-negativity and concentrated re-weighting as the two properties of softmax
  that matter, then reproduces them with a linear operator plus cosine distance
  re-weighting. Excluded: replaces the operator.

- **Child, R. et al., "Generating Long Sequences with Sparse Transformers", 2019.**
  <https://arxiv.org/abs/1904.10509>. Accessed 29 August 2026. Factorized local
  and strided attention patterns split across heads, giving `O(n sqrt n)`.
  Excluded as an approximation, but its block-factorization idea is the ancestor
  of the exact causal block-skipping we do want.

- **Beltagy, I., Peters, M. E., Cohan, A., "Longformer: The Long-Document
  Transformer", 2020.** <https://arxiv.org/abs/2004.05150>. Accessed
  29 August 2026. Sliding-window plus dilated-window plus task-motivated global
  attention for `O(n)`. Excluded: approximation.

- **Zaheer, M. et al., "Big Bird: Transformers for Longer Sequences", NeurIPS
  2020.** <https://arxiv.org/abs/2007.14062>. Accessed 29 August 2026. Random,
  windowed and global attention patterns with connectivity argued via
  Erdős-Rényi graphs. Excluded: approximation.

- **Kitaev, N., Kaiser, Ł., Levskaya, A., "Reformer: The Efficient Transformer",
  ICLR 2020.** <https://arxiv.org/abs/2001.04451>. Accessed 29 August 2026. LSH
  bucketing approximates nearest-neighbour attention in `O(n log n)`. Excluded:
  approximation, and reported gains only appear beyond N=2048, well above our
  largest in-scope N of 1024.

- **Yuan, J. et al. (DeepSeek-AI), "Native Sparse Attention: Hardware-Aligned and
  Natively Trainable Sparse Attention", ACL 2025.**
  <https://arxiv.org/abs/2502.11089>. Accessed 29 August 2026. Three parallel
  branches — compressed coarse-grained attention, fine-grained selected token
  blocks, and sliding local attention — co-designed with hardware. Excluded:
  approximation requiring training, but the clearest modern statement of
  hardware-aligned *block* granularity, which is the property lever L3 exploits.

- **Shazeer, N., "Fast Transformer Decoding: One Write-Head is All You Need",
  2019** (<https://arxiv.org/abs/1911.02150>) and **Ainslie, J. et al., "GQA:
  Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints", EMNLP 2023** (<https://arxiv.org/abs/2305.13245>). Both accessed
  29 August 2026. Share K/V heads across query heads to cut KV-cache bandwidth.
  Excluded: changes parameter count (breaks `strict` loading), and targets
  autoregressive decode bandwidth, which this benchmark does not exercise.

- **Wortsman, M. et al., "Replacing softmax with ReLU in Vision Transformers".**
  <https://arxiv.org/abs/2309.08586>. Accessed 29 August 2026. And **Ramapuram, J.
  et al., "Theory, Analysis, and Best Practices for Sigmoid Self-Attention"**,
  <https://arxiv.org/abs/2409.04431>, accessed 29 August 2026. Both replace the
  softmax normalizer. Excluded: change the operator, so they cannot match the
  reference.

- **Lu, J. et al., "SOFT: Softmax-free Transformer with Linear Complexity",
  NeurIPS 2021 (spotlight).** <https://arxiv.org/abs/2110.11945>. Accessed
  29 August 2026. Gaussian-kernel replacement for softmax with linear complexity.
  Excluded: changes the operator.

- **Hassani, A. et al., "Faster Neighborhood Attention: Reducing the O(n²) Cost of
  Self Attention at the Threadblock Level".** <https://arxiv.org/abs/2403.04690>.
  Accessed 29 August 2026. Implements neighbourhood attention as threadblock-level
  tiling rather than a masking overlay. The restriction is an approximation and is
  excluded; the threadblock tiling discipline is the transferable part.

- **Bikshandi, G., Shah, J., "A Case Study in CUDA Kernel Fusion: Implementing
  FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library".**
  <https://arxiv.org/abs/2312.11918>. Accessed 29 August 2026. Reference structure
  for a hand-written fused attention kernel. Consulted only as implementation
  guidance; the Hopper-specific features do not exist on our sm_89 target.

- **xFormers memory-efficient attention (Meta).**
  <https://github.com/facebookresearch/xformers>. Accessed 29 August 2026.
  CUTLASS 2.X FMHA kernel targeting SM50-SM80, keeping operands in shared memory
  where FlashAttention uses registers. This is the lineage of the backend PyTorch
  selects for float32 SDPA, which is our production path.
