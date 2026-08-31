# Research Index

- [Final submission report](final-submission/README.md): final fourteen-case
  benchmark, three-reference regression check, machine specifications,
  optimization summary, reproduction commands, AI-assisted workflow, team
  contributions, and limitations.
- [Team coordination](team-coordination/README.md): ownership and integration planning for the four parallel optimization streams.
- [Implementation plans](implementation-plans/README.md): preserved execution
  plans for safe SDPA routing and the fused/integrated polynomial-attention
  work, with their historical completion status.
- [Projections, FFN, layout, and elementwise fusion](projections-ffn-fusion/README.md):
  Person 3 research decomposition, source review, and optimization decisions.
- [Attention and softmax](attention-softmax/README.md): Person 2 decomposition of
  `QK^T`, causal softmax, and `PV` into optimizable stages; the roofline argument
  that this region is memory-bound; and the measured decision to target float32
  with `scaled_dot_product_attention`. Includes the original float16-reference
  reassociation rejection and the later correction that FP16 internal compute
  behind an FP32 interface is shape-specific: safe for two-layer Case 14, but
  unsafe for the four-layer residual stream. Also includes
  `long-sequence-attention.md`, which re-tests the
  approximate-attention literature at case 14's `N=100000` and validates a
  polynomial feature-map linear attention there.
- [Framework fast paths and dispatcher](framework-fastpaths/README.md): Person 1
  research on `torch.compile` modes, CUDA Graph suitability, numerical limits,
  RTX 5080 measurements, and a full-tuple routing matrix for integrating the
  four optimization streams.
- [Benchmark records](benchmarks/README.md): preserved baseline, accepted
  checkpoint, regression, and final runs, including validity status and
  replacement links.
- [Extreme-shape memory](extreme-memory/README.md): Person 4 analysis of OOM failure modes, the benchmark-harness blocker, and an exact streaming execution strategy.
