# Research Index

- [Team coordination](team-coordination/README.md): ownership and integration planning for the four parallel optimization streams.
- [Projections, FFN, layout, and elementwise fusion](projections-ffn-fusion/README.md):
  Person 3 research decomposition, source review, and optimization decisions.
- [Attention and softmax](attention-softmax/README.md): Person 2 decomposition of
  `QK^T`, causal softmax, and `PV` into optimizable stages; the roofline argument
  that this region is memory-bound; and the measured decision to target float32
  with `scaled_dot_product_attention`. Includes the finding that float16 fails the
  precision criterion for any arithmetic reassociation, which constrains all four
  streams.
- [Framework fast paths and dispatcher](framework-fastpaths/README.md): Person 1
  research on `torch.compile` modes, CUDA Graph suitability, numerical limits,
  RTX 5080 measurements, and a full-tuple routing matrix for integrating the
  four optimization streams.
- [Benchmark records](benchmarks/README.md): preserved baseline, accepted
  checkpoint, regression, and final runs, including validity status and
  replacement links.
- [Extreme-shape memory](extreme-memory/README.md): Person 4 analysis of OOM failure modes, the benchmark-harness blocker, and an exact streaming execution strategy.
