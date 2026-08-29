# Projections, FFN, Layout, and Elementwise Fusion

## Scope

This topic covers Person 3's ownership: Q/K/V and output projections, FFN,
LayerNorm, residual operations, and the layouts joining projections to attention.
The root [`torch_transformer_benchmark.py`](../../torch_transformer_benchmark.py)
is the executable definition of the required computation and correctness behavior.

## Research Structure

- [Major breakdown and research directions](major-breakdown.md): mathematical
  invariants, bottleneck classes, legal optimization boundaries, breakthrough
  hypotheses, and the evidence required before implementation.
- [Industry techniques against the benchmark](industry-frontier.md): August 2026
  kernel techniques filtered by the fixed graph, shapes, and correctness rule.
- [Benchmark graph analysis](benchmark-graph.md): exact operations, shape-family
  costs, benchmark semantics, and ranked Person 3 opportunities.
- [Numerical fusion boundaries](numerical-fusion.md): eager precision
  checkpoints, fusion safety, and the correctness test matrix.
- [Narrow FFN analysis](narrow-ffn.md): case #7 limits, viable fusion scopes,
  and implementation kill criteria.
- [Packed QKV layout contract](qkv-layout.md): stride analysis, attention
  compatibility, and the shared Person 2/Person 3 interface.

## Current Status

Broad reconnaissance and focused source validation were completed on 29 August
2026 across packed QKV, FFN epilogues, residual-LayerNorm fusion, layouts,
precision, and shape modeling. Benchmark reproduction is next. No exploratory
timing is an accepted performance result until preserved under
`research/benchmarks/` with the required environment and correctness metadata.
