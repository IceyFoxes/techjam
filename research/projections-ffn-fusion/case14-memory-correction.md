# Case #14 Memory Accounting Correction

Status: current as of 29 August 2026. This document corrects parts of the
original [`feasibility-gate.md`](feasibility-gate.md) without deleting that
historical analysis.

## Scope

Case #14 has `B=32`, `H=16`, `S=100000`, `D=1024`, `d_k=64`, `ffn_dim=1024`,
and two layers. The immutable benchmark explicitly materializes attention scores
with shape `[B,H,S,S]`. The calculations below establish a limitation of that
supplied baseline comparison. They do not establish that exact online attention,
chunking, streaming projections, or Case #14 itself is impossible.

## Corrected Storage Accounting

After `_split_heads`, `k` has shape `[32,16,100000,64]` and contains
3,276,800,000 elements, or 6.103515625 GiB in FP16. The expression
`k.transpose(-2, -1)` returns a `[32,16,64,100000]` view over the same storage.
It allocates no additional tensor storage. A backend could make an internal
copy, but that requires profiler evidence and cannot be charged statically to
`transpose`.

The score tensor contains:

```text
32 * 16 * 100000 * 100000 = 5,120,000,000,000 elements
```

Its storage alone is 10,240,000,000,000 bytes, or 9.313225746 TiB, in
FP16/BF16 and 20,480,000,000,000 bytes, or 18.626451492 TiB, in FP32.

The original exact softmax-peak claim is not supported by source inspection.
Tensor lifetimes, allocator reuse, softmax workspace, and backend behavior are
implementation-dependent. The score tensor is a firm lower bound; an exact peak
requires runtime allocation evidence on a runnable reduced shape.

## Corrected Model Weights

One layer has four attention linears, two FFN linears, and two affine
LayerNorms. Its parameter count is:

```text
4 * (D^2 + D) + (D*F + F) + (F*D + D) + 4D
= 4D^2 + 2DF + 9D + F
= 6,301,696 parameters when D=F=1024
```

Two layers contain 12,603,392 parameters. The final LayerNorm adds 2,048, for
12,605,440 total parameters. At two bytes per parameter, the complete model
weights occupy 25,210,880 bytes, or 24.04296875 MiB. The original analysis's
approximately 32 MiB total resulted from an arithmetic error.

## Supported Conclusion

The explicit baseline's full `[B,H,S,S]` materialization is infeasible on the
documented contemporary single-GPU capacities. Consequently, the supplied
baseline cannot provide an ordinary same-device comparison for Case #14 in those
environments.

Exact causal attention still represents extreme but finite work. Counting only
causally valid score positions gives 2,560,025,600,000 positions. QK and PV at
`d_k=64` total approximately 655.37 TFLOP per layer, or 1.311 PFLOP for two
layers. Feasibility and latency of a memory-safe implementation remain unmeasured
and belong to Person 4's extreme-shape investigation.

Until that route exists, integration should report Case #14 as explicitly
unsupported before attempting dense baseline or candidate execution. It should
not describe dense reference arithmetic as a memory-safe fallback.
