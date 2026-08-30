# Kernel-Level Profile of the Phase 1 Polynomial Path — RTX 4060 Laptop

Attribution run taken to decide what Phase 2 should target. This is a
**profile**, not a latency benchmark: it answers "where does the time go",
and section 2 explains why the absolute numbers here must not be quoted as
latency.

Date: 30 August 2026. Commit: `6dc9639` (branch `fused-kernal`, clean tree).
Motivates: [`../../attention-softmax/integrated-kernel-spec.md`](../../attention-softmax/integrated-kernel-spec.md).

## Command

```bash
PYTHONPATH=$PWD .venv/bin/python - <<'EOF'
import torch
from torch.profiler import profile, ProfilerActivity
from src.implementations.poly_attention import poly_attention_forward

B, H, N, D = 2, 16, 100000, 64
torch.manual_seed(0)
x = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16) * 0.577
split = lambda t: t.view(B, N, H, D).transpose(1, 2)   # strided, as _split_heads_view produces
q, k, v = split(x), split(x.roll(1, 0)), split(x.roll(2, 0))

with torch.inference_mode():
    poly_attention_forward(q, k, v, D ** -0.5, sigma=0.334)   # warm, absorbs autotuning
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        poly_attention_forward(q, k, v, D ** -0.5, sigma=0.334)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=22))
EOF
```

Shapes: `B=2, H=16, N=100000, d_h=64`, float16, causal, `chunk=512`,
`exact_prefix=4096`, `sigma=0.334`. `B=2` is what case 14's route actually
streams. Inputs are strided views, which is what `_split_heads_view` hands the
attention path — with contiguous inputs the internal `reshape` is free and the
profile measures a shape the module never produces.

Raw output: `profile-b2-n100000-rep1.txt`, `-rep2.txt`, `-rep3.txt`, and
`-rep4.csv` (machine-readable, with full kernel names and call counts).

## 1. Attribution

From `profile-b2-n100000-rep4.csv`, total self CUDA time 267.6 ms. Rows are
grouped by kernel family. The groups sum to 267.5 ms; one 8-call row worth
0.065 ms is not classified.

| what | ms | share | calls |
| --- | ---: | ---: | ---: |
| `_quad_update_kernel` | 68.8 | 25.7% | 196 |
| `_quad_apply_kernel` | 67.8 | 25.3% | 195 |
| **the two Triton kernels** | **136.6** | **51.1%** | 391 |
| `exp` on the `[M, C, C]` diagonal block | 22.1 | 8.3% | 196 |
| `masked_fill_` on the same block | 13.4 | 5.0% | 196 |
| all GEMM families (cuBLAS/CUTLASS) | 31.3 | 11.7% | 1369 |
| dtype conversion and copy | 20.5 | 7.7% | 3326 |
| scalar multiplies (`c0*`, `c1*`, `c2*`) | 13.4 | 5.0% | 1562 |
| adds | 12.9 | 4.8% | 2152 |
| reductions (row sums) | 12.0 | 4.5% | 783 |
| binary elementwise (the `num/den` divide) | 2.8 | 1.1% | 391 |
| `exact_prefix` flash call | 2.5 | 0.9% | 1 |
| causal mask construction | 0.005 | 0.00% | 2 |

**The two Triton kernels are about half the time. The other half is the PyTorch
glue Phase 1 deliberately left in place** — the exact diagonal block, the
constant/linear/Gram terms, the per-chunk dtype conversions, and the
normalisation. Roughly 7400 of the ~9700 launches are elementwise or reduction
work of a few microseconds each.

### 1.1 The exact diagonal block, isolated

The profile cannot cleanly separate the diagonal block's two GEMMs from the
state-side ones, so it was measured on its own at the real per-chunk shape
(`M=32, C=512, D=64`). Output in `diagonal-block-isolation.txt`; reproduce with:

```python
import torch, triton
M, C, D = 32, 512, 64
a, b, vc = (torch.randn(M, C, D, device="cuda", dtype=torch.float16) * 0.1 for _ in range(3))
blocked = torch.ones(C, C, device="cuda", dtype=torch.bool).triu(1)

def full():                                    # what the path does today
    w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
    return w @ vc, w.sum(-1, keepdim=True, dtype=torch.float32)

def blocked_lower(BK=128):                     # only the tiles a causal kernel visits
    nb = C // BK
    num = torch.zeros(M, C, D, device="cuda", dtype=torch.float16)
    den = torch.zeros(M, C, 1, device="cuda", dtype=torch.float32)
    for i in range(nb):
        for j in range(i + 1):
            w = torch.exp(a[:, i*BK:(i+1)*BK] @ b[:, j*BK:(j+1)*BK].transpose(-2, -1))
            if i == j:
                w = w.masked_fill_(blocked[:BK, :BK], 0.0)
            num[:, i*BK:(i+1)*BK] += w @ vc[:, j*BK:(j+1)*BK]
            den[:, i*BK:(i+1)*BK] += w.sum(-1, keepdim=True, dtype=torch.float32)
    return num, den

for name, fn in (("full C x C", full), ("lower-triangular tiles", blocked_lower)):
    ms = triton.testing.do_bench(fn)
    print(f"{name:24s} {ms:7.4f} ms/chunk -> {ms*196:7.1f} ms over 196 chunks")
```

Result:

```text
full C x C  (today)              0.4883 ms/chunk  ->   95.7 ms over 196 chunks
lower-triangular tiles, BK=128   1.3383 ms/chunk  ->  262.3 ms over 196 chunks
tile count: full=16, causal=10 (62%)
```

Cross-checking against the profile: `exp` + `masked_fill_` + the row-sum
reduction are 40.1 ms of confidently-attributed diagonal-only work, and its two
GEMMs put the in-loop total near 65 ms. The isolated figure of 95.7 ms is
higher because `do_bench` flushes the cache between reps and there is no overlap
with neighbouring work. **The honest bracket is 65-95 ms, i.e. 25-35% of the
whole path** — substantially more than the two Triton kernels' individual costs.

Two conclusions:

- The block computes the **full `C x C`** score matrix and masks half of it away.
  `exp`, `masked_fill_`, both GEMMs and the row sum all pay for the discarded
  upper triangle. A causal-tiled version visits 62% of the tiles at `BK=128`
  (56% at `BK=64`) and needs no mask except on the diagonal tile.
- **The causal skip cannot be done by sub-blocking in PyTorch.** The variant
  above is numerically equivalent (`max|dnum|` = 7.8e-03, one fp16 ulp;
  `max|dden|` = 0) but **2.7x slower**, because 10 tiles x 5 operations is 50
  launches per chunk against 5. It has to happen inside a Triton kernel.

## 2. Why these absolute numbers are not latency

Four profiles of the identical workload, taken back to back on this machine:

| sample | self CUDA total | the two kernels | their share |
| --- | ---: | ---: | ---: |
| unsaved first run | 334.2 ms | 184.7 ms | 55.3% |
| `rep1` | 280.3 ms | 145.7 ms | 52.0% |
| `rep2` | 263.7 ms | 135.1 ms | 51.2% |
| `rep3` | 571.1 ms | 292.5 ms | 51.2% |
| `rep4` | 267.6 ms | 136.6 ms | 51.1% |

**Absolute time varies by 2.17x across identical runs; the share is 51-55% in
every one.** The card is a laptop RTX 4060 that throttles under sustained load,
and `rep3` ran on a card that had been at 100% for a minute.

This is the same hazard `2026-08-30-rtx4060-poly/README.md` records and the
reason `src/bench_poly.py` interleaves variants. It has two consequences that
the Phase 2 spec turns into requirements:

1. **Proportional attribution is trustworthy here; absolute latency is not.**
   The table in section 1 is safe to design against. Its millisecond column is
   not safe to quote as a baseline.
2. **A noise floor must be established before any A/B is decidable.** Phase 1's
   acceptance run states plainly that none was ("no formal noise floor was
   established… 256-305 ms across repeats"). A 1.15x improvement is
   indistinguishable from a 2.17x measurement artefact without one.

The authoritative latency figures for this commit remain the interleaved ones in
[`../2026-08-30-rtx4060-poly/README.md`](../2026-08-30-rtx4060-poly/README.md):
**328.1 ms at B=2, 4.31x over exact Flash.**

## 3. Companion measurement: the tile-schedule sweep

A sweep of `BC` in {32…512}, `BI` in {1…8}, `num_warps` in {4, 8} and
`num_stages` in {2, 3} — well outside the 12- and 8-config spaces the kernels
ship with — was run to test whether the schedule leaves headroom.

**Its ranking is not usable and is deliberately not recorded here.** It timed
configs sequentially, and the shipped configuration measured 0.375 ms standalone
against 0.865 ms inside the sweep. The only conclusion it supports is a negative
one: no configuration was faster by a margin that survives a 2.17x measurement
spread, so **there is no order-of-magnitude headroom in the tile schedule**, and
Phase 2's gains have to come from structure rather than from tuning.

Re-running this sweep interleaved is Stage 0 task F4 in the Phase 2 spec.

## 4. Device

| item | value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GiB, 24 SMs, sm_89 |
| L2 cache | 33,554,432 bytes (32 MiB) |
| shared memory per block, opt-in | 101,376 bytes |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| CUDA | 13.0 |
| Python | 3.12 |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 |

The 32 MiB L2 is load-bearing for the design: the quadratic state is
`[M, D*D, V]` float32 = 1 MiB per `(batch, head)`, so the whole state working set
is **16 MiB at B=1 and 32 MiB at B=2 — exactly L2 capacity**. Chosen autotune
configs at these shapes were `BC=64, BI=2, num_warps=4` for apply and
`BC=64, BI=1, num_warps=4` for update.

## 5. Scope and limitations

- **Attention core only**, one sample-pair x one layer. Case 14 does not run end
  to end on this 8 GiB card.
- Single GPU. No RTX 5080 or L4 evidence.
- Profiler overhead is included and not subtracted; it inflates the CPU column
  materially and the CUDA column slightly. Since the design uses shares rather
  than absolutes, this does not affect the conclusions.
- One warm-up call precedes each profile, which absorbs Triton autotuning.
