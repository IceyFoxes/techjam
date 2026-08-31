# Integrated Polynomial Attention Kernel — Stage 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the six measured redundancies from case 14's polynomial attention path, each individually A/B'd, producing the cleaned baseline that Stage 1's integrated kernel will be measured against.

**Architecture:** The shipped path runs two Triton kernels (51% of GPU time) surrounded by ~15 PyTorch operations per chunk (the other ~49%). Stage 0 leaves that structure intact and attacks the redundancies inside it: a diagonal block that computes a full `C x C` score matrix and masks half of it away, eight chunks whose output is computed and then discarded, an apply kernel that re-reads the entire 1 MiB state once per row-block, and per-chunk dtype conversions. The largest fix — a causal-tiled diagonal block — is written as a standalone `@triton.jit` device function specifically so Stage 1 calls it rather than reimplementing it.

**Tech Stack:** PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, Python 3.12, `unittest`.

**Spec:** [`research/attention-softmax/integrated-kernel-spec.md`](../attention-softmax/integrated-kernel-spec.md)

**Scope note:** This plan covers **Stage 0 only**, plus the gate that closes it. Stage 1 is deliberately not planned here: the spec requires Stage 0's re-profile before Stage 1's design is chosen, because F1 alone may move the diagonal block from 25-35% of the path to under 12% and change what Stage 1 should target. Task 8 produces that re-profile and the Stage 1 go/no-go. A second plan follows it.

**Task F0 of the spec is already complete** (commit `708fe73`); its result is baked into the Global Constraints below.

**STATUS: EXECUTED AND ACCEPTED, 30 August 2026.** Stage 0 delivered **1.439x** (328.1 -> 228.0 ms at B=2), 6.135x over exact Flash, +61.4 MiB peak VRAM, zero correctness failures at all six oracle points, 194 tests passing. Two tasks ended in rejections (F6, and Design B at the gate) and one finding forced a fix outside the plan's scope (`SIGMA_CEILING` lowered from 0.45 to 0.40). Full record: [`research/benchmarks/2026-08-30-rtx4060-stage0/`](../benchmarks/2026-08-30-rtx4060-stage0/README.md).

## Global Constraints

- **Branch:** all work on `fused-kernal`, branched from `master`. Never commit code to `master` — research only.
- **All code under `src/`.** Never modify the root `torch_transformer_benchmark.py`.
- **Test runner is `unittest`, not pytest.** Run with `.venv/bin/python -m unittest ... -v`. Every test module guards `import torch` and skips when it is absent; CUDA-only tests additionally skip when `not torch.cuda.is_available()`.
- **Python is `.venv/bin/python`** from the repo root. It has no `pip`; do not attempt to install packages.
- **Target dtype is `float16`; the master state is `float32`.** Not a preference — a float16 master state passes at N=16384 and fails at N=65536 with 1,064,935 failures. A read-only fp16 *shadow* of an fp32 master (Task 4) is a different thing and is permitted.
- **Head dimension 64**, `d_model=1024`, `H=16`, `N=100000`, 2 layers, causal, `chunk=512`, `exact_prefix=4096`.
- **Correctness criterion:** `abs(user-ref) <= 0.002 OR abs(user-ref) <= 0.02*abs(ref)`, with **zero** failing elements.
- **Measured noise floor is 1.03x** (commit `d496539`). Three rules follow and are not negotiable:
  - **Both arms of every A/B run in the same session, interleaved.** Identical code drifted 17.5% between sessions minutes apart.
  - **Re-establish the floor at the start of every measuring session.** It was 0.6% cool and 2.7% warm.
  - **Any result inside the floor is reported as "no measurable change", never as a win.** `src/bench_poly.py` prints `RESOLVABLE` or `WITHIN NOISE`; quote the verdict.
- **`C` stays at 512 for all of Stage 0.** It becomes a tuning knob in Stage 1 only, after F1 lands.
- **The route stays opt-in.** `POLY_ATTENTION_ENABLED = False` remains the default throughout.
- Commit after every task. Push the branch after every task.

---

## File Structure

| file | responsibility | change |
| --- | --- | --- |
| `src/kernels/poly_attention_triton.py` | all Triton kernels, their shared device functions, and Python wrappers | **modify** — gains `causal_diag`, an fp16 shadow store in the update kernel, and a pre-baked config table |
| `src/kernels/poly_configs.py` | measured launch configurations keyed by shape and device capability | **create** |
| `src/implementations/poly_reference.py` | the chunked scan; the numerical oracle and the PyTorch fallback | **modify** — gains a `causal_diag` hook, prefix skipping, an optional fp16 state view, and fp16 shadows for the small states |
| `src/implementations/poly_attention.py` | wires the Triton back end into the scan | **modify** — passes the new hooks |
| `src/bench_poly.py` | interleaved latency harness with the noise floor | **modify** — gains a second polynomial arm with hooks disabled, so both sides of an A/B run in one session |
| `src/tests/test_poly_diag.py` | `causal_diag` equivalence against dense float32 | **create** |
| `src/tests/test_poly_reference.py` | scan correctness, including prefix skipping | **modify** |
| `src/tests/test_poly_kernel.py` | kernel equivalence, including the fp16 shadow | **modify** |
| `src/tests/test_poly_configs.py` | config table lookup and fallback | **create** |
| `research/attention-softmax/kernel-integration-notes.md` | what Persons 1 and 4 need to know | **modify** at the end of Stage 0 |

**Task ordering note.** The spec lists the fixes F1-F6. This plan runs them as Tasks 1-7 in a different order: **F3's fp16 shadow comes before F4's config table, and F3's `BC` half is folded into F4**, because the right `BC` cannot be chosen until the shadow exists (it changes how many bytes each row-block reads) and both are decided by the same interleaved sweep.

---

### Task 1: The causal-tiled diagonal block kernel (F1, part 1)

The largest single fix. The current diagonal block computes a full `C x C` score matrix and masks half of it away; `exp`, `masked_fill_`, both GEMMs and the row sum all pay for the discarded upper triangle. Measured at 25-35% of the whole path.

This task builds the kernel and proves it computes the right thing. Task 2 wires it in.

**Files:**
- Modify: `src/kernels/poly_attention_triton.py`
- Test: `src/tests/test_poly_diag.py` (create)

**Interfaces:**
- Consumes: `HAS_TRITON` from `src.kernels`.
- Produces: `causal_diag(a, b, v) -> tuple[Tensor, Tensor]`. `a` and `b` are `[M, C, D]` float16 and `v` is `[M, C, V]` float16; returns `(num, den)` where `num` is `[M, C, V]` float32 and `den` is `[M, C, 1]` float32. In this codebase `V == D == 64`, but the kernel does not assume it. Computes `w = tril(exp(a @ b.T))`, then `num = w @ v` and `den = w.sum(-1)`, with `w` rounded to float16 before both the `v` product and the row sum.

- [x] **Step 1: Write the failing test**

Create `src/tests/test_poly_diag.py`:

```python
"""The causal diagonal kernel must equal tril(exp(a @ b.T)) applied to v.

It replaces a dense float16 block that materialises the full C x C score matrix
and masks half of it away. Both compute the same sum in a different order and
round to float16, so they are judged against float32 truth rather than against
each other -- at these magnitudes one float16 ulp is already ~2e-3.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


def _cuda_missing():
    return torch is None or not torch.cuda.is_available()


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "Triton kernels require CUDA")
class CausalDiagTests(unittest.TestCase):
    def _case(self, M=2, C=512, D=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        kw = dict(generator=gen, device="cuda", dtype=torch.float16)
        # 0.2 reproduces the measured score scale; exp() must not saturate fp16.
        a = torch.randn(M, C, D, **kw) * 0.2
        b = torch.randn(M, C, D, **kw) * 0.2
        v = torch.randn(M, C, D, **kw) * 0.2
        return a, b, v

    def _truth_fp32(self, a, b, v):
        blocked = torch.ones(
            a.shape[1], b.shape[1], device="cuda", dtype=torch.bool
        ).triu(1)
        w = torch.exp(a.float() @ b.float().transpose(-2, -1))
        w = w.masked_fill(blocked, 0.0)
        return w @ v.float(), w.sum(-1, keepdim=True)

    def _dense_fp16(self, a, b, v):
        """The path the kernel replaces."""
        blocked = torch.ones(
            a.shape[1], b.shape[1], device="cuda", dtype=torch.bool
        ).triu(1)
        w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
        return (w @ v).float(), w.sum(-1, keepdim=True, dtype=torch.float32)

    def _assert_no_worse_than_dense(self, a, b, v):
        from src.kernels.poly_attention_triton import causal_diag

        t_num, t_den = self._truth_fp32(a, b, v)
        d_num, d_den = self._dense_fp16(a, b, v)
        k_num, k_den = causal_diag(a, b, v)
        for name, kern, dense, truth in (
            ("num", k_num, d_num, t_num),
            ("den", k_den, d_den, t_den),
        ):
            kernel_err = (kern.float() - truth).abs().max().item()
            dense_err = (dense.float() - truth).abs().max().item()
            self.assertLessEqual(
                kernel_err,
                max(3.0 * dense_err, 1e-6),
                f"{name}: kernel err {kernel_err:.3e} vs dense err {dense_err:.3e}",
            )

    def test_matches_dense_at_the_real_chunk_shape(self):
        self._assert_no_worse_than_dense(*self._case(M=2, C=512, D=64))

    def test_matches_dense_at_a_short_chunk(self):
        self._assert_no_worse_than_dense(*self._case(M=3, C=128, D=64))

    def test_matches_dense_at_a_narrow_head(self):
        self._assert_no_worse_than_dense(*self._case(M=2, C=256, D=32))

    def test_handles_a_ragged_final_chunk(self):
        """N=100000 is not a multiple of 512; the last chunk is 352 rows."""
        self._assert_no_worse_than_dense(*self._case(M=2, C=352, D=64))

    def test_is_strictly_causal(self):
        """Row i must be unaffected by any key j > i."""
        from src.kernels.poly_attention_triton import causal_diag

        a, b, v = self._case(M=1, C=128, D=64)
        num_before, den_before = causal_diag(a, b, v)
        b[:, 64:] = b[:, 64:] * 7.0   # perturb the second half of the keys
        v[:, 64:] = v[:, 64:] * 7.0
        num_after, den_after = causal_diag(a, b, v)
        self.assertEqual((num_before[:, :64] - num_after[:, :64]).abs().max().item(), 0.0)
        self.assertEqual((den_before[:, :64] - den_after[:, :64]).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_diag -v`
Expected: FAIL, `ImportError: cannot import name 'causal_diag'`.

- [x] **Step 3: Implement the kernel**

Add to `src/kernels/poly_attention_triton.py`, inside the `if HAS_TRITON:` block, after `_phi_tile`:

```python
    @triton.autotune(
        configs=[
            triton.Config({"BC": bc, "BK": bk}, num_warps=w, num_stages=2)
            for bc in (64, 128)
            for bk in (64, 128)
            for w in (4, 8)
        ],
        key=["C", "D", "V"],
    )
    @triton.jit
    def _causal_diag_kernel(
        a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
        stride_am, stride_ac, stride_ad,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_nm, stride_nc, stride_nv,
        stride_dm, stride_dc,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_c = pid_c * BC + tl.arange(0, BC)
        offs_d = tl.arange(0, D)
        offs_v = tl.arange(0, BV)
        mask_c = offs_c < C
        mask_v = offs_v < V

        a = tl.load(
            a_ptr + pid_m * stride_am
            + offs_c[:, None] * stride_ac + offs_d[None, :] * stride_ad,
            mask=mask_c[:, None], other=0.0,
        )

        acc_num = tl.zeros((BC, BV), dtype=tl.float32)
        acc_den = tl.zeros((BC,), dtype=tl.float32)

        # Key tiles beyond this row block are entirely masked out, so the loop
        # simply does not visit them. That is the whole point of the kernel:
        # at BC=BK=128 with C=512 it runs 10 of the 16 tiles, and only the
        # diagonal tile needs a mask at all.
        k_end = tl.minimum((pid_c + 1) * BC, C)
        for k0 in range(0, k_end, BK):
            offs_k = k0 + tl.arange(0, BK)
            mask_k = offs_k < C
            b = tl.load(
                b_ptr + pid_m * stride_bm
                + offs_k[:, None] * stride_bc + offs_d[None, :] * stride_bd,
                mask=mask_k[:, None], other=0.0,
            )
            s = tl.dot(a, tl.trans(b))
            w = tl.exp(s)
            w = tl.where(
                (offs_c[:, None] >= offs_k[None, :]) & mask_k[None, :], w, 0.0
            )
            # Rounded to float16 before the V product, matching what the dense
            # path does. den sums the rounded weights for the same reason: the
            # numerator and denominator must see the same w.
            wh = w.to(tl.float16)
            vt = tl.load(
                v_ptr + pid_m * stride_vm
                + offs_k[:, None] * stride_vc + offs_v[None, :] * stride_vv,
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            acc_num += tl.dot(wh, vt)
            acc_den += tl.sum(wh.to(tl.float32), axis=1)

        tl.store(
            num_ptr + pid_m * stride_nm
            + offs_c[:, None] * stride_nc + offs_v[None, :] * stride_nv,
            acc_num,
            mask=mask_c[:, None] & mask_v[None, :],
        )
        tl.store(
            den_ptr + pid_m * stride_dm + offs_c * stride_dc,
            acc_den,
            mask=mask_c,
        )
```

And the wrapper, next to `quad_apply` / `quad_update`:

```python
def causal_diag(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor):
    """``w = tril(exp(a @ b.T))``, then ``(w @ v, w.sum(-1))``, fused.

    ``a``, ``b``, ``v`` are ``[M, C, D]`` float16. Returns
    ``(num [M, C, V] float32, den [M, C, 1] float32)``.

    The dense path this replaces materialises the full ``[M, C, C]`` score
    matrix and masks half of it away. This visits only the tiles below the
    diagonal -- 10 of 16 at ``BC=BK=128, C=512`` -- and never writes the score
    matrix to HBM.

    No max subtraction. Scores are measured bounded to [-2.203, 2.404], so
    ``exp`` cannot overflow; the runtime sigma guard is what keeps that true.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = a.shape
    if b.shape != a.shape:
        raise ValueError(f"b shape {tuple(b.shape)} != a shape {tuple(a.shape)}")
    if v.shape[0] != M or v.shape[1] != C:
        raise ValueError(f"v shape {tuple(v.shape)} does not match a {tuple(a.shape)}")
    V = v.shape[2]
    a, b, v = a.contiguous(), b.contiguous(), v.contiguous()
    num = torch.empty((M, C, V), device=a.device, dtype=torch.float32)
    den = torch.empty((M, C), device=a.device, dtype=torch.float32)
    BV = max(16, triton.next_power_of_2(V))
    grid = lambda meta: (triton.cdiv(C, meta["BC"]), M)  # noqa: E731
    _causal_diag_kernel[grid](
        a, b, v, num, den,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        num.stride(0), num.stride(1), num.stride(2),
        den.stride(0), den.stride(1),
        C, D, V, BV=BV,
    )
    return num, den.unsqueeze(-1)
```

- [x] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m unittest src.tests.test_poly_diag -v`
Expected: PASS, 5 tests.

- [x] **Step 5: Commit**

```bash
git add src/kernels/poly_attention_triton.py src/tests/test_poly_diag.py
git commit -m "feat(kernels): add the causal-tiled diagonal block kernel"
git push origin fused-kernal
```

---

### Task 2: Wire the diagonal kernel into the scan and A/B it (F1, part 2)

**Files:**
- Modify: `src/implementations/poly_reference.py`
- Modify: `src/implementations/poly_attention.py`
- Test: `src/tests/test_poly_reference.py`

**Interfaces:**
- Consumes: `causal_diag(a, b, v) -> (num, den)` from Task 1.
- Produces: `poly_linear_attention(..., causal_diag=None)`. When `None`, the dense block runs exactly as today.

- [x] **Step 1: Write the failing test**

Add to `src/tests/test_poly_reference.py`:

```python
    def test_causal_diag_hook_produces_the_same_answer_as_the_dense_block(self):
        """The hook is an optimization, not a change of function."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        kw = dict(chunk=256, exact_prefix=0, sigma=0.334)

        def fake_diag(a, b, vc):
            """Dense reimplementation, to test the wiring without needing CUDA."""
            blocked = torch.ones(
                a.shape[1], b.shape[1], device=a.device, dtype=torch.bool
            ).triu(1)
            w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
            return (
                (w @ vc).float(),
                w.sum(-1, keepdim=True, dtype=torch.float32),
            )

        base = poly_linear_attention(q, k, v, scale, **kw)
        hooked = poly_linear_attention(q, k, v, scale, causal_diag=fake_diag, **kw)
        self.assertLess((base - hooked).abs().max().item(), 1e-5)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_reference -v -k causal_diag`
Expected: FAIL, `TypeError: poly_linear_attention() got an unexpected keyword argument 'causal_diag'`.

- [x] **Step 3: Implement the hook**

In `src/implementations/poly_reference.py`, add the parameter to the signature after `quad_update`:

```python
    causal_diag: Optional[Callable] = None,
```

Extend the docstring's hook paragraph:

```python
    ``causal_diag(a, b, vc) -> (num, den)`` optionally replaces the exact
    diagonal block. The dense form materialises an ``[M, C, C]`` score matrix
    and masks half of it away; a tiled kernel visits only the tiles below the
    diagonal. When ``None``, the dense block is used.
```

Replace the diagonal block (the `blocked = ...` through `del w` region) with:

```python
        # Exact diagonal block. No max subtraction: scores are measured bounded
        # to [-2.203, 2.404], so exp cannot overflow. The guard keeps that true.
        #
        # Kept in the compute dtype end to end. Upcasting to float32 here forced
        # the PV product onto an fp32 SGEMM, off the tensor cores, and tripled
        # the traffic of the [M, C, C] block through exp, the mask and the row
        # sum. Only the row sum accumulates in float32, where ~512 terms of
        # order 1 would otherwise lose bits.
        if causal_diag is None:
            blocked = blocked_full if C == chunk else blocked_full[:C, :C]
            w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
            d_num = w @ vc
            d_den = w.sum(-1, keepdim=True, dtype=torch.float32)
            del w
        else:
            d_num, d_den = causal_diag(a, b, vc)
        num = num + d_num.to(state_dtype)
        den = den + d_den.to(state_dtype)
        del d_num, d_den
```

Make `blocked_full` lazy, since the hooked path never needs it — replace its unconditional construction with:

```python
    blocked_full = (
        None
        if causal_diag is not None
        else torch.ones(chunk, chunk, device=dev, dtype=torch.bool).triu(1)
    )
```

In `src/implementations/poly_attention.py`, extend the Triton import and the call:

```python
    apply_fn = update_fn = diag_fn = None
    if use_triton and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        from src.kernels.poly_attention_triton import (
            causal_diag,
            quad_apply,
            quad_update,
        )

        apply_fn, update_fn, diag_fn = quad_apply, quad_update, causal_diag
```

and add `causal_diag=diag_fn,` to the `poly_linear_attention(...)` call.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS, all tests.

- [x] **Step 5: Verify end-to-end correctness against both oracles**

```bash
.venv/bin/python -m src.validate_poly --n 4096   --oracle dense
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
```

Expected: **zero** failing elements on all three. If any fails, stop — the fp32 `exp` in the kernel is one fewer rounding than the dense block, and the spec flagged "more accurate than the reference" as a real failure mode. Report the failure rather than loosening anything.

- [x] **Step 6: Build the A/B mechanism, then A/B the change**

Every Global Constraint about measurement requires both arms in one session, so
the harness needs a way to run the *old* path as a second arm. Add an opt-out
set rather than editing the benchmark per task.

In `src/implementations/poly_attention.py`, add a `disable` parameter:

```python
def poly_attention_forward(
    q, k, v, scale, *, sigma, use_triton: bool = True,
    disable: frozenset[str] = frozenset(),
) -> torch.Tensor:
    """...

    ``disable`` turns individual Stage 0 optimizations off by name, so the
    pre-optimization path can run as a second arm in the SAME benchmarking
    session. Identical code drifted 17.5% between sessions, so a cross-session
    A/B is not a measurement. Recognised names: ``"diag"``.
    """
    apply_fn = update_fn = diag_fn = None
    if use_triton and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        from src.kernels.poly_attention_triton import (
            causal_diag, quad_apply, quad_update,
        )

        apply_fn, update_fn = quad_apply, quad_update
        diag_fn = None if "diag" in disable else causal_diag
```

In `src/bench_poly.py`, add the arm next to `poly_triton`:

```python
    parser.add_argument(
        "--ab-disable",
        default="",
        help="comma-separated optimizations to switch off in a second arm, "
             "so both sides of an A/B run in one session (e.g. 'diag')",
    )
```

```python
        if args.ab_disable:
            off = frozenset(args.ab_disable.split(","))
            variants["poly_triton_ab_ms"] = lambda: poly_attention_forward(
                q, k, v, scale, sigma=0.3338, use_triton=True, disable=off
            )
```

Then measure both arms together:

```bash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7 --ab-disable diag
```

Compare `poly_triton_ms` against `poly_triton_ab_ms`, and check the ratio against
the A/A floor printed in the same run. Expected: the largest single gain in
Stage 0.

- [x] **Step 7: Commit**

```bash
git add src/implementations/poly_reference.py src/implementations/poly_attention.py src/tests/test_poly_reference.py
git commit -m "perf(poly): route the diagonal block through the causal-tiled kernel"
git push origin fused-kernal
```

---

### Task 3: Skip fully-prefixed chunks (F2)

Chunks entirely inside `exact_prefix` have their output overwritten by the closing SDPA call. At `exact_prefix=4096, C=512` that is chunks 0-7 of 196 — roughly 4% of apply and diagonal work, computed and discarded.

Worth ~4% against a 2.7% floor, so it is **marginally** measurable. The test asserts the work is skipped by counting hook calls, which does not depend on the floor at all.

**Files:**
- Modify: `src/implementations/poly_reference.py`
- Test: `src/tests/test_poly_reference.py`

**Interfaces:**
- Consumes: the `causal_diag` hook from Task 2.
- Produces: no signature change.

- [x] **Step 1: Write the failing test**

Add to `src/tests/test_poly_reference.py`:

```python
    def test_chunks_inside_the_exact_prefix_are_not_computed(self):
        """Their output is overwritten by SDPA, so computing it is pure waste."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        calls = []

        def counting_diag(a, b, vc):
            calls.append(a.shape[1])
            blocked = torch.ones(
                a.shape[1], b.shape[1], device=a.device, dtype=torch.bool
            ).triu(1)
            w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
            return (w @ vc).float(), w.sum(-1, keepdim=True, dtype=torch.float32)

        poly_linear_attention(
            q, k, v, scale, chunk=256, exact_prefix=512, sigma=0.334,
            causal_diag=counting_diag,
        )
        # 1024/256 = 4 chunks; the first two lie entirely inside the prefix.
        self.assertEqual(len(calls), 2)

    def test_prefix_skipping_does_not_change_the_output(self):
        """The skipped region is overwritten, so the answer must be identical."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        kw = dict(chunk=256, sigma=0.334)
        skipped = poly_linear_attention(q, k, v, scale, exact_prefix=512, **kw)
        # exact_prefix=0 then an explicit SDPA overwrite reproduces the old
        # behaviour: everything computed, the prefix then thrown away.
        import torch.nn.functional as F

        full = poly_linear_attention(q, k, v, scale, exact_prefix=0, **kw)
        full[:, :, :512] = F.scaled_dot_product_attention(
            q[:, :, :512], k[:, :, :512], v[:, :, :512], is_causal=True, scale=scale
        )
        self.assertLess((skipped - full).abs().max().item(), 1e-5)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest src.tests.test_poly_reference -v -k prefix`
Expected: FAIL on `test_chunks_inside_the_exact_prefix_are_not_computed` with `4 != 2`.

- [x] **Step 3: Implement the skip**

In `poly_linear_attention`, before the chunk loop:

```python
    # Chunks that lie entirely inside the exact prefix have their output
    # overwritten by the SDPA call at the end, so computing it is pure waste --
    # 8 of 196 chunks at exact_prefix=4096, C=512. The STATE UPDATE still runs:
    # every later chunk depends on it.
    prefix_end = min(exact_prefix, N) if exact_prefix > 0 else 0
```

Then guard the whole query side. The cleanest edit that does not reindent the
existing body is to extract it into a nested function and call it conditionally.
Replace everything from `if t0 == 0:` down to and including `del num, den` with:

```python
        def emit_chunk():
            """Numerator, denominator and output store for this chunk."""
            if t0 == 0:
                num = torch.zeros(M, C, D, device=dev, dtype=state_dtype)
                den = torch.zeros(M, C, 1, device=dev, dtype=state_dtype)
            else:
                af = a.to(cdt)
                num = (
                    c0 * s_const.expand(M, C, D)
                    + c1 * (af @ s_lin.to(cdt)).to(state_dtype)
                )
                den = (
                    c0 * z_const.expand(M, C, 1)
                    + c1 * (af @ z_lin.to(cdt)).to(state_dtype)
                )
                den = den + c2 * (
                    (af @ gram.to(cdt)) * af
                ).sum(-1, keepdim=True).to(state_dtype)
                if quad_apply is None:
                    aq = phi2(af)
                    num = num + c2 * (aq @ s_quad.to(cdt)).to(state_dtype)
                    del aq
                else:
                    num = num + c2 * quad_apply(af, s_quad_view).to(state_dtype)

            # Exact diagonal block. No max subtraction: scores are measured
            # bounded to [-2.203, 2.404], so exp cannot overflow.
            if causal_diag is None:
                blocked = blocked_full if C == chunk else blocked_full[:C, :C]
                w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
                d_num = w @ vc
                d_den = w.sum(-1, keepdim=True, dtype=torch.float32)
                del w
            else:
                d_num, d_den = causal_diag(a, b, vc)
            num = num + d_num.to(state_dtype)
            den = den + d_den.to(state_dtype)

            out[:, :, t0:t1] = (num / den).to(q.dtype).reshape(B, H, C, D)

        if t1 > prefix_end:
            emit_chunk()
```

Note `s_quad_view` in the `quad_apply` call: that name is introduced in Task 4.
Until Task 4 lands, it reads `s_quad`. If executing tasks out of order, use
whichever of the two currently exists.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS.

- [x] **Step 5: Verify correctness end to end**

```bash
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
```

Expected: zero failing elements.

- [x] **Step 6: A/B the change, same session**

```bash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7
```

Expect ~4% against a floor of 0.6-2.7%. **If the verdict is `WITHIN NOISE`, keep the change anyway and record it as unmeasured** — it strictly removes work and the hook-count test proves it, which is the honest basis here. Do not report it as a speedup.

- [x] **Step 7: Commit**

```bash
git add src/implementations/poly_reference.py src/tests/test_poly_reference.py
git commit -m "perf(poly): stop computing the chunks the exact prefix overwrites"
git push origin fused-kernal
```

---

### Task 4: The fp16 shadow of the quadratic state (F3, part 1)

`_quad_apply_kernel` is gridded over `(C/BC, M)` and **every program reads the entire `[D*D, V]` state** — at `BC=64, C=512` that is 8 full reads of 1 MiB per `(batch, head)` per chunk. The state working set is 32 MiB at B=2 against exactly 32 MiB of L2, so this survives on cache residency and is the first thing to degrade if `B` or `H` grows.

The apply kernel already does `s.to(tl.float16)` before its dot. Having the update kernel write an fp16 copy and the apply kernel read *that* is therefore **bitwise-identical** — the same round-to-nearest, just moved — and halves the bytes each row-block reads.

**The master state stays float32.** A read-only fp16 shadow of an fp32 master is not an fp16 accumulator.

**Files:**
- Modify: `src/kernels/poly_attention_triton.py`
- Modify: `src/implementations/poly_reference.py`
- Test: `src/tests/test_poly_kernel.py`

**Interfaces:**
- Consumes: `quad_apply(a, s)`, `quad_update(b, v, out)` as they exist.
- Produces: `quad_update(b, v, out, shadow=None) -> None`. When `shadow` is a `[M, D*D, V]` float16 tensor, the updated state is also written to it. `quad_apply(a, s)` gains support for `s` being float16.

- [x] **Step 1: Write the failing test**

Add to `src/tests/test_poly_kernel.py`:

```python
@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "Triton kernels require CUDA")
class QuadShadowTests(unittest.TestCase):
    def _case(self, M=2, C=128, D=64, V=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        kw = dict(generator=gen, device="cuda", dtype=torch.float16)
        b = torch.randn(M, C, D, **kw) * 0.2
        v = torch.randn(M, C, V, **kw) * 0.2
        state = torch.zeros(M, D * D, V, device="cuda", dtype=torch.float32)
        return b, v, state

    def test_shadow_is_the_float16_rounding_of_the_master(self):
        from src.kernels.poly_attention_triton import quad_update

        b, v, state = self._case()
        shadow = torch.zeros_like(state, dtype=torch.float16)
        quad_update(b, v, state, shadow=shadow)
        self.assertEqual(state.dtype, torch.float32)
        # Exactly the master, rounded -- not an independently accumulated value.
        self.assertTrue(torch.equal(shadow, state.to(torch.float16)))

    def test_applying_the_shadow_is_bitwise_identical_to_applying_the_master(self):
        """The apply kernel already converted to float16 internally."""
        from src.kernels.poly_attention_triton import quad_apply, quad_update

        b, v, state = self._case()
        shadow = torch.zeros_like(state, dtype=torch.float16)
        quad_update(b, v, state, shadow=shadow)
        a = torch.randn(
            2, 128, 64, device="cuda", dtype=torch.float16,
            generator=torch.Generator(device="cuda").manual_seed(7),
        ) * 0.2
        self.assertTrue(torch.equal(quad_apply(a, state), quad_apply(a, shadow)))

    def test_shadow_stays_in_step_across_repeated_updates(self):
        from src.kernels.poly_attention_triton import quad_update

        b, v, state = self._case()
        shadow = torch.zeros_like(state, dtype=torch.float16)
        for _ in range(4):
            quad_update(b, v, state, shadow=shadow)
        self.assertTrue(torch.equal(shadow, state.to(torch.float16)))
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest src.tests.test_poly_kernel -v -k Shadow`
Expected: FAIL, `TypeError: quad_update() got an unexpected keyword argument 'shadow'`.

- [x] **Step 3: Implement the shadow**

In `_quad_update_kernel`, add `sh_ptr` and strides after `o_ptr`'s, add `HAS_SHADOW: tl.constexpr` to the signature, and replace the final store with:

```python
        prev = tl.load(o_addr, mask=mask_v[None, :], other=0.0)
        updated = prev + acc
        tl.store(o_addr, updated, mask=mask_v[None, :])
        if HAS_SHADOW:
            # The apply kernel converts to float16 before its dot anyway, so
            # reading this instead of the master is bitwise-identical and moves
            # half the bytes.
            tl.store(
                sh_ptr + pid_m * stride_shm
                + offs_f[:, None] * stride_shf + offs_v[None, :] * stride_shv,
                updated.to(tl.float16),
                mask=mask_v[None, :],
            )
```

Add `"sh_ptr"` to the autotune `restore_value` list alongside `"o_ptr"` — it is accumulated in place for the same reason and has the same trial-repetition hazard.

In the `quad_update` wrapper, add the parameter and pass a dummy pointer when absent:

```python
def quad_update(b, v, out, shadow=None) -> None:
    ...
    if shadow is not None:
        if shadow.shape != out.shape or shadow.dtype != torch.float16:
            raise ValueError("shadow must be a float16 tensor shaped like out")
    target = shadow if shadow is not None else out
    _quad_update_kernel[grid](
        b, v, out, target,
        ...,
        target.stride(0), target.stride(1), target.stride(2),
        C, D, V, BV=BV, HAS_SHADOW=shadow is not None,
    )
```

In `quad_apply`, relax the state dtype so an fp16 state is accepted — the kernel's `s.to(tl.float16)` is already a no-op for it. Update its docstring to say `s` is `[M, D*D, V]` float32 **or float16**.

In `poly_linear_attention`, allocate the shadow when the Triton hooks are in use and pass the fp16 view to `quad_apply`:

```python
    # Read-only float16 copy of the float32 master, maintained by quad_update.
    # The apply kernel converts to float16 internally, so reading this is
    # bitwise-identical and halves the bytes each row-block reads -- and every
    # row-block reads the whole state, 8 times per chunk at BC=64, C=512.
    s_quad_shadow = (
        torch.zeros(M, D * D, D, device=dev, dtype=torch.float16)
        if quad_apply is not None and state_dtype == torch.float32
        else None
    )
    s_quad_view = s_quad if s_quad_shadow is None else s_quad_shadow
```

Use `s_quad_view` in the `quad_apply(af, ...)` call and pass `shadow=s_quad_shadow` to `quad_update`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS.

- [x] **Step 5: Verify correctness end to end**

```bash
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 65536  --oracle flash
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
```

Expected: zero failing elements. **N=65536 is the specific regression that catches an fp16 master state**; it must stay clean.

- [x] **Step 6: A/B the change, same session, and check VRAM**

```bash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7
```

The shadow adds `M * D*D * D * 2` bytes = **16 MiB at B=2**, so `poly_overhead` should rise from +67 MiB to roughly +83 MiB. That is inside the spec's +100 MiB ceiling but it consumes most of the headroom — record it explicitly. **If the latency verdict is `WITHIN NOISE`, revert this task**: it costs 16 MiB and buys nothing measurable, which fails the spec's own "kept only if it pays" rule.

- [x] **Step 7: Commit**

```bash
git add src/kernels/poly_attention_triton.py src/implementations/poly_reference.py src/tests/test_poly_kernel.py
git commit -m "perf(kernels): halve the apply kernel's state reads with a float16 shadow"
git push origin fused-kernal
```

---

### Task 5: Pre-baked launch configurations (F4, and F3 part 2)

The kernels ship with a deliberately narrow autotune space — 12 and 8 configs, `num_stages` pinned at 2 — because case 14 is a single forward pass, so compile time lands directly in the measured wall clock. That reasoning is sound but treats a symptom. `restore_value` additionally makes every update-kernel trial clone the 32 MiB state.

This task replaces autotune on the shipped path with a measured table, which also settles F3's `BC` question: the right `BC` depends on whether the shadow from Task 4 landed, so both are decided by one sweep.

**Files:**
- Create: `src/kernels/poly_configs.py`
- Create: `src/tests/test_poly_configs.py`
- Modify: `src/kernels/poly_attention_triton.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `lookup(kernel: str, key: tuple[int, ...], capability: tuple[int, int]) -> dict | None`. Returns a config dict such as `{"BC": 128, "BI": 2, "num_warps": 4, "num_stages": 2}`, or `None` when the combination has not been measured.

- [x] **Step 1: Write the failing test**

Create `src/tests/test_poly_configs.py`:

```python
"""Launch configurations are measured, not searched at runtime.

Case 14 is a single forward pass, so autotune compile time lands directly in the
measured wall clock -- a wide space ran for minutes at 1% GPU. A table keyed on
shape and device gets the benefit without the cost, and without the artificial
narrowness the compile budget forced.
"""

from __future__ import annotations

import unittest


class ConfigLookupTests(unittest.TestCase):
    def test_returns_a_measured_config_for_the_case_14_shape(self):
        from src.kernels.poly_configs import lookup

        got = lookup("quad_apply", (512, 64, 64), (8, 9))
        self.assertIsNotNone(got)
        self.assertIn("BC", got)
        self.assertIn("num_warps", got)

    def test_returns_none_for_an_unmeasured_shape(self):
        """An unknown key must fall back to autotune, not to a guessed config."""
        from src.kernels.poly_configs import lookup

        self.assertIsNone(lookup("quad_apply", (77, 13, 5), (8, 9)))

    def test_returns_none_for_an_unmeasured_device(self):
        from src.kernels.poly_configs import lookup

        self.assertIsNone(lookup("quad_apply", (512, 64, 64), (12, 0)))

    def test_every_table_entry_names_the_run_that_measured_it(self):
        """A config with no recorded provenance is a guess wearing a table."""
        from src.kernels.poly_configs import CONFIGS

        for (kernel, key, capability), entry in CONFIGS.items():
            self.assertIn("source", entry, f"{kernel} {key} {capability}")
            self.assertTrue(entry["source"].startswith("research/benchmarks/"))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m unittest src.tests.test_poly_configs -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.kernels.poly_configs'`.

- [x] **Step 3: Split each kernel into a device function with two entry points**

The sweep and the fast path both need to launch a kernel at an explicit
configuration, while the fallback still needs an autotuned one. Factor each
kernel body into a `@triton.jit` device function called by two thin entry
points, so there is one implementation and no chance of them drifting:

```python
    @triton.jit
    def _quad_apply_body(
        a_ptr, s_ptr, y_ptr,
        stride_am, stride_ac, stride_ad,
        stride_sm, stride_sf, stride_sv,
        stride_ym, stride_yc, stride_yv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        ...   # the existing body of _quad_apply_kernel, moved here verbatim

    @triton.autotune(configs=_APPLY_FALLBACK_CONFIGS, key=["C", "D", "V"])
    @triton.jit
    def _quad_apply_kernel(a_ptr, s_ptr, y_ptr, *strides_and_shape):
        _quad_apply_body(a_ptr, s_ptr, y_ptr, *strides_and_shape)

    @triton.jit
    def _quad_apply_kernel_static(a_ptr, s_ptr, y_ptr, *strides_and_shape):
        _quad_apply_body(a_ptr, s_ptr, y_ptr, *strides_and_shape)
```

Do the same for `_quad_update_kernel` and `_causal_diag_kernel`, producing
`_quad_update_kernel_static` and `_causal_diag_kernel_static`. Triton's variadic
forwarding across `@triton.jit` boundaries is limited; if `*args` forwarding does
not compile, spell the parameter lists out in full in both entry points rather
than duplicating the body.

Run `.venv/bin/python -m unittest discover -s src/tests` — the existing kernel
tests must still pass, since this step changes no arithmetic.

- [x] **Step 4: Run the interleaved sweep that produces the table**

Create `src/sweep_poly_configs.py` (code lives under `src/`, never under
`research/`). It must **interleave** configurations rather than time them
sequentially — a sequential sweep measured the shipped config at 0.375 ms
standalone and 0.865 ms inside the sweep, and its ranking was unusable.

```python
#!/usr/bin/env python3
"""Interleaved launch-configuration sweep for the polynomial kernels.

Sequential sweeps do not work on this hardware: the same configuration measured
0.375 ms alone and 0.865 ms inside a back-to-back sweep. Every candidate is
timed once per round, in rotation, and each one keeps its own minimum -- so
thermal drift is spread across all candidates instead of landing on whichever
ran last.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import triton

from src.infra.environment import collect_environment, collect_git

M, C, D, V = 32, 512, 64, 64   # case 14 at B=2: M = B*H = 32


def _candidates():
    """(kernel, config) pairs to time. OutOfResources entries drop out later."""
    for bc, bi, w, st in itertools.product((32, 64, 128, 256), (1, 2, 4), (4, 8), (2, 3)):
        yield "quad_apply", {"BC": bc, "BI": bi, "num_warps": w, "num_stages": st}
        yield "quad_update", {"BC": bc, "BI": bi, "num_warps": w, "num_stages": st}
    for bc, bk, w in itertools.product((64, 128), (64, 128), (4, 8)):
        yield "causal_diag", {"BC": bc, "BK": bk, "num_warps": w, "num_stages": 2}


def _runner(kernel, cfg, tensors):
    """A zero-argument callable launching this kernel with this configuration."""
    from src.kernels import poly_attention_triton as K

    a, b, v, state, shadow = tensors
    if kernel == "quad_apply":
        y = torch.empty(M, C, V, device="cuda", dtype=torch.float16)
        return lambda: K._quad_apply_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
            a, state, y,
            *a.stride(), *state.stride(), *y.stride(),
            C, D, V, BV=V, **cfg,
        )
    if kernel == "quad_update":
        return lambda: K._quad_update_kernel_static[(D // cfg["BI"], M)](
            b, v, state, shadow,
            *b.stride(), *v.stride(), *state.stride(), *shadow.stride(),
            C, D, V, BV=V, HAS_SHADOW=True, **cfg,
        )
    num = torch.empty(M, C, V, device="cuda", dtype=torch.float32)
    den = torch.empty(M, C, device="cuda", dtype=torch.float32)
    return lambda: K._causal_diag_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
        a, b, v, num, den,
        *a.stride(), *b.stride(), *v.stride(), *num.stride(), *den.stride(),
        C, D, V, BV=V, **cfg,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gen = torch.Generator(device="cuda").manual_seed(0)
    kw = dict(generator=gen, device="cuda", dtype=torch.float16)
    tensors = (
        torch.randn(M, C, D, **kw) * 0.2,
        torch.randn(M, C, D, **kw) * 0.2,
        torch.randn(M, C, V, **kw) * 0.2,
        torch.zeros(M, D * D, V, device="cuda", dtype=torch.float32),
        torch.zeros(M, D * D, V, device="cuda", dtype=torch.float16),
    )

    live = []
    for kernel, cfg in _candidates():
        fn = _runner(kernel, cfg, tensors)
        try:
            fn()                       # compile, and reject over-budget configs
            torch.cuda.synchronize()
        except Exception as exc:       # OutOfResources at BI>=4 exceeds 101,376 B
            print(f"skip {kernel} {cfg}: {type(exc).__name__}")
            continue
        live.append({"kernel": kernel, "config": cfg, "fn": fn, "best_ms": float("inf")})

    for _ in range(args.rounds):
        for entry in live:             # ROTATION, not one candidate at a time
            torch.cuda.synchronize()
            start = time.perf_counter()
            entry["fn"]()
            torch.cuda.synchronize()
            entry["best_ms"] = min(entry["best_ms"], (time.perf_counter() - start) * 1e3)

    results = [
        {"kernel": e["kernel"], "config": e["config"], "best_ms": e["best_ms"]}
        for e in live
    ]
    results.sort(key=lambda r: (r["kernel"], r["best_ms"]))
    for kernel in ("quad_apply", "quad_update", "causal_diag"):
        winner = min((r for r in results if r["kernel"] == kernel),
                     key=lambda r: r["best_ms"], default=None)
        print(f"{kernel}: {winner}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "shape": {"M": M, "C": C, "D": D, "V": V},
                "rounds": args.rounds,
                "environment": collect_environment(torch, torch.device("cuda")),
                "git": collect_git(),
                "results": results,
            },
            handle, indent=2, sort_keys=True,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it, substituting today's date and `git rev-parse --short HEAD`:

```bash
.venv/bin/python -m src.sweep_poly_configs --rounds 7 \
  --output research/benchmarks/$(date +%F)-rtx4060-$(git rev-parse --short HEAD)/config-sweep.json
```

The `_static` entry points it calls were created in Step 3.

- [x] **Step 5: Write the table**

Create `src/kernels/poly_configs.py`, filling the values from Step 3's sweep:

```python
"""Measured launch configurations, keyed by kernel, shape and device capability.

Autotune is the wrong mechanism for this workload. Case 14 is a single forward
pass, so search time is not amortised -- it lands directly in the measured wall
clock, and a wide space ran for minutes with the GPU at 1%. The update kernel is
worse still: it accumulates in place, so every trial has to clone the 32 MiB
state via restore_value.

Every entry names the benchmark run that measured it. An unknown key returns
None and the caller falls back to a narrow autotune, which is slow but correct;
a guessed entry would be fast and wrong.
"""

from __future__ import annotations

# (kernel, (C, D, V), (major, minor)) -> config
CONFIGS = {
    ("quad_apply", (512, 64, 64), (8, 9)): {
        "BC": 128, "BI": 2, "num_warps": 4, "num_stages": 2,
        "source": "research/benchmarks/2026-08-30-rtx4060-<commit>/config-sweep.json",
    },
    # ... one entry per kernel, filled from the sweep
}


def lookup(kernel, key, capability):
    """The measured config, or None when this combination was never measured."""
    entry = CONFIGS.get((kernel, tuple(key), tuple(capability)))
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if k != "source"}
```

- [x] **Step 6: Use the table in the kernel wrappers**

In each wrapper, consult the table before falling back to the autotuned entry point. Keep the autotuned kernel as a separate symbol so the fallback path still exists:

```python
    cfg = lookup("quad_apply", (C, D, V), torch.cuda.get_device_capability(a.device))
    if cfg is None:
        _quad_apply_kernel[grid](...)          # narrow autotune, unknown device/shape
    else:
        _quad_apply_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](..., **cfg, BV=BV)
```

`_quad_apply_kernel_static` is the same `@triton.jit` function without the `@triton.autotune` decorator. Factor the body into a shared `@triton.jit` device function so there is one implementation and two entry points.

- [x] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS.

- [x] **Step 8: Verify correctness and A/B, same session**

```bash
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7
```

Expected: zero failing elements. The gain here is partly **cold-start** — autotune compile time that no longer runs — so also record a cold-cache figure with `TRITON_CACHE_DIR` pointed at an empty directory, which is what a graded single forward pass would actually see.

- [x] **Step 9: Commit**

```bash
git add src/kernels/ src/sweep_poly_configs.py src/tests/test_poly_configs.py research/benchmarks/
git commit -m "perf(kernels): replace autotune with measured launch configurations"
git push origin fused-kernal
```

---

### Task 6: Per-chunk conversions and the z_const scalar (F5)

`s_lin`, `z_lin` and `gram` are converted float32 to float16 on all 195 chunks — part of 3326 conversion launches worth ~20.5 ms in the profile. `z_const` is a device tensor holding what is simply `t0`.

Worth ~4%, so like Task 3 it sits only marginally above the floor. **Stage 1A subsumes most of this**, so if Tasks 1-5 have consumed the available time, skip this task and record it as deferred rather than doing it badly.

**Files:**
- Modify: `src/implementations/poly_reference.py`
- Test: `src/tests/test_poly_reference.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change.

- [x] **Step 1: Write the failing test**

Add to `src/tests/test_poly_reference.py`:

```python
    def test_small_state_shadows_match_their_masters(self):
        """The float16 shadows must be rounded copies, not separate accumulators.

        An independently accumulated float16 state is the trap that passes at
        N=16384 and fails at N=65536 with over a million failures.
        """
        from src.implementations import poly_reference

        q, k, v, scale = self._qkv(N=1024)
        seen = {}
        original = poly_reference._fold_chunk_into_state

        def spy(state, *args, **kwargs):
            original(state, *args, **kwargs)
            seen["lin"] = (state.s_lin.clone(), state.s_lin_h.clone())
            seen["gram"] = (state.gram.clone(), state.gram_h.clone())

        poly_reference._fold_chunk_into_state = spy
        try:
            poly_reference.poly_linear_attention(
                q, k, v, scale, chunk=256, exact_prefix=0, sigma=0.334
            )
        finally:
            poly_reference._fold_chunk_into_state = original

        for name, (master, shadow) in seen.items():
            self.assertEqual(master.dtype, torch.float32, name)
            self.assertTrue(torch.equal(shadow, master.half()), name)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_reference -v -k shadows`
Expected: FAIL, `AttributeError: module has no attribute '_fold_chunk_into_state'`.

- [x] **Step 3: Implement**

Extract the state into a small dataclass with an explicit fold step, so masters and shadows cannot drift apart in separate places:

```python
@dataclass
class _ScanState:
    """Running prefix state. Masters are float32; the _h fields are rounded
    copies for the per-chunk matmuls, written only by ``fold``."""

    s_const: torch.Tensor
    s_lin: torch.Tensor
    gram: torch.Tensor
    z_lin: torch.Tensor
    s_lin_h: torch.Tensor
    gram_h: torch.Tensor
    z_lin_h: torch.Tensor
    z_const: float = 0.0   # a count, not a tensor: it is simply t0


def _fold_chunk_into_state(state, bf, vfc, count):
    state.s_const += vfc.sum(1, keepdim=True).float()
    state.z_const += float(count)
    state.s_lin += (bf.transpose(-2, -1) @ vfc).float()
    state.z_lin += bf.sum(1).unsqueeze(-1).float()
    state.gram += (bf.transpose(-2, -1) @ bf).float()
    # Refreshed once per chunk, replacing three .to(fp16) calls per use site.
    state.s_lin_h.copy_(state.s_lin)
    state.gram_h.copy_(state.gram)
    state.z_lin_h.copy_(state.z_lin)
```

Use `state.s_lin_h`, `state.gram_h`, `state.z_lin_h` at the numerator and denominator sites, and `state.z_const` as a Python float in the `c0 * z_const` term.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS.

- [x] **Step 5: Verify correctness and A/B, same session**

```bash
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7
```

Expected: zero failing elements. Keep the change if resolvable or if it is neutral and simplifies the code; revert if it measures slower.

- [x] **Step 6: Commit**

```bash
git add src/implementations/poly_reference.py src/tests/test_poly_reference.py
git commit -m "perf(poly): keep float16 shadows of the small states instead of reconverting"
git push origin fused-kernal
```

---

### Task 7: The redundant ai/bi loads (F6) — measure, expect to drop

Both kernels re-load `ai` (a column subset of `a`, already in registers) from global memory on every loop iteration.

**This task is expected to end in a deletion.** F0 measured the floor at 1.03x and these loads are L2- and L1-served; the spec already says F6 is "expected to be dropped", and the F0 record says it "cannot be A/B'd at all". It is here so the negative result is measured rather than assumed, and so the observation is recorded for whoever notices it next.

**Files:**
- Modify: `src/kernels/poly_attention_triton.py` (probably reverted)

**Interfaces:** none.

- [x] **Step 1: Attempt the hoist**

The loop currently reloads `ai` from global on every iteration:

```python
        for i0 in range(0, D, BI):
            offs_i = i0 + tl.arange(0, BI)
            ai = tl.load(
                a_base + offs_c[:, None] * stride_ac + offs_i[None, :] * stride_ad,
                mask=mask_c[:, None], other=0.0,
            )
            phi = _phi_tile(a, ai, BC, BI, D)
```

`a` is already resident and `ai` is `a[:, i0:i0+BI]`. With `tl.static_range` the
bound becomes a compile-time constant, which is what a static slice needs:

```python
        a3 = tl.reshape(a, (BC, D // BI, BI))          # [BC, groups, BI]
        for g in tl.static_range(0, D // BI):
            ai = tl.reshape(
                tl.sum(
                    a3 * (tl.arange(0, D // BI)[None, :, None] == g), axis=1
                ),
                (BC, BI),
            )
            phi = _phi_tile(a, ai, BC, BI, D)
```

**Judge this on the generated code, not on hope.** The `tl.sum`-with-mask idiom
above may well cost more arithmetic than the load it removes, in which case the
hoist is a pessimisation and the finding is that Triton has no cheap static
slice here. If `tl.static_range` cannot be used because `D // BI` is not a
compile-time constant in the autotuned entry point, **stop and record that**;
the negative result is the deliverable.

- [x] **Step 2: Judge it by proxy, not by A/B**

Latency cannot decide this. Use the two measures that can:

```bash
# instruction count and memory-instruction count, from the compiled artefact
.venv/bin/python -c "
from src.kernels.poly_attention_triton import _quad_apply_kernel
k = list(_quad_apply_kernel.cache[0].values())[0]
print(k.asm['ptx'].count('ld.global'))
"
```

Expected: fewer `ld.global` instructions if the hoist worked. If the count is unchanged, the hoist did nothing and is reverted.

- [x] **Step 3: Run the tests**

Run: `.venv/bin/python -m unittest discover -s src/tests`
Expected: PASS if kept.

- [x] **Step 4: Commit the outcome, whichever it is**

If kept:

```bash
git add src/kernels/poly_attention_triton.py
git commit -m "perf(kernels): hoist the redundant ai/bi loads out of the feature loop"
```

If dropped, commit **only** a comment recording the finding, so the next reader does not re-derive it:

```bash
git add src/kernels/poly_attention_triton.py
git commit -m "docs(kernels): record why the redundant ai load is not worth hoisting"
```

Then: `git push origin fused-kernal`

---

### Task 8: The Stage 0 gate — re-profile, record, and decide Stage 1

Stage 0 is not finished when the fixes land. It is finished when the new attribution is known, because Stage 1's design is chosen from it.

**Files:**
- Create: `research/benchmarks/<date>-rtx4060-<commit>/README.md`
- Modify: `research/attention-softmax/kernel-integration-notes.md`
- Modify: `research/attention-softmax/integrated-kernel-spec.md`
- Modify: `research/benchmarks/README.md`

**Interfaces:** none.

- [x] **Step 1: Re-run the full correctness table**

```bash
.venv/bin/python -m src.validate_poly --n 4096   --oracle dense
.venv/bin/python -m src.validate_poly --n 8192   --oracle dense
.venv/bin/python -m src.validate_poly --n 16384  --oracle flash
.venv/bin/python -m src.validate_poly --n 32768  --oracle flash
.venv/bin/python -m src.validate_poly --n 65536  --oracle flash
.venv/bin/python -m src.validate_poly --n 100000 --oracle flash
```

Expected: **zero** failing elements at every `N`. This is the gate; a single failure blocks Stage 0 regardless of speed.

- [x] **Step 2: Re-run the guard sweep**

```bash
for w in 1.0 1.1 1.2 1.25 1.3 1.4 1.5; do
  .venv/bin/python -m src.validate_poly --n 8192 --oracle dense --scale-qk-weights $w
done
```

Expected: reproduces the Phase 1 result — passes through `sigma` 0.4808, first failure at 0.5217 — confirming `SIGMA_CEILING = 0.45` still has margin.

- [x] **Step 3: Establish the floor and measure the Stage 0 baseline**

```bash
.venv/bin/python -m src.bench_poly --n 100000 --batch 1 --reps 7 \
  --output research/benchmarks/<date>-rtx4060-<commit>/stage0-b1.json
.venv/bin/python -m src.bench_poly --n 100000 --batch 2 --reps 7 \
  --output research/benchmarks/<date>-rtx4060-<commit>/stage0-b2.json
```

Gate: **>= 1.15x** over the shipped 328.1 ms at B=2, with the ratio marked `RESOLVABLE`. Record the peak VRAM overhead against the **+100 MiB** ceiling — Task 4's shadow alone consumes 16 MiB of the 33 MiB headroom.

- [x] **Step 4: Re-profile for attribution**

Re-run the kernel-level profile from
`research/benchmarks/2026-08-30-rtx4060-6dc9639/README.md` (the command is recorded there verbatim) and produce the new grouped table. Report **shares**, not milliseconds — four identical runs of the old path spread 2.17x while shares held at 51-55%.

- [x] **Step 5: Decide Stage 1, in writing**

The spec's Design B gate has two conditions. Answer both from Step 4's profile:

1. Is state traffic still the top cost after F1 and F3?
2. Does its projected peak overhead fit the +100 MiB ceiling, given the `[M, N, V]` float32 partial buffer is +819 MiB?

Record the answer either way. **"Design B rejected with evidence" is a legitimate outcome of Phase 2, not a failure of it.** If Design A is confirmed, note which of its components the new profile says to prioritise.

- [x] **Step 6: Write the run record**

Create `research/benchmarks/<date>-rtx4060-<commit>/README.md` carrying, per the repository's benchmark policy: the exact commands, commit, timestamp, input shapes, dtype, correctness result, latency with its noise floor and verdict, peak VRAM, and the CPU/GPU/OS/driver/CUDA/PyTorch versions. Add it to `research/benchmarks/README.md`.

State explicitly which fixes were kept, which were `WITHIN NOISE` and kept anyway on structural grounds, and which were reverted.

- [x] **Step 7: Update the integration notes and the spec**

In `research/attention-softmax/kernel-integration-notes.md`, replace the latency and VRAM tables with the Stage 0 figures. Persons 1 and 4 read this document; it must never carry a stale number.

In the spec, mark F1-F6 with their outcomes and record the Stage 1 decision from Step 5.

- [x] **Step 8: Commit**

```bash
git add research/ docs/
git commit -m "bench(poly): record the Stage 0 baseline and the Stage 1 decision"
git push origin fused-kernal
```

---

## What comes after this plan

Stage 1 gets its own plan, written against Task 8's profile. It builds the two complete per-chunk kernels (`poly_chunk_query` and `poly_chunk_state`) sketched in spec section 5.1, A/Bs `C` in {512, 1024, 2048} now that the diagonal block is causally tiled, and either builds or formally rejects the persistent-slab scan according to Task 8 Step 5.
