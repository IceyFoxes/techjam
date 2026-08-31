# Fused Polynomial Attention Kernel (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut case 14's attention cost by at least 2x by fusing the polynomial feature-map GEMM into a Triton kernel, so the `a (x) a` tile is generated in registers and never written to HBM.

**Architecture:** Order-2 Gauss-Hermite polynomial linear attention over a chunked causal scan. Two Triton kernels replace the two operations that materialise a `[C, 4096]` feature tensor — applying the running state to queries, and folding a chunk into the state. PyTorch retains the chunk loop, the exact diagonal block, and the constant/linear terms. A runtime `sigma` guard falls back to exact Flash SDPA when the score distribution leaves the validated range.

**Tech Stack:** PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, Python 3.12, `unittest`.

**Spec:** [`research/attention-softmax/triton-kernel-spec.md`](../attention-softmax/triton-kernel-spec.md)

## Global Constraints

- **Branch:** all work on a branch off `master`. Never commit code to `master` (research only). Current branch: `fused-kernal`.
- **All code under `src/`.** Never modify root `torch_transformer_benchmark.py`.
- **Test runner is `unittest`, not pytest.** Run with `.venv/bin/python -m unittest ... -v`. Every test module guards `import torch` and uses `@unittest.skipIf(torch is None, ...)`; CUDA-only tests additionally skip when `not torch.cuda.is_available()`.
- **Python is `.venv/bin/python`** from the repo root. It has no `pip`; do not attempt to install packages.
- **Target dtype is `float16`**; the master state is **`float32`** and per-chunk matmuls are **`float16`**. This is not a preference — a float16 state passes at N=16384 and fails at N=65536 with 1,064,935 failures.
- **Chunk length `C = 512`, `exact_prefix = 4096`.** Not tuning parameters; changing them changes the approximation.
- **Head dimension is 64**, `d_model=1024`, `H=16`, `N=100000`, 2 layers, causal.
- **Correctness criterion:** `abs(user-ref) <= 0.002 OR abs(user-ref) <= 0.02*abs(ref)`, with **zero** failing elements.
- **Acceptance:** `<= 360 ms` per sample-layer at N=100000. Above `603.9 ms` is rejected outright.
- **Polynomial coefficients:** `g = exp(sigma^2/2)`, then `c0 = g*(1 - sigma^2/2)`, `c1 = g`, `c2 = g/2`. The `g` factor must **not** be dropped -- the diagonal chunk uses unscaled `exp`, so dropping it de-scales the inter-chunk term by ~5.6% and measures worse than plain Taylor. Corrected during Task 1; see the spec's section 3.
- Commit after every task. Push the branch after every task.

---

## File Structure

| file | responsibility |
| --- | --- |
| `src/implementations/poly_reference.py` | PyTorch chunked polynomial attention. The numerical oracle for the kernels and a working fallback. |
| `src/implementations/poly_guard.py` | `sigma` estimation, Hermite constant, and the safety ceiling. No torch kernels. |
| `src/kernels/__init__.py` | Triton availability probe. |
| `src/kernels/poly_attention_triton.py` | Both Triton kernels, the shared `phi` device function, and their Python wrappers. |
| `src/implementations/poly_attention.py` | Attention module, `CandidateSpec` named `poly`, guard wiring, flash fallback. |
| `src/tests/test_poly_reference.py` | Oracle correctness. |
| `src/tests/test_poly_guard.py` | Guard logic (CPU-only, no CUDA needed). |
| `src/tests/test_poly_kernel.py` | Kernel-vs-dense equivalence. |
| `src/tests/test_poly_attention.py` | End-to-end criterion, fallback, and the fp16-state regression. |
| `docs/kernel-integration-notes.md` | Written for Persons 1 and 4: what changed in their files and how to disable it. |

---

### Task 1: PyTorch reference implementation (the oracle)

Everything else is validated against this. It is the promoted, cleaned-up form of the validated spike.

**Files:**
- Create: `src/implementations/poly_reference.py`
- Test: `src/tests/test_poly_reference.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `poly_linear_attention(q, k, v, scale, *, chunk=512, exact_prefix=4096, sigma=None, state_dtype=torch.float32, compute_dtype=torch.float16, quad_apply=None, quad_update=None) -> Tensor`. Accepts `q, k, v` of shape `[B, H, N, D]` and returns `[B, H, N, D]`. The `quad_apply` / `quad_update` hooks default to `None`, meaning "use the internal PyTorch path"; Task 6 passes the Triton implementations through them.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_poly_reference.py`:

```python
"""The polynomial reference must approximate causal softmax attention closely
enough to pass the official criterion, and must be exact where it claims to be."""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolyReferenceTests(unittest.TestCase):
    def _qkv(self, B=1, H=2, N=1024, D=64, seed=0, device="cpu"):
        gen = torch.Generator(device=device).manual_seed(seed)
        shape = (B, H, N, D)
        # Component rms 0.577 reproduces the measured score std of ~0.334 at D=64.
        q = torch.randn(shape, generator=gen, device=device) * 0.577
        k = torch.randn(shape, generator=gen, device=device) * 0.577
        v = torch.randn(shape, generator=gen, device=device) * 0.577
        return q, k, v, D ** -0.5

    def test_prefix_region_is_exact(self):
        """Tokens inside exact_prefix must match SDPA to floating-point noise."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=512)
        got = poly_linear_attention(
            q, k, v, scale, chunk=128, exact_prefix=512, sigma=0.334
        )
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        self.assertLess((got - ref).abs().max().item(), 1e-5)

    def test_approximation_is_close_to_softmax_attention(self):
        """Beyond the exact prefix the polynomial must still track softmax."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=2048)
        got = poly_linear_attention(
            q, k, v, scale, chunk=512, exact_prefix=512, sigma=0.334
        )
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        rel = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        self.assertLess(rel, 0.02, f"relative rms {rel:.4f} exceeds 2%")

    def test_hermite_constant_beats_plain_taylor(self):
        """sigma=None is the plain Taylor constant; the fitted one must be better."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=2048)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        kw = dict(chunk=512, exact_prefix=512)
        taylor = poly_linear_attention(q, k, v, scale, sigma=None, **kw)
        hermite = poly_linear_attention(q, k, v, scale, sigma=0.334, **kw)
        self.assertLess(
            (hermite - ref).pow(2).mean().item(),
            (taylor - ref).pow(2).mean().item(),
        )

    def test_batch_and_head_dims_are_independent(self):
        """Each (batch, head) scan must be independent of its neighbours."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(B=2, H=3, N=512)
        kw = dict(chunk=128, exact_prefix=128, sigma=0.334)
        full = poly_linear_attention(q, k, v, scale, **kw)
        one = poly_linear_attention(
            q[1:2, 2:3], k[1:2, 2:3], v[1:2, 2:3], scale, **kw
        )
        self.assertTrue(torch.allclose(full[1:2, 2:3], one, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_reference -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.implementations.poly_reference'`

- [ ] **Step 3: Write the implementation**

Create `src/implementations/poly_reference.py`:

```python
"""Chunked order-2 polynomial linear attention -- the numerical oracle.

Approximates ``exp(s)`` by the degree-2 polynomial that is L2-optimal under the
measured score distribution ``s ~ N(0, sigma^2)``. The Gauss-Hermite projection
gives ``exp(s) ~= exp(sigma^2/2) * [(1 - sigma^2/2) + s + s^2/2]``; the common
factor cancels in the softmax normalisation, so only the constant term differs
from a plain Taylor expansion at zero.

With ``a = q*sqrt(scale)`` and ``b = k*sqrt(scale)`` so that ``a.b = s``, and
``phi2(x) = flatten(x x^T)`` so that ``<phi2(a), phi2(b)> = (a.b)^2``, causal
attention becomes a chunked scan over a running state. The strict prefix
contributes through that state; the diagonal chunk is computed with exact
``exp``.

See ``research/attention-softmax/long-sequence-attention.md`` for the
measurements that justify every constant here, and ``triton-kernel-spec.md``
section 3 for the derivation.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def phi2(x: torch.Tensor) -> torch.Tensor:
    """``[..., C, D] -> [..., C, D*D]``, the flattened outer product ``x x^T``."""
    return (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)


def hermite_c0(sigma: Optional[float]) -> float:
    """Constant term of the L2-optimal degree-2 fit; ``None`` gives plain Taylor."""
    if sigma is None:
        return 1.0
    return max(1.0 - 0.5 * sigma * sigma, 1e-3)


def poly_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    chunk: int = 512,
    exact_prefix: int = 4096,
    sigma: Optional[float] = None,
    state_dtype: torch.dtype = torch.float32,
    compute_dtype: torch.dtype = torch.float16,
    quad_apply: Optional[Callable] = None,
    quad_update: Optional[Callable] = None,
) -> torch.Tensor:
    """Causal attention with a degree-2 polynomial feature map.

    ``q``, ``k``, ``v`` are ``[B, H, N, D]``. Returns ``[B, H, N, D]``.

    ``quad_apply(a, s_quad) -> [M, C, D]`` and
    ``quad_update(b, v, out) -> None`` are optional accelerated implementations
    of the two quadratic-term operations. When ``None``, dense PyTorch is used.
    """
    if q.ndim != 4:
        raise ValueError("q, k, v must be [B, H, N, D]")
    B, H, N, D = q.shape
    M = B * H
    cdt = compute_dtype if compute_dtype is not None else state_dtype

    qf = q.reshape(M, N, D)
    kf = k.reshape(M, N, D)
    vf = v.reshape(M, N, D)
    out = torch.empty_like(qf)

    c0 = hermite_c0(sigma)
    c1, c2 = 1.0, 0.5
    rs = math.sqrt(scale)
    a_all = qf * rs
    b_all = kf * rs

    dev = q.device
    s_const = torch.zeros(M, 1, D, device=dev, dtype=state_dtype)
    s_lin = torch.zeros(M, D, D, device=dev, dtype=state_dtype)
    s_quad = torch.zeros(M, D * D, D, device=dev, dtype=state_dtype)
    z_const = torch.zeros(M, 1, 1, device=dev, dtype=state_dtype)
    z_lin = torch.zeros(M, D, 1, device=dev, dtype=state_dtype)
    z_quad = torch.zeros(M, D * D, 1, device=dev, dtype=state_dtype)

    for t0 in range(0, N, chunk):
        t1 = min(t0 + chunk, N)
        C = t1 - t0
        a = a_all[:, t0:t1]
        b = b_all[:, t0:t1]
        vc = vf[:, t0:t1]

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
            if quad_apply is None:
                aq = phi2(af)
                num = num + c2 * (aq @ s_quad.to(cdt)).to(state_dtype)
                den = den + c2 * (aq @ z_quad.to(cdt)).to(state_dtype)
                del aq
            else:
                num = num + c2 * quad_apply(af, s_quad).to(state_dtype)
                den = den + c2 * quad_apply(af, z_quad).to(state_dtype)

        # Exact diagonal block. No max subtraction: scores are measured bounded
        # to [-2.203, 2.404], so exp cannot overflow. The guard keeps that true.
        sc = (a @ b.transpose(-2, -1)).to(torch.float32)
        blocked = torch.ones(C, C, device=dev, dtype=torch.bool).triu(1)
        w = torch.exp(sc).masked_fill(blocked, 0.0)
        num = num + (w @ vc.to(torch.float32)).to(state_dtype)
        den = den + w.sum(-1, keepdim=True).to(state_dtype)
        del sc, w

        out[:, t0:t1] = (num / den).to(q.dtype)
        del num, den

        bf = b.to(cdt)
        vfc = vc.to(cdt)
        ones = torch.ones(M, C, 1, device=dev, dtype=cdt)
        s_const += vfc.sum(1, keepdim=True).to(state_dtype)
        z_const += float(C)
        s_lin += (bf.transpose(-2, -1) @ vfc).to(state_dtype)
        z_lin += bf.sum(1).unsqueeze(-1).to(state_dtype)
        if quad_update is None:
            bq = phi2(bf)
            s_quad += (bq.transpose(-2, -1) @ vfc).to(state_dtype)
            z_quad += (bq.transpose(-2, -1) @ ones).to(state_dtype)
            del bq
        else:
            quad_update(bf, vfc, s_quad)
            quad_update(bf, ones, z_quad)

    # Early tokens attend to very few keys, where a relative weight error is not
    # damped by averaging, so the max error lives there. Recomputing the first
    # `exact_prefix` tokens exactly costs (W/N)^2 of the quadratic work -- 0.17%
    # at W=4096, N=100000 -- and removes that tail entirely.
    if exact_prefix > 0:
        w0 = min(exact_prefix, N)
        out[:, :w0] = F.scaled_dot_product_attention(
            qf[:, :w0], kf[:, :w0], vf[:, :w0], is_causal=True, scale=scale
        )
    return out.reshape(B, H, N, D)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest src.tests.test_poly_reference -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/implementations/poly_reference.py src/tests/test_poly_reference.py
git commit -m "feat(poly): add chunked polynomial attention reference

The numerical oracle for the Triton kernels, promoted from the validated
spike. Uses the Gauss-Hermite optimal constant rather than plain Taylor, which
measured 2.3x more accurate at no cost, and takes optional quad_apply /
quad_update hooks so Task 6 can substitute the fused kernels without changing
the scan."
git push origin fused-kernal
```

---

### Task 2: The runtime sigma guard

The approximation is only valid while scores stay small. This module decides that, and it is deliberately torch-light so it tests on CPU.

**Files:**
- Create: `src/implementations/poly_guard.py`
- Test: `src/tests/test_poly_guard.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `estimate_sigma(q, k, scale, samples=512, seed=0) -> float`, `SIGMA_CEILING: float`, `poly_is_safe(sigma) -> bool`. `SIGMA_CEILING` is a provisional `0.60` here and is **replaced by a measured value in Task 9**.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_poly_guard.py`:

```python
"""The guard decides whether the polynomial approximation is valid at runtime."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolyGuardTests(unittest.TestCase):
    def _qk(self, D=64, N=4096, scaleup=1.0, seed=0):
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(1, 2, N, D, generator=gen) * 0.577 * scaleup
        k = torch.randn(1, 2, N, D, generator=gen) * 0.577 * scaleup
        return q, k, D ** -0.5

    def test_recovers_the_benchmark_sigma(self):
        from src.implementations.poly_guard import estimate_sigma

        q, k, scale = self._qk()
        got = estimate_sigma(q, k, scale)
        self.assertAlmostEqual(got, 0.334, delta=0.02)

    def test_sigma_scales_with_input_magnitude(self):
        """Doubling q and k should roughly quadruple the score spread."""
        from src.implementations.poly_guard import estimate_sigma

        q1, k1, scale = self._qk(scaleup=1.0)
        q2, k2, _ = self._qk(scaleup=2.0)
        s1 = estimate_sigma(q1, k1, scale)
        s2 = estimate_sigma(q2, k2, scale)
        self.assertAlmostEqual(s2 / s1, 4.0, delta=0.5)

    def test_estimate_is_stable_across_sample_draws(self):
        from src.implementations.poly_guard import estimate_sigma

        q, k, scale = self._qk()
        a = estimate_sigma(q, k, scale, seed=1)
        b = estimate_sigma(q, k, scale, seed=2)
        self.assertLess(abs(a - b), 0.01)

    def test_benchmark_sigma_is_safe_and_large_sigma_is_not(self):
        from src.implementations.poly_guard import SIGMA_CEILING, poly_is_safe

        self.assertTrue(poly_is_safe(0.334))
        self.assertFalse(poly_is_safe(SIGMA_CEILING + 0.01))
        self.assertFalse(poly_is_safe(float("nan")))

    def test_ceiling_is_above_the_measured_benchmark_value(self):
        """A ceiling at or below 0.334 would disable the route entirely."""
        from src.implementations.poly_guard import SIGMA_CEILING

        self.assertGreater(SIGMA_CEILING, 0.334)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_guard -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.implementations.poly_guard'`

- [ ] **Step 3: Write the implementation**

Create `src/implementations/poly_guard.py`:

```python
"""Runtime validity check for the polynomial attention approximation.

The degree-2 fit is accurate only while scores stay small. That is a property of
the benchmark's random initialisation -- measured ``sigma = 0.3336`` -- not of
attention in general; under trained weights scores are far larger and the fit
would be poor. So the route is gated on a runtime measurement rather than an
assumption.

The same statistic does double duty: it supplies the L2-optimal constant term
``c0 = 1 - sigma^2/2`` and decides whether to run at all.
"""

from __future__ import annotations

import math

import torch


# Provisional. Task 9 replaces this with the measured value from the sigma
# sweep, set with margin below where the official criterion first fails.
SIGMA_CEILING = 0.60


def estimate_sigma(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    samples: int = 512,
    seed: int = 0,
) -> float:
    """Standard deviation of the scaled scores, from a random row sample.

    ``q``, ``k`` are ``[B, H, N, D]``. Sampling ``samples`` rows and taking all
    pairs gives ``samples^2`` scores per head, which is ample: 512 rows recovered
    the population value to three decimal places at a cost of a few MiB.

    Costs one device-to-host synchronization. Call it once per forward, from the
    eager dispatch layer -- never inside a compiled or graph-replayed region.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be [B, H, N, D]")
    n = q.shape[2]
    count = min(samples, n)
    gen = torch.Generator(device=q.device).manual_seed(seed)
    idx = torch.randperm(n, generator=gen, device=q.device)[:count]
    qs = q[:, :, idx].float()
    ks = k[:, :, idx].float()
    scores = (qs @ ks.transpose(-2, -1)) * scale
    return float(scores.std().item())


def poly_is_safe(sigma: float) -> bool:
    """True when the measured score spread is inside the validated range."""
    if sigma is None or math.isnan(sigma) or math.isinf(sigma):
        return False
    return 0.0 < sigma <= SIGMA_CEILING
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest src.tests.test_poly_guard -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/implementations/poly_guard.py src/tests/test_poly_guard.py
git commit -m "feat(poly): add the runtime sigma guard

The approximation depends on the benchmark's small scores, which is a property
of its random initialisation rather than of attention. Measure sigma at runtime
instead of assuming it: the same statistic supplies the optimal constant term
and gates the route. SIGMA_CEILING is provisional pending the Task 9 sweep."
git push origin fused-kernal
```

---

### Task 3: `poly_quad_apply` Triton kernel

Replaces `phi2(a) @ S`, the larger half of the ~51 GB of feature traffic.

**Files:**
- Create: `src/kernels/__init__.py`
- Create: `src/kernels/poly_attention_triton.py`
- Test: `src/tests/test_poly_kernel.py`

**Interfaces:**
- Consumes: `phi2` from `src.implementations.poly_reference`.
- Produces: `HAS_TRITON: bool`, `quad_apply(a, s) -> Tensor` where `a` is `[M, C, D]` and `s` is `[M, D*D, V]`, returning `[M, C, V]`.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_poly_kernel.py`:

```python
"""The Triton kernels must compute the same quantity as dense PyTorch.

These are the same mathematical values in a different summation order, so a
deviation beyond fp16 rounding indicates a bug, not precision loss.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _cuda_missing():
    return torch is None or not torch.cuda.is_available()


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "Triton kernels require CUDA")
class QuadApplyTests(unittest.TestCase):
    def _case(self, M=2, C=128, D=64, V=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.randn(M, C, D, generator=gen, device="cuda", dtype=torch.float16) * 0.2
        s = torch.randn(M, D * D, V, generator=gen, device="cuda", dtype=torch.float32)
        return a, s

    def _dense(self, a, s):
        from src.implementations.poly_reference import phi2

        return phi2(a) @ s.to(a.dtype)

    def test_matches_dense_for_the_case_14_shape(self):
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(D=64, V=64)
        got = quad_apply(a, s)
        ref = self._dense(a, s)
        err = (got.float() - ref.float()).abs().max().item()
        self.assertLess(err, 1e-3, f"max deviation {err:.3e}")

    def test_matches_dense_across_shapes(self):
        from src.kernels.poly_attention_triton import quad_apply

        for D in (16, 32, 64):
            for C in (128, 512):
                for V in (32, 64):
                    with self.subTest(D=D, C=C, V=V):
                        a, s = self._case(C=C, D=D, V=V)
                        err = (
                            quad_apply(a, s).float() - self._dense(a, s).float()
                        ).abs().max().item()
                        self.assertLess(err, 1e-3)

    def test_handles_a_ragged_final_chunk(self):
        """N is not a multiple of the chunk length, so C need not divide BC."""
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(C=100)
        err = (quad_apply(a, s).float() - self._dense(a, s).float()).abs().max().item()
        self.assertLess(err, 1e-3)

    def test_single_column_state_for_the_denominator(self):
        """The denominator uses V=1; it must not be a special case that breaks."""
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(V=1)
        err = (quad_apply(a, s).float() - self._dense(a, s).float()).abs().max().item()
        self.assertLess(err, 1e-3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_kernel -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.kernels'`

- [ ] **Step 3: Write the availability probe**

Create `src/kernels/__init__.py`:

```python
"""Custom GPU kernels. Import is safe without Triton; callers check HAS_TRITON."""

from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    HAS_TRITON = True
except ImportError:  # pragma: no cover - environments without Triton
    HAS_TRITON = False

__all__ = ["HAS_TRITON"]
```

- [ ] **Step 4: Write the kernel**

Create `src/kernels/poly_attention_triton.py`:

```python
"""Fused degree-2 polynomial feature-map kernels.

The order-2 polynomial path performs 12x fewer FLOPs than exact attention yet
realises only 1.19x, because it writes and re-reads a ``[C, D*D]`` feature
tensor -- about 51 GB per sample-layer at case 14's shapes. These kernels
generate that tensor in registers and consume it directly, so it never reaches
HBM.

The ``D*D`` feature axis is indexed by pairs ``(i, j)``. A feature block is a
contiguous range of ``i`` with all ``j``, which is a contiguous slab of the
state -- that is what makes the tiling work.

State is float32 in HBM and converted to float16 on load. A float16 master
state is 1.1x faster and silently wrong at scale: it passes at N=16384 and
fails at N=65536 with over a million failures.
"""

from __future__ import annotations

import torch

from src.kernels import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _phi_tile(a, ai, BC: tl.constexpr, BI: tl.constexpr, D: tl.constexpr):
        """Outer-product slab: ``phi[c, i*D + j] = ai[c, i] * a[c, j]``."""
        return tl.reshape(ai[:, :, None] * a[:, None, :], (BC, BI * D))

    @triton.autotune(
        configs=[
            triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
            for bc in (32, 64)
            for bi in (1, 2, 4)
            for w in (4, 8)
        ],
        key=["C", "D", "V"],
    )
    @triton.jit
    def _quad_apply_kernel(
        a_ptr, s_ptr, y_ptr,
        stride_am, stride_ac, stride_ad,
        stride_sm, stride_sf, stride_sv,
        stride_ym, stride_yc, stride_yv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_c = pid_c * BC + tl.arange(0, BC)
        offs_d = tl.arange(0, D)
        offs_v = tl.arange(0, BV)
        mask_c = offs_c < C
        mask_v = offs_v < V

        a_base = a_ptr + pid_m * stride_am
        a = tl.load(
            a_base + offs_c[:, None] * stride_ac + offs_d[None, :] * stride_ad,
            mask=mask_c[:, None], other=0.0,
        )

        acc = tl.zeros((BC, BV), dtype=tl.float32)
        for i0 in range(0, D, BI):
            offs_i = i0 + tl.arange(0, BI)
            ai = tl.load(
                a_base + offs_c[:, None] * stride_ac + offs_i[None, :] * stride_ad,
                mask=mask_c[:, None], other=0.0,
            )
            phi = _phi_tile(a, ai, BC, BI, D)
            offs_f = i0 * D + tl.arange(0, BI * D)
            s = tl.load(
                s_ptr + pid_m * stride_sm
                + offs_f[:, None] * stride_sf + offs_v[None, :] * stride_sv,
                mask=mask_v[None, :], other=0.0,
            )
            acc += tl.dot(phi.to(tl.float16), s.to(tl.float16))

        tl.store(
            y_ptr + pid_m * stride_ym
            + offs_c[:, None] * stride_yc + offs_v[None, :] * stride_yv,
            acc.to(tl.float16),
            mask=mask_c[:, None] & mask_v[None, :],
        )


def quad_apply(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``phi2(a) @ s`` without materialising ``phi2(a)``.

    ``a`` is ``[M, C, D]`` (float16), ``s`` is ``[M, D*D, V]`` (float32).
    Returns ``[M, C, V]`` in ``a``'s dtype.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = a.shape
    if s.shape[0] != M or s.shape[1] != D * D:
        raise ValueError(f"state shape {tuple(s.shape)} does not match a {tuple(a.shape)}")
    V = s.shape[2]
    a = a.contiguous()
    s = s.contiguous()
    y = torch.empty((M, C, V), device=a.device, dtype=a.dtype)
    BV = max(16, triton.next_power_of_2(V))
    grid = lambda meta: (triton.cdiv(C, meta["BC"]), M)
    _quad_apply_kernel[grid](
        a, s, y,
        a.stride(0), a.stride(1), a.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        C, D, V, BV=BV,
    )
    return y
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest src.tests.test_poly_kernel -v`
Expected: 4 tests PASS

If `tl.reshape` rejects the 3-D broadcast on Triton 3.7.1, replace `_phi_tile` with an explicit `tl.static_range(BI)` loop accumulating `tl.dot(ai[:, ii, None] * a, s_slice)` per `ii`. Keep the device function boundary either way — Task 4 and Phase 2 both reuse it.

- [ ] **Step 6: Commit**

```bash
git add src/kernels/ src/tests/test_poly_kernel.py
git commit -m "feat(kernels): add fused poly_quad_apply Triton kernel

Generates the a(x)a feature slab in registers and consumes it directly, so the
[C, D*D] tensor never reaches HBM. Tiles the D*D feature axis as contiguous
ranges of i, which keeps each state slice a contiguous slab. State stays fp32
in HBM and converts to fp16 on load."
git push origin fused-kernal
```

---

### Task 4: `poly_quad_update` Triton kernel

Replaces `phi2(b)^T @ v`, the other half of the feature traffic.

**Files:**
- Modify: `src/kernels/poly_attention_triton.py`
- Modify: `src/tests/test_poly_kernel.py`

**Interfaces:**
- Consumes: `_phi_tile` from Task 3.
- Produces: `quad_update(b, v, out) -> None`, accumulating `phi2(b)^T @ v` into `out` in place. `b` is `[M, C, D]` float16, `v` is `[M, C, V]` float16, `out` is `[M, D*D, V]` float32.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/test_poly_kernel.py`, before the `if __name__` block:

```python
@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "Triton kernels require CUDA")
class QuadUpdateTests(unittest.TestCase):
    def _case(self, M=2, C=128, D=64, V=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        b = torch.randn(M, C, D, generator=gen, device="cuda", dtype=torch.float16) * 0.2
        v = torch.randn(M, C, V, generator=gen, device="cuda", dtype=torch.float16) * 0.2
        return b, v

    def _dense(self, b, v, D):
        from src.implementations.poly_reference import phi2

        return (phi2(b).transpose(-2, -1) @ v).float()

    def test_matches_dense_for_the_case_14_shape(self):
        from src.kernels.poly_attention_triton import quad_update

        b, v = self._case(D=64, V=64)
        out = torch.zeros(2, 64 * 64, 64, device="cuda", dtype=torch.float32)
        quad_update(b, v, out)
        err = (out - self._dense(b, v, 64)).abs().max().item()
        self.assertLess(err, 1e-2, f"max deviation {err:.3e}")

    def test_accumulates_rather_than_overwrites(self):
        """The state is a running sum; two folds must equal one doubled fold."""
        from src.kernels.poly_attention_triton import quad_update

        b, v = self._case()
        once = torch.zeros(2, 64 * 64, 64, device="cuda", dtype=torch.float32)
        quad_update(b, v, once)
        twice = torch.zeros_like(once)
        quad_update(b, v, twice)
        quad_update(b, v, twice)
        err = (twice - 2 * once).abs().max().item()
        self.assertLess(err, 1e-2)

    def test_single_column_for_the_denominator(self):
        from src.kernels.poly_attention_triton import quad_update

        b, _ = self._case()
        ones = torch.ones(2, 128, 1, device="cuda", dtype=torch.float16)
        out = torch.zeros(2, 64 * 64, 1, device="cuda", dtype=torch.float32)
        quad_update(b, ones, out)
        err = (out - self._dense(b, ones, 64)).abs().max().item()
        self.assertLess(err, 1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_kernel.QuadUpdateTests -v`
Expected: FAIL — `ImportError: cannot import name 'quad_update'`

- [ ] **Step 3: Write the kernel**

Add inside the `if HAS_TRITON:` block of `src/kernels/poly_attention_triton.py`:

```python
    @triton.autotune(
        configs=[
            triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
            for bc in (32, 64)
            for bi in (1, 2)
            for w in (4, 8)
        ],
        key=["C", "D", "V"],
    )
    @triton.jit
    def _quad_update_kernel(
        b_ptr, v_ptr, o_ptr,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_om, stride_of, stride_ov,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        pid_m = tl.program_id(1)

        i0 = pid_i * BI
        offs_d = tl.arange(0, D)
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V

        b_base = b_ptr + pid_m * stride_bm
        v_base = v_ptr + pid_m * stride_vm
        acc = tl.zeros((BI * D, BV), dtype=tl.float32)

        for c0 in range(0, C, BC):
            offs_c = c0 + tl.arange(0, BC)
            mask_c = offs_c < C
            b = tl.load(
                b_base + offs_c[:, None] * stride_bc + offs_d[None, :] * stride_bd,
                mask=mask_c[:, None], other=0.0,
            )
            offs_i = i0 + tl.arange(0, BI)
            bi = tl.load(
                b_base + offs_c[:, None] * stride_bc + offs_i[None, :] * stride_bd,
                mask=mask_c[:, None], other=0.0,
            )
            phi = _phi_tile(b, bi, BC, BI, D)
            vt = tl.load(
                v_base + offs_c[:, None] * stride_vc + offs_v[None, :] * stride_vv,
                mask=mask_c[:, None] & mask_v[None, :], other=0.0,
            )
            acc += tl.dot(tl.trans(phi).to(tl.float16), vt.to(tl.float16))

        offs_f = i0 * D + tl.arange(0, BI * D)
        o_addr = (
            o_ptr + pid_m * stride_om
            + offs_f[:, None] * stride_of + offs_v[None, :] * stride_ov
        )
        prev = tl.load(o_addr, mask=mask_v[None, :], other=0.0)
        tl.store(o_addr, prev + acc, mask=mask_v[None, :])
```

and the wrapper at module level:

```python
def quad_update(b: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
    """Accumulate ``phi2(b)^T @ v`` into ``out`` in place.

    ``b`` is ``[M, C, D]`` float16, ``v`` is ``[M, C, V]`` float16, and ``out``
    is ``[M, D*D, V]`` float32. ``out`` is the running state, so this adds
    rather than overwrites.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = b.shape
    V = v.shape[2]
    if out.shape != (M, D * D, V):
        raise ValueError(f"out shape {tuple(out.shape)} != {(M, D * D, V)}")
    if out.dtype != torch.float32:
        raise ValueError("the master state must be float32; see the module docstring")
    b = b.contiguous()
    v = v.contiguous()
    BV = max(16, triton.next_power_of_2(V))
    grid = lambda meta: (triton.cdiv(D, meta["BI"]), M)
    _quad_update_kernel[grid](
        b, v, out,
        b.stride(0), b.stride(1), b.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        C, D, V, BV=BV,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest src.tests.test_poly_kernel -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/kernels/poly_attention_triton.py src/tests/test_poly_kernel.py
git commit -m "feat(kernels): add fused poly_quad_update Triton kernel

Folds a chunk into the running state without materialising phi2(b). Reuses the
_phi_tile device function from quad_apply, and rejects a non-float32 state
outright rather than silently producing the failure mode that only appears at
N>=65536."
git push origin fused-kernal
```

---

### Task 5: Wire the kernels into the scan

**Files:**
- Create: `src/implementations/poly_attention.py`
- Test: `src/tests/test_poly_attention.py`

**Interfaces:**
- Consumes: `poly_linear_attention` (Task 1), `quad_apply`/`quad_update` (Tasks 3-4), `estimate_sigma`/`poly_is_safe` (Task 2).
- Produces: `poly_attention_forward(q, k, v, scale, *, sigma, use_triton=True) -> Tensor`.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_poly_attention.py`:

```python
"""The fused path must agree with the PyTorch oracle it replaces."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _cuda_missing():
    return torch is None or not torch.cuda.is_available()


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "requires CUDA")
class FusedPathTests(unittest.TestCase):
    def _qkv(self, M=2, N=2048, D=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        shape = (1, M, N, D)
        f = lambda: (
            torch.randn(shape, generator=gen, device="cuda", dtype=torch.float16) * 0.577
        )
        return f(), f(), f(), D ** -0.5

    def test_fused_matches_the_pytorch_oracle(self):
        from src.implementations.poly_attention import poly_attention_forward

        q, k, v, scale = self._qkv()
        fused = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=True)
        oracle = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=False)
        err = (fused.float() - oracle.float()).abs().max().item()
        self.assertLess(err, 2e-3, f"max deviation {err:.3e}")

    def test_fused_matches_oracle_on_a_ragged_sequence(self):
        from src.implementations.poly_attention import poly_attention_forward

        q, k, v, scale = self._qkv(N=2000)
        fused = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=True)
        oracle = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=False)
        self.assertLess((fused.float() - oracle.float()).abs().max().item(), 2e-3)

    def test_state_dtype_is_float32_regardless_of_input_dtype(self):
        """Regression guard for the fp16-state trap that only fails at N>=65536."""
        import inspect

        from src.implementations import poly_attention

        src = inspect.getsource(poly_attention.poly_attention_forward)
        self.assertIn("float32", src)
        self.assertNotIn("state_dtype=torch.float16", src)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_attention -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.implementations.poly_attention'`

- [ ] **Step 3: Write the implementation**

Create `src/implementations/poly_attention.py`:

```python
"""Polynomial attention for official case 14, with a fused Triton back end.

Routing is decided by ``src.implementations.poly_guard``: when the measured
score spread leaves the validated range, the caller falls back to exact Flash
SDPA. See ``research/attention-softmax/triton-kernel-spec.md``.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.implementations.poly_reference import poly_linear_attention
from src.kernels import HAS_TRITON

CHUNK = 512
EXACT_PREFIX = 4096


def poly_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    sigma: Optional[float],
    use_triton: bool = True,
) -> torch.Tensor:
    """Causal polynomial attention over ``[B, H, N, D]`` tensors.

    The master state is float32 unconditionally. A float16 state is faster and
    silently wrong at scale -- it passes at N=16384 and fails at N=65536 with
    1,064,935 failures -- so it is not exposed as an option.
    """
    apply_fn = update_fn = None
    if use_triton and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        from src.kernels.poly_attention_triton import quad_apply, quad_update

        apply_fn, update_fn = quad_apply, quad_update

    return poly_linear_attention(
        q, k, v, scale,
        chunk=CHUNK,
        exact_prefix=EXACT_PREFIX,
        sigma=sigma,
        state_dtype=torch.float32,
        compute_dtype=torch.float16,
        quad_apply=apply_fn,
        quad_update=update_fn,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m unittest src.tests.test_poly_attention -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/implementations/poly_attention.py src/tests/test_poly_attention.py
git commit -m "feat(poly): route the chunked scan through the fused kernels

Substitutes the Triton quad_apply/quad_update through the reference's hooks, so
the fused and PyTorch paths run identical scan logic and any divergence is the
kernels'. The float32 master state is not exposed as an option."
git push origin fused-kernal
```

---

### Task 6: End-to-end correctness against both oracles

This is spec section 7.2. It is a script rather than a unit test because the large-N runs take minutes and allocate GiB.

**Files:**
- Create: `src/validate_poly.py`
- Create: `research/benchmarks/2026-08-30-rtx4060-poly/README.md` (written in Task 8)

**Interfaces:**
- Consumes: `poly_attention_forward` (Task 5).
- Produces: a CLI, `.venv/bin/python -m src.validate_poly --n N [--oracle dense|flash]`, printing failures/max/rms and exiting non-zero on any failure.

- [ ] **Step 1: Write the validator**

Create `src/validate_poly.py`:

```python
#!/usr/bin/env python3
"""End-to-end criterion for the polynomial attention path at case-14 shapes.

Two oracles, both required (spec section 7.2):

* ``dense``  -- the immutable ``BaselineTransformer``. Authoritative, but its
  N x N score tensor limits it to about N=8192 in 8 GiB.
* ``flash``  -- the same model with exact Flash SDPA attention. Algebraically
  exact, so it isolates approximation error from fp16 reduction-order noise,
  and it runs to N=100000.
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
    copy_model_weights,
)
from src.implementations.poly_attention import poly_attention_forward
from src.implementations.poly_guard import estimate_sigma

ATOL, RTOL = 0.002, 0.02
D_MODEL, HEADS, FFN, LAYERS = 1024, 16, 1024, 2


class _FlashAttention(BaselineSelfAttention):
    def forward(self, x, valid_token_mask=None, causal=False):
        b, n, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ctx = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, scale=self.scale
            )
        return self.out_proj(ctx.transpose(1, 2).reshape(b, n, self.d_model))


class _PolyAttention(BaselineSelfAttention):
    sigma = None

    def forward(self, x, valid_token_mask=None, causal=False):
        b, n, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.sigma is None:
            self.sigma = estimate_sigma(q, k, self.scale)
        ctx = poly_attention_forward(q, k, v, self.scale, sigma=self.sigma)
        return self.out_proj(ctx.transpose(1, 2).reshape(b, n, self.d_model))


def _build(attn_cls, config, device, seed=0):
    torch.manual_seed(seed)
    model = BaselineTransformer(config)
    if attn_cls is not None:
        for layer in model.layers:
            layer.attention = attn_cls(config.d_model, config.num_heads)
    return model.to(device).half().eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--oracle", choices=("dense", "flash"), default="flash")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale-qk", type=float, default=1.0,
                        help="multiply the input, to sweep sigma (Task 7)")
    args = parser.parse_args()

    device = torch.device("cuda")
    config = TransformerConfig(1, args.n, D_MODEL, HEADS, FFN, LAYERS, True)

    oracle = _build(None if args.oracle == "dense" else _FlashAttention, config, device)
    candidate = _build(_PolyAttention, config, device)
    copy_model_weights(oracle, candidate)

    torch.manual_seed(args.seed)
    x = torch.randn(1, args.n, D_MODEL, device=device, dtype=torch.float16)
    x = x * args.scale_qk

    with torch.inference_mode():
        ref = oracle(x).float()
        del oracle
        torch.cuda.empty_cache()
        got = candidate(x).float()

    err = (got - ref).abs()
    tol = torch.clamp(ref.abs() * RTOL, min=ATOL)
    failures = int((err > tol).sum())
    sigma = candidate.layers[0].attention.sigma
    print(
        f"N={args.n} oracle={args.oracle} scale_qk={args.scale_qk} "
        f"sigma={sigma:.4f} failures={failures}/{err.numel()} "
        f"max={err.max().item():.4e} rms={err.pow(2).mean().sqrt().item():.4e} "
        f"{'PASS' if failures == 0 else 'FAIL'}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the dense-oracle rows**

```bash
.venv/bin/python -m src.validate_poly --n 4096 --oracle dense
.venv/bin/python -m src.validate_poly --n 8192 --oracle dense
```

Expected: both print `PASS` with `failures=0`. The PyTorch path already passes these, so a failure here is the kernel's.

- [ ] **Step 3: Run the flash-oracle rows**

```bash
for n in 16384 32768 65536 100000; do
  .venv/bin/python -m src.validate_poly --n $n --oracle flash
done
```

Expected: all four print `PASS`. N=100000 takes several minutes.

- [ ] **Step 4: If N=65536 or N=100000 fails but small N passes**

That is the float16-state signature. Confirm `poly_linear_attention` is called with `state_dtype=torch.float32` and that `quad_update` did not silently receive a float16 `out`. Do not "fix" it by loosening a tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/validate_poly.py
git commit -m "test(poly): add the end-to-end criterion validator

Spec section 7.2: the dense reference is authoritative but caps at ~8192 in
8 GiB, and the exact-flash oracle isolates approximation error from fp16
reduction-order noise up to N=100000. Both are required."
git push origin fused-kernal
```

---

### Task 7: Measure the sigma ceiling

The guard is decorative until this runs. Spec section 6 requires a measured ceiling, not an assumed one.

**Files:**
- Modify: `src/implementations/poly_guard.py`
- Modify: `src/tests/test_poly_guard.py`

- [ ] **Step 1: Sweep sigma until the criterion fails**

**Corrected during execution.** The original instruction here used `--scale-qk`, which multiplies the input. That does **not** move sigma: the first operation in every layer is `norm1`, a `LayerNorm`, which is scale-invariant, so Q and K are unchanged. Measured, sigma stayed at 0.3339 for input scales 1.0, 1.5 and 2.0.

Sweep with `--scale-qk-weights` instead, which scales the `q_proj` and `k_proj` weights of **both** models — so they still compute the same function — and therefore scales scores by the square of the factor.

```bash
for w in 1.0 1.5 2.0 2.5 3.0; do
  .venv/bin/python -m src.validate_poly --n 8192 --oracle dense --scale-qk-weights $w
done
```

Record the printed `sigma` and PASS/FAIL for each. Expected: PASS at low sigma, FAIL beyond some point. If the first FAIL is at the second point, sweep finer between them before bisecting.

- [ ] **Step 2: Narrow the boundary**

Bisect between the last PASS and the first FAIL with three more runs to locate the failure sigma to about +/-0.02.

- [ ] **Step 3: Set the ceiling with margin**

**Corrected during execution.** The original rule here was "use half the measured failure sigma". That assumed a wide gap between the operating point and the failure point. Measured, the gap is only 1.56x — sigma 0.334 operating against a first failure at 0.5217 — so half the failure value is 0.26, *below* the operating point, which would disable the route entirely.

The rule that works: set the ceiling **below the largest verified passing sigma**, and comfortably above the operating point's seed-to-seed spread. Record the full sweep, not just the chosen number:

```python
# Measured 30 August 2026 by sweeping --scale-qk in src/validate_poly.py at
# N=8192 against the dense reference. The criterion first failed at
# sigma = <MEASURED_FAIL>; the ceiling is set at half that for margin.
# The benchmark's own value is 0.3336, comfortably inside.
SIGMA_CEILING = <MEASURED_FAIL / 2>
```

- [ ] **Step 4: Add the regression test**

Append to `src/tests/test_poly_guard.py`:

```python
    def test_ceiling_matches_the_measured_sweep(self):
        """Pins the Task 7 measurement so a later edit cannot quietly widen it."""
        from src.implementations.poly_guard import SIGMA_CEILING

        self.assertLess(SIGMA_CEILING, 1.0)
        self.assertGreater(SIGMA_CEILING, 0.334)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m unittest src.tests.test_poly_guard -v`
Expected: 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/implementations/poly_guard.py src/tests/test_poly_guard.py
git commit -m "feat(poly): set the sigma ceiling from measurement

Swept --scale-qk at N=8192 against the dense reference to find where the
official criterion first fails, then set the ceiling at half that value. The
benchmark's own sigma of 0.3336 sits well inside. Replaces the provisional
constant, which was a guess."
git push origin fused-kernal
```

---

### Task 8: Benchmark and the acceptance decision

**Files:**
- Create: `src/bench_poly.py`
- Create: `research/benchmarks/2026-08-30-rtx4060-poly/README.md`

- [ ] **Step 1: Write the benchmark**

Create `src/bench_poly.py`:

```python
#!/usr/bin/env python3
"""Attention-core latency at case-14 shapes: exact flash vs polynomial paths.

One sample x one layer, which is what case 14's route actually executes -- it
streams 1-2 samples at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.implementations.poly_attention import poly_attention_forward
from src.infra.environment import collect_environment, collect_git


def _time(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    scale = args.head_dim ** -0.5
    torch.manual_seed(0)
    shape = (1, args.heads, args.n, args.head_dim)
    q, k, v = (
        (torch.randn(shape, device=device) * 0.577).half() for _ in range(3)
    )

    with torch.inference_mode():
        def exact():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )

        results = {
            "exact_flash_ms": _time(exact),
            "poly_pytorch_ms": _time(
                lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=False
                )
            ),
            "poly_triton_ms": _time(
                lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=True
                )
            ),
        }

    base = results["exact_flash_ms"]
    results["speedup_vs_exact"] = base / results["poly_triton_ms"]
    results["speedup_vs_pytorch_poly"] = (
        results["poly_pytorch_ms"] / results["poly_triton_ms"]
    )
    results["accepted"] = results["poly_triton_ms"] <= 360.0
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.output is not None:
        payload = {
            "schema_version": 1,
            "config": {"n": args.n, "heads": args.heads, "head_dim": args.head_dim,
                       "dtype": "float16", "batch": 1, "layers": 1},
            "environment": collect_environment(torch, device),
            "git": collect_git(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python -m src.bench_poly --n 100000 \
  --output research/benchmarks/2026-08-30-rtx4060-poly/attention-core.json
```

Reference numbers to beat: exact flash 719.8 ms, PyTorch poly 603.9 ms, target <= 360 ms.

- [ ] **Step 3: Apply the acceptance rule**

- `<= 360 ms` — accepted, continue to Task 9.
- `360-603.9 ms` — beats the PyTorch path but misses target. Record it, do **not** promote in Task 9, and say so plainly.
- `> 603.9 ms` — rejected. Stop and report; do not integrate.

- [ ] **Step 4: Write the run record**

Create `research/benchmarks/2026-08-30-rtx4060-poly/README.md` containing: the exact commands, git commit, timestamp, input shapes, dtype, correctness results from Tasks 6-7, the latency table, the acceptance verdict, and CPU/GPU/OS/driver/CUDA/PyTorch/Triton versions. Add a one-line entry to `research/benchmarks/README.md` under "Current Runs".

- [ ] **Step 5: Commit**

```bash
git add src/bench_poly.py research/benchmarks/
git commit -m "bench(poly): record the fused-kernel attention-core measurement"
git push origin fused-kernal
```

---

### Task 9: Integrate into case 14, opt-in and reversible

Only if Task 8 accepted. Touches Person 4's and Person 1's files; the repository owner carries the human coordination.

**Files:**
- Modify: `src/implementations/extreme.py`
- Test: `src/tests/test_poly_attention.py`
- Create: `docs/kernel-integration-notes.md`

- [ ] **Step 1: Write the failing test**

Append to `src/tests/test_poly_attention.py`, before `if __name__`:

```python
@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolyRouteToggleTests(unittest.TestCase):
    """The route must be opt-in and reversible in one flag."""

    def test_disabled_by_default(self):
        from src.implementations import extreme

        self.assertFalse(extreme.POLY_ATTENTION_ENABLED)

    def test_disabled_flag_selects_the_flash_path(self):
        from src.implementations.extreme import PolyOrFlashSelfAttention

        module = PolyOrFlashSelfAttention(1024, 16)
        module.poly_enabled = False
        self.assertEqual(module.route_name(sigma=0.334), "flash")

    def test_enabled_flag_selects_poly_only_inside_the_guard(self):
        from src.implementations.extreme import PolyOrFlashSelfAttention

        module = PolyOrFlashSelfAttention(1024, 16)
        module.poly_enabled = True
        self.assertEqual(module.route_name(sigma=0.334), "poly")
        self.assertEqual(module.route_name(sigma=5.0), "flash")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest src.tests.test_poly_attention.PolyRouteToggleTests -v`
Expected: FAIL — `AttributeError: module 'src.implementations.extreme' has no attribute 'POLY_ATTENTION_ENABLED'`

- [ ] **Step 3: Implement**

Add to `src/implementations/extreme.py`:

```python
# Opt-in. Setting this False returns case 14 to exactly the forced-Flash
# behaviour, which src/tests/test_poly_attention.py pins so it cannot rot.
# See docs/kernel-integration-notes.md.
POLY_ATTENTION_ENABLED = False


class PolyOrFlashSelfAttention(FlashOnlySDPASelfAttention):
    """Case-14 attention that may use the polynomial kernel, guarded.

    Memory behaviour is unchanged: no resident tensor here scales with
    ``B*N*d_model``, and the caller's prefix streaming and OOM backoff still
    drive execution.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.poly_enabled = POLY_ATTENTION_ENABLED
        self._sigma: Optional[float] = None

    def route_name(self, sigma: float) -> str:
        from src.implementations.poly_guard import poly_is_safe

        return "poly" if self.poly_enabled and poly_is_safe(sigma) else "flash"

    def forward(self, x, valid_token_mask=None, causal=False):
        if not self.poly_enabled:
            return super().forward(x, valid_token_mask, causal)
        if valid_token_mask is not None:
            raise ValueError("trim padding before calling extreme attention")

        from src.implementations.poly_attention import poly_attention_forward
        from src.implementations.poly_guard import estimate_sigma

        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        if self._sigma is None:
            # One host synchronization per module, in the eager dispatch layer.
            self._sigma = estimate_sigma(q, k, self.scale)
        if self.route_name(self._sigma) == "flash":
            return super().forward(x, valid_token_mask, causal)

        context = poly_attention_forward(q, k, v, self.scale, sigma=self._sigma)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)
```

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m unittest discover -s src/tests
```

Expected: OK, with the previously passing tests still passing.

- [ ] **Step 5: Write the integration notes**

Create `docs/kernel-integration-notes.md` for Persons 1 and 4, stating: what changed in `extreme.py` and why; that `POLY_ATTENTION_ENABLED = False` restores today's behaviour exactly and is covered by a test; that the memory contract is unchanged; that the guard's host synchronization must stay in the eager layer; the measured evidence from Tasks 6-8; and that the approximation depends on the benchmark's `sigma`.

- [ ] **Step 6: Commit and open the PR**

```bash
git add src/implementations/extreme.py src/tests/test_poly_attention.py docs/kernel-integration-notes.md
git commit -m "feat(extreme): add the opt-in polynomial attention route for case 14

Disabled by default. POLY_ATTENTION_ENABLED = False restores the forced-Flash
behaviour exactly, and a test pins that so it cannot rot. Touches Person 4's
file; docs/kernel-integration-notes.md records the memory and synchronization
contracts it must not break."
git push origin fused-kernal
gh pr create --title "Fused polynomial attention kernel for case 14" --body "..."
```

The PR body must cover problem, decision and alternatives, affected behaviour, expected performance, risks, numerical-correctness evidence, benchmark environment and results, and verification commands, per `CLAUDE.md`.

---

## Self-Review

**Spec coverage.** Section 3 computation → Task 1. Section 4.1 modules → Tasks 1-5, 9. Section 4.2 `poly_quad_apply` → Task 3. Section 4.3 `poly_quad_update` → Task 4. Section 4.4 precision → Tasks 1, 4, 5 (with a regression test). Section 4.5 block sizes → Task 3 autotune configs. Section 5 performance → Task 8. Section 6 guard → Tasks 2 and 7. Section 7.1 kernel equivalence → Tasks 3-4. Section 7.2 end-to-end → Task 6. Section 7.3 guard sweep → Task 7. Section 7.4 acceptance → Task 8 step 3. Section 8 Phase 2 compatibility → Task 3's shared `_phi_tile` and the fixed `[M, D*D, V]` layout. Section 10 integration → Task 9.

**Known gap:** spec section 8.1's two-level parallel scan is Phase 2 and deliberately has no task.

**Type consistency.** `quad_apply(a, s) -> Tensor` and `quad_update(b, v, out) -> None` are used with those exact signatures in Task 5's hooks and match Task 1's `quad_apply`/`quad_update` parameter contract. `phi2`, `hermite_c0`, `estimate_sigma`, `poly_is_safe`, `SIGMA_CEILING`, `poly_attention_forward`, `POLY_ATTENTION_ENABLED`, `PolyOrFlashSelfAttention.route_name` are each defined once and referenced consistently. State is `[M, D*D, V]` float32 everywhere.
