"""The polynomial reference must approximate causal softmax attention closely
enough to pass the official criterion, and must be exact where it claims to be.

These run on CPU, so they pass ``compute_dtype=torch.float32``: CPU float16
matmul is slow to unsupported, and the compute dtype is orthogonal to the
properties under test here. The float16 compute path is covered on CUDA by
``test_poly_attention.py``.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolyReferenceTests(unittest.TestCase):
    CPU_KW = dict(state_dtype=torch.float32, compute_dtype=torch.float32) if torch else {}

    def _qkv(self, B=1, H=2, N=1024, D=64, seed=0):
        gen = torch.Generator().manual_seed(seed)
        shape = (B, H, N, D)
        # Component rms 0.577 reproduces the measured score std of ~0.334 at D=64.
        q = torch.randn(shape, generator=gen) * 0.577
        k = torch.randn(shape, generator=gen) * 0.577
        v = torch.randn(shape, generator=gen) * 0.577
        return q, k, v, D ** -0.5

    def test_prefix_region_is_exact(self):
        """Tokens inside exact_prefix must match SDPA to floating-point noise."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=512)
        got = poly_linear_attention(
            q, k, v, scale, chunk=128, exact_prefix=512, sigma=0.334, **self.CPU_KW
        )
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        self.assertLess((got - ref).abs().max().item(), 1e-5)

    def test_approximation_is_close_to_softmax_attention(self):
        """Beyond the exact prefix the polynomial must still track softmax."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=2048)
        got = poly_linear_attention(
            q, k, v, scale, chunk=512, exact_prefix=512, sigma=0.334, **self.CPU_KW
        )
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        rel = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        self.assertLess(rel, 0.02, f"relative rms {rel:.4f} exceeds 2%")

    def test_hermite_constant_beats_plain_taylor(self):
        """sigma=None is the plain Taylor constant; the fitted one must be better."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=2048)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        kw = dict(chunk=512, exact_prefix=512, **self.CPU_KW)
        taylor = poly_linear_attention(q, k, v, scale, sigma=None, **kw)
        hermite = poly_linear_attention(q, k, v, scale, sigma=0.334, **kw)
        self.assertLess(
            (hermite - ref).pow(2).mean().item(),
            (taylor - ref).pow(2).mean().item(),
        )

    def test_coefficients_match_the_mean_of_exp(self):
        """Regression pin for the dropped-factor bug.

        The diagonal chunk uses unscaled ``exp``, so the inter-chunk polynomial
        must be on the same scale. The invariant that catches a dropped constant
        factor is that both have the same mean under ``s ~ N(0, sigma^2)``:

            E[c0 + c1*s + c2*s^2] = c0 + c2*sigma^2 = exp(sigma^2/2) = E[exp(s)]
        """
        import math

        from src.implementations.poly_reference import hermite_coefficients

        for sigma in (0.2, 0.334, 0.5):
            with self.subTest(sigma=sigma):
                c0, c1, c2 = hermite_coefficients(sigma)
                self.assertAlmostEqual(
                    c0 + c2 * sigma * sigma, math.exp(0.5 * sigma * sigma), places=12
                )

    def test_taylor_coefficients_are_unscaled(self):
        """sigma=None must be the plain expansion, with no gain applied."""
        from src.implementations.poly_reference import hermite_coefficients

        self.assertEqual(hermite_coefficients(None), (1.0, 1.0, 0.5))

    def test_batch_and_head_dims_are_independent(self):
        """Each (batch, head) scan must be independent of its neighbours."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(B=2, H=3, N=512)
        kw = dict(chunk=128, exact_prefix=128, sigma=0.334, **self.CPU_KW)
        full = poly_linear_attention(q, k, v, scale, **kw)
        one = poly_linear_attention(
            q[1:2, 2:3], k[1:2, 2:3], v[1:2, 2:3], scale, **kw
        )
        self.assertTrue(torch.allclose(full[1:2, 2:3], one, atol=1e-6))


    def _dense_diag(self, a, b, vc):
        """Dense reimplementation of the diagonal block, for testing the wiring.

        Deliberately not the Triton kernel: this exercises the hook itself on
        CPU, so a wiring mistake fails everywhere rather than only on CUDA.
        """
        blocked = torch.ones(
            a.shape[1], b.shape[1], device=a.device, dtype=torch.bool
        ).triu(1)
        w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
        return (w @ vc).float(), w.sum(-1, keepdim=True, dtype=torch.float32)

    def test_causal_diag_hook_produces_the_same_answer_as_the_dense_block(self):
        """The hook is an optimization, not a change of function."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        kw = dict(chunk=256, exact_prefix=0, sigma=0.334, **self.CPU_KW)

        base = poly_linear_attention(q, k, v, scale, **kw)
        hooked = poly_linear_attention(
            q, k, v, scale, causal_diag=self._dense_diag, **kw
        )
        self.assertLess((base - hooked).abs().max().item(), 1e-5)


    def test_chunks_inside_the_exact_prefix_are_not_computed(self):
        """Their output is overwritten by SDPA, so computing it is pure waste."""
        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        calls = []

        def counting_diag(a, b, vc):
            calls.append(a.shape[1])
            return self._dense_diag(a, b, vc)

        poly_linear_attention(
            q, k, v, scale, chunk=256, exact_prefix=512, sigma=0.334,
            causal_diag=counting_diag, **self.CPU_KW,
        )
        # 1024/256 = 4 chunks; the first two lie entirely inside the prefix.
        self.assertEqual(len(calls), 2)

    def test_prefix_skipping_does_not_change_the_output(self):
        """The skipped region is overwritten, so the answer must be identical."""
        import torch.nn.functional as F

        from src.implementations.poly_reference import poly_linear_attention

        q, k, v, scale = self._qkv(N=1024)
        kw = dict(chunk=256, sigma=0.334, **self.CPU_KW)
        skipped = poly_linear_attention(q, k, v, scale, exact_prefix=512, **kw)
        # exact_prefix=0 then an explicit SDPA overwrite reproduces the old
        # behaviour: everything computed, the prefix then thrown away.
        full = poly_linear_attention(q, k, v, scale, exact_prefix=0, **kw)
        full[:, :, :512] = F.scaled_dot_product_attention(
            q[:, :, :512], k[:, :, :512], v[:, :, :512], is_causal=True, scale=scale
        )
        self.assertLess((skipped - full).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
