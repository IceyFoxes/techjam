"""The Triton kernels must compute the same quantity as dense PyTorch.

These are the same mathematical values in a different summation order, so a
deviation beyond float16 rounding indicates a bug, not precision loss.
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
class QuadApplyTests(unittest.TestCase):
    def _case(self, M=2, C=128, D=64, V=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        a = (
            torch.randn(M, C, D, generator=gen, device="cuda", dtype=torch.float16)
            * 0.2
        )
        s = torch.randn(M, D * D, V, generator=gen, device="cuda", dtype=torch.float32)
        return a, s

    def _dense_fp16(self, a, s):
        """The path the kernel replaces: dense phi2 in float16."""
        from src.implementations.poly_reference import phi2

        return phi2(a) @ s.to(a.dtype)

    def _truth_fp32(self, a, s):
        """Ground truth in float32."""
        from src.implementations.poly_reference import phi2

        return phi2(a.float()) @ s

    def _assert_no_worse_than_dense(self, a, s, quad_apply):
        """The kernel must be at least as accurate as the path it replaces.

        Both compute the same sum in different orders and round to float16, so
        they differ by a few ulps -- at these magnitudes one ulp is already
        ~2e-3, which is why an absolute tolerance is the wrong instrument.
        Judging both against float32 truth tests the thing that matters.
        """
        truth = self._truth_fp32(a, s)
        kernel_err = (quad_apply(a, s).float() - truth).abs().max().item()
        dense_err = (self._dense_fp16(a, s).float() - truth).abs().max().item()
        scale = truth.abs().max().item()
        self.assertLessEqual(
            kernel_err,
            max(3.0 * dense_err, 1e-6),
            f"kernel err {kernel_err:.3e} vs dense err {dense_err:.3e} "
            f"(values up to {scale:.2f})",
        )

    def test_matches_dense_for_the_case_14_shape(self):
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(D=64, V=64)
        self._assert_no_worse_than_dense(a, s, quad_apply)

    def test_matches_dense_across_shapes(self):
        from src.kernels.poly_attention_triton import quad_apply

        for D in (16, 32, 64):
            for C in (128, 512):
                for V in (32, 64):
                    with self.subTest(D=D, C=C, V=V):
                        a, s = self._case(C=C, D=D, V=V)
                        self._assert_no_worse_than_dense(a, s, quad_apply)

    def test_error_is_rounding_not_bias(self):
        """A systematic offset would show as a non-zero mean error."""
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(D=64, V=64, C=512)
        truth = self._truth_fp32(a, s)
        diff = quad_apply(a, s).float() - truth
        rel_bias = abs(diff.mean().item()) / truth.abs().mean().item()
        self.assertLess(rel_bias, 1e-3, f"relative mean bias {rel_bias:.3e}")

    def test_handles_a_ragged_final_chunk(self):
        """N is not a multiple of the chunk length, so C need not divide BC."""
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(C=100)
        self._assert_no_worse_than_dense(a, s, quad_apply)

    def test_single_column_state_for_the_denominator(self):
        """The denominator uses V=1; it must not be a special case that breaks."""
        from src.kernels.poly_attention_triton import quad_apply

        a, s = self._case(V=1)
        self._assert_no_worse_than_dense(a, s, quad_apply)


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "Triton kernels require CUDA")
class QuadUpdateTests(unittest.TestCase):
    def _case(self, M=2, C=128, D=64, V=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        b = (
            torch.randn(M, C, D, generator=gen, device="cuda", dtype=torch.float16)
            * 0.2
        )
        v = (
            torch.randn(M, C, V, generator=gen, device="cuda", dtype=torch.float16)
            * 0.2
        )
        return b, v

    def _truth_fp32(self, b, v):
        from src.implementations.poly_reference import phi2

        return phi2(b.float()).transpose(-2, -1) @ v.float()

    def _dense_fp16(self, b, v):
        from src.implementations.poly_reference import phi2

        return (phi2(b).transpose(-2, -1) @ v).float()

    def _assert_no_worse_than_dense(self, b, v, D=64):
        from src.kernels.poly_attention_triton import quad_update

        M, _, _ = b.shape
        out = torch.zeros(M, D * D, v.shape[2], device="cuda", dtype=torch.float32)
        quad_update(b, v, out)
        truth = self._truth_fp32(b, v)
        kernel_err = (out - truth).abs().max().item()
        dense_err = (self._dense_fp16(b, v) - truth).abs().max().item()
        self.assertLessEqual(
            kernel_err,
            max(3.0 * dense_err, 1e-6),
            f"kernel err {kernel_err:.3e} vs dense err {dense_err:.3e}",
        )

    def test_matches_dense_for_the_case_14_shape(self):
        b, v = self._case(D=64, V=64)
        self._assert_no_worse_than_dense(b, v)

    def test_matches_dense_across_shapes(self):
        for D in (16, 32, 64):
            for C in (128, 512):
                with self.subTest(D=D, C=C):
                    b, v = self._case(C=C, D=D)
                    self._assert_no_worse_than_dense(b, v, D=D)

    def test_accumulates_rather_than_overwrites(self):
        """The state is a running sum; two folds must equal one doubled fold."""
        from src.kernels.poly_attention_triton import quad_update

        b, v = self._case()
        once = torch.zeros(2, 64 * 64, 64, device="cuda", dtype=torch.float32)
        quad_update(b, v, once)
        twice = torch.zeros_like(once)
        quad_update(b, v, twice)
        quad_update(b, v, twice)
        self.assertLess((twice - 2 * once).abs().max().item(), 1e-3)

    def test_handles_a_ragged_final_chunk(self):
        b, v = self._case(C=100)
        self._assert_no_worse_than_dense(b, v)

    def test_single_column_for_the_denominator(self):
        from src.kernels.poly_attention_triton import quad_update

        b, _ = self._case()
        ones = torch.ones(2, 128, 1, device="cuda", dtype=torch.float16)
        out = torch.zeros(2, 64 * 64, 1, device="cuda", dtype=torch.float32)
        quad_update(b, ones, out)
        truth = self._truth_fp32(b, ones)
        self.assertLess((out - truth).abs().max().item(), 1e-2)

    def test_rejects_a_float16_state(self):
        """The fp16-state trap must fail loudly, not compute something plausible."""
        from src.kernels.poly_attention_triton import quad_update

        b, v = self._case()
        bad = torch.zeros(2, 64 * 64, 64, device="cuda", dtype=torch.float16)
        with self.assertRaises(ValueError):
            quad_update(b, v, bad)


if __name__ == "__main__":
    unittest.main()
