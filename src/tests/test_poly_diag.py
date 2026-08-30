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
        self.assertEqual(
            (num_before[:, :64] - num_after[:, :64]).abs().max().item(), 0.0
        )
        self.assertEqual(
            (den_before[:, :64] - den_after[:, :64]).abs().max().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
