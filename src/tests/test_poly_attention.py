"""The fused path must agree with the PyTorch oracle it replaces."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


def _cuda_missing():
    return torch is None or not torch.cuda.is_available()


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "requires CUDA")
class FusedPathTests(unittest.TestCase):
    def _qkv(self, M=2, N=2048, D=64, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        shape = (1, M, N, D)

        def draw():
            return (
                torch.randn(
                    shape, generator=gen, device="cuda", dtype=torch.float16
                )
                * 0.577
            )

        return draw(), draw(), draw(), D ** -0.5

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

    def test_fused_is_deterministic(self):
        """Two identical calls must agree bitwise; autotune must not leak state."""
        from src.implementations.poly_attention import poly_attention_forward

        q, k, v, scale = self._qkv()
        a = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=True)
        b = poly_attention_forward(q, k, v, scale, sigma=0.334, use_triton=True)
        self.assertTrue(torch.equal(a, b))

    def test_state_dtype_is_float32_regardless_of_input_dtype(self):
        """Regression guard for the fp16-state trap that only fails at N>=65536."""
        import inspect

        from src.implementations import poly_attention

        source = inspect.getsource(poly_attention.poly_attention_forward)
        self.assertIn("float32", source)
        self.assertNotIn("state_dtype=torch.float16", source)


if __name__ == "__main__":
    unittest.main()
