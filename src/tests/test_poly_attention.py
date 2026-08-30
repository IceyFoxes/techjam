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


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolyRouteToggleTests(unittest.TestCase):
    """The route must be opt-in and reversible in one flag.

    These are CPU-only: they exercise the routing decision, not the kernel.
    """

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
        # Beyond the measured ceiling it must fall back, not approximate.
        self.assertEqual(module.route_name(sigma=0.5217), "flash")
        self.assertEqual(module.route_name(sigma=5.0), "flash")

    def test_instances_default_to_the_module_flag(self):
        from src.implementations import extreme

        module = extreme.PolyOrFlashSelfAttention(1024, 16)
        self.assertEqual(module.poly_enabled, extreme.POLY_ATTENTION_ENABLED)

    def test_is_a_flash_only_subclass_so_the_fallback_is_person_4s_code(self):
        from src.implementations.extreme import (
            FlashOnlySDPASelfAttention,
            PolyOrFlashSelfAttention,
        )

        self.assertTrue(
            issubclass(PolyOrFlashSelfAttention, FlashOnlySDPASelfAttention)
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "requires CUDA")
class PolyRouteExecutionTests(unittest.TestCase):
    def _module(self, enabled):
        from src.implementations.extreme import PolyOrFlashSelfAttention

        module = PolyOrFlashSelfAttention(1024, 16).cuda().half().eval()
        module.poly_enabled = enabled
        return module

    def test_disabled_route_matches_flash_exactly(self):
        """Disabling must reproduce today's behaviour bitwise, not approximately."""
        from src.implementations.extreme import FlashOnlySDPASelfAttention

        torch.manual_seed(0)
        module = self._module(enabled=False)
        flash = FlashOnlySDPASelfAttention(1024, 16).cuda().half().eval()
        flash.load_state_dict(module.state_dict())

        x = torch.randn(1, 2048, 1024, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            self.assertTrue(torch.equal(module(x, None, True), flash(x, None, True)))

    def test_enabled_route_stays_within_tolerance_of_flash(self):
        torch.manual_seed(0)
        module = self._module(enabled=True)
        x = torch.randn(1, 2048, 1024, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            got = module(x, None, True).float()
            module.poly_enabled = False
            module._sigma = None
            ref = module(x, None, True).float()
        tol = torch.clamp(ref.abs() * 0.02, min=0.002)
        self.assertEqual(int(((got - ref).abs() > tol).sum()), 0)

    def test_rejects_a_mask_like_the_flash_route_does(self):
        module = self._module(enabled=True)
        x = torch.randn(1, 128, 1024, device="cuda", dtype=torch.float16)
        mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
        with self.assertRaises(ValueError):
            module(x, mask, True)


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(_cuda_missing(), "requires CUDA")
class PolyMemoryTests(unittest.TestCase):
    """Peak VRAM must stay close to the exact path's.

    Case 14 streams 1-2 samples at a time against a fixed budget, so the
    polynomial route's working set is a hard constraint, not a nicety. An
    earlier version passed 3-D tensors to the prefix SDPA, which silently
    selected the quadratic math backend and cost 2.4 GiB per sample -- and no
    test noticed, because every correctness test still passed.

    Inputs here are STRIDED VIEWS, matching what _split_heads_view actually
    produces. With contiguous inputs the internal reshape is free and this
    measures nothing.
    """

    def _strided_qkv(self, B=1, N=16384, H=16, D=64):
        torch.manual_seed(0)
        packed = [
            (torch.randn(B, N, H * D, device="cuda") * 0.577).half()
            for _ in range(3)
        ]
        views = tuple(t.view(B, N, H, D).transpose(1, 2) for t in packed)
        input_bytes = sum(t.numel() * t.element_size() for t in packed)
        return views, input_bytes

    def _peak_mib(self, fn, warm=True):
        """Peak allocation of one call, in MiB.

        ``warm`` runs the function once first, so the measurement is the
        steady-state working set rather than the one-off Triton autotuning
        allocation. Autotuning happens on the first chunk of a real run and is
        bounded; what must not grow with the workload is the steady state.
        """
        if warm:
            del_me = fn()
            del del_me
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        result = fn()
        peak = torch.cuda.max_memory_allocated() - base
        del result
        return peak / 2 ** 20

    def test_peak_is_bounded_by_the_input_size(self):
        from src.implementations.poly_attention import poly_attention_forward

        (q, k, v), input_bytes = self._strided_qkv()
        scale = q.shape[-1] ** -0.5
        with torch.inference_mode():
            peak = self._peak_mib(
                lambda: poly_attention_forward(q, k, v, scale, sigma=0.3338)
            )
        budget = 2.0 * input_bytes / 2 ** 20
        self.assertLess(
            peak,
            budget,
            f"peak {peak:.0f} MiB exceeds {budget:.0f} MiB "
            f"(2x the {input_bytes / 2 ** 20:.0f} MiB of q/k/v)",
        )

    def test_peak_does_not_blow_up_relative_to_flash(self):
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from src.implementations.poly_attention import poly_attention_forward

        (q, k, v), _ = self._strided_qkv()
        scale = q.shape[-1] ** -0.5
        with torch.inference_mode():

            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, scale=scale
                    )

            flash_peak = self._peak_mib(flash)
            poly_peak = self._peak_mib(
                lambda: poly_attention_forward(q, k, v, scale, sigma=0.3338)
            )
        self.assertLess(
            poly_peak,
            6.0 * flash_peak,
            f"poly peak {poly_peak:.0f} MiB against flash {flash_peak:.0f} MiB",
        )


if __name__ == "__main__":
    unittest.main()
