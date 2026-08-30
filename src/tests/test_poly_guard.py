"""The guard decides whether the polynomial approximation is valid at runtime.

The approximation depends on scores being small, which is a property of the
benchmark's random initialisation (measured sigma = 0.3336) rather than of
attention in general. These tests run on CPU: the guard is deliberately
torch-light so it needs no GPU to verify.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
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

    def test_sample_larger_than_sequence_is_clamped(self):
        """N smaller than the sample count must not raise."""
        from src.implementations.poly_guard import estimate_sigma

        q, k, scale = self._qk(N=64)
        got = estimate_sigma(q, k, scale, samples=512)
        self.assertGreater(got, 0.0)

    def test_benchmark_sigma_is_safe_and_large_sigma_is_not(self):
        from src.implementations.poly_guard import SIGMA_CEILING, poly_is_safe

        self.assertTrue(poly_is_safe(0.334))
        self.assertFalse(poly_is_safe(SIGMA_CEILING + 0.01))
        self.assertFalse(poly_is_safe(float("nan")))
        self.assertFalse(poly_is_safe(float("inf")))
        self.assertFalse(poly_is_safe(None))
        self.assertFalse(poly_is_safe(0.0))

    def test_ceiling_is_above_the_measured_benchmark_value(self):
        """A ceiling at or below 0.334 would disable the route entirely."""
        from src.implementations.poly_guard import SIGMA_CEILING

        self.assertGreater(SIGMA_CEILING, 0.334)

    def test_ceiling_sits_inside_the_measured_safe_band(self):
        """Pins the Task 7 sweep so a later edit cannot quietly widen it.

        Measured at N=8192 against the dense reference: sigma 0.4808 was the
        largest value passing the official criterion, and 0.5217 was the first
        to fail. The ceiling must stay below the last verified pass, so the
        route never runs in the untested band.
        """
        from src.implementations.poly_guard import SIGMA_CEILING, poly_is_safe

        # 0.4188 is the largest CLEAN pass measured after Stage 0. The
        # previous bound here was 0.4808, which passed before Stage 0's
        # numerics changes and fails after them -- so this assertion is what
        # would have caught a stale ceiling.
        self.assertLessEqual(SIGMA_CEILING, 0.4188)
        # Everything at or beyond the first observed failure must be rejected.
        for failing in (0.5217, 0.5642, 0.6544, 0.7512):
            self.assertFalse(poly_is_safe(failing), f"sigma {failing} must fall back")
        # The benchmark's own spread across seeds must be comfortably accepted.
        for operating in (0.3327, 0.3339, 0.3343):
            self.assertTrue(poly_is_safe(operating), f"sigma {operating} must run")

    def test_rejects_wrongly_shaped_inputs(self):
        from src.implementations.poly_guard import estimate_sigma

        q, k, scale = self._qk()
        with self.assertRaises(ValueError):
            estimate_sigma(q[0], k[0], scale)


if __name__ == "__main__":
    unittest.main()
