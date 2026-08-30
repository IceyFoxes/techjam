"""The benchmark harness must report a noise floor, not just a minimum.

Four identical profiles of the same workload on this machine measured 280.3,
263.7, 571.1 and 267.6 ms -- a 2.17x spread. Phase 1 shipped without ever
establishing a floor and said so ("no formal noise floor was established").
Every A/B in the Phase 2 spec is undecidable until one exists, so these tests
pin the statistic that decides whether a measured difference is real.

See research/attention-softmax/integrated-kernel-spec.md section 4, task F0.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SummariseTimingsTests(unittest.TestCase):
    def test_reports_min_median_max_and_spread(self):
        from src.bench_poly import summarise_timings

        got = summarise_timings({"a": [10.0, 12.0, 20.0]})
        self.assertAlmostEqual(got["variants"]["a"]["min_ms"], 10.0)
        self.assertAlmostEqual(got["variants"]["a"]["median_ms"], 12.0)
        self.assertAlmostEqual(got["variants"]["a"]["max_ms"], 20.0)
        # Spread is expressed against the minimum, which is the statistic the
        # harness reports as "the" time.
        self.assertAlmostEqual(got["variants"]["a"]["spread"], 1.0)

    def test_noise_floor_comes_from_the_a_a_control_pair(self):
        """Two runs of the SAME implementation bound what a real effect must beat."""
        from src.bench_poly import summarise_timings

        got = summarise_timings(
            {"poly": [100.0, 104.0], "poly_control": [103.0, 108.0]},
            control_pairs=[("poly", "poly_control")],
        )
        # min 100.0 against min 103.0 -- identical code differing by 3%.
        self.assertAlmostEqual(got["noise"]["aa_discrepancy"], 0.03)
        self.assertGreaterEqual(got["noise"]["minimum_detectable_effect"], 1.03)

    def test_floor_is_the_a_a_gap_not_the_within_variant_spread(self):
        """Dispersion of individual samples is not the error of a min-of-N compare.

        Measured at B=2, N=100000: identical code reproduced to 0.6% while its own
        repetitions varied by 10.3%. Using the spread would set the floor at
        1.103x and make every fix worth under 10% unmeasurable -- and since
        (max-min)/min only grows with more repetitions, it would penalise
        measuring more carefully.
        """
        from src.bench_poly import summarise_timings

        got = summarise_timings(
            {"poly": [100.0, 130.0], "poly_control": [100.0, 128.0]},
            control_pairs=[("poly", "poly_control")],
        )
        self.assertAlmostEqual(got["noise"]["aa_discrepancy"], 0.0)
        self.assertAlmostEqual(got["noise"]["worst_within_variant_spread"], 0.3)
        self.assertAlmostEqual(got["noise"]["minimum_detectable_effect"], 1.0)

    def test_reports_the_worst_paired_disagreement_as_context(self):
        """What a one-shot comparison would have suffered, hence why we repeat."""
        from src.bench_poly import summarise_timings

        got = summarise_timings(
            {"poly": [100.0, 130.0], "poly_control": [100.0, 104.0]},
            control_pairs=[("poly", "poly_control")],
        )
        # rep 2 had identical code reading 130 against 104.
        self.assertAlmostEqual(got["noise"]["worst_paired_aa_discrepancy"], 0.25)
        # ...but the floor still comes from the minima, which agree exactly.
        self.assertAlmostEqual(got["noise"]["minimum_detectable_effect"], 1.0)

    def test_decides_whether_a_measured_ratio_is_reportable(self):
        from src.bench_poly import is_resolvable, summarise_timings

        got = summarise_timings(
            {"poly": [100.0, 104.0], "poly_control": [103.0, 108.0]},
            control_pairs=[("poly", "poly_control")],
        )
        floor = got["noise"]["minimum_detectable_effect"]
        self.assertFalse(is_resolvable(1.02, floor))   # inside the floor
        self.assertFalse(is_resolvable(1.0 / 1.02, floor))  # and in either direction
        self.assertTrue(is_resolvable(1.40, floor))
        self.assertTrue(is_resolvable(1.0 / 1.40, floor))

    def test_no_control_pair_means_no_claimed_floor(self):
        """Without an A/A control the harness must not invent a floor."""
        from src.bench_poly import summarise_timings

        got = summarise_timings({"a": [10.0, 11.0]})
        self.assertIsNone(got["noise"]["aa_discrepancy"])
        # It still reports the within-variant spread, which is a lower bound.
        self.assertAlmostEqual(got["noise"]["minimum_detectable_effect"], 1.1)

    def test_single_rep_cannot_establish_a_floor(self):
        from src.bench_poly import summarise_timings

        got = summarise_timings({"a": [10.0]})
        self.assertAlmostEqual(got["variants"]["a"]["spread"], 0.0)
        self.assertIsNone(got["noise"]["aa_discrepancy"])
        self.assertAlmostEqual(got["noise"]["minimum_detectable_effect"], 1.0)


if __name__ == "__main__":
    unittest.main()
