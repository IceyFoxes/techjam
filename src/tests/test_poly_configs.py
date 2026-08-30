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

    def test_lookup_does_not_leak_provenance_into_launch_kwargs(self):
        """`source` documents the entry; passing it to Triton would be an error."""
        from src.kernels.poly_configs import lookup

        self.assertNotIn("source", lookup("quad_apply", (512, 64, 64), (8, 9)))

    def test_every_table_entry_names_the_run_that_measured_it(self):
        """A config with no recorded provenance is a guess wearing a table."""
        from src.kernels.poly_configs import CONFIGS

        for (kernel, key, capability), entry in CONFIGS.items():
            self.assertIn("source", entry, f"{kernel} {key} {capability}")
            self.assertTrue(entry["source"].startswith("research/benchmarks/"))

    def test_all_three_kernels_are_covered_for_the_target_shape(self):
        from src.kernels.poly_configs import lookup

        for kernel in ("quad_apply", "quad_update", "causal_diag"):
            self.assertIsNotNone(
                lookup(kernel, (512, 64, 64), (8, 9)), f"{kernel} has no entry"
            )


if __name__ == "__main__":
    unittest.main()
