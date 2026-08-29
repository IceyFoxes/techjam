from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskRoutedSDPASelfAttentionTests(unittest.TestCase):
    def _pair(self, d_model=16, num_heads=4):
        from torch_transformer_benchmark import BaselineSelfAttention
        from src.implementations.attention import MaskRoutedSDPASelfAttention

        torch.manual_seed(0)
        reference = BaselineSelfAttention(d_model, num_heads).eval()
        candidate = MaskRoutedSDPASelfAttention(d_model, num_heads).eval()
        candidate.load_state_dict(reference.state_dict())
        return reference, candidate

    def _inputs(self, batch=2, seq_len=8, d_model=16, lengths=None):
        torch.manual_seed(1)
        x = torch.randn(batch, seq_len, d_model)
        if lengths is None:
            return x, None
        positions = torch.arange(seq_len)[None, :]
        mask = positions < torch.tensor(lengths)[:, None]
        return x.masked_fill(~mask[..., None], 0), mask

    def _assert_within_tolerance(self, reference, candidate):
        from torch_transformer_benchmark import compare_outputs

        result = compare_outputs(reference, candidate, 0.02, 0.002)
        self.assertTrue(
            result.passed,
            msg=f"{result.failed_elements} failed, max_abs={result.max_abs_error}",
        )

    def test_defaults_to_upstream_behavior(self) -> None:
        """Constructing the module must not change what sdpa.py already does."""
        from src.implementations.attention import MaskRoutedSDPASelfAttention
        from src.implementations.attention_routing import Route
        from src.implementations.sdpa import StridedSDPASelfAttention

        torch.manual_seed(0)
        module = MaskRoutedSDPASelfAttention(16, 4).eval()
        self.assertIs(module.route, Route.SDPA_CAUSAL_KEYMASK)
        self.assertIsInstance(module, StridedSDPASelfAttention)

        upstream = StridedSDPASelfAttention(16, 4).eval()
        upstream.load_state_dict(module.state_dict())
        x, mask = self._inputs(lengths=[5, 8])
        with torch.inference_mode():
            delta = (upstream(x, mask, True) - module(x, mask, True)).abs().max()
        self.assertEqual(delta.item(), 0.0)

    def test_keeps_the_reference_parameter_contract(self) -> None:
        reference, candidate = self._pair()
        self.assertEqual(list(reference.state_dict()), list(candidate.state_dict()))

    def test_matches_reference_causal_without_a_mask(self) -> None:
        from src.implementations.attention_routing import Route

        reference, candidate = self._pair()
        x, _ = self._inputs()
        candidate.route = Route.SDPA_CAUSAL
        with torch.inference_mode():
            self._assert_within_tolerance(
                reference(x, None, True), candidate(x, None, True)
            )

    def test_every_route_matches_the_reference_on_a_prefix_mask(self) -> None:
        from src.implementations.attention_routing import Route

        x, mask = self._inputs(lengths=[5, 8])
        for route in (
            Route.SDPA_CAUSAL,
            Route.SDPA_CAUSAL_KEYMASK,
            Route.SDPA_FULLMASK,
            Route.EXACT_EAGER,
        ):
            with self.subTest(route=route):
                reference, candidate = self._pair()
                candidate.route = route
                with torch.inference_mode():
                    self._assert_within_tolerance(
                        reference(x, mask, True), candidate(x, mask, True)
                    )

    def test_dropping_the_mask_agrees_with_keeping_it(self) -> None:
        """The two causal routes are the same function (spec section 3)."""
        from src.implementations.attention_routing import Route

        x, mask = self._inputs(lengths=[3, 8])
        _, dropped = self._pair()
        _, kept = self._pair()
        dropped.route = Route.SDPA_CAUSAL
        kept.route = Route.SDPA_CAUSAL_KEYMASK
        with torch.inference_mode():
            self._assert_within_tolerance(kept(x, mask, True), dropped(x, mask, True))

    def test_general_mask_routes_match_the_reference(self) -> None:
        from src.implementations.attention_routing import Route

        torch.manual_seed(2)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor(
            [[False, True, True, True, True, False, True, True],
             [True, True, False, True, True, True, True, True]]
        )
        x = x.masked_fill(~mask[..., None], 0)
        for route, causal in (
            (Route.SDPA_FULLMASK, True),
            (Route.SDPA_KEYMASK, False),
            (Route.EXACT_EAGER, True),
            (Route.EXACT_EAGER, False),
        ):
            with self.subTest(route=route):
                reference, candidate = self._pair()
                candidate.route = route
                with torch.inference_mode():
                    self._assert_within_tolerance(
                        reference(x, mask, causal), candidate(x, mask, causal)
                    )

    def test_exact_routes_are_bitwise_identical(self) -> None:
        """Both exact routes must be exact, not merely within tolerance."""
        from src.implementations.attention_routing import Route

        x, mask = self._inputs(lengths=[5, 8])
        for route in (Route.EXACT_EAGER, Route.EXACT_EAGER_PREFIX):
            with self.subTest(route=route):
                reference, candidate = self._pair()
                candidate.route = route
                with torch.inference_mode():
                    delta = (
                        reference(x, mask, True) - candidate(x, mask, True)
                    ).abs().max()
                self.assertEqual(delta.item(), 0.0)

    def test_prefix_skip_is_wrong_for_a_general_mask(self) -> None:
        """Guards the bug TDD caught: the skip is invalid off a prefix mask.

        EXACT_EAGER_PREFIX must never be selected for a general mask. This pins
        that it genuinely would be wrong, so the routing rule is load-bearing
        rather than decorative.
        """
        from torch_transformer_benchmark import compare_outputs
        from src.implementations.attention_routing import Route

        torch.manual_seed(2)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor(
            [[False, True, True, True, True, False, True, True],
             [True, True, False, True, True, True, True, True]]
        )
        x = x.masked_fill(~mask[..., None], 0)
        reference, candidate = self._pair()
        candidate.route = Route.EXACT_EAGER_PREFIX
        with torch.inference_mode():
            result = compare_outputs(
                reference(x, mask, True), candidate(x, mask, True), 0.02, 0.002
            )
        self.assertFalse(result.passed)

    def test_head_views_avoid_copies(self) -> None:
        """Lever L9, inherited from sdpa.py: Q/K/V reach SDPA as views."""
        from src.implementations.attention import MaskRoutedSDPASelfAttention

        module = MaskRoutedSDPASelfAttention(16, 4)
        projected = torch.randn(2, 8, 16)
        view = module._split_heads_view(projected)
        self.assertEqual(tuple(view.shape), (2, 4, 8, 4))
        self.assertFalse(view.is_contiguous())
        self.assertEqual(
            view.untyped_storage().data_ptr(), projected.untyped_storage().data_ptr()
        )


if __name__ == "__main__":
    unittest.main()
