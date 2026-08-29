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


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionCandidateTests(unittest.TestCase):
    def _config(self, causal=True):
        from torch_transformer_benchmark import TransformerConfig

        return TransformerConfig(
            batch_size=2, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=causal,
        )

    def _models(self, config, prefer_keymask=False):
        from torch_transformer_benchmark import BaselineTransformer, copy_model_weights
        from src.implementations.attention import AttentionCandidate

        torch.manual_seed(3)
        baseline = BaselineTransformer(config).eval()
        candidate = AttentionCandidate(config, prefer_keymask=prefer_keymask).eval()
        copy_model_weights(baseline, candidate, strict=True)
        return baseline, candidate

    def test_strict_weight_copy_succeeds(self) -> None:
        baseline, candidate = self._models(self._config())
        self.assertEqual(list(baseline.state_dict()), list(candidate.state_dict()))

    def test_matches_reference_across_padding_ratios(self) -> None:
        from torch_transformer_benchmark import compare_outputs, generate_random_case

        config = self._config()
        for prefer_keymask in (False, True):
            for ratio in (0.0, 0.3, 0.9):
                with self.subTest(ratio=ratio, prefer_keymask=prefer_keymask):
                    baseline, candidate = self._models(config, prefer_keymask)
                    x, mask = generate_random_case(
                        config, torch.device("cpu"), torch.float32, 11, ratio, 1.0
                    )
                    with torch.inference_mode():
                        result = compare_outputs(
                            baseline(x, mask), candidate(x, mask), 0.02, 0.002
                        )
                    self.assertTrue(result.passed, msg=f"{result.failed_elements} failed")

    def test_non_causal_config_matches_reference(self) -> None:
        from torch_transformer_benchmark import compare_outputs, generate_random_case

        config = self._config(causal=False)
        baseline, candidate = self._models(config)
        x, mask = generate_random_case(
            config, torch.device("cpu"), torch.float32, 13, 0.3, 1.0
        )
        with torch.inference_mode():
            result = compare_outputs(baseline(x, mask), candidate(x, mask), 0.02, 0.002)
        self.assertTrue(result.passed, msg=f"{result.failed_elements} failed")

    def test_general_mask_config_matches_reference(self) -> None:
        """A non-prefix mask must route away from the prefix shortcut."""
        from torch_transformer_benchmark import compare_outputs

        config = self._config()
        baseline, candidate = self._models(config)
        torch.manual_seed(9)
        x = torch.randn(2, 8, 16)
        mask = torch.tensor(
            [[False, True, True, True, True, False, True, True],
             [True, True, False, True, True, True, True, True]]
        )
        x = x.masked_fill(~mask[..., None], 0)
        with torch.inference_mode():
            result = compare_outputs(baseline(x, mask), candidate(x, mask), 0.02, 0.002)
        self.assertTrue(result.passed, msg=f"{result.failed_elements} failed")

    def test_classifies_the_mask_once_per_forward(self) -> None:
        """The host sync must not scale with layer count."""
        from src.implementations import attention as attention_module

        config = self._config()
        _, candidate = self._models(config)
        calls = []
        original = attention_module.classify_mask

        def counting(mask):
            calls.append(mask)
            return original(mask)

        attention_module.classify_mask = counting
        self.addCleanup(setattr, attention_module, "classify_mask", original)

        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.inference_mode():
            candidate(x, mask)

        self.assertEqual(len(calls), 1, msg=f"{config.num_layers} layers")

    def test_route_reaches_every_layer(self) -> None:
        from src.implementations.attention_routing import Route

        _, candidate = self._models(self._config())
        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.inference_mode():
            candidate(x, mask)
        for layer in candidate.layers:
            self.assertIs(layer.attention.route, Route.SDPA_CAUSAL)

    def test_both_candidate_specs_are_loadable(self) -> None:
        from src.infra import load_candidate

        self.assertEqual(load_candidate("attention").name, "attention")
        self.assertEqual(
            load_candidate("attention:KEYMASK_CANDIDATE").name, "attention-keymask"
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExactEagerDtypeTests(unittest.TestCase):
    def test_reduced_precision_routes_are_bitwise_identical(self) -> None:
        """float16/bfloat16 fall to an exact route and reproduce the reference."""
        from torch_transformer_benchmark import (
            BaselineTransformer, TransformerConfig, copy_model_weights,
            generate_random_case,
        )
        from src.implementations.attention import AttentionCandidate

        config = TransformerConfig(
            batch_size=2, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=True,
        )
        for dtype in (torch.float16, torch.bfloat16):
            for ratio in (0.0, 0.3):
                with self.subTest(dtype=dtype, ratio=ratio):
                    torch.manual_seed(5)
                    baseline = BaselineTransformer(config).to(dtype).eval()
                    candidate = AttentionCandidate(config).to(dtype).eval()
                    copy_model_weights(baseline, candidate, strict=True)
                    x, mask = generate_random_case(
                        config, torch.device("cpu"), dtype, 17, ratio, 1.0
                    )
                    with torch.inference_mode():
                        delta = (baseline(x, mask) - candidate(x, mask)).abs().max()
                    self.assertEqual(delta.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
