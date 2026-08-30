"""The dispatcher's key-mask decision must be safe and correctly scoped.

Dropping the broadcast key mask is bitwise exact only under causal attention
with a prefix-valid mask. These tests pin that the dispatcher checks both
conditions rather than assuming them, and that the check does not run per
forward -- it costs ~85-99 us, which would erase the gain on small cases.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DropKeyMaskDecisionTests(unittest.TestCase):
    def _dispatcher(self, causal=True):
        from torch_transformer_benchmark import TransformerConfig
        from src.dispatcher import DispatchingTransformer

        config = TransformerConfig(
            batch_size=2, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=causal,
        )
        return DispatchingTransformer(config), config

    def _prefix_mask(self, batch=2, seq_len=8, lengths=(5, 8)):
        positions = torch.arange(seq_len)[None, :]
        return positions < torch.tensor(lengths)[:, None]

    def test_defaults_to_keeping_the_mask(self) -> None:
        """A freshly constructed model must behave exactly as before."""
        model, _ = self._dispatcher()
        for layer in model.layers:
            self.assertFalse(layer.attention.drop_key_mask)

    def test_reference_route_never_drops_the_mask(self) -> None:
        """The fallback must reproduce baseline arithmetic exactly."""
        from src.dispatcher import REFERENCE_BACKEND, RouteDecision

        model, _ = self._dispatcher()
        route = RouteDecision(1, REFERENCE_BACKEND, None, "test")
        self.assertFalse(model._may_drop_key_mask(route, self._prefix_mask()))

    def test_non_causal_config_never_drops_the_mask(self) -> None:
        """Without an upper triangle there is nothing to subsume the padding."""
        from src.dispatcher import COMPILED_SDPA_BACKEND, RouteDecision

        model, _ = self._dispatcher(causal=False)
        route = RouteDecision(1, COMPILED_SDPA_BACKEND, "default", "test")
        self.assertFalse(model._may_drop_key_mask(route, self._prefix_mask()))

    def test_general_mask_never_drops_the_mask(self) -> None:
        """A left-padded or gapped mask is NOT removable."""
        from src.dispatcher import COMPILED_SDPA_BACKEND, RouteDecision

        model, _ = self._dispatcher()
        route = RouteDecision(1, COMPILED_SDPA_BACKEND, "default", "test")
        gapped = torch.tensor(
            [[False, True, True, True, True, True, True, True],
             [True, True, False, True, True, True, True, True]]
        )
        self.assertFalse(model._may_drop_key_mask(route, gapped))

    def test_causal_prefix_mask_drops_the_mask(self) -> None:
        from src.dispatcher import COMPILED_SDPA_BACKEND, RouteDecision

        model, _ = self._dispatcher()
        route = RouteDecision(1, COMPILED_SDPA_BACKEND, "default", "test")
        self.assertTrue(model._may_drop_key_mask(route, self._prefix_mask()))

    def test_absent_mask_does_not_drop(self) -> None:
        from src.dispatcher import COMPILED_SDPA_BACKEND, RouteDecision

        model, _ = self._dispatcher()
        route = RouteDecision(1, COMPILED_SDPA_BACKEND, "default", "test")
        self.assertFalse(model._may_drop_key_mask(route, None))

    def test_dropping_the_mask_matches_the_reference(self) -> None:
        """End-to-end: the optimized path agrees with baseline arithmetic."""
        from torch_transformer_benchmark import (
            BaselineTransformer, compare_outputs, copy_model_weights,
        )

        model, config = self._dispatcher()
        torch.manual_seed(4)
        baseline = BaselineTransformer(config).eval()
        copy_model_weights(baseline, model, strict=True)
        model.eval()

        mask = self._prefix_mask()
        x = torch.randn(2, 8, 16).masked_fill(~mask[..., None], 0)
        model._set_drop_key_mask(True)
        with torch.inference_mode():
            result = compare_outputs(
                baseline(x, mask), model._forward_sdpa(x, mask), 0.02, 0.002
            )
        self.assertTrue(result.passed, msg=f"{result.failed_elements} failed")

    def test_decision_is_not_taken_per_forward(self) -> None:
        """The host sync costs ~85-99us; it must not run on every call."""
        from src.dispatcher import DispatchingTransformer

        model, config = self._dispatcher()
        model.eval()
        calls = []
        original = DispatchingTransformer._may_drop_key_mask

        def counting(self, route, mask):
            calls.append(route)
            return original(self, route, mask)

        DispatchingTransformer._may_drop_key_mask = counting
        self.addCleanup(
            setattr, DispatchingTransformer, "_may_drop_key_mask", original
        )

        mask = self._prefix_mask()
        x = torch.randn(2, 8, 16).masked_fill(~mask[..., None], 0)
        with torch.inference_mode():
            for _ in range(5):
                model(x, mask)

        # CPU routes to the reference backend, which resolves before the
        # decision is reached; either way it must not scale with call count.
        self.assertLessEqual(len(calls), 1, msg=f"{len(calls)} syncs in 5 calls")


if __name__ == "__main__":
    unittest.main()
