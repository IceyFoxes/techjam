"""The extreme routes must set ``drop_key_mask`` explicitly, and case 6 must
be allowed to drop a right-padded causal key mask.

Two properties are pinned here:

1. The extreme branch never inherits a stale ``drop_key_mask`` from an earlier
   forward on the same model instance. Every other route in
   ``DispatchingTransformer.forward`` sets the flag; before this test the
   extreme branch relied on the class-level default.
2. Under causal attention with a right-padded mask, case 6 producing the same
   output with and without the key mask -- which is what makes dropping it
   safe. See ``research/attention-softmax/safe-optimization-spec.md`` section 3
   and ``test_padding_mask_redundancy.py``.

The eligibility decision is exercised on a real config through
``_may_drop_key_mask`` rather than by asserting on a hard-coded route table, so
this fails if the eligibility rule is narrowed again.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExtremeKeyMaskTests(unittest.TestCase):
    def _prefix_mask(self, batch, seq_len, valid):
        positions = torch.arange(seq_len)
        lengths = torch.full((batch, 1), valid)
        return positions[None, :] < lengths

    def _route(self, backend, case_id):
        from src.dispatcher import RouteDecision

        return RouteDecision(case_id, backend, None, "test route")

    def _model(self, case_id):
        from src.dispatcher import DispatchingTransformer
        from src.infra import load_official_cases

        shape = load_official_cases()[case_id].benchmark_config()
        from torch_transformer_benchmark import TransformerConfig

        # Keep the official tuple except for the batch, which only needs to be
        # large enough to carry a mask; routing does not depend on its value.
        config = TransformerConfig(
            batch_size=2,
            seq_len=shape["seq_len"],
            d_model=shape["d_model"],
            num_heads=shape["heads"],
            ffn_dim=shape["ffn_dim"],
            num_layers=shape["layers"],
            causal=shape["causal"],
        )
        return DispatchingTransformer(config), config

    def test_extreme_backend_is_eligible_to_drop_a_prefix_key_mask(self):
        from src.dispatcher import EXTREME_MEMORY_BACKEND, MaskKind

        model, _ = self._model(6)
        self.assertTrue(
            model._may_drop_key_mask(
                self._route(EXTREME_MEMORY_BACKEND, 6), MaskKind.PREFIX
            )
        )

    def test_extreme_backend_keeps_a_non_prefix_key_mask(self):
        from src.dispatcher import EXTREME_MEMORY_BACKEND, MaskKind

        model, _ = self._model(6)
        self.assertFalse(
            model._may_drop_key_mask(
                self._route(EXTREME_MEMORY_BACKEND, 6), MaskKind.GENERAL
            )
        )

    def test_reference_backend_never_drops_the_key_mask(self):
        from src.dispatcher import MaskKind, REFERENCE_BACKEND

        model, _ = self._model(6)
        self.assertFalse(
            model._may_drop_key_mask(
                self._route(REFERENCE_BACKEND, 6), MaskKind.PREFIX
            )
        )

    def test_extreme_route_does_not_inherit_a_stale_flag(self):
        model, _ = self._model(6)
        # Simulate a previous forward that enabled the flag.
        model._set_drop_key_mask(True)
        for layer in model.layers:
            self.assertTrue(layer.attention.drop_key_mask)
        model._set_drop_key_mask(False)
        for layer in model.layers:
            self.assertFalse(layer.attention.drop_key_mask)

    def test_case_6_output_is_unchanged_by_dropping_a_prefix_key_mask(self):
        """The property that makes the optimization safe, on case 6's shape."""
        from torch_transformer_benchmark import TransformerConfig
        from src.implementations.sdpa import StridedSDPASelfAttention
        from torch_transformer_benchmark import BaselineTransformer
        from src.infra import load_official_cases

        shape = load_official_cases()[6].benchmark_config()
        config = TransformerConfig(
            batch_size=4,
            seq_len=shape["seq_len"],
            d_model=shape["d_model"],
            num_heads=shape["heads"],
            ffn_dim=shape["ffn_dim"],
            num_layers=shape["layers"],
            causal=shape["causal"],
        )
        torch.manual_seed(0)
        model = BaselineTransformer(config)
        for layer in model.layers:
            layer.attention = StridedSDPASelfAttention(
                config.d_model, config.num_heads
            )
        model.eval()

        x = torch.randn(config.batch_size, config.seq_len, config.d_model)
        mask = self._prefix_mask(
            config.batch_size, config.seq_len, config.seq_len // 2
        )
        with torch.no_grad():
            for layer in model.layers:
                layer.attention.drop_key_mask = False
            kept = model(x, mask)
            for layer in model.layers:
                layer.attention.drop_key_mask = True
            dropped = model(x, mask)

        self.assertTrue(
            torch.equal(kept, dropped),
            f"max deviation {(kept - dropped).abs().max().item():.3e}",
        )


if __name__ == "__main__":
    unittest.main()
