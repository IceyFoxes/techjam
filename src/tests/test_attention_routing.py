from __future__ import annotations

import unittest

from src.implementations.attention_routing import MaskKind, Route, select_route


class SelectRouteTests(unittest.TestCase):
    EXACT_ROUTES = (Route.EXACT_EAGER, Route.EXACT_EAGER_PREFIX)

    def test_non_float32_always_takes_an_exact_route(self) -> None:
        for causal in (True, False):
            for kind in MaskKind:
                self.assertIn(
                    select_route(False, causal, kind),
                    self.EXACT_ROUTES,
                    msg=f"causal={causal} kind={kind}",
                )

    def test_non_float32_skips_the_padding_mask_only_when_provably_redundant(
        self,
    ) -> None:
        # Redundant only under causal attention with a prefix mask.
        self.assertIs(
            select_route(False, True, MaskKind.PREFIX), Route.EXACT_EAGER_PREFIX
        )
        # A general mask must still be applied, even under causal attention.
        self.assertIs(
            select_route(False, True, MaskKind.GENERAL), Route.EXACT_EAGER
        )
        # Without causal there is no upper triangle to subsume the padding.
        self.assertIs(
            select_route(False, False, MaskKind.PREFIX), Route.EXACT_EAGER
        )
        self.assertIs(
            select_route(False, False, MaskKind.GENERAL), Route.EXACT_EAGER
        )

    def test_absent_mask_uses_plain_causal_sdpa(self) -> None:
        self.assertIs(select_route(True, True, MaskKind.ABSENT), Route.SDPA_CAUSAL)
        self.assertIs(select_route(True, False, MaskKind.ABSENT), Route.SDPA_CAUSAL)

    def test_causal_prefix_mask_drops_the_mask_by_default(self) -> None:
        self.assertIs(select_route(True, True, MaskKind.PREFIX), Route.SDPA_CAUSAL)

    def test_keymask_route_is_never_selected(self) -> None:
        """It is the construction default only, not a routing alternative.

        ``SDPA_CAUSAL_KEYMASK`` reproduces upstream ``sdpa.py`` behavior and
        measured slower on every in-scope case. Nothing should route to it.
        """
        for is_f32 in (True, False):
            for causal in (True, False):
                for kind in MaskKind:
                    self.assertIsNot(
                        select_route(is_f32, causal, kind),
                        Route.SDPA_CAUSAL_KEYMASK,
                        msg=f"f32={is_f32} causal={causal} kind={kind}",
                    )

    def test_causal_general_mask_needs_the_full_mask(self) -> None:
        self.assertIs(select_route(True, True, MaskKind.GENERAL), Route.SDPA_FULLMASK)

    def test_non_causal_masked_input_uses_the_broadcast_key_mask(self) -> None:
        self.assertIs(select_route(True, False, MaskKind.PREFIX), Route.SDPA_KEYMASK)
        self.assertIs(select_route(True, False, MaskKind.GENERAL), Route.SDPA_KEYMASK)

    def test_every_combination_resolves_to_a_route(self) -> None:
        for is_f32 in (True, False):
            for causal in (True, False):
                for kind in MaskKind:
                    self.assertIsInstance(select_route(is_f32, causal, kind), Route)


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ClassifyMaskTests(unittest.TestCase):
    def test_none_is_absent(self) -> None:
        from src.implementations.attention_routing import classify_mask

        self.assertIs(classify_mask(None), MaskKind.ABSENT)

    def test_all_true_mask_is_prefix(self) -> None:
        from src.implementations.attention_routing import classify_mask

        self.assertIs(classify_mask(torch.ones(3, 5, dtype=torch.bool)), MaskKind.PREFIX)

    def test_right_padded_mask_is_prefix(self) -> None:
        from src.implementations.attention_routing import classify_mask

        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        self.assertIs(classify_mask(mask), MaskKind.PREFIX)

    def test_left_padded_mask_is_general(self) -> None:
        from src.implementations.attention_routing import classify_mask

        self.assertIs(
            classify_mask(torch.tensor([[False, True, True, True]])), MaskKind.GENERAL
        )

    def test_interior_gap_is_general(self) -> None:
        from src.implementations.attention_routing import classify_mask

        self.assertIs(
            classify_mask(torch.tensor([[True, False, True, False]])), MaskKind.GENERAL
        )

    def test_one_mixed_row_makes_the_batch_general(self) -> None:
        from src.implementations.attention_routing import classify_mask

        mask = torch.tensor(
            [[True, True, False], [True, True, False], [False, True, True]]
        )
        self.assertIs(classify_mask(mask), MaskKind.GENERAL)

    def test_single_position_is_prefix(self) -> None:
        from src.implementations.attention_routing import classify_mask

        self.assertIs(
            classify_mask(torch.zeros(2, 1, dtype=torch.bool)), MaskKind.PREFIX
        )

    def test_matches_the_harness_generator(self) -> None:
        from src.implementations.attention_routing import classify_mask
        from torch_transformer_benchmark import TransformerConfig, generate_random_case

        config = TransformerConfig(
            batch_size=8, seq_len=16, d_model=8, num_heads=2,
            ffn_dim=8, num_layers=1, causal=True,
        )
        for ratio in (0.0, 0.3, 0.9):
            _, mask = generate_random_case(
                config, torch.device("cpu"), torch.float32, 7, ratio, 1.0
            )
            self.assertIs(classify_mask(mask), MaskKind.PREFIX, msg=f"ratio={ratio}")


if __name__ == "__main__":
    unittest.main()
