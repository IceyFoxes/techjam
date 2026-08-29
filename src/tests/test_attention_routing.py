from __future__ import annotations

import unittest

from src.implementations.attention_routing import MaskKind, Route, select_route


class SelectRouteTests(unittest.TestCase):
    def test_non_float32_always_takes_the_exact_eager_route(self) -> None:
        for causal in (True, False):
            for kind in MaskKind:
                self.assertIs(
                    select_route(False, causal, kind),
                    Route.EXACT_EAGER,
                    msg=f"causal={causal} kind={kind}",
                )

    def test_absent_mask_uses_plain_causal_sdpa(self) -> None:
        self.assertIs(select_route(True, True, MaskKind.ABSENT), Route.SDPA_CAUSAL)
        self.assertIs(select_route(True, False, MaskKind.ABSENT), Route.SDPA_CAUSAL)

    def test_causal_prefix_mask_drops_the_mask_by_default(self) -> None:
        self.assertIs(select_route(True, True, MaskKind.PREFIX), Route.SDPA_CAUSAL)

    def test_causal_prefix_mask_can_keep_the_key_mask(self) -> None:
        self.assertIs(
            select_route(True, True, MaskKind.PREFIX, prefer_keymask=True),
            Route.SDPA_CAUSAL_KEYMASK,
        )

    def test_prefer_keymask_is_ignored_when_it_would_change_semantics(self) -> None:
        # The preference is a measurement control for the causal+prefix case only.
        self.assertIs(
            select_route(True, True, MaskKind.GENERAL, prefer_keymask=True),
            Route.SDPA_FULLMASK,
        )
        self.assertIs(
            select_route(True, False, MaskKind.PREFIX, prefer_keymask=True),
            Route.SDPA_KEYMASK,
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


if __name__ == "__main__":
    unittest.main()
