"""Attention route selection for the Person 2 candidate.

The routing decision is deliberately separated from the attention module and
from torch. It is a pure function of three booleans-worth of state, so the whole
table is testable without a GPU, and Person 1 can call it to hoist the decision
above a compiled region as ``dispatcher-strategy.md`` requires.
"""

from __future__ import annotations

import enum
from typing import Any, Optional


class MaskKind(enum.Enum):
    """How much structure a ``valid_token_mask`` has."""

    ABSENT = "absent"
    PREFIX = "prefix"
    GENERAL = "general"


class Route(enum.Enum):
    """An attention implementation. All routes compute the same function."""

    SDPA_CAUSAL = "sdpa_causal"
    SDPA_CAUSAL_KEYMASK = "sdpa_causal_keymask"
    SDPA_KEYMASK = "sdpa_keymask"
    SDPA_FULLMASK = "sdpa_fullmask"
    EXACT_EAGER = "exact_eager"


def select_route(
    is_float32: bool,
    causal: bool,
    mask_kind: MaskKind,
    prefer_keymask: bool = False,
) -> Route:
    """Choose the attention implementation for one forward pass.

    ``prefer_keymask`` forces the upstream-equivalent route that keeps the
    broadcast key mask. The two causal routes are numerically identical;
    dropping the mask measured faster on all twelve in-scope cases at both
    padding ratios, so this exists only to reproduce upstream behavior as an
    A/B control, not as a routing alternative. It is ignored wherever the two
    routes would not agree.
    """
    if not is_float32:
        # float16 SDPA fails the pass criterion on 0/8 seeds for case 13. The
        # cause is the reference rounding probabilities to float16 before PV,
        # which a fused kernel does not reproduce. See sdpa-and-precision.md.
        return Route.EXACT_EAGER

    if mask_kind is MaskKind.ABSENT:
        return Route.SDPA_CAUSAL

    if not causal:
        # Without an upper-triangular mask to subsume it, the padding mask has
        # to be applied.
        return Route.SDPA_KEYMASK

    if mask_kind is MaskKind.PREFIX:
        # Causal masking already sets -inf everywhere a right-padding mask
        # would, so the mask is removable. Verified bitwise; see spec section 3.
        # Dropping it is also faster on every in-scope case, so the keymask
        # route is only ever selected explicitly, as a measurement control.
        return Route.SDPA_CAUSAL_KEYMASK if prefer_keymask else Route.SDPA_CAUSAL

    return Route.SDPA_FULLMASK
