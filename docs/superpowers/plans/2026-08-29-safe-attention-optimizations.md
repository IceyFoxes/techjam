# Safe Attention Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the eager attention core in Person 2's candidate with a routed `scaled_dot_product_attention` implementation that gains on every in-scope case and never loses correctness in any dtype.

**Architecture:** Person 1 and Person 3 have already landed strided-view SDPA in the shared `src/implementations/sdpa.py`, wired into a compiling `src/dispatcher.py`. Person 2's remaining contribution is narrower and specific: their `StridedSDPASelfAttention` **always** supplies the broadcast key mask, and under causal attention that mask is provably dead code. Dropping it measured faster in **24 of 24** comparisons on cu130 — 12 cases x 2 padding ratios — by ~2-5% on the large shapes and more on the small ones. This plan builds a mask-routed subclass of their module, proves the mask is removable, validates the change, and hands them a one-line diff. Non-float32 dtypes fall to a bitwise-exact eager route, because float16 SDPA fails the pass criterion for reasons rooted in the reference's own rounding.

**Tech Stack:** Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA 13.0, `unittest`, the repo's `src.benchmark` harness and `src/infra/timing.py` paired timer. This machine was migrated to the team-pinned stack on 30 August 2026; `.venv` is now that stack and `.venv-cu124-old` holds the previous torch 2.6.0+cu124 environment.

**Upstream state this plan builds on (read before starting):**
- `src/implementations/sdpa.py` — `StridedSDPASelfAttention` (Person 1) and `PackedQKVSDPASelfAttention` (Person 3). Shared module; **do not edit it in this plan.**
- `src/dispatcher.py` — per-case compile modes, `reduce-overhead` for cases 1-12, `default` for case 13, reference fallback otherwise.
- `src/implementations/attention.py` — still the Person 2 reference scaffold. This is the file this plan replaces.

**Spec:** [`research/attention-softmax/safe-optimization-spec.md`](../../../research/attention-softmax/safe-optimization-spec.md)

## Global Constraints

- Pass criterion is `abs_error <= 0.002 OR rel_error <= 0.02 * |reference|` with **zero** failed elements. No averaging, no percentile.
- **Never modify** the root `torch_transformer_benchmark.py`. All working code lives under `src/`.
- Implementation changes go on a branch, never directly on `master`. The current branch is `att-snx-ker`.
- Parameter names and module structure must stay reference-compatible so `copy_model_weights(..., strict=True)` loads without a custom `weight_loader`.
- **Exactly one** device-to-host synchronization per forward, above the layer loop. A per-layer sync turned a measured 1.15-1.43x gain into a 0.82-0.95x loss.
- **The host sync must never occur inside a compiled or graph-replayed region.** `sdpa.py` carries an explicit upstream warning: *"Do not inspect mask values on the host: that would synchronize and break graph replay."* The dispatcher runs cases 1-12 under `reduce-overhead` (CUDA graphs). Any mask classification therefore belongs in the **uncompiled** dispatch layer, before the compiled callable is invoked — never inside the model that gets captured. This is the single hardest constraint in the plan; a design that ignores it will appear to work and then silently produce stale results under replay.
- **Do not edit `src/implementations/sdpa.py`.** It is now shared by Persons 1 and 3. Person 2 extends it by subclassing from `src/implementations/attention.py`.
- The sync costs roughly 10-20 us. On case 2 (~0.14 ms compiled) that is ~10% and may not pay for itself; on case 13 (~25-60 ms) it is noise. This is exactly why Task 7 decides per case rather than globally.
- No unconditional `.contiguous()` at the head-reshape boundary (Person 3 contract, `qkv-layout.md`).
- Only `src/implementations/attention.py` and uniquely-named helper modules may be created or changed. Do **not** edit `src/dispatcher.py`, `src/implementations/sdpa.py`, `src/implementations/compiler.py`, `src/implementations/projections.py`, or `src/implementations/extreme.py`.
- **`src.dispatcher` is now measurable on this machine.** As of 30 August 2026 this host runs the pinned `2.13.0+cu130` on driver 616.56 / CUDA 13.4, so the dispatcher's version and capability gates both pass and it selects `compiled-sdpa` (`reduce-overhead` for cases 1-5 and 7-12, `default` for 13) rather than falling back to reference. Task 7 therefore measures the integrated route as well as the standalone candidate.
- Tests must run without a GPU: guard torch-dependent tests with `unittest.skipIf`, and use small CPU shapes.
- Case 14 is out of scope. Case 6 is correctness-and-memory only; its latency is not a claim.

---

### Task 1: Routing primitives

Pure decision logic, no torch, no GPU. Isolating it means the routing table is testable in milliseconds and Person 1 can import it to hoist the decision above a compiled region.

**Files:**
- Create: `src/implementations/attention_routing.py`
- Test: `src/tests/test_attention_routing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MaskKind` (enum: `ABSENT`, `PREFIX`, `GENERAL`), `Route` (enum: `SDPA_CAUSAL`, `SDPA_CAUSAL_KEYMASK`, `SDPA_KEYMASK`, `SDPA_FULLMASK`, `EXACT_EAGER`), and `select_route(is_float32: bool, causal: bool, mask_kind: MaskKind, prefer_keymask: bool = False) -> Route`.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_attention_routing.py`:

```python
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
        self.assertIs(
            select_route(True, True, MaskKind.ABSENT), Route.SDPA_CAUSAL
        )
        self.assertIs(
            select_route(True, False, MaskKind.ABSENT), Route.SDPA_CAUSAL
        )

    def test_causal_prefix_mask_drops_the_mask_by_default(self) -> None:
        self.assertIs(
            select_route(True, True, MaskKind.PREFIX), Route.SDPA_CAUSAL
        )

    def test_causal_prefix_mask_can_keep_the_key_mask(self) -> None:
        self.assertIs(
            select_route(True, True, MaskKind.PREFIX, prefer_keymask=True),
            Route.SDPA_CAUSAL_KEYMASK,
        )

    def test_prefer_keymask_is_ignored_when_it_would_change_semantics(self) -> None:
        # The preference is a performance knob for the causal+prefix case only.
        self.assertIs(
            select_route(True, True, MaskKind.GENERAL, prefer_keymask=True),
            Route.SDPA_FULLMASK,
        )
        self.assertIs(
            select_route(True, False, MaskKind.PREFIX, prefer_keymask=True),
            Route.SDPA_KEYMASK,
        )

    def test_causal_general_mask_needs_the_full_mask(self) -> None:
        self.assertIs(
            select_route(True, True, MaskKind.GENERAL), Route.SDPA_FULLMASK
        )

    def test_non_causal_masked_input_uses_the_broadcast_key_mask(self) -> None:
        self.assertIs(
            select_route(True, False, MaskKind.PREFIX), Route.SDPA_KEYMASK
        )
        self.assertIs(
            select_route(True, False, MaskKind.GENERAL), Route.SDPA_KEYMASK
        )

    def test_every_combination_resolves_to_a_route(self) -> None:
        for is_f32 in (True, False):
            for causal in (True, False):
                for kind in MaskKind:
                    self.assertIsInstance(
                        select_route(is_f32, causal, kind), Route
                    )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.tests.test_attention_routing -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.implementations.attention_routing'`

- [ ] **Step 3: Write minimal implementation**

Create `src/implementations/attention_routing.py`:

```python
"""Attention route selection for the Person 2 candidate.

The routing decision is deliberately separated from the attention module and
from torch. It is a pure function of three booleans-worth of state, so the whole
table is testable without a GPU, and Person 1 can call it to hoist the decision
above a compiled region as ``dispatcher-strategy.md`` requires.
"""

from __future__ import annotations

import enum


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.tests.test_attention_routing -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/implementations/attention_routing.py src/tests/test_attention_routing.py
git commit -m "feat(attention): add pure route-selection table"
```

---

### Task 2: Mask classification

The one host synchronization. It lives alone so the sync is auditable in a single place.

**Files:**
- Modify: `src/implementations/attention_routing.py`
- Test: `src/tests/test_attention_routing.py`

**Interfaces:**
- Consumes: `MaskKind` from Task 1.
- Produces: `classify_mask(valid_token_mask: Optional["torch.Tensor"]) -> MaskKind`.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/test_attention_routing.py`, above the `if __name__` block:

```python
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

        mask = torch.ones(3, 5, dtype=torch.bool)
        self.assertIs(classify_mask(mask), MaskKind.PREFIX)

    def test_right_padded_mask_is_prefix(self) -> None:
        from src.implementations.attention_routing import classify_mask

        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        self.assertIs(classify_mask(mask), MaskKind.PREFIX)

    def test_left_padded_mask_is_general(self) -> None:
        from src.implementations.attention_routing import classify_mask

        mask = torch.tensor([[False, True, True, True]])
        self.assertIs(classify_mask(mask), MaskKind.GENERAL)

    def test_interior_gap_is_general(self) -> None:
        from src.implementations.attention_routing import classify_mask

        mask = torch.tensor([[True, False, True, False]])
        self.assertIs(classify_mask(mask), MaskKind.GENERAL)

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
        from torch_transformer_benchmark import (
            TransformerConfig,
            generate_random_case,
        )

        config = TransformerConfig(
            batch_size=8, seq_len=16, d_model=8, num_heads=2,
            ffn_dim=8, num_layers=1, causal=True,
        )
        for ratio in (0.0, 0.3, 0.9):
            _, mask = generate_random_case(
                config, torch.device("cpu"), torch.float32, 7, ratio, 1.0
            )
            self.assertIs(
                classify_mask(mask), MaskKind.PREFIX, msg=f"ratio={ratio}"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.tests.test_attention_routing -v`
Expected: FAIL with `ImportError: cannot import name 'classify_mask'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/implementations/attention_routing.py`, after the imports:

```python
from typing import Any, Optional
```

and append at the end of the file:

```python
def classify_mask(valid_token_mask: Optional[Any]) -> MaskKind:
    """Classify a ``[B, N]`` boolean mask. Performs ONE host synchronization.

    A mask is ``PREFIX`` when it never rises along the sequence axis, which is
    exactly the right-padded shape ``generate_random_case`` produces. Callers
    must invoke this once per forward, above the layer loop: evaluating an
    equivalent predicate per layer was measured to turn a 1.15-1.43x gain into a
    0.82-0.95x loss.
    """
    if valid_token_mask is None:
        return MaskKind.ABSENT

    if valid_token_mask.shape[-1] <= 1:
        # A single position cannot rise, so it is trivially prefix-valid.
        return MaskKind.PREFIX

    non_increasing = bool(
        (valid_token_mask[:, :-1] >= valid_token_mask[:, 1:]).all()
    )
    return MaskKind.PREFIX if non_increasing else MaskKind.GENERAL
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.tests.test_attention_routing -v`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add src/implementations/attention_routing.py src/tests/test_attention_routing.py
git commit -m "feat(attention): classify masks with a single host sync"
```

---

### Task 3: The mask-routed attention module

Extends Person 1's `StridedSDPASelfAttention` rather than reimplementing it. The strided views, the scale, the output reshape and the output zeroing are all theirs and already validated; the only thing this subclass changes is **which attention mask is supplied**.

**Files:**
- Modify: `src/implementations/attention.py`
- Test: `src/tests/test_attention_candidate.py`
- Read first (do not edit): `src/implementations/sdpa.py`

**Interfaces:**
- Consumes: `MaskKind`, `Route`, `select_route`, `classify_mask` (Tasks 1-2); `StridedSDPASelfAttention` from `src.implementations.sdpa`.
- Produces: `MaskRoutedSDPASelfAttention(d_model: int, num_heads: int)` with attribute `route: Route` (default `Route.SDPA_CAUSAL_KEYMASK`, i.e. upstream behavior) and the reference `forward(x, valid_token_mask=None, causal=False)` signature.

The default is deliberately upstream's behavior, so that constructing the module changes nothing until a route is explicitly selected.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_attention_candidate.py`:

```python
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

        module = MaskRoutedSDPASelfAttention(16, 4)
        self.assertIs(module.route, Route.SDPA_CAUSAL_KEYMASK)
        self.assertIsInstance(module, StridedSDPASelfAttention)

        upstream = StridedSDPASelfAttention(16, 4).eval()
        upstream.load_state_dict(module.state_dict())
        x, mask = self._inputs(lengths=[5, 8])
        with torch.inference_mode():
            delta = (
                upstream(x, mask, True) - module.eval()(x, mask, True)
            ).abs().max()
        self.assertEqual(delta.item(), 0.0)

    def test_keeps_the_reference_parameter_contract(self) -> None:
        reference, candidate = self._pair()
        self.assertEqual(
            list(reference.state_dict()), list(candidate.state_dict())
        )

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
            self._assert_within_tolerance(
                kept(x, mask, True), dropped(x, mask, True)
            )

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
        ):
            with self.subTest(route=route):
                reference, candidate = self._pair()
                candidate.route = route
                with torch.inference_mode():
                    self._assert_within_tolerance(
                        reference(x, mask, causal), candidate(x, mask, causal)
                    )

    def test_exact_eager_route_is_bitwise_identical(self) -> None:
        """The EXACT_EAGER route must be exact, not merely within tolerance."""
        from src.implementations.attention_routing import Route

        x, mask = self._inputs(lengths=[5, 8])
        reference, candidate = self._pair()
        candidate.route = Route.EXACT_EAGER
        with torch.inference_mode():
            delta = (reference(x, mask, True) - candidate(x, mask, True)).abs().max()
        self.assertEqual(delta.item(), 0.0)

    def test_head_views_avoid_copies(self) -> None:
        """Lever L9, inherited from sdpa.py: Q/K/V reach SDPA as views."""
        from src.implementations.attention import MaskRoutedSDPASelfAttention

        module = MaskRoutedSDPASelfAttention(16, 4)
        projected = torch.randn(2, 8, 16)
        view = module._split_heads_view(projected)
        self.assertEqual(tuple(view.shape), (2, 4, 8, 4))
        self.assertFalse(view.is_contiguous())
        self.assertEqual(
            view.untyped_storage().data_ptr(),
            projected.untyped_storage().data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.tests.test_attention_candidate -v`
Expected: FAIL with `ImportError: cannot import name 'MaskRoutedSDPASelfAttention'`

- [ ] **Step 3: Write minimal implementation**

Replace the whole of `src/implementations/attention.py` with:

```python
"""Person 2 candidate: attention mask routing.

Person 1's ``src/implementations/sdpa.py`` already supplies strided-view SDPA.
It always passes the broadcast key mask, which is correct but not always fast:
measured on case 13, dropping the mask is 4.2% quicker at ``padding_ratio=0``
and 2.5x quicker at ``padding_ratio=0.3``, while case 11 prefers keeping it.

This module adds the route choice on top of theirs. Under causal attention a
right-padded key mask is provably dead code -- causal masking already writes
-inf everywhere the padding mask would -- so the two causal routes compute the
same function and may be selected on speed alone. See
``research/attention-softmax/safe-optimization-spec.md`` section 3.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineSelfAttention

from src.implementations.attention_routing import Route
from src.implementations.sdpa import StridedSDPASelfAttention


class MaskRoutedSDPASelfAttention(StridedSDPASelfAttention):
    """Strided SDPA whose attention mask is chosen by ``route``.

    ``route`` is assigned by the owning model once per forward. This module
    never classifies the mask itself: that costs a host synchronization, which
    both breaks CUDA-graph replay and, per layer, costs more than it saves.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        # Default to upstream behavior so constructing this changes nothing.
        self.route = Route.SDPA_CAUSAL_KEYMASK
        self._causal_masks: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    def _blocked_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Cached strict upper triangle. True marks a blocked position (L6)."""
        key = (seq_len, device)
        cached = self._causal_masks.get(key)
        if cached is None:
            cached = torch.ones(
                (seq_len, seq_len), device=device, dtype=torch.bool
            ).triu(1)
            self._causal_masks[key] = cached
        return cached

    def _eager_exact(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """Reference arithmetic, minus provably dead work. Bitwise exact.

        Uses the reference's own contiguous ``_split_heads`` rather than the
        strided view, because matmul may select a different kernel for strided
        inputs and this route's whole purpose is bitwise agreement.
        """
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            scores = scores.masked_fill(
                self._blocked_causal(seq_len, x.device), float("-inf")
            )
        # Under causal attention a right-padded key mask only writes -inf onto
        # positions the causal mask already blocked. Skipping it is exact.
        if valid_token_mask is not None and not causal:
            scores = scores.masked_fill(
                ~valid_token_mask[:, None, None, :], float("-inf")
            )
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        route = self.route

        if route is Route.EXACT_EAGER:
            return self._eager_exact(x, valid_token_mask, causal)

        if route is Route.SDPA_CAUSAL_KEYMASK:
            # Exactly upstream's behavior.
            return super().forward(x, valid_token_mask, causal)

        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        attn_mask = None
        is_causal = causal
        if route is Route.SDPA_KEYMASK:
            attn_mask = valid_token_mask[:, None, None, :]
            is_causal = False
        elif route is Route.SDPA_FULLMASK:
            keep = valid_token_mask[:, None, None, :]
            if causal:
                keep = keep & ~self._blocked_causal(seq_len, x.device)
            attn_mask = keep
            is_causal = False

        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.tests.test_attention_candidate -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/implementations/attention.py src/tests/test_attention_candidate.py
git commit -m "feat(attention): add mask-routed SDPA on top of the shared strided module"
```

---

### Task 4: Model wiring and the candidate export

**Files:**
- Modify: `src/implementations/attention.py`
- Test: `src/tests/test_attention_candidate.py`

**Interfaces:**
- Consumes: `MaskRoutedSDPASelfAttention` (Task 3), `classify_mask` / `select_route` (Tasks 1-2).
- Produces: `AttentionCandidate(config, prefer_keymask: bool = False)` and `CANDIDATE: CandidateSpec` named `"attention"`, plus `KEYMASK_CANDIDATE` for A/B.

**Where the sync lives.** `AttentionCandidate.forward` is the uncompiled dispatch layer for this candidate: it classifies the mask on the host, pushes the route into each layer, then runs the reference forward. Under `torch.compile` this whole method would be traced, so `classify_mask` would become a graph break or a captured constant. That is acceptable for this candidate, which is measured eagerly — but it is exactly why Task 8's handoff tells Person 1 to call `classify_mask` in `DispatchingTransformer.forward` *before* invoking the compiled callable, never inside it.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/test_attention_candidate.py`, above the `if __name__` block:

```python
@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionCandidateTests(unittest.TestCase):
    def _config(self, causal=True):
        from torch_transformer_benchmark import TransformerConfig

        return TransformerConfig(
            batch_size=2, seq_len=8, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=causal,
        )

    def _models(self, config, prefer_keymask=False):
        from torch_transformer_benchmark import (
            BaselineTransformer,
            copy_model_weights,
        )
        from src.implementations.attention import AttentionCandidate

        torch.manual_seed(3)
        baseline = BaselineTransformer(config).eval()
        candidate = AttentionCandidate(
            config, prefer_keymask=prefer_keymask
        ).eval()
        copy_model_weights(baseline, candidate, strict=True)
        return baseline, candidate

    def test_strict_weight_copy_succeeds(self) -> None:
        config = self._config()
        baseline, candidate = self._models(config)
        self.assertEqual(
            list(baseline.state_dict()), list(candidate.state_dict())
        )

    def test_matches_reference_across_padding_ratios(self) -> None:
        from torch_transformer_benchmark import (
            compare_outputs,
            generate_random_case,
        )

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
                    self.assertTrue(
                        result.passed, msg=f"{result.failed_elements} failed"
                    )

    def test_non_causal_config_matches_reference(self) -> None:
        from torch_transformer_benchmark import (
            compare_outputs,
            generate_random_case,
        )

        config = self._config(causal=False)
        baseline, candidate = self._models(config)
        x, mask = generate_random_case(
            config, torch.device("cpu"), torch.float32, 13, 0.3, 1.0
        )
        with torch.inference_mode():
            result = compare_outputs(
                baseline(x, mask), candidate(x, mask), 0.02, 0.002
            )
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

        config = self._config()
        _, candidate = self._models(config)
        x = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        with torch.inference_mode():
            candidate(x, mask)
        for layer in candidate.layers:
            self.assertIs(layer.attention.route, Route.SDPA_CAUSAL)

    def test_both_candidate_specs_are_loadable(self) -> None:
        from src.infra import load_candidate

        spec = load_candidate("attention")
        self.assertEqual(spec.name, "attention")
        keymask = load_candidate("attention:KEYMASK_CANDIDATE")
        self.assertEqual(keymask.name, "attention-keymask")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ExactEagerDtypeTests(unittest.TestCase):
    def test_reduced_precision_routes_are_bitwise_identical(self) -> None:
        """float16/bfloat16 fall to EXACT_EAGER and reproduce the reference exactly."""
        from torch_transformer_benchmark import (
            BaselineTransformer,
            TransformerConfig,
            copy_model_weights,
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
                        delta = (
                            baseline(x, mask) - candidate(x, mask)
                        ).abs().max()
                    self.assertEqual(delta.item(), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest src.tests.test_attention_candidate -v`
Expected: FAIL with `ImportError: cannot import name 'AttentionCandidate'`

(The name `AttentionCandidate` already exists in the scaffold, so the failure may instead be a `TypeError` on the `prefer_keymask` argument. Either failure is the expected red state.)

- [ ] **Step 3: Write minimal implementation**

In `src/implementations/attention.py`, extend the imports:

```python
from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
)

from src.infra import CandidateSpec
from src.implementations.attention_routing import (
    MaskKind,
    Route,
    classify_mask,
    select_route,
)
```

and append at the end of the file:

```python
class AttentionCandidate(BaselineTransformer):
    """Reference Transformer whose attention cores are mask-routed.

    The mask is classified once here, above the layer loop, and the resulting
    route is pushed into every layer before it runs. This method is the
    uncompiled dispatch layer: the host synchronization must stay here and
    never move inside a compiled or graph-replayed region.
    """

    def __init__(
        self, config: TransformerConfig, prefer_keymask: bool = False
    ) -> None:
        super().__init__(config)
        self.prefer_keymask = prefer_keymask
        for layer in self.layers:
            layer.attention = MaskRoutedSDPASelfAttention(
                config.d_model, config.num_heads
            )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The single host synchronization for this forward pass.
        route = select_route(
            x.dtype == torch.float32,
            self.config.causal,
            classify_mask(valid_token_mask),
            prefer_keymask=self.prefer_keymask,
        )
        for layer in self.layers:
            layer.attention.route = route
        return super().forward(x, valid_token_mask)


def _keymask_factory(config: TransformerConfig) -> AttentionCandidate:
    return AttentionCandidate(config, prefer_keymask=True)


CANDIDATE = CandidateSpec(
    name="attention",
    model_factory=AttentionCandidate,
    owner="Person 2",
    description="Mask-routed float32 SDPA with an exact eager fallback.",
)

KEYMASK_CANDIDATE = CandidateSpec(
    name="attention-keymask",
    model_factory=_keymask_factory,
    owner="Person 2",
    description="Mask-routed SDPA retaining the broadcast key mask under causal attention.",
)
```

`MaskKind` is imported for re-export so Person 1 can consume the whole routing
vocabulary from this module. Keep the import even though this file does not
reference it directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest src.tests.test_attention_candidate -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Confirm nothing upstream regressed**

```bash
python3 -m unittest discover -s src/tests -v
python3 -m compileall -q src
python3 -m src.benchmark --candidate attention --device cpu \
  --batch-size 1 --seq-len 8 --d-model 16 --heads 4 --ffn-dim 32 --layers 1 \
  --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1
```

Expected: the pre-existing `test_dispatcher` and `test_projections` suites still
pass — this plan must not disturb them — and the benchmark reports a PASS.

- [ ] **Step 6: Commit**

```bash
git add src/implementations/attention.py src/tests/test_attention_candidate.py
git commit -m "feat(attention): wire the mask-routed candidate with an A/B selector"
```

---

### Task 5: Lock in the dead-mask proof as a regression test

Spec section 3 rests on a property of the harness's mask generator. If a future harness change breaks it, this must fail loudly rather than silently produce wrong numbers.

**Files:**
- Create: `src/tests/test_padding_mask_redundancy.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Deliberately independent — it tests the *reference*, not our code.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Create `src/tests/test_padding_mask_redundancy.py`:

```python
"""The padding key mask is dead code under causal attention.

Spec section 3 removes the padding ``masked_fill`` on the strength of an
argument about the reference's own semantics: causal masking already writes
-inf everywhere a right-padded key mask would, and rows that genuinely differ
are zeroed by the reference before they can propagate.

This test pins that property to the reference implementation itself, with the
padding mask as the only variable. If the harness ever emits a mask that is not
right-padded, or stops zeroing invalid rows, this fails.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover - dependency-free environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PaddingMaskRedundancyTests(unittest.TestCase):
    def _no_pad_mask_model(self, config):
        from torch_transformer_benchmark import (
            BaselineSelfAttention,
            BaselineTransformer,
        )

        class NoPadMaskAttention(BaselineSelfAttention):
            """Reference arithmetic with the padding masked_fill removed."""

            def forward(self, x, valid_token_mask=None, causal=False):
                batch, seq_len, _ = x.shape
                q = self._split_heads(self.q_proj(x))
                k = self._split_heads(self.k_proj(x))
                v = self._split_heads(self.v_proj(x))
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                if causal:
                    blocked = torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).triu(1)
                    scores = scores.masked_fill(blocked, float("-inf"))
                probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
                context = torch.matmul(probs, v)
                context = (
                    context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
                )
                output = self.out_proj(context)
                if valid_token_mask is not None:
                    output = output.masked_fill(~valid_token_mask[..., None], 0)
                return output

        model = BaselineTransformer(config)
        for layer in model.layers:
            layer.attention = NoPadMaskAttention(config.d_model, config.num_heads)
        return model.eval()

    def test_removing_the_padding_mask_is_bitwise_exact(self) -> None:
        from torch_transformer_benchmark import (
            BaselineTransformer,
            TransformerConfig,
            copy_model_weights,
            generate_random_case,
        )

        config = TransformerConfig(
            batch_size=4, seq_len=16, d_model=16, num_heads=4,
            ffn_dim=16, num_layers=2, causal=True,
        )
        torch.manual_seed(19)
        reference = BaselineTransformer(config).eval()
        variant = self._no_pad_mask_model(config)
        copy_model_weights(reference, variant, strict=True)

        for ratio in (0.0, 0.3, 0.5, 0.9):
            for seed in (101, 202):
                with self.subTest(ratio=ratio, seed=seed):
                    x, mask = generate_random_case(
                        config, torch.device("cpu"), torch.float32,
                        seed, ratio, 1.0,
                    )
                    with torch.inference_mode():
                        delta = (
                            reference(x, mask) - variant(x, mask)
                        ).abs().max()
                    self.assertEqual(delta.item(), 0.0)

    def test_harness_masks_are_right_padded(self) -> None:
        """The precondition for the proof above."""
        from torch_transformer_benchmark import (
            TransformerConfig,
            generate_random_case,
        )

        config = TransformerConfig(
            batch_size=16, seq_len=32, d_model=8, num_heads=2,
            ffn_dim=8, num_layers=1, causal=True,
        )
        for ratio in (0.1, 0.3, 0.5, 0.9):
            for seed in (1, 2, 3):
                _, mask = generate_random_case(
                    config, torch.device("cpu"), torch.float32, seed, ratio, 1.0
                )
                rises = (mask[:, :-1] < mask[:, 1:]).any()
                self.assertFalse(
                    bool(rises), msg=f"ratio={ratio} seed={seed} is not right-padded"
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `python3 -m unittest src.tests.test_padding_mask_redundancy -v`
Expected: PASS, 2 tests.

This is a characterization test, so it passes on first write. That is correct and expected — it pins existing reference behavior that the spec depends on. If it *fails*, stop: spec section 3 is wrong and Task 3's `skip_padding_mask=causal` must be reverted to `False` before going further.

- [ ] **Step 3: Commit**

```bash
git add src/tests/test_padding_mask_redundancy.py
git commit -m "test(attention): pin the causal padding-mask redundancy property"
```

---

### Task 6: GPU correctness matrix

Spec section 6. Nothing here is a performance claim; this task establishes that the implementation is correct everywhere before any timing is trusted.

**Files:**
- Create: `src/tests/test_attention_matrix.py`

**Interfaces:**
- Consumes: `AttentionCandidate` (Task 4).
- Produces: nothing.

- [ ] **Step 1: Write the test**

Create `src/tests/test_attention_matrix.py`:

```python
"""GPU correctness matrix for the attention candidate (spec section 6).

Skipped without CUDA. Small-N cases only, so the suite stays minutes rather
than hours; the full official sweep is the benchmark task's job.
"""

from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

CASES = (1, 7, 9, 10, 11, 12)
RATIOS = (0.0, 0.3)
SCALES = (0.1, 1.0, 10.0)
SEEDS = (1234, 1235, 1236, 1237, 1238)


@unittest.skipIf(torch is None, "PyTorch is not installed")
@unittest.skipIf(
    torch is not None and not torch.cuda.is_available(), "CUDA is required"
)
class AttentionCorrectnessMatrixTests(unittest.TestCase):
    def test_matrix(self) -> None:
        from torch_transformer_benchmark import (
            BaselineTransformer,
            TransformerConfig,
            compare_outputs,
            copy_model_weights,
            generate_random_case,
        )
        from src.infra.cases import load_official_cases
        from src.implementations.attention import AttentionCandidate

        official = load_official_cases()
        device = torch.device("cuda")
        for case_id in CASES:
            case = official[case_id]
            config = TransformerConfig(
                batch_size=case.batch_size, seq_len=case.seq_len,
                d_model=case.qkv_dim, num_heads=case.heads,
                ffn_dim=case.ffn_dim, num_layers=case.layers,
                causal=case.causal,
            )
            baseline = BaselineTransformer(config).to(device, torch.float32).eval()
            candidate = AttentionCandidate(config).to(device, torch.float32).eval()
            copy_model_weights(baseline, candidate, strict=True)
            for ratio in RATIOS:
                for scale in SCALES:
                    for seed in SEEDS:
                        with self.subTest(
                            case=case_id, ratio=ratio, scale=scale, seed=seed
                        ):
                            x, mask = generate_random_case(
                                config, device, torch.float32, seed, ratio, scale
                            )
                            with torch.inference_mode():
                                result = compare_outputs(
                                    baseline(x, mask), candidate(x, mask),
                                    0.02, 0.002,
                                )
                            self.assertTrue(
                                result.passed,
                                msg=(
                                    f"case {case_id} ratio={ratio} scale={scale} "
                                    f"seed={seed}: {result.failed_elements} failed, "
                                    f"max_abs={result.max_abs_error}"
                                ),
                            )
            del baseline, candidate
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m unittest src.tests.test_attention_matrix -v`
Expected: PASS. 180 subtests across 6 cases.

If any cell fails, record the exact `(case, ratio, scale, seed)` and `max_abs` before changing anything — a failure here is evidence about the tolerance budget, and spec section 6 requires it to be reported, not silently fixed.

- [ ] **Step 3: Verify TF32-off behavior**

Run:

```bash
.venv/bin/python - <<'EOF'
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
import unittest
from src.tests import test_attention_matrix as m
unittest.main(module=m, argv=["x"], exit=False)
EOF
```

Expected: PASS, with a much smaller `max_abs` than the TF32-on run. Record both.

- [ ] **Step 4: Commit**

```bash
git add src/tests/test_attention_matrix.py
git commit -m "test(attention): add the GPU correctness matrix"
```

---

### Task 7: Validation sweep and preserved benchmarks

Spec sections 3 and 7. The route question is settled: dropping the mask was faster in 24 of 24 comparisons on cu130. This task is therefore a **validation sweep with a control**, not a per-case decision — it confirms the exploratory spike numbers under the official harness and preserves them as evidence.

**Files:**
- Create: `research/benchmarks/2026-08-30-rtx4060-<commit>/README.md` and one JSON per run
- Modify: `research/benchmarks/README.md`
- Modify: `research/attention-softmax/safe-optimization-spec.md` (replace the exploratory section 3 table with harness-produced figures)

**Interfaces:**
- Consumes: `AttentionCandidate(config, prefer_keymask=...)` and `KEYMASK_CANDIDATE` (Task 4).
- Produces: preserved per-case evidence confirming (or overturning) the section 3 table.

- [ ] **Step 1: Confirm the environment**

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())'
git rev-parse --short HEAD
```

Record both. Expected here: `2.13.0+cu130 13.0`, `NVIDIA GeForce RTX 4060 Laptop GPU`, driver 616.56 / CUDA 13.4.

Note the driver and toolkit are now identical to Person 1's RTX 5080 box, so
route decisions no longer differ by software stack — only by GPU.

- [ ] **Step 2: Run the sweep, both routes, in one process per case**

For each case in 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13 and each `--padding-ratio` in 0.0 and 0.3:

```bash
.venv/bin/python -m src.benchmark --candidate attention \
  --device cuda --dtype float32 --case <N> --padding-ratio <R> \
  --repeats 40 --settle-seconds 20 \
  --output research/benchmarks/2026-08-30-rtx4060-<commit>/case<N>-fp32-pr<R>-dropmask.json
```

**Critical, two rules learned the hard way:**

1. Compare the two routes only via their own concurrently-interleaved baselines. Case 11's drop-mask route measured 4.462x in one run and 3.422x in another — a 30% swing exceeding both runs' noise floors. Never compare a number from one process against a number from another.
2. **Any case under ~5 ms needs >= 100 repeats.** At 25 repeats cases 2 and 3 measured -68% and -12.6%; at 120 repeats the same comparisons gave +11.8% and +12.1%. Use `--repeats 120 --settle-seconds 20` for cases 2, 3, 4, 7, 9, 10, 12 and `--repeats 40` for 1, 5, 8, 11, 13.

Measure `prefer_keymask=True` with the selector Task 4 already exports:

```bash
.venv/bin/python -m src.benchmark --candidate attention:KEYMASK_CANDIDATE \
  --device cuda --dtype float32 --case <N> --padding-ratio <R> \
  --repeats 40 --settle-seconds 20 \
  --output research/benchmarks/2026-08-30-rtx4060-<commit>/case<N>-fp32-pr<R>-keymask.json
```

`attention:KEYMASK_CANDIDATE` reproduces upstream `StridedSDPASelfAttention`
behavior exactly, so it doubles as the incumbent control: this A/B measures
Person 2's change against what `sdpa.py` does today, not against the eager
reference.

**Also measure the integrated route.** The cu130 upgrade landed on 30 August
2026, so `src.dispatcher` now selects `compiled-sdpa` here instead of falling
back. Confirm the gate before trusting any dispatcher number, because a silent
fallback would measure the reference against itself:

```bash
.venv/bin/python -c "
import torch
from src.dispatcher import VALIDATED_TORCH_VERSION
print('validated:', VALIDATED_TORCH_VERSION, 'actual:', torch.__version__,
      '->', 'PASS' if torch.__version__ == VALIDATED_TORCH_VERSION else 'FALLBACK')
"
.venv/bin/python -m src.benchmark --candidate src.dispatcher \
  --device cuda --dtype float32 --case <N> --padding-ratio <R> \
  --repeats 40 --settle-seconds 20 \
  --output research/benchmarks/2026-08-30-rtx4060-<commit>/case<N>-fp32-pr<R>-dispatcher.json
```

The dispatcher composes Person 1's compilation with `sdpa.py`'s always-on key
mask. If the drop-mask route wins standalone, the interesting question is
whether it still wins *inside* `reduce-overhead` — compilation may already hide
the mask cost. Record both, and note that the host sync in
`AttentionCandidate.forward` is not present in the dispatcher path.

- [ ] **Step 3: Record case 6 as correctness and memory only**

```bash
.venv/bin/python -m src.benchmark --candidate attention \
  --device cuda --dtype float32 --case 6 --padding-ratio 0.0 \
  --repeats 10 --settle-seconds 10 \
  --output research/benchmarks/2026-08-30-rtx4060-<commit>/case6-fp32-correctness.json
```

Case 6 was observed at 11,901 MiB peak allocation on an 8,188 MiB card, completing only via WSL2 host-memory oversubscription. Record `torch.cuda.max_memory_allocated()` and mark the latency **not transferable** in the run README, per spec section 7.3. Do not quote a case 6 speedup.

- [ ] **Step 3b: Record the SDPA backend, output strides, and peak memory**

Spec acceptance criterion 4: if SDPA silently selects the **math** backend it
materializes the `N x N` scores and defeats lever L1 — a silent regression that
timing alone will not catch. Criterion 5 requires peak memory for cases 5, 8 and
13 as well as 6. Run once per case and paste the output into the run README:

```bash
.venv/bin/python - <<'EOF'
import torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.infra.cases import load_official_cases

for case_id in (1, 5, 8, 11, 13):
    c = load_official_cases()[case_id]
    dh = c.qkv_dim // c.heads
    q, k, v = (
        torch.randn(c.batch_size, c.heads, c.seq_len, dh,
                    device="cuda", dtype=torch.float32)
        for _ in range(3)
    )
    torch.cuda.reset_peak_memory_stats()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    chosen = []
    for backend in (SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            chosen.append(backend.name)
        except RuntimeError:
            pass
    print(f"case {case_id}: available={chosen} "
          f"out_stride={out.stride()} out_contiguous={out.is_contiguous()} "
          f"peak_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}")
    del q, k, v, out
    torch.cuda.empty_cache()
EOF
```

Expected: `EFFICIENT_ATTENTION` available and `FLASH_ATTENTION` absent in
float32 (flash rejects non-half inputs). If `EFFICIENT_ATTENTION` is *not*
available for some head dimension, that case is falling back to math — stop and
report it, because the measured gain would then be coming from somewhere other
than lever L1.

The `out_stride` figure is the Person 3 handoff required by spec section 4.5: it
says whether the output `reshape` still copies.

- [ ] **Step 4: Write the run README**

Create `research/benchmarks/2026-08-30-rtx4060-<commit>/README.md` with, for every run: the exact command, Git commit, timestamp, input shape, dtype, correctness result, latency, speedup, noise floor, peak memory, and the CPU/GPU/OS/driver/CUDA/PyTorch versions. For each case state which route was selected and by what margin. Mark any speedup inside its own noise floor as `WITHIN NOISE` and do not select on it — where both routes are within noise of each other, prefer `SDPA_CAUSAL`, which allocates no mask.

- [ ] **Step 5: Update the indexes**

Add the run directory to `research/benchmarks/README.md` under `## Current Runs`. Replace the exploratory section 3 table with the harness-produced figures. If any case contradicts the 24/24 exploratory result, say so explicitly rather than averaging it away — a single genuine reversal reopens the per-case routing question the spec just closed.

- [ ] **Step 6: Commit**

```bash
git add research/benchmarks/2026-08-30-rtx4060-<commit>/ research/benchmarks/README.md \
        research/attention-softmax/safe-optimization-spec.md src/implementations/attention.py
git commit -m "bench(attention): preserve per-case route selection evidence"
```

---

### Task 8: Handoff to Persons 1, 3 and 4

The finding is worth nothing until the people who own the shared modules have it. `sdpa.py` and `dispatcher.py` are theirs; this task delivers evidence and a concrete proposed change, and does not edit either file.

**Files:**
- Modify: `research/framework-fastpaths/dispatcher-strategy.md`
- Modify: `research/projections-ffn-fusion/qkv-layout.md`
- Modify: `research/attention-softmax/README.md`

**Interfaces:**
- Consumes: the per-case route table from Task 7.
- Produces: nothing executable.

- [ ] **Step 1: Write the Person 1 handoff**

Under `### Person 2 attention` in `research/framework-fastpaths/dispatcher-strategy.md`, add a dated note recording:

- `StridedSDPASelfAttention` and `PackedQKVSDPASelfAttention` in `src/implementations/sdpa.py` always supply the broadcast key mask. That is correct, but under causal attention the mask is dead code, and dropping it measured faster in **24 of 24** comparisons on cu130 — all twelve in-scope cases at `padding_ratio` 0.0 and 0.3, by ~2-5% on the large shapes (case 13: 48.81 ms vs 51.23 ms) and 5-21% on the small ones. This is a uniform change, not a per-case selection; Task 7's preserved runs are the evidence.
- Under causal attention a right-padded key mask is bitwise dead code — causal masking already writes `-inf` everywhere it would. Verified `0.000e+00` over 4 cases x 4 padding ratios x 4 seeds, and pinned by `src/tests/test_padding_mask_redundancy.py`.
- Both torch 2.6.0+cu124 and 2.13.0+cu130 accept `attn_mask` together with `is_causal=True` and apply both, contrary to the documentation — verified against a hand-computed reference at `max_abs` 3.6e-07 on each. The upstream code already relies on this; the note records that it is deliberate and tested on both stacks, not accidental.
- `classify_mask` and `select_route` are importable from `src/implementations/attention_routing.py`.

- [ ] **Step 2: Record the graph-replay constraint explicitly**

In the same section, state the integration constraint plainly, because it is the part most likely to be got wrong:

> `sdpa.py` correctly refuses to inspect mask values on the host, since that
> would synchronize and break graph replay under `reduce-overhead`. Selecting
> the drop-mask route therefore requires the decision to be made in
> `DispatchingTransformer.forward` **before** the compiled callable is invoked,
> with the resulting route either baked into a separately compiled callable per
> route or set on the attention modules ahead of the call. It must not be done
> inside the captured region. The sync costs roughly 10-20 us, which is noise on
> case 13 (~25-60 ms) and roughly 10% on case 2 (~0.14 ms) — another reason the
> selection is per case.

The minimal upstream change, for reference — **propose it, do not apply it**:

```python
# src/implementations/sdpa.py, StridedSDPASelfAttention.forward
attn_mask = (
    None
    if valid_token_mask is None or self.drop_key_mask
    else valid_token_mask[:, None, None, :]
)
```

with `drop_key_mask` defaulting to `False` and set by the dispatcher per case.

- [ ] **Step 3: Write the Person 3 handoff**

In `research/projections-ffn-fusion/qkv-layout.md`, under `## Ownership Contract`, record the SDPA output strides observed in Task 7 Step 3b and whether the output `reshape` still copies. That is the evidence the packed-producer decision was waiting on. Note that `PackedQKVSDPASelfAttention` inherits the same key-mask question, so whichever route wins for case 2 applies to it too.

- [ ] **Step 4: Write the Person 4 handoff**

In `research/attention-softmax/README.md`, under `## Scope boundaries`, note that spec section 3 applies to cases 6 and 14: under causal attention with right padding, no padding mask need ever be built, which removes a `B x N` term from any chunked design. Include the case 6 result from Task 7 Step 3 with its oversubscription caveat.

- [ ] **Step 5: Commit**

```bash
git add research/framework-fastpaths/dispatcher-strategy.md \
        research/projections-ffn-fusion/qkv-layout.md \
        research/attention-softmax/README.md
git commit -m "research: hand off per-case attention route evidence"
```

---

## Verification

Full check before opening the MR:

```bash
python3 -m unittest discover -s src/tests -v
python3 -m compileall -q src
.venv/bin/python -m unittest src.tests.test_attention_matrix -v
git diff --stat origin/master
```

The MR must explain, per `AGENTS.md`: the problem, the decision and alternatives, affected behavior, expected performance impact, risks, numerical-correctness evidence, benchmark environment and results, and verification commands. `git diff --stat origin/master` must show no changes to `torch_transformer_benchmark.py`, `src/dispatcher.py`, `src/implementations/sdpa.py`, or another stream's implementation module. If it does, the ownership contract has been broken and the change belongs in a handoff note instead.
