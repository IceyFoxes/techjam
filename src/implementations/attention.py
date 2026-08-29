"""Person 2 candidate: attention mask routing.

Person 1's ``src/implementations/sdpa.py`` already supplies strided-view SDPA.
It always passes the broadcast key mask, which is correct but not the fastest
option: under causal attention that mask is dead code, and dropping it measured
faster in 24 of 24 comparisons on the pinned cu130 stack -- all twelve in-scope
cases at padding_ratio 0.0 and 0.3.

This module adds the route choice on top of theirs. The two causal routes
compute the same function, so the selection is made on speed alone. See
``research/attention-softmax/safe-optimization-spec.md`` section 3.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

from src.implementations.attention_routing import (
    MaskKind,
    Route,
    classify_mask,
    select_route,
)
from src.implementations.sdpa import StridedSDPASelfAttention
from src.infra import CandidateSpec


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
        skip_padding_mask: bool,
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
        # positions the causal mask already blocked, so the caller may skip it.
        # That holds ONLY for prefix masks: a general mask still has to be
        # applied, which is why the route, not `causal`, decides.
        if valid_token_mask is not None and not skip_padding_mask:
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
            return self._eager_exact(
                x, valid_token_mask, causal, skip_padding_mask=False
            )

        if route is Route.EXACT_EAGER_PREFIX:
            return self._eager_exact(
                x, valid_token_mask, causal, skip_padding_mask=True
            )

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
    description="Mask-routed SDPA retaining the broadcast key mask; A/B control.",
)
