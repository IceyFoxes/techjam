"""Person 1 research probe: SDPA with strided head views.

This module exists to measure the composition selected by the Person 1 and
Person 2 research before it is promoted into an owned implementation.  It keeps
the reference parameter layout, causal behavior, and padding-key semantics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
)

from src.infra import CandidateSpec


class StridedSDPASelfAttention(BaselineSelfAttention):
    """Reference projections around PyTorch SDPA without Q/K/V copies."""

    def _split_heads_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        # PyTorch 2.13 accepts a broadcast key mask together with is_causal.
        # True denotes an element that participates in attention.
        attn_mask = (
            None
            if valid_token_mask is None
            else valid_token_mask[:, None, None, :]
        )
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class IntegratedProbeTransformer(BaselineTransformer):
    """Reference Transformer whose attention cores use the measured SDPA path."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.attention = StridedSDPASelfAttention(
                config.d_model,
                config.num_heads,
            )


CANDIDATE = CandidateSpec(
    name="integrated-probe",
    model_factory=IntegratedProbeTransformer,
    owner="Person 1 research integration",
    description="Float32 SDPA with strided Q/K/V views and padding semantics.",
)
