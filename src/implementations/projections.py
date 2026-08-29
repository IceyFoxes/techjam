"""Person 3 projection controls and the experimental Case-2 packed-QKV route."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, BaselineTransformerBlock

from src.implementations.sdpa import PackedQKVSDPASelfAttention
from src.infra import CandidateSpec


class FunctionalFFNControlBlock(BaselineTransformerBlock):
    """Reference-compatible block with an explicitly written functional FFN."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask,
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)

        residual = x
        ln2_out = self.norm2(x)
        x = F.gelu(
            F.linear(ln2_out, self.ffn_in.weight, self.ffn_in.bias),
            approximate="none",
        )
        x = residual + F.linear(x, self.ffn_out.weight, self.ffn_out.bias)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class ProjectionsControl(BaselineTransformer):
    """Reference-equivalent control that preserves model-dtype checkpoints."""

    def __init__(self, config) -> None:
        super().__init__(config)
        functional_layers = nn.ModuleList(
            [
                FunctionalFFNControlBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        for original, functional in zip(self.layers, functional_layers):
            functional.load_state_dict(original.state_dict(), strict=True)
        self.layers = functional_layers


class Case2PackedQKVTransformer(BaselineTransformer):
    """Full-model Case-2 experiment with packed QKV and strided-view SDPA."""

    def __init__(self, config) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.attention = PackedQKVSDPASelfAttention(
                config.d_model,
                config.num_heads,
            )


CANDIDATE = CandidateSpec(
    name="projections-control",
    model_factory=ProjectionsControl,
    owner="Person 3",
    description=(
        "Reference-equivalent FFN control expressed with F.linear and exact "
        "GELU; model-dtype intermediates are preserved."
    ),
)


PACKED_CASE2 = CandidateSpec(
    name="case2-packed-qkv",
    model_factory=Case2PackedQKVTransformer,
    owner="Person 3",
    description=(
        "Experimental Case-2 route: prepacked QKV projection, strided views, "
        "and SDPA; use external reduce-overhead compilation."
    ),
    unsupported_official_cases=(1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
)


PACKED_ALL = CandidateSpec(
    name="packed-qkv-cross-case-validation",
    model_factory=Case2PackedQKVTransformer,
    owner="Person 1 validation / Person 3 implementation",
    description=(
        "Validation-only packed-QKV, strided-view SDPA candidate for every "
        "memory-feasible official case; requires external compilation."
    ),
    unsupported_official_cases=(6, 14),
)
