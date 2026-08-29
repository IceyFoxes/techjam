"""Person 3 reference-equivalent control using functional FFN operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, BaselineTransformerBlock

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


CANDIDATE = CandidateSpec(
    name="projections-control",
    model_factory=ProjectionsControl,
    owner="Person 3",
    description=(
        "Reference-equivalent FFN control expressed with F.linear and exact "
        "GELU; model-dtype intermediates are preserved."
    ),
)
