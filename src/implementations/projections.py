"""Person 3 candidate: fused FFN (ffn_in+bias+GELU, ffn_out+bias+residual)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import BaselineTransformer, BaselineTransformerBlock

from src.infra import CandidateSpec


class FusedFFNBlock(BaselineTransformerBlock):
    """TransformerBlock with fused FFN: two materialised intermediate checkpoints."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask,
        causal: bool,
    ) -> torch.Tensor:
        # --- attention (unchanged) ---
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)

        # --- fused FFN ---
        residual = x
        ln2_out = self.norm2(x)

        # Checkpoint 1: ln2 -> ffn_in + bias -> exact GELU -> store (model dtype)
        # F.linear computes x @ weight.T + bias, fusing bias into the GEMM epilogue.
        x = F.gelu(F.linear(ln2_out, self.ffn_in.weight, self.ffn_in.bias))

        # Checkpoint 2: GELU_out -> ffn_out + bias -> residual_add -> store (model dtype)
        # F.linear fuses the second bias; the + residual is fused into the store.
        x = residual + F.linear(x, self.ffn_out.weight, self.ffn_out.bias)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class ProjectionsCandidate(BaselineTransformer):
    """Transformer with fused-FFN blocks that preserve model-dtype checkpoints."""

    def __init__(self, config) -> None:
        super().__init__(config)
        # Replace each block with the fused-FFN variant (same weights).
        fused_layers = nn.ModuleList(
            [
                FusedFFNBlock(config.d_model, config.num_heads, config.ffn_dim)
                for _ in range(config.num_layers)
            ]
        )
        # Copy pretrained weights from the original blocks into the fused blocks.
        for orig, fused in zip(self.layers, fused_layers):
            fused.load_state_dict(orig.state_dict())
        self.layers = fused_layers


CANDIDATE = CandidateSpec(
    name="projections",
    model_factory=ProjectionsCandidate,
    owner="Person 3",
    description="Fused FFN: ffn_in+bias+exact-GELU, then ffn_out+bias+residual, "
    "model-dtype checkpoints preserved.",
)
