"""Correct but intentionally slower candidate for end-to-end infrastructure checks."""

from __future__ import annotations

from typing import Optional

import torch

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class DummyCandidate(BaselineTransformer):
    """Reference execution plus bounded, discarded diagnostic GPU work."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = super().forward(x, valid_token_mask)

        # This work is deliberately unrelated to the returned value. It gives the
        # profiler an unmistakable candidate-only region without changing model
        # correctness or allocating tensors proportional to an extreme full case.
        # Keep the dummy in eager mode: a compiler may eliminate discarded work.
        with torch.profiler.record_function("dummy_extra_work"):
            tokens = output.reshape(-1, output.shape[-1])
            tile_size = min(256, tokens.shape[0], tokens.shape[1])
            tile = tokens[:tile_size, :tile_size].float()
            scratch = tile @ tile.transpose(0, 1)
            for _ in range(3):
                scratch = torch.sin(scratch) + torch.cos(scratch)
            torch.softmax(scratch, dim=-1)

        # The clone is bitwise exact and leaves one final, visible copy operation.
        return output.clone()


CANDIDATE = CandidateSpec(
    name="dummy",
    model_factory=DummyCandidate,
    owner="infrastructure",
    description=(
        "Correct reference model with discarded GEMM, trigonometric, softmax, "
        "and output-copy work for visible infrastructure profiling."
    ),
)
