"""Final integration candidate; accepted optimizations are routed here."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class DispatchingTransformer(BaselineTransformer):
    """Central integration point for shape-based backend selection.

    Keep all final shape checks in this module. The initial implementation is a
    correct reference fallback so integration can happen incrementally.
    """


CANDIDATE = CandidateSpec(
    name="dispatcher",
    model_factory=DispatchingTransformer,
    owner="Person 1 / integrator",
    description="Integrated shape dispatcher with a reference fallback.",
)
