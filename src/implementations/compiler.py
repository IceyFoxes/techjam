"""Person 1 candidate: framework fast paths and compilation."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class CompilerCandidate(BaselineTransformer):
    """Scaffold that remains correct until framework optimizations are added."""


CANDIDATE = CandidateSpec(
    name="compiler",
    model_factory=CompilerCandidate,
    owner="Person 1",
    description="Framework fast-path and compilation candidate (reference scaffold).",
)
