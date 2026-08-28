"""Person 4 candidate: memory-safe execution for extreme shapes."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class ExtremeShapeCandidate(BaselineTransformer):
    """Scaffold that remains correct until memory strategies are added."""


CANDIDATE = CandidateSpec(
    name="extreme",
    model_factory=ExtremeShapeCandidate,
    owner="Person 4",
    description="Extreme-shape memory candidate (reference scaffold).",
)
