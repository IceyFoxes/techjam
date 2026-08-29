"""Person 3 candidate: projections, FFN, normalization, and fusion."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class ProjectionsCandidate(BaselineTransformer):
    """Scaffold that remains correct until projection optimizations are added."""


CANDIDATE = CandidateSpec(
    name="projections",
    model_factory=ProjectionsCandidate,
    owner="Person 3",
    description="Projection and elementwise-fusion candidate (reference scaffold).",
)
