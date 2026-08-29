"""Person 2 candidate: attention and softmax kernels."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class AttentionCandidate(BaselineTransformer):
    """Scaffold that remains correct until attention optimizations are added."""


CANDIDATE = CandidateSpec(
    name="attention",
    model_factory=AttentionCandidate,
    owner="Person 2",
    description="Attention and softmax candidate (reference scaffold).",
)
