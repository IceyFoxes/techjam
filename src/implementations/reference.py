"""Known-correct no-op candidate used to validate the collaboration harness."""

from torch_transformer_benchmark import BaselineTransformer

from src.infra import CandidateSpec


class ReferenceCandidate(BaselineTransformer):
    """Deliberately identical to the authoritative reference."""


CANDIDATE = CandidateSpec(
    name="reference",
    model_factory=ReferenceCandidate,
    owner="infrastructure",
    description="Known-correct baseline used to smoke-test candidate loading.",
)
