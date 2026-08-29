"""Load and validate the disclosed organizer test-shape table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


DEFAULT_CASES_PATH = Path(__file__).resolve().parents[1] / "cases" / "task_shapes.json"


@dataclass(frozen=True)
class OfficialCase:
    id: int
    batch_size: int
    qkv_dim: int
    heads: int
    seq_len: int
    layers: int
    causal: bool
    ffn_dim: int

    def validate(self) -> None:
        numeric_fields = (
            self.id,
            self.batch_size,
            self.qkv_dim,
            self.heads,
            self.seq_len,
            self.layers,
            self.ffn_dim,
        )
        if any(value <= 0 for value in numeric_fields):
            raise ValueError(f"case {self.id} contains a non-positive value")
        if self.qkv_dim % self.heads != 0:
            raise ValueError(
                f"case {self.id}: qkv_dim must be divisible by heads"
            )

    def benchmark_config(self) -> Dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "d_model": self.qkv_dim,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "layers": self.layers,
            "causal": self.causal,
            "ffn_dim": self.ffn_dim,
        }


def load_official_cases(path: Path = DEFAULT_CASES_PATH) -> Dict[int, OfficialCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported shape schema in {path}")

    cases: Dict[int, OfficialCase] = {}
    for raw_case in payload.get("cases", []):
        case = OfficialCase(**raw_case)
        case.validate()
        if case.id in cases:
            raise ValueError(f"duplicate official case id {case.id}")
        cases[case.id] = case
    expected_ids = set(range(1, 15))
    if set(cases) != expected_ids:
        raise ValueError(
            f"expected official case ids 1-14, got {sorted(cases)}"
        )
    return cases
