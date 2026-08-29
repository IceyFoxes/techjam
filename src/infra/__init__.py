"""Shared infrastructure for loading and benchmarking optimization candidates."""

from .candidate import CandidateSpec, load_candidate
from .cases import OfficialCase, load_official_cases

__all__ = [
    "CandidateSpec",
    "OfficialCase",
    "load_candidate",
    "load_official_cases",
]
