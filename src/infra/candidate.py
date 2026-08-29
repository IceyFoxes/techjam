"""Stable candidate contract used by the collaboration benchmark harness."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


ModelFactory = Callable[[Any], Any]
WeightLoader = Callable[[Any, Any, bool], None]


@dataclass(frozen=True)
class CandidateSpec:
    """Description and construction hooks for one full-model candidate.

    ``config`` passed to ``model_factory`` is the authoritative benchmark's
    ``TransformerConfig``. A custom weight loader is only needed when a candidate
    deliberately changes parameter names or packing.
    """

    name: str
    model_factory: ModelFactory
    owner: str
    description: str
    weight_loader: Optional[WeightLoader] = None
    strict_weight_copy: bool = True
    self_compiling: bool = False
    unsupported_official_cases: Tuple[int, ...] = ()

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.name):
            raise ValueError(
                "candidate name must contain only lowercase letters, numbers, "
                "underscores, and hyphens"
            )
        if not callable(self.model_factory):
            raise TypeError("candidate model_factory must be callable")
        if self.weight_loader is not None and not callable(self.weight_loader):
            raise TypeError("candidate weight_loader must be callable")
        if not isinstance(self.self_compiling, bool):
            raise TypeError("candidate self_compiling must be a bool")
        if not isinstance(self.unsupported_official_cases, tuple):
            raise TypeError("candidate unsupported_official_cases must be a tuple")
        if any(
            not isinstance(case_id, int) or not 1 <= case_id <= 14
            for case_id in self.unsupported_official_cases
        ):
            raise ValueError(
                "unsupported official cases must contain only ids 1 through 14"
            )
        if len(set(self.unsupported_official_cases)) != len(
            self.unsupported_official_cases
        ):
            raise ValueError("unsupported official cases must not contain duplicates")


def load_candidate(selector: str) -> CandidateSpec:
    """Load ``module[:attribute]``; short modules resolve under implementations.

    A module uses ``CANDIDATE`` by default. This convention lets contributors add
    candidates without editing a shared registry.
    """

    module_selector, separator, attribute = selector.partition(":")
    if not module_selector:
        raise ValueError("candidate selector must not be empty")
    if "." not in module_selector:
        module_selector = f"src.implementations.{module_selector}"
    attribute = attribute if separator else "CANDIDATE"
    if not attribute:
        raise ValueError("candidate attribute must not be empty")

    module = importlib.import_module(module_selector)
    try:
        candidate = getattr(module, attribute)
    except AttributeError as error:
        raise AttributeError(
            f"candidate module {module_selector!r} has no {attribute!r} attribute"
        ) from error
    if not isinstance(candidate, CandidateSpec):
        raise TypeError(
            f"{module_selector}:{attribute} must be a CandidateSpec, "
            f"got {type(candidate).__name__}"
        )
    candidate.validate()
    return candidate


def validate_candidate_execution(
    spec: CandidateSpec,
    official_case_id: Optional[int],
    *,
    compile_user: bool,
) -> None:
    """Reject evaluator options that violate a candidate's declared contract."""

    if spec.self_compiling and compile_user:
        raise ValueError(
            f"candidate {spec.name!r} compiles itself; remove --compile-user "
            "to avoid nested compilation"
        )
    if official_case_id in spec.unsupported_official_cases:
        raise ValueError(
            f"candidate {spec.name!r} explicitly does not support official "
            f"case {official_case_id}; refusing before model/input allocation"
        )
