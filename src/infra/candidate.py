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
    official_case_dtypes: Tuple[Tuple[int, Tuple[str, ...]], ...] = ()
    candidate_only_official_cases: Tuple[int, ...] = ()
    default_input_scale_only_cases: Tuple[int, ...] = ()
    cuda_only_official_cases: Tuple[int, ...] = ()
    official_case_min_cuda_capability: Tuple[
        Tuple[int, Tuple[int, int]], ...
    ] = ()
    official_case_torch_versions: Tuple[Tuple[int, Tuple[str, ...]], ...] = ()

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
        seen_dtype_cases = set()
        for case_id, dtypes in self.official_case_dtypes:
            if not isinstance(case_id, int) or not 1 <= case_id <= 14:
                raise ValueError("dtype contracts require official case ids 1 through 14")
            if case_id in seen_dtype_cases:
                raise ValueError("dtype contracts must not repeat an official case")
            if not isinstance(dtypes, tuple) or not dtypes:
                raise TypeError("dtype contracts must contain a non-empty dtype tuple")
            if any(dtype not in ("float32", "float16", "bfloat16") for dtype in dtypes):
                raise ValueError("dtype contracts contain an unsupported dtype name")
            seen_dtype_cases.add(case_id)
        if any(
            not isinstance(case_id, int) or not 1 <= case_id <= 14
            for case_id in self.candidate_only_official_cases
        ):
            raise ValueError(
                "candidate-only official cases must contain ids 1 through 14"
            )
        if not isinstance(self.default_input_scale_only_cases, tuple):
            raise TypeError("default-input-scale cases must be a tuple")
        if any(
            not isinstance(case_id, int) or not 1 <= case_id <= 14
            for case_id in self.default_input_scale_only_cases
        ):
            raise ValueError(
                "default-input-scale cases must contain ids 1 through 14"
            )
        for case_ids, label in (
            (self.cuda_only_official_cases, "CUDA-only"),
        ):
            if not isinstance(case_ids, tuple):
                raise TypeError(f"{label} official cases must be a tuple")
            if any(
                not isinstance(case_id, int) or not 1 <= case_id <= 14
                for case_id in case_ids
            ):
                raise ValueError(
                    f"{label} official cases must contain ids 1 through 14"
                )
        seen_capability_cases = set()
        for case_id, capability in self.official_case_min_cuda_capability:
            if not isinstance(case_id, int) or not 1 <= case_id <= 14:
                raise ValueError("capability contracts require case ids 1 through 14")
            if case_id in seen_capability_cases:
                raise ValueError("capability contracts must not repeat a case")
            if (
                not isinstance(capability, tuple)
                or len(capability) != 2
                or any(not isinstance(value, int) or value < 0 for value in capability)
            ):
                raise TypeError("CUDA capabilities must be non-negative integer pairs")
            seen_capability_cases.add(case_id)
        seen_version_cases = set()
        for case_id, versions in self.official_case_torch_versions:
            if not isinstance(case_id, int) or not 1 <= case_id <= 14:
                raise ValueError("version contracts require case ids 1 through 14")
            if case_id in seen_version_cases:
                raise ValueError("version contracts must not repeat a case")
            if (
                not isinstance(versions, tuple)
                or not versions
                or any(not isinstance(version, str) or not version for version in versions)
            ):
                raise TypeError("version contracts require non-empty version strings")
            seen_version_cases.add(case_id)


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
    dtype_name: Optional[str] = None,
    candidate_only: bool = False,
    input_scale: float = 1.0,
    device_type: Optional[str] = None,
    cuda_capability: Optional[Tuple[int, int]] = None,
    torch_version: Optional[str] = None,
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
    dtype_contract = dict(spec.official_case_dtypes).get(official_case_id)
    if dtype_contract is not None and dtype_name not in dtype_contract:
        choices = ", ".join(dtype_contract)
        raise ValueError(
            f"candidate {spec.name!r} supports official case {official_case_id} "
            f"only with {choices}; received {dtype_name}; refusing before "
            "model/input allocation"
        )
    if official_case_id in spec.candidate_only_official_cases and not candidate_only:
        raise ValueError(
            f"official case {official_case_id} has no runnable dense baseline; "
            "use the candidate-only extreme smoke runner"
        )
    if (
        official_case_id in spec.default_input_scale_only_cases
        and input_scale != 1.0
    ):
        raise ValueError(
            f"official case {official_case_id} is validated only at the default "
            "input scale 1.0"
        )
    if official_case_id in spec.cuda_only_official_cases and device_type != "cuda":
        raise ValueError(
            f"official case {official_case_id} requires CUDA; refusing before "
            "model/input allocation"
        )
    minimum_capability = dict(spec.official_case_min_cuda_capability).get(
        official_case_id
    )
    if minimum_capability is not None and (
        cuda_capability is None or cuda_capability < minimum_capability
    ):
        raise ValueError(
            f"official case {official_case_id} requires CUDA capability "
            f"{minimum_capability} or newer; refusing before model/input allocation"
        )
    supported_versions = dict(spec.official_case_torch_versions).get(
        official_case_id
    )
    if supported_versions is not None and torch_version not in supported_versions:
        raise ValueError(
            f"official case {official_case_id} requires PyTorch "
            f"{', '.join(supported_versions)}; refusing before model/input allocation"
        )
