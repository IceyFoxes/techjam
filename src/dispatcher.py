"""Shape-aware integration candidate for the twelve feasible official cases.

The evaluator constructs the model on CPU, copies reference weights, and only
then moves it to the final device and dtype. Compilation is therefore lazy and
per model instance. Unsupported runtime contracts use the executable reference
arithmetic rather than silently trying an unvalidated fast path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
)

from src.infra import CandidateSpec, OfficialCase, load_official_cases


ConfigKey = Tuple[int, int, int, int, int, bool, int]
ForwardCallable = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

REFERENCE_BACKEND = "reference"
COMPILED_SDPA_BACKEND = "compiled-sdpa"

# Cases 6 and 14 are intentionally absent. Their extreme batch/sequence sizes
# need the separately owned memory-safe backend before they can be promoted.
CASE_COMPILE_MODES: Dict[int, str] = {
    1: "reduce-overhead",
    2: "reduce-overhead",
    3: "reduce-overhead",
    4: "reduce-overhead",
    5: "reduce-overhead",
    7: "reduce-overhead",
    8: "reduce-overhead",
    9: "reduce-overhead",
    10: "reduce-overhead",
    11: "reduce-overhead",
    12: "reduce-overhead",
    # CUDA Graph replay did not credibly help this long-sequence case.
    13: "default",
}

# The route is measured on capability 12.0 (RTX 5080). SDPA and Inductor are
# mature on Ampere and newer; older/unknown CUDA capabilities take the exact
# reference path until separately validated.
MIN_COMPILED_CUDA_CAPABILITY = (8, 0)


def _config_key(config: TransformerConfig) -> ConfigKey:
    return (
        config.batch_size,
        config.seq_len,
        config.d_model,
        config.num_heads,
        config.ffn_dim,
        config.causal,
        config.num_layers,
    )


def _official_config_key(case: OfficialCase) -> ConfigKey:
    return (
        case.batch_size,
        case.seq_len,
        case.qkv_dim,
        case.heads,
        case.ffn_dim,
        case.causal,
        case.layers,
    )


OFFICIAL_CASE_BY_CONFIG: Dict[ConfigKey, int] = {
    _official_config_key(case): case_id
    for case_id, case in load_official_cases().items()
}


@dataclass(frozen=True)
class RouteDecision:
    """Resolved backend for one static model and runtime contract."""

    case_id: Optional[int]
    backend: str
    compile_mode: Optional[str]
    reason: str


@dataclass(frozen=True)
class RuntimeKey:
    """Inputs that can change compiler guards or numerical behavior."""

    input_shape: Tuple[int, ...]
    device_type: str
    device_index: Optional[int]
    dtype: torch.dtype
    device_capability: Optional[Tuple[int, int]]
    mask_present: bool
    inference_mode: bool
    grad_enabled: bool
    matmul_precision: str
    allow_tf32: bool


def select_route(
    config: TransformerConfig,
    *,
    device_type: str,
    dtype: torch.dtype,
    device_capability: Optional[Tuple[int, int]],
) -> RouteDecision:
    """Select only routes validated for the complete official configuration."""

    case_id = OFFICIAL_CASE_BY_CONFIG.get(_config_key(config))
    if case_id is None:
        return RouteDecision(
            None,
            REFERENCE_BACKEND,
            None,
            "configuration is not an official disclosed case",
        )
    if case_id not in CASE_COMPILE_MODES:
        return RouteDecision(
            case_id,
            REFERENCE_BACKEND,
            None,
            "extreme case requires a memory-safe backend",
        )
    if device_type != "cuda":
        return RouteDecision(
            case_id,
            REFERENCE_BACKEND,
            None,
            "compiled SDPA route is GPU-only",
        )
    if dtype != torch.float32:
        return RouteDecision(
            case_id,
            REFERENCE_BACKEND,
            None,
            "only float32 has integrated numerical evidence",
        )
    if (
        device_capability is None
        or device_capability < MIN_COMPILED_CUDA_CAPABILITY
    ):
        return RouteDecision(
            case_id,
            REFERENCE_BACKEND,
            None,
            "CUDA capability is outside the validated compiler family",
        )
    if not callable(getattr(torch, "compile", None)):
        return RouteDecision(
            case_id,
            REFERENCE_BACKEND,
            None,
            "torch.compile is unavailable",
        )
    return RouteDecision(
        case_id,
        COMPILED_SDPA_BACKEND,
        CASE_COMPILE_MODES[case_id],
        "validated official float32 CUDA route",
    )


class StridedSDPASelfAttention(BaselineSelfAttention):
    """Reference-compatible projections around SDPA without Q/K/V copies."""

    def _split_heads_view(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward_reference(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Execute the immutable benchmark's explicit attention arithmetic."""

        return BaselineSelfAttention.forward(self, x, valid_token_mask, causal)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        # True means that a key participates in attention. Do not inspect mask
        # values on the host: that would synchronize and break graph replay.
        attn_mask = (
            None
            if valid_token_mask is None
            else valid_token_mask[:, None, None, :]
        )
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class DispatchingTransformer(BaselineTransformer):
    """Lazy per-instance dispatcher with a strict reference fallback."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.attention = StridedSDPASelfAttention(
                config.d_model,
                config.num_heads,
            )

        self._compiled_forwards: Dict[RuntimeKey, ForwardCallable] = {}
        self._compile_failures: Dict[RuntimeKey, str] = {}
        self._last_route: Optional[RouteDecision] = None

    @property
    def last_route(self) -> Optional[RouteDecision]:
        """Most recent route, exposed for diagnostics and contract tests."""

        return self._last_route

    @property
    def compile_failures(self) -> Dict[RuntimeKey, str]:
        """A copy of lazy compilation failures that triggered fallback."""

        return dict(self._compile_failures)

    def clear_runtime_cache(self) -> None:
        """Discard callables bound to the current parameter/device state."""

        self._compiled_forwards.clear()
        self._compile_failures.clear()
        self._last_route = None

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ):
        # nn.Module.to()/cuda()/half() all flow through _apply. A compiled
        # callable must never survive a parameter move or dtype conversion.
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_compiled_forwards"):
            self.clear_runtime_cache()
        return result

    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.clear_runtime_cache()
        return result

    def _forward_sdpa(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return BaselineTransformer.forward(self, x, valid_token_mask)

    # Dynamo associates recompilation limits with a Python code object. A test
    # runner may construct every official case in one process, so compiling one
    # shared method would exhaust that limit after enough distinct shapes. Keep
    # a separate, deliberately tiny entry point per routed case; the common
    # implementation is still inlined by Dynamo.
    def _forward_case_1(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_2(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_3(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_4(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_5(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_7(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_8(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_9(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_10(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_11(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_12(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _forward_case_13(self, x, valid_token_mask=None):
        return self._forward_sdpa(x, valid_token_mask)

    def _compile_entrypoint(self, case_id: int) -> ForwardCallable:
        entrypoints = {
            1: self._forward_case_1,
            2: self._forward_case_2,
            3: self._forward_case_3,
            4: self._forward_case_4,
            5: self._forward_case_5,
            7: self._forward_case_7,
            8: self._forward_case_8,
            9: self._forward_case_9,
            10: self._forward_case_10,
            11: self._forward_case_11,
            12: self._forward_case_12,
            13: self._forward_case_13,
        }
        return entrypoints[case_id]

    def _forward_reference(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # This mirrors BaselineTransformer/Block.forward while explicitly
        # selecting the baseline attention implementation on shared parameters.
        for layer in self.layers:
            attention = layer.attention
            if not isinstance(attention, StridedSDPASelfAttention):
                raise TypeError(
                    "dispatcher attention module was unexpectedly replaced"
                )
            x = x + attention.forward_reference(
                layer.norm1(x),
                valid_token_mask,
                self.config.causal,
            )
            x = x + layer.ffn_out(
                F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none")
            )
            if valid_token_mask is not None:
                x = x.masked_fill(~valid_token_mask[..., None], 0)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x

    @staticmethod
    def _device_capability(device: torch.device) -> Optional[Tuple[int, int]]:
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(device)

    def _runtime_key(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> RuntimeKey:
        return RuntimeKey(
            input_shape=tuple(x.shape),
            device_type=x.device.type,
            device_index=x.device.index,
            dtype=x.dtype,
            device_capability=self._device_capability(x.device),
            mask_present=valid_token_mask is not None,
            inference_mode=torch.is_inference_mode_enabled(),
            grad_enabled=torch.is_grad_enabled(),
            matmul_precision=torch.get_float32_matmul_precision(),
            allow_tf32=(
                bool(torch.backends.cuda.matmul.allow_tf32)
                if x.device.type == "cuda"
                else False
            ),
        )

    def _resolve_route(self, key: RuntimeKey) -> RouteDecision:
        expected_shape = (
            self.config.batch_size,
            self.config.seq_len,
            self.config.d_model,
        )
        if key.input_shape != expected_shape:
            return RouteDecision(
                OFFICIAL_CASE_BY_CONFIG.get(_config_key(self.config)),
                REFERENCE_BACKEND,
                None,
                "runtime input shape does not match the static configuration",
            )
        return select_route(
            self.config,
            device_type=key.device_type,
            dtype=key.dtype,
            device_capability=key.device_capability,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = self._runtime_key(x, valid_token_mask)
        cached = self._compiled_forwards.get(key)
        if cached is not None:
            return cached(x, valid_token_mask)

        route = self._resolve_route(key)
        self._last_route = route
        if route.backend == REFERENCE_BACKEND:
            self._compiled_forwards[key] = self._forward_reference
            return self._forward_reference(x, valid_token_mask)

        assert route.compile_mode is not None
        assert route.case_id is not None
        try:
            compiled = torch.compile(
                self._compile_entrypoint(route.case_id),
                mode=route.compile_mode,
            )
            # Inductor compilation is lazy. Execute once here so a first-call
            # backend failure can atomically demote this runtime key.
            output = compiled(x, valid_token_mask)
        except Exception as error:
            self._compile_failures[key] = f"{type(error).__name__}: {error}"
            self._compiled_forwards[key] = self._forward_reference
            self._last_route = RouteDecision(
                route.case_id,
                REFERENCE_BACKEND,
                None,
                "lazy compilation failed; using reference fallback",
            )
            return self._forward_reference(x, valid_token_mask)

        self._compiled_forwards[key] = compiled
        return output


CANDIDATE = CandidateSpec(
    name="dispatcher",
    model_factory=DispatchingTransformer,
    owner="Person 1 / integrator",
    description=(
        "Lazy twelve-case float32 CUDA dispatcher: strided-view SDPA with "
        "shape-specific compile modes and an exact reference fallback."
    ),
)
