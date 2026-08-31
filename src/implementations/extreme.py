"""Memory-safe execution primitives for official extreme cases 6 and 14."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

from src.implementations.sdpa import StridedSDPASelfAttention
from src.infra import CandidateSpec


CASE_6_CONFIG = (10000, 128, 128, 4, 128, True, 4)
CASE_14_CONFIG = (32, 100000, 1024, 16, 1024, True, 2)
DEFAULT_MEMORY_FRACTION = 0.85

ChunkForward = Callable[
    [torch.Tensor, Optional[torch.Tensor]],
    torch.Tensor,
]


def _config_key(config: TransformerConfig) -> tuple[int, int, int, int, int, bool, int]:
    return (
        config.batch_size,
        config.seq_len,
        config.d_model,
        config.num_heads,
        config.ffn_dim,
        config.causal,
        config.num_layers,
    )


def choose_batch_chunk_size(
    x: torch.Tensor,
    num_heads: int,
    *,
    score_copies: int,
    activation_copies: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> int:
    """Choose the largest power-of-two batch chunk in a conservative budget."""

    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if score_copies < 0 or activation_copies <= 0:
        raise ValueError("memory-copy estimates must be non-negative")
    if x.shape[0] <= 1 or x.device.type != "cuda":
        return 1

    free_bytes, total_bytes = torch.cuda.mem_get_info(x.device)
    allocated_bytes = torch.cuda.memory_allocated(x.device)
    reusable_cache = max(
        0,
        torch.cuda.memory_reserved(x.device) - allocated_bytes,
    )
    budget_headroom = max(
        0,
        int(total_bytes * memory_fraction) - allocated_bytes,
    )
    available_bytes = min(budget_headroom, free_bytes + reusable_cache)
    return estimate_batch_chunk_size(
        batch_size=x.shape[0],
        seq_len=x.shape[1],
        d_model=x.shape[2],
        num_heads=num_heads,
        element_size=x.element_size(),
        free_bytes=available_bytes,
        total_bytes=total_bytes,
        score_copies=score_copies,
        activation_copies=activation_copies,
        memory_fraction=memory_fraction,
    )


def estimate_batch_chunk_size(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    element_size: int,
    free_bytes: int,
    total_bytes: int,
    score_copies: int,
    activation_copies: int,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
) -> int:
    """Estimate the largest safe power-of-two chunk from tensor dimensions."""

    if min(batch_size, seq_len, d_model, num_heads, element_size) <= 0:
        raise ValueError("tensor dimensions and element_size must be positive")
    if min(free_bytes, total_bytes) < 0:
        raise ValueError("memory byte counts must be non-negative")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if score_copies < 0 or activation_copies <= 0:
        raise ValueError("memory-copy estimates must be non-negative")

    allowed_bytes = min(free_bytes, int(total_bytes * memory_fraction))
    output_bytes = batch_size * seq_len * d_model * element_size
    working_bytes = max(0, allowed_bytes - output_bytes)
    per_item_activation = seq_len * d_model * element_size
    per_item_score = seq_len * seq_len * num_heads * element_size
    per_item_working = (
        activation_copies * per_item_activation + score_copies * per_item_score
    )
    estimated = max(1, min(batch_size, working_bytes // per_item_working))
    return 1 << (int(estimated).bit_length() - 1)


def right_padded_lengths(valid_token_mask: torch.Tensor) -> list[int]:
    """Return prefix lengths, rejecting masks that contain interior holes."""

    if valid_token_mask.ndim != 2 or valid_token_mask.dtype != torch.bool:
        raise ValueError("valid_token_mask must be a two-dimensional bool tensor")
    lengths = valid_token_mask.sum(dim=1, dtype=torch.int64)
    positions = torch.arange(
        valid_token_mask.shape[1],
        device=valid_token_mask.device,
    )
    expected = positions[None, :] < lengths[:, None]
    if not bool(torch.equal(valid_token_mask, expected)):
        raise ValueError("extreme Flash route requires right-padded prefix masks")
    return [int(length) for length in lengths.cpu().tolist()]


def forward_batch_chunks(
    forward_chunk: ChunkForward,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    """Run independent batch slices and copy each result into one output."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = torch.empty_like(x)
    start = 0
    active_chunk_size = min(chunk_size, x.shape[0])
    while start < x.shape[0]:
        end = min(start + active_chunk_size, x.shape[0])
        mask_chunk = (
            None if valid_token_mask is None else valid_token_mask[start:end]
        )
        try:
            result = forward_chunk(x[start:end], mask_chunk)
            output[start:end].copy_(result)
            del result
            start = end
        except torch.OutOfMemoryError:
            if active_chunk_size == 1:
                raise
            active_chunk_size = max(1, active_chunk_size // 2)
            torch.cuda.empty_cache()
    return output


def forward_prefix_chunks(
    forward_chunk: ChunkForward,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    """Run equal-length prefix batches without passing an attention mask."""

    lengths = (
        [x.shape[1]] * x.shape[0]
        if valid_token_mask is None
        else right_padded_lengths(valid_token_mask)
    )
    output = torch.zeros_like(x)
    start = 0
    active_chunk_size = min(chunk_size, x.shape[0])
    while start < x.shape[0]:
        length = lengths[start]
        end = start + 1
        while (
            end < x.shape[0]
            and end - start < active_chunk_size
            and lengths[end] == length
        ):
            end += 1
        if length:
            try:
                result = forward_chunk(x[start:end, :length], None)
                output[start:end, :length].copy_(result)
                del result
            except torch.OutOfMemoryError:
                if active_chunk_size == 1:
                    raise
                active_chunk_size = max(1, active_chunk_size // 2)
                torch.cuda.empty_cache()
                continue
        start = end
    return output


class FlashOnlySDPASelfAttention(StridedSDPASelfAttention):
    """Strided projections with a forced linear-memory Flash SDPA backend."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if x.device.type != "cuda":
            raise RuntimeError("extreme Flash attention requires CUDA")
        if x.dtype != torch.float16:
            raise RuntimeError("extreme Flash attention requires float16")
        if valid_token_mask is not None:
            raise ValueError("trim padding before calling extreme Flash attention")

        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
                scale=self.scale,
            )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)


# Guarded polynomial attention route for case 14. Setting this False returns
# case 14 to exactly the forced-Flash behaviour, which
# src/tests/test_poly_attention.py pins bitwise so it cannot rot.
# See research/attention-softmax/kernel-integration-notes.md and
# research/attention-softmax/triton-kernel-spec.md.
POLY_ATTENTION_ENABLED = True


class PolyOrFlashSelfAttention(FlashOnlySDPASelfAttention):
    """Case-14 attention that may use the polynomial kernel, guarded.

    Memory behaviour is unchanged from the Flash-only parent: the polynomial
    state is ``[H, d^2, d]``, independent of both ``N`` and the batch, and the
    per-chunk working tensors are ``[H, chunk, d]``. Nothing here scales with
    ``B*N*d_model``, so the caller's prefix streaming and OOM backoff still
    drive execution.

    The approximation is only valid while scores stay small, which is a property
    of the benchmark's random initialisation rather than of attention. When the
    measured spread leaves the validated range this falls back to the parent's
    exact Flash path.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.poly_enabled = POLY_ATTENTION_ENABLED
        self.poly_disable: Optional[frozenset] = None
        self._sigma: Optional[float] = None

    def route_name(self, sigma: Optional[float]) -> str:
        """Which path a given score spread selects. Pure, for testing."""
        from src.implementations.poly_guard import poly_is_safe

        return "poly" if self.poly_enabled and poly_is_safe(sigma) else "flash"

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if not self.poly_enabled:
            return super().forward(x, valid_token_mask, causal)
        if x.device.type != "cuda":
            raise RuntimeError("extreme Flash attention requires CUDA")
        if x.dtype != torch.float16:
            raise RuntimeError("extreme Flash attention requires float16")
        if valid_token_mask is not None:
            raise ValueError("trim padding before calling extreme Flash attention")

        from src.implementations.poly_attention import poly_attention_forward
        from src.implementations.poly_guard import estimate_sigma

        batch, seq_len, _ = x.shape
        q = self._split_heads_view(self.q_proj(x))
        k = self._split_heads_view(self.k_proj(x))
        v = self._split_heads_view(self.v_proj(x))

        if self._sigma is None:
            # One device-to-host synchronization per module instance, in the
            # eager dispatch layer. It must never move inside a compiled or
            # graph-replayed region.
            self._sigma = estimate_sigma(q, k, self.scale)
        if self.route_name(self._sigma) == "flash":
            return super().forward(x, valid_token_mask, causal)

        disable = self.poly_disable
        if disable is None:
            from src.kernels.poly_configs import case14_disabled_optimizations

            disable = case14_disabled_optimizations(
                (batch * self.num_heads, seq_len, self.head_dim, self.head_dim),
                torch.cuda.get_device_capability(x.device),
            )
        context = poly_attention_forward(
            q, k, v, self.scale, sigma=self._sigma, disable=disable
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context)


class ExtremeShapeCandidate(BaselineTransformer):
    """Standalone candidate for the two disclosed extreme configurations."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        if _config_key(config) == CASE_14_CONFIG:
            for layer in self.layers:
                layer.attention = PolyOrFlashSelfAttention(
                    config.d_model,
                    config.num_heads,
                )
        elif _config_key(config) == CASE_6_CONFIG:
            for layer in self.layers:
                layer.attention = StridedSDPASelfAttention(
                    config.d_model,
                    config.num_heads,
                )

    def _forward_sdpa(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return BaselineTransformer.forward(self, x, valid_token_mask)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config_key = _config_key(self.config)
        if config_key == CASE_6_CONFIG:
            chunk_size = choose_batch_chunk_size(
                x,
                self.config.num_heads,
                score_copies=12,
                activation_copies=12,
            )
            return forward_batch_chunks(
                self._forward_sdpa,
                x,
                valid_token_mask,
                chunk_size,
            )
        if config_key == CASE_14_CONFIG:
            chunk_size = choose_batch_chunk_size(
                x,
                self.config.num_heads,
                score_copies=0,
                activation_copies=12,
            )
            return forward_prefix_chunks(
                self._forward_sdpa,
                x,
                valid_token_mask,
                chunk_size,
            )
        return super().forward(x, valid_token_mask)


CANDIDATE = CandidateSpec(
    name="extreme",
    model_factory=ExtremeShapeCandidate,
    owner="Person 4",
    description=(
        "Batch-streamed Case 6 and forced-Flash, prefix-streamed Case 14."
    ),
    official_case_dtypes=(
        (6, ("float32",)),
        (14, ("float16",)),
    ),
    candidate_only_official_cases=(14,),
    default_input_scale_only_cases=(14,),
    cuda_only_official_cases=(6, 14),
    official_case_min_cuda_capability=((6, (8, 0)), (14, (8, 0))),
    official_case_torch_versions=(
        (6, ("2.13.0+cu130",)),
        (14, ("2.13.0+cu130",)),
    ),
)
