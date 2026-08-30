"""Linear-memory FP32 reference primitives for official Case 14.

This is validation infrastructure, not a submitted candidate.  It preserves the
immutable reference model around attention and replaces only the impossible
explicit ``[N, N]`` score/probability tensors with algebraically equivalent
scaled-dot-product attention.  CUDA execution explicitly excludes the math
backend so an unsupported fused route fails instead of attempting a
multi-terabyte allocation.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
)


CASE_14_BATCH = 32
CASE_14_SEQ_LEN = 100_000
CASE_14_D_MODEL = 1024
CASE_14_HEADS = 16
CASE_14_FFN_DIM = 1024
CASE_14_LAYERS = 2

# Both backends avoid materialising the complete attention matrix.  Availability
# is hardware/PyTorch dependent, so the forced context deliberately has no math
# fallback.
FP32_LINEAR_MEMORY_BACKENDS = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
)


def case14_config(
    *,
    batch_size: int = CASE_14_BATCH,
    seq_len: int = CASE_14_SEQ_LEN,
) -> TransformerConfig:
    """Return Case 14's model shape, optionally reduced for oracle validation."""

    return TransformerConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=CASE_14_D_MODEL,
        num_heads=CASE_14_HEADS,
        ffn_dim=CASE_14_FFN_DIM,
        num_layers=CASE_14_LAYERS,
        causal=True,
    )


def _right_padded_prefix_mask(valid_token_mask: torch.Tensor) -> bool:
    if valid_token_mask.ndim != 2 or valid_token_mask.dtype != torch.bool:
        return False
    lengths = valid_token_mask.sum(dim=1, dtype=torch.int64)
    positions = torch.arange(valid_token_mask.shape[1], device=valid_token_mask.device)
    return bool(torch.equal(valid_token_mask, positions[None, :] < lengths[:, None]))


class LinearMemoryFP32SelfAttention(BaselineSelfAttention):
    """Reference-like FP32 attention with no quadratic-memory CUDA fallback.

    Projection, head splitting, output projection, and invalid-query handling
    are copied from :class:`BaselineSelfAttention`.  SDPA changes only the
    reduction order used to evaluate the same causal softmax expression.
    """

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if x.dtype != torch.float32:
            raise RuntimeError("the Case 14 FP32 oracle requires float32 input")
        if valid_token_mask is not None and not _right_padded_prefix_mask(
            valid_token_mask
        ):
            raise ValueError("the Case 14 FP32 oracle requires right-padded masks")

        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # With causal right-padded prefixes, the key mask is redundant for all
        # valid queries.  Invalid query outputs are zeroed below, exactly as in
        # the immutable attention module.
        attention_mask = None
        if valid_token_mask is not None and not causal:
            attention_mask = valid_token_mask[:, None, None, :]

        backend_context = (
            sdpa_kernel(list(FP32_LINEAR_MEMORY_BACKENDS))
            if x.device.type == "cuda"
            else nullcontext()
        )
        with backend_context:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=causal,
                scale=self.scale,
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class LinearMemoryFP32Reference(BaselineTransformer):
    """Baseline Transformer with only its attention evaluator replaced."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        for layer in self.layers:
            original_attention = layer.attention
            replacement = LinearMemoryFP32SelfAttention(
                config.d_model,
                config.num_heads,
            )
            replacement.load_state_dict(original_attention.state_dict())
            layer.attention = replacement


@dataclass(frozen=True)
class StreamedAccuracy:
    """Accuracy totals accumulated without full-batch checker temporaries."""

    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float


def compare_outputs_streamed(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float = 0.02,
    atol: float = 0.002,
    token_chunk: int = 2048,
) -> StreamedAccuracy:
    """Apply the benchmark's exact OR criterion in bounded token chunks."""

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    if reference.ndim != 3:
        raise ValueError("reference and candidate must be [B, N, D]")
    if token_chunk <= 0:
        raise ValueError("token_chunk must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")

    total = reference.numel()
    failed = 0
    abs_sum = 0.0
    max_abs = 0.0
    max_rel = 0.0
    for start in range(0, reference.shape[1], token_chunk):
        end = min(start + token_chunk, reference.shape[1])
        ref = reference[:, start:end].detach().float()
        got = candidate[:, start:end].detach().float()
        finite = torch.isfinite(ref) & torch.isfinite(got)
        error = (got - ref).abs()
        passed = finite & ((error <= atol) | (error <= rtol * ref.abs()))
        failed += int((~passed).sum().item())
        abs_sum += float(error.sum(dtype=torch.float64).item())
        max_abs = max(max_abs, float(error.max().item()))
        relative = error / ref.abs().clamp_min(1e-12)
        max_rel = max(max_rel, float(relative.max().item()))

    return StreamedAccuracy(
        passed=failed == 0,
        total_elements=total,
        failed_elements=failed,
        max_abs_error=max_abs,
        max_relative_error=max_rel,
        mean_abs_error=abs_sum / total,
    )
