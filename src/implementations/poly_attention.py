"""Polynomial attention for official case 14, with a fused Triton back end.

Routing is decided by ``src.implementations.poly_guard``: when the measured
score spread leaves the validated range, the caller falls back to exact Flash
SDPA. See ``research/attention-softmax/triton-kernel-spec.md``.
"""

from __future__ import annotations

from typing import Optional

import torch

from src.implementations.poly_reference import poly_linear_attention
from src.kernels import HAS_TRITON

CHUNK = 512
EXACT_PREFIX = 4096


def poly_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    sigma: Optional[float],
    use_triton: bool = True,
    disable: frozenset = frozenset(),
) -> torch.Tensor:
    """Causal polynomial attention over ``[B, H, N, D]`` tensors.

    The master state is float32 unconditionally. A float16 state is faster and
    silently wrong at scale -- it passes at N=16384 and fails at N=65536 with
    1,064,935 failures -- so it is not exposed as an option.

    The fused kernels are used only for float16 CUDA inputs, which is what case
    14's route supplies. Everything else falls through to dense PyTorch, which
    computes the same function.

    ``disable`` turns individual optimizations off by name, so the
    pre-optimization path can run as a second arm in the SAME benchmarking
    session. Identical code drifted 17.5% between sessions on this hardware, so
    a cross-session A/B is not a measurement. Recognised names: ``"diag"``,
    ``"prefix"``, ``"shadow"``, ``"configs"``.
    """
    apply_fn = update_fn = diag_fn = None
    if use_triton and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        from src.kernels.poly_attention_triton import (
            causal_diag,
            quad_apply,
            quad_update,
        )

        from src.kernels import poly_attention_triton as _kernels

        _kernels.USE_MEASURED_CONFIGS = "configs" not in disable
        apply_fn, update_fn = quad_apply, quad_update
        diag_fn = None if "diag" in disable else causal_diag

    return poly_linear_attention(
        q, k, v, scale,
        chunk=CHUNK,
        exact_prefix=EXACT_PREFIX,
        sigma=sigma,
        state_dtype=torch.float32,
        compute_dtype=torch.float16,
        quad_apply=apply_fn,
        quad_update=update_fn,
        causal_diag=diag_fn,
        skip_prefix_chunks="prefix" not in disable,
        quad_shadow="shadow" not in disable,
    )
