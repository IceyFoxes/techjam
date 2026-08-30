"""Chunked order-2 polynomial linear attention -- the numerical oracle.

Approximates ``exp(s)`` by the degree-2 polynomial that is L2-optimal under the
measured score distribution ``s ~ N(0, sigma^2)``. The Gauss-Hermite projection
gives ``exp(s) ~= exp(sigma^2/2) * [(1 - sigma^2/2) + s + s^2/2]``; the common
factor cancels in the softmax normalisation, so only the constant term differs
from a plain Taylor expansion at zero. That constant measured 2.3x more accurate
than plain Taylor at no cost.

With ``a = q*sqrt(scale)`` and ``b = k*sqrt(scale)`` so that ``a.b = s``, and
``phi2(x) = flatten(x x^T)`` so that ``<phi2(a), phi2(b)> = (a.b)^2``, causal
attention becomes a chunked scan over a running state. The strict prefix
contributes through that state; the diagonal chunk is computed with exact
``exp``.

See ``research/attention-softmax/long-sequence-attention.md`` for the
measurements that justify every constant here, and ``triton-kernel-spec.md``
section 3 for the derivation.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def phi2(x: torch.Tensor) -> torch.Tensor:
    """``[..., C, D] -> [..., C, D*D]``, the flattened outer product ``x x^T``."""
    return (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)


def hermite_coefficients(sigma: Optional[float]) -> tuple[float, float, float]:
    """``(c0, c1, c2)`` for ``w(s) = c0 + c1*s + c2*s^2``; ``None`` gives Taylor.

    The L2-optimal degree-2 fit to ``exp(s)`` under ``s ~ N(0, sigma^2)`` is

        exp(s) ~= exp(sigma^2/2) * [ (1 - sigma^2/2) + s + s^2/2 ]

    **The ``exp(sigma^2/2)`` factor must be kept.** An earlier version dropped it,
    reasoning that a constant factor cancels in the softmax normalisation. It
    does not: the diagonal chunk is computed with unscaled ``exp``, so dropping
    the factor puts the inter-chunk polynomial on a different scale from the
    intra-chunk exact block -- about 5.6% low at sigma=0.334. Measured relative
    rms against SDPA, the three candidates are:

        N=2048/4096/8192   plain Taylor    0.00873 / 0.01073 / 0.00632
                           factor dropped  0.01105 / 0.01237 / 0.00525
                           scale-consistent 0.00576 / 0.00708 / 0.00420

    so dropping the factor is *worse than plain Taylor* at two of the three.
    """
    if sigma is None:
        return 1.0, 1.0, 0.5
    gain = math.exp(0.5 * sigma * sigma)
    return gain * max(1.0 - 0.5 * sigma * sigma, 1e-3), gain, 0.5 * gain


def poly_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    chunk: int = 512,
    exact_prefix: int = 4096,
    sigma: Optional[float] = None,
    state_dtype: torch.dtype = torch.float32,
    compute_dtype: torch.dtype = torch.float16,
    quad_apply: Optional[Callable] = None,
    quad_update: Optional[Callable] = None,
    causal_diag: Optional[Callable] = None,
    skip_prefix_chunks: bool = True,
) -> torch.Tensor:
    """Causal attention with a degree-2 polynomial feature map.

    ``q``, ``k``, ``v`` are ``[B, H, N, D]``. Returns ``[B, H, N, D]``.

    ``quad_apply(a, s) -> [M, C, V]`` and ``quad_update(b, v, out) -> None`` are
    optional accelerated implementations of the two quadratic-term operations.
    When ``None``, dense PyTorch is used.

    ``causal_diag(a, b, vc) -> (num, den)`` optionally replaces the exact
    diagonal block. The dense form materialises an ``[M, C, C]`` score matrix
    and masks half of it away; a tiled kernel visits only the tiles below the
    diagonal. When ``None``, the dense block is used.
    """
    if q.ndim != 4:
        raise ValueError("q, k, v must be [B, H, N, D]")
    B, H, N, D = q.shape
    M = B * H
    cdt = compute_dtype if compute_dtype is not None else state_dtype

    # q/k/v arrive as strided views from ``_split_heads_view``, so flattening
    # [B, H] into one axis is NOT free -- it forces a full contiguous copy of
    # each tensor. Slicing first keeps those copies at [M, chunk, D] instead of
    # [M, N, D]: 2 MiB per chunk rather than 410 MiB per tensor at case 14's
    # shape. For the same reason ``sqrt(scale)`` is folded into the per-chunk
    # slice rather than applied to a full-length copy up front.
    out = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)

    c0, c1, c2 = hermite_coefficients(sigma)
    rs = math.sqrt(scale)

    dev = q.device
    s_const = torch.zeros(M, 1, D, device=dev, dtype=state_dtype)
    s_lin = torch.zeros(M, D, D, device=dev, dtype=state_dtype)
    s_quad = torch.zeros(M, D * D, D, device=dev, dtype=state_dtype)
    z_const = torch.zeros(M, 1, 1, device=dev, dtype=state_dtype)
    z_lin = torch.zeros(M, D, 1, device=dev, dtype=state_dtype)
    # The denominator's quadratic term does NOT need a d^2 feature state.
    gram = torch.zeros(M, D, D, device=dev, dtype=state_dtype)

    # Only the dense path needs the mask. The tiled kernel masks the diagonal
    # tile in registers and never visits the tiles above it.
    blocked_full = (
        None
        if causal_diag is not None
        else torch.ones(chunk, chunk, device=dev, dtype=torch.bool).triu(1)
    )

    # Chunks that lie entirely inside the exact prefix have their output
    # overwritten by the SDPA call at the end, so computing it is pure waste --
    # 8 of 196 chunks at exact_prefix=4096, C=512. The STATE UPDATE still runs:
    # every later chunk depends on it.
    # skip_prefix_chunks=False restores the pre-optimization behaviour so it
    # can run as a second arm in the same benchmarking session.
    prefix_end = (
        min(exact_prefix, N) if exact_prefix > 0 and skip_prefix_chunks else 0
    )

    for t0 in range(0, N, chunk):
        t1 = min(t0 + chunk, N)
        C = t1 - t0
        a = (q[:, :, t0:t1] * rs).reshape(M, C, D)
        b = (k[:, :, t0:t1] * rs).reshape(M, C, D)
        vc = v[:, :, t0:t1].reshape(M, C, D)

        def emit_chunk():
            """Numerator, denominator and output store for this chunk."""
            if t0 == 0:
                num = torch.zeros(M, C, D, device=dev, dtype=state_dtype)
                den = torch.zeros(M, C, 1, device=dev, dtype=state_dtype)
            else:
                af = a.to(cdt)
                num = (
                    c0 * s_const.expand(M, C, D)
                    + c1 * (af @ s_lin.to(cdt)).to(state_dtype)
                )
                den = (
                    c0 * z_const.expand(M, C, 1)
                    + c1 * (af @ z_lin.to(cdt)).to(state_dtype)
                )
                den = den + c2 * (
                    (af @ gram.to(cdt)) * af
                ).sum(-1, keepdim=True).to(state_dtype)
                if quad_apply is None:
                    aq = phi2(af)
                    num = num + c2 * (aq @ s_quad.to(cdt)).to(state_dtype)
                    del aq
                else:
                    num = num + c2 * quad_apply(af, s_quad).to(state_dtype)

            # Exact diagonal block. No max subtraction: scores are measured
            # bounded to [-2.203, 2.404], so exp cannot overflow. The guard
            # keeps that true.
            #
            # Kept in the compute dtype end to end. Upcasting to float32 here
            # forced the PV product onto an fp32 SGEMM, off the tensor cores,
            # and tripled the traffic of the [M, C, C] block through exp, the
            # mask and the row sum. Only the row sum accumulates in float32,
            # where ~512 terms of order 1 would otherwise lose bits.
            if causal_diag is None:
                blocked = blocked_full if C == chunk else blocked_full[:C, :C]
                w = torch.exp(a @ b.transpose(-2, -1)).masked_fill_(blocked, 0.0)
                d_num = w @ vc
                d_den = w.sum(-1, keepdim=True, dtype=torch.float32)
                del w
            else:
                d_num, d_den = causal_diag(a, b, vc)
            num = num + d_num.to(state_dtype)
            den = den + d_den.to(state_dtype)

            out[:, :, t0:t1] = (num / den).to(q.dtype).reshape(B, H, C, D)

        if t1 > prefix_end:
            emit_chunk()

        bf = b.to(cdt)
        vfc = vc.to(cdt)
        s_const += vfc.sum(1, keepdim=True).to(state_dtype)
        z_const += float(C)
        s_lin += (bf.transpose(-2, -1) @ vfc).to(state_dtype)
        z_lin += bf.sum(1).unsqueeze(-1).to(state_dtype)
        gram += (bf.transpose(-2, -1) @ bf).to(state_dtype)
        if quad_update is None:
            bq = phi2(bf)
            s_quad += (bq.transpose(-2, -1) @ vfc).to(state_dtype)
            del bq
        else:
            quad_update(bf, vfc, s_quad)

    # Early tokens attend to very few keys, where a relative weight error is not
    # damped by averaging, so the max error lives there. Recomputing the first
    # `exact_prefix` tokens exactly costs (W/N)^2 of the quadratic work -- 0.17%
    # at W=4096, N=100000 -- and removes that tail entirely.
    if exact_prefix > 0:
        w0 = min(exact_prefix, N)
        # Pass 4-D slices. Every fused SDPA backend rejects 3-D input, and the
        # fallback is the quadratic math backend, which materialises an
        # [M, w0, w0] score matrix -- 2.4 GiB and ~72 ms per call at w0=4096.
        # That is invisible to correctness tests, so the cost is pinned by
        # src/tests/test_poly_attention.py::PolyMemoryTests instead.
        out[:, :, :w0] = F.scaled_dot_product_attention(
            q[:, :, :w0], k[:, :, :w0], v[:, :, :w0], is_causal=True, scale=scale
        )
    return out
