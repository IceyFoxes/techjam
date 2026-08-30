"""Fused degree-2 polynomial feature-map kernels.

The order-2 polynomial path performs 12x fewer FLOPs than exact attention yet
realises only 1.19x, because it writes and re-reads a ``[C, D*D]`` feature
tensor -- about 51 GB per sample-layer at case 14's shapes. These kernels
generate that tensor in registers and consume it directly, so it never reaches
HBM.

The ``D*D`` feature axis is indexed by pairs ``(i, j)``. A feature block is a
contiguous range of ``i`` with all ``j``, which is a contiguous slab of the
state -- that is what makes the tiling work.

State is float32 in HBM and converted to float16 on load. A float16 master state
is about 1.1x faster and silently wrong at scale: it passes at N=16384 and fails
at N=65536 with over a million failures. See
``research/attention-softmax/long-sequence-attention.md`` section 5.2.
"""

from __future__ import annotations

import torch

from src.kernels import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _phi_tile(a, ai, BC: tl.constexpr, BI: tl.constexpr, D: tl.constexpr):
        """Outer-product slab: ``phi[c, i*D + j] = ai[c, i] * a[c, j]``.

        Shared by both kernels, and by the Phase 2 fused scan when it lands.
        """
        return tl.reshape(ai[:, :, None] * a[:, None, :], (BC, BI * D))

    @triton.autotune(
        configs=[
            triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
            for bc in (32, 64)
            for bi in (1, 2, 4)
            for w in (4, 8)
        ],
        key=["C", "D", "V"],
    )
    @triton.jit
    def _quad_apply_kernel(
        a_ptr, s_ptr, y_ptr,
        stride_am, stride_ac, stride_ad,
        stride_sm, stride_sf, stride_sv,
        stride_ym, stride_yc, stride_yv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_c = pid_c * BC + tl.arange(0, BC)
        offs_d = tl.arange(0, D)
        offs_v = tl.arange(0, BV)
        mask_c = offs_c < C
        mask_v = offs_v < V

        a_base = a_ptr + pid_m * stride_am
        a = tl.load(
            a_base + offs_c[:, None] * stride_ac + offs_d[None, :] * stride_ad,
            mask=mask_c[:, None], other=0.0,
        )

        acc = tl.zeros((BC, BV), dtype=tl.float32)
        for i0 in range(0, D, BI):
            offs_i = i0 + tl.arange(0, BI)
            ai = tl.load(
                a_base + offs_c[:, None] * stride_ac + offs_i[None, :] * stride_ad,
                mask=mask_c[:, None], other=0.0,
            )
            phi = _phi_tile(a, ai, BC, BI, D)
            offs_f = i0 * D + tl.arange(0, BI * D)
            s = tl.load(
                s_ptr + pid_m * stride_sm
                + offs_f[:, None] * stride_sf + offs_v[None, :] * stride_sv,
                mask=mask_v[None, :], other=0.0,
            )
            acc += tl.dot(phi.to(tl.float16), s.to(tl.float16))

        tl.store(
            y_ptr + pid_m * stride_ym
            + offs_c[:, None] * stride_yc + offs_v[None, :] * stride_yv,
            acc.to(tl.float16),
            mask=mask_c[:, None] & mask_v[None, :],
        )

    @triton.autotune(
        configs=[
            triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
            for bc in (32, 64)
            for bi in (1, 2)
            for w in (4, 8)
        ],
        key=["C", "D", "V"],
        # This kernel accumulates into o_ptr in place, and the autotuner runs
        # each candidate config several times to time it. Without restore_value
        # every tuning trial folds the chunk in again, so the state comes out
        # multiplied by the trial count -- and only on the first call for a
        # given key, since later calls hit the config cache and run once. That
        # makes it a shape-dependent, cache-order-dependent wrong answer.
        restore_value=["o_ptr"],
    )
    @triton.jit
    def _quad_update_kernel(
        b_ptr, v_ptr, o_ptr,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_om, stride_of, stride_ov,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        pid_m = tl.program_id(1)

        i0 = pid_i * BI
        offs_d = tl.arange(0, D)
        offs_i = i0 + tl.arange(0, BI)
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V

        b_base = b_ptr + pid_m * stride_bm
        v_base = v_ptr + pid_m * stride_vm
        acc = tl.zeros((BI * D, BV), dtype=tl.float32)

        for c0 in range(0, C, BC):
            offs_c = c0 + tl.arange(0, BC)
            mask_c = offs_c < C
            b = tl.load(
                b_base + offs_c[:, None] * stride_bc + offs_d[None, :] * stride_bd,
                mask=mask_c[:, None], other=0.0,
            )
            bi = tl.load(
                b_base + offs_c[:, None] * stride_bc + offs_i[None, :] * stride_bd,
                mask=mask_c[:, None], other=0.0,
            )
            phi = _phi_tile(b, bi, BC, BI, D)
            vt = tl.load(
                v_base + offs_c[:, None] * stride_vc + offs_v[None, :] * stride_vv,
                mask=mask_c[:, None] & mask_v[None, :], other=0.0,
            )
            acc += tl.dot(tl.trans(phi).to(tl.float16), vt.to(tl.float16))

        offs_f = i0 * D + tl.arange(0, BI * D)
        o_addr = (
            o_ptr + pid_m * stride_om
            + offs_f[:, None] * stride_of + offs_v[None, :] * stride_ov
        )
        prev = tl.load(o_addr, mask=mask_v[None, :], other=0.0)
        tl.store(o_addr, prev + acc, mask=mask_v[None, :])


def quad_apply(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``phi2(a) @ s`` without materialising ``phi2(a)``.

    ``a`` is ``[M, C, D]`` (float16), ``s`` is ``[M, D*D, V]`` (float32).
    Returns ``[M, C, V]`` in ``a``'s dtype.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = a.shape
    if s.shape[0] != M or s.shape[1] != D * D:
        raise ValueError(
            f"state shape {tuple(s.shape)} does not match a {tuple(a.shape)}"
        )
    V = s.shape[2]
    a = a.contiguous()
    s = s.contiguous()
    y = torch.empty((M, C, V), device=a.device, dtype=a.dtype)
    BV = max(16, triton.next_power_of_2(V))
    grid = lambda meta: (triton.cdiv(C, meta["BC"]), M)  # noqa: E731
    _quad_apply_kernel[grid](
        a, s, y,
        a.stride(0), a.stride(1), a.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        C, D, V, BV=BV,
    )
    return y


def quad_update(b: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
    """Accumulate ``phi2(b)^T @ v`` into ``out`` in place.

    ``b`` is ``[M, C, D]`` float16, ``v`` is ``[M, C, V]`` float16, and ``out``
    is ``[M, D*D, V]`` float32. ``out`` is the running state, so this adds
    rather than overwrites.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = b.shape
    V = v.shape[2]
    if out.shape != (M, D * D, V):
        raise ValueError(f"out shape {tuple(out.shape)} != {(M, D * D, V)}")
    if out.dtype != torch.float32:
        # Not a style preference. A float16 accumulator passes at N=16384 and
        # fails at N=65536 with over a million failures, so it must fail loudly
        # here rather than compute something plausible.
        raise ValueError("the master state must be float32; see the module docstring")
    b = b.contiguous()
    v = v.contiguous()
    BV = max(16, triton.next_power_of_2(V))
    grid = lambda meta: (triton.cdiv(D, meta["BI"]), M)  # noqa: E731
    _quad_update_kernel[grid](
        b, v, out,
        b.stride(0), b.stride(1), b.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        C, D, V, BV=BV,
    )
