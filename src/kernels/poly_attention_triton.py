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

**On the autotune space.** It is deliberately narrow -- 12 configs for apply, 8
for update, ``num_stages`` fixed at 2. Widening it to 54 each (adding
``BC=128`` and ``num_stages`` 3 and 4) was tried and abandoned: with four
distinct shape keys per run it is roughly 400 kernel compilations, which ran for
several minutes with the GPU at 1% before being killed. Case 14 is a single
forward pass, not a training loop, so that compile time is not amortised -- on a
cold Triton cache it lands directly in the measured wall clock. Widen this only
with evidence that the compile cost is paid back.
"""

from __future__ import annotations

import torch

from src.kernels import HAS_TRITON
from src.kernels.poly_configs import lookup as _measured_config

# Benchmark switch only. Setting it False forces every launch back onto the
# narrow autotune, so the pre-F4 path can run as a second arm in the SAME
# session -- cross-session comparisons are not measurements on this hardware.
# Production code must leave this True; src/tests/test_poly_configs.py pins the
# table itself.
USE_MEASURED_CONFIGS = True


def _lookup_config(kernel, key, capability):
    if not USE_MEASURED_CONFIGS:
        return None
    return _measured_config(kernel, key, capability)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _phi_tile(a, ai, BC: tl.constexpr, BI: tl.constexpr, D: tl.constexpr):
        """Outer-product slab: ``phi[c, i*D + j] = ai[c, i] * a[c, j]``.

        Shared by both kernels, and by the Phase 2 fused scan when it lands.

        **Do not try to hoist the caller's ``ai`` load.** ``ai`` is a column
        subset of ``a``, which is already resident, so re-loading it each
        iteration looks redundant. It is not worth removing, and the obvious
        removal is a disaster. Measured at ``BC=128, BI=1``, M=32, C=512, D=64:

            as shipped   9 ld.global, 248 regs,  0 spills, 0.4606 ms
            hoisted    520 ld.global, 255 regs, 32 spills, 8.8300 ms

        Slicing ``ai`` out of the resident tile needs a compile-time index,
        which means ``tl.static_range``, which fully unrolls the 64-iteration
        feature loop and blows the register budget -- the 520 loads are spill
        traffic, not data. The output was bitwise identical (max diff 0.0), so
        the transformation was correct and 19x slower.

        Triton already compiles the "redundant" load into 9 global loads total
        for the whole kernel. There is nothing there to win. Recorded as Stage 0
        task F6 in research/attention-softmax/integrated-kernel-spec.md.
        """
        return tl.reshape(ai[:, :, None] * a[:, None, :], (BC, BI * D))

    @triton.jit
    def _causal_diag_body(
        a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
        stride_am, stride_ac, stride_ad,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_nm, stride_nc, stride_nv,
        stride_dm, stride_dc,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)

        offs_c = pid_c * BC + tl.arange(0, BC)
        offs_d = tl.arange(0, D)
        offs_v = tl.arange(0, BV)
        mask_c = offs_c < C
        mask_v = offs_v < V

        a = tl.load(
            a_ptr + pid_m * stride_am
            + offs_c[:, None] * stride_ac + offs_d[None, :] * stride_ad,
            mask=mask_c[:, None], other=0.0,
        )

        acc_num = tl.zeros((BC, BV), dtype=tl.float32)
        acc_den = tl.zeros((BC,), dtype=tl.float32)

        # Key tiles beyond this row block are entirely masked out, so the loop
        # simply does not visit them. That is the whole point of the kernel: at
        # BC=BK=128 with C=512 it runs 10 of the 16 tiles, and only the diagonal
        # tile needs a mask at all. The dense path it replaces computes all 16
        # and then masks 6 of them away.
        k_end = tl.minimum((pid_c + 1) * BC, C)
        for k0 in range(0, k_end, BK):
            offs_k = k0 + tl.arange(0, BK)
            mask_k = offs_k < C
            b = tl.load(
                b_ptr + pid_m * stride_bm
                + offs_k[:, None] * stride_bc + offs_d[None, :] * stride_bd,
                mask=mask_k[:, None], other=0.0,
            )
            s = tl.dot(a, tl.trans(b))
            w = tl.exp(s)
            w = tl.where(
                (offs_c[:, None] >= offs_k[None, :]) & mask_k[None, :], w, 0.0
            )
            # Rounded to float16 before the V product, matching what the dense
            # path does. den sums the rounded weights for the same reason: the
            # numerator and denominator must see the same w.
            wh = w.to(tl.float16)
            vt = tl.load(
                v_ptr + pid_m * stride_vm
                + offs_k[:, None] * stride_vc + offs_v[None, :] * stride_vv,
                mask=mask_k[:, None] & mask_v[None, :], other=0.0,
            )
            acc_num += tl.dot(wh, vt)
            acc_den += tl.sum(wh.to(tl.float32), axis=1)

        tl.store(
            num_ptr + pid_m * stride_nm
            + offs_c[:, None] * stride_nc + offs_v[None, :] * stride_nv,
            acc_num,
            mask=mask_c[:, None] & mask_v[None, :],
        )
        tl.store(
            den_ptr + pid_m * stride_dm + offs_c * stride_dc,
            acc_den,
            mask=mask_c,
        )

    @triton.jit
    def _quad_apply_body(
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

    @triton.jit
    def _quad_update_body(
        b_ptr, v_ptr, o_ptr, sh_ptr,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_om, stride_of, stride_ov,
        stride_shm, stride_shf, stride_shv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
        HAS_SHADOW: tl.constexpr,
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
        updated = prev + acc
        tl.store(o_addr, updated, mask=mask_v[None, :])
        if HAS_SHADOW:
            # The apply kernel converts to float16 before its dot anyway, so
            # reading this instead of the master is bitwise-identical and moves
            # half the bytes -- and every apply program reads the whole state.
            tl.store(
                sh_ptr + pid_m * stride_shm
                + offs_f[:, None] * stride_shf + offs_v[None, :] * stride_shv,
                updated.to(tl.float16),
                mask=mask_v[None, :],
            )


    # --- Entry points -------------------------------------------------------
    #
    # Each kernel has two: an autotuned one for unknown shapes and devices, and
    # a static one launched at a configuration measured offline (see
    # ``src/kernels/poly_configs.py``). Autotune is the wrong default here --
    # case 14 is a single forward pass, so search time is not amortised and
    # lands directly in the measured wall clock, and the update kernel has to
    # clone the 32 MiB state on every trial because it accumulates in place.
    #
    # Both entry points call the SAME body, so they cannot drift apart.

    _DIAG_CONFIGS = [
        triton.Config({"BC": bc, "BK": bk}, num_warps=w, num_stages=2)
        for bc in (64, 128)
        for bk in (64, 128)
        for w in (4, 8)
    ]
    _APPLY_CONFIGS = [
        triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
        for bc in (32, 64)
        for bi in (1, 2, 4)
        for w in (4, 8)
    ]
    _UPDATE_CONFIGS = [
        triton.Config({"BC": bc, "BI": bi}, num_warps=w, num_stages=2)
        for bc in (32, 64)
        for bi in (1, 2)
        for w in (4, 8)
    ]

    @triton.autotune(configs=_DIAG_CONFIGS, key=["C", "D", "V"])
    @triton.jit
    def _causal_diag_kernel(
        a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
        stride_am, stride_ac, stride_ad,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_nm, stride_nc, stride_nv,
        stride_dm, stride_dc,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        _causal_diag_body(
            a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
            stride_am, stride_ac, stride_ad,
            stride_bm, stride_bc, stride_bd,
            stride_vm, stride_vc, stride_vv,
            stride_nm, stride_nc, stride_nv,
            stride_dm, stride_dc,
            C, D, V, BC, BK, BV,
        )

    @triton.jit
    def _causal_diag_kernel_static(
        a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
        stride_am, stride_ac, stride_ad,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_nm, stride_nc, stride_nv,
        stride_dm, stride_dc,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        _causal_diag_body(
            a_ptr, b_ptr, v_ptr, num_ptr, den_ptr,
            stride_am, stride_ac, stride_ad,
            stride_bm, stride_bc, stride_bd,
            stride_vm, stride_vc, stride_vv,
            stride_nm, stride_nc, stride_nv,
            stride_dm, stride_dc,
            C, D, V, BC, BK, BV,
        )

    @triton.autotune(configs=_APPLY_CONFIGS, key=["C", "D", "V"])
    @triton.jit
    def _quad_apply_kernel(
        a_ptr, s_ptr, y_ptr,
        stride_am, stride_ac, stride_ad,
        stride_sm, stride_sf, stride_sv,
        stride_ym, stride_yc, stride_yv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        _quad_apply_body(
            a_ptr, s_ptr, y_ptr,
            stride_am, stride_ac, stride_ad,
            stride_sm, stride_sf, stride_sv,
            stride_ym, stride_yc, stride_yv,
            C, D, V, BC, BI, BV,
        )

    @triton.jit
    def _quad_apply_kernel_static(
        a_ptr, s_ptr, y_ptr,
        stride_am, stride_ac, stride_ad,
        stride_sm, stride_sf, stride_sv,
        stride_ym, stride_yc, stride_yv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
    ):
        _quad_apply_body(
            a_ptr, s_ptr, y_ptr,
            stride_am, stride_ac, stride_ad,
            stride_sm, stride_sf, stride_sv,
            stride_ym, stride_yc, stride_yv,
            C, D, V, BC, BI, BV,
        )

    @triton.autotune(
        configs=_UPDATE_CONFIGS,
        key=["C", "D", "V"],
        # This kernel accumulates into o_ptr (and sh_ptr) in place, and the
        # autotuner runs each candidate several times to time it. Without
        # restore_value every trial folds the chunk in again, so the state comes
        # out multiplied by the trial count -- and only on the first call for a
        # given key, since later calls hit the config cache and run once. That
        # makes it a shape-dependent, cache-order-dependent wrong answer.
        restore_value=["o_ptr", "sh_ptr"],
    )
    @triton.jit
    def _quad_update_kernel(
        b_ptr, v_ptr, o_ptr, sh_ptr,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_om, stride_of, stride_ov,
        stride_shm, stride_shf, stride_shv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
        HAS_SHADOW: tl.constexpr,
    ):
        _quad_update_body(
            b_ptr, v_ptr, o_ptr, sh_ptr,
            stride_bm, stride_bc, stride_bd,
            stride_vm, stride_vc, stride_vv,
            stride_om, stride_of, stride_ov,
            stride_shm, stride_shf, stride_shv,
            C, D, V, BC, BI, BV, HAS_SHADOW,
        )

    @triton.jit
    def _quad_update_kernel_static(
        b_ptr, v_ptr, o_ptr, sh_ptr,
        stride_bm, stride_bc, stride_bd,
        stride_vm, stride_vc, stride_vv,
        stride_om, stride_of, stride_ov,
        stride_shm, stride_shf, stride_shv,
        C, D: tl.constexpr, V: tl.constexpr,
        BC: tl.constexpr, BI: tl.constexpr, BV: tl.constexpr,
        HAS_SHADOW: tl.constexpr,
    ):
        _quad_update_body(
            b_ptr, v_ptr, o_ptr, sh_ptr,
            stride_bm, stride_bc, stride_bd,
            stride_vm, stride_vc, stride_vv,
            stride_om, stride_of, stride_ov,
            stride_shm, stride_shf, stride_shv,
            C, D, V, BC, BI, BV, HAS_SHADOW,
        )


def quad_apply(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``phi2(a) @ s`` without materialising ``phi2(a)``.

    ``a`` is ``[M, C, D]`` (float16), ``s`` is ``[M, D*D, V]`` float32 **or
    float16**. Returns ``[M, C, V]`` in ``a``'s dtype.

    A float16 ``s`` gives a bitwise-identical result, because the kernel
    converts the state to float16 before its dot regardless -- see
    ``quad_update``'s ``shadow`` argument.
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
    args = (
        a, s, y,
        a.stride(0), a.stride(1), a.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        C, D, V,
    )
    cfg = _lookup_config(
        "quad_apply", (C, D, V), torch.cuda.get_device_capability(a.device)
    )
    if cfg is None:
        _quad_apply_kernel[lambda meta: (triton.cdiv(C, meta["BC"]), M)](
            *args, BV=BV
        )
    else:
        _quad_apply_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
            *args, BV=BV, **cfg
        )
    return y


def quad_update(
    b: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    shadow: torch.Tensor | None = None,
) -> None:
    """Accumulate ``phi2(b)^T @ v`` into ``out`` in place.

    ``b`` is ``[M, C, D]`` float16, ``v`` is ``[M, C, V]`` float16, and ``out``
    is ``[M, D*D, V]`` float32. ``out`` is the running state, so this adds
    rather than overwrites.

    ``shadow``, when given, is a float16 tensor shaped like ``out`` that
    receives the updated state rounded to float16. ``quad_apply`` can then read
    it instead of the master and move half the bytes, which matters because
    every apply program reads the *whole* state -- 8 times per chunk at
    ``BC=64, C=512``. This is not an fp16 accumulator: the master stays float32
    and the shadow is derived from it every time.
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
    if shadow is not None:
        if shadow.shape != out.shape or shadow.dtype != torch.float16:
            raise ValueError(
                f"shadow must be float16 shaped {tuple(out.shape)}, "
                f"got {shadow.dtype} {tuple(shadow.shape)}"
            )
    b = b.contiguous()
    v = v.contiguous()
    # When there is no shadow the kernel still needs a valid pointer, so `out`
    # is passed twice and HAS_SHADOW switches the store off at compile time.
    target = shadow if shadow is not None else out
    BV = max(16, triton.next_power_of_2(V))
    args = (
        b, v, out, target,
        b.stride(0), b.stride(1), b.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        target.stride(0), target.stride(1), target.stride(2),
        C, D, V,
    )
    has_shadow = shadow is not None
    cfg = _lookup_config(
        "quad_update", (C, D, V), torch.cuda.get_device_capability(b.device)
    )
    if cfg is None:
        _quad_update_kernel[lambda meta: (triton.cdiv(D, meta["BI"]), M)](
            *args, BV=BV, HAS_SHADOW=has_shadow
        )
    else:
        _quad_update_kernel_static[(triton.cdiv(D, cfg["BI"]), M)](
            *args, BV=BV, HAS_SHADOW=has_shadow, **cfg
        )


def causal_diag(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor):
    """``w = tril(exp(a @ b.T))``, then ``(w @ v, w.sum(-1))``, fused.

    ``a`` and ``b`` are ``[M, C, D]`` float16 and ``v`` is ``[M, C, V]``
    float16. Returns ``(num [M, C, V] float32, den [M, C, 1] float32)``.

    The dense path this replaces materialises the full ``[M, C, C]`` score
    matrix and masks half of it away, so ``exp``, the mask, both GEMMs and the
    row sum all pay for the discarded upper triangle. This visits only the tiles
    below the diagonal -- 10 of 16 at ``BC=BK=128, C=512`` -- and never writes
    the score matrix to HBM.

    No max subtraction. Scores are measured bounded to [-2.203, 2.404], so
    ``exp`` cannot overflow; the runtime sigma guard is what keeps that true.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, C, D = a.shape
    if b.shape != a.shape:
        raise ValueError(f"b shape {tuple(b.shape)} != a shape {tuple(a.shape)}")
    if v.shape[0] != M or v.shape[1] != C:
        raise ValueError(f"v shape {tuple(v.shape)} does not match a {tuple(a.shape)}")
    V = v.shape[2]
    a = a.contiguous()
    b = b.contiguous()
    v = v.contiguous()
    num = torch.empty((M, C, V), device=a.device, dtype=torch.float32)
    den = torch.empty((M, C), device=a.device, dtype=torch.float32)
    BV = max(16, triton.next_power_of_2(V))
    args = (
        a, b, v, num, den,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        num.stride(0), num.stride(1), num.stride(2),
        den.stride(0), den.stride(1),
        C, D, V,
    )
    cfg = _lookup_config(
        "causal_diag", (C, D, V), torch.cuda.get_device_capability(a.device)
    )
    if cfg is None:
        _causal_diag_kernel[lambda meta: (triton.cdiv(C, meta["BC"]), M)](
            *args, BV=BV
        )
    else:
        _causal_diag_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
            *args, BV=BV, **cfg
        )
    return num, den.unsqueeze(-1)
