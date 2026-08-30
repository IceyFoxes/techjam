"""Measured launch configurations, keyed by kernel, shape and device capability.

Autotune is the wrong mechanism for this workload. Case 14 is a single forward
pass, so search time is not amortised -- it lands directly in the measured wall
clock, and a wide space ran for minutes with the GPU at 1%. The update kernel is
worse still: it accumulates in place, so every trial has to clone the 32 MiB
state via ``restore_value``.

That compile budget is what forced the shipped autotune spaces to be narrow, and
the narrowness cost real performance. The interleaved sweep that produced this
table found the best configuration **outside** the shipped space in every case:

* ``quad_apply``  wanted ``BC=128``; the space held only 32 and 64.
* ``quad_update`` wanted ``BC=128``; the space held only 32 and 64.
* ``causal_diag`` wanted ``num_stages=3``; the space pinned it at 2.

A measured table gets the benefit without the search. Every entry names the
benchmark run that produced it -- an unknown key returns ``None`` and the caller
falls back to the narrow autotune, which is slow but correct, whereas a guessed
entry would be fast and wrong.
"""

from __future__ import annotations

_RTX4060_SOURCE = (
    "research/benchmarks/2026-08-30-rtx4060-bfbea79/config-sweep.json"
)
_RTX5080_C512_SOURCE = (
    "research/benchmarks/2026-08-30-rtx5080-8567f3f-sm120/"
    "config-sweep-m16-c512.json"
)
_RTX5080_C352_SOURCE = (
    "research/benchmarks/2026-08-30-rtx5080-8567f3f-sm120/"
    "config-sweep-m16-c352.json"
)

# The final Case 14 chunk has C=352. Padding it to the regular C=512 launch
# reuses the already-compiled kernel specializations; causal masking makes the
# added future keys invisible to the 352 real query rows. The direct C=352
# sweep above is retained as evidence for the alternative we rejected after
# cold-start measurement.
PADDED_CHUNKS = {
    ((16, 352, 64, 64), (12, 0)): {
        "chunk": 512,
        "source": _RTX5080_C352_SOURCE,
    },
}

_RTX5080_END_TO_END_SOURCE = (
    "research/benchmarks/2026-08-30-rtx5080-8567f3f-sm120/"
    "full-case14-poly-updateonly-cold-warm.json"
)

# One-shot end-to-end policy. On sm_120 the apply and diagonal kernels save
# steady-state time but their two extra JIT compilations cost more than they
# save in the official forward. Keeping only quad_update wins both cold and
# warm against exact Flash; every entry must name its end-to-end measurement.
CASE14_POLICIES = {
    ((16, 100000, 64, 64), (12, 0)): {
        "disable": frozenset(("apply", "diag")),
        "source": _RTX5080_END_TO_END_SOURCE,
    },
}

# (kernel, (M, C, D, V), (capability major, minor)) -> config
CONFIGS = {
    # Ada (sm_89), RTX 4060 Laptop, at case 14's chunk shape M=32, C=512, D=V=64.
    ("quad_apply", (32, 512, 64, 64), (8, 9)): {
        "BC": 128, "BI": 1, "num_warps": 4, "num_stages": 2,
        "source": _RTX4060_SOURCE,
    },
    ("quad_update", (32, 512, 64, 64), (8, 9)): {
        "BC": 128, "BI": 1, "num_warps": 4, "num_stages": 2,
        "source": _RTX4060_SOURCE,
    },
    ("causal_diag", (32, 512, 64, 64), (8, 9)): {
        "BC": 64, "BK": 64, "num_warps": 4, "num_stages": 3,
        "source": _RTX4060_SOURCE,
    },
    # Blackwell (sm_120), RTX 5080, official Case 14 streamed at batch 1.
    ("quad_apply", (16, 512, 64, 64), (12, 0)): {
        "BC": 128, "BI": 2, "num_warps": 4, "num_stages": 3,
        "source": _RTX5080_C512_SOURCE,
    },
    ("quad_update", (16, 512, 64, 64), (12, 0)): {
        "BC": 32, "BI": 1, "num_warps": 4, "num_stages": 2,
        "source": _RTX5080_C512_SOURCE,
    },
    ("causal_diag", (16, 512, 64, 64), (12, 0)): {
        "BC": 64, "BK": 64, "num_warps": 4, "num_stages": 3,
        "source": _RTX5080_C512_SOURCE,
    },
}


def lookup(kernel, key, capability):
    """The measured config, or ``None`` when this combination was never measured.

    ``key`` is ``(M, C, D, V)`` and ``capability`` is
    ``torch.cuda.get_device_capability()``.
    The returned dict is safe to splat into a Triton launch: the provenance field
    is stripped.
    """
    entry = CONFIGS.get((kernel, tuple(key), tuple(capability)))
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if k != "source"}


def padded_chunk_size(key, capability):
    """Return a measured canonical chunk size, or ``None`` for no padding."""
    entry = PADDED_CHUNKS.get((tuple(key), tuple(capability)))
    return None if entry is None else entry["chunk"]


def case14_disabled_optimizations(key, capability):
    """Return the measured end-to-end policy for a Case 14 attention call."""
    entry = CASE14_POLICIES.get((tuple(key), tuple(capability)))
    return frozenset() if entry is None else entry["disable"]
