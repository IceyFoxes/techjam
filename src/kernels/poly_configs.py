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

_SOURCE = "research/benchmarks/2026-08-30-rtx4060-bfbea79/config-sweep.json"

# (kernel, (C, D, V), (capability major, minor)) -> config
CONFIGS = {
    # Ada (sm_89), RTX 4060 Laptop, at case 14's chunk shape M=32, C=512, D=V=64.
    ("quad_apply", (512, 64, 64), (8, 9)): {
        "BC": 128, "BI": 1, "num_warps": 4, "num_stages": 2, "source": _SOURCE,
    },
    ("quad_update", (512, 64, 64), (8, 9)): {
        "BC": 128, "BI": 1, "num_warps": 4, "num_stages": 2, "source": _SOURCE,
    },
    ("causal_diag", (512, 64, 64), (8, 9)): {
        "BC": 64, "BK": 64, "num_warps": 4, "num_stages": 3, "source": _SOURCE,
    },
}


def lookup(kernel, key, capability):
    """The measured config, or ``None`` when this combination was never measured.

    ``key`` is ``(C, D, V)`` and ``capability`` is ``torch.cuda.get_device_capability()``.
    The returned dict is safe to splat into a Triton launch: the provenance field
    is stripped.
    """
    entry = CONFIGS.get((kernel, tuple(key), tuple(capability)))
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if k != "source"}
