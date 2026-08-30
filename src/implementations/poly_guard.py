"""Runtime validity check for the polynomial attention approximation.

The degree-2 fit is accurate only while scores stay small. That is a property of
the benchmark's random initialisation -- measured ``sigma = 0.3336`` -- not of
attention in general; under trained weights scores are far larger and the fit
would be poor. So the route is gated on a runtime measurement rather than an
assumption.

The same statistic does double duty: it supplies the coefficients of the
L2-optimal degree-2 fit (see ``poly_reference.hermite_coefficients``) and decides
whether to run the approximation at all.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# Measured 30 August 2026. Swept by scaling the q_proj/k_proj weights of both
# the reference and the candidate (``src/validate_poly.py --scale-qk-weights``)
# and recording where the official criterion first fails, at N=8192 against the
# dense reference:
#
#     sigma 0.3339  0 failures      <- the benchmark's own operating point
#     sigma 0.4040  0 failures
#     sigma 0.4808  0 failures      <- largest verified pass
#     sigma 0.5217  21 failures     <- first observed failure
#     sigma 0.5642  308 failures
#     sigma 0.6544  9,566 failures
#     sigma 0.7512  56,801 failures
#
# 0.45 sits 35% above the operating point -- far outside its seed-to-seed spread
# of 0.3327 to 0.3343, so it cannot cause a spurious fallback -- and below 0.4808,
# so the route never runs in the untested band between the last pass and the
# first failure.
#
# The provisional value before this sweep was 0.60, which would have permitted
# sigma values that fail. A guessed ceiling is weaker protection than it looks.
#
# Note this is conservative for the target shape: it was measured at N=8192,
# and attention contributes less to the residual stream as N grows, so the
# tolerance at N=100000 is more forgiving, not less.
SIGMA_CEILING = 0.45


def estimate_sigma(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    samples: int = 512,
    seed: int = 0,
) -> float:
    """Standard deviation of the scaled scores, from a random row sample.

    ``q``, ``k`` are ``[B, H, N, D]``. Sampling ``samples`` rows and taking all
    pairs gives ``samples^2`` scores per head, which is ample: 512 rows recovers
    the population value to about three decimal places for a few MiB.

    Costs one device-to-host synchronization. Call it once per forward from the
    eager dispatch layer -- never inside a compiled or graph-replayed region.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be [B, H, N, D]")
    n = q.shape[2]
    count = min(samples, n)
    generator = torch.Generator(device=q.device).manual_seed(seed)
    index = torch.randperm(n, generator=generator, device=q.device)[:count]
    qs = q[:, :, index].float()
    ks = k[:, :, index].float()
    scores = (qs @ ks.transpose(-2, -1)) * scale
    return float(scores.std().item())


def poly_is_safe(sigma: Optional[float]) -> bool:
    """True when the measured score spread is inside the validated range."""
    if sigma is None:
        return False
    if math.isnan(sigma) or math.isinf(sigma):
        return False
    return 0.0 < sigma <= SIGMA_CEILING
