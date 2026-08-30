"""Custom GPU kernels. Import is safe without Triton; callers check HAS_TRITON."""

from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    HAS_TRITON = True
except ImportError:  # pragma: no cover - environments without Triton
    HAS_TRITON = False

__all__ = ["HAS_TRITON"]
