#!/usr/bin/env python3
"""Attention-core latency at case-14 shapes: exact flash vs polynomial paths.

One sample x one layer, which is what case 14's route actually executes -- it
streams 1-2 samples at a time (``choose_batch_chunk_size`` selected 2 on a 24 GB
L4), so a B=1 slice is the representative unit of work rather than a
simplification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.implementations.poly_attention import poly_attention_forward
from src.infra.environment import collect_environment, collect_git

# Reference points measured on the RTX 4060 Laptop before this kernel existed.
EXACT_FLASH_MS = 719.8
POLY_PYTORCH_MS = 603.9
ACCEPTANCE_SPEEDUP = 2.0


def _time_interleaved(variants, reps=3):
    """Time several callables by alternating them, and take each one's min.

    Measuring variants back to back is biased on a laptop GPU: by the third
    variant the card has been at 100% for seconds and has dropped clocks. That
    inflated the polynomial path by ~40% (392 ms measured last, against 275 ms
    measured alone). Interleaving spreads any drift across all variants instead
    of loading it onto whichever runs last. The repository hit the same class of
    bug before -- see the boost-clock fix in
    research/benchmarks/README.md -- so this is a known hazard here.
    """
    for fn in variants.values():
        fn()  # warm, and absorb Triton autotuning
    torch.cuda.synchronize()
    best = {name: float("inf") for name in variants}
    for _ in range(reps):
        for name, fn in variants.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best[name] = min(best[name], time.perf_counter() - start)
    return {name: value * 1e3 for name, value in best.items()}


def _peak_mib_table(q, k, v, scale):
    """Peak allocation of each path, MiB. VRAM is a binding constraint for
    case 14, so it is recorded alongside latency rather than assumed."""
    out = {}
    with torch.inference_mode():
        def measure(fn):
            fn()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            result = fn()
            peak = torch.cuda.max_memory_allocated() - base
            del result
            return peak / 2 ** 20

        def exact():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )

        out["exact_flash"] = measure(exact)
        out["poly_triton"] = measure(
            lambda: poly_attention_forward(q, k, v, scale, sigma=0.3338)
        )
    out["poly_overhead"] = out["poly_triton"] - out["exact_flash"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="samples per chunk; case 14's route selected 2 on a 24 GB L4",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    scale = args.head_dim ** -0.5
    torch.manual_seed(0)
    # STRIDED VIEWS, matching what _split_heads_view actually hands the module.
    # Building contiguous [B, H, N, D] tensors instead makes the internal
    # reshape free and understates the real cost -- an earlier version of this
    # script did exactly that and reported 342 ms where the real figure was 470.
    packed = [
        (torch.randn(args.batch, args.n, args.heads * args.head_dim,
                     device=device) * 0.577).half()
        for _ in range(3)
    ]
    q, k, v = (
        t.view(args.batch, args.n, args.heads, args.head_dim).transpose(1, 2)
        for t in packed
    )

    with torch.inference_mode():

        def exact():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )

        timings = _time_interleaved(
            {
                "exact_flash_ms": exact,
                "poly_pytorch_ms": lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=False
                ),
                "poly_triton_ms": lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=True
                ),
            },
            args.reps,
        )
        results = dict(timings)

    results["peak_mib"] = _peak_mib_table(q, k, v, scale)
    results["speedup_vs_exact"] = (
        results["exact_flash_ms"] / results["poly_triton_ms"]
    )
    results["speedup_vs_pytorch_poly"] = (
        results["poly_pytorch_ms"] / results["poly_triton_ms"]
    )
    # Ratio, not milliseconds. The absolute 360 ms budget was derived for a
    # single sample; it says nothing at --batch 2, where there is twice the work.
    results["accepted"] = results["speedup_vs_exact"] >= ACCEPTANCE_SPEEDUP
    results["beats_pytorch_poly"] = (
        results["poly_triton_ms"] < results["poly_pytorch_ms"]
    )
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.output is not None:
        payload = {
            "schema_version": 1,
            "config": {
                "n": args.n,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dtype": "float16",
                "batch": args.batch,
                "layers": 1,
                "reps": args.reps,
            },
            "acceptance_threshold_speedup": ACCEPTANCE_SPEEDUP,
            "environment": collect_environment(torch, device),
            "git": collect_git(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"result saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
