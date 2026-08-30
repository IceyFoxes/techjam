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
ACCEPTANCE_MS = 360.0


def _time(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    scale = args.head_dim ** -0.5
    torch.manual_seed(0)
    shape = (1, args.heads, args.n, args.head_dim)
    q, k, v = (
        (torch.randn(shape, device=device) * 0.577).half() for _ in range(3)
    )

    with torch.inference_mode():

        def exact():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale
                )

        results = {
            "exact_flash_ms": _time(exact, args.reps),
            "poly_pytorch_ms": _time(
                lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=False
                ),
                args.reps,
            ),
            "poly_triton_ms": _time(
                lambda: poly_attention_forward(
                    q, k, v, scale, sigma=0.3338, use_triton=True
                ),
                args.reps,
            ),
        }

    results["speedup_vs_exact"] = (
        results["exact_flash_ms"] / results["poly_triton_ms"]
    )
    results["speedup_vs_pytorch_poly"] = (
        results["poly_pytorch_ms"] / results["poly_triton_ms"]
    )
    results["accepted"] = results["poly_triton_ms"] <= ACCEPTANCE_MS
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
                "batch": 1,
                "layers": 1,
                "reps": args.reps,
            },
            "acceptance_threshold_ms": ACCEPTANCE_MS,
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
