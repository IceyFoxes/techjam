#!/usr/bin/env python3
"""Interleaved launch-configuration sweep for the polynomial kernels.

Sequential sweeps do not work on this hardware: the same configuration measured
0.375 ms alone and 0.865 ms inside a back-to-back sweep. Every candidate is
timed once per round, in rotation, and each one keeps its own minimum -- so
thermal drift is spread across all candidates instead of landing on whichever
ran last.

Output feeds ``src/kernels/poly_configs.py``. Record the JSON under
``research/benchmarks/`` so every table entry can name the run that measured it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import triton

from src.infra.environment import collect_environment, collect_git

def _candidates():
    """(kernel, config) pairs to time. OutOfResources entries drop out later."""
    for bc, bi, w, st in itertools.product(
        (32, 64, 128, 256), (1, 2, 4), (4, 8), (2, 3)
    ):
        yield "quad_apply", {"BC": bc, "BI": bi, "num_warps": w, "num_stages": st}
        yield "quad_update", {"BC": bc, "BI": bi, "num_warps": w, "num_stages": st}
    for bc, bk, w, st in itertools.product((64, 128, 256), (32, 64, 128), (4, 8), (2, 3)):
        yield "causal_diag", {"BC": bc, "BK": bk, "num_warps": w, "num_stages": st}


def _runner(kernel, cfg, tensors, shape):
    """A zero-argument callable launching this kernel with this configuration."""
    from src.kernels import poly_attention_triton as K

    M, C, D, V = shape
    a, b, v, state, shadow, y, num, den = tensors
    if kernel == "quad_apply":
        return lambda: K._quad_apply_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
            a, state, y,
            *a.stride(), *state.stride(), *y.stride(),
            C, D, V, BV=V, **cfg,
        )
    if kernel == "quad_update":
        return lambda: K._quad_update_kernel_static[(D // cfg["BI"], M)](
            b, v, state, shadow,
            *b.stride(), *v.stride(), *state.stride(), *shadow.stride(),
            C, D, V, BV=V, HAS_SHADOW=True, **cfg,
        )
    return lambda: K._causal_diag_kernel_static[(triton.cdiv(C, cfg["BC"]), M)](
        a, b, v, num, den,
        *a.stride(), *b.stride(), *v.stride(), *num.stride(), *den.stride(),
        C, D, V, BV=V, **cfg,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--c", type=int, default=512)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--v", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    M, C, D, V = args.m, args.c, args.d, args.v
    if min(M, C, D, V, args.rounds) <= 0:
        raise ValueError("shape dimensions and rounds must be positive")
    shape = (M, C, D, V)

    gen = torch.Generator(device="cuda").manual_seed(0)
    kw = dict(generator=gen, device="cuda", dtype=torch.float16)
    tensors = (
        torch.randn(M, C, D, **kw) * 0.2,                                  # a
        torch.randn(M, C, D, **kw) * 0.2,                                  # b
        torch.randn(M, C, V, **kw) * 0.2,                                  # v
        torch.zeros(M, D * D, V, device="cuda", dtype=torch.float32),      # state
        torch.zeros(M, D * D, V, device="cuda", dtype=torch.float16),      # shadow
        torch.empty(M, C, V, device="cuda", dtype=torch.float16),          # y
        torch.empty(M, C, V, device="cuda", dtype=torch.float32),          # num
        torch.empty(M, C, device="cuda", dtype=torch.float32),             # den
    )

    live = []
    for kernel, cfg in _candidates():
        fn = _runner(kernel, cfg, tensors, shape)
        try:
            fn()                       # compile, and reject over-budget configs
            torch.cuda.synchronize()
        except Exception as exc:       # OutOfResources exceeds the 101,376 B limit
            print(f"skip {kernel} {cfg}: {type(exc).__name__}")
            continue
        live.append(
            {"kernel": kernel, "config": cfg, "fn": fn, "best_ms": float("inf")}
        )
    print(f"{len(live)} configurations compiled")

    for _ in range(args.rounds):
        for entry in live:             # ROTATION, not one candidate at a time
            torch.cuda.synchronize()
            start = time.perf_counter()
            entry["fn"]()
            torch.cuda.synchronize()
            entry["best_ms"] = min(
                entry["best_ms"], (time.perf_counter() - start) * 1e3
            )

    results = [
        {"kernel": e["kernel"], "config": e["config"], "best_ms": e["best_ms"]}
        for e in live
    ]
    results.sort(key=lambda r: (r["kernel"], r["best_ms"]))
    for kernel in ("quad_apply", "quad_update", "causal_diag"):
        ranked = [r for r in results if r["kernel"] == kernel][:3]
        print(f"\n{kernel}:")
        for r in ranked:
            print(f"  {r['best_ms']:.4f} ms  {r['config']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "shape": {"M": M, "C": C, "D": D, "V": V},
                "rounds": args.rounds,
                "environment": collect_environment(torch, torch.device("cuda")),
                "git": collect_git(),
                "results": results,
            },
            handle, indent=2, sort_keys=True,
        )
        handle.write("\n")
    print(f"\nsaved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
