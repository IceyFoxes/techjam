#!/usr/bin/env python3
"""Run an extreme dispatcher case without constructing its unsafe baseline."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from torch_transformer_benchmark import TransformerConfig, resolve_dtype

from src.dispatcher import DispatchingTransformer
from src.infra import load_official_cases
from src.infra.environment import collect_environment, collect_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-only CUDA smoke test for official Case 6 or 14"
    )
    parser.add_argument("--case", type=int, choices=(6, 14), required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument(
        "--forwards",
        type=int,
        default=1,
        help="number of forwards in one process; use 2 to separate cold and warm time",
    )
    parser.add_argument(
        "--disable-poly",
        action="store_true",
        help="force the guarded Case 14 attention module onto exact Flash",
    )
    parser.add_argument(
        "--poly-disable",
        default="",
        help="comma-separated polynomial subpaths to disable for an A/B",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON result path; an existing file is never overwritten",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allowed = {6: ("float32",), 14: ("float16",)}
    if args.dtype not in allowed[args.case]:
        raise ValueError(
            f"Case {args.case} supports only {', '.join(allowed[args.case])}"
        )
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.forwards <= 0:
        raise ValueError("forwards must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("extreme smoke test requires CUDA")
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError("extreme smoke test requires PyTorch 2.13.0+cu130")
    if torch.cuda.get_device_capability() < (8, 0):
        raise RuntimeError("extreme smoke test requires Ampere or newer")

    shape = load_official_cases()[args.case]
    config = TransformerConfig(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.qkv_dim,
        num_heads=shape.heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.layers,
        causal=shape.causal,
    )
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = DispatchingTransformer(config).to(device=device, dtype=dtype).eval()
    if args.disable_poly:
        for layer in model.layers:
            if hasattr(layer.attention, "poly_enabled"):
                layer.attention.poly_enabled = False
    poly_disable = frozenset(filter(None, args.poly_disable.split(",")))
    if poly_disable:
        for layer in model.layers:
            if hasattr(layer.attention, "poly_disable"):
                layer.attention.poly_disable = poly_disable
    x = torch.empty(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=device,
        dtype=dtype,
    )
    x.normal_()
    lengths = torch.full(
        (config.batch_size,),
        config.seq_len,
        device=device,
        dtype=torch.int64,
    )
    if args.padding_ratio:
        minimum = max(1, round(config.seq_len * (1.0 - args.padding_ratio)))
        lengths.random_(minimum, config.seq_len + 1)
    positions = torch.arange(config.seq_len, device=device)
    valid_mask = positions[None, :] < lengths[:, None]
    x.masked_fill_(~valid_mask[..., None], 0)

    elapsed_by_forward = []
    peak_allocated_by_forward = []
    peak_reserved_by_forward = []
    output_shape = [config.batch_size, config.seq_len, config.d_model]
    with torch.inference_mode():
        for _ in range(args.forwards):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            output = model(x, valid_mask)
            torch.cuda.synchronize(device)
            elapsed_by_forward.append(time.perf_counter() - started)
            peak_allocated_by_forward.append(torch.cuda.max_memory_allocated(device))
            peak_reserved_by_forward.append(torch.cuda.max_memory_reserved(device))
            for batch_slice in output.split(1):
                if not bool(torch.isfinite(batch_slice).all()):
                    raise RuntimeError("candidate smoke output contains non-finite values")
            del batch_slice
            del output
    print(f"case={args.case} dtype={args.dtype} route={model.last_route.backend}")
    print(f"shape={tuple(output_shape)} elapsed_s={elapsed_by_forward}")
    print(
        "peak_allocated_mib="
        f"{max(peak_allocated_by_forward) / 2**20:.3f} "
        "peak_reserved_mib="
        f"{max(peak_reserved_by_forward) / 2**20:.3f}"
    )
    if args.output is not None:
        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "git": collect_git(),
            "environment": collect_environment(torch, device),
            "official_case_id": args.case,
            "config": asdict(config),
            "dtype": args.dtype,
            "padding_ratio": args.padding_ratio,
            "route": model.last_route.backend,
            "poly_disabled": args.disable_poly,
            "poly_disabled_optimizations": sorted(poly_disable),
            "batch_chunk_size": model.last_extreme_chunk_size,
            "output_shape": output_shape,
            "output_finite": True,
            "numerical_correctness": (
                "not checked: immutable Case 14 baseline is not runnable at "
                "this scale"
                if args.case == 14
                else "not checked by candidate-only smoke runner"
            ),
            "elapsed_seconds": elapsed_by_forward[0],
            "elapsed_seconds_by_forward": elapsed_by_forward,
            "peak_allocated_bytes": max(peak_allocated_by_forward),
            "peak_allocated_bytes_by_forward": peak_allocated_by_forward,
            "peak_reserved_bytes": max(peak_reserved_by_forward),
            "peak_reserved_bytes_by_forward": peak_reserved_by_forward,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"result saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
