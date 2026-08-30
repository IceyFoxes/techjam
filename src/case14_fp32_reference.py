#!/usr/bin/env python3
"""Run the linear-memory FP32 Case 14 validation oracle sample by sample.

The immutable benchmark cannot hold the explicit attention matrices for Case 14,
and a 16 GiB GPU cannot retain the full FP32 input and output together.  This
runner keeps the reference model resident, generates one batch item at a time,
and immediately reduces each output into a numerical fingerprint.

This command is an oracle feasibility run, not a performance score and not a
replacement for the immutable benchmark.
"""

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

from torch_transformer_benchmark import (
    BaselineTransformer,
    copy_model_weights,
    generate_random_case,
)

from src.implementations.fp32_reference import (
    LinearMemoryFP32Reference,
    case14_config,
    compare_outputs_streamed,
)
from src.infra.environment import collect_environment, collect_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear-memory FP32 reference runner for official Case 14"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument(
        "--validate-dense-n",
        type=int,
        default=1024,
        help="first compare the oracle with the immutable dense reference",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("batch-size and seq-len must be positive")
    if args.validate_dense_n < 0:
        raise ValueError("validate-dense-n must be non-negative")
    if args.input_scale <= 0:
        raise ValueError("input-scale must be positive")


def _dense_validation(
    oracle: LinearMemoryFP32Reference,
    *,
    seq_len: int,
    seed: int,
    input_scale: float,
    device: torch.device,
) -> dict:
    if seq_len == 0:
        return {"status": "skipped"}
    config = case14_config(batch_size=1, seq_len=seq_len)
    dense = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
    copy_model_weights(oracle, dense)
    x, mask = generate_random_case(
        config,
        device,
        torch.float32,
        seed,
        padding_ratio=0.0,
        input_scale=input_scale,
    )
    with torch.inference_mode():
        expected = dense(x, mask)
        actual = oracle(x, mask)
    accuracy = compare_outputs_streamed(expected, actual)
    del dense, x, mask, expected, actual
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "pass" if accuracy.passed else "fail", **asdict(accuracy)}


def main() -> int:
    args = parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = case14_config(batch_size=args.batch_size, seq_len=args.seq_len)
    oracle = LinearMemoryFP32Reference(config).to(
        device=device, dtype=torch.float32
    ).eval()
    dense_validation = _dense_validation(
        oracle,
        seq_len=min(args.validate_dense_n, args.seq_len),
        seed=args.seed,
        input_scale=args.input_scale,
        device=device,
    )
    if dense_validation["status"] == "fail":
        print(f"dense validation: FAIL {dense_validation}")
        return 2
    print(f"dense validation: {dense_validation}")

    # A dedicated generator makes the streamed data deterministic.  Candidate
    # validation must consume these same sample tensors; this is not claimed to
    # reproduce the exact RNG partitioning of one monolithic [32,N,D] randn.
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 100_000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output_sum = 0.0
    output_square_sum = 0.0
    output_abs_max = 0.0
    finite = True
    started = time.perf_counter()
    with torch.inference_mode():
        for sample_index in range(args.batch_size):
            x = torch.randn(
                1,
                args.seq_len,
                config.d_model,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ).mul_(args.input_scale)
            mask = torch.ones(
                1, args.seq_len, device=device, dtype=torch.bool
            )
            output = oracle(x, mask)
            finite &= bool(torch.isfinite(output).all())
            output_sum += float(output.sum(dtype=torch.float64).item())
            output_square_sum += float(output.square().sum(dtype=torch.float64).item())
            output_abs_max = max(output_abs_max, float(output.abs().max().item()))
            print(f"sample {sample_index + 1}/{args.batch_size}: complete")
            del x, mask, output
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": collect_git(),
        "environment": collect_environment(torch, device),
        "purpose": "validation oracle only; not a benchmark score or candidate",
        "config": asdict(config),
        "dtype": "float32",
        "dense_validation": dense_validation,
        "streaming": {
            "batch_chunk_size": 1,
            "input_rng_note": (
                "deterministic streamed samples; not asserted bitwise-identical "
                "to one monolithic batch randn"
            ),
        },
        "output_fingerprint": {
            "finite": finite,
            "sum": output_sum,
            "square_sum": output_square_sum,
            "abs_max": output_abs_max,
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"result saved to {args.output}")
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
