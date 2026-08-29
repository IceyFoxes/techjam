#!/usr/bin/env python3
"""Create Perfetto-compatible PyTorch traces for baseline and candidate models."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

import torch_transformer_benchmark as reference
from src.infra import CandidateSpec, load_candidate, load_official_cases
from src.infra.environment import collect_environment, collect_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch CPU/GPU traces for an official task shape"
    )
    parser.add_argument("--candidate", default="dummy")
    parser.add_argument("--case", type=int, choices=range(1, 15), default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--models",
        choices=("baseline", "candidate", "both"),
        default="both",
        help="export separate traces for the selected model or for both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "new directory for trace JSON, tables, and metadata; defaults to "
            "artifacts/profiles/<timestamp>-<candidate>-case<N>"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--with-flops", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _configure(args: argparse.Namespace) -> Tuple[torch.device, torch.dtype]:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.warmup < 0 or args.active <= 0:
        raise ValueError("warmup must be non-negative and active must be positive")

    device = reference.resolve_device(args.device)
    dtype = reference.resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
    return device, dtype


def _build_models(
    spec: CandidateSpec,
    config: reference.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, torch.nn.Module, bool]:
    baseline = reference.BaselineTransformer(config)
    candidate = spec.model_factory(config)
    if not isinstance(candidate, torch.nn.Module):
        raise TypeError(
            f"candidate factory returned {type(candidate).__name__}, expected nn.Module"
        )
    strict = spec.strict_weight_copy and not args.non_strict_weight_copy
    weight_loader = spec.weight_loader or reference.copy_model_weights
    weight_loader(baseline, candidate, strict)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()
    baseline = reference.maybe_compile(
        baseline, args.compile_baseline, args.compile_mode
    )
    candidate = reference.maybe_compile(
        candidate, args.compile_user, args.compile_mode
    )
    return baseline, candidate, strict


def _activities(device: torch.device) -> list[Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        cuda_activity = torch.profiler.ProfilerActivity.CUDA
        if cuda_activity not in torch.profiler.supported_activities():
            raise RuntimeError("this PyTorch build cannot collect CUDA profiler traces")
        activities.append(cuda_activity)
    return activities


def _nvtx_range(device: torch.device, label: str) -> Any:
    if device.type != "cuda":
        return nullcontext()
    return torch.cuda.nvtx.range(label)


def _profile_one(
    label: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.inference_mode(), torch.profiler.profile(
        activities=_activities(device),
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as profiler:
        for iteration in range(args.active):
            range_name = f"{label}_forward_{iteration + 1}"
            with torch.profiler.record_function(range_name), _nvtx_range(
                device, range_name
            ):
                model(x, valid_mask)
            profiler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    trace_path = output_dir / f"{label}-trace.json"
    table_path = output_dir / f"{label}-operators.txt"
    profiler.export_chrome_trace(str(trace_path))
    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    table = profiler.key_averages(group_by_input_shape=True).table(
        sort_by=sort_key,
        row_limit=50,
    )
    with table_path.open("x", encoding="utf-8") as output:
        output.write(table)
        output.write("\n")

    stack_path = None
    if args.with_stack:
        stack_path = output_dir / f"{label}-stacks.txt"
        profiler.export_stacks(str(stack_path), metric=sort_key)
    return {
        "trace": trace_path.name,
        "operator_table": table_path.name,
        "stacks": stack_path.name if stack_path else None,
    }


def _selected_models(
    selection: str,
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
) -> Iterable[Tuple[str, torch.nn.Module]]:
    if selection in ("baseline", "both"):
        yield "baseline", baseline
    if selection in ("candidate", "both"):
        yield "candidate", candidate


def main() -> int:
    args = parse_args()
    spec = load_candidate(args.candidate)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            Path("artifacts")
            / "profiles"
            / f"{timestamp}-{spec.name}-case{args.case}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device, dtype = _configure(args)
    official_case = load_official_cases()[args.case]
    shape = official_case.benchmark_config()
    config = reference.TransformerConfig(
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["layers"],
        causal=shape["causal"],
    )
    config.validate()
    baseline, candidate, strict = _build_models(
        spec, config, device, dtype, args
    )
    x, valid_mask = reference.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )

    artifacts: Dict[str, Any] = {}
    for label, model in _selected_models(args.models, baseline, candidate):
        print(f"profiling {label} for official case {args.case}")
        artifacts[label] = _profile_one(
            label, model, x, valid_mask, device, args.output_dir, args
        )

    metadata = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": collect_git(),
        "environment": collect_environment(torch, device),
        "candidate": {
            "selector": args.candidate,
            "name": spec.name,
            "owner": spec.owner,
            "description": spec.description,
            "strict_weight_copy": strict,
        },
        "official_case_id": args.case,
        "config": asdict(config),
        "dtype": args.dtype,
        "device": str(device),
        "warmup_iterations": args.warmup,
        "profiled_iterations": args.active,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": args.with_stack,
        "with_flops": args.with_flops,
        "artifacts": artifacts,
    }
    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("x", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"traces saved under {args.output_dir}")
    print("open *-trace.json in https://ui.perfetto.dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
