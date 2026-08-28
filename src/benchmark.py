#!/usr/bin/env python3
"""Benchmark any independently loadable candidate against the immutable reference."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import torch_transformer_benchmark as reference
from src.infra import CandidateSpec, load_candidate, load_official_cases
from src.infra.environment import collect_environment, collect_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a loadable candidate with torch_transformer_benchmark.py"
    )
    parser.add_argument(
        "--candidate",
        default="reference",
        help="short module name or module[:CandidateSpec attribute]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON result path; an existing file is never overwritten",
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=range(1, 15),
        help="load disclosed test shape 1-14 from src/cases/task_shapes.json",
    )

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--ffn-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="annotate timed rounds for Nsight Systems; adds profiling overhead",
    )
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


def _resolve_shape(args: argparse.Namespace) -> Optional[int]:
    defaults: Dict[str, object] = {
        "batch_size": 8,
        "seq_len": 128,
        "d_model": 512,
        "heads": 8,
        "ffn_dim": 2048,
        "layers": 6,
        "causal": False,
    }
    official_case_id: Optional[int] = args.case
    if official_case_id is not None:
        official = load_official_cases()[official_case_id].benchmark_config()
        for field, expected in official.items():
            supplied = getattr(args, field)
            if supplied is not None and supplied != expected:
                cli_field = field.replace("_", "-")
                raise ValueError(
                    f"--case {official_case_id} fixes --{cli_field}={expected}; "
                    f"received conflicting value {supplied}"
                )
            setattr(args, field, expected)
    else:
        for field, default in defaults.items():
            if getattr(args, field) is None:
                setattr(args, field, default)
    return official_case_id


def _accuracy_trials(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    config: reference.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for trial_index in range(args.accuracy_trials):
            x, valid_mask = reference.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=args.seed + trial_index,
                padding_ratio=args.padding_ratio,
                input_scale=args.input_scale,
            )
            expected = baseline(x, valid_mask)
            actual = candidate(x, valid_mask)
            result = reference.compare_outputs(
                expected,
                actual,
                rtol=args.rtol,
                atol=args.atol,
            )
            trial = asdict(result)
            trial["trial"] = trial_index + 1
            trial["seed"] = args.seed + trial_index
            trials.append(trial)
            print(
                f"accuracy {trial_index + 1:02d}/{args.accuracy_trials}: "
                f"{'PASS' if result.passed else 'FAIL'} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

    return {
        "passed": all(trial["passed"] for trial in trials),
        "rtol": args.rtol,
        "atol": args.atol,
        "failed_elements": sum(trial["failed_elements"] for trial in trials),
        "total_elements": sum(trial["total_elements"] for trial in trials),
        "max_abs_error": max(trial["max_abs_error"] for trial in trials),
        "max_relative_error": max(
            trial["max_relative_error"] for trial in trials
        ),
        "trials": trials,
    }


def _timing_summary(samples: List[float]) -> Dict[str, Any]:
    timing = reference.TimingResult(samples)
    return {
        "samples_ms": samples,
        "median_ms": timing.median_ms,
        "mean_ms": timing.mean_ms,
        "p90_ms": timing.p90_ms,
        "min_ms": timing.min_ms,
    }


def _nvtx_range(device: torch.device, enabled: bool, label: str) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.cuda.nvtx.range(label)


def _benchmark(
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    config: reference.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    x, valid_mask = reference.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed + 100000,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    reference.warmup_model(baseline, x, valid_mask, args.warmup, device)
    reference.warmup_model(candidate, x, valid_mask, args.warmup, device)

    baseline_samples: List[float] = []
    candidate_samples: List[float] = []
    for round_index in range(args.benchmark_rounds):
        first, second = (
            (
                ("baseline", baseline, baseline_samples),
                ("candidate", candidate, candidate_samples),
            )
            if round_index % 2 == 0
            else (
                ("candidate", candidate, candidate_samples),
                ("baseline", baseline, baseline_samples),
            )
        )
        for label, model, samples in (first, second):
            range_name = f"{label}_timed_round_{round_index + 1}"
            with _nvtx_range(device, args.nvtx, range_name):
                samples.extend(
                    reference.benchmark_once(
                        model,
                        x,
                        valid_mask,
                        args.repeats,
                        device,
                    )
                )

    baseline_summary = _timing_summary(baseline_samples)
    candidate_summary = _timing_summary(candidate_samples)
    speedup = baseline_summary["median_ms"] / candidate_summary["median_ms"]
    tokens = config.batch_size * config.seq_len
    baseline_summary["tokens_per_second"] = (
        tokens * 1000.0 / baseline_summary["median_ms"]
    )
    candidate_summary["tokens_per_second"] = (
        tokens * 1000.0 / candidate_summary["median_ms"]
    )
    print(
        f"baseline median={baseline_summary['median_ms']:.4f} ms | "
        f"candidate median={candidate_summary['median_ms']:.4f} ms | "
        f"speedup={speedup:.3f}x"
    )
    return {
        "warmup_iterations": args.warmup,
        "repeats": args.repeats,
        "rounds": args.benchmark_rounds,
        "timing_method": (
            "torch.cuda.Event" if device.type == "cuda" else "perf_counter_ns"
        ),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "speedup": speedup,
    }


def _configure(args: argparse.Namespace) -> Tuple[torch.device, torch.dtype]:
    device = reference.resolve_device(args.device)
    dtype = reference.resolve_dtype(args.dtype)
    reference.validate_args(args, device, dtype)
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


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"result saved to {path}")


def main() -> int:
    args = parse_args()
    official_case_id = _resolve_shape(args)
    spec = load_candidate(args.candidate)
    device, dtype = _configure(args)
    config = reference.TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()
    baseline, candidate, strict = _build_models(
        spec, config, device, dtype, args
    )

    print(
        f"candidate={spec.name} device={device} dtype={dtype} "
        f"config={config}"
    )
    accuracy = _accuracy_trials(
        baseline, candidate, config, device, dtype, args
    )
    performance: Optional[Dict[str, Any]] = None
    if accuracy["passed"] or args.benchmark_on_failure:
        performance = _benchmark(
            baseline, candidate, config, device, dtype, args
        )
    else:
        print("benchmark skipped because correctness failed")

    result: Dict[str, Any] = {
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
        "config": asdict(config),
        "official_case_id": official_case_id,
        "official_case_source": "task_shapes.png" if official_case_id else None,
        "dtype": args.dtype,
        "device": str(device),
        "padding_ratio": args.padding_ratio,
        "input_scale": args.input_scale,
        "seed": args.seed,
        "nvtx_enabled": args.nvtx,
        "accuracy": accuracy,
        "performance": performance,
    }
    if args.output is not None:
        _write_result(args.output, result)
    return 0 if accuracy["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
