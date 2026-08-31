"""Export one real baseline/candidate Perfetto trace for every official case.

Cases 1--13 use the immutable dense reference and the final dispatcher. Case 14
uses the validated streamed FP32 oracle and the dispatcher's FP32-facing sample
entrypoint because the dense reference is infeasible. The exported files use a
plain top-level Chrome-event array with timestamps rebased to zero, which is the
most portable legacy JSON form accepted by Perfetto UI.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Iterator

import torch

import torch_transformer_benchmark as reference
from src import profile as profile_runner
from src.dispatcher import DispatchingTransformer
from src.implementations.fp32_reference import LinearMemoryFP32Reference, case14_config
from src.infra import load_candidate, load_official_cases
from src.infra.environment import collect_environment, collect_git


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export final reference/candidate Perfetto traces for Cases 1--14"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=1)
    return parser.parse_args()


def _portable_trace(source: Path, destination: Path) -> dict[str, int | float]:
    """Convert a PyTorch Chrome trace to a timestamp-rebased event array."""

    document = json.loads(source.read_text(encoding="utf-8"))
    events = document.get("traceEvents")
    if not isinstance(events, list) or not events:
        raise ValueError(f"{source} contains no traceEvents")
    timestamps = [
        float(event["ts"])
        for event in events
        if isinstance(event, dict) and isinstance(event.get("ts"), (int, float))
    ]
    if not timestamps:
        raise ValueError(f"{source} contains no timestamped events")
    origin = min(timestamps)
    normalized = []
    for original in events:
        event = dict(original)
        if isinstance(event.get("ts"), (int, float)):
            event["ts"] = float(event["ts"]) - origin
        normalized.append(event)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(normalized, output, separators=(",", ":"))
        output.write("\n")
    return {
        "events": len(normalized),
        "kernel_events": sum(event.get("cat") == "kernel" for event in normalized),
        "gpu_memcpy_events": sum(
            event.get("cat") == "gpu_memcpy" for event in normalized
        ),
        "duration_us": max(timestamps) - origin,
    }


def _profile_args(device: str, warmup: int, active: int) -> argparse.Namespace:
    return argparse.Namespace(
        warmup=warmup,
        active=active,
        with_stack=False,
        with_flops=False,
    )


@contextmanager
def _temporary_profile_dir(case_id: int) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=f"techjam-case{case_id}-profile-") as raw:
        yield Path(raw)


def _official_config(case_id: int) -> reference.TransformerConfig:
    shape = load_official_cases()[case_id].benchmark_config()
    return reference.TransformerConfig(
        batch_size=shape["batch_size"],
        seq_len=shape["seq_len"],
        d_model=shape["d_model"],
        num_heads=shape["heads"],
        ffn_dim=shape["ffn_dim"],
        num_layers=shape["layers"],
        causal=shape["causal"],
    )


def _trace_direct_case(
    case_id: int,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    spec = load_candidate("src.dispatcher")
    config = _official_config(case_id)
    build_args = argparse.Namespace(
        non_strict_weight_copy=False,
        compile_baseline=False,
        compile_user=False,
        compile_mode="default",
    )
    baseline, candidate, _ = profile_runner._build_models(
        spec, config, device, torch.float32, build_args
    )
    x, mask = reference.generate_random_case(
        config,
        device,
        torch.float32,
        1234,
        padding_ratio=0.0,
        input_scale=1.0,
    )
    records = {}
    with _temporary_profile_dir(case_id) as raw:
        for arm, model in (("reference", baseline), ("candidate", candidate)):
            print(f"Case {case_id}: profiling {arm}", flush=True)
            artifacts = profile_runner._profile_one(
                arm, model, x, mask, device, raw, args
            )
            name = f"case{case_id}_{arm}.perfetto.json"
            records[arm] = {
                "file": name,
                **_portable_trace(raw / artifacts["trace"], output_dir / name),
            }
    del x, mask, baseline, candidate
    torch.cuda.empty_cache()
    return {"config": asdict(config), "arms": records}


class _Case14Candidate(torch.nn.Module):
    def __init__(self, model: DispatchingTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.forward_case14_streamed_sample(x, mask)


def _trace_case14(
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    config = case14_config()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    oracle = LinearMemoryFP32Reference(config)
    candidate = DispatchingTransformer(config)
    reference.copy_model_weights(oracle, candidate)
    oracle = oracle.to(device=device, dtype=torch.float32).eval()
    candidate = candidate.to(device=device, dtype=torch.float32).eval()
    candidate_wrapper = _Case14Candidate(candidate).eval()

    generator = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(
        1,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    mask = torch.ones(1, config.seq_len, device=device, dtype=torch.bool)
    records = {}
    with _temporary_profile_dir(14) as raw:
        for arm, profile_label, model in (
            ("reference", "oracle", oracle),
            ("candidate", "candidate", candidate_wrapper),
        ):
            print(f"Case 14: profiling streamed {profile_label} sample 1/32", flush=True)
            artifacts = profile_runner._profile_one(
                profile_label, model, x, mask, device, raw, args
            )
            name = f"case14_{arm}.perfetto.json"
            records[arm] = {
                "file": name,
                "implementation": (
                    "streamed_fp32_oracle" if arm == "reference" else "dispatcher"
                ),
                **_portable_trace(raw / artifacts["trace"], output_dir / name),
            }
    del x, mask, oracle, candidate, candidate_wrapper
    torch.cuda.empty_cache()
    return {
        "config": asdict(config),
        "streaming_note": (
            "Trace covers one full-length sample of the official B=32 streamed "
            "execution; the validated harness repeats this independent path 32 times."
        ),
        "arms": records,
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.warmup < 0 or args.active <= 0:
        raise ValueError("warmup must be non-negative and active must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the final dispatcher trace export requires CUDA")

    args.output_dir.mkdir(parents=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    runner_args = _profile_args(args.device, args.warmup, args.active)
    manifest = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": collect_git(),
        "environment": collect_environment(torch, device),
        "capture": {
            "dtype": "float32",
            "warmup_iterations": args.warmup,
            "profiled_iterations": args.active,
            "record_shapes": True,
            "profile_memory": True,
            "timing_claim": "none; profiler-instrumented runs are visualization only",
        },
        "cases": {},
    }
    try:
        for case_id in range(1, 14):
            manifest["cases"][str(case_id)] = _trace_direct_case(
                case_id, args.output_dir, device, runner_args
            )
        manifest["cases"]["14"] = _trace_case14(
            args.output_dir, device, runner_args
        )
    finally:
        manifest_path = args.output_dir / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
    print(f"Final Perfetto traces saved under {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
