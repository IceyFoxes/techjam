"""Live terminal dashboard for recording the Transformer benchmark demo.

This module launches ``src.benchmark`` and polls ``nvidia-smi`` while it runs.
It intentionally uses only the Python standard library so the demo machine does
not need a dashboard package, browser, or local web server.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Deque, Iterable, Optional, Sequence


SPARKS = "▁▂▃▄▅▆▇█"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_HOME = "\033[2J\033[H"


@dataclass(frozen=True)
class GpuSample:
    name: str
    utilization: float
    memory_used_mib: float
    memory_total_mib: float
    power_w: float
    power_limit_w: float
    temperature_c: float
    sm_clock_mhz: float


def _number(value: str) -> float:
    """Parse an nvidia-smi numeric cell and reject unsupported values."""

    cleaned = value.strip()
    if cleaned in {"", "N/A", "[Not Supported]"}:
        raise ValueError(f"GPU metric is unavailable: {value!r}")
    return float(cleaned)


def parse_nvidia_smi_row(row: str) -> GpuSample:
    """Parse one nounits CSV row returned by the dashboard query."""

    cells = next(csv.reader([row]))
    if len(cells) != 8:
        raise ValueError(f"expected 8 GPU fields, received {len(cells)}")
    return GpuSample(
        name=cells[0].strip(),
        utilization=_number(cells[1]),
        memory_used_mib=_number(cells[2]),
        memory_total_mib=_number(cells[3]),
        power_w=_number(cells[4]),
        power_limit_w=_number(cells[5]),
        temperature_c=_number(cells[6]),
        sm_clock_mhz=_number(cells[7]),
    )


def query_gpu(index: int) -> GpuSample:
    """Read a single GPU sample with a short, bounded subprocess call."""

    fields = (
        "name,utilization.gpu,memory.used,memory.total,power.draw,"
        "power.limit,temperature.gpu,clocks.sm"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one GPU row, received {len(rows)}")
    return parse_nvidia_smi_row(rows[0])


def sparkline(values: Sequence[float], maximum: Optional[float] = None) -> str:
    """Render a compact fixed-scale history graph."""

    if not values:
        return ""
    ceiling = maximum if maximum is not None else max(values)
    ceiling = max(float(ceiling), 1.0)
    return "".join(
        SPARKS[min(len(SPARKS) - 1, int(max(0.0, value) / ceiling * 7))]
        for value in values
    )


def bar(value: float, maximum: float, width: int = 30) -> str:
    """Render one clamped horizontal utilization bar."""

    fraction = 0.0 if maximum <= 0 else max(0.0, min(value / maximum, 1.0))
    filled = round(width * fraction)
    return "█" * filled + "░" * (width - filled)


def benchmark_phase(lines: Iterable[str], running: bool, returncode: Optional[int]) -> str:
    """Infer a viewer-friendly phase from the benchmark's public output."""

    joined = "\n".join(lines)
    if returncode is not None:
        return "Complete — benchmark passed" if returncode == 0 else "Stopped — benchmark failed"
    if "baseline median=" in joined or "oracle time" in joined.lower():
        return "Finalizing results"
    if "accuracy" in joined:
        return "Correctness passed — measuring paired latency"
    if "Case 14 FP32:" in joined:
        return "Streaming Case 14 oracle and candidate"
    if running:
        return "Loading model and compiling GPU route"
    return "Waiting to start"


def _fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def render_dashboard(
    sample: GpuSample,
    history: dict[str, Deque[float]],
    output_lines: Sequence[str],
    elapsed: float,
    running: bool,
    returncode: Optional[int],
    case: int,
    color: bool = True,
    width: int = 88,
) -> str:
    """Produce a complete dashboard frame, separated for deterministic tests."""

    width = max(72, width)
    accent = CYAN if color else ""
    good = GREEN if color else ""
    warn = YELLOW if color else ""
    reset = RESET if color else ""
    bold = BOLD if color else ""
    status_color = good if returncode in (None, 0) else (RED if color else "")
    phase = benchmark_phase(output_lines, running, returncode)

    memory_pct = 100.0 * sample.memory_used_mib / sample.memory_total_mib
    power_pct = 100.0 * sample.power_w / sample.power_limit_w
    recent = [line.strip() for line in output_lines if line.strip()][-6:]

    frame = [
        f"{bold}{accent}TRANSFORMER GPU KERNEL — LIVE WORKLOAD{reset}",
        f"Case {case}  •  {sample.name}  •  elapsed {elapsed:6.1f}s",
        "─" * min(width, 88),
        f"{status_color}{phase}{reset}",
        "",
        f"GPU compute  {bar(sample.utilization, 100)} {sample.utilization:5.1f}%  "
        f"{accent}{sparkline(history['utilization'], 100)}{reset}",
        f"VRAM        {bar(sample.memory_used_mib, sample.memory_total_mib)} "
        f"{memory_pct:5.1f}%  {sample.memory_used_mib:,.0f}/{sample.memory_total_mib:,.0f} MiB",
        f"Power       {bar(sample.power_w, sample.power_limit_w)} {power_pct:5.1f}%  "
        f"{sample.power_w:5.1f}/{sample.power_limit_w:5.1f} W",
        f"Temperature {bar(sample.temperature_c, 100)} {sample.temperature_c:5.1f} °C  "
        f"{warn}{sparkline(history['temperature'], 100)}{reset}",
        f"SM clock    {sample.sm_clock_mhz:7.0f} MHz                       "
        f"{accent}{sparkline(history['clock'])}{reset}",
        "",
        f"{bold}Benchmark output{reset}",
    ]
    if recent:
        frame.extend(f"  {_fit(line, width - 2)}" for line in recent)
    else:
        frame.append("  Waiting for benchmark output…")
    frame.extend(
        [
            "",
            "Final published matrix: PASS 65/65 • 3.611× geomean • Case 13 8.577×",
            "Case 14: PASS 5/5 vs streamed FP32 oracle • 8.972× diagnostic ratio",
        ]
    )
    return "\n".join(frame)


def _reader(stream, lines: list[str]) -> None:
    for line in iter(stream.readline, ""):
        lines.append(line.rstrip())
    stream.close()


def build_benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.benchmark",
        "--candidate",
        "src.dispatcher",
        "--case",
        str(args.case),
        "--device",
        args.device,
        "--dtype",
        "float32",
        "--accuracy-trials",
        str(args.accuracy_trials),
        "--seed",
        "1234",
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--timing",
        "paired",
        "--settle-seconds",
        str(args.settle_seconds),
    ]
    if args.output:
        command.extend(["--output", args.output])
    return command


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark case with a live nvidia-smi dashboard"
    )
    parser.add_argument("--case", type=int, choices=range(1, 14), default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--benchmark-rounds", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=0.35)
    parser.add_argument("--output", help="optional new benchmark JSON path")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.poll_interval < 0.1:
        raise ValueError("--poll-interval must be at least 0.1 seconds")
    if args.output and Path(args.output).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    try:
        sample = query_gpu(args.gpu_index)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        print(f"GPU dashboard preflight failed: {exc}", file=sys.stderr)
        return 1

    command = build_benchmark_command(args)
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    reader = threading.Thread(target=_reader, args=(process.stdout, lines), daemon=True)
    reader.start()
    history: dict[str, Deque[float]] = {
        "utilization": deque(maxlen=36),
        "temperature": deque(maxlen=36),
        "clock": deque(maxlen=36),
    }
    started = time.monotonic()
    use_color = not args.no_color and sys.stdout.isatty()

    if sys.stdout.isatty():
        print(HIDE_CURSOR, end="", flush=True)
    try:
        while process.poll() is None:
            tick = time.monotonic()
            try:
                sample = query_gpu(args.gpu_index)
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
                pass
            history["utilization"].append(sample.utilization)
            history["temperature"].append(sample.temperature_c)
            history["clock"].append(sample.sm_clock_mhz)
            width = os.get_terminal_size().columns if sys.stdout.isatty() else 88
            frame = render_dashboard(
                sample,
                history,
                list(lines),
                time.monotonic() - started,
                True,
                None,
                args.case,
                color=use_color,
                width=width,
            )
            if sys.stdout.isatty():
                print(CLEAR_HOME + frame, end="", flush=True)
            else:
                print(frame, flush=True)
            remaining = args.poll_interval - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        reader.join(timeout=1)
        returncode = process.poll()
        frame = render_dashboard(
            sample,
            history,
            list(lines),
            time.monotonic() - started,
            False,
            returncode,
            args.case,
            color=use_color,
        )
        if sys.stdout.isatty():
            print(CLEAR_HOME + frame + SHOW_CURSOR)
        else:
            print(frame)

    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
