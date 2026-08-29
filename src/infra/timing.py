"""Drift-resistant paired timing for A/B benchmark comparisons.

The measured GPU (laptop RTX 4060) sustains roughly 2520 MHz at 60 W for about
two seconds of continuous load, then settles near 1890 MHz at 33 W under
``SwPowerCap``. Latency changes by about 2x across that transition, so any
schedule that measures one implementation earlier than the other reports a
fabricated speedup. Timing both sides in an order that balances their mean
position cancels that drift instead of trying to outrun it.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence

BASELINE = "baseline"
CANDIDATE = "candidate"

Runner = Callable[[], float]


def paired_schedule(repeats: int) -> List[str]:
    """Order ``repeats`` samples of each side so neither is measured earlier.

    Emits ``AB`` and ``BA`` pairs alternately, which gives both labels the same
    mean position and therefore cancels monotonic drift.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    schedule: List[str] = []
    for index in range(repeats):
        pair = (BASELINE, CANDIDATE) if index % 2 == 0 else (CANDIDATE, BASELINE)
        schedule.extend(pair)
    return schedule


def _resolve(sample: Any) -> float:
    """Read a sample, whether it is already a float or a pending device timer."""
    resolver = getattr(sample, "milliseconds", None)
    return resolver() if callable(resolver) else float(sample)


def measure_pair(
    runners: Mapping[str, Runner],
    repeats: int,
    device: Any = None,
) -> Dict[str, List[float]]:
    """Run the paired schedule, collecting per-sample milliseconds per side.

    Every sample is recorded before any is resolved. A device sync between
    samples would let the GPU idle back up to its boost clocks, widening the
    spread and hiding exactly the improvements this harness exists to detect.
    """
    missing = {BASELINE, CANDIDATE} - set(runners)
    if missing:
        raise KeyError(f"runners must provide {sorted(missing)}")

    pending: Dict[str, List[Any]] = {BASELINE: [], CANDIDATE: []}
    for label in paired_schedule(repeats):
        pending[label].append(runners[label]())

    if _is_cuda(device):
        import torch

        torch.cuda.synchronize(device)

    return {label: [_resolve(s) for s in values] for label, values in pending.items()}


def paired_summary(samples: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    """Reduce paired samples to medians and the resulting speedup."""
    baseline_median = statistics.median(samples[BASELINE])
    candidate_median = statistics.median(samples[CANDIDATE])
    return {
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "baseline_min_ms": min(samples[BASELINE]),
        "candidate_min_ms": min(samples[CANDIDATE]),
        "speedup": baseline_median / candidate_median,
    }


def paired_ratios(samples: Mapping[str, Sequence[float]]) -> List[float]:
    """Per-pair baseline/candidate ratios.

    Neighbouring samples are measured microseconds apart, so a ratio taken
    within a pair sees the same clock state on both sides and carries no drift.
    """
    baseline = samples[BASELINE]
    candidate = samples[CANDIDATE]
    return [baseline[i] / candidate[i] for i in range(min(len(baseline), len(candidate)))]


def noise_floor_ratio(samples: Mapping[str, Sequence[float]]) -> float:
    """Half-width of the 95% interval around the measured speedup.

    Derived from disagreement *between* paired samples rather than raw spread,
    because shared boost-clock decay moves both sides together and is already
    cancelled by the schedule. A speedup within this band is indistinguishable
    from measurement noise.
    """
    ratios = paired_ratios(samples)
    if len(ratios) < 2:
        return float("inf")

    # 1.2533 converts the standard error of the mean to that of the median.
    standard_error = 1.2533 * statistics.stdev(ratios) / math.sqrt(len(ratios))
    return 1.96 * standard_error


def is_significant(speedup: float, noise_floor: float) -> bool:
    """True when the measured change exceeds the run's own noise floor."""
    return abs(speedup - 1.0) > noise_floor


def choose_block_size(
    per_iteration_ms: float,
    target_ms: float = 50.0,
    maximum: int = 200,
) -> int:
    """Pick how many forward passes one timed sample should cover.

    Launch-bound shapes carry per-call CPU scheduling jitter that no schedule can
    cancel, so their samples are batched until each covers ``target_ms``.
    Compute-bound shapes already exceed the target in a single pass and stay at 1.
    """
    if per_iteration_ms <= 0:
        raise ValueError(f"per_iteration_ms must be positive, got {per_iteration_ms}")

    blocks = max(1, round(target_ms / per_iteration_ms))
    return min(blocks, maximum)


def _is_cuda(device: Any) -> bool:
    return device is not None and getattr(device, "type", None) == "cuda"


class _PendingCudaSample:
    """A recorded CUDA event pair, resolved after the final synchronise."""

    __slots__ = ("start", "end", "block_size")

    def __init__(self, start: Any, end: Any, block_size: int) -> None:
        self.start = start
        self.end = end
        self.block_size = block_size

    def milliseconds(self) -> float:
        return self.start.elapsed_time(self.end) / self.block_size


def _inference_mode():
    import torch

    return torch.inference_mode()


def make_block_runner(
    model: Any,
    x: Any,
    valid_mask: Any,
    device: Any,
    block_size: int = 1,
) -> Runner:
    """Build a timer for ``block_size`` forward passes, reported per iteration.

    Launch-bound shapes are dominated by per-call CPU scheduling jitter, which a
    block larger than one averages away. Compute-bound shapes use ``block_size=1``
    so individual samples stay meaningful.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if _is_cuda(device):
        import torch

        def cuda_runner() -> _PendingCudaSample:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with _inference_mode():
                start.record()
                for _ in range(block_size):
                    model(x, valid_mask)
                end.record()
            return _PendingCudaSample(start, end, block_size)

        return cuda_runner

    def host_runner() -> float:
        started = time.perf_counter_ns()
        for _ in range(block_size):
            model(x, valid_mask)
        return (time.perf_counter_ns() - started) / 1e6 / block_size

    return host_runner


def settle(
    model: Any,
    x: Any,
    valid_mask: Any,
    device: Any,
    seconds: float,
    chunk: int = 4,
) -> int:
    """Hold the device under continuous load until its clocks stop drifting.

    Iteration-count warmup is not enough: the boost budget is spent in wall-clock
    time, so a fast shape can finish its warmup entirely inside the boost window
    and then be timed while the clocks are still falling. Returns the number of
    forward passes executed.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")

    calls = 0
    deadline = time.perf_counter() + seconds
    while True:
        for _ in range(chunk):
            model(x, valid_mask)
            calls += 1
        if _is_cuda(device):
            import torch

            torch.cuda.synchronize(device)
        if time.perf_counter() >= deadline:
            return calls
