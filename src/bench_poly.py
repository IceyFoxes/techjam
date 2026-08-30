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
    """Time several callables by alternating them; return every sample, in ms.

    Measuring variants back to back is biased on a laptop GPU: by the third
    variant the card has been at 100% for seconds and has dropped clocks. That
    inflated the polynomial path by ~40% (392 ms measured last, against 275 ms
    measured alone). Interleaving spreads any drift across all variants instead
    of loading it onto whichever runs last. The repository hit the same class of
    bug before -- see the boost-clock fix in
    research/benchmarks/README.md -- so this is a known hazard here.

    Every sample is returned rather than only the minimum, because the minimum
    alone cannot say whether a difference between two variants is real. See
    ``summarise_timings``.
    """
    for fn in variants.values():
        fn()  # warm, and absorb Triton autotuning
    torch.cuda.synchronize()
    samples = {name: [] for name in variants}
    for _ in range(reps):
        for name, fn in variants.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1e3)
    return samples


def summarise_timings(samples, control_pairs=()):
    """Per-variant statistics plus the noise floor that makes A/Bs decidable.

    ``samples`` maps a variant name to its list of millisecond timings.
    ``control_pairs`` names A/A pairs -- two entries running the *same*
    implementation. The gap between such a pair is a direct measurement of what
    this machine reports as a difference when there is none.

    **The floor is the A/A discrepancy between the two minima**, because that is
    exactly the statistic an A/B compares. It is not the within-variant spread.
    Measured here at B=2, those two differ by more than an order of magnitude:
    identical code reproduced to **0.6%** while individual repetitions of it
    varied by **10.3%**. Folding the spread in would have set the floor at 1.103x
    and made every fix worth less than 10% unmeasurable -- and since
    ``(max - min) / min`` only grows as repetitions are added, it would have
    punished measuring more carefully. The spread is reported alongside as
    dispersion, which is what it is.

    The within-variant spread is used as the floor only when there is no control
    pair, where it is a crude lower bound and better than claiming nothing.

    Returned as a ratio: ``minimum_detectable_effect`` of 1.06 means only
    speedups of 1.06x or better (or slowdowns past 1/1.06) are reportable.
    """
    variants = {}
    for name, values in samples.items():
        ordered = sorted(values)
        low = ordered[0]
        mid = len(ordered) // 2
        median = (
            ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        )
        variants[name] = {
            "min_ms": low,
            "median_ms": median,
            "max_ms": ordered[-1],
            "spread": (ordered[-1] - low) / low if low else 0.0,
            "reps": len(ordered),
        }

    discrepancy = None
    paired = None
    for left, right in control_pairs:
        a, b = variants[left]["min_ms"], variants[right]["min_ms"]
        gap = abs(a - b) / min(a, b)
        discrepancy = gap if discrepancy is None else max(discrepancy, gap)
        # The worst disagreement between identical code within a single
        # repetition. Reported as context: it bounds how bad a one-shot
        # comparison would be, which is why the harness does not do one.
        for x, y in zip(samples[left], samples[right]):
            step = abs(x - y) / min(x, y)
            paired = step if paired is None else max(paired, step)

    worst_spread = max((v["spread"] for v in variants.values()), default=0.0)
    floor = discrepancy if discrepancy is not None else worst_spread
    return {
        "variants": variants,
        "noise": {
            "aa_discrepancy": discrepancy,
            "worst_paired_aa_discrepancy": paired,
            "worst_within_variant_spread": worst_spread,
            "minimum_detectable_effect": 1.0 + floor,
        },
    }


def is_resolvable(ratio, minimum_detectable_effect):
    """True when a measured ratio is outside the noise floor, in either direction."""
    if ratio <= 0:
        return False
    return max(ratio, 1.0 / ratio) >= minimum_detectable_effect


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
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="interleaved repetitions after one warm-up; 5 is the Phase 2 minimum",
    )
    parser.add_argument(
        "--no-aa-control",
        dest="aa_control",
        action="store_false",
        help="skip the A/A control variant, leaving the noise floor unmeasured",
    )
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

        def poly_triton():
            return poly_attention_forward(q, k, v, scale, sigma=0.3338, use_triton=True)

        variants = {
            "exact_flash_ms": exact,
            "poly_pytorch_ms": lambda: poly_attention_forward(
                q, k, v, scale, sigma=0.3338, use_triton=False
            ),
            "poly_triton_ms": poly_triton,
        }
        control_pairs = []
        if args.aa_control:
            # The SAME callable, timed as if it were a second variant. Whatever
            # gap appears between the two is this machine reporting a difference
            # where there is none, and it is the bar a real effect must clear.
            variants["poly_triton_control_ms"] = poly_triton
            control_pairs.append(("poly_triton_ms", "poly_triton_control_ms"))

        samples = _time_interleaved(variants, args.reps)
        stats = summarise_timings(samples, control_pairs)
        results = {name: v["min_ms"] for name, v in stats["variants"].items()}
        results["samples_ms"] = samples
        results["noise"] = stats["noise"]
        results["variant_stats"] = stats["variants"]

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

    floor = results["noise"]["minimum_detectable_effect"]
    # A speedup inside the noise floor is not a speedup. Both headline ratios
    # carry that verdict so a later reader cannot quote one without it.
    results["speedup_vs_exact_resolvable"] = is_resolvable(
        results["speedup_vs_exact"], floor
    )
    results["speedup_vs_pytorch_poly_resolvable"] = is_resolvable(
        results["speedup_vs_pytorch_poly"], floor
    )

    print(f"{'variant':28s} {'min':>9s} {'median':>9s} {'max':>9s} {'spread':>8s}")
    for name, stat in results["variant_stats"].items():
        print(
            f"{name:28s} {stat['min_ms']:8.1f}m {stat['median_ms']:8.1f}m "
            f"{stat['max_ms']:8.1f}m {stat['spread']:7.1%}"
        )
    noise = results["noise"]
    aa = noise["aa_discrepancy"]
    paired = noise["worst_paired_aa_discrepancy"]
    print(f"\nA/A discrepancy (the floor):  {'unmeasured' if aa is None else f'{aa:.1%}'}")
    print(
        "  worst single-rep A/A gap:   "
        f"{'unmeasured' if paired is None else f'{paired:.1%}'}"
        "   <- what one unrepeated compare would risk"
    )
    print(
        f"  within-variant spread:      {noise['worst_within_variant_spread']:.1%}"
        "   <- dispersion, not compare error"
    )
    print(f"minimum detectable effect:    {floor:.3f}x")
    print(
        f"\nspeedup vs exact flash:       {results['speedup_vs_exact']:.3f}x "
        f"({'RESOLVABLE' if results['speedup_vs_exact_resolvable'] else 'WITHIN NOISE'})"
    )
    print(
        f"speedup vs pytorch poly:      {results['speedup_vs_pytorch_poly']:.3f}x "
        f"({'RESOLVABLE' if results['speedup_vs_pytorch_poly_resolvable'] else 'WITHIN NOISE'})"
    )
    print(f"peak MiB:                     {results['peak_mib']}")
    print(f"accepted (>= {ACCEPTANCE_SPEEDUP}x):           {results['accepted']}")

    if args.output is not None:
        payload = {
            "schema_version": 2,
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
