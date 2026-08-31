#!/usr/bin/env python3
"""Regenerate every figure in docs/technical_report.tex.

Run from the repository root:

    python docs/make_figures.py

Numeric inputs come from the preserved benchmark evidence under
``research/benchmarks/`` wherever a machine-readable record exists, so a figure
cannot drift away from the run it claims to plot. The three inputs that have no
JSON record -- the eager operator attribution, the sigma calibration sweep, and
the polynomial crossover formula -- are transcribed here with the research
document that owns each one named at the point of use.

Palette: the eight-slot categorical set, sequential blue ramp and chrome inks
documented in the ``dataviz`` reference palette. The two-hue subset used here
(blue/orange) was validated all-pairs against a white surface in light mode:
worst CVD dE 24.7, worst normal-vision dE 33.6, both slots >= 3:1 contrast.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "research" / "benchmarks" / "2026-08-31-rtx5080-775c820"
OUT = Path(__file__).resolve().parent / "fig"

# --- palette -----------------------------------------------------------------
SERIES_1 = "#2a78d6"   # blue    -- candidate / emphasis
SERIES_2 = "#eb6834"   # orange  -- baseline / contrast
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

COL_W = 3.45      # IEEE single column, inches
PAGE_W = 7.10     # IEEE full text width, inches


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def load_direct_cases() -> list:
    """Cases 1-13: the directly comparable immutable-baseline runs."""
    rows = []
    for path in sorted(RUN.glob("latest-dispatcher-case*.json")):
        record = json.loads(path.read_text())
        if "official_case_id" not in record:
            continue  # case 14 has no dense baseline and a different schema
        perf = record["performance"]
        memory = perf["memory"]
        rows.append(
            {
                "case": record["official_case_id"],
                "config": record["config"],
                "baseline_ms": perf["baseline"]["median_ms"],
                "candidate_ms": perf["candidate"]["median_ms"],
                "speedup": perf["speedup"],
                "noise": perf["noise_floor_ratio"],
                "significant": perf["significant"],
                "baseline_mib": memory["baseline"]["peak_allocated_bytes"] / 2**20,
                "candidate_mib": memory["candidate"]["peak_allocated_bytes"] / 2**20,
                "max_abs": record["accuracy"]["max_abs_error"],
                "failed": record["accuracy"]["failed_elements"],
            }
        )
    rows.sort(key=lambda row: row["case"])
    return rows


def geomean(values) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


# --- figure 1: per-case speedup ----------------------------------------------
def fig_speedup(rows) -> None:
    fig, ax = plt.subplots(figsize=(COL_W, 3.05))
    cases = [r["case"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    errors = [r["speedup"] * r["noise"] for r in rows]
    positions = list(range(len(rows)))

    gm = geomean(speedups)
    ax.axvline(1.0, color=AXIS, linewidth=0.8, zorder=2)
    ax.axvline(gm, color=SERIES_2, linewidth=1.0, zorder=2)
    ax.text(gm * 1.05, -0.85, "geomean %.3fx" % gm,
            color=SERIES_2, fontsize=6.8, va="center")

    # One series, one hue: bar length already encodes magnitude.
    ax.barh(positions, speedups, height=0.62, color=SERIES_1, zorder=3)
    ax.errorbar(speedups, positions, xerr=errors, fmt="none", ecolor=INK_2,
                elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    for pos, value, err in zip(positions, speedups, errors):
        ax.text(value + err + 0.2, pos, "%.2fx" % value, va="center",
                fontsize=6.6, color=INK_2)

    ax.set_yticks(positions)
    ax.set_yticklabels(["Case %d" % c for c in cases])
    # Headroom above the first bar for the geometric-mean label.
    ax.set_ylim(len(rows) - 0.4, -1.25)
    ax.set_xlabel("Speedup over immutable dense reference")
    ax.set_xlim(0, 13.7)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    fig.savefig(OUT / "speedup-by-case.png")
    plt.close(fig)


# --- figure 2: latency and peak memory ---------------------------------------
def _grouped_log_panel(ax, rows, base_key, cand_key, ylabel) -> None:
    positions = list(range(len(rows)))
    width = 0.38
    gap = 0.03  # the surface gap that separates adjacent fills
    ax.bar([p - width / 2 - gap / 2 for p in positions],
           [r[base_key] for r in rows], width, color=SERIES_2,
           label="Original", zorder=3)
    ax.bar([p + width / 2 + gap / 2 for p in positions],
           [r[cand_key] for r in rows], width, color=SERIES_1,
           label="Dispatcher", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(r["case"]) for r in rows])
    ax.set_xlabel("Official case")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    despine(ax)


def fig_latency_memory(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.4))
    _grouped_log_panel(axes[0], rows, "baseline_ms", "candidate_ms",
                       "Median latency (ms, log scale)")
    _grouped_log_panel(axes[1], rows, "baseline_mib", "candidate_mib",
                       "Peak allocated (MiB, log scale)")
    axes[0].set_title("(a) Steady-state latency", color=INK, pad=5)
    axes[1].set_title("(b) Peak allocated memory", color=INK, pad=5)

    # No per-bar labels: on a log axis the gridlines already carry magnitude,
    # and the exact figures are tabulated beside the figure in the report.
    axes[0].legend(loc="upper left", handlelength=1.1, borderpad=0.1)
    fig.tight_layout(pad=0.4, w_pad=1.8)
    fig.savefig(OUT / "latency-and-memory.png")
    plt.close(fig)


# --- figure 3: eager operator attribution ------------------------------------
def fig_attribution() -> None:
    """Case 13 eager attention, device time by operator.

    Transcribed from research/attention-softmax/measurements.md, profiled on the
    RTX 4060 Laptop development machine. It is an attribution measurement used
    to choose what to optimize, not a latency claim on the RTX 5080 target.
    """
    labels = [
        "aten::copy_ + DtoD\n(head-layout .contiguous())",
        "dtype casts\n(.float() / .to(dtype))",
        "aten::masked_fill_\n(causal + padding)",
        "aten::_softmax",
        "aten::bmm\n(the attention matmul)",
        "aten::mul\n(the * scale pass)",
    ]
    values = [131.1, 53.9, 38.6, 36.9, 18.6, 18.4]
    # Emphasis, not a value ramp: highlight the one operator that is arithmetic.
    colors = [SERIES_1 if label.startswith("aten::bmm") else MUTED
              for label in labels]

    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    positions = list(range(len(labels)))
    ax.barh(positions, values, height=0.6, color=colors, zorder=3)
    for pos, value in zip(positions, values):
        ax.text(value + 2.5, pos, "%.1f ms" % value, va="center",
                fontsize=6.6, color=INK_2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 160)
    ax.set_xlabel("Device time (ms)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    fig.savefig(OUT / "attention-time-attribution.png")
    plt.close(fig)


# --- figure 4: polynomial crossover ------------------------------------------
def fig_crossover(rows) -> None:
    """Where linear attention becomes cheaper than exact causal attention.

    Per token per head, exact causal attention costs about N*d_h MACs and the
    degree-2 polynomial feature map about 2*d_h^3, so the polynomial only wins
    when N > 2*d_h^2. Derivation and the per-case measurements that confirm it:
    research/attention-softmax/long-sequence-attention.md, section 5.5.
    """
    configs = {r["case"]: r["config"] for r in rows}
    configs[14] = {"seq_len": 100000, "d_model": 1024, "num_heads": 16}
    points = {cid: (cfg["seq_len"], cfg["d_model"] // cfg["num_heads"])
              for cid, cfg in configs.items()}

    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    steps = 96
    head_dims = [4.0 * (100.0 / 4.0) ** (i / steps) for i in range(steps + 1)]
    boundary_n = [2 * d * d for d in head_dims]
    ax.plot(boundary_n, head_dims, color=INK_2, linewidth=1.1, zorder=4,
            label=r"crossover $N = 2d_h^2$")
    ax.fill_betweenx(head_dims, boundary_n, 1e7, color=SERIES_1, alpha=0.07,
                     zorder=1, linewidth=0)

    exact = [(n, d) for cid, (n, d) in points.items() if cid != 14]
    ax.scatter([p[0] for p in exact], [p[1] for p in exact], s=26,
               color=SERIES_2, zorder=5, edgecolor=SURFACE, linewidth=0.8,
               label="exact SDPA selected")
    ax.scatter([points[14][0]], [points[14][1]], s=54, color=SERIES_1, zorder=6,
               marker="D", edgecolor=SURFACE, linewidth=0.8,
               label="polynomial selected")

    seen = set()
    for cid, (n, d) in sorted(points.items()):
        if (n, d) in seen:
            continue
        seen.add((n, d))
        ax.annotate(str(cid), (n, d), textcoords="offset points",
                    xytext=(0, 6.5), ha="center", fontsize=6.2,
                    color=SERIES_1 if cid == 14 else INK_2,
                    fontweight="bold" if cid == 14 else "normal")

    ax.text(1.15e5, 6.6, "polynomial\ncheaper", fontsize=6.4, color=SERIES_1,
            ha="center", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(18, 6e5)
    # Headroom above the tallest point (case 8, d_h = 256) for the legend.
    ax.set_ylim(4.6, 1500)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Head dimension $d_h$")
    ax.grid(True, which="major", zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.legend(loc="upper left", handlelength=1.2, borderpad=0.15,
              labelspacing=0.3)
    despine(ax)
    fig.savefig(OUT / "poly-crossover.png")
    plt.close(fig)


# --- figure 5: sigma guard calibration ---------------------------------------
def fig_sigma_guard() -> None:
    """Failed elements against score spread, at N=8192 vs the dense reference.

    Sweep transcribed from the calibration table in
    src/implementations/poly_guard.py, for the shipped kernel. Zero failures
    are drawn at an axis floor because the y axis is logarithmic.
    """
    sigma = [0.3339, 0.3681, 0.4040, 0.4188, 0.4416, 0.4649,
             0.4808, 0.5217, 0.5642, 0.6544, 0.7512]
    failures = [0, 0, 0, 0, 1, 0, 1, 19, 319, 9482, 56758]

    floor = 0.35  # visual stand-in for "zero failures" on a log axis
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    ax.axvspan(0.0, 0.40, color=SERIES_1, alpha=0.07, zorder=1, linewidth=0)
    ax.axvline(0.40, color=SERIES_1, linewidth=1.1, zorder=4)
    ax.axvline(0.3336, color=INK_2, linewidth=0.9, zorder=4)

    ax.plot(sigma, [max(v, floor) for v in failures], color=SERIES_1,
            marker="s", markersize=3.6, zorder=6, markeredgecolor=SURFACE,
            markeredgewidth=0.7)

    ax.set_yscale("log")
    ax.set_ylim(0.22, 2.0e5)
    ax.set_xlim(0.30, 0.79)
    ax.set_xlabel(r"score standard deviation $\sigma$")
    ax.set_ylabel("Failed elements of 8,388,608")
    ax.set_yticks([floor, 1, 1e2, 1e4])
    ax.set_yticklabels(["0", "1", "100", "10,000"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    # Rotated inline labels: the two vertical rules sit 0.066 apart, which is
    # too close for horizontal text at this width.
    ax.text(0.3336, 1.2, "operating point 0.334", rotation=90, fontsize=6.2,
            color=INK_2, va="bottom", ha="right")
    ax.text(0.40, 1.2, "validated ceiling 0.40", rotation=90, fontsize=6.2,
            color=SERIES_1, va="bottom", ha="right")
    despine(ax)
    fig.savefig(OUT / "sigma-guard-calibration.png")
    plt.close(fig)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_direct_cases()
    if len(rows) != 13:
        raise SystemExit("expected 13 directly comparable cases, found %d"
                         % len(rows))
    fig_speedup(rows)
    fig_latency_memory(rows)
    fig_attribution()
    fig_crossover(rows)
    fig_sigma_guard()
    print("geometric-mean speedup over cases 1-13: %.4fx"
          % geomean([r["speedup"] for r in rows]))
    for path in sorted(OUT.glob("*.png")):
        print("wrote %s" % path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
