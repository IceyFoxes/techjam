"""Generate five regime-specific Perfetto Data Explorer dashboards."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "regime_dashboards"

SOURCES = {
    "kernel": "6",
    "memcpy": "7",
    "cuda": "8",
    "cpu": "9",
    "forward": "10",
}


def _graph(title: str, story: str) -> str:
    categories = (
        ("1", "kernel", "6", "GPU kernel slices"),
        ("2", "gpu_memcpy", "7", "GPU memcpy slices"),
        ("3", "cuda_runtime", "8", "CUDA runtime slices"),
        ("4", "cpu_op", "9", "CPU operator slices"),
        ("5", "user_annotation", "10", "Forward annotations"),
    )
    nodes = [
        {
            "nodeId": "0",
            "type": "table",
            "state": {"sqlTable": "slice"},
            "nextNodes": [item[0] for item in categories],
        }
    ]
    for filter_id, category, dashboard_id, export_name in categories:
        nodes.extend(
            [
                {
                    "nodeId": filter_id,
                    "type": "filter",
                    "state": {
                        "filterMode": "structured",
                        "filters": [
                            {"column": "category", "op": "=", "value": category}
                        ],
                        "filterOperator": "AND",
                    },
                    "nextNodes": [dashboard_id],
                    "primaryInputId": "0",
                },
                {
                    "nodeId": dashboard_id,
                    "type": "dashboard",
                    # Perfetto v57.2 DashboardNodeAttrs uses `name`.  The old
                    # `exportName` key deserializes but never publishes a live
                    # source, leaving imported charts orphaned.
                    "state": {"name": export_name},
                    "nextNodes": [],
                    "primaryInputId": filter_id,
                },
            ]
        )
    graph = {
        "nodes": nodes,
        "rootNodeIds": ["0"],
        # Select an export leaf so Data Explorer eagerly initializes the
        # imported graph's dashboard sources.
        "selectedNodeId": "6",
        "nodeLayouts": {
            "0": {"x": 0, "y": 380},
            "1": {"x": 300, "y": 0},
            "2": {"x": 300, "y": 190},
            "3": {"x": 300, "y": 380},
            "4": {"x": 300, "y": 570},
            "5": {"x": 300, "y": 760},
            "6": {"x": 620, "y": 0},
            "7": {"x": 620, "y": 190},
            "8": {"x": 620, "y": 380},
            "9": {"x": 620, "y": 570},
            "10": {"x": 620, "y": 760},
        },
        "labels": [
            {
                "id": "regime-story",
                "x": 20,
                "y": 930,
                "width": 900,
                "text": f"{title}: {story}",
            }
        ],
        "isExplorerCollapsed": True,
        "sidebarWidth": 360,
    }
    return json.dumps(graph, indent=2)


def label(identifier: str, text: str, row: int, row_span: int = 2) -> dict:
    return {
        "kind": "label",
        "id": identifier,
        "text": text,
        "col": 0,
        "row": row,
        "colSpan": 24,
        "rowSpan": row_span,
    }


def divider(identifier: str, text: str, row: int) -> dict:
    return {"kind": "divider", "id": identifier, "row": row, "label": text}


def scorecard(
    source: str,
    identifier: str,
    name: str,
    row: int,
    col: int,
    *,
    aggregation: str,
    column: str,
) -> dict:
    config = {
        "id": identifier,
        "name": name,
        "column": column,
        "chartType": "scorecard",
        "aggregation": aggregation,
    }
    if aggregation == "SUM":
        config["measureColumn"] = column
    return {
        "kind": "chart",
        "sourceNodeId": SOURCES[source],
        "config": config,
        "col": col,
        "row": row,
        "colSpan": 6,
        "rowSpan": 4,
    }


def bar_chart(
    source: str,
    identifier: str,
    name: str,
    row: int,
    col: int,
    col_span: int,
    *,
    aggregation: str,
    measure: str | None = None,
    row_span: int = 9,
) -> dict:
    config = {
        "id": identifier,
        "name": name,
        "column": "name",
        "chartType": "bar",
        "orientation": "horizontal",
        "aggregation": aggregation,
    }
    if measure is not None:
        config["measureColumn"] = measure
    return {
        "kind": "chart",
        "sourceNodeId": SOURCES[source],
        "config": config,
        "col": col,
        "row": row,
        "colSpan": col_span,
        "rowSpan": row_span,
    }


def distribution(
    source: str,
    identifier: str,
    name: str,
    chart_type: str,
    row: int,
    col: int,
    col_span: int,
) -> dict:
    config = {
        "id": identifier,
        "name": name,
        "column": "dur",
        "chartType": chart_type,
    }
    if chart_type == "histogram":
        config["binCount"] = 24
    return {
        "kind": "chart",
        "sourceNodeId": SOURCES[source],
        "config": config,
        "col": col,
        "row": row,
        "colSpan": col_span,
        "rowSpan": 8,
    }


def timeline(source: str, identifier: str, name: str, row: int) -> dict:
    return {
        "kind": "chart",
        "sourceNodeId": SOURCES[source],
        "config": {
            "id": identifier,
            "name": name,
            "column": "ts",
            "chartType": "scatter",
            "yColumn": "dur",
        },
        "col": 0,
        "row": row,
        "colSpan": 24,
        "rowSpan": 9,
    }


def _common_header(title: str, subtitle: str) -> list[dict]:
    return [
        label("title", title, 0),
        label("story", subtitle, 2),
    ]


REGIMES = {
    "01_launch_bound": {
        "title": "Launch-bound — fewer launches, less host overhead",
        "story": (
            "Representative Case 3 (also Cases 2, 4, 12): packed QKV and CUDA Graph replay. "
            "Profiled GPU kernels fall 115 → 33; final paired speedup 5.959×."
        ),
        "items": _common_header(
            "LAUNCH-BOUND  •  CASE 3",
            "The work is tiny; dispatch is the workload. Compare launch count and CUDA API time before reading kernel duration.",
        )
        + [
            scorecard("kernel", "kernel-time", "GPU kernel time", 4, 0, aggregation="SUM", column="dur"),
            scorecard("kernel", "kernel-count", "GPU kernel launches", 4, 6, aggregation="COUNT", column="name"),
            scorecard("cuda", "cuda-time", "CUDA API time", 4, 12, aggregation="SUM", column="dur"),
            scorecard("cuda", "cuda-count", "CUDA API calls", 4, 18, aggregation="COUNT", column="name"),
            divider("mechanism", "Mechanism: three projections become one; repeated dispatch becomes graph replay", 8),
            bar_chart("kernel", "launches-kernel", "Launches by GPU kernel", 9, 0, 12, aggregation="COUNT"),
            bar_chart("cuda", "cuda-api-time", "CUDA runtime time by API", 9, 12, 12, aggregation="SUM", measure="dur"),
            divider("shape", "Kernel duration distribution", 18),
            bar_chart("kernel", "kernel-time-name", "GPU time by kernel", 19, 0, 12, aggregation="SUM", measure="dur", row_span=8),
            distribution("kernel", "kernel-cdf", "Kernel-duration CDF", "cdf", 19, 12, 12),
            divider("timeline", "Launch pattern across the forward", 27),
            timeline("kernel", "kernel-timeline", "GPU kernel duration over trace time", 28),
        ],
    },
    "02_memory_bound": {
        "title": "Memory-bound — stop materializing and copying attention scores",
        "story": (
            "Representative Case 13 (also Cases 1, 5, 7, 9–11): strided SDPA removes score traffic. "
            "GPU memcpy slices fall 17 → 0; final speedup 8.577× and peak allocation falls 90.4%."
        ),
        "items": _common_header(
            "MEMORY-BOUND  •  CASE 13",
            "The decisive signal is disappearing transfer and elementwise work: 95.7 ms → 13.2 ms profiled GPU kernel time.",
        )
        + [
            scorecard("kernel", "kernel-time", "GPU kernel time", 4, 0, aggregation="SUM", column="dur"),
            scorecard("memcpy", "memcpy-time", "GPU memcpy time", 4, 6, aggregation="SUM", column="dur"),
            scorecard("memcpy", "memcpy-count", "GPU memcpy events", 4, 12, aggregation="COUNT", column="name"),
            scorecard("kernel", "kernel-count", "GPU kernel launches", 4, 18, aggregation="COUNT", column="name"),
            divider("traffic", "Traffic removed from the critical path", 8),
            bar_chart("kernel", "kernel-time-name", "GPU time by kernel", 9, 0, 14, aggregation="SUM", measure="dur"),
            bar_chart("memcpy", "memcpy-time-name", "Transfer time by operation", 9, 14, 10, aggregation="SUM", measure="dur"),
            divider("distributions", "Duration distributions", 18),
            distribution("kernel", "kernel-cdf", "Kernel-duration CDF", "cdf", 19, 0, 12),
            distribution("memcpy", "memcpy-hist", "Memcpy-duration histogram", "histogram", 19, 12, 12),
            divider("timeline", "Before: score construction and copies; after: fused SDPA", 27),
            timeline("kernel", "kernel-timeline", "GPU kernel duration over trace time", 28),
        ],
    },
    "03_projection_bound": {
        "title": "Projection-bound — attention is no longer the ceiling",
        "story": (
            "Representative Case 8: dense projection/FFN GEMMs dominate and attention is only 15.7% of work. "
            "The accepted end-to-end gain is 1.116×; universal packed QKV measured only 1.003×."
        ),
        "items": _common_header(
            "PROJECTION-BOUND  •  CASE 8",
            "SDPA removes attention overhead, but the same long GEMM family remains. This dashboard makes the Amdahl ceiling visible.",
        )
        + [
            scorecard("kernel", "kernel-time", "GPU kernel time", 4, 0, aggregation="SUM", column="dur"),
            scorecard("kernel", "kernel-count", "GPU kernel launches", 4, 6, aggregation="COUNT", column="name"),
            scorecard("memcpy", "memcpy-time", "GPU memcpy time", 4, 12, aggregation="SUM", column="dur"),
            scorecard("cpu", "cpu-time", "CPU operator time", 4, 18, aggregation="SUM", column="dur"),
            divider("ceiling", "Where the time remains", 8),
            bar_chart("kernel", "kernel-time-name", "GPU time by kernel — GEMMs should dominate", 9, 0, 16, aggregation="SUM", measure="dur"),
            bar_chart("kernel", "kernel-launches-name", "Launches by kernel", 9, 16, 8, aggregation="COUNT"),
            divider("duration", "Long-kernel tail", 18),
            distribution("kernel", "kernel-hist", "Kernel-duration histogram", "histogram", 19, 0, 12),
            distribution("kernel", "kernel-cdf", "Kernel-duration CDF", "cdf", 19, 12, 12),
            divider("timeline", "GEMM-dominated forward", 27),
            timeline("kernel", "kernel-timeline", "GPU kernel duration over trace time", 28),
        ],
    },
    "04_capacity_bound": {
        "title": "Capacity-bound — stream the batch to stay within VRAM",
        "story": (
            "Representative Case 6: B=10,000 is divided into bounded chunks. The candidate deliberately launches "
            "more repeated work (335 kernels) while cutting peak allocation 78.3% and latency 2.394×."
        ),
        "items": _common_header(
            "CAPACITY-BOUND  •  CASE 6",
            "More launches are intentional here: repeated waves expose batch streaming, trading bounded working memory for tractability.",
        )
        + [
            scorecard("kernel", "kernel-time", "GPU kernel time", 4, 0, aggregation="SUM", column="dur"),
            scorecard("kernel", "kernel-count", "GPU kernel launches", 4, 6, aggregation="COUNT", column="name"),
            scorecard("memcpy", "memcpy-time", "GPU memcpy time", 4, 12, aggregation="SUM", column="dur"),
            scorecard("cuda", "cuda-count", "CUDA API calls", 4, 18, aggregation="COUNT", column="name"),
            divider("waves", "Streaming waves across the complete B=10,000 forward", 8),
            timeline("kernel", "kernel-timeline", "Repeated GPU kernel waves", 9),
            divider("composition", "What each chunk executes", 18),
            bar_chart("kernel", "kernel-time-name", "GPU time by kernel", 19, 0, 12, aggregation="SUM", measure="dur"),
            bar_chart("kernel", "kernel-count-name", "Launches by kernel", 19, 12, 12, aggregation="COUNT"),
            divider("transfer", "Transfer distribution after chunking", 28),
            distribution("kernel", "kernel-cdf", "Kernel-duration CDF", "cdf", 29, 0, 12),
            distribution("memcpy", "memcpy-hist", "Memcpy-duration histogram", "histogram", 29, 12, 12),
        ],
    },
    "05_allocation_bound": {
        "title": "Allocation-bound — change the algorithm, not the allocator",
        "story": (
            "Representative Case 14: dense FP32 scores require 18.63 TiB. The reference trace is the streamed FP32 oracle; "
            "the candidate uses guarded polynomial Triton attention. Full validation: PASS 5/5, 3.6 GiB, 8.972× diagnostic ratio."
        ),
        "items": _common_header(
            "ALLOCATION-BOUND  •  CASE 14",
            "One full N=100,000 streamed sample per trace. Oracle: 53 large GPU kernels; candidate: a 4,590-kernel linear scan with no quadratic tensor.",
        )
        + [
            scorecard("kernel", "kernel-time", "GPU kernel time", 4, 0, aggregation="SUM", column="dur"),
            scorecard("kernel", "kernel-count", "GPU kernel launches", 4, 6, aggregation="COUNT", column="name"),
            scorecard("cuda", "cuda-count", "CUDA API calls", 4, 12, aggregation="COUNT", column="name"),
            scorecard("cpu", "cpu-count", "CPU operator slices", 4, 18, aggregation="COUNT", column="name"),
            divider("algorithm", "Exact fused attention versus guarded linear scan", 8),
            bar_chart("kernel", "kernel-time-name", "GPU time by kernel", 9, 0, 14, aggregation="SUM", measure="dur"),
            bar_chart("kernel", "kernel-count-name", "Launches by kernel", 9, 14, 10, aggregation="COUNT"),
            divider("scan", "Kernel granularity", 18),
            distribution("kernel", "kernel-hist", "Kernel-duration histogram", "histogram", 19, 0, 12),
            distribution("kernel", "kernel-cdf", "Kernel-duration CDF", "cdf", 19, 12, 12),
            divider("timeline", "Chunked scan across 100,000 causal positions", 27),
            timeline("kernel", "kernel-timeline", "GPU kernel duration over trace time", 28),
        ],
    },
}


def build(output: Path = OUTPUT) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, spec in REGIMES.items():
        document = {
            "version": 1,
            "title": spec["title"],
            "graph": _graph(spec["title"], spec["story"]),
            "dashboards": [{"id": stem, "items": spec["items"]}],
        }
        destination = output / f"{stem}.perfetto-dashboard.json"
        destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        written.append(destination)
    return written


if __name__ == "__main__":
    for path in build():
        print(path)
