# Five-regime Perfetto dashboards

These dashboards are designed for a narrated A/B demo. Each one focuses on the
signal that explains its regime rather than presenting a generic profiler grid.

| Dashboard | Official cases | Recommended trace pair | Story |
| --- | --- | --- | --- |
| `01_launch_bound` | 2, 3, 4, 12 | Case 3 | Packed QKV and graph replay collapse launch/API overhead |
| `02_memory_bound` | 1, 5, 7, 9--11, 13 | Case 13 | SDPA removes score materialization and GPU copies |
| `03_projection_bound` | 8 | Case 8 | Dense projection/FFN GEMMs remain the Amdahl ceiling |
| `04_capacity_bound` | 6 | Case 6 | Repeated kernel waves expose bounded batch streaming |
| `05_allocation_bound` | 14 | Case 14 | The FP32 oracle and guarded polynomial route use different algorithms |

## Import workflow

1. In Perfetto, choose **Open trace file** and load the recommended
   `caseN_reference.perfetto.json` from the demo evidence bundle.
2. Open **Data Explorer**, choose **Import**, and select the matching
   `*.perfetto-dashboard.json` file from this directory.
3. Switch from **Graph** to **Dashboard** and collapse the sidebar.
4. Repeat in a second browser tab with `caseN_candidate.perfetto.json`.
5. During the demo, switch between the two tabs while narrating the static
   headline and the live scorecards.

The trace is opened with **Open trace file**. The dashboard alone is imported
through Data Explorer; importing a trace through the dashboard importer produces
an empty view.

## Demo sequence

- Use Perfetto dark mode and browser fullscreen.
- Start on the reference tab for roughly five seconds.
- Point to the scorecards, then the dominant-name bars and timeline.
- Switch to the candidate tab without changing the dashboard.
- Treat profiler durations as attribution only. Quote speedup and peak memory
  from the static dashboard text, which comes from the preserved final matrix.

Case 14's `reference` trace is the validated streamed FP32 oracle, not the
infeasible dense implementation. Both Case 14 traces contain one complete
`N=100000`, `D=1024` streamed sample of the official `B=32` execution.
