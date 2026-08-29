# Perfetto Data Explorer dashboard

[`transformer_profile_dashboard.json`](transformer_profile_dashboard.json) is
an importable Perfetto Data Explorer tab. It uses Perfetto's own query graph and
dashboard components; it is not a custom report or visualization layer.

To use it:

1. Open `baseline-trace.json` or `candidate-trace.json` in
   <https://ui.perfetto.dev>.
2. Open **Data Explorer**.
3. Choose **Import tab** and select
   `src/perfetto/transformer_profile_dashboard.json`.
4. Switch from **Graph** to **Dashboard**.

Open baseline and candidate traces in separate browser tabs and import the same
dashboard into both. This keeps the queries and chart layout identical while
the underlying trace changes.

The graph uses Perfetto's typed `slice` table rather than custom SQL result
columns. It filters only the standard `kernel` and `gpu_memcpy` categories
emitted by PyTorch Profiler, so it does not depend on a candidate name or on
candidate-specific annotations.

The dashboard includes:

- scorecards for total GPU kernel time, kernel launches, and GPU memcpy time;
- GPU time and launch counts grouped by the actual kernel name;
- a kernel-duration histogram; and
- a kernel timeline, kernel-duration CDF, and memcpy-duration histogram.

The `ts` and `dur` columns keep their native Perfetto timestamp and duration
types, allowing Data Explorer to format their units appropriately and making
numeric charts work without manual column-type repair after import.
