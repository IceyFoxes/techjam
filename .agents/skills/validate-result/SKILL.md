---
name: validate-result
description: Use when validating an implementation's correctness and performance against three reference points: the immutable baseline, the commit the branch was based on, and the latest result. Covers accuracy checks, benchmark runs, and result comparison across all three.
---

# Validate Result

Validate an implementation against three reference points and report the
combined outcome. Every validation run must cover all three; do not report
only a subset.

## Reference Points

1. **Baseline** — the unmodified `torch_transformer_benchmark.py`
   `BaselineTransformer`. This is the immutable correctness oracle.
2. **Branch base commit** — the Git commit on `master` that the
   implementation branch forked from. Record its SHA, timestamp, and the
   benchmark result at that commit for regression detection.
3. **Latest result** — the current state of the implementation on the active
   branch. This is what is being validated.

## Step 1: Identify the Branch Base Commit

```bash
git merge-base master HEAD
```

Record the full SHA and short SHA. If on `master`, the base commit is HEAD
itself and the branch/commit comparison is a no-op (report this).

## Step 2: Run Baseline Accuracy and Benchmark

```bash
python src/benchmark.py \
  --candidate reference \
  --case <CASE_ID> \
  --dtype float32 \
  --accuracy-trials 5 \
  --seed 1234 \
  --output research/benchmarks/<DATE>-<GPU>-<BASE_COMMIT_SHORT>/baseline-case<CASE>.json
```

This produces the baseline timing and accuracy for the given official case.
Repeat for every official case the implementation claims to support.

## Step 3: Run Candidate Accuracy and Benchmark

```bash
python src/benchmark.py \
  --candidate dispatcher \
  --case <CASE_ID> \
  --dtype float32 \
  --accuracy-trials 5 \
  --seed 1234 \
  --output research/benchmarks/<DATE>-<GPU>-<LATEST_COMMIT_SHORT>/candidate-case<CASE>.json
```

Record the output JSON. The harness prints accuracy PASS/FAIL, latency
median, speedup, noise floor, and significance.

## Step 4: Validate Correctness Criteria

For every trial on every case, the output must satisfy:

```
abs(candidate - baseline) <= 0.002
OR
abs(candidate - baseline) <= 0.02 * abs(baseline)
```

A result is only valid if:
- All accuracy trials PASS (0 failed elements).
- The timing significance verdict is `SIGNIFICANT` or the speedup is >= 1.0x.
- The candidate JSON records the exact command, Git SHA, timestamp, environment,
  and raw latency samples.

## Step 5: Detect Regression Against Branch Base

Compare the latest candidate result against the base-commit result:

1. **Accuracy regression**: latest has any FAIL trial that the base commit
   passed → regression, block the result.
2. **Performance regression**: latest median latency is >5% slower than the
   base commit on any case → flag as potential regression, investigate.
3. **New failures**: cases that the base commit supported but the latest
   refuses → regression unless deliberately dropped with documented reason.

## Step 6: Report Combined Outcome

Present all three reference points together in a single table. Example:

```
=== Validation Summary — case <N> ===

Reference       | Commit     | Accuracy    | Median ms | Speedup
----------------|------------|-------------|-----------|--------
Baseline        | (immutable)| PASS 5/5    | <X.XXXX>  | 1.000x
Branch base     | <short-sha>| PASS 5/5    | <X.XXXX>  | <X.XXX>x
Latest candidate| <short-sha>| PASS 5/5    | <X.XXXX>  | <X.XXX>x

Delta vs baseline:  <X.XXX>x speedup, <N> failed elements
Delta vs base commit: <+/-X.X%> latency change, no accuracy regression
```

### Reporting Rules

- Always show all three columns (baseline, branch base, latest) even if
  two are identical.
- State the Git SHA for the branch base and latest candidate.
- Report accuracy as `PASS N/N` with the max absolute error.
- Report speedup relative to the baseline (always 1.000x for baseline).
- Flag any regression with `⚠ REGRESSION` and a one-line explanation.
- If a case is unsupported, show `UNSUPPORTED` with the reason, not a blank.

## Step 7: Preserve the Run

Save results under:

```
research/benchmarks/<DATE>-<GPU>-<LATEST_COMMIT_SHORT>/
```

Include:
- One JSON per case (candidate result).
- One JSON per case (baseline result, if not already recorded at the base commit).
- A `README.md` with the summary table, environment, and links to JSONs.
- Never overwrite an existing run; mark stale runs in
  `research/benchmarks/README.md`.

## Environment Metadata

Every preserved run must record (from the benchmark JSON or manually):

- CPU model and core count
- GPU name, driver version, CUDA runtime version
- PyTorch version and build
- OS and kernel version
- The exact `python src/benchmark.py ...` command
- Git commit SHA and dirty-tree status
- Timestamp (UTC ISO-8601)

## Quick Reference

| What | Command |
| --- | --- |
| Find branch base | `git merge-base master HEAD` |
| Run baseline | `python src/benchmark.py --candidate reference --case N` |
| Run candidate | `python src/benchmark.py --candidate dispatcher --case N` |
| Save result | `--output research/benchmarks/<dir>/caseN.json` |
| All official cases | `for c in 1 2 3 4 5 7 8 9 10 11 12 13; do ... done` |
| Accuracy rule | `abs(err) <= 0.002 OR abs(err) <= 0.02 * abs(ref)` |
