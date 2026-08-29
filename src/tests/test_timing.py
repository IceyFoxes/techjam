from __future__ import annotations

import statistics
import time
import unittest

from src.infra.timing import (
    is_significant,
    choose_block_size,
    make_block_runner,
    measure_pair,
    noise_floor_ratio,
    paired_schedule,
    paired_summary,
    settle,
)

BASELINE = "baseline"
CANDIDATE = "candidate"


def drifting_runners(baseline_ms: float, candidate_ms: float, final_multiplier: float,
                     total_samples: int):
    """Fake timers whose latency drifts, reproducing GPU boost-budget decay.

    The observed RTX 4060 behaviour is a ~2x slowdown from the first samples to
    steady state, so a schedule that measures one side earlier than the other
    reports a fabricated speedup.
    """
    state = {"call": 0}

    def multiplier() -> float:
        progress = state["call"] / max(1, total_samples - 1)
        state["call"] += 1
        return 1.0 + (final_multiplier - 1.0) * progress

    return {
        BASELINE: lambda: baseline_ms * multiplier(),
        CANDIDATE: lambda: candidate_ms * multiplier(),
    }


class PairedScheduleTests(unittest.TestCase):
    def test_runs_each_side_the_requested_number_of_times(self) -> None:
        schedule = paired_schedule(8)

        self.assertEqual(schedule.count(BASELINE), 8)
        self.assertEqual(schedule.count(CANDIDATE), 8)

    def test_balances_mean_position_so_drift_cannot_favour_either_side(self) -> None:
        schedule = paired_schedule(64)

        baseline_positions = [i for i, s in enumerate(schedule) if s == BASELINE]
        candidate_positions = [i for i, s in enumerate(schedule) if s == CANDIDATE]

        self.assertAlmostEqual(
            statistics.mean(baseline_positions),
            statistics.mean(candidate_positions),
            places=9,
        )

    def test_rejects_non_positive_repeats(self) -> None:
        with self.assertRaisesRegex(ValueError, "repeats"):
            paired_schedule(0)


class DriftCancellationTests(unittest.TestCase):
    def test_recovers_true_speedup_despite_two_fold_drift(self) -> None:
        repeats = 200
        runners = drifting_runners(
            baseline_ms=100.0,
            candidate_ms=80.0,
            final_multiplier=2.0,
            total_samples=repeats * 2,
        )

        samples = measure_pair(runners, repeats)
        summary = paired_summary(samples)

        self.assertAlmostEqual(summary["speedup"], 1.25, delta=0.02)

    def test_identical_sides_report_unit_speedup_despite_drift(self) -> None:
        repeats = 200
        runners = drifting_runners(
            baseline_ms=100.0,
            candidate_ms=100.0,
            final_multiplier=2.0,
            total_samples=repeats * 2,
        )

        summary = paired_summary(measure_pair(runners, repeats))

        self.assertAlmostEqual(summary["speedup"], 1.0, delta=0.01)


class DeferredSampleTests(unittest.TestCase):
    """CUDA samples must not force a device sync per sample.

    Synchronising after every sample leaves the GPU idle long enough to regain
    its boost clocks, which inflates the spread and destroys sensitivity. Runners
    may therefore return a pending sample resolved once, after all timing.
    """

    class PendingSample:
        def __init__(self, value: float) -> None:
            self.value = value
            self.resolved = False

        def milliseconds(self) -> float:
            self.resolved = True
            return self.value

    def test_resolves_pending_samples_into_milliseconds(self) -> None:
        pending = [self.PendingSample(float(i + 1)) for i in range(4)]
        issued = iter(pending)
        runner = lambda: next(issued)

        samples = measure_pair({BASELINE: runner, CANDIDATE: runner}, repeats=2)

        self.assertEqual(sorted(samples[BASELINE] + samples[CANDIDATE]),
                         [1.0, 2.0, 3.0, 4.0])
        self.assertTrue(all(p.resolved for p in pending))

    def test_defers_resolution_until_every_sample_is_recorded(self) -> None:
        recorded: List[str] = []

        class Tracking(DeferredSampleTests.PendingSample):
            def milliseconds(self) -> float:
                recorded.append("resolve")
                return super().milliseconds()

        def runner() -> "DeferredSampleTests.PendingSample":
            recorded.append("record")
            return Tracking(1.0)

        measure_pair({BASELINE: runner, CANDIDATE: runner}, repeats=3)

        self.assertEqual(recorded, ["record"] * 6 + ["resolve"] * 6)


class NoiseFloorTests(unittest.TestCase):
    """The floor must measure disagreement BETWEEN paired samples.

    Raw per-sample spread is the wrong quantity: under boost-clock decay both
    sides drift together, which the paired schedule already cancels. Charging
    that drift to the noise floor reports ~31% on a run whose A/A ratio is
    accurate to 2%, and would veto every real improvement.
    """

    def test_noise_floor_of_a_noiseless_pair_is_zero(self) -> None:
        samples = {BASELINE: [10.0] * 20, CANDIDATE: [10.0] * 20}

        self.assertAlmostEqual(noise_floor_ratio(samples), 0.0, places=9)

    def test_noise_floor_grows_with_pairwise_disagreement(self) -> None:
        tight = {BASELINE: [10.0, 10.1, 9.9] * 8, CANDIDATE: [10.0] * 24}
        loose = {BASELINE: [10.0, 14.0, 6.0] * 8, CANDIDATE: [10.0] * 24}

        self.assertLess(noise_floor_ratio(tight), noise_floor_ratio(loose))

    def test_shared_drift_does_not_inflate_the_noise_floor(self) -> None:
        # Both sides slow by 2x together; every pair still agrees exactly.
        drift = [10.0 * (1.0 + i / 40.0) for i in range(40)]
        samples = {BASELINE: drift, CANDIDATE: list(drift)}

        self.assertLess(noise_floor_ratio(samples), 0.01)

    def test_more_samples_tighten_the_noise_floor(self) -> None:
        pattern_short = {BASELINE: [10.0, 11.0] * 5, CANDIDATE: [10.0] * 10}
        pattern_long = {BASELINE: [10.0, 11.0] * 50, CANDIDATE: [10.0] * 100}

        self.assertLess(
            noise_floor_ratio(pattern_long), noise_floor_ratio(pattern_short)
        )

    def test_speedup_inside_the_noise_floor_is_not_significant(self) -> None:
        self.assertFalse(is_significant(speedup=1.05, noise_floor=0.10))

    def test_speedup_outside_the_noise_floor_is_significant(self) -> None:
        self.assertTrue(is_significant(speedup=1.40, noise_floor=0.10))

    def test_regression_outside_the_noise_floor_is_significant(self) -> None:
        self.assertTrue(is_significant(speedup=0.70, noise_floor=0.10))


class ChooseBlockSizeTests(unittest.TestCase):
    def test_compute_bound_shapes_are_timed_one_iteration_at_a_time(self) -> None:
        self.assertEqual(choose_block_size(per_iteration_ms=425.0, target_ms=50.0), 1)

    def test_launch_bound_shapes_are_batched_up_to_the_target(self) -> None:
        self.assertEqual(choose_block_size(per_iteration_ms=1.0, target_ms=50.0), 50)

    def test_block_size_is_capped(self) -> None:
        block = choose_block_size(per_iteration_ms=0.0001, target_ms=50.0, maximum=200)

        self.assertEqual(block, 200)

    def test_rejects_non_positive_iteration_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_iteration_ms"):
            choose_block_size(per_iteration_ms=0.0, target_ms=50.0)


class RecordingModel:
    """Stands in for a Transformer; records calls and burns a known duration."""

    def __init__(self, seconds_per_call: float = 0.0) -> None:
        self.calls = 0
        self.seconds_per_call = seconds_per_call

    def __call__(self, x, valid_token_mask=None):
        self.calls += 1
        if self.seconds_per_call:
            time.sleep(self.seconds_per_call)
        return x


class BlockRunnerTests(unittest.TestCase):
    def test_invokes_the_model_once_per_block_iteration(self) -> None:
        model = RecordingModel()
        runner = make_block_runner(model, x=None, valid_mask=None,
                                   device=None, block_size=5)

        runner()

        self.assertEqual(model.calls, 5)

    def test_reports_per_iteration_time_not_block_total(self) -> None:
        model = RecordingModel(seconds_per_call=0.004)
        runner = make_block_runner(model, x=None, valid_mask=None,
                                   device=None, block_size=5)

        per_iteration_ms = runner()

        self.assertGreater(per_iteration_ms, 2.0)
        self.assertLess(per_iteration_ms, 8.0)

    def test_rejects_non_positive_block_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "block_size"):
            make_block_runner(RecordingModel(), x=None, valid_mask=None,
                              device=None, block_size=0)


class SettleTests(unittest.TestCase):
    def test_runs_the_model_until_the_deadline_elapses(self) -> None:
        model = RecordingModel(seconds_per_call=0.001)

        started = time.perf_counter()
        settle(model, x=None, valid_mask=None, device=None, seconds=0.05)
        elapsed = time.perf_counter() - started

        self.assertGreaterEqual(elapsed, 0.05)
        self.assertGreater(model.calls, 0)

    def test_zero_duration_still_warms_the_model_once(self) -> None:
        model = RecordingModel()

        settle(model, x=None, valid_mask=None, device=None, seconds=0.0)

        self.assertGreater(model.calls, 0)


if __name__ == "__main__":
    unittest.main()
