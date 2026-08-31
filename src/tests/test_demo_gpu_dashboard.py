import argparse
import unittest

from src.demo_gpu_dashboard import (
    GpuSample,
    benchmark_phase,
    build_benchmark_command,
    parse_nvidia_smi_row,
    render_dashboard,
    sparkline,
)


class DemoGpuDashboardTests(unittest.TestCase):
    def test_parses_nvidia_smi_csv(self):
        sample = parse_nvidia_smi_row(
            "NVIDIA GeForce RTX 5080, 97, 3644, 16303, 280.5, 360.0, 66, 2715"
        )
        self.assertEqual(sample.name, "NVIDIA GeForce RTX 5080")
        self.assertEqual(sample.utilization, 97)
        self.assertEqual(sample.memory_total_mib, 16303)
        self.assertEqual(sample.sm_clock_mhz, 2715)

    def test_sparkline_uses_a_fixed_scale(self):
        self.assertEqual(sparkline([0, 50, 100], maximum=100), "▁▄█")

    def test_phase_tracks_correctness_and_completion(self):
        self.assertIn("compiling", benchmark_phase([], True, None))
        self.assertIn(
            "Correctness passed",
            benchmark_phase(["accuracy 01/1: PASS"], True, None),
        )
        self.assertIn("passed", benchmark_phase([], False, 0))
        self.assertIn("failed", benchmark_phase([], False, 2))

    def test_frame_contains_metrics_and_published_context(self):
        sample = GpuSample("Test GPU", 80, 8000, 16000, 200, 400, 60, 2500)
        history = {
            "utilization": [10, 80],
            "temperature": [55, 60],
            "clock": [2000, 2500],
        }
        frame = render_dashboard(
            sample,
            history,
            ["accuracy 01/1: PASS", "speedup=8.000x"],
            12.5,
            True,
            None,
            13,
            color=False,
        )
        self.assertIn("GPU compute", frame)
        self.assertIn("8,000/16,000 MiB", frame)
        self.assertIn("accuracy 01/1: PASS", frame)
        self.assertIn("3.611× geomean", frame)

    def test_command_uses_abbreviated_paired_demo_settings(self):
        args = argparse.Namespace(
            case=13,
            device="cuda",
            accuracy_trials=1,
            warmup=5,
            repeats=20,
            benchmark_rounds=1,
            settle_seconds=2.0,
            output=None,
        )
        command = build_benchmark_command(args)
        self.assertIn("src.benchmark", command)
        self.assertIn("paired", command)
        self.assertEqual(command[command.index("--case") + 1], "13")


if __name__ == "__main__":
    unittest.main()
