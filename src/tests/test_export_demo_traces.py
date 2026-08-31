import json
from pathlib import Path
import tempfile
import unittest

from src.export_demo_traces import _portable_trace


class ExportDemoTracesTests(unittest.TestCase):
    def test_portable_trace_rebases_and_preserves_events(self):
        document = {
            "traceEvents": [
                {"ph": "M", "name": "process_name", "ts": 1000.0},
                {"ph": "X", "cat": "kernel", "name": "k", "ts": 1010.0, "dur": 2.0},
                {"ph": "X", "cat": "gpu_memcpy", "name": "m", "ts": 1020.0, "dur": 1.0},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            destination = Path(directory) / "case1_candidate.perfetto.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            stats = _portable_trace(source, destination)
            events = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual([event["ts"] for event in events], [0.0, 10.0, 20.0])
        self.assertEqual(stats["events"], 3)
        self.assertEqual(stats["kernel_events"], 1)
        self.assertEqual(stats["gpu_memcpy_events"], 1)


if __name__ == "__main__":
    unittest.main()
