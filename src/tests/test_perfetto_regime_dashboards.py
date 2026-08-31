import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "perfetto" / "regime_dashboards"


class PerfettoRegimeDashboardTests(unittest.TestCase):
    def test_five_dashboards_are_well_formed_and_self_consistent(self):
        paths = sorted(ROOT.glob("*.perfetto-dashboard.json"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(path=path.name):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(document["version"], 1)
                self.assertEqual(len(document["dashboards"]), 1)
                graph = json.loads(document["graph"])
                nodes = graph["nodes"]
                node_ids = [node["nodeId"] for node in nodes]
                self.assertEqual(len(node_ids), len(set(node_ids)))
                self.assertEqual(graph["rootNodeIds"], ["0"])
                self.assertTrue(graph["isExplorerCollapsed"])

                dashboard_nodes = {
                    node["nodeId"] for node in nodes if node["type"] == "dashboard"
                }
                for node in nodes:
                    if node["type"] == "dashboard":
                        self.assertIn("name", node["state"])
                        self.assertNotIn("exportName", node["state"])
                items = document["dashboards"][0]["items"]
                charts = [item for item in items if item["kind"] == "chart"]
                self.assertGreaterEqual(len(charts), 9)
                for chart in charts:
                    self.assertIn(chart["sourceNodeId"], dashboard_nodes)
                    self.assertIn(
                        chart["config"]["chartType"],
                        {"scorecard", "bar", "histogram", "cdf", "scatter"},
                    )

                chart_ids = [chart["config"]["id"] for chart in charts]
                self.assertEqual(len(chart_ids), len(set(chart_ids)))

    def test_every_dashboard_names_its_representative_case(self):
        expected = {
            "01_launch_bound.perfetto-dashboard.json": "CASE 3",
            "02_memory_bound.perfetto-dashboard.json": "CASE 13",
            "03_projection_bound.perfetto-dashboard.json": "CASE 8",
            "04_capacity_bound.perfetto-dashboard.json": "CASE 6",
            "05_allocation_bound.perfetto-dashboard.json": "CASE 14",
        }
        for name, case_label in expected.items():
            document = json.loads((ROOT / name).read_text(encoding="utf-8"))
            labels = [
                item["text"]
                for item in document["dashboards"][0]["items"]
                if item["kind"] == "label"
            ]
            self.assertTrue(any(case_label in text for text in labels))


if __name__ == "__main__":
    unittest.main()
