import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.datasets import load_narrative_reference, summarize_reference


class DatasetTests(unittest.TestCase):
    def test_join_reports_missing_and_incomplete_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            narratives = {
                "1": {"source": "Human", "title": "One", "synopsis": ["A.", "B.", "C.", "D.", "E."]},
                "2": {"source": "GPT", "title": "Two", "synopsis": ["A."]},
            }
            arcs = {"1": "Man in Hole", "2": "Cinderella"}
            turning_points = {
                "1": {"tp1": 1, "tp2": 2, "tp3": 3, "tp4": 4, "tp5": 5},
                "3": {"tp1": 1, "tp2": 2, "tp3": 3, "tp4": 4, "tp5": 5},
            }
            for name, value in (
                ("narratives.json", narratives),
                ("ground_truth_arc.json", arcs),
                ("ground_truth_tp.json", turning_points),
            ):
                with (root / name).open("w", encoding="utf-8") as handle:
                    json.dump(value, handle)
            dataset = load_narrative_reference(root)
            summary = summarize_reference(dataset)
            self.assertEqual(summary["dataset"]["joined_annotation_count"], 1)
            self.assertEqual(summary["dataset"]["missing_narrative_count"], 1)
            self.assertEqual(summary["dataset"]["incomplete_annotation_count"], 1)
            self.assertEqual(summary["sources"]["human"]["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
