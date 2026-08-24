import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.linearity import analyze_linearity_corpus, chronology_metrics, linearity_markdown


class LinearityTests(unittest.TestCase):
    def test_marker_families_are_inspectable(self):
        result = chronology_metrics(
            "Years later, Mira remembered the summer before. Then she finally returned home.",
            sentence_count=2,
        )
        self.assertGreaterEqual(result["retrospective_count"], 2)
        self.assertGreaterEqual(result["temporal_jump_count"], 1)
        self.assertGreaterEqual(result["forward_count"], 2)
        self.assertIn("remember", result["matches"]["retrospective"])

    def test_empty_text_remains_finite(self):
        result = chronology_metrics("")
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["retrospective_per_100_sentences"], 0.0)
        self.assertIsNone(result["forward_share"])

    def test_balanced_corpus_produces_human_minus_gpt_contrasts(self):
        corpus = {
            "h1": {
                "source": "Human",
                "title": "H1",
                "synopsis": ["Years ago, Ana remembered her childhood.", "A decade later, she returned."],
            },
            "h2": {
                "source": "Human",
                "title": "H2",
                "synopsis": ["Before the journey, Ravi recalled an earlier promise."],
            },
            "g1": {
                "source": "GPT",
                "title": "G1",
                "synopsis": ["First he left.", "Then he arrived.", "Finally he celebrated."],
            },
            "g2": {
                "source": "GPT",
                "title": "G2",
                "synopsis": ["Next she opened the door and eventually found the prize."],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "narratives.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(corpus, handle)
            result = analyze_linearity_corpus(path, bootstrap_iterations=50, seed=9)
        self.assertEqual(result["metadata"]["story_count"], 4)
        self.assertGreater(
            result["contrasts"]["retrospective_per_100_sentences"]["human_minus_gpt"], 0
        )
        self.assertTrue(result["storyscope_directional_comparison"]["local_discontinuity_proxies_align"])
        self.assertIn("lexical chronology proxy", linearity_markdown(result))


if __name__ == "__main__":
    unittest.main()
