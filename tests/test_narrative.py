from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.narrative import (
    affective_curve,
    arc_distribution,
    discourse_feature_metrics,
    plot_diversity,
    turning_point_summary,
)
from multistory.text import jensen_shannon_distance


class NarrativeTests(unittest.TestCase):
    def test_arc_distribution_sums_to_one(self):
        result = arc_distribution(["Man in Hole", "Cinderella", "Man in Hole"])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertEqual(result["counts"]["Man in Hole"], 2)

    def test_turning_points_are_normalized_by_sentence_count(self):
        record = {
            "story": "One. Two. Three. Four. Five.",
            "turning_points": {"tp1": 1, "tp2": 2, "tp3": 3, "tp4": 4, "tp5": 5},
        }
        result = turning_point_summary([record])
        self.assertAlmostEqual(result["mean_normalized_position"]["tp3"], 0.6)
        self.assertEqual(result["ordering_violation_count"], 0)

    def test_turning_point_order_violation_is_counted(self):
        record = {
            "story": "One. Two. Three. Four. Five.",
            "turning_points": {"tp1": 1, "tp2": 3, "tp3": 2, "tp4": 4, "tp5": 5},
        }
        self.assertEqual(turning_point_summary([record])["ordering_violation_count"], 1)

    def test_affective_curve_has_requested_length(self):
        result = affective_curve("Hope returned. Fear remained.", {"hope": (0.9, 0.5), "fear": (0.1, 0.9)}, bins=7)
        self.assertEqual(len(result["valence"]), 7)
        self.assertEqual(len(result["arousal"]), 7)
        self.assertEqual(result["recognized_token_count"], 2)

    def test_js_distance_identical_is_zero(self):
        distribution = {"a": 0.25, "b": 0.75}
        self.assertAlmostEqual(jensen_shannon_distance(distribution, distribution), 0.0)

    def test_discourse_features_capture_tension_and_affect_range(self):
        lexicon = {
            "hope": (0.9, 0.5),
            "fear": (0.1, 0.9),
            "danger": (0.1, 0.9),
            "victory": (0.95, 0.8),
        }
        result = discourse_feature_metrics(
            "Fear and danger grew. Hope returned after victory.", lexicon
        )
        self.assertGreater(result["suspense_proxy"], 0)
        self.assertGreater(result["valence_range"], 0.7)
        self.assertGreater(result["arousal_range"], 0.3)

    def test_plot_diversity_counts_repeated_signatures(self):
        records = [
            {
                "story": "Fear rose. Hope returned.",
                "declared_arc": "Man in Hole",
                "narrative_parameters": {
                    "plot_complexity": "single_track",
                    "temporal_structure": "linear",
                },
            },
            {
                "story": "Fear rose again. Hope returned.",
                "declared_arc": "Man in Hole",
                "narrative_parameters": {
                    "plot_complexity": "single_track",
                    "temporal_structure": "linear",
                },
            },
        ]
        result = plot_diversity(records)
        self.assertEqual(result["unique_signature_count"], 1)
        self.assertEqual(result["duplicate_signature_count"], 1)
        self.assertEqual(result["duplicate_signature_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
