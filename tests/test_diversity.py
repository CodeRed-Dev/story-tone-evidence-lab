import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.diversity import corpus_diversity, maximum_entropy, normalized_entropy


class DiversityTests(unittest.TestCase):
    def test_maximum_entropy_when_samples_fewer_than_categories(self):
        self.assertAlmostEqual(maximum_entropy(3, 7), math.log(3))
        self.assertAlmostEqual(normalized_entropy(["a", "b", "c"], 7), 1.0)

    def test_maximum_entropy_when_samples_equal_categories(self):
        self.assertAlmostEqual(maximum_entropy(4, 4), math.log(4))

    def test_maximum_entropy_uneven_allocation(self):
        expected = -(2 * (3 / 8) * math.log(3 / 8) + (2 / 8) * math.log(2 / 8))
        self.assertAlmostEqual(maximum_entropy(8, 3), expected)

    def test_empty_and_singleton_are_zero(self):
        self.assertEqual(maximum_entropy(0, 7), 0.0)
        self.assertEqual(normalized_entropy([], 7), 0.0)
        self.assertEqual(normalized_entropy(["a"], 7), 0.0)

    def test_identical_stories_have_no_pairwise_distance(self):
        result = corpus_diversity(["one quiet night", "one quiet night"])
        self.assertEqual(result["pairwise_token_jaccard_distance"], 0.0)

    def test_different_stories_have_positive_distance(self):
        result = corpus_diversity(["one quiet night", "storms break every window"])
        self.assertGreater(result["pairwise_token_jaccard_distance"], 0.0)
        self.assertGreater(result["pairwise_bigram_jaccard_distance"], 0.0)


if __name__ == "__main__":
    unittest.main()
