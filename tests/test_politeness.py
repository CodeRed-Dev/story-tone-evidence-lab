from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.politeness import score_politeness, script_ratios


class PolitenessTests(unittest.TestCase):
    def test_english_positive_strategies(self):
        result = score_politeness(
            "Certainly. Please consider this option, and thank you for asking. I hope this helps.", "en"
        )
        self.assertEqual(result["strategy_presence"]["request_softener"], 1)
        self.assertEqual(result["strategy_presence"]["gratitude"], 1)
        self.assertEqual(result["strategy_presence"]["warmth_or_support"], 1)
        self.assertGreaterEqual(result["politeness_strategy_coverage"], 0.5)

    def test_hindi_deference_and_script(self):
        result = score_politeness(
            "कृपया आप इस सुझाव पर विचार करें। यदि संभव हो तो मुझे बताइए। धन्यवाद।", "hi"
        )
        self.assertEqual(result["strategy_presence"]["request_softener"], 1)
        self.assertEqual(result["strategy_presence"]["respect_or_honorific"], 1)
        self.assertEqual(result["strategy_presence"]["hedging_or_indirectness"], 1)
        self.assertGreater(result["target_script_ratio"], 0.9)

    def test_informal_form_is_reported_but_not_called_total_impoliteness(self):
        result = score_politeness("तुम यह काम करो।", "hi")
        self.assertEqual(result["informal_or_blunt_presence"], 1)
        self.assertIn("does not measure", result["validity_warning"])

    def test_script_ratios_detect_mixed_output(self):
        ratios = script_ratios("यह Hindi mixed text है")
        self.assertGreater(ratios["devanagari"], 0)
        self.assertGreater(ratios["latin"], 0)
        self.assertAlmostEqual(ratios["devanagari"] + ratios["latin"], 1.0)


if __name__ == "__main__":
    unittest.main()
