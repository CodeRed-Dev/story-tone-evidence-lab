import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.tone_experiment import analyze_tone_generation, run_tone_generation, tone_markdown


class FakeTextProvider:
    name = "fake"
    model = "fixed-responses"

    def __init__(self):
        self.calls = 0

    def generate_text(self, prompt, seed, max_tokens=300):
        self.calls += 1
        if "केवल हिंदी" in prompt and "कृपया" in prompt:
            return "निश्चित रूप से। कृपया आप इस उदाहरण पर विचार करें। धन्यवाद।"
        if "केवल हिंदी" in prompt:
            return "आप इस उदाहरण से उत्तर समझ सकते हैं।"
        if "please" in prompt.lower():
            return "Certainly. I hope this explanation helps."
        return "The answer follows from the example."


class ToneExperimentTests(unittest.TestCase):
    def test_factorial_run_analysis_and_resume(self):
        prompts = [
            {
                "id": "t_en_direct",
                "task": "t",
                "language": "en",
                "input_tone": "direct",
                "prompt": "Explain the example. Answer only in English.",
            },
            {
                "id": "t_en_deferential",
                "task": "t",
                "language": "en",
                "input_tone": "deferential",
                "prompt": "Could you please explain the example? Answer only in English.",
            },
            {
                "id": "t_hi_direct",
                "task": "t",
                "language": "hi",
                "input_tone": "direct",
                "prompt": "उदाहरण समझाओ। केवल हिंदी में उत्तर दो।",
            },
            {
                "id": "t_hi_deferential",
                "task": "t",
                "language": "hi",
                "input_tone": "deferential",
                "prompt": "कृपया उदाहरण समझाइए। केवल हिंदी में उत्तर दीजिए।",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_path = root / "prompts.json"
            raw_path = root / "raw.json"
            with prompt_path.open("w", encoding="utf-8") as handle:
                json.dump(prompts, handle, ensure_ascii=False)
            provider = FakeTextProvider()
            raw = run_tone_generation(provider, prompt_path, raw_path, base_seed=7)
            self.assertEqual(provider.calls, 4)
            resumed = run_tone_generation(provider, prompt_path, raw_path, base_seed=7)
            self.assertEqual(provider.calls, 4)

        result = analyze_tone_generation(resumed)
        self.assertEqual(result["metadata"]["successful_count"], 4)
        self.assertGreater(
            result["matched_contrasts"]["hindi_minus_english/direct"]
            ["politeness_strategy_coverage"]["mean_matched_difference"],
            0,
        )
        self.assertIn("cannot establish", tone_markdown(result))


if __name__ == "__main__":
    unittest.main()
