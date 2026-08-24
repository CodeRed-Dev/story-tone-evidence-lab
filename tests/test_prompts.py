from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.prompts import build_requests, render_prompt, research_narrative_parameters
from multistory.schema import StoryBrief


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.brief = StoryBrief("fixed", "Fixed Title", "mystery", "Fixed setting text.", 123)
        self.requests = build_requests([self.brief])

    def test_balanced_core_design(self):
        counts = {
            condition: sum(request.condition == condition for request in self.requests)
            for condition in ("monolingual", "multicultural", "multilingual")
        }
        self.assertEqual(counts, {"monolingual": 3, "multicultural": 3, "multilingual": 3})

    def test_brief_is_held_fixed_and_story_contract_is_present(self):
        for request in self.requests:
            prompt = render_prompt(request)
            self.assertIn("Fixed Title", prompt)
            self.assertIn("Fixed setting text.", prompt)
            self.assertIn("123", prompt)
            self.assertIn("Return only the story prose", prompt)
            self.assertIn("no JSON", prompt)

    def test_prompt_ids_are_unique(self):
        identifiers = [request.prompt_id for request in self.requests]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_multilingual_prompts_use_expected_scripts(self):
        by_code = {
            request.profile.code: render_prompt(request)
            for request in self.requests
            if request.condition == "multilingual"
        }
        self.assertIn("你是一位", by_code["zh"])
        self.assertIn("あなたは", by_code["ja"])
        self.assertTrue(by_code["en"].startswith("You are"))

    def test_multicultural_prompt_language_stays_english(self):
        for request in self.requests:
            if request.condition == "multicultural":
                self.assertEqual(request.prompt_language, "English")
                self.assertTrue(render_prompt(request).startswith("Assume you are"))

    def test_narrative_parameters_are_explicit_prompt_preconditions(self):
        request = build_requests(
            [self.brief],
            conditions=("monolingual",),
            languages=("en",),
            narrative_parameters=research_narrative_parameters()[1:2],
        )[0]
        prompt = render_prompt(request)
        self.assertIn("Narrative parameter cell: literary_high_suspense", prompt)
        self.assertIn("Style register: literary", prompt)
        self.assertIn("Content/style separation precondition", prompt)
        self.assertIn("Discourse-structure precondition", prompt)

    def test_research_parameter_set_is_factorial_axis(self):
        requests = build_requests(
            [self.brief],
            conditions=("monolingual",),
            languages=("en",),
            narrative_parameters=research_narrative_parameters(),
        )
        self.assertEqual(len(requests), 12)
        self.assertEqual(
            {request.narrative_parameters.parameter_id for request in requests},
            {
                "neutral",
                "literary_high_suspense",
                "common_flattening_probe",
                "time_jump_plot_diversity",
            },
        )


if __name__ == "__main__":
    unittest.main()
