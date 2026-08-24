import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multistory.schema import (
    GenerationRequest,
    LanguageProfile,
    NarrativeParameters,
    StoryBrief,
    StoryRecord,
)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        brief = StoryBrief("brief", "Title", "drama", "A room.", 100)
        profile = LanguageProfile("en", "English", "Emily", "US", "NY", "pizza")
        self.request = GenerationRequest("p1", brief, "multilingual", profile, 1, 42)

    def response(self):
        return {
            "story": "One event begins. A choice follows. Trouble arrives. Hope fails. The end changes everything.",
            "story_arc": "Man in Hole",
            "turning_points": {"tp1": 1, "tp2": 2, "tp3": 3, "tp4": 4, "tp5": 5},
        }

    def test_parses_direct_json(self):
        record = StoryRecord.from_model_response(json.dumps(self.response()), self.request)
        self.assertEqual(record.declared_arc, "Man in Hole")
        self.assertEqual(record.turning_points["tp5"], 5.0)
        self.assertEqual(record.warnings, [])

    def test_parses_fenced_json(self):
        text = "```json\n%s\n```" % json.dumps(self.response())
        self.assertEqual(StoryRecord.from_model_response(text, self.request).brief_id, "brief")

    def test_rejects_missing_story(self):
        value = self.response()
        value.pop("story")
        with self.assertRaisesRegex(ValueError, "story"):
            StoryRecord.from_model_response(json.dumps(value), self.request)

    def test_warns_for_turning_point_order(self):
        value = self.response()
        value["turning_points"]["tp4"] = 2
        record = StoryRecord.from_model_response(json.dumps(value), self.request)
        self.assertTrue(any("not monotonically" in warning for warning in record.warnings))

    def test_preserves_narrative_parameters_in_record(self):
        request = GenerationRequest(
            "p2",
            self.request.brief,
            "multilingual",
            self.request.profile,
            1,
            43,
            NarrativeParameters(
                parameter_id="literary_high_suspense",
                style_register="literary",
                plot_complexity="branching",
                suspense_level="high",
                affective_range="high_contrast",
                temporal_structure="retrospective",
            ),
        )
        record = StoryRecord.from_model_response(json.dumps(self.response()), request)
        self.assertEqual(record.narrative_parameters["parameter_id"], "literary_high_suspense")
        self.assertEqual(record.narrative_parameters["style_register"], "literary")

    def test_rejects_unknown_narrative_parameter_value(self):
        with self.assertRaisesRegex(ValueError, "style_register"):
            NarrativeParameters(style_register="unsupported")


if __name__ == "__main__":
    unittest.main()
