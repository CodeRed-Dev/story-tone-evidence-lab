"""Generation providers. The default implementation talks only to local Ollama."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Mapping, Protocol

from .schema import STORY_ARCS, TURNING_POINTS, GenerationRequest
from .text import split_sentences, tokenize


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest, prompt: str) -> str:
        ...


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        timeout_seconds: int = 600,
        max_tokens: int = 800,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Ollama base_url must be a local HTTP address")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def _call(self, prompt: str, output_format: object, seed: int, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": seed,
                "num_predict": max_tokens,
            },
        }
        if output_format is not None:
            payload["format"] = output_format
        http_request = urllib.request.Request(
            self.base_url + "/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError("Ollama HTTP %s: %s" % (exc.code, body[:500])) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError("Ollama request failed: %s" % exc) from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("Ollama returned invalid response JSON") from exc
        if not isinstance(result, Mapping) or not isinstance(result.get("response"), str):
            raise ProviderError("Ollama response is missing the 'response' text")
        if result.get("error"):
            raise ProviderError("Ollama error: %s" % result["error"])
        return result["response"]

    def generate_text(self, prompt: str, seed: int, max_tokens: int = 300) -> str:
        """Generate plain text for non-story experiments such as tone analysis."""

        text = self._call(prompt, None, seed, max_tokens).strip()
        if not text:
            raise ProviderError("generated response is empty")
        return text

    def generate(self, request: GenerationRequest, prompt: str) -> str:
        """Generate prose, then annotate that immutable prose in a separate pass."""

        story = self._call(prompt, None, request.seed, self.max_tokens).strip()
        if len(tokenize(story)) < 40:
            raise ProviderError("generated story is shorter than 40 words")
        sentences = split_sentences(story)
        if not sentences:
            raise ProviderError("generated story contains no sentences")
        numbered = "\n".join(
            "%d. %s" % (index, sentence) for index, sentence in enumerate(sentences, 1)
        )
        annotation_prompt = """Analyze the numbered story below. Choose exactly one of the seven story arcs and identify five chronological turning points using only sentence numbers from 1 through {count}.

Story arcs: Rags to Riches; Riches to Rags; Man in Hole; Double Man in Hole; Icarus; Cinderella; Oedipus.
Turning points: tp1 Opportunity; tp2 Change of Plans; tp3 Point of No Return; tp4 Major Setback; tp5 Climax.
Return only the requested JSON annotation. Do not rewrite the story.

{story}""".format(count=len(sentences), story=numbered)
        annotation_schema = {
            "type": "object",
            "properties": {
                "story_arc": {"type": "string", "enum": list(STORY_ARCS)},
                "turning_points": {
                    "type": "object",
                    "properties": {
                        key: {"type": "integer", "minimum": 1, "maximum": len(sentences)}
                        for key in TURNING_POINTS
                    },
                    "required": list(TURNING_POINTS),
                    "additionalProperties": False,
                },
            },
            "required": ["story_arc", "turning_points"],
            "additionalProperties": False,
        }
        annotation_text = self._call(
            annotation_prompt,
            annotation_schema,
            request.seed + 1_000_000,
            180,
        )
        try:
            annotation = json.loads(annotation_text)
        except json.JSONDecodeError as exc:
            raise ProviderError("annotation pass returned invalid JSON") from exc
        if not isinstance(annotation, Mapping):
            raise ProviderError("annotation pass did not return an object")
        combined = {
            "story": story,
            "story_arc": annotation.get("story_arc"),
            "turning_points": annotation.get("turning_points"),
        }
        return json.dumps(combined, ensure_ascii=False)


class FixtureProvider:
    """Deterministic provider used only by tests and explicit demo runs."""

    name = "fixture"
    model = "fixture-records"

    def __init__(self, records_by_prompt_id: Dict[str, str]) -> None:
        self.records_by_prompt_id = dict(records_by_prompt_id)

    def generate(self, request: GenerationRequest, prompt: str) -> str:
        try:
            return self.records_by_prompt_id[request.prompt_id]
        except KeyError as exc:
            raise ProviderError("no fixture for %s" % request.prompt_id) from exc
