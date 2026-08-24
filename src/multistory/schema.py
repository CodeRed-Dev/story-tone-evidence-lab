"""Validated experiment records shared by generation and analysis."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


STORY_ARCS: Tuple[str, ...] = (
    "Rags to Riches",
    "Riches to Rags",
    "Man in Hole",
    "Double Man in Hole",
    "Icarus",
    "Cinderella",
    "Oedipus",
)
TURNING_POINTS: Tuple[str, ...] = ("tp1", "tp2", "tp3", "tp4", "tp5")
CONDITIONS: Tuple[str, ...] = ("monolingual", "multicultural", "multilingual")
STYLE_REGISTERS: Tuple[str, ...] = ("neutral", "common_fiction", "literary", "oral_tradition")
PLOT_COMPLEXITIES: Tuple[str, ...] = ("single_track", "branching", "nested")
SUSPENSE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
AFFECTIVE_RANGES: Tuple[str, ...] = ("narrow_positive", "balanced", "high_contrast")
TEMPORAL_STRUCTURES: Tuple[str, ...] = ("linear", "retrospective", "time_jump")


@dataclass(frozen=True)
class NarrativeParameters:
    """Narrative controls held separately from language and cultural persona.

    These fields operationalize the storytelling literature as explicit experimental
    parameters: style is not allowed to stand in for content, plot diversity is
    inspectable, and discourse features such as suspense, affect range, and temporal
    structure are first-class prompt conditions.
    """

    parameter_id: str = "neutral"
    style_register: str = "neutral"
    plot_complexity: str = "single_track"
    suspense_level: str = "medium"
    affective_range: str = "balanced"
    temporal_structure: str = "linear"
    require_turning_points: bool = True
    preserve_content_elements: bool = True

    def __post_init__(self) -> None:
        if not self.parameter_id.strip():
            raise ValueError("parameter_id must not be empty")
        checks = (
            ("style_register", self.style_register, STYLE_REGISTERS),
            ("plot_complexity", self.plot_complexity, PLOT_COMPLEXITIES),
            ("suspense_level", self.suspense_level, SUSPENSE_LEVELS),
            ("affective_range", self.affective_range, AFFECTIVE_RANGES),
            ("temporal_structure", self.temporal_structure, TEMPORAL_STRUCTURES),
        )
        for name, value, allowed in checks:
            if value not in allowed:
                raise ValueError("%s must be one of: %s" % (name, ", ".join(allowed)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NarrativeParameters":
        return cls(
            parameter_id=str(value.get("parameter_id", "neutral")),
            style_register=str(value.get("style_register", "neutral")),
            plot_complexity=str(value.get("plot_complexity", "single_track")),
            suspense_level=str(value.get("suspense_level", "medium")),
            affective_range=str(value.get("affective_range", "balanced")),
            temporal_structure=str(value.get("temporal_structure", "linear")),
            require_turning_points=bool(value.get("require_turning_points", True)),
            preserve_content_elements=bool(value.get("preserve_content_elements", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoryBrief:
    """The content held fixed while prompt condition varies."""

    brief_id: str
    title: str
    genre: str
    initial_setting: str
    target_words: int = 300

    def __post_init__(self) -> None:
        for name in ("brief_id", "title", "genre", "initial_setting"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s must not be empty" % name)
        if self.target_words <= 0:
            raise ValueError("target_words must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StoryBrief":
        return cls(
            brief_id=str(value["brief_id"]),
            title=str(value["title"]),
            genre=str(value["genre"]),
            initial_setting=str(value["initial_setting"]),
            target_words=int(value.get("target_words", 300)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LanguageProfile:
    """A prompt-language and culturally grounded persona configuration."""

    code: str
    language: str
    persona_name: str
    location: str
    birthplace: str
    favorite_food: str

    def __post_init__(self) -> None:
        if not self.code or not self.language:
            raise ValueError("language profile code and language are required")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    """One controlled model call."""

    prompt_id: str
    brief: StoryBrief
    condition: str
    profile: LanguageProfile
    replicate: int
    seed: int
    narrative_parameters: NarrativeParameters = field(default_factory=NarrativeParameters)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError("unknown condition: %s" % self.condition)
        if self.replicate < 1:
            raise ValueError("replicate must be one-based")

    @property
    def prompt_language(self) -> str:
        return self.profile.language if self.condition == "multilingual" else "English"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "brief": self.brief.to_dict(),
            "condition": self.condition,
            "profile": self.profile.to_dict(),
            "prompt_language": self.prompt_language,
            "replicate": self.replicate,
            "seed": self.seed,
            "narrative_parameters": self.narrative_parameters.to_dict(),
        }


def _extract_json_object(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    candidates: List[str] = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    first, last = stripped.find("{"), stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    error: Optional[Exception] = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, Mapping):
                return value
        except (json.JSONDecodeError, TypeError) as exc:
            error = exc
    raise ValueError("response does not contain a valid JSON object") from error


def _estimate_sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]
    return max(1, len(parts)) if text.strip() else 0


def validate_tp_positions(
    positions: Mapping[str, float], sentence_count: int
) -> List[str]:
    """Return non-fatal warnings for generated turning-point metadata."""

    warnings: List[str] = []
    ordered: List[float] = []
    for key in TURNING_POINTS:
        if key not in positions:
            warnings.append("missing %s" % key)
            continue
        value = positions[key]
        if value < 1 or (sentence_count and value > sentence_count):
            warnings.append("%s=%s is outside 1..%s" % (key, value, sentence_count))
        ordered.append(value)
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        warnings.append("turning points are not monotonically ordered")
    return warnings


@dataclass
class StoryRecord:
    """A parsed model generation and its self-reported narrative metadata."""

    prompt_id: str
    brief_id: str
    condition: str
    prompt_language: str
    language_code: str
    replicate: int
    seed: int
    narrative_parameters: Dict[str, Any]
    story: str
    declared_arc: str
    turning_points: Dict[str, float]
    provider: str = "unknown"
    model: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_model_response(
        cls,
        text: str,
        request: GenerationRequest,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> "StoryRecord":
        value = _extract_json_object(text)
        story_value = value.get("story")
        if isinstance(story_value, Sequence) and not isinstance(story_value, str):
            story = " ".join(str(item).strip() for item in story_value if str(item).strip())
        else:
            story = str(story_value or "").strip()
        if not story:
            raise ValueError("response field 'story' must not be empty")

        arc = str(value.get("story_arc") or value.get("declared_arc") or "").strip()
        if arc not in STORY_ARCS:
            raise ValueError("story_arc must be one of: %s" % ", ".join(STORY_ARCS))

        raw_tps = value.get("turning_points")
        if not isinstance(raw_tps, Mapping):
            raise ValueError("turning_points must be an object")
        positions: Dict[str, float] = {}
        for key in TURNING_POINTS:
            try:
                positions[key] = float(raw_tps[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("turning_points.%s must be numeric" % key) from exc

        warnings = validate_tp_positions(positions, _estimate_sentence_count(story))
        return cls(
            prompt_id=request.prompt_id,
            brief_id=request.brief.brief_id,
            condition=request.condition,
            prompt_language=request.prompt_language,
            language_code=request.profile.code,
            replicate=request.replicate,
            seed=request.seed,
            narrative_parameters=request.narrative_parameters.to_dict(),
            story=story,
            declared_arc=arc,
            turning_points=positions,
            provider=provider,
            model=model,
            warnings=warnings,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StoryRecord":
        return cls(
            prompt_id=str(value["prompt_id"]),
            brief_id=str(value["brief_id"]),
            condition=str(value["condition"]),
            prompt_language=str(value["prompt_language"]),
            language_code=str(value["language_code"]),
            replicate=int(value["replicate"]),
            seed=int(value["seed"]),
            narrative_parameters=dict(
                value.get("narrative_parameters", NarrativeParameters().to_dict())
            ),
            story=str(value["story"]),
            declared_arc=str(value["declared_arc"]),
            turning_points={key: float(number) for key, number in value["turning_points"].items()},
            provider=str(value.get("provider", "unknown")),
            model=str(value.get("model", "unknown")),
            warnings=list(value.get("warnings", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunFailure:
    prompt_id: str
    error_type: str
    message: str
    raw_response: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
