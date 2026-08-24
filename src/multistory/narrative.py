"""Macro, meso, and micro narrative measurements."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .diversity import categorical_entropy, normalized_entropy
from .schema import STORY_ARCS, TURNING_POINTS
from .text import interpolate, jensen_shannon_distance, mean, population_std, split_sentences, tokenize


# Transparent offline proxy values on a 0..1 scale. This is not the NRC VAD lexicon.
_PROXY_VAD: Dict[str, Tuple[float, float]] = {
    "afraid": (0.18, 0.82), "anger": (0.12, 0.88), "angry": (0.10, 0.90),
    "anxious": (0.22, 0.79), "ashamed": (0.15, 0.62), "calm": (0.72, 0.20),
    "celebrate": (0.91, 0.74), "confident": (0.85, 0.61), "cried": (0.16, 0.68),
    "danger": (0.13, 0.86), "dead": (0.05, 0.50), "defeat": (0.10, 0.64),
    "delight": (0.94, 0.70), "despair": (0.05, 0.72), "dream": (0.78, 0.42),
    "fear": (0.10, 0.90), "fought": (0.25, 0.85), "friend": (0.86, 0.42),
    "grief": (0.08, 0.65), "happy": (0.95, 0.72), "hate": (0.06, 0.84),
    "hope": (0.88, 0.58), "joy": (0.98, 0.80), "kind": (0.90, 0.35),
    "laughed": (0.93, 0.76), "lonely": (0.13, 0.40), "lost": (0.18, 0.55),
    "love": (0.98, 0.72), "peace": (0.93, 0.18), "proud": (0.91, 0.68),
    "rage": (0.04, 0.96), "regret": (0.14, 0.48), "relief": (0.84, 0.38),
    "sad": (0.08, 0.45), "safe": (0.88, 0.20), "smiled": (0.91, 0.61),
    "sorrow": (0.07, 0.52), "success": (0.94, 0.72), "terrified": (0.04, 0.98),
    "threat": (0.11, 0.89), "triumph": (0.96, 0.84), "trust": (0.86, 0.33),
    "victory": (0.97, 0.87), "violent": (0.05, 0.95), "warm": (0.82, 0.38),
    "worried": (0.20, 0.72), "wound": (0.10, 0.78), "wonder": (0.86, 0.62),
}

_TENSION_WORDS = {
    "afraid", "anger", "angry", "anxious", "ashamed", "cried", "danger", "dead",
    "defeat", "despair", "fear", "fought", "grief", "hate", "lost", "rage",
    "regret", "sad", "sorrow", "terrified", "threat", "violent", "worried", "wound",
}
_RESOLUTION_WORDS = {
    "calm", "celebrate", "confident", "delight", "dream", "friend", "happy", "hope",
    "joy", "kind", "laughed", "love", "peace", "proud", "relief", "safe", "smiled",
    "success", "triumph", "trust", "victory", "warm", "wonder",
}
_AGENCY_WORDS = {
    "chose", "decided", "refused", "risked", "searched", "built", "escaped",
    "returned", "saved", "fought", "promised", "confessed", "challenged",
}


def load_vad_lexicon(path: Optional[Path] = None) -> Dict[str, Tuple[float, float]]:
    """Load word/valence/arousal TSV or return the documented proxy lexicon."""

    if path is None:
        return dict(_PROXY_VAD)
    lexicon: Dict[str, Tuple[float, float]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"word", "valence", "arousal"}
        if not reader.fieldnames or not required.issubset(
            {name.casefold() for name in reader.fieldnames}
        ):
            raise ValueError("VAD TSV needs word, valence, and arousal columns")
        columns = {name.casefold(): name for name in reader.fieldnames}
        for row in reader:
            word = row[columns["word"]].strip().casefold()
            if word:
                lexicon[word] = (
                    float(row[columns["valence"]]),
                    float(row[columns["arousal"]]),
                )
    if not lexicon:
        raise ValueError("VAD lexicon is empty")
    return lexicon


def arc_distribution(labels: Iterable[str]) -> Dict[str, Any]:
    materialized = [label for label in labels if label in STORY_ARCS]
    counts = Counter(materialized)
    total = len(materialized)
    probabilities = {
        arc: counts.get(arc, 0) / total if total else 0.0 for arc in STORY_ARCS
    }
    return {
        "sample_count": total,
        "counts": {arc: counts.get(arc, 0) for arc in STORY_ARCS},
        "probabilities": probabilities,
        "entropy": categorical_entropy(materialized),
        "normalized_entropy": normalized_entropy(materialized, len(STORY_ARCS)),
    }


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def turning_point_summary(records: Iterable[Any]) -> Dict[str, Any]:
    positions: Dict[str, List[float]] = {key: [] for key in TURNING_POINTS}
    violation_count = 0
    story_count = 0
    for record in records:
        story = str(_field(record, "story"))
        sentence_count = len(split_sentences(story))
        turning_points = _field(record, "turning_points")
        if not sentence_count or not all(key in turning_points for key in TURNING_POINTS):
            continue
        normalized = [float(turning_points[key]) / sentence_count for key in TURNING_POINTS]
        for key, value in zip(TURNING_POINTS, normalized):
            positions[key].append(value)
        if any(right < left for left, right in zip(normalized, normalized[1:])):
            violation_count += 1
        story_count += 1
    return {
        "sample_count": story_count,
        "mean_normalized_position": {key: mean(values) for key, values in positions.items()},
        "std_normalized_position": {
            key: population_std(values) for key, values in positions.items()
        },
        "ordering_violation_count": violation_count,
        "ordering_violation_rate": violation_count / story_count if story_count else 0.0,
    }


def affective_curve(
    text: str,
    lexicon: Mapping[str, Tuple[float, float]],
    bins: int = 20,
) -> Dict[str, Any]:
    sentence_scores: List[Tuple[float, float]] = []
    recognized = 0
    for sentence in split_sentences(text):
        matches = [lexicon[token] for token in tokenize(sentence) if token in lexicon]
        recognized += len(matches)
        if matches:
            sentence_scores.append(
                (mean(pair[0] for pair in matches), mean(pair[1] for pair in matches))
            )
        else:
            sentence_scores.append((0.5, 0.5))
    return {
        "bins": bins,
        "sentence_count": len(sentence_scores),
        "recognized_token_count": recognized,
        "valence": interpolate([pair[0] for pair in sentence_scores], bins),
        "arousal": interpolate([pair[1] for pair in sentence_scores], bins),
    }


def discourse_feature_metrics(
    text: str,
    lexicon: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """Inspectable discourse proxies for recent narrative-parameter findings.

    The metrics deliberately use transparent lexical proxies. They are suitable for
    comparing cells and catching prompt/annotation failures, not for replacing human
    literary judgment.
    """

    tokens = tokenize(text)
    token_count = len(tokens)
    sentences = split_sentences(text)
    sentence_count = len(sentences)
    tension_hits = sum(1 for token in tokens if token in _TENSION_WORDS)
    resolution_hits = sum(1 for token in tokens if token in _RESOLUTION_WORDS)
    agency_hits = sum(1 for token in tokens if token in _AGENCY_WORDS)
    vad_matches = [lexicon[token] for token in tokens if token in lexicon]
    valences = [pair[0] for pair in vad_matches]
    arousals = [pair[1] for pair in vad_matches]
    valence_range = max(valences) - min(valences) if valences else 0.0
    arousal_range = max(arousals) - min(arousals) if arousals else 0.0
    positive_rate = (
        sum(1 for value in valences if value >= 0.75) / len(valences) if valences else 0.0
    )
    negative_rate = (
        sum(1 for value in valences if value <= 0.25) / len(valences) if valences else 0.0
    )
    suspense_proxy = (
        (tension_hits + agency_hits) / token_count if token_count else 0.0
    )
    flattening_proxy = 0.0
    if vad_matches:
        narrow_affect = 1.0 - min(1.0, (valence_range + arousal_range) / 2.0)
        positivity_bias = max(0.0, positive_rate - negative_rate)
        low_tension = 1.0 - min(1.0, suspense_proxy * 20.0)
        flattening_proxy = mean((narrow_affect, positivity_bias, low_tension))
    return {
        "sentence_count": sentence_count,
        "token_count": token_count,
        "tension_marker_count": tension_hits,
        "resolution_marker_count": resolution_hits,
        "agency_marker_count": agency_hits,
        "tension_markers_per_100_tokens": 100 * tension_hits / token_count if token_count else 0.0,
        "agency_markers_per_100_tokens": 100 * agency_hits / token_count if token_count else 0.0,
        "suspense_proxy": suspense_proxy,
        "valence_range": valence_range,
        "arousal_range": arousal_range,
        "positive_vad_rate": positive_rate,
        "negative_vad_rate": negative_rate,
        "flattening_proxy": flattening_proxy,
    }


def plot_signature(record: Any) -> str:
    """Compact signature for Echoes-style repeated plot-combination analysis."""

    story = str(_field(record, "story"))
    parameters = _field(record, "narrative_parameters") if hasattr(record, "narrative_parameters") or (
        isinstance(record, Mapping) and "narrative_parameters" in record
    ) else {}
    tokens = set(tokenize(story))
    tension = "tension" if tokens & _TENSION_WORDS else "low_tension"
    agency = "agency" if tokens & _AGENCY_WORDS else "low_agency"
    resolution = "resolution" if tokens & _RESOLUTION_WORDS else "low_resolution"
    arc = str(_field(record, "declared_arc")) if (
        hasattr(record, "declared_arc") or (isinstance(record, Mapping) and "declared_arc" in record)
    ) else "unknown_arc"
    plot = str(parameters.get("plot_complexity", "unknown_plot")) if isinstance(parameters, Mapping) else "unknown_plot"
    time = str(parameters.get("temporal_structure", "unknown_time")) if isinstance(parameters, Mapping) else "unknown_time"
    return "|".join((arc, plot, time, tension, agency, resolution))


def plot_diversity(records: Sequence[Any]) -> Dict[str, Any]:
    signatures = [plot_signature(record) for record in records]
    counts = Counter(signatures)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "sample_count": len(signatures),
        "unique_signature_count": len(counts),
        "duplicate_signature_count": duplicate_count,
        "duplicate_signature_rate": duplicate_count / len(signatures) if signatures else 0.0,
        "signature_entropy": categorical_entropy(signatures),
        "normalized_signature_entropy": normalized_entropy(signatures, len(counts) or 1),
        "top_signatures": [
            {"signature": signature, "count": count}
            for signature, count in counts.most_common(10)
        ],
    }


def mean_affective_curve(curves: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    if not curves:
        return {"valence": [], "arousal": []}
    bins = min(len(curve["valence"]) for curve in curves)
    return {
        "valence": [mean(curve["valence"][index] for curve in curves) for index in range(bins)],
        "arousal": [mean(curve["arousal"][index] for curve in curves) for index in range(bins)],
    }


def compare_to_human(
    condition_summary: Mapping[str, Any], human_summary: Mapping[str, Any]
) -> Dict[str, float]:
    condition_arcs = condition_summary["arc_distribution"]["probabilities"]
    human_arcs = human_summary["arc_distribution"]["probabilities"]
    condition_tp = condition_summary["turning_points"]["mean_normalized_position"]
    human_tp = human_summary["turning_points"]["mean_normalized_position"]
    return {
        "arc_jensen_shannon_distance": jensen_shannon_distance(condition_arcs, human_arcs),
        "turning_point_mean_absolute_distance": mean(
            abs(float(condition_tp[key]) - float(human_tp[key])) for key in TURNING_POINTS
        ),
    }
