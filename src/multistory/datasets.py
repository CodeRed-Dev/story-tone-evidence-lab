"""Load and summarize the Narrative-Discourse release."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .narrative import arc_distribution, turning_point_summary
from .schema import STORY_ARCS, TURNING_POINTS
from .text import jensen_shannon_distance


@dataclass
class ReferenceDataset:
    records: List[Dict[str, Any]]
    corpus_size: int
    corpus_source_counts: Dict[str, int]
    arc_label_count: int
    turning_point_label_count: int
    missing_narrative_ids: List[str]
    incomplete_annotation_ids: List[str]


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_name(value: Any) -> str:
    lowered = str(value).strip().casefold()
    if lowered in {"gpt", "gpt-4", "ai", "machine"}:
        return "gpt"
    if lowered in {"human", "humans"}:
        return "human"
    return lowered or "unknown"


def load_narrative_reference(data_dir: Path) -> ReferenceDataset:
    data_dir = Path(data_dir)
    narratives = load_json(data_dir / "narratives.json")
    arcs = load_json(data_dir / "ground_truth_arc.json")
    turning_points = load_json(data_dir / "ground_truth_tp.json")
    if not all(isinstance(item, Mapping) for item in (narratives, arcs, turning_points)):
        raise ValueError("reference JSON files must contain top-level objects")

    source_counts = Counter(
        _source_name(value.get("source"))
        for value in narratives.values()
        if isinstance(value, Mapping)
    )
    records: List[Dict[str, Any]] = []
    missing: List[str] = []
    incomplete: List[str] = []
    for story_id in sorted(set(arcs) | set(turning_points)):
        if story_id not in narratives:
            missing.append(story_id)
            continue
        if story_id not in arcs or story_id not in turning_points:
            incomplete.append(story_id)
            continue
        narrative = narratives[story_id]
        sentences = narrative.get("synopsis") if isinstance(narrative, Mapping) else None
        arc = arcs[story_id]
        tps = turning_points[story_id]
        if not isinstance(sentences, list) or not sentences or arc not in STORY_ARCS:
            incomplete.append(story_id)
            continue
        if not isinstance(tps, Mapping) or not all(key in tps for key in TURNING_POINTS):
            incomplete.append(story_id)
            continue
        try:
            numeric_tps = {key: float(tps[key]) for key in TURNING_POINTS}
        except (TypeError, ValueError):
            incomplete.append(story_id)
            continue
        records.append(
            {
                "story_id": story_id,
                "source": _source_name(narrative.get("source")),
                "title": str(narrative.get("title", "")),
                "sentences": [str(sentence) for sentence in sentences],
                "story": " ".join(str(sentence) for sentence in sentences),
                "declared_arc": arc,
                "turning_points": numeric_tps,
            }
        )
    return ReferenceDataset(
        records=records,
        corpus_size=len(narratives),
        corpus_source_counts=dict(sorted(source_counts.items())),
        arc_label_count=len(arcs),
        turning_point_label_count=len(turning_points),
        missing_narrative_ids=missing,
        incomplete_annotation_ids=incomplete,
    )


def _source_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_count": len(records),
        "arc_distribution": arc_distribution(record["declared_arc"] for record in records),
        "turning_points": turning_point_summary(records),
    }


def summarize_reference(dataset: ReferenceDataset) -> Dict[str, Any]:
    source_names = sorted(set(record["source"] for record in dataset.records))
    sources = {
        source: _source_summary(
            [record for record in dataset.records if record["source"] == source]
        )
        for source in source_names
    }
    comparison: Dict[str, float] = {}
    if "human" in sources and "gpt" in sources:
        comparison["arc_jensen_shannon_distance"] = jensen_shannon_distance(
            sources["human"]["arc_distribution"]["probabilities"],
            sources["gpt"]["arc_distribution"]["probabilities"],
        )
        human_tp = sources["human"]["turning_points"]["mean_normalized_position"]
        gpt_tp = sources["gpt"]["turning_points"]["mean_normalized_position"]
        comparison["gpt_minus_human_tp_position"] = {
            key: gpt_tp[key] - human_tp[key] for key in TURNING_POINTS
        }
    return {
        "result_tier": "direct_local_analysis_of_official_released_annotations",
        "dataset": {
            "corpus_size": dataset.corpus_size,
            "corpus_source_counts": dataset.corpus_source_counts,
            "arc_label_count": dataset.arc_label_count,
            "turning_point_label_count": dataset.turning_point_label_count,
            "joined_annotation_count": len(dataset.records),
            "missing_narrative_count": len(dataset.missing_narrative_ids),
            "incomplete_annotation_count": len(dataset.incomplete_annotation_ids),
        },
        "sources": sources,
        "human_vs_gpt": comparison,
    }
