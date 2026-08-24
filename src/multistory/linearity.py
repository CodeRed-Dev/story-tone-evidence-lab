"""Deterministic chronology proxies for comparing human and AI narratives.

These lexical measurements are deliberately narrower than StoryScope's learned
304-feature analysis.  They make every counted cue inspectable and can therefore
serve as inexpensive local evidence, not as a complete definition of linearity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FORWARD_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("then", r"\bthen\b"),
    ("next", r"\bnext\b"),
    ("later", r"\blater\b"),
    ("afterwards", r"\bafterwards?\b"),
    ("eventually", r"\beventually\b"),
    ("finally", r"\bfinally\b"),
    ("subsequently", r"\bsubsequently\b"),
    ("soon", r"\bsoon\b"),
    ("after_that", r"\bafter that\b"),
    ("as_time_passed", r"\bas (?:time|the days|days|the years|years) pass(?:ed|es)?\b"),
    ("over_the_next", r"\bover the next\b"),
)

RETROSPECTIVE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("earlier", r"\bearlier\b"),
    ("previously", r"\bpreviously\b"),
    ("before", r"\bbefore\b"),
    ("formerly", r"\bformerly\b"),
    ("back_then", r"\bback then\b"),
    ("ago", r"\b(?:days?|weeks?|months?|years?|decades?) ago\b"),
    ("remember", r"\bremember(?:ed|s|ing)?\b"),
    ("recall", r"\brecall(?:ed|s|ing)?\b"),
    ("reminisce", r"\breminisc(?:e|ed|es|ing)\b"),
    ("memory", r"\bmemor(?:y|ies)\b"),
    ("flashback", r"\bflashbacks?\b"),
    ("had_once", r"\bhad (?:once|previously|formerly)\b"),
)

JUMP_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        "elapsed_time_jump",
        r"\b(?:days?|weeks?|months?|years?|decades?) (?:later|earlier|afterwards?|before)\b",
    ),
    ("next_period", r"\bthe next (?:day|morning|evening|week|month|year)\b"),
    ("following_period", r"\bthe following (?:day|morning|evening|week|month|year)\b"),
    ("time_passage", r"\b(?:time|years?|months?|weeks?|days?) (?:had )?passed\b"),
    ("meanwhile", r"\bmeanwhile\b"),
    ("flashback", r"\bflashbacks?\b"),
)

REPORT_METRICS: Tuple[str, ...] = (
    "retrospective_presence",
    "retrospective_per_100_sentences",
    "temporal_jump_presence",
    "temporal_jump_per_100_sentences",
    "forward_presence",
    "forward_per_100_sentences",
    "forward_share",
)


def _count_patterns(text: str, patterns: Sequence[Tuple[str, str]]) -> Tuple[int, Dict[str, int]]:
    details: Dict[str, int] = {}
    total = 0
    for name, pattern in patterns:
        count = len(re.findall(pattern, text, flags=re.IGNORECASE))
        if count:
            details[name] = count
            total += count
    return total, details


def chronology_metrics(text: str, sentence_count: Optional[int] = None) -> Dict[str, Any]:
    """Return transparent lexical chronology measurements for one narrative."""

    normalized = " ".join(str(text).split())
    if sentence_count is None:
        pieces = [part for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
        sentence_count = len(pieces)
    sentence_count = max(1, int(sentence_count))
    word_count = len(re.findall(r"\b[\w'-]+\b", normalized, flags=re.UNICODE))
    forward, forward_matches = _count_patterns(normalized, FORWARD_PATTERNS)
    retrospective, retrospective_matches = _count_patterns(normalized, RETROSPECTIVE_PATTERNS)
    jumps, jump_matches = _count_patterns(normalized, JUMP_PATTERNS)
    ordered_total = forward + retrospective
    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "forward_count": forward,
        "retrospective_count": retrospective,
        "temporal_jump_count": jumps,
        "forward_presence": int(forward > 0),
        "retrospective_presence": int(retrospective > 0),
        "temporal_jump_presence": int(jumps > 0),
        "forward_per_100_sentences": 100.0 * forward / sentence_count,
        "retrospective_per_100_sentences": 100.0 * retrospective / sentence_count,
        "temporal_jump_per_100_sentences": 100.0 * jumps / sentence_count,
        "forward_share": forward / ordered_total if ordered_total else None,
        "matches": {
            "forward": forward_matches,
            "retrospective": retrospective_matches,
            "temporal_jump": jump_matches,
        },
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return statistics.median(clean) if clean else None


def _cohens_d(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    left_var = statistics.variance(left)
    right_var = statistics.variance(right)
    pooled_numerator = (len(left) - 1) * left_var + (len(right) - 1) * right_var
    pooled_denominator = len(left) + len(right) - 2
    if pooled_denominator <= 0:
        return None
    pooled_sd = math.sqrt(pooled_numerator / pooled_denominator)
    if pooled_sd == 0:
        return None
    return (statistics.fmean(left) - statistics.fmean(right)) / pooled_sd


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_difference(
    human: Sequence[float],
    gpt: Sequence[float],
    seed: int,
    iterations: int = 2000,
) -> List[float]:
    generator = random.Random(seed)
    differences: List[float] = []
    for _ in range(iterations):
        human_sample = [human[generator.randrange(len(human))] for _ in human]
        gpt_sample = [gpt[generator.randrange(len(gpt))] for _ in gpt]
        differences.append(statistics.fmean(human_sample) - statistics.fmean(gpt_sample))
    differences.sort()
    return [_percentile(differences, 0.025), _percentile(differences, 0.975)]


def _numeric_values(records: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
    return [float(record[metric]) for record in records if record.get(metric) is not None]


def _source_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"story_count": len(records)}
    for metric in REPORT_METRICS:
        values = _numeric_values(records, metric)
        summary[metric] = {
            "mean": _mean(values),
            "median": _median(values),
            "measured_story_count": len(values),
        }
    summary["mean_sentence_count"] = _mean(record.get("sentence_count") for record in records)
    summary["mean_word_count"] = _mean(record.get("word_count") for record in records)
    return summary


def analyze_linearity_corpus(
    narratives_path: Path,
    bootstrap_iterations: int = 2000,
    seed: int = 260403136,
) -> Dict[str, Any]:
    """Analyze the balanced Narrative-Discourse human/GPT synopsis corpus."""

    path = Path(narratives_path)
    with path.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)
    if not isinstance(corpus, Mapping):
        raise ValueError("narratives file must contain a JSON object")

    records: List[Dict[str, Any]] = []
    for narrative_id, item in corpus.items():
        if not isinstance(item, Mapping):
            continue
        source = str(item.get("source", "Unknown"))
        synopsis = item.get("synopsis", [])
        if isinstance(synopsis, list):
            sentences = [str(sentence).strip() for sentence in synopsis if str(sentence).strip()]
            text = " ".join(sentences)
            sentence_count = len(sentences)
        else:
            text = str(synopsis)
            sentence_count = None
        metrics = chronology_metrics(text, sentence_count=sentence_count)
        records.append(
            {
                "narrative_id": str(narrative_id),
                "source": source,
                "title": str(item.get("title", "")),
                **metrics,
            }
        )

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    if "Human" not in by_source or "GPT" not in by_source:
        raise ValueError("corpus must contain both Human and GPT sources")

    summaries = {source: _source_summary(items) for source, items in sorted(by_source.items())}
    contrasts: Dict[str, Any] = {}
    for metric_index, metric in enumerate(REPORT_METRICS):
        human_values = _numeric_values(by_source["Human"], metric)
        gpt_values = _numeric_values(by_source["GPT"], metric)
        contrasts[metric] = {
            "human_minus_gpt": statistics.fmean(human_values) - statistics.fmean(gpt_values),
            "bootstrap_95_ci": _bootstrap_difference(
                human_values,
                gpt_values,
                seed=seed + metric_index,
                iterations=bootstrap_iterations,
            ),
            "cohens_d": _cohens_d(human_values, gpt_values),
        }

    discontinuity_metrics = (
        "retrospective_per_100_sentences",
        "temporal_jump_per_100_sentences",
    )
    aligns = all(contrasts[name]["human_minus_gpt"] > 0 for name in discontinuity_metrics)
    return {
        "metadata": {
            "analysis": "deterministic lexical chronology proxy",
            "input_path": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "story_count": len(records),
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
        },
        "definitions": {
            "retrospective": [name for name, _ in RETROSPECTIVE_PATTERNS],
            "temporal_jump": [name for name, _ in JUMP_PATTERNS],
            "forward": [name for name, _ in FORWARD_PATTERNS],
            "warning": (
                "Marker counts are lexical cues, not a complete narrative-structure judgment. "
                "They do not reproduce StoryScope's learned 304-feature pipeline."
            ),
        },
        "source_summaries": summaries,
        "contrasts": contrasts,
        "storyscope_directional_comparison": {
            "paper_claim": "Human fiction has more temporal complexity; AI fiction is more linear.",
            "local_discontinuity_proxies_align": aligns,
            "status": "directional comparison, not a reproduction",
        },
        "records": records,
    }


def _format(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def linearity_markdown(result: Mapping[str, Any]) -> str:
    """Render the compact, human-readable evidence report."""

    summaries = result["source_summaries"]
    contrasts = result["contrasts"]
    labels = {
        "retrospective_presence": "Stories with retrospective cues",
        "retrospective_per_100_sentences": "Retrospective cues / 100 sentences",
        "temporal_jump_presence": "Stories with time-jump cues",
        "temporal_jump_per_100_sentences": "Time-jump cues / 100 sentences",
        "forward_presence": "Stories with forward cues",
        "forward_per_100_sentences": "Forward cues / 100 sentences",
        "forward_share": "Forward share (when ordered cue exists)",
    }
    lines = [
        "# AI versus human narrative linearity evidence",
        "",
        "This report analyzes all %d local synopses using an inspectable lexical chronology proxy. "
        "It is a directional extension of StoryScope, not a reproduction of its 304-feature model."
        % result["metadata"]["story_count"],
        "",
        "| Measure | Human mean | GPT mean | Human - GPT | Bootstrap 95% CI | Cohen's d |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in REPORT_METRICS:
        contrast = contrasts[metric]
        interval = contrast["bootstrap_95_ci"]
        lines.append(
            "| %s | %s | %s | %s | [%s, %s] | %s |"
            % (
                labels[metric],
                _format(summaries["Human"][metric]["mean"]),
                _format(summaries["GPT"][metric]["mean"]),
                _format(contrast["human_minus_gpt"]),
                _format(interval[0]),
                _format(interval[1]),
                _format(contrast["cohens_d"]),
            )
        )
    comparison = result["storyscope_directional_comparison"]
    direction = "align" if comparison["local_discontinuity_proxies_align"] else "do not uniformly align"
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The two local discontinuity proxies **%s** with StoryScope's reported direction that human fiction is more temporally complex than AI fiction."
            % direction,
            "",
            "Presence measures are proportions from 0 to 1. The confidence intervals resample stories within each source. "
            "A positive human-minus-GPT discontinuity contrast supports the claimed direction; it does not prove that every human story is nonlinear or every AI story is linear.",
            "",
            "## Provenance",
            "",
            "- Input SHA-256: `%s`" % result["metadata"]["input_sha256"],
            "- Bootstrap iterations: %d" % result["metadata"]["bootstrap_iterations"],
            "- StoryScope: https://arxiv.org/abs/2604.03136",
        ]
    )
    return "\n".join(lines) + "\n"


def write_linearity_evidence(result: Mapping[str, Any], results_dir: Path) -> Tuple[Path, Path]:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "linearity_evidence.json"
    markdown_path = output_dir / "linearity_evidence.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    markdown_path.write_text(linearity_markdown(result), encoding="utf-8")
    return json_path, markdown_path
