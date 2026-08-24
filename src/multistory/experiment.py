"""End-to-end baseline reproduction, generation, and analysis."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .datasets import load_narrative_reference, summarize_reference
from .diversity import corpus_diversity
from .narrative import (
    affective_curve,
    arc_distribution,
    compare_to_human,
    discourse_feature_metrics,
    load_vad_lexicon,
    mean_affective_curve,
    plot_diversity,
    turning_point_summary,
)
from .prompts import build_requests, render_prompt
from .providers import Provider
from .reporting import write_json
from .schema import RunFailure, StoryBrief, StoryRecord
from .schema import NarrativeParameters
from .text import mean


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_baseline(data_dir: Path) -> Dict[str, Any]:
    result = summarize_reference(load_narrative_reference(Path(data_dir)))
    result["generated_at"] = _timestamp()
    return result


def run_generation(
    provider: Provider,
    briefs: Iterable[StoryBrief],
    output_path: Path,
    conditions: Sequence[str],
    languages: Optional[Sequence[str]] = None,
    replicates: int = 1,
    base_seed: int = 1729,
    narrative_parameters: Optional[Sequence[NarrativeParameters]] = None,
) -> Dict[str, Any]:
    requests = build_requests(
        briefs,
        conditions=conditions,
        languages=languages,
        replicates=replicates,
        base_seed=base_seed,
        narrative_parameters=narrative_parameters,
    )
    run: Dict[str, Any] = {
        "run_type": "exploratory_local_generation",
        "result_tier": "fresh_single_model_local_experiment",
        "started_at": _timestamp(),
        "completed_at": None,
        "provider": provider.name,
        "model": provider.model,
        "request_count": len(requests),
        "successful_count": 0,
        "failure_count": 0,
        "conditions": list(conditions),
        "languages": list(languages or ("en", "zh", "ja")),
        "replicates": replicates,
        "base_seed": base_seed,
        "narrative_parameter_ids": [
            request.narrative_parameters.parameter_id for request in requests
        ],
        "attempts": [],
    }
    write_json(output_path, run)
    for request in requests:
        prompt = render_prompt(request)
        started = time.perf_counter()
        raw_response = ""
        attempt: Dict[str, Any] = {
            "request": request.to_dict(),
            "prompt": prompt,
            "status": "failure",
        }
        try:
            raw_response = provider.generate(request, prompt)
            record = StoryRecord.from_model_response(
                raw_response,
                request,
                provider=provider.name,
                model=provider.model,
            )
            attempt.update(
                {
                    "status": "success",
                    "raw_response": raw_response,
                    "record": record.to_dict(),
                }
            )
            run["successful_count"] += 1
        except Exception as exc:  # Preserve a partial run after provider or parser errors.
            failure = RunFailure(
                prompt_id=request.prompt_id,
                error_type=type(exc).__name__,
                message=str(exc),
                raw_response=raw_response,
            )
            attempt.update({"raw_response": raw_response, "failure": failure.to_dict()})
            run["failure_count"] += 1
        attempt["elapsed_seconds"] = time.perf_counter() - started
        run["attempts"].append(attempt)
        write_json(output_path, run)
    run["completed_at"] = _timestamp()
    write_json(output_path, run)
    return run


def _group_summary(records: Sequence[StoryRecord], lexicon: Mapping[str, Any]) -> Dict[str, Any]:
    curves = [affective_curve(record.story, lexicon) for record in records]
    discourse_metrics = [discourse_feature_metrics(record.story, lexicon) for record in records]
    discourse_keys = (
        "suspense_proxy",
        "valence_range",
        "arousal_range",
        "positive_vad_rate",
        "negative_vad_rate",
        "flattening_proxy",
        "tension_markers_per_100_tokens",
        "agency_markers_per_100_tokens",
    )
    return {
        "sample_count": len(records),
        "arc_distribution": arc_distribution(record.declared_arc for record in records),
        "turning_points": turning_point_summary(records),
        "lexical_diversity": corpus_diversity([record.story for record in records]),
        "plot_diversity": plot_diversity(records),
        "discourse_features": {
            "mean_%s" % key: mean(metric[key] for metric in discourse_metrics)
            for key in discourse_keys
        },
        "mean_affective_curve": mean_affective_curve(curves),
        "affective_proxy_coverage": {
            "recognized_token_count": sum(curve["recognized_token_count"] for curve in curves),
            "sentence_count": sum(curve["sentence_count"] for curve in curves),
        },
        "warning_count": sum(len(record.warnings) for record in records),
    }


def _delta(
    conditions: Mapping[str, Mapping[str, Any]],
    treatment: str,
    metric_path: Sequence[str],
) -> Optional[float]:
    if treatment not in conditions or "monolingual" not in conditions:
        return None
    left: Any = conditions[treatment]
    right: Any = conditions["monolingual"]
    for key in metric_path:
        left, right = left[key], right[key]
    return float(left) - float(right)


def analyze_generation(
    raw_run: Mapping[str, Any],
    baseline: Mapping[str, Any],
    vad_path: Optional[Path] = None,
) -> Dict[str, Any]:
    records = [
        StoryRecord.from_dict(attempt["record"])
        for attempt in raw_run.get("attempts", [])
        if attempt.get("status") == "success"
    ]
    failures = [
        attempt["failure"]
        for attempt in raw_run.get("attempts", [])
        if attempt.get("status") == "failure" and "failure" in attempt
    ]
    lexicon = load_vad_lexicon(vad_path)
    by_condition: Dict[str, List[StoryRecord]] = defaultdict(list)
    by_language: Dict[str, List[StoryRecord]] = defaultdict(list)
    by_parameter: Dict[str, List[StoryRecord]] = defaultdict(list)
    by_condition_parameter: Dict[str, List[StoryRecord]] = defaultdict(list)
    for record in records:
        by_condition[record.condition].append(record)
        by_language["%s/%s" % (record.condition, record.language_code)].append(record)
        parameter_id = str(record.narrative_parameters.get("parameter_id", "neutral"))
        by_parameter[parameter_id].append(record)
        by_condition_parameter["%s/%s" % (record.condition, parameter_id)].append(record)

    condition_summaries = {
        condition: _group_summary(group, lexicon)
        for condition, group in sorted(by_condition.items())
    }
    human = baseline.get("sources", {}).get("human")
    if human:
        for summary in condition_summaries.values():
            summary["distance_from_human"] = compare_to_human(summary, human)

    language_summaries = {
        language: _group_summary(group, lexicon)
        for language, group in sorted(by_language.items())
    }
    parameter_summaries = {
        parameter: _group_summary(group, lexicon)
        for parameter, group in sorted(by_parameter.items())
    }
    condition_parameter_summaries = {
        key: _group_summary(group, lexicon)
        for key, group in sorted(by_condition_parameter.items())
    }
    arc_delta = _delta(
        condition_summaries, "multilingual", ("arc_distribution", "normalized_entropy")
    )
    lexical_delta = _delta(
        condition_summaries,
        "multilingual",
        ("lexical_diversity", "pairwise_token_jaccard_distance"),
    )
    multicultural_arc_delta = _delta(
        condition_summaries, "multicultural", ("arc_distribution", "normalized_entropy")
    )
    if arc_delta is None:
        h1 = "Unavailable: monolingual and multilingual successful samples are both required."
    else:
        h1 = "Multilingual minus monolingual normalized arc entropy = %.4f." % arc_delta
    if lexical_delta is None:
        h2 = "Unavailable: monolingual and multilingual successful samples are both required."
    else:
        h2 = "Multilingual minus monolingual token Jaccard diversity = %.4f." % lexical_delta
    if arc_delta is None or multicultural_arc_delta is None:
        h3 = "Unavailable: all three conditions need successful samples."
    else:
        h3 = (
            "Arc-entropy gain over monolingual: language shift %.4f versus persona-only %.4f."
            % (arc_delta, multicultural_arc_delta)
        )
    violation_count = sum(
        summary["turning_points"]["ordering_violation_count"]
        for summary in condition_summaries.values()
    )
    if violation_count:
        h4 = (
            "Turning-point timing is not interpreted: %d/%d annotations were non-monotonic."
            % (violation_count, len(records))
        )
    else:
        h4 = "Turning-point annotations were monotonic; inspect condition timing in the JSON results."
    if "monolingual" in condition_summaries and "multilingual" in condition_summaries:
        mono_human = condition_summaries["monolingual"].get("distance_from_human", {}).get(
            "arc_jensen_shannon_distance"
        )
        multi_human = condition_summaries["multilingual"].get("distance_from_human", {}).get(
            "arc_jensen_shannon_distance"
        )
        h5 = (
            "Multilingual minus monolingual human arc distance = %.4f; diversity and human-likeness are reported separately."
            % (float(multi_human) - float(mono_human))
        )
    else:
        h5 = "Unavailable: monolingual and multilingual summaries are both required."
    if len(parameter_summaries) < 2:
        h6 = "Unavailable: at least two narrative parameter cells are required to test parameter sensitivity."
    else:
        flattening_values = {
            key: value["discourse_features"]["mean_flattening_proxy"]
            for key, value in parameter_summaries.items()
        }
        lowest = min(flattening_values, key=flattening_values.get)
        highest = max(flattening_values, key=flattening_values.get)
        h6 = (
            "Narrative-parameter flattening spread = %.4f (%s %.4f vs %s %.4f)."
            % (
                flattening_values[highest] - flattening_values[lowest],
                highest,
                flattening_values[highest],
                lowest,
                flattening_values[lowest],
            )
        )
    if parameter_summaries:
        repeated = {
            key: value["plot_diversity"]["duplicate_signature_rate"]
            for key, value in parameter_summaries.items()
        }
        h7 = "Highest repeated plot-signature rate = %.4f in parameter cell `%s`." % (
            max(repeated.values()),
            max(repeated, key=repeated.get),
        )
    else:
        h7 = "Unavailable: no successful parameter summaries."

    return {
        "metadata": {
            "result_tier": raw_run.get("result_tier", "exploratory_local_experiment"),
            "analyzed_at": _timestamp(),
            "provider": raw_run.get("provider", "unknown"),
            "model": raw_run.get("model", "unknown"),
            "request_count": raw_run.get("request_count", len(records) + len(failures)),
            "successful_count": len(records),
            "failure_count": len(failures),
            "conditions_requested": raw_run.get("conditions", []),
            "languages_requested": raw_run.get("languages", []),
            "affective_method": "small built-in proxy lexicon" if vad_path is None else str(vad_path),
            "generation_method": "plain-prose generation followed by a separate constrained annotation pass",
            "parameter_precondition": (
                "Narrative parameters are tracked as an orthogonal factor so style, plot complexity, "
                "suspense, affective range, and temporal structure are not conflated with prompt language."
            ),
        },
        "conditions": condition_summaries,
        "condition_languages": language_summaries,
        "narrative_parameters": parameter_summaries,
        "condition_narrative_parameters": condition_parameter_summaries,
        "hypothesis_checks": {
            "H1": h1,
            "H2": h2,
            "H3": h3,
            "H4": h4,
            "H5": h5,
            "H6": h6,
            "H7": h7,
        },
        "failures": failures,
    }
