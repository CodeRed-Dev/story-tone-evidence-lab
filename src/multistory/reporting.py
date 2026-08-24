"""Stable JSON and human-readable Markdown result writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import STORY_ARCS, TURNING_POINTS


def write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return ("%.*f" % (digits, float(value))).rstrip("0").rstrip(".")
    return "—" if value is None else str(value)


def baseline_markdown(result: Mapping[str, Any]) -> str:
    dataset = result["dataset"]
    sources = result["sources"]
    lines = [
        "# Baseline reproduction",
        "",
        "This report is a direct local analysis of the official released narrative annotations.",
        "It is not a fresh generation run.",
        "",
        "## Dataset",
        "",
        "| Measure | Count |",
        "|---|---:|",
        "| Full narrative corpus | %s |" % dataset["corpus_size"],
        "| Story-arc labels | %s |" % dataset["arc_label_count"],
        "| Turning-point labels | %s |" % dataset["turning_point_label_count"],
        "| Joined annotated stories | %s |" % dataset["joined_annotation_count"],
        "| Missing narrative IDs | %s |" % dataset["missing_narrative_count"],
        "| Incomplete annotation IDs | %s |" % dataset["incomplete_annotation_count"],
        "",
        "## Story arcs",
        "",
        "| Arc | Human | GPT |",
        "|---|---:|---:|",
    ]
    human = sources.get("human", {}).get("arc_distribution", {})
    gpt = sources.get("gpt", {}).get("arc_distribution", {})
    human_probs = human.get("probabilities", {})
    gpt_probs = gpt.get("probabilities", {})
    for arc in STORY_ARCS:
        lines.append(
            "| %s | %.1f%% | %.1f%% |"
            % (arc, 100 * human_probs.get(arc, 0.0), 100 * gpt_probs.get(arc, 0.0))
        )
    lines.extend(
        [
            "| **Normalized entropy** | %s | %s |"
            % (_number(human.get("normalized_entropy")), _number(gpt.get("normalized_entropy"))),
            "",
            "## Mean normalized turning-point positions",
            "",
            "| Turning point | Human | GPT | GPT − Human |",
            "|---|---:|---:|---:|",
        ]
    )
    human_tp = sources.get("human", {}).get("turning_points", {}).get(
        "mean_normalized_position", {}
    )
    gpt_tp = sources.get("gpt", {}).get("turning_points", {}).get(
        "mean_normalized_position", {}
    )
    deltas = result.get("human_vs_gpt", {}).get("gpt_minus_human_tp_position", {})
    for key in TURNING_POINTS:
        lines.append(
            "| %s | %s | %s | %s |"
            % (key.upper(), _number(human_tp.get(key)), _number(gpt_tp.get(key)), _number(deltas.get(key)))
        )
    lines.extend(
        [
            "",
            "Arc Jensen–Shannon distance: **%s**."
            % _number(result.get("human_vs_gpt", {}).get("arc_jensen_shannon_distance")),
            "",
            "## Interpretation boundary",
            "",
            "These values describe aggregate structural differences in the released annotated subset. "
            "They do not classify individual texts or measure overall writing quality.",
        ]
    )
    return "\n".join(lines)


def experiment_markdown(result: Mapping[str, Any]) -> str:
    metadata = result["metadata"]
    lines = [
        "# Multilingual storytelling experiment",
        "",
        "Result tier: **%s**" % metadata["result_tier"],
        "",
        "Model: `%s` via `%s`  " % (metadata["model"], metadata["provider"]),
        "Successful generations: **%s / %s**  "
        % (metadata["successful_count"], metadata["request_count"]),
        "Failed generations: **%s**" % metadata["failure_count"],
        "",
        "## Condition comparison",
        "",
        "| Condition | n | Arc entropy | Plot repeat rate | Suspense | Flattening | Token distance | Human arc distance | TP order violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, summary in result["conditions"].items():
        lexical = summary["lexical_diversity"]
        discourse = summary["discourse_features"]
        human = summary.get("distance_from_human", {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s/%s |"
            % (
                condition,
                summary["sample_count"],
                _number(summary["arc_distribution"]["normalized_entropy"]),
                _number(summary["plot_diversity"]["duplicate_signature_rate"]),
                _number(discourse["mean_suspense_proxy"]),
                _number(discourse["mean_flattening_proxy"]),
                _number(lexical["pairwise_token_jaccard_distance"]),
                _number(human.get("arc_jensen_shannon_distance")),
                summary["turning_points"]["ordering_violation_count"],
                summary["turning_points"]["sample_count"],
            )
        )
    if result.get("narrative_parameters"):
        lines.extend(
            [
                "",
                "## Narrative parameter breakdown",
                "",
                "| Parameter cell | n | Arc entropy | Plot repeat rate | Suspense | Valence range | Arousal range | Flattening |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for parameter, summary in result["narrative_parameters"].items():
            discourse = summary["discourse_features"]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    parameter,
                    summary["sample_count"],
                    _number(summary["arc_distribution"]["normalized_entropy"]),
                    _number(summary["plot_diversity"]["duplicate_signature_rate"]),
                    _number(discourse["mean_suspense_proxy"]),
                    _number(discourse["mean_valence_range"]),
                    _number(discourse["mean_arousal_range"]),
                    _number(discourse["mean_flattening_proxy"]),
                )
            )
    lines.extend(["", "## Language breakdown", "", "| Group | n | Arc entropy | Token distance |", "|---|---:|---:|---:|"])
    for group, summary in result["condition_languages"].items():
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                group,
                summary["sample_count"],
                _number(summary["arc_distribution"]["normalized_entropy"]),
                _number(summary["lexical_diversity"]["pairwise_token_jaccard_distance"]),
            )
        )
    hypotheses = result.get("hypothesis_checks", {})
    lines.extend(["", "## Descriptive hypothesis checks", ""])
    for name, check in hypotheses.items():
        lines.append("- **%s:** %s" % (name, check))
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        for failure in result["failures"]:
            lines.append("- `%s`: %s" % (failure["prompt_id"], failure["message"]))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an exploratory single-model local experiment with only three stories per condition. "
            "A separate local-model pass assigned arcs and turning points after generation; those labels "
            "require blind independent annotation before publication-quality conclusions. Non-monotonic "
            "turning-point annotations are retained and counted rather than silently repaired, so the human "
            "TP distances are diagnostic only. The affective analysis uses a small transparent proxy "
            "lexicon, not the paper's full GPT-4 plus NRC VAD pipeline. The requested English output "
            "makes lexical comparison direct but tests prompt-language effects rather than native-language prose.",
        ]
    )
    return "\n".join(lines)
